import torch, math, random, cv2
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Union, List, Optional, Tuple
from torchvision.utils import save_image
from depth_anything_3.api import DepthAnything3

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoProcessor, AutoTokenizer
from ..models.flux2_text_encoder import Flux2TextEncoder
from ..models.flux2_dit import Flux2DiT
from ..models.flux2_vae import Flux2VAE
from ..models.z_image_text_encoder import ZImageTextEncoder
from ..utils.model_cache import get_da3_cache_dir


def resample_erp_noise(
    base_noise,
    output_height,
    output_width,
    erp_f,
    erp_full_size,
    erp_crop,
    erp_content_size,
    latitude_aware=True,
    vae_scale_factor=16,
):
    """Sample a cropped ERP noise field from a global noise canvas.

    The latitude-aware branch follows PanoWan: horizontal sampling is
    compressed by cos(latitude), then the interpolated magnitude is restored
    from the locally interpolated second moment.
    """
    if base_noise.ndim != 4:
        raise ValueError(f"base_noise must have shape [B, C, H, W], got {base_noise.shape}.")

    full_w, full_h = (float(erp_full_size[0]), float(erp_full_size[1]))
    crop_x0, crop_x1, crop_y0, crop_y1 = (float(value) for value in erp_crop)
    content_w, content_h = (float(erp_content_size[0]), float(erp_content_size[1]))
    if erp_f <= 0 or full_w <= 0 or full_h <= 0 or content_w <= 0 or content_h <= 0:
        raise ValueError("ERP focal length and canvas sizes must be positive.")

    device = base_noise.device
    dtype = base_noise.dtype
    y = (torch.arange(output_height, device=device, dtype=dtype) + 0.5) * vae_scale_factor
    x = (torch.arange(output_width, device=device, dtype=dtype) + 0.5) * vae_scale_factor

    # The generated canvas can be padded beyond the projected content to meet
    # the 1024 AR window minimum. Extrapolate the same pixel-to-ERP scale there.
    v_abs = crop_y0 + y * ((crop_y1 - crop_y0) / content_h)
    u_abs = crop_x0 + x * ((crop_x1 - crop_x0) / content_w)
    v_abs = v_abs.clamp(0.0, max(full_h - 1e-4, 0.0))
    u_abs = u_abs.clamp(0.0, max(full_w - 1e-4, 0.0))

    if latitude_aware:
        latitude = math.pi / 2 - v_abs / float(erp_f)
        full_center_x = full_w / 2
        u_sample = full_center_x + (u_abs - full_center_x) * torch.cos(latitude)[:, None]
    else:
        u_sample = u_abs[None, :].expand(output_height, -1)
    v_sample = v_abs[:, None].expand(-1, output_width)
    if u_sample.shape[-1] == 1 and output_width != 1:
        u_sample = u_sample.expand(-1, output_width)

    base_h, base_w = base_noise.shape[-2:]
    grid_x = 2 * (u_sample / vae_scale_factor) / base_w - 1
    grid_y = 2 * (v_sample / vae_scale_factor) / base_h - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    grid = grid.expand(base_noise.shape[0], -1, -1, -1)

    interpolated = torch.nn.functional.grid_sample(
        base_noise,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    second_moment = torch.nn.functional.grid_sample(
        base_noise.square(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return torch.sign(interpolated) * torch.sqrt(second_moment.clamp_min(1e-12))


class Flux2ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16, is_training=True):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: Flux2TextEncoder = None
        self.text_encoder_qwen3: ZImageTextEncoder = None
        self.dit: Flux2DiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Flux2Unit_ShapeChecker(),
            Flux2Unit_PromptEmbedder(),
            Flux2Unit_Qwen3PromptEmbedder(),
            Flux2Unit_NoiseInitializer(),
            Flux2Unit_InputImageEmbedder(),
            Flux2Unit_EditImageEmbedder(is_training=is_training),
            Flux2Unit_ImageIDs(),
            Flux2Unit_AR() if is_training else Flux2Unit_AR_Inference(),
        ]
        self.model_fn = model_fn_flux2
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
        is_training: bool = True,
    ):
        # Initialize pipeline
        pipe = Flux2ImagePipeline(device=device, torch_dtype=torch_dtype, is_training=is_training)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
        pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("flux2_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # Depth Anything V3
        pipe.da3 = DepthAnything3.from_pretrained(
            "depth-anything/DA3NESTED-GIANT-LARGE",
            cache_dir=get_da3_cache_dir(),
        ).to(device)
        pipe.da3.eval()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Edit
        edit_image: Union[Image.Image, List[Image.Image]] = None,
        edit_image_auto_resize: bool = True,
        # AR
        pano_latent: torch.Tensor = None,
        crop_region_latent: List[int] = None,
        weight_map_latent: torch.Tensor = None,
        gt_height=None,
        gt_width=None,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Progress bar
        progress_bar_cmd = tqdm,
        coord_maps = None,
        coord_maps_depth = None,
        camera_params = None,
        source_view_top_k: int = 0,
        separate_median_rope_normalization: bool = False,
        edit_token_rectification_mode: str = "none",
        edit_token_rectification_strength: float = 1.0,
        edit_token_rectification_min_scale: float = 0.25,
        edit_token_rectification_max_scale: float = 4.0,
        edit_token_rectification_patch_size: int = 48,
        edit_token_rectification_batch_size: int = 64,
        edit_token_rectification_transform: str = "rotation_only",
        spherical_noise_mode: str = "none",
        spherical_noise_metadata = None,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=height//16*width//16)

        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "coord_maps": coord_maps,
            "coord_maps_depth": coord_maps_depth,
            "camera_params": camera_params,
            "source_view_top_k": source_view_top_k,
            "separate_median_rope_normalization": separate_median_rope_normalization,
            "edit_token_rectification_mode": edit_token_rectification_mode,
            "edit_token_rectification_strength": edit_token_rectification_strength,
            "edit_token_rectification_min_scale": edit_token_rectification_min_scale,
            "edit_token_rectification_max_scale": edit_token_rectification_max_scale,
            "edit_token_rectification_patch_size": edit_token_rectification_patch_size,
            "edit_token_rectification_batch_size": edit_token_rectification_batch_size,
            "edit_token_rectification_transform": edit_token_rectification_transform,
            "spherical_noise_mode": spherical_noise_mode,
            "spherical_noise_metadata": spherical_noise_metadata,
            "pano_latent": pano_latent, "crop_region_latent": crop_region_latent, "weight_map_latent": weight_map_latent, "gt_height": gt_height, "gt_width": gt_width,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        latents = rearrange(inputs_shared["latents"], "B (H W) C -> B C H W", H=inputs_shared["height"]//16, W=inputs_shared["width"]//16)
        image = self.vae.decode(latents)
        # image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image, latents


import random
import torch
from einops import rearrange
from ..diffusion.base_pipeline import PipelineUnit

class Flux2Unit_AR(PipelineUnit):
    def __init__(self, crop_latent_size=64, context_prob=0.85):
        """
        :param crop_latent_size: 目标裁剪区域大小 (默认 64x64 latent)
        :param context_prob: 训练时有多少概率会加入 Context (模拟推理起步第一张没有先验的情况)
        """
        super().__init__(
            input_params=("input_latents", "image_ids", "edit_latents", "edit_image_ids", "gt_height", "gt_width"),
            output_params=(
                "input_latents", "image_ids", "edit_latents", "edit_image_ids",
                "training_crop_region_latent",
            ),
        )
        self.crop_latent_size = crop_latent_size
        self.context_prob = context_prob

    def process(self, pipe, input_latents, image_ids, edit_latents, edit_image_ids, gt_height, gt_width):
        H_latent = gt_height // 16
        W_latent = gt_width // 16

        # assert H_latent >= self.crop_latent_size and W_latent >= self.crop_latent_size
        
        cH = self.crop_latent_size
        cW = self.crop_latent_size

        # 1. 随机生成 Target Crop 的左上角起点
        h_start = random.randint(0, max(0, H_latent - cH))
        w_start = random.randint(0, max(0, W_latent - cW))
        h_end = min(H_latent, h_start + cH)
        w_end = min(W_latent, w_start + cW)

        # ==========================================
        # Step A: 对目标画布 (Noisy GT & IDs) 进行 Crop
        # ==========================================
        B, _, C = input_latents.shape
        
        latents_2d = input_latents.view(B, H_latent, W_latent, C)
        latents_cropped = latents_2d[:, h_start:h_end, w_start:w_end, :].reshape(B, (h_end - h_start) * (w_end - w_start), C)

        ids_2d = image_ids.view(1, H_latent, W_latent, 2, 4)
        ids_cropped = ids_2d[:, h_start:h_end, w_start:w_end, :, :].reshape(1, (h_end - h_start) * (w_end - w_start), 2, 4)

        # ==========================================
        # Step B: 对条件图像 (Edit Latents) 进行 Bounding Box 过滤
        # ==========================================
        # Both RoPE branches must describe a valid target position.  The
        # coord-map converter uses 1000 as the invalid sentinel; filtering on
        # only the true-depth branch lets a token with a valid depth position
        # but invalid median-depth position leak 1000 into projected RoPE.
        true_coords = edit_image_ids[0, :, 0, 1:3]
        median_coords = edit_image_ids[0, :, 1, 1:3]
        invalid_coord = 1000.0
        valid_true = (
            torch.isfinite(true_coords).all(dim=1)
            & (true_coords >= 0).all(dim=1)
            & (true_coords < invalid_coord).all(dim=1)
        )
        valid_median = (
            torch.isfinite(median_coords).all(dim=1)
            & (median_coords >= 0).all(dim=1)
            & (median_coords < invalid_coord).all(dim=1)
        )
        h_warped = true_coords[:, 0]
        w_warped = true_coords[:, 1]

        mask = (
            valid_true & valid_median
            & (h_warped >= h_start) & (h_warped < h_end)
            & (w_warped >= w_start) & (w_warped < w_end)
        )

        edit_latents_filtered = edit_latents[:, mask, :]
        edit_image_ids_filtered = edit_image_ids[:, mask, :, :]

        # random_id_candidates = torch.tensor(
        #     [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        #     device=edit_image_ids_filtered.device,
        #     dtype=edit_image_ids_filtered.dtype,
        # )
        # token_random_indices = torch.randint(
        #     0,
        #     random_id_candidates.shape[0],
        #     (edit_image_ids_filtered.shape[0], edit_image_ids_filtered.shape[1], 1),
        #     device=edit_image_ids_filtered.device,
        # )
        # token_random_ids = random_id_candidates[token_random_indices]
        # edit_image_ids_filtered_random_ids = token_random_ids.expand_as(edit_image_ids_filtered[:, :, :, 0])
        # edit_image_ids_filtered[:, :, :, 0] = edit_image_ids_filtered_random_ids

        # ==========================================
        # Step C: 【高阶】全方位 Context 模拟 (Boolean Mask 提取)
        # ==========================================
        if random.random() < self.context_prob and H_latent > cH and W_latent > cW:
            # 创建全局布尔掩码，用于防止多个 Context 窗口叠加导致的 Token 重复
            context_mask = torch.zeros((H_latent, W_latent), dtype=torch.bool, device=input_latents.device)
            
            # 8 向偏移 (包含上下左右 + 四个对角线角)
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),    # 边缘相连
                (-1, -1), (-1, 1), (1, -1), (1, 1)   # 对角线/角落相连
            ]
            
            dh_sign, dw_sign = random.sample(directions, 1)[0]

            # 动态重叠率：步长跨度从 20% 到 90% 不等，让模型适应各种 Overlap 大小
            if random.random() < 0.75:
                stride_h = int(cH * 0.5) if dh_sign != 0 else 0
                stride_w = int(cW * 0.5) if dw_sign != 0 else 0
            else:
                stride_h = random.randint(int(cH * 0.5), cH) if dh_sign != 0 else 0
                stride_w = random.randint(int(cW * 0.5), cW) if dw_sign != 0 else 0

            h_prev = h_start + dh_sign * stride_h
            w_prev = w_start + dw_sign * stride_w

            # 约束在图像边界内
            # h_prev = max(0, min(H_latent - cH, h_prev))
            # w_prev = max(0, min(W_latent - cW, w_prev))

            overlap_h_start = max(h_prev, h_start)
            overlap_w_start = max(w_prev, w_start)
            overlap_h_end = min(h_prev + cH, h_start + cH)
            overlap_w_end = min(w_prev + cW, w_start + cW)

            # # 过滤掉完全与 target 叠合的无效 context
            # if h_prev == h_start and w_prev == w_start:
            #     continue
            
            # 将该 Context 窗口区域涂 True
            context_mask[h_start:h_start+cH, w_start:w_start+cW] = True
            context_mask[overlap_h_start:overlap_h_end, overlap_w_start:overlap_w_end] = False

            # 将 2D 掩码展平，利用布尔索引一次性提取全部 Context Tokens (天然去重！)
            mask_1d = context_mask.view(-1)
            
            # 只有当 Mask 里真的有内容时才 Concat
            if mask_1d.any():
                # 从 Input Latents (干净的 GT) 中抠出所有标记的 Context Tokens
                context_latents = input_latents[:, mask_1d, :]
                context_ids = image_ids[:, mask_1d, :, :]

                # 无缝拼接给 Edit Latents 供模型 Attention
                edit_latents_filtered = torch.cat([edit_latents_filtered, context_latents], dim=1)
                edit_image_ids_filtered = torch.cat([edit_image_ids_filtered, context_ids], dim=1)

        if edit_image_ids_filtered.numel() > 0:
            norm_h_start = edit_image_ids_filtered[0, :, :, 1].min()
            norm_w_start = edit_image_ids_filtered[0, :, :, 2].min()

            edit_image_ids_filtered[0, :, :, 1] = edit_image_ids_filtered[0, :, :, 1] - norm_h_start
            edit_image_ids_filtered[0, :, :, 2] = edit_image_ids_filtered[0, :, :, 2] - norm_w_start
        else:
            norm_h_start = ids_cropped[0, :, :, 1].min()
            norm_w_start = ids_cropped[0, :, :, 2].min() 

        ids_cropped[0, :, :, 1] = ids_cropped[0, :, :, 1] - norm_h_start
        ids_cropped[0, :, :, 2] = ids_cropped[0, :, :, 2] - norm_w_start

        edit_image_ids_filtered[:, :, :, 0] = edit_image_ids_filtered[:, :, :, 0] + 10.

        return {
            "input_latents": latents_cropped,
            "image_ids": ids_cropped,
            "edit_latents": edit_latents_filtered,
            "edit_image_ids": edit_image_ids_filtered,
            "training_crop_region_latent": [w_start, h_start, w_end, h_end],
        }


class Flux2Unit_AR_Inference(PipelineUnit):
    def __init__(self, crop_latent_size=64, context_prob=0.5):
        """
        :param crop_latent_size: 目标裁剪区域大小 (默认 64x64 latent)
        :param context_prob: 训练时有多少概率会加入 Context (模拟推理起步第一张没有先验的情况)
        """
        super().__init__(
            input_params=("image_ids", "edit_latents", "edit_image_ids", "gt_height", "gt_width", "pano_latent", "crop_region_latent", "weight_map_latent", "latents", "source_view_top_k", "separate_median_rope_normalization"),
            output_params=("latents", "input_latents", "image_ids", "edit_latents", "edit_image_ids"),
        )
        self.crop_latent_size = crop_latent_size
        self.context_prob = context_prob

    def process(self, pipe, image_ids, edit_latents, edit_image_ids, gt_height, gt_width, pano_latent, crop_region_latent, weight_map_latent, latents, source_view_top_k=0, separate_median_rope_normalization=False):
        H_latent = gt_height // 16
        W_latent = gt_width // 16

        # assert H_latent >= self.crop_latent_size and W_latent >= self.crop_latent_size
        
        w_start = crop_region_latent[0]
        h_start = crop_region_latent[1]
        w_end = crop_region_latent[2]
        h_end = crop_region_latent[3]

        # ==========================================
        # Step A: 对目标画布 (Noisy GT & IDs) 进行 Crop
        # ==========================================
        B, C, _, _ = pano_latent.shape

        weight_map_latent_cropped = weight_map_latent[h_start:h_end, w_start:w_end].reshape((h_end - h_start) * (w_end - w_start))

        latents_2d = pano_latent
        latents_cropped = latents_2d[:, :, h_start:h_end, w_start:w_end].permute(0, 2, 3, 1).reshape(B, (h_end - h_start) * (w_end - w_start), C)

        noisy_latents_cropped = latents.view(B, H_latent, W_latent, C)[:, h_start:h_end, w_start:w_end, :].reshape(B, (h_end - h_start) * (w_end - w_start), C)

        ids_2d = image_ids.view(1, H_latent, W_latent, 2, 4)
        ids_cropped = ids_2d[:, h_start:h_end, w_start:w_end, :, :].reshape(1, (h_end - h_start) * (w_end - w_start), 2, 4)

        mask = weight_map_latent_cropped > 0
        latents_cropped_selected = latents_cropped[:, mask, :]
        ids_cropped_selected = ids_cropped[:, mask, :, :]

        # ==========================================
        # Step B: 对条件图像 (Edit Latents) 进行 Bounding Box 过滤
        # ==========================================
        true_coords = edit_image_ids[0, :, 0, 1:3]
        median_coords = edit_image_ids[0, :, 1, 1:3]
        invalid_coord = 1000.0
        valid_true = (
            torch.isfinite(true_coords).all(dim=1)
            & (true_coords >= 0).all(dim=1)
            & (true_coords < invalid_coord).all(dim=1)
        )
        valid_median = (
            torch.isfinite(median_coords).all(dim=1)
            & (median_coords >= 0).all(dim=1)
            & (median_coords < invalid_coord).all(dim=1)
        )
        h_warped = edit_image_ids[0, :, 0, 1] 
        w_warped = edit_image_ids[0, :, 0, 2]

        larger_h_start = h_start
        larger_w_start = w_start
        larger_h_end = larger_h_start + 64
        larger_w_end = larger_w_start + 64

        larger_h_start = max(0, larger_h_start)
        larger_w_start = max(0, larger_w_start)
        larger_h_end = min(H_latent, larger_h_end)
        larger_w_end = min(W_latent, larger_w_end)

        mask = (
            valid_true & valid_median
            & (h_warped >= larger_h_start) & (h_warped < larger_h_end)
            & (w_warped >= larger_w_start) & (w_warped < larger_w_end)
        )

        edit_latents_filtered = edit_latents[:, mask, :]
        edit_image_ids_filtered = edit_image_ids[:, mask, :, :]

        frame_ids = edit_image_ids_filtered[:, :, :, 0]
        frame_id_classes_before = torch.sort(torch.unique(torch.round(frame_ids).to(torch.long)))[0]
        print("frame id classes before re-encode:", frame_id_classes_before.detach().cpu().tolist())

        # Count source rows per frame (the true-depth and median-depth RoPE
        # branches share one source token and therefore count only once).
        source_frame_ids = torch.round(edit_image_ids_filtered[0, :, 0, 0]).to(torch.long)
        count_ids, count_values = torch.unique(
            source_frame_ids, sorted=True, return_counts=True
        )
        print(
            "frame id token counts before top-k:",
            {int(frame_id): int(count) for frame_id, count in zip(count_ids, count_values)},
        )

        # K=0 means no view filtering.  For K>0, retain every token belonging
        # to the K frame ids with the largest token counts in this patch.
        top_k = int(source_view_top_k or 0)
        if top_k > 0 and count_ids.numel() > top_k:
            ranked_ids = sorted(
                zip(count_ids.detach().cpu().tolist(), count_values.detach().cpu().tolist()),
                key=lambda item: (-item[1], item[0]),
            )
            keep_ids = {frame_id for frame_id, _ in ranked_ids[:top_k]}
            keep_mask = torch.zeros_like(source_frame_ids, dtype=torch.bool)
            for frame_id in keep_ids:
                keep_mask |= source_frame_ids == int(frame_id)
            edit_latents_filtered = edit_latents_filtered[:, keep_mask, :]
            edit_image_ids_filtered = edit_image_ids_filtered[:, keep_mask, :, :]
            kept_ids, kept_counts = torch.unique(
                torch.round(edit_image_ids_filtered[0, :, 0, 0]).to(torch.long),
                sorted=True,
                return_counts=True,
            )
            print(
                f"top-k source views applied (k={top_k}):",
                {int(frame_id): int(count) for frame_id, count in zip(kept_ids, kept_counts)},
            )

        # Recompute classes after optional filtering so re-encoding never
        # leaves gaps for views removed from this patch.
        frame_ids = edit_image_ids_filtered[:, :, :, 0]
        frame_id_classes_before = torch.sort(torch.unique(torch.round(frame_ids).to(torch.long)))[0]

        frame_ids_int = torch.round(frame_ids).to(torch.long)
        frame_ids_reencoded = torch.empty_like(frame_ids_int)
        for new_frame_id, old_frame_id in enumerate(frame_id_classes_before):
            frame_ids_reencoded[frame_ids_int == old_frame_id] = new_frame_id * 10
        edit_image_ids_filtered[:, :, :, 0] = frame_ids_reencoded.to(edit_image_ids_filtered.dtype)
        edit_image_ids_filtered[:, :, :, 0] = edit_image_ids_filtered[:, :, :, 0] + 10.

        frame_id_classes_after = torch.sort(torch.unique(torch.round(edit_image_ids_filtered[:, :, :, 0]).to(torch.long)))[0]
        print("frame id classes after re-encode:", frame_id_classes_after.detach().cpu().tolist())

        source_token_count = edit_image_ids_filtered.shape[1]
        edit_latents_filtered = torch.cat([edit_latents_filtered, latents_cropped_selected], dim=1)
        edit_image_ids_filtered = torch.cat([edit_image_ids_filtered, ids_cropped_selected], dim=1)

        edit_image_ids_filtered[:, :, :, 0] = edit_image_ids_filtered[:, :, :, 0] + 10.

        if edit_image_ids_filtered.shape[1] > 0:
            # The source-token mask above removes all invalid map sentinels.
            # Keep this assertion close to normalization so future changes do
            # not silently reintroduce very large invalid RoPE positions.
            source_ids = edit_image_ids_filtered[:, :, :2, 1:3]
            if not bool(torch.isfinite(source_ids).all()):
                raise RuntimeError("non-finite projected RoPE coordinate after coord-map filtering")
            if separate_median_rope_normalization:
                # Group A: true-depth source branch and target/context IDs
                # share one coordinate origin.  Group B: median-depth source
                # branch gets an independent origin for this patch.
                true_h_values = torch.cat(
                    [edit_image_ids_filtered[0, :, 0, 1], ids_cropped[0, :, :, 1].reshape(-1)]
                )
                true_w_values = torch.cat(
                    [edit_image_ids_filtered[0, :, 0, 2], ids_cropped[0, :, :, 2].reshape(-1)]
                )
                norm_h_start = true_h_values.min()
                norm_w_start = true_w_values.min()
                if source_token_count > 0:
                    median_h_start = edit_image_ids_filtered[0, :source_token_count, 1, 1].min()
                    median_w_start = edit_image_ids_filtered[0, :source_token_count, 1, 2].min()
                else:
                    median_h_start = norm_h_start
                    median_w_start = norm_w_start

                edit_image_ids_filtered[0, :, 0, 1] -= norm_h_start
                edit_image_ids_filtered[0, :, 0, 2] -= norm_w_start
                if source_token_count > 0:
                    edit_image_ids_filtered[0, :source_token_count, 1, 1] -= median_h_start
                    edit_image_ids_filtered[0, :source_token_count, 1, 2] -= median_w_start
                if source_token_count < edit_image_ids_filtered.shape[1]:
                    # Context rows come from ids_cropped and therefore stay
                    # in the true/target coordinate group.
                    edit_image_ids_filtered[0, source_token_count:, 1, 1] -= norm_h_start
                    edit_image_ids_filtered[0, source_token_count:, 1, 2] -= norm_w_start
                ids_cropped[0, :, :, 1] -= norm_h_start
                ids_cropped[0, :, :, 2] -= norm_w_start
                print(
                    "separate RoPE normalization applied:",
                    f"true_target_h_offset={float(norm_h_start):.6f}",
                    f"true_target_w_offset={float(norm_w_start):.6f}",
                    f"median_h_offset={float(median_h_start):.6f}",
                    f"median_w_offset={float(median_w_start):.6f}",
                )
            else:
                norm_h_start = edit_image_ids_filtered[0, :, :, 1].min()
                norm_w_start = edit_image_ids_filtered[0, :, :, 2].min()
                edit_image_ids_filtered[0, :, :, 1] -= norm_h_start
                edit_image_ids_filtered[0, :, :, 2] -= norm_w_start
                ids_cropped[0, :, :, 1] -= norm_h_start
                ids_cropped[0, :, :, 2] -= norm_w_start
            print("h: ", edit_image_ids_filtered[0, :, :, 1].min(), edit_image_ids_filtered[0, :, :, 1].max(), edit_image_ids_filtered[0, :, :, 1].mean())
            print("w: ", edit_image_ids_filtered[0, :, 0, 2].min(), edit_image_ids_filtered[0, :, 0, 2].max(), edit_image_ids_filtered[0, :, 0, 2].mean())
            print("w: ", edit_image_ids_filtered[0, :, 1, 2].min(), edit_image_ids_filtered[0, :, 1, 2].max(), edit_image_ids_filtered[0, :, 1, 2].mean())

        print("latents_cropped.shape", latents_cropped.shape)
        print("ids_cropped.shape", ids_cropped.shape)
        print("latents_cropped_selected.shape", latents_cropped_selected.shape)
        print("ids_cropped_selected.shape", ids_cropped_selected.shape)
        print("edit_latents_filtered.shape", edit_latents_filtered.shape)
        print("edit_image_ids_filtered.shape", edit_image_ids_filtered.shape)

        return {
            "latents": noisy_latents_cropped,
            "input_latents": latents_cropped,
            "image_ids": ids_cropped,
            "edit_latents": edit_latents_filtered,
            "edit_image_ids": edit_image_ids_filtered,
        }


class Flux2Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: Flux2ImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Flux2Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        self.system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."

    def format_text_input(self, prompts: List[str], system_message: str = None):
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def get_mistral_3_small_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        # fmt: off
        system_message: str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.",
        # fmt: on
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = self.format_text_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds
    
    def prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self.get_mistral_3_small_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2ImagePipeline, prompt):
        # Skip if Qwen3 text encoder is available (handled by Qwen3PromptEmbedder)
        if pipe.text_encoder_qwen3 is not None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=pipe.device,
        )
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2Unit_Qwen3PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder_qwen3",)
        )
        self.hidden_states_layers = (9, 18, 27)  # Qwen3 layers

    def get_qwen3_prompt_embeds(
        self,
        text_encoder: ZImageTextEncoder,
        tokenizer: AutoTokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        with torch.inference_mode():
            output = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    def prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            # 1. 确定序列索引
            l_idx = torch.arange(L, device=x.device)
            
            # 2. 文本没有 2D 空间和投影属性，填充 0
            zeros = torch.zeros(L, device=x.device)
            
            # 3. 堆叠成 4 维坐标 (严格对齐图像的 [t, h, w, c] 格式)
            # 文本没有时空维度，所以前三维给 0，序列索引放在最后一维
            coords = torch.stack([
                zeros,  # Dim 0: 对应图像的 t (Time/View)
                zeros,  # Dim 1: 对应图像的 h (Height)
                zeros,  # Dim 2: 对应图像的 w (Width)
                l_idx   # Dim 3: 对应图像的 c (Channel/Sequence)
            ], dim=1)
            
            out_ids.append(coords)

        return torch.stack(out_ids) # 返回 (B, L, 4)

    def encode_prompt(
        self,
        text_encoder: ZImageTextEncoder,
        tokenizer: AutoTokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self.get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2ImagePipeline, prompt):
        # Check if Qwen3 text encoder is available
        if pipe.text_encoder_qwen3 is None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder_qwen3, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=pipe.device,
        )
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "gt_height", "gt_width", "seed", "rand_device",
                "spherical_noise_mode", "spherical_noise_metadata",
            ),
            output_params=("noise",),
        )

    def process(
        self,
        pipe: Flux2ImagePipeline,
        gt_height,
        gt_width,
        seed,
        rand_device,
        spherical_noise_mode,
        spherical_noise_metadata,
    ):
        latent_h, latent_w = gt_height // 16, gt_width // 16
        spherical_noise_mode = spherical_noise_mode or "none"
        if spherical_noise_mode == "none":
            noise = pipe.generate_noise(
                (1, 128, latent_h, latent_w),
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=pipe.torch_dtype,
            )
        elif spherical_noise_mode in {"erp_iid", "panowan"}:
            if spherical_noise_metadata is None:
                raise ValueError(f"{spherical_noise_mode=} requires spherical_noise_metadata.")
            if spherical_noise_metadata.get("projection_mode") != "equirectangular":
                raise ValueError(f"{spherical_noise_mode=} is only valid for equirectangular projection.")

            full_w, full_h = spherical_noise_metadata["full_size"]
            full_latent_w = max(1, math.ceil(full_w / 16))
            full_latent_h = max(1, math.ceil(full_h / 16))
            base_noise = pipe.generate_noise(
                (1, 128, full_latent_h, full_latent_w),
                seed=seed,
                rand_device=rand_device,
                rand_torch_dtype=torch.float32,
                torch_dtype=torch.float32,
            )
            noise = resample_erp_noise(
                base_noise,
                output_height=latent_h,
                output_width=latent_w,
                erp_f=float(spherical_noise_metadata["f"]),
                erp_full_size=spherical_noise_metadata["full_size"],
                erp_crop=spherical_noise_metadata["crop"],
                erp_content_size=spherical_noise_metadata["content_size"],
                latitude_aware=spherical_noise_mode == "panowan",
            ).to(pipe.torch_dtype)
        else:
            raise ValueError(
                f"Unsupported spherical_noise_mode: {spherical_noise_mode}. "
                "Expected one of: none, erp_iid, panowan."
            )

        noise = noise.reshape(1, 128, gt_height//16 * gt_width//16).permute(0, 2, 1)
        return {"noise": noise}


class Flux2Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "edit_image", "noise", "gt_height", "gt_width", "coord_maps", "coord_maps_depth", "camera_params"),
            output_params=("latents", "input_latents", "coord_maps", "coord_maps_depth"),
            onload_model_names=("vae",)
        )
    
    def slice_image_adaptive(self, H, W, patch_size=1024, stride=512):
        nx = int((W - patch_size + stride) / stride)
        ny = int((H - patch_size + stride) / stride)
        x_coord_list = [stride * i for i in range(nx)]
        y_coord_list = [stride * i for i in range(ny)]

        if x_coord_list[-1] + patch_size != W:
            x_coord_list.append(W - patch_size)
        if y_coord_list[-1] + patch_size != H:
            y_coord_list.append(H - patch_size)
        
        coords = []
        for x in x_coord_list:
            for y in y_coord_list:
                coords.append([x, y, x + patch_size, y + patch_size])
        
        return coords
    
    def get_linear_mask(self, size, min_weight=1e-3, device="cuda", dtype=torch.float32):
        """
        生成二维线性渐变掩码 (Bilinear Blending Mask)
        中心权重为 1.0，向四周边缘绝对线性衰减
        """
        # 生成从 -1 到 1 的坐标
        coords = torch.linspace(-1, 1, size, device=device, dtype=dtype)
        
        # 1D 线性衰减：公式为 1 - |x|
        # 中心 0 处为 1.0，边缘 -1 和 1 处为 0.0
        window_1d = 1.0 - torch.abs(coords)
        
        # 外积生成 2D 掩码（形状像一个四棱锥形 / 金字塔形）
        window_2d = torch.outer(window_1d, window_1d)
        
        # 依然保留这个 1e-3 的极小保底权重，用来彻底防止四个纯边缘的角点变黑
        window_2d = torch.clamp(window_2d, min=min_weight)
        
        return window_2d


    def forward_warp_using_map(self, src_img, coord_map, target_size=None, background_color=(0, 0, 0)):
        """
        使用 Forward Coord Map 将原图变换到投影图
        
        Args:
            src_img: (H_src, W_src, 3) 原图
            coord_map: (H_src, W_src, 2) 映射表，最后一维是 (target_u, target_v)
            target_size: (W_tgt, H_tgt) 目标画布尺寸。如果不传，自动根据 map 最大值推断。
            background_color: 背景填充色 (B, G, R)
            
        Returns:
            warped_img: (H_tgt, W_tgt, 3) 结果图
            mask: (H_tgt, W_tgt) 哪些位置被填充了 (0/255)
        """
        H_src, W_src = src_img.shape[:2]
        
        # 1. 展平数据 (N, ...)
        # 这样方便进行统一的索引操作
        src_pixels = src_img.reshape(-1, 3)
        coords = coord_map.reshape(-1, 2)
        
        # 2. 筛选有效映射点
        # 假设 -1 是无效值的标记
        valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)
        
        valid_src_pixels = src_pixels[valid_mask]
        valid_coords = coords[valid_mask]
        
        # 3. 确定目标画布尺寸
        if target_size is None:
            # 如果没指定，就找坐标里的最大值作为边界
            max_u = np.max(valid_coords[:, 0])
            max_v = np.max(valid_coords[:, 1])
            W_tgt, H_tgt = int(max_u) + 1, int(max_v) + 1
        else:
            W_tgt, H_tgt = target_size

        # 4. 坐标离散化 (Float -> Int)
        # 因为像素坐标必须是整数
        u_tgt = np.round(valid_coords[:, 0]).astype(np.int32)
        v_tgt = np.round(valid_coords[:, 1]).astype(np.int32)
        
        # 5. 边界检查 (Clip)
        # 这一步是为了防止 Resize/Crop 后有极少数点溢出
        keep_mask = (u_tgt >= 0) & (u_tgt < W_tgt) & (v_tgt >= 0) & (v_tgt < H_tgt)
        
        u_tgt = u_tgt[keep_mask]
        v_tgt = v_tgt[keep_mask]
        colors = valid_src_pixels[keep_mask]
        
        # 6. 投射 (Splatting)
        # 初始化画布
        warped_img = np.full((H_tgt, W_tgt, 3), background_color, dtype=np.uint8)
        fill_mask = np.zeros((H_tgt, W_tgt), dtype=np.uint8)
        
        # 【核心操作】利用 NumPy 高级索引直接赋值
        # 如果有多个点投射到同一个位置，后面的会覆盖前面的 (简单的 Z-buffer 替代方案)
        warped_img[v_tgt, u_tgt] = colors
        fill_mask[v_tgt, u_tgt] = 255
        
        return warped_img, fill_mask

    def unproject_points(self, image, depth, K):
        """Step 1: 2D + Depth -> Local 3D"""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 【修改点 1】: 保留原图的 u, v 坐标用于后续建立映射
        u_src_grid, v_src_grid = u.flatten(), v.flatten() # 备份一份原图网格坐标
        u, v, z = u.flatten(), v.flatten(), depth.flatten()
        
        valid_mask = z > 0
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]
        
        # 【修改点 2】: 对原图坐标做同样的筛选
        u_src = u_src_grid[valid_mask]
        v_src = v_src_grid[valid_mask]
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_3d_local = np.stack([x, y, z], axis=1)
        colors = image.reshape(-1, 3)[valid_mask]
        
        # 【修改点 3】: 返回 points, colors 的同时，返回对应的原图坐标
        return points_3d_local, colors, u_src, v_src


    def project_to_target_camera(self, points_local, K, W, H):
        """
        Step 4: Local 3D -> Target 2D Pixel Coordinates
        直接使用目标相机的内参 K 投影到 [0, W] x [0, H] 的图像空间
        """
        X, Y, Z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
        
        # 剔除相机背后的点
        valid_mask = Z > 0.01 
        
        u_proj = np.zeros_like(X)
        v_proj = np.zeros_like(Y)
        
        if np.any(valid_mask):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # 针孔相机投影模型
            u_proj[valid_mask] = fx * (X[valid_mask] / Z[valid_mask]) + cx
            v_proj[valid_mask] = fy * (Y[valid_mask] / Z[valid_mask]) + cy
            
        # 可选：如果你只关心落在目标图像画面内的点，可以进一步卡可见范围
        # in_screen_mask = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
        # valid_mask = valid_mask & in_screen_mask
            
        return np.stack([u_proj, v_proj], axis=1), valid_mask

    def estimate_geometry(self, pipe: Flux2ImagePipeline, input_video, camera_params=None):
        assert isinstance(input_video[0], Image.Image), "input_video must be a list of PIL.Image, rather than {}".format(type(input_video[0]))
        org_W, org_H = input_video[0].size

        inference_kwargs = {}
        if camera_params is not None:
            if len(camera_params) != len(input_video):
                raise ValueError(
                    f"Expected {len(input_video)} camera records, "
                    f"received {len(camera_params)}."
                )
            known_intrinsics = []
            known_extrinsics = []
            for camera in camera_params:
                K = np.asarray(camera["K"], dtype=np.float64)
                R = np.asarray(camera["R"], dtype=np.float64)
                T = np.asarray(camera["T"], dtype=np.float64)
                if K.shape != (3, 3) or R.shape != (3, 3) or T.shape != (3,):
                    raise ValueError(
                        "Each camera record must contain K(3,3), R(3,3), "
                        f"and T(3,), got {K.shape}, {R.shape}, {T.shape}."
                    )
                extrinsic = np.eye(4, dtype=np.float64)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = T
                known_intrinsics.append(K)
                known_extrinsics.append(extrinsic)
            inference_kwargs = {
                "intrinsics": np.stack(known_intrinsics),
                "extrinsics": np.stack(known_extrinsics),
                "align_to_input_ext_scale": True,
            }

        with torch.no_grad():
            prediction = pipe.da3.inference(
                input_video,
                **inference_kwargs,
            )
        # pipe.da3.to("cpu")
        # torch.cuda.empty_cache()

        imgs = np.split(prediction.processed_images, prediction.processed_images.shape[0], axis=0)
        imgs = [x[0] for x in imgs]
        new_H, new_W = imgs[0].shape[:2]
        sx = org_W / new_W
        sy = org_H / new_H

        depths = np.split(prediction.depth, prediction.depth.shape[0], axis=0)
        depths = [x[0] for x in depths]

        if camera_params is not None:
            if len(camera_params) != len(input_video):
                raise ValueError(
                    f"Expected {len(input_video)} camera records, "
                    f"received {len(camera_params)}."
                )
            res = []
            for depth, image, camera in zip(depths, input_video, camera_params):
                image_w, image_h = image.size
                depth = cv2.resize(
                    depth,
                    (image_w, image_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                K = np.asarray(camera["K"], dtype=np.float64)
                R = np.asarray(camera["R"], dtype=np.float64)
                T = np.asarray(camera["T"], dtype=np.float64)
                if K.shape != (3, 3) or R.shape != (3, 3) or T.shape != (3,):
                    raise ValueError(
                        "Each camera record must contain K(3,3), R(3,3), "
                        f"and T(3,), got {K.shape}, {R.shape}, {T.shape}."
                    )
                res.append({"depth": depth, "K": K, "R": R, "T": T})
            return res

        depths = [
            cv2.resize(depth, image.size, interpolation=cv2.INTER_LINEAR)
            for depth, image in zip(depths, input_video)
        ]

        extrinsics = np.asarray(prediction.extrinsics)
        if extrinsics.shape[0] != len(input_video):
            raise ValueError(
                f"DA3 returned {extrinsics.shape[0]} extrinsics for "
                f"{len(input_video)} input views."
            )
        Rs = [extrinsic[:, :3] for extrinsic in extrinsics]
        Ts = [extrinsic[:, 3] for extrinsic in extrinsics]

        # Preserve DA3's per-view calibration. Reusing intrinsics[0] for every
        # view makes rotation/focal errors compensate for view-specific K.
        intrinsics = np.asarray(prediction.intrinsics)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None]
        if intrinsics.shape[0] != len(input_video):
            raise ValueError(
                f"DA3 returned {intrinsics.shape[0]} intrinsics for "
                f"{len(input_video)} input views."
            )
        Ks = []
        for K, processed_image, image in zip(intrinsics, imgs, input_video):
            image_w, image_h = image.size
            processed_h, processed_w = processed_image.shape[:2]
            scaled_K = np.asarray(K, dtype=np.float64).copy()
            scaled_K[0, :] *= image_w / processed_w
            scaled_K[1, :] *= image_h / processed_h
            Ks.append(scaled_K)

        res = []
        for d, K, R, T in zip(depths, Ks, Rs, Ts):
            res.append({'depth': d, 'K': K, 'R': R, 'T': T})
        return res
    
    def transform_to_world(self, points_local, R, T):
        """Step 2: Local 3D -> World 3D"""
        return (points_local - T) @ R

    def transform_world_to_local(self, points_world, R, T):
        """
        Step 3: World 3D -> Local 3D (Target Camera)
        基于你 transform_to_world: (P - T) @ R 推导的逆变换
        """
        return (points_world @ R.T) + T
    
    def get_coord_maps(self, da3_res, input_video, source_idx=0, target_indices=None):
        coord_maps = []
        coord_maps_depth = []

        if target_indices is None:
            target_indices = range(source_idx + 1, len(input_video))

        first_cam = da3_res[source_idx]
        source_image = input_video[source_idx]
        W_in, H_in = source_image.size 

        # 1. 估算深度并从第一帧 Unproject
        median_depth = np.median(first_cam['depth'][first_cam['depth'] > 0])
        if np.isnan(median_depth) or median_depth < 0: 
            print("Warning: Median depth is nan or less than 0, set to 3.0!")
            median_depth = 3
            
        flat_depth = np.full_like(first_cam['depth'], median_depth)
        pts_local_0, clrs, u_src, v_src = self.unproject_points(np.array(source_image), flat_depth, first_cam['K'])
        pts_local_0_depth, clrs2, u_src2, v_src2 = self.unproject_points(np.array(source_image), first_cam['depth'], first_cam['K'])

        # 2. 将第一帧的局部 3D 点统一转换到 World 空间 (只需要做一次)
        # 注意：这里必须用 first_cam 的 R 和 T！
        pts_world_flat = self.transform_to_world(pts_local_0, first_cam['R'], first_cam['T'])
        pts_world_dense = self.transform_to_world(pts_local_0_depth, first_cam['R'], first_cam['T'])

        for tgt_idx in target_indices:
            tgt_cam = da3_res[tgt_idx]
            W_tgt, H_tgt = input_video[tgt_idx].size
            # 3. 将 World 3D 点转换到目标相机 N 的局部 3D 空间
            pts_local_tgt_flat = self.transform_world_to_local(pts_world_flat, tgt_cam['R'], tgt_cam['T'])
            pts_local_tgt_dense = self.transform_world_to_local(pts_world_dense, tgt_cam['R'], tgt_cam['T'])

            # 4. 投影到目标相机的 2D 像素平面
            # 这里不需要 canvas_w, canvas_h，直接传入目标图像尺寸
            uv_final, valid_proj_mask = self.project_to_target_camera(pts_local_tgt_flat, tgt_cam['K'], W_tgt, H_tgt)
            uv_final_dense, valid_proj_mask_dense = self.project_to_target_camera(pts_local_tgt_dense, tgt_cam['K'], W_tgt, H_tgt)

            # ====== 处理 Flat Depth Map ======
            u_src_valid = u_src[valid_proj_mask]
            v_src_valid = v_src[valid_proj_mask]
            uv_valid_proj = uv_final[valid_proj_mask]

            coord_map = np.full((H_in, W_in, 2), -1.0, dtype=np.float32)
            # coord_map 记录的是：第一帧的 (y, x) 在当前帧的目标坐标
            coord_map[v_src_valid, u_src_valid] = uv_valid_proj
            coord_maps.append(coord_map)
            
            # ====== 处理 Dense Depth Map ======
            u_src_valid2 = u_src2[valid_proj_mask_dense]
            v_src_valid2 = v_src2[valid_proj_mask_dense]
            uv_valid_proj2 = uv_final_dense[valid_proj_mask_dense]

            coord_map_depth = np.full((H_in, W_in, 2), -1.0, dtype=np.float32)
            coord_map_depth[v_src_valid2, u_src_valid2] = uv_valid_proj2
            coord_maps_depth.append(coord_map_depth)
        
        # highres_coord_maps_depth = [map.copy() for map in coord_maps_depth]
        coord_maps = [cv2.resize(coord_map, (W_in // 16, H_in // 16), interpolation=cv2.INTER_LINEAR) for coord_map in coord_maps]
        coord_maps = [coord_map / 16 for coord_map in coord_maps]
        coord_maps_depth = [cv2.resize(coord_map, (W_in // 16, H_in // 16), interpolation=cv2.INTER_LINEAR) for coord_map in coord_maps_depth]
        coord_maps_depth = [coord_map / 16 for coord_map in coord_maps_depth]
        # Match the synthetic stitching datasets: invalid/out-of-canvas tokens
        # use a far-away RoPE sentinel instead of clustering near (-1 / 16).
        for coord_map in coord_maps:
            coord_map[coord_map < 0] = 1000
        for coord_map_depth in coord_maps_depth:
            coord_map_depth[coord_map_depth < 0] = 1000

        # coord_maps_depth = coord_maps_depth[::4]
        # coord_maps = coord_maps[::4]

        return coord_maps, coord_maps_depth

    def process(self, pipe: Flux2ImagePipeline, input_image, edit_image, noise, gt_height, gt_width, coord_maps, coord_maps_depth, camera_params):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        edit_images = [edit_image] if isinstance(edit_image, Image.Image) else list(edit_image)
        projected_imgs, projected_imgs_depth = [], []
        if coord_maps is None or coord_maps_depth is None:
            pipe.load_models_to_device(['vae', 'da3'])
            coord_maps, coord_maps_depth = [], []
            input_video = edit_images + [input_image]
            da3_res = self.estimate_geometry(
                pipe,
                input_video,
                camera_params=camera_params,
            )
            input_idx = len(input_video) - 1

            for src_idx, src_img in enumerate(edit_images):
                src_coord_maps, src_coord_maps_depth = self.get_coord_maps(
                    da3_res,
                    input_video,
                    source_idx=src_idx,
                    target_indices=[input_idx],
                )
                coord_maps.append(src_coord_maps[0])
                coord_maps_depth.append(src_coord_maps_depth[0])

                # if not pipe.scheduler.training:
                # resized_coord_maps_depth = cv2.resize(src_coord_maps_depth[0] * 16, src_img.size, interpolation=cv2.INTER_LINEAR)
                # resized_coord_maps = cv2.resize(src_coord_maps[0] * 16, src_img.size, interpolation=cv2.INTER_LINEAR)
                # projected_img_depth, _ = self.forward_warp_using_map(np.array(src_img), resized_coord_maps_depth, src_img.size)
                # projected_img, _ = self.forward_warp_using_map(np.array(src_img), resized_coord_maps, src_img.size)
                # projected_imgs_depth.append(projected_img_depth)
                # projected_imgs.append(projected_img)
        else:
            pipe.load_models_to_device(['vae'])

        # for idx, img in enumerate(projected_imgs_depth):
        #     cv2.imwrite(f"test_depth_{str(idx).zfill(3)}.png", img[:, :, [2, 1, 0]])
        # for idx, img in enumerate(projected_imgs):
        #     cv2.imwrite(f"test_{str(idx).zfill(3)}.png", img[:, :, [2, 1, 0]])
        # input_image.save(f"test_depth_gt.png")
        # quit()
        # print(coord_maps_depth[0].min(), coord_maps_depth[0].max(), resized_coord_maps_depth.min(), resized_coord_maps_depth.max(), resized_coord_maps.min(), resized_coord_maps.max())

        image = pipe.preprocess_image(input_image)
        if gt_height < 1024 or gt_width < 1024:
            input_latents = pipe.vae.encode(image)
        else:
            crop_coords = self.slice_image_adaptive(gt_height, gt_width, patch_size=1024, stride=768)
            weight_map_latent = torch.zeros((gt_height // 16, gt_width // 16), device=pipe.device, dtype=torch.float32)
            input_latents = torch.zeros((1, 128, gt_height // 16, gt_width // 16), device=pipe.device, dtype=torch.float32)
            g_mask_latent = self.get_linear_mask(1024 // 16, min_weight=1e-3, device=pipe.device, dtype=torch.float32)
            g_mask_latent_bc = g_mask_latent.view(1, 1, 1024 // 16, 1024 // 16)
            for cc in crop_coords:
                ccl = [x // 16 for x in cc]
                tmp_latent = pipe.vae.encode(image[:, :, cc[1]:cc[3], cc[0]:cc[2]])
                input_latents[:, :, ccl[1]:ccl[3], ccl[0]:ccl[2]] += tmp_latent.to(device=pipe.device, dtype=torch.float32) * g_mask_latent_bc
                weight_map_latent[ccl[1]:ccl[3], ccl[0]:ccl[2]] += g_mask_latent
            input_latents = input_latents / (weight_map_latent + 1e-6)
            input_latents = input_latents.to(pipe.torch_dtype)
        input_latents = rearrange(input_latents, "B C H W -> B (H W) C")

        # pipe.load_models_to_device([])

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents, "coord_maps": coord_maps, "coord_maps_depth": coord_maps_depth}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": noise, "input_latents": input_latents, "coord_maps": coord_maps, "coord_maps_depth": coord_maps_depth, "projected_img": projected_imgs, "projected_img_depth": projected_imgs_depth}


class Flux2Unit_EditImageEmbedder(PipelineUnit):
    def __init__(self, is_training=True):
        super().__init__(
            input_params=(
                "edit_image", "edit_image_auto_resize", "coord_maps", "coord_maps_depth",
                "edit_token_rectification_mode", "edit_token_rectification_strength",
                "edit_token_rectification_min_scale", "edit_token_rectification_max_scale",
                "edit_token_rectification_patch_size", "edit_token_rectification_batch_size",
                "edit_token_rectification_transform",
                "crop_region_latent",
            ),
            output_params=("edit_latents", "edit_image_ids"),
            onload_model_names=("vae",)
        )
        self.is_training = is_training

    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height

    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return edit_image.resize((calculated_width, calculated_height))

    @staticmethod
    def _valid_coord_map(map_np, invalid_coord=1000.0):
        map_np = np.asarray(map_np)
        return (
            np.isfinite(map_np).all(axis=-1) &
            (np.abs(map_np[..., 0]) < invalid_coord) &
            (np.abs(map_np[..., 1]) < invalid_coord)
        )

    @classmethod
    def build_token_internal_rectification_grid(
        cls,
        map_np,
        strength=1.0,
        min_scale=0.25,
        max_scale=4.0,
        invalid_coord=1000.0,
    ):
        """Build a center-preserving grid for the VAE token's packed 2x2 cells.

        Flux2VAE packs a [C, 2, 2] pre-patch feature block into each final
        token. The map Jacobian describes source-token offsets in target RoPE
        space. Its inverse therefore tells us where each target-oriented
        internal sub-cell should sample in the source pre-patch feature map.
        Token centers and token indices never move.
        """
        map_np = np.asarray(map_np, dtype=np.float64)
        if map_np.ndim != 3 or map_np.shape[-1] != 2:
            raise ValueError(f"Expected coord map [H, W, 2], got {map_np.shape}.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Rectification strength must be in [0, 1], got {strength}.")
        if not 0.0 < min_scale <= max_scale:
            raise ValueError(
                f"Expected 0 < min_scale <= max_scale, got {min_scale}, {max_scale}."
            )
        height, width = map_np.shape[:2]
        prepatch_h, prepatch_w = height * 2, width * 2
        grid_y, grid_x = np.meshgrid(
            np.arange(prepatch_h, dtype=np.float64),
            np.arange(prepatch_w, dtype=np.float64),
            indexing="ij",
        )
        sample_x = grid_x.copy()
        sample_y = grid_y.copy()
        valid = cls._valid_coord_map(map_np, invalid_coord=invalid_coord)
        corrected = np.zeros((height, width), dtype=bool)
        singular_values = []
        offsets = (
            (-0.5, -0.5),
            (0.5, -0.5),
            (-0.5, 0.5),
            (0.5, 0.5),
        )

        def derivative(row, col, axis):
            if axis == 0:
                before, after = (row, col - 1), (row, col + 1)
            else:
                before, after = (row - 1, col), (row + 1, col)
            before_valid = (
                0 <= before[0] < height and 0 <= before[1] < width and
                valid[before]
            )
            after_valid = (
                0 <= after[0] < height and 0 <= after[1] < width and
                valid[after]
            )
            if before_valid and after_valid:
                return (map_np[after] - map_np[before]) * 0.5
            if after_valid:
                return map_np[after] - map_np[row, col]
            if before_valid:
                return map_np[row, col] - map_np[before]
            return None

        for row in range(height):
            for col in range(width):
                if not valid[row, col]:
                    continue
                derivative_x = derivative(row, col, axis=0)
                derivative_y = derivative(row, col, axis=1)
                if derivative_x is None or derivative_y is None:
                    continue
                jacobian = np.stack([derivative_x, derivative_y], axis=1)
                if not np.isfinite(jacobian).all() or np.linalg.det(jacobian) <= 1e-8:
                    continue
                u, values, vt = np.linalg.svd(jacobian)
                values = np.clip(values, min_scale, max_scale)
                stable_jacobian = u @ np.diag(values) @ vt
                inverse_jacobian = np.linalg.inv(stable_jacobian)
                center_x = 2.0 * col + 0.5
                center_y = 2.0 * row + 0.5
                proposed = []
                for offset_x, offset_y in offsets:
                    identity_offset = np.array([offset_x, offset_y], dtype=np.float64)
                    rectified_offset = inverse_jacobian @ identity_offset
                    source_offset = (
                        (1.0 - strength) * identity_offset +
                        strength * rectified_offset
                    )
                    proposed.append(
                        (center_x + source_offset[0], center_y + source_offset[1])
                    )
                if not all(
                    0.0 <= x <= prepatch_w - 1.0 and 0.0 <= y <= prepatch_h - 1.0
                    for x, y in proposed
                ):
                    continue
                for sub_index, (x, y) in enumerate(proposed):
                    sub_row, sub_col = divmod(sub_index, 2)
                    sample_x[2 * row + sub_row, 2 * col + sub_col] = x
                    sample_y[2 * row + sub_row, 2 * col + sub_col] = y
                corrected[row, col] = True
                singular_values.append(values)

        normalized_x = 2.0 * (sample_x + 0.5) / float(prepatch_w) - 1.0
        normalized_y = 2.0 * (sample_y + 0.5) / float(prepatch_h) - 1.0
        grid = np.stack([normalized_x, normalized_y], axis=-1).astype(np.float32)
        if singular_values:
            singular_values = np.stack(singular_values, axis=0)
            stats = {
                "corrected_tokens": int(corrected.sum()),
                "valid_tokens": int(valid.sum()),
                "min_singular": float(singular_values.min()),
                "max_singular": float(singular_values.max()),
            }
        else:
            stats = {
                "corrected_tokens": 0,
                "valid_tokens": int(valid.sum()),
                "min_singular": None,
                "max_singular": None,
            }
        return grid, corrected, stats

    @classmethod
    def rectify_token_internal_features(
        cls,
        pipe,
        latents,
        map_np,
        strength=1.0,
        min_scale=0.25,
        max_scale=4.0,
    ):
        """Rectify packed 2x2 VAE sub-cells without changing latent centers."""
        if latents.ndim != 4 or latents.shape[1] != 128:
            raise ValueError(f"Expected Flux2 edit latents [B, 128, H, W], got {latents.shape}.")
        if tuple(map_np.shape[:2]) != tuple(latents.shape[-2:]):
            raise ValueError(
                f"Coord map shape {map_np.shape[:2]} does not match latent shape "
                f"{tuple(latents.shape[-2:])}."
            )
        grid, _, stats = cls.build_token_internal_rectification_grid(
            map_np,
            strength=strength,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        vae = pipe.vae
        latent_dtype = latents.dtype
        running_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=latents.device, dtype=torch.float32
        )
        running_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1).to(
                device=latents.device, dtype=torch.float32
            ) + vae.bn.eps
        )
        denormalized = latents.to(torch.float32) * running_std + running_mean
        prepatch = rearrange(
            denormalized,
            "B (C P Q) H W -> B C (H P) (W Q)",
            P=2,
            Q=2,
        )
        grid_tensor = torch.from_numpy(grid).unsqueeze(0).to(
            device=latents.device, dtype=torch.float32
        )
        grid_tensor = grid_tensor.expand(latents.shape[0], -1, -1, -1)
        rectified_prepatch = torch.nn.functional.grid_sample(
            prepatch,
            grid_tensor,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        rectified = rearrange(
            rectified_prepatch,
            "B C (H P) (W Q) -> B (C P Q) H W",
            P=2,
            Q=2,
        )
        rectified = (rectified - running_mean) / running_std
        return rectified.to(latent_dtype), stats

    @classmethod
    def rectify_tokens_from_rgb_patches(
        cls,
        pipe,
        image,
        latents,
        map_np,
        map_depth_np,
        crop_region_latent,
        patch_size=48,
        batch_size=64,
        strength=1.0,
        min_scale=0.25,
        max_scale=4.0,
        transform_mode="rotation_only",
        invalid_coord=1000.0,
        identity_threshold=1e-3,
    ):
        """Replace selected edit tokens with centers of rectified RGB patch encodes.

        Each selected source token keeps its original index and RoPE coordinates.
        By default, the polar rotation of the local Jacobian samples a
        target-oriented RGB patch around that token's source-image center;
        scale and shear are discarded. Encoding a 48x48 patch produces a 3x3
        Flux2 latent, whose center token replaces the original full-image token.
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected one preprocessed image [1, C, H, W], got {image.shape}.")
        if latents.ndim != 4 or latents.shape[0] != 1 or latents.shape[1] != 128:
            raise ValueError(f"Expected Flux2 edit latents [1, 128, H, W], got {latents.shape}.")
        if patch_size <= 0 or patch_size % 16 != 0 or (patch_size // 16) % 2 != 1:
            raise ValueError(
                "RGB rectification patch_size must be a positive multiple of 16 "
                f"with an odd latent size; got {patch_size}."
            )
        if batch_size <= 0:
            raise ValueError(f"RGB rectification batch_size must be positive, got {batch_size}.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Rectification strength must be in [0, 1], got {strength}.")
        if not 0.0 < min_scale <= max_scale:
            raise ValueError(
                f"Expected 0 < min_scale <= max_scale, got {min_scale}, {max_scale}."
            )
        if transform_mode not in {"rotation_only", "full_jacobian"}:
            raise ValueError(
                "RGB patch transform_mode must be 'rotation_only' or "
                f"'full_jacobian', got {transform_mode}."
            )

        map_np = np.asarray(map_np, dtype=np.float64)
        map_depth_np = np.asarray(map_depth_np, dtype=np.float64)
        latent_height, latent_width = latents.shape[-2:]
        expected_map_shape = (latent_height, latent_width, 2)
        if map_np.shape != expected_map_shape or map_depth_np.shape != expected_map_shape:
            raise ValueError(
                "Coord maps must match the edit latent grid: expected "
                f"{expected_map_shape}, got {map_np.shape} and {map_depth_np.shape}."
            )

        valid_appearance = cls._valid_coord_map(map_np, invalid_coord=invalid_coord)
        valid_depth = cls._valid_coord_map(map_depth_np, invalid_coord=invalid_coord)
        candidates = valid_appearance & valid_depth
        if crop_region_latent is not None:
            if len(crop_region_latent) != 4:
                raise ValueError(
                    "crop_region_latent must be [x_start, y_start, x_end, y_end], "
                    f"got {crop_region_latent}."
                )
            x_start, y_start, x_end, y_end = crop_region_latent
            candidates &= (
                (map_depth_np[..., 0] >= x_start) &
                (map_depth_np[..., 0] < x_end) &
                (map_depth_np[..., 1] >= y_start) &
                (map_depth_np[..., 1] < y_end)
            )

        image_height, image_width = image.shape[-2:]
        offsets_1d = np.arange(patch_size, dtype=np.float64) - (patch_size - 1) * 0.5
        offset_y, offset_x = np.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        target_offsets = np.stack([offset_x, offset_y], axis=0).reshape(2, -1)
        identity = np.eye(2, dtype=np.float64)
        patch_grids, token_rows, token_cols, singular_values = [], [], [], []
        rotation_degrees = []
        skipped_invalid_jacobian = 0
        skipped_identity = 0
        skipped_boundary = 0

        def derivative(row, col, axis):
            if axis == 0:
                before, after = (row, col - 1), (row, col + 1)
            else:
                before, after = (row - 1, col), (row + 1, col)
            before_valid = (
                0 <= before[0] < latent_height and 0 <= before[1] < latent_width and
                valid_appearance[before]
            )
            after_valid = (
                0 <= after[0] < latent_height and 0 <= after[1] < latent_width and
                valid_appearance[after]
            )
            if before_valid and after_valid:
                return (map_np[after] - map_np[before]) * 0.5
            if after_valid:
                return map_np[after] - map_np[row, col]
            if before_valid:
                return map_np[row, col] - map_np[before]
            return None

        for row, col in np.argwhere(candidates):
            derivative_x = derivative(row, col, axis=0)
            derivative_y = derivative(row, col, axis=1)
            if derivative_x is None or derivative_y is None:
                skipped_invalid_jacobian += 1
                continue
            jacobian = np.stack([derivative_x, derivative_y], axis=1)
            if not np.isfinite(jacobian).all() or np.linalg.det(jacobian) <= 1e-8:
                skipped_invalid_jacobian += 1
                continue
            u, values, vt = np.linalg.svd(jacobian)
            rotation = u @ vt
            rotation_angle = math.atan2(rotation[1, 0], rotation[0, 0])
            if transform_mode == "rotation_only":
                # J maps source-patch offsets to target-canvas offsets. Backward
                # RGB sampling therefore uses the inverse 2D rotation. Angle
                # interpolation keeps partial strengths rotation-only too.
                sampling_angle = -strength * rotation_angle
                cos_angle = math.cos(sampling_angle)
                sin_angle = math.sin(sampling_angle)
                sampling_transform = np.array(
                    [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
                    dtype=np.float64,
                )
                stable_values = values
            else:
                stable_values = np.clip(values, min_scale, max_scale)
                stable_jacobian = u @ np.diag(stable_values) @ vt
                inverse_jacobian = np.linalg.inv(stable_jacobian)
                sampling_transform = (
                    (1.0 - strength) * identity + strength * inverse_jacobian
                )
            if np.linalg.norm(sampling_transform - identity) <= identity_threshold:
                skipped_identity += 1
                continue
            source_offsets = sampling_transform @ target_offsets
            center_x = (float(col) + 0.5) * 16.0 - 0.5
            center_y = (float(row) + 0.5) * 16.0 - 0.5
            sample_x = center_x + source_offsets[0]
            sample_y = center_y + source_offsets[1]
            if (
                sample_x.min() < 0.0 or sample_x.max() > image_width - 1.0 or
                sample_y.min() < 0.0 or sample_y.max() > image_height - 1.0
            ):
                skipped_boundary += 1
                continue
            normalized_x = 2.0 * (sample_x + 0.5) / float(image_width) - 1.0
            normalized_y = 2.0 * (sample_y + 0.5) / float(image_height) - 1.0
            patch_grids.append(
                np.stack([normalized_x, normalized_y], axis=-1).reshape(
                    patch_size, patch_size, 2
                ).astype(np.float32)
            )
            token_rows.append(int(row))
            token_cols.append(int(col))
            singular_values.append(stable_values)
            rotation_degrees.append(math.degrees(rotation_angle))

        rectified = latents.clone()
        center_latent_index = (patch_size // 16) // 2
        for start in range(0, len(patch_grids), batch_size):
            end = min(start + batch_size, len(patch_grids))
            grid = torch.from_numpy(np.stack(patch_grids[start:end])).to(
                device=image.device, dtype=torch.float32
            )
            image_batch = image.to(torch.float32).expand(end - start, -1, -1, -1)
            rgb_patches = torch.nn.functional.grid_sample(
                image_batch,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).to(dtype=image.dtype)
            patch_latents = pipe.vae.encode(rgb_patches)
            expected_patch_latent_size = patch_size // 16
            if patch_latents.shape[-2:] != (
                expected_patch_latent_size,
                expected_patch_latent_size,
            ):
                raise RuntimeError(
                    "Unexpected VAE patch latent shape: expected spatial size "
                    f"{expected_patch_latent_size}x{expected_patch_latent_size}, "
                    f"got {tuple(patch_latents.shape[-2:])}."
                )
            center_tokens = patch_latents[
                :, :, center_latent_index, center_latent_index
            ].to(dtype=rectified.dtype)
            rows = torch.as_tensor(token_rows[start:end], device=rectified.device)
            cols = torch.as_tensor(token_cols[start:end], device=rectified.device)
            rectified[0, :, rows, cols] = center_tokens.transpose(0, 1)

        if singular_values:
            singular_values_np = np.stack(singular_values, axis=0)
            min_singular = float(singular_values_np.min())
            max_singular = float(singular_values_np.max())
        else:
            min_singular = None
            max_singular = None
        stats = {
            "crop_candidate_tokens": int(candidates.sum()),
            "corrected_tokens": len(patch_grids),
            "skipped_invalid_jacobian": skipped_invalid_jacobian,
            "skipped_identity": skipped_identity,
            "skipped_boundary": skipped_boundary,
            "vae_patch_batches": int(math.ceil(len(patch_grids) / batch_size)),
            "transform_mode": transform_mode,
            "min_rotation_degrees": (
                float(min(rotation_degrees)) if rotation_degrees else None
            ),
            "max_rotation_degrees": (
                float(max(rotation_degrees)) if rotation_degrees else None
            ),
            "min_singular": min_singular,
            "max_singular": max_singular,
        }
        return rectified, stats

    def process_image_ids_warp(self, image_latents, coord_maps, coord_maps_depth, scale=10):
        if self.is_training:
            t_coords = [scale + scale * t for t in torch.randperm(8)[:2]]
        else:
            t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        
        for x, t, map_np, map_d_np in zip(image_latents, t_coords, coord_maps, coord_maps_depth):
            x = x.squeeze(0) # [C, H, W]
            _, h_latent, w_latent = x.shape
            
            map_tensor = torch.from_numpy(map_np).float().to(x.device) 
            map_depth_tensor = torch.from_numpy(map_d_np).float().to(x.device) 
            
            u_coords = map_tensor[..., 0].flatten() 
            v_coords = map_tensor[..., 1].flatten() 
            u_depth_coords = map_depth_tensor[..., 0].flatten() 
            v_depth_coords = map_depth_tensor[..., 1].flatten() 
            
            num_tokens = h_latent * w_latent
            
            t_col = t.repeat(num_tokens).to(x.device)
            c_col = torch.zeros(num_tokens, device=x.device)

            # 分别构造 true depth 和 avg depth 的 4 维 ID [t, h, w, c]
            ids_avg = torch.stack([t_col, v_coords, u_coords, c_col], dim=1)
            ids_true = torch.stack([t_col, v_depth_coords, u_depth_coords, c_col], dim=1)
            
            # 增加一个组别维度 dim=1，形状变为 (Num_tokens, 2, 4)
            x_ids = torch.stack([ids_true, ids_avg], dim=1)
            
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0) # (1, Total_Tokens, 2, 4)

        return image_latent_ids
    
    # def process_image_ids_training(self, image_latents, scale=10):
    #     # 1. 定义候选 ID 列表
    #     candidates = torch.arange(scale, scale + scale * 10, scale)
        
    #     image_latent_ids = []
        
    #     for x in image_latents:
    #         x = x.squeeze(0)
    #         _, height, width = x.shape

    #         # 2. 【关键修改】为当前这张图随机抽一个 ID
    #         # 随机选一个索引
    #         rand_idx = torch.randint(0, len(candidates), (1,))
    #         # 取出数值，比如 50
    #         t_val = candidates[rand_idx] 

    #         # 3. 使用 cartesian_prod (保持原有逻辑，只是 t 变了)
    #         # t_val 扩展成 tensor([50])
    #         # 结果形状: (H*W, 4)，所有行的第一列都是 50
    #         x_ids = torch.cartesian_prod(
    #             t_val.view(-1), 
    #             torch.arange(height), 
    #             torch.arange(width), 
    #             torch.arange(1)
    #         )
    #         image_latent_ids.append(x_ids)

    #     image_latent_ids = torch.cat(image_latent_ids, dim=0)
    #     image_latent_ids = image_latent_ids.unsqueeze(0)

    #     return image_latent_ids

    def process(
        self,
        pipe: Flux2ImagePipeline,
        edit_image,
        edit_image_auto_resize,
        coord_maps,
        coord_maps_depth,
        edit_token_rectification_mode,
        edit_token_rectification_strength,
        edit_token_rectification_min_scale,
        edit_token_rectification_max_scale,
        edit_token_rectification_patch_size,
        edit_token_rectification_batch_size,
        edit_token_rectification_transform,
        crop_region_latent,
    ):
        if edit_image is None:
            return {}
        edit_token_rectification_mode = edit_token_rectification_mode or "none"
        if edit_token_rectification_mode not in {"none", "local_jacobian", "rgb_patch"}:
            raise ValueError(
                f"Unsupported edit_token_rectification_mode: "
                f"{edit_token_rectification_mode}."
            )
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]
        resized_edit_image, edit_latents, valid_masks = [], [], []
        for image in edit_image:
            # Preprocess
            # if edit_image_auto_resize is None or edit_image_auto_resize:
            #     image = self.edit_image_auto_resize(image)
            resized_edit_image.append(image)
            # Encode
            image = pipe.preprocess_image(image)
            # valid_mask = (image > -1).type(torch.float32).mean(dim=1, keepdim=True)
            # valid_mask = torch.nn.functional.interpolate(valid_mask, scale_factor=1/16)
            # valid_mask = rearrange(valid_mask, "B C H W -> B 1 (H W) C")
            # valid_masks.append(valid_mask)
            latents = pipe.vae.encode(image)
            if edit_token_rectification_mode == "local_jacobian":
                map_index = len(edit_latents)
                latents, rectification_stats = self.rectify_token_internal_features(
                    pipe,
                    latents,
                    coord_maps[map_index],
                    strength=float(edit_token_rectification_strength),
                    min_scale=float(edit_token_rectification_min_scale),
                    max_scale=float(edit_token_rectification_max_scale),
                    transform_mode=edit_token_rectification_transform,
                )
                print(
                    f"edit view {map_index} token-internal rectification:",
                    rectification_stats,
                )
            elif edit_token_rectification_mode == "rgb_patch":
                map_index = len(edit_latents)
                latents, rectification_stats = self.rectify_tokens_from_rgb_patches(
                    pipe,
                    image,
                    latents,
                    coord_maps[map_index],
                    coord_maps_depth[map_index],
                    crop_region_latent=crop_region_latent,
                    patch_size=int(edit_token_rectification_patch_size),
                    batch_size=int(edit_token_rectification_batch_size),
                    strength=float(edit_token_rectification_strength),
                    min_scale=float(edit_token_rectification_min_scale),
                    max_scale=float(edit_token_rectification_max_scale),
                )
                print(
                    f"edit view {map_index} 48x48-RGB token rectification:",
                    rectification_stats,
                )
            edit_latents.append(latents)
            edit_image_ids = self.process_image_ids_warp(edit_latents, coord_maps, coord_maps_depth).to(pipe.device)
        edit_latents = torch.concat([rearrange(latents, "B C H W -> B (H W) C") for latents in edit_latents], dim=1)
        return {"edit_latents": edit_latents, "edit_image_ids": edit_image_ids}


class Flux2Unit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("gt_height", "gt_width"),
            output_params=("image_ids",),
        )

    def prepare_latent_ids(self, h_latent, w_latent):
        # 1. 生成基础的 H 和 W 序列
        h_range = torch.arange(h_latent)
        w_range = torch.arange(w_latent)

        # 2. 生成标准的 2D 网格坐标 [H*W, 2]
        grid = torch.cartesian_prod(h_range, w_range) 
        h_coords = grid[:, 0]
        w_coords = grid[:, 1]
        num_tokens = h_coords.shape[0]

        # 3. 构造 4 维坐标列: [t, h, w, c]
        # Dim 0: View ID / Time (GT 画布统一设为 0)
        t_col = torch.zeros(num_tokens)
        
        # Dim 3: Layer/Reserved (统一设为 0)
        c_col = torch.zeros(num_tokens)

        # 4. 核心逻辑：对于目标画布，True和Avg的坐标是完全等价的
        ids_true = torch.stack([t_col, h_coords, w_coords, c_col], dim=1)
        ids_avg = torch.stack([t_col, h_coords, w_coords, c_col], dim=1)

        # 5. 使用 stack 组合成 [H*W, 2, 4]，与 edit_image_ids 形状完美对齐
        latent_ids = torch.stack([ids_true, ids_avg], dim=1)

        # 6. 扩充 batch 维度并返回 [1, H*W, 2, 4]
        return latent_ids.unsqueeze(0)

    def process(self, pipe: Flux2ImagePipeline, gt_height, gt_width):
        # 注意：确认你的 VAE 缩放比例，如果是 16 则用 // 16
        image_ids = self.prepare_latent_ids(gt_height // 16, gt_width // 16).to(pipe.device)
        return {"image_ids": image_ids}


def select_topk_tokens_with_rope(features, score_map_list, full_rope, k=2):
    """
    features: [B, m, N, C]
    score_map_list: list of [B, 1, N, 1]
    full_rope: [m*N, D]
    """
    scores = torch.concat(score_map_list, dim=1) 
    B, m, N, C = features.shape
    rope_dim = full_rope.shape[-1]
    rope_reshaped = full_rope.view(1, m, N, rope_dim).expand(B, -1, -1, -1) # [B, m, N, D]
    _, topk_indices = torch.topk(scores, k=k, dim=1)
    feat_indices = topk_indices.expand(-1, -1, -1, C) # [B, k, N, C]
    selected_features = torch.gather(features, dim=1, index=feat_indices) # [B, k, N, C]
    selected_features = rearrange(selected_features, "B k N C -> B (k N) C")
    rope_indices = topk_indices.expand(-1, -1, -1, rope_dim) # [B, k, N, D]
    selected_rope = torch.gather(rope_reshaped, dim=1, index=rope_indices) # [B, k, N, D]
    selected_rope = rearrange(selected_rope, "B k N D -> B (k N) D") # [B, kN, D]
    return selected_features, selected_rope

def model_fn_flux2(
    dit: Flux2DiT,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    edit_latents=None,
    edit_image_ids=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    image_seq_len = latents.shape[1]
    if edit_latents is not None:
        image_seq_len = latents.shape[1]
        latents = torch.concat([latents, edit_latents], dim=1)
        image_ids = torch.concat([image_ids, edit_image_ids], dim=1)
    embedded_guidance = torch.tensor([embedded_guidance], device=latents.device)
    model_output = dit(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=embedded_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    model_output = model_output[:, :image_seq_len]
    return model_output
