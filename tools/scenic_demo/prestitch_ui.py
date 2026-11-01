#!/usr/bin/env python3
"""Interactive, memory-only DA3/Pi3X pre-stitching demo.

The UI deliberately keeps the prediction and geometry in a server-side
``gr.State`` object.  DA3 is run once per uploaded frame set; projection and
camera controls only re-render the in-memory views.  No prediction cache or
rendered image is written unless a caller explicitly adds an export action.

Run from the repository root with::

    python tools/scenic_demo/prestitch_ui.py --device cuda

Gradio is an optional dependency because the geometry module remains usable in
headless environments and in the existing test suite.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:  # Support both ``python -m tools.scenic_demo...`` and direct execution.
    from .da3_prestitch import (
        GENERAL_PANINI_BOTTOM_SQUEEZE,
        GENERAL_PANINI_COMPRESSION,
        GENERAL_PANINI_TOP_SQUEEZE,
        PANINI_DISTANCE,
        PROJECTION_MODES,
        REFERENCE_MODES,
        ViewGeometry,
        _canvas_transform,
        _resize_valid_coord_map,
        load_images,
        prepare_views,
        prediction_to_arrays,
        project_views,
        reference_frame,
        reference_camera,
        render_prestitch_memory,
        target_camera_components,
        target_camera_focal_y,
        unproject_views,
    )
except ImportError:  # pragma: no cover - exercised by the CLI invocation.
    from da3_prestitch import (  # type: ignore
        GENERAL_PANINI_BOTTOM_SQUEEZE,
        GENERAL_PANINI_COMPRESSION,
        GENERAL_PANINI_TOP_SQUEEZE,
        PANINI_DISTANCE,
        PROJECTION_MODES,
        REFERENCE_MODES,
        ViewGeometry,
        _canvas_transform,
        _resize_valid_coord_map,
        load_images,
        prepare_views,
        prediction_to_arrays,
        project_views,
        reference_frame,
        reference_camera,
        render_prestitch_memory,
        target_camera_components,
        target_camera_focal_y,
        unproject_views,
    )

try:
    from .colmap_geometry import load_colmap_views
except ImportError:  # pragma: no cover - direct-script invocation.
    from colmap_geometry import load_colmap_views  # type: ignore


_MODEL_LOCK = threading.Lock()
_MODEL: Any | None = None
_MODEL_KEY: tuple[str, str | None, str | None] | None = None
_PI3X_LOCK = threading.Lock()
_PI3X_MODEL: Any | None = None
_PI3X_MODEL_KEY: tuple[str, str, str | None] | None = None
_VGGT_OMEGA_LOCK = threading.Lock()
_VGGT_OMEGA_MODEL: Any | None = None
_VGGT_OMEGA_MODEL_KEY: tuple[str, str, str | None] | None = None
_FLUX_LOCK = threading.Lock()
_FLUX_PIPE: Any | None = None
_FLUX_KEY: tuple[str, str, str | None] | None = None
_FLUX_CANCEL = threading.Event()
_VIEWER_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None


def _viewer_temp_dir() -> Path:
    """Return a process-lifetime directory for transient browser GLB assets."""

    global _VIEWER_TEMP_DIR
    if _VIEWER_TEMP_DIR is None:
        _VIEWER_TEMP_DIR = tempfile.TemporaryDirectory(prefix="da3_prestitch_viewer_")
    return Path(_VIEWER_TEMP_DIR.name)


def _uploaded_paths(files: Any) -> list[str]:
    """Normalize Gradio file values and sort them in frame order."""

    if files is None:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    paths = []
    for item in files:
        value = getattr(item, "name", item)
        if value:
            paths.append(str(value))
    return sorted(paths, key=lambda value: Path(value).name)


def _get_model(model_id: str, model_cache_dir: str | None, device: str | None) -> Any:
    """Load one DA3 model per UI process/device and reuse it across sessions."""

    global _MODEL, _MODEL_KEY
    key = (model_id, model_cache_dir, device)
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return _MODEL
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError(
                "depth_anything_3 is required for a DA3 cache miss. "
                "Install DA3 before launching the UI."
            ) from exc
        kwargs = {"cache_dir": model_cache_dir} if model_cache_dir else {}
        model = DepthAnything3.from_pretrained(model_id, **kwargs)
        if device and hasattr(model, "to"):
            model = model.to(device)
        _MODEL, _MODEL_KEY = model, key
        return model


def _run_da3(
    paths: Sequence[str],
    model_id: str,
    model_cache_dir: str | None,
    device: str | None,
    process_res: int,
) -> Any:
    model = _get_model(model_id, model_cache_dir, device)
    return model.inference(list(paths), process_res=process_res)


def _get_pi3x_model(
    model_id: str,
    pi3_root: str,
    device: str | None,
) -> Any:
    """Load and cache a Pi3X model independently from the DA3 cache."""

    global _PI3X_MODEL, _PI3X_MODEL_KEY
    root = str(Path(pi3_root).expanduser())
    key = (str(model_id), root, device)
    with _PI3X_LOCK:
        if root and root not in sys.path:
            sys.path.insert(0, root)
        if _PI3X_MODEL is not None and _PI3X_MODEL_KEY == key:
            return _PI3X_MODEL
        try:
            from pi3.models.pi3x import Pi3X
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                f"Pi3X import failed from {root!r}. Set the Pi3 repository path "
                "to a checkout containing pi3/models/pi3x.py."
            ) from exc
        model = Pi3X.from_pretrained(str(model_id))
        # RGB-only inference does not need the multimodal branches and uses
        # substantially less memory.  Keep this guarded for older Pi3X builds.
        if hasattr(model, "disable_multimodal"):
            model.disable_multimodal()
        if device and hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model = model.eval()
        _PI3X_MODEL, _PI3X_MODEL_KEY = model, key
        return model


def _pi3x_input_tensor(images: Sequence[np.ndarray], pixel_limit: int = 255000):
    """Resize RGB frames to Pi3X's uniform 14-pixel patch grid.

    The original-resolution images are kept in ``ViewGeometry``; this tensor
    is only the model input.  The sizing follows Pi3's ``load_images_as_tensor``
    convention (roughly 255k pixels and dimensions divisible by 14).
    """

    import torch

    if not images:
        raise ValueError("Pi3X requires at least one image")
    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"Pi3X expects RGB HxWx3 images, got {first.shape}")
    orig_h, orig_w = first.shape[:2]
    scale = (float(pixel_limit) / max(1, orig_h * orig_w)) ** 0.5
    target_w = max(14, int(round(orig_w * scale / 14.0)) * 14)
    target_h = max(14, int(round(orig_h * scale / 14.0)) * 14)
    while target_w * target_h > pixel_limit and target_w > 14 and target_h > 14:
        if target_w / target_h >= orig_w / max(1, orig_h):
            target_w -= 14
        else:
            target_h -= 14
    tensors = []
    for image in images:
        array = np.asarray(image)
        pil = Image.fromarray(array.astype(np.uint8), mode="RGB")
        resized = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        data = np.asarray(resized, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(data).permute(2, 0, 1).contiguous())
    return torch.stack(tensors, dim=0), (target_w, target_h)


def _resize_pi3x_float(array: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.shape == (height, width):
        return array.copy()
    return np.asarray(
        Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _run_pi3x(
    images: Sequence[np.ndarray],
    paths: Sequence[str],
    model_id: str,
    pi3_root: str,
    device: str | None,
) -> list[ViewGeometry]:
    """Run Pi3X and convert its OpenCV camera-to-world output to DA3 convention."""

    import torch

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _get_pi3x_model(model_id, pi3_root, run_device)
    input_tensor, (processed_w, processed_h) = _pi3x_input_tensor(images)
    input_tensor = input_tensor.to(run_device)
    # Pi3X expects (B, N, C, H, W), while the loader returns (N, C, H, W).
    model_input = input_tensor.unsqueeze(0)
    with torch.no_grad():
        if str(run_device).startswith("cuda"):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(imgs=model_input)
        else:
            prediction = model(imgs=model_input)

    local_points = prediction["local_points"][0].detach().float().cpu().numpy()
    camera_poses = prediction["camera_poses"][0].detach().float().cpu().numpy()
    confidence = prediction.get("conf")
    if confidence is not None:
        confidence = torch.sigmoid(confidence)[0, ..., 0].detach().float().cpu().numpy()
    rays = prediction.get("rays")
    if rays is None:
        rays = torch.nn.functional.normalize(prediction["local_points"], dim=-1)
    # Pi3's helper recovers per-view pinhole intrinsics from its ray field.
    pi3_geometry = __import__("pi3.utils.geometry", fromlist=["recover_intrinsic_from_rays_d"])
    rays = rays[0].detach().float()
    with torch.no_grad():
        K_processed = pi3_geometry.recover_intrinsic_from_rays_d(
            rays, force_center_principal_point=True
        ).detach().cpu().numpy()

    if local_points.shape[0] != len(images) or camera_poses.shape[0] != len(images):
        raise ValueError(
            f"Pi3X returned {local_points.shape[0]} views for {len(images)} inputs"
        )
    views: list[ViewGeometry] = []
    for index, (image, path) in enumerate(zip(images, paths)):
        image = np.asarray(image).astype(np.uint8, copy=False)
        original_h, original_w = image.shape[:2]
        Kp = np.asarray(K_processed[index], dtype=np.float64).reshape(3, 3)
        if not np.isfinite(Kp).all() or Kp[0, 0] <= 0 or Kp[1, 1] <= 0:
            # Extremely textureless/invalid ray pixels can make Pi3's global
            # least-squares intrinsic recovery rank-deficient.  Recover a
            # stable centered pinhole fit from finite rays before falling back
            # to a conservative field-of-view prior.
            ray_np = rays[index].cpu().numpy()
            yy, xx = np.mgrid[0:processed_h, 0:processed_w]
            valid = np.isfinite(ray_np).all(axis=-1) & (np.abs(ray_np[..., 2]) > 1e-6)
            xproj = ray_np[..., 0] / np.abs(ray_np[..., 2])
            yproj = ray_np[..., 1] / np.abs(ray_np[..., 2])
            cx0, cy0 = (processed_w - 1) / 2.0, (processed_h - 1) / 2.0
            xvalid, yvalid = xproj[valid], yproj[valid]
            uvalid, vvalid = xx[valid] - cx0, yy[valid] - cy0
            fx = float(np.sum(xvalid * uvalid) / max(np.sum(xvalid * xvalid), 1e-8))
            fy = float(np.sum(yvalid * vvalid) / max(np.sum(yvalid * yvalid), 1e-8))
            if not np.isfinite(fx) or fx <= 1e-6:
                fx = 0.8 * processed_w
            if not np.isfinite(fy) or fy <= 1e-6:
                fy = 0.8 * processed_h
            Kp = np.array([[fx, 0.0, cx0], [0.0, fy, cy0], [0.0, 0.0, 1.0]], dtype=np.float64)
        K = Kp.copy()
        K[0, :] *= original_w / processed_w
        K[1, :] *= original_h / processed_h
        # Pi3X emits metric local camera points and OpenCV camera-to-world
        # matrices.  Convert to the row-vector world-to-camera convention used
        # by da3_prestitch: world = (camera - T) @ R.
        c2w = np.asarray(camera_poses[index], dtype=np.float64).reshape(4, 4)
        c2w_R = c2w[:3, :3]
        c2w_t = c2w[:3, 3]
        R = c2w_R.T
        T = -(c2w_t.reshape(1, 3) @ c2w_R).reshape(3)
        depth = _resize_pi3x_float(local_points[index, ..., 2], original_w, original_h)
        conf = None if confidence is None else _resize_pi3x_float(confidence[index], original_w, original_h)
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"Pi3X produced invalid intrinsics for view {index}: {K}")
        views.append(
            ViewGeometry(
                index=index,
                path=str(path),
                image=image,
                depth=depth,
                K=K,
                K_processed=Kp,
                R=R,
                T=T,
                processed_size=(processed_w, processed_h),
                confidence=conf,
            )
        )
    return views


def _get_vggt_omega_model(
    checkpoint_path: str,
    vggt_root: str,
    device: str | None,
) -> Any:
    """Load and cache a local VGGT-Omega checkpoint."""

    import torch

    global _VGGT_OMEGA_MODEL, _VGGT_OMEGA_MODEL_KEY
    checkpoint = str(Path(checkpoint_path).expanduser())
    root = str(Path(vggt_root).expanduser())
    key = (checkpoint, root, device)
    with _VGGT_OMEGA_LOCK:
        if root and root not in sys.path:
            sys.path.insert(0, root)
        if _VGGT_OMEGA_MODEL is not None and _VGGT_OMEGA_MODEL_KEY == key:
            return _VGGT_OMEGA_MODEL
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
        if not device or not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("VGGT-Omega inference requires an available CUDA device")
        try:
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                f"VGGT-Omega import failed from {root!r}. Set the repository path "
                "to a checkout containing vggt_omega/models."
            ) from exc
        model = VGGTOmega().eval()
        state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        _VGGT_OMEGA_MODEL, _VGGT_OMEGA_MODEL_KEY = model, key
        return model


def _run_vggt_omega(
    images: Sequence[np.ndarray],
    paths: Sequence[str],
    checkpoint_path: str,
    vggt_root: str,
    device: str | None,
    image_resolution: int = 512,
) -> list[ViewGeometry]:
    """Run VGGT-Omega and convert its predictions to ``ViewGeometry``."""

    import torch

    model = _get_vggt_omega_model(checkpoint_path, vggt_root, device)
    root = str(Path(vggt_root).expanduser())
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("VGGT-Omega utility modules could not be imported") from exc

    processed = load_and_preprocess_images(
        list(paths), image_resolution=int(image_resolution), mode="balanced", patch_size=16
    ).to(device)
    with torch.inference_mode():
        prediction = model(processed)
    if "pose_enc" not in prediction or "depth" not in prediction:
        raise ValueError("VGGT-Omega output must contain pose_enc and depth")
    proc_h, proc_w = prediction["images"].shape[-2:]
    extrinsics, intrinsics = encoding_to_camera(
        prediction["pose_enc"], (proc_h, proc_w)
    )
    depth = prediction["depth"][..., 0]
    depth_conf = prediction.get("depth_conf")
    if depth_conf is not None and depth_conf.ndim == 5:
        depth_conf = depth_conf[..., 0]

    extrinsics = extrinsics[0].detach().float().cpu().numpy()
    intrinsics = intrinsics[0].detach().float().cpu().numpy()
    depth = depth[0].detach().float().cpu().numpy()
    if depth_conf is not None:
        depth_conf = depth_conf[0].detach().float().cpu().numpy()
    if extrinsics.shape[0] != len(images) or depth.shape[0] != len(images):
        raise ValueError(
            f"VGGT-Omega returned {depth.shape[0]} views for {len(images)} inputs"
        )

    views: list[ViewGeometry] = []
    for index, (image, path) in enumerate(zip(images, paths)):
        image = np.asarray(image).astype(np.uint8, copy=False)
        original_h, original_w = image.shape[:2]
        Kp = np.asarray(intrinsics[index], dtype=np.float64).reshape(3, 3)
        K = Kp.copy()
        K[0, :] *= original_w / proc_w
        K[1, :] *= original_h / proc_h
        E = np.asarray(extrinsics[index], dtype=np.float64)
        if E.shape == (4, 4):
            E = E[:3, :]
        if E.shape != (3, 4):
            raise ValueError(f"VGGT-Omega extrinsic {index} has invalid shape {E.shape}")
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"VGGT-Omega produced invalid intrinsics for view {index}: {K}")
        views.append(
            ViewGeometry(
                index=index,
                path=str(path),
                image=image,
                depth=_resize_pi3x_float(depth[index], original_w, original_h),
                K=K,
                K_processed=Kp,
                # encoding_to_camera returns world-to-camera OpenCV extrinsics,
                # which already match da3_prestitch's R/T convention.
                R=E[:3, :3].copy(),
                T=E[:3, 3].copy(),
                processed_size=(int(proc_w), int(proc_h)),
                confidence=(
                    _resize_pi3x_float(depth_conf[index], original_w, original_h)
                    if depth_conf is not None else None
                ),
            )
        )
    return views


def _resize_views(
    views: Sequence[ViewGeometry], scale: float,
) -> list[ViewGeometry]:
    """Resize RGB/depth grids while keeping metric geometry consistent.

    Pixel-space quantities (image, depth-map sampling grid, and the four
    pinhole intrinsic entries) are resized.  Depth values themselves remain
    metric and the world-space extrinsics ``R,T`` are unchanged.
    """

    import cv2

    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("input resize scale must be a positive finite number")
    resized: list[ViewGeometry] = []
    for view in views:
        old_h, old_w = view.image.shape[:2]
        new_w = max(1, int(round(old_w * scale)))
        new_h = max(1, int(round(old_h * scale)))
        if new_w == old_w and new_h == old_h:
            resized.append(view)
            continue
        image = cv2.resize(view.image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        depth = cv2.resize(
            np.asarray(view.depth, dtype=np.float32),
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        confidence = None
        if view.confidence is not None:
            confidence = cv2.resize(
                np.asarray(view.confidence, dtype=np.float32),
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )
        K = np.asarray(view.K, dtype=np.float64).copy()
        K[0, :] *= new_w / old_w
        K[1, :] *= new_h / old_h
        resized.append(replace(
            view,
            image=np.asarray(image, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            K=K,
            K_processed=K.copy(),
            processed_size=(new_w, new_h),
            confidence=confidence,
        ))
    return resized


def _resize_input_mask(mask: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Resize a boolean input mask with nearest-neighbour semantics."""

    if mask is None:
        return np.zeros((height, width), dtype=bool)
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape == (height, width):
        return array.astype(bool, copy=True)
    image = Image.fromarray((array.astype(np.uint8) * 255))
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 127


def _editor_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    """Extract exclusion pixels (red/legacy brush colors, but not blue)."""

    return _editor_masks(value, shape)[0]


def _editor_masks(value: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode red exclusion and blue depth-edit layers, topmost paint wins.

    Explicit layers are authoritative, including entirely erased layers. Do
    not recover stale paint from the composite after the user erases it.
    """

    height, width = shape
    if value is None:
        return (np.zeros((height, width), dtype=bool), np.zeros((height, width), dtype=bool))
    layers: list[Any] = []
    if isinstance(value, dict):
        raw_layers = value.get("layers")
        if isinstance(raw_layers, (list, tuple)):
            layers.extend(raw_layers)
        # Older Gradio versions may return only a composite image.  It is
        # handled below by comparing it with the untouched background.
        composite = value.get("composite")
        background = value.get("background")
    else:
        composite, background = value, None

    mask = np.zeros((height, width), dtype=bool)
    depth_mask = np.zeros_like(mask)
    for layer in layers:
        array = np.asarray(layer)
        if array.ndim not in (2, 3):
            continue
        if array.shape[:2] != (height, width):
            array = np.asarray(Image.fromarray(array.astype(np.uint8)).resize(
                (width, height), Image.Resampling.NEAREST
            ))
        if array.ndim == 2:
            alpha = array
        elif array.ndim == 3 and array.shape[-1] >= 4:
            alpha = array[..., 3]
        elif array.ndim == 3:
            alpha = np.max(array[..., :3], axis=-1)
        else:
            continue
        painted = alpha > 8
        blue = np.zeros_like(mask)
        if array.ndim == 3 and array.shape[-1] >= 3:
            blue = (array[..., 2] > array[..., 0]) & (array[..., 2] > array[..., 1])
        mask[painted] = ~blue[painted]
        depth_mask[painted] = blue[painted]

    # If no explicit painted layer is present, detect changed pixels in the
    # composite.  This keeps the feature compatible with Gradio 4/5 editor
    # payloads that omit ``layers`` for a single-layer brush.
    if not layers and composite is not None and background is not None:
        comp = np.asarray(composite)
        back = np.asarray(background)
        comp = np.asarray(Image.fromarray(comp.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST))
        back = np.asarray(Image.fromarray(back.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST))
        if comp.ndim == 3 and back.ndim == 3:
            delta = comp[..., :3].astype(np.int16) - back[..., :3].astype(np.int16)
            painted = np.any(np.abs(delta) > 8, axis=-1)
            # A translucent blue stroke increases blue and decreases red/green.
            depth_mask = painted & (delta[..., 2] >= 0) & (delta[..., 0] <= 0) & (delta[..., 1] <= 0)
            mask = painted & ~depth_mask
    return mask, depth_mask


def _apply_depth_average_masks(
    views: Sequence[ViewGeometry],
    masks: Sequence[np.ndarray] | None,
    *,
    reference_views: Sequence[ViewGeometry] | None = None,
) -> tuple[list[ViewGeometry], dict[str, int]]:
    """Flatten each 8-connected blue region using immutable native depth.

    Label BEFORE input resize so separate regions cannot accidentally merge.
    Resize labels with nearest-neighbour, then overwrite the resized depth:
    every selected pixel gets exactly its region mean, without boundary blur.
    Never modify RGB, intrinsics, extrinsics or the original depth arrays.
    """
    import cv2

    stats = {"regions": 0, "skipped": 0, "pixels": 0}
    if masks is None:
        return list(views), stats
    originals = views if reference_views is None else reference_views
    if len(masks) != len(views) or len(originals) != len(views):
        raise ValueError("Depth masks and reference views must match the input view count")
    output = []
    for view, original, mask in zip(views, originals, masks):
        source_depth = np.asarray(original.depth)
        mask = _resize_input_mask(mask, source_depth.shape[1], source_depth.shape[0])
        if not mask.any():
            output.append(view)
            continue
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        valid = mask & np.isfinite(source_depth) & (source_depth > 0)
        counts = np.bincount(labels[valid], minlength=count)
        sums = np.bincount(labels[valid], weights=source_depth[valid].astype(np.float64), minlength=count)
        usable = counts > 0
        usable[0] = False
        means = sums / np.maximum(counts, 1)
        stats["regions"] += int(usable.sum())
        stats["skipped"] += int(count - 1 - usable.sum())
        height, width = view.depth.shape
        if labels.shape != (height, width):
            labels = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(
                (width, height), Image.Resampling.NEAREST
            ))
        selected = usable[labels]
        depth = np.asarray(view.depth, dtype=np.float32).copy()
        depth[selected] = means[labels[selected]]
        stats["pixels"] += int(selected.sum())
        output.append(replace(view, depth=depth))
    return output, stats


def _invalidate_mask_render(state: dict[str, Any], *, depth_changed: bool) -> None:
    """Invalidate dependent state so Flux cannot use pre-edit coord maps."""
    state["input_mask_version"] = int(state.get("input_mask_version", 0)) + 1
    state["sample_cache_key"] = None
    state["sample_cache"] = None
    state["last_render"] = None
    state["target_rotation_optimization_cache"] = {}
    if depth_changed:
        state["depth_edit_version"] = int(state.get("depth_edit_version", 0)) + 1
        state["all_sample_cache_key"] = None
        state["all_sample_cache"] = None
        state["testtime_optimization_cache"] = {}
        state["testtime_acceptance_cache"] = {}


def _filter_samples_by_input_mask(
    samples: Sequence[tuple[np.ndarray, ...]],
    views: Sequence[ViewGeometry],
    masks: Sequence[np.ndarray] | None,
) -> list[tuple[np.ndarray, ...]]:
    """Remove masked source pixels from reusable unprojected samples."""

    if masks is None:
        return list(samples)
    filtered: list[tuple[np.ndarray, ...]] = []
    for sample, view, mask in zip(samples, views, masks):
        if not len(sample) or sample[4] is None:
            filtered.append(sample)
            continue
        source_xy = np.asarray(sample[4])
        rows = np.rint(source_xy[:, 1]).astype(np.int64)
        cols = np.rint(source_xy[:, 0]).astype(np.int64)
        mask = np.asarray(mask, dtype=bool)
        inside = (
            (rows >= 0) & (rows < mask.shape[0]) &
            (cols >= 0) & (cols < mask.shape[1])
        )
        keep = np.ones(len(source_xy), dtype=bool)
        keep[inside] = ~mask[rows[inside], cols[inside]]
        filtered.append(tuple(np.asarray(item)[keep] for item in sample))
    return filtered


def _sample_depth_at_points(depth: np.ndarray, points: np.ndarray) -> np.ndarray:
    import cv2

    map_x = points[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = points[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(
        np.asarray(depth, dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).reshape(-1).astype(np.float64)


def _sift_magsac_matches_pair(views: Sequence[ViewGeometry]) -> tuple[np.ndarray, np.ndarray]:
    """Return spatially balanced, epipolar-consistent SIFT matches for a pair."""

    import cv2

    if len(views) != 2:
        raise ValueError("a SIFT/MAGSAC match pair must contain exactly two views")
    gray = [cv2.cvtColor(np.asarray(view.image), cv2.COLOR_RGB2GRAY) for view in views]
    sift = cv2.SIFT_create()
    keypoints, descriptors = zip(*(sift.detectAndCompute(image, None) for image in gray))
    if descriptors[0] is None or descriptors[1] is None:
        raise RuntimeError("SIFT could not extract descriptors from both views")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors[0], descriptors[1], k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 8:
        raise RuntimeError(f"only {len(good)} SIFT matches passed the ratio test")
    points_0 = np.asarray([keypoints[0][match.queryIdx].pt for match in good], dtype=np.float32)
    points_1 = np.asarray([keypoints[1][match.trainIdx].pt for match in good], dtype=np.float32)
    _fundamental, inlier_mask = cv2.findFundamentalMat(
        points_0, points_1, cv2.USAC_MAGSAC, 1.5, 0.999, 10000
    )
    if inlier_mask is None:
        raise RuntimeError("USAC_MAGSAC failed to estimate a fundamental matrix")
    inliers = np.flatnonzero(np.asarray(inlier_mask).reshape(-1).astype(bool))
    if len(inliers) < 16:
        raise RuntimeError(f"only {len(inliers)} USAC_MAGSAC inliers remain")

    # Prevent a large, repetitive skyline/building band from dominating the
    # reprojection objective. Keep at most 12 matches in each cell of both
    # 8x6 image grids, preferring the lowest descriptor distances.
    counts = [np.zeros((6, 8), dtype=np.int32), np.zeros((6, 8), dtype=np.int32)]
    sizes = [(views[0].image.shape[1], views[0].image.shape[0]),
             (views[1].image.shape[1], views[1].image.shape[0])]

    def cell(point: np.ndarray, size: tuple[int, int]) -> tuple[int, int]:
        width, height = size
        return (
            min(5, max(0, int(point[1] / max(height, 1) * 6))),
            min(7, max(0, int(point[0] / max(width, 1) * 8))),
        )

    selected = []
    for index in sorted(inliers.tolist(), key=lambda item: good[item].distance):
        cell_0, cell_1 = cell(points_0[index], sizes[0]), cell(points_1[index], sizes[1])
        if counts[0][cell_0] >= 12 or counts[1][cell_1] >= 12:
            continue
        selected.append(index)
        counts[0][cell_0] += 1
        counts[1][cell_1] += 1
    if len(selected) < 16:
        raise RuntimeError(f"grid balancing retained only {len(selected)} matches")
    selected = np.asarray(selected, dtype=np.int64)
    return points_0[selected].astype(np.float64), points_1[selected].astype(np.float64)


def _sift_magsac_matches(views: Sequence[ViewGeometry]) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for the two-view matcher."""

    if len(views) != 2:
        raise ValueError("_sift_magsac_matches expects exactly two views")
    return _sift_magsac_matches_pair(views)


def _optimization_pairs(num_views: int, reference_index: int) -> list[tuple[int, int]]:
    """Choose adjacent sequence pairs plus links to the reference view."""

    if num_views < 2:
        return []
    reference_index = max(0, min(int(reference_index), num_views - 1))
    pairs = {(index, index + 1) for index in range(num_views - 1)}
    pairs.update(
        (min(reference_index, index), max(reference_index, index))
        for index in range(num_views)
        if index != reference_index
    )
    return sorted(pairs)


def _refine_multiview_camera(
    views: Sequence[ViewGeometry],
    reference_index: int,
) -> tuple[list[ViewGeometry], dict[str, Any]]:
    """Jointly refine non-reference poses over all views.

    Relative camera centers are allowed to move with a strong depth-normalized
    prior. This is important for a translating sequence such as ``shenzhen_7``;
    rotation-only refinement cannot explain its 250--560 px frame-to-frame
    displacement. The selected reference center remains fixed, and all
    updates are still gated by held-out reprojection error.
    """

    if len(views) < 2:
        raise ValueError("test-time camera optimization requires at least two views")
    try:
        import cv2
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - optional refinement dependency.
        raise RuntimeError("test-time optimization requires opencv-python and scipy") from exc

    reference_index = max(0, min(int(reference_index), len(views) - 1))
    pair_records: list[dict[str, Any]] = []
    for index_0, index_1 in _optimization_pairs(len(views), reference_index):
        try:
            points_0, points_1 = _sift_magsac_matches_pair((views[index_0], views[index_1]))
            depth_0 = _sample_depth_at_points(views[index_0].depth, points_0)
            depth_1 = _sample_depth_at_points(views[index_1].depth, points_1)
            confidence_0 = _sample_depth_at_points(
                views[index_0].confidence
                if views[index_0].confidence is not None
                else np.ones_like(views[index_0].depth),
                points_0,
            )
            confidence_1 = _sample_depth_at_points(
                views[index_1].confidence
                if views[index_1].confidence is not None
                else np.ones_like(views[index_1].depth),
                points_1,
            )
        except Exception:
            # Frames without enough overlap do not invalidate other pairs.
            continue
        valid = (
            np.isfinite(depth_0) & np.isfinite(depth_1)
            & np.isfinite(confidence_0) & np.isfinite(confidence_1)
            & (depth_0 > 1e-4) & (depth_1 > 1e-4)
        )
        if int(valid.sum()) < 16:
            continue
        pair_records.append({
            "indices": (index_0, index_1),
            "points_0": points_0[valid],
            "points_1": points_1[valid],
            "depth_0": depth_0[valid],
            "depth_1": depth_1[valid],
            "weights": np.sqrt(
                np.maximum(confidence_0[valid], 0.0)
                * np.maximum(confidence_1[valid], 0.0)
            ),
        })
    if not pair_records:
        raise RuntimeError("no view pair has at least 16 valid SIFT/MAGSAC matches")
    total_matches = sum(len(record["points_0"]) for record in pair_records)
    rng = np.random.default_rng(20260805)
    shuffled = rng.permutation(total_matches)
    num_validation = min(max(8, int(round(0.2 * total_matches))), total_matches - 8)
    validation_indices = shuffled[:num_validation]
    training_indices = shuffled[num_validation:]
    cursor = 0
    for record in pair_records:
        count = len(record["points_0"])
        scale = max(float(np.percentile(record["weights"], 90)), 1e-8)
        record["weights"] = np.clip(record["weights"] / scale, 0.05, 1.0)
        record["global_indices"] = np.arange(cursor, cursor + count, dtype=np.int64)
        cursor += count

    base_K = [np.asarray(view.K, dtype=np.float64) for view in views]
    base_R = [np.asarray(view.R, dtype=np.float64) for view in views]
    base_T = [np.asarray(view.T, dtype=np.float64) for view in views]
    centers = [-base_T[index] @ base_R[index] for index in range(len(views))]
    # Fixed scene-depth statistic used only to express camera-center updates
    # in scale-independent units.  This is not an optimized depth multiplier.
    scene_depth_scale = float(np.median(np.concatenate(
        [np.concatenate((record["depth_0"], record["depth_1"])) for record in pair_records]
    )))
    scene_depth_scale = max(scene_depth_scale, 1e-3)
    variable_views = [index for index in range(len(views)) if index != reference_index]
    rotation_slices = {
        index: slice(3 * position, 3 * position + 3)
        for position, index in enumerate(variable_views)
    }
    translation_offset = 3 * len(variable_views)
    translation_slices = {
        index: slice(translation_offset + 3 * position, translation_offset + 3 * position + 3)
        for position, index in enumerate(variable_views)
    }
    focal_position = translation_offset + 3 * len(variable_views)
    # A focal scale at the old 0.8 bound was compensating for pose/depth
    # errors on shenzhen_7 and changed the panorama scale substantially.
    # Keep this calibration correction conservative; center/rotation variables
    # explain the dominant lateral motion instead.
    # Keep focal effectively fixed and let the pose terms explain motion.
    min_focal_scale, max_focal_scale = 0.999, 1.001

    def scale_focal(K: np.ndarray, scale: float) -> np.ndarray:
        result = K.copy()
        result[0, 0] *= scale
        result[1, 1] *= scale
        return result

    def geometry(parameters: np.ndarray):
        rotations = list(base_R)
        translations = list(base_T)
        current_centers = list(centers)
        for index in variable_views:
            delta_rotation, _ = cv2.Rodrigues(parameters[rotation_slices[index]])
            rotations[index] = delta_rotation @ base_R[index]
            center = centers[index] + scene_depth_scale * parameters[translation_slices[index]]
            current_centers[index] = center
            translations[index] = -center @ rotations[index].T
        focal_scale = float(np.exp(parameters[focal_position]))
        intrinsics = [scale_focal(K, focal_scale) for K in base_K]
        return intrinsics, rotations, translations, current_centers

    # The UI target for this case is vertical-plane/avg. Add a direct plane
    # consistency term so the optimizer improves the coordinates actually
    # used by pre-stitch, not only camera-to-camera reprojection.
    target_center, target_scene_rotation = reference_frame(views, "avg")
    target_focal = float(np.median([0.5 * (K[0, 0] + K[1, 1]) for K in base_K]))
    plane_scale = target_focal / scene_depth_scale

    def unproject(points: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        return np.stack(
            [(points[:, 0] - K[0, 2]) * depth / K[0, 0],
             (points[:, 1] - K[1, 2]) * depth / K[1, 1], depth], axis=1
        )

    def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
        z = points[:, 2]
        safe_z = np.where(z > 1e-4, z, 1e-4)
        result = np.stack([K[0, 0] * points[:, 0] / safe_z + K[0, 2],
                           K[1, 1] * points[:, 1] / safe_z + K[1, 2]], axis=1)
        result[z <= 1e-4] = 1e4
        return result

    def residual(
        parameters: np.ndarray,
        global_indices: np.ndarray,
        regularize: bool,
        balance_pairs: bool = False,
        include_plane: bool = False,
    ) -> np.ndarray:
        intrinsics, rotations, translations, current_centers = geometry(parameters)
        selected = np.zeros(total_matches, dtype=bool)
        selected[global_indices] = True
        chunks = []
        for record in pair_records:
            local_indices = np.flatnonzero(selected[record["global_indices"]])
            if local_indices.size == 0:
                continue
            index_0, index_1 = record["indices"]
            points_0 = record["points_0"][local_indices]
            points_1 = record["points_1"][local_indices]
            depth_0 = record["depth_0"][local_indices]
            depth_1 = record["depth_1"][local_indices]
            match_weights = record["weights"][local_indices]
            local_0 = unproject(points_0, depth_0, intrinsics[index_0])
            world_0 = (local_0 - translations[index_0]) @ rotations[index_0]
            predicted_1 = project(
                world_0 @ rotations[index_1].T + translations[index_1], intrinsics[index_1]
            )
            local_1 = unproject(points_1, depth_1, intrinsics[index_1])
            world_1 = (local_1 - translations[index_1]) @ rotations[index_1]
            predicted_0 = project(
                world_1 @ rotations[index_0].T + translations[index_0], intrinsics[index_0]
            )
            # Interleave the two directional residuals for each match. This
            # lets ``errors`` reduce both directions to one pixel error per
            # correspondence without mixing neighboring matches.
            chunk = np.stack(
                (predicted_1 - points_1, predicted_0 - points_0), axis=1
            ).reshape(-1, 2)
            if balance_pairs:
                chunk *= np.sqrt(np.repeat(match_weights, 2))[:, None]
            if balance_pairs:
                # A pair with 250 matches must not dominate a neighboring
                # pair with 45 matches simply because it has more texture.
                # Normalize each pair to comparable least-squares mass.
                chunk *= np.sqrt(total_matches / max(len(points_0), 1))
            chunks.append(chunk)
        output = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2))
        # Directly constrain the XY coordinates on the avg vertical plane.
        plane_chunks = []
        for record in pair_records:
            local_indices = np.flatnonzero(selected[record["global_indices"]])
            if local_indices.size == 0:
                continue
            index_0, index_1 = record["indices"]
            points_0 = record["points_0"][local_indices]
            points_1 = record["points_1"][local_indices]
            depth_0 = record["depth_0"][local_indices]
            depth_1 = record["depth_1"][local_indices]
            p0 = unproject(points_0, depth_0, intrinsics[index_0])
            p1 = unproject(points_1, depth_1, intrinsics[index_1])
            world_0 = (p0 - translations[index_0]) @ rotations[index_0]
            world_1 = (p1 - translations[index_1]) @ rotations[index_1]
            plane = ((world_0 - target_center) @ target_scene_rotation
                     - (world_1 - target_center) @ target_scene_rotation)[:, :2]
            plane *= plane_scale
            if balance_pairs:
                plane *= np.sqrt(total_matches / max(len(points_0), 1))
            plane_chunks.append(plane)
        if include_plane and plane_chunks:
            output = np.concatenate([output, np.concatenate(plane_chunks, axis=0)], axis=0)
        if regularize:
            correction_centers = np.zeros((len(views), 3), dtype=np.float64)
            for index in variable_views:
                correction_centers[index] = parameters[translation_slices[index]]
            smoothness = []
            for index in range(1, len(views) - 1):
                # The source is a video sequence, so correction jitter between
                # neighboring frames is usually a sign of a bad match. Keep
                # this prior weak enough to preserve genuine motion changes.
                smoothness.append(
                    (correction_centers[index - 1]
                     - 2.0 * correction_centers[index]
                     + correction_centers[index + 1]) / 0.10
                )
            output = np.concatenate([
                output.reshape(-1),
                parameters[:translation_offset] / np.deg2rad(5.0),
                parameters[translation_offset:focal_position] / 0.05,
                [parameters[focal_position] / np.log(1.05)],
                np.concatenate(smoothness) if smoothness else np.empty(0),
            ])
        return output.reshape(-1)

    def errors(parameters: np.ndarray, indices: np.ndarray) -> np.ndarray:
        # Each match contributes two directional 2D residuals. Reduce those
        # into one pixel error per match before taking medians.
        values = residual(parameters, indices, False).reshape(-1, 2)
        return np.linalg.norm(values.reshape(-1, 2, 2), axis=2).mean(axis=1)

    initial = np.zeros(focal_position + 1, dtype=np.float64)
    optimized = least_squares(
        lambda value: residual(
            value, training_indices, True, balance_pairs=True, include_plane=True
        ),
        initial,
        bounds=(
            np.array(
                [-np.deg2rad(5)] * translation_offset
                + [-0.5] * (focal_position - translation_offset)
                + [np.log(min_focal_scale)]
            ),
            np.array(
                [np.deg2rad(5)] * translation_offset
                + [0.5] * (focal_position - translation_offset)
                + [np.log(max_focal_scale)]
            ),
        ),
        loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=300,
    )
    initial_train, optimized_train = errors(initial, training_indices), errors(optimized.x, training_indices)
    initial_valid, optimized_valid = errors(initial, validation_indices), errors(optimized.x, validation_indices)
    accepted = bool(
        optimized.success
        and np.median(optimized_train) < np.median(initial_train)
        and np.median(optimized_valid) < np.median(initial_valid)
    )
    # Keep the candidate even when strict validation rejects it; force-apply
    # can select it from the cache without running optimization a second time.
    output = list(views)
    if optimized.success:
        intrinsics, rotations, translations, _ = geometry(optimized.x)
        for index in variable_views:
            output[index] = replace(
                views[index], K=intrinsics[index], K_processed=intrinsics[index].copy(),
                R=rotations[index], T=translations[index],
            )
        output[reference_index] = replace(
            views[reference_index], K=intrinsics[reference_index],
            K_processed=intrinsics[reference_index].copy(),
        )
    normalized_translation_shift = float(max(
        np.linalg.norm(optimized.x[translation_slices[index]])
        for index in variable_views
    ))
    stats = {
        # The renderer adds a second, projection-specific dense gate before
        # this sparse candidate is allowed to replace the original geometry.
        "sparse_accepted": accepted,
        "accepted": accepted,
        "optimizer_success": bool(optimized.success),
        "num_matches": int(total_matches),
        "num_pairs": int(len(pair_records)),
        "num_views": int(len(views)),
        "initial_median_px": float(np.median(errors(initial, np.arange(total_matches)))),
        "optimized_median_px": float(np.median(errors(optimized.x, np.arange(total_matches)))),
        "initial_heldout_median_px": float(np.median(initial_valid)),
        "optimized_heldout_median_px": float(np.median(optimized_valid)),
        "rotation_degrees": float(max(
            np.linalg.norm(optimized.x[rotation_slices[index]]) for index in variable_views
        ) * 180.0 / np.pi),
        "translation_shift": normalized_translation_shift * scene_depth_scale,
        "translation_shift_depth_ratio": normalized_translation_shift,
        "depth_scale": 1.0,
        "focal_scale": float(np.exp(optimized.x[focal_position])),
    }
    return output, stats


def _refine_two_view_camera(
    views: Sequence[ViewGeometry],
    reference_index: int,
) -> tuple[list[ViewGeometry], dict[str, Any]]:
    """Backward-compatible wrapper; the multiview solver covers two views."""

    if len(views) != 2:
        raise ValueError("_refine_two_view_camera expects exactly two views")
    return _refine_multiview_camera(views, reference_index)


def _coarse_tto_render_metrics(
    views: Sequence[ViewGeometry],
    mode: str,
    reference: str,
    filter_canvas_outliers: bool,
    visibility_aware: bool,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
    canvas_bounds_quantile: float = 0.001,
) -> dict[str, Any]:
    """Render a bounded low-resolution diagnostic for the TTO dense gate."""

    total_pixels = sum(int(view.image.shape[0] * view.image.shape[1]) for view in views)
    stride = max(2, int(np.ceil(np.sqrt(total_pixels / 1_200_000.0))))
    samples = unproject_views(views, stride)
    target_camera = _edit_target_camera(
        views, reference, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    _image, _heatmap, metrics = render_prestitch_memory(
        views,
        mode=mode,
        reference_mode=reference,
        render_stride=stride,
        bounds_quantile=canvas_bounds_quantile if filter_canvas_outliers else 0.0,
        padding=0,
        max_canvas_pixels=400_000,
        canvas_size=None,
        unprojected=samples,
        target_camera=target_camera,
        visibility_aware=bool(visibility_aware),
        return_coord_maps=False,
        return_heatmap=False,
        panini_distance=panini_distance,
        general_panini_compression=general_panini_compression,
        general_panini_top=general_panini_top,
        general_panini_bottom=general_panini_bottom,
    )
    metrics["dense_gate_stride"] = stride
    return metrics


def _apply_tto_acceptance(
    stats: dict[str, Any],
    before_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Combine sparse validation, dense-render quality, and pose-step limits."""

    result = dict(stats)
    sparse_accepted = bool(result.get("sparse_accepted", result.get("accepted", False)))
    before_coverage = float(before_metrics["canvas_coverage_ratio"])
    candidate_coverage = float(candidate_metrics["canvas_coverage_ratio"])
    before_score = float(before_metrics["prestitch_score"])
    candidate_score = float(candidate_metrics["prestitch_score"])
    before_disagreement = float(before_metrics["overlap_color_disagreement"])
    candidate_disagreement = float(candidate_metrics["overlap_color_disagreement"])
    coverage_drop = before_coverage - candidate_coverage
    score_drop = before_score - candidate_score
    disagreement_limit = max(
        before_disagreement + 0.01,
        before_disagreement * 1.25,
    )
    dense_checks = {
        "coverage_drop_le_0.025": bool(coverage_drop <= 0.025),
        "score_drop_le_2.0": bool(score_drop <= 2.0),
        "disagreement_within_limit": bool(candidate_disagreement <= disagreement_limit),
    }
    rotation_degrees = float(result.get("rotation_degrees", float("inf")))
    center_shift_ratio = float(
        result.get("translation_shift_depth_ratio", result.get("translation_shift", float("inf")))
    )
    pose_checks = {
        "rotation_le_6deg": bool(rotation_degrees <= 6.0),
        "center_shift_depth_ratio_le_0.10": bool(center_shift_ratio <= 0.10),
    }
    dense_accepted = all(dense_checks.values())
    pose_accepted = all(pose_checks.values())
    accepted = bool(
        result.get("optimizer_success")
        and sparse_accepted
        and dense_accepted
        and pose_accepted
    )
    reasons = []
    if not sparse_accepted:
        reasons.append("sparse held-out gate rejected")
    reasons.extend(name for name, passed in dense_checks.items() if not passed)
    reasons.extend(name for name, passed in pose_checks.items() if not passed)
    result.update({
        "accepted": accepted,
        "sparse_accepted": sparse_accepted,
        "dense_accepted": dense_accepted,
        "pose_step_accepted": pose_accepted,
        "acceptance_reasons": reasons,
        "dense_initial_coverage": before_coverage,
        "dense_candidate_coverage": candidate_coverage,
        "dense_coverage_drop": coverage_drop,
        "dense_initial_hole_ratio": float(before_metrics["canvas_hole_ratio"]),
        "dense_candidate_hole_ratio": float(candidate_metrics["canvas_hole_ratio"]),
        "dense_initial_score": before_score,
        "dense_candidate_score": candidate_score,
        "dense_score_drop": score_drop,
        "dense_initial_disagreement": before_disagreement,
        "dense_candidate_disagreement": candidate_disagreement,
        "dense_disagreement_limit": disagreement_limit,
        "dense_gate_stride": int(before_metrics.get("dense_gate_stride", 1)),
        "acceptance_thresholds": {
            "max_coverage_drop": 0.025,
            "max_score_drop": 2.0,
            "max_disagreement_absolute_increase": 0.01,
            "max_disagreement_relative_factor": 1.25,
            "max_rotation_degrees": 6.0,
            "max_center_shift_depth_ratio": 0.10,
        },
    })
    return result


def _testtime_optimize_views(
    state: dict[str, Any],
    reference: str,
    force_apply: bool = False,
    mode: str = "vertical_plane",
    filter_canvas_outliers: bool = True,
    visibility_aware: bool = False,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
    canvas_bounds_quantile: float = 0.001,
) -> tuple[list[ViewGeometry], str]:
    """Run TTO and apply it only after sparse, dense, and pose-step gates."""

    reference_index = 0 if reference == "first" else len(state["da3_views"]) // 2
    cache = state.setdefault("testtime_optimization_cache", {})
    cache_key = (
        int(reference_index),
        round(float(state.get("_active_resize_scale", 1.0)), 6),
        int(state.get("depth_edit_version", 0)),
    )
    if cache_key in cache:
        views, sparse_stats = cache[cache_key]
    else:
        try:
            views, sparse_stats = _refine_multiview_camera(
                state["da3_views"], reference_index
            )
        except Exception as exc:
            return state["da3_views"], f"TTO skipped: {type(exc).__name__}: {exc}"
        cache[cache_key] = (views, sparse_stats)
    acceptance_cache = state.setdefault("testtime_acceptance_cache", {})
    acceptance_key = (
        cache_key,
        str(mode),
        str(reference),
        bool(filter_canvas_outliers),
        float(canvas_bounds_quantile),
        bool(visibility_aware),
        round(float(panini_distance), 6),
        round(float(general_panini_compression), 6),
        round(float(general_panini_top), 6),
        round(float(general_panini_bottom), 6),
    )
    if acceptance_key in acceptance_cache:
        stats = acceptance_cache[acceptance_key]
    else:
        stats = dict(sparse_stats)
        if stats.get("optimizer_success"):
            try:
                before_metrics = _coarse_tto_render_metrics(
                    state["da3_views"], mode, reference,
                    filter_canvas_outliers, visibility_aware,
                    panini_distance,
                    general_panini_compression,
                    general_panini_top,
                    general_panini_bottom,
                    canvas_bounds_quantile=canvas_bounds_quantile,
                )
                candidate_metrics = _coarse_tto_render_metrics(
                    views, mode, reference,
                    filter_canvas_outliers, visibility_aware,
                    panini_distance,
                    general_panini_compression,
                    general_panini_top,
                    general_panini_bottom,
                    canvas_bounds_quantile=canvas_bounds_quantile,
                )
                stats = _apply_tto_acceptance(
                    stats, before_metrics, candidate_metrics
                )
            except Exception as exc:
                stats["accepted"] = False
                stats["dense_accepted"] = False
                stats["acceptance_reasons"] = [
                    f"dense gate failed: {type(exc).__name__}: {exc}"
                ]
        else:
            stats["accepted"] = False
            stats["dense_accepted"] = False
            stats["acceptance_reasons"] = ["optimizer did not converge"]
        acceptance_cache[acceptance_key] = stats
    apply_candidate = bool(stats["accepted"] or (force_apply and stats["optimizer_success"]))
    if stats["accepted"]:
        decision = "accepted"
    elif apply_candidate:
        decision = "FORCED apply (acceptance gates bypassed)"
    else:
        decision = "rejected (original DA3 restored)"
    selected_views = views if apply_candidate else state["da3_views"]
    gate_detail = (
        "; gate=" + ",".join(stats.get("acceptance_reasons", []))
        if stats.get("acceptance_reasons") else ""
    )
    return selected_views, (
        f"TTO {decision}; views={stats.get('num_views', len(selected_views))}; "
        f"pairs={stats.get('num_pairs', 1)}; matches={stats['num_matches']}; "
        f"held-out median {stats['initial_heldout_median_px']:.2f}→"
        f"{stats['optimized_heldout_median_px']:.2f}px; "
        f"rotation={stats['rotation_degrees']:.2f}°; "
        f"center_shift={stats.get('translation_shift', 0.0):.3f}; "
        f"depth_scale={stats.get('depth_scale', 1.0):.3f}; "
        f"focal={stats['focal_scale']:.3f}; "
        f"dense score={stats.get('dense_initial_score', float('nan')):.1f}→"
        f"{stats.get('dense_candidate_score', float('nan')):.1f}; "
        f"coverage={stats.get('dense_initial_coverage', float('nan')):.3f}→"
        f"{stats.get('dense_candidate_coverage', float('nan')):.3f}"
        f"{gate_detail}"
    )


def _get_flux_pipeline(model_id: str, device: str, lora_path: str | None = None) -> Any:
    """Load Flux.2 Klein once, on the first explicit Generate Flux click."""

    global _FLUX_PIPE, _FLUX_KEY
    normalized_lora = str(lora_path).strip() if lora_path and str(lora_path).strip() else None
    key = (str(model_id), str(device or "cuda"), normalized_lora)
    with _FLUX_LOCK:
        if _FLUX_PIPE is not None and _FLUX_KEY == key:
            return _FLUX_PIPE
        try:
            import torch
            from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
        except ImportError as exc:  # pragma: no cover - optional GPU dependency.
            raise RuntimeError(
                "Flux generation requires torch and the repository's DiffSynth."
            ) from exc
        # This is the same split used by the supplied validation script:
        # base-9B provides the transformer/tokenizer while Klein-9B provides
        # the text encoder and VAE.
        pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(
                    model_id="black-forest-labs/FLUX.2-klein-9B",
                    origin_file_pattern="text_encoder/*.safetensors",
                ),
                ModelConfig(
                    model_id=model_id,
                    origin_file_pattern="transformer/*.safetensors",
                ),
                ModelConfig(
                    model_id="black-forest-labs/FLUX.2-klein-9B",
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id=model_id,
                origin_file_pattern="tokenizer/",
            ),
            is_training=False,
        )
        # We already ran DA3 in this UI and always provide explicit maps below;
        # Flux's optional internal DA3 copy would only duplicate GPU/CPU memory.
        if hasattr(pipe, "da3"):
            pipe.da3 = None
        if normalized_lora:
            lora_file = Path(normalized_lora).expanduser()
            if not lora_file.is_file():
                raise FileNotFoundError(f"Flux LoRA 不存在：{lora_file}")
            # The reference validation script loads the stitching LoRA before
            # moving the pipeline to CUDA.  Without it this is only the base
            # image model and does not learn the multi-view stitching task.
            pipe.load_lora(pipe.dit, str(lora_file))
        if hasattr(pipe, "to"):
            pipe.to(device or "cuda")
        _FLUX_PIPE, _FLUX_KEY = pipe, key
        return pipe


def _flux_tile_regions(
    width: int,
    height: int,
    tile: int = 1024,
    stride: int = 768,
    sequence: str = "center",
) -> list[tuple[int, int, int, int]]:
    """Return Flux windows in the validation script's adaptive order.

    The default ``center`` sequence visits the middle x/y windows first, then
    expands outward.  This matches ``slice_image_adaptive(..., mode="center")``
    in FLUX.2-klein-base-9B.py and gives the next patch the most useful
    already-generated context.
    """

    def starts(length: int, window: int) -> list[int]:
        if length <= window:
            return [0]
        values = list(range(0, length - window + 1, stride))
        if values[-1] != length - window:
            values.append(length - window)
        return values

    window_w, window_h = min(tile, width), min(tile, height)
    x_starts = starts(width, window_w)
    y_starts = starts(height, window_h)
    if sequence == "center":
        x_mid, y_mid = len(x_starts) // 2, len(y_starts) // 2
        x_starts = x_starts[x_mid:] + x_starts[:x_mid][::-1]
        y_starts = y_starts[y_mid:] + y_starts[:y_mid][::-1]
    elif sequence != "left":
        raise ValueError(f"Unsupported Flux generation sequence: {sequence}")
    return [
        (x, y, x + window_w, y + window_h)
        for x in x_starts
        for y in y_starts
    ]


def _resize_flux_source(
    image: Image.Image,
    coord_map: np.ndarray,
) -> tuple[Image.Image, np.ndarray]:
    """Keep the original source resolution, aligning only to 16 pixels.

    This is only a small source-resolution resize required by the Flux VAE
    (never a camera/projective warp).  Unlike the offline validation script's
    fixed 1024 working-resolution preparation, the interactive UI preserves the
    user's input resolution.
    """

    width, height = image.size
    target_w = max(16, int(((width + 8) // 16) * 16))
    target_h = max(16, int(((height + 8) // 16) * 16))
    if (target_w, target_h) == (width, height):
        return image, coord_map
    resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    map_array = np.asarray(coord_map, dtype=np.float32)
    target_latent_size = (target_w // 16, target_h // 16)
    if map_array.shape[:2] == (target_latent_size[1], target_latent_size[0]):
        return resized, map_array
    # Both -1 and 1000 corrupt neighbouring coordinates if interpolated as
    # numbers. Carry validity separately through this second grid resize too.
    valid = (
        np.isfinite(map_array).all(axis=-1)
        & (map_array >= 0.0).all(axis=-1)
        & (map_array < 1000.0).all(axis=-1)
    )
    resized_map = _resize_valid_coord_map(map_array, valid, target_latent_size)
    return resized, resized_map


# Backward-compatible private name for callers that imported the old helper.
_pad_flux_source = _resize_flux_source


def _tensor_to_rgb(value: Any) -> np.ndarray:
    array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] in (1, 3, 4):
        array = np.transpose(array[:3], (1, 2, 0))
    array = ((array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return array


def generate_flux(
    state: dict[str, Any] | None,
    prompt: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    source_view_top_k: int,
    separate_median_rope_normalization: bool,
    flux_model_id: str,
    flux_lora_path: str,
    flux_device: str,
):
    """Run Flux directly on the last displayed pre-stitch canvas."""

    if not state or not state.get("views"):
        yield None, "请先点击 Load geometry。"
        return
    # A new generation starts from a clean cancellation state.  The cancel
    # button can set this event while a tile is being inferred; the active tile
    # finishes and the generator exits before starting the next one.
    _FLUX_CANCEL.clear()
    render = state.get("last_render")
    if not render or not render.get("metrics"):
        yield None, "请先点击 Generate pre-stitch；Flux 将使用当前显示的画布。"
        return
    composite = np.asarray(render["composite"], dtype=np.uint8)
    height, width = composite.shape[:2]
    # Match the reference validation script: round only the right/bottom edge
    # up to Flux VAE's 16-pixel lattice.  Existing pixels and coord-map values
    # remain in the displayed pre-stitch coordinate system; the returned image
    # is cropped back to the original displayed extent after generation.
    flux_width = max(16, int(np.ceil(width / 16.0) * 16))
    flux_height = max(16, int(np.ceil(height / 16.0) * 16))
    metrics = render["metrics"]
    try:
        source_view_top_k = int(source_view_top_k or 0)
    except (TypeError, ValueError):
        yield None, "source view top-K 必须是整数；设置为 0 表示不限制。"
        return
    if source_view_top_k < 0:
        yield None, "source view top-K 必须大于等于 0；设置为 0 表示不限制。"
        return
    coord_maps = metrics.get("_coord_maps")
    coord_maps_depth = metrics.get("_coord_maps_depth")
    if coord_maps is None or coord_maps_depth is None:
        yield None, "当前 pre-stitch 没有生成 coord_map，请重新 Generate pre-stitch。"
        return
    try:
        import torch

        yield None, "Flux 模型加载中（首次运行会较慢）……"
        if not flux_lora_path or not str(flux_lora_path).strip():
            yield None, (
                "警告：未提供 stitching LoRA，当前将使用裸 Flux Klein base；"
                "它通常不会充分利用多帧 input。"
            )
        pipe = _get_flux_pipeline(flux_model_id, flux_device, flux_lora_path)
        # Native sources are passed untouched (apart from VAE-required resize);
        # in particular, no rotation homography/pre-warp is applied.
        edit_images, maps, maps_depth = [], [], []
        for view, cmap, dmap in zip(state["views"], coord_maps, coord_maps_depth):
            image, resized_map = _resize_flux_source(Image.fromarray(view.image), np.asarray(cmap))
            _, resized_depth_map = _resize_flux_source(Image.fromarray(view.image), np.asarray(dmap))
            edit_images.append(image)
            maps.append(resized_map / 1.0)
            maps_depth.append(resized_depth_map / 1.0)

        latent_h, latent_w = flux_height // 16, flux_width // 16
        pano_latent = torch.zeros((1, 128, latent_h, latent_w), device=flux_device, dtype=torch.float32)
        weight_map = torch.zeros((latent_h, latent_w), device=flux_device, dtype=torch.float32)
        output = torch.zeros((1, 3, flux_height, flux_width), device=flux_device, dtype=torch.float32)
        output_weight = torch.zeros((1, 1, flux_height, flux_width), device=flux_device, dtype=torch.float32)
        tile_regions = _flux_tile_regions(flux_width, flux_height)
        for tile_index, (x0, y0, x1, y1) in enumerate(tile_regions):
            if _FLUX_CANCEL.is_set():
                partial = _tensor_to_rgb(output / output_weight.clamp_min(1e-6))[:height, :width]
                yield partial, (
                    f"Flux 已取消：完成 {tile_index}/{len(tile_regions)} 个 patch。"
                    "当前部分结果已保留；再次点击 Generate with Flux 可重新开始。"
                )
                return
            tile_w, tile_h = x1 - x0, y1 - y0
            tile_latent = [x0 // 16, y0 // 16, x1 // 16, y1 // 16]
            context_latent = pano_latent / (weight_map.view(1, 1, latent_h, latent_w) + 1e-6)
            tile_out, tile_latent_out = pipe(
                prompt or "Stitch input figures. Generated results must be natural and seamless.",
                pano_latent=context_latent.to(dtype=torch.bfloat16),
                edit_image=edit_images,
                edit_image_auto_resize=False,
                coord_maps=maps,
                coord_maps_depth=maps_depth,
                seed=int(seed),
                num_inference_steps=max(1, int(steps)),
                height=tile_h,
                width=tile_w,
                cfg_scale=float(cfg_scale),
                crop_region_latent=tile_latent,
                weight_map_latent=weight_map,
                gt_height=flux_height,
                gt_width=flux_width,
                source_view_top_k=source_view_top_k,
                separate_median_rope_normalization=separate_median_rope_normalization,
            )
            tile = tile_out.float()
            window_y = (1.0 - torch.abs(torch.linspace(-1.0, 1.0, tile_h, device=flux_device))).clamp_min(0.001).view(1, 1, tile_h, 1)
            window_x = (1.0 - torch.abs(torch.linspace(-1.0, 1.0, tile_w, device=flux_device))).clamp_min(0.001).view(1, 1, 1, tile_w)
            blend = torch.minimum(window_y, window_x)
            output[:, :, y0:y1, x0:x1] += tile * blend
            output_weight[:, :, y0:y1, x0:x1] += blend
            latent_window_h, latent_window_w = tile_h // 16, tile_w // 16
            latent_y = torch.linspace(-1.0, 1.0, latent_window_h, device=flux_device)
            latent_x = torch.linspace(-1.0, 1.0, latent_window_w, device=flux_device)
            latent_blend = torch.minimum(
                (1.0 - torch.abs(latent_y)).view(latent_window_h, 1),
                (1.0 - torch.abs(latent_x)).view(1, latent_window_w),
            ).clamp_min(0.001)
            pano_latent[:, :, y0 // 16:y1 // 16, x0 // 16:x1 // 16] += tile_latent_out.to(dtype=torch.float32) * latent_blend.view(1, 1, latent_window_h, latent_window_w)
            weight_map[y0 // 16:y1 // 16, x0 // 16:x1 // 16] += latent_blend
            # Stream the partially blended panorama after every completed tile.
            partial = _tensor_to_rgb(output / output_weight.clamp_min(1e-6))[:height, :width]
            yield partial, (
                f"Flux 生成中：已完成 {tile_index + 1}/{len(tile_regions)} 个 patch；"
                f"显示画布 {width}×{height}（Flux 内部 {flux_width}×{flux_height}），未使用 pre-warp。"
            )
        result = _tensor_to_rgb(output / output_weight.clamp_min(1e-6))[:height, :width]
    except Exception as exc:
        yield None, f"Flux 生成失败：{type(exc).__name__}: {exc}"
        return
    size_note = (
        f"（Flux 内部右/下补齐为 {flux_width}×{flux_height}，返回时裁回原尺寸）"
        if (flux_width, flux_height) != (width, height) else ""
    )
    yield result, f"Flux 完成：使用当前 pre-stitch {width}×{height} 画布{size_note}；未使用 pre-warp。"


def cancel_flux() -> str:
    """Request cancellation of the currently running Flux generation."""

    _FLUX_CANCEL.set()
    return "Flux cancellation requested"


def load_case(
    files: Any,
    model_id: str,
    model_cache_dir: str,
    device: str,
    process_res: int,
    geometry_source: str = "DA3",
    colmap_workspace: str = "",
    pi3x_model_id: str = "yyfz233/Pi3X",
    pi3x_root: str = "third_party/Pi3",
    vggt_omega_checkpoint: str = "models/VGGT-Omega/vggt_omega_1b_512.pt",
    vggt_omega_root: str = "third_party/vggt-omega",
    vggt_omega_image_resolution: int = 512,
) -> tuple[dict[str, Any] | None, list[np.ndarray], str, np.ndarray | None]:
    """Load one case using DA3, Pi3X, VGGT-Omega, or COLMAP."""

    paths = _uploaded_paths(files)
    if not paths:
        return None, [], "请先上传至少两张按顺序排列的 RGB 帧。", None
    if len(paths) < 2:
        return None, [], "至少需要两张帧才能进行 pre-stitch。", None
    try:
        images = load_images(paths)
        source = str(geometry_source or "DA3").strip().upper()
        if source == "COLMAP":
            if not str(colmap_workspace).strip():
                raise ValueError("COLMAP geometry source requires a workspace path")
            views, colmap_info = load_colmap_views(images, paths, colmap_workspace)
            da3_views = views
        elif source == "DA3":
            prediction = _run_da3(
                paths, model_id, model_cache_dir or None, device or None, int(process_res)
            )
            arrays = prediction_to_arrays(prediction)
            da3_views = prepare_views(images, SimpleNamespace(**arrays), paths)
            views = da3_views
            colmap_info = None
        elif source == "PI3X":
            views = _run_pi3x(
                images,
                paths,
                str(pi3x_model_id or "yyfz233/Pi3X"),
                str(pi3x_root or "third_party/Pi3"),
                device or None,
            )
            da3_views = views
            colmap_info = None
        elif source in {"VGGT-OMEGA", "VGGT_OMEGA", "VGGTOMEGA"}:
            views = _run_vggt_omega(
                images,
                paths,
                str(vggt_omega_checkpoint or "models/VGGT-Omega/vggt_omega_1b_512.pt"),
                str(vggt_omega_root or "third_party/vggt-omega"),
                device or None,
                int(vggt_omega_image_resolution),
            )
            da3_views = views
            colmap_info = None
        else:
            raise ValueError(
                f"Unknown geometry source: {geometry_source!r}; choose DA3, Pi3X, VGGT-Omega, or COLMAP"
            )
    except Exception as exc:  # Surface actionable errors in the UI.
        return None, [], f"Case 加载失败：{type(exc).__name__}: {exc}", None

    state = {
        "views": views,
        "base_views": views,
        "da3_views": da3_views,
        "paths": paths,
        "model_id": model_id,
        "process_res": int(process_res),
        "geometry_source": source,
        "colmap_workspace": str(colmap_workspace or ""),
        "colmap_info": colmap_info,
        "pi3x_model_id": str(pi3x_model_id or "yyfz233/Pi3X"),
        "pi3x_root": str(pi3x_root or "third_party/Pi3"),
        "vggt_omega_checkpoint": str(vggt_omega_checkpoint or "models/VGGT-Omega/vggt_omega_1b_512.pt"),
        "vggt_omega_root": str(vggt_omega_root or "third_party/vggt-omega"),
        "vggt_omega_image_resolution": int(vggt_omega_image_resolution),
        "sample_cache_key": None,
        "sample_cache": None,
        "all_sample_cache_key": None,
        "all_sample_cache": None,
        "input_masks": [np.zeros(view.image.shape[:2], dtype=bool) for view in views],
        "input_mask_version": 0,
        "depth_average_masks": [np.zeros(view.image.shape[:2], dtype=bool) for view in views],
        "depth_edit_version": 0,
        "testtime_optimization_cache": {},
        "testtime_acceptance_cache": {},
        "target_rotation_optimization_cache": {},
        "viewer_scene_path": None,
        "last_render": None,
    }
    preview = [image for image in images[: min(8, len(images))]]
    if source == "COLMAP":
        depth_types = ", ".join(str(x) for x in colmap_info["depth_types"])
        status = (
            f"COLMAP 完成：从 full reconstruction 匹配并加载 {len(views)} 个 input；"
            f"depth={depth_types}；camera pose 与 dense depth 已缓存。"
        )
    elif source == "PI3X":
        status = (
            f"Pi3X 完成：{len(views)} 个 input 已加载；metric depth、camera pose 与 "
            "per-view intrinsic 已缓存，后续调整不会重复 Pi3X 推理。"
        )
    elif source in {"VGGT-OMEGA", "VGGT_OMEGA", "VGGTOMEGA"}:
        status = (
            f"VGGT-Omega 完成：{len(views)} 个 input 已加载；camera pose、per-view "
            "intrinsic、dense depth 已缓存，后续调整不会重复 VGGT-Omega 推理。"
        )
    else:
        status = (
            f"DA3 完成：{len(views)} 个 input 已加载；dense depth 已缓存；"
            "后续调整不会重复 DA3 推理。"
        )
    return state, preview, status, np.asarray(images[0]).copy()


def select_mask_frame(
    state: dict[str, Any] | None,
    frame_index: int,
) -> tuple[Any | None, str]:
    """Load one input frame into the brush editor without rerunning geometry."""

    if not state or not state.get("base_views"):
        return None, "请先点击 Load geometry。"
    views = state["base_views"]
    index = max(0, min(int(frame_index or 0), len(views) - 1))
    frame = np.asarray(views[index].image).copy()
    masks = state.get("input_masks")
    mask = (
        _resize_input_mask(masks[index], frame.shape[1], frame.shape[0])
        if masks else np.zeros(frame.shape[:2], dtype=bool)
    )
    depth_masks = state.get("depth_average_masks")
    depth_mask = (
        _resize_input_mask(depth_masks[index], frame.shape[1], frame.shape[0])
        if depth_masks else np.zeros(frame.shape[:2], dtype=bool)
    ) & ~mask
    # Both operations round-trip in one color-coded RGBA layer.
    layer = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
    layer[mask] = [255, 0, 0, 150]
    layer[depth_mask] = [0, 0, 255, 150]
    composite = frame.copy()
    composite[mask] = (0.45 * composite[mask] + 0.55 * np.asarray([255, 0, 0])).astype(np.uint8)
    composite[depth_mask] = (0.45 * composite[depth_mask] + 0.55 * np.asarray([0, 0, 255])).astype(np.uint8)
    masked = int(mask.sum())
    total = int(np.prod(views[index].image.shape[:2]))
    return {"background": frame, "layers": [layer], "composite": composite}, (
        f"当前编辑 frame {index}；红色屏蔽 {100.0 * masked / max(1, total):.2f}%；"
        f"蓝色深度平均化 {100.0 * depth_mask.sum() / max(1, total):.2f}%。"
    )


def update_input_mask(
    state: dict[str, Any] | None,
    frame_index: int,
    editor_value: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Persist exclusion and depth-average masks; leave base geometry intact."""

    if not state or not state.get("base_views"):
        return state, "请先点击 Load geometry。"
    views = state["base_views"]
    index = max(0, min(int(frame_index or 0), len(views) - 1))
    masks = state.setdefault(
        "input_masks", [np.zeros(view.image.shape[:2], dtype=bool) for view in views]
    )
    depth_masks = state.setdefault(
        "depth_average_masks", [np.zeros(view.image.shape[:2], dtype=bool) for view in views]
    )
    mask, depth_mask = _editor_masks(editor_value, views[index].image.shape[:2])
    depth_changed = not np.array_equal(depth_masks[index], depth_mask)
    masks[index] = mask
    depth_masks[index] = depth_mask
    _invalidate_mask_render(state, depth_changed=depth_changed)
    _, depth_stats = _apply_depth_average_masks([views[index]], [depth_mask])
    masked = int(masks[index].sum())
    total = int(np.prod(masks[index].shape))
    return state, (
        f"frame {index} mask 已保存：红色屏蔽 {100.0 * masked / max(1, total):.2f}%；"
        f"蓝色 {depth_stats['regions']} 个连通区域分别取平均 depth"
        f"（跳过 {depth_stats['skipped']} 个无有效深度区域）。"
        "点击 Generate pre-stitch 更新点云、画布及 coord_map 后再运行 Flux。"
    )


def clear_input_mask(
    state: dict[str, Any] | None,
    frame_index: int,
) -> tuple[dict[str, Any] | None, np.ndarray | None, str]:
    """Clear one frame's mask and reset the brush editor background."""

    if not state or not state.get("base_views"):
        return state, None, "请先点击 Load geometry。"
    views = state["base_views"]
    index = max(0, min(int(frame_index or 0), len(views) - 1))
    masks = state.setdefault(
        "input_masks", [np.zeros(view.image.shape[:2], dtype=bool) for view in views]
    )
    masks[index] = np.zeros(views[index].image.shape[:2], dtype=bool)
    depth_masks = state.setdefault(
        "depth_average_masks", [np.zeros(view.image.shape[:2], dtype=bool) for view in views]
    )
    depth_changed = bool(np.asarray(depth_masks[index]).any())
    depth_masks[index] = np.zeros(views[index].image.shape[:2], dtype=bool)
    _invalidate_mask_render(state, depth_changed=depth_changed)
    return state, np.asarray(views[index].image).copy(), (
        f"frame {index} 红色和蓝色 mask 已清除。点击 Generate pre-stitch 恢复原始深度并重新渲染。"
    )


def _rotation_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return a row-vector rotation matrix for XYZ camera-path offsets."""

    ax, ay, az = np.deg2rad([float(rx), float(ry), float(rz)])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx_m = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rx_m @ ry_m @ rz_m


def _adjust_views(
    views: Sequence[ViewGeometry],
    focal_scale: float,
    cx_offset: float,
    cy_offset: float,
    rx: float,
    ry: float,
    rz: float,
    tx: float,
    ty: float,
    tz: float,
    target_index: int = 0,
) -> list[ViewGeometry]:
    """Apply intrinsic/extrinsic edits only to the reference (target) view."""

    selected = max(0, min(int(target_index), len(views) - 1))

    delta_rotation = _rotation_xyz(rx, ry, rz)
    delta_translation = np.asarray([tx, ty, tz], dtype=np.float64)
    adjusted = []
    for index, view in enumerate(views):
        K = view.K.copy()
        if index == selected:
            K[0, 0] *= float(focal_scale)
            K[1, 1] *= float(focal_scale)
            K[0, 2] += float(cx_offset)
            K[1, 2] += float(cy_offset)

        # Existing geometry uses C=-T@R. A rotation control means rotating the
        # target camera in place; it must not orbit the camera center around the
        # world origin. Translation is the only control that changes C.
        if index == selected:
            R = view.R @ delta_rotation
            center = view.camera_center + delta_translation
            T = -center @ R.T
        else:
            R, T = view.R.copy(), view.T.copy()
        adjusted.append(replace(view, K=K, K_processed=K.copy(), R=R, T=T))
    return adjusted


def _edit_target_camera(
    views: Sequence[ViewGeometry],
    reference: str,
    focal_scale: float,
    rx: float,
    ry: float,
    rz: float,
    tx: float,
    ty: float,
    tz: float,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Create an editable target camera while leaving source views untouched."""

    center, scene_rotation, _focal = reference_camera(views, reference)
    if reference == "first":
        target_source = views[0]
        focal_x, focal_y = float(target_source.K[0, 0]), float(target_source.K[1, 1])
    elif reference == "mid":
        target_source = views[len(views) // 2]
        focal_x, focal_y = float(target_source.K[0, 0]), float(target_source.K[1, 1])
    else:
        median_k = np.median(np.stack([view.K for view in views], axis=0), axis=0)
        focal_x, focal_y = float(median_k[0, 0]), float(median_k[1, 1])
    delta_rotation = _rotation_xyz(rx, ry, rz)
    # scene_rotation is R_target.T under the row-vector convention.
    scene_rotation = delta_rotation.T @ scene_rotation
    # Translation controls are expressed in the edited target camera frame,
    # not in DA3's arbitrary world frame.  With row vectors, camera-to-world
    # rotation is R_target == scene_rotation.T.
    local_translation = np.asarray([tx, ty, tz], dtype=np.float64)
    center = center + local_translation @ scene_rotation.T
    return (
        center,
        scene_rotation,
        focal_x * float(focal_scale),
        focal_y * float(focal_scale),
        float(cx_offset),
        float(cy_offset),
    )


def _target_rotation_coverage(
    views: Sequence[ViewGeometry],
    samples: Sequence[tuple[np.ndarray, ...]],
    mode: str,
    reference: str,
    rotation_degrees: Sequence[float],
    focal_scale: float,
    cx_offset: float,
    cy_offset: float,
    tx: float,
    ty: float,
    tz: float,
    bounds_quantile: float,
    max_canvas_pixels: int = 300_000,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> dict[str, Any]:
    """Estimate occupied area in a tight canvas without compositing RGB."""

    rx, ry, rz = (float(value) for value in rotation_degrees)
    target_camera = _edit_target_camera(
        views, reference, focal_scale, rx, ry, rz, tx, ty, tz,
        cx_offset, cy_offset,
    )
    projected, projection_meta = project_views(
        views,
        mode,
        reference,
        render_stride=1,
        unprojected=samples,
        target_camera=target_camera,
        panini_distance=panini_distance,
        general_panini_compression=general_panini_compression,
        general_panini_top=general_panini_top,
        general_panini_bottom=general_panini_bottom,
    )
    transformed, (width, height), canvas_meta = _canvas_transform(
        projected,
        quantile=float(bounds_quantile),
        padding=0,
        max_canvas_pixels=max(1, int(max_canvas_pixels)),
        requested_size=None,
    )
    occupied = np.zeros(max(1, width * height), dtype=bool)
    for item, uv in zip(projected, transformed):
        if not len(uv):
            continue
        base_x = np.floor(uv[:, 0]).astype(np.int64)
        base_y = np.floor(uv[:, 1]).astype(np.int64)
        frac_x = uv[:, 0] - base_x
        frac_y = uv[:, 1] - base_y
        source_weights = np.asarray(item.feather, dtype=np.float64)
        for dx, dy, bilinear in (
            (0, 0, (1.0 - frac_x) * (1.0 - frac_y)),
            (1, 0, frac_x * (1.0 - frac_y)),
            (0, 1, (1.0 - frac_x) * frac_y),
            (1, 1, frac_x * frac_y),
        ):
            x, y = base_x + dx, base_y + dy
            valid = (
                (x >= 0) & (x < width) & (y >= 0) & (y < height)
                & (bilinear * source_weights > 1e-6)
            )
            occupied[y[valid] * width + x[valid]] = True
    coverage = float(occupied.mean())
    return {
        "coverage": coverage,
        "black_fraction": 1.0 - coverage,
        "projection_valid_fraction": float(projection_meta["projection_valid_fraction"]),
        "canvas_size": [int(width), int(height)],
        "raw_canvas_size": list(canvas_meta["raw_canvas_size"]),
    }


def _optimize_target_rotation(
    views: Sequence[ViewGeometry],
    mode: str,
    reference: str,
    initial_rotation: Sequence[float],
    focal_scale: float,
    cx_offset: float,
    cy_offset: float,
    tx: float,
    ty: float,
    tz: float,
    bounds_quantile: float = 0.001,
    source_masks: Sequence[np.ndarray] | None = None,
    max_delta_degrees: float = 30.0,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Numerically find a target orientation with less empty tight-canvas area.

    Source cameras, target center, intrinsics, and depth stay fixed.  A coarse
    point set and a small raster are used during optimization; the selected
    orientation is subsequently rendered at the user's full settings.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - TTO already depends on scipy.
        raise RuntimeError("target rotation optimization requires scipy") from exc
    if not views:
        raise ValueError("target rotation optimization requires at least one view")
    initial = np.asarray(initial_rotation, dtype=np.float64).reshape(3)
    if not np.isfinite(initial).all():
        raise ValueError("initial target rotation must be finite")

    total_pixels = sum(int(view.image.shape[0] * view.image.shape[1]) for view in views)
    optimization_stride = max(2, int(np.ceil(np.sqrt(total_pixels / 400_000.0))))
    samples = unproject_views(views, optimization_stride)
    if source_masks is not None:
        samples = _filter_samples_by_input_mask(samples, views, source_masks)

    evaluations: dict[tuple[float, float, float], dict[str, Any]] = {}

    def evaluate(rotation: Sequence[float]) -> dict[str, Any]:
        key = tuple(round(float(value), 6) for value in rotation)
        if key not in evaluations:
            evaluations[key] = _target_rotation_coverage(
                views,
                samples,
                mode,
                reference,
                rotation,
                focal_scale,
                cx_offset,
                cy_offset,
                tx,
                ty,
                tz,
                bounds_quantile,
                panini_distance=panini_distance,
                general_panini_compression=general_panini_compression,
                general_panini_top=general_panini_top,
                general_panini_bottom=general_panini_bottom,
            )
        return evaluations[key]

    initial_metrics = evaluate(initial)
    initial_valid = float(initial_metrics["projection_valid_fraction"])

    # Black-area minimization alone is ill-posed: an orthographic target can
    # rotate nearly edge-on, collapse the scene into a thin strip, and obtain
    # deceptively high occupancy.  Keep the optimized target in the local
    # neighbourhood of the selected first/mid/avg view and mildly regularize
    # optical-axis changes. Roll remains freer because it primarily aligns the
    # panorama with the axis-aligned bounding rectangle.
    delta = abs(float(max_delta_degrees))
    tilt_delta = min(delta, 12.0)
    roll_delta = min(max(delta, 12.0), 45.0)

    def objective(rotation: np.ndarray) -> float:
        metrics = evaluate(rotation)
        # Prevent pinhole projections from winning merely by rotating content
        # behind the target camera.  All-visible periodic projections receive
        # no penalty and optimize their tight-canvas black fraction directly.
        lost_valid = max(
            0.0,
            initial_valid - float(metrics["projection_valid_fraction"]),
        )
        offset = np.asarray(rotation, dtype=np.float64) - initial
        tilt_regularizer = 0.08 * float(np.sum((offset[:2] / max(tilt_delta, 1e-6)) ** 2))
        roll_regularizer = 0.005 * float((offset[2] / max(roll_delta, 1e-6)) ** 2)
        return (
            float(metrics["black_fraction"])
            + 4.0 * lost_valid
            + tilt_regularizer
            + roll_regularizer
        )

    lower = initial - np.asarray([tilt_delta, tilt_delta, roll_delta])
    upper = initial + np.asarray([tilt_delta, tilt_delta, roll_delta])
    # Avoid the pinhole singularity at exactly +/-90 degrees while allowing
    # roll to search a wider equivalent angular interval.
    lower[:2] = np.maximum(lower[:2], -89.0)
    upper[:2] = np.minimum(upper[:2], 89.0)
    lower[2] = max(lower[2], -180.0)
    upper[2] = min(upper[2], 180.0)
    optimized = minimize(
        objective,
        initial,
        method="Powell",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 24, "maxfev": 72, "xtol": 0.2, "ftol": 5e-4},
    )
    # Powell's last point is not guaranteed to be its best sampled point for
    # a quantized occupancy objective. Select the best evaluation explicitly.
    best_key, best_metrics = min(
        evaluations.items(),
        key=lambda item: objective(np.asarray(item[0], dtype=np.float64)),
    )
    best = np.asarray(best_key, dtype=np.float64)
    applied = bool(objective(best) + 1e-6 < objective(initial))
    selected = best if applied else initial
    selected_metrics = best_metrics if applied else initial_metrics
    stats = {
        "applied": applied,
        "optimizer_success": bool(optimized.success),
        "message": str(optimized.message),
        "num_evaluations": len(evaluations),
        "optimization_stride": optimization_stride,
        "max_tilt_delta_degrees": tilt_delta,
        "max_roll_delta_degrees": roll_delta,
        "initial_rotation_degrees": initial.tolist(),
        "optimized_rotation_degrees": selected.tolist(),
        "initial_black_fraction": float(initial_metrics["black_fraction"]),
        "optimized_black_fraction": float(selected_metrics["black_fraction"]),
        "initial_coverage": float(initial_metrics["coverage"]),
        "optimized_coverage": float(selected_metrics["coverage"]),
        "initial_projection_valid_fraction": initial_valid,
        "optimized_projection_valid_fraction": float(
            selected_metrics["projection_valid_fraction"]
        ),
        "optimization_canvas_size": selected_metrics["canvas_size"],
        "optimization_raw_canvas_size": selected_metrics["raw_canvas_size"],
    }
    return tuple(float(value) for value in selected), stats


def _camera_frustum_segments(
    center: np.ndarray,
    rotation_w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    """Return eight world-space line segments for a pinhole camera."""

    corners = np.asarray(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]],
        dtype=np.float64,
    )
    rays = corners @ np.linalg.inv(K).T
    rays /= np.where(np.abs(rays[:, 2:3]) > 1e-12, rays[:, 2:3], 1.0)
    plane_camera = rays * float(scale)
    translation = -np.asarray(center, dtype=np.float64) @ np.asarray(rotation_w2c).T
    plane_world = (plane_camera - translation) @ np.asarray(rotation_w2c)
    segments = [np.stack((center, corner)) for corner in plane_world]
    segments.extend(
        np.stack((plane_world[first], plane_world[second]))
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0))
    )
    return np.asarray(segments, dtype=np.float64)


def build_scene_glb(
    views: Sequence[ViewGeometry],
    target_camera: tuple[np.ndarray, np.ndarray, float, float, float],
    target_index: int,
    samples: Sequence[tuple[np.ndarray, ...]] | None = None,
    max_points: int = 500_000,
    reference_pose: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> str | None:
    """Export a transient GLB point cloud for Gradio's native 3D viewer.

    This deliberately mirrors the official DA3 Space: dense colored points and
    camera wireframes are packed into GLB and displayed by ``gr.Model3D``.  The
    file lives only in a process-lifetime temporary directory.
    """

    try:
        import trimesh
    except ImportError:  # pragma: no cover - optional UI dependency.
        return None

    point_sets, color_sets = [], []
    if samples is not None and len(samples) == len(views):
        budget = max(1, int(max_points) // max(1, len(views)))
        for view, sample in zip(views, samples):
            local, colors = sample[0], sample[1]
            if not len(local):
                continue
            count = min(budget, len(local))
            indices = np.linspace(0, len(local) - 1, count, dtype=np.int64)
            point_sets.append((local[indices] - view.T) @ view.R)
            color_sets.append(np.clip(colors[indices], 0, 255).astype(np.uint8))
    if not point_sets:
        return None
    points_world = np.concatenate(point_sets, axis=0)
    colors = np.concatenate(color_sets, axis=0)
    finite = np.isfinite(points_world).all(axis=1)
    points_world, colors = points_world[finite], colors[finite]
    if not len(points_world):
        return None

    # Express the complete scene in the selected reference camera frame, then
    # flip CV Y/Z to glTF's Y-up/Z-backward convention.  Unlike the official
    # generic exporter, do not recenter by point-cloud median: the origin must
    # remain the reference camera center for this camera-editing UI.
    if reference_pose is None:
        reference_view = views[target_index]
        reference_center = reference_view.camera_center
        reference_scene_rotation = reference_view.R.T
        reference_focal = float((reference_view.K[0, 0] + reference_view.K[1, 1]) / 2)
    else:
        reference_center, reference_scene_rotation, reference_focal = reference_pose
    reference_center = np.asarray(reference_center, dtype=np.float64)
    reference_rotation_w2c = np.asarray(reference_scene_rotation, dtype=np.float64).T
    reference_translation = -reference_center @ reference_rotation_w2c.T
    axis_flip = np.asarray([1.0, -1.0, -1.0])
    aligned = (points_world @ reference_rotation_w2c.T + reference_translation) * axis_flip

    low, high = np.percentile(aligned, [5, 95], axis=0)
    scene_scale = max(float(np.linalg.norm(high - low)), 1e-3)
    frustum_scale = scene_scale * 0.015

    def align_world(points: np.ndarray) -> np.ndarray:
        return (
            np.asarray(points) @ reference_rotation_w2c.T + reference_translation
        ) * axis_flip

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.points.PointCloud(vertices=aligned.astype(np.float32), colors=colors),
        geom_name="DA3 RGB point cloud",
    )

    def add_path(name: str, segments: np.ndarray, color: tuple[int, int, int, int]) -> None:
        path = trimesh.load_path(align_world(segments.reshape(-1, 3)).reshape(-1, 2, 3))
        if hasattr(path, "colors"):
            path.colors = np.tile(np.asarray(color, dtype=np.uint8), (len(path.entities), 1))
        scene.add_geometry(path, geom_name=name)

    centers = np.asarray([view.camera_center for view in views])
    if len(centers) > 1:
        trajectory = np.stack((centers[:-1], centers[1:]), axis=1)
        add_path("input camera trajectory", trajectory, (96, 165, 250, 255))
    for index, view in enumerate(views):
        segments = _camera_frustum_segments(
            view.camera_center,
            view.R,
            view.K,
            int(view.image.shape[1]),
            int(view.image.shape[0]),
            frustum_scale,
        )
        add_path(f"input camera {index}", segments, (59, 130, 246, 255))

    target_source = views[target_index]
    reference_K = np.asarray(
        [
            [reference_focal, 0, target_source.image.shape[1] / 2],
            [0, reference_focal, target_source.image.shape[0] / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    reference_segments = _camera_frustum_segments(
        reference_center,
        reference_rotation_w2c,
        reference_K,
        int(target_source.image.shape[1]),
        int(target_source.image.shape[0]),
        frustum_scale,
    )
    add_path("reference camera", reference_segments, (245, 158, 11, 255))

    (
        target_center,
        _,
        target_rotation_w2c,
        _,
        target_focal,
        target_principal_offset,
    ) = target_camera_components(target_camera)
    target_fy = target_camera_focal_y(target_camera)
    cx_offset, cy_offset = target_principal_offset
    target_K = np.asarray(
        [
            [target_focal, 0, target_source.image.shape[1] / 2 + cx_offset],
            [0, target_fy, target_source.image.shape[0] / 2 + cy_offset],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    target_segments = _camera_frustum_segments(
        np.asarray(target_center),
        target_rotation_w2c,
        target_K,
        int(target_source.image.shape[1]),
        int(target_source.image.shape[0]),
        frustum_scale * 1.15,
    )
    add_path("edited target camera", target_segments, (220, 38, 38, 255))

    with tempfile.NamedTemporaryFile(
        dir=_viewer_temp_dir(), prefix="scene_", suffix=".glb", delete=False
    ) as output:
        output_path = output.name
    scene.export(output_path, file_type="glb")
    return output_path


def render_case(
    state: dict[str, Any] | None,
    mode: str,
    reference: str,
    render_stride: int,
    input_resize_scale: float,
    testtime_optimization: bool,
    force_apply_optimized_camera: bool,
    auto_canvas: bool,
    filter_canvas_outliers: bool,
    visibility_aware: bool,
    auto_canvas_downscale: bool,
    auto_canvas_max_mp: float,
    canvas_width: int,
    canvas_height: int,
    focal_scale: float,
    cx_offset: float,
    cy_offset: float,
    rx: float,
    ry: float,
    rz: float,
    tx: float,
    ty: float,
    tz: float,
    optimize_target_rotation: bool = False,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
    canvas_bounds_quantile: float = 0.001,
) -> tuple[np.ndarray | None, Any | None, str]:
    """Render one in-memory parameter setting for the Gradio callbacks."""

    if not state or not state.get("views"):
        return None, None, "请先点击 Load geometry。"
    width, height = int(canvas_width), int(canvas_height)
    if width < 1 or height < 1:
        return None, None, "画布宽度和高度必须是正整数。"
    max_canvas_pixels = (
        max(1, int(float(auto_canvas_max_mp) * 1_000_000))
        if auto_canvas and auto_canvas_downscale
        else None
    )
    try:
        canvas_bounds_quantile = float(canvas_bounds_quantile)
        if not np.isfinite(canvas_bounds_quantile) or not 0.0 <= canvas_bounds_quantile < 0.5:
            raise ValueError("canvas_bounds_quantile must be finite and in [0, 0.5)")
        base_views = state.get("base_views", state["da3_views"])
        resize_scale = float(input_resize_scale)
        resized_views = _resize_views(base_views, resize_scale)
        resized_views, depth_edit_stats = _apply_depth_average_masks(
            resized_views, state.get("depth_average_masks"), reference_views=base_views
        )
        base_masks = state.get("input_masks")
        resized_masks = None
        if base_masks is not None:
            resized_masks = [
                _resize_input_mask(mask, view.image.shape[1], view.image.shape[0])
                for mask, view in zip(base_masks, resized_views)
            ]
        # TTO and all subsequent rendering operate on this same resized view
        # set. The immutable base set lets the user change scale repeatedly
        # without accumulating resampling or intrinsic changes.
        state["da3_views"] = resized_views
        state["_active_resize_scale"] = resize_scale
        views = resized_views
        optimization_status = ""
        if testtime_optimization:
            views, optimization_status = _testtime_optimize_views(
                state,
                reference,
                force_apply=force_apply_optimized_camera,
                mode=mode,
                filter_canvas_outliers=filter_canvas_outliers,
                canvas_bounds_quantile=canvas_bounds_quantile,
                visibility_aware=visibility_aware,
                panini_distance=panini_distance,
                general_panini_compression=general_panini_compression,
                general_panini_top=general_panini_top,
                general_panini_bottom=general_panini_bottom,
            )
        state["views"] = views
        target_index = 0 if reference == "first" else len(views) // 2
        effective_rx, effective_ry, effective_rz = float(rx), float(ry), float(rz)
        target_rotation_status = ""
        target_rotation_stats = None
        if optimize_target_rotation:
            rotation_cache = state.setdefault("target_rotation_optimization_cache", {})
            rotation_cache_key = (
                str(mode), str(reference), round(resize_scale, 6),
                round(float(panini_distance), 6),
                round(float(general_panini_compression), 6),
                round(float(general_panini_top), 6),
                round(float(general_panini_bottom), 6),
                bool(testtime_optimization), bool(force_apply_optimized_camera),
                bool(filter_canvas_outliers), canvas_bounds_quantile,
                int(state.get("input_mask_version", 0)),
                *(round(float(value), 6) for value in (
                    focal_scale, cx_offset, cy_offset, rx, ry, rz, tx, ty, tz
                )),
            )
            if rotation_cache_key in rotation_cache:
                optimized_rotation, target_rotation_stats = rotation_cache[rotation_cache_key]
            else:
                try:
                    optimized_rotation, target_rotation_stats = _optimize_target_rotation(
                        views,
                        mode,
                        reference,
                        (rx, ry, rz),
                        focal_scale,
                        cx_offset,
                        cy_offset,
                        tx,
                        ty,
                        tz,
                        bounds_quantile=canvas_bounds_quantile if filter_canvas_outliers else 0.0,
                        source_masks=resized_masks,
                        panini_distance=panini_distance,
                        general_panini_compression=general_panini_compression,
                        general_panini_top=general_panini_top,
                        general_panini_bottom=general_panini_bottom,
                    )
                except Exception as exc:
                    optimized_rotation = (float(rx), float(ry), float(rz))
                    target_rotation_stats = {
                        "skipped": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rotation_cache[rotation_cache_key] = (
                    optimized_rotation, target_rotation_stats
                )
            effective_rx, effective_ry, effective_rz = optimized_rotation
            if target_rotation_stats.get("skipped"):
                target_rotation_status = (
                    "target rotation optimization skipped: "
                    f"{target_rotation_stats['error']}"
                )
            else:
                target_rotation_status = (
                    "target rotation optimized "
                    f"({float(rx):.2f},{float(ry):.2f},{float(rz):.2f})°→"
                    f"({effective_rx:.2f},{effective_ry:.2f},{effective_rz:.2f})°; "
                    f"coarse black={100.0 * target_rotation_stats['initial_black_fraction']:.1f}%→"
                    f"{100.0 * target_rotation_stats['optimized_black_fraction']:.1f}%"
                )
        target_camera = _edit_target_camera(
            views, reference, focal_scale,
            effective_rx, effective_ry, effective_rz, tx, ty, tz,
            cx_offset, cy_offset,
        )
        state["views"] = views
        # Source-view unprojection is independent of the editable target
        # camera, so it can be reused for every target intrinsic/extrinsic edit.
        all_sample_key = (
            round(resize_scale, 6),
            int(state.get("depth_edit_version", 0)),
            bool(testtime_optimization),
            bool(force_apply_optimized_camera),
            reference,
            int(render_stride),
        )
        if state.get("all_sample_cache_key") == all_sample_key and state.get("all_sample_cache") is not None:
            all_samples = state["all_sample_cache"]
        else:
            all_samples = unproject_views(views, int(render_stride))
            state["all_sample_cache_key"] = all_sample_key
            state["all_sample_cache"] = all_samples
        samples = _filter_samples_by_input_mask(all_samples, views, resized_masks)
        sample_key = (*all_sample_key, int(state.get("input_mask_version", 0)))
        state["sample_cache_key"] = sample_key
        state["sample_cache"] = samples
        has_mask = bool(resized_masks and any(np.asarray(mask, dtype=bool).any() for mask in resized_masks))
        composite, _heatmap, metrics = render_prestitch_memory(
            views,
            mode=mode,
            reference_mode=reference,
            render_stride=max(1, int(render_stride)),
            bounds_quantile=canvas_bounds_quantile if filter_canvas_outliers else 0.0,
            padding=0,
            max_canvas_pixels=max_canvas_pixels,
            canvas_size=None if auto_canvas else (width, height),
            unprojected=samples,
            # Build coord maps from the unmasked geometry, then overwrite only
            # the masked latent cells after map interpolation. This prevents
            # invalid sentinels from bleeding into neighbouring source cells.
            coord_unprojected=all_samples if has_mask else None,
            coord_source_masks=resized_masks if has_mask else None,
            target_camera=target_camera,
            visibility_aware=bool(visibility_aware),
            return_coord_maps=True,
            return_heatmap=False,
            panini_distance=panini_distance,
            general_panini_compression=general_panini_compression,
            general_panini_top=general_panini_top,
            general_panini_bottom=general_panini_bottom,
        )
    except Exception as exc:
        return None, None, f"渲染失败：{type(exc).__name__}: {exc}"
    status = (
        f"geometry={state.get('geometry_source', 'DA3')} · input_scale={float(input_resize_scale):.3f}× · "
        f"{mode}/{reference} · "
        f"canvas={metrics['canvas_size'][0]}×{metrics['canvas_size'][1]} · "
        f"raw={metrics['raw_canvas_size'][0]}×{metrics['raw_canvas_size'][1]} · "
        f"canvas_scale={metrics['canvas_scale']:.3f} · "
        f"valid3D={metrics['projection_valid_fraction']:.3f} · "
        f"target_depth={metrics['target_depth_median']:.3f} · "
        f"score={metrics['prestitch_score']:.1f} · coverage={metrics['canvas_coverage_ratio']:.3f} · "
        f"translation={metrics['translation_motion_fraction']:.3f} · "
        f"extension={metrics['horizontal_extension']:.2f}× · target_view={target_index}"
    )
    if visibility_aware:
        status += " · fusion=visibility-aware z-buffer (near surfaces selected)"
    if depth_edit_stats["regions"] or depth_edit_stats["skipped"]:
        status += (
            f" · depth averaged={depth_edit_stats['regions']} regions"
            f" ({depth_edit_stats['pixels']} pixels; {depth_edit_stats['skipped']} skipped)"
        )
    if mode == "panini":
        status += f" · panini_d={float(panini_distance):.3f}"
    if mode == "general_panini":
        status += (
            f" · general_panini=cmpr:{float(general_panini_compression):.1f},"
            f"tops:{float(general_panini_top):.1f},"
            f"bots:{float(general_panini_bottom):.1f}"
        )
    masks = state.get("input_masks")
    if masks:
        masked_pixels = sum(int(np.asarray(mask, dtype=bool).sum()) for mask in masks)
        total_pixels = sum(int(np.asarray(view.image).shape[0] * np.asarray(view.image).shape[1]) for view in state["base_views"])
        if masked_pixels:
            status += f" · input mask={100.0 * masked_pixels / max(1, total_pixels):.2f}% pixels hidden"
    if optimization_status:
        status += f" · {optimization_status}"
    if target_rotation_status:
        status += f" · {target_rotation_status}"
    if mode != "rectilinear":
        status += " · 注意：3D frustum 只精确对应 rectilinear；当前模式是非 pinhole 投影"
    if metrics["projection_valid_fraction"] < 0.99:
        behind = 100.0 * (1.0 - metrics["projection_valid_fraction"])
        status += f" · ⚠ target camera 已穿过约 {behind:.1f}% 的投影点"
    if auto_canvas and auto_canvas_downscale and metrics["canvas_scale"] < 0.95:
        status += (
            f" · ⚠ Auto canvas 为满足 8MP 上限将投影缩放至 "
            f"{metrics['canvas_scale']:.3f}×；请关闭 Auto canvas 比较相机尺度"
        )
    scene_path = build_scene_glb(
        state["views"],
        target_camera,
        target_index,
        samples=samples,
        reference_pose=reference_camera(state["views"], reference),
    )
    previous_scene = state.get("viewer_scene_path")
    state["viewer_scene_path"] = scene_path
    if previous_scene and previous_scene != scene_path:
        try:
            Path(previous_scene).unlink(missing_ok=True)
        except OSError:
            pass
    if scene_path is None:
        status += " · 3D viewer 生成失败；请确认已安装 trimesh"
    # This snapshot is the contract for Flux: generation consumes precisely
    # this displayed canvas coordinate system and its maps, without another
    # projection, crop, or warp pass.
    state["last_render"] = {
        "composite": composite.copy(),
        "metrics": metrics,
        "depth_edit_stats": depth_edit_stats,
        "mode": mode,
        "reference": reference,
        "target_camera": target_camera,
        "target_rotation_degrees": [effective_rx, effective_ry, effective_rz],
        "target_rotation_optimization": target_rotation_stats,
        "panini_distance": float(panini_distance),
        "general_panini_compression": float(general_panini_compression),
        "general_panini_top": float(general_panini_top),
        "general_panini_bottom": float(general_panini_bottom),
    }
    return composite, scene_path, status


def build_demo(
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    model_cache_dir: str = "models/depth-anything-3",
    device: str = "cuda",
):
    """Construct the Gradio Blocks app (imported lazily)."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("Gradio is required: pip install 'gradio>=4.0'") from exc

    with gr.Blocks(title="DA3 / Pi3X Pre-stitch Lab") as demo:
        gr.Markdown(
            "# DA3 / Pi3X Pre-stitch Lab\n"
            "点击 **Load geometry** 后，DA3、Pi3X 或 COLMAP 只加载一次；调整下面参数只重新渲染内存中的几何，默认不写文件。"
        )
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="Ordered input frames",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                geometry_source = gr.Dropdown(
                    ["DA3", "Pi3X", "VGGT-Omega", "COLMAP"],
                    value="DA3",
                    label="Geometry source",
                    info=(
                        "DA3 runs joint depth/pose inference; Pi3X runs RGB-only metric "
                        "depth/pose inference; VGGT-Omega uses the local 1B checkpoint; "
                        "COLMAP loads precomputed pose + dense depth "
                        "from a full reconstruction by frame filename."
                    ),
                )
                pi3x_model_id = gr.Textbox(
                    value="yyfz233/Pi3X",
                    label="Pi3X model id/checkpoint",
                    info="Used only when Geometry source=Pi3X; e.g. yyfz233/Pi3X.",
                )
                pi3x_root = gr.Textbox(
                    value="third_party/Pi3",
                    label="Pi3 repository path",
                    info="Checkout containing pi3/models/pi3x.py.",
                )
                vggt_omega_checkpoint = gr.Textbox(
                    value="models/VGGT-Omega/vggt_omega_1b_512.pt",
                    label="VGGT-Omega checkpoint",
                    info="Used only when Geometry source=VGGT-Omega.",
                )
                vggt_omega_root = gr.Textbox(
                    value="third_party/vggt-omega",
                    label="VGGT-Omega repository path",
                    info="Checkout containing vggt_omega/models and vggt_omega/utils.",
                )
                vggt_omega_image_resolution = gr.Slider(
                    256, 1024, value=512, step=16,
                    label="VGGT-Omega image resolution",
                    info="Official 1B-512 checkpoint is normally used at 512.",
                )
                colmap_workspace = gr.Textbox(
                    value="",
                    label="COLMAP full reconstruction workspace",
                    info="Used only when Geometry source=COLMAP; input basenames must exist in this reconstruction.",
                )
                process_res = gr.Slider(256, 768, value=504, step=8, label="DA3 process resolution")
                run_button = gr.Button("Load geometry", variant="primary")
                preview = gr.Gallery(label="Input preview", columns=4, height=220)
                load_status = gr.Markdown()
                gr.Markdown(
                    "### Input mask brush\n"
                    "红色：屏蔽区域（原功能）。蓝色：保留 RGB，将每个连通区域的 depth "
                    "替换为该区域原始有效 depth 的平均值。相连（含对角接触）的蓝色笔迹视为同一区域。\n\n"
                    "点击 **Apply mask** 保存，再点击 **Generate pre-stitch** 更新点云、"
                    "画布和 coord_map。擦除蓝色或 Clear current mask 后重新渲染可恢复深度；"
                    "Clear 会清除当前帧两种颜色。\n\n"
                    "画笔修改不会自动上传；切换 frame 前请先 Apply mask，"
                    "否则未保存的修改会丢失。"
                )
                mask_frame_index = gr.Number(
                    value=0,
                    minimum=0,
                    precision=0,
                    label="Mask frame index",
                )
                mask_editor = gr.ImageEditor(
                    value=None,
                    type="numpy",
                    image_mode="RGBA",
                    label="Red: exclude · Blue: average region depth",
                    brush=gr.Brush(
                        default_size=40, colors=["#ff0000", "#0000ff"],
                        default_color="#ff0000", color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=40),
                    canvas_size=(800, 600),
                    transforms=(),
                    layers=True,
                )
                with gr.Row():
                    apply_mask_button = gr.Button("Apply mask", variant="primary")
                    clear_mask_button = gr.Button("Clear current mask")
                mask_status = gr.Markdown()
            with gr.Column(scale=1):
                mode = gr.Dropdown(list(PROJECTION_MODES), value="rectilinear", label="Projection")
                panini_distance = gr.Slider(
                    0.0,
                    1.0,
                    value=PANINI_DISTANCE,
                    step=0.05,
                    label="Panini distance (d)",
                    info=(
                        "Used only for Projection=panini. d=0 matches rectilinear; "
                        "d=1 is classic Panini and supports the widest horizontal field."
                    ),
                )
                general_panini_compression = gr.Slider(
                    0.0,
                    150.0,
                    value=GENERAL_PANINI_COMPRESSION,
                    step=1.0,
                    label="General Panini compression (Cmpr)",
                    info=(
                        "Used only for Projection=general_panini. 0 is rectilinear, "
                        "100 is standard Panini, and 150 approaches cylindrical orthographic."
                    ),
                )
                with gr.Row():
                    general_panini_top = gr.Slider(
                        -100.0,
                        100.0,
                        value=GENERAL_PANINI_TOP_SQUEEZE,
                        step=1.0,
                        label="General Panini Tops",
                        info="Positive=hard squeeze; negative=soft squeeze; 0=off.",
                    )
                    general_panini_bottom = gr.Slider(
                        -100.0,
                        100.0,
                        value=GENERAL_PANINI_BOTTOM_SQUEEZE,
                        step=1.0,
                        label="General Panini Bots",
                        info="Positive=hard squeeze; negative=soft squeeze; 0=off.",
                    )
                reference = gr.Dropdown(list(REFERENCE_MODES), value="mid", label="Reference frame")
                render_stride = gr.Slider(
                    1, 8, value=4, step=1,
                    label="Render stride (4 recommended for preview; 1 for final)",
                )
                input_resize_scale = gr.Number(
                    value=1.0,
                    step=0.05,
                    precision=2,
                    label="Input resize scale",
                    info=(
                        "Resizes RGB/depth grids and scales fx, fy, cx, cy. "
                        "Metric depth values and camera extrinsics stay unchanged."
                    ),
                )
                testtime_optimization = gr.Checkbox(
                    False,
                    label="Test-time camera optimization (multi-view)",
                    info=(
                        "Jointly optimizes non-reference rotations and camera centers "
                        "over adjacent/reference pairs; focal/depth scale stay near "
                        "1.0 for vertical-plane; accepts only held-out improvement."
                    ),
                )
                force_apply_optimized_camera = gr.Checkbox(
                    False,
                    label="Force apply optimized camera",
                    info=(
                        "Apply the optimizer candidate when optimization succeeds, "
                        "even if held-out reprojection error becomes worse."
                    ),
                )
                optimize_target_rotation = gr.Checkbox(
                    False,
                    label="Optimize target rotation for minimum black area",
                    info=(
                        "Starts from the selected first/mid/avg target and the current "
                        "Rotate XYZ values. Only target rotation is optimized; source "
                        "cameras, target translation/intrinsics, and depth stay fixed."
                    ),
                )
                auto_canvas = gr.Checkbox(
                    False,
                    label="Auto canvas size (tight bounds; may downscale above 8MP)",
                )
                filter_canvas_outliers = gr.Checkbox(
                    True,
                    label="Filter canvas-bound outliers (0.1% tails)",
                    info="Only affects Auto canvas boundary estimation; points are not deleted.",
                )
                visibility_aware = gr.Checkbox(
                    False,
                    label="Visibility-aware z-buffer fusion",
                    info=(
                        "Optional depth-aware compositing: nearer target-camera surfaces "
                        "are kept instead of averaging through duplicated patches."
                    ),
                )
                auto_canvas_downscale = gr.Checkbox(
                    False,
                    label="Downscale Auto canvas to a pixel budget",
                )
                auto_canvas_max_mp = gr.Number(
                    value=8,
                    minimum=1,
                    precision=1,
                    label="Auto canvas budget (megapixels)",
                )
                with gr.Row():
                    canvas_width = gr.Number(value=2048, minimum=1, precision=0, label="Canvas width")
                    canvas_height = gr.Number(value=1024, minimum=1, precision=0, label="Canvas height")
                gr.Markdown(
                    "Manual canvas preserves the target camera's 1:1 pixel scale and clips overflow."
                )
                gr.Markdown("### Camera adjustment (applied only to target view)")
                focal_scale = gr.Slider(0.5, 2.0, value=1.0, step=0.01, label="Focal scale")
                with gr.Row():
                    cx_offset = gr.Number(0, label="cx offset (px)")
                    cy_offset = gr.Number(0, label="cy offset (px)")
                with gr.Row():
                    rx = gr.Slider(-30, 30, value=0, step=0.5, label="Rotate X (deg)")
                    ry = gr.Slider(-30, 30, value=0, step=0.5, label="Rotate Y (deg)")
                    rz = gr.Slider(-30, 30, value=0, step=0.5, label="Rotate Z (deg)")
                with gr.Row():
                    tx = gr.Number(0, label="Translate X (camera right)")
                    ty = gr.Number(0, label="Translate Y (camera down)")
                    tz = gr.Number(0, label="Translate Z (camera forward)")
                render_button = gr.Button("Generate pre-stitch", variant="primary")
                gr.Markdown("### Flux generation (uses the current pre-stitch exactly)")
                flux_prompt = gr.Textbox(
                    value="Stitch input figures. Generated results must be natural and seamless.",
                    label="Prompt",
                )
                with gr.Row():
                    flux_seed = gr.Number(value=219, precision=0, label="Seed")
                    flux_steps = gr.Slider(1, 50, value=40, step=1, label="Inference steps")
                    flux_cfg = gr.Number(value=1.0, minimum=0, precision=2, label="CFG scale")
                source_view_top_k = gr.Number(
                    value=0,
                    minimum=0,
                    precision=0,
                    label="Max source views per Flux patch (top-K)",
                    info="按当前 patch 中的 source token 数保留 top-K 个 frame；设为 0 表示不限制。",
                )
                separate_median_rope_normalization = gr.Checkbox(
                    value=False,
                    label="Separate median-depth RoPE normalization",
                    info="开启后：true-depth source RoPE 与 target/context 共用归一化原点；median/flat source RoPE 单独归一化。",
                )
                flux_model_id = gr.Textbox(
                    value="black-forest-labs/FLUX.2-klein-base-9B",
                    label="Flux transformer/tokenizer model id",
                )
                flux_lora_path = gr.Textbox(
                    value="",
                    label="Stitching LoRA path (optional, recommended)",
                    info="Use the trained stitching LoRA from the validation script for strong input conditioning.",
                )
                flux_button = gr.Button("Generate with Flux", variant="primary")
                cancel_flux_button = gr.Button("Cancel Flux", variant="stop")
            with gr.Column(scale=1):
                # WebP has a hard 16383-pixel per-dimension limit.  Skyline
                # canvases can exceed it, so use PNG for Gradio serialization
                # while keeping the full-resolution array in ``state`` for
                # Flux and coord-map generation.
                result = gr.Image(label="Pre-stitch composite", type="numpy", format="png")
                pose_viewer = gr.Model3D(
                    label="3D point cloud · blue=input · orange=reference · red=target",
                    height=520,
                    zoom_speed=0.5,
                    pan_speed=0.5,
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                )
                render_status = gr.Markdown()
                flux_result = gr.Image(label="Flux result", type="numpy", format="png")
                flux_status = gr.Markdown()

        run_button.click(
            load_case,
            inputs=[
                files, gr.State(model_id), gr.State(model_cache_dir), gr.State(device),
                process_res, geometry_source, colmap_workspace, pi3x_model_id, pi3x_root,
                vggt_omega_checkpoint, vggt_omega_root, vggt_omega_image_resolution,
            ],
            outputs=[state, preview, load_status, mask_editor],
            # Gradio 6.26 reads component values during loading-status updates.
            # For ImageEditor that uploads background/layers/composite, even
            # without change/input listeners. Keep it out of progress targets.
            show_progress_on=[load_status],
        )
        mask_frame_index.change(
            select_mask_frame,
            inputs=[state, mask_frame_index],
            outputs=[mask_editor, mask_status],
            show_progress_on=[mask_status],
        )
        # ImageEditor enables realtime blob uploads when change/input listeners
        # are attached. Keep brush edits local; only this explicit action reads
        # the editor and uploads its background, layers, and composite.
        apply_mask_button.click(
            update_input_mask,
            inputs=[state, mask_frame_index, mask_editor],
            outputs=[state, mask_status],
            show_progress_on=[mask_status],
        )
        clear_mask_button.click(
            clear_input_mask,
            inputs=[state, mask_frame_index],
            outputs=[state, mask_editor, mask_status],
            show_progress_on=[mask_status],
        )
        render_inputs = [
            state, mode, reference, render_stride, input_resize_scale,
            testtime_optimization,
            force_apply_optimized_camera,
            auto_canvas, filter_canvas_outliers,
            visibility_aware,
            auto_canvas_downscale, auto_canvas_max_mp, canvas_width, canvas_height,
            focal_scale, cx_offset, cy_offset, rx, ry, rz, tx, ty, tz,
            optimize_target_rotation,
            panini_distance,
            general_panini_compression,
            general_panini_top,
            general_panini_bottom,
        ]
        # Deliberately render only on an explicit click.  A full CPU splat can
        # process millions of samples; binding every slider's ``change`` event
        # makes ordinary parameter editing feel blocked and queues stale jobs.
        render_button.click(
            render_case,
            inputs=render_inputs,
            outputs=[result, pose_viewer, render_status],
        )
        flux_button.click(
            generate_flux,
            inputs=[
                state, flux_prompt, flux_seed, flux_steps, flux_cfg, source_view_top_k,
                separate_median_rope_normalization,
                flux_model_id, flux_lora_path, gr.State(device),
            ],
            outputs=[flux_result, flux_status],
        )
        # This listener bypasses the generation queue so cancellation can be
        # requested while a Flux tile is still running.
        cancel_flux_button.click(
            cancel_flux,
            inputs=[],
            outputs=[flux_status],
            queue=False,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--model-cache-dir", default="models/depth-anything-3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()
    demo = build_demo(args.model_id, args.model_cache_dir, args.device)
    launch_kwargs = {}
    if args.server_name:
        launch_kwargs["server_name"] = args.server_name
    if args.server_port:
        launch_kwargs["server_port"] = args.server_port
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
