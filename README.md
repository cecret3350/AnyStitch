# Geometry-guided image stitching

Estimate multi-view geometry with DA3, render a pre-stitch canvas, and refine it
patch by patch with FLUX.2 Klein and a stitching LoRA. This guide covers the
GitHub source checkout and its batch inference workflow.

## 1. Get the code and create a separate environment

Clone this repository (or download and extract its source archive), then enter
the project root. Use Linux with an NVIDIA GPU and a compatible driver.
Model weights are not included in the repository.

```bash
conda create -n anystitch python=3.12 -y
conda activate anystitch
python --version
python -m pip install --upgrade pip setuptools wheel
```

**Run all subsequent commands from the project root with the `anystitch` environment activated.**

## 2. Install PyTorch and ensure it can use the GPU

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

The command above follows the recorded environment in
[runtime_versions.json](runtime_versions.json), which is not a portable lockfile.
Your NVIDIA driver must support the selected PyTorch CUDA build.

## 3. Install the project and DA3

```bash
python -m pip install -e '.[batch]'
python -m pip install --no-deps \
  'depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
```

Important details:

- DA3 is a separate Python package and must be installed; downloading this project does not install DA3.
- Keep `--no-deps` when installing DA3: `.[batch]` supplies its runtime dependencies, preventing the xFormers dependency from changing torch.
- There is no need to install xFormers, gsplat or Gradio separately; the `gsplat` warning about 3DGS does not affect this workflow.
- This project pins `transformers==4.57.1` and `peft==0.14.0`; do not upgrade them independently.
- Use the modified DiffSynth included here; do not overwrite it by installing upstream DiffSynth separately.

## 4. Verify the installation

The following commands neither download base weights nor run inference:

```bash
python - <<'PY'
import torch, transformers, diffsynth
from depth_anything_3.api import DepthAnything3
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline
print('DiffSynth loaded from:', diffsynth.__file__)
assert transformers.__version__ == '4.57.1', transformers.__version__
assert torch.cuda.is_available()
print('Imports OK')
PY
python tools/scenic_demo/batch_prestitch.py --help
```

Expect `Imports OK` and the argument help with no traceback. The DiffSynth path
should point to this checkout.

## 5. Prepare the weights and input images

### Base models

The following base models are required and are downloaded automatically under
the project's `models/` directory on first use:

- `depth-anything/DA3NESTED-GIANT-LARGE`: geometry estimation, downloaded through Hugging Face.
- `black-forest-labs/FLUX.2-klein-base-9B`: transformer/tokenizer.
- `black-forest-labs/FLUX.2-klein-9B`: text encoder/VAE; Flux downloads use DiffSynth/ModelScope.

The first download may take some time and requires network access and permission
to access the models. Do not set `HF_HUB_OFFLINE=1` before caching the weights.
Where access terms apply, obtain authorization yourself; installing Python packages
does not grant access to model weights. For an offline machine, transfer the complete
model caches in advance; copying the source code alone is not enough.

### Training data and stitching LoRA

Both the **training data and stitching LoRA** are available in this
[Google Drive folder](https://drive.google.com/drive/folders/1mZIELHAqLJ7cP-nuhiv4f6KY0n-cNgH1?usp=sharing):

- `anystitch.safetensors`: the stitching LoRA from
  `stitch_v17.3.2.2_flux_klein_9b_lora_r32_try2`.
- `training_data.tar.gz`: the training data archive, with
  `equirectangular_hr/`, `cylindrical_hr/` and `rectilinear_hr/` at its top level.

Training data is not required for batch inference. Access to these files does
not replace applicable source-data or model license requirements.

Obtain the checkpoint before running Flux; the base Flux model cannot replace
this LoRA. Save it at `models/stitch_lora/anystitch.safetensors`, or pass its actual path as
the third argument to `scripts/run_batch.sh`. The script does not download the
LoRA automatically. Pre-stitch-only inference does not require the LoRA or Flux weights.

### Input images

Place each scene in a separate subdirectory with at least two images, using
zero-padded filenames in capture order. `/data/stitch_inputs` below is only an
example; replace it with the actual directory on your machine.

```text
/data/stitch_inputs/
  scene_001/
    frame_000.jpg
    frame_001.jpg
  scene_002/
    frame_000.jpg
    frame_001.jpg
```

If the root contains images directly, it is treated as a single scene; otherwise,
its immediate subdirectories are scanned. Use `--case-glob '**/*'` for deeper
directories. Do not place output images in input directories, as they may be
treated as inputs. Batch processing uses natural numeric sorting. TTO uses
adjacent-frame matching and a weak sequence-smoothness constraint, so do not
arbitrarily reorder frames.

## 6. Run one scene first, then all scenes

Start with a small scene containing only two images to verify the full pipeline.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_batch.sh \
  /data/stitch_inputs/scene_001 /data/stitch_results/first_scene \
  models/stitch_lora/anystitch.safetensors panini
```

Replace all example paths. After a successful run, the output directory's
`scene_001/` subdirectory should contain `prestitch.png`, `flux.png` and
`case_result.json`. Flux prints progress for each patch.

To process all scenes:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_batch.sh \
  /data/stitch_inputs /data/stitch_results/panini \
  models/stitch_lora/anystitch.safetensors panini
```

The last argument can be changed to `cylindrical` or `equirectangular`.
With three available GPUs, run the commands in three separate terminals
(with only one GPU, run them sequentially, using GPU 0 for each):

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_batch.sh \
  /data/stitch_inputs /data/stitch_results/panini \
  models/stitch_lora/anystitch.safetensors panini

# Terminal 2
CUDA_VISIBLE_DEVICES=1 bash scripts/run_batch.sh \
  /data/stitch_inputs /data/stitch_results/cylindrical \
  models/stitch_lora/anystitch.safetensors cylindrical

# Terminal 3
CUDA_VISIBLE_DEVICES=2 bash scripts/run_batch.sh \
  /data/stitch_inputs /data/stitch_results/equirectangular \
  models/stitch_lora/anystitch.safetensors equirectangular
```

### Preset settings and customization

`scripts/run_batch.sh` uses DA3 (`process-res=504`), original-resolution inputs
(`input-resize-scale=1`), render stride 1, and the average target view. It enables
canvas-bound outlier filtering (0.1% tails), automatic canvas sizing, multi-view
TTO and target-rotation optimization for minimum black area. Visibility-aware
fusion is disabled. TTO keeps depth fixed and applies a candidate only if accepted;
target-rotation optimization runs afterwards.

Flux uses the supplied LoRA, 40 steps, CFG 1 and seed 219. Source-view top-K is 0
(no top-K limit), and separate median-depth RoPE normalization is disabled.
Panini uses distance 1 and the default General Panini controls.

The wrapper accepts only input root, output root, LoRA path and projection. For
other settings, invoke `python tools/scenic_demo/batch_prestitch.py` directly and
consult `--help`; direct CLI defaults are not identical to the wrapper preset.
For example, to generate only pre-stitch images with the same geometry settings:

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/scenic_demo/batch_prestitch.py \
  --input-root /data/stitch_inputs \
  --output-root /data/stitch_results/prestitch_only \
  --geometry-source da3 --device cuda --process-res 504 \
  --projection panini --panini-distance 1 --reference avg \
  --input-resize-scale 1 --render-stride 1 \
  --filter-canvas-outliers --no-visibility-aware \
  --tto --optimize-target-rotation --resume
```

## 7. Outputs, resume and troubleshooting

Each scene has its own output directory:

- `prestitch.png`: the rendered canvas before Flux.
- `flux.png`: the completed Flux result (only with Flux enabled).
- `case_result.json`: success/failure status, settings-related metadata, timings,
  TTO/target-rotation metrics and any reported error.

The output root also contains `summary.json` and `summary.csv`. Check these for
failed cases, even if other scenes completed successfully.

The wrapper enables `--resume`: scenes with a successful result and matching
configuration hash are skipped. It does not resume within a partially generated
scene. Input contents and checkpoint contents are not hashed; if you replace
files at the same paths, use a fresh output directory or call the CLI without
`--resume`.

- First-run delays can be model downloads; check network access and model permissions.
- For CUDA out-of-memory errors, test a small scene and reduce
  `--input-resize-scale` through the direct CLI. `--process-res` controls DA3's
  processing resolution, not the source image resolution used for stitching.
- Automatic canvases have no megapixel cap by default. Use
  `--max-canvas-megapixels` explicitly if you want to limit the pre-stitch canvas.
- Gradio is optional and not required for batch inference. The shared UI/adapters
  are in `tools/scenic_demo/prestitch_ui.py`; install `.[ui]` to use the UI.

## 8. Development and acknowledgements

Run the CPU regression tests after installing the development dependencies:

```bash
python -m pip install -e '.[batch,dev]'
python -m pytest tools/scenic_demo/tests -q
```

This repository includes modified [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
code and uses [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
and [FLUX.2 Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B).
See [LICENSE](LICENSE) for the repository code license and retain upstream notices.
Model weights have their own licenses and access terms; the code license does
not override them.
