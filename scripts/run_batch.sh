#!/usr/bin/env bash
# One projection per process/GPU. Run from anywhere; outputs resolve at repo root.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'Usage: bash scripts/run_batch.sh INPUT_ROOT OUTPUT_ROOT LORA_PATH [PROJECTION]' >&2
  exit 2
fi
exec python -u tools/scenic_demo/batch_prestitch.py \
  --input-root "$1" --output-root "$2" \
  --geometry-source da3 --device cuda --process-res 504 \
  --projection "${4:-panini}" --panini-distance 1 \
  --reference avg --input-resize-scale 1 --render-stride 1 \
  --filter-canvas-outliers --no-visibility-aware \
  --tto --optimize-target-rotation \
  --flux --flux-lora-path "$3" \
  --flux-model-id black-forest-labs/FLUX.2-klein-base-9B \
  --flux-seed 219 --flux-steps 40 --flux-cfg-scale 1 \
  --flux-source-view-top-k 0 --no-flux-separate-median-rope-normalization \
  --resume
