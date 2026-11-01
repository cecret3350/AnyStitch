# Batch stitching

See the repository [README](../../README.md) for installation, models, commands
and outputs.

- `batch_prestitch.py`: batch orchestration, resume and reporting.
- `da3_prestitch.py`: geometry, projection, fusion and coordinate maps.
- `prestitch_ui.py`: shared model adapters, TTO, target optimization and Flux;
  also retains optional Gradio controls. Batch does not require Gradio.
- `colmap_geometry.py`: optional loading of existing COLMAP reconstructions.
- `tests/`: CPU regression tests.

The release retains batch's full-render TTO acceptance, not UI's coarse metrics.
