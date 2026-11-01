from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from PIL import Image

from tools.scenic_demo.batch_prestitch import (
    discover_cases,
    natural_sort_paths,
    parse_args,
    run_batch,
)
from tools.scenic_demo.da3_prestitch import ViewGeometry, load_images
import tools.scenic_demo.batch_prestitch as batch_prestitch


def _mock_geometry(paths: list[Path], _args):
    images = load_images(paths)
    views = []
    for index, (path, image) in enumerate(zip(paths, images)):
        height, width = image.shape[:2]
        K = np.array(
            [[30.0, 0.0, (width - 1) / 2], [0.0, 30.0, (height - 1) / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        views.append(
            ViewGeometry(
                index=index,
                path=str(path),
                image=image,
                depth=np.full((height, width), 4.0, dtype=np.float32),
                K=K,
                K_processed=K.copy(),
                R=np.eye(3, dtype=np.float64),
                T=np.array([-0.1 * index, 0.0, 0.0], dtype=np.float64),
                processed_size=(width, height),
            )
        )
    return views, {"source": "mock"}


def test_natural_sort_paths_orders_numbered_frames():
    paths = [Path("frame10.png"), Path("frame2.png"), Path("frame1.png")]
    assert [path.name for path in natural_sort_paths(paths)] == [
        "frame1.png",
        "frame2.png",
        "frame10.png",
    ]


def test_batch_preserves_unicode_case_suffix(tmp_path):
    for name in ('20260210_053544', '20260210_053544_副本'):
        case = tmp_path / 'inputs' / name
        case.mkdir(parents=True)
        for index in range(2):
            Image.new('RGB', (32, 24), (30, 40, 50)).save(case / f'{index}.png')
    args = parse_args([
        '--input-root', str(tmp_path / 'inputs'),
        '--output-root', str(tmp_path / 'output'), '--resume',
    ])
    results = run_batch(args, geometry_loader=_mock_geometry)
    assert len(results) == 2
    assert all(result['status'] == 'success' for result in results)
    assert {Path(result['output_dir']).name for result in results} == {
        '20260210_053544', '20260210_053544_副本',
    }


def test_batch_rejects_colliding_output_names(tmp_path):
    import pytest

    for name in ('case+a', 'case=a'):
        case = tmp_path / 'inputs' / name
        case.mkdir(parents=True)
        Image.new('RGB', (32, 24)).save(case / '0.png')
    args = parse_args([
        '--input-root', str(tmp_path / 'inputs'),
        '--output-root', str(tmp_path / 'output'),
    ])
    with pytest.raises(ValueError, match='same output directory'):
        run_batch(args, geometry_loader=lambda *_: (_ for _ in ()).throw(AssertionError()))


def test_batch_parses_general_panini_parameters(tmp_path):
    args = parse_args([
        "--input-root", str(tmp_path),
        "--output-root", str(tmp_path / "output"),
        "--projection", "general_panini",
        "--general-panini-compression", "72",
        "--general-panini-top", "35",
        "--general-panini-bottom", "-20",
    ])

    assert args.projection == "general_panini"
    assert args.general_panini_compression == 72.0
    assert args.general_panini_top == 35.0
    assert args.general_panini_bottom == -20.0


def test_discover_cases_and_resumable_batch_without_numpy_artifacts(tmp_path):
    input_root = tmp_path / "inputs"
    case = input_root / "case10"
    case.mkdir(parents=True)
    for name, color in (("frame10.png", 30), ("frame2.png", 20), ("frame1.png", 10)):
        Image.fromarray(np.full((24, 32, 3), color, dtype=np.uint8)).save(case / name)
    assert discover_cases(input_root) == [case]

    output_root = tmp_path / "outputs"
    args = parse_args([
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--geometry-source", "da3",
        "--projection", "panini",
        "--panini-distance", "0.35",
        "--reference", "avg",
        "--no-filter-canvas-outliers",
        "--resume",
    ])
    first = run_batch(args, geometry_loader=_mock_geometry)
    second = run_batch(args, geometry_loader=lambda *_: (_ for _ in ()).throw(AssertionError()))

    assert first[0]["status"] == "success"
    assert second[0]["status"] == "success"
    assert first[0]["input_order"] == ["frame1.png", "frame2.png", "frame10.png"]
    assert first[0]["metrics"]["projection_mode"] == "panini"
    assert first[0]["metrics"]["panini_distance"] == 0.35
    assert (output_root / "case10" / "prestitch.png").is_file()
    assert (output_root / "summary.json").is_file()
    assert (output_root / "summary.csv").is_file()
    assert not list(output_root.rglob("*.npy"))
    assert not list(output_root.rglob("*.npz"))


def test_batch_flux_reuses_gui_generator_and_keeps_coord_maps_in_memory(
    tmp_path, monkeypatch
):
    case = tmp_path / "case"
    case.mkdir()
    for index in range(2):
        Image.fromarray(np.full((24, 32, 3), 30 + index, dtype=np.uint8)).save(
            case / f"frame{index}.png"
        )
    seen = {}

    def fake_generate_flux(state, prompt, seed, steps, cfg, top_k, separate, model, lora, device):
        metrics = state["last_render"]["metrics"]
        assert len(metrics["_coord_maps"]) == 2
        assert len(metrics["_coord_maps_depth"]) == 2
        seen.update(
            prompt=prompt, seed=seed, steps=steps, cfg=cfg, top_k=top_k,
            separate=separate, model=model, lora=lora, device=device,
        )
        shape = state["last_render"]["composite"].shape
        yield None, "Flux 模型加载中（首次运行会较慢）……"
        yield np.full(shape, 80, dtype=np.uint8), "Flux 生成中：已完成 1/1 个 patch"
        yield np.full(shape, 90, dtype=np.uint8), "Flux 完成：test"

    monkeypatch.setattr(batch_prestitch, "generate_flux", fake_generate_flux)
    output_root = tmp_path / "output"
    args = parse_args([
        "--input-root", str(case),
        "--output-root", str(output_root),
        "--projection", "rectilinear",
        "--no-filter-canvas-outliers",
        "--flux",
        "--flux-prompt", "test prompt",
        "--flux-seed", "7",
        "--flux-steps", "3",
        "--flux-source-view-top-k", "1",
        "--flux-separate-median-rope-normalization",
        "--save-flux-patches",
    ])
    result = run_batch(args, geometry_loader=_mock_geometry)[0]

    case_output = output_root / "case"
    assert result["status"] == "success"
    assert (case_output / "flux.png").is_file()
    assert (case_output / "flux_patches" / "patch_0001.png").is_file()
    assert seen == {
        "prompt": "test prompt", "seed": 7, "steps": 3, "cfg": 1.0,
        "top_k": 1, "separate": True,
        "model": "black-forest-labs/FLUX.2-klein-base-9B",
        "lora": "", "device": "cuda",
    }
    saved = json.loads((case_output / "case_result.json").read_text())
    assert saved["flux"]["patches_completed"] == 1
    assert not any(key.startswith("_") for key in saved["metrics"])
    assert not list(output_root.rglob("*.npy"))
    assert not list(output_root.rglob("*.npz"))
