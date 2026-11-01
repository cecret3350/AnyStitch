#!/usr/bin/env python3
"""Sequential, resumable batch runner for the interactive pre-stitch pipeline.

The runner intentionally does not save depth, camera, or coordinate arrays.
Geometry models are cached by ``prestitch_ui`` and therefore loaded only once
per process while all cases are handled sequentially on one GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

if __package__ in {None, ""}:  # Support direct execution from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.scenic_demo.da3_prestitch import (  # noqa: E402
    PROJECTION_MODES,
    REFERENCE_MODES,
    ViewGeometry,
    discover_images,
    load_images,
    prepare_views,
    prediction_to_arrays,
    render_prestitch_memory,
    unproject_views,
)
from tools.scenic_demo.prestitch_ui import (  # noqa: E402
    _apply_tto_acceptance,
    _edit_target_camera,
    _optimize_target_rotation,
    _refine_multiview_camera,
    _resize_views,
    _run_da3,
    _run_pi3x,
    _run_vggt_omega,
    generate_flux,
)
from tools.scenic_demo.colmap_geometry import load_colmap_views  # noqa: E402


GeometryLoader = Callable[
    [Sequence[Path], argparse.Namespace],
    tuple[list[ViewGeometry], dict[str, Any]],
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_AUXILIARY_STEMS = {"contact", "contact_sheet", "preview", "thumbnail"}
_RUNNER_VERSION = 3


def natural_sort_key(value: str | Path) -> tuple[tuple[int, Any], ...]:
    """Sort numbered frames as 1, 2, 10 while remaining deterministic."""

    pieces = re.split(r"(\d+)", str(value).lower())
    return tuple((0, int(piece)) if piece.isdigit() else (1, piece) for piece in pieces)


def natural_sort_paths(paths: Sequence[str | Path]) -> list[Path]:
    return sorted((Path(path) for path in paths), key=lambda path: natural_sort_key(path.name))


def _contains_images(path: Path) -> bool:
    return path.is_dir() and any(
        item.is_file()
        and item.suffix.lower() in _IMAGE_SUFFIXES
        and item.stem.lower() not in _AUXILIARY_STEMS
        for item in path.iterdir()
    )


def discover_cases(input_root: str | Path, case_glob: str = "*") -> list[Path]:
    """Return either the root itself or matching image-containing directories."""

    root = Path(input_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    if _contains_images(root):
        return [root]
    cases = [path for path in root.glob(case_glob) if _contains_images(path)]
    return sorted(cases, key=lambda path: natural_sort_key(str(path.relative_to(root))))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, ensure_ascii=False, sort_keys=True)


def _case_slug(case: Path, input_root: Path) -> str:
    if case.resolve() == input_root.resolve():
        raw = case.name
    else:
        raw = "__".join(case.relative_to(input_root).parts)
    # Preserve Unicode letters: e.g. a Chinese "copy" suffix must not collapse
    # onto the original case name and silently overwrite/skip that case.
    slug = re.sub(r"[^\w.-]+", "_", raw).strip("._")
    return slug or "case"


def _config(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"input_root", "output_root", "resume", "fail_fast", "max_cases"}
    return {
        "runner_version": _RUNNER_VERSION,
        **{key: value for key, value in vars(args).items() if key not in excluded},
    }


def _config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(_json_ready(_config(args)), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_colmap_workspace(template: str, case: Path) -> str:
    return str(template).replace("{case}", case.name).replace("{case_path}", str(case))


def load_geometry(
    paths: Sequence[Path], args: argparse.Namespace
) -> tuple[list[ViewGeometry], dict[str, Any]]:
    """Load one case through the same model adapters used by the Gradio demo."""

    string_paths = [str(path) for path in paths]
    images = load_images(paths)
    source = args.geometry_source
    if source == "da3":
        prediction = _run_da3(
            string_paths,
            args.model_id,
            args.model_cache_dir or None,
            args.device or None,
            int(args.process_res),
        )
        arrays = prediction_to_arrays(prediction)
        views = prepare_views(images, SimpleNamespace(**arrays), paths)
        info: dict[str, Any] = {"source": "DA3"}
    elif source == "pi3x":
        views = _run_pi3x(
            images, string_paths, args.pi3x_model_id, args.pi3x_root, args.device or None
        )
        info = {"source": "Pi3X"}
    elif source == "vggt-omega":
        views = _run_vggt_omega(
            images,
            string_paths,
            args.vggt_omega_checkpoint,
            args.vggt_omega_root,
            args.device or None,
            int(args.vggt_omega_image_resolution),
        )
        info = {"source": "VGGT-Omega"}
    elif source == "colmap":
        workspace = _resolve_colmap_workspace(args.colmap_workspace, paths[0].parent)
        if not workspace:
            raise ValueError("--colmap-workspace is required for COLMAP geometry")
        views, colmap_info = load_colmap_views(images, paths, workspace)
        info = {"source": "COLMAP", "workspace": workspace, **colmap_info}
    else:  # argparse choices should make this unreachable.
        raise ValueError(f"Unknown geometry source: {source}")
    return views, info


def _render(
    views: Sequence[ViewGeometry],
    args: argparse.Namespace,
    rotation_degrees: Sequence[float] | None = None,
    return_coord_maps: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    rx, ry, rz = (
        tuple(float(value) for value in rotation_degrees)
        if rotation_degrees is not None
        else (float(args.rx), float(args.ry), float(args.rz))
    )
    target_camera = _edit_target_camera(
        views,
        args.reference,
        args.focal_scale,
        rx,
        ry,
        rz,
        args.tx,
        args.ty,
        args.tz,
        args.cx_offset,
        args.cy_offset,
    )
    samples = unproject_views(views, int(args.render_stride))
    canvas_size = None
    if args.canvas_width is not None or args.canvas_height is not None:
        if args.canvas_width is None or args.canvas_height is None:
            raise ValueError("--canvas-width and --canvas-height must be supplied together")
        canvas_size = (int(args.canvas_width), int(args.canvas_height))
    max_canvas_pixels = (
        max(1, int(args.max_canvas_megapixels * 1_000_000))
        if args.max_canvas_megapixels is not None
        else None
    )
    composite, _heatmap, metrics = render_prestitch_memory(
        views,
        mode=args.projection,
        reference_mode=args.reference,
        render_stride=max(1, int(args.render_stride)),
        bounds_quantile=0.001 if args.filter_canvas_outliers else 0.0,
        padding=0,
        max_canvas_pixels=max_canvas_pixels,
        canvas_size=canvas_size,
        unprojected=samples,
        target_camera=target_camera,
        visibility_aware=bool(args.visibility_aware),
        return_coord_maps=bool(return_coord_maps),
        return_heatmap=False,
        panini_distance=float(args.panini_distance),
        general_panini_compression=float(args.general_panini_compression),
        general_panini_top=float(args.general_panini_top),
        general_panini_bottom=float(args.general_panini_bottom),
    )
    return composite, metrics


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop in-memory coord-map arrays before serializing case metadata."""

    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, format="PNG")


def _run_flux_for_case(
    case_dir: Path,
    views: Sequence[ViewGeometry],
    composite: np.ndarray,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Consume the exact Gradio Flux generator and persist its final result."""

    state = {
        "views": list(views),
        "last_render": {
            "composite": np.asarray(composite, dtype=np.uint8),
            "metrics": metrics,
            "mode": args.projection,
            "reference": args.reference,
        },
    }
    started = time.perf_counter()
    statuses: list[str] = []
    patch_count = 0
    completed = False
    final_image: np.ndarray | None = None
    flux_device = args.flux_device or args.device
    for image, status in generate_flux(
        state,
        args.flux_prompt,
        int(args.flux_seed),
        int(args.flux_steps),
        float(args.flux_cfg_scale),
        int(args.flux_source_view_top_k),
        bool(args.flux_separate_median_rope_normalization),
        args.flux_model_id,
        args.flux_lora_path,
        flux_device,
    ):
        status = str(status)
        statuses.append(status)
        print(f"             {status}", flush=True)
        if image is not None:
            final_image = np.asarray(image, dtype=np.uint8)
        if status.startswith("Flux 生成中") and image is not None:
            patch_count += 1
            if args.save_flux_patches:
                _save_png(
                    case_dir / "flux_patches" / f"patch_{patch_count:04d}.png",
                    final_image,
                )
        if status.startswith("Flux 完成") and image is not None:
            completed = True
    if not completed or final_image is None:
        detail = statuses[-1] if statuses else "Flux generator returned no updates"
        raise RuntimeError(detail)
    output_path = case_dir / "flux.png"
    _save_png(output_path, final_image)
    return {
        "status": "success",
        "output": str(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "patches_completed": patch_count,
        "saved_patch_snapshots": bool(args.save_flux_patches),
        "prompt": args.flux_prompt,
        "seed": int(args.flux_seed),
        "steps": int(args.flux_steps),
        "cfg_scale": float(args.flux_cfg_scale),
        "source_view_top_k": int(args.flux_source_view_top_k),
        "separate_median_rope_normalization": bool(
            args.flux_separate_median_rope_normalization
        ),
        "model_id": args.flux_model_id,
        "lora_path": args.flux_lora_path or None,
        "device": flux_device,
        "statuses": statuses,
    }


def process_case(
    case: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    geometry_loader: GeometryLoader = load_geometry,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_dir = output_root / _case_slug(case, input_root)
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = natural_sort_paths(discover_images(case))
    if len(paths) < 2:
        raise ValueError(f"At least two frames are required; found {len(paths)} in {case}")
    _write_json(
        case_dir / "input_order.json",
        {"case": str(case), "frames": [str(path) for path in paths]},
    )

    geometry_started = time.perf_counter()
    base_views, geometry_info = geometry_loader(paths, args)
    views = _resize_views(base_views, float(args.input_resize_scale))
    geometry_seconds = time.perf_counter() - geometry_started

    tto_stats: dict[str, Any] | None = None
    selected_views = views
    if args.tto:
        before_image, before_metrics = _render(views, args)
        _save_png(case_dir / "before_tto.png", before_image)
        reference_index = 0 if args.reference == "first" else len(views) // 2
        try:
            candidate_views, tto_stats = _refine_multiview_camera(views, reference_index)
            candidate_image, candidate_metrics = _render(candidate_views, args)
            _save_png(case_dir / "after_tto.png", candidate_image)
            tto_stats = _apply_tto_acceptance(
                tto_stats, before_metrics, candidate_metrics
            )
            apply_candidate = bool(
                tto_stats.get("accepted")
                or (args.force_apply_tto and tto_stats.get("optimizer_success"))
            )
            if apply_candidate:
                selected_views, composite, metrics = candidate_views, candidate_image, candidate_metrics
            else:
                composite, metrics = before_image, before_metrics
            comparison = {
                "applied": apply_candidate,
                "forced": bool(args.force_apply_tto and not tto_stats.get("accepted")),
                "stats": tto_stats,
                "before_metrics": before_metrics,
                "candidate_metrics": candidate_metrics,
            }
        except Exception as exc:
            # Match the UI behavior: an unoptimizable case still gets a valid
            # pre-stitch from the original geometry and records why TTO skipped.
            composite, metrics = before_image, before_metrics
            tto_stats = {"skipped": True, "error": f"{type(exc).__name__}: {exc}"}
            comparison = {
                "applied": False,
                "forced": False,
                "stats": tto_stats,
                "before_metrics": before_metrics,
                "candidate_metrics": None,
            }
        _write_json(case_dir / "tto_comparison.json", comparison)
    else:
        composite, metrics = _render(selected_views, args)

    target_rotation_stats = None
    optimized_rotation: Sequence[float] | None = None
    if args.optimize_target_rotation:
        try:
            optimized_rotation, target_rotation_stats = _optimize_target_rotation(
                selected_views,
                args.projection,
                args.reference,
                (args.rx, args.ry, args.rz),
                args.focal_scale,
                args.cx_offset,
                args.cy_offset,
                args.tx,
                args.ty,
                args.tz,
                bounds_quantile=0.001 if args.filter_canvas_outliers else 0.0,
                panini_distance=float(args.panini_distance),
                general_panini_compression=float(args.general_panini_compression),
                general_panini_top=float(args.general_panini_top),
                general_panini_bottom=float(args.general_panini_bottom),
            )
        except Exception as exc:
            optimized_rotation = None
            target_rotation_stats = {
                "skipped": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Flux needs the exact coord maps of the final selected geometry, target
    # camera, and canvas. Generate them only for the final render and keep them
    # in memory, exactly as render_case() does for the Gradio state.
    if args.optimize_target_rotation or args.flux:
        composite, metrics = _render(
            selected_views,
            args,
            optimized_rotation,
            return_coord_maps=bool(args.flux),
        )

    _save_png(case_dir / "prestitch.png", composite)
    flux_info = None
    if args.flux:
        flux_info = _run_flux_for_case(
            case_dir, selected_views, composite, metrics, args
        )
    elapsed = time.perf_counter() - started
    result = {
        "status": "success",
        "case": str(case),
        "output_dir": str(case_dir),
        "prestitch": str(case_dir / "prestitch.png"),
        "num_frames": len(paths),
        "input_order": [path.name for path in paths],
        "geometry": geometry_info,
        "metrics": _public_metrics(metrics),
        "tto": tto_stats,
        "target_rotation_optimization": target_rotation_stats,
        "flux": flux_info,
        "flux_output": flux_info["output"] if flux_info else None,
        "geometry_seconds": geometry_seconds,
        "elapsed_seconds": elapsed,
        "config_hash": _config_hash(args),
    }
    _write_json(case_dir / "case_result.json", result)
    return result


def _write_summary(output_root: Path, results: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    successful = sum(result.get("status") == "success" for result in results)
    failed = sum(result.get("status") == "failed" for result in results)
    _write_json(
        output_root / "summary.json",
        {
            "config": _config(args),
            "config_hash": _config_hash(args),
            "successful": successful,
            "failed": failed,
            "cases": list(results),
        },
    )
    fields = [
        "status", "case", "num_frames", "prestitch", "flux_output",
        "canvas_width", "canvas_height", "prestitch_score", "coverage",
        "elapsed_seconds", "error",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result.get("metrics") or {}
            canvas = metrics.get("canvas_size") or [None, None]
            writer.writerow({
                "status": result.get("status"),
                "case": result.get("case"),
                "num_frames": result.get("num_frames"),
                "prestitch": result.get("prestitch"),
                "flux_output": result.get("flux_output"),
                "canvas_width": canvas[0],
                "canvas_height": canvas[1],
                "prestitch_score": metrics.get("prestitch_score"),
                "coverage": metrics.get("canvas_coverage_ratio"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            })


def run_batch(
    args: argparse.Namespace, geometry_loader: GeometryLoader = load_geometry
) -> list[dict[str, Any]]:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(input_root, args.case_glob)
    # An output directory nested below input-root may itself contain PNG
    # results from a previous run; never rediscover it as a new input case.
    cases = [case for case in cases if not case.resolve().is_relative_to(output_root)]
    if args.max_cases is not None:
        cases = cases[: max(0, int(args.max_cases))]
    if not cases:
        raise ValueError(f"No image cases found below {input_root} with glob {args.case_glob!r}")

    output_names: dict[str, Path] = {}
    for case in cases:
        slug = _case_slug(case, input_root)
        if slug in output_names:
            raise ValueError(
                f"Cases map to the same output directory {slug!r}: "
                f"{output_names[slug]} and {case}"
            )
        output_names[slug] = case

    results: list[dict[str, Any]] = []
    config_hash = _config_hash(args)
    print(f"Found {len(cases)} case(s); geometry model will be reused in this process.", flush=True)
    for index, case in enumerate(cases, start=1):
        case_dir = output_root / _case_slug(case, input_root)
        result_path = case_dir / "case_result.json"
        if args.resume and result_path.is_file():
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous.get("status") == "success" and previous.get("config_hash") == config_hash:
                    results.append(previous)
                    print(f"[{index}/{len(cases)}] skip {case.name} (already complete)", flush=True)
                    _write_summary(output_root, results, args)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        print(f"[{index}/{len(cases)}] run  {case}", flush=True)
        started = time.perf_counter()
        try:
            result = process_case(case, input_root, output_root, args, geometry_loader)
            metrics = result["metrics"]
            canvas = metrics.get("canvas_size", ["?", "?"])
            print(
                f"             ok {canvas[0]}x{canvas[1]}, "
                f"score={metrics.get('prestitch_score', float('nan')):.1f}, "
                f"{result['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "case": str(case),
                "output_dir": str(case_dir),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
                "config_hash": config_hash,
            }
            _write_json(case_dir / "case_result.json", result)
            print(f"             FAILED {result['error']}", flush=True)
            if args.fail_fast:
                results.append(result)
                _write_summary(output_root, results, args)
                raise
        results.append(result)
        _write_summary(output_root, results, args)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="One case or a root of case directories")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--case-glob", default="*", help="Directory glob below input-root")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--geometry-source", choices=("da3", "pi3x", "vggt-omega", "colmap"), default="da3"
    )
    parser.add_argument("--projection", choices=PROJECTION_MODES, default="vertical_plane")
    parser.add_argument(
        "--panini-distance",
        type=float,
        default=1.0,
        help="Panini d in [0, 1]: 0 is rectilinear, 1 is classic Panini",
    )
    parser.add_argument(
        "--general-panini-compression",
        type=float,
        default=100.0,
        help="General Panini Cmpr in [0, 150]",
    )
    parser.add_argument(
        "--general-panini-top",
        type=float,
        default=0.0,
        help="General Panini Tops squeeze in [-100, 100]",
    )
    parser.add_argument(
        "--general-panini-bottom",
        type=float,
        default=0.0,
        help="General Panini Bots squeeze in [-100, 100]",
    )
    parser.add_argument("--reference", choices=REFERENCE_MODES, default="avg")
    parser.add_argument("--input-resize-scale", type=float, default=1.0)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument(
        "--filter-canvas-outliers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--visibility-aware", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--max-canvas-megapixels", type=float, default=None,
        help="Optional auto-canvas downscale limit; omitted means no limit",
    )
    parser.add_argument("--canvas-width", type=int, default=None)
    parser.add_argument("--canvas-height", type=int, default=None)

    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--cx-offset", type=float, default=0.0)
    parser.add_argument("--cy-offset", type=float, default=0.0)
    parser.add_argument("--rx", type=float, default=0.0)
    parser.add_argument("--ry", type=float, default=0.0)
    parser.add_argument("--rz", type=float, default=0.0)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--tz", type=float, default=0.0)

    parser.add_argument("--tto", action="store_true", help="Run fixed-depth multiview TTO")
    parser.add_argument(
        "--force-apply-tto", action="store_true",
        help="Apply a converged TTO candidate even when held-out validation rejects it",
    )
    parser.add_argument(
        "--optimize-target-rotation",
        action="store_true",
        help="Optimize target XYZ rotation for minimum tight-canvas black area",
    )
    parser.add_argument("--model-id", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--model-cache-dir", default="models/depth-anything-3")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pi3x-model-id", default="yyfz233/Pi3X")
    parser.add_argument("--pi3x-root", default="third_party/Pi3")
    parser.add_argument(
        "--vggt-omega-checkpoint",
        default="models/VGGT-Omega/vggt_omega_1b_512.pt",
    )
    parser.add_argument("--vggt-omega-root", default="third_party/vggt-omega")
    parser.add_argument("--vggt-omega-image-resolution", type=int, default=512)
    parser.add_argument(
        "--colmap-workspace", default="",
        help="Workspace path; {case} and {case_path} placeholders are supported",
    )
    flux_group = parser.add_argument_group("Flux generation (identical to Gradio)")
    flux_group.add_argument("--flux", action="store_true", help="Run Flux after pre-stitch")
    flux_group.add_argument(
        "--flux-prompt",
        default="Stitch input figures. Generated results must be natural and seamless.",
    )
    flux_group.add_argument("--flux-seed", type=int, default=219)
    flux_group.add_argument("--flux-steps", type=int, default=40)
    flux_group.add_argument("--flux-cfg-scale", type=float, default=1.0)
    flux_group.add_argument("--flux-source-view-top-k", type=int, default=0)
    flux_group.add_argument(
        "--flux-separate-median-rope-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    flux_group.add_argument(
        "--flux-model-id", default="black-forest-labs/FLUX.2-klein-base-9B"
    )
    flux_group.add_argument("--flux-lora-path", default="")
    flux_group.add_argument(
        "--flux-device", default=None,
        help="Defaults to --device; normally cuda or cuda:0",
    )
    flux_group.add_argument(
        "--save-flux-patches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save each cumulative patch update; disabled by default to limit disk usage",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.render_stride < 1:
        parser.error("--render-stride must be at least 1")
    if not np.isfinite(args.input_resize_scale) or args.input_resize_scale <= 0:
        parser.error("--input-resize-scale must be a positive finite number")
    if not np.isfinite(args.panini_distance) or not 0.0 <= args.panini_distance <= 1.0:
        parser.error("--panini-distance must be finite and in [0, 1]")
    if (
        not np.isfinite(args.general_panini_compression)
        or not 0.0 <= args.general_panini_compression <= 150.0
    ):
        parser.error("--general-panini-compression must be finite and in [0, 150]")
    for name in ("general_panini_top", "general_panini_bottom"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not -100.0 <= value <= 100.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and in [-100, 100]")
    if (args.canvas_width is None) != (args.canvas_height is None):
        parser.error("--canvas-width and --canvas-height must be supplied together")
    if args.canvas_width is not None and (args.canvas_width < 1 or args.canvas_height < 1):
        parser.error("manual canvas dimensions must be positive")
    if args.max_canvas_megapixels is not None and args.max_canvas_megapixels <= 0:
        parser.error("--max-canvas-megapixels must be positive")
    if args.force_apply_tto and not args.tto:
        parser.error("--force-apply-tto requires --tto")
    if args.flux_steps < 1:
        parser.error("--flux-steps must be at least 1")
    if args.flux_cfg_scale < 0:
        parser.error("--flux-cfg-scale must be non-negative")
    if args.flux_source_view_top_k < 0:
        parser.error("--flux-source-view-top-k must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_batch(args)
    failed = sum(result.get("status") == "failed" for result in results)
    print(f"Done: {len(results) - failed} successful, {failed} failed.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
