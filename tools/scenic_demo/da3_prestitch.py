#!/usr/bin/env python3
"""Depth Anything 3 pre-stitching without loading a diffusion model.

The module deliberately imports neither DiffSynth nor torch.  DA3 is imported
only inside :func:`run_da3`, which also makes the geometry and rendering code
usable in CPU-only tests with a mock prediction.

DA3 camera convention
---------------------
This follows the convention used by the repository's stitching experiments::

    point_world = (point_camera - T) @ R
    camera_center_world = -T @ R

Every view keeps its own ``prediction.intrinsics[i]``.  Its first and second
rows are scaled independently from the DA3 processed-image resolution to that
view's original resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


PROJECTION_MODES = (
    "vertical_plane",
    "pushbroom",
    "rectilinear",
    "panini",
    "general_panini",
    "cylindrical",
    "equirectangular",
)
REFERENCE_MODES = ("first", "mid", "avg")
CACHE_VERSION = 1

# The classic Panini projection.  d=0 is rectilinear and increasing d
# progressively compresses the horizontal edges; d=1 is the usual balanced
# choice for wide architectural panoramas.
PANINI_DISTANCE = 1.0
GENERAL_PANINI_COMPRESSION = 100.0
GENERAL_PANINI_TOP_SQUEEZE = 0.0
GENERAL_PANINI_BOTTOM_SQUEEZE = 0.0
GENERAL_PANINI_MAX_PROJECTION_ANGLE_DEGREES = 80.0


@dataclass
class ViewGeometry:
    """One original RGB view and its DA3 geometry at original resolution."""

    index: int
    path: str
    image: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    K_processed: np.ndarray
    R: np.ndarray
    T: np.ndarray
    processed_size: tuple[int, int]
    confidence: np.ndarray | None = None

    @property
    def camera_center(self) -> np.ndarray:
        return -self.T @ self.R


@dataclass
class ProjectedView:
    """Projected samples before rasterization."""

    image: np.ndarray
    depth: np.ndarray
    uv: np.ndarray
    colors: np.ndarray
    feather: np.ndarray
    # Depth in the editable target-camera coordinate system.  ``depth`` keeps
    # the original source-camera depth for diagnostics/coord-map compatibility;
    # visibility decisions must use this target-space depth instead.
    target_depth: np.ndarray | None = None
    source_xy: np.ndarray | None = None
    flat_uv: np.ndarray | None = None
    flat_source_xy: np.ndarray | None = None


def _to_numpy(value: Any) -> np.ndarray:
    """Convert numpy/torch-like values without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def discover_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    # Prepared-frame directories may also contain a visual audit sheet.  It is
    # not a camera view and must never be passed to DA3 as an extra input.
    auxiliary_stems = {"contact", "contact_sheet", "preview", "thumbnail"}
    paths = sorted(
        p
        for p in path.iterdir()
        if p.suffix.lower() in suffixes and p.stem.lower() not in auxiliary_stems
    )
    if not paths:
        raise ValueError(f"No supported images found in {path}")
    return paths


def load_images(paths: Sequence[str | Path]) -> list[np.ndarray]:
    images = []
    for path in paths:
        with Image.open(path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            images.append(np.asarray(rgb).copy())
    return images


def _input_signature(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for item in paths:
        path = Path(item)
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def prediction_to_arrays(prediction: Any) -> dict[str, np.ndarray]:
    required = ("processed_images", "depth", "intrinsics", "extrinsics")
    missing = [name for name in required if not hasattr(prediction, name)]
    if missing:
        raise ValueError(f"DA3 prediction is missing: {', '.join(missing)}")
    arrays = {name: _to_numpy(getattr(prediction, name)) for name in required}
    if hasattr(prediction, "conf") and getattr(prediction, "conf") is not None:
        arrays["conf"] = _to_numpy(getattr(prediction, "conf"))
    return arrays


def save_prediction_cache(
    cache_path: str | Path,
    prediction: Any,
    image_paths: Sequence[str | Path],
    model_id: str,
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = prediction_to_arrays(prediction)
    metadata = {
        "cache_version": CACHE_VERSION,
        "input_signature": _input_signature(image_paths),
        "images": [str(Path(path).resolve()) for path in image_paths],
        "model_id": model_id,
    }
    np.savez_compressed(
        cache_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_prediction_cache(
    cache_path: str | Path,
    image_paths: Sequence[str | Path] | None = None,
) -> SimpleNamespace:
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        if metadata.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"Unsupported prediction cache version: {metadata}")
        if image_paths is not None:
            signature = _input_signature(image_paths)
            if metadata.get("input_signature") != signature:
                raise ValueError(
                    "Prediction cache does not match the input images. "
                    "Use --force-prediction to replace it."
                )
        arrays = {
            name: np.asarray(cache[name])
            for name in ("processed_images", "depth", "intrinsics", "extrinsics")
        }
    return SimpleNamespace(**arrays, metadata=metadata)


def run_da3(
    image_paths: Sequence[str | Path],
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    model_cache_dir: str | None = None,
    device: str | None = None,
    process_res: int = 504,
) -> Any:
    """Run joint multi-view DA3 inference; the import is intentionally lazy."""

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "depth_anything_3 is required only for a cache miss. Install DA3 or "
            "provide an existing --prediction-cache."
        ) from exc

    load_kwargs: dict[str, Any] = {}
    if model_cache_dir:
        load_kwargs["cache_dir"] = model_cache_dir
    model = DepthAnything3.from_pretrained(model_id, **load_kwargs)
    if device and hasattr(model, "to"):
        model = model.to(device)
    # Passing the complete list in one call is essential: poses and depths are
    # jointly inferred rather than estimated independently per frame.
    return model.inference(
        [str(Path(path)) for path in image_paths],
        process_res=process_res,
    )


def get_or_run_prediction(
    image_paths: Sequence[str | Path],
    cache_path: str | Path,
    model_id: str,
    force: bool = False,
    inference_fn: Callable[[Sequence[str | Path]], Any] | None = None,
) -> tuple[Any, bool]:
    """Return ``(prediction, cache_hit)`` and persist cache misses."""

    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_prediction_cache(cache_path, image_paths), True
    if inference_fn is None:
        def default_inference(paths: Sequence[str | Path]) -> Any:
            return run_da3(paths, model_id=model_id)

        inference_fn = default_inference
    prediction = inference_fn(image_paths)
    save_prediction_cache(cache_path, prediction, image_paths, model_id)
    return prediction, False


def _view_axis(array: np.ndarray, expected_views: int, name: str) -> list[np.ndarray]:
    array = np.asarray(array)
    if array.shape[0] != expected_views:
        raise ValueError(
            f"DA3 returned {array.shape[0]} {name} records for {expected_views} views"
        )
    return [np.asarray(array[index]) for index in range(expected_views)]


def _processed_hw(processed_image: np.ndarray) -> tuple[int, int]:
    item = np.squeeze(np.asarray(processed_image))
    if item.ndim == 2:
        return int(item.shape[0]), int(item.shape[1])
    if item.ndim != 3:
        raise ValueError(f"Unsupported processed image shape: {item.shape}")
    if item.shape[-1] in (1, 3, 4):
        return int(item.shape[0]), int(item.shape[1])
    if item.shape[0] in (1, 3, 4):
        return int(item.shape[1]), int(item.shape[2])
    raise ValueError(f"Cannot infer H/W from processed image shape: {item.shape}")


def _resize_float(array: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.squeeze(np.asarray(array)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {array.shape}")
    if array.shape == (height, width):
        return array.copy()
    pil = Image.fromarray(array)
    return np.asarray(pil.resize((width, height), Image.Resampling.BILINEAR)).copy()


def prepare_views(
    images: Sequence[np.ndarray],
    prediction: Any,
    image_paths: Sequence[str | Path] | None = None,
) -> list[ViewGeometry]:
    """Convert DA3 output to original-resolution per-view geometry.

    Crucially, this indexes ``intrinsics[i]`` and derives ``sx_i``/``sy_i``
    from each individual original and processed image.
    """

    arrays = prediction_to_arrays(prediction)
    count = len(images)
    if count == 0:
        raise ValueError("At least one input image is required")
    paths = [str(p) for p in image_paths] if image_paths is not None else [""] * count
    if len(paths) != count:
        raise ValueError("image_paths and images must have the same length")

    processed = _view_axis(arrays["processed_images"], count, "processed images")
    depths = _view_axis(arrays["depth"], count, "depth maps")
    intrinsics = _view_axis(arrays["intrinsics"], count, "intrinsics")
    extrinsics = _view_axis(arrays["extrinsics"], count, "extrinsics")
    confidences = (
        _view_axis(arrays["conf"], count, "confidence maps")
        if "conf" in arrays else [None] * count
    )

    views = []
    for index, (image, proc, depth, intrinsic, extrinsic, path, confidence) in enumerate(
        zip(images, processed, depths, intrinsics, extrinsics, paths, confidences)
    ):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image {index} must be RGB HxWx3, got {image.shape}")
        original_h, original_w = image.shape[:2]
        processed_h, processed_w = _processed_hw(proc)

        K_processed = np.asarray(intrinsic, dtype=np.float64).squeeze().copy()
        if K_processed.shape != (3, 3):
            raise ValueError(f"Intrinsic {index} must be 3x3, got {K_processed.shape}")
        K = K_processed.copy()
        K[0, :] *= original_w / processed_w
        K[1, :] *= original_h / processed_h

        E = np.asarray(extrinsic, dtype=np.float64).squeeze()
        if E.shape not in ((3, 4), (4, 4)):
            raise ValueError(f"Extrinsic {index} must be 3x4 or 4x4, got {E.shape}")
        R = E[:3, :3].copy()
        T = E[:3, 3].copy()
        if not np.isfinite(K).all() or not np.isfinite(E).all():
            raise ValueError(f"Non-finite camera parameters in view {index}")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"Non-positive focal length in view {index}: {K}")

        views.append(
            ViewGeometry(
                index=index,
                path=path,
                image=image.astype(np.uint8, copy=False),
                depth=_resize_float(depth, original_w, original_h),
                K=K,
                K_processed=K_processed,
                R=R,
                T=T,
                processed_size=(processed_w, processed_h),
                confidence=(
                    _resize_float(confidence, original_w, original_h)
                    if confidence is not None else None
                ),
            )
        )
    return views


def _orthogonal_rotation_mean(rotations: Iterable[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(list(rotations)), axis=0)
    U, _, Vt = np.linalg.svd(mean)
    result = U @ Vt
    if np.linalg.det(result) < 0:
        U[:, -1] *= -1
        result = U @ Vt
    return result


def reference_frame(
    views: Sequence[ViewGeometry], reference_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"Unsupported reference mode: {reference_mode}")
    if reference_mode == "first":
        view = views[0]
        return view.camera_center, view.R.T
    if reference_mode == "mid":
        view = views[len(views) // 2]
        return view.camera_center, view.R.T
    center = np.mean(np.stack([view.camera_center for view in views]), axis=0)
    rotation = _orthogonal_rotation_mean(view.R for view in views)
    return center, rotation.T


def reference_camera(
    views: Sequence[ViewGeometry], reference_mode: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(center, scene_rotation, focal)`` for an editable target camera.

    The returned camera is separate from the source views.  Editing it changes
    the coordinate system into which every source view is projected, rather
    than perturbing one source image and creating a duplicate patch.
    """

    center, scene_rotation = reference_frame(views, reference_mode)
    if reference_mode == "first":
        target = views[0]
        focal = float((target.K[0, 0] + target.K[1, 1]) / 2)
    elif reference_mode == "mid":
        target = views[len(views) // 2]
        focal = float((target.K[0, 0] + target.K[1, 1]) / 2)
    else:
        focal = float(np.mean([(view.K[0, 0] + view.K[1, 1]) / 2 for view in views]))
    return center.copy(), scene_rotation.copy(), focal


def target_camera_components(
    target_camera: tuple[np.ndarray, np.ndarray, float, float, float]
    | tuple[np.ndarray, np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Decode one target camera for both projection and visualization.

    Returns ``(center, scene_rotation, R_w2c, T_w2c, focal, principal_offset)``
    under this module's row-vector convention.  Keeping this conversion in one
    place prevents the GLB frustum from silently using the inverse pose of the
    pre-stitch projection.
    """

    if len(target_camera) == 3:
        center, scene_rotation, focal = target_camera
        principal_offset = np.zeros(2, dtype=np.float64)
    elif len(target_camera) == 6:
        center, scene_rotation, focal, _focal_y, cx_offset, cy_offset = target_camera
        principal_offset = np.asarray([cx_offset, cy_offset], dtype=np.float64)
    else:
        center, scene_rotation, focal, cx_offset, cy_offset = target_camera
        principal_offset = np.asarray([cx_offset, cy_offset], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    scene_rotation = np.asarray(scene_rotation, dtype=np.float64)
    rotation_w2c = scene_rotation.T
    translation_w2c = -center @ rotation_w2c.T
    return (
        center,
        scene_rotation,
        rotation_w2c,
        translation_w2c,
        float(focal),
        principal_offset,
    )


def target_camera_focal_y(target_camera: tuple) -> float:
    """Return target fy while preserving compatibility with scalar cameras."""

    if len(target_camera) == 6:
        return float(target_camera[3])
    return float(target_camera_components(target_camera)[4])


def _unproject(view: ViewGeometry, stride: int) -> tuple[np.ndarray, ...]:
    height, width = view.depth.shape
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    z = view.depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    fx, fy = view.K[0, 0], view.K[1, 1]
    cx, cy = view.K[0, 2], view.K[1, 2]
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    colors = view.image[::stride, ::stride][valid].astype(np.float32)

    # Linear distance-to-border feathering.  A small floor preserves boundary
    # samples while ensuring central pixels dominate multi-view overlaps.
    edge = np.minimum.reduce((u + 1, v + 1, width - u, height - v))
    edge_scale = max(1.0, 0.12 * min(width, height))
    feather = np.clip(edge / edge_scale, 0.02, 1.0).astype(np.float32)
    source_xy = np.stack((u, v), axis=1)
    return points, colors, z.astype(np.float32), feather, source_xy


def unproject_views(
    views: Sequence[ViewGeometry], render_stride: int = 1
) -> list[tuple[np.ndarray, ...]]:
    """Build reusable per-view samples for repeated projection renders."""

    if render_stride < 1:
        raise ValueError("render_stride must be >= 1")
    return [_unproject(view, render_stride) for view in views]


def _project(
    points: np.ndarray,
    mode: str,
    focal: float,
    global_scale: float,
    focal_y: float | None = None,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y, Z = points.T
    focal_y = focal if focal_y is None else float(focal_y)
    finite = np.isfinite(points).all(axis=1)
    uv = np.zeros((len(points), 2), dtype=np.float64)
    if mode == "vertical_plane":
        valid = finite
        uv[valid] = np.stack((global_scale * X[valid], global_scale * Y[valid]), axis=1)
    elif mode == "pushbroom":
        valid = finite & (Z > 1e-4)
        uv[valid] = np.stack(
            (global_scale * X[valid], focal * Y[valid] / Z[valid]), axis=1
        )
    elif mode == "rectilinear":
        valid = finite & (Z > 1e-4)
        uv[valid] = np.stack(
            (focal * X[valid] / Z[valid], focal_y * Y[valid] / Z[valid]), axis=1
        )
    elif mode == "panini":
        # General Panini projection for longitude lambda and latitude phi:
        #   S = (d + 1) / (d + cos(lambda))
        #   x = fx * S * sin(lambda), y = fy * S * tan(phi)
        # Here sin(lambda)=X/rho, cos(lambda)=Z/rho, tan(phi)=Y/rho.
        # Unlike the angular cylindrical/equirectangular maps, classic Panini
        # has an intentional singularity on the rear optical axis, so it must
        # not be passed through the periodic longitude unwrap below.
        distance = float(panini_distance)
        if not np.isfinite(distance) or not 0.0 <= distance <= 1.0:
            raise ValueError("panini_distance must be finite and in [0, 1]")
        rho = np.sqrt(X * X + Z * Z)
        denominator = np.full_like(rho, np.nan, dtype=np.float64)
        nonzero_rho = finite & (rho > 1e-4)
        denominator[nonzero_rho] = (
            distance + Z[nonzero_rho] / rho[nonzero_rho]
        )
        valid = nonzero_rho & (denominator > 1e-4)
        scale = (distance + 1.0) / denominator[valid]
        uv[valid] = np.stack(
            (
                focal * scale * X[valid] / rho[valid],
                focal_y * scale * Y[valid] / rho[valid],
            ),
            axis=1,
        )
    elif mode == "general_panini":
        compression = float(general_panini_compression)
        top = float(general_panini_top)
        bottom = float(general_panini_bottom)
        if not np.isfinite(compression) or not 0.0 <= compression <= 150.0:
            raise ValueError("general_panini_compression must be finite and in [0, 150]")
        if not np.isfinite(top) or not -100.0 <= top <= 100.0:
            raise ValueError("general_panini_top must be finite and in [-100, 100]")
        if not np.isfinite(bottom) or not -100.0 <= bottom <= 100.0:
            raise ValueError("general_panini_bottom must be finite and in [-100, 100]")

        # libpano13's GUI-to-working-parameter conversion.  Cmpr=0 maps to
        # d=0 (rectilinear), Cmpr=100 to approximately d=1 (standard Panini),
        # and Cmpr=150 to a large finite approximation of cylindrical
        # orthographic projection.
        compression_t = (150.0 - compression) / 50.0
        distance = 1.5 / (compression_t + 0.0001) - 1.5 / 3.0001

        rho = np.sqrt(X * X + Z * Z)
        nonzero_rho = finite & (rho > 1e-4)
        longitude = np.zeros_like(rho, dtype=np.float64)
        longitude[nonzero_rho] = np.arctan2(X[nonzero_rho], Z[nonzero_rho])
        cos_longitude = np.zeros_like(rho, dtype=np.float64)
        cos_longitude[nonzero_rho] = Z[nonzero_rho] / rho[nonzero_rho]
        denominator = distance + cos_longitude

        # Match the feasible longitude domain used by libpano13.  Besides
        # avoiding singularities, this prevents the d>1 branch from folding
        # rear-facing content back over the visible panorama.
        theoretical_limit = np.arccos(-distance if distance <= 1.0 else -1.0 / distance)
        projection_limit_argument = distance * np.sin(
            np.radians(GENERAL_PANINI_MAX_PROJECTION_ANGLE_DEGREES)
        )
        max_longitude = theoretical_limit
        if abs(projection_limit_argument) <= 1.0:
            projection_limited = np.arcsin(projection_limit_argument) + np.radians(
                GENERAL_PANINI_MAX_PROJECTION_ANGLE_DEGREES
            )
            max_longitude = min(max_longitude, projection_limited)
        valid = (
            nonzero_rho
            & (denominator > 1e-4)
            & (np.abs(longitude) <= max_longitude)
        )

        scale = (distance + 1.0) / denominator[valid]
        horizontal = scale * X[valid] / rho[valid]
        vertical = scale * Y[valid] / rho[valid]
        squeeze = np.where(Y[valid] < 0.0, top, bottom) / 100.0

        # The following two branches reproduce PaniniGeneral.c's forward
        # hard/soft squeeze.  They only alter vertical coordinates, retaining
        # General Panini's straight vertical lines.
        soft = squeeze < 0.0
        if np.any(soft):
            side_curve = np.cos(0.92 * longitude[valid][soft]) - 1.0
            squeeze_scale = 2.0 * distance / (distance + 1.0)
            vertical[soft] *= 1.0 + squeeze_scale * squeeze[soft] * side_curve
        hard = squeeze > 0.0
        if np.any(hard):
            cosine = cos_longitude[valid][hard]
            straightened = cosine.copy()
            usable = cosine > 0.01
            straightened[usable] = vertical[hard][usable] / cosine[usable]
            vertical[hard] += squeeze[hard] * (straightened - vertical[hard])

        uv[valid] = np.stack(
            (focal * horizontal, focal_y * vertical), axis=1
        )
    elif mode == "cylindrical":
        rho = np.sqrt(X * X + Z * Z)
        valid = finite & (rho > 1e-4)
        uv[valid] = np.stack(
            (focal * np.arctan2(X[valid], Z[valid]), focal * Y[valid] / rho[valid]),
            axis=1,
        )
    elif mode == "equirectangular":
        radius = np.linalg.norm(points, axis=1)
        valid = finite & (radius > 1e-4)
        uv[valid] = np.stack(
            (
                focal * np.arctan2(X[valid], Z[valid]),
                focal * np.arcsin(np.clip(Y[valid] / radius[valid], -1.0, 1.0)),
            ),
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported projection mode: {mode}")
    return uv, valid


def _unwrap_periodic_projection(
    projected: Sequence[ProjectedView],
    views: Sequence[ViewGeometry],
    scene_rotation: np.ndarray,
    focal: float,
) -> None:
    """Put periodic panorama projections on one geometry-defined branch.

    Cylindrical and equirectangular horizontal coordinates are periodic at
    ``-pi/+pi``.  Choosing the branch from input-list adjacency is fragile:
    lexicographic filenames such as ``0, 10, 1, ...`` can move an otherwise
    adjacent view by a full circumference.  Instead, choose the seam in the
    largest gap between source-camera optical axes.  Each view's samples are
    then unwrapped relative to its own optical axis, which also prevents a
    single view crossing the seam from being split across both canvas edges.

    The operation mutates ``uv`` and ``flat_uv`` in ``projected`` in place.
    """

    if not projected:
        return
    if len(projected) != len(views):
        raise ValueError("projected views and source views must have the same length")
    if not np.isfinite(focal) or abs(float(focal)) < 1e-12:
        raise ValueError("periodic projection unwrap requires a finite non-zero focal")

    scene_rotation = np.asarray(scene_rotation, dtype=np.float64)
    optical_axes = []
    for view, item in zip(views, projected):
        # Under the row-vector convention, local camera +Z expressed in world
        # coordinates is e_z @ R.  scene_rotation then expresses it in the
        # target-camera coordinate system used by _project.
        forward_world = (
            np.asarray([0.0, 0.0, 1.0])
            @ np.asarray(view.R, dtype=np.float64)
        )
        forward_target = forward_world @ scene_rotation
        angle = float(np.arctan2(forward_target[0], forward_target[2]))
        if not np.isfinite(angle):
            # A malformed camera should not poison every other view's branch.
            # Fall back to its projected center only for that view.
            angle = float(np.median(item.uv[:, 0]) / focal) if len(item.uv) else 0.0
        optical_axes.append(angle)
    optical_axes = np.asarray(optical_axes, dtype=np.float64)

    axes_2pi = np.mod(optical_axes, 2.0 * np.pi)
    sorted_axes = np.sort(axes_2pi)
    circular_gaps = np.diff(
        np.concatenate((sorted_axes, sorted_axes[:1] + 2.0 * np.pi))
    )
    largest_gap_index = int(np.argmax(circular_gaps))
    branch_start = float(sorted_axes[(largest_gap_index + 1) % len(sorted_axes)])
    branch_axes = branch_start + np.mod(axes_2pi - branch_start, 2.0 * np.pi)

    # Only the branch modulo 2pi matters.  Select the equivalent global turn
    # nearest zero so manual canvases retain their target-camera-centered
    # interpretation instead of acquiring an arbitrary full-width offset.
    global_turn = (
        np.round(np.median(branch_axes) / (2.0 * np.pi)) * (2.0 * np.pi)
    )
    branch_axes -= global_turn

    def unwrap_values(
        values: np.ndarray | None, raw_axis: float, branch_axis: float
    ) -> None:
        if values is None or not len(values):
            return
        raw_angles = values[:, 0] / focal
        relative = (raw_angles - raw_axis + np.pi) % (2.0 * np.pi) - np.pi
        values[:, 0] = focal * (branch_axis + relative)

    for item, raw_axis, branch_axis in zip(projected, optical_axes, branch_axes):
        unwrap_values(item.uv, float(raw_axis), float(branch_axis))
        unwrap_values(item.flat_uv, float(raw_axis), float(branch_axis))


def project_views(
    views: Sequence[ViewGeometry],
    mode: str,
    reference_mode: str,
    render_stride: int = 1,
    unprojected: Sequence[tuple[np.ndarray, ...]] | None = None,
    reference_views: Sequence[ViewGeometry] | None = None,
    target_camera: tuple[np.ndarray, np.ndarray, float, float, float] | tuple[np.ndarray, np.ndarray, float] | None = None,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> tuple[list[ProjectedView], dict[str, float]]:
    if mode not in PROJECTION_MODES:
        raise ValueError(f"Unsupported projection mode: {mode}")
    if render_stride < 1:
        raise ValueError("render_stride must be >= 1")
    if unprojected is not None and len(unprojected) != len(views):
        raise ValueError("unprojected samples and views must have the same length")
    if target_camera is not None:
        center, scene_rotation, _, _, focal, principal_offset = target_camera_components(
            target_camera
        )
        focal_y = target_camera_focal_y(target_camera)
    else:
        reference_source = reference_views if reference_views is not None else views
        center, scene_rotation = reference_frame(reference_source, reference_mode)
        focal = float(np.median([(v.K[0, 0] + v.K[1, 1]) / 2 for v in views]))
        focal_y = focal
        principal_offset = np.zeros(2, dtype=np.float64)
    positive_depths = [v.depth[np.isfinite(v.depth) & (v.depth > 0)] for v in views]
    medians = [float(np.median(depth)) for depth in positive_depths if depth.size]
    z_reference = float(np.mean(medians)) if medians else 10.0
    global_scale = focal / max(z_reference, 1e-6)

    projected = []
    total_sample_count = 0
    valid_projected_count = 0
    per_view_target_depth = []
    for index, view in enumerate(views):
        local, colors, depth, feather, source_xy = (
            unprojected[index]
            if unprojected is not None
            else _unproject(view, render_stride)
        )
        world = (local - view.T) @ view.R
        referenced = (world - center) @ scene_rotation
        finite_target_depth = referenced[:, 2][np.isfinite(referenced[:, 2])]
        if len(finite_target_depth):
            per_view_target_depth.append(float(np.median(finite_target_depth)))
        uv, valid = _project(
            referenced,
            mode,
            focal,
            global_scale,
            focal_y=focal_y,
            panini_distance=panini_distance,
            general_panini_compression=general_panini_compression,
            general_panini_top=general_panini_top,
            general_panini_bottom=general_panini_bottom,
        )
        total_sample_count += len(local)
        valid_projected_count += int(valid.sum())
        if np.any(valid) and np.any(principal_offset):
            uv[valid] += principal_offset

        # A second map uses one median-depth surface per source view, matching
        # Flux.2's ``coord_maps`` (while the measured-depth map is passed as
        # ``coord_maps_depth``).  Both maps use the exact same target camera
        # and therefore the exact same final canvas coordinate system.
        positive = local[:, 2] > 1e-8
        median_depth = float(np.median(view.depth[np.isfinite(view.depth) & (view.depth > 0)]))
        if not np.isfinite(median_depth) or median_depth <= 0:
            median_depth = z_reference
        flat_local = local.copy()
        flat_local[positive] *= (median_depth / flat_local[positive, 2])[:, None]
        flat_world = (flat_local - view.T) @ view.R
        flat_referenced = (flat_world - center) @ scene_rotation
        flat_uv_all, flat_valid = _project(
            flat_referenced,
            mode,
            focal,
            global_scale,
            focal_y=focal_y,
            panini_distance=panini_distance,
            general_panini_compression=general_panini_compression,
            general_panini_top=general_panini_top,
            general_panini_bottom=general_panini_bottom,
        )
        if np.any(flat_valid) and np.any(principal_offset):
            flat_uv_all[flat_valid] += principal_offset
        projected.append(
            ProjectedView(
                image=view.image,
                depth=depth[valid],
                uv=uv[valid],
                colors=colors[valid],
                feather=feather[valid],
                target_depth=referenced[:, 2][valid].astype(np.float32),
                source_xy=source_xy[valid],
                flat_uv=flat_uv_all[flat_valid],
                flat_source_xy=source_xy[flat_valid],
            )
        )

    if mode in {"cylindrical", "equirectangular"}:
        _unwrap_periodic_projection(projected, views, scene_rotation, focal)

    return projected, {
        "focal_pixels": focal,
        "panini_distance": float(panini_distance) if mode == "panini" else None,
        "general_panini_compression": (
            float(general_panini_compression) if mode == "general_panini" else None
        ),
        "general_panini_top": (
            float(general_panini_top) if mode == "general_panini" else None
        ),
        "general_panini_bottom": (
            float(general_panini_bottom) if mode == "general_panini" else None
        ),
        "median_depth": z_reference,
        "linear_projection_scale": global_scale,
        "projection_valid_fraction": valid_projected_count / max(1, total_sample_count),
        "target_depth_median": (
            float(np.median(per_view_target_depth)) if per_view_target_depth else float("nan")
        ),
    }


def _canvas_transform(
    projected: Sequence[ProjectedView],
    quantile: float,
    padding: int,
    max_canvas_pixels: int | None,
    requested_size: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], tuple[int, int], dict[str, Any]]:
    all_uv = np.concatenate([item.uv for item in projected if len(item.uv)], axis=0)
    if not len(all_uv):
        raise ValueError("No valid projected points")
    if not 0 <= quantile < 0.5:
        raise ValueError("bounds_quantile must be in [0, 0.5)")
    low = np.quantile(all_uv, quantile, axis=0)
    high = np.quantile(all_uv, 1.0 - quantile, axis=0)
    # The renderer/cropper operates on integer pixels (the reference script
    # applies np.round before taking inclusive min/max bounds).  Quantize the
    # estimated bounds before deriving the extent; ceil(high-low)+1 on the
    # continuous UVs can overestimate one edge pixel.
    low = np.rint(low)
    high = np.rint(high)
    extent = np.maximum(high - low, 0.0)
    raw_width = int(extent[0]) + 2 * padding + 1
    raw_height = int(extent[1]) + 2 * padding + 1
    if requested_size is not None:
        width = max(1, int(requested_size[0]))
        height = max(1, int(requested_size[1]))
        # A manual canvas represents the target camera's actual pixel plane.
        # Preserve the camera projection at 1:1 scale and place its zero
        # principal point at the canvas center; points outside are clipped by
        # the splat.  Scaling the complete bounding box to fit here would
        # silently change the target focal length after projection.
        scale = 1.0
        offset = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
        transformed = [item.uv + offset for item in projected]
    else:
        if max_canvas_pixels is None:
            scale = 1.0
        else:
            scale = min(1.0, math.sqrt(max_canvas_pixels / max(1, raw_width * raw_height)))
        width = max(1, int(math.ceil(raw_width * scale)))
        height = max(1, int(math.ceil(raw_height * scale)))
        transformed = [(item.uv - low + padding) * scale for item in projected]
    return transformed, (width, height), {
        "raw_bounds_xyxy": [float(low[0]), float(low[1]), float(high[0]), float(high[1])],
        "raw_canvas_size": [raw_width, raw_height],
        "canvas_scale": scale,
        "canvas_size": [width, height],
        "bounds_quantile": quantile,
        "padding": int(padding),
        "requested_canvas_size": list(requested_size) if requested_size is not None else None,
    }


def _apply_canvas_transform(uv: np.ndarray, canvas_meta: dict[str, Any]) -> np.ndarray:
    """Apply the already-selected canvas transform to another UV set."""

    uv = np.asarray(uv, dtype=np.float64)
    requested = canvas_meta.get("requested_canvas_size")
    if requested is not None:
        return uv + np.asarray(
            [float(requested[0]) / 2.0, float(requested[1]) / 2.0],
            dtype=np.float64,
        )
    bounds = np.asarray(canvas_meta["raw_bounds_xyxy"], dtype=np.float64)
    low = bounds[:2]
    padding = float(canvas_meta.get("padding", 0))
    scale = float(canvas_meta.get("canvas_scale", 1.0))
    return (uv - low + padding) * scale


def _resize_valid_coord_map(
    coordinates: np.ndarray,
    valid: np.ndarray,
    size: tuple[int, int],
    invalid_value: float = 1000.0,
) -> np.ndarray:
    """Bilinearly resize coordinates without interpolating invalid sentinels.

    Resize zero-filled coordinates and their validity weights with the same
    kernel, then renormalize. Partial support uses only valid neighbours;
    cells with no support receive the sentinel *after* interpolation. This
    does not fill depth holes or extrapolate beyond the resize kernel.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(coordinates).all(axis=-1)
    if not np.any(valid):
        return np.full((size[1], size[0], coordinates.shape[-1]), invalid_value, dtype=np.float32)
    numerator = np.where(valid[..., None], coordinates, 0.0)
    try:
        import cv2

        if np.all(valid):
            # Preserve the previous dense-map interpolation exactly.
            return cv2.resize(coordinates, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # Coordinates and weights must use the *same* sampling implementation.
        # Separate float32 C=2/C=1 resize paths can disagree slightly near
        # sparse boundaries; division by tiny support magnifies that error.
        packed = np.concatenate((numerator.astype(np.float64), valid[..., None]), axis=-1)
        resized = cv2.resize(packed, size, interpolation=cv2.INTER_LINEAR)
        numerator, weights = resized[..., :-1], resized[..., -1]
    except ImportError:  # pragma: no cover - OpenCV is a repository dependency.
        source_h, source_w = valid.shape
        x = np.clip((np.arange(size[0]) + 0.5) * source_w / size[0] - 0.5, 0, source_w - 1)
        y = np.clip((np.arange(size[1]) + 0.5) * source_h / size[1] - 0.5, 0, source_h - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, source_w - 1), np.minimum(y0 + 1, source_h - 1)
        dx, dy = x - x0, y - y0
        sampled = np.zeros((size[1], size[0], coordinates.shape[-1]), dtype=np.float64)
        weights = np.zeros((size[1], size[0]), dtype=np.float64)
        for cols, rows, kernel in (
            (x0, y0, (1-dy)[:, None] * (1-dx)[None, :]),
            (x1, y0, (1-dy)[:, None] * dx[None, :]),
            (x0, y1, dy[:, None] * (1-dx)[None, :]),
            (x1, y1, dy[:, None] * dx[None, :]),
        ):
            support = kernel * valid[rows[:, None], cols[None, :]]
            sampled += numerator[rows[:, None], cols[None, :]] * support[..., None]
            weights += support
        numerator = sampled
    result = np.full(numerator.shape, invalid_value, dtype=np.float32)
    supported = weights > 1e-6
    result[supported] = numerator[supported] / weights[supported, None]
    return result


def _latent_coord_map(
    source_xy: np.ndarray,
    target_uv: np.ndarray,
    source_shape: tuple[int, int],
    canvas_size: tuple[int, int],
    patch_size: int = 16,
    source_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a pixel-level source map to the Flux latent grid.

    Keep the reference script's ``ceil(H/16) x ceil(W/16)`` grid and target
    coordinate units, but use validity-normalized interpolation so missing
    depth / clipped points cannot drag adjacent coordinates toward -1.
    """

    height, width = (int(source_shape[0]), int(source_shape[1]))
    if height <= 0 or width <= 0:
        return np.empty((0, 0, 2), dtype=np.float32)
    if patch_size != 16:
        raise ValueError("Flux coord-map conversion uses a 16-pixel patch size")

    pixel_map = np.zeros((height, width, 2), dtype=np.float32)
    pixel_valid = np.zeros((height, width), dtype=bool)
    if len(source_xy) > 0 and len(target_uv) > 0:
        source_xy = np.asarray(source_xy, dtype=np.float64)
        target_uv = np.asarray(target_uv, dtype=np.float64)
        finite = np.isfinite(source_xy).all(axis=1) & np.isfinite(target_uv).all(axis=1)
        # Keep Flux conditioning consistent with the displayed pre-stitch:
        # points clipped by a manual/quantile canvas must not survive in the
        # source-to-target map as out-of-canvas coordinates.
        canvas_w, canvas_h = canvas_size
        finite &= (
            (target_uv[:, 0] >= 0.0) & (target_uv[:, 0] < float(canvas_w)) &
            (target_uv[:, 1] >= 0.0) & (target_uv[:, 1] < float(canvas_h))
        )
        source_xy = source_xy[finite]
        target_uv = target_uv[finite]
        if len(source_xy):
            # _unproject uses integer mesh-grid coordinates.  Rounding here
            # also keeps the helper safe for callers that provide float indices.
            cols = np.rint(source_xy[:, 0]).astype(np.int64)
            rows = np.rint(source_xy[:, 1]).astype(np.int64)
            inside = (
                (rows >= 0) & (rows < height) &
                (cols >= 0) & (cols < width)
            )
            pixel_map[rows[inside], cols[inside]] = target_uv[inside].astype(np.float32)
            pixel_valid[rows[inside], cols[inside]] = True

    mod_pad_h = (patch_size - height % patch_size) % patch_size
    mod_pad_w = (patch_size - width % patch_size) % patch_size
    latent_map_size = (
        (width + mod_pad_w) // patch_size,
        (height + mod_pad_h) // patch_size,
    )
    latent_map = _resize_valid_coord_map(
        pixel_map, pixel_valid, latent_map_size, invalid_value=1000.0 * patch_size,
    )
    latent_map /= float(patch_size)
    if source_mask is not None:
        # Do not insert invalid sentinels before interpolation: that would
        # bleed into neighbouring latent cells.  First determine the complete
        # coordinate map, then overwrite only latent cells covered by the
        # source mask.
        # Keep fractional coverage during downsampling; using uint8 values
        # (0/1) with INTER_AREA would round small covered cells back to zero.
        mask_array = np.asarray(source_mask, dtype=np.float32)
        try:
            import cv2
            mask_latent = cv2.resize(
                mask_array,
                latent_map_size,
                interpolation=cv2.INTER_AREA,
            ) > 0
        except ImportError:  # pragma: no cover - OpenCV is a repository dependency.
            mask_latent = np.asarray(
                Image.fromarray(mask_array).resize(latent_map_size, Image.Resampling.BILINEAR)
            ) > 0
        latent_map[mask_latent] = 1000.0
    return latent_map


def _splat(
    uv: np.ndarray,
    colors: np.ndarray,
    depth: np.ndarray,
    source_weights: np.ndarray,
    canvas_size: tuple[int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Bilinear forward splat, returning color/depth/mask and accumulators."""

    width, height = canvas_size
    color_sum = np.zeros((height, width, 3), dtype=np.float32)
    color_sq_sum = np.zeros_like(color_sum)
    depth_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    if len(uv):
        base_x = np.floor(uv[:, 0]).astype(np.int64)
        base_y = np.floor(uv[:, 1]).astype(np.int64)
        frac_x = (uv[:, 0] - base_x).astype(np.float32)
        frac_y = (uv[:, 1] - base_y).astype(np.float32)
        for dx, dy, bilinear in (
            (0, 0, (1 - frac_x) * (1 - frac_y)),
            (1, 0, frac_x * (1 - frac_y)),
            (0, 1, (1 - frac_x) * frac_y),
            (1, 1, frac_x * frac_y),
        ):
            x, y = base_x + dx, base_y + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x, y = x[valid], y[valid]
            weights = bilinear[valid] * source_weights[valid]
            values = colors[valid]
            np.add.at(color_sum, (y, x), values * weights[:, None])
            np.add.at(color_sq_sum, (y, x), values * values * weights[:, None])
            np.add.at(depth_sum, (y, x), depth[valid] * weights)
            np.add.at(weight_sum, (y, x), weights)
    mask = weight_sum > 1e-6
    image = np.zeros_like(color_sum, dtype=np.uint8)
    depth_image = np.zeros_like(depth_sum)
    image[mask] = np.clip(color_sum[mask] / weight_sum[mask, None], 0, 255).astype(np.uint8)
    depth_image[mask] = depth_sum[mask] / weight_sum[mask]
    return image, depth_image, mask, color_sum, color_sq_sum, weight_sum


def _zbuffer_nearest_depth(
    uv: np.ndarray,
    depth: np.ndarray,
    source_weights: np.ndarray,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Compute the nearest target-camera depth reaching each canvas pixel.

    The first pass is deliberately separate from colour accumulation: this
    makes visibility independent of view iteration order and prevents a later
    view from averaging through a nearer surface.  Bilinear support with a
    positive weight is retained, so the z-buffer does not depend on integer
    rounding of projected coordinates.
    """

    width, height = canvas_size
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    if not len(uv):
        return nearest
    base_x = np.floor(uv[:, 0]).astype(np.int64)
    base_y = np.floor(uv[:, 1]).astype(np.int64)
    frac_x = (uv[:, 0] - base_x).astype(np.float32)
    frac_y = (uv[:, 1] - base_y).astype(np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    source_weights = np.asarray(source_weights, dtype=np.float32)
    for dx, dy, bilinear in (
        (0, 0, (1 - frac_x) * (1 - frac_y)),
        (1, 0, frac_x * (1 - frac_y)),
        (0, 1, (1 - frac_x) * frac_y),
        (1, 1, frac_x * frac_y),
    ):
        x, y = base_x + dx, base_y + dy
        valid = (
            (x >= 0) & (x < width) & (y >= 0) & (y < height)
            & (bilinear * source_weights > 1e-6)
            # vertical-plane/pushbroom projections can legitimately contain
            # points behind the editable target camera, so do not impose a
            # positive-Z test here (pinhole projection already filters Z>0).
            & np.isfinite(depth)
        )
        if np.any(valid):
            np.minimum.at(nearest, (y[valid], x[valid]), depth[valid])
    return nearest


def _splat_visibility(
    uv: np.ndarray,
    colors: np.ndarray,
    source_depth: np.ndarray,
    visibility_depth: np.ndarray,
    source_weights: np.ndarray,
    canvas_size: tuple[int, int],
    nearest_depth: np.ndarray,
    relative_tolerance: float = 0.03,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Splat only samples close to the front-most surface at each pixel.

    A small relative depth tolerance keeps nearly coplanar samples from
    producing single-view pinholes while still rejecting the far surfaces that
    cause the duplicated-patch artefact in ordinary averaging.  The function
    returns the same accumulators as :func:`_splat`, allowing the visibility
    mode to reuse all existing metrics and coordinate-map logic.
    """

    width, height = canvas_size
    color_sum = np.zeros((height, width, 3), dtype=np.float32)
    color_sq_sum = np.zeros_like(color_sum)
    depth_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    if len(uv):
        base_x = np.floor(uv[:, 0]).astype(np.int64)
        base_y = np.floor(uv[:, 1]).astype(np.int64)
        frac_x = (uv[:, 0] - base_x).astype(np.float32)
        frac_y = (uv[:, 1] - base_y).astype(np.float32)
        source_depth = np.asarray(source_depth, dtype=np.float32)
        visibility_depth = np.asarray(visibility_depth, dtype=np.float32)
        source_weights = np.asarray(source_weights, dtype=np.float32)
        for dx, dy, bilinear in (
            (0, 0, (1 - frac_x) * (1 - frac_y)),
            (1, 0, frac_x * (1 - frac_y)),
            (0, 1, (1 - frac_x) * frac_y),
            (1, 1, frac_x * frac_y),
        ):
            x, y = base_x + dx, base_y + dy
            valid = (
                (x >= 0) & (x < width) & (y >= 0) & (y < height)
                & (bilinear * source_weights > 1e-6)
                & np.isfinite(visibility_depth)
            )
            if not np.any(valid):
                continue
            x, y = x[valid], y[valid]
            weights = bilinear[valid] * source_weights[valid]
            depths = visibility_depth[valid]
            nearest = nearest_depth[y, x]
            # Use an absolute-depth tolerance instead of multiplying by
            # ``(1+tolerance)``: vertical-plane targets may contain signed Z,
            # and multiplying a negative nearest depth would make the nearest
            # sample fail its own visibility test.
            tolerance = np.maximum(
                np.abs(nearest) * max(0.0, relative_tolerance), 1e-4
            )
            keep = depths <= nearest + tolerance
            if not np.any(keep):
                continue
            x, y = x[keep], y[keep]
            weights = weights[keep]
            values = colors[valid][keep]
            np.add.at(color_sum, (y, x), values * weights[:, None])
            np.add.at(color_sq_sum, (y, x), values * values * weights[:, None])
            np.add.at(depth_sum, (y, x), source_depth[valid][keep] * weights)
            np.add.at(weight_sum, (y, x), weights)
    mask = weight_sum > 1e-6
    image = np.zeros_like(color_sum, dtype=np.uint8)
    depth_image = np.zeros_like(depth_sum)
    image[mask] = np.clip(color_sum[mask] / weight_sum[mask, None], 0, 255).astype(np.uint8)
    depth_image[mask] = depth_sum[mask] / weight_sum[mask]
    return image, depth_image, mask, color_sum, color_sq_sum, weight_sum


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _depth_visualization(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0)
    output = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        if high <= low:
            high = low + 1e-6
        output[valid] = np.clip((depth[valid] - low) * 255 / (high - low), 0, 255)
    return output


def _rotation_path_degrees(views: Sequence[ViewGeometry]) -> float:
    total = 0.0
    for first, second in zip(views, views[1:]):
        relative = second.R @ first.R.T
        cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
        total += math.degrees(math.acos(float(cosine)))
    return total


def camera_report(views: Sequence[ViewGeometry]) -> dict[str, Any]:
    centers = np.stack([view.camera_center for view in views])
    path_length = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    endpoint = float(np.linalg.norm(centers[-1] - centers[0]))
    records = []
    for view in views:
        valid_depth = view.depth[np.isfinite(view.depth) & (view.depth > 0)]
        records.append(
            {
                "index": view.index,
                "path": view.path,
                "original_size": [int(view.image.shape[1]), int(view.image.shape[0])],
                "processed_size": list(view.processed_size),
                "intrinsic_processed": view.K_processed.tolist(),
                "intrinsic_original": view.K.tolist(),
                "R": view.R.tolist(),
                "T": view.T.tolist(),
                "camera_center": view.camera_center.tolist(),
                "depth_median": float(np.median(valid_depth)) if valid_depth.size else None,
                "depth_p10": float(np.percentile(valid_depth, 10)) if valid_depth.size else None,
                "depth_p90": float(np.percentile(valid_depth, 90)) if valid_depth.size else None,
            }
        )
    return {
        "views": records,
        "camera_path_length": path_length,
        "camera_endpoint_displacement": endpoint,
        "rotation_path_degrees": _rotation_path_degrees(views),
    }


def render_prestitch(
    views: Sequence[ViewGeometry],
    output_dir: str | Path | None,
    mode: str = "vertical_plane",
    reference_mode: str = "mid",
    render_stride: int = 1,
    bounds_quantile: float = 0.001,
    padding: int = 8,
    max_canvas_pixels: int | None = 40_000_000,
    save_numpy: bool = False,
    return_images: bool = False,
    canvas_size: tuple[int, int] | None = None,
    unprojected: Sequence[tuple[np.ndarray, ...]] | None = None,
    coord_unprojected: Sequence[tuple[np.ndarray, ...]] | None = None,
    coord_source_masks: Sequence[np.ndarray] | None = None,
    reference_views: Sequence[ViewGeometry] | None = None,
    target_camera: tuple[np.ndarray, np.ndarray, float, float, float] | tuple[np.ndarray, np.ndarray, float] | None = None,
    visibility_aware: bool = False,
    zbuffer_relative_tolerance: float = 0.03,
    return_coord_maps: bool = False,
    return_heatmap: bool = True,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> dict[str, Any]:
    """Project, feather-composite, optionally save diagnostics, and return metrics.

    Passing ``output_dir=None`` keeps the complete render in memory.  This is
    used by the interactive UI so changing projection/camera parameters does
    not create a file per slider update.
    """

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    projected, projection_meta = project_views(
        views,
        mode,
        reference_mode,
        render_stride=render_stride,
        unprojected=unprojected,
        reference_views=reference_views,
        target_camera=target_camera,
        panini_distance=panini_distance,
        general_panini_compression=general_panini_compression,
        general_panini_top=general_panini_top,
        general_panini_bottom=general_panini_bottom,
    )
    coord_projected = projected
    if coord_unprojected is not None:
        if len(coord_unprojected) != len(views):
            raise ValueError("coord_unprojected samples and views must have the same length")
        coord_projected, _ = project_views(
            views,
            mode,
            reference_mode,
            render_stride=render_stride,
            unprojected=coord_unprojected,
            reference_views=reference_views,
            target_camera=target_camera,
            panini_distance=panini_distance,
            general_panini_compression=general_panini_compression,
            general_panini_top=general_panini_top,
            general_panini_bottom=general_panini_bottom,
        )
    transformed_uv, canvas_size, canvas_meta = _canvas_transform(
        projected,
        bounds_quantile,
        padding,
        max_canvas_pixels,
        requested_size=canvas_size,
    )
    width, height = canvas_size

    nearest_depth = None
    if visibility_aware:
        # All projected views participate in one global depth pass.  This is
        # what makes the result order-independent and prevents far patches
        # from being averaged into nearer surfaces.
        nearest_depth = _zbuffer_nearest_depth(
            np.concatenate([item.uv for item in projected if len(item.uv)], axis=0),
            np.concatenate([
                (item.target_depth if item.target_depth is not None else item.depth)
                for item in projected if len(item.uv)
            ], axis=0),
            np.concatenate([item.feather for item in projected if len(item.uv)], axis=0),
            canvas_size,
        )

    combined_color = np.zeros((height, width, 3), dtype=np.float32)
    combined_sq = np.zeros_like(combined_color)
    combined_weight = np.zeros((height, width), dtype=np.float32)
    support = np.zeros((height, width), dtype=np.uint16)
    masks = []

    projected_dir = output_dir / "projected_views" if output_dir is not None else None
    for index, (item, uv) in enumerate(zip(projected, transformed_uv)):
        if visibility_aware:
            visibility_depth = item.target_depth if item.target_depth is not None else item.depth
            image, depth, mask, color_sum, color_sq, weights = _splat_visibility(
                uv,
                item.colors,
                item.depth,
                visibility_depth,
                item.feather,
                canvas_size,
                nearest_depth,
                relative_tolerance=zbuffer_relative_tolerance,
            )
        else:
            image, depth, mask, color_sum, color_sq, weights = _splat(
                uv, item.colors, item.depth, item.feather, canvas_size
            )
        combined_color += color_sum
        combined_sq += color_sq
        combined_weight += weights
        support += mask.astype(np.uint16)
        masks.append(mask)

        stem = f"{index:06d}"
        if projected_dir is not None:
            _save_rgb(projected_dir / f"{stem}_image.png", image)
        if save_numpy and projected_dir is not None:
            np.save(projected_dir / f"{stem}_depth.npy", depth)
        if projected_dir is not None:
            _save_gray(projected_dir / f"{stem}_depth.png", _depth_visualization(depth, mask))
            _save_gray(projected_dir / f"{stem}_mask.png", mask.astype(np.uint8) * 255)

    visibility_fallback_pixels = 0
    if visibility_aware and np.any(combined_weight <= 1e-6):
        # A noisy depth outlier can win the z-buffer at an isolated sample and
        # leave neighbouring pixels without an accepted contributor. Fill only
        # those holes from the ordinary splat; overlapping pixels remain
        # visibility-selected, so this fallback cannot reintroduce a far
        # surface duplicate patch.
        fallback_color = np.zeros_like(combined_color)
        fallback_sq = np.zeros_like(combined_sq)
        fallback_weight = np.zeros_like(combined_weight)
        fallback_masks = []
        for item, uv in zip(projected, transformed_uv):
            _, _, fallback_mask, color_sum, color_sq, weights = _splat(
                uv, item.colors, item.depth, item.feather, canvas_size
            )
            fallback_color += color_sum
            fallback_sq += color_sq
            fallback_weight += weights
            fallback_masks.append(fallback_mask)
        missing = (combined_weight <= 1e-6) & (fallback_weight > 1e-6)
        visibility_fallback_pixels = int(missing.sum())
        if visibility_fallback_pixels:
            combined_color[missing] = fallback_color[missing]
            combined_sq[missing] = fallback_sq[missing]
            combined_weight[missing] = fallback_weight[missing]
            for index, fallback_mask in enumerate(fallback_masks):
                masks[index] = masks[index] | (missing & fallback_mask)
            support[missing] = np.sum(
                np.stack([fallback_mask[missing] for fallback_mask in fallback_masks], axis=0),
                axis=0,
            ).astype(np.uint16)

    valid = combined_weight > 1e-6
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    mean = np.zeros_like(combined_color)
    mean[valid] = combined_color[valid] / combined_weight[valid, None]
    composite[valid] = np.clip(mean[valid], 0, 255).astype(np.uint8)
    if output_dir is not None:
        _save_rgb(output_dir / "composite.png", composite)

    if return_heatmap:
        max_support = max(1, int(support.max()))
        normalized = support.astype(np.float32) / max_support
        heatmap = np.stack(
            (
                np.clip(2 * normalized, 0, 1),
                np.clip(2 - 2 * np.abs(normalized - 0.5), 0, 1),
                np.clip(2 * (1 - normalized), 0, 1),
            ),
            axis=2,
        )
        heatmap[support == 0] = 0
        if output_dir is not None:
            _save_rgb(output_dir / "overlap_heatmap.png", (heatmap * 255).astype(np.uint8))
    else:
        heatmap = np.empty((0, 0, 3), dtype=np.uint8)
    if save_numpy and output_dir is not None:
        np.save(output_dir / "overlap_count.npy", support)

    overlap = support >= 2
    variance = np.zeros_like(combined_color)
    variance[valid] = np.maximum(
        combined_sq[valid] / combined_weight[valid, None] - mean[valid] ** 2,
        0,
    )
    disagreement = (
        float(np.sqrt(variance[overlap]).mean() / 255.0) if np.any(overlap) else 1.0
    )
    adjacent_ious = []
    for first, second in zip(masks, masks[1:]):
        union = np.logical_or(first, second).sum()
        adjacent_ious.append(float(np.logical_and(first, second).sum() / max(1, union)))

    camera = camera_report(views)
    median_depth = projection_meta["median_depth"]
    path_ratio = camera["camera_path_length"] / max(median_depth, 1e-6)
    rotation_radians = math.radians(camera["rotation_path_degrees"])
    translation_fraction = path_ratio / max(path_ratio + rotation_radians, 1e-6)
    coverage = float(valid.mean())
    overlap_ratio = float(overlap.sum() / max(1, valid.sum()))
    adjacent_mean = float(np.mean(adjacent_ious)) if adjacent_ious else 1.0
    prestitch_score = 100 * (
        0.35 * coverage
        + 0.35 * min(1.0, adjacent_mean / 0.25)
        + 0.30 * math.exp(-5 * disagreement)
    )
    challenge_score = 100 * (
        0.55 * math.tanh(5 * path_ratio)
        + 0.25 * translation_fraction
        + 0.20 * math.tanh(5 * disagreement)
    )

    source_width = float(np.median([view.image.shape[1] for view in views]))
    metrics = {
        "projection_mode": mode,
        "reference_mode": reference_mode,
        "num_views": len(views),
        **projection_meta,
        **canvas_meta,
        "canvas_aspect_ratio": width / max(1, height),
        "horizontal_extension": width / max(1.0, source_width),
        "canvas_coverage_ratio": coverage,
        "canvas_hole_ratio": 1.0 - coverage,
        "overlap_ratio_of_covered": overlap_ratio,
        "mean_view_support": float(support[valid].mean()) if np.any(valid) else 0.0,
        "max_view_support": int(support.max()),
        "adjacent_iou": adjacent_ious,
        "adjacent_iou_mean": adjacent_mean,
        "overlap_color_disagreement": disagreement,
        "camera_path_to_depth_ratio": path_ratio,
        "translation_motion_fraction": translation_fraction,
        "prestitch_score": prestitch_score,
        "challenge_score": challenge_score,
        "render_stride": render_stride,
        "fusion_mode": "visibility_aware_zbuffer" if visibility_aware else "weighted_forward_splat",
        "zbuffer_relative_tolerance": float(zbuffer_relative_tolerance) if visibility_aware else None,
        "visibility_fallback_pixels": visibility_fallback_pixels,
    }
    if return_coord_maps:
        # Keep these in memory only.  They are derived after the exact canvas
        # transform used by ``composite`` and are consumed directly by Flux.
        coord_maps = []
        coord_maps_depth = []
        if coord_projected is projected:
            coord_transformed_uv = transformed_uv
        else:
            coord_transformed_uv = [
                _apply_canvas_transform(item.uv, canvas_meta) for item in coord_projected
            ]
        for index, (item, uv) in enumerate(zip(coord_projected, coord_transformed_uv)):
            flat_source_xy = item.flat_source_xy if item.flat_source_xy is not None else np.empty((0, 2))
            flat_uv = item.flat_uv if item.flat_uv is not None else np.empty((0, 2))
            source_xy = item.source_xy if item.source_xy is not None else np.empty((0, 2))
            source_mask = (
                coord_source_masks[index]
                if coord_source_masks is not None and index < len(coord_source_masks)
                else None
            )
            coord_maps.append(
                _latent_coord_map(
                    flat_source_xy,
                    _apply_canvas_transform(flat_uv, canvas_meta),
                    views[index].image.shape[:2],
                    canvas_size,
                    source_mask=source_mask,
                )
            )
            coord_maps_depth.append(
                _latent_coord_map(
                    source_xy,
                    uv,
                    views[index].image.shape[:2],
                    canvas_size,
                    source_mask=source_mask,
                )
            )
        metrics["_coord_maps"] = coord_maps
        metrics["_coord_maps_depth"] = coord_maps_depth
        # Pixel-space copies are useful for debugging/inspection.  Flux uses
        # the latent-space versions above (target pixels divided by 16).
        metrics["_coord_maps_pixel"] = [item * 16.0 for item in coord_maps]
        metrics["_coord_maps_depth_pixel"] = [item * 16.0 for item in coord_maps_depth]
    if output_dir is not None:
        with (output_dir / "camera_intrinsics.json").open("w", encoding="utf-8") as handle:
            json.dump(camera, handle, indent=2, ensure_ascii=False)
        with (output_dir / "geometry_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
    if return_images:
        metrics["_composite"] = composite
        metrics["_overlap_heatmap"] = (heatmap * 255).astype(np.uint8)
    return metrics


def render_prestitch_memory(
    views: Sequence[ViewGeometry],
    mode: str = "vertical_plane",
    reference_mode: str = "mid",
    render_stride: int = 1,
    bounds_quantile: float = 0.001,
    padding: int = 8,
    max_canvas_pixels: int | None = 8_000_000,
    canvas_size: tuple[int, int] | None = None,
    unprojected: Sequence[tuple[np.ndarray, ...]] | None = None,
    coord_unprojected: Sequence[tuple[np.ndarray, ...]] | None = None,
    coord_source_masks: Sequence[np.ndarray] | None = None,
    reference_views: Sequence[ViewGeometry] | None = None,
    target_camera: tuple[np.ndarray, np.ndarray, float, float, float] | tuple[np.ndarray, np.ndarray, float] | None = None,
    visibility_aware: bool = False,
    zbuffer_relative_tolerance: float = 0.03,
    return_coord_maps: bool = False,
    return_heatmap: bool = True,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Render a pre-stitch result without creating any files.

    Returns ``(composite_rgb, overlap_heatmap_rgb, metrics)``.  The helper is
    intentionally separate from :func:`run_pipeline`, whose CLI behavior
    continues to save its diagnostics to disk.
    """

    metrics = render_prestitch(
        views,
        output_dir=None,
        mode=mode,
        reference_mode=reference_mode,
        render_stride=render_stride,
        bounds_quantile=bounds_quantile,
        padding=padding,
        max_canvas_pixels=max_canvas_pixels,
        save_numpy=False,
        return_images=True,
        canvas_size=canvas_size,
        unprojected=unprojected,
        coord_unprojected=coord_unprojected,
        coord_source_masks=coord_source_masks,
        reference_views=reference_views,
        target_camera=target_camera,
        visibility_aware=visibility_aware,
        zbuffer_relative_tolerance=zbuffer_relative_tolerance,
        return_coord_maps=return_coord_maps,
        return_heatmap=return_heatmap,
        panini_distance=panini_distance,
        general_panini_compression=general_panini_compression,
        general_panini_top=general_panini_top,
        general_panini_bottom=general_panini_bottom,
    )
    composite = metrics.pop("_composite")
    heatmap = metrics.pop("_overlap_heatmap")
    return composite, heatmap, metrics


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    prediction_cache: str | Path | None = None,
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    model_cache_dir: str | None = None,
    device: str | None = None,
    process_res: int = 504,
    force_prediction: bool = False,
    modes: Sequence[str] = ("vertical_plane",),
    references: Sequence[str] = ("mid",),
    render_stride: int = 1,
    bounds_quantile: float = 0.001,
    max_canvas_pixels: int = 40_000_000,
    save_numpy: bool = False,
    inference_fn: Callable[[Sequence[str | Path]], Any] | None = None,
    panini_distance: float = PANINI_DISTANCE,
    general_panini_compression: float = GENERAL_PANINI_COMPRESSION,
    general_panini_top: float = GENERAL_PANINI_TOP_SQUEEZE,
    general_panini_bottom: float = GENERAL_PANINI_BOTTOM_SQUEEZE,
) -> dict[str, dict[str, Any]]:
    paths = discover_images(input_path)
    images = load_images(paths)
    output_dir = Path(output_dir)
    if prediction_cache is None:
        prediction_cache = output_dir / "da3_prediction.npz"
    if inference_fn is None:
        def default_inference(items: Sequence[str | Path]) -> Any:
            return run_da3(
                items,
                model_id=model_id,
                model_cache_dir=model_cache_dir,
                device=device,
                process_res=process_res,
            )

        inference_fn = default_inference
    prediction, cache_hit = get_or_run_prediction(
        paths,
        prediction_cache,
        model_id,
        force=force_prediction,
        inference_fn=inference_fn,
    )
    views = prepare_views(images, prediction, paths)

    results = {}
    for mode in modes:
        for reference in references:
            key = f"{mode}/{reference}"
            results[key] = render_prestitch(
                views,
                output_dir / mode / reference,
                mode=mode,
                reference_mode=reference,
                render_stride=render_stride,
                bounds_quantile=bounds_quantile,
                max_canvas_pixels=max_canvas_pixels,
                save_numpy=save_numpy,
                panini_distance=panini_distance,
                general_panini_compression=general_panini_compression,
                general_panini_top=general_panini_top,
                general_panini_bottom=general_panini_bottom,
            )
    summary = {
        "input": str(Path(input_path).resolve()),
        "prediction_cache": str(Path(prediction_cache).resolve()),
        "prediction_cache_hit": cache_hit,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return results


def _expand_choice(value: str, choices: Sequence[str]) -> tuple[str, ...]:
    if value == "all":
        return tuple(choices)
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = set(requested) - set(choices)
    if invalid:
        raise ValueError(f"Unsupported values {sorted(invalid)}; choose from {choices}")
    return requested


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory of ordered input frames")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-cache", default=None)
    parser.add_argument("--force-prediction", action="store_true")
    parser.add_argument("--model-id", default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--device", default=None, help="For example cuda or cuda:0")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--mode",
        default="vertical_plane",
        help="Projection mode, comma-separated modes, or all",
    )
    parser.add_argument(
        "--reference-mode",
        default="mid",
        help="Reference mode, comma-separated modes, or all",
    )
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument(
        "--panini-distance",
        type=float,
        default=PANINI_DISTANCE,
        help="Panini d in [0, 1]: 0 is rectilinear, 1 is classic Panini",
    )
    parser.add_argument(
        "--general-panini-compression",
        type=float,
        default=GENERAL_PANINI_COMPRESSION,
        help="General Panini Cmpr in [0, 150]",
    )
    parser.add_argument(
        "--general-panini-top",
        type=float,
        default=GENERAL_PANINI_TOP_SQUEEZE,
        help="General Panini Tops squeeze in [-100, 100]",
    )
    parser.add_argument(
        "--general-panini-bottom",
        type=float,
        default=GENERAL_PANINI_BOTTOM_SQUEEZE,
        help="General Panini Bots squeeze in [-100, 100]",
    )
    parser.add_argument("--bounds-quantile", type=float, default=0.001)
    parser.add_argument("--max-canvas-megapixels", type=float, default=40.0)
    parser.add_argument(
        "--save-npy",
        dest="save_numpy",
        action="store_true",
        help="Persist large per-view depth/overlap NumPy caches (disabled by default)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not np.isfinite(args.panini_distance) or not 0.0 <= args.panini_distance <= 1.0:
        raise ValueError("--panini-distance must be finite and in [0, 1]")
    if (
        not np.isfinite(args.general_panini_compression)
        or not 0.0 <= args.general_panini_compression <= 150.0
    ):
        raise ValueError("--general-panini-compression must be finite and in [0, 150]")
    for name in ("general_panini_top", "general_panini_bottom"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not -100.0 <= value <= 100.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and in [-100, 100]")
    modes = _expand_choice(args.mode, PROJECTION_MODES)
    references = _expand_choice(args.reference_mode, REFERENCE_MODES)
    results = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        prediction_cache=args.prediction_cache,
        model_id=args.model_id,
        model_cache_dir=args.model_cache_dir,
        device=args.device,
        process_res=args.process_res,
        force_prediction=args.force_prediction,
        modes=modes,
        references=references,
        render_stride=args.render_stride,
        bounds_quantile=args.bounds_quantile,
        max_canvas_pixels=max(1, int(args.max_canvas_megapixels * 1_000_000)),
        save_numpy=args.save_numpy,
        panini_distance=args.panini_distance,
        general_panini_compression=args.general_panini_compression,
        general_panini_top=args.general_panini_top,
        general_panini_bottom=args.general_panini_bottom,
    )
    ranking = sorted(
        (
            (name, metrics["prestitch_score"], metrics["challenge_score"])
            for name, metrics in results.items()
        ),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    print(json.dumps({"ranking": ranking}, indent=2))


if __name__ == "__main__":
    main()
