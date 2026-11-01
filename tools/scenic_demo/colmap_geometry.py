"""Load per-frame camera/depth geometry from a COLMAP dense workspace.

The loader intentionally returns the same :class:`ViewGeometry` objects used
by the DA3 path.  This lets the interactive renderer use COLMAP geometry
without introducing a second projection or coordinate convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:  # Support both package and direct-script invocation.
    from .da3_prestitch import ViewGeometry
except ImportError:  # pragma: no cover - direct CLI path.
    from da3_prestitch import ViewGeometry  # type: ignore


def _read_colmap_array(path: Path) -> np.ndarray:
    """Read COLMAP's ``width&height&channels&<float32>`` dense-array format."""

    with path.open("rb") as handle:
        fields: list[bytes] = []
        for _ in range(3):
            field = bytearray()
            while True:
                byte = handle.read(1)
                if not byte:
                    raise ValueError(f"Truncated COLMAP dense array header: {path}")
                if byte == b"&":
                    break
                field.extend(byte)
            fields.append(bytes(field))
        width, height, channels = (int(item) for item in fields)
        values = np.fromfile(handle, dtype=np.float32)
    expected = width * height * channels
    if values.size != expected:
        raise ValueError(
            f"Invalid COLMAP dense array {path}: expected {expected} float32 values, "
            f"got {values.size}"
        )
    array = values.reshape((height, width, channels))
    return array[:, :, 0] if channels == 1 else array


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == (height, width):
        return depth.copy()
    # Match DA3's original-resolution conversion. COLMAP's depth maps are
    # measured in the undistorted dense-camera pixel grid.
    return np.asarray(
        Image.fromarray(depth).resize((width, height), Image.Resampling.BILINEAR)
    ).copy()


def _workspace_paths(root: Path) -> tuple[Path, Path]:
    """Resolve ``(sparse_model, dense_root)`` from common workspace layouts."""

    candidates = [
        (root / "dense" / "sparse", root / "dense"),
        (root / "sparse_fixed" / "0", root / "dense"),
        (root / "sparse" / "0", root / "dense"),
        (root / "sparse", root),
    ]
    for sparse, dense in candidates:
        if sparse.exists() and (sparse / "images.bin").exists():
            return sparse, dense
    raise FileNotFoundError(
        f"No COLMAP sparse model found below {root}; expected sparse/0 or dense/sparse"
    )


def load_colmap_views(
    images: Sequence[np.ndarray],
    image_paths: Sequence[str | Path],
    workspace: str | Path,
) -> tuple[list[ViewGeometry], dict[str, object]]:
    """Load geometry for the requested frames from a full COLMAP reconstruction.

    ``image_paths`` may be any ordered subset of the full reconstruction. Frame
    matching is by basename, while RGB pixels remain the user-provided input
    images. COLMAP's dense camera is typically an undistorted image grid, so
    its intrinsics and depth map are scaled to each input image's resolution.
    """

    if len(images) != len(image_paths):
        raise ValueError("images and image_paths must have the same length")
    if not images:
        raise ValueError("At least one input image is required")
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("COLMAP geometry requires the pycolmap package") from exc

    sparse_model, dense_root = _workspace_paths(Path(workspace).expanduser())
    depth_root = dense_root / "stereo" / "depth_maps"
    if not depth_root.is_dir():
        raise FileNotFoundError(f"COLMAP depth-map directory does not exist: {depth_root}")
    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    by_name = {Path(image.name).name: image for image in reconstruction.images.values()}

    views: list[ViewGeometry] = []
    depth_types: list[str] = []
    for index, (rgb, image_path) in enumerate(zip(images, image_paths)):
        basename = Path(image_path).name
        colmap_image = by_name.get(basename)
        if colmap_image is None:
            raise KeyError(
                f"Frame {basename!r} is not present in COLMAP reconstruction {sparse_model}."
            )
        camera = reconstruction.cameras[colmap_image.camera_id]
        dense_width, dense_height = int(camera.width), int(camera.height)
        depth_path = depth_root / f"{basename}.geometric.bin"
        depth_kind = "geometric"
        if not depth_path.exists():
            depth_path = depth_root / f"{basename}.photometric.bin"
            depth_kind = "photometric"
        if not depth_path.exists():
            raise FileNotFoundError(f"No dense depth map found for frame {basename}: {depth_root}")

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image {index} must be RGB HxWx3, got {image.shape}")
        height, width = image.shape[:2]
        depth = _resize_depth(_read_colmap_array(depth_path), width, height)
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        K[0, :] *= width / dense_width
        K[1, :] *= height / dense_height
        cam_from_world = colmap_image.cam_from_world
        # COLMAP stores p_cam = R_cw p_world + t_cw (column-vector form).
        # ViewGeometry's row-vector convention is p_world=(p_cam-T)@R,
        # therefore R=R_cw and T=t_cw.
        R = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        T = np.asarray(cam_from_world.translation, dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0)
        views.append(
            ViewGeometry(
                index=index,
                path=str(image_path),
                image=image.astype(np.uint8, copy=False),
                depth=depth,
                K=K.copy(),
                K_processed=K.copy(),
                R=R,
                T=T,
                processed_size=(dense_width, dense_height),
                confidence=valid.astype(np.float32),
            )
        )
        depth_types.append(depth_kind)

    return views, {
        "workspace": str(Path(workspace).expanduser().resolve()),
        "sparse_model": str(sparse_model),
        "dense_root": str(dense_root),
        "num_registered_images": reconstruction.num_images(),
        "num_input_views": len(views),
        "depth_types": depth_types,
        "camera_model": "COLMAP calibrated dense camera",
    }
