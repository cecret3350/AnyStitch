import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "da3_prestitch.py"
SPEC = importlib.util.spec_from_file_location("da3_prestitch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def mock_prediction(count=2, processed_height=12, processed_width=16):
    processed = np.zeros(
        (count, processed_height, processed_width, 3), dtype=np.uint8
    )
    depth = np.stack(
        [
            np.full((processed_height, processed_width), 4.0 + index)
            for index in range(count)
        ]
    )
    intrinsics = []
    extrinsics = []
    for index in range(count):
        intrinsics.append(
            np.array(
                [
                    [10.0 + index * 2, 0.0, 7.5],
                    [0.0, 11.0 + index * 3, 5.5],
                    [0.0, 0.0, 1.0],
                ]
            )
        )
        extrinsic = np.eye(4)
        extrinsic[0, 3] = -0.15 * index
        extrinsics.append(extrinsic)
    return SimpleNamespace(
        processed_images=processed,
        depth=depth,
        intrinsics=np.stack(intrinsics),
        extrinsics=np.stack(extrinsics),
    )


def make_inputs(tmp_path, sizes=((32, 24), (48, 36))):
    paths = []
    images = []
    for index, (width, height) in enumerate(sizes):
        yy, xx = np.mgrid[:height, :width]
        image = np.stack(
            (
                (xx * 7 + index * 30) % 255,
                (yy * 11 + index * 20) % 255,
                ((xx + yy) * 5) % 255,
            ),
            axis=2,
        ).astype(np.uint8)
        path = tmp_path / f"{index:06d}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
        images.append(image)
    return paths, images


def test_discover_images_excludes_contact_sheet(tmp_path):
    paths, _ = make_inputs(tmp_path)
    Image.new("RGB", (64, 64)).save(tmp_path / "contact.jpg")

    assert MODULE.discover_images(tmp_path) == paths


def test_latent_coord_map_invalid_depth_does_not_pull_coordinates_toward_origin():
    # Only half the bilinear footprint has geometry. Previously the -1 holes
    # changed (50,25) into (24.96875,12.46875) in latent coordinates.
    result = MODULE._latent_coord_map(
        np.array([[7, 7], [7, 8]]), np.array([[800, 400], [800, 400]]),
        (16, 16), (1600, 1000),
    )
    np.testing.assert_array_equal(result, [[[50.0, 25.0]]])


def test_latent_coord_map_without_valid_support_is_sentinel():
    result = MODULE._latent_coord_map(
        np.array([[7, 7], [8, 8]]), np.array([[1601, 400], [np.nan, 400]]),
        (16, 16), (1600, 1000),
    )
    np.testing.assert_array_equal(result, [[[1000.0, 1000.0]]])


def test_latent_coord_map_manual_mask_still_applies_after_interpolation():
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 0] = True
    result = MODULE._latent_coord_map(
        np.array([[7, 7], [7, 8]]), np.array([[800, 400], [800, 400]]),
        (16, 16), (1600, 1000), source_mask=mask,
    )
    np.testing.assert_array_equal(result, [[[1000.0, 1000.0]]])


def test_valid_coordinate_resize_matches_bilinear_for_dense_maps():
    import cv2
    values = np.arange(11 * 17 * 2, dtype=np.float32).reshape(11, 17, 2)
    result = MODULE._resize_valid_coord_map(values, np.ones((11, 17), dtype=bool), (5, 4))
    np.testing.assert_array_equal(result, cv2.resize(values, (5, 4), interpolation=cv2.INTER_LINEAR))


def test_valid_coordinate_resize_excludes_nan_and_invalid_pairs():
    values = np.array([[[50, 25], [np.nan, 9]], [[1000, 1000], [-1, -1]]], dtype=np.float32)
    valid = np.array([[True, True], [False, False]])
    result = MODULE._resize_valid_coord_map(values, valid, (1, 1))
    np.testing.assert_array_equal(result, [[[50, 25]]])


def test_sparse_resize_tiny_weights_do_not_amplify_numerical_error():
    rng = np.random.default_rng(219)
    valid = rng.random((1077, 1916)) > .3
    coordinates = np.empty((1077, 1916, 2), dtype=np.float32)
    coordinates[:] = [4474.999, 901.999]
    result = MODULE._resize_valid_coord_map(coordinates, valid, (120, 68))
    supported = (result != 1000).any(axis=-1)
    np.testing.assert_allclose(result[supported], np.broadcast_to(coordinates[0, 0], result[supported].shape), atol=.001, rtol=0)


def test_prepare_views_uses_each_intrinsic_and_each_size(tmp_path):
    paths, images = make_inputs(tmp_path)
    prediction = mock_prediction()
    views = MODULE.prepare_views(images, prediction, paths)

    # View 0: 16x12 -> 32x24, so both intrinsic rows scale by 2.
    np.testing.assert_allclose(views[0].K[0], prediction.intrinsics[0, 0] * 2)
    np.testing.assert_allclose(views[0].K[1], prediction.intrinsics[0, 1] * 2)
    # View 1 has a distinct K and a distinct 3x per-view resize.
    np.testing.assert_allclose(views[1].K[0], prediction.intrinsics[1, 0] * 3)
    np.testing.assert_allclose(views[1].K[1], prediction.intrinsics[1, 1] * 3)
    assert not np.array_equal(views[0].K, views[1].K)
    assert views[0].depth.shape == (24, 32)
    assert views[1].depth.shape == (36, 48)


def test_prediction_cache_avoids_second_inference(tmp_path):
    paths, _ = make_inputs(tmp_path)
    calls = []

    def inference(items):
        calls.append(tuple(items))
        return mock_prediction()

    cache = tmp_path / "prediction.npz"
    first, first_hit = MODULE.get_or_run_prediction(
        paths, cache, "mock-model", inference_fn=inference
    )
    second, second_hit = MODULE.get_or_run_prediction(
        paths, cache, "mock-model", inference_fn=inference
    )
    assert not first_hit
    assert second_hit
    assert len(calls) == 1
    np.testing.assert_array_equal(first.intrinsics, second.intrinsics)


def test_manual_canvas_preserves_target_pixel_scale():
    item = MODULE.ProjectedView(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones(2),
        uv=np.array([[-10.0, -5.0], [10.0, 5.0]]),
        colors=np.zeros((2, 3), dtype=np.float32),
        feather=np.ones(2, dtype=np.float32),
    )
    transformed, size, meta = MODULE._canvas_transform(
        [item], quantile=0, padding=8, max_canvas_pixels=1, requested_size=(100, 80)
    )
    np.testing.assert_allclose(transformed[0], item.uv + [50.0, 40.0])
    assert size == (100, 80)
    assert meta["canvas_scale"] == 1.0


def test_auto_canvas_uses_integer_raster_bounds_without_extra_pixel():
    item = MODULE.ProjectedView(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones(2),
        # Continuous extent is 1.8 px, but np.round produces pixels 1 and 2,
        # so the inclusive raster crop is exactly 2 px wide.
        uv=np.array([[0.6, 0.6], [2.4, 2.4]]),
        colors=np.zeros((2, 3), dtype=np.float32),
        feather=np.ones(2, dtype=np.float32),
    )
    _transformed, size, meta = MODULE._canvas_transform(
        [item], quantile=0, padding=0, max_canvas_pixels=None
    )
    assert size == (2, 2)
    assert meta["raw_canvas_size"] == [2, 2]


def test_classic_panini_projection_has_expected_center_and_edge_scale():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    uv, valid = MODULE._project(
        points, "panini", focal=100.0, focal_y=120.0, global_scale=1.0
    )

    assert valid.all()
    np.testing.assert_allclose(uv[0], [0.0, 0.0], atol=1e-12)
    # At the central meridian, Panini has the same vertical scale as a
    # rectilinear target camera.
    np.testing.assert_allclose(uv[1], [0.0, 120.0], atol=1e-12)
    # d=1 maps +/-45 degrees to +/-2*tan(22.5 degrees), less stretched
    # than rectilinear's +/-1 while retaining left/right symmetry.
    expected_x = 200.0 * np.tan(np.radians(22.5))
    np.testing.assert_allclose(uv[2:, 0], [expected_x, -expected_x], rtol=1e-12)


def test_panini_distance_zero_matches_rectilinear_in_front_hemisphere():
    points = np.array(
        [[-0.5, 0.2, 1.0], [0.0, -0.3, 2.0], [1.0, 0.5, 1.5]],
        dtype=np.float64,
    )
    rectilinear, rect_valid = MODULE._project(
        points, "rectilinear", focal=100.0, focal_y=120.0, global_scale=1.0
    )
    panini, panini_valid = MODULE._project(
        points,
        "panini",
        focal=100.0,
        focal_y=120.0,
        global_scale=1.0,
        panini_distance=0.0,
    )

    np.testing.assert_array_equal(panini_valid, rect_valid)
    np.testing.assert_allclose(panini, rectilinear, rtol=1e-12, atol=1e-12)


def test_panini_rejects_only_the_rear_axis_singularity():
    points = np.array(
        [[0.0, 0.0, -1.0], [0.1, 0.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    uv, valid = MODULE._project(
        points, "panini", focal=100.0, global_scale=1.0
    )

    np.testing.assert_array_equal(valid, [False, True, True])
    assert np.isfinite(uv[valid]).all()


def test_general_panini_compression_zero_matches_rectilinear_coordinates():
    # All rays are inside Hugin's 160-degree feasible FOV at Cmpr=0.
    points = np.array(
        [[-0.5, 0.2, 1.0], [0.0, -0.3, 2.0], [1.0, 0.5, 1.5]],
        dtype=np.float64,
    )
    rectilinear, rect_valid = MODULE._project(
        points, "rectilinear", focal=100.0, focal_y=120.0, global_scale=1.0
    )
    general, general_valid = MODULE._project(
        points,
        "general_panini",
        focal=100.0,
        focal_y=120.0,
        global_scale=1.0,
        general_panini_compression=0.0,
    )

    np.testing.assert_array_equal(general_valid, rect_valid)
    np.testing.assert_allclose(general, rectilinear, rtol=1e-12, atol=1e-12)


def test_general_panini_top_hard_squeeze_does_not_change_bottom_half():
    points = np.array(
        [[1.0, -0.4, 1.0], [1.0, 0.4, 1.0]], dtype=np.float64
    )
    base, base_valid = MODULE._project(
        points,
        "general_panini",
        focal=100.0,
        global_scale=1.0,
        general_panini_compression=100.0,
    )
    squeezed, squeezed_valid = MODULE._project(
        points,
        "general_panini",
        focal=100.0,
        global_scale=1.0,
        general_panini_compression=100.0,
        general_panini_top=100.0,
        general_panini_bottom=0.0,
    )

    assert base_valid.all() and squeezed_valid.all()
    # A 100% hard squeeze maps v to v/cos(longitude) in the upper half.
    np.testing.assert_allclose(
        squeezed[0, 1], base[0, 1] / np.cos(np.radians(45.0)), rtol=1e-12
    )
    np.testing.assert_allclose(squeezed[1], base[1], rtol=1e-12, atol=1e-12)


def test_general_panini_compression_150_prevents_rear_foldover():
    angles = np.radians([80.0, 120.0])
    points = np.stack(
        (np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=1
    )
    _uv, valid = MODULE._project(
        points,
        "general_panini",
        focal=100.0,
        global_scale=1.0,
        general_panini_compression=150.0,
    )

    np.testing.assert_array_equal(valid, [True, False])


def _yaw_view(angle_degrees, name):
    angle = np.radians(angle_degrees)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    height, width = 12, 16
    intrinsic = np.array(
        [[8.0, 0.0, 7.5], [0.0, 8.0, 5.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return MODULE.ViewGeometry(
        index=int(angle_degrees),
        path=name,
        image=np.zeros((height, width, 3), dtype=np.uint8),
        depth=np.ones((height, width), dtype=np.float32),
        K=intrinsic,
        K_processed=intrinsic.copy(),
        R=rotation,
        T=np.zeros(3, dtype=np.float64),
        processed_size=(width, height),
    )


def test_periodic_projection_unwrap_uses_geometry_not_input_order():
    # Deliberately place +137 and -128 degrees next to each other in the input
    # list, analogous to lexicographic filename order 0, 10, 1, ... .  The
    # result must depend only on camera directions.  With this test's wide
    # horizontal FOV, the +137-degree view also crosses atan2's +/-pi seam
    # internally and must remain one compact patch.
    angles = [137, -128, 110, 81, 55, 28, 4, -26, -52, -80, -110]
    views = [_yaw_view(angle, f"view_{angle}") for angle in angles]
    target_camera = (np.zeros(3), np.eye(3), 8.0)

    for mode in ("cylindrical", "equirectangular"):
        def centers_by_path(items):
            samples = MODULE.unproject_views(items, 1)
            projected, _ = MODULE.project_views(
                items,
                mode,
                "avg",
                unprojected=samples,
                target_camera=target_camera,
            )
            # No individual source view may remain split across a full-period
            # seam after the geometry-relative per-point unwrap.
            assert all(np.ptp(item.uv[:, 0]) < np.pi * 8.0 for item in projected)
            return {
                view.path: float(np.median(item.uv[:, 0]) / 8.0)
                for view, item in zip(items, projected)
            }

        forward = centers_by_path(views)
        shuffled = centers_by_path([views[i] for i in (7, 2, 10, 4, 0, 8, 1, 5, 3, 9, 6)])
        assert forward.keys() == shuffled.keys()
        for path in forward:
            np.testing.assert_allclose(forward[path], shuffled[path], atol=1e-12)

        left_to_right = [
            path for path, _ in sorted(forward.items(), key=lambda pair: pair[1])
        ]
        assert left_to_right == [
            "view_-128", "view_-110", "view_-80", "view_-52", "view_-26",
            "view_4", "view_28", "view_55", "view_81", "view_110", "view_137",
        ]


def test_mock_pipeline_renders_every_projection_and_reference(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    make_inputs(input_dir, sizes=((32, 24), (32, 24)))
    output_dir = tmp_path / "outputs"

    results = MODULE.run_pipeline(
        input_path=input_dir,
        output_dir=output_dir,
        modes=MODULE.PROJECTION_MODES,
        references=MODULE.REFERENCE_MODES,
        render_stride=2,
        bounds_quantile=0,
        max_canvas_pixels=1_000_000,
        inference_fn=lambda _: mock_prediction(),
    )

    assert len(results) == len(MODULE.PROJECTION_MODES) * len(MODULE.REFERENCE_MODES)
    for mode in MODULE.PROJECTION_MODES:
        for reference in MODULE.REFERENCE_MODES:
            result_dir = output_dir / mode / reference
            metrics = json.loads((result_dir / "geometry_metrics.json").read_text())
            camera = json.loads((result_dir / "camera_intrinsics.json").read_text())
            assert metrics["projection_mode"] == mode
            assert metrics["reference_mode"] == reference
            assert metrics["num_views"] == 2
            assert len(camera["views"]) == 2
            assert (result_dir / "composite.png").is_file()
            assert (result_dir / "overlap_heatmap.png").is_file()
            for index in range(2):
                base = result_dir / "projected_views" / f"{index:06d}"
                assert Path(f"{base}_image.png").is_file()
                assert not Path(f"{base}_depth.npy").exists()
                assert Path(f"{base}_depth.png").is_file()
                assert Path(f"{base}_mask.png").is_file()

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["prediction_cache_hit"] is False
