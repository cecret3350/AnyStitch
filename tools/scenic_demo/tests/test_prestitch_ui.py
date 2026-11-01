from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.scenic_demo.da3_prestitch import (
    prepare_views,
    project_views,
    render_prestitch_memory,
    target_camera_components,
    unproject_views,
)
from tools.scenic_demo.prestitch_ui import (
    _apply_tto_acceptance,
    _adjust_views,
    _camera_frustum_segments,
    _edit_target_camera,
    _optimize_target_rotation,
    _rotation_xyz,
    build_scene_glb,
    _resize_flux_source,
    _flux_tile_regions,
)
import tools.scenic_demo.prestitch_ui as prestitch_ui
from PIL import Image


def _views():
    images = [np.zeros((24, 32, 3), dtype=np.uint8), np.full((24, 32, 3), 120, dtype=np.uint8)]
    processed = np.zeros((2, 12, 16, 3), dtype=np.uint8)
    depth = np.full((2, 12, 16), 4.0, dtype=np.float32)
    intrinsics = np.array(
        [
            [[10, 0, 7.5], [0, 11, 5.5], [0, 0, 1]],
            [[12, 0, 7.5], [0, 14, 5.5], [0, 0, 1]],
        ],
        dtype=np.float64,
    )
    extrinsics = np.stack([np.eye(4), np.eye(4)])
    extrinsics[1, 0, 3] = -0.2
    prediction = SimpleNamespace(
        processed_images=processed,
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    return prepare_views(images, prediction, ["000.png", "001.png"])


def test_mask_editor_uploads_only_on_explicit_apply(monkeypatch):
    import pytest

    pytest.importorskip("gradio")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    demo = prestitch_ui.build_demo(device="cpu")
    config = demo.get_config_file()
    editor = next(c for c in config["components"] if c["type"] == "imageeditor")
    assert editor["props"]["brush"]["colors"] == ["#ff0000", "#0000ff"]
    assert editor["props"]["brush"]["default_color"] == "#ff0000"
    assert editor["props"]["brush"]["color_mode"] == "fixed"
    apply_button = next(
        c for c in config["components"]
        if c["type"] == "button" and c["props"]["value"] == "Apply mask"
    )
    # Even a listener without editor inputs would turn on frontend realtime
    # blob uploads, so inspect event targets as well as callback inputs.
    assert not any(
        target == editor["id"] and event in {"change", "input"}
        for dependency in config["dependencies"]
        for target, event in dependency["targets"]
    )
    readers = [f for f in demo.fns.values() if any(c._id == editor["id"] for c in f.inputs)]
    assert len(readers) == 1
    assert readers[0].fn is prestitch_ui.update_input_mask
    assert readers[0].targets == [(apply_button["id"], "click")]
    assert any(f.fn is prestitch_ui.render_case for f in demo.fns.values())
    # Gradio retains progress targets from earlier events and refreshes them
    # on later events. A status-only update reads the component's get_data(),
    # which makes ImageEditor upload even with no change/input listeners.
    # Checking callback inputs alone misses this second automatic upload path.
    for dependency in config["dependencies"]:
        progress_targets = dependency.get("show_progress_on")
        if progress_targets is None:
            progress_targets = dependency["outputs"]
        assert editor["id"] not in progress_targets


def test_memory_render_returns_requested_canvas_without_files(tmp_path):
    composite, heatmap, metrics = render_prestitch_memory(
        _views(), mode="vertical_plane", canvas_size=(320, 160), render_stride=2
    )
    assert composite.shape == (160, 320, 3)
    assert heatmap.shape == composite.shape
    assert metrics["canvas_size"] == [320, 160]
    assert list(tmp_path.iterdir()) == []


def test_render_case_canvas_quantile_uses_same_demo_projection_and_maps(monkeypatch):
    monkeypatch.setattr(prestitch_ui, "build_scene_glb", lambda *args, **kwargs: None)
    views = _views()
    kwargs = dict(mode="vertical_plane", reference="avg", render_stride=1,
        input_resize_scale=1., testtime_optimization=False,
        force_apply_optimized_camera=False, auto_canvas=True,
        filter_canvas_outliers=True, visibility_aware=False,
        auto_canvas_downscale=False, auto_canvas_max_mp=8.,
        canvas_width=160, canvas_height=96, focal_scale=1.,
        cx_offset=0., cy_offset=0., rx=0., ry=0., rz=0., tx=0., ty=0., tz=0.)
    for quantile, enabled in [(None, True), (.02, True), (.02, False)]:
        state = dict(views=views, base_views=views, da3_views=views)
        options = dict(kwargs, filter_canvas_outliers=enabled)
        if quantile is not None:
            options["canvas_bounds_quantile"] = quantile
        actual, _, status = prestitch_ui.render_case(state, **options)
        assert actual is not None, status
        q = (0.001 if quantile is None else quantile) if enabled else 0.0
        snapshot = state["last_render"]
        assert snapshot["metrics"]["bounds_quantile"] == q
        expected, _, metrics = render_prestitch_memory(
            views, mode="vertical_plane", reference_mode="avg", render_stride=1,
            bounds_quantile=q, padding=0, max_canvas_pixels=None,
            target_camera=snapshot["target_camera"], return_coord_maps=True,
            return_heatmap=False)
        np.testing.assert_array_equal(actual, expected)
        for key in ["_coord_maps", "_coord_maps_depth"]:
            for got, want in zip(snapshot["metrics"][key], metrics[key]):
                np.testing.assert_array_equal(got, want)
    for bad in [float("nan"), -.01, .5]:
        image, _, status = prestitch_ui.render_case(state, **kwargs, canvas_bounds_quantile=bad)
        assert image is None
        assert "canvas_bounds_quantile" in status


def test_visibility_aware_fusion_prefers_nearer_target_surface():
    base = _views()[0]
    far = replace(
        base,
        index=1,
        path="001.png",
        image=np.full_like(base.image, 120),
        depth=np.full_like(base.depth, 8.0),
    )
    near = replace(base, index=0, depth=np.full_like(base.depth, 4.0))
    regular, _, regular_metrics = render_prestitch_memory(
        [near, far], mode="vertical_plane", reference_mode="first",
        render_stride=1, canvas_size=(32, 24), return_heatmap=False,
    )
    visible, _, visible_metrics = render_prestitch_memory(
        [near, far], mode="vertical_plane", reference_mode="first",
        render_stride=1, canvas_size=(32, 24), visibility_aware=True,
        return_heatmap=False,
    )
    # Both views project onto the same plane.  Ordinary averaging mixes black
    # and white-ish input, while the z-buffer keeps the nearer (black) view.
    assert float(regular.mean()) > 20.0
    assert float(visible.mean()) < float(regular.mean())
    assert visible_metrics["fusion_mode"] == "visibility_aware_zbuffer"
    assert regular_metrics["fusion_mode"] == "weighted_forward_splat"


def test_flux_source_resize_does_not_interpolate_invalid_sentinels():
    image = Image.new("RGB", (19, 17))
    coords = np.full((2, 2, 2), 1000.0, dtype=np.float32)
    coords[0, 0] = [50, 25]
    resized, actual = _resize_flux_source(image, coords)
    assert resized.size == (16, 16)
    np.testing.assert_array_equal(actual, [[[50, 25]]])


def test_flux_source_resize_keeps_unsupported_cells_invalid():
    image = Image.new("RGB", (19, 17))
    coords = np.full((2, 2, 2), 1000.0, dtype=np.float32)
    _, actual = _resize_flux_source(image, coords)
    np.testing.assert_array_equal(actual, [[[1000, 1000]]])


def test_flux_coord_maps_share_displayed_canvas_coordinates():
    views = _views()
    composite, _, metrics = render_prestitch_memory(
        views,
        mode="vertical_plane",
        reference_mode="first",
        render_stride=1,
        canvas_size=(320, 160),
        return_coord_maps=True,
    )
    assert composite.shape == (160, 320, 3)
    maps = metrics["_coord_maps"]
    maps_depth = metrics["_coord_maps_depth"]
    assert len(maps) == len(views) == len(maps_depth)
    assert maps[0].shape == (2, 2, 2)
    # Flux maps are target-pixel coordinates in latent units.  With a manual
    # 320x160 canvas, the first-view origin is exactly (160, 80) pixels.
    valid = maps_depth[0][..., 0] < 1000
    assert np.any(valid)
    assert np.all(maps_depth[0][valid] >= 0)
    assert np.all(maps_depth[0][valid, 0] < 320 / 16)
    assert np.all(maps_depth[0][valid, 1] < 160 / 16)


def test_flux_source_is_resized_to_16_multiple_without_black_padding():
    image = Image.fromarray(np.full((17, 19, 3), 128, dtype=np.uint8))
    coord_map = np.zeros((2, 2, 2), dtype=np.float32)
    resized, resized_map = _resize_flux_source(image, coord_map)
    assert resized.size == (16, 16)
    assert np.asarray(resized).mean() > 100
    assert resized_map.shape == (1, 1, 2)
    np.testing.assert_array_equal(resized_map, coord_map.mean(axis=(0, 1), keepdims=True))


def test_flux_tiles_follow_reference_center_sequence():
    regions = _flux_tile_regions(3000, 1500, tile=1024, stride=768)
    assert regions[0] == (1536, 476, 2560, 1500)
    assert regions[1] == (1536, 0, 2560, 1024)
    assert regions[-1] == (0, 0, 1024, 1024)


def test_cached_unprojection_matches_regular_render():
    views = _views()
    samples = unproject_views(views, render_stride=2)
    cached, _, cached_metrics = render_prestitch_memory(
        views, mode="vertical_plane", render_stride=2, unprojected=samples
    )
    regular, _, regular_metrics = render_prestitch_memory(
        views, mode="vertical_plane", render_stride=2
    )
    np.testing.assert_array_equal(cached, regular)
    assert cached_metrics["canvas_size"] == regular_metrics["canvas_size"]


def test_adjust_views_does_not_mutate_da3_geometry():
    views = _views()
    original_k = views[0].K.copy()
    original_center = views[0].camera_center.copy()
    adjusted = _adjust_views(views, 1.2, 3, -2, 5, 0, 0, 1, 2, 3)
    assert not np.array_equal(adjusted[0].K, original_k)
    assert not np.array_equal(adjusted[0].camera_center, original_center)
    np.testing.assert_array_equal(views[0].K, original_k)
    np.testing.assert_array_equal(views[0].camera_center, original_center)
    np.testing.assert_allclose(_rotation_xyz(0, 0, 0), np.eye(3))


def test_selected_view_extrinsic_edit_changes_relative_pose():
    views = _views()
    adjusted = _adjust_views(views, 1.0, 0, 0, 0, 0, 0, 1, 0, 0, 1)
    np.testing.assert_array_equal(adjusted[0].R, views[0].R)
    np.testing.assert_array_equal(adjusted[0].T, views[0].T)
    assert not np.array_equal(adjusted[1].camera_center, views[1].camera_center)


def test_force_apply_uses_rejected_optimization_candidate(monkeypatch):
    views = _views()
    candidate = list(views)
    candidate[0] = replace(candidate[0], T=candidate[0].T + np.array([0.1, 0.0, 0.0]))
    stats = {
        "accepted": False,
        "optimizer_success": True,
        "num_matches": 78,
        "initial_heldout_median_px": 7.61,
        "optimized_heldout_median_px": 7.92,
        "rotation_degrees": 1.12,
        "focal_scale": 0.963,
    }
    monkeypatch.setattr(prestitch_ui, "_refine_multiview_camera", lambda *_args: (candidate, stats))
    state = {"da3_views": views, "testtime_optimization_cache": {}}

    strict, strict_status = prestitch_ui._testtime_optimize_views(state, "mid", False)
    forced, forced_status = prestitch_ui._testtime_optimize_views(state, "mid", True)
    np.testing.assert_array_equal(strict[0].T, views[0].T)
    np.testing.assert_array_equal(forced[0].T, candidate[0].T)
    assert "rejected" in strict_status
    assert "FORCED apply" in forced_status


def test_tto_dense_gate_rejects_sparse_improvement_with_worse_prestitch():
    stats = {
        "accepted": True,
        "sparse_accepted": True,
        "optimizer_success": True,
        "rotation_degrees": 8.33,
        "translation_shift_depth_ratio": 0.08,
    }
    before = {
        "canvas_coverage_ratio": 0.938,
        "canvas_hole_ratio": 0.062,
        "prestitch_score": 91.86,
        "overlap_color_disagreement": 0.044,
    }
    candidate = {
        "canvas_coverage_ratio": 0.844,
        "canvas_hole_ratio": 0.156,
        "prestitch_score": 85.98,
        "overlap_color_disagreement": 0.067,
    }
    gated = _apply_tto_acceptance(stats, before, candidate)
    assert not gated["accepted"]
    assert not gated["dense_accepted"]
    assert not gated["pose_step_accepted"]
    assert "coverage_drop_le_0.025" in gated["acceptance_reasons"]
    assert "rotation_le_6deg" in gated["acceptance_reasons"]


def test_tto_dense_gate_accepts_small_non_degrading_update():
    stats = {
        "accepted": True,
        "sparse_accepted": True,
        "optimizer_success": True,
        "rotation_degrees": 1.5,
        "translation_shift_depth_ratio": 0.02,
    }
    before = {
        "canvas_coverage_ratio": 0.90,
        "canvas_hole_ratio": 0.10,
        "prestitch_score": 85.0,
        "overlap_color_disagreement": 0.05,
    }
    candidate = {
        "canvas_coverage_ratio": 0.91,
        "canvas_hole_ratio": 0.09,
        "prestitch_score": 86.0,
        "overlap_color_disagreement": 0.048,
    }
    gated = _apply_tto_acceptance(stats, before, candidate)
    assert gated["accepted"]
    assert gated["dense_accepted"]
    assert gated["pose_step_accepted"]
    assert gated["acceptance_reasons"] == []


def test_multiview_refinement_uses_all_sequence_pairs(monkeypatch):
    views = _views()
    views.append(replace(views[0], index=2, path="002.png", image=np.full_like(views[0].image, 80)))
    original_depths = [view.depth.copy() for view in views]
    points = np.asarray(
        [[x, y] for y in (5, 10, 15, 20) for x in (5, 10, 15, 20)],
        dtype=np.float64,
    )
    monkeypatch.setattr(prestitch_ui, "_sift_magsac_matches_pair", lambda _pair: (points, points.copy()))
    refined, stats = prestitch_ui._refine_multiview_camera(views, reference_index=1)
    assert len(refined) == 3
    assert stats["num_views"] == 3
    assert stats["num_pairs"] == 2  # (0,1) and (1,2); reference links are duplicates.
    assert stats["num_matches"] == 32
    assert stats["depth_scale"] == 1.0
    for original_depth, refined_view in zip(original_depths, refined):
        np.testing.assert_array_equal(refined_view.depth, original_depth)


def test_target_rotation_is_in_place():
    views = _views()
    adjusted = _adjust_views(views, 1.0, 0, 0, 0, 0, 15, 0, 0, 0, 1)
    np.testing.assert_allclose(adjusted[1].camera_center, views[1].camera_center, atol=1e-12)
    assert not np.array_equal(adjusted[1].R, views[1].R)


def test_target_rotation_optimizer_reduces_tight_canvas_black_area():
    views = _views()
    original_poses = [(view.R.copy(), view.T.copy()) for view in views]
    rotation, stats = _optimize_target_rotation(
        views,
        mode="rectilinear",
        reference="mid",
        initial_rotation=(0.0, 0.0, 25.0),
        focal_scale=1.0,
        cx_offset=0.0,
        cy_offset=0.0,
        tx=0.0,
        ty=0.0,
        tz=0.0,
        bounds_quantile=0.0,
    )
    assert stats["applied"]
    assert stats["optimized_black_fraction"] < stats["initial_black_fraction"]
    assert abs(rotation[2]) < 10.0
    for view, (rotation_before, translation_before) in zip(views, original_poses):
        np.testing.assert_array_equal(view.R, rotation_before)
        np.testing.assert_array_equal(view.T, translation_before)


def test_target_translation_uses_edited_camera_axes():
    views = _views()
    base_rotation = _rotation_xyz(0, 35, 0)
    base_center = views[1].camera_center.copy()
    views[1] = replace(
        views[1],
        R=base_rotation,
        T=-base_center @ base_rotation.T,
    )
    target = _edit_target_camera(
        views, "mid", 1.0, 0, 0, 0, 0, 0, 2.0
    )
    target_center, target_scene_rotation = target[:2]
    delta_world = target_center - base_center
    delta_in_target_camera = delta_world @ target_scene_rotation
    np.testing.assert_allclose(delta_in_target_camera, [0, 0, 2.0], atol=1e-12)


def test_frustum_and_prestitch_share_target_pose_direction():
    views = _views()
    samples = unproject_views(views, 1)
    target_at_zero = _edit_target_camera(views, "mid", 1.0, 0, 0, 0, 0, 0, 0)
    target_forward = _edit_target_camera(views, "mid", 1.0, 0, 0, 0, 0, 0, 0.5)
    center0, scene_rotation0, rotation0, _, focal0, _ = target_camera_components(
        target_at_zero
    )
    center1, scene_rotation1, rotation1, _, focal1, _ = target_camera_components(
        target_forward
    )
    np.testing.assert_allclose(scene_rotation0, scene_rotation1)
    np.testing.assert_allclose(rotation0, rotation1)

    # The visible frustum points along the exact same local +Z axis used by
    # the translation controls and by project_views.
    K = np.array([[focal1, 0, 16], [0, focal1, 12], [0, 0, 1]], dtype=np.float64)
    segments = _camera_frustum_segments(center1, rotation1, K, 32, 24, 1.0)
    visible_forward = segments[:4, 1].mean(axis=0) - center1
    optical_forward = np.array([0.0, 0.0, 1.0]) @ rotation1
    assert np.dot(visible_forward, optical_forward) > 0
    np.testing.assert_allclose(center1 - center0, 0.5 * optical_forward, atol=1e-12)

    # Moving +Z forward reduces target-space depth and enlarges the raw
    # rectilinear projection before any canvas fitting is applied.
    world_point = (samples[0][0][0] - views[0].T) @ views[0].R
    depth0 = ((world_point - center0) @ scene_rotation0)[2]
    depth1 = ((world_point - center1) @ scene_rotation1)[2]
    np.testing.assert_allclose(depth1, depth0 - 0.5, atol=1e-12)
    projected0, meta0 = project_views(
        views, "rectilinear", "mid", unprojected=samples, target_camera=target_at_zero
    )
    projected1, meta1 = project_views(
        views, "rectilinear", "mid", unprojected=samples, target_camera=target_forward
    )
    extent0 = np.ptp(np.concatenate([item.uv for item in projected0]), axis=0)
    extent1 = np.ptp(np.concatenate([item.uv for item in projected1]), axis=0)
    assert np.all(extent1 > extent0)
    np.testing.assert_allclose(
        meta1["target_depth_median"], meta0["target_depth_median"] - 0.5
    )


def test_scene_glb_uses_target_camera_without_mutating_sources():
    views = _views()
    target = _edit_target_camera(views, "mid", 1.0, 5, 0, 10, 0, 0, 0)
    scene_path = build_scene_glb(
        views, target, target_index=1, samples=unproject_views(views, 2), max_points=100
    )
    assert scene_path is not None
    assert scene_path.endswith(".glb")
    assert Path(scene_path).is_file()
    assert Path(scene_path).stat().st_size > 0
    np.testing.assert_allclose(views[1].camera_center, np.array([0.2, 0.0, 0.0]))


def test_scene_glb_uses_reference_camera_as_origin():
    import trimesh

    views = _views()
    reference = (
        views[1].camera_center,
        views[1].R.T,
        float((views[1].K[0, 0] + views[1].K[1, 1]) / 2),
    )
    target = _edit_target_camera(views, "mid", 1.0, 0, 0, 0, 0, 0, 0)
    scene_path = build_scene_glb(
        views,
        target,
        target_index=1,
        samples=unproject_views(views, 2),
        max_points=100,
        reference_pose=reference,
    )
    scene = trimesh.load(scene_path, force="scene")
    reference_vertices = np.asarray(scene.geometry["reference camera"].vertices)
    target_vertices = np.asarray(scene.geometry["edited target camera"].vertices)
    assert np.min(np.linalg.norm(reference_vertices, axis=1)) < 1e-7
    assert np.min(np.linalg.norm(target_vertices, axis=1)) < 1e-7
