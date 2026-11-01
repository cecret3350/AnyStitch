"""Color brush, reversible region depth edits, and renderer propagation."""
import numpy as np
import pytest

from tools.scenic_demo import prestitch_ui as ui
from tools.scenic_demo.da3_prestitch import ViewGeometry


def _view(depth, index=0):
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    k = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=float)
    return ViewGeometry(index, str(index), np.full((h, w, 3), 90, np.uint8), depth,
                        k, k.copy(), np.eye(3), np.array([-0.5 * index, 0., 0.]), (w, h))


def _payload(red, blue):
    layer = np.zeros((*red.shape, 4), np.uint8)
    layer[red] = [255, 0, 0, 150]
    layer[blue] = [0, 0, 255, 150]
    return {"layers": [layer]}


def test_brush_colors_and_top_layer_take_precedence():
    bottom = np.zeros((4, 5, 4), np.uint8)
    bottom[0, 0] = [255, 0, 0, 150]
    bottom[1, 1] = [0, 0, 255, 150]
    bottom[2, 2] = [0, 0, 255, 255]
    top = np.zeros_like(bottom)
    top[0, 0] = [0, 0, 255, 255]
    top[2, 2] = [255, 0, 0, 255]
    top[3, 3] = [0, 0, 255, 4]  # ignore near-transparent edge
    red, blue = ui._editor_masks({"layers": [bottom, top]}, (4, 5))
    assert red.sum() == 1 and red[2, 2]
    assert blue.sum() == 2 and blue[0, 0] and blue[1, 1]
    assert not np.any(red & blue)
    big_red, big_blue = ui._editor_masks({"layers": [bottom, top]}, (8, 10))
    assert big_red.sum() == 4 and big_blue.sum() == 8


def test_erased_layers_do_not_resurrect_composite_mask():
    payload = {"layers": [np.zeros((4, 4, 4), np.uint8)],
               "background": np.zeros((4, 4, 3), np.uint8),
               "composite": np.full((4, 4, 3), 255, np.uint8)}
    red, blue = ui._editor_masks(payload, (4, 4))
    assert not red.any() and not blue.any()


def test_composite_only_and_legacy_grayscale_masks():
    back = np.full((4, 4, 3), 100, np.uint8)
    comp = back.copy()
    comp[0, 0] = [180, 45, 45]
    comp[2, 2] = [45, 45, 180]
    red, blue = ui._editor_masks({"background": back, "composite": comp}, (8, 8))
    assert red.sum() == blue.sum() == 4
    red, blue = ui._editor_masks({"layers": [np.full((4, 4), 255, np.uint8)]}, (4, 4))
    assert red.all() and not blue.any()


def test_separate_components_valid_depth_and_immutable_geometry():
    depth = np.full((8, 10), 42., np.float32)
    depth[1:3, 1:3] = [[2, 6], [0, np.nan]]
    depth[5:7, 1:3] = [[10, 20], [30, np.inf]]
    depth[1:3, 7:9] = [[0, np.nan], [-1, np.inf]]
    mask = np.zeros_like(depth, dtype=bool)
    mask[1:3, 1:3] = mask[5:7, 1:3] = mask[1:3, 7:9] = True
    view = _view(depth)
    original = depth.copy()
    edited, stats = ui._apply_depth_average_masks([view], [mask])
    np.testing.assert_array_equal(edited[0].depth[1:3, 1:3], 4.)
    np.testing.assert_array_equal(edited[0].depth[5:7, 1:3], 20.)
    np.testing.assert_array_equal(edited[0].depth[1:3, 7:9], original[1:3, 7:9])
    np.testing.assert_array_equal(edited[0].depth[~mask], original[~mask])
    np.testing.assert_array_equal(view.depth, original)
    for field in ('image', 'K', 'R', 'T'):
        assert getattr(edited[0], field) is getattr(view, field)
    assert stats == dict(regions=2, skipped=1, pixels=8)


def test_diagonal_contact_is_one_component():
    view = _view([[2., 1.], [1., 6.]])
    edited, stats = ui._apply_depth_average_masks([view], [np.eye(2, dtype=bool)])
    assert stats['regions'] == 1
    np.testing.assert_array_equal(edited[0].depth, [[4., 1.], [1., 4.]])


@pytest.mark.parametrize('scale', [0.5, 1., 2., 3.])
def test_native_means_and_labels_survive_resize_without_boundary_blur(scale):
    depth = np.arange(64, dtype=np.float32).reshape(8, 8) + 1
    view = _view(depth)
    mask = np.zeros((8, 8), bool)
    mask[:, :3] = True
    mask[:, 4:] = True  # one native pixel gap can disappear after downsample
    resized = ui._resize_views([view], scale)
    edited, stats = ui._apply_depth_average_masks(resized, [mask], reference_views=[view])
    left = ui._resize_input_mask(mask & (np.indices(mask.shape)[1] < 3),
                                 edited[0].depth.shape[1], edited[0].depth.shape[0])
    right = ui._resize_input_mask(mask & (np.indices(mask.shape)[1] >= 4),
                                  edited[0].depth.shape[1], edited[0].depth.shape[0])
    np.testing.assert_array_equal(edited[0].depth[left], depth[:, :3].mean())
    np.testing.assert_array_equal(edited[0].depth[right], depth[:, 4:].mean())
    np.testing.assert_array_equal(edited[0].depth[~(left | right)], resized[0].depth[~(left | right)])
    assert stats['regions'] == 2


def test_apply_reload_and_clear_preserve_other_frames_and_original_depth():
    views = [_view(np.arange(64).reshape(8, 8) + 1, i) for i in range(2)]
    state = dict(base_views=views, views=views, da3_views=views,
                 last_render={'stale': True}, all_sample_cache=['stale'],
                 testtime_optimization_cache={'stale': True}, testtime_acceptance_cache={'stale': True},
                 target_rotation_optimization_cache={'stale': True})
    red = np.zeros((8, 8), bool)
    blue = red.copy()
    red[0, 0] = True
    blue[3:6, 3:6] = True
    ui.update_input_mask(state, 0, _payload(red, blue))
    assert state['depth_edit_version'] == 1
    assert state['last_render'] is state['all_sample_cache'] is None
    assert not state['testtime_optimization_cache'] and not state['testtime_acceptance_cache']
    assert not state['target_rotation_optimization_cache']
    payload, _ = ui.select_mask_frame(state, 0)
    actual_red, actual_blue = ui._editor_masks(payload, (8, 8))
    np.testing.assert_array_equal(actual_red, red)
    np.testing.assert_array_equal(actual_blue, blue)
    ui.update_input_mask(state, 0, payload)
    assert state['depth_edit_version'] == 1  # identical application is idempotent
    ui.update_input_mask(state, 1, _payload(red, blue))
    ui.clear_input_mask(state, 0)
    assert not state['input_masks'][0].any() and not state['depth_average_masks'][0].any()
    np.testing.assert_array_equal(state['depth_average_masks'][1], blue)
    edited, _ = ui._apply_depth_average_masks(views, state['depth_average_masks'])
    np.testing.assert_array_equal(edited[0].depth, views[0].depth)
    np.testing.assert_array_equal(views[0].depth, np.arange(64).reshape(8, 8) + 1)


def test_renderer_recomputes_pointcloud_and_both_coord_maps_after_edit(monkeypatch):
    depth = np.tile(np.linspace(2., 8., 64, dtype=np.float32), (64, 1))
    views = [_view(depth, 0), _view(np.full_like(depth, 4.), 1)]
    state = dict(base_views=views, views=views, da3_views=views)
    scene_calls = []

    def scene(scene_views, *_args, **kwargs):
        scene_calls.append((scene_views, kwargs['samples']))
        return None

    monkeypatch.setattr(ui, 'build_scene_glb', scene)
    kwargs = dict(mode='rectilinear', reference='mid', render_stride=1, input_resize_scale=1.,
                  testtime_optimization=False, force_apply_optimized_camera=False,
                  auto_canvas=False, filter_canvas_outliers=False, visibility_aware=False,
                  auto_canvas_downscale=False, auto_canvas_max_mp=8., canvas_width=160, canvas_height=96,
                  focal_scale=1., cx_offset=0., cy_offset=0., rx=0., ry=0., rz=0., tx=0., ty=0., tz=0.)
    output, _, status = ui.render_case(state, **kwargs)
    assert output is not None, status
    original_maps = state['last_render']['metrics']
    original_samples = state['all_sample_cache']
    blue = np.zeros_like(depth, dtype=bool)
    blue[:, :48] = True
    red = np.zeros_like(blue)
    red[48:, 48:] = True
    ui.update_input_mask(state, 0, _payload(red, blue))
    assert state['last_render'] is None
    output, _, status = ui.render_case(state, **kwargs)
    assert output is not None, status
    assert state['all_sample_cache'] is not original_samples
    np.testing.assert_array_equal(state['views'][0].depth[blue], depth[blue].mean())
    assert 'depth averaged=1' in status
    # 3D viewer receives exactly the edited views and the renderer's samples.
    assert scene_calls[-1][0] is state['views']
    assert scene_calls[-1][1] is state['sample_cache']
    local, _, _, _, xy = state['sample_cache'][0]
    in_blue = blue[xy[:, 1].astype(int), xy[:, 0].astype(int)]
    np.testing.assert_allclose(local[in_blue, 2], depth[blue].mean())
    for key in ('_coord_maps', '_coord_maps_depth'):
        cmap = state['last_render']['metrics'][key][0]
        assert not np.allclose(cmap, original_maps[key][0])
        assert np.all(cmap[-1, -1] == 1000)  # old exclusion contract
        assert np.all(cmap[:, :3] != 1000)  # blue retains conditioning
    ui.clear_input_mask(state, 0)
    output, _, status = ui.render_case(state, **kwargs)
    assert output is not None, status
    for key in ('_coord_maps', '_coord_maps_depth'):
        np.testing.assert_allclose(state['last_render']['metrics'][key][0], original_maps[key][0])


def test_tto_cache_uses_current_depth_revision(monkeypatch):
    calls = []

    def refine(views, _index):
        calls.append(views[0].depth.copy())
        return list(views), dict(optimizer_success=False, num_matches=0,
                                initial_heldout_median_px=1., optimized_heldout_median_px=1.,
                                rotation_degrees=0., focal_scale=1.)

    monkeypatch.setattr(ui, '_refine_multiview_camera', refine)
    view = _view([[2., 6.], [2., 6.]])
    state = dict(da3_views=[view], depth_edit_version=0)
    ui._testtime_optimize_views(state, 'first')
    ui._testtime_optimize_views(state, 'first')
    assert len(calls) == 1
    state['da3_views'], _ = ui._apply_depth_average_masks([view], [np.ones((2, 2), bool)])
    state['depth_edit_version'] = 1
    ui._testtime_optimize_views(state, 'first')
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[-1], 4.)
