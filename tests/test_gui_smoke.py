"""pytest-qt smoke tests for the M4 GUI widgets.

These tests exercise widget construction and basic interactions without
requiring a live napari viewer or file I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.project.schema import (
    AtlasRef,
    ChannelLevels,
    PlaneParams,
    Project,
    Section,
    Slide,
)

# ---------------------------------------------------------------------------
# Schema: new flip/levels fields
# ---------------------------------------------------------------------------

def test_channel_levels_defaults() -> None:
    levels = ChannelLevels()
    assert levels.low == [0.0, 0.0, 0.0]
    assert levels.high == [1.0, 1.0, 1.0]


def test_slide_flip_fields() -> None:
    slide = Slide(image_path="x.png")
    assert slide.flip_h is False
    assert slide.flip_v is False
    assert slide.levels is None


def test_section_flip_fields() -> None:
    sec = Section(index=0, slide_idx=0, bbox_px=(0, 0, 100, 80))
    assert sec.flip_h is False
    assert sec.flip_v is False
    assert sec.levels is None


def test_project_roundtrip_with_flip_levels(tmp_path) -> None:
    from histo_to_ccf.project.io import load_project, save_project

    sec = Section(
        index=0, slide_idx=0, bbox_px=(0, 0, 100, 80),
        flip_h=True, levels=ChannelLevels(low=[0.1, 0.0, 0.0], high=[0.9, 1.0, 1.0]),
    )
    slide = Slide(image_path="s.png", sections=[sec], flip_v=True)
    project = Project(atlas=AtlasRef(), slides=[slide], probes=[])
    path = tmp_path / "test.json"
    save_project(project, path)
    reloaded = load_project(path)
    assert reloaded.slides[0].flip_v is True
    assert reloaded.slides[0].sections[0].flip_h is True
    assert reloaded.slides[0].sections[0].levels.high[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_add_slide() -> None:
    state = WorkflowState()
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    idx = state.add_slide("path/to/slide.png", img)
    assert idx == 0
    assert len(state.project.slides) == 1
    assert state.slide_images[0] is img
    idx2 = state.add_slide("path/to/slide2.png", img)
    assert idx2 == 1


def test_workflow_state_atlas_property() -> None:
    state = WorkflowState()
    assert state.atlas is None
    state.atlas = "mock_atlas"
    assert state.atlas == "mock_atlas"


# ---------------------------------------------------------------------------
# Widget smoke tests (require Qt via pytest-qt)
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_slide_loader_widget_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.slide_loader import SlideLoaderWidget

    state = WorkflowState()
    widget = SlideLoaderWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_image_tools_widget_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    state = WorkflowState()
    widget = ImageToolsWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_probe_picker_adds_probe(qtbot) -> None:
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget

    state = WorkflowState()
    widget = ProbePickerWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    widget._add_probe()
    assert len(state.project.probes) == 1
    assert state.project.probes[0].label == "probe1"


@pytest.mark.qt
def test_ordering_panel_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget

    state = WorkflowState()
    widget = OrderingPanelWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_save_panel_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget

    state = WorkflowState()
    widget = SavePanelWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_image_tools_section_scope_flip(qtbot) -> None:
    """Selecting a section in the dropdown makes section-scoped flips apply to it."""
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    state = WorkflowState()
    state.add_slide("fake.png", np.zeros((40, 40), dtype=np.uint8))
    state.active_slide_idx = 0
    slide = state.project.slides[0]
    slide.sections.append(Section(index=2, slide_idx=0, bbox_px=(0, 0, 20, 20)))

    widget = ImageToolsWidget(state)
    qtbot.addWidget(widget)
    # Choose section scope → dropdown populates and the active section is set.
    widget._scope_section.setChecked(True)
    assert state.active_section_idx == 2
    # A section-scoped flip now toggles that section's flag (previously a no-op).
    widget._flip_section("h")
    sec = next(s for s in slide.sections if s.index == 2)
    assert sec.flip_h is True


@pytest.mark.qt
def test_image_tools_flip_h_updates_state(qtbot) -> None:
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    state = WorkflowState()
    img = np.arange(6, dtype=np.uint8).reshape(2, 3)
    state.add_slide("fake.png", img.copy())
    state.active_slide_idx = 0

    widget = ImageToolsWidget(state)
    qtbot.addWidget(widget)
    # Flip H: image should be flipped and schema flag set
    widget._flip_slide("h")
    assert state.project.slides[0].flip_h is True
    np.testing.assert_array_equal(state.slide_images[0], np.fliplr(img))


@pytest.mark.qt
def test_save_panel_saves_json(qtbot, tmp_path) -> None:
    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget
    from histo_to_ccf.project.io import load_project

    state = WorkflowState()
    out_path = tmp_path / "out.json"
    widget = SavePanelWidget(state)
    qtbot.addWidget(widget)
    widget._path_edit.setText(str(out_path))
    widget._save()
    assert out_path.exists()
    p = load_project(out_path)
    assert p.version == 1


# ---------------------------------------------------------------------------
# Viewer-dependent widgets (atlas browser, click overlay)
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_atlas_browser_ap_is_bregma_relative(qtbot) -> None:
    import napari
    from histo_to_ccf.config import AppSettings
    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM

    viewer = napari.Viewer(show=False)
    try:
        widget = AtlasBrowserWidget(WorkflowState(), viewer, settings=AppSettings())
        qtbot.addWidget(widget)
        # Default shows bregma (0) and converts to the absolute origin AP.
        assert widget._ap_spin.value() == 0.0
        assert widget._bregma_to_absolute(0.0) == pytest.approx(BREGMA_AP_FROM_ORIGIN_UM)
        assert widget._absolute_to_bregma(BREGMA_AP_FROM_ORIGIN_UM) == pytest.approx(0.0)
        # The removed midline / dorsal / px controls should no longer exist.
        assert not hasattr(widget, "_midline_spin")
        assert not hasattr(widget, "_dorsal_spin")
    finally:
        viewer.close()


@pytest.mark.qt
def test_atlas_browser_assign_stores_absolute_ap(qtbot) -> None:
    import napari
    from histo_to_ccf.config import AppSettings
    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        img = np.zeros((40, 40), dtype=np.uint8)
        state.add_slide("s.png", img)
        state.active_slide_idx = 0
        state.project.slides[0].sections.append(
            Section(index=0, slide_idx=0, bbox_px=(0, 0, 20, 20))
        )
        widget = AtlasBrowserWidget(state, viewer, settings=AppSettings())
        qtbot.addWidget(widget)
        widget._ap_spin.setValue(-2000.0)  # 2 mm posterior to bregma
        widget._sec_spin.setValue(0)
        widget._assign_ap()
        plane = state.project.slides[0].sections[0].plane
        assert plane is not None
        assert plane.ap_um == pytest.approx(BREGMA_AP_FROM_ORIGIN_UM + 2000.0)
    finally:
        viewer.close()


class _FakeAtlas:
    """Minimal stand-in for a BrainGlobeAtlas (reference/annotation/resolution)."""

    def __init__(self) -> None:
        self.resolution = (25.0, 25.0, 25.0)
        # AP span must cover bregma-relative test positions (~ -2000 µm => abs 7400).
        shape = (320, 16, 18)  # (AP, DV, ML)
        n = int(np.prod(shape))
        self.reference = np.arange(n, dtype=np.uint16).reshape(shape)
        self.annotation = (np.arange(n) % 5).astype(np.int32).reshape(shape)
        self.atlas_name = "fake_25um"


@pytest.mark.qt
def test_atlas_matcher_navigate_and_assign(qtbot) -> None:
    from histo_to_ccf.gui.widgets.atlas_matcher import AtlasMatcherDialog
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM

    state = WorkflowState()
    state.add_slide("s.png", np.zeros((40, 60), dtype=np.uint8))
    state.active_slide_idx = 0
    slide = state.project.slides[0]
    slide.sections.append(Section(index=0, slide_idx=0, bbox_px=(0, 0, 20, 20), ap_order=0))
    slide.sections.append(Section(index=1, slide_idx=0, bbox_px=(20, 0, 40, 20), ap_order=1))
    state.atlas = _FakeAtlas()

    dlg = AtlasMatcherDialog(state)
    qtbot.addWidget(dlg)

    # Navigate to the second section (by ap_order).
    dlg._step_section(1)
    assert dlg._pos == 1
    assert state.active_section_idx == 1

    # Assign the current AP: stored as absolute, bregma-converted.
    dlg._ap_spin.setValue(-2000.0)
    dlg._assign_current()
    sec1 = next(s for s in slide.sections if s.index == 1)
    assert sec1.plane is not None
    assert sec1.plane.ap_um == pytest.approx(BREGMA_AP_FROM_ORIGIN_UM + 2000.0)

    # Linked assign-all fills every section from the anchor + spacing.
    dlg._link_check.setChecked(True)
    dlg._spacing_spin.setValue(100.0)
    dlg._assign_all()
    assert all(s.plane is not None for s in slide.sections)

    # Overlay mode must render without raising.
    dlg._overlay_radio.setChecked(True)
    assert dlg._stack.currentIndex() == 1


@pytest.mark.qt
def test_gl_report_never_raises(qtbot) -> None:
    from histo_to_ccf.gui.gl_diagnostics import format_gl_report, gl_report

    rep = gl_report()  # must not raise, even with no usable GL context
    assert {"ok", "vendor", "renderer", "version", "error"} <= set(rep)
    text = format_gl_report(rep)
    assert "OpenGL diagnostic" in text
    assert "Roll Back Driver" in text  # remediation guidance is included


@pytest.mark.qt
def test_build_panel_constructs_full_app(qtbot) -> None:
    """Build the entire dock panel exactly as launch() does (all tabs wired)."""
    import napari
    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        assert panel is not None and viz_panel is not None
        # No empty Tips/Entries marker layers at launch - adding those triggered
        # vispy "Unsupported framebuffer format" shader errors on some GPUs.
        assert "Tips" not in viewer.layers
        assert "Entries" not in viewer.layers
        # Only "Project" and "Registration" menus are visible; napari's defaults
        # (File / View / Plugins / Window / Help) are hidden.
        visible = [
            (a.menu().title() if a.menu() else a.text()).replace("&", "")
            for a in viewer.window._qt_window.menuBar().actions()
            if a.isVisible()
        ]
        assert set(visible) == {"Project", "Registration"}, visible
        # Project menu wires Ctrl+S (save) as an application-wide shortcut.
        proj_menu = next(
            a.menu() for a in viewer.window._qt_window.menuBar().actions()
            if a.menu() and a.menu().title().replace("&", "") == "Project"
        )
        shortcuts = {a.text(): a.shortcut().toString() for a in proj_menu.actions()}
        assert shortcuts.get("Save Project") == "Ctrl+S"
        assert shortcuts.get("Load Project…") == "Ctrl+O"
        # 3D/export buttons live on the permanent viz panel, not the Register tab.
        assert hasattr(viz_panel, "_view_napari3d") and hasattr(viz_panel, "_export_plotly")
    finally:
        viewer.close()


@pytest.mark.qt
def test_reload_restores_slide_flip(qtbot, tmp_path) -> None:
    """Loading a project re-applies the persisted slide flip to the raw image."""
    import napari
    from histo_to_ccf.gui.app import _reload_project_display
    from histo_to_ccf.io.image import load_image

    # A non-symmetric image so a flip is detectable.
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    img_path = tmp_path / "slide.png"
    import imageio.v3 as iio

    iio.imwrite(img_path, img)
    expected = np.fliplr(load_image(img_path))

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        state.project = Project(
            slides=[Slide(image_path=str(img_path), flip_h=True)]
        )
        _reload_project_display(viewer, state)
        np.testing.assert_array_equal(state.slide_images[0], expected)
    finally:
        viewer.close()


@pytest.mark.qt
def test_add_probe_arms_tip_mode(qtbot) -> None:
    import napari
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        overlay = ClickOverlayWidget(state, viewer)
        picker = ProbePickerWidget(state)
        qtbot.addWidget(overlay)
        qtbot.addWidget(picker)
        picker.on_probe_added = overlay.arm_tip
        picker._add_probe()
        # Tip + Marker selected and the Tips layer armed, no extra clicks needed.
        assert overlay._mode_tip.isChecked()
        assert overlay._entry_marker.isChecked()
        assert viewer.layers.selection.active is overlay._tip_layer
        assert overlay._tip_layer.mode == "add"
    finally:
        viewer.close()


@pytest.mark.qt
def test_edit_boxes_resize_and_delete(qtbot) -> None:
    import napari
    from histo_to_ccf.gui.widgets.slide_loader import SlideLoaderWidget

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        state.add_slide("s.png", np.zeros((400, 400), dtype=np.uint8))
        state.active_slide_idx = 0
        slide = state.project.slides[0]
        for i, box in enumerate([(0, 0, 80, 80), (100, 0, 180, 80), (200, 0, 280, 80)]):
            slide.sections.append(Section(index=i, slide_idx=0, bbox_px=box, ap_order=i))

        widget = SlideLoaderWidget(state, viewer=viewer)
        qtbot.addWidget(widget)
        widget._edit_boxes()
        layer = widget._box_layer
        assert layer is not None and len(layer.data) == 3

        # Resize section 0's rectangle (taller box) and sync.
        new = list(layer.data)
        new[0] = np.array([[0, 0], [0, 120], [200, 120], [200, 0]], dtype=float)
        layer.data = new
        widget._sync_boxes_from_shapes()
        sec0 = next(s for s in slide.sections if s.index == 0)
        assert sec0.bbox_px == (0, 0, 120, 200)

        # Delete the last rectangle and sync → that section is removed.
        layer.data = list(layer.data)[:2]
        layer.features = {"idx": [0, 1]}
        widget._sync_boxes_from_shapes()
        assert {s.index for s in slide.sections} == {0, 1}
    finally:
        viewer.close()


@pytest.mark.qt
def test_ordering_resort_and_interpolate(qtbot) -> None:
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget
    from histo_to_ccf.project.schema import PlaneParams

    state = WorkflowState()
    state.add_slide("s.png", np.zeros((10, 10), dtype=np.uint8))
    state.active_slide_idx = 0
    slide = state.project.slides[0]
    # 2x2 grid; ap_order intentionally scrambled.
    boxes = [(0, 0, 80, 80), (100, 0, 180, 80), (0, 100, 80, 180), (100, 100, 180, 180)]
    for i, b in enumerate(boxes):
        slide.sections.append(Section(index=i, slide_idx=0, bbox_px=b, ap_order=99))

    widget = OrderingPanelWidget(state)
    qtbot.addWidget(widget)
    widget._col_first.setChecked(True)
    widget._resort_sections()
    top_left = next(s for s in slide.sections if s.bbox_px == (0, 0, 80, 80))
    bottom_left = next(s for s in slide.sections if s.bbox_px == (0, 100, 80, 180))
    assert top_left.ap_order == 0  # column-first: down column 0 first
    assert bottom_left.ap_order == 1

    # Interpolate: assign ends, fill the middle linearly.
    ordered = sorted(slide.sections, key=lambda s: s.ap_order)
    ordered[0].plane = PlaneParams(ap_um=1000.0)
    ordered[3].plane = PlaneParams(ap_um=1300.0)
    widget._interpolate_ap()
    aps = [s.plane.ap_um for s in sorted(slide.sections, key=lambda s: s.ap_order)]
    assert aps == [1000.0, 1100.0, 1200.0, 1300.0]


@pytest.mark.qt
def _two_shank_state() -> WorkflowState:
    from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Shank

    state = WorkflowState()
    state.add_slide("s.png", np.zeros((100, 100), dtype=np.uint8))
    state.active_slide_idx = 0
    state.project.probes.append(
        ProbeSpec(label="P0", type=ProbeType(name="NP", n_shanks=2),
                  shanks=[Shank(index=0), Shank(index=1)])
    )
    return state


def _color_of(layer, p_idx, s_idx):
    """RGBA of the point whose features mark it as (probe p_idx, shank s_idx)."""
    feats = layer.features
    p = np.asarray(feats["p"], dtype=int)
    s = np.asarray(feats["s"], dtype=int)
    fc = np.asarray(layer.face_color)
    for i in range(len(p)):
        if p[i] == p_idx and s[i] == s_idx:
            return fc[i]
    return None


@pytest.mark.qt
def test_markers_color_per_shank_and_one_per_shank(qtbot) -> None:
    """Tip+entry of a shank share a colour; another shank cycles; one tip/shank."""
    import napari
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget

    viewer = napari.Viewer(show=False)
    try:
        state = _two_shank_state()
        w = ClickOverlayWidget(state, viewer)
        qtbot.addWidget(w)
        w._ensure_points_layers()

        # Drop a tip for shank 0, then shank 1 (simulating clicks via layer.add).
        w._probe_combo.setCurrentIndex(0)
        w._shank_combo.setCurrentIndex(0)
        w._tip_layer.add([[10.0, 20.0]])
        w._shank_combo.setCurrentIndex(1)
        w._tip_layer.add([[30.0, 40.0]])

        shanks = state.project.probes[0].shanks
        assert shanks[0].tip_px is not None and shanks[1].tip_px is not None
        assert (shanks[0].tip_px.x_px, shanks[0].tip_px.y_px) == (20.0, 10.0)
        # Two shanks -> two different colours.
        c0 = _color_of(w._tip_layer, 0, 0)
        c1 = _color_of(w._tip_layer, 0, 1)
        assert c0 is not None and c1 is not None and not np.allclose(c0, c1)

        # Entry for shank 0 shares shank 0's colour.
        w._shank_combo.setCurrentIndex(0)
        w._entry_layer.add([[12.0, 22.0]])
        assert np.allclose(_color_of(w._entry_layer, 0, 0), c0)

        # A second tip for shank 0 REPLACES it (still one tip point per shank).
        w._tip_layer.add([[50.0, 60.0]])
        assert len(w._tip_layer.data) == 2  # shank0 (moved) + shank1, not 3
        assert (shanks[0].tip_px.x_px, shanks[0].tip_px.y_px) == (60.0, 50.0)
    finally:
        viewer.close()


@pytest.mark.qt
def test_markers_clear_selected_removes_one(qtbot) -> None:
    import napari
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget

    viewer = napari.Viewer(show=False)
    try:
        state = _two_shank_state()
        w = ClickOverlayWidget(state, viewer)
        qtbot.addWidget(w)
        w._ensure_points_layers()
        w._shank_combo.setCurrentIndex(0)
        w._tip_layer.add([[10.0, 20.0]])
        w._shank_combo.setCurrentIndex(1)
        w._tip_layer.add([[30.0, 40.0]])

        # Select the shank-0 point and clear only it.
        feats = w._tip_layer.features
        s = np.asarray(feats["s"], dtype=int)
        sel = {i for i in range(len(s)) if s[i] == 0}
        w._tip_layer.selected_data = sel
        w._clear_selected()

        shanks = state.project.probes[0].shanks
        assert shanks[0].tip_px is None  # cleared
        assert shanks[1].tip_px is not None  # kept
        assert len(w._tip_layer.data) == 1
    finally:
        viewer.close()


@pytest.mark.qt
def test_wheel_pan_moves_camera(qtbot) -> None:
    """Ctrl+wheel pans horizontally, Shift+wheel vertically; neither zooms."""
    import napari
    from histo_to_ccf.gui.app import _install_wheel_pan

    viewer = napari.Viewer(show=False)
    try:
        _install_wheel_pan(viewer)
        cb = viewer.mouse_wheel_callbacks[-1]
        viewer.camera.center = (0.0, 0.0, 0.0)
        viewer.camera.zoom = 1.0
        zoom_before = viewer.camera.zoom

        class _Evt:
            def __init__(self, mods):
                self.modifiers = mods
                self.delta = (0.0, 1.0)
                self.native = None

        cb(viewer, _Evt(("Control",)))
        assert viewer.camera.center[2] != 0.0  # x moved
        assert viewer.camera.center[1] == 0.0
        cb(viewer, _Evt(("Shift",)))
        assert viewer.camera.center[1] != 0.0  # y moved
        # Panning must not change zoom.
        assert viewer.camera.zoom == zoom_before
        # No modifier -> no pan.
        before = tuple(viewer.camera.center)
        cb(viewer, _Evt(()))
        assert tuple(viewer.camera.center) == before
    finally:
        viewer.close()


@pytest.mark.qt
def test_update_coordinates_remaps_moved_points(qtbot) -> None:
    """'Update coordinates' re-maps tip/entry pixels (incl. moved ones) into CCF."""
    import napari
    from histo_to_ccf.project.schema import (
        Point2D, ProbeSpec, ProbeType, RegistrationResult, Section, Shank,
    )
    from histo_to_ccf.gui.widgets.viz_export_panel import VizExportPanelWidget

    class _FakeAtlas:
        resolution = (25.0, 25.0, 25.0)

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        state.add_slide("s.png", np.zeros((20, 20), dtype=np.uint8))
        state.active_slide_idx = 0
        state.atlas = _FakeAtlas()
        # A registered section with NO B-spline (so no .h5 needed): plane only.
        state.project.slides[0].sections.append(Section(
            index=5, slide_idx=0, bbox_px=(0, 0, 20, 20), ap_order=0,
            registration=RegistrationResult(
                anchoring=[100, 50, 80, 40, 0, 0, 0, 0, 40],
                output_size_px=(20, 20), bspline_transform_path=None, residual=0.1),
        ))
        shank = Shank(index=0, tip_px=Point2D(x_px=5.0, y_px=5.0), tip_section_idx=5)
        state.project.probes.append(
            ProbeSpec(label="P", type=ProbeType(name="NP", n_shanks=1), shanks=[shank]))

        panel = VizExportPanelWidget(state, viewer)
        qtbot.addWidget(panel)

        assert shank.tip_ccf_um is None
        panel._do_update_coordinates()
        first = shank.tip_ccf_um
        assert first is not None

        # Move the tip and update again -> CCF changes.
        shank.tip_px = Point2D(x_px=15.0, y_px=12.0)
        panel._do_update_coordinates()
        assert shank.tip_ccf_um is not None and shank.tip_ccf_um != first
    finally:
        viewer.close()


@pytest.mark.qt
def test_reset_morph_drops_bspline_keeps_plane(qtbot, tmp_path) -> None:
    """Reset-morph nulls the B-spline + manual corrections but keeps the anchoring."""
    import napari
    from histo_to_ccf.project.schema import ManualLandmarks, RegistrationResult, Section
    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        state.add_slide(str(tmp_path / "s.png"), np.zeros((10, 10), dtype=np.uint8))
        state.active_slide_idx = 0
        state.project_path = tmp_path / "p.histo2ccf.json"
        anchoring = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sec = Section(
            index=5, slide_idx=0, bbox_px=(0, 0, 5, 5), ap_order=0,
            registration=RegistrationResult(
                anchoring=list(anchoring), output_size_px=(5, 5),
                bspline_transform_path="transforms/section_005.h5", residual=0.1),
            manual_landmarks=ManualLandmarks(source=[[1.0, 1.0]], target=[[2.0, 2.0]]),
        )
        state.project.slides[0].sections.append(sec)
        panel = RegisterPanelWidget(state, viewer)
        qtbot.addWidget(panel)
        panel._populate_adjust_combo()
        panel._adjust_combo.setCurrentIndex(panel._adjust_combo.findData(5))

        panel._reset_morph()

        # Morph + manual corrections dropped; the plane (anchoring) is kept.
        assert sec.registration.bspline_transform_path is None
        assert sec.manual_landmarks is None
        assert sec.manual_affine is None
        assert sec.registration.anchoring == anchoring
        assert state.project_path.exists()  # auto-saved
    finally:
        viewer.close()


@pytest.mark.qt
def test_residuals_table_bregma_and_ap_order(qtbot) -> None:
    """Residuals table: AP from bregma (matches Atlas tab) and sorted by AP order."""
    import napari
    from histo_to_ccf.project.schema import RegistrationResult, Section
    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        state.add_slide("s.png", np.zeros((10, 10), dtype=np.uint8))
        state.active_slide_idx = 0
        # Stored out of AP order: section 3 (ap_order 1, posterior) before
        # section 7 (ap_order 0, anterior). Anchorings: AP voxel = ox.
        def reg(ox):
            return RegistrationResult(anchoring=[ox, 0, 0, 0, 0, 0, 0, 0, 0],
                                      output_size_px=(10, 10), residual=0.1)
        state.project.slides[0].sections.extend([
            Section(index=3, slide_idx=0, bbox_px=(0, 0, 5, 5), ap_order=1,
                    registration=reg(420.0)),   # bregma 5400 - 420*25 = -5100
            Section(index=7, slide_idx=0, bbox_px=(5, 5, 9, 9), ap_order=0,
                    registration=reg(400.0)),   # bregma 5400 - 400*25 = -4600
        ])
        panel = RegisterPanelWidget(state, viewer)
        qtbot.addWidget(panel)
        panel._refresh_residuals()

        t = panel._residuals_table
        assert t.rowCount() == 2
        # Sorted by ap_order: section 7 (anterior) first, then 3.
        assert t.item(0, 0).text() == "7"
        assert t.item(1, 0).text() == "3"
        # AP shown from bregma (negative = posterior), anterior section less negative.
        assert t.item(0, 1).text() == "-4600"
        assert t.item(1, 1).text() == "-5100"
    finally:
        viewer.close()


@pytest.mark.qt
def test_click_overlay_modes_and_nearest_section(qtbot) -> None:
    import napari
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        img = np.zeros((100, 100), dtype=np.uint8)
        state.add_slide("s.png", img)
        state.active_slide_idx = 0
        state.project.slides[0].sections.append(
            Section(index=3, slide_idx=0, bbox_px=(10, 10, 30, 30))
        )
        widget = ClickOverlayWidget(state, viewer)
        qtbot.addWidget(widget)

        # Trajectory-line entry mode arms a Trajectory shapes layer.
        widget._mode_entry.setChecked(True)
        widget._entry_line.setChecked(True)
        widget._activate_pick_mode()
        assert "Trajectory" in viewer.layers

        # A point just OUTSIDE the tight bbox still resolves to that section.
        assert widget._find_section_for_point(32.0, 20.0) == 3
        assert widget._find_section_for_point(20.0, 20.0) == 3  # inside
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Ephys alignment tab
# ---------------------------------------------------------------------------

class _EphysFakeAtlas:
    """Fake atlas with structure_from_coords + structures for region lookup."""

    def __init__(self) -> None:
        self.resolution = (25.0, 25.0, 25.0)
        self.structures = {
            "A": {"rgb_triplet": [10, 20, 30]},
            "B": {"rgb_triplet": [40, 50, 60]},
        }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        if dv < 0 or dv > 6000:
            return "Outside atlas"
        return "A" if dv < 3000 else "B"


def _fake_lfp_result(n_ch: int = 16, n_freq: int = 20) -> dict:
    rng = np.arange(n_ch * n_freq, dtype=np.uint8).reshape(n_ch, n_freq)
    return {
        "freqs": np.linspace(0, 300, n_freq),
        "psd": rng.astype(float),
        "image": rng,
        "depths_um": np.linspace(0.0, 3000.0, n_ch),
        "x_um": np.zeros(n_ch),
        "channel_ids": list(range(n_ch)),
        "stream_name": "ProbeA-LFP",
        "derived_from_ap": False,
    }


def _registered_probe_state() -> WorkflowState:
    from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Shank

    state = WorkflowState()
    shank = Shank(
        index=0,
        tip_ccf_um=(1000.0, 2000.0, 5000.0),
        entry_ccf_um=(1000.0, 2000.0, 1000.0),
    )
    state.project.probes.append(
        ProbeSpec(label="probe1", type=ProbeType(name="NP", n_shanks=1), shanks=[shank])
    )
    state.atlas = _EphysFakeAtlas()
    return state


@pytest.mark.qt
def test_ephys_panel_lists_probes(qtbot) -> None:
    import napari
    from histo_to_ccf.gui.widgets.ephys_panel import EphysPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        state = _registered_probe_state()
        widget = EphysPanelWidget(state, viewer)
        qtbot.addWidget(widget)
        widget.refresh_probes()
        assert widget._probe_combo.count() == 1
        assert widget._shank_combo.count() == 1
    finally:
        viewer.close()


@pytest.mark.qt
def test_ephys_alignment_dialog_apply_writes_ccf(qtbot) -> None:
    from histo_to_ccf.gui.widgets.ephys_align_dialog import EphysAlignmentDialog

    state = _registered_probe_state()
    dlg = EphysAlignmentDialog(state, 0, 0, _fake_lfp_result())
    qtbot.addWidget(dlg)

    # Region strip rendered without raising and produced labels.
    assert "Regions" in dlg._regions_label.text()
    # Channel ids carried through (used for the depth/channel axis labels).
    assert len(dlg._channel_ids) == 16

    # Per-frequency normalization toggle re-renders without raising.
    dlg._per_freq_check.setChecked(True)
    assert dlg._img_feat.shape[0] == 16

    # Double-click placement adds an anchor at the clicked depth, then clear.
    dlg._add_anchor_at_scene_y(300.0)
    assert len(dlg.anchors()) == 1
    dlg.clear_anchors()
    assert dlg.anchors() == []

    # Add an anchor, then apply -> per-channel CCF stored on the shank.
    dlg.add_anchor(feature_depth=1000.0, track_depth=1200.0)
    assert len(dlg.anchors()) == 1
    dlg._apply()

    eph = state.project.probes[0].shanks[0].ephys
    assert eph is not None
    assert eph.stream_name == "ProbeA-LFP"
    assert len(eph.channel_ccf_um) == 16
    assert len(eph.channel_depths_um) == 16
    assert eph.anchors and eph.anchors[0][0] == pytest.approx(1000.0)


@pytest.mark.qt
def test_napari_probe_layer_uses_placed_ml_no_offset(qtbot) -> None:
    """The 3D probe line uses the placed tip/entry ML, with no shank offset added."""
    import napari
    from histo_to_ccf.project.schema import Project, ProbeSpec, ProbeType, Shank
    from histo_to_ccf.viz.napari3d import add_probe_layers

    viewer = napari.Viewer(show=False)
    try:
        shank = Shank(index=3, tip_ccf_um=(10700.0, 5800.0, 5000.0),
                      entry_ccf_um=(10700.0, 5800.0, 1000.0))
        project = Project(probes=[ProbeSpec(
            label="P", type=ProbeType(name="NP4", n_shanks=4, shank_pitch_um=250.0),
            shanks=[shank])])
        layers = add_probe_layers(viewer, project)
        assert layers
        line = np.asarray(layers[0].data[0])  # (2, 3) [tip, entry] in (AP, ML, DV)
        assert line[0][1] == 5800.0 and line[1][1] == 5800.0  # ML unshifted
    finally:
        viewer.close()


@pytest.mark.qt
def test_default_3d_camera_is_posterior_dorsal_up(qtbot) -> None:
    """Default 3D camera: from behind (anterior view dir), dorsal up, tilted down."""
    import napari
    from histo_to_ccf.viz.napari3d import _set_default_camera

    viewer = napari.Viewer(show=False)
    try:
        # Data extent in (AP, ML, DV) so reset_view has something to fit.
        viewer.add_points(np.array([[0, 0, 0], [13000, 11000, 8000]], float), ndim=3)
        viewer.dims.ndisplay = 3
        _set_default_camera(viewer)
        vd = np.asarray(viewer.camera.view_direction)
        assert vd[0] < 0          # looking toward anterior (-AP) = viewing from behind
        assert abs(vd[1]) < 0.05  # symmetric left-right (no ML)
        assert vd[2] > 0          # tilted slightly downward (+DV)
        up = np.asarray(viewer.camera.up_direction)
        assert up[2] < 0          # dorsal (-DV) is up
    finally:
        viewer.close()


@pytest.mark.qt
def test_show_3d_scene_adds_ephys_channel_layer(qtbot) -> None:
    import napari
    from histo_to_ccf.project.schema import EphysAlignment, ProbeSpec, ProbeType, Shank
    from histo_to_ccf.viz.napari3d import show_3d_scene

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        shank = Shank(
            index=0,
            tip_ccf_um=(4000.0, 2000.0, 5000.0),
            entry_ccf_um=(4000.0, 2000.0, 1000.0),
            ephys=EphysAlignment(
                channel_ccf_um=[(4000.0, 2000.0, 5000.0), (4000.0, 2000.0, 3000.0)]
            ),
        )
        state.project.probes.append(
            ProbeSpec(label="probeA", type=ProbeType(name="NP", n_shanks=1), shanks=[shank])
        )
        # No atlas -> still draws probe + ephys channel layers.
        show_3d_scene(viewer, state.project, None)
        assert any(str(lyr.name).startswith("Ephys channels") for lyr in viewer.layers)
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Reload repopulates tab fields
# ---------------------------------------------------------------------------

def test_project_section_spacing_round_trips(tmp_path) -> None:
    from histo_to_ccf.project.io import load_project, save_project

    proj = Project(section_spacing_um=123.0)
    path = tmp_path / "p.histo2ccf.json"
    save_project(proj, path)
    assert load_project(path).section_spacing_um == 123.0


def _populated_state() -> WorkflowState:
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM  # noqa: F401
    from histo_to_ccf.project.schema import (
        AtlasRef,
        Point2D,
        ProbeSpec,
        ProbeType,
        RegistrationResult,
        Shank,
    )

    state = WorkflowState()
    state.add_slide("s.png", np.zeros((200, 200), dtype=np.uint8))
    state.active_slide_idx = 0
    slide = state.project.slides[0]
    slide.sections.append(
        Section(
            index=0, slide_idx=0, bbox_px=(0, 0, 80, 80), ap_order=0,
            plane=PlaneParams(ap_um=4000.0),
            registration=RegistrationResult(anchoring=[0.0] * 9, output_size_px=(80, 80), residual=0.5),
        )
    )
    slide.sections.append(Section(index=1, slide_idx=0, bbox_px=(100, 0, 180, 80), ap_order=1))
    shank = Shank(
        index=0,
        tip_px=Point2D(x_px=40.0, y_px=40.0),
        entry_px=Point2D(x_px=40.0, y_px=10.0),
        tip_ccf_um=(4000.0, 2000.0, 5000.0),
        entry_ccf_um=(4000.0, 2000.0, 1000.0),
    )
    state.project.probes.append(
        ProbeSpec(label="probeA", type=ProbeType(name="NP", n_shanks=1), shanks=[shank])
    )
    state.project.atlas = AtlasRef(name="kim_mouse_25um")
    state.project.section_spacing_um = 120.0
    return state


@pytest.mark.qt
def test_reload_repopulates_widgets(qtbot) -> None:
    import napari
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM
    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget
    from histo_to_ccf.gui.widgets.ephys_panel import EphysPanelWidget
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget
    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        state = _populated_state()
        probe_picker = ProbePickerWidget(state)
        click_overlay = ClickOverlayWidget(state, viewer)
        atlas_browser = AtlasBrowserWidget(state, viewer)
        ordering = OrderingPanelWidget(state)
        register_panel = RegisterPanelWidget(state, viewer)
        ephys_panel = EphysPanelWidget(state, viewer)
        for w in (probe_picker, click_overlay, atlas_browser, ordering, register_panel, ephys_panel):
            qtbot.addWidget(w)
            w.refresh_after_load()

        # Ordering: spacing restored from the project.
        assert ordering._spacing.value() == 120.0

        # Atlas: combo points at the project's atlas; AP reflects assigned section 0.
        assert atlas_browser._current_atlas_id() == "kim_mouse_25um"
        assert atlas_browser._ap_spin.value() == pytest.approx(BREGMA_AP_FROM_ORIGIN_UM - 4000.0)

        # Click overlay: tip + entry restored to the table and markers drawn.
        assert click_overlay._table.rowCount() == 2
        assert click_overlay._tip_layer is not None and len(click_overlay._tip_layer.data) == 1
        assert len(click_overlay._entry_layer.data) == 1

        # Probe picker status + ephys combos reflect the probe.
        assert "probeA" in probe_picker._status.text()
        assert ephys_panel._probe_combo.count() == 1

        # Register: residuals table has the one registered section.
        assert register_panel._residuals_table.rowCount() == 1
    finally:
        viewer.close()


@pytest.mark.qt
def test_auto_load_atlas_skips_when_already_loaded(qtbot) -> None:
    """auto_load_atlas is a no-op when the matching atlas is already loaded or unset."""
    import napari
    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.project.schema import AtlasRef

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        widget = AtlasBrowserWidget(state, viewer)
        qtbot.addWidget(widget)

        # No atlas name recorded -> nothing happens, status untouched.
        state.project.atlas = AtlasRef(name="")
        widget._atlas_status.setText("untouched")
        widget.auto_load_atlas()
        assert widget._atlas_status.text() == "untouched"

        # Atlas already loaded with the same name -> skip (no "Loading…").
        state.project.atlas = AtlasRef(name="kim_mouse_25um")
        state.atlas = _EphysFakeAtlas()
        state.atlas.atlas_name = "kim_mouse_25um"
        widget.auto_load_atlas()
        assert widget._atlas_status.text() == "untouched"
    finally:
        viewer.close()


@pytest.mark.qt
def test_manual_atlas_adjustment(qtbot, tmp_path) -> None:
    """Drag-on-canvas manual correction stores a section-local affine + re-maps."""
    import napari
    import numpy as np
    from napari.utils.transforms import Affine

    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        state = _populated_state()
        state.project_path = tmp_path / "p.histo2ccf.json"
        panel = RegisterPanelWidget(state, viewer)
        qtbot.addWidget(panel)

        # Simulate "Show atlas overlay": add the section-0 overlay layer.
        viewer.add_labels(
            np.zeros((80, 80), dtype=np.uint8), name="Atlas overlay 0", translate=(0, 0)
        )
        panel._populate_adjust_combo()
        assert panel._adjust_combo.count() == 1  # only section 0 is registered

        # Enter transform mode (what the user gets to drag).
        panel._adjust_btn.setChecked(True)
        layer = viewer.layers["Atlas overlay 0"]
        assert layer.mode == "transform"

        # Simulate a drag: a +4 row / +8 col translation of the overlay.
        layer.affine = Affine(
            affine_matrix=np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]])
        )

        # Apply.
        panel._adjust_btn.setChecked(False)
        sec = state.project.slides[0].sections[0]
        assert layer.mode == "pan_zoom"
        assert sec.manual_affine is not None
        # bbox origin is (0, 0), so section-local affine == the world affine.
        assert np.allclose(
            np.array(sec.manual_affine), [[1, 0, 4], [0, 1, 8], [0, 0, 1]], atol=1e-6
        )
        assert state.project_path.exists()  # auto-saved

        # Reset clears it and restores identity.
        panel._reset_adjustment()
        assert sec.manual_affine is None
        assert np.allclose(np.asarray(layer.affine.affine_matrix), np.eye(3), atol=1e-6)
    finally:
        viewer.close()


@pytest.mark.qt
def test_landmark_warp_apply_and_reset(qtbot, tmp_path) -> None:
    """Place/apply/reset landmark correction stores ManualLandmarks + cleans up."""
    import napari
    import numpy as np

    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        state = _populated_state()
        state.project_path = tmp_path / "p.histo2ccf.json"
        panel = RegisterPanelWidget(state, viewer)
        qtbot.addWidget(panel)

        # Stand in for "Show overlay" + "Place landmarks" (atlas-independent path):
        # build the landmarks layer the way _place_landmarks does (source in features).
        viewer.add_labels(np.zeros((80, 80), dtype=np.uint8), name="Atlas overlay 0",
                          translate=(0, 0))
        src = np.array([[10, 10], [70, 10], [40, 40], [10, 70], [70, 70]], float)  # (x,y)
        data = src[:, ::-1].copy()  # (row, col); target == source initially
        lm = viewer.add_points(data, name="Atlas landmarks 0", size=12,
                               features={"sy": src[:, 1], "sx": src[:, 0]})
        panel._landmark_idx = 0
        panel._lm_prev_data = np.asarray(lm.data, dtype=float).copy()
        lm.events.data.connect(panel._on_landmark_data)
        panel._populate_adjust_combo()
        sec = state.project.slides[0].sections[0]

        # (1) Plain drag = warp: move centre target +12 in x, source unchanged.
        d = np.asarray(lm.data, dtype=float); d[2, 1] += 12.0; lm.data = d
        assert np.asarray(lm.features["sx"])[2] == pytest.approx(40.0)  # anchor stayed

        # (2) Ctrl/move drag = relocate: anchor follows the point.
        panel._lm_move_btn.setChecked(True)
        d = np.asarray(lm.data, dtype=float); d[0, 0] += 5.0; lm.data = d  # move row(y) of pt0
        assert np.asarray(lm.features["sy"])[0] == pytest.approx(15.0)  # anchor moved with it
        panel._lm_move_btn.setChecked(False)

        # (3) Add a point: its anchor is set to where it was dropped.
        lm.add(np.array([[33.0, 44.0]]))  # (row, col)
        assert len(lm.data) == 6
        assert np.asarray(lm.features["sx"])[-1] == pytest.approx(44.0)
        assert np.asarray(lm.features["sy"])[-1] == pytest.approx(33.0)

        # (4) Delete a point: features stay aligned.
        lm.selected_data = {1}
        lm.remove_selected()
        assert len(lm.data) == 5 and len(lm.features["sx"]) == 5

        panel._apply_landmarks()
        assert sec.manual_landmarks is not None
        assert sec.manual_affine is None  # mutually exclusive
        assert len(sec.manual_landmarks.target) == 5
        # The warped centre point's target moved +12 in x relative to its anchor.
        tgt = np.array(sec.manual_landmarks.target)
        srcs = np.array(sec.manual_landmarks.source)
        assert np.any(np.isclose(tgt[:, 0] - srcs[:, 0], 12.0, atol=1e-6))
        assert state.project_path.exists()

        panel._reset_adjustment()
        assert sec.manual_landmarks is None
        assert "Atlas landmarks 0" not in viewer.layers
    finally:
        viewer.close()
