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
        panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        assert panel is not None
        # No empty Tips/Entries marker layers at launch — adding those triggered
        # vispy "Unsupported framebuffer format" shader errors on some GPUs.
        assert "Tips" not in viewer.layers
        assert "Entries" not in viewer.layers
        # Project save/load moved to a menu (no "Save" tab); the menu exists.
        titles = [a.text() for a in viewer.window._qt_window.menuBar().actions()]
        assert any("Histo" in t for t in titles)
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
