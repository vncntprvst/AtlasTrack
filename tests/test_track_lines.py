"""Where a probe track's line is drawn on the sections."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.gui.widgets.click_overlay import track_segments


def test_one_section_line_runs_on_past_the_entry_to_the_box_edge() -> None:
    (a, b), = track_segments((50, 80), (0, 0, 100, 100), (50, 40), (0, 0, 100, 100))
    assert np.allclose(a, (50, 80))
    assert np.allclose(b, (50, 0))          # straight up through the entry, to the top


def test_tip_and_entry_on_different_sections() -> None:
    tip_box, entry_box = (0, 0, 100, 100), (200, 0, 300, 100)
    (t0, t1), (e0, e1) = track_segments((50, 80), tip_box, (260, 40), entry_box)
    # Tip's section: from the tip toward the entry placed at the same spot in this
    # box (60, 40), on to the box edge.
    assert np.allclose(t0, (50, 80)) and np.allclose(t1, (70, 0))
    # Entry's section: from the entry away from the tip placed likewise (250, 80).
    assert np.allclose(e0, (260, 40)) and np.allclose(e1, (270, 0))


def test_no_line_until_both_markers_are_placed() -> None:
    assert track_segments((50, 80), (0, 0, 100, 100), None, None) == []


def _two_section_widget(qtbot, tissue_boxes):
    """A widget over a slide with two sections whose tissue sits at ``tissue_boxes``."""
    import napari

    from atlastrack.gui.widgets.click_overlay import ClickOverlayWidget
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.project.schema import Section

    img = np.zeros((120, 320), dtype=np.uint8)
    for x0, y0, x1, y1 in tissue_boxes:
        img[y0:y1, x0:x1] = 200
    state = WorkflowState()
    state.add_slide("s.png", img)
    state.active_slide_idx = 0
    state.project.slides[0].sections += [
        Section(index=0, slide_idx=0, bbox_px=(0, 0, 150, 120), ap_order=0),
        Section(index=1, slide_idx=0, bbox_px=(160, 0, 320, 120), ap_order=1),
    ]
    viewer = napari.Viewer(show=False)
    widget = ClickOverlayWidget(state, viewer)
    qtbot.addWidget(widget)
    return widget, viewer


def test_carried_against_the_tissue_not_the_box(qtbot) -> None:
    """Tissue placed differently in each box: the same spot relative to the tissue."""
    # Section 0 tissue: x 20-120 (midline 70); section 1 tissue: x 200-280 (midline 240).
    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        # 25 px left of section 1's midline (half-width 40) -> 31 px left of section
        # 0's midline (half-width 50), at the same fraction of the tissue height.
        out = widget._carry_by_tissue((215.0, 40.0), 1, 0)
        assert out is not None
        assert abs(out[0] - (70 - 25 * 50 / 40)) < 3
        assert abs(out[1] - 40.0) < 3
    finally:
        viewer.close()


def test_carried_through_the_atlas_when_both_sections_are_registered(qtbot) -> None:
    """With registrations, the carried point has the same atlas left-right and depth."""
    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])

    class _Linear:
        """Section pixel -> (AP, ML, DV): ML = a + x, DV = y (one um per pixel)."""

        def __init__(self, ml0, ap):
            self.ml0, self.ap = ml0, ap

        def apply_many(self, px):
            px = np.asarray(px, dtype=float).reshape(-1, 2)
            return np.stack([np.full(len(px), self.ap), self.ml0 + px[:, 0], px[:, 1]], 1)

    # Section 1's image is shifted 30 px relative to section 0 in ML.
    transforms = {0: _Linear(1000.0, 9000.0), 1: _Linear(970.0, 9500.0)}
    widget._section_transform = lambda idx: transforms.get(idx)
    from atlastrack.project.schema import RegistrationResult

    for sec in widget._state.project.slides[0].sections:
        sec.registration = RegistrationResult(anchoring=[0.0] * 9, output_size_px=(120, 150))
    widget._state.atlas = object()          # any atlas: the transforms are faked
    widget._atlas_in_background = False     # build them right away, not in a thread
    try:
        out = widget._carry_through_atlas((260.0, 50.0), 1, 0)   # ML 970+100, DV 50
        assert out is not None
        assert abs(out[0] - 70.0) < 2 and abs(out[1] - 50.0) < 2  # ML 1000+70, DV 50
        # The line uses it rather than the tissue estimate.
        point, final = widget._carry_over((260.0, 50.0), 1, 0)
        assert final and np.allclose(point, out)
    finally:
        viewer.close()


def test_without_an_image_the_box_is_used(qtbot) -> None:
    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        widget._state.slide_images.clear()
        assert widget._carry_over((215.0, 40.0), 1, 0)[0] is None   # -> same spot in the box
    finally:
        viewer.close()


def test_atlas_transforms_are_built_off_the_gui_thread(qtbot) -> None:
    """Until a section's transform is ready, the tissue estimate is used and the
    transform is built in the background (opening the Probes tab froze for 9 s)."""
    from atlastrack.project.schema import RegistrationResult

    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        for s in widget._state.project.slides[0].sections:
            s.registration = RegistrationResult(anchoring=[0.0] * 9, output_size_px=(120, 150))
        widget._state.atlas = object()
        started = []
        widget._build_transforms_in_background = lambda idx: started.append(list(idx))
        out, final = widget._carry_over((215.0, 40.0), 1, 0)
        assert started == [[1, 0]]
        assert np.allclose(out, widget._carry_by_tissue((215.0, 40.0), 1, 0))
        assert not final  # redone through the atlas once the transforms are built
    finally:
        viewer.close()


def test_moving_one_shanks_marker_leaves_the_other_lines_alone(qtbot) -> None:
    """Dragging a marker used to redraw every shank's line the quick way, so all the
    lines moved, and some did not land back where they were."""
    from atlastrack.project.schema import Point2D, ProbeSpec, ProbeType, Shank

    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        def shank(i, x):
            return Shank(index=i, tip_px=Point2D(x_px=x, y_px=90.0), tip_section_idx=0,
                         entry_px=Point2D(x_px=x + 160.0, y_px=30.0), entry_section_idx=1)

        widget._state.project.probes.append(ProbeSpec(
            label="P", type=ProbeType(name="Neuropixels 2.0 (4-shank)", n_shanks=2),
            shanks=[shank(0, 50.0), shank(1, 80.0)]))
        quick = []
        real = widget._carry_over

        def spy(xy, a, b, *, quick_flag=None, **kw):
            quick.append(kw.get("quick", False))
            return real(xy, a, b, **kw)

        widget._carry_over = lambda xy, a, b, **kw: spy(xy, a, b, **kw)
        widget._rebuild_markers()
        before = [np.array(seg) for seg in widget._segments_for(0, 1)]

        quick.clear()
        widget._move_marker(0, 0, "entry", 215.0, 35.0, commit=False)   # dragging
        assert quick and all(quick)   # only the dragged shank was recomputed, quickly
        assert all(np.array_equal(a, b) for a, b in zip(widget._segments_for(0, 1), before))
        widget._move_marker(0, 0, "entry", 215.0, 35.0, commit=True)    # dropped
        after = widget._segments_for(0, 1)
        assert len(after) == len(before)
        assert all(np.array_equal(a, b) for a, b in zip(after, before))
    finally:
        viewer.close()


def _two_probes(widget):
    from atlastrack.project.schema import Point2D, ProbeSpec, ProbeType, Shank

    placed = Shank(index=0, tip_px=Point2D(x_px=50.0, y_px=90.0), tip_section_idx=0,
                   entry_px=Point2D(x_px=210.0, y_px=30.0), entry_section_idx=1)
    coords_only = Shank(index=0, tip_ccf_um=(9000.0, 1050.0, 90.0),
                        entry_ccf_um=(9500.0, 1080.0, 30.0))
    widget._state.project.probes += [
        ProbeSpec(label="A", type=ProbeType(name="Neuropixels 2.0 (4-shank)", n_shanks=1),
                  shanks=[placed]),
        ProbeSpec(label="B", type=ProbeType(name="Neuropixels 2.0 (4-shank)", n_shanks=1),
                  shanks=[coords_only]),
    ]
    widget._refresh_probe_combo()
    widget._refresh_table()


def test_the_list_shows_shanks_without_markers_and_the_dropdowns_select(qtbot) -> None:
    """A probe placed from coordinates had no rows, so it could not be found or picked."""
    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        _two_probes(widget)
        labels = [widget._table.item(r, 0).text() for r in range(widget._table.rowCount())]
        assert labels == ["A", "B"]
        assert widget._table.item(1, 2).text() == "not placed"
        widget._probe_combo.setCurrentIndex(1)
        assert widget._selected == (1, 0)
        assert widget._table.selectedItems()[0].row() == 1
        assert "Markers from coordinates" in widget._status.text()
    finally:
        viewer.close()


def test_arrow_keys_nudge_the_chosen_marker(qtbot) -> None:
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QApplication

    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        _two_probes(widget)
        widget.show()   # the keys are the Probes tab's only while it is shown
        widget._select_track(0, 0, "tip")
        canvas = widget._canvas_widget()
        for key, mods in ((Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier),
                          (Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)):
            QApplication.sendEvent(canvas, QKeyEvent(QEvent.Type.KeyPress, key, mods))
        tip = widget._state.project.probes[0].shanks[0].tip_px
        assert (tip.x_px, tip.y_px) == pytest.approx((51.0, 90.1))
        entry = widget._state.project.probes[0].shanks[0].entry_px
        assert (entry.x_px, entry.y_px) == (210.0, 30.0)     # the other marker stays
    finally:
        viewer.close()


class _Linear:
    """Section pixel -> (AP, ML, DV): ML = a + x, DV = y (one um per pixel)."""

    def __init__(self, ml0, ap):
        self.ml0, self.ap = ml0, ap

    def apply_many(self, px):
        px = np.asarray(px, dtype=float).reshape(-1, 2)
        return np.stack([np.full(len(px), self.ap), self.ml0 + px[:, 0], px[:, 1]], 1)


def test_the_pixel_search_is_finer_than_a_pixel(qtbot) -> None:
    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        px, err = widget._pixel_at(_Linear(1000.0, 9000.0), (1033.37, 41.62), (0, 0, 150, 120))
        assert err < 0.01
        assert np.allclose(px, (33.37, 41.62), atol=0.01)
    finally:
        viewer.close()


def test_markers_from_coordinates_go_on_the_nearest_section_in_ap(qtbot) -> None:
    from atlastrack.project.schema import RegistrationResult

    widget, viewer = _two_section_widget(qtbot, [(20, 20, 120, 100), (200, 20, 280, 100)])
    try:
        _two_probes(widget)
        for sec in widget._state.project.slides[0].sections:
            sec.registration = RegistrationResult(anchoring=[0.0] * 9, output_size_px=(120, 150))
        widget._state.atlas = object()
        transforms = {0: _Linear(1000.0, 9000.0), 1: _Linear(970.0, 9500.0)}
        widget._section_transform = lambda idx: transforms.get(idx)
        widget._atlas_in_background = False     # the transforms are faked: no thread
        widget._probe_combo.setCurrentIndex(1)
        widget._markers_from_coordinates()
        shank = widget._state.project.probes[1].shanks[0]
        assert shank.tip_section_idx == 0 and shank.entry_section_idx == 1
        assert (shank.tip_px.x_px, shank.tip_px.y_px) == pytest.approx((50.0, 90.0), abs=0.05)
        # Section 1 starts at x = 160 in the slide; ML 1080 is its pixel 110.
        assert (shank.entry_px.x_px, shank.entry_px.y_px) == pytest.approx((270.0, 30.0), abs=0.05)
    finally:
        viewer.close()
