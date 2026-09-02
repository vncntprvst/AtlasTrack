"""Landmark alignment on the feature panels: handles, history, and the fit plot."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from atlastrack.ephys.penetration import PenetrationProfile, RecordingProfile
from atlastrack.ephys.recordings import NP2_ROW_PITCH_UM, recording_span
from atlastrack.gui.widgets.ephys_alignment_panel import EphysAlignmentPanel

pytestmark = pytest.mark.qt

INSERTION = 4945.0
ENTRY = (5000.0, 1000.0, 0.0)
TIP = (5000.0, 1000.0, INSERTION)
DEEP = np.arange(48) * NP2_ROW_PITCH_UM


class _DepthAtlas:
    def __init__(self) -> None:
        self.structures = {
            "A": {"rgb_triplet": [10, 20, 30]},
            "B": {"rgb_triplet": [40, 50, 60]},
        }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        if dv < 0 or dv > 6000:
            return "Outside atlas"
        return "A" if dv < 2000 else "B"


def _profile() -> PenetrationProfile:
    span = recording_span(DEEP, label="001", insertion_depth_um=INSERTION,
                          electrode_range=(1, 96))
    rng = np.random.default_rng(0)
    rec = RecordingProfile(label="001", span=span)
    rec.spike_depth_um = rng.uniform(span.top_um, span.bottom_um, 300)
    rec.spike_times_s = rng.uniform(0, 100, 300)
    rec.spike_amplitude = rng.normal(1.0, 0.2, 300)
    rec.duration_s = 100.0
    return PenetrationProfile([rec])


def _handle_indices(panel) -> list[int]:
    """Distinct landmark indices with a handle - each landmark has one per panel."""
    return sorted({line.landmark_index for line in panel._lines})


def _handles_for(panel, index: int) -> list:
    """Draggable handles only - the anatomy bar's black backing line is not one."""
    return [line for line in panel._lines
            if line.landmark_index == index and line.landmark_slot >= 0]


def _panel(qtbot, *, track: bool = True) -> EphysAlignmentPanel:
    panel = EphysAlignmentPanel()
    qtbot.addWidget(panel)
    panel.set_penetration(_profile(), track_length_um=INSERTION)
    if track:
        panel.set_track(_DepthAtlas(), TIP, ENTRY)
    return panel


# -- loading ---------------------------------------------------------------


def test_a_fresh_track_starts_with_no_landmarks(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.landmarks().n_user == 0
    assert "No landmarks" in panel.status_text()


def test_the_landmark_span_is_the_histology_track(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.landmarks().track_extent_um == pytest.approx((0.0, INSERTION))


def test_an_unregistered_shank_says_so_instead_of_offering_an_alignment(qtbot) -> None:
    panel = EphysAlignmentPanel()
    qtbot.addWidget(panel)

    panel.set_track(_DepthAtlas(), None, None)

    assert panel.landmarks() is None
    assert "no registered tip/entry" in panel.status_text()
    assert not panel._add_btn.isEnabled()


# -- editing ---------------------------------------------------------------


def test_adding_a_landmark_moves_nothing_until_aligned(qtbot) -> None:
    """Adding states an intention; Align is what acts on it."""
    panel = _panel(qtbot)
    before = panel.view().drawn_bands()

    panel.add_landmark_at(2000.0)

    assert panel.pending_pairs() == [(2000.0, 2000.0)]  # both handles together
    assert panel.landmarks().n_user == 0                # nothing applied yet
    assert panel.view().drawn_bands() == before


def test_the_two_handles_are_independent(qtbot) -> None:
    """The point of a landmark: the feature bar and the anatomy bar may disagree.

    An earlier version tied them to the same value, so the anatomy handle could never
    say anything the feature handle had not already said.
    """
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)

    panel.move_landmark(0, 1600.0, slot=0)  # feature bar up 400 µm

    assert panel.pending_pairs() == [(1600.0, 2000.0)]  # anatomy bar stayed put
    assert panel.landmarks().n_user == 0                # and still nothing applied


def test_align_moves_the_anatomy_onto_the_feature(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    before = panel.view().drawn_bands()
    panel.move_landmark(0, 1600.0, slot=0)

    panel.align()

    after = panel.view().drawn_bands()
    assert after[0][2] == pytest.approx(before[0][2] - 400.0, abs=25.0)
    assert panel.landmarks().user_pairs() == [(1600.0, 2000.0)]
    # Once applied the pair coincides, so the handles come back together.
    assert panel.pending_pairs() == pytest.approx([(1600.0, 1600.0)])


def test_dragging_the_brain_surface_marker_makes_a_landmark(qtbot) -> None:
    """The marker already claims where the brain starts; let the ephys contradict it."""
    panel = _panel(qtbot)

    panel.view().endMarkerDragged.emit(0.0, 300.0)

    # One landmark, anatomy side pinned to the surface (track depth 0).
    assert panel.pending_pairs() == [(300.0, 0.0)]

    panel.align()

    assert panel.landmarks().user_pairs() == [(300.0, 0.0)]
    # The whole track now sits 300 µm deeper on the feature axis.
    assert panel.landmarks().to_track(1300.0) == pytest.approx(1000.0)


def test_dragging_the_surface_again_moves_the_same_landmark(qtbot) -> None:
    """Re-dragging must not stack up a new landmark each time."""
    panel = _panel(qtbot)

    panel.view().endMarkerDragged.emit(0.0, 300.0)
    panel.view().endMarkerDragged.emit(0.0, -150.0)

    assert panel.pending_pairs() == [(-150.0, 0.0)]


def test_the_tip_marker_is_draggable_too(qtbot) -> None:
    """The dye marks the physical tip; the LFP only reaches the lowest electrode."""
    panel = _panel(qtbot)

    panel.view().endMarkerDragged.emit(INSERTION, INSERTION - 200.0)

    assert panel.pending_pairs() == [(INSERTION - 200.0, INSERTION)]
    # Independent of the surface: dragging one end must not move the other.
    panel.view().endMarkerDragged.emit(0.0, 120.0)
    assert sorted(panel.pending_pairs()) == [
        (120.0, 0.0), (INSERTION - 200.0, INSERTION),
    ]


def test_dragging_the_anatomy_bar_also_defines_a_correction(qtbot) -> None:
    """Either bar can be the one you move - the gap is what matters."""
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)

    panel.move_landmark(0, 2400.0, slot=1)  # anatomy bar down 400 µm
    panel.align()

    assert panel.landmarks().user_pairs() == [(2000.0, 2400.0)]


def test_a_handle_is_drawn_for_each_landmark(qtbot) -> None:
    panel = _panel(qtbot)

    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    # Each landmark gets a handle on the ephys panel *and* on the region column, so
    # it can be placed against the feature you can actually see.
    assert _handle_indices(panel) == [0, 1]
    assert len(_handles_for(panel, 0)) == 2
    assert [line.value() for line in _handles_for(panel, 0)] == pytest.approx(
        [1500.0, 1500.0]
    )
    assert all(line.movable for line in _handles_for(panel, 0))
    assert all(line.movable for line in _handles_for(panel, 1))


def test_dragging_the_feature_bar_carries_the_anatomy_bar(qtbot) -> None:
    """The order the work happens in: find the feature, the boundary follows."""
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    ephys_bar = next(h for h in _handles_for(panel, 0) if h.landmark_slot == 0)
    anatomy_bar = next(h for h in _handles_for(panel, 0) if h.landmark_slot == 1)

    ephys_bar.setValue(1600.0)
    panel._on_line_dragged(ephys_bar)

    assert anatomy_bar.value() == pytest.approx(1600.0)
    assert panel.pending_pairs() == [(1600.0, 1600.0)]


def test_dragging_the_anatomy_bar_does_not_move_the_feature_bar(qtbot) -> None:
    """The reverse must not hold - that drag *is* the disagreement being stated."""
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    ephys_bar = next(h for h in _handles_for(panel, 0) if h.landmark_slot == 0)
    anatomy_bar = next(h for h in _handles_for(panel, 0) if h.landmark_slot == 1)

    anatomy_bar.setValue(2400.0)
    panel._on_line_dragged(anatomy_bar)

    assert ephys_bar.value() == pytest.approx(2000.0)
    assert panel.pending_pairs() == [(2000.0, 2400.0)]


def test_the_anatomy_bar_has_a_backing_line_for_legibility(qtbot) -> None:
    """White alone vanishes on the pale bands, black alone on the dark ones."""
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)

    anatomy_bar = next(h for h in _handles_for(panel, 0) if h.landmark_slot == 1)
    assert anatomy_bar.halo is not None

    anatomy_bar.setValue(2400.0)
    panel._on_line_dragging(anatomy_bar)
    assert anatomy_bar.halo.value() == pytest.approx(2400.0)


def test_removing_and_clearing(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    panel.remove_landmark(0)
    assert panel.pending_pairs() == [(3000.0, 3000.0)]
    assert _handle_indices(panel) == [0]

    panel.clear_landmarks()
    assert panel.pending_pairs() == []
    assert panel.landmarks().n_user == 0
    assert panel._lines == []


def test_a_drag_that_crosses_a_neighbour_is_refused_and_explained(qtbot) -> None:
    """IBL would silently re-pair the two. We snap back and say which pair crossed."""
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    # Drag the deeper landmark's anatomy bar above the shallower one's: no monotonic
    # warp can honour that, and it is caught when Align tries to apply it.
    panel.move_landmark(1, 900.0, slot=1)
    panel.align()

    assert panel.landmarks().n_user == 0  # nothing applied
    assert "Cannot align" in panel.status_text()


def test_double_click_near_a_landmark_removes_it(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    lo, hi = panel.view().region_plot.getViewBox().viewRange()[1]

    assert panel._landmark_near(2000.0 + 0.002 * abs(hi - lo)) == 0
    assert panel._landmark_near(2000.0 + 0.4 * abs(hi - lo)) is None


# -- history ---------------------------------------------------------------


def test_previous_and_next_walk_the_applied_history(qtbot) -> None:
    """History records what was *applied*, so each Align is one step."""
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)
    panel.move_landmark(0, 1400.0, slot=0)
    panel.align()
    panel.add_landmark_at(3000.0)
    panel.move_landmark(1, 2900.0, slot=0)
    panel.align()

    assert panel.landmarks().n_user == 2
    panel.undo()
    assert panel.landmarks().n_user == 1
    panel.redo()
    assert panel.landmarks().n_user == 2


def test_snap_keeps_the_landmarks_but_drops_the_correction(qtbot) -> None:
    """Unlike Clear: the depths you decided were interesting stay pinned."""
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    panel.move_landmark(0, 1600.0, slot=0)
    panel.align()
    assert panel.landmarks().user_pairs() == [(1600.0, 2000.0)]

    panel.snap_to_no_correction()

    assert panel.landmarks().n_user == 1  # still there...
    assert panel.landmarks().user_pairs() == [(2000.0, 2000.0)]  # ...but not warping
    assert panel.landmarks().to_track(1234.0) == pytest.approx(1234.0)


def test_history_buttons_reflect_what_is_possible(qtbot) -> None:
    panel = _panel(qtbot)
    assert not panel._prev_btn.isEnabled()
    assert not panel._next_btn.isEnabled()

    panel.add_landmark_at(1500.0)
    panel.move_landmark(0, 1400.0, slot=0)
    panel.align()
    assert panel._prev_btn.isEnabled()
    assert not panel._next_btn.isEnabled()

    panel.undo()
    assert panel._next_btn.isEnabled()


def test_reset_discards_the_landmarks_and_the_history(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)

    panel.reset()

    assert panel.landmarks().n_user == 0
    assert not panel._prev_btn.isEnabled()
    assert not panel._next_btn.isEnabled()


def test_every_edit_emits_landmarks_changed(qtbot) -> None:
    panel = _panel(qtbot)

    with qtbot.waitSignal(panel.landmarksChanged, timeout=1000):
        panel.add_landmark_at(1500.0)
    with qtbot.waitSignal(panel.landmarksChanged, timeout=1000):
        panel.move_landmark(0, 1400.0, slot=0)
    with qtbot.waitSignal(panel.landmarksChanged, timeout=1000):
        panel.align()
    with qtbot.waitSignal(panel.landmarksChanged, timeout=1000):
        panel.undo()


# -- extremes mode and reporting ------------------------------------------


def test_switching_the_extremes_mode_changes_the_tails(qtbot) -> None:
    panel = _panel(qtbot)
    state = panel.landmarks()
    for feature, track in ((800.0, 900.0), (2000.0, 2200.0), (3600.0, 3950.0)):
        state = state.added(feature, track)
    panel._commit(state)

    uniform = panel.landmarks().to_track(0.0, "uniform")
    linear = panel.landmarks().to_track(0.0, "linear")
    assert uniform != pytest.approx(linear)

    panel.set_extremes_mode("linear")
    assert panel.extremes_mode() == "linear"
    with pytest.raises(ValueError, match="unknown extremes mode"):
        panel.set_extremes_mode("quadratic")


def test_status_reports_the_offset_and_the_worst_stretch(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(1000.0)
    panel.move_landmark(0, 800.0)

    text = panel.status_text()
    assert "1 landmark(s)" in text
    assert "+200 µm" in text


def test_a_wild_stretch_is_called_out(qtbot) -> None:
    panel = _panel(qtbot)
    panel._commit(panel.landmarks().added(1000.0, 1000.0).added(1200.0, 3000.0))

    assert "big local stretch" in panel.status_text()


def test_the_fit_plot_tracks_the_landmarks(qtbot) -> None:
    panel = _panel(qtbot)
    n_empty = len(panel._fit_plot.items)

    panel.add_landmark_at(2000.0)
    panel.move_landmark(0, 1600.0, slot=0)
    panel.align()

    assert len(panel._fit_plot.items) > n_empty
