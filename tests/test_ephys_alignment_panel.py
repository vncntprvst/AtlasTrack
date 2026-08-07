"""Landmark alignment on the feature panels: handles, history, and the fit plot."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from histo_to_ccf.ephys.penetration import PenetrationProfile, RecordingProfile
from histo_to_ccf.ephys.recordings import NP2_ROW_PITCH_UM, recording_span
from histo_to_ccf.gui.widgets.ephys_alignment_panel import EphysAlignmentPanel

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


def test_adding_a_landmark_pins_what_is_currently_shown(qtbot) -> None:
    """A new landmark must not move anything - it records, it does not correct."""
    panel = _panel(qtbot)
    before = panel.view().drawn_bands()

    panel.add_landmark_at(2000.0)

    assert panel.landmarks().user_pairs() == [(2000.0, 2000.0)]
    assert panel.view().drawn_bands() == before


def test_dragging_a_landmark_moves_the_anatomy(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    before = panel.view().drawn_bands()

    panel.move_landmark(0, 1600.0)

    after = panel.view().drawn_bands()
    assert after[0][2] == pytest.approx(before[0][2] - 400.0, abs=25.0)
    assert panel.landmarks().user_pairs() == [(1600.0, 2000.0)]


def test_a_handle_is_drawn_for_each_landmark(qtbot) -> None:
    panel = _panel(qtbot)

    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    assert [line.value() for line in panel._lines] == pytest.approx([1500.0, 3000.0])
    assert [line.landmark_index for line in panel._lines] == [0, 1]
    assert all(line.movable for line in panel._lines)


def test_removing_and_clearing(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    panel.remove_landmark(0)
    assert panel.landmarks().user_pairs() == [(3000.0, 3000.0)]
    assert len(panel._lines) == 1

    panel.clear_landmarks()
    assert panel.landmarks().n_user == 0
    assert panel._lines == []


def test_a_drag_that_crosses_a_neighbour_is_refused_and_explained(qtbot) -> None:
    """IBL would silently re-pair the two. We snap back and say which pair crossed."""
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    line = panel._lines[1]
    line.setValue(900.0)  # dragged up past the shallower landmark
    panel._on_line_dragged(line)

    assert panel.landmarks().user_pairs() == [(1500.0, 1500.0), (3000.0, 3000.0)]
    assert "not moved" in panel.status_text()
    assert line.value() == pytest.approx(3000.0)  # the handle snapped back


def test_double_click_near_a_landmark_removes_it(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(2000.0)
    lo, hi = panel.view().region_plot.getViewBox().viewRange()[1]

    assert panel._landmark_near(2000.0 + 0.002 * abs(hi - lo)) == 0
    assert panel._landmark_near(2000.0 + 0.4 * abs(hi - lo)) is None


# -- history ---------------------------------------------------------------


def test_previous_and_next_walk_the_landmark_history(qtbot) -> None:
    panel = _panel(qtbot)
    panel.add_landmark_at(1500.0)
    panel.add_landmark_at(3000.0)

    panel.undo()
    assert panel.landmarks().n_user == 1
    panel.undo()
    assert panel.landmarks().n_user == 0
    panel.redo()
    assert panel.landmarks().n_user == 1
    assert len(panel._lines) == 1


def test_history_buttons_reflect_what_is_possible(qtbot) -> None:
    panel = _panel(qtbot)
    assert not panel._prev_btn.isEnabled()
    assert not panel._next_btn.isEnabled()

    panel.add_landmark_at(1500.0)
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
    panel.move_landmark(0, 1600.0)

    assert len(panel._fit_plot.items) > n_empty
