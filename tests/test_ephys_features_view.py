"""The depth-resolved feature panels."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from histo_to_ccf.ephys.penetration import (
    PenetrationProfile,
    RecordingProfile,
)
from histo_to_ccf.ephys.recordings import NP2_ROW_PITCH_UM, recording_span
from histo_to_ccf.gui.widgets.ephys_features_view import (
    EphysFeaturesView,
)

pytestmark = pytest.mark.qt

INSERTION = 4945.0
DEEP = np.arange(48) * NP2_ROW_PITCH_UM
SHALLOW = 720.0 + np.arange(48) * NP2_ROW_PITCH_UM


def _rec(label, axial, erange, *, spikes=0, seed=0):
    span = recording_span(axial, label=label, insertion_depth_um=INSERTION,
                          electrode_range=erange)
    prof = RecordingProfile(label=label, span=span)
    if spikes:
        rng = np.random.default_rng(seed)
        prof.spike_depth_um = rng.uniform(span.top_um, span.bottom_um, spikes)
        prof.spike_times_s = rng.uniform(0, 100, spikes)
        prof.spike_amplitude = rng.normal(1.0, 0.2, spikes)
        prof.duration_s = 100.0
    return prof


def _view(qtbot) -> EphysFeaturesView:
    view = EphysFeaturesView()
    qtbot.addWidget(view)
    return view


def test_empty_view_says_so(qtbot) -> None:
    view = _view(qtbot)

    view.set_profile(PenetrationProfile())

    assert "No recordings" in view.summary_text()


def test_panels_share_one_depth_axis(qtbot) -> None:
    """The point of the layout: a depth in one panel is that depth in all of them."""
    view = _view(qtbot)

    view.set_profile(PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=500)]))

    assert len(view._plots) >= 2
    for plot in view._plots[1:]:
        linked = plot.getViewBox().linkedView(1)  # 1 = Y axis
        assert linked is view._raster.getViewBox()


def test_depth_increases_downwards(qtbot) -> None:
    """Surface at the top, tip at the bottom - as the probe sits."""
    view = _view(qtbot)

    view.set_profile(PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=100)]))

    for plot in view._plots:
        assert plot.getViewBox().yInverted()


def test_summary_reports_coverage_against_the_track(qtbot) -> None:
    view = _view(qtbot)
    profile = PenetrationProfile([
        _rec("001", DEEP, (1, 96), spikes=400),
        _rec("002", SHALLOW, (97, 192), spikes=400, seed=1),
    ])

    view.set_profile(profile, track_length_um=INSERTION)

    text = view.summary_text()
    assert "2 recording(s)" in text
    # 3520-4945 µm covered contiguously = 1425 µm of a 4945 µm track.
    assert "29%" in text


def test_gaps_are_named_in_the_summary(qtbot) -> None:
    view = _view(qtbot)
    high = 2880.0 + np.arange(48) * NP2_ROW_PITCH_UM
    profile = PenetrationProfile([
        _rec("001", DEEP, (1, 96), spikes=200),
        _rec("004", high, (385, 480), spikes=200, seed=2),
    ])

    view.set_profile(profile, track_length_um=INSERTION)

    assert "no coverage at" in view.summary_text()


def test_unsorted_recordings_are_flagged_not_hidden(qtbot) -> None:
    """A recording with no sorting still contributes; say so rather than drop it."""
    view = _view(qtbot)
    profile = PenetrationProfile([
        _rec("001", DEEP, (1, 96), spikes=200),
        _rec("003", SHALLOW, (97, 192)),  # never sorted
    ])

    view.set_profile(profile, track_length_um=INSERTION)

    assert "no sorting for 003" in view.summary_text()


def test_overlapping_recordings_are_reported(qtbot) -> None:
    view = _view(qtbot)
    profile = PenetrationProfile([
        _rec("a", DEEP, (1, 96), spikes=200),
        _rec("b", DEEP[:24] + 300.0, (1, 96), spikes=200, seed=3),
    ])

    view.set_profile(profile, track_length_um=INSERTION)

    assert "overlapping pair" in view.summary_text()


def test_a_recording_with_no_spikes_does_not_break_the_raster(qtbot) -> None:
    view = _view(qtbot)

    view.set_profile(PenetrationProfile([_rec("003", DEEP, (1, 96))]))

    assert "1 recording(s)" in view.summary_text()


def test_redrawing_replaces_rather_than_accumulates(qtbot) -> None:
    """set_profile twice must not leave the first profile's items behind."""
    view = _view(qtbot)
    first = PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=300)])
    view.set_profile(first)
    n_after_first = len(view._raster.items)

    view.set_profile(first)

    assert len(view._raster.items) == n_after_first
