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

    # Not "No recordings loaded": that read as a broken recording when an LFP map was
    # on screen, because spikes and LFP arrive by different routes.
    assert "Nothing loaded yet" in view.summary_text()


def test_with_lfp_but_no_spikes_the_status_says_which(qtbot) -> None:
    view = _view(qtbot)
    depths = np.linspace(0.0, 3000.0, 32)

    view.set_lfp(depths, np.random.default_rng(0).random((32, 20)), np.linspace(0, 300, 20))
    view.set_profile(PenetrationProfile())

    text = view.summary_text()
    assert "LFP loaded" in text
    assert "No sorted spikes" in text
    assert view.available_modes() == ["lfp"]


def test_panels_share_one_depth_axis(qtbot) -> None:
    """The point of the layout: a depth in one panel is that depth in all of them."""
    view = _view(qtbot)

    view.set_profile(PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=500)]))

    # Two panels only: one toggleable ephys panel, plus the atlas region column.
    assert len(view._plots) == 2
    others = [p for p in view._plots if p is not view._ephys_plot]
    assert len(others) == 1
    for plot in others:
        linked = plot.getViewBox().linkedView(1)  # 1 = Y axis
        assert linked is view._ephys_plot.getViewBox()


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


# -- atlas region column ---------------------------------------------------

ENTRY = (5000.0, 1000.0, 0.0)
TIP = (5000.0, 1000.0, INSERTION)


class _DepthAtlas:
    """Fake atlas: two regions, boundary at DV 2000 µm, nothing past 6000."""

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


def test_region_column_shares_the_depth_axis(qtbot) -> None:
    view = _view(qtbot)

    view.set_profile(PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=200)]))
    view.set_track(_DepthAtlas(), TIP, ENTRY)

    assert view.region_plot is not None
    assert view.region_plot.getViewBox().linkedView(1) is view._raster.getViewBox()
    assert view.region_plot.getViewBox().yInverted()


def test_regions_are_found_along_the_track(qtbot) -> None:
    view = _view(qtbot)

    view.set_track(_DepthAtlas(), TIP, ENTRY)

    bands = view.bands()
    assert [b.acronym for b in bands] == ["A", "B"]
    assert bands[0].bottom_um == pytest.approx(2000.0, abs=20.0)


def test_no_atlas_leaves_the_column_empty_without_raising(qtbot) -> None:
    view = _view(qtbot)

    view.set_track(None, TIP, ENTRY)

    assert view.bands() == []
    assert view.drawn_bands() == []


def test_an_unregistered_shank_leaves_the_column_empty(qtbot) -> None:
    view = _view(qtbot)

    view.set_track(_DepthAtlas(), None, None)

    assert view.bands() == []


def test_landmarks_stretch_the_drawn_regions_not_the_ephys(qtbot) -> None:
    """The whole point of the column: anatomy moves, the measured features do not."""
    from histo_to_ccf.ephys.landmarks import Landmarks

    view = _view(qtbot)
    view.set_profile(PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=200)]))
    view.set_track(_DepthAtlas(), TIP, ENTRY)
    before = view.drawn_bands()
    raster_before = len(view._raster.items)

    # Pin the A/B boundary 400 µm shallower than the histology puts it.
    lm = Landmarks.identity(0.0, INSERTION).added(1600.0, 2000.0)
    view.set_landmarks(lm)

    after = view.drawn_bands()
    assert [b[0] for b in after] == [b[0] for b in before]
    assert after[0][2] == pytest.approx(before[0][2] - 400.0, abs=25.0)
    assert len(view._raster.items) == raster_before  # the ephys panels never moved


def test_clearing_the_landmarks_puts_the_regions_back(qtbot) -> None:
    from histo_to_ccf.ephys.landmarks import Landmarks

    view = _view(qtbot)
    view.set_track(_DepthAtlas(), TIP, ENTRY)
    before = view.drawn_bands()

    view.set_landmarks(Landmarks.identity(0.0, INSERTION).added(1600.0, 2000.0))
    view.set_landmarks(None)

    assert view.drawn_bands() == before


def test_region_redraw_replaces_rather_than_accumulates(qtbot) -> None:
    view = _view(qtbot)
    view.set_track(_DepthAtlas(), TIP, ENTRY)
    n_first = len(view.region_plot.items)

    view.set_track(_DepthAtlas(), TIP, ENTRY)

    assert len(view.region_plot.items) == n_first


def test_gap_shading_reaches_the_region_column_too(qtbot) -> None:
    view = _view(qtbot)
    high = 2880.0 + np.arange(48) * NP2_ROW_PITCH_UM
    profile = PenetrationProfile([
        _rec("001", DEEP, (1, 96), spikes=100),
        _rec("004", high, (385, 480), spikes=100, seed=2),
    ])

    view.set_track(_DepthAtlas(), TIP, ENTRY)
    view.set_profile(profile, track_length_um=INSERTION)

    assert view.drawn_bands()  # the region bands survived the gap overlay
    assert len(view.region_plot.items) > len(view.drawn_bands())


def test_redrawing_replaces_rather_than_accumulates(qtbot) -> None:
    """set_profile twice must not leave the first profile's items behind."""
    view = _view(qtbot)
    first = PenetrationProfile([_rec("001", DEEP, (1, 96), spikes=300)])
    view.set_profile(first)
    n_after_first = len(view._raster.items)

    view.set_profile(first)

    assert len(view._raster.items) == n_after_first
