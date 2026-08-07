"""Merging several recordings of one penetration onto the shared depth axis."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.penetration import (
    PenetrationProfile,
    RecordingProfile,
    to_shared_axis,
)
from histo_to_ccf.ephys.recordings import NP2_ROW_PITCH_UM, recording_span

INSERTION = 4945.0


def _profile(label, axial, electrode_range, *, spikes=0, seed=0):
    span = recording_span(axial, label=label, insertion_depth_um=INSERTION,
                          electrode_range=electrode_range)
    prof = RecordingProfile(label=label, span=span)
    if spikes:
        rng = np.random.default_rng(seed)
        depths_axial = rng.uniform(axial.min(), axial.max(), spikes)
        prof.spike_depth_um = to_shared_axis(
            depths_axial, insertion_depth_um=INSERTION,
            electrode_range=electrode_range, reference_axial_um=axial,
        )
        prof.spike_times_s = rng.uniform(0, 100, spikes)
        prof.spike_amplitude = rng.normal(1.0, 0.1, spikes)
        prof.duration_s = 100.0
    return prof


DEEP_AXIAL = np.arange(48) * NP2_ROW_PITCH_UM            # bank 1-96, y 0..705
SHALLOW_AXIAL = 720.0 + np.arange(48) * NP2_ROW_PITCH_UM  # bank 97-192, y 720..1425


def test_to_shared_axis_uses_the_channels_not_the_spikes_to_pick_the_convention() -> None:
    """Spikes localise beyond the site span, so they must not decide the offset.

    Measured on real data: a 0-705 µm bank yields spike depths from -31 to 734 µm.
    Judging the bank convention from those could be off by a row.
    """
    spike_axial = np.array([-31.0, 734.0])

    out = to_shared_axis(
        spike_axial, insertion_depth_um=INSERTION,
        electrode_range=(1, 96), reference_axial_um=DEEP_AXIAL,
    )

    assert out == pytest.approx([INSERTION + 31.0, INSERTION - 734.0])


def test_to_shared_axis_handles_an_absolute_geometry() -> None:
    out = to_shared_axis(
        np.array([720.0]), insertion_depth_um=INSERTION,
        electrode_range=(97, 192), reference_axial_um=SHALLOW_AXIAL,
    )

    assert out == pytest.approx([INSERTION - 720.0])


def test_depth_range_spans_all_recordings() -> None:
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96)),
        _profile("002", SHALLOW_AXIAL, (97, 192)),
    ])

    top, bottom = pen.depth_range_um()

    assert bottom == pytest.approx(INSERTION)
    assert top == pytest.approx(INSERTION - 1425.0)


def test_two_abutting_banks_leave_no_gap() -> None:
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96)),
        _profile("002", SHALLOW_AXIAL, (97, 192)),
    ])

    assert pen.gaps_um() == []


def test_a_missing_bank_shows_as_a_gap() -> None:
    """LO_06 also recorded bank 385-480, well above the two adjacent ones."""
    high = 2880.0 + np.arange(48) * NP2_ROW_PITCH_UM
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96)),
        _profile("004", high, (385, 480)),
    ])

    gaps = pen.gaps_um()

    assert len(gaps) == 1
    lo, hi = gaps[0]
    assert hi - lo == pytest.approx(2880.0 - 705.0)


def test_overlaps_are_reported_not_merged() -> None:
    """Where two recordings overlap they must stay comparable.

    A disagreement there is the evidence that an insertion depth or bank label is
    wrong, so averaging them would destroy the only independent check available.
    """
    pen = PenetrationProfile([
        _profile("a", DEEP_AXIAL, (1, 96)),
        _profile("b", DEEP_AXIAL[:24] + 300.0, (1, 96)),
    ])

    overlaps = pen.overlaps_um()

    assert len(overlaps) == 1
    lo, hi, first, second = overlaps[0]
    assert hi > lo
    assert {first, second} == {"a", "b"}


def test_no_overlap_between_separate_banks() -> None:
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96)),
        _profile("002", SHALLOW_AXIAL, (97, 192)),
    ])

    assert pen.overlaps_um() == []


def test_coverage_fraction_counts_overlap_once() -> None:
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96)),
        _profile("002", SHALLOW_AXIAL, (97, 192)),
    ])

    # 1425 µm of contiguous coverage on a 4945 µm track.
    assert pen.coverage_fraction(INSERTION) == pytest.approx(1425.0 / INSERTION, rel=1e-3)


def test_coverage_fraction_of_one_bank_is_the_small_number_that_motivates_this() -> None:
    pen = PenetrationProfile([_profile("001", DEEP_AXIAL, (1, 96))])

    assert pen.coverage_fraction(INSERTION) < 0.15


def test_all_spikes_concatenates_recordings_with_sorting() -> None:
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96), spikes=500),
        _profile("002", SHALLOW_AXIAL, (97, 192), spikes=300),
    ])

    depth, amp, times = pen.all_spikes()

    assert depth.size == amp.size == times.size == 800


def test_recordings_without_sorting_are_skipped_not_fatal() -> None:
    """LO_06 2026-02-07/003 and /004 were never sorted; they still contribute LFP."""
    pen = PenetrationProfile([
        _profile("001", DEEP_AXIAL, (1, 96), spikes=200),
        _profile("004", SHALLOW_AXIAL, (97, 192)),  # no spikes
    ])

    depth, _amp, _t = pen.all_spikes()

    assert depth.size == 200
    assert pen.profiles[1].has_spikes is False


def test_an_empty_penetration_is_harmless() -> None:
    pen = PenetrationProfile()

    assert pen.depth_range_um() == (0.0, 0.0)
    assert pen.gaps_um() == []
    assert pen.coverage_fraction(1000.0) == 0.0
    assert pen.all_spikes()[0].size == 0
