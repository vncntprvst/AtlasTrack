"""Placing several recordings from one penetration on a shared depth axis.

The bank numbers here are the real ones from docs/dataset.md, because getting this
arithmetic wrong silently shifts every channel of a recording.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.recordings import (
    NP2_ROW_PITCH_UM,
    RecordingSpan,
    bank_offset_um,
    coverage_gaps_um,
    depth_below_surface_um,
    depth_from_tip_um,
    recording_span,
    resolve_bank_offset,
)


@pytest.mark.parametrize(
    "electrode_range, expected_um",
    [
        (None, 0.0),
        ((1, 96), 0.0),        # tip bank
        ((97, 192), 720.0),    # 96 sites / 2 columns = 48 rows x 15 µm
        ((193, 288), 1440.0),  # LO_06 2026-02-09 / 004
        ((385, 480), 2880.0),  # LO_06 2026-02-07 / 004
        ((1153, 1248), 8640.0),  # LO_06 2026-02-07 / 003
    ],
)
def test_bank_offset_matches_the_dataset_banks(electrode_range, expected_um) -> None:
    assert bank_offset_um(electrode_range) == pytest.approx(expected_um)


def test_one_bank_spans_720_um_of_shank() -> None:
    """The fact that motivates multi-recording alignment at all."""
    assert bank_offset_um((97, 192)) - bank_offset_um((1, 96)) == pytest.approx(720.0)


def test_electrode_numbering_is_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        bank_offset_um((0, 95))


def test_depth_from_tip_adds_the_bank_offset() -> None:
    local = np.array([0.0, 15.0, 30.0])

    out = depth_from_tip_um(local, bank_offset_um((97, 192)))

    assert out == pytest.approx([720.0, 735.0, 750.0])


def test_depth_below_surface_inverts_depth_from_tip() -> None:
    """The tip is deepest, so a site further up the shank is shallower."""
    depths = depth_below_surface_um(np.array([0.0, 720.0]), insertion_depth_um=4945.0)

    assert depths == pytest.approx([4945.0, 4225.0])


def test_sites_above_the_brain_surface_are_negative_not_clipped() -> None:
    """LO_06's 1153-1248 recording sits mostly out of the brain.

    That is a landmark (LFP collapses at the surface), not an error, so the value
    must stay negative rather than being clamped to 0.
    """
    span = recording_span(
        np.array([0.0, 15.0, 30.0]),
        label="LO_06_003",
        insertion_depth_um=4945.0,
        electrode_range=(1153, 1248),
    )

    assert span.above_surface is True
    assert span.bottom_um < 0.0


def test_bank_local_positions_get_the_offset_added() -> None:
    """Probe map rebuilt per bank: positions restart at 0, so add the offset."""
    axial = np.arange(48) * NP2_ROW_PITCH_UM  # 0..705

    assert resolve_bank_offset(axial, (97, 192)) == pytest.approx(720.0)


def test_absolute_positions_do_not_get_the_offset_added_twice() -> None:
    """The real case, measured on LO_06 2026-02-07/002.

    SpikeInterface reports y = 720-1410 µm for bank 97-192 because the probe map
    covers the whole shank. Adding the bank offset again would push the recording
    720 µm too shallow and open a phantom gap against the bank below it.
    """
    axial = 720.0 + np.arange(48) * NP2_ROW_PITCH_UM  # 720..1425

    assert resolve_bank_offset(axial, (97, 192)) == 0.0


def test_two_real_banks_abut_rather_than_leaving_a_gap() -> None:
    """LO_06 2026-02-07: 001 (bank 1-96) and 002 (97-192), one insertion.

    Site positions as SpikeInterface actually reports them.
    """
    deep = recording_span(np.arange(48) * NP2_ROW_PITCH_UM, label="001",
                          insertion_depth_um=4945.0, electrode_range=(1, 96))
    shallow = recording_span(720.0 + np.arange(48) * NP2_ROW_PITCH_UM, label="002",
                             insertion_depth_um=4945.0, electrode_range=(97, 192))

    assert deep.bottom_um == pytest.approx(4945.0)
    assert deep.top_um == pytest.approx(4240.0)
    assert shallow.bottom_um == pytest.approx(4225.0)
    # One row pitch between them, not the 735 µm a double-counted offset produced.
    assert deep.top_um - shallow.bottom_um == pytest.approx(NP2_ROW_PITCH_UM)
    assert coverage_gaps_um([deep, shallow]) == []


def test_an_explicit_bank_offset_overrides_the_detection() -> None:
    """The escape hatch for the LO_04 2025-08-26 bank-label discrepancy."""
    axial = np.arange(48) * NP2_ROW_PITCH_UM

    span = recording_span(axial, label="x", insertion_depth_um=1000.0,
                          electrode_range=(1, 96), bank_offset=720.0)

    assert span.bottom_um == pytest.approx(280.0)


def test_advancing_the_probe_moves_the_same_electrodes_deeper() -> None:
    """LO_07 ProbeA: same bank, insertion 4576 then 4976 µm."""
    axial = np.arange(48) * NP2_ROW_PITCH_UM
    before = recording_span(axial, label="001", insertion_depth_um=4576.0,
                            electrode_range=(1, 96))
    after = recording_span(axial, label="003", insertion_depth_um=4976.0,
                           electrode_range=(1, 96))

    assert after.bottom_um - before.bottom_um == pytest.approx(400.0)


def test_recording_span_rejects_an_empty_channel_set() -> None:
    with pytest.raises(ValueError, match="no channel positions"):
        recording_span(np.array([]), label="x", insertion_depth_um=1.0)


def test_coverage_gaps_finds_the_uncovered_stretch() -> None:
    spans = [
        RecordingSpan("deep", 4200.0, 4945.0, 96, False),
        RecordingSpan("high", 1000.0, 1700.0, 96, False),
    ]

    assert coverage_gaps_um(spans) == [(1700.0, 4200.0)]


def test_overlapping_recordings_leave_no_gap() -> None:
    spans = [
        RecordingSpan("a", 0.0, 800.0, 96, False),
        RecordingSpan("b", 700.0, 1500.0, 96, False),
    ]

    assert coverage_gaps_um(spans) == []


def test_coverage_gaps_of_nothing_is_empty() -> None:
    assert coverage_gaps_um([]) == []


def test_a_site_pitch_apart_is_not_a_coverage_gap() -> None:
    """Abutting banks are one row pitch apart; that is spacing, not a blind spot."""
    spans = [
        RecordingSpan("deep", 4240.0, 4945.0, 96, False),
        RecordingSpan("shallow", 3535.0, 4225.0, 96, False),
    ]

    assert coverage_gaps_um(spans) == []
    # A genuinely missing bank still shows up.
    assert coverage_gaps_um(spans, min_gap_um=1.0) == [(4225.0, 4240.0)]
