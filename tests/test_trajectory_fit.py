"""Fitting a placement to detected boundaries by order-preserving assignment.

The objective this replaces summed the step profile at every atlas boundary. On real
LO_07 data that never converged: the boundary count changed with the placement (22-26
over a +/-300 µm offset), so totals compared different questions; the landscape had a
step-to-step roughness of 0.14 of its own range; and grid search landed on the edge of
whatever range it was given. The fix is to hold the *ephys* fixed and ask how much of
it a placement explains.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from atlastrack.probes.trajectory_fit import (
    ParameterScan,
    PlacementScore,
    ShankEvidence,
    fit_trajectory,
    leave_one_out,
    match_ordered,
    scan_parameter,
    score_placement,
)


class _StripedAtlas:
    """An atlas of flat slabs stacked in DV, so track anatomy is known exactly.

    Regions change every ``slab_um`` of depth, which gives boundaries at predictable
    places without needing a real atlas in the test suite.
    """

    structures: ClassVar[dict] = {}

    def __init__(self, slab_um: float = 400.0, dv0: float = 1000.0):
        self.slab_um = float(slab_um)
        self.dv0 = float(dv0)

    def structure_from_coords(self, coords, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        if dv < self.dv0:
            return "Outside atlas"
        return f"S{int((dv - self.dv0) // self.slab_um)}"


def _vertical_array(n=4, tip_dv=5000.0, track=4000.0, pitch=250.0):
    tips = np.array([[8000.0, 5000.0 + pitch * i, tip_dv] for i in range(n)])
    entries = tips - np.array([0.0, 0.0, track])
    return tips, entries


def _evidence_at(atlas, tips, entries, *, shanks=(0, 1, 2, 3), weight=5.0,
                 shift_um=0.0):
    """Detected boundaries placed exactly on this array's atlas boundaries, shifted."""
    from atlastrack.probes.trajectory_fit import atlas_boundaries_from_tip

    out = {}
    for s in shanks:
        edges = atlas_boundaries_from_tip(atlas, tips[s], entries[s])
        depths = np.array(sorted(e[0] + shift_um for e in edges))
        track = float(np.linalg.norm(tips[s] - entries[s]))
        keep = (depths > 0) & (depths < track)
        out[s] = ShankEvidence(s, depths[keep],
                               np.full(int(keep.sum()), float(weight)))
    return out


# ------------------------------------------------------------------- the matcher


def test_matches_are_order_preserving():
    pairs = match_ordered([100.0, 500.0], [1.0, 1.0], [120.0, 520.0])

    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1)]


def test_a_crossed_pairing_is_never_returned():
    """Two transitions cannot swap places along a shank under any rigid move."""
    # Nearest-neighbour would pair feature 0 -> atlas 1 and feature 1 -> atlas 0.
    pairs = match_ordered([100.0, 140.0], [1.0, 1.0], [130.0, 110.0][::-1])

    idx = [(i, j) for i, j, _ in pairs]
    assert idx == sorted(idx), idx
    assert [j for _, j, _ in pairs] == sorted(j for _, j, _ in pairs)


def test_a_feature_beyond_the_match_radius_is_left_unmatched():
    """Past a few hundred µm the nearer explanation is a different structure."""
    assert match_ordered([100.0], [1.0], [900.0], max_match_um=350.0) == []


def test_either_side_may_be_skipped():
    """Real boundaries are often LFP-silent, and vice versa."""
    pairs = match_ordered([100.0, 900.0], [1.0, 1.0], [110.0, 500.0, 910.0])

    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 2)]


def test_gain_falls_off_with_distance_and_scales_with_weight():
    (_, _, near), = match_ordered([100.0], [2.0], [100.0])
    (_, _, far), = match_ordered([100.0], [2.0], [220.0], sigma_um=120.0)

    assert near == pytest.approx(2.0)
    assert far == pytest.approx(2.0 * np.exp(-0.5), rel=1e-6)


def test_empty_input_matches_nothing():
    assert match_ordered([], [], [100.0]) == []
    assert match_ordered([100.0], [1.0], []) == []


# ------------------------------------------------------------------ the evidence


def test_evidence_is_sorted_and_weighted_from_detections():
    from atlastrack.ephys.autolandmarks import DetectedBoundary

    ev = ShankEvidence.from_boundaries(2, [
        DetectedBoundary(900.0, 5.0, 4.0, 60.0),
        DetectedBoundary(300.0, 3.0, 9.0, 100.0),
    ])

    assert ev.shank_index == 2
    assert ev.depths_from_tip_um.tolist() == [300.0, 900.0]
    assert ev.weights.tolist() == [3.0, 4.0]  # min(z, prominence)
    assert ev.total_weight == pytest.approx(7.0)


# -------------------------------------------------------------------- the score


def test_the_registered_placement_explains_its_own_boundaries():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries)

    score = score_placement(tips, entries, ev, atlas)

    assert score.explained > 0.9
    assert score.matched == score.available
    assert abs(score.mean_residual_um) < 1.0


def test_a_displaced_probe_explains_less():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries)

    here = score_placement(tips, entries, ev, atlas)
    moved = score_placement(tips, entries, ev, atlas, offset_um=250.0)

    assert moved.explained < here.explained


def test_the_denominator_is_the_ephys_so_scores_compare_across_placements():
    """The old objective divided by a boundary count that moved with the probe."""
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries)

    scores = [score_placement(tips, entries, ev, atlas, offset_um=o)
              for o in (-200.0, 0.0, 200.0)]

    assert len({s.available for s in scores}) == 1
    assert len({round(s.total_weight, 6) for s in scores}) == 1
    assert all(0.0 <= s.explained <= 1.0 for s in scores)


def test_residual_spread_reports_how_consistent_the_matches_are():
    tidy = PlacementScore(0.5, 3, 3, 3.0, matches=[])
    assert tidy.residual_spread_um == 0.0

    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries, shift_um=60.0)
    consistent = score_placement(tips, entries, ev, atlas)

    assert consistent.residual_spread_um < 30.0
    assert consistent.mean_residual_um == pytest.approx(60.0, abs=15.0)


# ------------------------------------------------------------------ the scanning


def test_a_known_offset_is_recovered():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    # The ephys sits 150 µm further from the tip than the atlas says: the probe must
    # move deeper by 150 µm to explain it.
    ev = _evidence_at(atlas, tips, entries, shift_um=150.0)

    scan = scan_parameter(tips, entries, ev, atlas, "offset_um",
                          np.arange(-400.0, 401.0, 25.0))

    assert abs(scan.best_value - 150.0) <= 50.0, scan.best_value
    assert not scan.at_edge


def test_a_flat_scan_is_reported_as_not_identifiable():
    flat = ParameterScan("roll_deg", np.arange(-10.0, 10.1, 2.5),
                         np.full(9, 0.4))

    assert flat.contrast == 0.0
    assert not flat.identifiable()


def test_an_edge_optimum_is_reported_as_not_identifiable():
    """The failure mode of the old objective: best value pinned to the scan limit."""
    rising = ParameterScan("offset_um", np.arange(0.0, 5.0),
                           np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    assert rising.at_edge
    assert not rising.identifiable()


def test_a_rough_scan_is_reported_as_not_identifiable():
    rng = np.random.default_rng(0)
    noisy = ParameterScan("tilt_deg", np.arange(21.0),
                          rng.uniform(0.0, 1.0, size=21))

    assert noisy.roughness > 0.12
    assert not noisy.identifiable()


def test_a_smooth_interior_peak_is_identifiable():
    x = np.arange(-10.0, 10.1, 1.0)
    hill = ParameterScan("offset_um", x, 0.2 + 0.6 * np.exp(-0.5 * (x / 3.0) ** 2))

    assert hill.identifiable()
    assert hill.best_value == pytest.approx(0.0)


# ---------------------------------------------------------------------- the fit


def test_fit_recovers_the_offset_and_reports_identifiability():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries, shift_um=150.0)

    fit = fit_trajectory(tips, entries, ev, atlas,
                         rolls_deg=[0.0], tilts_deg=[0.0])

    assert abs(fit.offset_um - 150.0) <= 50.0
    assert fit.improvement > 0.0
    assert set(fit.scans) == {"offset_um", "roll_deg", "tilt_deg"}
    assert "explains" in fit.summary()


def test_the_fit_never_scores_worse_than_the_registered_placement():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries)

    fit = fit_trajectory(tips, entries, ev, atlas,
                         offsets_um=[-100.0, 0.0, 100.0],
                         rolls_deg=[0.0], tilts_deg=[0.0])

    assert fit.score.explained >= fit.baseline.explained


def test_no_evidence_scores_zero_rather_than_dividing_by_it():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()

    score = score_placement(tips, entries, {}, atlas)

    assert score.explained == 0.0
    assert score.matched == 0


# --------------------------------------------------------------- leave-one-out


def test_leave_one_out_agrees_when_every_shank_says_the_same():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries, shift_um=150.0)

    loo = leave_one_out(tips, entries, ev, atlas, name="offset_um",
                        values=np.arange(-400.0, 401.0, 25.0))

    assert len(loo.per_shank) == 4
    assert loo.spread <= 50.0
    assert loo.stable(100.0)


def test_leave_one_out_exposes_a_single_shank_carrying_the_estimate():
    """Roll is identifiable only from shanks disagreeing in a coordinated way, so one
    dominant shank can produce a confident answer that says nothing about the array."""
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries)
    # Shank 3 alone claims boundaries 300 µm deeper than everyone else.
    ev[3] = ShankEvidence(3, ev[3].depths_from_tip_um + 300.0,
                          ev[3].weights * 8.0)

    loo = leave_one_out(tips, entries, ev, atlas, name="offset_um",
                        values=np.arange(-400.0, 401.0, 25.0))

    assert loo.dominant_shank == 3
    assert loo.max_influence > 100.0
    assert not loo.stable(100.0)
    assert "rests on that shank" in loo.summary(100.0, "um")


def test_weight_share_is_reported_so_dominance_is_visible_up_front():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array()
    ev = _evidence_at(atlas, tips, entries)
    ev[0] = ShankEvidence(0, ev[0].depths_from_tip_um, ev[0].weights * 10.0)

    loo = leave_one_out(tips, entries, ev, atlas, name="offset_um",
                        values=np.arange(-200.0, 201.0, 50.0))

    assert loo.weight_share[0] > 0.5
    assert sum(loo.weight_share.values()) == pytest.approx(1.0)


def test_leave_one_out_needs_at_least_three_subsets_to_call_anything_stable():
    atlas = _StripedAtlas()
    tips, entries = _vertical_array(n=2)
    ev = _evidence_at(atlas, tips, entries, shanks=(0, 1))

    loo = leave_one_out(tips, entries, ev, atlas, name="offset_um",
                        values=np.arange(-200.0, 201.0, 50.0))

    # With two shanks, holding one out leaves one - too few to refit.
    assert loo.per_shank == {}
    assert not loo.stable(100.0)
