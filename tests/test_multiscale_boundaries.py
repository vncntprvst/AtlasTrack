"""Multi-scale step detection: several window widths, best evidence kept.

Written because a single 250 µm window, tuned on the 5745 µm single-column recording,
missed both boundaries on a 705 µm bank by 132-158 µm - while an 80 µm window found
one of them to within 2 µm. No single width works when the structures differ in size.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.autolandmarks import (
    MIN_BOUNDARY_Z,
    WINDOW_LADDER_UM,
    DetectedBoundary,
    _phase_randomised,
    adaptive_windows,
    calibrate_scales,
    detect_boundaries,
    multiscale_step_profile,
    step_profile,
)


def _stepped(depths, edges, *, amp=4.0, noise=0.25, seed=0):
    """A feature that steps by ``amp`` at each depth in ``edges``."""
    rng = np.random.default_rng(seed)
    level = np.zeros(depths.size, dtype=float)
    for i, e in enumerate(edges):
        level += amp * (depths >= e) * (-1.0) ** i
    return level + rng.normal(scale=noise, size=depths.size)


# ------------------------------------------------------------------- the windows


def test_a_short_recording_only_gets_windows_it_can_support():
    """A 250 µm window on a 705 µm bank leaves 205 µm of scorable grid."""
    assert adaptive_windows(705.0) == (60.0, 100.0, 150.0, 220.0)
    assert adaptive_windows(400.0) == (60.0, 100.0)
    assert adaptive_windows(5745.0) == WINDOW_LADDER_UM


def test_a_tiny_recording_still_gets_one_window_rather_than_none():
    """An empty profile reads as 'no boundaries here', which is a different claim."""
    assert adaptive_windows(30.0) == (60.0,)


# ------------------------------------------------------------------- the profile


def test_a_sharp_step_is_found_at_the_narrow_scale():
    depths = np.arange(0.0, 700.0, 15.0)
    values = _stepped(depths, [350.0])

    prof = multiscale_step_profile(depths, values)
    found = detect_boundaries(prof)

    assert found, "a 4-sigma step must be detected"
    best = max(found, key=lambda b: b.z_score)
    assert abs(best.depth_um - 350.0) < 40.0


def test_the_profile_peaks_at_both_scales_of_transition():
    """One boundary sharp at 60-100 µm, another gradual - both must show in the
    combined profile, which is what taking the best scale rather than averaging buys.

    Asserted on the profile, not on the detections, because a *lone* gradual ramp
    cannot clear a spectrum-matched null: a surrogate with the same power spectrum
    contains smooth ramps too. On real data the gradual ones do get detected (LO_07
    shank 0 finds transitions at the 220 µm scale) because they sit alongside other
    structure that fixes the scale of the null.
    """
    rng = np.random.default_rng(1)
    depths = np.arange(0.0, 2000.0, 10.0)
    sharp = _stepped(depths, [500.0], amp=5.0, noise=0.3, seed=1)
    gradual = 4.0 * np.tanh((depths - 1400.0) / 250.0) + rng.normal(scale=0.4,
                                                                    size=depths.size)
    prof = multiscale_step_profile(depths, np.column_stack([sharp, gradual]))

    def scale_at(depth):
        i = int(np.argmin(np.abs(prof.grid_um - depth)))
        window = slice(max(0, i - 12), i + 13)
        j = window.start + int(np.nanargmax(prof.score[window]))
        return float(prof.scale_um[j]), float(prof.score[j])

    sharp_scale, sharp_z = scale_at(500.0)
    grad_scale, grad_z = scale_at(1400.0)

    assert sharp_z > 0 and grad_z > 0
    assert sharp_scale <= grad_scale, (sharp_scale, grad_scale)
    assert detect_boundaries(prof), "the sharp step at least must be detected"


def test_a_lone_gradual_ramp_is_honestly_reported_as_indistinguishable():
    """Not a defect - the null is right, and this is worth pinning down.

    A phase-randomised surrogate keeps the power spectrum, so the surrogate of a
    smooth ramp is itself smooth and full of ramp-like stretches. One gradual
    transition on an otherwise featureless trace therefore does not stand above what
    that trace produces by chance, and the detector says nothing rather than
    manufacturing a confident boundary.
    """
    rng = np.random.default_rng(8)
    depths = np.arange(0.0, 2000.0, 10.0)
    gradual = 4.0 * np.tanh((depths - 1000.0) / 300.0) + rng.normal(scale=0.4,
                                                                    size=depths.size)
    prof = multiscale_step_profile(depths, gradual)

    # The profile still points at the right depth with the right scale...
    i = int(np.nanargmax(prof.score))
    assert abs(float(prof.grid_um[i]) - 1000.0) < 200.0
    assert float(prof.scale_um[i]) >= 100.0
    # ...it just does not clear the bar, and says so.
    assert np.nanmax(prof.score) < prof.null_z * 1.5


def test_multi_scale_beats_the_fixed_window_on_a_bank_sized_recording():
    """The regression this module exists for, as a synthetic 705 µm bank."""
    depths = np.arange(0.0, 705.0, 15.0)
    values = _stepped(depths, [180.0], amp=5.0, noise=0.3, seed=2)

    multi = detect_boundaries(multiscale_step_profile(depths, values))
    grid, score = step_profile(depths, values, window_um=250.0, step_um=5.0)

    multi_err = min(abs(b.depth_um - 180.0) for b in multi)
    fixed_err = (abs(float(grid[int(np.argmax(score))]) - 180.0)
                 if grid.size else float("inf"))
    assert multi_err < fixed_err


def test_a_real_step_stands_far_above_its_null_and_noise_does_not():
    """The discriminant, stated as the ratio it actually is.

    Not "noise yields nothing": the floor is the 90th percentile of the surrogate
    peaks, so ~10% of pure-noise traces exceed it *by construction*, and a test
    asserting zero false positives would be asserting something untrue. What separates
    them is how far above the null the peak sits - measured on LO_07 ProbeA, real
    boundaries reach 4.4-19.2 against surrogate maxima of 3.4-4.7.
    """
    depths = np.arange(0.0, 705.0, 15.0)
    stepped = _stepped(depths, [350.0], amp=6.0, noise=0.3, seed=11)
    noise = np.random.default_rng(12).normal(size=depths.size)

    real = multiscale_step_profile(depths, stepped)
    flat = multiscale_step_profile(depths, noise)

    assert np.nanmax(real.score) > 1.5 * real.null_z
    assert np.nanmax(flat.score) < 1.5 * flat.null_z


def test_the_floor_comes_from_the_signals_own_null_by_default():
    depths = np.arange(0.0, 705.0, 15.0)
    prof = multiscale_step_profile(depths, _stepped(depths, [350.0], seed=13))

    assert np.isfinite(prof.null_z)
    # Passing the same floor explicitly must give the same answer as defaulting to it.
    assert detect_boundaries(prof) == detect_boundaries(prof, min_z=prof.null_z)


def test_skipping_the_null_falls_back_to_a_fixed_floor():
    depths = np.arange(0.0, 705.0, 15.0)
    prof = multiscale_step_profile(depths, _stepped(depths, [350.0], seed=14),
                                   n_surrogates=0)

    assert np.isnan(prof.null_z)
    assert detect_boundaries(prof) == detect_boundaries(prof, min_z=MIN_BOUNDARY_Z)


def test_the_null_is_calibrated_per_scale_not_from_the_profiles_own_spread():
    """A surrogate is featureless, so its own spread is small and its z inflated."""
    depths = np.arange(0.0, 705.0, 15.0)
    values = _stepped(depths, [350.0], amp=6.0, noise=0.3, seed=15)

    stats, null_z = calibrate_scales(depths, values, adaptive_windows(705.0))

    assert set(stats) == set(adaptive_windows(705.0))
    for _centre, spread in stats.values():
        assert spread > 0
    assert np.isfinite(null_z)


def test_a_phase_randomised_surrogate_keeps_the_roughness_and_loses_the_step():
    rng = np.random.default_rng(16)
    depths = np.arange(0.0, 1500.0, 10.0)
    values = _stepped(depths, [700.0], amp=6.0, noise=0.3, seed=17)[:, None]

    surrogate = _phase_randomised(values, rng)

    # Same power spectrum (hence roughness), so the same variance...
    assert float(np.std(surrogate)) == pytest.approx(float(np.std(values)), rel=0.05)
    # ...but the step is gone.
    real_jump = abs(values[depths >= 700].mean() - values[depths < 700].mean())
    sur_jump = abs(surrogate[depths >= 700].mean() - surrogate[depths < 700].mean())
    assert sur_jump < 0.5 * real_jump


def test_too_few_depths_is_empty_not_an_error():
    prof = multiscale_step_profile(np.array([0.0, 15.0]), np.array([1.0, 2.0]))

    assert prof.grid_um.size == 0
    assert detect_boundaries(prof) == []


# ------------------------------------------------------------------- the peaks


def test_peaks_come_back_in_depth_order():
    depths = np.arange(0.0, 2000.0, 10.0)
    values = _stepped(depths, [400.0, 1200.0], amp=5.0, noise=0.3, seed=4)

    found = detect_boundaries(multiscale_step_profile(depths, values))

    assert [b.depth_um for b in found] == sorted(b.depth_um for b in found)


def test_the_same_transition_seen_twice_is_reported_once():
    depths = np.arange(0.0, 1500.0, 10.0)
    values = _stepped(depths, [700.0], amp=6.0, noise=0.3, seed=5)

    found = detect_boundaries(multiscale_step_profile(depths, values),
                              min_separation_um=200.0)

    close = [b for b in found if abs(b.depth_um - 700.0) < 200.0]
    assert len(close) == 1


def test_the_stronger_of_two_close_peaks_survives_thinning():
    depths = np.arange(0.0, 2000.0, 10.0)
    strong = _stepped(depths, [900.0], amp=8.0, noise=0.2, seed=6)
    weak = 0.8 * (depths >= 1000.0)

    found = detect_boundaries(multiscale_step_profile(depths, strong + weak),
                              min_separation_um=300.0)

    close = [b for b in found if 800.0 <= b.depth_um <= 1200.0]
    assert len(close) == 1
    assert abs(close[0].depth_um - 900.0) < abs(close[0].depth_um - 1000.0)


def test_max_n_caps_the_number_returned_keeping_the_strongest():
    depths = np.arange(0.0, 3000.0, 10.0)
    values = _stepped(depths, [500.0, 1200.0, 2000.0], amp=5.0, noise=0.3, seed=7)

    capped = detect_boundaries(multiscale_step_profile(depths, values), max_n=2)

    assert len(capped) <= 2


def test_weight_is_the_smaller_of_strength_and_prominence():
    """Either alone is fooled: a plateau is strong, a spike on noise is prominent."""
    plateau = DetectedBoundary(depth_um=0.0, z_score=6.0, prominence=0.4, scale_um=60.0)
    spike = DetectedBoundary(depth_um=0.0, z_score=2.1, prominence=5.0, scale_um=60.0)
    real = DetectedBoundary(depth_um=0.0, z_score=5.0, prominence=4.5, scale_um=60.0)

    assert plateau.weight == pytest.approx(0.4)
    assert spike.weight == pytest.approx(2.1)
    assert real.weight == pytest.approx(4.5)


def test_weight_is_never_negative():
    assert DetectedBoundary(0.0, -3.0, -1.0, 60.0).weight == 0.0
