"""Array roll and along-track offset, plus the boundary-contrast measure they score with."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from atlastrack.ephys.features import boundary_contrast, contrast_null
from atlastrack.probes.trajectory_refine import (
    array_axes,
    lateral_sign,
    pitch_deg,
    refine_trajectory,
    roll_deg,
    rolled_array,
    row_direction,
    score_trajectory,
    shank_row_positions,
    shift_along_track,
    tilted_array,
    transformed_array,
)

PITCH = 250.0


MIDLINE = 5700.0


def _array(roll_degrees: float = 0.0, *, n: int = 4, depth: float = 4000.0,
           ml: float = 4000.0, pitch_degrees: float = 0.0):
    """A rigid comb in the lab's convention: 0° roll = row along AP.

    ``ml`` places the array; 4000 µm is medial of the 5700 µm midline, so "lateral"
    there is -ML. Built from :func:`row_direction` so the test states the convention
    once and the geometry follows from it.
    """
    lateral = 1.0 if ml >= MIDLINE else -1.0
    u = np.array([np.sin(np.radians(pitch_degrees)), 0.0, np.cos(np.radians(pitch_degrees))])
    row = row_direction(roll_degrees, u, lateral)
    offsets = (np.arange(n, dtype=float) - (n - 1) / 2.0) * PITCH
    entry = np.array([5000.0, ml, 1000.0])
    entries = entry[None, :] + offsets[:, None] * row[None, :]
    tips = entries + depth * u[None, :]
    return tips, entries


# -- axes ------------------------------------------------------------------


def test_axes_recover_the_insertion_direction_and_row():
    tips, entries = _array(0.0)

    u, r, _centre = array_axes(tips, entries)

    assert np.allclose(np.abs(u), [0.0, 0.0, 1.0], atol=1e-6)  # straight down in DV
    assert abs(float(u @ r)) < 1e-9  # row is perpendicular to the insertion
    assert np.allclose(np.abs(r), [1.0, 0.0, 0.0], atol=1e-6)  # 0 deg roll = along AP


# -- the lab's conventions -------------------------------------------------


def test_pitch_is_the_angle_away_from_vertical():
    assert pitch_deg(*_array(0.0, pitch_degrees=0.0)) == pytest.approx(0.0, abs=1e-6)
    assert pitch_deg(*_array(0.0, pitch_degrees=10.0)) == pytest.approx(10.0, abs=1e-6)
    assert pitch_deg(*_array(30.0, pitch_degrees=20.0)) == pytest.approx(20.0, abs=1e-6)


def test_positive_roll_puts_the_anterior_shank_more_lateral():
    """The lab's definition, checked on the actual coordinates.

    CCF AP increases posteriorly, so the most anterior shank is the one with the
    smallest AP. On a medial-of-midline array (ML 4000 < 5700), lateral is -ML.
    """
    tips, _entries = _array(45.0, ml=4000.0)
    anterior = tips[np.argmin(tips[:, 0])]
    posterior = tips[np.argmax(tips[:, 0])]

    assert anterior[1] < posterior[1]  # anterior is further from the midline (-ML side)


def test_the_same_roll_mirrors_on_the_other_hemisphere():
    """The lab uses one number for both sides, so CCF ML must flip with hemisphere."""
    left_tips, _ = _array(45.0, ml=4000.0)   # medial of midline
    right_tips, _ = _array(45.0, ml=7400.0)  # lateral of midline

    left_anterior = left_tips[np.argmin(left_tips[:, 0])]
    right_anterior = right_tips[np.argmin(right_tips[:, 0])]

    assert left_anterior[1] < left_tips[:, 1].mean()    # anterior toward -ML
    assert right_anterior[1] > right_tips[:, 1].mean()  # anterior toward +ML
    assert lateral_sign(left_tips) == -1.0
    assert lateral_sign(right_tips) == 1.0


def test_row_direction_inverts_roll_deg():
    tips, entries = _array(37.0)
    u, r, _c = array_axes(tips, entries)

    rebuilt = row_direction(roll_deg(tips, entries), u, lateral_sign(tips))

    assert np.allclose(np.abs(rebuilt @ r), 1.0, atol=1e-6)  # same line, either sense


def test_row_positions_are_evenly_spaced_at_the_pitch():
    tips, entries = _array(30.0)

    positions = np.sort(shank_row_positions(tips, entries))

    assert np.allclose(np.diff(positions), PITCH, atol=1e-6)


def test_axes_reject_a_malformed_array():
    with pytest.raises(ValueError, match="n_shanks >= 2"):
        array_axes(np.zeros((1, 3)), np.zeros((1, 3)))


def test_coincident_shanks_still_give_a_perpendicular_row():
    """Degenerate input must not produce a row axis parallel to the insertion."""
    tips = np.tile([5000.0, 4000.0, 5000.0], (4, 1))
    entries = np.tile([5000.0, 4000.0, 1000.0], (4, 1))

    u, r, _c = array_axes(tips, entries)

    assert abs(float(u @ r)) < 1e-9
    assert np.isclose(np.linalg.norm(r), 1.0)


# -- roll ------------------------------------------------------------------


@pytest.mark.parametrize("truth", [0.0, 15.0, 30.0, 45.0, -25.0, 80.0])
def test_roll_is_recovered_from_the_geometry(truth):
    tips, entries = _array(truth)

    assert roll_deg(tips, entries) == pytest.approx(truth, abs=1e-6)


def test_roll_is_folded_so_a_flipped_row_is_not_a_different_array():
    """A row is an undirected line: +100 deg and -80 deg are the same physical array."""
    tips, entries = _array(100.0)

    assert roll_deg(tips, entries) == pytest.approx(-80.0, abs=1e-6)


def test_rolling_by_a_known_angle_moves_the_roll_by_that_angle():
    tips, entries = _array(10.0)

    rolled_tips, rolled_entries = rolled_array(tips, entries, 35.0)

    assert roll_deg(rolled_tips, rolled_entries) == pytest.approx(45.0, abs=1e-6)


def test_roll_preserves_spacing_and_the_array_centre():
    tips, entries = _array(0.0)

    rolled_tips, rolled_entries = rolled_array(tips, entries, 37.0)

    assert np.allclose(np.sort(shank_row_positions(rolled_tips, rolled_entries)),
                       np.sort(shank_row_positions(tips, entries)), atol=1e-6)
    assert np.allclose(rolled_tips.mean(0), tips.mean(0), atol=1e-6)
    assert np.allclose(rolled_entries.mean(0), entries.mean(0), atol=1e-6)


def test_roll_does_not_move_any_shank_along_the_track():
    """The geometric fact that rules out fitting roll from per-shank depth offsets."""
    tips, entries = _array(0.0)
    u, _r, _c = array_axes(tips, entries)

    rolled_tips, rolled_entries = rolled_array(tips, entries, 45.0)

    assert np.allclose(rolled_tips @ u, tips @ u, atol=1e-6)
    assert np.allclose(rolled_entries @ u, entries @ u, atol=1e-6)
    # Track length per shank is untouched too.
    assert np.allclose(np.linalg.norm(rolled_tips - rolled_entries, axis=1),
                       np.linalg.norm(tips - entries, axis=1), atol=1e-6)


# -- along-track offset ----------------------------------------------------


def test_shift_moves_the_array_deeper_without_stretching_it():
    tips, entries = _array(20.0)

    new_tips, new_entries = shift_along_track(tips, entries, 300.0)

    assert np.allclose(new_tips[:, 2], tips[:, 2] + 300.0)  # deeper in DV
    assert np.allclose(new_entries[:, 2], entries[:, 2] + 300.0)
    assert np.allclose(np.linalg.norm(new_tips - new_entries, axis=1),
                       np.linalg.norm(tips - entries, axis=1))


def test_shift_leaves_the_roll_alone():
    tips, entries = _array(33.0)

    new_tips, new_entries = shift_along_track(tips, entries, -250.0)

    assert roll_deg(new_tips, new_entries) == pytest.approx(33.0, abs=1e-6)


# -- rigid probe adjustment ------------------------------------------------


def test_tilt_changes_pitch_and_pivots_at_the_entry():
    """A manipulator angle change swings the tips; it cannot drag the entry points."""
    tips, entries = _array(0.0, pitch_degrees=0.0)

    new_tips, new_entries = tilted_array(tips, entries, 8.0)

    assert pitch_deg(new_tips, new_entries) == pytest.approx(8.0, abs=1e-6)
    assert np.allclose(new_entries.mean(0), entries.mean(0), atol=1e-6)
    assert not np.allclose(new_tips, tips)


def test_transformed_array_composes_all_three():
    # Start vertical: pitch is the unsigned angle from vertical, so it only adds
    # arithmetically when the tilt is applied to an already-vertical probe. From a
    # tipped start, tilting about the row axis rotates in a different plane and the
    # angles compose trigonometrically, not by addition.
    tips, entries = _array(10.0, pitch_degrees=0.0)

    out_t, out_e = transformed_array(tips, entries, offset_um=200.0, roll_deg=6.0,
                                     tilt_deg=4.0)

    assert pitch_deg(out_t, out_e) == pytest.approx(4.0, abs=0.01)
    assert roll_deg(out_t, out_e) == pytest.approx(16.0, abs=0.5)
    # Track lengths are untouched: this moves the probe, it does not stretch it.
    assert np.allclose(np.linalg.norm(out_t - out_e, axis=1),
                       np.linalg.norm(tips - entries, axis=1), atol=1e-6)


def test_a_zero_transform_is_the_identity():
    tips, entries = _array(25.0, pitch_degrees=12.0)

    out_t, out_e = transformed_array(tips, entries)

    assert np.allclose(out_t, tips)
    assert np.allclose(out_e, entries)


class _SlabAtlas:
    """Fake anatomy: three thick DV slabs, so boundaries sit at known depths."""

    structures: ClassVar[dict] = {
        "A": {"rgb_triplet": [1, 1, 1]},
        "B": {"rgb_triplet": [2, 2, 2]},
        "C": {"rgb_triplet": [3, 3, 3]},
    }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        if dv < 1000 or dv > 5000:
            return "Outside atlas"
        if dv < 2500:
            return "A"
        return "B" if dv < 3800 else "C"


def _profile_for(tips, entries, shank: int = 0, *, offset_um: float = 0.0):
    """A step profile peaking where this placement's atlas boundaries actually fall."""
    from atlastrack.ephys.autolandmarks import candidate_boundaries
    from atlastrack.ephys.regions import region_bands, regions_along_track

    tip, entry = np.asarray(tips)[shank], np.asarray(entries)[shank]
    track = float(np.linalg.norm(tip - entry))
    depths = np.arange(-200.0, track + 220.0, 20.0)
    bands = region_bands(regions_along_track(_SlabAtlas(), tip, entry, depths), depths)
    grid = np.arange(0.0, track, 10.0)
    score = np.zeros_like(grid)
    for below, _label in candidate_boundaries(bands):
        from_tip = track - below + offset_um
        score += np.exp(-0.5 * ((grid - from_tip) / 60.0) ** 2)
    return {shank: (grid, score)}


def test_scoring_peaks_at_the_placement_the_features_came_from():
    tips, entries = _array(0.0, depth=3500.0)
    profiles = _profile_for(tips, entries)

    truth, n = score_trajectory(tips, entries, profiles, _SlabAtlas())
    moved_t, moved_e = shift_along_track(tips, entries, 400.0)
    wrong, _n = score_trajectory(moved_t, moved_e, profiles, _SlabAtlas())

    assert n > 0
    assert truth > wrong


def test_refine_recovers_a_known_offset():
    """Synthesise features displaced by a known amount, and see if the search finds it.

    ``_profile_for(offset_um=200)`` puts every feature 200 µm further from the tip
    than this placement predicts. The fit returns the offset to pass to
    ``shift_along_track`` that best explains them - so a recovered +200 means the
    search found exactly the displacement that was put in.
    """
    tips, entries = _array(0.0, depth=3500.0)
    profiles = _profile_for(tips, entries, offset_um=200.0)

    fit = refine_trajectory(tips, entries, profiles, _SlabAtlas(),
                            offsets_um=np.arange(-400.0, 401.0, 50.0),
                            rolls_deg=[0.0], tilts_deg=[0.0])

    assert fit.offset_um == pytest.approx(200.0, abs=60.0)
    assert fit.n_boundaries > 0
    assert fit.mean_score > 0.5


def test_refine_finds_nothing_to_correct_when_there_is_nothing_wrong():
    """The control: features generated from the placement itself must fit at zero."""
    tips, entries = _array(0.0, depth=3500.0)
    profiles = _profile_for(tips, entries, offset_um=0.0)

    fit = refine_trajectory(tips, entries, profiles, _SlabAtlas(),
                            offsets_um=np.arange(-400.0, 401.0, 50.0),
                            rolls_deg=[0.0], tilts_deg=[0.0])

    assert fit.offset_um == pytest.approx(0.0, abs=60.0)


def test_refine_returns_an_empty_fit_when_there_is_nothing_to_score():
    tips, entries = _array(0.0)

    fit = refine_trajectory(tips, entries, {}, _SlabAtlas(),
                            offsets_um=[0.0], rolls_deg=[0.0], tilts_deg=[0.0])

    assert fit.n_boundaries == 0
    assert fit.mean_score == 0.0


# -- the contrast measure --------------------------------------------------


def test_contrast_finds_a_step_the_adjacent_sample_measure_would_miss():
    """A 300 um ramp: big overall, unremarkable between neighbours."""
    depths = np.arange(0.0, 2000.0, 15.0)
    values = np.clip((depths - 850.0) / 300.0, 0.0, 1.0) * 10.0

    d = boundary_contrast(depths, values, 1000.0, window_um=150.0)
    adjacent = np.abs(np.diff(values)).max()

    assert d > 3.0                 # the level step is unmistakable...
    assert adjacent < 0.6          # ...while no two neighbours differ by much


def test_contrast_is_near_zero_where_nothing_happens():
    depths = np.arange(0.0, 2000.0, 15.0)
    rng = np.random.default_rng(0)
    flat = rng.normal(0.0, 1.0, depths.size)

    assert boundary_contrast(depths, flat, 1000.0) < 1.0


def test_contrast_needs_samples_on_both_sides():
    depths = np.arange(0.0, 500.0, 15.0)
    values = np.ones_like(depths)

    assert np.isnan(boundary_contrast(depths, values, 0.0))     # nothing above
    assert np.isnan(boundary_contrast(depths, values, 490.0))   # nothing below


def test_contrast_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="must match"):
        boundary_contrast(np.arange(10.0), np.arange(5.0), 5.0)


def test_null_distribution_puts_a_real_boundary_in_its_tail():
    depths = np.arange(0.0, 3000.0, 15.0)
    rng = np.random.default_rng(1)
    values = np.where(depths < 1500.0, 0.0, 6.0) + rng.normal(0.0, 1.0, depths.size)

    null = contrast_null(depths, values)
    d = boundary_contrast(depths, values, 1500.0)

    assert null.size > 20
    assert d > np.percentile(null, 95)


def test_null_is_empty_for_a_track_shorter_than_the_window():
    assert contrast_null(np.arange(0.0, 100.0, 15.0), np.zeros(7)).size == 0
