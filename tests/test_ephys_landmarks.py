"""The IBL landmark alignment model: the map, the tails, the guard, the history."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.landmarks import (
    AlignmentHistory,
    LandmarkCrossingError,
    Landmarks,
    adjust_extremes_linear,
    adjust_extremes_uniform,
    feature2track,
    segment_scales,
    track2feature,
)

TOP, BOTTOM = 0.0, 5000.0


def _lm(*pairs: tuple[float, float]) -> Landmarks:
    out = Landmarks.identity(TOP, BOTTOM)
    for f, t in pairs:
        out = out.added(f, t)
    return out


# -- the map ---------------------------------------------------------------


def test_no_landmarks_is_the_identity():
    lm = Landmarks.identity(TOP, BOTTOM)
    depths = np.array([-500.0, 0.0, 1234.0, 5000.0, 6000.0])
    assert np.allclose(lm.to_track(depths), depths)
    assert lm.n_user == 0


def test_one_landmark_is_a_pure_shift_everywhere():
    """IBL's uniform extremes make a single landmark an offset, not a stretch."""
    lm = _lm((1000.0, 1200.0))
    depths = np.array([-500.0, 0.0, 1000.0, 3000.0, 5000.0, 9000.0])
    assert np.allclose(lm.to_track(depths), depths + 200.0)


def test_two_landmarks_scale_only_between_them():
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    # Interior segment carries the scale...
    assert lm.to_track(2000.0) == pytest.approx(2300.0)
    # ...and the tails are pure translation (slope 1) by the landmark's own offset.
    assert lm.to_track(0.0) == pytest.approx(200.0)
    assert lm.to_track(1000.0 - 500.0) == pytest.approx(1200.0 - 500.0)
    assert lm.to_track(3000.0 + 500.0) == pytest.approx(3400.0 + 500.0)


def test_uniform_extremes_give_the_outer_segments_slope_one():
    lm = _lm((800.0, 900.0), (2400.0, 2900.0))
    feature, track = lm.fit("uniform")
    slopes = np.diff(track) / np.diff(feature)
    assert slopes[0] == pytest.approx(1.0)
    assert slopes[-1] == pytest.approx(1.0)
    assert slopes[1] != pytest.approx(1.0)  # the pinned interval really is scaled


def test_track_to_feature_inverts_feature_to_track():
    lm = _lm((900.0, 1100.0), (2600.0, 3000.0), (4100.0, 4400.0))
    depths = np.array([100.0, 1500.0, 3300.0, 4800.0])
    back = lm.to_feature(lm.to_track(depths))
    assert np.allclose(back, depths)


def test_scalar_input_returns_a_scalar():
    lm = _lm((1000.0, 1200.0))
    assert isinstance(lm.to_track(500.0), float)


def test_bare_functions_agree_with_the_dataclass():
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    feature, track = lm.fit("uniform")
    assert feature2track(2000.0, feature, track) == pytest.approx(2300.0)
    assert track2feature(2300.0, feature, track) == pytest.approx(2000.0)


# -- the tails -------------------------------------------------------------


def test_linear_extremes_continue_the_global_regression():
    """Landmarks on a line t = 1.1f + 50 make the *whole* map that line."""
    slope, intercept = 1.1, 50.0
    features = [800.0, 2000.0, 3600.0]
    lm = _lm(*[(f, slope * f + intercept) for f in features])
    for depth in (-1000.0, 0.0, 1500.0, 4800.0, 7000.0):
        assert lm.to_track(depth, "linear") == pytest.approx(slope * depth + intercept)


def test_linear_and_uniform_differ_in_the_tails_only():
    features = [800.0, 2000.0, 3600.0]
    lm = _lm(*[(f, 1.1 * f + 50.0) for f in features])
    # Between the outer landmarks the two modes are the same piecewise map.
    assert lm.to_track(2500.0, "linear") == pytest.approx(lm.to_track(2500.0, "uniform"))
    # Outside, uniform translates (slope 1) where linear keeps stretching (slope 1.1).
    assert lm.to_track(0.0, "uniform") == pytest.approx(1.1 * 800.0 + 50.0 - 800.0)
    assert lm.to_track(0.0, "linear") == pytest.approx(50.0)


def test_linear_falls_back_to_uniform_below_three_landmarks():
    """IBL's own fallback sends the tail points to zero; ours must not."""
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    assert lm.n_user < 3
    assert np.allclose(lm.fit("linear")[1], lm.fit("uniform")[1])
    # The telltale of the IBL bug would be a tail collapsing onto zero.
    assert lm.to_track(0.0, "linear") == pytest.approx(200.0)


def test_adjust_extremes_uniform_matches_the_ibl_formula():
    feature = np.array([0.0, 1000.0, 3000.0, 5000.0])
    track = np.array([0.0, 1200.0, 3400.0, 5000.0])
    _f, adjusted = adjust_extremes_uniform(feature, track)
    diff = np.diff(feature - track)
    assert adjusted[0] == pytest.approx(0.0 - diff[0])
    assert adjusted[-1] == pytest.approx(5000.0 + diff[-1])
    # Inputs untouched: the stored state must never be an adjusted one.
    assert track[0] == 0.0 and track[-1] == 5000.0


def test_adjust_extremes_functions_do_not_mutate_their_arguments():
    feature = np.array([0.0, 900.0, 2000.0, 3600.0, 5000.0])
    track = np.array([0.0, 1040.0, 2250.0, 4010.0, 5000.0])
    before = (feature.copy(), track.copy())
    adjust_extremes_linear(feature, track)
    assert np.array_equal(feature, before[0])
    assert np.array_equal(track, before[1])


def test_unknown_extremes_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown extremes mode"):
        _lm((1000.0, 1200.0)).fit("quadratic")


# -- the monotonic guard ---------------------------------------------------


def test_crossed_landmarks_are_refused_not_silently_repaired():
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    # Drag the deeper landmark up past the shallower one *in track space only*.
    with pytest.raises(LandmarkCrossingError, match="crossed"):
        lm.moved(1, track_um=900.0)


def test_duplicate_feature_depth_is_refused():
    lm = _lm((1000.0, 1200.0))
    with pytest.raises(LandmarkCrossingError, match="share the feature depth"):
        lm.added(1000.0, 2000.0)


def test_a_legal_move_is_accepted():
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    moved = lm.moved(0, feature_um=1500.0)
    assert moved.user_pairs() == [(1500.0, 1200.0), (3000.0, 3400.0)]
    assert lm.user_pairs()[0][0] == 1000.0  # original untouched


# -- state -----------------------------------------------------------------


def test_landmarks_stay_sorted_by_feature_depth():
    lm = _lm((3000.0, 3400.0), (1000.0, 1200.0), (2000.0, 2100.0))
    assert [f for f, _ in lm.user_pairs()] == [1000.0, 2000.0, 3000.0]


def test_remove_and_clear():
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    assert lm.removed(0).user_pairs() == [(3000.0, 3400.0)]
    assert lm.cleared().n_user == 0
    with pytest.raises(IndexError):
        lm.removed(5)


def test_identity_rejects_an_inverted_track_extent():
    with pytest.raises(ValueError, match="top < bottom"):
        Landmarks.identity(5000.0, 0.0)


def test_offset_and_shift():
    lm = _lm((1000.0, 1200.0), (3000.0, 3300.0))
    assert lm.offset_um() == pytest.approx(250.0)
    assert lm.shifted(-100.0).offset_um() == pytest.approx(150.0)


def test_shift_with_no_landmarks_still_moves_the_map():
    shifted = Landmarks.identity(TOP, BOTTOM).shifted(300.0)
    assert shifted.to_track(1000.0) == pytest.approx(1300.0)


def test_segment_scales_reports_the_stretch_per_interval():
    lm = _lm((1000.0, 1200.0), (3000.0, 3400.0))
    edges, scale = segment_scales(*lm.fit("uniform"))
    assert edges.size == 4
    assert scale == pytest.approx([1.0, 1.1, 1.0])


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        Landmarks(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))


# -- history ---------------------------------------------------------------


def test_history_undo_and_redo():
    first = Landmarks.identity(TOP, BOTTOM)
    history = AlignmentHistory(first)
    second = first.added(1000.0, 1200.0)
    history.push(second)
    third = second.added(3000.0, 3400.0)
    history.push(third)

    assert history.current().n_user == 2
    assert history.previous().n_user == 1
    assert history.previous().n_user == 0
    assert history.previous() is None  # stops at the oldest, never wraps
    assert history.next().n_user == 1
    assert history.next().n_user == 2
    assert history.next() is None


def test_history_is_bounded_at_ten_and_drops_the_oldest():
    lm = Landmarks.identity(TOP, BOTTOM)
    history = AlignmentHistory(lm)
    for i in range(1, 15):
        lm = lm.added(100.0 * i, 100.0 * i + 50.0)
        history.push(lm)
    assert history.n_states == 10
    for _ in range(9):
        assert history.previous() is not None
    assert history.previous() is None
    assert history.current().n_user == 5  # the oldest still kept, not the original


def test_pushing_after_an_undo_discards_the_redo_branch():
    lm = Landmarks.identity(TOP, BOTTOM)
    history = AlignmentHistory(lm)
    history.push(lm.added(1000.0, 1200.0))
    history.push(lm.added(1000.0, 1200.0).added(3000.0, 3400.0))
    history.previous()
    assert history.can_redo
    history.push(lm.added(2000.0, 2500.0))
    assert not history.can_redo
    assert history.current().user_pairs() == [(2000.0, 2500.0)]


def test_history_reset():
    lm = Landmarks.identity(TOP, BOTTOM)
    history = AlignmentHistory(lm)
    history.push(lm.added(1000.0, 1200.0))
    history.reset(lm)
    assert history.n_states == 1
    assert not history.can_undo and not history.can_redo


def test_history_depth_must_be_positive():
    with pytest.raises(ValueError, match="depth"):
        AlignmentHistory(Landmarks.identity(TOP, BOTTOM), depth=0)
