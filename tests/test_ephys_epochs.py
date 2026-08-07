"""Excerpt selection: keep informative windows, reject artifact ones.

The discrimination that matters is artifact vs activity. Both are large-amplitude;
they differ in whether the excursion is shared across depth.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.epochs import (
    EpochVerdict,
    activity_score,
    artifact_score,
    candidate_windows,
    common_median_reference,
    rank_epochs,
    screen_window,
)

FS = 2500.0
N_CH = 64


def _noise(n_samples: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 10.0, size=(n_samples, N_CH))


def test_clean_noise_scores_no_artifact() -> None:
    assert artifact_score(_noise()) == pytest.approx(0.0, abs=1e-3)


def test_a_cross_channel_transient_is_flagged() -> None:
    """The lick-artifact signature: every channel jumps at the same instant."""
    traces = _noise()
    traces[1000:1100, :] += 500.0

    assert artifact_score(traces) > 0.01


def test_large_activity_on_some_channels_is_not_an_artifact() -> None:
    """A strongly responding population must not be mistaken for an artifact.

    Same amplitude as the transient above, but confined to a quarter of the
    channels - which is what makes it signal.
    """
    traces = _noise()
    traces[1000:1100, : N_CH // 4] += 500.0

    assert artifact_score(traces) == pytest.approx(0.0, abs=1e-3)


def test_activity_score_rewards_depth_contrast_not_loudness() -> None:
    """A uniformly loud window carries no depth information."""
    rng = np.random.default_rng(1)
    uniform = rng.normal(0.0, 50.0, size=(4000, N_CH))
    varied = rng.normal(0.0, 1.0, size=(4000, N_CH)) * np.linspace(1, 100, N_CH)

    assert activity_score(varied) > activity_score(uniform)


def test_common_median_reference_removes_the_shared_transient() -> None:
    traces = _noise()
    traces[1000:1100, :] += 500.0

    cleaned = common_median_reference(traces)

    assert artifact_score(cleaned) < artifact_score(traces)


def test_common_median_reference_keeps_local_signal() -> None:
    traces = _noise()
    traces[1000:1100, :4] += 500.0

    cleaned = common_median_reference(traces)

    assert cleaned[1000:1100, :4].mean() > 100.0


def test_dead_channels_do_not_look_like_permanent_excursions() -> None:
    traces = _noise()
    traces[:, 0] = 0.0

    assert artifact_score(traces) == pytest.approx(0.0, abs=1e-3)


def test_candidate_windows_are_spread_across_the_recording() -> None:
    windows = candidate_windows(int(600 * FS), FS, window_s=10.0, n_windows=6)

    assert len(windows) == 6
    starts = [w[0] for w in windows]
    assert starts == sorted(starts)
    assert starts[0] < 100.0 and starts[-1] > 400.0
    assert all(pytest.approx(10.0) == (b - a) for a, b in windows)


def test_a_window_longer_than_the_recording_is_clipped() -> None:
    windows = candidate_windows(int(3 * FS), FS, window_s=10.0, n_windows=1)

    assert windows == [(0.0, 3.0)]


def test_candidate_windows_needs_a_positive_rate() -> None:
    with pytest.raises(ValueError, match="fs must be positive"):
        candidate_windows(1000, 0.0)


def test_screen_window_keeps_a_clean_window() -> None:
    verdict = screen_window(_noise(), 0.0, 2.0)

    assert verdict.kept is True
    assert verdict.reject_reason is None


def test_screen_window_rejects_and_says_why() -> None:
    traces = _noise()
    traces[::10, :] += 500.0  # frequent cross-channel transients

    verdict = screen_window(traces, 0.0, 2.0)

    assert verdict.kept is False
    assert "artifact" in verdict.reject_reason


def test_rank_epochs_puts_the_most_informative_first() -> None:
    verdicts = [
        EpochVerdict(0.0, 1.0, True, 0.0, activity=1.0),
        EpochVerdict(1.0, 2.0, True, 0.0, activity=9.0),
        EpochVerdict(2.0, 3.0, False, 0.5, activity=5.0, reject_reason="artifact"),
    ]

    ranked = rank_epochs(verdicts)

    assert [v.activity for v in ranked[:2]] == [9.0, 1.0]
    assert ranked[-1].kept is False


def test_rank_epochs_reports_what_trimming_dropped() -> None:
    """Trimming to the best N must not silently lose windows."""
    verdicts = [
        EpochVerdict(0.0, 1.0, True, 0.0, activity=1.0),
        EpochVerdict(1.0, 2.0, True, 0.0, activity=9.0),
    ]

    ranked = rank_epochs(verdicts, keep=1)

    assert len(ranked) == 2
    assert ranked[0].kept is True and ranked[0].activity == 9.0
    assert ranked[1].kept is False
    assert "most informative" in ranked[1].reject_reason
