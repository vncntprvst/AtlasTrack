"""Choosing which excerpts of a recording to build alignment features from.

Alignment does not need whole recordings - a handful of short windows carries the
depth structure, and reading 52 GB of raw data per recording off a spinning disk to
compute a power spectrum is waste. What matters is *which* windows:

* **Active periods discriminate better.** A quiet stretch gives a flat firing-rate
  profile and a featureless LFP map, and there is nothing to align to.
* **Artifact periods actively mislead.** The lick artifacts flagged throughout this
  dataset are large transients that appear on *many channels at once*. Averaged into
  a power map they raise every channel together, flattening exactly the
  depth contrast the alignment depends on, and they look like signal.

The two are easy to confuse - both are "high amplitude" - so they are separated by
how *shared across depth* the excursion is: real activity is local to some channels,
an artifact is not. That is what :func:`artifact_score` measures.

Pure numpy, no SpikeInterface, so it is testable without ephys extras installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# An excursion counts when a channel exceeds this many robust SDs from its own
# median. Deliberately generous: we are looking for gross transients, not spikes.
_EXCURSION_K = 6.0
# Fraction of channels that must excurse *simultaneously* for a sample to look like
# an artifact rather than neural activity.
_CHANNEL_FRACTION = 0.5
# Reject a window when this fraction of its samples are artifact-like.
_ARTIFACT_TOLERANCE = 0.02

_MAD_TO_SD = 1.4826


def _robust_sd(traces: np.ndarray) -> np.ndarray:
    """Per-channel SD estimated from the MAD, so spikes don't inflate it."""
    med = np.median(traces, axis=0)
    mad = np.median(np.abs(traces - med), axis=0)
    sd = _MAD_TO_SD * mad
    # A dead (flat) channel has mad == 0; give it a positive SD so it never counts
    # as permanently excursing.
    sd[sd <= 0] = np.inf
    return sd


def artifact_score(traces: np.ndarray, *, k: float = _EXCURSION_K,
                   channel_fraction: float = _CHANNEL_FRACTION) -> float:
    """Fraction of samples where many channels excurse together.

    ``traces`` is ``(n_samples, n_channels)``. Returns 0.0 for clean data and rises
    towards 1.0 for a window dominated by cross-channel transients. Neural activity
    is local in depth and so scores near 0 however large it is.
    """
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2 or traces.shape[0] == 0 or traces.shape[1] == 0:
        return 0.0
    med = np.median(traces, axis=0)
    sd = _robust_sd(traces)
    excursion = np.abs(traces - med) > (k * sd)
    return float((excursion.mean(axis=1) >= channel_fraction).mean())


def activity_score(traces: np.ndarray) -> float:
    """How much *depth-varying* signal a window has.

    The spread of per-channel RMS across channels, not the mean: a window where
    every channel is equally loud carries no depth information, however loud it is.
    """
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2 or traces.shape[0] == 0 or traces.shape[1] == 0:
        return 0.0
    rms = np.sqrt(np.mean((traces - traces.mean(axis=0)) ** 2, axis=0))
    return float(np.std(rms))


def common_median_reference(traces: np.ndarray) -> np.ndarray:
    """Subtract the across-channel median from every sample.

    The standard first defence against shared transients: whatever appears on all
    channels at once (licking, movement, reference noise) is removed, and anything
    local to a few channels survives. Applied per shank by the caller - referencing
    across shanks that sit in different tissue would mix unrelated signals.
    """
    traces = np.asarray(traces, dtype=float)
    return traces - np.median(traces, axis=1, keepdims=True)


def candidate_windows(
    n_samples: int, fs: float, *, window_s: float = 10.0, n_windows: int = 6
) -> list[tuple[float, float]]:
    """Evenly spread candidate excerpts, in seconds.

    Spread rather than contiguous: a recording drifts, and sampling across it
    averages that out instead of characterising one moment.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    duration = n_samples / float(fs)
    if duration <= 0 or n_windows < 1:
        return []
    window = min(float(window_s), duration)
    if n_windows == 1:
        start = max(0.0, (duration - window) / 2.0)
        return [(start, start + window)]
    # Centre each window in its own equal share of the recording.
    edges = np.linspace(0.0, duration, n_windows + 1)
    out: list[tuple[float, float]] = []
    for i in range(n_windows):
        centre = 0.5 * (edges[i] + edges[i + 1])
        start = float(np.clip(centre - window / 2.0, 0.0, duration - window))
        out.append((start, start + window))
    return out


@dataclass(frozen=True)
class EpochVerdict:
    """Whether one candidate window is fit to use, and why."""

    t_start_s: float
    t_end_s: float
    kept: bool
    artifact: float
    activity: float
    reject_reason: str | None = None


def screen_window(
    traces: np.ndarray,
    t_start_s: float,
    t_end_s: float,
    *,
    artifact_tolerance: float = _ARTIFACT_TOLERANCE,
) -> EpochVerdict:
    """Score one window and decide whether to keep it."""
    artifact = artifact_score(traces)
    activity = activity_score(traces)
    if artifact > artifact_tolerance:
        return EpochVerdict(
            t_start_s, t_end_s, False, artifact, activity,
            f"cross-channel artifact in {100 * artifact:.1f}% of samples",
        )
    if activity <= 0.0:
        return EpochVerdict(
            t_start_s, t_end_s, False, artifact, activity,
            "no depth-varying signal (flat across channels)",
        )
    return EpochVerdict(t_start_s, t_end_s, True, artifact, activity)


def rank_epochs(verdicts: list[EpochVerdict], *, keep: int | None = None
                ) -> list[EpochVerdict]:
    """Order kept windows by how informative they are, most first.

    ``keep`` trims to the best N **without discarding the rejected ones** - they are
    returned after, so the UI can show what was thrown away and why.
    """
    kept = sorted((v for v in verdicts if v.kept), key=lambda v: -v.activity)
    dropped = [v for v in verdicts if not v.kept]
    if keep is not None:
        dropped = [
            EpochVerdict(v.t_start_s, v.t_end_s, False, v.artifact, v.activity,
                         v.reject_reason or "not among the most informative windows")
            for v in kept[keep:]
        ] + dropped
        kept = kept[:keep]
    return kept + dropped
