"""LFP feature computation for ephys alignment.

The classic IBL alignment view is a depth x frequency LFP power map: power
transitions line up with anatomical boundaries, which is what the user matches
to the histology region profile. These helpers are pure numpy/scipy so they can
be tested with synthetic traces, independent of SpikeInterface.
"""
from __future__ import annotations

import numpy as np


def lfp_psd(
    traces: np.ndarray,
    fs: float,
    *,
    fmin: float = 0.0,
    fmax: float = 300.0,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel power spectral density via Welch's method.

    Parameters
    ----------
    traces
        ``(n_samples, n_channels)`` LFP samples.
    fs
        Sampling rate (Hz).
    fmin, fmax
        Frequency band to keep (Hz).
    nperseg
        Welch segment length; defaults to ~1 s (capped at the trace length).

    Returns
    -------
    freqs
        ``(n_freq,)`` frequency bins within ``[fmin, fmax]``.
    psd
        ``(n_channels, n_freq)`` power for each channel.
    """
    from scipy.signal import welch

    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces must be (n_samples, n_channels)")
    n_samples = traces.shape[0]
    if nperseg is None:
        nperseg = int(min(n_samples, max(256, round(fs))))
    nperseg = max(16, min(nperseg, n_samples))

    freqs, pxx = welch(traces, fs=fs, axis=0, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], pxx[mask].T  # (n_channels, n_freq)


def power_image(psd: np.ndarray, *, log: bool = True, per_freq: bool = False) -> np.ndarray:
    """Normalise a ``(n_channels, n_freq)`` PSD to a uint8 image for display.

    With ``log`` the power is log10-compressed first (LFP power spans orders of
    magnitude). By default the result is scaled to span 0-255 across the whole
    map. With ``per_freq`` each frequency column is normalised independently
    (min-max down the depth axis), which removes the strong 1/f gradient across
    frequencies and makes depth-dependent power changes - the features that line
    up with region boundaries - far more visible.
    """
    a = np.asarray(psd, dtype=float)
    if log:
        a = np.log10(a + 1e-12)
    if per_freq:
        lo = np.nanmin(a, axis=0, keepdims=True)
        hi = np.nanmax(a, axis=0, keepdims=True)
    else:
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
    rng = np.where(np.asarray(hi) <= np.asarray(lo), 1.0, np.asarray(hi) - np.asarray(lo))
    return (np.clip((a - lo) / rng, 0.0, 1.0) * 255.0).astype(np.uint8)
