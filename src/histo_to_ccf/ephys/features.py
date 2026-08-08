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


# The bands IBL's alignment GUI plots on the probe view. Splitting the spectrum this
# way separates effects that a single broadband number merges: slow-wave power and
# high-frequency power change at different anatomical boundaries.
LFP_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 4.0),
    (4.0, 10.0),
    (10.0, 30.0),
    (30.0, 80.0),
    (80.0, 200.0),
)


def lfp_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    *,
    bands: "tuple[tuple[float, float], ...]" = LFP_BANDS_HZ,
) -> np.ndarray:
    """Mean power per band for each channel: ``(n_channels, n_bands)``.

    Bands with no frequency bin in range come back as NaN rather than 0, so a band
    that simply was not sampled is distinguishable from one with no power.
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if psd.ndim != 2 or psd.shape[1] != freqs.size:
        raise ValueError(
            f"psd {psd.shape} does not match {freqs.size} frequencies; expected "
            "(n_channels, n_freq)"
        )
    out = np.full((psd.shape[0], len(bands)), np.nan, dtype=float)
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            out[:, i] = psd[:, mask].mean(axis=1)
    return out


def normalise_band_power(band_power: np.ndarray) -> np.ndarray:
    """Strip one recording's overall power level, keeping its depth structure.

    **Band power is not comparable across recordings, and stitching it raw draws a
    boundary that is not there.** Measured on LO_06 2026-02-09, four shanks: the
    median 10-30 Hz power differs by **0.69-0.93 log10 (5.0-8.5x)** between the three
    recordings on the *same* penetration - a gain / reference / noise-floor
    difference between acquisitions, not anatomy. Left alone it puts a large step at
    every bank junction, exactly where a user would read a region boundary and place
    a landmark.

    Returns log10 power **relative to that recording's own median channel**, one
    median per band. That deliberately discards any genuine overall difference in
    power between the depths each recording covers - which is unmeasurable anyway
    while the instrumental offset is 5-8x larger - and keeps the local, depth-to-depth
    transitions, which are what an alignment is read from.

    Apply per recording *and* per shank, before putting anything on a shared axis.
    """
    a = np.log10(np.maximum(np.asarray(band_power, dtype=float), 1e-12))
    if a.ndim != 2:
        raise ValueError("band_power must be (n_channels, n_bands)")
    if a.shape[0] == 0:
        return a
    return a - np.nanmedian(a, axis=0, keepdims=True)


def depth_profiles(
    depths_um: np.ndarray,
    amplitudes: np.ndarray,
    duration_s: float,
    *,
    bin_um: float = 10.0,
    min_spikes: int = 50,
    depth_range: "tuple[float, float] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Firing rate and mean amplitude against depth.

    Returns ``(bin_centres_um, rate_hz, mean_amplitude)``. Bins holding fewer than
    ``min_spikes`` come back as NaN, following IBL: a bin with a handful of spikes
    produces a wild rate and a meaningless mean amplitude, and plotting it as a real
    value invites aligning to noise. NaN leaves a visible gap instead.
    """
    depths = np.asarray(depths_um, dtype=float)
    amps = np.asarray(amplitudes, dtype=float)
    if depths.shape != amps.shape:
        raise ValueError(
            f"depths {depths.shape} and amplitudes {amps.shape} must match"
        )
    if bin_um <= 0:
        raise ValueError("bin_um must be positive")
    if depths.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    lo, hi = depth_range if depth_range is not None else (depths.min(), depths.max())
    if hi <= lo:
        hi = lo + bin_um
    edges = np.arange(lo, hi + bin_um, bin_um, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])

    counts, _ = np.histogram(depths, bins=edges)
    amp_sum, _ = np.histogram(depths, bins=edges, weights=amps)

    enough = counts >= int(min_spikes)
    rate = np.full(centres.shape, np.nan)
    mean_amp = np.full(centres.shape, np.nan)
    if duration_s > 0:
        rate[enough] = counts[enough] / float(duration_s)
    mean_amp[enough] = amp_sum[enough] / counts[enough]
    return centres, rate, mean_amp


def raster_points(
    times_s: np.ndarray,
    depths_um: np.ndarray,
    amplitudes: np.ndarray,
    *,
    max_points: int = 200_000,
    seed: int = 0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Thin a spike raster down to something a plot can draw.

    A recording here holds ~1e6 spikes (960,899 in the LO_03 export) and drawing
    them all makes the panel unusable. Thinning is a **uniform random subsample**,
    not a time or depth crop, so the visible density stays proportional to the real
    density everywhere - which is the only property the raster is read for.
    """
    times = np.asarray(times_s, dtype=float)
    depths = np.asarray(depths_um, dtype=float)
    amps = np.asarray(amplitudes, dtype=float)
    if not (times.shape == depths.shape == amps.shape):
        raise ValueError("times, depths and amplitudes must have the same shape")
    if times.size <= max_points:
        return times, depths, amps
    rng = np.random.default_rng(seed)
    pick = rng.choice(times.size, size=int(max_points), replace=False)
    pick.sort()
    return times[pick], depths[pick], amps[pick]
