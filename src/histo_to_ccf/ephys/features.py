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

    Rows may be **NaN**: a depth-binned map covering several recordings has bins no
    recording reached. Those come back as 0 and are excluded from the min/max, so an
    uncovered stretch neither takes part in the scaling nor turns into whatever
    ``astype(uint8)`` makes of a NaN. Use :func:`covered_rows` to draw them as empty
    rather than as zero power.
    """
    a = np.asarray(psd, dtype=float)
    if a.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    if log:
        with np.errstate(invalid="ignore"):
            a = np.log10(a + 1e-12)
    good = np.isfinite(a)
    if not good.any():
        return np.zeros(a.shape, dtype=np.uint8)
    filled = np.where(good, a, np.nan)
    with np.errstate(invalid="ignore", all="ignore"):
        if per_freq:
            lo = np.nanmin(filled, axis=0, keepdims=True)
            hi = np.nanmax(filled, axis=0, keepdims=True)
            # A frequency column with no finite value at all: nanmin/nanmax are NaN,
            # which would poison the whole column. Neutralise it instead.
            lo = np.where(np.isfinite(lo), lo, 0.0)
            hi = np.where(np.isfinite(hi), hi, 1.0)
        else:
            lo = float(np.nanmin(filled))
            hi = float(np.nanmax(filled))
    rng = np.where(np.asarray(hi) <= np.asarray(lo), 1.0, np.asarray(hi) - np.asarray(lo))
    scaled = np.zeros(a.shape, dtype=float)
    np.divide(a - lo, rng, out=scaled, where=good)
    return (np.clip(np.where(good, scaled, 0.0), 0.0, 1.0) * 255.0).astype(np.uint8)


def covered_rows(psd: np.ndarray) -> np.ndarray:
    """Which rows of a depth x frequency map hold data at all: ``(n_rows,)`` bool.

    A binned multi-recording map has rows nothing reached. Zero power and no
    measurement look identical once the map is an image, so the display needs to be
    told them apart - a dark band that means "no recording covers this depth" must
    not be readable as "the LFP is quiet here".
    """
    a = np.asarray(psd, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return np.zeros(0 if a.ndim != 2 else a.shape[0], dtype=bool)
    return np.isfinite(a).any(axis=1)


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
    bands: tuple[tuple[float, float], ...] = LFP_BANDS_HZ,
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


def boundary_contrast(depths_um, values, boundary_um: float, *, window_um: float = 150.0,
                      min_samples: int = 3) -> float:
    """How big a **step in level** a feature makes across one depth, as a Cohen's d.

    Mean of the ``window_um`` above minus the mean below, over the pooled within-window
    spread. NaN when either side is too thin to judge.

    **This measures the right thing, and an obvious alternative does not.** Comparing
    the difference between *adjacent* depth samples is a high-pass measure: an
    anatomical transition is spread over 100-300 µm, so its sample-to-sample steps look
    ordinary even when the total change is 100x. Measured on LO_07_005, this statistic
    finds the cerebellum→brainstem crossing (``chpl|V4|MV``) at d = 3.8-4.9 and
    cerebellar white matter → cortex (``arb|CUL4,5``) at d = 4.5, where the adjacent-
    sample version reported nothing at all.
    """
    depths = np.asarray(depths_um, dtype=float)
    vals = np.asarray(values, dtype=float)
    if depths.shape != vals.shape:
        raise ValueError(f"depths {depths.shape} and values {vals.shape} must match")
    at = float(boundary_um)
    above = vals[(depths >= at - window_um) & (depths < at)]
    below = vals[(depths >= at) & (depths < at + window_um)]
    if above.size < min_samples or below.size < min_samples:
        return float("nan")
    spread = np.sqrt(0.5 * (np.nanvar(above) + np.nanvar(below)))
    if not np.isfinite(spread) or spread <= 0:
        return float("nan")
    return float(abs(np.nanmean(above) - np.nanmean(below)) / spread)


def contrast_null(depths_um, values, *, window_um: float = 150.0, step_um: float = 25.0
                  ) -> np.ndarray:
    """:func:`boundary_contrast` at every depth, i.e. the null distribution.

    Brain tissue varies continuously, so a contrast is only interesting relative to
    what the same feature does at an arbitrary depth. Comparing a boundary's d against
    this is what separates "a real step" from "this feature is just rough".
    """
    depths = np.asarray(depths_um, dtype=float)
    if depths.size == 0 or step_um <= 0:
        return np.empty(0)
    grid = np.arange(depths.min() + window_um, depths.max() - window_um, float(step_um))
    out = np.array([boundary_contrast(depths, values, g, window_um=window_um) for g in grid])
    return out[np.isfinite(out)]


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
    depth_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
