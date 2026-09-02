"""Depth-resolved spike features: band power, depth profiles, raster thinning."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.features import (
    LFP_BANDS_HZ,
    depth_profiles,
    lfp_band_power,
    raster_points,
)


def test_band_power_averages_within_each_band() -> None:
    freqs = np.arange(0.0, 210.0, 1.0)
    psd = np.ones((4, freqs.size))
    psd[:, (freqs >= 4) & (freqs < 10)] = 5.0

    bands = lfp_band_power(psd, freqs)

    assert bands.shape == (4, len(LFP_BANDS_HZ))
    assert bands[:, 1] == pytest.approx(5.0)   # 4-10 Hz
    assert bands[:, 0] == pytest.approx(1.0)   # 0-4 Hz


def test_band_with_no_frequency_bins_is_nan_not_zero() -> None:
    """An unsampled band must be distinguishable from a band with no power."""
    freqs = np.arange(0.0, 5.0, 1.0)
    psd = np.ones((2, freqs.size))

    bands = lfp_band_power(psd, freqs)

    assert not np.isnan(bands[:, 0]).any()
    assert np.isnan(bands[:, 4]).all()  # 80-200 Hz never sampled


def test_band_power_rejects_a_mismatched_psd() -> None:
    with pytest.raises(ValueError, match="does not match"):
        lfp_band_power(np.ones((4, 10)), np.arange(5.0))


def test_depth_profiles_recovers_a_known_rate() -> None:
    # 600 spikes in one 10 µm bin over 60 s -> 10 Hz.
    depths = np.full(600, 105.0)
    amps = np.full(600, 3.0)

    centres, rate, mean_amp = depth_profiles(
        depths, amps, duration_s=60.0, bin_um=10.0, depth_range=(100.0, 120.0)
    )

    hit = np.nanargmax(rate)
    assert centres[hit] == pytest.approx(105.0)
    assert rate[hit] == pytest.approx(10.0)
    assert mean_amp[hit] == pytest.approx(3.0)


def test_sparse_bins_are_nan_not_a_wild_rate() -> None:
    """IBL drops bins under 50 spikes; a 3-spike bin would otherwise plot a rate."""
    depths = np.concatenate([np.full(100, 50.0), np.full(3, 200.0)])
    amps = np.ones_like(depths)

    centres, rate, _ = depth_profiles(
        depths, amps, duration_s=10.0, bin_um=10.0, min_spikes=50
    )

    dense = np.argmin(np.abs(centres - 50.0))
    sparse = np.argmin(np.abs(centres - 200.0))
    assert np.isfinite(rate[dense])
    assert np.isnan(rate[sparse])


def test_depth_profiles_of_nothing_is_empty() -> None:
    centres, rate, amp = depth_profiles(np.array([]), np.array([]), duration_s=1.0)

    assert centres.size == 0 and rate.size == 0 and amp.size == 0


def test_depth_profiles_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="must match"):
        depth_profiles(np.zeros(5), np.zeros(4), duration_s=1.0)


def test_zero_duration_gives_no_rate_rather_than_dividing_by_zero() -> None:
    depths = np.full(100, 10.0)

    _, rate, _ = depth_profiles(depths, np.ones(100), duration_s=0.0, bin_um=10.0)

    assert np.isnan(rate).all()


def test_raster_thinning_preserves_the_depth_distribution() -> None:
    """Thinning must be uniform, so visible density stays proportional to real."""
    rng = np.random.default_rng(0)
    n = 500_000
    depths = np.concatenate([rng.normal(200, 20, n // 5), rng.normal(600, 20, 4 * n // 5)])
    times = rng.uniform(0, 100, depths.size)
    amps = rng.normal(1.0, 0.1, depths.size)

    t, d, _a = raster_points(times, depths, amps, max_points=50_000)

    assert t.size == d.size == _a.size == 50_000
    # The 20/80 split between the two depth clusters must survive.
    assert (d < 400).mean() == pytest.approx(0.2, abs=0.02)


def test_raster_returns_everything_when_under_the_cap() -> None:
    times, depths, amps = np.arange(10.0), np.arange(10.0), np.arange(10.0)

    t, d, _a = raster_points(times, depths, amps, max_points=100)

    assert t.size == 10 and np.array_equal(d, depths)


def test_raster_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        raster_points(np.zeros(3), np.zeros(4), np.zeros(3))
