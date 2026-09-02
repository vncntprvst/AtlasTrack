"""Tests for probes/fitting.py."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.probes.fitting import (
    fit_rigid_array,
    fit_trajectory,
    line_point_distances,
    ordered_endpoints,
    pca_line_fit,
    ransac_line_fit,
)


# ---------------------------------------------------------------------------
# fit_rigid_array
# ---------------------------------------------------------------------------

def _noisy_array():
    """A 4-shank array (250 µm along ML, inserted along DV) with uneven, noisy picks."""
    # Ideal even tips at ML = 0,250,500,750; DV=6000; AP=11000. Entries 5000 shallower.
    ranks = np.array([0, 1, 2, 3], dtype=float)
    ml = np.array([0.0, 380.0, 470.0, 750.0])   # uneven spacing (picking noise)
    tips = np.stack([np.full(4, 11000.0), ml, np.full(4, 6000.0)], axis=1)
    tips[1, 2] += 120.0                          # one shank off-axis in DV
    entries = tips.copy()
    entries[:, 2] -= 5000.0                       # 5 mm shallower (toward surface)
    entries[2, 0] += 90.0                         # noise on an entry
    return tips, entries, ranks


def test_fit_rigid_array_strict_even_spacing() -> None:
    tips, entries, _ = _noisy_array()
    nt, ne, info = fit_rigid_array(tips, entries, tolerance=0.0)
    # Consecutive tip gaps become equal (strict).
    gaps = np.linalg.norm(np.diff(nt, axis=0), axis=1)
    assert np.allclose(gaps, gaps[0], atol=1e-6)
    # Shanks are parallel: every tip→entry vector is identical.
    dirs = nt - ne
    assert np.allclose(dirs, dirs[0], atol=1e-6)
    assert info["spacing_um"] > 0


def test_fit_rigid_array_tolerance_blends() -> None:
    tips, entries, _ = _noisy_array()
    strict_t, _, _ = fit_rigid_array(tips, entries, tolerance=0.0)
    mid_t, _, _ = fit_rigid_array(tips, entries, tolerance=0.5)
    # tolerance=1 leaves the picks untouched.
    keep_t, keep_e, _ = fit_rigid_array(tips, entries, tolerance=1.0)
    assert np.allclose(keep_t, tips) and np.allclose(keep_e, entries)
    # tolerance=0.5 sits between strict and the original picks.
    assert np.all(np.abs(mid_t - tips) <= np.abs(strict_t - tips) + 1e-9)


def test_fit_rigid_array_lock_spacing() -> None:
    tips, entries, _ = _noisy_array()
    nt, _, info = fit_rigid_array(tips, entries, tolerance=0.0, lock_spacing_um=250.0)
    gaps = np.linalg.norm(np.diff(nt, axis=0), axis=1)
    assert np.allclose(gaps, 250.0, atol=1e-6)
    assert info["spacing_um"] == 250.0


def test_fit_rigid_array_too_few_shanks_noop() -> None:
    tips = np.array([[0.0, 0, 0], [0, 250, 0]])
    entries = tips - np.array([0, 0, 5000.0])
    nt, ne, info = fit_rigid_array(tips, entries)
    assert np.allclose(nt, tips) and np.allclose(ne, entries)
    assert info == {}


# ---------------------------------------------------------------------------
# pca_line_fit
# ---------------------------------------------------------------------------

def test_pca_exact_collinear() -> None:
    """PCA should recover the exact direction of collinear points."""
    direction = np.array([0.0, 0.0, 1.0])  # DV axis
    pts = np.array([[5000.0, 5700.0, d] for d in range(0, 3000, 100)], dtype=float)
    centroid, d_hat = pca_line_fit(pts)
    np.testing.assert_allclose(np.abs(d_hat), np.abs(direction), atol=1e-6)
    np.testing.assert_allclose(centroid, pts.mean(axis=0), atol=1e-6)


def test_pca_noisy_line() -> None:
    """PCA on noisy collinear points should recover direction within tolerance."""
    rng = np.random.default_rng(0)
    n = 20
    t = np.linspace(0, 3000, n)
    true_dir = np.array([0.0, 0.0, 1.0])
    pts = np.stack([5000 + rng.normal(0, 5, n),
                    5700 + rng.normal(0, 5, n),
                    t], axis=1)
    _, d_hat = pca_line_fit(pts)
    # Angle between fitted and true direction should be < 5°.
    cos_angle = abs(float(d_hat @ true_dir))
    assert cos_angle > np.cos(np.deg2rad(5))


def test_pca_requires_2_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        pca_line_fit(np.array([[1.0, 2.0, 3.0]]))


def test_pca_wrong_shape() -> None:
    with pytest.raises(ValueError):
        pca_line_fit(np.zeros((5, 2)))


# ---------------------------------------------------------------------------
# line_point_distances
# ---------------------------------------------------------------------------

def test_distances_on_axis_are_zero() -> None:
    anchor = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    pts = np.array([[0.0, 0.0, d] for d in range(10)], dtype=float)
    dists = line_point_distances(pts, anchor, direction)
    np.testing.assert_allclose(dists, 0.0, atol=1e-10)


def test_distances_perpendicular() -> None:
    anchor = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    pts = np.array([[100.0, 0.0, 500.0], [0.0, 50.0, 0.0]])
    dists = line_point_distances(pts, anchor, direction)
    assert dists[0] == pytest.approx(100.0)
    assert dists[1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# ransac_line_fit
# ---------------------------------------------------------------------------

def test_ransac_rejects_outlier() -> None:
    """RANSAC should exclude a gross outlier."""
    rng = np.random.default_rng(1)
    # Clean collinear points along DV axis.
    t = np.linspace(0, 2000, 15)
    pts = np.stack([np.full(15, 5000.0),
                    np.full(15, 5700.0),
                    t], axis=1)
    pts = pts + rng.normal(0, 5, pts.shape)
    # Add one gross outlier.
    pts = np.vstack([pts, [[5000.0, 8000.0, 1000.0]]])
    _, _, mask = ransac_line_fit(pts, inlier_threshold_um=50.0, seed=7)
    # The outlier (last point) should NOT be an inlier.
    assert not mask[-1], "Outlier should be excluded by RANSAC"
    assert mask[:-1].sum() >= 10


def test_ransac_all_inliers() -> None:
    """All points on a clean line → all inliers."""
    t = np.linspace(0, 3000, 20)
    pts = np.stack([np.full(20, 5000.0), np.full(20, 5700.0), t], axis=1)
    _, _, mask = ransac_line_fit(pts, inlier_threshold_um=10.0, seed=0)
    assert mask.all()


# ---------------------------------------------------------------------------
# ordered_endpoints
# ---------------------------------------------------------------------------

def test_ordered_endpoints_tip_is_deeper() -> None:
    """The 'tip' endpoint should have a larger DV value than 'entry'."""
    centroid = np.array([5000.0, 5700.0, 1500.0])
    direction = np.array([0.0, 0.0, 1.0])
    pts = np.array([[5000.0, 5700.0, 0.0],
                    [5000.0, 5700.0, 3000.0]])
    entry, tip = ordered_endpoints(centroid, direction, pts)
    assert tip[2] > entry[2]  # DV: tip is deeper (larger DV)


# ---------------------------------------------------------------------------
# fit_trajectory
# ---------------------------------------------------------------------------

def test_fit_trajectory_returns_ordered_pair() -> None:
    pts = np.array([[5000.0, 5700.0, d] for d in [200, 500, 800, 1200, 2000]], dtype=float)
    entry, tip, mask = fit_trajectory(pts, use_ransac=False)
    assert entry[2] < tip[2]  # entry shallower than tip
    assert mask.all()


def test_fit_trajectory_ransac_with_noise() -> None:
    rng = np.random.default_rng(2)
    t = np.linspace(0, 2500, 20)
    pts = np.stack([5000 + rng.normal(0, 10, 20),
                    5700 + rng.normal(0, 10, 20),
                    t + rng.normal(0, 10, 20)], axis=1)
    pts = np.vstack([pts, [[5000.0, 9000.0, 1000.0]]])  # outlier in ML
    entry, tip, mask = fit_trajectory(pts, use_ransac=True, inlier_threshold_um=100.0)
    assert not mask[-1], "Outlier should be excluded"
    assert entry[2] < tip[2]
    # Entry and tip should be within a reasonable range of 0 and 2500 µm DV.
    assert 0 <= entry[2] < 1000
    assert 1500 < tip[2] <= 3000


def test_fit_trajectory_requires_2_points() -> None:
    with pytest.raises(ValueError):
        fit_trajectory(np.array([[5000.0, 5700.0, 1000.0]]))
