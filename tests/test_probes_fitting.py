"""Tests for probes/fitting.py."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.probes.fitting import (
    fit_trajectory,
    line_point_distances,
    ordered_endpoints,
    pca_line_fit,
    ransac_line_fit,
)


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
