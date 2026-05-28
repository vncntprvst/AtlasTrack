"""PCA-SVD line fitting + small RANSAC for probe trajectory estimation.

All coordinates are (AP, ML, DV) in µm, matching the project schema.
No scikit-learn dependency — PCA uses ``numpy.linalg.svd``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core fitting primitives
# ---------------------------------------------------------------------------

def pca_line_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 3D line through ``points`` via PCA (SVD).

    Parameters
    ----------
    points
        Array of shape (N, 3).

    Returns
    -------
    centroid
        Shape (3,) — mean of ``points``.
    direction
        Shape (3,) unit vector — first principal component (best-fit axis).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to fit a line")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    return centroid, direction


def line_point_distances(
    points: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Return perpendicular distance from each point to an infinite 3D line.

    Parameters
    ----------
    points
        Shape (N, 3).
    anchor
        A point on the line, shape (3,).
    direction
        Unit vector along the line, shape (3,).

    Returns
    -------
    distances
        Shape (N,).
    """
    pts = np.asarray(points, dtype=float)
    a = np.asarray(anchor, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    v = pts - a
    proj = (v @ d)[:, np.newaxis] * d
    perp = v - proj
    return np.linalg.norm(perp, axis=1)


def ransac_line_fit(
    points: np.ndarray,
    *,
    n_iter: int = 100,
    inlier_threshold_um: float = 150.0,
    min_inliers: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust line fit via RANSAC.

    At each iteration two points are sampled; inliers within
    ``inlier_threshold_um`` are counted.  The iteration with the most inliers
    (ties broken by smallest mean residual) wins.  A final PCA is run on the
    winning inlier set.

    Parameters
    ----------
    points
        Shape (N, 3), N ≥ 2.
    n_iter
        Number of RANSAC iterations.
    inlier_threshold_um
        Maximum perpendicular distance (µm) to count a point as an inlier.
    min_inliers
        If the best set has fewer inliers than this, fall back to a full PCA.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    centroid
        Shape (3,) anchor point of the fitted line (centroid of inliers).
    direction
        Shape (3,) unit vector.
    inlier_mask
        Boolean array of shape (N,).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        raise ValueError("Need at least 2 points")
    rng = np.random.default_rng(seed)

    best_mask: np.ndarray = np.zeros(n, dtype=bool)
    best_count = 0
    best_residual = float("inf")

    for _ in range(n_iter):
        idx = rng.choice(n, size=2, replace=False)
        a, b = pts[idx[0]], pts[idx[1]]
        d = b - a
        dnorm = np.linalg.norm(d)
        if dnorm < 1e-9:
            continue
        d = d / dnorm
        dists = line_point_distances(pts, a, d)
        mask = dists <= inlier_threshold_um
        count = int(mask.sum())
        if count > best_count or (
            count == best_count and float(dists[mask].mean()) < best_residual
        ):
            best_count = count
            best_mask = mask
            best_residual = float(dists[mask].mean()) if count > 0 else float("inf")

    # Final PCA on the best inlier set (or all points if too few inliers).
    if best_count >= max(min_inliers, 2):
        final_pts = pts[best_mask]
    else:
        best_mask = np.ones(n, dtype=bool)
        final_pts = pts

    centroid, direction = pca_line_fit(final_pts)
    return centroid, direction, best_mask


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def ordered_endpoints(
    centroid: np.ndarray,
    direction: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``points`` onto the line and return the extreme endpoints.

    Returns ``(shallowest, deepest)`` — i.e. (entry, tip) — where deepest
    means the point that projects furthest along ``direction`` (toward the
    brain tip).
    """
    pts = np.asarray(points, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    projections = (pts - centroid) @ d
    i_min = int(np.argmin(projections))
    i_max = int(np.argmax(projections))
    # Choose ordering so that the larger DV value is the "tip" (deeper in brain).
    entry_candidate = centroid + projections[i_min] * d
    tip_candidate = centroid + projections[i_max] * d
    if tip_candidate[2] < entry_candidate[2]:
        entry_candidate, tip_candidate = tip_candidate, entry_candidate
    return entry_candidate, tip_candidate


def fit_trajectory(
    points: np.ndarray,
    *,
    use_ransac: bool = True,
    inlier_threshold_um: float = 150.0,
    n_iter: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a probe trajectory through a set of 3D CCF points.

    Intended use: pass all tip_ccf and/or entry_ccf values collected for one
    shank across sections, and recover a single best-fit trajectory.

    Returns
    -------
    entry_ccf
        Estimated entry-point at the brain surface, shape (3,).
    tip_ccf
        Estimated tip position, shape (3,).
    inlier_mask
        Boolean array of shape (N,) — True for points used in the final fit.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points to fit a trajectory")

    if use_ransac and len(pts) >= 4:
        centroid, direction, mask = ransac_line_fit(
            pts,
            n_iter=n_iter,
            inlier_threshold_um=inlier_threshold_um,
            seed=seed,
        )
        inlier_pts = pts[mask]
    else:
        centroid, direction = pca_line_fit(pts)
        mask = np.ones(len(pts), dtype=bool)
        inlier_pts = pts

    entry, tip = ordered_endpoints(centroid, direction, inlier_pts)
    return entry, tip, mask
