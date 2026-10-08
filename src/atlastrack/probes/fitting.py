"""PCA-SVD line fitting + small RANSAC for probe trajectory estimation.

All coordinates are (AP, ML, DV) in µm, matching the project schema.
No scikit-learn dependency - PCA uses ``numpy.linalg.svd``.
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
        Shape (3,) - mean of ``points``.
    direction
        Shape (3,) unit vector - first principal component (best-fit axis).
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

    Returns ``(shallowest, deepest)`` - i.e. (entry, tip) - where deepest
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


def fit_rigid_array(
    tips: np.ndarray,
    entries: np.ndarray,
    *,
    tolerance: float = 0.25,
    lock_spacing_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bring a multi-shank probe's entries onto the probe's own row, keeping each track.

    The shanks leave the probe base side by side, a fixed distance apart (250 µm on a
    Neuropixels 2.0), so where they enter the brain they form an evenly spaced row.
    Lower down the shanks bend, so the tips may spread apart; they follow the dye.
    Picks made shank by shank are noisy at the entries (uneven spacing, one shank off
    the row), so this fits the row to the entries and moves each **whole track** by
    its own entry's correction. A track keeps its direction and its length, so a tip
    that really diverged still does.

    The entries are compared at one depth: each is slid along its own track to the
    plane through the mean entry, square to the mean insertion direction (entries
    clicked at different heights would otherwise read as uneven spacing). In that
    plane the shanks keep their order (shank 0 first) and are placed
    ``lock_spacing_um`` apart around the entries' centre, along the direction that
    best fits them in that order.

    Parameters
    ----------
    tips, entries
        ``(N, 3)`` CCF (AP, ML, DV) µm, in shank order; ``N >= 3``.
    tolerance
        In ``[0, 1]``. ``0`` moves the entries exactly onto the row; ``1`` leaves the
        picks unchanged. Each track moves by ``(1 - tolerance)`` of its correction.
    lock_spacing_um
        The distance between neighbouring shanks (the probe's shank pitch). ``None``
        estimates it from the entries, which is only a fallback: a free estimate
        follows the picking noise (it once gave 404 µm for a 250 µm probe).

    Returns
    -------
    new_tips, new_entries, info
        The moved coordinates plus a dict with ``spacing_um``, ``old_entry_gaps_um``
        and ``new_entry_gaps_um`` (neighbour distances at the common depth),
        ``max_shift_um`` and ``order``.
    """
    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    n = len(tips)
    if n < 3 or entries.shape != tips.shape:
        return tips, entries, {}

    ups = entries - tips
    ups = ups / (np.linalg.norm(ups, axis=1, keepdims=True) + 1e-12)
    u = ups.mean(0)
    u = u / (np.linalg.norm(u) + 1e-12)

    def at_level(ends: np.ndarray, centre: np.ndarray) -> np.ndarray:
        # Slide each point along its own track onto the plane through ``centre``.
        steps = ((centre - ends) @ u) / np.clip(ups @ u, 1e-6, None)
        return ends + steps[:, None] * ups

    ce = entries.mean(0)
    level = at_level(entries, ce)
    flat = level - level.mean(0)
    flat = flat - (flat @ u)[:, None] * u
    # Row axis: the direction that best fits the shanks *in their known order*, at
    # even steps. (The entries' principal axis ignores the order, and a zigzag of
    # picks then turns the row sideways - LO_02 moved 417 µm that way.)
    k = np.arange(n) - (n - 1) / 2.0
    r = k @ flat
    r = r - (r @ u) * u
    r = r / (np.linalg.norm(r) + 1e-12)

    old_gaps = np.linalg.norm(np.diff(level, axis=0), axis=1)
    spacing = (
        float(lock_spacing_um) if lock_spacing_um
        else float(k @ (flat @ r)) / float(k @ k)
    )
    row = level.mean(0) + (k * spacing)[:, None] * r

    t = float(np.clip(tolerance, 0.0, 1.0))
    shift = (1.0 - t) * (row - level)
    new_tips = tips + shift
    new_entries = entries + shift
    new_gaps = np.linalg.norm(np.diff(level + shift, axis=0), axis=1)
    info = {
        "spacing_um": spacing,
        "old_entry_gaps_um": [float(g) for g in old_gaps],
        "new_entry_gaps_um": [float(g) for g in new_gaps],
        "max_shift_um": float(np.linalg.norm(shift, axis=1).max()),
        "order": list(range(n)),
    }
    return new_tips, new_entries, info


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
        Boolean array of shape (N,) - True for points used in the final fit.
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


def enforce_rigid_arrays(
    project,
    *,
    tolerance: float = 0.25,
    lock_spacing_um: float | None = None,
) -> dict[str, dict]:
    """Apply :func:`fit_rigid_array` to every multi-shank probe in ``project``.

    Rewrites each participating shank's ``tip_ccf_um`` / ``entry_ccf_um`` in place.
    Only shanks carrying both a tip and an entry take part; a probe with fewer than
    three of those is left alone (the row is underdetermined). The spacing is the
    probe's own shank pitch unless ``lock_spacing_um`` says otherwise. Returns the
    per-probe ``info`` dicts from :func:`fit_rigid_array`, keyed by probe label.

    Call this *after* the pixel→CCF re-map, otherwise the re-map overwrites it.
    """
    out: dict[str, dict] = {}
    for probe in project.probes:
        shanks = sorted(
            (s for s in probe.shanks
             if s.tip_ccf_um is not None and s.entry_ccf_um is not None),
            key=lambda s: s.index,
        )
        if len(shanks) < 3:
            continue
        tips = np.array([s.tip_ccf_um for s in shanks], dtype=float)
        entries = np.array([s.entry_ccf_um for s in shanks], dtype=float)
        spacing = lock_spacing_um or probe.type.shank_pitch_um
        new_tips, new_entries, info = fit_rigid_array(
            tips, entries, tolerance=tolerance, lock_spacing_um=spacing
        )
        for s, t, e in zip(shanks, new_tips, new_entries, strict=True):
            s.tip_ccf_um = tuple(float(v) for v in t)
            s.entry_ccf_um = tuple(float(v) for v in e)
        out[probe.label] = info
    return out
