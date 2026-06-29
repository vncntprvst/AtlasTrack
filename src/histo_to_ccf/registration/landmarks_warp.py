"""Landmark (thin-plate-spline) manual correction of the atlas overlay.

VisuAlign / BigWarp style: a handful of points are auto-placed on the registered
atlas overlay; the user drags each onto the matching tissue feature, and a
thin-plate spline interpolates a smooth non-rigid warp from the dragged
displacements. Unlike the box-handle affine this can fix *local* distortions
(e.g. a damaged region) while pinning the rest.

Conventions: all points are section-local ``(x, y)`` = (col, row) pixels. The TPS
maps **source -> target** (where the user dragged things) for the overlay, and
**target -> source** for the probe->CCF inverse. Headless (numpy / scipy only).
"""
from __future__ import annotations

import numpy as np


def auto_landmarks(
    extent_mask: np.ndarray, *, n_perimeter: int = 6, n_inside: int = 3
) -> np.ndarray:
    """Auto-place landmarks on a warped-atlas extent: a ring + interior points.

    Returns an ``(n_perimeter + n_inside, 2)`` array of section-local ``(x, y)``
    points: ``n_perimeter`` on the silhouette boundary (evenly spaced by angle
    from the centroid) and ``n_inside`` down the vertical midline. Falls back to
    a centred grid if the mask is degenerate.
    """
    ys, xs = np.nonzero(extent_mask)
    h, w = extent_mask.shape
    if xs.size < 16:
        cx, cy = w / 2.0, h / 2.0
        return np.array([[cx, cy]], dtype=float)
    cx, cy = float(xs.mean()), float(ys.mean())

    pts: list[tuple[float, float]] = []
    # Perimeter: march outward from the centroid along each angle to the last
    # in-mask pixel (the silhouette boundary in that direction).
    reach = float(np.hypot(h, w))
    for k in range(n_perimeter):
        ang = 2.0 * np.pi * k / n_perimeter
        dx, dy = np.cos(ang), np.sin(ang)
        last = (cx, cy)
        for r in np.arange(2.0, reach, 2.0):
            x, y = cx + dx * r, cy + dy * r
            ix, iy = round(x), round(y)
            if 0 <= iy < h and 0 <= ix < w and extent_mask[iy, ix]:
                last = (x, y)
            else:
                break
        # Pull just inside the boundary so the handle sits on the contour.
        pts.append((cx + (last[0] - cx) * 0.92, cy + (last[1] - cy) * 0.92))

    # Interior: down the vertical midline at evenly spaced heights.
    ys_at_cx = np.nonzero(extent_mask[:, int(round(cx))])[0]
    y_lo, y_hi = (float(ys_at_cx.min()), float(ys_at_cx.max())) if ys_at_cx.size else (cy, cy)
    for k in range(n_inside):
        frac = (k + 1) / (n_inside + 1)
        pts.append((cx, y_lo + frac * (y_hi - y_lo)))
    return np.array(pts, dtype=float)


def _boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean edge map: True where a region label differs from a 4-neighbour.

    Inlined (not imported from ``transforms``) to keep this module dependency-free.
    """
    labels = np.asarray(labels)
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[1:, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edges


def _region_junctions(labels: np.ndarray) -> list[tuple[float, float]]:
    """Centroids of points where >=3 region outlines meet (skeleton branch points)."""
    from scipy.ndimage import convolve
    from scipy.ndimage import label as cc_label
    from skimage.morphology import skeletonize

    sk = skeletonize(_boundaries(labels))
    if not sk.any():
        return []
    deg = convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    branch = sk & ((deg - sk.astype(np.uint8)) >= 3)  # >=3 skeleton neighbours
    if not branch.any():
        return []
    lab, n = cc_label(branch, structure=np.ones((3, 3)))
    out: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        out.append((float(xs.mean()), float(ys.mean())))  # (x, y)
    return out


def _silhouette_corners(extent: np.ndarray) -> list[tuple[float, float]]:
    """High-curvature corners of the silhouette (Harris response peaks)."""
    from skimage.feature import corner_harris, corner_peaks

    h, w = extent.shape
    resp = corner_harris(extent.astype(float))
    peaks = corner_peaks(
        resp, min_distance=max(5, int(0.05 * min(h, w))), threshold_rel=0.05
    )
    cy, cx = float(np.nonzero(extent)[0].mean()), float(np.nonzero(extent)[1].mean())
    out: list[tuple[float, float]] = []
    for r, c in peaks:
        # Nudge a hair toward the centroid so the handle sits on the overlay.
        out.append((c + (cx - c) * 0.04, r + (cy - r) * 0.04))
    return out


def _silhouette_extremes(extent: np.ndarray) -> list[tuple[float, float]]:
    """The four tips of the silhouette (top / bottom / left / right pixels)."""
    ys, xs = np.nonzero(extent)
    out: list[tuple[float, float]] = []
    for sel in (ys.argmin(), ys.argmax(), xs.argmin(), xs.argmax()):
        out.append((float(xs[sel]), float(ys[sel])))
    return out


def _spread_select(
    cands: list[tuple[float, float]], min_dist: float, max_points: int
) -> list[tuple[float, float]]:
    """Greedily keep candidates (in priority order) that are >= min_dist apart."""
    chosen: list[tuple[float, float]] = []
    d2 = min_dist * min_dist
    for x, y in cands:
        if all((x - cx) ** 2 + (y - cy) ** 2 >= d2 for cx, cy in chosen):
            chosen.append((x, y))
            if len(chosen) >= max_points:
                break
    return chosen


def salient_landmarks(
    labels: np.ndarray, *, max_points: int = 12
) -> np.ndarray:
    """Place landmarks on distinctive spots of the atlas overlay, with coverage.

    Candidates are taken in priority order: the four silhouette **tips**, then
    **junctions** where region outlines meet (skeleton branch points), then
    high-curvature silhouette **corners**, then the geometric
    :func:`auto_landmarks` ring as fill. A greedy spread keeps every point >= 10%
    of the image apart, so the result favours grabbable features where they exist
    (busy dorsal cortex) while the tips + ring guarantee the rest of the section
    (cerebellum, brainstem) is still pinned. Returns ``(N, 2)`` section-local
    ``(x, y)``; falls back to the plain ring on a degenerate (tiny) extent.
    """
    extent = np.asarray(labels) > 0
    h, w = extent.shape
    if int(extent.sum()) < 64:
        return auto_landmarks(extent)
    min_dist = max(10.0, 0.10 * min(h, w))

    ring = [(float(x), float(y)) for x, y in auto_landmarks(extent)]
    cands = (
        _silhouette_extremes(extent)
        + _region_junctions(labels)
        + _silhouette_corners(extent)
        + ring
    )
    chosen = _spread_select(cands, min_dist, max_points)
    if len(chosen) < 4:  # last resort: never starve the TPS
        return auto_landmarks(extent)
    return _snap_into_extent(np.array(chosen, dtype=float), extent)


def _snap_into_extent(pts: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Move any point that landed off the silhouette onto the nearest overlay pixel.

    Silhouette corners can sit a pixel outside the mask; snapping keeps every
    handle on the atlas overlay so it's grabbable.
    """
    from scipy.ndimage import distance_transform_edt

    h, w = extent.shape
    nearest = distance_transform_edt(~extent, return_indices=True)[1]  # (2, h, w)
    out = pts.copy()
    for k, (x, y) in enumerate(pts):
        ix = min(max(int(round(x)), 0), w - 1)
        iy = min(max(int(round(y)), 0), h - 1)
        if not extent[iy, ix]:
            out[k] = (float(nearest[1, iy, ix]), float(nearest[0, iy, ix]))
    return out


def _rbf(src: np.ndarray, dst: np.ndarray):
    from scipy.interpolate import RBFInterpolator

    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    # smoothing=0 -> exact interpolation (handles land exactly on their targets).
    return RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=0.0)


def warp_points(source: np.ndarray, target: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Forward TPS (source -> target) applied to ``pts`` (N, 2) in (x, y)."""
    return _rbf(source, target)(np.asarray(pts, dtype=float).reshape(-1, 2))


def invert_points(source: np.ndarray, target: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Reverse TPS (target -> source) - maps a corrected-frame point back."""
    return _rbf(target, source)(np.asarray(pts, dtype=float).reshape(-1, 2))


def warp_contour_image(
    edge_rc: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Forward-TPS boundary pixels and rasterise them into a binary edge image.

    ``edge_rc`` is an ``(N, 2)`` array of ``(row, col)`` boundary pixels of the
    *un-warped* atlas overlay. They are pushed source -> target and splatted into
    a fresh ``shape`` image (1 on the warped contour, 0 elsewhere). This is the
    cheap, live counterpart of :func:`warp_label_image` (a few hundred points, no
    per-pixel pull-back) used for the real-time landmark-drag preview.
    """
    edge_rc = np.asarray(edge_rc)
    h, w = int(shape[0]), int(shape[1])
    img = np.zeros((h, w), dtype=np.uint8)
    if edge_rc.size == 0:
        return img
    pts_xy = np.column_stack([edge_rc[:, 1], edge_rc[:, 0]]).astype(float)
    warped = warp_points(source, target, pts_xy)  # (N, 2) (x, y)
    cx = np.round(warped[:, 0]).astype(int)
    cy = np.round(warped[:, 1]).astype(int)
    ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    img[cy[ok], cx[ok]] = 1
    return img


def warp_label_image(
    labels: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Resample a label image through the TPS so ``labels`` move source -> target.

    ``corrected[q] = labels[reverseTPS(q)]`` (pull-back), nearest-neighbour, 0
    outside - i.e. the atlas content originally at ``source`` ends up at ``target``.
    """
    from scipy.ndimage import map_coordinates

    h, w = labels.shape
    rev = _rbf(target, source)  # corrected (x,y) -> registered/source (x,y)
    yy, xx = np.mgrid[0:h, 0:w]
    q = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    s = rev(q)
    out = map_coordinates(
        labels, [s[:, 1], s[:, 0]], order=0, mode="constant", cval=0.0
    )
    return out.reshape(h, w).astype(labels.dtype)
