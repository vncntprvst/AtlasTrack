"""Extract the outer contour of a section's tissue mask."""
from __future__ import annotations

import numpy as np
from skimage import measure


def section_contour(mask: np.ndarray, *, level: float = 0.5) -> np.ndarray:
    """Return the largest closed outline of ``mask`` as an ``(N, 2)`` polygon.

    Coordinates are in (x, y) pixel order, ready for plotting.
    """
    contours = measure.find_contours(mask.astype(float), level=level)
    if not contours:
        return np.empty((0, 2), dtype=float)
    # measure.find_contours returns (row, col); pick the longest, convert to (x, y).
    longest = max(contours, key=len)
    return np.column_stack([longest[:, 1], longest[:, 0]])


def contour_centroid(contour_xy: np.ndarray) -> tuple[float, float]:
    """Centroid of a polygon contour (simple mean of vertices)."""
    if len(contour_xy) == 0:
        return float("nan"), float("nan")
    return float(contour_xy[:, 0].mean()), float(contour_xy[:, 1].mean())
