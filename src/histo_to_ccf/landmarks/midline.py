"""Estimate the midline of a coronal section.

Default strategy: PCA on the tissue-mask pixels. For a coronal section the
medio-lateral axis is the longer principal axis; the midline is the line
through the centroid perpendicular to it (i.e., along the minor axis).

When the section is mounted near-vertical (the common case), the PCA midline
matches the visual symmetry axis well enough. ``refine_by_reflection`` adds an
optional second pass that searches a small angular window around the PCA
estimate for the orientation that maximizes mask self-overlap when reflected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Midline:
    """A midline line, parameterized by a point and a unit direction."""

    centroid_px: tuple[float, float]  # (cx, cy)
    direction: tuple[float, float]  # unit vector along the midline (dorsal→ventral)
    angle_deg_from_vertical: float

    def line_at(self, t_range: tuple[float, float], steps: int = 50) -> np.ndarray:
        """Sample points along the midline; returns ``(steps, 2)`` (x, y)."""
        cx, cy = self.centroid_px
        dx, dy = self.direction
        ts = np.linspace(t_range[0], t_range[1], steps)
        return np.column_stack([cx + ts * dx, cy + ts * dy])


def estimate_midline_pca(mask: np.ndarray) -> Midline | None:
    """Estimate the midline of a section by PCA on the mask pixels."""
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    # vh rows are principal directions, in decreasing variance order.
    # Major axis (vh[0]) ≈ ML; minor (vh[1]) ≈ DV - the midline direction.
    minor = vh[1]
    # Orient minor so its y component is positive (dorsal → ventral, image-down).
    if minor[1] < 0:
        minor = -minor
    angle = float(np.degrees(np.arctan2(minor[0], minor[1])))
    return Midline(
        centroid_px=(float(centroid[0]), float(centroid[1])),
        direction=(float(minor[0]), float(minor[1])),
        angle_deg_from_vertical=angle,
    )


def refine_by_reflection(
    mask: np.ndarray,
    initial: Midline,
    *,
    angle_search_deg: float = 8.0,
    n_steps: int = 17,
) -> Midline:
    """Search a small angle window around ``initial`` for max reflection overlap.

    The score is the fraction of mask pixels whose mirrored counterpart (about
    the candidate midline) is also tissue. Centroid is held fixed.
    """
    cx, cy = initial.centroid_px
    base_angle = initial.angle_deg_from_vertical
    angles = np.linspace(
        base_angle - angle_search_deg, base_angle + angle_search_deg, n_steps
    )

    best_angle = base_angle
    best_score = -1.0
    for angle in angles:
        # Rotate the mask so the candidate midline lies along the y-axis.
        rotated = ndimage.rotate(
            mask.astype(float),
            -float(angle),
            reshape=False,
            order=0,
            mode="constant",
            cval=0.0,
        )
        # Reflect about the vertical line at x = cx (approximate; we ignore
        # the small centroid shift induced by rotation about image center -
        # the angle window is small so this is OK).
        h, w = rotated.shape
        cx_int = int(round(cx))
        cx_int = max(1, min(w - 2, cx_int))
        left_width = min(cx_int, w - cx_int)
        left = rotated[:, cx_int - left_width : cx_int]
        right = rotated[:, cx_int : cx_int + left_width][:, ::-1]
        overlap = float((left * right).sum())
        union = float((left + right > 0).sum()) + 1e-9
        score = overlap / union
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    angle_rad = np.radians(best_angle)
    direction = (float(np.sin(angle_rad)), float(np.cos(angle_rad)))
    if direction[1] < 0:
        direction = (-direction[0], -direction[1])
    return Midline(
        centroid_px=(cx, cy),
        direction=direction,
        angle_deg_from_vertical=best_angle,
    )


def estimate_midline(
    mask: np.ndarray, *, refine: bool = False
) -> Midline | None:
    """Convenience: PCA estimate, optionally followed by reflection refinement."""
    initial = estimate_midline_pca(mask)
    if initial is None or not refine:
        return initial
    return refine_by_reflection(mask, initial)
