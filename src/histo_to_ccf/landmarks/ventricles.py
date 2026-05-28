"""Detect ventricles as holes inside a section's tissue mask.

Algorithm: morphologically fill the tissue mask, subtract the original mask to
isolate the holes, then filter by area and convexity to keep ventricle-like
shapes. Works for lateral, third, and fourth ventricles when they appear as
clean dark regions inside the section.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure


@dataclass(frozen=True)
class Ventricle:
    """One detected ventricle."""

    centroid_px: tuple[float, float]  # (cx, cy)
    area_px: int
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    mask: np.ndarray  # same shape as input section mask
    eccentricity: float


def detect_ventricles(
    section_mask: np.ndarray,
    *,
    min_area_px: int = 50,
    max_area_frac: float = 0.15,
    min_distance_to_edge_px: int = 5,
) -> list[Ventricle]:
    """Find ventricle-like holes inside ``section_mask`` (a bool array).

    ``min_distance_to_edge_px`` filters out holes that touch the outer contour
    (those are usually background, not ventricles).
    """
    mask = section_mask.astype(bool)
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask

    h, w = mask.shape
    max_area_px = int(max_area_frac * h * w)

    out: list[Ventricle] = []
    labeled = measure.label(holes, connectivity=2)
    for region in measure.regionprops(labeled):
        if region.area < min_area_px or region.area > max_area_px:
            continue
        minr, minc, maxr, maxc = region.bbox
        edge_dist = min(minr, minc, h - maxr, w - maxc)
        if edge_dist < min_distance_to_edge_px:
            continue
        hole_mask = labeled == region.label
        cy, cx = region.centroid
        out.append(
            Ventricle(
                centroid_px=(float(cx), float(cy)),
                area_px=int(region.area),
                bbox_px=(int(minc), int(minr), int(maxc), int(maxr)),
                mask=hole_mask,
                eccentricity=float(region.eccentricity),
            )
        )
    # Sort by area descending — lateral ventricles tend to dominate.
    out.sort(key=lambda v: -v.area_px)
    return out
