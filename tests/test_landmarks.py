"""Tests for contour, ventricle, and midline landmark extraction."""
from __future__ import annotations

import numpy as np
from skimage.draw import disk, ellipse

from atlastrack.landmarks.contour import contour_centroid, section_contour
from atlastrack.landmarks.midline import estimate_midline, estimate_midline_pca
from atlastrack.landmarks.ventricles import detect_ventricles


def _ellipse_mask(h: int = 200, w: int = 300, ry: int = 70, rx: int = 120) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = ellipse(h // 2, w // 2, ry, rx, shape=mask.shape)
    mask[rr, cc] = True
    return mask


def test_contour_centroid_matches_mask_center() -> None:
    mask = _ellipse_mask()
    contour = section_contour(mask)
    assert contour.shape[1] == 2
    cx, cy = contour_centroid(contour)
    assert abs(cx - mask.shape[1] / 2) < 1.0
    assert abs(cy - mask.shape[0] / 2) < 1.0


def test_midline_pca_is_vertical_for_horizontal_ellipse() -> None:
    """A horizontally elongated ellipse (ML > DV) yields a near-vertical midline."""
    mask = _ellipse_mask(ry=60, rx=140)
    midline = estimate_midline_pca(mask)
    assert midline is not None
    # Vertical = direction roughly (0, 1). Angle from vertical should be small.
    assert abs(midline.angle_deg_from_vertical) < 2.0


def test_midline_refinement_preserves_symmetry_axis() -> None:
    mask = _ellipse_mask(ry=60, rx=140)
    refined = estimate_midline(mask, refine=True)
    assert refined is not None
    assert abs(refined.angle_deg_from_vertical) < 3.0


def test_ventricles_detected_from_two_holes() -> None:
    """Two symmetric circular holes should be found as ventricles."""
    mask = _ellipse_mask()
    # Punch two holes inside the ellipse (lateral-ventricle-like).
    h, w = mask.shape
    for cx in (w // 2 - 40, w // 2 + 40):
        rr, cc = disk((h // 2, cx), 8, shape=mask.shape)
        mask[rr, cc] = False
    vents = detect_ventricles(mask, min_area_px=20)
    assert len(vents) == 2
    # Areas should be comparable (within 30 %).
    areas = sorted(v.area_px for v in vents)
    assert areas[1] / areas[0] < 1.3
