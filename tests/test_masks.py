"""Tests for section tissue / label masks (headless)."""
from __future__ import annotations

import numpy as np

from atlastrack.registration.masks import (
    moment_similarity,
    registration_moving_mask,
    section_label_mask,
    section_tissue_mask,
)


def _section(h=120, w=160) -> np.ndarray:
    """Blue tissue disk on black, plus a green and a magenta label blob."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    tissue = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 < (h // 3) ** 2
    img[tissue, 2] = 180  # blue
    green = (yy - h // 2) ** 2 + (xx - w // 2 + 15) ** 2 < 8 ** 2
    img[green] = (0, 220, 0)
    magenta = (yy - h // 2 - 10) ** 2 + (xx - w // 2) ** 2 < 6 ** 2
    img[magenta] = (220, 0, 220)
    return img


def test_tissue_mask_covers_body_not_background() -> None:
    img = _section()
    m = section_tissue_mask(img)
    assert m.dtype == bool
    assert m[60, 80]  # centre is tissue
    assert not m[5, 5]  # corner is background
    # single connected component
    from scipy import ndimage as ndi

    assert ndi.label(m)[1] == 1
    assert 0.1 < m.mean() < 0.6


def test_label_mask_flags_labels_only() -> None:
    img = _section()
    labels = section_label_mask(img, dilate=0)
    # green + magenta pixels flagged, pure-blue tissue not.
    assert labels[60, 65]  # green blob (centre x≈65)
    assert labels[70, 80]  # magenta blob (centre y≈70, x≈80)
    assert not labels[60, 100]  # blue tissue, no label
    # Grayscale / blue-only inputs => no labels.
    assert not section_label_mask(np.zeros((10, 10), dtype=np.uint8)).any()
    blue = np.zeros((20, 20, 3), dtype=np.uint8)
    blue[..., 2] = 200
    assert not section_label_mask(blue).any()


def test_registration_moving_mask_excludes_labels_keeps_outline() -> None:
    img = _section()
    mm = registration_moving_mask(img)
    assert mm.dtype == np.uint8
    assert mm.sum() > 0
    # The green label pixel is inside the tissue body but must be excluded.
    assert mm[60, 65] == 0
    # Tissue away from labels is kept.
    assert mm[60, 100] == 1


def test_registration_mask_never_empty_on_flat_image() -> None:
    flat = np.zeros((30, 30, 3), dtype=np.uint8)
    assert registration_moving_mask(flat).all()


def test_registration_mask_keeps_cyan_tissue_not_treated_as_label() -> None:
    """A DAPI section rendered cyan (high green+blue) must not be gutted.

    Regression: ``section_label_mask`` flags green/red as fluorescent label, so a
    cyan-stained tissue body had ~65% of its pixels excised, leaving a sparse
    moving mask that made elastix abort ("too many samples map outside moving
    image buffer"). The label exclusion must back off when it would remove most
    of the tissue.
    """
    h, w = 120, 160
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    tissue = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 < (h // 3) ** 2
    img[tissue, 1] = 170  # green  -> cyan body (both channels high)
    img[tissue, 2] = 200  # blue

    body = float(tissue.mean())
    mm = registration_moving_mask(img)
    # The mask must still cover (most of) the tissue, not collapse to fragments.
    assert mm.mean() >= 0.8 * body
    assert mm[60, 80] == 1  # tissue centre kept


def _disk_mask(h, w, cy, cx, r) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r * r


def test_moment_similarity_recovers_scale_and_shift() -> None:
    fixed = _disk_mask(120, 120, 60, 60, 18)
    moving = _disk_mask(120, 120, 72, 48, 27)  # +12 row, -12 col, 1.5x
    s = moment_similarity(fixed, moving)
    # Maps the fixed centroid (x=60, y=60) onto the moving centroid (x=48, y=72).
    x, y, _ = s @ np.array([60.0, 60.0, 1.0])
    assert np.allclose([x, y], [48.0, 72.0], atol=1.0)
    assert np.isclose(s[0, 0], 1.5, atol=0.05) and np.isclose(s[1, 1], 1.5, atol=0.05)


def test_moment_similarity_identity_on_degenerate() -> None:
    empty = np.zeros((20, 20), dtype=bool)
    assert np.allclose(moment_similarity(empty, empty), np.eye(3))
    one = _disk_mask(40, 40, 20, 20, 8)
    # Same mask -> identity (no shift, unit scale).
    assert np.allclose(moment_similarity(one, one), np.eye(3), atol=1e-6)
