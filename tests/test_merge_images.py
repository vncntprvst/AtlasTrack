"""Tests for merging multiple slide images into one combined image."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.io.image import merge_images, slide_bands


def test_single_image_returned_unchanged() -> None:
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = merge_images([img])
    np.testing.assert_array_equal(out, img)


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        merge_images([])


def test_two_grayscale_stack_with_gap() -> None:
    a = np.ones((3, 4), dtype=np.uint8)
    b = np.full((2, 6), 2, dtype=np.uint8)
    gap = 5
    out = merge_images([a, b], gap_px=gap)
    # Height = 3 + gap + 2; width = max(4, 6).
    assert out.shape == (3 + gap + 2, 6)
    # First image lands top-left.
    np.testing.assert_array_equal(out[:3, :4], a)
    # Second image lands below the gap, top-left.
    np.testing.assert_array_equal(out[3 + gap:3 + gap + 2, :6], b)
    # The gap rows are background (zero).
    assert out[3:3 + gap].sum() == 0
    # Padding to the right of the narrower first image is background.
    assert out[:3, 4:].sum() == 0


def test_mixed_gray_and_rgb_promotes_to_rgb() -> None:
    gray = np.full((2, 2), 7, dtype=np.uint8)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 9
    out = merge_images([gray, rgb], gap_px=0)
    assert out.ndim == 3 and out.shape[2] == 3
    # Grayscale promoted by replication across channels.
    np.testing.assert_array_equal(out[:2, :2, :], np.stack([gray] * 3, axis=-1))
    np.testing.assert_array_equal(out[2:4, :2, :], rgb)


def test_merge_is_deterministic_for_reload() -> None:
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    b = np.arange(6, 12, dtype=np.uint8).reshape(2, 3)
    out1 = merge_images([a, b])
    out2 = merge_images([a, b])
    np.testing.assert_array_equal(out1, out2)


def test_slide_bands_match_merge_placement() -> None:
    """Each band's (y0, y1) is exactly where merge_images places that source."""
    a = np.ones((3, 4), dtype=np.uint8)
    b = np.full((2, 6), 2, dtype=np.uint8)
    gap = 5
    bands = slide_bands([a.shape[0], b.shape[0]], gap_px=gap)
    assert bands == [(0, 3), (3 + gap, 3 + gap + 2)]

    out = merge_images([a, b], gap_px=gap)
    for (y0, y1), src in zip(bands, [a, b], strict=True):
        np.testing.assert_array_equal(out[y0:y1, : src.shape[1]], src)


def test_slide_bands_single_source_spans_image() -> None:
    assert slide_bands([120]) == [(0, 120)]
