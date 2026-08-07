"""Detected boxes keep background around the tissue.

A box flush against the tissue leaves the registration nothing to work with on
that side - the mask and the boundary snap take the image border for the tissue
edge, which flattens the atlas contour there.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.sectioning.split import detect_sections


def _slide(blobs, shape=(400, 600)):
    """A dark slide with bright filled rectangles as 'sections'."""
    img = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in blobs:
        img[y0:y1, x0:x1] = 220
    return img


def _boxes(img, **kw):
    found = detect_sections(img, min_area_px=500, opening_radius_px=0, **kw)
    return sorted((s.bbox_px for s in found), key=lambda b: (b[0], b[1]))


def test_margin_leaves_background_around_the_tissue() -> None:
    blob = (200, 150, 320, 260)
    img = _slide([blob])

    tight = _boxes(img, margin_frac=0.0, equalize_boxes=False)[0]
    padded = _boxes(img, margin_frac=0.1, equalize_boxes=False)[0]

    assert tight[0] <= blob[0] and tight[2] >= blob[2]
    # Every side moved outwards, so the tissue no longer touches the edge.
    assert padded[0] < tight[0]
    assert padded[1] < tight[1]
    assert padded[2] > tight[2]
    assert padded[3] > tight[3]


def test_margin_scales_with_box_size() -> None:
    small = _boxes(_slide([(100, 100, 160, 160)]), margin_frac=0.1,
                   equalize_boxes=False)[0]
    big = _boxes(_slide([(100, 100, 340, 340)]), margin_frac=0.1,
                 equalize_boxes=False)[0]

    small_pad = 100 - small[0]
    big_pad = 100 - big[0]
    assert big_pad > small_pad, "a larger section should get a larger margin"


def test_margin_never_eats_into_a_neighbour() -> None:
    # Two sections 20 px apart: neither may grow more than 10 px towards the other.
    left = (100, 150, 220, 260)
    right = (240, 150, 360, 260)
    boxes = _boxes(_slide([left, right]), margin_frac=0.5, equalize_boxes=False)

    assert len(boxes) == 2
    assert boxes[0][2] <= boxes[1][0], "boxes must not overlap"
    assert boxes[0][2] <= left[2] + 10
    assert boxes[1][0] >= right[0] - 10


def test_margin_is_clipped_to_the_image() -> None:
    # A section hard against the top-left corner cannot grow off-canvas.
    boxes = _boxes(_slide([(0, 0, 150, 150)]), margin_frac=0.5,
                   equalize_boxes=False)
    assert boxes[0][0] >= 0 and boxes[0][1] >= 0


def test_margin_zero_is_the_old_behaviour() -> None:
    img = _slide([(200, 150, 320, 260)])
    assert _boxes(img, margin_frac=0.0, equalize_boxes=False) == _boxes(
        img, margin_frac=0.0, equalize_boxes=False
    )


def test_mask_is_unchanged_by_the_margin() -> None:
    """The margin is background, so the tissue mask must not grow with it."""
    img = _slide([(200, 150, 320, 260)])
    tight = detect_sections(img, min_area_px=500, opening_radius_px=0,
                            equalize_boxes=False, margin_frac=0.0)[0]
    padded = detect_sections(img, min_area_px=500, opening_radius_px=0,
                             equalize_boxes=False, margin_frac=0.15)[0]
    assert padded.mask.sum() == tight.mask.sum()
    assert padded.area_px == tight.area_px


def test_default_adds_a_margin() -> None:
    img = _slide([(200, 150, 320, 260)])
    default = _boxes(img, equalize_boxes=False)[0]
    tight = _boxes(img, margin_frac=0.0, equalize_boxes=False)[0]
    assert default[0] < tight[0], "detection should pad by default"


def test_margin_is_fast_with_many_boxes() -> None:
    """Detection on a noisy slide can return thousands of components.

    The first cut capped the margin with a Python-level pairwise loop, which took
    minutes on that many boxes and hung a GUI test outright.
    """
    import time

    from histo_to_ccf.sectioning.split import DetectedSection, _add_box_margin

    rng = np.random.default_rng(0)
    n = 2000
    xs = rng.integers(0, 4000, n)
    ys = rng.integers(0, 4000, n)
    dummy = np.zeros((1, 1), dtype=bool)
    boxes = [
        DetectedSection(
            bbox_px=(int(x), int(y), int(x) + 20, int(y) + 20),
            mask=dummy, area_px=400, centroid_px=(0.0, 0.0), aspect_ratio=1.0,
        )
        for x, y in zip(xs, ys)
    ]

    t0 = time.perf_counter()
    out = _add_box_margin(boxes, (4100, 4100), margin_frac=0.1)
    elapsed = time.perf_counter() - t0

    assert len(out) == n
    assert elapsed < 5.0, f"took {elapsed:.1f}s for {n} boxes"


def test_margin_on_empty_input() -> None:
    from histo_to_ccf.sectioning.split import _add_box_margin

    assert _add_box_margin([], (100, 100), margin_frac=0.1) == []


def test_single_box_uses_the_full_margin() -> None:
    """With no neighbour there is nothing to cap against."""
    from histo_to_ccf.sectioning.split import DetectedSection, _add_box_margin

    dummy = np.zeros((1, 1), dtype=bool)
    box = DetectedSection(bbox_px=(100, 100, 200, 200), mask=dummy, area_px=1,
                          centroid_px=(0.0, 0.0), aspect_ratio=1.0)
    out = _add_box_margin([box], (500, 500), margin_frac=0.1)
    assert out[0].bbox_px == (90, 90, 210, 210)
