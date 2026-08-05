"""Tests for group_fragmented_sections - re-joining pieces of a broken section."""
from __future__ import annotations

import numpy as np

from histo_to_ccf.sectioning.split import DetectedSection, group_fragmented_sections

SHAPE = (400, 400)


def _piece(x0, y0, x1, y1) -> DetectedSection:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[y0:y1, x0:x1] = True
    area = int(mask.sum())
    return DetectedSection(
        bbox_px=(x0, y0, x1, y1),
        mask=mask,
        area_px=area,
        centroid_px=((x0 + x1) / 2, (y0 + y1) / 2),
        aspect_ratio=max((x1 - x0) / (y1 - y0), (y1 - y0) / (x1 - x0)),
    )


def test_stacked_pieces_are_joined() -> None:
    # Cerebellum on top, detached brainstem directly below - same x span.
    cerebellum = _piece(100, 20, 300, 120)
    brainstem = _piece(120, 140, 280, 240)

    out = group_fragmented_sections([cerebellum, brainstem])

    assert len(out) == 1
    assert out[0].bbox_px == (100, 20, 300, 240)
    assert out[0].area_px == cerebellum.area_px + brainstem.area_px
    assert out[0].mask.sum() == out[0].area_px


def test_side_by_side_debris_is_kept_separate() -> None:
    section = _piece(100, 20, 300, 240)
    neighbour_debris = _piece(310, 30, 390, 120)  # beside it, not below

    out = group_fragmented_sections([section, neighbour_debris])

    assert len(out) == 2, "debris from an adjacent section must not be absorbed"
    assert out[0].bbox_px == section.bbox_px


def test_a_chain_of_three_pieces_collapses() -> None:
    top = _piece(100, 10, 300, 80)
    middle = _piece(110, 100, 290, 170)
    bottom = _piece(130, 190, 270, 260)

    out = group_fragmented_sections([top, middle, bottom])

    assert len(out) == 1
    assert out[0].bbox_px == (100, 10, 300, 260)


def test_two_separate_sections_stay_separate() -> None:
    left = _piece(20, 20, 180, 200)
    right = _piece(220, 20, 380, 200)

    out = group_fragmented_sections([left, right])

    assert len(out) == 2


def test_overlap_threshold_is_respected() -> None:
    main = _piece(100, 20, 300, 120)
    # Only a sliver of this piece's width sits under `main`.
    grazing = _piece(280, 140, 400, 240)

    strict = group_fragmented_sections([main, grazing], x_overlap_frac=0.5)
    loose = group_fragmented_sections([main, grazing], x_overlap_frac=0.1)

    assert len(strict) == 2
    assert len(loose) == 1


def test_results_are_ordered_largest_first() -> None:
    small = _piece(20, 20, 60, 60)
    big = _piece(200, 20, 380, 300)

    out = group_fragmented_sections([small, big])

    assert [s.area_px for s in out] == sorted((s.area_px for s in out), reverse=True)


def test_single_or_empty_input_is_returned_unchanged() -> None:
    one = _piece(10, 10, 50, 50)
    assert group_fragmented_sections([]) == []
    assert group_fragmented_sections([one]) == [one]
