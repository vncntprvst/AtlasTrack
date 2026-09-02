"""Tests for sectioning/ap_series.py - AP progression along a cutting series."""
from __future__ import annotations

import pytest

from atlastrack.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM
from atlastrack.project.schema import PlaneParams, Section
from atlastrack.sectioning.ap_series import ap_offsets, assign_section_ap


def _sections(slide_numbers, *, with_plane_ap=None):
    out = []
    for i, num in enumerate(slide_numbers):
        plane = None if with_plane_ap is None else PlaneParams(ap_um=with_plane_ap)
        out.append(
            Section(
                index=i,
                slide_idx=0,
                bbox_px=(0, 0, 10, 10),
                ap_order=i,
                slide_number=num,
                plane=plane,
            )
        )
    return out


# ---------------------------------------------------------------------------
# ap_offsets
# ---------------------------------------------------------------------------

def test_no_slide_numbers_falls_back_to_even_spacing() -> None:
    offsets, mode = ap_offsets([None] * 4, anchor_pos=0, spacing_um=100.0)
    assert mode == "ordinal"
    assert offsets == [0.0, 100.0, 200.0, 300.0]


def test_consecutive_slide_numbers_match_even_spacing() -> None:
    even, _ = ap_offsets([None] * 4, anchor_pos=0, spacing_um=100.0)
    by_slide, mode = ap_offsets([10, 11, 12, 13], anchor_pos=0, spacing_um=100.0)
    assert mode == "slide_number"
    assert by_slide == even


def test_gaps_in_slide_numbers_stretch_the_ap_series() -> None:
    # LO_05's actual anterior run: slides 76, 74, 72, then a jump to 59.
    offsets, mode = ap_offsets([76, 74, 72, 59], anchor_pos=0, spacing_um=50.0)
    assert mode == "slide_number"
    # Numbers descend along ap_order, so the trend flips them to ascend.
    assert offsets == [0.0, 100.0, 200.0, 850.0]


def test_ascending_and_descending_numbering_agree() -> None:
    desc, _ = ap_offsets([76, 74, 72, 59], anchor_pos=0, spacing_um=50.0)
    asc, _ = ap_offsets([18, 20, 22, 35], anchor_pos=0, spacing_um=50.0)
    assert desc == asc, "the numbering direction must not change the AP progression"


def test_anchor_position_shifts_the_origin() -> None:
    offsets, _ = ap_offsets([76, 74, 72, 59], anchor_pos=2, spacing_um=50.0)
    assert offsets[2] == 0.0
    assert offsets == [-200.0, -100.0, 0.0, 650.0]


def test_forward_false_reverses_the_direction() -> None:
    fwd, _ = ap_offsets([76, 74, 72, 59], anchor_pos=0, spacing_um=50.0, forward=True)
    rev, _ = ap_offsets([76, 74, 72, 59], anchor_pos=0, spacing_um=50.0, forward=False)
    assert rev == [-v for v in fwd]


def test_identical_slide_numbers_are_ignored() -> None:
    _, mode = ap_offsets([5, 5, 5], anchor_pos=0, spacing_um=100.0)
    assert mode == "ordinal", "a constant slide number carries no spacing information"


def test_partial_slide_numbers_fall_back_to_ordinal() -> None:
    _, mode = ap_offsets([76, None, 72], anchor_pos=0, spacing_um=100.0)
    assert mode == "ordinal"


def test_anchor_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="anchor_pos"):
        ap_offsets([1, 2, 3], anchor_pos=7, spacing_um=100.0)


def test_empty_series() -> None:
    assert ap_offsets([], anchor_pos=0, spacing_um=100.0) == ([], "ordinal")


# ---------------------------------------------------------------------------
# assign_section_ap
# ---------------------------------------------------------------------------

def test_assign_writes_ap_onto_every_section() -> None:
    sections = _sections([76, 74, 72, 59])
    n, mode = assign_section_ap(sections, spacing_um=50.0, anchor_ap_um=7000.0)
    assert (n, mode) == (4, "slide_number")
    assert [s.plane.ap_um for s in sections] == [7000.0, 7100.0, 7200.0, 7850.0]


def test_assign_keeps_the_anchor_section_ap_when_not_overridden() -> None:
    sections = _sections([76, 74, 72], with_plane_ap=9000.0)
    assign_section_ap(sections, spacing_um=50.0, anchor_index=1)
    assert sections[1].plane.ap_um == 9000.0
    assert sections[0].plane.ap_um == 8900.0
    assert sections[2].plane.ap_um == 9100.0


def test_assign_defaults_to_bregma_when_no_plane_exists() -> None:
    sections = _sections([None, None])
    assign_section_ap(sections, spacing_um=100.0)
    assert sections[0].plane.ap_um == BREGMA_AP_FROM_ORIGIN_UM


def test_assign_preserves_other_plane_fields() -> None:
    sections = _sections([1, 2])
    for s in sections:
        s.plane = PlaneParams(ap_um=0.0, ml_tilt_deg=3.5, pixel_size_um=6.25)
    assign_section_ap(sections, spacing_um=100.0, anchor_ap_um=7000.0)
    assert sections[1].plane.ml_tilt_deg == 3.5
    assert sections[1].plane.pixel_size_um == 6.25


def test_assign_follows_ap_order_not_list_order() -> None:
    sections = _sections([76, 74, 72])
    sections[0].ap_order, sections[2].ap_order = 2, 0  # reverse the series
    assign_section_ap(sections, spacing_um=50.0, anchor_ap_um=7000.0)
    # The section now first in ap_order (slide 72) holds the anchor AP.
    assert sections[2].plane.ap_um == 7000.0
    assert sections[0].plane.ap_um == 7200.0


def test_assign_empty_is_a_noop() -> None:
    assert assign_section_ap([], spacing_um=100.0) == (0, "ordinal")
