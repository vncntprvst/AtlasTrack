"""Bregma is per-atlas, not a universal constant.

Written after finding that selecting "CCFv3-BBP Augmented" silently shifted every
bregma-relative and Paxinos AP by ~346 µm: the augmented CCFv3 is Allen CCFv3 padded
to 566 AP slices, and the code carried Allen's 5400 µm anchor as a module constant.

The 5746 µm figure is measured, not assumed - see the derivation note on
:data:`BREGMA_AP_BY_ATLAS`. ``test_the_augmented_offset_matches_the_measurement``
re-derives it from the real atlases when they are downloaded.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from histo_to_ccf.io.ccf_coords import (
    BREGMA_AP_BY_ATLAS,
    BREGMA_AP_FROM_ORIGIN_UM,
    MIDLINE_ML_UM,
    AtlasAnchors,
    anchors_for_atlas,
    anchors_for_atlas_name,
    bregma_ap_for_display,
    bregma_ap_from_origin_um,
    ccf_um_to_paxinos_mm,
)

# The measured shift of the BBP augmented CCFv3 relative to Allen (µm), and the
# tolerance the 25-nucleus measurement supports (sd 2.5 µm, so 15 µm is generous).
AUGMENTED_SHIFT_UM = 346.0
SHIFT_TOL_UM = 15.0


class _FakeAtlas:
    """Enough of BrainGlobeAtlas for the anchor derivation."""

    def __init__(self, name, shape, resolution=(25.0, 25.0, 25.0)):
        self.atlas_name = name
        self.annotation = np.zeros(shape, dtype=np.int32)
        self.resolution = resolution


# ------------------------------------------------------------------- the lookup


def test_every_allen_resolution_shares_the_one_anchor():
    """The table is keyed by family: bregma does not move when you resample."""
    for name in ("allen_mouse_10um", "allen_mouse_25um", "allen_mouse_100um"):
        assert bregma_ap_from_origin_um(name) == BREGMA_AP_FROM_ORIGIN_UM


def test_the_augmented_atlas_does_not_inherit_allens_anchor():
    """The bug this module exists for."""
    allen = bregma_ap_from_origin_um("allen_mouse_25um")
    augmented = bregma_ap_from_origin_um("ccfv3augmented_mouse_25um")

    assert augmented != allen
    assert augmented - allen == pytest.approx(AUGMENTED_SHIFT_UM, abs=SHIFT_TOL_UM)


def test_kim_shares_allens_frame_because_it_is_a_reannotation():
    """Same 528x320x456 grid, same space - only the labels differ."""
    assert bregma_ap_from_origin_um("kim_mouse_25um") == BREGMA_AP_FROM_ORIGIN_UM


def test_an_unknown_atlas_returns_none_rather_than_allens_value():
    """Silently substituting Allen's anchor is how the augmented bug happened."""
    assert bregma_ap_from_origin_um("perens_stereotaxic_mri_mouse_25um") is None
    assert bregma_ap_from_origin_um("whitewashed_zebrafish_1um") is None
    assert bregma_ap_from_origin_um(None) is None
    assert bregma_ap_from_origin_um("") is None


def test_the_lookup_is_case_and_whitespace_tolerant():
    assert bregma_ap_from_origin_um("  Allen_Mouse_25um ") == BREGMA_AP_FROM_ORIGIN_UM


def test_a_longer_prefix_wins_over_a_shorter_one():
    """Guards the table against a future entry that is a prefix of another."""
    table = dict(BREGMA_AP_BY_ATLAS)
    for key in table:
        others = [k for k in table if k != key and k.startswith(key)]
        assert not others, f"{key!r} shadows {others!r}; the sort must break the tie"


# ------------------------------------------------------------------- the anchors


def test_a_known_atlas_yields_a_usable_bregma():
    anchors = anchors_for_atlas_name("allen_mouse_25um")

    assert anchors.has_bregma
    assert anchors.require_bregma() == BREGMA_AP_FROM_ORIGIN_UM
    assert anchors.midline_ml_um == MIDLINE_ML_UM


def test_an_unknown_atlas_raises_a_message_naming_the_atlas_and_the_fix():
    anchors = anchors_for_atlas_name("mystery_mouse_25um")

    assert not anchors.has_bregma
    with pytest.raises(ValueError) as exc:
        anchors.require_bregma()
    assert "mystery_mouse_25um" in str(exc.value)
    assert "BREGMA_AP_BY_ATLAS" in str(exc.value)


def test_extents_come_off_the_grid_when_the_atlas_is_loaded():
    """The augmented atlas is 566 slices, so its AP extent is not Allen's 13200."""
    anchors = anchors_for_atlas(_FakeAtlas("ccfv3augmented_mouse_25um", (566, 320, 456)))

    assert anchors.ap_um == pytest.approx(14150.0)
    assert anchors.dv_um == pytest.approx(8000.0)
    assert anchors.ml_um == pytest.approx(11400.0)


def test_the_midline_is_half_the_ml_extent():
    """Exact for the symmetric mouse atlases; all three give the familiar 5700."""
    for name, shape in (
        ("allen_mouse_25um", (528, 320, 456)),
        ("ccfv3augmented_mouse_25um", (566, 320, 456)),
        ("kim_mouse_25um", (528, 320, 456)),
    ):
        anchors = anchors_for_atlas(_FakeAtlas(name, shape))
        assert anchors.midline_ml_um == pytest.approx(MIDLINE_ML_UM)


def test_a_non_isotropic_atlas_uses_each_axis_own_resolution():
    anchors = anchors_for_atlas(
        _FakeAtlas("allen_mouse_25um", (100, 50, 40), resolution=(10.0, 20.0, 25.0))
    )

    assert anchors.ap_um == pytest.approx(1000.0)
    assert anchors.dv_um == pytest.approx(1000.0)
    assert anchors.ml_um == pytest.approx(1000.0)
    assert anchors.midline_ml_um == pytest.approx(500.0)


# ------------------------------------------------------------------- the display


def test_the_display_helper_never_returns_none():
    """A spin box has to show a number even for an atlas we cannot anchor."""
    assert bregma_ap_for_display("mystery_mouse_25um") == BREGMA_AP_FROM_ORIGIN_UM
    assert bregma_ap_for_display(None) == BREGMA_AP_FROM_ORIGIN_UM


def test_the_display_helper_still_prefers_a_known_anchor():
    assert bregma_ap_for_display("ccfv3augmented_mouse_25um") == pytest.approx(
        BREGMA_AP_FROM_ORIGIN_UM + AUGMENTED_SHIFT_UM, abs=SHIFT_TOL_UM
    )


# ------------------------------------------------------------------- the transform


def test_paxinos_without_anchors_is_unchanged():
    """Back-compat: the Allen default must not move."""
    plain = ccf_um_to_paxinos_mm(5400.0, 5700.0, 440.0, alignment="qiu2018")
    allen = ccf_um_to_paxinos_mm(
        5400.0, 5700.0, 440.0, alignment="qiu2018",
        anchors=anchors_for_atlas_name("allen_mouse_25um"),
    )

    np.testing.assert_allclose(plain, allen)


def test_the_same_structure_lands_at_the_same_stereotaxic_ap_in_both_atlases():
    """The point of the fix, stated as the invariant it restores.

    A nucleus sits 346 µm further along the augmented atlas's AP axis, so feeding
    that shifted CCF coordinate through the augmented anchors must return the same
    bregma-relative millimetre as the Allen coordinate through Allen's.
    """
    allen_ap, ml, dv = 8963.0, 5700.0, 3000.0
    aug_ap = allen_ap + AUGMENTED_SHIFT_UM

    a = ccf_um_to_paxinos_mm(
        allen_ap, ml, dv, anchors=anchors_for_atlas_name("allen_mouse_25um")
    )
    b = ccf_um_to_paxinos_mm(
        aug_ap, ml, dv, anchors=anchors_for_atlas_name("ccfv3augmented_mouse_25um")
    )

    np.testing.assert_allclose(a, b, atol=1e-6)


def test_using_allens_anchor_on_the_augmented_atlas_is_off_by_the_shift():
    """What the bug actually cost, in the units the CSV reports."""
    aug_ap = 8963.0 + AUGMENTED_SHIFT_UM

    wrong, _, _ = ccf_um_to_paxinos_mm(aug_ap, 5700.0, 3000.0, alignment="none")
    right, _, _ = ccf_um_to_paxinos_mm(
        aug_ap, 5700.0, 3000.0, alignment="none",
        anchors=anchors_for_atlas_name("ccfv3augmented_mouse_25um"),
    )

    assert abs(float(right) - float(wrong)) == pytest.approx(
        AUGMENTED_SHIFT_UM / 1000.0, abs=SHIFT_TOL_UM / 1000.0
    )


def test_paxinos_refuses_an_atlas_it_cannot_anchor():
    with pytest.raises(ValueError, match="No bregma anchor"):
        ccf_um_to_paxinos_mm(
            5400.0, 5700.0, 440.0, anchors=anchors_for_atlas_name("mystery_mouse")
        )


def test_the_none_alignment_keeps_dv_untouched_even_with_anchors():
    """"none" is the no-correction baseline; anchors must not smuggle a DV shift in."""
    _, _, dv = ccf_um_to_paxinos_mm(
        5400.0, 5700.0, 3000.0, alignment="none",
        anchors=anchors_for_atlas_name("allen_mouse_25um"),
    )

    assert float(dv) == pytest.approx(3.0)


def test_anchors_are_frozen_so_a_caller_cannot_corrupt_the_table():
    anchors = anchors_for_atlas_name("allen_mouse_25um")

    with pytest.raises(dataclasses.FrozenInstanceError):
        anchors.bregma_ap_um = 1.0  # type: ignore[misc]
    assert isinstance(anchors, AtlasAnchors)


# ------------------------------------------------------------------- real atlases


@pytest.mark.slow
def test_the_augmented_offset_matches_the_measurement():
    """Re-derive the tabulated shift from the atlases themselves.

    Skipped unless both atlases are already downloaded - this must never trigger a
    network fetch. Uses compact nuclei rather than parent regions: the augmented
    atlas redraws MY/CB/CTX, so their centroids move by +438..+567 µm while the
    nuclei inside them all move by the rigid +346.
    """
    pytest.importorskip("brainglobe_atlasapi")
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_atlasapi.list_atlases import get_downloaded_atlases

    have = set(get_downloaded_atlases())
    needed = {"allen_mouse_25um", "ccfv3augmented_mouse_25um"}
    if not needed <= have:
        pytest.skip(f"needs {sorted(needed - have)} downloaded")

    nuclei = ["VII", "IO", "PG", "XII", "LC", "SNr", "VTA", "IP", "DN", "FN"]

    def centroid(atlas, acronym):
        sid = atlas.structures[acronym]["id"]
        desc = list(atlas.structures.tree.expand_tree(sid))
        mask = np.isin(atlas.annotation, desc)
        profile = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(float)
        ap = np.arange(profile.size) * atlas.resolution[0]
        return float((profile * ap).sum() / profile.sum())

    allen = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    aug = BrainGlobeAtlas("ccfv3augmented_mouse_25um", check_latest=False)
    deltas = [centroid(aug, a) - centroid(allen, a) for a in nuclei]

    measured = float(np.median(deltas))
    assert measured == pytest.approx(AUGMENTED_SHIFT_UM, abs=SHIFT_TOL_UM)
    # A rigid shift, not a stretch: every nucleus must agree closely.
    assert float(np.std(deltas)) < 10.0
    tabulated = bregma_ap_from_origin_um("ccfv3augmented_mouse_25um")
    assert tabulated - BREGMA_AP_FROM_ORIGIN_UM == pytest.approx(measured, abs=SHIFT_TOL_UM)
