"""User-assigned AP guides DeepSlice: its predicted AP is anchored to the values."""
from __future__ import annotations

import numpy as np

import atlastrack.registration.pipeline as pipeline
from atlastrack.atlas.planes import Anchoring
from atlastrack.project.schema import PlaneParams, Project, Section, Slide


class _FakeAtlas:
    """anchoring_from_plane_params is monkeypatched, so the atlas is unused."""


def _anchoring9(ap_origin: float) -> list[float]:
    # ox=ap_origin, oy=oz=0, ux=20 (AP tilt across plane), rest 0, vx=0.
    return [ap_origin, 0, 0, 20.0, 0, 0, 0.0, 0, 0]


def _project_with_planes(planes: dict[int, float], sources: dict | None = None) -> Project:
    sources = sources or {}
    secs = [
        Section(index=i, slide_idx=0, bbox_px=(0, 0, 10, 10),
                plane=PlaneParams(ap_um=ap) if ap is not None else None,
                ap_source=sources.get(i, "manual" if ap is not None else None))
        for i, ap in planes.items()
    ]
    return Project(slides=[Slide(image_path="x.png", sections=secs)])


def _patch_user_anchoring(monkeypatch, mapping: dict[float, float]) -> None:
    """Make anchoring_from_plane_params return an anchoring whose AP centre is
    the user's desired voxel AP (looked up from the plane's ap_um)."""
    def fake(atlas, plane):
        center = mapping[plane.ap_um]
        return Anchoring.from_iterable([center - 10.0, 0, 0, 20.0, 0, 0, 0.0, 0, 0])
    monkeypatch.setattr(pipeline, "anchoring_from_plane_params", fake)


def test_single_anchor_offsets_all_planes(monkeypatch) -> None:
    # DeepSlice predicts AP centres 100, 110, 120 for sections 0,1,2.
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0), 2: _anchoring9(110.0)}
    # section 0 AP centres at 100 (ox 90 + 0.5*20). User wants it at 300.
    proj = _project_with_planes({0: -5300.0, 1: None, 2: None})
    _patch_user_anchoring(monkeypatch, {-5300.0: 300.0})

    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    # Offset = +200 applied to every section's AP centre; spacing preserved.
    assert pipeline._ap_center(out[0]) == 300.0
    assert pipeline._ap_center(out[1]) == 310.0
    assert pipeline._ap_center(out[2]) == 320.0


def test_two_anchors_rescale_ap(monkeypatch) -> None:
    # DeepSlice centres: 100, 110, 120. User anchors sections 0 and 2.
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0), 2: _anchoring9(110.0)}
    proj = _project_with_planes({0: -5000.0, 1: None, 2: -5400.0})
    # Want section0 centre -> 100, section2 centre -> 140 (double the 20-unit gap).
    _patch_user_anchoring(monkeypatch, {-5000.0: 100.0, -5400.0: 140.0})

    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    assert np.isclose(pipeline._ap_center(out[0]), 100.0)
    assert np.isclose(pipeline._ap_center(out[2]), 140.0)
    # Middle section interpolated by the same line (slope 2): centre 110 -> 120.
    assert np.isclose(pipeline._ap_center(out[1]), 120.0)


def test_assigned_sections_land_exactly_even_when_deepslice_is_noisy(monkeypatch) -> None:
    """Each pinned section lands on its AP exactly (not a best-fit) - the user's ask."""
    # DeepSlice predicts NON-collinear AP centres (noisy): 100, 137, 119.
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(127.0), 2: _anchoring9(109.0)}
    # User pins all three to a clean sequence 200, 210, 220.
    proj = _project_with_planes({0: -5000.0, 1: -5100.0, 2: -5200.0})
    _patch_user_anchoring(monkeypatch, {-5000.0: 200.0, -5100.0: 210.0, -5200.0: 220.0})

    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    # Despite DeepSlice's noise, every pinned section sits EXACTLY on its value.
    assert pipeline._ap_center(out[0]) == 200.0
    assert pipeline._ap_center(out[1]) == 210.0
    assert pipeline._ap_center(out[2]) == 220.0


def test_unassigned_section_interpolates_shift(monkeypatch) -> None:
    """An unpinned section is shifted by interpolating the pinned sections' shifts."""
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0), 2: _anchoring9(110.0)}
    # ds centres 100, 110, 120; pin only 0 and 2 (shifts +100 and +120).
    proj = _project_with_planes({0: -5000.0, 1: None, 2: -5200.0})
    _patch_user_anchoring(monkeypatch, {-5000.0: 200.0, -5200.0: 240.0})

    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    assert pipeline._ap_center(out[0]) == 200.0          # pinned exact
    assert pipeline._ap_center(out[2]) == 240.0          # pinned exact
    # section 1 (ds 110) shift interpolated between +100 and +120 -> +110 -> 220.
    assert np.isclose(pipeline._ap_center(out[1]), 220.0)


def test_after_a_pre_match_hand_edits_carry_the_neighbours(monkeypatch) -> None:
    """Pre-match gives every section a DeepSlice AP; those are not anchors.

    Sections 0 and 2 are corrected by hand (+100 and +120). Section 1 still has
    the AP the pre-match wrote, and must follow its corrected neighbours (+110)
    instead of staying pinned at DeepSlice's raw value.
    """
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0), 2: _anchoring9(110.0)}
    proj = _project_with_planes({0: -5000.0, 1: -5100.0, 2: -5200.0},
                                sources={1: "deepslice"})
    _patch_user_anchoring(monkeypatch, {-5000.0: 200.0, -5100.0: 110.0, -5200.0: 240.0})

    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    assert pipeline._ap_center(out[0]) == 200.0
    assert pipeline._ap_center(out[2]) == 240.0
    assert np.isclose(pipeline._ap_center(out[1]), 220.0)


def test_even_spacing_and_unrecorded_sources_still_pin(monkeypatch) -> None:
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0)}
    proj = _project_with_planes({0: -5000.0, 1: -5100.0},
                                sources={0: "even_spacing", 1: None})
    _patch_user_anchoring(monkeypatch, {-5000.0: 150.0, -5100.0: 300.0})

    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    assert pipeline._ap_center(out[0]) == 150.0
    assert pipeline._ap_center(out[1]) == 300.0


def test_a_pre_match_alone_is_left_as_predicted() -> None:
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0)}
    proj = _project_with_planes({0: -5000.0, 1: -5100.0},
                                sources={0: "deepslice", 1: "deepslice"})
    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    assert out is anchorings


def test_no_anchors_returns_unchanged() -> None:
    anchorings = {0: _anchoring9(90.0), 1: _anchoring9(100.0)}
    proj = _project_with_planes({0: None, 1: None})
    out = pipeline.guide_anchorings_with_planes(anchorings, proj, _FakeAtlas())
    assert out is anchorings


def test_deepslice_anchoring_used_when_present() -> None:
    sec = Section(index=0, slide_idx=0, bbox_px=(0, 0, 10, 10),
                  plane=PlaneParams(ap_um=-5300.0))
    anch = pipeline.anchoring_for_section(sec, {0: _anchoring9(42.0)}, _FakeAtlas())
    assert isinstance(anch, Anchoring) and anch.ox == 42.0
