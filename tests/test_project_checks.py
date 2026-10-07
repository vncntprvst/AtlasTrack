"""Which face the sections are seen from, and the check that planes agree."""
from __future__ import annotations

import pytest

from atlastrack.project.checks import mirrored_sections, mirrored_sections_note
from atlastrack.project.orientation import (
    implied_view,
    mirror_anchoring,
    project_view,
    set_view,
)
from atlastrack.project.schema import Project, RegistrationResult, Section, Slide


def _section(index: int, u_ml: float, *, registered: bool = True) -> Section:
    anchoring = [400.0, 0.0, 228 - u_ml / 2, 0.0, 0.0, u_ml, 0.0, 320.0, 0.0]
    return Section(
        index=index,
        slide_idx=0,
        bbox_px=(0, 0, 10, 10),
        registration=RegistrationResult(anchoring=anchoring, output_size_px=(10, 10))
        if registered else None,
        deepslice_anchoring=None if registered else anchoring,
    )


def _project(*u_ml: float, seen_from=None) -> Project:
    return Project(seen_from=seen_from, slides=[Slide(image_path="x.tif", sections=[
        _section(i, u) for i, u in enumerate(u_ml)
    ])])


def test_an_unset_view_follows_the_planes() -> None:
    project = _project(-456.0, -456.0, 456.0)
    assert implied_view(project) == "back"
    assert mirrored_sections(project) == [2]
    assert implied_view(_project()) is None
    assert project_view(_project(), default="back") == "back"


def test_planes_are_checked_against_the_set_view() -> None:
    project = _project(-456.0, -300.0, seen_from="front")
    assert mirrored_sections(project) == [0, 1]
    note = mirrored_sections_note(project)
    assert note is not None and "(0, 1)" in note and "seen from the front" in note
    assert mirrored_sections_note(_project(-456.0, seen_from="back")) is None


def test_set_view_mirrors_the_planes_placed_for_the_other_face() -> None:
    project = _project(456.0, -456.0)
    before = project.slides[0].sections[0].registration.anchoring
    mirrored, skipped = set_view(project, "back", mirror_planes=True)
    assert mirrored == [0] and skipped == []
    assert project.slides[0].sections[0].registration.anchoring == pytest.approx(
        mirror_anchoring(before, 456.0)
    )
    assert mirrored_sections(project) == []


def test_mirroring_keeps_ap_and_dv_and_mirrors_ml() -> None:
    a = [400.0, 10.0, 30.0, 2.0, 1.0, 400.0, -1.0, 300.0, 5.0]
    m = mirror_anchoring(a, 456.0)
    for su, sv in ((0.0, 0.0), (0.3, 0.7), (1.0, 1.0)):
        p = [a[k] + su * a[3 + k] + sv * a[6 + k] for k in range(3)]
        q = [m[k] + su * m[3 + k] + sv * m[6 + k] for k in range(3)]
        assert q[0] == pytest.approx(p[0]) and q[1] == pytest.approx(p[1])
        assert q[2] == pytest.approx(456.0 - p[2])
