"""Manual per-section atlas correction: affine math, probe composition, schema."""
from __future__ import annotations

import numpy as np

from histo_to_ccf.atlas.planes import Anchoring
from histo_to_ccf.registration.manual import (
    invert_apply,
    is_identity,
    section_to_world,
    world_to_section,
)
from histo_to_ccf.registration.transforms import RegisteredSectionTransform


def test_world_section_roundtrip() -> None:
    origin = (120.0, 300.0)  # (y0, x0)
    a = np.array([[1.1, -0.05, 4.0], [0.05, 0.9, -7.0], [0.0, 0.0, 1.0]])
    world = section_to_world(a, origin)
    back = world_to_section(world, origin)
    assert np.allclose(back, a, atol=1e-9)


def test_identity_world_is_identity_section() -> None:
    origin = (50.0, 80.0)
    a = world_to_section(np.eye(3), origin)
    assert is_identity(a)


def test_invert_apply_undoes_translation() -> None:
    # Section-local affine that shifts the atlas +10 col, +4 row.
    a = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]])
    # A point the user clicks at (x=110, y=54) should map back to (100, 50).
    x, y = invert_apply(a, 110.0, 54.0)
    assert np.allclose([x, y], [100.0, 50.0])


def test_manual_affine_shifts_probe_mapping() -> None:
    """A pure-translation manual affine shifts the CCF result by the same pixels."""
    anchoring = Anchoring(ox=20.0, oy=0.0, oz=0.0, ux=0.0, uy=0.0, uz=80.0,
                          vx=0.0, vy=40.0, vz=0.0)
    base = RegisteredSectionTransform(
        anchoring=anchoring, output_size_px=(40, 80), bspline=None,
        atlas_resolution_um=(25.0, 25.0, 25.0),
    )
    # Manual affine: atlas dragged +8 col (x), +0 row (y).
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]])
    adjusted = RegisteredSectionTransform(
        anchoring=anchoring, output_size_px=(40, 80), bspline=None,
        atlas_resolution_um=(25.0, 25.0, 25.0), manual_affine=a,
    )
    # Clicking at x+8 on the adjusted map should equal clicking at x on the base.
    assert np.allclose(adjusted.apply(48.0, 20.0), base.apply(40.0, 20.0))


def test_schema_round_trips_manual_affine(tmp_path) -> None:
    from histo_to_ccf.project.io import load_project, save_project
    from histo_to_ccf.project.schema import AtlasRef, Project, Section, Slide

    a = [[1.2, 0.0, 3.0], [0.0, 0.8, -5.0], [0.0, 0.0, 1.0]]
    sec = Section(index=0, slide_idx=0, bbox_px=(0, 0, 80, 40), manual_affine=a)
    proj = Project(atlas=AtlasRef(), slides=[Slide(image_path="x.png", sections=[sec])])
    path = tmp_path / "p.histo2ccf.json"
    save_project(proj, path)
    loaded = load_project(path)
    assert loaded.slides[0].sections[0].manual_affine == a
    # Default stays None.
    sec2 = Section(index=1, slide_idx=0, bbox_px=(0, 0, 10, 10))
    assert sec2.manual_affine is None
