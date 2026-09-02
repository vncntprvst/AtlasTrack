"""Tests for mesh array extraction and registration overlay helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atlastrack.atlas.meshes import mesh_vertices_faces


class _FakeMeshio:
    """Mimics meshio.Mesh: vertices on .points, faces in .cells_dict."""

    def __init__(self, points, tris):
        self.points = np.asarray(points, dtype=float)
        self.cells_dict = {"triangle": np.asarray(tris, dtype=int)}


class _FakeTrimesh:
    """Mimics a trimesh-style mesh: .vertices / .faces."""

    def __init__(self, verts, faces):
        self.vertices = np.asarray(verts, dtype=float)
        self.faces = np.asarray(faces, dtype=int)


def test_mesh_vertices_faces_meshio() -> None:
    m = _FakeMeshio([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    v, f = mesh_vertices_faces(m)
    assert v.shape == (3, 3)
    assert f.tolist() == [[0, 1, 2]]


def test_mesh_vertices_faces_trimesh() -> None:
    m = _FakeTrimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    v, f = mesh_vertices_faces(m)
    assert v.shape == (3, 3)
    assert f.tolist() == [[0, 1, 2]]


def test_mesh_vertices_faces_missing_faces_raises() -> None:
    class _NoFaces:
        points = np.zeros((3, 3))

    with pytest.raises(ValueError):
        mesh_vertices_faces(_NoFaces())


class _FakeAtlas:
    """Minimal atlas stub for region lookup tests."""

    structures = {
        "IRN": {"rgb_triplet": [10, 20, 30]},
        "VII": {"rgb_triplet": [1, 2, 3]},
    }

    def structure_from_coords(self, coords, microns=False, as_acronym=False):
        ap, dv, ml = coords  # ASR order
        if ap < 0:
            return "Outside atlas"
        return "IRN" if dv > 5000 else "root"


def test_region_acronyms_at_points() -> None:
    from atlastrack.atlas.meshes import region_acronyms_at_points

    # points are (AP, ML, DV); the deep one (DV=6000) is in IRN.
    pts = [(5000, 5700, 6000), (5000, 5700, 100), (-1, 0, 0), (5000, 5700, 6000)]
    acrs = region_acronyms_at_points(_FakeAtlas(), pts)
    assert acrs == ["IRN"]  # root + outside excluded, deduped


def test_structure_rgb() -> None:
    from atlastrack.atlas.meshes import structure_rgb

    assert structure_rgb(_FakeAtlas(), "IRN") == (10, 20, 30)
    assert structure_rgb(_FakeAtlas(), "nonexistent") == (180, 180, 180)


def test_resolve_regions_tips_plus_extra_minus_context() -> None:
    from atlastrack.project.schema import ProbeSpec, ProbeType, Project, Shank
    from atlastrack.viz.plotly3d import resolve_regions

    project = Project(
        probes=[
            ProbeSpec(
                label="p",
                type=ProbeType(name="np1", n_shanks=1),
                shanks=[Shank(index=0, tip_ccf_um=(5000.0, 5700.0, 6000.0))],
            )
        ]
    )
    regions = resolve_regions(
        project, _FakeAtlas(), extra_regions=["VII", "Isocortex"]
    )
    # tip → IRN; extra adds VII; Isocortex dropped (it's a context shell region).
    assert regions == ["IRN", "VII"]


def test_deepslice_anchoring_permutation_and_flips() -> None:
    import pytest

    from atlastrack.registration.deepslice_adapter import _quicknii_to_atlas_anchoring

    # Real DeepSlice section-0 anchoring (QuickNII (ML, AP, DV) order).
    a = [379.03, 107.44, 270.03, -320.33, -29.73, -10.68, -6.81, 19.19, -231.72]
    out = _quicknii_to_atlas_anchoring(a, (528, 320, 456))  # 25 µm atlas → scale 1
    # Permute (x,y,z)->(AP,DV,ML), then flip AP (528-o, -u, -v) and DV (320-o).
    assert out == pytest.approx([528 - 107.44, 320 - 270.03, 379.03,
                                 29.73, 10.68, -320.33,
                                 -19.19, 231.72, -6.81])
    # Anatomy sanity: AP posterior, DV near the dorsal (top) end.
    assert out[0] > 400  # posterior brainstem section
    assert out[1] < 60   # dorsal surface near the top
    # u dominated by ML, v by DV - as sample_plane expects.
    assert abs(out[5]) == max(abs(out[3]), abs(out[4]), abs(out[5]))  # u → ML
    assert abs(out[7]) == max(abs(out[6]), abs(out[7]), abs(out[8]))  # v → DV


def test_sample_plane_out_dtype_matches_float_cast() -> None:
    from atlastrack.atlas.planes import Anchoring, sample_plane

    vol = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)  # (AP, DV, ML)
    anch = Anchoring(0, 0, 0, 0, 0, 3, 0, 2, 0)  # u→ML, v→DV
    direct = sample_plane(vol, anch, (3, 4), order=1, out_dtype=np.float32)
    cast_first = sample_plane(vol.astype(np.float32), anch, (3, 4), order=1)
    assert direct.dtype == np.float32
    np.testing.assert_allclose(direct, cast_first, atol=1e-4)


def test_predict_anchorings_runs_subprocess_and_parses(tmp_path, monkeypatch) -> None:
    import subprocess

    from atlastrack.io.quicknii import QuickNiiDocument, QuickNiiSlice, save_quicknii
    from atlastrack.registration import deepslice_adapter as ds

    class _Annot:
        shape = (528, 320, 456)

    class _Atlas:
        annotation = _Annot()

    anchoring = [379.03, 107.44, 270.03, -320.33, -29.73, -10.68, -6.81, 19.19, -231.72]

    def fake_run(cmd, **kwargs):
        # cmd = [exe, "-m", module, workdir, species]
        workdir = Path(cmd[-2])
        doc = QuickNiiDocument(
            slices=[QuickNiiSlice(
                filename="section_s000.png", nr=1, width=10, height=10,
                anchoring=anchoring,
            )]
        )
        save_quicknii(doc, workdir / "deepslice_predictions.json")

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    section_images = {0: np.zeros((10, 10), dtype=np.uint8)}
    out = ds.predict_anchorings(section_images, _Atlas(), workdir=tmp_path)
    assert set(out) == {0}
    expected = ds._quicknii_to_atlas_anchoring(anchoring, (528, 320, 456))
    assert out[0] == pytest.approx(expected)


def test_predict_anchorings_orders_by_ap_sequence(tmp_path, monkeypatch) -> None:
    """`order` numbers the DeepSlice files by AP rank; results map back to index."""
    import subprocess

    from atlastrack.io.quicknii import QuickNiiDocument, QuickNiiSlice, save_quicknii
    from atlastrack.registration import deepslice_adapter as ds

    class _Atlas:
        class annotation:
            shape = (528, 320, 456)

    # A distinct anchoring per token so we can verify the token->index mapping.
    def anchoring_for(tok: int) -> list[float]:
        return [float(tok), 0, 270, -320, 0, 0, 0, 0, -231]

    written: list[str] = []

    def fake_run(cmd, **kwargs):
        workdir = Path(cmd[-2])
        slices = []
        for p in sorted(workdir.glob("section_s*.png")):
            written.append(p.name)
            tok = ds._parse_section_index(p.name)
            slices.append(QuickNiiSlice(filename=p.name, nr=1, width=10, height=10,
                                        anchoring=anchoring_for(tok)))
        save_quicknii(QuickNiiDocument(slices=slices),
                      workdir / "deepslice_predictions.json")

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    section_images = {10: np.zeros((10, 10), np.uint8), 11: np.zeros((10, 10), np.uint8)}
    # User's AP sequence: section 11 is first (rank 0), section 10 second (rank 1).
    order = {11: 0, 10: 1}
    out = ds.predict_anchorings(section_images, _Atlas(), workdir=tmp_path, order=order)

    # Files were named by AP rank, not by section index.
    assert sorted(written) == ["section_s000.png", "section_s001.png"]
    # Token 0 (rank 0) maps back to section 11; token 1 to section 10. The token
    # rode in the ML origin (index 2) through the QuickNII->atlas conversion.
    assert out[11][2] == 0.0
    assert out[10][2] == 1.0


def test_deepslice_anchoring_scales_with_resolution() -> None:
    from atlastrack.registration.deepslice_adapter import _quicknii_to_atlas_anchoring

    a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    # u/v components scale linearly with resolution (origin has a flip offset,
    # so compare the direction vectors, which are pure scale).
    out = _quicknii_to_atlas_anchoring(a, (264, 160, 228))
    full = _quicknii_to_atlas_anchoring(a, (528, 320, 456))
    assert out[3:] == pytest.approx([v * 0.5 for v in full[3:]])


def test_deepslice_parse_section_index() -> None:
    from atlastrack.registration.deepslice_adapter import (
        _parse_section_index,
        _section_filename,
    )

    assert _section_filename(7) == "section_s007.png"
    assert _parse_section_index("section_s007.png") == 7
    assert _parse_section_index("/tmp/x/section_s012.png") == 12
    assert _parse_section_index(_section_filename(3)) == 3
    assert _parse_section_index("no_section_number.png") is None


def test_region_styling_palette_and_distinct_fallbacks() -> None:
    from atlastrack.viz.plotly3d import hex_to_rgb, region_style, styled_regions

    assert hex_to_rgb("#d4c8a8") == (0xD4, 0xC8, 0xA8)
    # Curated palette entries keep their fixed colour/opacity.
    assert region_style("VII") == ("#e05030", 0.40)
    assert region_style("Isocortex")[0] == "#d4c8a8"

    styled = styled_regions(["Isocortex", "FOO", "BS", "BAR", "FOO"])
    acronyms = [s[0] for s in styled]
    assert acronyms == ["Isocortex", "FOO", "BS", "BAR"]  # deduped, order kept
    colors = {a: c for a, c, _ in styled}
    # Palette regions use the palette; unknown siblings get DISTINCT fallbacks.
    assert colors["Isocortex"] == "#d4c8a8"
    assert colors["BS"] == "#b8a0c8"
    assert colors["FOO"] != colors["BAR"]


def test_annotation_boundaries() -> None:
    from atlastrack.registration.transforms import annotation_boundaries

    labels = np.zeros((5, 5), dtype=int)
    labels[:, 3:] = 7  # vertical region boundary between col 2 and 3
    edges = annotation_boundaries(labels)
    assert edges[:, 2].all() and edges[:, 3].all()  # both sides of the seam
    assert not edges[:, 0].any()  # interior of a uniform region has no edge
