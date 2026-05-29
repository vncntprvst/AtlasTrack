"""Tests for mesh array extraction and registration overlay helpers."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.atlas.meshes import mesh_vertices_faces


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


def test_annotation_boundaries() -> None:
    from histo_to_ccf.registration.transforms import annotation_boundaries

    labels = np.zeros((5, 5), dtype=int)
    labels[:, 3:] = 7  # vertical region boundary between col 2 and 3
    edges = annotation_boundaries(labels)
    assert edges[:, 2].all() and edges[:, 3].all()  # both sides of the seam
    assert not edges[:, 0].any()  # interior of a uniform region has no edge
