"""Turn BrainGlobe region meshes into plain vertex/face arrays.

BrainGlobe's ``BrainGlobeAtlas.mesh_from_structure`` returns a
:class:`meshio.Mesh`, whose vertices live on ``.points`` and whose triangular
faces live on ``.cells_dict['triangle']``. Earlier visualization code assumed
``.vertices`` / ``.faces`` (a trimesh-style API meshio does not provide), which
raised ``'Mesh' object has no attribute 'vertices'`` on export and in 3D view.
This helper normalizes both shapes.

Pure-core module — no napari / Qt imports (import-linter contract).
"""
from __future__ import annotations

import numpy as np


def mesh_vertices_faces(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices (N, 3), faces (M, 3))`` from a BrainGlobe/meshio mesh.

    Accepts a ``meshio.Mesh`` (``.points`` + ``.cells_dict['triangle']``) or any
    object exposing trimesh-style ``.vertices`` / ``.faces``.

    Raises
    ------
    ValueError
        If no vertices or no triangular faces can be found.
    """
    verts = getattr(mesh, "points", None)
    if verts is None:
        verts = getattr(mesh, "vertices", None)
    if verts is None:
        raise ValueError("mesh has no vertices (.points / .vertices)")
    verts = np.asarray(verts, dtype=float)

    faces = None
    cells_dict = getattr(mesh, "cells_dict", None)
    if isinstance(cells_dict, dict) and "triangle" in cells_dict:
        faces = cells_dict["triangle"]
    if faces is None:
        faces = getattr(mesh, "faces", None)
    if faces is None:
        # meshio < cells_dict, or a raw CellBlock list.
        for block in getattr(mesh, "cells", []) or []:
            if getattr(block, "type", None) == "triangle":
                faces = block.data
                break
    if faces is None:
        raise ValueError("mesh has no triangular faces")
    return verts, np.asarray(faces, dtype=int)
