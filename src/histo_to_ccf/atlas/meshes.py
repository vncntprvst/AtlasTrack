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


def region_acronyms_at_points(atlas, points_ap_ml_dv_um, *, exclude=("root",)):
    """Region acronyms at a list of CCF ``(AP, ML, DV)`` µm points.

    The atlas indexes coordinates in ASR order ``(AP, DV, ML)``, so each point is
    reordered before lookup. Points outside the atlas (or in ``exclude``) are
    dropped. Returns a de-duplicated list preserving first-seen order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for p in points_ap_ml_dv_um:
        if p is None:
            continue
        ap, ml, dv = float(p[0]), float(p[1]), float(p[2])
        try:
            acr = atlas.structure_from_coords((ap, dv, ml), microns=True, as_acronym=True)
        except Exception:
            continue
        if not acr or acr == "Outside atlas" or acr in exclude or acr in seen:
            continue
        seen.add(acr)
        out.append(acr)
    return out


def structure_rgb(atlas, acronym, default=(180, 180, 180)) -> tuple[int, int, int]:
    """Return the atlas's RGB triplet (0–255) for a structure acronym."""
    try:
        rgb = atlas.structures[acronym]["rgb_triplet"]
        return int(rgb[0]), int(rgb[1]), int(rgb[2])
    except Exception:
        return default
