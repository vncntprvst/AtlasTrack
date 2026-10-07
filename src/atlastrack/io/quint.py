"""Import a QUINT registration: QuickNII, DeepSlice or VisuAlign.

The QUINT tools keep a whole series in one JSON file next to the section images::

    {"target": "ABA_Mouse_CCFv3_2017_25um.cutlas", "target-resolution": [456, 528, 320],
     "slices": [{"filename": "brain_s001.png", "nr": 1, "width": 1500, "height": 1000,
                 "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz],
                 "markers": [[x, y, x', y'], ...]}, ...]}

``anchoring`` places the section on a plane (QuickNII's linear fit, or DeepSlice's
prediction); VisuAlign's ``markers`` then move the point the plane puts at
``(x, y)`` to ``(x', y')``, in the slice's ``width`` x ``height`` pixels. VisuAlign
workspace files (``.waln``) say ``sections`` and ``ouv`` instead.

The markers deform the plane piecewise-linearly: a Delaunay triangulation of the
moved positions, with the four corners 10% outside the image pinned in place,
and each triangle mapped as a whole (as PyNutil, the QUINT developers' Python
reader, does it). AtlasTrack's landmark correction is a smooth spline and cannot
reproduce that, so the deformation comes in as the section's warp instead: a
displacement field in the same file a registration writes. AtlasTrack's own
landmarks can still be added on top.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from atlastrack.io.quicknii import quicknii_to_atlas_anchoring
from atlastrack.io.section_import import ImportedSection, write_imported_project

#: The pinned corners sit this far outside the image, as a fraction of its size.
CORNER_MARGIN = 0.1

#: Allen CCFv3 extent in µm, (AP, DV, ML): the grid every CCF-space atlas shares.
_CCF_EXTENT_UM = (13200.0, 8000.0, 11400.0)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def atlas_for_target(target: str, resolution_um: float = 25.0) -> str:
    """The BrainGlobe atlas for a QUINT ``target``, or an error saying why not."""
    t = target.lower()
    res = f"{resolution_um:g}um"
    if "kim" in t:
        return f"kim_mouse_{res}"
    if "aba_mouse" in t or "allen" in t or "ccfv3" in t:
        return f"allen_mouse_{res}"
    raise ValueError(
        f"QUINT atlas {target!r} is not supported: only atlases in the Allen CCFv3 "
        "space (Allen, Kim) can be imported so far."
    )


def triangulation(width: float, height: float, markers: np.ndarray):
    """Triangles over the markers' moved positions plus the four pinned corners.

    Returns ``(original, moved, simplices)``: vertex positions before and after
    the markers, and the vertex indices of each triangle.
    """
    from scipy.spatial import Delaunay

    m = CORNER_MARGIN
    corners = np.array([
        [-m * width, -m * height], [(1 + m) * width, -m * height],
        [-m * width, (1 + m) * height], [(1 + m) * width, (1 + m) * height],
    ])
    markers = np.asarray(markers, dtype=float).reshape(-1, 4)
    original = np.vstack([corners, markers[:, :2]])
    moved = np.vstack([corners, markers[:, 2:4]])
    return original, moved, Delaunay(moved).simplices


def deform_forward(width, height, markers, x, y):
    """Where the markers move plane positions ``(x, y)`` (slice pixels).

    Each point is located in the triangles laid over the *original* positions and
    carried to the same spot of the moved triangle. Points outside every triangle
    (beyond the pinned corners) stay put.
    """
    original, moved, simplices = triangulation(width, height, markers)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    out_x, out_y = x.copy(), y.copy()
    todo = np.ones(x.shape, dtype=bool)
    for tri in simplices:
        if not todo.any():
            break
        a, b, c = original[tri]
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        det = np.linalg.det(m)
        if abs(det) < 1e-12:
            continue
        idx = np.nonzero(todo)[0]
        rel = np.stack([x[idx] - a[0], y[idx] - a[1]])
        u, v = np.linalg.solve(m, rel)
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        if not inside.any():
            continue
        hit = idx[inside]
        a2, b2, c2 = moved[tri]
        out_x[hit] = a2[0] + (b2[0] - a2[0]) * u[inside] + (c2[0] - a2[0]) * v[inside]
        out_y[hit] = a2[1] + (b2[1] - a2[1]) * u[inside] + (c2[1] - a2[1]) * v[inside]
        todo[hit] = False
    return out_x, out_y


def marker_warp(
    markers, reg_size: tuple[float, float], image_size: tuple[int, int]
) -> np.ndarray:
    """VisuAlign's deformation as a displacement field on the image's pixel grid.

    ``reg_size`` is the slice's ``(width, height)`` in the JSON, which the markers
    use; ``image_size`` the image file's ``(w, h)``. The field gives, for each
    image pixel ``p`` on the plane, ``moved(p) - p`` - the warp AtlasTrack stores
    (atlas-slice pixel -> section pixel).
    """
    rw, rh = reg_size
    w, h = image_size
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sx, sy = rw / w, rh / h
    mx, my = deform_forward(rw, rh, markers, xx * sx, yy * sy)
    field = np.empty((h, w, 2), dtype=np.float64)
    field[..., 0] = mx.reshape(h, w) / sx - xx
    field[..., 1] = my.reshape(h, w) / sy - yy
    return field


def deform_inverse(width, height, markers, x, y):
    """Where moved positions ``(x, y)`` came from: the inverse of :func:`deform_forward`.

    The triangles are laid over the moved positions, where they never overlap, so
    unlike the forward direction this one is defined everywhere inside them.
    """
    original, moved, _ = triangulation(width, height, markers)
    from scipy.spatial import Delaunay

    tri = Delaunay(moved)
    pts = np.c_[np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()]
    simplex = tri.find_simplex(pts)
    out = pts.copy()
    inside = simplex >= 0
    t = tri.transform[simplex[inside]]
    bary = np.einsum("nij,nj->ni", t[:, :2], pts[inside] - t[:, 2])
    weights = np.c_[bary, 1 - bary.sum(axis=1)]
    out[inside] = np.einsum("ni,nij->nj", weights, original[tri.simplices[simplex[inside]]])
    return out[:, 0], out[:, 1]


def marker_warp_inverse(
    markers, reg_size: tuple[float, float], image_size: tuple[int, int]
) -> np.ndarray:
    """The exact inverse of :func:`marker_warp`, on the same grid: for each image
    pixel ``q``, ``original(q) - q``. Section points are mapped with this one, as
    PyNutil maps them, so they are right even where the forward warp folds."""
    rw, rh = reg_size
    w, h = image_size
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    sx, sy = rw / w, rh / h
    ox, oy = deform_inverse(rw, rh, markers, xx * sx, yy * sy)
    field = np.empty((h, w, 2), dtype=np.float64)
    field[..., 0] = ox.reshape(h, w) / sx - xx
    field[..., 1] = oy.reshape(h, w) / sy - yy
    return field


def _find_image(folder: Path, filename: str) -> Path:
    path = folder / filename
    if path.is_file():
        return path
    stem = Path(filename).stem
    for p in sorted(folder.glob(f"{stem}.*")):
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            return p
    raise FileNotFoundError(f"image {filename!r} is not in {folder}")


def _series_number(entry: dict, filename: str, fallback: int) -> int:
    if entry.get("nr"):
        return int(entry["nr"])
    m = re.search(r"_s(\d+)", Path(filename).stem)
    return int(m.group(1)) if m else fallback


def import_quint(
    series_json: str | Path,
    out_dir: str | Path | None = None,
    *,
    image_dir: str | Path | None = None,
    resolution_um: float = 25.0,
) -> Path:
    """Write an AtlasTrack project from a QuickNII/DeepSlice/VisuAlign JSON file.

    Section images are looked up next to the JSON (or in ``image_dir``) by the
    file names it lists. Nothing there is changed; the output goes to ``out_dir``
    (default: a folder named after the JSON file, ``_atlastrack`` added, beside it).
    Returns the path of the written ``.atlastrack.json``.
    """
    from atlastrack.io.image import load_image

    series_json = Path(series_json).resolve()
    data = json.loads(series_json.read_text(encoding="utf-8"))
    workspace = "sections" in data and "slices" not in data
    entries = data.get("sections" if workspace else "slices") or []
    key = "ouv" if workspace else "anchoring"
    entries = [e for e in entries if e.get(key)]
    if not entries:
        raise ValueError(f"{series_json} has no anchored sections")
    entries.sort(key=lambda e: _series_number(e, e.get("filename", ""), 0))

    atlas_name = atlas_for_target(str(data.get("target", "ABA_Mouse_CCFv3")), resolution_um)
    atlas_shape = tuple(round(e / resolution_um) for e in _CCF_EXTENT_UM)
    # QuickNII's own grid, (ML, AP, DV) in the file -> (AP, DV, ML).
    tr = data.get("target-resolution") or [456, 528, 320]
    qn_dims = (round(tr[1]), round(tr[2]), round(tr[0]))

    folder = Path(image_dir) if image_dir else series_json.parent
    imported = []
    for n, entry in enumerate(entries, start=1):
        filename = entry.get("filename", "")
        image = load_image(_find_image(folder, filename))
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        h, w = image.shape[:2]
        anchoring = quicknii_to_atlas_anchoring(entry[key], atlas_shape, quicknii_dims=qn_dims)
        markers = entry.get("markers") or []
        warp = warp_inverse = None
        if markers:
            reg_size = (float(entry.get("width", w)), float(entry.get("height", h)))
            warp = marker_warp(markers, reg_size, (w, h))
            warp_inverse = marker_warp_inverse(markers, reg_size, (w, h))
        imported.append(ImportedSection(
            image=image, anchoring=anchoring,
            label=_series_number(entry, filename, n), warp=warp, warp_inverse=warp_inverse,
        ))

    name = series_json.stem
    out = Path(out_dir) if out_dir else series_json.with_name(f"{name}_atlastrack")
    n_cols = max(1, math.ceil(math.sqrt(len(imported))))
    for i, s in enumerate(imported):
        s.grid_cell = divmod(i, n_cols)
    return write_imported_project(
        imported, out, name, atlas_name=atlas_name, resolution_um=resolution_um,
    )
