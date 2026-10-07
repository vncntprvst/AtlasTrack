"""Import a registration made with slicereg (github.com/mvdokh/cell-counting).

slicereg keeps a project as a folder::

    <name>_project/            (or a folder called Registration)
        project.json           channels, atlas, per-slice pixel size, col/row on the slide
        slices/slice_NN/
            image.tif          cropped, downsampled, mirrored to the atlas's view (H, W, C)
            alignment.json     plane, global fit and landmarks
            cells.csv          counted cells with atlas coordinates

Its model (``slicereg/transform.py`` and ``slicereg/atlas.py``), reproduced here:

* A **plane** through the BrainGlobe atlas, given by ``ap_um`` (AP at the atlas
  centre), ``pitch_deg`` and ``yaw_deg``. Plane point ``(s, t)`` µm sits at
  ``origin + s*u + t*v`` with ``u, v`` the ML and DV axes turned by
  ``R = R_yaw @ R_pitch``.
* A **global fit** from plane µm to image pixels: ``x = M (p - c) + t`` with
  ``M = rot(rotation_deg) @ diag(scale_x, scale_y) @ diag(-1 if flip, 1)``,
  ``c`` the plane centre and ``t = (tx, ty)``.
* A **landmark warp** on top, in image pixels: a thin-plate spline of the
  displacements from where the global fit puts each landmark's atlas point to
  where the user put it, held at zero on 8 points 10% outside the image edges.

In AtlasTrack the global fit becomes the section's anchoring and the warp its
landmark correction: the landmarks plus those 8 fixed points, source = the global
fit's position, target = the user's, marked ``forward`` - fitted source -> target
and inverted exactly, as slicereg does (AtlasTrack's own landmarks are fitted the
other way round). Both the overlay and the atlas position of every pixel then
match slicereg's exactly, and the landmarks stay editable.

Sections are rescaled to one pixel size (the coarsest, by default) and laid out by
their column and row on the slide. Channels become RGB by the colours in
``project.json``, each stretched to full range with one scale for all sections.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from atlastrack.io.section_import import (
    ImportedSection,
    rescale,
    rescale_points,
    write_imported_project,
)
from atlastrack.project.cells import add_slicereg_cells
from atlastrack.project.schema import ManualLandmarks, PlaneParams

#: RGB weights for slicereg's channel colour names.
_COLOURS = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "magenta": (1, 0, 1),
    "cyan": (0, 1, 1),
    "yellow": (1, 1, 0),
    "gray": (1, 1, 1),
    "grey": (1, 1, 1),
    "white": (1, 1, 1),
}

#: How far outside the image slicereg pins its warp to zero, as a fraction of size.
ANCHOR_MARGIN = 0.1


def is_slicereg_project(path: str | Path) -> bool:
    """True if ``path`` looks like a slicereg project folder."""
    path = Path(path)
    return (path / "project.json").is_file() and any(path.glob("slices/*/alignment.json"))


def atlas_resolution_um(atlas_name: str) -> float:
    m = re.search(r"_(\d+(?:\.\d+)?)um$", atlas_name)
    if not m:
        raise ValueError(f"cannot read the atlas resolution from its name {atlas_name!r}")
    return float(m.group(1))


def plane_basis(alignment: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """slicereg's plane: origin, u (ML-like) and v (DV-like), in (AP, DV, ML) µm."""
    p, y = math.radians(alignment["pitch_deg"]), math.radians(alignment["yaw_deg"])
    r_pitch = np.array([[math.cos(p), math.sin(p), 0], [-math.sin(p), math.cos(p), 0], [0, 0, 1]])
    r_yaw = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    r = r_yaw @ r_pitch
    u, v = r @ np.array([0.0, 0.0, 1.0]), r @ np.array([0.0, 1.0, 0.0])
    w, h = alignment["plane_size_um"]
    centre = np.array([alignment["ap_um"], h / 2, w / 2])
    return centre - (w / 2) * u - (h / 2) * v, u, v


def plane_to_atlas_um(alignment: dict, st: np.ndarray) -> np.ndarray:
    """Plane ``(s, t)`` µm -> atlas (AP, DV, ML) µm."""
    origin, u, v = plane_basis(alignment)
    st = np.atleast_2d(np.asarray(st, dtype=float))
    return origin + st[:, :1] * u + st[:, 1:2] * v


def global_fit(alignment: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(M, c, t)`` of slicereg's plane -> image fit ``x = M (p - c) + t``."""
    th = math.radians(alignment.get("rotation_deg", 0.0))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    flip = np.diag([-1.0 if alignment.get("flip") else 1.0, 1.0])
    m = rot @ np.diag([alignment.get("scale_x", 1.0), alignment.get("scale_y", 1.0)]) @ flip
    c = np.asarray(alignment["plane_size_um"], dtype=float) / 2
    t = np.array([alignment.get("tx", 0.0), alignment.get("ty", 0.0)])
    return m, c, t


def anchor_points(image_size: tuple[int, int]) -> np.ndarray:
    """The 8 points around the image where slicereg holds its warp at zero."""
    w, h = image_size
    xs = [-ANCHOR_MARGIN * w, w / 2, (1 + ANCHOR_MARGIN) * w]
    ys = [-ANCHOR_MARGIN * h, h / 2, (1 + ANCHOR_MARGIN) * h]
    return np.array([[x, y] for x in xs for y in ys if not (x == w / 2 and y == h / 2)])


def section_registration(
    alignment: dict, factor: float, size_wh: tuple[int, int], res_um: float
) -> tuple[list[float], ManualLandmarks | None]:
    """Anchoring and landmark correction of one section, at ``factor`` x its pixels."""
    m, c, t = global_fit(alignment)
    m_inv = np.linalg.inv(m)
    w, h = size_wh
    # Image corners at the new pixel size, back to the original pixels, onto the plane.
    corners = rescale_points(np.array([[0.0, 0.0], [w, 0.0], [0.0, h]]), 1 / factor)
    st = (corners - t) @ m_inv.T + c
    o, x_end, y_end = plane_to_atlas_um(alignment, st) / res_um
    anchoring = [*o, *(x_end - o), *(y_end - o)]

    lm = np.asarray(alignment.get("landmarks") or [], dtype=float).reshape(-1, 4)
    if not len(lm):
        return [float(a) for a in anchoring], None
    source = (lm[:, :2] - c) @ m.T + t
    anchors = anchor_points(tuple(alignment["image_size"]))
    source = np.vstack([source, anchors])
    target = np.vstack([lm[:, 2:4], anchors])
    return [float(a) for a in anchoring], ManualLandmarks(
        source=rescale_points(source, factor).tolist(),
        target=rescale_points(target, factor).tolist(),
        # slicereg fits its spline this way round and inverts it exactly.
        forward=True,
    )


def _to_rgb(images: list[np.ndarray], channels: list[dict]) -> list[np.ndarray]:
    """Colour each section's channels into uint8 RGB, one stretch per channel."""
    n_ch = images[0].shape[2]
    if len(channels) != n_ch:
        channels = [{"color": "gray"}] * n_ch
    tops = []
    for k in range(n_ch):
        values = np.concatenate([im[..., k][::4, ::4].ravel() for im in images])
        tops.append(max(float(np.percentile(values, 99.5)), 1.0))
    out = []
    for im in images:
        rgb = np.zeros((*im.shape[:2], 3), dtype=np.float32)
        for k, ch in enumerate(channels):
            weights = _COLOURS.get(str(ch.get("color", "gray")).lower(), (1, 1, 1))
            scaled = np.clip(im[..., k].astype(np.float32) / tops[k], 0, 1)
            for j in range(3):
                if weights[j]:
                    rgb[..., j] = np.maximum(rgb[..., j], scaled)
        out.append((rgb * 255 + 0.5).astype(np.uint8))
    return out


def _channel_colour(colour: str) -> str | None:
    """The RGB channel a slicereg colour lands in, when it is just one."""
    weights = _COLOURS.get(colour.lower())
    if weights is None or sum(weights) != 1:
        return None
    return ("red", "green", "blue")[weights.index(1)]


def import_slicereg(
    project_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    pixel_um: float | None = None,
) -> Path:
    """Write an AtlasTrack project from a slicereg project folder.

    ``pixel_um`` is the pixel size sections are brought to; by default the
    coarsest in the project, so none is enlarged. Nothing in ``project_dir`` is
    changed; the output goes to ``out_dir`` (default ``<project_dir>_atlastrack``).
    Returns the path of the written ``.atlastrack.json``.
    """
    import tifffile

    project_dir = Path(project_dir).resolve()
    meta = json.loads((project_dir / "project.json").read_text())
    slices = sorted(meta.get("slices") or [], key=lambda s: s["id"])
    if not slices:
        raise ValueError(f"{project_dir / 'project.json'} lists no slices")
    if pixel_um is None:
        pixel_um = max(float(s["pixel_um"]) for s in slices)
    name = project_dir.name
    if name.lower() == "registration":
        name = project_dir.parent.name
    name = re.sub(r"_project$", "", name)
    out_dir = Path(out_dir) if out_dir else project_dir.with_name(f"{project_dir.name}_atlastrack")
    atlas_name = meta.get("atlas", "allen_mouse_25um")
    res_um = atlas_resolution_um(atlas_name)

    images, alignments, factors = [], [], []
    for s in slices:
        folder = project_dir / "slices" / f"slice_{s['id']:02d}"
        image = np.asarray(tifffile.imread(str(folder / "image.tif")))
        if image.ndim == 2:
            image = image[..., None]
        alignment = json.loads((folder / "alignment.json").read_text())
        h, w = image.shape[:2]
        if tuple(alignment["image_size"]) != (w, h):
            raise ValueError(
                f"{folder.name}: alignment.json is for a {alignment['image_size']} image, "
                f"image.tif is {[w, h]}"
            )
        factor = float(s["pixel_um"]) / pixel_um
        images.append(rescale(image, factor))
        alignments.append(alignment)
        factors.append(factor)

    channels = meta.get("channels") or []
    rgb = _to_rgb(images, channels)
    if all("row" in s and "col" in s for s in slices):
        rows = sorted({s["row"] for s in slices})
        cols = sorted({s["col"] for s in slices})
        cells = [(rows.index(s["row"]), cols.index(s["col"])) for s in slices]
    else:
        cells = [None] * len(slices)

    imported, planes = [], {}
    for i, (s, im, alignment, factor, cell) in enumerate(
        zip(slices, rgb, alignments, factors, cells, strict=True)
    ):
        h, w = im.shape[:2]
        anchoring, landmarks = section_registration(alignment, factor, (w, h), res_um)
        imported.append(ImportedSection(
            image=im, anchoring=anchoring, label=int(s["id"]),
            grid_cell=cell, landmarks=landmarks,
        ))
        # The tilts as coronal-plane parameters, for a later re-register. These
        # are rotations the other way round from slicereg's pitch and yaw.
        planes[i] = PlaneParams(
            ap_um=float(alignment["ap_um"]),
            ml_tilt_deg=-float(alignment["yaw_deg"]),
            dv_tilt_deg=-float(alignment["pitch_deg"]),
            pixel_size_um=pixel_um,
        )

    names: dict = {}
    for ch in channels:
        colour = _channel_colour(str(ch.get("color", "")))
        if colour and ch.get("name"):
            names[colour] = str(ch["name"])
    align = None
    if meta.get("align_channel"):
        align = next(
            (c for c, n in names.items() if n == meta["align_channel"]), None
        )

    return write_imported_project(
        imported, Path(out_dir), name,
        atlas_name=atlas_name, resolution_um=res_um, plane_for=planes,
        channel_names=names, align_channel=align, pixel_um=pixel_um,
        section_spacing_um=meta.get("section_spacing_um"),
        finish=lambda project: add_slicereg_cells(project, project_dir),
    )
