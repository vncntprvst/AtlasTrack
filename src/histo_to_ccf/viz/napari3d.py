"""napari 3D layer builders for probe trajectories and atlas meshes.

This module IS allowed to import napari (it lives under viz/, not core modules).
All functions accept a live napari Viewer and add layers to it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari
    from brainglobe_atlasapi import BrainGlobeAtlas
    from histo_to_ccf.project.schema import Project

_PROBE_COLORS = [
    (1.0, 0.1, 0.1, 1.0),
    (0.1, 0.8, 0.2, 1.0),
    (0.2, 0.4, 1.0, 1.0),
    (1.0, 0.5, 0.1, 1.0),
    (0.7, 0.2, 0.9, 1.0),
]


def add_probe_layers(
    viewer: "napari.Viewer",
    project: "Project",
    *,
    line_width: float = 4.0,
) -> list:
    """Add probe tip→entry trajectories as 3D Shapes layers.

    Returns the list of added layers.
    """
    added = []
    for p_idx, probe in enumerate(project.probes):
        color = _PROBE_COLORS[p_idx % len(_PROBE_COLORS)]
        lines = []
        for shank in probe.shanks:
            if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
                continue
            # Each shank's tip/entry was placed + registered individually, so its
            # CCF already carries the correct ML. Do NOT add a geometric shank
            # offset here - that double-counted the shank separation and pushed
            # outer shanks across the midline. (The per-channel export uses these
            # coords directly too.) Stored as (AP, ML, DV).
            tip = np.array(shank.tip_ccf_um, dtype=float)
            entry = np.array(shank.entry_ccf_um, dtype=float)
            lines.append(np.array([tip, entry]))

        if not lines:
            continue
        layer = viewer.add_shapes(
            lines,
            name=f"Probe {probe.label}",
            shape_type="line",
            edge_color=[color] * len(lines),
            edge_width=line_width,
            ndim=3,
        )
        added.append(layer)
    return added


def add_ephys_channel_layers(
    viewer: "napari.Viewer",
    project: "Project",
    *,
    size: float = 60.0,
) -> list:
    """Add ephys-aligned per-channel CCF positions as 3D Points layers.

    One layer per probe that has any shank with ``ephys.channel_ccf_um`` filled
    (from the Ephys tab). Makes the depth refinement visible in 3D - without this
    the scene only shows the tip→entry line, which the alignment does not move.
    Returns the added layers.
    """
    added = []
    for p_idx, probe in enumerate(project.probes):
        color = _PROBE_COLORS[p_idx % len(_PROBE_COLORS)]
        pts = []
        for shank in probe.shanks:
            if shank.ephys is None:
                continue
            pts.extend([tuple(float(v) for v in c) for c in shank.ephys.channel_ccf_um])
        if not pts:
            continue
        layer = viewer.add_points(
            np.array(pts, dtype=float),  # (AP, ML, DV)
            name=f"Ephys channels {probe.label}",
            face_color=[color] * len(pts),
            size=size,
            ndim=3,
        )
        added.append(layer)
    return added


def _solid_colormap(rgb: tuple[int, int, int], name: str):
    """A single-colour napari Colormap (both ends = ``rgb``) for flat shading."""
    from napari.utils.colormaps import Colormap

    c = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0]
    return Colormap([c, c], name=name or "region")


def _add_region_surface(
    viewer: "napari.Viewer",
    atlas: "BrainGlobeAtlas",
    acronym: str,
    *,
    rgb: tuple[int, int, int],
    opacity: float,
    blending: str = "translucent_no_depth",
):
    """Add one region mesh as a flat-coloured napari Surface layer."""
    from histo_to_ccf.atlas.meshes import mesh_vertices_faces

    try:
        mesh = atlas.mesh_from_structure(acronym)
        verts, faces = mesh_vertices_faces(mesh)  # (N,3) AP,DV,ML
    except Exception:
        return None
    # Rearrange to (AP, ML, DV) to match probe layer convention.
    verts_aml = np.stack([verts[:, 0], verts[:, 2], verts[:, 1]], axis=1)
    try:
        layer = viewer.add_surface(
            (verts_aml, faces, np.ones(len(verts_aml))),
            name=acronym, opacity=opacity, blending=blending,
            colormap=_solid_colormap(rgb, acronym),
        )
    except Exception:
        layer = viewer.add_surface((verts_aml, faces), name=acronym, opacity=opacity)
    return layer


def add_region_layers(
    viewer: "napari.Viewer",
    atlas: "BrainGlobeAtlas",
    regions: list[str],
    *,
    blending: str = "translucent",
) -> list:
    """Add atlas region meshes as napari Surface layers using the palette."""
    from histo_to_ccf.viz.plotly3d import hex_to_rgb, styled_regions

    added = []
    for acronym, color, opacity in styled_regions(regions):
        layer = _add_region_surface(
            viewer, atlas, acronym, rgb=hex_to_rgb(color),
            opacity=opacity, blending=blending,
        )
        if layer is not None:
            added.append(layer)
    return added


def add_region_by_acronym(
    viewer: "napari.Viewer",
    atlas: "BrainGlobeAtlas",
    acronym: str,
):
    """Add a single searched-for region mesh (palette-coloured) to a 3D scene."""
    from histo_to_ccf.viz.plotly3d import hex_to_rgb, region_style

    color, opacity = region_style(acronym)
    return _add_region_surface(
        viewer, atlas, acronym, rgb=hex_to_rgb(color), opacity=opacity
    )


def switch_to_3d(viewer: "napari.Viewer") -> None:
    """Switch the napari viewer into 3D rendering mode."""
    viewer.dims.ndisplay = 3


def show_3d_scene(
    viewer: "napari.Viewer",
    project: "Project",
    atlas: "BrainGlobeAtlas | None" = None,
    *,
    extra_regions: "list[str] | tuple[str, ...]" = (),
    show_tip_regions: bool = True,
    line_width: float = 40.0,
) -> list:
    """Build a clean 3D scene: brain shell + tip regions + probe tracks.

    The 2D working layers are hidden first so they do not clutter the 3D view.
    The whole-brain outline and the large context divisions use *additive*
    blending so they never hide the probe tracks running inside the brain.
    Internal regions shown are only those containing a shank tip, plus any
    ``extra_regions`` the user searched for.
    """
    from histo_to_ccf.viz.plotly3d import (
        CONTEXT_REGIONS,
        hex_to_rgb,
        resolve_regions,
        styled_regions,
    )

    for layer in list(viewer.layers):
        layer.visible = False

    added: list = []
    if atlas is not None:
        # Faint whole-brain shell (additive so it never hides the probe).
        root = _add_region_surface(
            viewer, atlas, "root", rgb=(170, 170, 170), opacity=0.04, blending="additive"
        )
        if root is not None:
            added.append(root)

        internal = resolve_regions(
            project, atlas, extra_regions=extra_regions, show_tip_regions=show_tip_regions
        )
        for acronym, color, opacity in styled_regions(list(CONTEXT_REGIONS) + internal):
            # Context shell additive (see-through); internal nuclei use
            # translucent_no_depth so a region surrounding a shank tip colours
            # the volume but never writes depth - the probe stays visible inside.
            blending = "additive" if acronym in CONTEXT_REGIONS else "translucent_no_depth"
            layer = _add_region_surface(
                viewer, atlas, acronym, rgb=hex_to_rgb(color),
                opacity=opacity, blending=blending,
            )
            if layer is not None:
                added.append(layer)

    added += add_probe_layers(viewer, project, line_width=line_width)
    added += add_ephys_channel_layers(viewer, project)
    _apply_bregma_display(added)  # show everything bregma-referenced (display only)
    switch_to_3d(viewer)
    _set_default_camera(viewer)
    return added


# Layer data is in CCF (AP, ML, DV) µm. The 3D scene is shown **bregma-referenced**
# via a per-layer display affine - the stored data is untouched. It is a PURE
# TRANSLATION (offsets only): AP 0 at bregma, ML 0 at the midline, DV unchanged.
# Crucially it does NOT negate/reverse any axis - negating one axis is a reflection
# that would MIRROR left/right (the bug the Plotly z-reversal caused). AP therefore
# stays posterior-positive here; orientation (dorsal up, posterior view) is all from
# the camera.
def _bregma_affine() -> "np.ndarray":
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM, MIDLINE_ML_UM

    return np.array([
        [1.0, 0.0, 0.0, -BREGMA_AP_FROM_ORIGIN_UM],  # AP -> AP - bregma (0 at bregma)
        [0.0, 1.0, 0.0, -MIDLINE_ML_UM],             # ML -> 0 at midline
        [0.0, 0.0, 1.0, 0.0],                        # DV unchanged
        [0.0, 0.0, 0.0, 1.0],
    ])


# Default 3D view: from behind (posterior), dorsal up, tilted slightly down from the
# top. AP stays posterior-positive (offset only), so "look toward anterior" is -AP.
_VIEW_DIRECTION = (-1.0, 0.0, 0.5)   # look toward anterior (-AP), slightly down (+DV)
_UP_DIRECTION = (0.0, 0.0, -1.0)     # dorsal is up (-DV)


def _apply_bregma_display(layers: list) -> None:
    """Set the bregma display affine on every 3D-scene layer (display only)."""
    affine = _bregma_affine()
    for layer in layers:
        try:
            layer.affine = affine
        except Exception:  # noqa: BLE001 - affine is best-effort per layer
            pass


def _set_default_camera(viewer: "napari.Viewer") -> None:
    """Fit the data, then orient the camera to the standard posterior-top view."""
    try:
        viewer.reset_view()  # fit zoom/centre to the data
        viewer.camera.set_view_direction(
            view_direction=_VIEW_DIRECTION, up_direction=_UP_DIRECTION
        )
    except Exception:  # noqa: BLE001 - camera orientation is best-effort
        pass
