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
    from histo_to_ccf.probes.geometry import shank_offsets

    added = []
    for p_idx, probe in enumerate(project.probes):
        color = _PROBE_COLORS[p_idx % len(_PROBE_COLORS)]
        n = probe.type.n_shanks
        pitch = probe.type.shank_pitch_um
        ml_offsets = shank_offsets(n, pitch)
        lines = []
        for shank in probe.shanks:
            if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
                continue
            tip = np.array(shank.tip_ccf_um, dtype=float)    # (AP, ML, DV)
            entry = np.array(shank.entry_ccf_um, dtype=float)
            ml_offset = float(ml_offsets[shank.index]) if shank.index < len(ml_offsets) else 0.0
            tip_adj = tip.copy(); tip_adj[1] += ml_offset
            entry_adj = entry.copy(); entry_adj[1] += ml_offset
            # napari 3D: (z=DV, y=AP, x=ML) or just direct if 3D mode uses (row,col,depth)
            # napari uses (y, x) for 2D and (z, y, x) for 3D by default. In our case:
            # We'll store as (AP, ML, DV) matching the atlas dimensions.
            lines.append(np.array([[tip_adj[0], tip_adj[1], tip_adj[2]],
                                   [entry_adj[0], entry_adj[1], entry_adj[2]]]))

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
    blending: str = "translucent",
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
            # Context shell additive (see-through); internal nuclei translucent.
            blending = "additive" if acronym in CONTEXT_REGIONS else "translucent"
            layer = _add_region_surface(
                viewer, atlas, acronym, rgb=hex_to_rgb(color),
                opacity=opacity, blending=blending,
            )
            if layer is not None:
                added.append(layer)

    added += add_probe_layers(viewer, project, line_width=line_width)
    switch_to_3d(viewer)
    try:
        viewer.reset_view()
    except Exception:
        pass
    return added
