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


def add_region_layers(
    viewer: "napari.Viewer",
    atlas: "BrainGlobeAtlas",
    regions: list[str],
    *,
    opacity: float = 0.3,
) -> list:
    """Add atlas region meshes as napari Surface layers.

    Returns the list of added layers.
    """
    from histo_to_ccf.atlas.meshes import mesh_vertices_faces

    added = []
    for acronym in regions:
        try:
            mesh = atlas.mesh_from_structure(acronym)
            verts, faces = mesh_vertices_faces(mesh)  # (N,3) AP,DV,ML
        except Exception:
            continue
        # Rearrange to (AP, ML, DV) to match probe layer convention.
        verts_aml = np.stack([verts[:, 0], verts[:, 2], verts[:, 1]], axis=1)
        layer = viewer.add_surface(
            (verts_aml, faces),
            name=acronym,
            opacity=opacity,
        )
        added.append(layer)
    return added


def switch_to_3d(viewer: "napari.Viewer") -> None:
    """Switch the napari viewer into 3D rendering mode."""
    viewer.dims.ndisplay = 3


def show_3d_scene(
    viewer: "napari.Viewer",
    project: "Project",
    atlas: "BrainGlobeAtlas | None" = None,
    *,
    regions: list[str] | None = None,
    line_width: float = 30.0,
) -> list:
    """Build a clean 3D scene: brain outline + region meshes + probe tracks.

    The 2D working layers (slide image, section outlines/numbers, tip/entry
    markers, atlas preview) are hidden first so they do not clutter the 3D view
    as flat sheets or tiny floating text. Returns the list of layers added.

    ``line_width`` is in atlas µm (world units), so probe tracks are visible at
    brain scale — the old 4 px default rendered as near-invisible hairlines.
    """
    # Hide everything currently shown; 3D content is added fresh on top.
    for layer in list(viewer.layers):
        layer.visible = False

    added: list = []
    if atlas is not None:
        # Translucent whole-brain outline for context, then a few key regions.
        added += add_region_layers(viewer, atlas, ["root"], opacity=0.08)
        if regions:
            added += add_region_layers(viewer, atlas, regions, opacity=0.25)

    added += add_probe_layers(viewer, project, line_width=line_width)
    switch_to_3d(viewer)
    try:
        viewer.reset_view()
    except Exception:
        pass
    return added
