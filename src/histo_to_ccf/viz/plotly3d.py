"""Plotly-based 3D probe + atlas visualization.

No Qt or napari imports allowed here (see import-linter contract).
"""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from brainglobe_atlasapi import BrainGlobeAtlas
    from histo_to_ccf.project.schema import Project

# Regions displayed by default when an atlas is available.
DEFAULT_REGIONS = ["CB", "CTX", "HPF", "TH", "STR", "BS"]

# Soft colours for each default region (same order as DEFAULT_REGIONS).
_REGION_COLORS = [
    "rgba(150,200,150,0.25)",
    "rgba(200,150,100,0.18)",
    "rgba(100,170,220,0.22)",
    "rgba(220,180,100,0.22)",
    "rgba(190,130,210,0.22)",
    "rgba(160,160,160,0.18)",
]

# Probe trajectory colours (cycling).
_PROBE_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


# ---------------------------------------------------------------------------
# Atlas mesh helpers
# ---------------------------------------------------------------------------

def _mesh_for_region(atlas: "BrainGlobeAtlas", acronym: str) -> "go.Mesh3d | None":
    """Return a Plotly Mesh3d for one brain region, or None if unavailable."""
    import plotly.graph_objects as go

    from histo_to_ccf.atlas.meshes import mesh_vertices_faces

    try:
        mesh = atlas.mesh_from_structure(acronym)
    except Exception:
        return None
    if mesh is None:
        return None

    try:
        verts, faces = mesh_vertices_faces(mesh)  # (N,3) AP,DV,ML µm ; (M,3)
    except ValueError:
        return None
    # Plotly axes: x=ML, y=AP, z=DV
    color_idx = DEFAULT_REGIONS.index(acronym) if acronym in DEFAULT_REGIONS else 0
    color = _REGION_COLORS[color_idx % len(_REGION_COLORS)]
    return go.Mesh3d(
        x=verts[:, 2].tolist(),  # ML
        y=verts[:, 0].tolist(),  # AP
        z=verts[:, 1].tolist(),  # DV
        i=faces[:, 0].tolist(),
        j=faces[:, 1].tolist(),
        k=faces[:, 2].tolist(),
        color=color,
        opacity=0.25,
        name=acronym,
        showlegend=True,
        lighting={"diffuse": 0.5, "specular": 0.1, "roughness": 0.8},
        flatshading=True,
    )


def add_atlas_meshes(
    fig: "go.Figure",
    atlas: "BrainGlobeAtlas",
    regions: list[str] = DEFAULT_REGIONS,
) -> None:
    """Load BrainGlobe meshes and add as Mesh3d traces."""
    for region in regions:
        mesh = _mesh_for_region(atlas, region)
        if mesh is not None:
            fig.add_trace(mesh)


# ---------------------------------------------------------------------------
# Probe trajectory helpers
# ---------------------------------------------------------------------------

def add_probe_traces(
    fig: "go.Figure",
    project: "Project",
    *,
    style: str = "line",
) -> None:
    """Add probe tip→entry trajectories to ``fig``.

    ``style`` can be ``'line'``, ``'mesh'``, or ``'both'``.
    Probes without both tip_ccf_um and entry_ccf_um are skipped.
    """
    import plotly.graph_objects as go
    from histo_to_ccf.probes.geometry import probe_prism_mesh, shank_offsets

    probe_count = 0
    for probe in project.probes:
        color = _PROBE_COLORS[probe_count % len(_PROBE_COLORS)]
        probe_count += 1
        n = probe.type.n_shanks
        pitch = probe.type.shank_pitch_um
        ml_offsets = shank_offsets(n, pitch)

        for shank in probe.shanks:
            if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
                continue
            tip = np.array(shank.tip_ccf_um, dtype=float)    # (AP, ML, DV)
            entry = np.array(shank.entry_ccf_um, dtype=float)
            ml_offset = float(ml_offsets[shank.index]) if shank.index < len(ml_offsets) else 0.0
            tip_adj = tip.copy(); tip_adj[1] += ml_offset
            entry_adj = entry.copy(); entry_adj[1] += ml_offset

            if style in ("line", "both"):
                fig.add_trace(go.Scatter3d(
                    x=[tip_adj[1], entry_adj[1]],  # ML
                    y=[tip_adj[0], entry_adj[0]],  # AP
                    z=[tip_adj[2], entry_adj[2]],  # DV
                    mode="lines+markers",
                    line={"color": color, "width": 4},
                    marker={"size": [5, 3], "color": color},
                    name=f"{probe.label} s{shank.index}",
                    showlegend=True,
                ))

            if style in ("mesh", "both"):
                try:
                    mesh_dict = probe_prism_mesh(tuple(tip_adj), tuple(entry_adj))  # type: ignore[arg-type]
                    fig.add_trace(go.Mesh3d(
                        **mesh_dict,
                        color=color,
                        opacity=0.85,
                        name=f"{probe.label} s{shank.index} (mesh)",
                        showlegend=False,
                    ))
                except Exception:
                    pass  # fall back silently if geometry fails


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    project: "Project",
    atlas: "BrainGlobeAtlas | None" = None,
    *,
    regions: list[str] = DEFAULT_REGIONS,
    style: str = "line",
    title: str = "Histo-to-CCF: Probe trajectories",
) -> "go.Figure":
    """Build a Plotly 3D figure with probe trajectories and optional atlas meshes.

    Parameters
    ----------
    project
        The project holding CCF-registered shank coordinates.
    atlas
        If provided, brain-region meshes are added.
    regions
        List of BrainGlobe structure acronyms to render.  Skipped if atlas is None.
    style
        ``'line'``, ``'mesh'``, or ``'both'`` for probe representation.
    title
        Figure title shown in the HTML.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    if atlas is not None:
        add_atlas_meshes(fig, atlas, regions)

    add_probe_traces(fig, project, style=style)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="ML (µm)",
            yaxis_title="AP (µm)",
            zaxis_title="DV (µm)",
            # Invert DV so dorsal is up.
            zaxis={"autorange": "reversed"},
            aspectmode="data",
            bgcolor="rgb(20,20,30)",
            xaxis={"gridcolor": "rgb(60,60,80)", "zerolinecolor": "rgb(80,80,100)"},
            yaxis={"gridcolor": "rgb(60,60,80)", "zerolinecolor": "rgb(80,80,100)"},
        ),
        paper_bgcolor="rgb(20,20,30)",
        font={"color": "rgb(220,220,220)"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    return fig


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

def save_html(fig: "go.Figure", path: str | Path, *, open_browser: bool = False) -> Path:
    """Write ``fig`` to an interactive HTML file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    if open_browser:
        webbrowser.open(out.as_uri())
    return out
