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

# Large outer divisions drawn as a faint translucent shell for context.
CONTEXT_REGIONS = ["Isocortex", "CB", "BS"]

# Kept for backward compatibility (older callers / tests import this).
DEFAULT_REGIONS = CONTEXT_REGIONS

# Curated region styling: acronym -> (hex colour, opacity). Hand-picked so that
# distinct structures - including separate nuclei within one parent (e.g. VII /
# XII / IRN in the brainstem) - are clearly separable. The native atlas colours
# are intentionally NOT used (too muddy / too similar between siblings).
REGION_STYLE: dict[str, tuple[str, float]] = {
    "Isocortex": ("#d4c8a8", 0.07),
    "CB": ("#a8c8d4", 0.09),
    "OLF": ("#c8d4a8", 0.09),
    "BS": ("#b8a0c8", 0.13),
    "VII": ("#e05030", 0.40),
    "XII": ("#3080c8", 0.40),
    "IRN": ("#40a850", 0.28),
    "TH": ("#d8a010", 0.18),
    "STR": ("#c86060", 0.13),
    "MOp": ("#6090d0", 0.13),
}

# Qualitative fallback palette for regions not in REGION_STYLE - kept mutually
# distinct so sibling nuclei never collide on colour.
_FALLBACK_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabed4", "#469990", "#dcbeff",
]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """``'#rrggbb'`` -> ``(r, g, b)`` in 0–255."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def region_style(acronym: str, fallback_index: int = 0) -> tuple[str, float]:
    """Return ``(hex colour, opacity)`` for a region; cycle fallbacks otherwise."""
    if acronym in REGION_STYLE:
        return REGION_STYLE[acronym]
    return (_FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)], 0.30)


def styled_regions(regions: list[str]) -> list[tuple[str, str, float]]:
    """Map a region list to ``(acronym, hex colour, opacity)``, deduped in order.

    Fallback colours advance only for regions missing from REGION_STYLE, so the
    curated entries keep their fixed colour regardless of ordering.
    """
    out: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    fb = 0
    for acr in regions:
        if acr in seen:
            continue
        seen.add(acr)
        if acr in REGION_STYLE:
            color, op = REGION_STYLE[acr]
        else:
            color, op = _FALLBACK_COLORS[fb % len(_FALLBACK_COLORS)], 0.30
            fb += 1
        out.append((acr, color, op))
    return out

# Probe trajectory colours (cycling).
_PROBE_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
]


# ---------------------------------------------------------------------------
# Atlas mesh helpers
# ---------------------------------------------------------------------------

def _mesh_for_region(
    atlas: "BrainGlobeAtlas", acronym: str, *, color: str, opacity: float
) -> "go.Mesh3d | None":
    """Return a Plotly Mesh3d for one brain region in the given hex ``color``."""
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
    r, g, b = hex_to_rgb(color)
    return go.Mesh3d(
        x=verts[:, 2].tolist(),  # ML
        y=verts[:, 0].tolist(),  # AP
        z=verts[:, 1].tolist(),  # DV
        i=faces[:, 0].tolist(),
        j=faces[:, 1].tolist(),
        k=faces[:, 2].tolist(),
        color=f"rgb({r},{g},{b})",
        opacity=opacity,
        name=acronym,
        showlegend=True,
        lighting={"diffuse": 0.5, "specular": 0.1, "roughness": 0.8},
        flatshading=True,
    )


def add_atlas_meshes(
    fig: "go.Figure",
    atlas: "BrainGlobeAtlas",
    regions: list[str] = CONTEXT_REGIONS,
) -> None:
    """Load BrainGlobe meshes and add them as styled Mesh3d traces."""
    for acronym, color, opacity in styled_regions(list(regions)):
        mesh = _mesh_for_region(atlas, acronym, color=color, opacity=opacity)
        if mesh is not None:
            fig.add_trace(mesh)


def _tip_points(project: "Project") -> list[tuple[float, float, float]]:
    """CCF (AP, ML, DV) of every registered shank tip."""
    pts = []
    for probe in project.probes:
        for shank in probe.shanks:
            if shank.tip_ccf_um is not None:
                pts.append(tuple(shank.tip_ccf_um))
    return pts


def resolve_regions(
    project: "Project",
    atlas: "BrainGlobeAtlas | None",
    *,
    extra_regions: "list[str] | tuple[str, ...]" = (),
    show_tip_regions: bool = True,
) -> list[str]:
    """Internal regions to draw: those at shank tips plus any user extras."""
    if atlas is None:
        return list(extra_regions)
    from histo_to_ccf.atlas.meshes import region_acronyms_at_points

    internal: list[str] = []
    if show_tip_regions:
        internal += region_acronyms_at_points(atlas, _tip_points(project))
    for acr in extra_regions:
        if acr and acr not in internal:
            internal.append(acr)
    # Don't duplicate the context shell.
    return [a for a in internal if a not in CONTEXT_REGIONS]


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
    context_regions: list[str] = CONTEXT_REGIONS,
    extra_regions: "list[str] | tuple[str, ...]" = (),
    show_tip_regions: bool = True,
    style: str = "line",
    title: str = "Histo-to-CCF: Probe trajectories",
) -> "go.Figure":
    """Build a Plotly 3D figure with probe trajectories and atlas meshes.

    Parameters
    ----------
    project
        The project holding CCF-registered shank coordinates.
    atlas
        If provided, brain-region meshes are added.
    context_regions
        Large outer divisions drawn as a faint shell (default Isocortex/CB/BS).
    extra_regions
        Additional structure acronyms the user asked to display.
    show_tip_regions
        When True, the region containing each shank tip is drawn (opaque-ish).
    style
        ``'line'``, ``'mesh'``, or ``'both'`` for probe representation.
    title
        Figure title shown in the HTML.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    if atlas is not None:
        internal = resolve_regions(
            project, atlas, extra_regions=extra_regions, show_tip_regions=show_tip_regions
        )
        # Context shell first, then tip/extra regions - each with its style.
        for acronym, color, opacity in styled_regions(list(context_regions) + internal):
            mesh = _mesh_for_region(atlas, acronym, color=color, opacity=opacity)
            if mesh is not None:
                fig.add_trace(mesh)

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
