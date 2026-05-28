#!/usr/bin/env python3
"""
Visualize HERBS probe trajectories in Allen CCF atlas space.

Loads one or more HERBS pkl files, converts the registered electrode positions
to Allen CCF µm coordinates, overlays them on semi-transparent brain structure
meshes from BrainGlobe, and produces an interactive Plotly 3D figure.

Works identically as a standalone script or in a Jupyter notebook
(call make_figure() directly and display with fig.show()).

Dependencies
------------
    pip install brainglobe-atlasapi plotly numpy
The first run will download the requested atlas (~100 MB for 100 µm resolution).

See the companion how-to for a step-by-step explanation:
    docs/how-to/herbs_probe_mapping.md
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from brainglobe_atlasapi import BrainGlobeAtlas


# ── Configure this block for your data ────────────────────────────────────────

HERBS_PKL_FILES: list[tuple[str, str]] = [
    # Each entry: (path_to_pkl_file, session_label)
    # Any section pkl (_1, _2, ...) encodes the same trajectory — pick any one.
    # Example paths for the Sabatini/Shijia M249 dataset:
    (
        "/mnt/raid5/datasets/dataset_shijia/O2 output/HERBS Mapping File/M249"
        "/probe M249_IRt_Red_R_1.pkl",
        "20241025 (Red / DiI)",
    ),
    (
        "/mnt/raid5/datasets/dataset_shijia/O2 output/HERBS Mapping File/M249"
        "/probe M249_IRt_Purple_R_1.pkl",
        "20241026 (Purple / DiD)",
    ),
    (
        "/mnt/raid5/datasets/dataset_shijia/O2 output/HERBS Mapping File/M249"
        "/probe M249_IRt_Green_R_1.pkl",
        "20241028 (Green / DiO)",
    ),
]

# Brain structures rendered as semi-transparent meshes.
# Acronyms follow Allen CCF; browse the full list at https://atlas.brain-map.org
# (acronym, display_name, hex_color, opacity 0-1)
BRAIN_STRUCTURES: list[tuple[str, str, str, float]] = [
    ("BS",  "Brainstem",            "#b0a0c8", 0.07),
    ("CB",  "Cerebellum",           "#a0b8d0", 0.07),
    ("MY",  "Medulla",              "#c8a080", 0.18),
    ("VII", "Facial nucleus (VII)", "#e05030", 0.40),
    ("IRN", "IRt / IRN",            "#40a850", 0.30),
    ("TH",  "Thalamus",             "#d8a010", 0.15),
    ("STR", "Striatum",             "#c86060", 0.12),
    ("MOp", "Motor cortex (MOp)",   "#6090d0", 0.12),
]

# Atlas used for brain structure meshes.
# "allen_mouse_100um" is fast and sufficient for visualization.
# Switch to "allen_mouse_10um" only if you need finer mesh detail.
ATLAS_RESOLUTION = "allen_mouse_100um"

# Output: set to None to open in browser, or a Path to save as a standalone HTML file.
OUTPUT_HTML: Optional[Path] = Path("probe_trajectories.html")


# ── HERBS → Allen CCF coordinate conversion ───────────────────────────────────

# The Allen CCF 10 µm atlas has shape (AP=1320, DV=800, ML=1140).
# HERBS stores voxel indices into this atlas with two axes reversed:
#   axis0: ML, reversed  (0 = right edge)
#   axis1: AP, reversed  (0 = caudal end)
#   axis2: DV, standard  (0 = dorsal surface)
# Conversion to Allen CCF µm (x=ML, y=DV, z=AP):
_AP_MAX = 1319   # atlas AP size − 1  (1320 − 1)
_ML_MAX = 1139   # atlas ML size − 1  (1140 − 1)


def load_herbs_pkl(pkl_path: str | Path) -> list[dict]:
    """
    Load a HERBS pkl and return per-shank trajectory data in Allen CCF µm.

    Parameters
    ----------
    pkl_path : path to the HERBS pkl file (any section _1–_4 suffix works)

    Returns
    -------
    List of shank dicts (length = number of shanks), each containing:
        ccf     (N, 3) float  — Allen CCF µm columns: [ML, DV, AP]
        regions list[str]     — brain region acronym per site (from HERBS)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)["data"]

    sites_vox    = data["sites_vox"]               # list of (N_i, 3) voxel coords
    region_sites = [int(r) for r in data["region_sites"]]
    label_acr    = data["label_acronym"]

    # Expand run-length encoded region labels into one label per site
    labels: list[str] = []
    for acr, n in zip(label_acr, region_sites):
        labels.extend([str(acr).strip()] * n)

    shanks = []
    offset = 0
    for sv_raw in sites_vox:
        sv  = np.array(sv_raw, dtype=float)
        n   = len(sv)
        ccf = np.column_stack([
            (_ML_MAX - sv[:, 0]) * 10,   # ML (µm)
            sv[:, 2] * 10,               # DV (µm)
            (_AP_MAX - sv[:, 1]) * 10,   # AP (µm)
        ])
        shanks.append(dict(ccf=ccf, regions=labels[offset : offset + n]))
        offset += n

    return shanks


# ── Atlas mesh loading ─────────────────────────────────────────────────────────

def load_atlas_meshes(atlas: BrainGlobeAtlas,
                      structures: list[tuple]) -> list[dict]:
    """Extract triangle meshes for each requested brain structure.

    BrainGlobe mesh vertices are in Allen CCF µm, ASR orientation
    (axis0=AP, axis1=DV, axis2=ML).  The Plotly axes used here are
    x=ML, y=AP, z=DV, so vertices are reordered accordingly.
    """
    meshes = []
    for acronym, name, color, opacity in structures:
        try:
            mesh = atlas.mesh_from_structure(acronym)
        except Exception as e:
            print(f"  skip '{acronym}': {e}")
            continue

        v = mesh.points   # (V, 3) µm in (AP, DV, ML) order

        # Extract triangle faces
        tris = None
        for block in mesh.cells:
            if block.type in ("triangle", "tri"):
                tris = block.data
                break
        if tris is None:
            for k, data in mesh.cells_dict.items():
                if "tri" in k.lower():
                    tris = data
                    break
        if tris is None:
            print(f"  skip '{acronym}': no triangle faces found")
            continue

        meshes.append(dict(
            name=name, color=color, opacity=opacity,
            x=v[:, 2].tolist(),          # ML
            y=v[:, 0].tolist(),          # AP  → Plotly y
            z=v[:, 1].tolist(),          # DV  → Plotly z
            i=tris[:, 0].tolist(),
            j=tris[:, 1].tolist(),
            k=tris[:, 2].tolist(),
        ))
        print(f"  {acronym} ({name}): {len(v):,} vertices")

    return meshes


# ── Colors ─────────────────────────────────────────────────────────────────────

# One color per shank (cycles through the list if you have many shanks/sessions)
_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#9a6324", "#469990", "#dcbeff", "#800000",
]


# ── Figure construction ────────────────────────────────────────────────────────

def make_figure(
    pkl_files: list[tuple[str, str]],
    structures: list[tuple] = BRAIN_STRUCTURES,
    atlas_name: str = ATLAS_RESOLUTION,
) -> go.Figure:
    """
    Build an interactive Plotly 3D figure with brain meshes + probe trajectories.

    Parameters
    ----------
    pkl_files  : list of (pkl_path, label) — one entry per session/probe
    structures : list of (acronym, name, color, opacity) — brain regions to show
    atlas_name : BrainGlobe atlas identifier

    Returns
    -------
    plotly.graph_objects.Figure
    """
    print(f"Loading atlas '{atlas_name}'…")
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)

    print("Loading brain structure meshes…")
    meshes = load_atlas_meshes(atlas, structures)

    traces: list = []

    # Semi-transparent brain structure meshes
    for m in meshes:
        traces.append(go.Mesh3d(
            x=m["x"], y=m["y"], z=m["z"],
            i=m["i"], j=m["j"], k=m["k"],
            color=m["color"], opacity=m["opacity"],
            name=m["name"], showlegend=True,
            hoverinfo="name", flatshading=False,
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.1),
        ))

    # Probe trajectories: one scatter per shank per pkl
    color_idx = 0
    for pkl_path, label in pkl_files:
        print(f"Loading {Path(pkl_path).name}…")
        shanks = load_herbs_pkl(pkl_path)
        for s, shank in enumerate(shanks):
            ccf     = shank["ccf"]       # (N, 3): ML, DV, AP
            regions = shank["regions"]
            hover   = [f"{label} — shank {s}<br>region: {r}" for r in regions]
            traces.append(go.Scatter3d(
                x=ccf[:, 0].tolist(),    # ML
                y=ccf[:, 2].tolist(),    # AP  (Plotly y = AP)
                z=ccf[:, 1].tolist(),    # DV  (Plotly z = DV)
                mode="markers",
                name=f"{label} — shank {s}",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=3,
                    color=_COLORS[color_idx % len(_COLORS)],
                    opacity=0.90,
                ),
            ))
            color_idx += 1

    layout = go.Layout(
        title="Probe Trajectories — Allen CCF (µm)",
        template="plotly_white",
        height=680,
        legend=dict(x=1.0, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
        scene=dict(
            xaxis=dict(title="ML (µm)", showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(title="AP (µm)", showgrid=True, gridcolor="#e0e0e0"),
            zaxis=dict(title="DV (µm)", showgrid=True, gridcolor="#e0e0e0",
                       autorange="reversed"),   # dorsal = top
            aspectmode="data",
            bgcolor="rgba(248,248,255,1)",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=1.6, y=1.3, z=-0.4),
            ),
        ),
    )

    return go.Figure(data=traces, layout=layout)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    fig = make_figure(HERBS_PKL_FILES, BRAIN_STRUCTURES)
    if OUTPUT_HTML is not None:
        fig.write_html(str(OUTPUT_HTML), include_plotlyjs="cdn")
        print(f"\nSaved → {OUTPUT_HTML.resolve()}")
        print("Open that file in any browser to view the interactive figure.")
    else:
        fig.show()


if __name__ == "__main__":
    main()
