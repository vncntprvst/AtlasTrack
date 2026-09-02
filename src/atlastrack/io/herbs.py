"""Read HERBS pickle output and convert HERBS voxel indices ↔ Allen CCF µm.

HERBS stores per-shank sample sites as voxel indices into the Allen CCF atlas
(default 10 µm grid), with two axes reversed:

    HERBS axis0: ML, reversed   (0 = right edge)
    HERBS axis1: AP, reversed   (0 = caudal end)
    HERBS axis2: DV, standard   (0 = dorsal surface)

Conversion to CCF µm (AP, ML, DV order in our package):

    ML_µm = (ml_max - sv[:, 0]) * grid_µm
    AP_µm = (ap_max - sv[:, 1]) * grid_µm
    DV_µm =          sv[:, 2]   * grid_µm

where ``*_max = int(extent_µm // grid_µm) - 1``.

Mirrors ``legacy/HERBS_to_AllenCCF/herbs_probe_mapping.py``'s ``load_herbs_pkl``
and ``probe_visualization.py``'s ``infer_voxel_size_um``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np

from atlastrack.io.ccf_coords import AP_UM, DV_UM, ML_UM

# Default HERBS voxel grid. HERBS itself uses 10 µm regardless of the BrainGlobe
# atlas resolution the downstream code uses for visualization.
DEFAULT_HERBS_GRID_UM = 10


class HerbsShank(TypedDict):
    """One shank's trajectory after HERBS-voxel→CCF-µm conversion."""

    ccf_um: np.ndarray  # (N, 3): AP, ML, DV
    regions: list[str]


def infer_voxel_size_um(sites_vox: list[np.ndarray]) -> int:
    """Guess the HERBS voxel grid (10 or 25 µm) from the max indices observed."""
    all_sites = np.concatenate(
        [np.asarray(sites, dtype=float) for sites in sites_vox], axis=0
    )
    max_ap = float(all_sites[:, 1].max())
    max_ml = float(all_sites[:, 0].max())
    # 25 µm atlas has AP=528, ML=456; 10 µm atlas has AP=1320, ML=1140.
    return 10 if (max_ap > 800 or max_ml > 700) else 25


def _voxel_extents(grid_um: int) -> tuple[int, int, int]:
    ap_max = int(AP_UM // grid_um) - 1
    dv_max = int(DV_UM // grid_um) - 1
    ml_max = int(ML_UM // grid_um) - 1
    return ap_max, dv_max, ml_max


def herbs_voxel_to_ccf_um(
    sv: np.ndarray, *, grid_um: int = DEFAULT_HERBS_GRID_UM
) -> np.ndarray:
    """Convert a (N, 3) HERBS voxel array to (N, 3) AP/ML/DV µm in CCF."""
    sv = np.asarray(sv, dtype=float)
    ap_max, _dv_max, ml_max = _voxel_extents(grid_um)
    ml_um = (ml_max - sv[:, 0]) * grid_um
    ap_um = (ap_max - sv[:, 1]) * grid_um
    dv_um = sv[:, 2] * grid_um
    return np.column_stack([ap_um, ml_um, dv_um])


def ccf_um_to_herbs_voxel(
    ccf_um: np.ndarray, *, grid_um: int = DEFAULT_HERBS_GRID_UM
) -> np.ndarray:
    """Inverse of :func:`herbs_voxel_to_ccf_um`. Returns float voxel coords."""
    ccf_um = np.asarray(ccf_um, dtype=float)
    ap_max, _dv_max, ml_max = _voxel_extents(grid_um)
    ap = ccf_um[:, 0]
    ml = ccf_um[:, 1]
    dv = ccf_um[:, 2]
    sv0 = ml_max - ml / grid_um
    sv1 = ap_max - ap / grid_um
    sv2 = dv / grid_um
    return np.column_stack([sv0, sv1, sv2])


def load_herbs_pkl(
    pkl_path: str | Path, *, grid_um: int | str = "auto"
) -> list[HerbsShank]:
    """Load a HERBS pkl file, returning one entry per shank in CCF µm.

    Parameters
    ----------
    pkl_path
        Path to the HERBS pkl (any section suffix `_1`–`_4` encodes the same
        trajectory; pick one).
    grid_um
        HERBS voxel grid. ``"auto"`` infers from the data; otherwise pass 10 or 25.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)["data"]

    sites_vox = data["sites_vox"]
    region_sites = [int(r) for r in data.get("region_sites", [])]
    label_acr = data.get("label_acronym", [])

    labels: list[str] = []
    for acr, n in zip(label_acr, region_sites, strict=False):
        labels.extend([str(acr).strip()] * n)

    g = infer_voxel_size_um(sites_vox) if grid_um == "auto" else int(grid_um)

    shanks: list[HerbsShank] = []
    offset = 0
    for sv_raw in sites_vox:
        sv = np.asarray(sv_raw, dtype=float)
        ccf = herbs_voxel_to_ccf_um(sv, grid_um=g)
        n = len(sv)
        shanks.append(
            HerbsShank(
                ccf_um=ccf,
                regions=labels[offset : offset + n] if labels else [""] * n,
            )
        )
        offset += n
    return shanks
