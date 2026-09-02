"""Probe geometry constants and 3D mesh builders.

Coordinate convention: all CCF positions stored as (AP, ML, DV) in µm,
matching the project schema.  Plotly display uses x=ML, y=AP, z=DV.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shank physical constants (µm)
# ---------------------------------------------------------------------------
SHANK_WIDTH_UM: float = 70.0
SHANK_THICKNESS_UM: float = 24.0
SHANK_TIP_LENGTH_UM: float = 175.0
SHANK_PITCH_UM: float = 250.0
ELECTRODE_COLUMN_CENTER_UM: float = 16.0


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def probe_prism_mesh(
    tip_ccf: tuple[float, float, float],
    entry_ccf: tuple[float, float, float],
    *,
    shank_width_um: float = SHANK_WIDTH_UM,
    shank_thickness_um: float = SHANK_THICKNESS_UM,
    tip_length_um: float = SHANK_TIP_LENGTH_UM,
) -> dict[str, list[float]]:
    """Build a Neuropixels-style rectangular prism mesh for one shank.

    Returns a dict with keys ``x, y, z, i, j, k`` suitable for
    ``plotly.graph_objects.Mesh3d(**mesh)``.

    Input coordinates are (AP, ML, DV) µm; output is in Plotly space
    (x=ML, y=AP, z=DV).
    """
    tip = np.array(tip_ccf, dtype=float)    # (AP, ML, DV)
    entry = np.array(entry_ccf, dtype=float)

    axis = entry - tip
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1.0:
        axis_len = 1.0
    axis_hat = axis / axis_len

    # Construct orthogonal width/thickness vectors.
    # width_hat: perpendicular to axis, in the ML–DV plane.
    # thickness_hat: perpendicular to both.
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(axis_hat, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    width_hat = np.cross(axis_hat, ref)
    w_len = np.linalg.norm(width_hat)
    width_hat = width_hat / w_len if w_len > 1e-9 else np.array([1.0, 0.0, 0.0])
    thick_hat = np.cross(axis_hat, width_hat)
    t_len = np.linalg.norm(thick_hat)
    thick_hat = thick_hat / t_len if t_len > 1e-9 else np.array([0.0, 0.0, 1.0])

    hw = shank_width_um / 2.0
    ht = shank_thickness_um / 2.0
    tip_start = tip + axis_hat * tip_length_um  # base of taper on shank body

    # 8 corners of the rectangular body.
    corners = []
    for s in (tip_start, entry):
        for w in (-hw, hw):
            for t in (-ht, ht):
                corners.append(s + w * width_hat + t * thick_hat)

    # 2 tip taper points.
    tip_left = tip + (-hw / 2) * width_hat
    tip_right = tip + (hw / 2) * width_hat
    corners.append(tip_left)
    corners.append(tip_right)

    verts = np.array(corners)  # (10, 3) in (AP, ML, DV)

    # Map to Plotly x=ML, y=AP, z=DV.
    x = verts[:, 1].tolist()
    y = verts[:, 0].tolist()
    z = verts[:, 2].tolist()

    # Faces (triangles).  Indices into the 10-vertex array:
    # 0-7 = body corners (tip_start plane then entry plane), 8-9 = taper tip.
    # Body: two rectangular faces per pair of corners.
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []

    def _quad(a: int, b: int, c: int, d: int) -> None:
        i.extend([a, a]); j.extend([b, c]); k.extend([c, d])

    # Bottom (t=-ht): 0,1,5,4
    _quad(0, 1, 5, 4)
    # Top (t=+ht): 2,3,7,6
    _quad(2, 3, 7, 6)
    # Left (w=-hw): 0,2,6,4
    _quad(0, 2, 6, 4)
    # Right (w=+hw): 1,3,7,5
    _quad(1, 3, 7, 5)
    # Entry cap: 4,5,7,6
    _quad(4, 5, 7, 6)
    # Taper faces (tip → body base)
    i.extend([8, 9, 8, 9])
    j.extend([0, 1, 2, 3])
    k.extend([1, 3, 0, 2])

    return {"x": x, "y": y, "z": z, "i": i, "j": j, "k": k}


def shank_offsets(n_shanks: int, pitch_um: float = SHANK_PITCH_UM) -> np.ndarray:
    """Return ML offsets (µm) for each shank, centered around 0."""
    offsets = np.arange(n_shanks, dtype=float) * pitch_um
    offsets -= offsets.mean()
    return offsets
