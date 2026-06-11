"""Allen CCF coordinate helpers - physical extents, bregma↔CCF, atlas surface lookup.

Ported from legacy/HERBS_to_AllenCCF/probe_visualization.py.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


# Allen CCF physical extents (µm). These are independent of atlas voxel resolution.
AP_UM = 13200.0
DV_UM = 8000.0
ML_UM = 11400.0

# Stereotaxic-frame anchors (µm in CCF).
BREGMA_AP_UM = 6600.0
MIDLINE_ML_UM = 5700.0

# Bregma position measured from the anterior-most coronal slice of the atlas
# volume (i.e. in the same "distance from the AP origin" frame the GUI uses to
# index the reference volume: ap_index = ap_um / ap_resolution). Allen CCFv3
# places bregma ≈ 5400 µm caudal to the anterior edge. Used to show AP as a
# bregma-relative value (bregma = 0, anterior positive) in the atlas browser
# while still storing the absolute-from-origin AP that the resampler expects.
BREGMA_AP_FROM_ORIGIN_UM = 5400.0

# CCF DV (µm from the dorsal-most slice) of bregma: ~44 voxels at 10 µm. Used to
# re-reference DV to the bregma surface for stereotaxic output. ESTIMATE.
BREGMA_DV_FROM_ORIGIN_UM = 440.0


def relative_ap_ml_dv_to_ccf(ap_rel: float, ml_rel: float, dv_um: float) -> np.ndarray:
    """Convert bregma/midline-relative AP/ML/DV (µm) to Allen CCF AP/ML/DV (µm).

    Positive ``ap_rel`` is anterior to bregma. Positive ``ml_rel`` is lateral on
    the right hemisphere. CCF AP and ML increase in the opposite directions
    used by stereotaxic conventions, so both are subtracted from the anchors.
    """
    return np.array(
        [BREGMA_AP_UM - ap_rel, MIDLINE_ML_UM - ml_rel, dv_um],
        dtype=float,
    )


# CCF -> stereotaxic (Paxinos/bregma) alignment presets.
#
# The Allen CCFv3 template is pitched ~5 deg nose-DOWN relative to a flat-skull
# stereotaxic frame (bregma & lambda at equal DV), so a pure mirror is only correct
# near bregma; the error grows with distance (especially in DV / far AP). Each preset
# therefore un-pitches by 5 deg in the sagittal plane about the ML axis through
# bregma, then applies published per-axis scale factors. These are ESTIMATES with
# real variance - validate against pilot injections / histology.
#
# Refs: community.brain-map.org/t/.../1858 ; virtualbrainlab.org Pinpoint in-vivo
# alignment (Qiu 2018 is Pinpoint's recommended default; Dorr 2008 alternative).
#
# value = (pitch_deg, scale_ap, scale_ml, scale_dv, bregma_dv_um)
PAXINOS_ALIGNMENTS: dict[str, tuple[float, float, float, float, float]] = {
    "none":        (0.0, 1.000, 1.000, 1.0000, 0.0),    # legacy pure mirror (no tilt)
    "allen_forum": (5.0, 1.000, 1.000, 0.9434, BREGMA_DV_FROM_ORIGIN_UM),
    "qiu2018":     (5.0, 1.031, 0.952, 0.8850, BREGMA_DV_FROM_ORIGIN_UM),
    "dorr2008":    (5.0, 1.087, 1.000, 0.9520, BREGMA_DV_FROM_ORIGIN_UM),
}
DEFAULT_PAXINOS_ALIGNMENT = "qiu2018"


def ccf_um_to_paxinos_mm(
    ap_um: np.ndarray | float,
    ml_um: np.ndarray | float,
    dv_um: np.ndarray | float,
    *,
    alignment: str = DEFAULT_PAXINOS_ALIGNMENT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allen CCF (AP, ML, DV) µm -> Paxinos (Franklin-Paxinos) stereotaxic mm.

    Steps (see :data:`PAXINOS_ALIGNMENTS`):

    1. Bregma-relative µm, CCF signs: ``ap-5400``, ``ml-5700``, ``dv-bregma_dv``.
    2. Un-pitch by ``pitch_deg`` in the sagittal (AP-DV) plane about the ML axis.
    3. Per-axis scale (published estimates).
    4. Paxinos signs + mm: AP anterior-positive, ML 0 at midline, DV ventral (depth).

    With ``alignment="none"`` this reduces to the plain mirror
    ``(5400-AP, 5700-ML, DV)/1000`` (no tilt, DV = raw CCF depth) - useful as a
    no-correction baseline. The default ``"qiu2018"`` matches Pinpoint's recommended
    transform. Uses the corrected bregma AP (5400 µm); the legacy 6600 placed regions
    ~1.2 mm too anterior.
    """
    if alignment not in PAXINOS_ALIGNMENTS:
        raise ValueError(
            f"unknown alignment {alignment!r}; options: {sorted(PAXINOS_ALIGNMENTS)}"
        )
    pitch_deg, s_ap, s_ml, s_dv, bregma_dv = PAXINOS_ALIGNMENTS[alignment]

    ap_c = np.asarray(ap_um, dtype=float) - BREGMA_AP_FROM_ORIGIN_UM  # + posterior
    ml_c = np.asarray(ml_um, dtype=float) - MIDLINE_ML_UM             # + right (CCF)
    dv_c = np.asarray(dv_um, dtype=float) - bregma_dv                 # + ventral

    th = math.radians(pitch_deg)
    cos_t, sin_t = math.cos(th), math.sin(th)
    ap_r = ap_c * cos_t - dv_c * sin_t
    dv_r = ap_c * sin_t + dv_c * cos_t

    ap_mm = -(ap_r * s_ap) / 1000.0   # anterior positive
    ml_mm = -(ml_c * s_ml) / 1000.0   # 0 at midline
    dv_mm = (dv_r * s_dv) / 1000.0    # ventral positive (depth below bregma)
    return ap_mm, ml_mm, dv_mm


def atlas_resolution_um(atlas: "BrainGlobeAtlas") -> tuple[float, float, float]:
    """Return ``(ap_res, dv_res, ml_res)`` µm for a BrainGlobe atlas."""
    resolution = atlas.resolution
    if isinstance(resolution, tuple | list | np.ndarray):
        return float(resolution[0]), float(resolution[1]), float(resolution[2])
    value = float(resolution)
    return value, value, value


def dorsal_surface_dv_um(
    atlas: "BrainGlobeAtlas", ap_um: float, ml_um: float
) -> float | None:
    """Return the dorsal-most non-background DV (µm) at one AP/ML coordinate.

    Searches a small AP/ML neighborhood when the exact column lies outside the
    brain (e.g. due to rounding). Returns ``None`` if no surface is found.
    """
    annotation = atlas.annotation
    ap_res, dv_res, ml_res = atlas_resolution_um(atlas)
    ap_index = int(np.clip(round(ap_um / ap_res), 0, annotation.shape[0] - 1))
    ml_index = int(np.clip(round(ml_um / ml_res), 0, annotation.shape[2] - 1))

    def surface_index(a_idx: int, m_idx: int) -> int | None:
        hits = np.flatnonzero(annotation[a_idx, :, m_idx] > 0)
        return int(hits[0]) if len(hits) else None

    hit = surface_index(ap_index, ml_index)
    if hit is not None:
        return hit * dv_res

    candidates: list[int] = []
    for radius in range(1, 9):
        for da in range(-radius, radius + 1):
            for dm in range(-radius, radius + 1):
                if abs(da) != radius and abs(dm) != radius:
                    continue
                a_idx = int(np.clip(ap_index + da, 0, annotation.shape[0] - 1))
                m_idx = int(np.clip(ml_index + dm, 0, annotation.shape[2] - 1))
                near = surface_index(a_idx, m_idx)
                if near is not None:
                    candidates.append(near)
        if candidates:
            return min(candidates) * dv_res
    return None
