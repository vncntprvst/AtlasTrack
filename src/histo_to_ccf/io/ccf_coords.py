"""Allen CCF coordinate helpers — physical extents, bregma↔CCF, atlas surface lookup.

Ported from legacy/HERBS_to_AllenCCF/probe_visualization.py.
"""
from __future__ import annotations

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
