"""Allen CCF coordinate helpers - physical extents, bregma↔CCF, atlas surface lookup.

Ported from legacy/HERBS_to_AllenCCF/probe_visualization.py.

The module-level constants describe the **Allen** CCFv3 and remain the defaults, but
they are not true of every BrainGlobe atlas: see :func:`anchors_for_atlas_name`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Per-atlas anchors
# ---------------------------------------------------------------------------
#
# Bregma is a *skull* landmark. Its position inside an atlas volume is an empirical
# measurement, not something derivable from the voxel grid, so it has to be tabulated
# per atlas family rather than guessed. Getting this wrong is silent: every
# bregma-relative and Paxinos coordinate shifts, and nothing in the output says so.
#
# Keys are matched against the atlas name by prefix, so all resolutions of a family
# ("allen_mouse_10um", "allen_mouse_25um", ...) share an entry.
#
#   allen_mouse         5400 µm - Allen forum measurement; see BREGMA_AP_FROM_ORIGIN_UM.
#   kim_mouse           5400 µm - the Chon/Kim atlas is a *re-annotation* of Allen CCFv3
#                       on the identical (528, 320, 456) 25 µm grid, so it inherits
#                       Allen's frame. Confirmed here: the AP centroids of IO, DN and
#                       MED sit within 43 µm (< 2 voxels) of their Allen counterparts.
#                       Kim's acronym set overlaps Allen's without being a superset,
#                       so structure-matched calibration is unreliable for it - some
#                       shared acronyms name different delineations entirely.
#   ccfv3augmented      5746 µm - the BBP augmented CCFv3 is Allen CCFv3 padded along
#   _mouse              AP (566 slices, not 528). Measured as a rigid posterior shift
#                       of +345.7 ± 2.5 µm over 25 compact nuclei spanning AP 6.7-12.5
#                       mm (VII, IO, PG, VI, XII, AMB, LC, SNr, VTA, ...), whose voxel
#                       counts agree with Allen's to better than 1% - i.e. the same
#                       annotation relocated, not redrawn. 5400 + 346 = 5746.
#
# Structures are the right probe for this and *whole-brain* regions are not: the
# augmented atlas redraws its parent regions, so the centroids of MY, CB and CTX shift
# by +438 to +567 µm while the nuclei inside them all move by +346.
BREGMA_AP_BY_ATLAS: dict[str, float] = {
    "allen_mouse": 5400.0,
    "kim_mouse": 5400.0,
    "ccfv3augmented_mouse": 5746.0,
    # The isotropic Chon/Kim v2 samples the same volume at 20 um, but its annotation
    # is translated +102 um along AP relative to the 25 um release. Measured as volume
    # centroids over the 811 structures whose volume agrees between the two releases
    # within 10%: AP +101.8 +/- 26.6 um, DV +5.8, ML +7.7, and a shift-vs-AP slope of
    # +0.0008 um/um - i.e. a pure translation, not a scaling difference. Longest
    # prefix wins in the lookup below, so this beats the "kim_mouse" entry.
    "kim_mouse_isotropic": 5502.0,
}


@dataclass(frozen=True)
class AtlasAnchors:
    """Where bregma and the midline sit in one atlas's voxel frame.

    ``bregma_ap_um`` is ``None`` for an atlas we have no measurement for. Callers must
    treat that as "cannot express this in stereotaxic coordinates" rather than
    substituting Allen's value - an unrecognised atlas is exactly the case where the
    Allen anchor is most likely to be wrong.
    """

    atlas_name: str
    bregma_ap_um: float | None
    midline_ml_um: float
    bregma_dv_um: float = BREGMA_DV_FROM_ORIGIN_UM
    ap_um: float = AP_UM
    dv_um: float = DV_UM
    ml_um: float = ML_UM

    @property
    def has_bregma(self) -> bool:
        return self.bregma_ap_um is not None

    def require_bregma(self) -> float:
        """The bregma AP anchor, or a targeted error naming the atlas."""
        if self.bregma_ap_um is None:
            raise ValueError(
                f"No bregma anchor is known for atlas {self.atlas_name!r}, so "
                "bregma-relative and Paxinos coordinates cannot be computed for it. "
                f"Known atlas families: {sorted(BREGMA_AP_BY_ATLAS)}. Register one by "
                "adding an entry to histo_to_ccf.io.ccf_coords.BREGMA_AP_BY_ATLAS."
            )
        return float(self.bregma_ap_um)


def bregma_ap_from_origin_um(atlas_name: str | None) -> float | None:
    """Bregma AP (µm from the anterior edge) for a BrainGlobe atlas name.

    Name lookup only - deliberately does not load the atlas, so the export path stays
    cheap. Returns ``None`` when the atlas is unknown.
    """
    if not atlas_name:
        return None
    name = str(atlas_name).strip().lower()
    for prefix, value in sorted(
        BREGMA_AP_BY_ATLAS.items(), key=lambda kv: -len(kv[0])
    ):
        if name.startswith(prefix):
            return float(value)
    return None


def bregma_ap_for_display(atlas_name: str | None) -> float:
    """Bregma AP for showing an AP as bregma-relative in the UI.

    Unlike :func:`bregma_ap_from_origin_um` this never returns ``None``: a spin box
    has to show *some* number. Unknown atlases fall back to the Allen anchor, and the
    caller is expected to say so. Do **not** use this for exports - those should raise
    via :meth:`AtlasAnchors.require_bregma` rather than ship a guessed frame.
    """
    return bregma_ap_from_origin_um(atlas_name) or BREGMA_AP_FROM_ORIGIN_UM


def anchors_for_atlas_name(atlas_name: str | None) -> AtlasAnchors:
    """Anchors from an atlas *name*, using the Allen grid extents.

    Use :func:`anchors_for_atlas` instead when the atlas object is loaded - it reads
    the true extents off the voxel grid rather than assuming Allen's.
    """
    return AtlasAnchors(
        atlas_name=str(atlas_name or ""),
        bregma_ap_um=bregma_ap_from_origin_um(atlas_name),
        midline_ml_um=MIDLINE_ML_UM,
    )


def anchors_for_atlas(atlas: "BrainGlobeAtlas") -> AtlasAnchors:
    """Anchors for a loaded atlas: extents and midline off its grid, bregma by name.

    The midline is half the ML extent, which is exact for the symmetric mouse atlases
    (all of Allen/Kim/augmented are 456 x 25 µm, giving the familiar 5700 µm).
    """
    name = getattr(atlas, "atlas_name", "") or ""
    shape = tuple(int(s) for s in atlas.annotation.shape)
    ap_res, dv_res, ml_res = atlas_resolution_um(atlas)
    ml_extent = shape[2] * ml_res
    return AtlasAnchors(
        atlas_name=str(name),
        bregma_ap_um=bregma_ap_from_origin_um(name),
        midline_ml_um=ml_extent / 2.0,
        ap_um=shape[0] * ap_res,
        dv_um=shape[1] * dv_res,
        ml_um=ml_extent,
    )


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
    anchors: AtlasAnchors | None = None,
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

    ``anchors`` re-bases the transform onto a non-Allen atlas (see
    :func:`anchors_for_atlas_name`). Omitting it keeps the Allen anchors, which is
    correct for ``allen_mouse_*`` and ``kim_mouse_*`` but **not** for
    ``ccfv3augmented_mouse_*``, whose frame is shifted 346 µm posteriorly.
    """
    if alignment not in PAXINOS_ALIGNMENTS:
        raise ValueError(
            f"unknown alignment {alignment!r}; options: {sorted(PAXINOS_ALIGNMENTS)}"
        )
    pitch_deg, s_ap, s_ml, s_dv, bregma_dv = PAXINOS_ALIGNMENTS[alignment]
    bregma_ap = BREGMA_AP_FROM_ORIGIN_UM
    midline_ml = MIDLINE_ML_UM
    if anchors is not None:
        bregma_ap = anchors.require_bregma()
        midline_ml = float(anchors.midline_ml_um)
        # "none" is the no-correction baseline: a pure mirror with DV untouched.
        if bregma_dv:
            bregma_dv = float(anchors.bregma_dv_um)

    ap_c = np.asarray(ap_um, dtype=float) - bregma_ap  # + posterior
    ml_c = np.asarray(ml_um, dtype=float) - midline_ml  # + right (CCF)
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
