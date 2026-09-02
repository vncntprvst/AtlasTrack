"""Internal-feature snap: align the atlas ventricle + midline to the section.

Why this exists
---------------
:mod:`boundary_snap` only corrects the atlas **outer silhouette**. The intensity
B-spline (elastix MI) is blind to featureless CSF and to the midline raphe, so on
brainstem sections the atlas 4th ventricle and the ventral midline notch land off
the tissue even when the silhouette fits (the classic "the brainstem midline is
askew, shifted sideways"). This module adds a conservative, **gated**, fold-proof
correction that pulls those interior features onto the section's own anatomy:

- **Ventricle.** Detect the section's dark CSF cavity near where the atlas
  ventricle (V4 / V3 / aqueduct) landed. Only when that cavity is *clean* (compact
  and plausibly sized - not a ragged sectioning cleft) do we open the atlas
  ventricle onto it, via a clamped region (ellipse) affine so it can't blow up.
- **Midline.** Nudge the atlas midline horizontally onto the section's symmetry
  axis, with a **small cap** and *skipped entirely when the implied shift is large*
  - a large shift means the whole-section symmetry axis is biased (asymmetric
  tissue / label), and applying it over-corrects. This gate is what keeps the snap
  from making already-centred sections worse.

Both feature sets become TPS control points, the tissue contour is pinned so only
the interior moves, and the whole thing is fit with the **same fold-proof TPS** as
:mod:`boundary_snap` (:func:`boundary_snap.fit_foldproof_tps`). The function returns
``None`` (no correction) whenever the gates aren't met or the field would fold - it
must never worsen a section.

Design notes
------------
- **Headless core.** Everything here takes *section-frame* numpy masks / images -
  no ``BrainGlobeAtlas`` dependency - so it is unit-testable on synthetic data. The
  pipeline computes the warped atlas ventricle / midline masks and calls in.
- **Composes like the boundary snap.** The returned object is a
  ``sitk.DisplacementFieldTransform`` in the section (moving) frame; the caller
  composes it with :func:`boundary_snap.compose_snap`, so every downstream consumer
  (overlay, probe->CCF, ``.h5`` persistence, the iterative inverse) is unchanged.
"""
from __future__ import annotations

import numpy as np

# Gates chosen from LO_06 validation (brainstem sections 1-11): the ventricle
# opening is a clear win on clean cavities (s5/s8) and the midline cap prevents the
# over-correction seen on biased-axis sections (s14, ~21 px implied shift).
_DARK_THRESH = 0.17          # luminance below this (0..1) reads as CSF/void
_CAVITY_DILATE = 25          # search radius around the atlas ventricle (px)
_MIN_CAVITY_AREA = 300       # px; below this there's nothing worth opening onto
_MIN_SOLIDITY = 0.55         # reject ragged clefts (area / convex-hull area)
_CAVITY_RATIO = (1.3, 6.0)   # cavity / atlas-ventricle area: a real, sane mismatch
_AFFINE_SCALE = (0.85, 2.4)  # clamp the region-affine singular values
_MAX_CENTROID_SHIFT = 55.0   # px; clamp how far the ventricle centroid may move
_N_VENT_BOUNDARY = 24        # atlas-ventricle boundary samples -> control points
_MIDLINE_MAX_SHIFT = 15.0    # px; per-row cap on the midline horizontal nudge
_MIDLINE_MAX_MEDIAN = 12.0   # px; skip midline entirely if the median shift exceeds
_N_MIDLINE = 20              # midline ridge samples -> control points
_N_CONTOUR_PINS = 44         # tissue-contour anchors (source == target)


def _lum(section_image: np.ndarray) -> np.ndarray:
    """Normalised luminance in [0, 1] from an RGB or grayscale section crop."""
    arr = np.asarray(section_image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    hi = float(arr.max())
    return arr / hi if hi > 0 else arr


def detect_cavity(
    section_lum: np.ndarray,
    brain: np.ndarray,
    atlas_ventricle: np.ndarray,
    *,
    dark_thresh: float = _DARK_THRESH,
    dilate: int = _CAVITY_DILATE,
    min_area: int = _MIN_CAVITY_AREA,
    min_solidity: float = _MIN_SOLIDITY,
    ratio_range: tuple[float, float] = _CAVITY_RATIO,
) -> np.ndarray | None:
    """The section's CSF cavity matching the atlas ventricle, or ``None`` if unclean.

    Takes the largest dark component inside ``brain`` near ``atlas_ventricle`` and
    accepts it only if it is compact (``solidity > min_solidity``) and its area is a
    plausible multiple of the atlas ventricle (``ratio_range``). Rejecting keeps the
    snap off ragged sectioning clefts and off sections where the atlas ventricle is
    already the right size.
    """
    from scipy import ndimage as ndi
    from skimage import measure

    av = np.asarray(atlas_ventricle, dtype=bool)
    if int(av.sum()) < 80:
        return None
    near = ndi.binary_dilation(av, iterations=dilate)
    dark = (np.asarray(section_lum) < dark_thresh) & np.asarray(brain, dtype=bool) & near
    dark = ndi.binary_closing(ndi.binary_opening(dark, iterations=1), iterations=2)
    lab, n = ndi.label(dark)
    if n == 0:
        return None
    # Pick the component overlapping the atlas ventricle most (the true cavity).
    av_core = ndi.binary_dilation(av, iterations=8)
    best, best_ov = 0, -1
    for i in range(1, n + 1):
        ov = int(((lab == i) & av_core).sum())
        if ov > best_ov:
            best_ov, best = ov, i
    if best_ov <= 0:
        return None
    cavity = lab == best
    area = int(cavity.sum())
    ratio = area / max(int(av.sum()), 1)
    if area < min_area or not (ratio_range[0] < ratio < ratio_range[1]):
        return None
    props = measure.regionprops(cavity.astype(np.uint8))
    if not props or float(props[0].solidity) < min_solidity:
        return None
    return cavity


def _moments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(float)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T) + np.eye(2)  # +I regularises a near-degenerate blob
    return c, cov


def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return v @ np.diag(np.sqrt(np.clip(w, 1e-6, None))) @ v.T


def _clamped_region_affine(
    src_mask: np.ndarray,
    dst_mask: np.ndarray,
    *,
    scale_range: tuple[float, float] = _AFFINE_SCALE,
    max_shift: float = _MAX_CENTROID_SHIFT,
):
    """A clamped ellipse-match affine mapping ``src_mask`` onto ``dst_mask``.

    Matches the two regions' mean + covariance (an ellipse), then clamps the linear
    part's singular values and the centroid translation so a noisy/oversized cavity
    can't produce a huge warp. Returns ``A(p)`` mapping a source point to its image.
    """
    cs, ss = _moments(src_mask)
    ct, st = _moments(dst_mask)
    m = _sqrtm_psd(st) @ np.linalg.inv(_sqrtm_psd(ss))
    u, s, vt = np.linalg.svd(m)
    m = u @ np.diag(np.clip(s, *scale_range)) @ vt
    shift = ct - cs
    norm = float(np.linalg.norm(shift))
    if norm > max_shift:
        shift = shift * (max_shift / norm)
    ct = cs + shift

    def apply(p: np.ndarray) -> np.ndarray:
        return ct + m @ (np.asarray(p, dtype=float) - cs)

    return apply, cs, ct


def ventricle_correspondences(
    atlas_ventricle: np.ndarray,
    cavity: np.ndarray,
    *,
    n_boundary: int = _N_VENT_BOUNDARY,
) -> tuple[np.ndarray, np.ndarray]:
    """Control points opening the atlas ventricle boundary onto ``cavity``.

    Samples the atlas-ventricle boundary and maps each point (plus the centroid)
    through the clamped region-affine. Returns ``(src, dst)`` (N, 2) in (x, y).
    """
    from atlastrack.registration.transforms import annotation_boundaries

    av = np.asarray(atlas_ventricle, dtype=bool)
    apply, cs, ct = _clamped_region_affine(av, np.asarray(cavity, dtype=bool))
    bnd = annotation_boundaries(av.astype(int)) & av
    ys, xs = np.nonzero(bnd)
    if xs.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    if xs.size > n_boundary:
        sel = np.linspace(0, xs.size - 1, n_boundary).round().astype(int)
        ys, xs = ys[sel], xs[sel]
    src = [[float(x), float(y)] for x, y in zip(xs, ys)]
    dst = [apply((x, y)).tolist() for x, y in zip(xs, ys)]
    src.append([float(cs[0]), float(cs[1])])  # centroid moves the fill, not just rim
    dst.append([float(ct[0]), float(ct[1])])
    return np.asarray(src, dtype=float), np.asarray(dst, dtype=float)


def midline_correspondences(
    atlas_midline: np.ndarray,
    tissue: np.ndarray,
    *,
    n: int = _N_MIDLINE,
    max_shift: float = _MIDLINE_MAX_SHIFT,
    max_median_shift: float = _MIDLINE_MAX_MEDIAN,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Control points nudging the atlas midline onto the section's symmetry axis.

    ``atlas_midline`` is a thin section-frame mask of where the atlas midline
    landed. Its per-row x is pulled toward the section's reflection-symmetry axis
    (:func:`landmarks.midline.estimate_midline`). Each row's shift is capped at
    ``max_shift``; if the **median** shift exceeds ``max_median_shift`` the axis is
    judged biased and the whole midline term is dropped (returns ``None``) - this is
    the guard against over-correcting an asymmetric section.
    """
    from atlastrack.landmarks.midline import estimate_midline

    am = np.asarray(atlas_midline, dtype=bool)
    if int(am.sum()) < 10:
        return None
    ml = estimate_midline(np.asarray(tissue, dtype=bool), refine=True)
    if ml is None:
        return None
    cx, cy = ml.centroid_px
    dx, dy = ml.direction
    if abs(dy) < 1e-6:
        return None

    rows = np.unique(np.nonzero(am)[0])
    if rows.size == 0:
        return None
    if rows.size > n:
        rows = rows[np.linspace(0, rows.size - 1, n).round().astype(int)]

    x_atlas = np.array([np.nonzero(am[y])[0].mean() for y in rows])
    x_sec = cx + ((rows - cy) / dy) * dx
    shift = x_sec - x_atlas
    if float(abs(np.median(shift))) > max_median_shift:
        return None  # biased axis - do not apply
    shift = np.clip(shift, -max_shift, max_shift)
    src = np.stack([x_atlas, rows.astype(float)], axis=1)
    dst = np.stack([x_atlas + shift, rows.astype(float)], axis=1)
    return src, dst


def contour_pins(
    tissue: np.ndarray, *, n: int = _N_CONTOUR_PINS
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` anchor points on the tissue contour with ``source == target``.

    Pinning the outer boundary keeps the correction interior-only (it can't drift
    the whole atlas) and stiffens the TPS against folding.
    """
    from scipy import ndimage as ndi

    tissue = np.asarray(tissue, dtype=bool)
    edge = tissue & ~ndi.binary_erosion(tissue)
    ey, ex = np.nonzero(edge)
    if ex.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    sel = np.linspace(0, ex.size - 1, min(n, ex.size)).round().astype(int)
    pins = np.stack([ex[sel], ey[sel]], axis=1).astype(float)
    return pins, pins.copy()


def internal_feature_snap_transform(
    atlas_ventricle: np.ndarray,
    section_image: np.ndarray,
    tissue: np.ndarray,
    *,
    brain: np.ndarray | None = None,
    atlas_midline: np.ndarray | None = None,
    use_midline: bool = True,
):
    """Build a fold-free snap aligning the atlas ventricle + midline to the section.

    Parameters are all *section-frame* arrays:

    - ``atlas_ventricle`` - bool mask of the atlas V4/V3/AQ as it currently landed.
    - ``section_image`` - the RGB (or gray) section crop; luminance is derived here.
    - ``tissue`` - bool tissue silhouette (contour pins + midline axis).
    - ``brain`` - where the atlas landed (warped extent); defaults to ``tissue``.
    - ``atlas_midline`` - optional thin mask of the atlas midline as it landed.

    Returns a ``sitk.DisplacementFieldTransform`` (compose with
    :func:`boundary_snap.compose_snap`) or ``None`` when no gated feature yields a
    correction, or when every smoothing level still folds.
    """
    from atlastrack.registration.boundary_snap import fit_foldproof_tps

    tissue = np.asarray(tissue, dtype=bool)
    shape = tissue.shape
    brain_mask = tissue if brain is None else np.asarray(brain, dtype=bool)
    lum = _lum(section_image)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []

    cavity = detect_cavity(lum, brain_mask, np.asarray(atlas_ventricle, dtype=bool))
    if cavity is not None:
        vs, vd = ventricle_correspondences(atlas_ventricle, cavity)
        if len(vs):
            src_parts.append(vs)
            dst_parts.append(vd)

    if use_midline and atlas_midline is not None:
        mid = midline_correspondences(atlas_midline, tissue)
        if mid is not None:
            src_parts.append(mid[0])
            dst_parts.append(mid[1])

    if not src_parts:
        return None  # nothing passed the gates - leave the section alone

    pin_s, pin_d = contour_pins(tissue)
    if len(pin_s):
        src_parts.append(pin_s)
        dst_parts.append(pin_d)

    src = np.concatenate(src_parts, axis=0)
    dst = np.concatenate(dst_parts, axis=0)
    return fit_foldproof_tps(src, dst, shape)
