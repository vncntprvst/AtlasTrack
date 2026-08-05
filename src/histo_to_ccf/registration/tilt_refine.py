"""Per-section plane-tilt refinement.

DeepSlice (or a hand-assigned plane) can leave a small residual tilt: the section's
left and right sides then sample slightly different atlas AP levels, so a paramedian
structure sits on tissue on one side but falls into a gap on the other (the classic
"vestibular nucleus over the brainstem-cerebellum split"). This module searches a
small perturbation of the anchoring's tilt terms that best fits the atlas plane to
the section.

Two tilt terms are searched (in atlas-voxel units, added to the anchoring):
- ``ux`` (index 3): AP change across the **horizontal** axis -> the left/right tilt.
- ``vx`` (index 6): AP change across the **vertical** axis -> the dorsal/ventral tilt.

A full elastix fit per candidate would be too slow for a grid, so a cheap
pre-align + mutual-information **proxy** ranks candidates; only the top few are
confirmed with the real registration (whose residual is the final objective).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from histo_to_ccf.atlas.planes import Anchoring, sample_plane

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else a


def _mutual_information(a: np.ndarray, b: np.ndarray, mask: np.ndarray, bins: int = 32) -> float:
    """MI of two normalized images over ``mask`` (both already in [0, 1])."""
    av, bv = a[mask], b[mask]
    if av.size < 100:
        return 0.0
    h, _, _ = np.histogram2d(av, bv, bins=bins, range=[[0, 1], [0, 1]])
    pxy = h / h.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    return float((pxy[nz] * np.log(pxy[nz] / denom[nz])).sum())


def tilt_proxy_score(
    section_image: np.ndarray,
    reference_volume: np.ndarray,
    anchoring: Anchoring,
) -> float:
    """Cheap tilt-quality score: MI of the atlas plane vs the section after a
    closed-form silhouette pre-align (no B-spline). Higher = better tilt.
    """
    from histo_to_ccf.registration.elastix_bspline import (
        _affine_xy_to_sitk,
        _resample,
        atlas_foreground_mask,
    )
    from histo_to_ccf.registration.masks import moment_similarity, section_tissue_mask

    import SimpleITK as sitk

    h, w = section_image.shape[:2]
    ref = sample_plane(reference_volume, anchoring, (h, w), order=1, out_dtype=np.float32)
    if float(ref.max()) - float(ref.min()) < 1e-6:
        return -1.0
    ref_n = _norm(ref)
    tissue = section_image
    if tissue.ndim == 3:
        tissue = tissue[..., :3].astype(np.float32).mean(-1)
    tis_n = _norm(tissue)
    amask = atlas_foreground_mask(ref_n)
    tmask = section_tissue_mask(tis_n)
    s_xy = moment_similarity(amask.astype(np.uint8), tmask.astype(np.uint8), isotropic=True)
    inv = _affine_xy_to_sitk(s_xy).GetInverse()
    ref_w = _norm(_resample(ref_n, inv, sitk.sitkLinear))
    overlap = (ref_w > 0.02) & tmask
    return _mutual_information(ref_w, tis_n, overlap)


def _perturbed(anchoring: Anchoring, d_ux: float, d_vx: float) -> Anchoring:
    a = list(anchoring.as_tuple())
    a[3] += d_ux   # AP-per-horizontal (left/right tilt)
    a[6] += d_vx   # AP-per-vertical (dorsal/ventral tilt)
    return Anchoring.from_iterable(a)


def refine_tilt(
    section_image: np.ndarray,
    atlas: "BrainGlobeAtlas",
    anchoring: Anchoring,
    *,
    reference_volume: np.ndarray | None = None,
    max_delta: float = 8.0,
    step: float = 4.0,
    min_improvement: float = 0.006,
    search_dv: bool = True,
    register_kwargs: dict | None = None,
) -> tuple[Anchoring, dict]:
    """Return a conservatively tilt-refined anchoring (+info).

    Coordinate-descent over ``ux`` (left/right tilt) then ``vx`` (dorsal/ventral),
    using the **real** elastix residual as the objective (the cheap proxy only finds
    the neighbourhood, not the optimum). Two safety rails keep it from chasing a big
    tilt for a marginal residual gain (residual is an imperfect proxy for anatomical
    correctness):

    - the search range is small (``±max_delta`` atlas voxels), and
    - a perturbation is accepted only if it beats the current best by more than
      ``min_improvement`` residual.

    ``register_kwargs`` are forwarded to :func:`register_section_image` for every
    candidate - pass the pipeline's settings so the residual is comparable. Returns
    the input anchoring unchanged when nothing clears the bar.
    """
    from histo_to_ccf.registration.pipeline import register_section_image

    ref_vol = reference_volume if reference_volume is not None else atlas.reference
    rk = dict(register_kwargs or {})
    rk.pop("reference_volume", None)
    rk.setdefault("boundary_snap", False)  # residual is pre-snap; skip its cost

    def residual_at(du: float, dv: float) -> float:
        a = _perturbed(anchoring, du, dv)
        try:
            reg, _ = register_section_image(
                section_image, atlas, anchoring=a, reference_volume=ref_vol, **rk
            )
        except Exception:  # noqa: BLE001 - a degenerate tilt must not abort the search
            return float("inf")
        return float(reg.residual) if reg.residual is not None else float("inf")

    grid = [float(d) for d in np.arange(-max_delta, max_delta + 1e-6, step) if abs(d) > 1e-9]
    base_r = residual_at(0.0, 0.0)
    best_du, best_dv, best_r = 0.0, 0.0, base_r
    n = 1

    axes = ["ux"] + (["vx"] if search_dv else [])
    for axis in axes:
        for d in grid:
            du = d if axis == "ux" else best_du
            dv = d if axis == "vx" else best_dv
            if (du, dv) == (best_du, best_dv):
                continue
            r = residual_at(du, dv)
            n += 1
            # Accept only a MEANINGFUL improvement, so a marginal residual gain can't
            # pull the plane into a large (overfit) tilt.
            if r < best_r - min_improvement:
                best_du, best_dv, best_r = du, dv, r

    best_anchoring = _perturbed(anchoring, best_du, best_dv)
    info = {
        "refined": (best_du, best_dv) != (0.0, 0.0),
        "d_ux": best_du,
        "d_vx": best_dv,
        "residual": best_r,
        "baseline_residual": base_r,
        "improvement": (base_r - best_r) if np.isfinite(base_r) else None,
        "n_evaluated": n,
    }
    return best_anchoring, info
