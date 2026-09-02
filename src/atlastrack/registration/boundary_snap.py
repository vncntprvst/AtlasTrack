"""Automatic outer-contour snap: pull the registered atlas boundary onto tissue.

Why this exists
---------------
The intensity B-spline (elastix MI) aligns interior structure but does **not**
optimise the silhouette, so the atlas *outer contour* is left wherever the
affine pre-align put it - typically a few percent off the tissue border, the
"atlas lines just outside the bottom of the section" look. Measured on the real
example slide the silhouette Dice plateaus at ~0.89 no matter how the B-spline
is tuned, because MI simply doesn't see the boundary.

This module adds a final, **fold-proof** deformation that snaps the warped-atlas
silhouette onto the section's tissue silhouette - the automatic equivalent of the
manual landmark drag. It is the headless core; the GUI just toggles it on.

How it stays safe (these choices matter - don't loosen them blindly)
--------------------------------------------------------------------
- **Boundary correspondences, not a free metric.** We sample points on the warped
  atlas boundary and push each toward the nearest tissue edge (tissue signed
  distance field + its gradient). A free mean-squares B-spline on the silhouettes
  folds the asymmetric forebrain (tried and rejected, see handoff); a sparse,
  smoothed thin-plate spline cannot.
- **Drop large mismatches.** A correspondence whose push exceeds ``drop_frac`` of
  the image diagonal is *excluded entirely* (not pinned to zero - pinning next to
  a 50 px-snapped neighbour creases the field and folds it). Dropping protects
  genuinely damaged/asymmetric tissue (e.g. a torn cerebellar midline): where the
  atlas has no matching tissue edge, the atlas is left alone there.
- **Interior anchors.** A ring of interior points is pinned (zero displacement) so
  the snap is a boundary correction, not a global drift.
- **Heavy smoothing + a fold check.** The TPS uses a large ``smoothing`` and we
  verify the forward field's minimum Jacobian determinant is positive; if it
  isn't we escalate the smoothing and, failing that, return ``None`` (no snap)
  rather than ship a folded warp.

Integration: the returned object is a ``sitk.DisplacementFieldTransform`` in the
section (moving) frame mapping *where the registered atlas landed* -> *tissue*.
The pipeline composes it as ``CompositeTransform([snap, registration])`` so every
downstream consumer (overlay, probe->CCF, ``.h5`` persistence, the iterative
inverse) is unchanged - it is still one ``sitk.Transform``.
"""
from __future__ import annotations

import numpy as np

# Operating point chosen empirically on the real example slide (15 sections):
# mean silhouette Dice 0.894 -> 0.935, worst-case Jacobian +0.11 (fold-free).
_N_BOUNDARY = 56
_N_INTERIOR = 16
_DROP_FRAC = 0.06
_SMOOTHING = 2000.0
_SMOOTHING_ESCALATION = (2000.0, 4000.0, 8000.0)
_JAC_EPS = 0.02  # require min Jacobian determinant above this to accept a field


def _tissue_sdf(mask: np.ndarray) -> np.ndarray:
    """Signed distance to the mask boundary: negative inside, positive outside."""
    from scipy import ndimage as ndi

    mask = mask.astype(bool)
    return ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)


def _boundary_points(mask: np.ndarray, n: int) -> np.ndarray:
    """Evenly angularly-sampled (x, y) points on a silhouette's outer boundary."""
    from scipy import ndimage as ndi

    edge = mask & ~ndi.binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    if xs.size == 0:
        return np.empty((0, 2))
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    order = np.argsort(ang)
    xs, ys, ang = xs[order], ys[order], ang[order]
    targets = np.linspace(-np.pi, np.pi, n, endpoint=False)
    idx = np.searchsorted(ang, targets).clip(0, len(ang) - 1)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(float)
    return np.unique(pts, axis=0)


def boundary_correspondences(
    extent: np.ndarray,
    tissue: np.ndarray,
    *,
    n_boundary: int = _N_BOUNDARY,
    n_interior: int = _N_INTERIOR,
    drop_frac: float = _DROP_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Source (atlas-boundary) -> target (tissue-edge) point pairs for the TPS.

    ``extent`` is the warped-atlas silhouette in section space; ``tissue`` is the
    section tissue silhouette. Returns ``(src, dst)`` arrays of shape (N, 2) in
    (x, y). Boundary points are pushed to the nearest tissue edge; pushes larger
    than ``drop_frac`` of the diagonal are dropped; ``n_interior`` pinned interior
    anchors hold the inside still.
    """
    from scipy import ndimage as ndi

    h, w = extent.shape
    diag = float(np.hypot(h, w))
    tsdf = _tissue_sdf(tissue)
    gy, gx = np.gradient(tsdf)
    src: list[list[float]] = []
    dst: list[list[float]] = []
    for x, y in _boundary_points(extent, n_boundary):
        xi, yi = int(np.rint(y)), int(np.rint(x))
        sd = tsdf[xi, yi]
        g = np.array([gx[xi, yi], gy[xi, yi]])
        ng = float(np.linalg.norm(g))
        if ng < 1e-6:
            continue
        disp = -sd * (g / ng)  # toward the nearest tissue edge
        if float(np.hypot(*disp)) > drop_frac * diag:
            continue  # genuine mismatch / damage - leave the atlas alone here
        src.append([x, y])
        dst.append([x + disp[0], y + disp[1]])
    inner = ndi.binary_erosion(extent, iterations=max(1, int(0.12 * min(h, w))))
    iy, ix = np.nonzero(inner)
    if ix.size:
        for j in np.linspace(0, ix.size - 1, min(n_interior, ix.size)).astype(int):
            src.append([float(ix[j]), float(iy[j])])
            dst.append([float(ix[j]), float(iy[j])])
    return np.asarray(src, dtype=float), np.asarray(dst, dtype=float)


def _forward_field(
    src: np.ndarray, dst: np.ndarray, shape: tuple[int, int], smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dense (dx, dy) displacement that maps atlas-landing -> tissue (forward)."""
    from scipy.interpolate import RBFInterpolator

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    q = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)
    mapped = RBFInterpolator(src, dst, kernel="thin_plate_spline", smoothing=smoothing)(q)
    dx = (mapped[:, 0] - xx.ravel()).reshape(h, w)
    dy = (mapped[:, 1] - yy.ravel()).reshape(h, w)
    return dx, dy


def _min_jacobian(dx: np.ndarray, dy: np.ndarray) -> float:
    """Minimum Jacobian determinant of (x+dx, y+dy); <=0 means the field folds."""
    dxdx = np.gradient(dx, axis=1)
    dxdy = np.gradient(dx, axis=0)
    dydx = np.gradient(dy, axis=1)
    dydy = np.gradient(dy, axis=0)
    jac = (1.0 + dxdx) * (1.0 + dydy) - dxdy * dydx
    return float(jac.min())


def fit_foldproof_tps(
    src: np.ndarray,
    dst: np.ndarray,
    shape: tuple[int, int],
    *,
    smoothing_escalation: tuple[float, ...] = _SMOOTHING_ESCALATION,
    jac_eps: float = _JAC_EPS,
):
    """Fit a fold-free ``sitk.DisplacementFieldTransform`` from ``src -> dst`` pairs.

    Escalates the TPS smoothing until the forward field's minimum Jacobian
    determinant clears ``jac_eps`` (positive => no fold); returns ``None`` if every
    level still folds or there are too few pairs. Shared by :func:`boundary_snap`
    and the internal-feature snap so the fold-proof guarantee lives in one place.
    """
    import SimpleITK as sitk

    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        return None
    for smoothing in smoothing_escalation:
        dx, dy = _forward_field(src, dst, shape, smoothing)
        if _min_jacobian(dx, dy) > jac_eps:
            field = np.stack([dx, dy], axis=-1).astype(np.float64)
            disp = sitk.Cast(sitk.GetImageFromArray(field, isVector=True),
                             sitk.sitkVectorFloat64)
            return sitk.DisplacementFieldTransform(disp)
    return None


def boundary_snap_transform(
    extent: np.ndarray,
    tissue: np.ndarray,
    *,
    drop_frac: float = _DROP_FRAC,
    smoothing_escalation: tuple[float, ...] = _SMOOTHING_ESCALATION,
):
    """Build a fold-free ``sitk.DisplacementFieldTransform`` snapping atlas->tissue.

    Returns ``None`` when there isn't enough signal (degenerate masks, too few
    correspondences) or when every smoothing level still folds - i.e. the caller
    keeps the un-snapped registration rather than a creased one.
    """
    if extent.shape != tissue.shape:
        return None
    if int(extent.sum()) < 16 or int(tissue.sum()) < 16:
        return None
    src, dst = boundary_correspondences(extent, tissue, drop_frac=drop_frac)
    if len(src) < 8:
        return None
    return fit_foldproof_tps(
        src, dst, extent.shape, smoothing_escalation=smoothing_escalation
    )


def compose_snap(transform, snap):
    """``CompositeTransform([snap, transform])`` - registration first, then snap.

    Flattened so it never nests a composite inside a composite: the elastix
    engine already returns a ``CompositeTransform`` and HDF5 only allows a
    composite as the first transform in the file, so a nested composite fails to
    persist. ``FlattenTransform`` splices the registration's sub-transforms in.
    """
    import SimpleITK as sitk

    composite = sitk.CompositeTransform([snap, transform])
    composite.FlattenTransform()
    return composite
