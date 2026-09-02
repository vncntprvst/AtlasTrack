"""Tissue / label masks for a histology section crop.

Used in two places:

- **Registration metric mask** (:func:`registration_moving_mask`): the elastix
  metric should see brain, not the black border or the bright fluorescent
  *labels* (green / magenta), which have no atlas counterpart and otherwise pull
  the fit. The mask keeps a dilated tissue body (so the brain/background outline
  - the strongest cue - is preserved) and subtracts the saturated label blobs.

The overlay's far-off boundary "stripes" are clipped to the *warped atlas
extent* in :func:`atlastrack.registration.transforms.warp_annotation_to_section`,
not to a tissue mask - clipping to tissue would wrongly delete region outlines
over damaged/dim tissue.

All functions are headless (numpy / scipy / scikit-image only).
"""
from __future__ import annotations

import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    """Brightest-channel projection (captures dim blue tissue AND labels)."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=-1)
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo) if hi > lo else arr * 0.0


def section_tissue_mask(image: np.ndarray, *, close_iter: int = 12) -> np.ndarray:
    """Clean boolean tissue mask: Otsu split, fill, close, keep largest body.

    The section sits on a near-black background, so a low Otsu fraction captures
    even dim cortex; heavy closing + largest-component selection yields one solid
    silhouette and drops debris / neighbouring-section fragments.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    g = _gray(image)
    if float(g.max()) - float(g.min()) < 1e-6:
        return np.ones(g.shape, dtype=bool)
    try:
        thr = float(threshold_otsu(g)) * 0.5
    except Exception:  # noqa: BLE001 - degenerate histogram
        return np.ones(g.shape, dtype=bool)

    m = ndi.binary_fill_holes(g > thr)
    m = ndi.binary_closing(m, iterations=close_iter)
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_opening(m, iterations=2)
    lbl, n = ndi.label(m)
    if n == 0:
        return np.ones(g.shape, dtype=bool)
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def section_label_mask(
    image: np.ndarray, *, red_thresh: int = 70, green_thresh: int = 70, dilate: int = 3
) -> np.ndarray:
    """Boolean mask of bright fluorescent labels (green / magenta / red).

    DAPI tissue is blue (low R, low G), so a significant red or green signal
    marks a label. Returns an all-False mask for grayscale input (no channels to
    judge labels from).
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return np.zeros(arr.shape[:2], dtype=bool)
    from scipy import ndimage as ndi

    r, g = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16)
    labels = (r > red_thresh) | (g > green_thresh)
    if dilate > 0:
        labels = ndi.binary_dilation(labels, iterations=dilate)
    return labels


def registration_moving_mask(
    image: np.ndarray, *, rim_frac: float = 0.08, exclude_labels: bool = True
) -> np.ndarray:
    """uint8 metric mask for elastix: dilated tissue body minus label blobs.

    The dilation keeps the brain/background outline (a tight mask would discard
    it and hurt the fit); the subtraction drops the fluorescent labels.
    """
    from scipy import ndimage as ndi

    tissue = section_tissue_mask(image)
    rim = max(1, round(rim_frac * min(tissue.shape)))
    mask = ndi.binary_dilation(tissue, iterations=rim)
    if exclude_labels:
        kept = mask & ~section_label_mask(image)
        # Only subtract "labels" when they are the sparse bright blobs the mask is
        # meant for. If excluding them removes most of the tissue, the bright R/G
        # signal IS the stain itself - e.g. a DAPI section rendered cyan (high
        # green) or magenta - not a fluorescent label. Excising it would gut the
        # silhouette (the strongest alignment cue) and starve the elastix metric
        # of valid samples ("too many samples map outside moving image buffer"),
        # which aborts the whole registration. Keep the tissue in that case.
        if kept.sum() >= 0.5 * mask.sum():
            mask = kept
    if mask.sum() == 0:  # never hand elastix an empty mask
        return np.ones(tissue.shape, dtype=np.uint8)
    return mask.astype(np.uint8)


def moment_similarity(
    fixed_mask: np.ndarray, moving_mask: np.ndarray, *, isotropic: bool = False
) -> np.ndarray:
    """Closed-form pre-alignment of two silhouettes (translation + scale, no rotation).

    Matches the **centroid** (translation) and the **per-axis spread** (scale) of
    ``fixed_mask`` to ``moving_mask`` - a 4-DOF similarity (or isotropic 3-DOF)
    that *cannot shear or fold*. Returns a 3x3 homogeneous affine in **(x, y)**
    (col, row) order mapping a fixed-mask pixel to its moving-mask counterpart;
    identity if either silhouette is degenerate.

    The rotation DOF is intentionally dropped: the atlas plane is already oriented
    by the anchoring, so residual rotation is small and the principal-axis angle
    has a sign/180-degree ambiguity that does more harm than good.
    """
    fy, fx = np.nonzero(fixed_mask)
    my, mx = np.nonzero(moving_mask)
    if fx.size < 8 or mx.size < 8:
        return np.eye(3)
    fcx, fcy = float(fx.mean()), float(fy.mean())
    mcx, mcy = float(mx.mean()), float(my.mean())
    fsx, fsy = float(fx.std()), float(fy.std())
    msx, msy = float(mx.std()), float(my.std())
    if min(fsx, fsy) < 1e-3:
        return np.eye(3)
    sx, sy = msx / fsx, msy / fsy
    if isotropic:
        s = float(np.sqrt(max(sx * sy, 1e-9)))
        sx = sy = s
    return np.array(
        [[sx, 0.0, mcx - sx * fcx],
         [0.0, sy, mcy - sy * fcy],
         [0.0, 0.0, 1.0]]
    )
