"""Regularized 2D registration via elastix (ABBA-style), returning a SimpleITK transform.

Why this exists
---------------
The plain SimpleITK B-spline in :mod:`bspline` minimizes Mattes mutual
information with *no* deformation penalty and *no* tissue mask. On real
fluorescence sections that lets the warp chase bright labels and background,
so atlas boundaries drift well outside the tissue (the "lines off the bottom"
artefact). elastix — the engine ABBA uses — fixes this with two levers:

1. a **bending-energy penalty** (``TransformBendingEnergyPenalty``) that keeps
   the B-spline smooth, and
2. a **tissue mask** on both images so the metric only sees brain, not the
   black border or the magenta/green label blobs.

Integration trick
-----------------
elastix produces its own transform-parameter format, but the rest of this
package consumes a :class:`SimpleITK.Transform` (point mapping, the iterative
inverse in :mod:`transforms`, and ``.h5`` persistence). So we run elastix, pull
the **combined deformation field** (affine + B-spline baked into one field via
transformix), and wrap it as a :class:`SimpleITK.DisplacementFieldTransform`
inside a ``CompositeTransform``. That object *is* a ``sitk.Transform`` — it
composes, inverts (via the existing displacement-inversion fallback) and
round-trips through ``.h5`` exactly like the native B-spline, so nothing
downstream changes.

The module imports ``itk`` lazily; :data:`ELASTIX_AVAILABLE` reports whether the
optional ``itk-elastix`` dependency is installed so callers can fall back to the
SimpleITK engine.
"""
from __future__ import annotations

import numpy as np

from histo_to_ccf.registration.bspline import RegisterResult

try:  # itk-elastix is an optional extra
    import itk

    ELASTIX_AVAILABLE = True
except Exception:  # noqa: BLE001 — any import failure means "not available"
    itk = None  # type: ignore[assignment]
    ELASTIX_AVAILABLE = False


def _normalize(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] float32 for stable MI binning (mirrors bspline._to_sitk)."""
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


# Tissue masks are DILATED by this fraction of the smaller image dimension. A
# tight tissue mask would exclude the brain/background boundary — the strongest
# alignment cue — and actually hurt the fit. Dilating keeps that outline (plus a
# background rim) while still excluding far-field junk: the canvas border, debris
# and neighbouring-section pixels that fall inside a crop's bbox.
_MASK_RIM_FRAC = 0.10


def _rim_px(shape: tuple[int, int]) -> int:
    return max(1, round(_MASK_RIM_FRAC * min(shape)))


def atlas_foreground_mask(reference_norm: np.ndarray, *, thresh: float = 0.02) -> np.ndarray:
    """Dilated tissue mask for an atlas reference slice (background is ~0)."""
    from scipy import ndimage as ndi

    mask = reference_norm > thresh
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_dilation(mask, iterations=_rim_px(reference_norm.shape))
    return mask.astype(np.uint8)


def histo_foreground_mask(moving_norm: np.ndarray) -> np.ndarray:
    """Dilated tissue mask for a histology section (grayscale fallback).

    Used only when the caller doesn't supply an explicit moving mask. The
    RGB-aware, label-excluding mask lives in
    :func:`histo_to_ccf.registration.masks.registration_moving_mask` and is what
    the pipeline passes in; this grayscale version keeps the brain outline (see
    :data:`_MASK_RIM_FRAC`) but can't see labels.
    """
    from scipy import ndimage as ndi

    from histo_to_ccf.registration.masks import section_tissue_mask

    finite = moving_norm[np.isfinite(moving_norm)]
    if finite.size == 0 or float(finite.max()) - float(finite.min()) < 1e-6:
        return np.ones(moving_norm.shape, dtype=np.uint8)
    mask = section_tissue_mask(moving_norm)
    mask = ndi.binary_dilation(mask, iterations=_rim_px(moving_norm.shape))
    return mask.astype(np.uint8)


_METRICS = {
    "mi": "AdvancedMattesMutualInformation",
    "meansquares": "AdvancedMeanSquares",
}


def _parameter_object(
    grid_size: tuple[int, int],
    image_shape: tuple[int, int],
    bending_weight: float,
    max_iterations: int,
    seed: int,
    metric: str = "mi",
    deformable: bool = True,
):
    """Build the affine→B-spline elastix parameter object with a bending penalty.

    When ``deformable`` is False only the affine stage is added (a low-DOF global
    correction that cannot overfit) — useful when the only error is a global
    scale/shift of the atlas plane.
    """
    po = itk.ParameterObject.New()

    # Cap the resolution pyramid so the coarsest level stays ~>=16 px — small
    # section crops (or the tiny atlas slices in tests) can't survive 4 levels.
    h, w = image_shape
    n_res = max(1, min(4, int(np.floor(np.log2(max(min(h, w), 1) / 16.0))) + 1))

    metric_name = _METRICS.get(metric, _METRICS["mi"])

    affine = po.GetDefaultParameterMap("affine", n_res)
    affine["Metric"] = (metric_name,)
    affine["MaximumNumberOfIterations"] = (str(max_iterations),)
    affine["RandomSeed"] = (str(seed),)
    # Tolerate samples that fall outside a (tight) tissue mask during the search;
    # the default 0.25 makes elastix abort with "too many samples map outside".
    affine["RequiredRatioOfValidSamples"] = ("0.05",)
    po.AddParameterMap(affine)

    if deformable:
        # Control-point spacing (in pixels) that yields ~grid_size knots across the
        # image; the bending penalty controls smoothness independently of the grid.
        gy, gx = max(grid_size[0], 1), max(grid_size[1], 1)
        spacing = max(min(h / gy, w / gx), 4.0)

        bspline = po.GetDefaultParameterMap("bspline", n_res, float(spacing))
        bspline["Registration"] = ("MultiMetricMultiResolutionRegistration",)
        bspline["Metric"] = (metric_name, "TransformBendingEnergyPenalty")
        bspline["Metric0Weight"] = ("1.0",)
        bspline["Metric1Weight"] = (str(float(bending_weight)),)
        bspline["MaximumNumberOfIterations"] = (str(max_iterations),)
        bspline["RandomSeed"] = (str(seed),)
        bspline["RequiredRatioOfValidSamples"] = ("0.05",)
        po.AddParameterMap(bspline)
    return po


def _itk_mask(mask: np.ndarray):
    """numpy uint8 → itk unsigned-char image, or None if the mask is empty."""
    if mask is None or int(mask.sum()) == 0:
        return None
    return itk.image_from_array(np.ascontiguousarray(mask.astype(np.uint8)))


def refine_with_elastix(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    grid_size: tuple[int, int] = (8, 8),
    bending_weight: float = 20.0,
    max_iterations: int = 100,
    use_masks: bool = True,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    metric: str = "mi",
    deformable: bool = True,
    seed: int = 12345,
) -> RegisterResult:
    """Register ``moving`` onto ``fixed`` with a masked, bending-penalized B-spline.

    Returns the same :class:`RegisterResult` contract as
    :func:`histo_to_ccf.registration.bspline.refine_with_bspline`: a
    ``sitk.Transform`` mapping FIXED → MOVING (in pixel coordinates), a residual
    (lower = better), and an iteration count. The transform is a
    ``CompositeTransform`` wrapping a displacement field, so it persists and
    inverts like the native B-spline.

    Parameters
    ----------
    grid_size
        Approximate B-spline control-point mesh (knots across the image).
    bending_weight
        Weight of the ``TransformBendingEnergyPenalty``. Higher = smoother /
        stiffer (closer to affine); lower = more local freedom.
    use_masks
        Restrict the metric to a tissue mask on each image (derived
        automatically when ``fixed_mask`` / ``moving_mask`` are not given).
    """
    if not ELASTIX_AVAILABLE:  # pragma: no cover - guarded by callers
        raise RuntimeError(
            "itk-elastix is not installed; install the 'elastix' extra or use the "
            "SimpleITK engine"
        )
    import SimpleITK as sitk

    fixed_n = _normalize(fixed)
    moving_n = _normalize(moving)
    fixed_img = itk.image_from_array(fixed_n)
    moving_img = itk.image_from_array(moving_n)

    elastix_kwargs = {"log_to_console": False}
    if use_masks:
        if fixed_mask is None:
            fixed_mask = atlas_foreground_mask(fixed_n)
        if moving_mask is None:
            moving_mask = histo_foreground_mask(moving_n)
        fm = _itk_mask(fixed_mask)
        mm = _itk_mask(moving_mask)
        if fm is not None:
            elastix_kwargs["fixed_mask"] = fm
        if mm is not None:
            elastix_kwargs["moving_mask"] = mm

    po = _parameter_object(
        grid_size, fixed_n.shape, bending_weight, max_iterations, seed, metric, deformable
    )

    result_image, transform_params = itk.elastix_registration_method(
        fixed_img, moving_img, parameter_object=po, **elastix_kwargs
    )

    # Combined (affine ∘ B-spline) displacement field on the FIXED grid, mapping
    # fixed → moving — the convention every consumer of `bspline` expects.
    # transformix insists on writing `deformationField.nii`; point it at a temp
    # dir (an empty OutputDirectory is treated as ".") so it never litters cwd.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="histo2ccf_elx_") as _td:
        tx = itk.TransformixFilter.New(moving_img)
        tx.SetTransformParameterObject(transform_params)
        tx.SetComputeDeformationField(True)
        tx.SetOutputDirectory(_td)
        tx.SetLogToConsole(False)
        tx.UpdateLargestPossibleRegion()
        darr = np.array(
            itk.array_from_image(tx.GetOutputDeformationField()), dtype=np.float64
        )  # (H, W, 2); copied before the temp dir is removed

    disp = sitk.GetImageFromArray(darr, isVector=True)
    disp = sitk.Cast(disp, sitk.sitkVectorFloat64)
    dft = sitk.DisplacementFieldTransform(disp)
    composite = sitk.CompositeTransform([dft])

    residual = _residual_rms(fixed_n, np.asarray(itk.array_from_image(result_image)),
                             fixed_mask if use_masks else None)
    return RegisterResult(
        transform=composite,
        residual_rms=residual,
        n_iterations=int(max_iterations),
    )


def _residual_rms(
    fixed_n: np.ndarray, result: np.ndarray, mask: np.ndarray | None
) -> float:
    """Normalized-intensity RMS difference over the fixed foreground.

    Unlike the raw MI metric the SimpleITK path reports, this is interpretable
    (0 = identical) and comparable across sections.
    """
    res_n = _normalize(result)
    diff = (fixed_n - res_n) ** 2
    if mask is not None and int(np.asarray(mask).sum()) > 0:
        sel = np.asarray(mask).astype(bool)
        diff = diff[sel]
    return float(np.sqrt(np.mean(diff))) if diff.size else float("nan")
