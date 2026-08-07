"""2D B-spline registration of a histology section to an atlas slice.

Built on SimpleITK. Strategy:

    1. Coarse affine pre-alignment (centered + identity scale).
    2. Multi-resolution B-spline refinement minimizing Mattes Mutual
       Information.
    3. Return the composite transform (Affine ∘ BSpline) plus a residual that
       approximates the per-pixel displacement RMS in input-image pixels.

The result is a ``RegisterResult`` carrying:
    - ``transform``     : the SimpleITK transform mapping FIXED → MOVING
                          (input-image coords, in the SITK convention)
    - ``residual_rms``  : final metric value (lower = better)
    - ``n_iterations``  : iterations the optimizer ran for

To persist: ``sitk.WriteTransform(result.transform, "out.tfm")``.
To reload: ``sitk.ReadTransform("out.tfm")``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class RegisterResult:
    """Output of :func:`refine_with_bspline`."""

    transform: sitk.Transform
    residual_rms: float
    n_iterations: int
    # Set when the masked attempt failed and the section was rescued by an
    # unmasked retry (see ``register_section_image``). Worth surfacing: the
    # section IS registered, but without the label-excluding mask, so it
    # deserves a look.
    used_mask_fallback: bool = False


def _to_sitk(image: np.ndarray) -> sitk.Image:
    """Convert a 2D numpy array to a float32 SimpleITK image."""
    arr = np.asarray(image, dtype=np.float32)
    # Normalize to [0, 1] for robust MI binning.
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return sitk.GetImageFromArray(arr)


def refine_with_bspline(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    grid_size: tuple[int, int] = (8, 8),
    shrink_factors: tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0),
    learning_rate: float = 1.0,
    max_iterations: int = 100,
    sampling_percentage: float = 0.20,
    histogram_bins: int = 50,
    seed: int = 12345,
) -> RegisterResult:
    """Register ``moving`` to ``fixed`` with affine + B-spline.

    Parameters
    ----------
    fixed
        Target image (e.g. atlas reference slice resampled at the plane).
    moving
        Source image (the histology section converted to grayscale).
    grid_size
        B-spline control-point mesh size. 8×8 is a reasonable default; raise
        for tighter local deformations.
    shrink_factors, smoothing_sigmas
        Multi-resolution pyramid. Length must match.
    learning_rate, max_iterations
        Gradient-descent settings, applied at the finest level.
    sampling_percentage
        Fraction of voxels the metric samples each iteration (random).
    histogram_bins
        Mattes-MI histogram bins.
    seed
        RNG seed for reproducible sampling.
    """
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError("shrink_factors and smoothing_sigmas must be same length")

    fixed_img = _to_sitk(fixed)
    moving_img = _to_sitk(moving)

    # 1. Affine pre-alignment.
    initial = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogram_bins)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(sampling_percentage, seed=seed)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=max_iterations,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetInitialTransform(initial, inPlace=True)
    method.SetShrinkFactorsPerLevel(list(shrink_factors))
    method.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine = method.Execute(fixed_img, moving_img)  # returns `initial` in place

    # 2. B-spline refinement seeded with the affine.
    composite = sitk.CompositeTransform([affine])
    bspline = sitk.BSplineTransformInitializer(fixed_img, list(grid_size))
    composite.AddTransform(bspline)

    method2 = sitk.ImageRegistrationMethod()
    method2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=histogram_bins)
    method2.SetMetricSamplingStrategy(method2.RANDOM)
    method2.SetMetricSamplingPercentage(sampling_percentage, seed=seed + 1)
    method2.SetInterpolator(sitk.sitkLinear)
    method2.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=max_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )
    # Only the B-spline part is optimized; affine stays fixed.
    method2.SetInitialTransformAsBSpline(bspline, inPlace=True)
    method2.SetMovingInitialTransform(affine)
    method2.SetShrinkFactorsPerLevel(list(shrink_factors))
    method2.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    method2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    method2.Execute(fixed_img, moving_img)
    final_metric = float(method2.GetMetricValue())
    n_iters = int(method2.GetOptimizerIteration())

    return RegisterResult(
        transform=composite,
        residual_rms=final_metric,
        n_iterations=n_iters,
    )


def warp_moving_to_fixed(
    moving: np.ndarray,
    fixed_shape: tuple[int, int],
    transform: sitk.Transform,
    *,
    default_pixel_value: float = 0.0,
) -> np.ndarray:
    """Apply ``transform`` to ``moving`` to produce an image of ``fixed_shape``."""
    h, w = fixed_shape
    moving_img = _to_sitk(moving)
    ref = sitk.Image(int(w), int(h), sitk.sitkFloat32)
    out = sitk.Resample(
        moving_img,
        ref,
        transform,
        sitk.sitkLinear,
        default_pixel_value,
    )
    return sitk.GetArrayFromImage(out)
