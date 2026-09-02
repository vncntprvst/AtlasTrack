"""SimpleITK B-spline registration recovers a known displacement."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from skimage.draw import disk, ellipse

from atlastrack.registration.bspline import refine_with_bspline, warp_moving_to_fixed


def _make_fixed(h: int = 128, w: int = 192) -> np.ndarray:
    """A synthetic 'atlas slice' - an ellipse with two darker holes."""
    img = np.zeros((h, w), dtype=np.float32)
    rr, cc = ellipse(h // 2, w // 2, h // 2 - 6, w // 2 - 12, shape=img.shape)
    img[rr, cc] = 1.0
    for cx in (w // 2 - 25, w // 2 + 25):
        rr, cc = disk((h // 2 + 6, cx), 6, shape=img.shape)
        img[rr, cc] = 0.2
    return img


def _affine_warp(img: np.ndarray, *, tx: float, ty: float, angle_deg: float) -> np.ndarray:
    """Apply a known rigid+translation warp to ``img`` via SimpleITK."""
    src = sitk.GetImageFromArray(img.astype(np.float32))
    transform = sitk.Euler2DTransform()
    cy = img.shape[0] / 2
    cx = img.shape[1] / 2
    transform.SetCenter((cx, cy))
    transform.SetAngle(np.deg2rad(angle_deg))
    transform.SetTranslation((tx, ty))
    warped = sitk.Resample(src, src, transform, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(warped)


def test_bspline_recovers_small_warp() -> None:
    """The registered moving image should match fixed within a small MSE."""
    fixed = _make_fixed()
    moving = _affine_warp(fixed, tx=4.0, ty=-3.0, angle_deg=3.0)

    # Pre-registration MSE - the baseline we must beat by a wide margin.
    pre_mse = float(np.mean((fixed - moving) ** 2))

    result = refine_with_bspline(
        fixed,
        moving,
        grid_size=(6, 6),
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        max_iterations=80,
        sampling_percentage=0.5,
    )
    aligned = warp_moving_to_fixed(moving, fixed.shape, result.transform)
    post_mse = float(np.mean((fixed - aligned) ** 2))
    # Expect a meaningful improvement on the synthetic example. (Exact factor
    # varies per SITK version; 0.5× is conservative.)
    assert post_mse < 0.5 * pre_mse, (
        f"B-spline did not reduce MSE: pre={pre_mse:.4f} post={post_mse:.4f}"
    )
    # Sanity on the result fields.
    assert result.n_iterations >= 0
    assert np.isfinite(result.residual_rms)
