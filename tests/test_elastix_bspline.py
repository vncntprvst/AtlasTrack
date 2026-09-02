"""Tests for the elastix (regularized + masked) registration engine.

Skipped entirely when the optional ``itk-elastix`` dependency is absent, so the
base test suite still runs. The integration path (``register_section_image``
with ``engine="elastix"``) and the SimpleITK→displacement-field round-trip are
the important things to verify, since the whole transform-consumption layer
assumes a ``sitk.Transform``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk
from skimage.draw import disk, ellipse

from atlastrack.atlas.planes import Anchoring, resample_atlas_at_plane
from atlastrack.registration import elastix_bspline as eb
from atlastrack.registration.bspline import warp_moving_to_fixed
from atlastrack.registration.pipeline import _resolve_engine, register_section_image

pytestmark = pytest.mark.skipif(
    not eb.ELASTIX_AVAILABLE, reason="itk-elastix not installed"
)


# ---------------------------------------------------------------------------
# Helpers (shared shape with test_registration_synthetic)
# ---------------------------------------------------------------------------

def _brain_slice(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.float32)
    rr, cc = ellipse(h // 2, w // 2, h // 2 - 3, w // 2 - 6, shape=img.shape)
    img[rr, cc] = 1.0
    for cx in (w // 2 - 14, w // 2 + 14):
        rr, cc = disk((h // 2 + 3, cx), 4, shape=img.shape)
        img[rr, cc] = 0.2
    return img


def _structured_atlas(ap: int = 60, dv: int = 64, ml: int = 96) -> SimpleNamespace:
    slice_2d = _brain_slice(dv, ml)
    reference = np.broadcast_to(slice_2d[np.newaxis], (ap, dv, ml)).copy().astype(np.float32)
    annotation = np.zeros((ap, dv, ml), dtype=np.int32)
    annotation[:, dv // 2:, :] = 1
    return SimpleNamespace(reference=reference, annotation=annotation, resolution=(25.0,) * 3)


_ANCHORING = Anchoring(ox=20.0, oy=0.0, oz=0.0, ux=0.0, uy=0.0, uz=96.0, vx=0.0, vy=64.0, vz=0.0)


def _warp(img: np.ndarray, *, tx: float = 3.0, ty: float = -2.0, angle_deg: float = 3.0) -> np.ndarray:
    src = sitk.GetImageFromArray(img.astype(np.float32))
    t = sitk.Euler2DTransform()
    t.SetCenter((img.shape[1] / 2.0, img.shape[0] / 2.0))
    t.SetAngle(np.deg2rad(angle_deg))
    t.SetTranslation((tx, ty))
    return sitk.GetArrayFromImage(sitk.Resample(src, src, t, sitk.sitkLinear, 0.0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_refine_with_elastix_returns_sitk_transform_and_reduces_mse() -> None:
    fixed = _brain_slice(96, 128)
    moving = _warp(fixed)
    pre_mse = float(np.mean((fixed - moving) ** 2))

    result = eb.refine_with_elastix(fixed, moving, grid_size=(8, 8), max_iterations=80)

    # Contract parity with refine_with_bspline.
    assert isinstance(result.transform, sitk.Transform)
    assert np.isfinite(result.residual_rms)

    aligned = warp_moving_to_fixed(moving, fixed.shape, result.transform)
    post_mse = float(np.mean((fixed - aligned) ** 2))
    assert post_mse < pre_mse, f"elastix did not reduce MSE: pre={pre_mse} post={post_mse}"


def test_elastix_transform_round_trips_through_h5(tmp_path: Path) -> None:
    fixed = _brain_slice(96, 128)
    moving = _warp(fixed)
    result = eb.refine_with_elastix(fixed, moving, max_iterations=40)

    path = tmp_path / "t.h5"
    sitk.WriteTransform(result.transform, str(path))
    back = sitk.ReadTransform(str(path))

    p = (64.0, 48.0)
    assert np.allclose(back.TransformPoint(p), result.transform.TransformPoint(p), atol=1e-3)
    # The displacement-field inverse path used by transforms.py must work.
    h, w = fixed.shape
    ref = sitk.Image(w, h, sitk.sitkVectorFloat64)
    field = sitk.TransformToDisplacementField(
        back, sitk.sitkVectorFloat64, ref.GetSize(), ref.GetOrigin(),
        ref.GetSpacing(), ref.GetDirection(),
    )
    inv = sitk.InvertDisplacementField(field, maximumNumberOfIterations=20)
    assert np.all(np.isfinite(sitk.DisplacementFieldTransform(inv).TransformPoint(p)))


def test_bending_weight_increases_smoothness() -> None:
    """A higher bending penalty yields a smoother displacement field."""
    rng = np.random.default_rng(0)
    fixed = _brain_slice(96, 128)
    moving = _warp(fixed, tx=5.0, ty=-4.0, angle_deg=5.0)
    moving = moving + rng.normal(0, 0.03, moving.shape).astype(np.float32)

    def roughness(weight: float) -> float:
        r = eb.refine_with_elastix(fixed, moving, bending_weight=weight, max_iterations=80)
        field = sitk.TransformToDisplacementField(
            r.transform, sitk.sitkVectorFloat64, (128, 96), (0.0, 0.0), (1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        )
        d = sitk.GetArrayFromImage(field)  # (H, W, 2)
        return float(np.mean(np.abs(np.gradient(d[..., 0])[0])) + np.mean(np.abs(np.gradient(d[..., 1])[0])))

    assert roughness(200.0) <= roughness(0.1) + 1e-6


def test_masks_are_sensible() -> None:
    fixed = _brain_slice(96, 128)
    atlas_mask = eb.atlas_foreground_mask(fixed)
    histo_mask = eb.histo_foreground_mask(_warp(fixed))
    # Dilated masks cover most of the brain (incl. outline rim) but not all.
    for m in (atlas_mask, histo_mask):
        frac = m.mean()
        assert 0.1 < frac < 0.98
    # Flat image → all-ones fallback (so registration still runs unmasked).
    assert eb.histo_foreground_mask(np.zeros((20, 20), dtype=np.float32)).all()


def test_register_section_image_elastix_engine(tmp_path: Path) -> None:
    atlas = _structured_atlas()
    reference_slice, _ = resample_atlas_at_plane(atlas, _ANCHORING, out_shape=(64, 96))
    section_image = _warp(reference_slice)
    pre_mse = float(np.mean((reference_slice - section_image) ** 2))

    reg, transform = register_section_image(
        section_image, atlas, anchoring=_ANCHORING, engine="elastix",
        bspline_grid=(8, 8), max_iterations=80, boundary_snap=False,
    )
    assert isinstance(transform, sitk.Transform)
    assert np.isfinite(reg.residual)

    aligned = warp_moving_to_fixed(section_image, reference_slice.shape, transform)
    post_mse = float(np.mean((reference_slice - aligned) ** 2))
    assert post_mse < pre_mse


def test_prealign_handles_scale_mismatch_and_round_trips(tmp_path: Path) -> None:
    """With a big atlas/tissue scale gap, pre-align still yields a valid composite."""
    fixed = _brain_slice(120, 160)
    moving = _warp(fixed, tx=10.0, ty=-6.0, angle_deg=2.0)
    res = eb.refine_with_elastix(fixed, moving, prealign=True, max_iterations=80)

    assert isinstance(res.transform, sitk.Transform)
    aligned = warp_moving_to_fixed(moving, fixed.shape, res.transform)
    assert np.isfinite(aligned).all()
    # The composite (residual ∘ similarity) must persist + reload.
    path = tmp_path / "pre.h5"
    sitk.WriteTransform(res.transform, str(path))
    back = sitk.ReadTransform(str(path))
    p = (80.0, 60.0)
    assert np.allclose(back.TransformPoint(p), res.transform.TransformPoint(p), atol=1e-3)


def test_resolve_engine_auto_prefers_elastix_when_available() -> None:
    # This test only runs when elastix IS available (module-level skip).
    assert _resolve_engine("auto") == "elastix"
    assert _resolve_engine("sitk") == "sitk"
    assert _resolve_engine("elastix") == "elastix"
