"""Integration test: synthetically warp a known atlas slice, recover the transform.

Uses a fake in-memory atlas (no network download) to exercise the full
register_project_with_atlas pipeline including bspline refinement.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
from skimage.draw import disk, ellipse

from histo_to_ccf.atlas.planes import Anchoring, resample_atlas_at_plane
from histo_to_ccf.project.schema import (
    AtlasRef,
    PlaneParams,
    Point2D,
    ProbeSpec,
    ProbeType,
    Project,
    Section,
    Shank,
    Slide,
)
from histo_to_ccf.registration.bspline import warp_moving_to_fixed
from histo_to_ccf.registration.pipeline import register_project_with_atlas, register_section_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brain_slice(h: int = 40, w: int = 80) -> np.ndarray:
    """A brain-like 2D float32 pattern (ellipse + two dark holes)."""
    img = np.zeros((h, w), dtype=np.float32)
    rr, cc = ellipse(h // 2, w // 2, h // 2 - 3, w // 2 - 6, shape=img.shape)
    img[rr, cc] = 1.0
    for cx in (w // 2 - 14, w // 2 + 14):
        rr, cc = disk((h // 2 + 3, cx), 4, shape=img.shape)
        img[rr, cc] = 0.2
    return img


def _structured_atlas(ap: int = 60, dv: int = 40, ml: int = 80) -> SimpleNamespace:
    """Atlas whose coronal slices are all identical brain-like patterns."""
    slice_2d = _brain_slice(h=dv, w=ml)
    reference = np.broadcast_to(slice_2d[np.newaxis], (ap, dv, ml)).copy().astype(np.float32)
    annotation = np.zeros((ap, dv, ml), dtype=np.int32)
    annotation[:, dv // 2:, :] = 1  # ventral half = region 1
    return SimpleNamespace(
        reference=reference,
        annotation=annotation,
        resolution=(25.0, 25.0, 25.0),
    )


# Anchoring for AP=500 µm in a 25-µm atlas with dv=40, ml=80.
_ANCHORING = Anchoring(
    ox=20.0, oy=0.0, oz=0.0,
    ux=0.0, uy=0.0, uz=80.0,
    vx=0.0, vy=40.0, vz=0.0,
)


def _affine_warp(img: np.ndarray, *, tx: float = 3.0, ty: float = -2.0, angle_deg: float = 2.0) -> np.ndarray:
    src = sitk.GetImageFromArray(img.astype(np.float32))
    t = sitk.Euler2DTransform()
    t.SetCenter((img.shape[1] / 2.0, img.shape[0] / 2.0))
    t.SetAngle(np.deg2rad(angle_deg))
    t.SetTranslation((tx, ty))
    warped = sitk.Resample(src, src, t, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(warped)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_register_project_with_atlas(tmp_path: Path) -> None:
    """Full pipeline: structured atlas + synthetic warp → RegistrationResult on section."""
    atlas = _structured_atlas()
    reference_slice, _ = resample_atlas_at_plane(atlas, _ANCHORING, out_shape=(40, 80))
    section_image = _affine_warp(reference_slice)

    plane = PlaneParams(
        ap_um=500.0,
        midline_px=40.0,
        dorsal_surface_px=0.0,
        pixel_size_um=25.0,
    )
    section = Section(index=0, slide_idx=0, bbox_px=(0, 0, 80, 40), ap_order=0, plane=plane)
    probe = ProbeSpec(
        label="probe1",
        type=ProbeType(name="neuropixels-1.0", n_shanks=1),
        shanks=[
            Shank(
                index=0,
                tip_px=Point2D(x_px=40.0, y_px=35.0),
                tip_section_idx=0,
                entry_px=Point2D(x_px=40.0, y_px=5.0),
                entry_section_idx=0,
            )
        ],
    )
    project = Project(
        atlas=AtlasRef(),
        slides=[Slide(image_path="fake_slide.png", sections=[section])],
        probes=[probe],
    )

    register_project_with_atlas(
        project,
        atlas,
        section_images={0: section_image},
        transforms_dir=tmp_path / "transforms",
        bspline_grid=(6, 6),
        max_iterations=60,
    )

    reg = project.slides[0].sections[0].registration
    assert reg is not None, "expected RegistrationResult on section"
    assert len(reg.anchoring) == 9
    assert reg.output_size_px == (40, 80)
    assert np.isfinite(reg.residual)
    # .tfm sidecar written next to transforms dir.
    tfm_files = list((tmp_path / "transforms").glob("*.h5"))
    assert len(tfm_files) == 1

    # Shank CCF coords populated and finite.
    shank = project.probes[0].shanks[0]
    assert shank.tip_ccf_um is not None
    assert shank.entry_ccf_um is not None
    assert all(np.isfinite(shank.tip_ccf_um)), f"tip CCF not finite: {shank.tip_ccf_um}"
    assert all(np.isfinite(shank.entry_ccf_um)), f"entry CCF not finite: {shank.entry_ccf_um}"


def test_registration_reduces_mse() -> None:
    """B-spline alignment should reduce image MSE vs. unregistered section."""
    atlas = _structured_atlas()
    reference_slice, _ = resample_atlas_at_plane(atlas, _ANCHORING, out_shape=(40, 80))
    section_image = _affine_warp(reference_slice)
    pre_mse = float(np.mean((reference_slice - section_image) ** 2))

    reg, sitk_transform = register_section_image(
        section_image,
        atlas,
        anchoring=_ANCHORING,
        bspline_grid=(6, 6),
        max_iterations=80,
    )

    aligned = warp_moving_to_fixed(section_image, reference_slice.shape, sitk_transform)
    post_mse = float(np.mean((reference_slice - aligned) ** 2))

    assert post_mse < 0.5 * pre_mse, (
        f"registration did not reduce MSE: pre={pre_mse:.5f} post={post_mse:.5f}"
    )
