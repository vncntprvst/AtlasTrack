"""Internal-feature snap: gated ventricle opening + capped midline, fold-proof.

All synthetic (no atlas): section-frame masks stand in for the warped atlas
ventricle / midline the pipeline computes.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.registration.internal_feature_snap import (
    detect_cavity,
    internal_feature_snap_transform,
    midline_correspondences,
    ventricle_correspondences,
)

sitk = pytest.importorskip("SimpleITK")


def _disk(h, w, cy, cx, r) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _ring(h, w, cy, cx, r_out, r_in) -> np.ndarray:
    return _disk(h, w, cy, cx, r_out) & ~_disk(h, w, cy, cx, r_in)


def _dice(a, b) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def _scene():
    """A section blob with a dark CSF cavity, and a smaller/shifted atlas ventricle."""
    h, w = 200, 240
    tissue = _disk(h, w, 100, 120, 78)
    cavity = _disk(h, w, 100, 130, 18)          # true CSF cavity in the tissue
    atlas_vent = _disk(h, w, 100, 118, 12)       # atlas V4 landed small + left
    lum = np.zeros((h, w), dtype=np.float32)
    lum[tissue] = 0.5
    lum[cavity] = 0.0                            # cavity reads as void
    return h, w, tissue, cavity, atlas_vent, lum


def _warp_mask_forward(mask, transform):
    h, w = mask.shape
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, (w, h), (0.0, 0.0), (1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0))
    disp = sitk.GetArrayFromImage(field)
    ys, xs = np.nonzero(mask)
    sx = np.round(xs + disp[ys, xs, 0]).astype(int)
    sy = np.round(ys + disp[ys, xs, 1]).astype(int)
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out = np.zeros_like(mask)
    out[sy[ok], sx[ok]] = True
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(ndi.binary_closing(out, iterations=2))


def _min_jacobian(transform, shape):
    h, w = shape
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, (w, h), (0.0, 0.0), (1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0))
    d = sitk.GetArrayFromImage(field)
    dx, dy = d[..., 0], d[..., 1]
    jac = ((1 + np.gradient(dx, axis=1)) * (1 + np.gradient(dy, axis=0))
           - np.gradient(dx, axis=0) * np.gradient(dy, axis=1))
    return float(jac.min())


# ---------------------------------------------------------------- detect_cavity

def test_detect_cavity_accepts_clean_blob() -> None:
    h, w, tissue, cavity, atlas_vent, lum = _scene()
    got = detect_cavity(lum, tissue, atlas_vent)
    assert got is not None
    assert _dice(got, cavity) > 0.8  # it found the planted cavity


def test_detect_cavity_rejects_ragged_low_solidity() -> None:
    h, w, tissue, _cav, atlas_vent, _lum = _scene()
    ring = _ring(h, w, 100, 130, 22, 17)  # thin ring: solidity well below 0.55
    lum = np.zeros((h, w), dtype=np.float32)
    lum[tissue] = 0.5
    lum[ring] = 0.0
    assert detect_cavity(lum, tissue, atlas_vent) is None


def test_detect_cavity_rejects_oversized_and_tiny() -> None:
    h, w, tissue, _c, atlas_vent, _l = _scene()
    for r in (45, 8):  # ratio > 6, and area < min_area
        lum = np.zeros((h, w), dtype=np.float32)
        lum[tissue] = 0.5
        lum[_disk(h, w, 100, 130, r)] = 0.0
        assert detect_cavity(lum, tissue, atlas_vent) is None


def test_detect_cavity_none_when_atlas_ventricle_absent() -> None:
    h, w, tissue, _c, _a, lum = _scene()
    tiny = _disk(h, w, 100, 118, 3)  # < 80 px
    assert detect_cavity(lum, tissue, tiny) is None


# ------------------------------------------------------ ventricle_correspondences

def test_ventricle_correspondences_open_toward_cavity() -> None:
    h, w, tissue, cavity, atlas_vent, _lum = _scene()
    src, dst = ventricle_correspondences(atlas_vent, cavity)
    assert len(src) == len(dst) >= 8
    cav_c = np.array([np.nonzero(cavity)[1].mean(), np.nonzero(cavity)[0].mean()])
    # The mapped centroid (last pair) is nearer the cavity centre than the source.
    assert np.linalg.norm(dst[-1] - cav_c) < np.linalg.norm(src[-1] - cav_c)


# ------------------------------------------------------- midline_correspondences

def _vstrip(h, w, x, half=1, y0=40, y1=160):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x - half:x + half + 1] = True
    return m


def test_midline_accepts_small_shift() -> None:
    h, w, tissue, *_ = _scene()
    atlas_mid = _vstrip(h, w, 112)   # ~8 px left of the section axis (x=120)
    res = midline_correspondences(atlas_mid, tissue)
    assert res is not None
    src, dst = res
    assert float(np.mean(dst[:, 0] - src[:, 0])) > 0      # net nudge toward the axis
    assert abs(np.median(dst[:, 0]) - 120) < abs(np.median(src[:, 0]) - 120)


def test_midline_rejects_biased_large_shift() -> None:
    h, w, tissue, *_ = _scene()
    atlas_mid = _vstrip(h, w, 95)    # ~25 px off -> median shift over the cap
    assert midline_correspondences(atlas_mid, tissue) is None


# ---------------------------------------------------- internal_feature_snap_transform

def test_snap_foldfree_and_improves_ventricle() -> None:
    h, w, tissue, cavity, atlas_vent, lum = _scene()
    section = np.zeros((h, w), dtype=np.float32)
    section[tissue] = 0.5
    section[cavity] = 0.0
    atlas_mid = _vstrip(h, w, 112)
    snap = internal_feature_snap_transform(
        atlas_vent, section, tissue, brain=tissue, atlas_midline=atlas_mid
    )
    assert snap is not None
    assert _min_jacobian(snap, (h, w)) > 0.0
    warped = _warp_mask_forward(atlas_vent, snap)
    assert _dice(warped, cavity) > _dice(atlas_vent, cavity) + 0.02


def test_snap_none_when_no_features() -> None:
    h, w, tissue, _c, _a, _l = _scene()
    section = np.zeros((h, w), dtype=np.float32)
    section[tissue] = 0.5  # no dark cavity at all
    tiny_vent = _disk(h, w, 100, 118, 3)
    assert internal_feature_snap_transform(
        tiny_vent, section, tissue, brain=tissue, atlas_midline=None
    ) is None


def test_snap_composes_and_persists(tmp_path) -> None:
    from atlastrack.registration.boundary_snap import compose_snap

    h, w, tissue, cavity, atlas_vent, lum = _scene()
    section = np.zeros((h, w), dtype=np.float32)
    section[tissue] = 0.5
    section[cavity] = 0.0
    snap = internal_feature_snap_transform(atlas_vent, section, tissue, brain=tissue)
    assert snap is not None
    inner = sitk.CompositeTransform([sitk.AffineTransform(2)])
    composed = compose_snap(inner, snap)
    path = tmp_path / "isnap.h5"
    sitk.WriteTransform(composed, str(path))
    back = sitk.ReadTransform(str(path))
    for p in [(60.0, 60.0), (130.0, 100.0), (150.0, 120.0)]:
        assert np.allclose(composed.TransformPoint(p), back.TransformPoint(p), atol=1e-4)
