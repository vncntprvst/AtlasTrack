"""Boundary snap: improves silhouette Dice, never folds, drops mismatches, persists."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.registration.boundary_snap import (
    boundary_correspondences,
    boundary_snap_transform,
    compose_snap,
)

sitk = pytest.importorskip("SimpleITK")


def _disk(h, w, cy, cx, ry, rx) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _dice(a, b) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def _warp_mask_forward(mask, transform):
    """Forward-splat a boolean mask through a 2D sitk transform."""
    h, w = mask.shape
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, (w, h), (0.0, 0.0), (1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0))
    disp = sitk.GetArrayFromImage(field)  # (h, w, 2)
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
    dxdx = np.gradient(dx, axis=1)
    dxdy = np.gradient(dx, axis=0)
    dydx = np.gradient(dy, axis=1)
    dydy = np.gradient(dy, axis=0)
    return float(((1 + dxdx) * (1 + dydy) - dxdy * dydx).min())


def test_snap_improves_and_does_not_fold() -> None:
    # Atlas landed slightly too small and shifted; tissue is the true silhouette.
    h, w = 200, 240
    tissue = _disk(h, w, 100, 120, 70, 90)
    extent = _disk(h, w, 92, 112, 58, 78)
    snap = boundary_snap_transform(extent, tissue)
    assert snap is not None
    warped = _warp_mask_forward(extent, snap)
    assert _dice(warped, tissue) > _dice(extent, tissue) + 0.02
    # Fold-proof: the forward field's Jacobian stays positive.
    assert _min_jacobian(snap, (h, w)) > 0.0


def test_snap_none_when_degenerate() -> None:
    h, w = 120, 120
    empty = np.zeros((h, w), dtype=bool)
    disk = _disk(h, w, 60, 60, 40, 40)
    assert boundary_snap_transform(empty, disk) is None
    assert boundary_snap_transform(disk, empty) is None


def test_correspondences_drop_large_mismatch() -> None:
    # Tissue far from the atlas extent: every boundary push is huge -> dropped,
    # leaving only the pinned interior anchors (so the atlas is left alone).
    h, w = 200, 200
    extent = _disk(h, w, 60, 60, 30, 30)
    tissue = _disk(h, w, 150, 150, 25, 25)  # disjoint
    src, dst = boundary_correspondences(extent, tissue)
    # No boundary correspondence should propose a large move.
    moved = np.hypot(*(dst - src).T)
    assert float(moved.max(initial=0.0)) < 0.06 * np.hypot(h, w) + 1e-6


def test_compose_snap_flattens_and_persists(tmp_path) -> None:
    h, w = 160, 160
    tissue = _disk(h, w, 80, 80, 55, 60)
    extent = _disk(h, w, 76, 76, 48, 52)
    snap = boundary_snap_transform(extent, tissue)
    assert snap is not None
    # Mimic the elastix engine output: a CompositeTransform (so composing naively
    # would nest a composite, which HDF5 rejects).
    inner = sitk.CompositeTransform([sitk.AffineTransform(2)])
    composed = compose_snap(inner, snap)
    # Flattened: no nested composite -> writes and reads back exactly.
    path = tmp_path / "snap.h5"
    sitk.WriteTransform(composed, str(path))
    back = sitk.ReadTransform(str(path))
    for p in [(40.0, 40.0), (90.0, 70.0), (120.0, 100.0)]:
        a = composed.TransformPoint(p)
        b = back.TransformPoint(p)
        assert np.allclose(a, b, atol=1e-4)
