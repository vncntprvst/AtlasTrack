"""Unit tests for the M1 :class:`SectionTransform`."""
from __future__ import annotations

import numpy as np

from histo_to_ccf.io.ccf_coords import MIDLINE_ML_UM
from histo_to_ccf.project.schema import PlaneParams
from histo_to_ccf.registration.transforms import SectionTransform


def _plane(**kw: float) -> PlaneParams:
    base = dict(
        ap_um=5400.0,
        midline_px=500.0,
        dorsal_surface_px=100.0,
        pixel_size_um=2.0,
    )
    base.update(kw)
    return PlaneParams(**base)  # type: ignore[arg-type]


def test_midline_maps_to_ccf_midline() -> None:
    tx = SectionTransform(plane=_plane())
    ap, ml, dv = tx.apply(500.0, 100.0)  # at (midline_px, dorsal_surface_px)
    assert ap == 5400.0
    assert ml == MIDLINE_ML_UM
    assert dv == 0.0


def test_right_of_midline_is_lateral_right() -> None:
    tx = SectionTransform(plane=_plane())
    # 100 px right of midline at pixel_size_um=2.0 → 200 µm lateral right
    _ap, ml, _dv = tx.apply(600.0, 100.0)
    assert ml == MIDLINE_ML_UM + 200.0


def test_flip_lr_inverts_ml_sign() -> None:
    tx = SectionTransform(plane=_plane(image_right_is_anatomical_right=False))
    _ap, ml, _dv = tx.apply(600.0, 100.0)
    assert ml == MIDLINE_ML_UM - 200.0


def test_ventral_is_positive_dv() -> None:
    tx = SectionTransform(plane=_plane())
    _ap, _ml, dv = tx.apply(500.0, 350.0)  # 250 px below dorsal surface
    assert dv == 500.0  # 250 px * 2 µm/px


def test_apply_many_vectorized() -> None:
    tx = SectionTransform(plane=_plane())
    pts = np.array([[500.0, 100.0], [600.0, 350.0]])
    out = tx.apply_many(pts)
    np.testing.assert_allclose(out[0], [5400.0, MIDLINE_ML_UM, 0.0])
    np.testing.assert_allclose(out[1], [5400.0, MIDLINE_ML_UM + 200.0, 500.0])
