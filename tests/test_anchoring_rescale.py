"""Restating an anchoring on a different atlas voxel grid.

Every component of an anchoring is a count of voxels, so one measured on the Allen
25 µm grid points somewhere else on the Chon/Kim isotropic 20 µm grid even though the
two atlases cover the identical physical volume (13.2 x 8.0 x 11.4 mm). Without this
the region-atlas picker would silently label each section from the wrong plane.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.atlas.planes import Anchoring, rescale_atlas_anchoring

ALLEN_SHAPE = (528, 320, 456)
KIM_ISO_SHAPE = (660, 400, 570)


def test_the_allen_to_isotropic_step_is_a_clean_1_25():
    """Same extent, 25 µm against 20 µm, so every axis scales by exactly 1.25."""
    anchoring = Anchoring.from_iterable([100, 50, 25, 0, 0, 456, 0, 320, 0])

    out = rescale_atlas_anchoring(
        anchoring, source_shape=ALLEN_SHAPE, target_shape=KIM_ISO_SHAPE
    )

    assert out.as_tuple() == pytest.approx(
        tuple(v * 1.25 for v in anchoring.as_tuple())
    )


def test_every_triple_is_scaled_not_just_the_origin():
    """o, u and v are all voxel counts; scaling only the origin skews the plane."""
    anchoring = Anchoring.from_iterable([10, 20, 30, 40, 50, 60, 70, 80, 90])

    out = rescale_atlas_anchoring(
        anchoring, source_shape=(100, 100, 100), target_shape=(200, 400, 800)
    )

    assert out.as_tuple() == pytest.approx(
        (20, 80, 240, 80, 200, 480, 140, 320, 720)
    )


def test_the_same_grid_changes_nothing():
    anchoring = Anchoring.from_iterable([1, 2, 3, 4, 5, 6, 7, 8, 9])

    out = rescale_atlas_anchoring(
        anchoring, source_shape=ALLEN_SHAPE, target_shape=ALLEN_SHAPE
    )

    assert out.as_tuple() == pytest.approx(anchoring.as_tuple())


def test_axes_scale_independently():
    """A per-axis factor, not one number: an anisotropic pair must still work."""
    anchoring = Anchoring.from_iterable([10, 10, 10, 10, 10, 10, 10, 10, 10])

    out = rescale_atlas_anchoring(
        anchoring, source_shape=(100, 100, 100), target_shape=(100, 200, 300)
    )

    assert out.as_tuple() == pytest.approx(
        (10, 20, 30, 10, 20, 30, 10, 20, 30)
    )


def test_it_matches_building_the_anchoring_on_the_target_grid():
    """The property that makes this correct, on synthetic atlases.

    Rescaling a plane from grid A must give the same plane as constructing it on
    grid B directly - otherwise the region atlas draws a different slice.
    """
    from histo_to_ccf.atlas.planes import coronal_anchoring

    class _Atlas:
        def __init__(self, shape, res):
            self.reference = np.zeros(shape, dtype=np.float32)
            self.annotation = np.zeros(shape, dtype=np.int32)
            self.resolution = (res, res, res)
            self.shape = shape

    coarse, fine = _Atlas(ALLEN_SHAPE, 25.0), _Atlas(KIM_ISO_SHAPE, 20.0)

    rescaled = rescale_atlas_anchoring(
        coronal_anchoring(coarse, 7000.0),
        source_shape=ALLEN_SHAPE,
        target_shape=KIM_ISO_SHAPE,
    )

    assert rescaled.as_tuple() == pytest.approx(
        coronal_anchoring(fine, 7000.0).as_tuple()
    )
