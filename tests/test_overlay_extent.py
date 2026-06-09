"""The overlay clip uses a FORWARD-warped atlas extent (not an inverse resample),
so it can't paint coverage where the atlas never reached. These tests pin that
behaviour down on simple transforms."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from histo_to_ccf.registration.transforms import _warped_atlas_extent


def _disk(h=80, w=80, cy=40, cx=40, r=20) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r * r


def test_identity_extent_matches_foreground() -> None:
    fg = _disk()
    ident = sitk.Transform(2, sitk.sitkIdentity)
    ext = _warped_atlas_extent(ident, fg, fg.shape, fg.shape)
    # Closed/filled identity warp covers the disk and nothing far outside it.
    assert ext[40, 40]
    assert not ext[2, 2]
    assert abs(ext.sum() - fg.sum()) < 0.1 * fg.sum()


def test_translation_shifts_extent() -> None:
    fg = _disk(cx=40)
    t = sitk.TranslationTransform(2, (15.0, 0.0))  # +15 in x (fixed->moving)
    ext = _warped_atlas_extent(t, fg, fg.shape, fg.shape)
    # Disk centre lands near x=55; original centre x=40 is now near the left rim.
    assert ext[40, 55]
    xs = np.nonzero(ext)[1]
    assert xs.mean() > 48  # shifted right of the original centre (40)


def test_extent_bounded_to_section_shape() -> None:
    fg = _disk()
    big = sitk.TranslationTransform(2, (500.0, 500.0))  # pushes atlas off-frame
    ext = _warped_atlas_extent(big, fg, fg.shape, fg.shape)
    assert ext.sum() == 0  # nothing maps inside -> no stray coverage
