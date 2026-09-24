"""Sections whose brainstem came away from the cerebellum.

The tissue mask used to keep the largest body only. On such a section that
dropped the brainstem - a third of the tissue - and the metric mask, the
pre-alignment and the boundary snap all fitted the whole atlas brain onto the
cerebellum. On a real slide the three detached sections scored Dice 0.61-0.85
against their tissue, worse than the unwarped plane (0.82).
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from atlastrack.registration import pipeline
from atlastrack.registration.bspline import RegisterResult
from atlastrack.registration.masks import section_tissue_mask


def _ellipse(shape, cy, cx, ry, rx):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _rgb(mask: np.ndarray) -> np.ndarray:
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask, 2] = 180
    return img


def test_a_detached_brainstem_stays_in_the_tissue_mask() -> None:
    shape = (200, 240)
    cerebellum = _ellipse(shape, 50, 120, 35, 90)
    brainstem = _ellipse(shape, 160, 120, 30, 60)   # a gap wider than the closing bridges
    m = section_tissue_mask(_rgb(cerebellum | brainstem))
    assert m[50, 120] and m[160, 120]
    assert ndi.label(m)[1] == 2


def test_debris_and_a_neighbours_fragment_are_still_dropped() -> None:
    shape = (200, 240)
    body = _ellipse(shape, 90, 120, 60, 80)
    debris = _ellipse(shape, 180, 30, 5, 5)                 # well under 10% of the body
    neighbour = _ellipse(shape, 100, 250, 50, 30)           # big, but cut by the crop edge
    m = section_tissue_mask(_rgb(body | debris | neighbour))
    assert m[90, 120]
    assert not m[180, 30], "debris kept"
    assert not m[100, 235], "a neighbouring section's fragment kept"


def _fake_atlas():
    class _FakeAtlas:
        resolution = (25.0, 25.0, 25.0)

        def __init__(self) -> None:
            self.reference = np.zeros((10, 80, 100), dtype=np.float32)
            self.reference[:, 15:65, 20:80] = 1.0
            self.annotation = np.zeros((10, 80, 100), dtype=np.uint32)
            self.annotation[:, 15:65, 20:80] = 7

    return _FakeAtlas()


def _register(monkeypatch, transform, **kw):
    from atlastrack.atlas.planes import Anchoring

    monkeypatch.setattr(
        pipeline, "_refine",
        lambda reference, moving, **k: RegisterResult(transform, 0.1, 5),
    )
    section = np.zeros((80, 100, 3), dtype=np.float32)
    section[15:65, 20:80] = 0.8                     # tissue exactly where the plane is
    return pipeline.register_section_image(
        section, _fake_atlas(),
        anchoring=Anchoring(5.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 80.0, 0.0),
        reference_volume=None, use_masks=False, prealign=False, boundary_snap=False,
        **kw,
    )


def _shrink(factor: float) -> sitk.Transform:
    """Fixed -> moving map that draws the atlas brain ``factor`` times its size."""
    t = sitk.ScaleTransform(2, (1.0 / factor, 1.0 / factor))
    t.SetCenter((50.0, 40.0))
    return t


def test_a_morph_worse_than_the_plane_is_dropped(monkeypatch) -> None:
    reg, transform = _register(monkeypatch, _shrink(0.5))
    assert transform is None
    assert reg.morph_fallback
    assert reg.bspline_transform_path is None


def test_a_morph_that_fits_is_kept(monkeypatch) -> None:
    identity = sitk.Transform(2, sitk.sitkIdentity)
    reg, transform = _register(monkeypatch, identity)
    assert transform is identity
    assert not reg.morph_fallback


def test_a_near_tie_keeps_the_morph(monkeypatch) -> None:
    """A morph a hair worse than the plane is mask noise, not a failed fit."""
    slightly_small = _shrink(0.97)
    reg, transform = _register(monkeypatch, slightly_small)
    assert transform is slightly_small
    assert not reg.morph_fallback


def test_the_fallback_can_be_switched_off(monkeypatch) -> None:
    bad = _shrink(0.5)
    reg, transform = _register(monkeypatch, bad, plane_fallback=False)
    assert transform is bad
    assert not reg.morph_fallback


def test_projects_saved_before_the_flag_load_without_it() -> None:
    from atlastrack.project.schema import RegistrationResult

    reg = RegistrationResult.model_validate(
        {"anchoring": [0.0] * 9, "output_size_px": (10, 10)}
    )
    assert reg.morph_fallback is False
