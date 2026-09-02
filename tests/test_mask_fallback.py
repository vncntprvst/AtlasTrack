"""A section elastix cannot fit *with* a mask must still be registered without one.

Sections whose tissue mask under-covers the brain (low-contrast lightsheet
renders) made elastix abort with too few valid metric samples. Four of LO_03's
eight sections failed that way, and the run still reported 8/8 - 100%.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.registration import pipeline
from atlastrack.registration.bspline import RegisterResult


class _FakeAtlas:
    resolution = (25.0, 25.0, 25.0)


def _reference(shape=(40, 50)):
    ref = np.zeros(shape, dtype=np.float32)
    ref[8:32, 10:40] = 1.0
    return ref


@pytest.fixture
def section_rgb():
    img = np.zeros((40, 50, 3), dtype=np.float32)
    img[8:32, 10:40, :] = 0.8
    return img


def _patch_common(monkeypatch, calls, refine):
    monkeypatch.setattr(pipeline, "_refine", refine)
    monkeypatch.setattr(
        pipeline, "resample_atlas_at_plane", lambda *a, **k: (_reference(), None)
    )
    monkeypatch.setattr(
        pipeline, "_apply_boundary_snap", lambda tf, *a, **k: tf
    )
    return calls


def test_masked_failure_is_retried_without_the_mask(monkeypatch, section_rgb):
    calls = []

    def refine(reference, moving, **kw):
        calls.append(kw["use_masks"])
        if kw["use_masks"]:
            raise RuntimeError("ITK ERROR: Internal elastix error: See elastix log")
        import SimpleITK as sitk

        return RegisterResult(sitk.Transform(2, sitk.sitkIdentity), 0.2, 5)

    _patch_common(monkeypatch, calls, refine)

    reg = pipeline.register_section_image(
        section_rgb, _FakeAtlas(), anchoring=_anchoring(), use_masks=True,
        boundary_snap=False,
    )[0]

    assert calls == [True, False], "expected one masked attempt then one unmasked"
    assert reg.used_mask_fallback is True
    assert reg.residual == pytest.approx(0.2)


def test_successful_masked_fit_does_not_set_the_flag(monkeypatch, section_rgb):
    def refine(reference, moving, **kw):
        import SimpleITK as sitk

        return RegisterResult(sitk.Transform(2, sitk.sitkIdentity), 0.1, 5)

    _patch_common(monkeypatch, [], refine)

    reg = pipeline.register_section_image(
        section_rgb, _FakeAtlas(), anchoring=_anchoring(), use_masks=True,
        boundary_snap=False,
    )[0]

    assert reg.used_mask_fallback is False


def test_an_unrelated_failure_still_propagates(monkeypatch, section_rgb):
    """The retry must not swallow genuine errors."""
    def refine(reference, moving, **kw):
        raise ValueError("something else entirely")

    _patch_common(monkeypatch, [], refine)

    with pytest.raises(ValueError, match="something else entirely"):
        pipeline.register_section_image(
            section_rgb, _FakeAtlas(), anchoring=_anchoring(), use_masks=True,
            boundary_snap=False,
        )


def test_no_retry_when_masks_were_already_off(monkeypatch, section_rgb):
    calls = []

    def refine(reference, moving, **kw):
        calls.append(kw["use_masks"])
        raise RuntimeError("Internal elastix error")

    _patch_common(monkeypatch, calls, refine)

    with pytest.raises(RuntimeError):
        pipeline.register_section_image(
            section_rgb, _FakeAtlas(), anchoring=_anchoring(), use_masks=False,
            boundary_snap=False,
        )

    assert calls == [False], "an unmasked failure has no safer setting to retry"


@pytest.mark.parametrize(
    "message, retryable",
    [
        ("ITK ERROR: Internal elastix error: See elastix log", True),
        ("Too many samples map outside moving image buffer: 101 / 2048", True),
        ("could not open file", False),
        ("", False),
    ],
)
def test_failure_classification(message, retryable):
    assert pipeline._is_sample_coverage_failure(RuntimeError(message)) is retryable


def _anchoring():
    from atlastrack.io.quicknii import Anchoring

    return Anchoring.from_iterable([0, 0, 0, 1, 0, 0, 0, 1, 0])
