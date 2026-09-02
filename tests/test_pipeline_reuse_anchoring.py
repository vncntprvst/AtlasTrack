"""``register_project_with_atlas(reuse_stored_anchoring=...)`` plane selection.

``PlaneParams`` can only describe a coronal plane plus two tilts, so rebuilding an
anchoring from it discards the oblique plane DeepSlice predicted. Re-running a
registration therefore used to move sections silently. These tests pin down which
plane each mode feeds to the fit; ``register_section_image`` is stubbed so no
atlas download or real optimisation is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.project.schema import (
    AtlasRef,
    PlaneParams,
    Project,
    RegistrationResult,
    Section,
    Slide,
)
from atlastrack.registration import pipeline

# A plane that is genuinely oblique - no coronal PlaneParams can reproduce it.
OBLIQUE = [
    120.0, 30.0, 20.0,   # o (origin)
    400.0, 11.0, -7.0,   # u (row axis, tilted)
    5.0, 300.0, 9.0,     # v (col axis, tilted)
]


class _FakeAtlas:
    resolution = (25.0, 25.0, 25.0)
    reference = np.zeros((8, 8, 8), dtype=np.uint16)
    annotation = np.zeros((8, 8, 8), dtype=np.uint32)


@pytest.fixture
def captured(monkeypatch):
    """Record the anchoring handed to each section's fit."""
    seen: list[list[float]] = []

    class _FakeTransform:
        pass

    def _fake_register_section_image(img, atlas, *, anchoring, **kwargs):
        seen.append(list(anchoring.as_tuple()))
        return (
            RegistrationResult(
                anchoring=list(anchoring.as_tuple()),
                bspline_transform_path="",
                residual=0.0,
                output_size_px=(int(img.shape[0]), int(img.shape[1])),
            ),
            _FakeTransform(),
        )

    monkeypatch.setattr(
        pipeline, "register_section_image", _fake_register_section_image
    )
    return seen


def _project(*, with_registration: bool) -> Project:
    section = Section(
        index=0,
        slide_idx=0,
        bbox_px=(0, 0, 16, 16),
        plane=PlaneParams(ap_um=7000.0, pixel_size_um=25.0),
    )
    if with_registration:
        section.registration = RegistrationResult(
            anchoring=list(OBLIQUE),
            bspline_transform_path="transforms/section_000.h5",
            residual=0.3,
            output_size_px=(16, 16),
        )
    return Project(
        atlas=AtlasRef(),
        slides=[Slide(image_path="slide.png", sections=[section])],
        probes=[],
    )


def _run(project, tmp_path, monkeypatch, *, reuse: bool) -> None:
    import SimpleITK as sitk

    monkeypatch.setattr(sitk, "WriteTransform", lambda *a, **k: None)
    pipeline.register_project_with_atlas(
        project,
        _FakeAtlas(),
        section_images={0: np.zeros((16, 16), dtype=np.float32)},
        transforms_dir=tmp_path / "transforms",
        reuse_stored_anchoring=reuse,
    )


def test_reuse_keeps_the_stored_oblique_plane(captured, tmp_path, monkeypatch) -> None:
    _run(_project(with_registration=True), tmp_path, monkeypatch, reuse=True)
    assert captured == [OBLIQUE], "a stored anchoring must survive a re-run"


def test_without_reuse_the_plane_is_rebuilt_from_plane_params(
    captured, tmp_path, monkeypatch
) -> None:
    _run(_project(with_registration=True), tmp_path, monkeypatch, reuse=False)
    assert captured and captured[0] != OBLIQUE, (
        "default behaviour re-derives a coronal plane from PlaneParams"
    )


def test_reuse_falls_back_to_plane_params_when_never_registered(
    captured, tmp_path, monkeypatch
) -> None:
    _run(_project(with_registration=False), tmp_path, monkeypatch, reuse=True)
    assert captured and captured[0] != OBLIQUE
