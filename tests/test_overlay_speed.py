"""'Show atlas overlay' went from ~10 s to under 5 s (0.7 s when repeated).

Three changes, each pinned here, because each is easy to undo by accident:

* Plane sampling no longer casts the *whole* 3D atlas to another dtype to read one
  2D slice. That cast copied 154 MB (reference) and 308 MB (uint32 annotation) per
  section, and the overlay then threw the reference slice away. The output must stay
  bit-identical, or registration results would silently shift.
* Warped overlays are cached, keyed on everything they depend on - including the
  transform sidecar's modification time, because re-registering rewrites the same
  path.
* Sections are warped in parallel, which is only safe because reads of the ``.h5``
  sidecars are serialised: SimpleITK's bundled HDF5 is not thread-safe.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from atlastrack.atlas.planes import (
    Anchoring,
    annotation_at_plane,
    resample_atlas_at_plane,
    sample_plane,
)
from atlastrack.project.schema import RegistrationResult


class _FakeAtlas:
    """Just enough of a BrainGlobeAtlas for plane sampling and cache keys."""

    atlas_name = "fake_atlas_25um"

    def __init__(self) -> None:
        rng = np.random.default_rng(0)
        self.reference = rng.integers(0, 4000, (40, 30, 36), dtype=np.uint16)
        # Real CCF ids exceed both 2**16 and 2**24; keep one that large so a lossy
        # path through float32 would show up.
        self.annotation = rng.integers(0, 5, (40, 30, 36), dtype=np.uint32)
        self.annotation[self.annotation == 4] = 614_454_277


def _anchorings() -> list[Anchoring]:
    return [
        Anchoring(20.0, 0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 30.0, 0.0),  # plain coronal
        Anchoring(18.0, 1.0, 2.0, 3.5, 0.0, 35.0, -2.0, 29.0, 1.5),  # tilted
    ]


def _old_resample(atlas, anchoring, shape):
    """The implementation this replaced: cast the whole volume, then sample."""
    ref = sample_plane(atlas.reference.astype(np.float32), anchoring, shape, order=1)
    ann = sample_plane(
        atlas.annotation.astype(np.int32), anchoring, shape, order=0
    ).astype(atlas.annotation.dtype)
    return ref, ann


# ------------------------------------------------------------ plane sampling


@pytest.mark.parametrize("anchoring", _anchorings(), ids=["coronal", "tilted"])
def test_sampling_without_the_whole_volume_cast_is_bit_identical(anchoring) -> None:
    """Registration uses the reference slice, so it must not move by one ulp."""
    atlas = _FakeAtlas()
    ref_old, ann_old = _old_resample(atlas, anchoring, (25, 33))
    ref_new, ann_new = resample_atlas_at_plane(atlas, anchoring, (25, 33))

    assert ref_new.dtype == ref_old.dtype == np.float32
    assert np.array_equal(ref_new, ref_old)
    assert ann_new.dtype == ann_old.dtype == np.uint32
    assert np.array_equal(ann_new, ann_old)


def test_large_region_ids_survive_the_annotation_sample() -> None:
    """CCF ids reach ~6e8; a float32 detour would round them to neighbouring ids."""
    atlas = _FakeAtlas()
    ann = annotation_at_plane(atlas, _anchorings()[0], (25, 33))
    assert 614_454_277 in set(np.unique(ann).tolist())


def test_the_annotation_only_path_matches_the_full_one() -> None:
    atlas = _FakeAtlas()
    for anchoring in _anchorings():
        _ref, ann = resample_atlas_at_plane(atlas, anchoring, (25, 33))
        assert np.array_equal(annotation_at_plane(atlas, anchoring, (25, 33)), ann)


def test_the_volumes_are_not_copied_to_sample_a_plane(monkeypatch) -> None:
    """The regression itself: a full-volume ``astype`` per section."""
    atlas = _FakeAtlas()
    original = np.ndarray.astype
    calls: list[tuple[int, ...]] = []

    class _Watch(np.ndarray):
        def astype(self, *a, **k):  # pragma: no cover - only hit on regression
            calls.append(self.shape)
            return original(self, *a, **k)

    atlas.reference = atlas.reference.view(_Watch)
    atlas.annotation = atlas.annotation.view(_Watch)
    big = {atlas.reference.shape}

    resample_atlas_at_plane(atlas, _anchorings()[1], (25, 33))

    assert not [s for s in calls if s in big], f"whole-volume casts: {calls}"


# ------------------------------------------------------------------- cache


def _result(tmp_path: Path) -> RegistrationResult:
    sidecar = tmp_path / "transforms" / "section_000.h5"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(b"first fit")
    return RegistrationResult(
        anchoring=[20.0, 0.0, 0.0, 0.0, 0.0, 36.0, 0.0, 30.0, 0.0],
        output_size_px=(25, 33),
        bspline_transform_path="transforms/section_000.h5",
    )


@pytest.fixture
def counted(monkeypatch):
    """Replace the expensive warp with a counter, and start from an empty cache."""
    from atlastrack.registration import transforms

    transforms.clear_warp_cache()
    calls: list[int] = []

    def _fake(result, atlas, section_shape, **_k):
        calls.append(1)
        return np.full(section_shape, len(calls), dtype=np.uint32)

    monkeypatch.setattr(transforms, "_warp_annotation_to_section_uncached", _fake)
    yield transforms, calls
    transforms.clear_warp_cache()


def test_a_repeat_warp_is_served_from_the_cache(counted, tmp_path) -> None:
    transforms, calls = counted
    atlas, result = _FakeAtlas(), _result(tmp_path)

    first = transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)
    second = transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)

    assert len(calls) == 1, "the second call must not recompute"
    assert np.array_equal(first, second)


def test_editing_a_returned_overlay_cannot_poison_the_cache(counted, tmp_path) -> None:
    """Callers zero out pixels (extent clipping); that must not leak into the next caller."""
    transforms, _calls = counted
    atlas, result = _FakeAtlas(), _result(tmp_path)

    first = transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)
    first[:] = 0
    again = transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)

    assert again.any(), "the cached copy was modified through a returned array"


def test_re_registering_invalidates_the_cached_overlay(counted, tmp_path) -> None:
    """A new fit rewrites the *same* sidecar path - the key must notice."""
    transforms, calls = counted
    atlas, result = _FakeAtlas(), _result(tmp_path)
    transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)

    sidecar = tmp_path / "transforms" / "section_000.h5"
    sidecar.write_bytes(b"second fit, different size")
    later = time.time() + 5
    os.utime(sidecar, (later, later))

    transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)
    assert len(calls) == 2, "a stale overlay was served after re-registration"


def test_a_different_section_size_is_a_different_entry(counted, tmp_path) -> None:
    transforms, calls = counted
    atlas, result = _FakeAtlas(), _result(tmp_path)
    transforms.warp_annotation_to_section(result, atlas, (20, 30), project_dir=tmp_path)
    transforms.warp_annotation_to_section(result, atlas, (21, 30), project_dir=tmp_path)
    assert len(calls) == 2


# -------------------------------------------------------------------- batch


def test_the_batch_keeps_order_and_isolates_failures(monkeypatch, tmp_path) -> None:
    """One bad sidecar must not cost every other section its overlay."""
    from atlastrack.registration import transforms

    transforms.clear_warp_cache()

    def _fake(result, atlas, section_shape, **_k):
        if section_shape[0] == 13:
            raise RuntimeError("corrupt sidecar")
        return np.full(section_shape, section_shape[0], dtype=np.uint32)

    monkeypatch.setattr(transforms, "_warp_annotation_to_section_uncached", _fake)
    result = _result(tmp_path)
    shapes = [(10, 5), (13, 5), (11, 5), (12, 5)]

    out = transforms.warp_annotations_to_sections(
        [(result, s) for s in shapes], _FakeAtlas(), project_dir=tmp_path, max_workers=3
    )

    assert isinstance(out[1], RuntimeError)
    assert [int(o[0, 0]) for i, o in enumerate(out) if i != 1] == [10, 11, 12]
    transforms.clear_warp_cache()


def test_the_batch_restores_itk_threading(monkeypatch, tmp_path) -> None:
    """ITK's thread count is process-wide; leaving it lowered would slow registration."""
    import SimpleITK as sitk

    from atlastrack.registration import transforms

    transforms.clear_warp_cache()
    monkeypatch.setattr(
        transforms, "_warp_annotation_to_section_uncached",
        lambda r, a, s, **k: np.zeros(s, dtype=np.uint32),
    )
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    transforms.warp_annotations_to_sections(
        [(_result(tmp_path), (8 + i, 4)) for i in range(4)],
        _FakeAtlas(), project_dir=tmp_path, max_workers=4,
    )
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == before
    transforms.clear_warp_cache()


def test_sidecar_reads_are_serialised() -> None:
    """Concurrent ReadTransform crashes SimpleITK's bundled HDF5 (H5Gopen2 failed)."""
    import inspect

    from atlastrack.registration import transforms

    source = inspect.getsource(transforms.build_registered_transform)
    assert "with _HDF5_LOCK:" in source
    assert "sitk.ReadTransform" in source.split("with _HDF5_LOCK:")[1].split("\n")[1]
