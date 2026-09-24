"""The outer-contour snap must align the atlas *brain*, not the atlas slice frame.

It used to take the atlas silhouette from the reference image thresholded at 2% of
its range. On many planes the template's background clears 2%, so that silhouette
covered 95-99% of the slice - the rectangular frame - against ~65% for the brain.
The snap then pulled the frame onto the tissue and shrank the brain inside it: on a
real slide the drawn outline stopped 20-40 px short of the tissue edge, worse than
no snap at all, and only on the planes whose background happened to clear 2%.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from atlastrack.registration import pipeline
from atlastrack.registration.bspline import RegisterResult
from atlastrack.registration.transforms import _warped_atlas_extent


def _disk(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def test_the_snap_brings_a_short_brain_out_to_the_tissue_edge() -> None:
    """The behaviour itself: a brain that lands short of the tissue is pulled out."""
    shape = (120, 160)
    brain = _disk(shape, 60, 80, 38)       # where the atlas brain landed
    tissue = _disk(shape, 60, 80, 46)      # the real tissue, a little larger
    section = np.zeros((*shape, 3), dtype=np.float32)
    section[tissue] = 0.8
    identity = sitk.Transform(2, sitk.sitkIdentity)

    snapped = pipeline._apply_boundary_snap(identity, brain, section, shape)

    before = _dice(brain, tissue)
    after = _dice(_warped_atlas_extent(snapped, brain, shape, shape), tissue)
    assert after > before + 0.03, f"snap did not help: {before:.3f} -> {after:.3f}"
    assert after > 0.95


def test_an_empty_brain_leaves_the_registration_alone() -> None:
    """A plane that misses the brain entirely has nothing to snap."""
    shape = (40, 50)
    identity = sitk.Transform(2, sitk.sitkIdentity)
    out = pipeline._apply_boundary_snap(
        identity, np.zeros(shape, bool), np.zeros((*shape, 3), np.float32), shape
    )
    assert out is identity


class _FakeAtlas:
    """A plane whose reference background clears 2% everywhere - the bug's trigger."""

    resolution = (25.0, 25.0, 25.0)

    def __init__(self) -> None:
        self.reference = np.full((10, 40, 50), 0.10, dtype=np.float32)  # bright background
        self.reference[:, 8:32, 10:40] = 1.0
        self.annotation = np.zeros((10, 40, 50), dtype=np.uint32)
        self.annotation[:, 12:28, 15:35] = 7                            # the brain


def test_the_pipeline_hands_the_snap_the_annotation_not_a_reference_threshold(monkeypatch):
    """Wiring: what reaches the snap must be ``annotation > 0``.

    With this fake atlas a 2% reference threshold selects the entire slice, while
    the annotation selects the brain block only - so the two are told apart.
    """
    from atlastrack.atlas.planes import Anchoring, annotation_at_plane

    atlas = _FakeAtlas()
    anchoring = Anchoring(5.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 40.0, 0.0)
    section = np.zeros((40, 50, 3), dtype=np.float32)
    section[10:30, 12:38] = 0.8
    seen: list[np.ndarray] = []

    monkeypatch.setattr(
        pipeline, "_refine",
        lambda reference, moving, **kw: RegisterResult(
            sitk.Transform(2, sitk.sitkIdentity), 0.1, 5),
    )

    def _capture(transform, atlas_brain, section_image, out_shape):
        seen.append(np.asarray(atlas_brain))
        return transform

    monkeypatch.setattr(pipeline, "_apply_boundary_snap", _capture)

    pipeline.register_section_image(
        section, atlas, anchoring=anchoring, reference_volume=atlas.reference,
        use_masks=False, prealign=False, boundary_snap=True,
    )

    assert seen, "the snap was never called"
    expected = annotation_at_plane(atlas, anchoring, (40, 50)) > 0
    assert seen[0].dtype == bool
    assert np.array_equal(seen[0], expected)
    assert seen[0].mean() < 0.5, "that is the whole frame, not the brain"


def test_no_snap_means_no_annotation_sampling(monkeypatch):
    """Switching the snap and the plane check off must not cost an annotation resample."""
    from atlastrack.atlas import planes

    atlas = _FakeAtlas()
    calls: list[int] = []
    real = planes.annotation_at_plane
    monkeypatch.setattr(planes, "annotation_at_plane",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(
        pipeline, "_refine",
        lambda reference, moving, **kw: RegisterResult(
            sitk.Transform(2, sitk.sitkIdentity), 0.1, 5),
    )
    from atlastrack.atlas.planes import Anchoring

    pipeline.register_section_image(
        np.zeros((40, 50, 3), np.float32), atlas,
        anchoring=Anchoring(5.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 40.0, 0.0),
        reference_volume=atlas.reference, use_masks=False, prealign=False,
        boundary_snap=False, plane_fallback=False,
    )
    assert calls == []
