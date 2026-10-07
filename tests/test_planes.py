"""Atlas plane resampling tests using a synthetic 3D volume."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from atlastrack.atlas.planes import (
    Anchoring,
    coronal_anchoring,
    resample_atlas_at_plane,
    sample_plane,
)


def _fake_atlas(ap: int = 40, dv: int = 30, ml: int = 50) -> SimpleNamespace:
    """A toy atlas object exposing the BrainGlobe-style fields we use."""
    rng = np.random.default_rng(0)
    reference = rng.random((ap, dv, ml)).astype(np.float32)
    annotation = np.arange(ap * dv * ml, dtype=np.int32).reshape(ap, dv, ml) % 17
    return SimpleNamespace(
        reference=reference,
        annotation=annotation,
        resolution=(25.0, 25.0, 25.0),
    )


def test_coronal_anchoring_no_tilt() -> None:
    atlas = _fake_atlas()
    a = coronal_anchoring(atlas, ap_um=500.0)
    # AP idx = 500 / 25 = 20
    assert a.ox == 20.0
    assert (a.oy, a.oz) == (0.0, 0.0)
    # u along +ML, length = ML size; v along +DV, length = DV size.
    assert (a.ux, a.uy, a.uz) == (0.0, 0.0, 50.0)
    assert (a.vx, a.vy, a.vz) == (0.0, 30.0, 0.0)


def test_sample_plane_returns_correct_slab() -> None:
    """At zero tilt the sampled plane should equal the corresponding AP slice."""
    atlas = _fake_atlas()
    a = coronal_anchoring(atlas, ap_um=500.0)  # ap_idx = 20
    sampled = sample_plane(atlas.reference, a, out_shape=(30, 50), order=1)
    # The reference slice at AP=20 is reference[20].
    np.testing.assert_allclose(sampled, atlas.reference[20], atol=1e-4)


def test_annotation_uses_nearest_neighbor() -> None:
    """Integer labels must survive resampling without interpolation artifacts."""
    atlas = _fake_atlas()
    a = coronal_anchoring(atlas, ap_um=500.0)
    _ref, annot = resample_atlas_at_plane(atlas, a, out_shape=(30, 50))
    assert annot.dtype == atlas.annotation.dtype
    np.testing.assert_array_equal(annot, atlas.annotation[20])


def test_anchoring_round_trip() -> None:
    values = [10.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 30.0, 0.0]
    a = Anchoring.from_iterable(values)
    assert list(a.as_tuple()) == values


def test_tilts_turn_the_plane_about_its_centre(monkeypatch) -> None:
    import atlastrack.atlas.planes as planes
    from atlastrack.registration.pipeline import anchoring_center_ap_um

    class _Atlas:
        class annotation:
            shape = (528, 320, 456)

    monkeypatch.setattr(planes, "atlas_resolution_um", lambda atlas: (25.0, 25.0, 25.0))
    for tilts in ({"dv_tilt_deg": 5.0}, {"ml_tilt_deg": -5.0}, {"dv_tilt_deg": 4.7, "ml_tilt_deg": 1.0}):
        a = coronal_anchoring(_Atlas(), 10000.0, **tilts)
        assert anchoring_center_ap_um(a.as_tuple(), 25.0) == pytest.approx(10000.0)


def test_tilted_plane_is_slicereg_plane(monkeypatch) -> None:
    """dv_tilt = -pitch, ml_tilt = -yaw gives slicereg's plane exactly."""
    import atlastrack.atlas.planes as planes
    from atlastrack.io.slicereg import plane_to_atlas_um

    class _Atlas:
        class annotation:
            shape = (528, 320, 456)

    monkeypatch.setattr(planes, "atlas_resolution_um", lambda atlas: (25.0, 25.0, 25.0))
    al = {"ap_um": 10300.0, "pitch_deg": -4.7, "yaw_deg": 2.0, "plane_size_um": [11400.0, 8000.0]}
    a = coronal_anchoring(_Atlas(), al["ap_um"], ml_tilt_deg=-al["yaw_deg"],
                          dv_tilt_deg=-al["pitch_deg"])
    o, u, v = (np.array(a.as_tuple()[k:k + 3]) * 25.0 for k in (0, 3, 6))
    st = np.random.default_rng(0).uniform(0, 1, (20, 2)) * [11400.0, 8000.0]
    ours = o + np.outer(st[:, 0] / 11400.0, u) + np.outer(st[:, 1] / 8000.0, v)
    np.testing.assert_allclose(ours, plane_to_atlas_um(al, st), atol=1e-6)


def test_ml_tilt_rotates_basis() -> None:
    atlas = _fake_atlas()
    a = coronal_anchoring(atlas, ap_um=500.0, ml_tilt_deg=10.0)
    # With a small ML tilt, the u-vector should still be mostly along ML but
    # acquire an AP component.
    u = np.array([a.ux, a.uy, a.uz])
    assert abs(u[2]) > abs(u[0])  # ML component dominates
    assert u[0] != 0.0  # but AP component is now non-zero


def test_a_back_view_plane_is_the_front_one_mirrored(monkeypatch) -> None:
    import atlastrack.atlas.planes as planes
    from atlastrack.project.orientation import mirror_anchoring

    class _Atlas:
        class annotation:
            shape = (528, 320, 456)

    monkeypatch.setattr(planes, "atlas_resolution_um", lambda atlas: (25.0, 25.0, 25.0))
    front = coronal_anchoring(_Atlas(), 9000.0, dv_tilt_deg=4.0, ml_tilt_deg=1.0)
    back = coronal_anchoring(_Atlas(), 9000.0, dv_tilt_deg=4.0, ml_tilt_deg=1.0, seen_from="back")
    assert list(back.as_tuple()) == pytest.approx(mirror_anchoring(front.as_tuple(), 456))
    assert front.uz > 0 > back.uz
