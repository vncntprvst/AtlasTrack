"""Readable plane values and the series check."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.atlas.planes import coronal_anchoring
from atlastrack.project.schema import Project, RegistrationResult, Section, Slide
from atlastrack.project.series import plane_values, series_flags, series_rows
from atlastrack.registration.landmarks_warp import invert_points, warp_points


class _Atlas:
    class annotation:
        shape = (528, 320, 456)


@pytest.fixture(autouse=True)
def _resolution(monkeypatch):
    import atlastrack.atlas.planes as planes

    monkeypatch.setattr(planes, "atlas_resolution_um", lambda atlas: (25.0, 25.0, 25.0))


def _anchoring(**tilts) -> list[float]:
    a = coronal_anchoring(_Atlas(), 9000.0, **tilts)
    return [a.ox, a.oy, a.oz, a.ux, a.uy, a.uz, a.vx, a.vy, a.vz]


def test_plane_values_read_tilts_back() -> None:
    v = plane_values(_anchoring(dv_tilt_deg=4.0, ml_tilt_deg=-2.0), (320, 456), 25.0)
    # A positive dv_tilt moves the ventral edge anterior: deeper is further forward.
    assert v.pitch_deg == pytest.approx(-4.0, abs=0.05)
    assert v.yaw_deg == pytest.approx(2.0, abs=0.05)
    assert v.scale_x_um_per_px == pytest.approx(25.0, rel=1e-3)
    assert not v.mirrored


def test_series_flags_point_at_the_odd_section() -> None:
    aps = [9000, 9100, 9050, 9300, 9400]
    sections = []
    for i, ap in enumerate(aps):
        tilt = 9.0 if i == 4 else 4.0
        anchoring = _anchoring(dv_tilt_deg=tilt)
        anchoring[0] += (ap - 9000) / 25.0
        sections.append(Section(
            index=i, slide_idx=0, bbox_px=(0, 0, 456, 320), ap_order=i,
            registration=RegistrationResult(anchoring=anchoring, output_size_px=(320, 456)),
        ))
    rows = series_rows(Project(slides=[Slide(image_path="x.tif", sections=sections)]))
    flags = series_flags(rows)
    assert set(flags) == {2, 4}
    assert flags[2][0].startswith("AP out of order")
    assert flags[4][0].startswith("pitch")


@pytest.mark.parametrize("forward", [False, True])
def test_invert_points_undoes_the_drawn_warp(forward: bool) -> None:
    rng = np.random.default_rng(0)
    source = rng.uniform(0, 100, (8, 2))
    target = source + rng.normal(0, 4, source.shape)
    pts = rng.uniform(10, 90, (40, 2))
    back = invert_points(source, target, pts, forward=forward)
    np.testing.assert_allclose(warp_points(source, target, back, forward=forward), pts, atol=1e-3)


def test_own_landmarks_map_points_as_they_always_did() -> None:
    """AtlasTrack's landmarks: points go back through the spline fitted target -> source."""
    from scipy.interpolate import RBFInterpolator

    rng = np.random.default_rng(1)
    source = rng.uniform(0, 100, (8, 2))
    target = source + rng.normal(0, 6, source.shape)
    pts = rng.uniform(10, 90, (40, 2))
    reverse = RBFInterpolator(target, source, kernel="thin_plate_spline", smoothing=0.0)
    np.testing.assert_allclose(invert_points(source, target, pts), reverse(pts))
