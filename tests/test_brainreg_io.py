"""Tests for io/brainreg.py - mapping sample-volume points into CCF micrometres."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from histo_to_ccf.io.brainreg import BrainregRegistration

SHAPE = (12, 10, 8)


def _make_brainreg_dir(tmp_path, fields=None):
    """A minimal brainreg output directory with known deformation fields."""
    d = tmp_path / "sample_brainreg"
    d.mkdir()
    if fields is None:
        # Field i encodes a simple, distinguishable ramp in mm.
        ap = np.zeros(SHAPE, dtype=np.float32)
        dv = np.zeros(SHAPE, dtype=np.float32)
        ml = np.zeros(SHAPE, dtype=np.float32)
        i, j, k = np.indices(SHAPE)
        ap[:] = 1.0 + 0.1 * i        # mm
        dv[:] = 2.0 + 0.2 * j
        ml[:] = 3.0 + 0.3 * k
        fields = (ap, dv, ml)
    for n, arr in enumerate(fields):
        tifffile.imwrite(d / f"deformation_field_{n}.tiff", arr)
    return d


def test_rejects_a_directory_without_deformation_fields(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="not a brainreg output directory"):
        BrainregRegistration(tmp_path / "empty")


def test_shape_matches_the_fields(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    assert reg.shape == SHAPE


def test_maps_voxels_to_ccf_um_in_ap_ml_dv_order(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))

    out = reg.sample_voxels_to_ccf_um([[0, 0, 0], [5, 4, 3]])

    # Fields are (AP, DV, ML) in mm; output must be (AP, ML, DV) in um.
    assert out.shape == (2, 3)
    assert out[0] == pytest.approx([1000.0, 3000.0, 2000.0])
    assert out[1] == pytest.approx([1500.0, 3900.0, 2800.0])


def test_accepts_a_single_point(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    out = reg.sample_voxels_to_ccf_um([1, 1, 1])
    assert out.shape == (1, 3)
    assert out[0] == pytest.approx([1100.0, 3300.0, 2200.0])


def test_fractional_voxels_round_to_nearest(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    near = reg.sample_voxels_to_ccf_um([[4.6, 3.4, 2.5]])
    exact = reg.sample_voxels_to_ccf_um([[5, 3, 2]])
    assert near == pytest.approx(exact)


def test_out_of_range_voxels_clamp_to_the_volume(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    clamped = reg.sample_voxels_to_ccf_um([[-5, -5, -5], [999, 999, 999]])
    edge = reg.sample_voxels_to_ccf_um(
        [[0, 0, 0], [SHAPE[0] - 1, SHAPE[1] - 1, SHAPE[2] - 1]]
    )
    assert clamped == pytest.approx(edge)


def test_unmapped_voxels_come_back_as_nan(tmp_path) -> None:
    zeros = [np.zeros(SHAPE, dtype=np.float32) for _ in range(3)]
    for arr in zeros:
        arr[2, 2, 2] = 0.0        # explicitly unmapped
        arr[3, 3, 3] = 1.0        # mapped
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path, fields=zeros))

    out = reg.sample_voxels_to_ccf_um([[2, 2, 2], [3, 3, 3]])

    assert np.all(np.isnan(out[0]))
    assert not np.any(np.isnan(out[1]))


def test_bad_point_shape_is_rejected(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        reg.sample_voxels_to_ccf_um([[1, 2], [3, 4]])


def test_sample_um_to_voxels_uses_the_voxel_size(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    vox = reg.sample_um_to_voxels([[180.0, 180.0, 200.0]], voxel_size_um=(1.8, 1.8, 2.0))
    assert vox[0] == pytest.approx([100.0, 100.0, 100.0])


def test_sample_um_to_voxels_rejects_non_positive_sizes(tmp_path) -> None:
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path))
    with pytest.raises(ValueError, match="must be positive"):
        reg.sample_um_to_voxels([[1.0, 1.0, 1.0]], voxel_size_um=0.0)


# ---------------------------------------------------------------------------
# Geometry guard
# ---------------------------------------------------------------------------

def test_plausible_dv_extent_passes(tmp_path) -> None:
    fields = [np.ones((100, 320, 100), dtype=np.float32) for _ in range(3)]
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path, fields=fields))
    ok, msg = reg.check_geometry()
    assert ok, msg
    assert reg.dv_extent_um() == pytest.approx(8000.0)


def test_squashed_volume_is_flagged(tmp_path) -> None:
    # LO_04's real geometry: 198 DV voxels = 4950 um, far too shallow for a brain.
    fields = [np.ones((100, 198, 100), dtype=np.float32) for _ in range(3)]
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path, fields=fields))
    ok, msg = reg.check_geometry()
    assert not ok
    assert "voxel sizes" in msg


def test_stretched_volume_is_flagged(tmp_path) -> None:
    # LO_03's real geometry: 635 DV voxels = 15875 um, about twice a brain.
    fields = [np.ones((100, 635, 100), dtype=np.float32) for _ in range(3)]
    reg = BrainregRegistration(_make_brainreg_dir(tmp_path, fields=fields))
    ok, msg = reg.check_geometry()
    assert not ok
    assert "1.98x" in msg or "2.0" in msg
