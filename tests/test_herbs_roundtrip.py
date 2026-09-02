"""Verify the HERBS pkl reader ↔ writer round-trip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atlastrack.io.herbs import (
    DEFAULT_HERBS_GRID_UM,
    ccf_um_to_herbs_voxel,
    herbs_voxel_to_ccf_um,
    load_herbs_pkl,
)
from atlastrack.io.herbs_writer import write_herbs_pkl


@pytest.fixture
def sample_ccf_um() -> np.ndarray:
    """A short fake shank trajectory in CCF AP/ML/DV µm."""
    rng = np.random.default_rng(42)
    n = 16
    return np.column_stack(
        [
            np.full(n, 7000.0) + rng.normal(0, 10, n),  # AP near 7000 µm
            np.linspace(4000.0, 4200.0, n),  # ML
            np.linspace(0.0, 4000.0, n),  # DV (descending into the brain)
        ]
    )


def test_voxel_um_inverse(sample_ccf_um: np.ndarray) -> None:
    """Round-trip through voxel space recovers the original µm coords."""
    vox = ccf_um_to_herbs_voxel(sample_ccf_um, grid_um=DEFAULT_HERBS_GRID_UM)
    recovered = herbs_voxel_to_ccf_um(vox, grid_um=DEFAULT_HERBS_GRID_UM)
    np.testing.assert_allclose(recovered, sample_ccf_um, atol=1e-6)


def test_pkl_round_trip(tmp_path: Path, sample_ccf_um: np.ndarray) -> None:
    """Writer → reader returns the same CCF µm within voxel-quantization tolerance."""
    out = tmp_path / "shank.pkl"
    regions = ["root"] * len(sample_ccf_um)
    write_herbs_pkl(out, [sample_ccf_um], regions_per_shank=[regions])

    [shank] = load_herbs_pkl(out, grid_um=DEFAULT_HERBS_GRID_UM)
    # The pkl stores voxel-quantized coords. With a 10 µm grid the round-trip
    # is exact because writer stores float voxel indices, but to be safe allow
    # one voxel.
    np.testing.assert_allclose(shank["ccf_um"], sample_ccf_um, atol=DEFAULT_HERBS_GRID_UM)
    assert shank["regions"] == regions
