"""Verify HERBS pkl reader ↔ writer round-trip and cross-check against legacy code."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from histo_to_ccf.io.herbs import (
    DEFAULT_HERBS_GRID_UM,
    ccf_um_to_herbs_voxel,
    herbs_voxel_to_ccf_um,
    load_herbs_pkl,
)
from histo_to_ccf.io.herbs_writer import write_herbs_pkl


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


def test_matches_legacy_reader(tmp_path: Path, sample_ccf_um: np.ndarray) -> None:
    """Our writer produces a pkl the legacy `load_herbs_pkl` can read.

    Loads the legacy module from ``legacy/HERBS_to_AllenCCF/herbs_probe_mapping.py``
    by file path so we don't depend on it being on ``sys.path``.
    """
    legacy_module_path = (
        Path(__file__).parent.parent / "legacy" / "HERBS_to_AllenCCF" / "herbs_probe_mapping.py"
    )
    if not legacy_module_path.exists():
        pytest.skip(f"legacy module not present at {legacy_module_path}")

    spec = importlib.util.spec_from_file_location("_legacy_hpm", legacy_module_path)
    assert spec is not None and spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    out = tmp_path / "shank.pkl"
    write_herbs_pkl(out, [sample_ccf_um])

    legacy_shanks = legacy.load_herbs_pkl(out)
    assert len(legacy_shanks) == 1

    # Legacy returns columns (ML, DV, AP); ours returns (AP, ML, DV). Reorder.
    legacy_ccf = legacy_shanks[0]["ccf"]  # (N, 3): ML, DV, AP
    legacy_ap_ml_dv = np.column_stack(
        [legacy_ccf[:, 2], legacy_ccf[:, 0], legacy_ccf[:, 1]]
    )
    # Both paths use a 10 µm grid by default; values should agree to << 1 µm.
    np.testing.assert_allclose(legacy_ap_ml_dv, sample_ccf_um, atol=1e-6)
