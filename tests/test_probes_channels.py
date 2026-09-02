"""Tests for probes/channels.py and probes/catalog.py.

Includes a cross-check against the legacy project_feature_points_to_ccf logic.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from atlastrack.io.ccf_coords import ccf_um_to_paxinos_mm
from atlastrack.probes.catalog import CATALOG, get_layout
from atlastrack.probes.channels import (
    channel_ccf_coords,
    export_channel_csv,
    export_paxinos_csv,
    project_channel_coords,
    shank_channel_coords,
)
from atlastrack.probes.geometry import ELECTRODE_COLUMN_CENTER_UM
from atlastrack.project.schema import (
    AtlasRef,
    Point2D,
    ProbeSpec,
    ProbeType,
    Project,
    Shank,
)

# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def test_catalog_contains_known_probes() -> None:
    assert "Neuropixels 1.0" in CATALOG
    assert "Neuropixels 2.0 (4-shank)" in CATALOG
    assert "NeuroNexus A1x32-Poly3-10mm-25s-177-OA32LP" in CATALOG


def test_neuronexus_poly3_layout() -> None:
    layout = CATALOG["NeuroNexus A1x32-Poly3-10mm-25s-177-OA32LP"]
    depths = layout.site_depths_from_tip_um()
    offsets = layout.site_lateral_offsets_um()
    assert layout.n_channels == 32
    assert len(depths) == 32 and len(offsets) == 32
    # Ordered tip → base.
    assert list(depths) == sorted(depths)
    # Lowest site is the centre column at 62 µm above the tip; centred laterally.
    assert depths[0] == pytest.approx(62.0)
    assert offsets[0] == pytest.approx(0.0)
    assert offsets.mean() == pytest.approx(0.0, abs=1e-6)  # symmetric about centreline
    # 3 columns at -18 / 0 / +18 µm.
    assert sorted(set(round(o, 3) for o in offsets)) == [-18.0, 0.0, 18.0]
    # Side columns sit half a pitch above the centre rows, not level with them.
    centre_rows = np.unique(depths[offsets == 0.0])
    side_rows = np.unique(depths[offsets != 0.0])
    assert len(centre_rows) == 12 and len(side_rows) == 10
    assert side_rows == pytest.approx(centre_rows[:10] + 12.5)
    assert depths.max() - depths.min() == pytest.approx(275.0)
    assert layout.fiber_offset_above_top_site_um == pytest.approx(50.0)


def test_np10_depths_start_at_175() -> None:
    layout = CATALOG["Neuropixels 1.0"]
    depths = layout.site_depths_from_tip_um()
    assert depths[0] == pytest.approx(175.0)
    assert len(depths) == 384


def test_np10_lateral_offsets_are_symmetric() -> None:
    layout = CATALOG["Neuropixels 1.0"]
    lats = layout.site_lateral_offsets_um()
    assert lats.mean() == pytest.approx(0.0, abs=1.0)  # centred on shank


def test_np20_depths_start_near_zero() -> None:
    layout = CATALOG["Neuropixels 2.0 (4-shank)"]
    depths = layout.site_depths_from_tip_um()
    assert depths[0] == pytest.approx(0.0)


def test_get_layout_fallback() -> None:
    layout = get_layout("completely unknown probe X999")
    assert layout.name == "Neuropixels 1.0"


def test_get_layout_case_insensitive_prefix() -> None:
    layout = get_layout("neuropixels 1.0")
    assert layout.n_channels == 384


# ---------------------------------------------------------------------------
# channel_ccf_coords
# ---------------------------------------------------------------------------

def _vertical_probe():
    """A probe inserted along the DV axis from entry=(5400,5700,0) to tip=(5400,5700,3000)."""
    entry = np.array([5400.0, 5700.0, 0.0])
    tip = np.array([5400.0, 5700.0, 3000.0])
    return entry, tip


def test_channel_coords_vertical_probe_no_lateral() -> None:
    """Channels of a vertical probe should all have the same AP and ML."""
    entry, tip = _vertical_probe()
    depths = np.array([175.0, 195.0, 215.0, 235.0])  # 4 sites
    coords = channel_ccf_coords(entry, tip, depths)
    assert coords.shape == (4, 3)
    # AP and ML should be constant.
    np.testing.assert_allclose(coords[:, 0], 5400.0, atol=1e-6)  # AP
    np.testing.assert_allclose(coords[:, 1], 5700.0, atol=1e-6)  # ML
    # DV: tip is at 3000, so site at depth 175 from tip is at DV = 3000 - 175 = 2825
    np.testing.assert_allclose(coords[0, 2], 3000.0 - 175.0, atol=1e-6)
    np.testing.assert_allclose(coords[1, 2], 3000.0 - 195.0, atol=1e-6)


def test_channel_coords_deep_site_is_close_to_tip() -> None:
    """A site at depth 0 from the tip should be at the tip position."""
    entry, tip = _vertical_probe()
    depths = np.array([0.0])
    coords = channel_ccf_coords(entry, tip, depths)
    np.testing.assert_allclose(coords[0], tip, atol=1e-6)


def test_channel_coords_shallow_site_is_close_to_entry() -> None:
    """A site at depth=length from the tip should be at the entry."""
    entry, tip = _vertical_probe()
    length = float(np.linalg.norm(np.array(tip) - np.array(entry)))
    depths = np.array([length])
    coords = channel_ccf_coords(entry, tip, depths)
    np.testing.assert_allclose(coords[0], entry, atol=1e-6)


def test_channel_coords_degenerate_probe() -> None:
    """A zero-length probe should return the entry point for all channels."""
    entry = np.array([5400.0, 5700.0, 1000.0])
    tip = entry.copy()
    coords = channel_ccf_coords(entry, tip, np.array([0.0, 175.0]))
    assert coords.shape == (2, 3)
    np.testing.assert_allclose(coords[0], entry, atol=1e-6)
    np.testing.assert_allclose(coords[1], entry, atol=1e-6)


def test_channel_coords_oblique_probe() -> None:
    """An oblique probe: channel 0 (at tip) has the highest AP and DV."""
    entry = np.array([5000.0, 5700.0, 0.0])
    tip = np.array([5500.0, 5700.0, 2000.0])   # moves both AP and DV
    length = float(np.linalg.norm(tip - entry))
    # Use depths within the trajectory so we stay between entry and tip.
    depths = np.linspace(0.0, length, 100)
    coords = channel_ccf_coords(entry, tip, depths)
    assert coords.shape == (100, 3)
    # Channel 0 is at the tip (depth_from_tip=0) → highest AP and DV.
    assert coords[0, 0] > coords[-1, 0], "Tip channel should have largest AP"
    assert coords[0, 2] > coords[-1, 2], "Tip channel should have largest DV"


# ---------------------------------------------------------------------------
# Cross-check against legacy project_feature_points_to_ccf logic
# ---------------------------------------------------------------------------

def _legacy_single_channel_ccf(
    entry_aml: tuple[float, float, float],
    tip_aml: tuple[float, float, float],
    depth_from_surface: float,
    x_um_local: float,
) -> np.ndarray:
    """Reimplementation of the legacy probe_frame + project_feature_points_to_ccf
    for a single point, using (AP, ML, DV) input (our convention).

    This mirrors what probe_visualization.py does so we can cross-check.
    """
    from atlastrack.probes.geometry import SHANK_THICKNESS_UM

    # Legacy ccf array is (AP, ML, DV); legacy surface = ccf[0] = entry point.
    entry = np.array(entry_aml, dtype=float)  # (AP, ML, DV)
    tip = np.array(tip_aml, dtype=float)

    # probe_frame: axis from entry (surface) to tip.
    ccf_arr = np.stack([entry, tip])  # shape (2, 3): ccf[0]=entry, ccf[-1]=tip
    axis = ccf_arr[-1] - ccf_arr[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return entry.copy()
    axis = axis / axis_norm

    # rotation_face_normal_ap_ml(270°) → (0, -1, 0) in (AP, ML, DV)
    # theta = deg2rad(270 + 90) = deg2rad(360) = 0
    # normal = (-sin(0), -cos(0), 0) = (0, -1, 0)
    thick_ref = np.array([0.0, -1.0, 0.0])
    thick_vec = thick_ref - np.dot(thick_ref, axis) * axis
    thick_norm = float(np.linalg.norm(thick_vec))
    if thick_norm < 1e-6:
        thick_ref = np.array([0.0, 1.0, 0.0])
        thick_vec = thick_ref - np.dot(thick_ref, axis) * axis
        thick_norm = float(np.linalg.norm(thick_vec))
    thick_vec = thick_vec / thick_norm
    width_vec = np.cross(thick_vec, axis)
    width_vec = width_vec / float(np.linalg.norm(width_vec))

    depth = float(np.clip(depth_from_surface, 0.0, axis_norm))
    x_offset = x_um_local - ELECTRODE_COLUMN_CENTER_UM
    surface = ccf_arr[0].copy()
    ccf = (
        surface
        + axis * depth
        + width_vec * x_offset
        + thick_vec * (SHANK_THICKNESS_UM / 2.0 + 2.0)
    )
    return ccf


def test_crosscheck_legacy_channel_zero() -> None:
    """channel_ccf_coords at depth 0 (tip) matches legacy axial position within 1 µm.

    The legacy code adds a ~14 µm thick_vec transverse offset to project the site onto the
    probe face.  We skip that display artefact and return the shank-centreline position,
    so only the axial direction (AP and DV for a vertical probe) is cross-checked.
    """
    entry = (5400.0, 5700.0, 100.0)   # (AP, ML, DV) entry point
    tip = (5400.0, 5700.0, 2500.0)    # tip is deeper (larger DV)

    depth_from_surface = float(np.linalg.norm(np.array(tip) - np.array(entry)))
    x_local = ELECTRODE_COLUMN_CENTER_UM  # centred → zero lateral offset

    depths = np.array([0.0])
    our_coords = channel_ccf_coords(np.array(entry), np.array(tip), depths)
    legacy = _legacy_single_channel_ccf(entry, tip, depth_from_surface, x_local)

    # AP: should match exactly (no lateral component for vertical probe).
    assert our_coords[0, 0] == pytest.approx(legacy[0], abs=1.0)
    # DV: should match the axial depth (legacy adds ~14 µm thick offset in transverse direction,
    #     but for a vertical probe thick_vec is along ML so DV is unaffected).
    assert our_coords[0, 2] == pytest.approx(legacy[2], abs=1.0)


def test_crosscheck_legacy_midpoint() -> None:
    """Channel at half insertion depth matches legacy AP/DV within 2 µm.

    For an oblique probe the legacy thick_vec has an AP component (~1–2 µm),
    which explains the small tolerance.  ML is not checked here since the
    legacy thick_vec offset in ML can be ~14 µm for certain probe orientations.
    """
    entry = (5000.0, 5700.0, 0.0)
    tip = (5500.0, 5700.0, 2000.0)

    trajectory_len = float(np.linalg.norm(np.array(tip) - np.array(entry)))
    mid_depth_from_surface = trajectory_len / 2.0
    depth_from_tip = trajectory_len - mid_depth_from_surface
    x_local = ELECTRODE_COLUMN_CENTER_UM

    depths = np.array([depth_from_tip])
    our_coords = channel_ccf_coords(np.array(entry), np.array(tip), depths)
    legacy = _legacy_single_channel_ccf(entry, tip, mid_depth_from_surface, x_local)

    # AP and DV agree within 2 µm; ML not checked (thick_vec offset).
    assert our_coords[0, 0] == pytest.approx(legacy[0], abs=2.0)   # AP
    assert our_coords[0, 2] == pytest.approx(legacy[2], abs=2.0)   # DV


# ---------------------------------------------------------------------------
# shank_channel_coords + project_channel_coords
# ---------------------------------------------------------------------------

def _make_project_with_coords() -> Project:
    shank = Shank(
        index=0,
        tip_px=Point2D(x_px=50.0, y_px=70.0),
        tip_section_idx=0,
        tip_ccf_um=(5400.0, 5700.0, 3000.0),
        entry_px=Point2D(x_px=50.0, y_px=10.0),
        entry_section_idx=0,
        entry_ccf_um=(5400.0, 5700.0, 100.0),
    )
    probe = ProbeSpec(
        label="p1",
        type=ProbeType(name="Neuropixels 1.0", n_shanks=1),
        shanks=[shank],
    )
    return Project(atlas=AtlasRef(), slides=[], probes=[probe])


def test_shank_channel_coords_shape() -> None:
    project = _make_project_with_coords()
    shank = project.probes[0].shanks[0]
    layout = get_layout("Neuropixels 1.0")
    coords = shank_channel_coords(shank, layout)
    assert coords is not None
    assert coords.shape == (384, 3)


def test_shank_channel_coords_none_when_missing() -> None:
    shank = Shank(index=0)  # no CCF coords
    layout = get_layout("Neuropixels 1.0")
    assert shank_channel_coords(shank, layout) is None


def test_project_channel_coords_keys() -> None:
    project = _make_project_with_coords()
    result = project_channel_coords(project)
    assert ("p1", 0) in result
    assert result[("p1", 0)].shape[1] == 3


def test_channel_coords_dv_monotonic_np10() -> None:
    """For a vertical insertion, NP 1.0 channel DV should increase from tip to entry."""
    layout = get_layout("Neuropixels 1.0")
    entry = np.array([5400.0, 5700.0, 0.0])
    tip = np.array([5400.0, 5700.0, 3000.0])
    depths = layout.site_depths_from_tip_um()
    coords = channel_ccf_coords(entry, tip, depths)
    # Channel 0 is near the tip (large DV); last channel near entry (small DV).
    # depths[0] < depths[-1] → coords[0, DV] > coords[-1, DV] for downward probe.
    dv_values = coords[:, 2]
    assert dv_values[0] > dv_values[-1], "Deepest channel should have largest DV"


# ---------------------------------------------------------------------------
# export_channel_csv
# ---------------------------------------------------------------------------

def test_export_channel_csv(tmp_path: Path) -> None:
    project = _make_project_with_coords()
    out = tmp_path / "channels.csv"
    n = export_channel_csv(project, out)
    assert n == 384
    assert out.exists()
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 384
    assert set(rows[0].keys()) == {
        "probe", "shank", "channel", "ap_um", "ml_um", "dv_um", "depth_source"
    }
    assert rows[0]["probe"] == "p1"
    assert rows[0]["channel"] == "0"


def test_export_channel_csv_filters_probe(tmp_path: Path) -> None:
    project = _make_project_with_coords()
    # Add a second probe without coords → should produce 0 rows for it.
    shank2 = Shank(index=0)  # no CCF coords
    probe2 = ProbeSpec(
        label="p2",
        type=ProbeType(name="Neuropixels 1.0", n_shanks=1),
        shanks=[shank2],
    )
    project.probes.append(probe2)
    out = tmp_path / "filtered.csv"
    n = export_channel_csv(project, out, probe_label="p1")
    assert n == 384


# ---------------------------------------------------------------------------
# CCF -> Paxinos
# ---------------------------------------------------------------------------

def test_ccf_um_to_paxinos_mm_none_is_plain_mirror() -> None:
    # alignment="none" reduces to the plain mirror (no tilt, DV = raw CCF depth).
    ap, ml, dv = ccf_um_to_paxinos_mm(5400.0, 5700.0, 3000.0, alignment="none")
    assert float(ap) == pytest.approx(0.0)
    assert float(ml) == pytest.approx(0.0)
    assert float(dv) == pytest.approx(3.0)
    ap2, ml2, _ = ccf_um_to_paxinos_mm(4400.0, 4700.0, 0.0, alignment="none")
    assert float(ap2) == pytest.approx(1.0)
    assert float(ml2) == pytest.approx(1.0)


def test_ccf_um_to_paxinos_mm_bregma_maps_to_origin() -> None:
    # Bregma in CCF (5400, 5700, 440) -> Paxinos origin under a corrected alignment.
    ap, ml, dv = ccf_um_to_paxinos_mm(5400.0, 5700.0, 440.0, alignment="qiu2018")
    assert float(ap) == pytest.approx(0.0, abs=1e-9)
    assert float(ml) == pytest.approx(0.0, abs=1e-9)
    assert float(dv) == pytest.approx(0.0, abs=1e-9)


def test_ccf_um_to_paxinos_mm_pitch_shifts_ap() -> None:
    """The 5° pitch maps a point straight below bregma (CCF) to a non-zero AP."""
    # 1 mm below bregma in CCF DV (5400, 5700, 1440).
    ap_n, _, _ = ccf_um_to_paxinos_mm(5400.0, 5700.0, 1440.0, alignment="none")
    ap_q, _, _ = ccf_um_to_paxinos_mm(5400.0, 5700.0, 1440.0, alignment="qiu2018")
    assert float(ap_n) == pytest.approx(0.0)          # no tilt -> stays on the AP axis
    # 1 mm * sin(5°) * scale_ap ≈ 0.0899 mm anterior.
    assert float(ap_q) == pytest.approx(0.0899, abs=2e-3)
    assert float(ap_q) > 0.05


def test_export_paxinos_csv(tmp_path: Path) -> None:
    project = _make_project_with_coords()
    out = tmp_path / "paxinos.csv"
    n = export_paxinos_csv(project, out)
    assert n == 384
    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert set(rows[0].keys()) == {"probe", "shank", "channel", "ap_mm", "ml_mm", "dv_mm"}
    # The CSV is exactly the Paxinos transform of the per-channel CCF coords.
    coords = project_channel_coords(project)[("p1", 0)]
    ap, ml, dv = ccf_um_to_paxinos_mm(coords[0, 0], coords[0, 1], coords[0, 2])
    assert float(rows[0]["ap_mm"]) == pytest.approx(float(ap), abs=1e-3)
    assert float(rows[0]["ml_mm"]) == pytest.approx(float(ml), abs=1e-3)
    assert float(rows[0]["dv_mm"]) == pytest.approx(float(dv), abs=1e-3)
