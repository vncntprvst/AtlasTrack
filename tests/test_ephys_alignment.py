"""Tests for the pure ephys alignment math and LFP feature helpers."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.alignment import (
    apply_depth_alignment,
    channel_ccf_um,
    invert_anchors,
)
from histo_to_ccf.ephys.features import lfp_psd, power_image


# --- depth warp -------------------------------------------------------------

def test_no_anchors_is_identity() -> None:
    f = np.array([0.0, 100.0, 500.0])
    np.testing.assert_array_equal(apply_depth_alignment(f, []), f)


def test_single_anchor_is_pure_shift() -> None:
    f = np.array([0.0, 100.0, 500.0])
    out = apply_depth_alignment(f, [(100.0, 150.0)])  # shift +50
    np.testing.assert_allclose(out, f + 50.0)


def test_two_anchors_piecewise_linear_interpolates() -> None:
    # feature 0->0, 1000->2000  => slope 2 between them.
    out = apply_depth_alignment([0.0, 500.0, 1000.0], [(0.0, 0.0), (1000.0, 2000.0)])
    np.testing.assert_allclose(out, [0.0, 1000.0, 2000.0])


def test_extrapolation_uses_end_segment_slope() -> None:
    anchors = [(0.0, 0.0), (1000.0, 2000.0)]  # slope 2 everywhere
    # Below the first anchor and above the last extrapolate with the same slope.
    out = apply_depth_alignment([-100.0, 1500.0], anchors)
    np.testing.assert_allclose(out, [-200.0, 3000.0])


def test_invert_anchors_round_trips() -> None:
    anchors = [(0.0, 100.0), (1000.0, 1800.0)]
    inv = invert_anchors(anchors)
    f = np.array([0.0, 500.0, 1000.0])
    t = apply_depth_alignment(f, anchors)
    back = apply_depth_alignment(t, inv)
    np.testing.assert_allclose(back, f, atol=1e-9)


# --- channel -> CCF ---------------------------------------------------------

def test_channel_ccf_places_on_tip_entry_line() -> None:
    tip = (1000.0, 2000.0, 5000.0)   # (AP, ML, DV)
    entry = (1000.0, 2000.0, 1000.0)  # straight up in DV, 4000 µm insertion
    depths = [0.0, 2000.0, 4000.0]
    out = channel_ccf_um(tip, entry, depths, [])
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[0], tip)            # depth 0 = tip
    np.testing.assert_allclose(out[2], entry)          # depth = length = entry
    np.testing.assert_allclose(out[1], (1000.0, 2000.0, 3000.0))  # halfway


def test_channel_ccf_anchor_warps_position() -> None:
    tip = (0.0, 0.0, 4000.0)
    entry = (0.0, 0.0, 0.0)  # 4000 µm up
    # Anchor says feature depth 1000 actually sits at track depth 2000.
    out = channel_ccf_um(tip, entry, [1000.0], [(1000.0, 2000.0)])
    # track 2000 along a 4000 line from DV 4000 -> 0 lands at DV 2000.
    np.testing.assert_allclose(out[0], (0.0, 0.0, 2000.0))


def test_channel_ccf_zero_length_track_returns_tip() -> None:
    tip = (1.0, 2.0, 3.0)
    out = channel_ccf_um(tip, tip, [0.0, 100.0], [])
    np.testing.assert_allclose(out, [tip, tip])


# --- LFP features -----------------------------------------------------------

def test_lfp_psd_peaks_at_injected_frequency() -> None:
    fs = 2500.0
    t = np.arange(0, 4.0, 1 / fs)
    # Two channels, sine waves at 10 Hz and 50 Hz.
    traces = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 50 * t)], axis=1)
    freqs, psd = lfp_psd(traces, fs, fmin=0, fmax=120)
    assert psd.shape[0] == 2
    assert freqs[np.argmax(psd[0])] == pytest.approx(10.0, abs=2.0)
    assert freqs[np.argmax(psd[1])] == pytest.approx(50.0, abs=2.0)


def test_power_image_is_uint8_full_range() -> None:
    psd = np.array([[1.0, 10.0], [100.0, 1000.0]])
    img = power_image(psd)
    assert img.dtype == np.uint8
    assert img.min() == 0 and img.max() == 255
    assert img.shape == psd.shape


# --- region strip -----------------------------------------------------------

class _RegionAtlas:
    """Fake atlas: region acronym from the DV index, with an RGB table."""

    def __init__(self) -> None:
        self.structures = {
            "A": {"rgb_triplet": [10, 20, 30]},
            "B": {"rgb_triplet": [40, 50, 60]},
        }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        ap, dv, ml = coords
        if dv < 0 or dv > 4000:
            return "Outside atlas"
        return "A" if dv < 2000 else "B"


def test_regions_at_ccf_returns_acronym_and_rgb() -> None:
    from histo_to_ccf.ephys.regions import regions_at_ccf

    atlas = _RegionAtlas()
    pts = [(0.0, 0.0, 1000.0), (0.0, 0.0, 3000.0), (0.0, 0.0, 9999.0)]
    hits = regions_at_ccf(atlas, pts)
    assert [h[0] for h in hits] == ["A", "B", ""]
    assert hits[0][1] == (10, 20, 30)
    assert hits[2][1] == (0, 0, 0)  # outside -> black


def test_region_strip_image_shape_and_top_bottom() -> None:
    from histo_to_ccf.ephys.regions import region_strip_image

    hits = [("A", (10, 20, 30)), ("B", (40, 50, 60))]
    img = region_strip_image(hits, height=10, width=4)
    assert img.shape == (10, 4, 3)
    np.testing.assert_array_equal(img[0, 0], (10, 20, 30))   # top = first hit
    np.testing.assert_array_equal(img[-1, 0], (40, 50, 60))  # bottom = last hit


# --- schema round-trip ------------------------------------------------------

def test_ephys_alignment_persists(tmp_path) -> None:
    from histo_to_ccf.project.io import load_project, save_project
    from histo_to_ccf.project.schema import (
        EphysAlignment,
        Project,
        ProbeSpec,
        ProbeType,
        Shank,
    )

    shank = Shank(
        index=0,
        tip_ccf_um=(1000.0, 2000.0, 5000.0),
        entry_ccf_um=(1000.0, 2000.0, 1000.0),
        ephys=EphysAlignment(
            recording_path="rec",
            stream_name="ProbeA-LFP",
            channel_depths_um=[0.0, 100.0],
            anchors=[(100.0, 150.0)],
            channel_ccf_um=[(1000.0, 2000.0, 5000.0), (1000.0, 2000.0, 4900.0)],
        ),
    )
    project = Project(
        probes=[ProbeSpec(label="p", type=ProbeType(name="NP", n_shanks=1), shanks=[shank])]
    )
    path = tmp_path / "p.histo2ccf.json"
    save_project(project, path)
    loaded = load_project(path)
    eph = loaded.probes[0].shanks[0].ephys
    assert eph is not None
    assert eph.stream_name == "ProbeA-LFP"
    assert eph.anchors == [(100.0, 150.0)]
    assert len(eph.channel_ccf_um) == 2
