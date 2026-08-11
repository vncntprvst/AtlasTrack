"""Exporting the depth features an alignment was read from."""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.export import (
    ShankFeatureExport,
    build_payload,
    default_export_path,
    load_feature_export,
    save_feature_export,
)


def _shank(index: int = 0) -> ShankFeatureExport:
    return ShankFeatureExport(
        shank_index=index,
        track_length_um=4000.0,
        lfp_freqs_hz=np.linspace(0.0, 300.0, 31),
        lfp_psd=np.random.default_rng(index).random((16, 31)),
        channel_depth_from_tip_um=np.linspace(0.0, 3000.0, 16),
        channel_depth_below_surface_um=np.linspace(4000.0, 1000.0, 16),
        channel_ids=[f"ch{i}" for i in range(16)],
        profile_depth_um=np.linspace(0.0, 4000.0, 40),
        firing_rate_hz=np.linspace(1.0, 20.0, 40),
        mean_amplitude=np.linspace(-100.0, -50.0, 40),
        region_top_um=np.array([0.0, 1500.0]),
        region_bottom_um=np.array([1500.0, 4000.0]),
        region_acronym=["A", "B"],
        landmark_feature_um=np.array([0.0, 1400.0, 4000.0]),
        landmark_track_um=np.array([0.0, 1500.0, 4000.0]),
        extremes_mode="linear",
    )


# -- where it lands ---------------------------------------------------------


def test_default_path_sits_beside_the_project_file(tmp_path):
    project = tmp_path / "LO_07" / "LO_07_green_whole.json"

    path = default_export_path(project, "ProbeA")

    assert path.parent.parent == project.parent
    assert path.parent.name == "LO_07_green_whole_ephys_features"
    assert path.name == "ProbeA_depth_features.npz"


def test_probe_labels_with_awkward_characters_are_slugged(tmp_path):
    path = default_export_path(tmp_path / "p.json", "Probe A/2 (red)")

    assert "/" not in path.name
    assert path.name.endswith("_depth_features.npz")


def test_no_project_path_still_yields_a_usable_suggestion():
    path = default_export_path(None, "ProbeA")

    assert path.name == "ProbeA_depth_features.npz"


# -- what it contains -------------------------------------------------------


def test_payload_carries_every_feature_family():
    payload = build_payload("ProbeA", [_shank(0)])

    for name in ("lfp_psd", "lfp_freqs_hz", "profile_depth_um", "firing_rate_hz",
                 "region_acronym", "landmark_feature_um"):
        assert f"shank0_{name}" in payload, name
    assert "meta" in payload


def test_both_depth_conventions_are_recorded():
    """Every mix-up in this codebase came from one being read as the other."""
    payload = build_payload("ProbeA", [_shank(0)])

    assert "shank0_channel_depth_from_tip_um" in payload
    assert "shank0_channel_depth_below_surface_um" in payload


def test_empty_arrays_are_omitted_rather_than_stored_as_zeros():
    bare = ShankFeatureExport(shank_index=1, track_length_um=4000.0)

    payload = build_payload("ProbeA", [bare])

    assert not any(k.startswith("shank1_lfp") for k in payload)
    assert "meta" in payload


def test_multiple_shanks_are_keyed_separately():
    payload = build_payload("ProbeA", [_shank(0), _shank(3)])

    assert "shank0_lfp_psd" in payload
    assert "shank3_lfp_psd" in payload
    assert not np.array_equal(payload["shank0_lfp_psd"], payload["shank3_lfp_psd"])


# -- round trip -------------------------------------------------------------


def test_round_trip_preserves_arrays_and_metadata(tmp_path):
    shank = _shank(2)

    written = save_feature_export(tmp_path / "out.npz", "ProbeB", [shank])
    arrays, meta = load_feature_export(written)

    assert written.exists()
    assert np.allclose(arrays["shank2_lfp_psd"], shank.lfp_psd)
    assert arrays["shank2_region_acronym"].tolist() == ["A", "B"]
    assert meta["probe"] == "ProbeB"
    assert meta["shanks"][0]["index"] == 2
    assert meta["shanks"][0]["extremes_mode"] == "linear"
    assert meta["shanks"][0]["n_landmarks"] == 1  # 3 entries minus the two ends


def test_the_folder_is_created_if_missing(tmp_path):
    target = tmp_path / "nested" / "deeper" / "out.npz"

    written = save_feature_export(target, "ProbeA", [_shank(0)])

    assert written.exists()


def test_the_archive_is_actually_compressed(tmp_path):
    """Lightweight was the requirement, so check it rather than assume."""
    big = _shank(0)
    # Highly compressible: a constant block is the clearest test that it is deflated.
    big.lfp_psd = np.ones((384, 301))

    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [big])

    raw_bytes = big.lfp_psd.nbytes  # 384*301*8 ≈ 925 kB
    assert written.stat().st_size < 0.1 * raw_bytes


def test_load_rejects_pickled_objects(tmp_path):
    """allow_pickle stays off - an export must never be able to execute code."""
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [_shank(0)])

    arrays, _meta = load_feature_export(written)

    assert all(a.dtype != object for a in arrays.values())


def test_a_bare_export_still_round_trips(tmp_path):
    written = save_feature_export(
        tmp_path / "out.npz", "ProbeA", [ShankFeatureExport(shank_index=0)]
    )

    _arrays, meta = load_feature_export(written)

    assert meta["shanks"][0]["n_channels"] == 0
    assert meta["shanks"][0]["n_landmarks"] == 0


def test_missing_npz_suffix_is_reported_as_written(tmp_path):
    written = save_feature_export(tmp_path / "out", "ProbeA", [_shank(0)])

    assert written.suffix == ".npz"
    assert written.exists()


@pytest.mark.qt
def test_the_dialog_assembles_an_export_for_every_shank(qtbot) -> None:
    pytest.importorskip("pyqtgraph")
    import napari

    from histo_to_ccf.gui.widgets.ephys_alignment_panel import EphysProbeAlignmentDialog
    from histo_to_ccf.gui.workflow import WorkflowState
    from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Shank

    class _Atlas:
        structures = {"A": {"rgb_triplet": [1, 2, 3]}}  # noqa: RUF012

        def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
            return "A"

    state = WorkflowState()
    state.atlas = _Atlas()
    state.project.probes.append(
        ProbeSpec(
            label="ProbeA", type=ProbeType(name="NP", n_shanks=2),
            shanks=[
                Shank(index=i, tip_ccf_um=(5000.0, 2000.0, 5000.0),
                      entry_ccf_um=(5000.0, 2000.0, 1000.0))
                for i in range(2)
            ],
        )
    )
    viewer = napari.Viewer(show=False)
    try:
        dlg = EphysProbeAlignmentDialog(state, 0)
        qtbot.addWidget(dlg)
        dlg.panels[0].add_landmark_at(2000.0)
        dlg.panels[0].move_landmark(0, 1800.0, slot=0)
        dlg.panels[0].align()

        exports = dlg.feature_exports()

        assert [e.shank_index for e in exports] == [0, 1]
        assert exports[0].landmark_feature_um.size == 3  # two ends + one landmark
        assert exports[0].region_acronym  # the atlas column made it in
        assert exports[0].track_length_um == pytest.approx(4000.0)
    finally:
        viewer.close()
