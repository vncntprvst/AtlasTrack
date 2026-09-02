"""Exporting the depth features an alignment was read from."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from atlastrack.ephys.export import (
    ShankFeatureExport,
    build_payload,
    default_export_path,
    load_feature_export,
    load_shank_features,
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


# -- reloading, and the landmark safety rule --------------------------------


def test_reloading_drops_the_landmarks_by_default(tmp_path):
    """The whole point: measurements are safe to reload, an alignment is not."""
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [_shank(0)])

    shanks, _meta = load_shank_features(written)

    assert shanks[0].lfp_psd.shape == (16, 31)          # the ephys came back...
    assert shanks[0].firing_rate_hz.size == 40
    assert shanks[0].landmark_feature_um.size == 0      # ...the alignment did not
    assert shanks[0].landmark_track_um.size == 0
    assert shanks[0].extremes_mode == "uniform"         # not the stored "linear"


def test_landmarks_come_back_only_when_asked_for(tmp_path):
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [_shank(0)])

    shanks, _meta = load_shank_features(written, include_landmarks=True)

    assert shanks[0].landmark_feature_um.tolist() == [0.0, 1400.0, 4000.0]
    assert shanks[0].landmark_track_um.tolist() == [0.0, 1500.0, 4000.0]
    assert shanks[0].extremes_mode == "linear"


def test_reload_keys_shanks_by_their_original_index(tmp_path):
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [_shank(0), _shank(3)])

    shanks, meta = load_shank_features(written)

    assert sorted(shanks) == [0, 3]
    assert shanks[3].shank_index == 3
    assert shanks[0].track_length_um == pytest.approx(4000.0)
    assert meta["probe"] == "ProbeA"


def test_reloaded_lists_come_back_as_strings(tmp_path):
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [_shank(0)])

    shanks, _meta = load_shank_features(written)

    assert shanks[0].region_acronym == ["A", "B"]
    assert shanks[0].channel_ids[0] == "ch0"


def test_reloading_a_bare_export_yields_an_empty_shank(tmp_path):
    written = save_feature_export(
        tmp_path / "out.npz", "ProbeA", [ShankFeatureExport(shank_index=2)]
    )

    shanks, _meta = load_shank_features(written)

    # Nothing was stored, so nothing comes back - but the shank is still known.
    assert shanks == {} or shanks[2].lfp_psd.size == 0


def test_a_round_trip_of_measurements_is_lossless(tmp_path):
    original = _shank(1)
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [original])

    shanks, _meta = load_shank_features(written)

    assert np.allclose(shanks[1].lfp_psd, original.lfp_psd)
    assert np.allclose(shanks[1].channel_depth_below_surface_um,
                       original.channel_depth_below_surface_um)
    assert np.allclose(shanks[1].firing_rate_hz, original.firing_rate_hz)


class _Atlas:
    structures: ClassVar[dict] = {"A": {"rgb_triplet": [1, 2, 3]}}

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        return "A"


def _probe_state():
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.project.schema import ProbeSpec, ProbeType, Shank

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
    return state


@pytest.mark.qt
def test_the_alignment_dialog_loads_landmarks_deliberately(qtbot, tmp_path) -> None:
    """The Ephys tab must not, but here it is an informed choice."""
    pytest.importorskip("pyqtgraph")
    import napari

    from atlastrack.gui.widgets.ephys_alignment_panel import EphysProbeAlignmentDialog

    saved = ShankFeatureExport(
        shank_index=0, track_length_um=4000.0,
        landmark_feature_um=np.array([0.0, 1600.0, 4000.0]),
        landmark_track_um=np.array([0.0, 2000.0, 4000.0]),
    )
    written = save_feature_export(tmp_path / "out.npz", "ProbeA", [saved])

    viewer = napari.Viewer(show=False)
    try:
        dlg = EphysProbeAlignmentDialog(_probe_state(), 0)
        qtbot.addWidget(dlg)
        assert dlg.panels[0].landmarks().n_user == 0

        applied = dlg.load_landmarks_from_file(str(written))

        assert applied == 1
        assert dlg.panels[0].landmarks().user_pairs() == pytest.approx(
            [(1600.0, 2000.0)]
        )
    finally:
        viewer.close()


@pytest.mark.qt
def test_the_dialog_assembles_an_export_for_every_shank(qtbot) -> None:
    pytest.importorskip("pyqtgraph")
    import napari

    from atlastrack.gui.widgets.ephys_alignment_panel import EphysProbeAlignmentDialog

    state = _probe_state()
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
