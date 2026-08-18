"""Saving and loading features straight from the Ephys tab."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Shank

pytestmark = pytest.mark.qt

N_CH = 16


def _state():
    from histo_to_ccf.gui.workflow import WorkflowState

    state = WorkflowState()
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


def _lfp_result():
    """Two shanks' worth of channels, tagged by shank id as a real recording is."""
    y = np.tile(np.linspace(0.0, 750.0, N_CH), 2)
    return {
        "depths_um": y,
        "psd": np.random.default_rng(0).random((2 * N_CH, 24)),
        "freqs": np.linspace(0.0, 300.0, 24),
        "shank_ids": np.array(["0"] * N_CH + ["1"] * N_CH),
        "stream_name": "ProbeA-AP",
    }


def _panel(qtbot):
    import napari

    from histo_to_ccf.gui.widgets.ephys_panel import EphysPanelWidget

    viewer = napari.Viewer(show=False)
    widget = EphysPanelWidget(_state(), viewer)
    qtbot.addWidget(widget)
    return widget, viewer


def test_computed_exports_split_by_shank_id(qtbot) -> None:
    """By shank_ids, never by x: a NP2.0 shank has two electrode columns."""
    widget, viewer = _panel(qtbot)
    try:
        widget._lfp_result = _lfp_result()

        exports = widget._computed_exports(widget._state.project.probes[0])

        assert [e.shank_index for e in exports] == [0, 1]
        assert all(e.lfp_psd.shape == (N_CH, 24) for e in exports)
        assert all(e.track_length_um == pytest.approx(4000.0) for e in exports)
    finally:
        viewer.close()


def test_exports_carry_both_depth_conventions_and_the_tip_offset(qtbot) -> None:
    widget, viewer = _panel(qtbot)
    try:
        widget._lfp_result = _lfp_result()

        first = widget._computed_exports(widget._state.project.probes[0])[0]

        # The lowest channel sits 175 µm above the physical tip, not on it.
        assert first.channel_depth_from_tip_um.min() == pytest.approx(175.0)
        assert np.allclose(
            first.channel_depth_below_surface_um,
            4000.0 - first.channel_depth_from_tip_um,
        )
    finally:
        viewer.close()


def test_saving_computed_features_round_trips(qtbot, tmp_path) -> None:
    from histo_to_ccf.ephys.export import load_shank_features, save_feature_export

    widget, viewer = _panel(qtbot)
    try:
        widget._lfp_result = _lfp_result()
        exports = widget._computed_exports(widget._state.project.probes[0])
        written = save_feature_export(tmp_path / "out.npz", "ProbeA", exports)

        shanks, meta = load_shank_features(written)

        assert sorted(shanks) == [0, 1]
        assert meta["probe"] == "ProbeA"
        # No landmarks exist at this stage, so this file can never disturb an
        # alignment when it is reloaded.
        assert all(s.get("n_landmarks", 0) == 0 for s in meta["shanks"])
    finally:
        viewer.close()


def test_nothing_computed_yields_no_exports(qtbot) -> None:
    widget, viewer = _panel(qtbot)
    try:
        assert widget._computed_exports(widget._state.project.probes[0]) == []
    finally:
        viewer.close()


def test_the_io_buttons_exist_and_are_plainly_named(qtbot) -> None:
    widget, viewer = _panel(qtbot)
    try:
        assert widget._load_btn.text() == "Load saved features"
        assert widget._save_btn.text() == "Save computed features"
        assert widget._align_btn.text() == "Open alignment"
    finally:
        viewer.close()
