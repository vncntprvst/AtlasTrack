"""The Ephys tab's join between attached recordings and computed features.

The bug these cover: discovery attached ``LO_07_004`` (all four shanks, one bank) and
``LO_07_005`` (one shank, whole track), and Compute still read the single path in the
box - so shank 0 got features and shanks 1-3 came back blank.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.combine import RecordingFeatures, stack_penetration
from atlastrack.project.schema import (
    EphysRecordingRef,
    ProbeSpec,
    ProbeType,
    Shank,
)

pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

FREQS = np.linspace(0.0, 300.0, 16)


def _state_with_probe(n_shanks=4):
    from atlastrack.gui.workflow import WorkflowState

    state = WorkflowState()
    state.project.probes.append(ProbeSpec(
        label="ProbeA", type=ProbeType(name="NP2.0", n_shanks=n_shanks),
        shanks=[Shank(index=i, tip_ccf_um=(5000.0, 2000.0 + 250.0 * i, 6000.0),
                      entry_ccf_um=(5000.0, 2000.0 + 250.0 * i, 1000.0))
                for i in range(n_shanks)],
    ))
    return state


def _panel(qtbot, state):
    import napari

    from atlastrack.gui.widgets.ephys_panel import EphysPanelWidget

    viewer = napari.Viewer(show=False)
    panel = EphysPanelWidget(state, viewer)
    qtbot.addWidget(panel)
    panel.refresh_probes()
    return panel, viewer


def _rec(label, *, shanks, n_rows, columns=2, insertion=4976.0):
    xs, ys = [], []
    for s in shanks:
        for r in range(n_rows):
            for c in range(columns):
                xs.append(250.0 * s + 32.0 * c)
                ys.append(15.0 * r)
    y = np.asarray(ys, dtype=float)
    return RecordingFeatures(
        label=label, stream_name=f"{label}.ProbeA", insertion_depth_um=insertion,
        freqs_hz=FREQS, psd=np.ones((y.size, FREQS.size)),
        axial_um=y, x_um=np.asarray(xs, dtype=float),
    )


def _lo07_result():
    """The real shape of the case: 004 on all shanks, 005 on shank 0 only."""
    recs = [_rec("LO_07_004", shanks=(0, 1, 2, 3), n_rows=48),
            _rec("LO_07_005", shanks=(0,), n_rows=384, columns=1)]
    return {"stacks": stack_penetration(recs, [0, 1, 2, 3]),
            "recordings": recs, "failed": []}


def test_compute_uses_the_attached_recordings_not_the_path_box(qtbot, monkeypatch):
    state = _state_with_probe()
    probe = state.project.probes[0]
    probe.recordings = [
        EphysRecordingRef(path="F:/x/004", label="LO_07_004", insertion_depth_um=4976.0),
        EphysRecordingRef(path="F:/x/005", label="LO_07_005", insertion_depth_um=4976.0),
    ]
    panel, viewer = _panel(qtbot, state)
    try:
        panel._path_edit.setText("F:/some/other/Record Node 104")
        seen = {}

        def fake_multi(refs, shanks, **kw):
            seen["refs"] = list(refs)
            seen["shanks"] = list(shanks)
            raise RuntimeError("stop here")

        monkeypatch.setattr(
            "atlastrack.gui.workers.multi_lfp_power_worker", fake_multi
        )
        with pytest.raises(RuntimeError, match="stop here"):
            panel._compute()

        assert [r.label for r in seen["refs"]] == ["LO_07_004", "LO_07_005"]
        assert seen["shanks"] == [0, 1, 2, 3]
    finally:
        viewer.close()


def test_every_shank_gets_features_from_whichever_recording_reached_it(qtbot):
    state = _state_with_probe()
    panel, viewer = _panel(qtbot, state)
    try:
        panel._on_multi_computed(_lo07_result())

        features = panel._stack_features()
        assert sorted(features) == [0, 1, 2, 3], "shanks 1-3 were blank before"
        assert features[0].lfp_psd.shape[0] > 300      # the whole column
        assert 40 < features[1].lfp_psd.shape[0] < 60  # one bank
        for shank in (1, 2, 3):
            assert np.isfinite(features[shank].lfp_psd).any()
    finally:
        viewer.close()


def test_the_status_names_the_recordings_and_the_reach_of_each_shank(qtbot):
    state = _state_with_probe()
    panel, viewer = _panel(qtbot, state)
    try:
        panel._on_multi_computed(_lo07_result())

        text = panel._status.text()
        assert "LO_07_004" in text and "LO_07_005" in text
        assert "shank 3" in text
    finally:
        viewer.close()


def test_a_recording_that_failed_to_read_is_reported_not_swallowed(qtbot):
    state = _state_with_probe()
    panel, viewer = _panel(qtbot, state)
    try:
        result = _lo07_result()
        result["failed"] = [("LO_07_003", "every candidate window was artifact-dominated")]
        panel._on_multi_computed(result)

        assert "LO_07_003" in panel._status.text()
        assert "artifact-dominated" in panel._status.text()
    finally:
        viewer.close()


def test_nothing_readable_says_so_rather_than_looking_computed(qtbot):
    state = _state_with_probe()
    panel, viewer = _panel(qtbot, state)
    try:
        panel._on_multi_computed(
            {"stacks": {}, "recordings": [], "failed": [("LO_07_004", "disk gone")]}
        )

        assert "No shank got any data" in panel._status.text()
        assert "disk gone" in panel._status.text()
        assert panel._stack_features() == {}
    finally:
        viewer.close()


def test_saving_exports_every_stacked_shank(qtbot):
    state = _state_with_probe()
    panel, viewer = _panel(qtbot, state)
    try:
        panel._on_multi_computed(_lo07_result())
        exports = panel._computed_exports(state.project.probes[0])

        assert [e.shank_index for e in exports] == [0, 1, 2, 3]
        for e in exports:
            below = np.asarray(e.channel_depth_below_surface_um)
            from_tip = np.asarray(e.channel_depth_from_tip_um)
            # The two conventions must stay consistent; mixing them is the recurring
            # bug this export records both to prevent.
            assert np.allclose(below, e.track_length_um - from_tip)
    finally:
        viewer.close()


def test_mixed_insertion_depths_with_one_missing_are_refused(qtbot, monkeypatch):
    """Without every depth the recordings cannot be placed relative to each other."""
    from atlastrack.gui.widgets import ephys_panel as mod

    state = _state_with_probe()
    state.project.probes[0].recordings = [
        EphysRecordingRef(path="F:/x/001", label="a", insertion_depth_um=4576.0),
        EphysRecordingRef(path="F:/x/003", label="b", insertion_depth_um=0.0),
    ]
    panel, viewer = _panel(qtbot, state)
    try:
        warned = {}
        monkeypatch.setattr(mod.QMessageBox, "warning",
                            lambda *a, **k: warned.setdefault("msg", a[2]))
        monkeypatch.setattr(
            "atlastrack.gui.workers.multi_lfp_power_worker",
            lambda *a, **k: pytest.fail("must not compute with a missing depth"),
        )
        panel._compute()

        assert "insertion depth" in warned["msg"].lower()
        assert "b" in warned["msg"]
    finally:
        viewer.close()


def test_the_button_says_which_recordings_it_will_read(qtbot):
    state = _state_with_probe()
    panel, viewer = _panel(qtbot, state)
    try:
        assert "attached" not in panel._compute_btn.text()

        state.project.probes[0].recordings = [
            EphysRecordingRef(path="F:/x/004", label="LO_07_004",
                              insertion_depth_um=4976.0),
        ]
        panel.refresh_compute_button()

        assert panel._compute_btn.text() == "Compute features from 1 attached recording"
        assert "LO_07_004" in panel._compute_btn.toolTip()
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Choosing a probe map
#
# Intan recordings carry no geometry, so the map is the difference between real
# depths and a 32-channel probe reporting a span of "31". The control has to be
# reachable from the panel or the format is only supported on paper.
# ---------------------------------------------------------------------------


def _map_entries(panel):
    return [panel._map_combo.itemText(i) for i in range(panel._map_combo.count())]


def _select(panel, text):
    index = panel._map_combo.findText(text)
    assert index >= 0, f"{text!r} not offered; have {_map_entries(panel)}"
    panel._map_combo.setCurrentIndex(index)
    return index


class _Signal:
    def connect(self, *_args, **_kwargs) -> None:
        return None


class _FakeWorker:
    yielded = _Signal()
    returned = _Signal()
    errored = _Signal()

    def start(self) -> None:
        return None


def _fake_worker(*_args, **_kwargs) -> _FakeWorker:
    return _FakeWorker()


def test_the_default_is_to_take_geometry_from_the_recording(qtbot):
    """Open Ephys and SpikeGLX store the probe; overriding that would be a step back."""
    from atlastrack.gui.widgets.ephys_panel import MAP_FROM_RECORDING

    panel, _ = _panel(qtbot, _state_with_probe())

    assert panel._map_combo.currentText() == MAP_FROM_RECORDING
    assert panel._selected_probe_map() is None


def test_the_wired_poly3_map_is_offered_and_resolves_to_real_micrometres(qtbot):
    from atlastrack.ephys.probemap import (
        NEURONEXUS_POLY3_A32_RHD2132,
        resolve_probe_map,
    )

    panel, _ = _panel(qtbot, _state_with_probe())
    _select(panel, NEURONEXUS_POLY3_A32_RHD2132)

    assert panel._selected_probe_map() == NEURONEXUS_POLY3_A32_RHD2132
    resolved = resolve_probe_map(panel._selected_probe_map(), n_channels=32)
    assert resolved.extent_um == pytest.approx(275.0)


def test_the_file_browser_entry_is_not_itself_a_probe_map(qtbot):
    """Selecting it opens a dialog; it must never be handed on as a map name."""
    from atlastrack.gui.widgets.ephys_panel import MAP_CHOOSE_FILE

    panel, _ = _panel(qtbot, _state_with_probe())
    index = panel._map_combo.findText(MAP_CHOOSE_FILE)

    assert index >= 0
    assert panel._map_combo.itemData(index) == MAP_CHOOSE_FILE
    assert panel._selected_probe_map() is None


def test_a_catalog_layout_is_offered_but_labelled_as_layout_only(qtbot):
    """It says where the sites are, not which channel each one is on."""
    from atlastrack.probes.catalog import NEURONEXUS_A1X32_POLY3

    panel, _ = _panel(qtbot, _state_with_probe())
    texts = _map_entries(panel)
    entry = next(t for t in texts if t.startswith(NEURONEXUS_A1X32_POLY3))

    assert "layout only" in entry
    _select(panel, entry)
    assert panel._selected_probe_map() == NEURONEXUS_A1X32_POLY3


def test_the_chosen_map_reaches_recordings_that_have_none(qtbot, monkeypatch):
    """The panel's choice is the penetration's default, applied at compute time."""
    from atlastrack.ephys.probemap import NEURONEXUS_POLY3_A32_RHD2132
    from atlastrack.gui import workers

    state = _state_with_probe(n_shanks=1)
    probe = state.project.probes[0]
    probe.recordings = [
        EphysRecordingRef(path="a", label="A", insertion_depth_um=5028.0),
        EphysRecordingRef(path="b", label="B", insertion_depth_um=5028.0,
                          probe_map="already/set.csv"),
    ]
    monkeypatch.setattr(workers, "multi_lfp_power_worker", _fake_worker)

    panel, _ = _panel(qtbot, state)
    _select(panel, NEURONEXUS_POLY3_A32_RHD2132)
    panel._compute_multi()

    assert probe.recordings[0].probe_map == NEURONEXUS_POLY3_A32_RHD2132
    # A per-recording choice already made is not overwritten by the panel default.
    assert probe.recordings[1].probe_map == "already/set.csv"
