"""The discovery dialog: coverage as the thing on screen, and what Add writes."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from atlastrack.ephys.discovery import (
    Penetration,
    RecordingCandidate,
    ShankCoverage,
    StreamInfo,
)
from atlastrack.project.schema import ProbeSpec, ProbeType, Shank

pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _stream(recording, probe, coverage) -> StreamInfo:
    return StreamInfo(
        recording_dir=f"F:/TJO/LO_07/2026_05_08/{recording}/raw_ephys_data/Record Node 104",
        stream_name=f"Record Node 104#Neuropix-PXI-100.{probe}",
        probe_label=probe, n_channels=384, sampling_rate_hz=30000.0, duration_s=900.0,
        coverage=tuple(coverage), subject="LO_07", session_date=date(2026, 5, 8),
        recording_label=recording,
    )


def _bank(shank, top=0.0, bottom=705.0, n=96, cols=2) -> ShankCoverage:
    return ShankCoverage(shank, n, top, bottom, 15.0, cols)


def _cand(recording, probe, coverage, depth=None) -> RecordingCandidate:
    return RecordingCandidate(
        stream=_stream(recording, probe, coverage), insertion_depth_um=depth,
        dye="green", depth_source="sidecar" if depth is not None else "unknown",
    )


def _lo07_probe_a() -> Penetration:
    """The real shape: four bank recordings at two depths plus one deep single shank."""
    return Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("LO_07_001", "ProbeA", [_bank(i) for i in range(4)], 4576.0),
        _cand("LO_07_002", "ProbeA", [_bank(i) for i in range(4)], 4576.0),
        _cand("LO_07_003", "ProbeA", [_bank(i) for i in range(4)], 4976.0),
        _cand("LO_07_004", "ProbeA", [_bank(i) for i in range(4)], 4976.0),
        _cand("LO_07_005", "ProbeA", [ShankCoverage(0, 384, 0.0, 5745.0, 15.0, 2)],
              4976.0),
    ])


def _state_with_probe(label="ProbeA", n_shanks=4):
    from atlastrack.gui.workflow import WorkflowState

    state = WorkflowState()
    state.project.probes.append(
        ProbeSpec(
            label=label, type=ProbeType(name="NP2.0", n_shanks=n_shanks),
            shanks=[Shank(index=i, tip_ccf_um=(5000.0, 2000.0 + 250.0 * i, 6000.0),
                          entry_ccf_um=(5000.0, 2000.0 + 250.0 * i, 1000.0))
                    for i in range(n_shanks)],
        )
    )
    return state


def _dialog(qtbot, state=None, pens=None):
    from atlastrack.gui.widgets.ephys_discovery_dialog import EphysDiscoveryDialog

    dlg = EphysDiscoveryDialog(state or _state_with_probe())
    qtbot.addWidget(dlg)
    if pens is not None:
        dlg.set_penetrations(pens)
    return dlg


# ------------------------------------------------------------------ populating


def test_penetrations_populate_the_combo_and_table(qtbot):
    dlg = _dialog(qtbot, pens=[_lo07_probe_a()])

    assert dlg._pen_combo.count() == 1
    assert "ProbeA" in dlg._pen_combo.itemText(0)
    assert dlg._table.rowCount() == 5
    assert dlg._table.item(4, 1).text() == "LO_07_005"


def test_every_recording_starts_ticked(qtbot):
    dlg = _dialog(qtbot, pens=[_lo07_probe_a()])

    assert len(dlg.ticked()) == 5


def test_an_empty_scan_says_so_and_disables_add(qtbot):
    dlg = _dialog(qtbot, pens=[])

    assert dlg._table.rowCount() == 0
    assert dlg._add_btn.isEnabled() is False
    assert "No Open Ephys recordings" in dlg._status.text()


def test_the_deep_recording_is_described_as_one_shank(qtbot):
    dlg = _dialog(qtbot, pens=[_lo07_probe_a()])

    assert dlg._table.item(0, 2).text() == "all 4 shanks, 0-705 µm from tip"
    assert dlg._table.item(4, 2).text() == "shank 0, 0-5745 µm from tip"


# -------------------------------------------------------------------- coverage


def test_coverage_is_what_the_plot_shows_not_a_file_count(qtbot):
    """Ticking the shallower pair is what turns 705 µm of shank 1 into 1105."""
    from atlastrack.ephys.discovery import coverage_from_tip

    pen = _lo07_probe_a()
    dlg = _dialog(qtbot, pens=[pen])
    dlg.refresh_coverage()

    view = dlg.selected_penetration()
    assert coverage_from_tip(view, 0) == [(0.0, 5745.0)]
    assert coverage_from_tip(view, 1) == [(0.0, 1105.0)]


def test_unticking_the_shallow_pair_shrinks_the_span(qtbot):
    from qtpy.QtWidgets import QCheckBox

    dlg = _dialog(qtbot, pens=[_lo07_probe_a()])
    for row in (0, 1):  # the two 4576 µm recordings
        dlg._table.cellWidget(row, 0).findChild(QCheckBox).setChecked(False)

    from atlastrack.ephys.discovery import coverage_from_tip

    remaining = dlg.ticked()
    assert len(remaining) == 3
    view = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", remaining)
    assert coverage_from_tip(view, 1) == [(0.0, 705.0)]


def test_refresh_coverage_survives_a_penetration_with_no_depths(qtbot):
    pen = Penetration("X", None, "ProbeA", None, [_cand("r1", "ProbeA", [_bank(0)])])
    dlg = _dialog(qtbot, pens=[pen])
    dlg.refresh_coverage()

    assert "no insertion depths" in dlg._status.text().lower()


# ----------------------------------------------------------------- depth entry


def test_typing_a_depth_marks_it_as_the_users_and_places_the_recording(qtbot):
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", None, [
        _cand("a", "ProbeA", [_bank(0)], 4976.0),
        _cand("b", "ProbeA", [_bank(0)], None),
    ])
    dlg = _dialog(qtbot, pens=[pen])
    assert dlg._table.item(1, 4).text() == "unknown"

    dlg._table.item(1, 3).setText("4576")

    assert pen.recordings[1].insertion_depth_um == pytest.approx(4576.0)
    assert pen.recordings[1].depth_source == "user"
    assert dlg._table.item(1, 4).text() == "user"


def test_a_nonsense_depth_is_refused_and_the_old_value_restored(qtbot):
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", None,
                      [_cand("a", "ProbeA", [_bank(0)], 4976.0)])
    dlg = _dialog(qtbot, pens=[pen])

    dlg._table.item(0, 3).setText("deepish")

    assert pen.recordings[0].insertion_depth_um == pytest.approx(4976.0)
    assert dlg._table.item(0, 3).text() == "4976"
    assert "not a depth" in dlg._status.text()


def test_clearing_a_depth_makes_it_unknown_again(qtbot):
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", None,
                      [_cand("a", "ProbeA", [_bank(0)], 4976.0)])
    dlg = _dialog(qtbot, pens=[pen])

    dlg._table.item(0, 3).setText("")

    assert pen.recordings[0].insertion_depth_um is None
    assert dlg._table.item(0, 4).text() == "unknown"


def test_status_names_how_many_recordings_still_need_a_depth(qtbot):
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", None, [
        _cand("a", "ProbeA", [_bank(0)], 4976.0),
        _cand("b", "ProbeA", [_bank(0)], None),
    ])
    dlg = _dialog(qtbot, pens=[pen])

    assert "1 of 2" in dlg._status.text()


# -------------------------------------------------------------------- applying


def test_add_writes_recording_refs_onto_the_probe(qtbot):
    state = _state_with_probe()
    dlg = _dialog(qtbot, state=state, pens=[_lo07_probe_a()])

    assert dlg.apply_to_project() == 5

    probe = state.project.probes[0]
    assert len(probe.recordings) == 5
    ref = probe.recordings[0]
    assert ref.label == "LO_07_001"
    assert ref.insertion_depth_um == pytest.approx(4576.0)
    assert ref.stream_name.endswith("ProbeA")
    # Positions are absolute on the shank, so the bank offset is already included.
    assert ref.bank_offset_um == 0.0


def test_add_derives_the_electrode_range_for_a_full_bank(qtbot):
    state = _state_with_probe()
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("bank2", "ProbeA", [_bank(i, 720.0, 1425.0) for i in range(4)], 4900.0),
    ])
    dlg = _dialog(qtbot, state=state, pens=[pen])
    dlg.apply_to_project()

    assert state.project.probes[0].recordings[0].electrode_range == (97, 192)


def test_add_leaves_the_range_unset_for_the_single_column_scan(qtbot):
    """384 sites over 5745 µm is one per row; calling that 1-768 doubles the count."""
    state = _state_with_probe()
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("deep", "ProbeA", [ShankCoverage(0, 384, 0.0, 5745.0, 15.0, 2)], 4976.0),
    ])
    dlg = _dialog(qtbot, state=state, pens=[pen])
    dlg.apply_to_project()

    assert state.project.probes[0].recordings[0].electrode_range is None


def test_add_is_idempotent(qtbot):
    state = _state_with_probe()
    dlg = _dialog(qtbot, state=state, pens=[_lo07_probe_a()])

    assert dlg.apply_to_project() == 5
    assert dlg.apply_to_project() == 0
    assert len(state.project.probes[0].recordings) == 5


def test_add_only_writes_ticked_rows(qtbot):
    from qtpy.QtWidgets import QCheckBox

    state = _state_with_probe()
    dlg = _dialog(qtbot, state=state, pens=[_lo07_probe_a()])
    dlg._table.cellWidget(0, 0).findChild(QCheckBox).setChecked(False)

    assert dlg.apply_to_project() == 4
    assert "LO_07_001" not in [r.label for r in state.project.probes[0].recordings]


def test_add_refuses_when_no_probe_matches(qtbot, monkeypatch):
    """Recordings attach to a probe; without one there is nowhere to put them."""
    from qtpy.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    state = _state_with_probe(label="ProbeZ")
    dlg = _dialog(qtbot, state=state, pens=[_lo07_probe_a()])

    assert dlg.apply_to_project() == -1
    assert state.project.probes[0].recordings == []


# ------------------------------------------------------------------ pre-select


def test_the_penetration_matching_the_projects_dye_is_preselected(qtbot):
    """Projects are named for the dye, so an open project picks its own recordings."""
    state = _state_with_probe()
    # A Path, as the app sets it - not a str. The str version passed here while the
    # GUI threw "'WindowsPath' object has no attribute 'lower'" on the same line.
    state.project_path = Path("E:/TJO/LO_07/Registration/LO_07_green_whole.json")
    red = Penetration("LO_07", date(2026, 5, 6), "ProbeA", "red", [
        _cand("LO_07_001", "ProbeA", [_bank(0)], 4532.0)])
    dlg = _dialog(qtbot, state=state, pens=[red, _lo07_probe_a()])

    assert dlg._pen_combo.currentIndex() == 1
    assert dlg.selected_penetration().dye == "green"


def test_switching_penetration_reloads_the_table(qtbot):
    red = Penetration("LO_07", date(2026, 5, 6), "ProbeA", "red", [
        _cand("LO_07_001", "ProbeA", [_bank(0)], 4532.0)])
    dlg = _dialog(qtbot, pens=[red, _lo07_probe_a()])

    dlg._pen_combo.setCurrentIndex(0)
    assert dlg._table.rowCount() == 1
    dlg._pen_combo.setCurrentIndex(1)
    assert dlg._table.rowCount() == 5


# -------------------------------------------------------------- cross-checking


def test_a_stated_config_contradicting_the_geometry_is_shown_in_the_table(qtbot):
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        RecordingCandidate(
            stream=_stream("LO_07_005", "ProbeA", [_bank(0)]),
            insertion_depth_um=4976.0, depth_source="sidecar",
            stated_config="all shanks 1-96",
        )
    ])
    dlg = _dialog(qtbot, pens=[pen])

    assert "all shanks" in dlg._table.item(0, 5).text()


def test_panel_opens_the_dialog(qtbot, monkeypatch):
    """The Ephys tab's entry point exists and builds without a scan."""
    import napari

    from atlastrack.gui.widgets.ephys_discovery_dialog import EphysDiscoveryDialog
    from atlastrack.gui.widgets.ephys_panel import EphysPanelWidget

    monkeypatch.setattr(EphysDiscoveryDialog, "exec", lambda self: 0)
    viewer = napari.Viewer(show=False)
    try:
        state = _state_with_probe()
        panel = EphysPanelWidget(state, viewer)
        qtbot.addWidget(panel)
        panel.refresh_probes()

        dlg = panel._open_discovery()
        assert isinstance(dlg, EphysDiscoveryDialog)
    finally:
        viewer.close()


# ------------------------------------------- which penetration, and which shank


def test_the_selected_probes_penetration_is_preselected_not_just_the_dye(qtbot):
    """One session, one dye, two probes: dye alone always picked the first."""
    state = _state_with_probe()
    state.project_path = Path("E:/TJO/LO_07/Registration/LO_07_green_whole.json")
    probe_a = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("LO_07_005", "ProbeA", [_bank(0)], 4976.0)])
    probe_b = Penetration("LO_07", date(2026, 5, 8), "ProbeB", "green", [
        _cand("LO_07_005", "ProbeB", [_bank(3)], 5400.0)])

    dlg = _dialog(qtbot, state=state, pens=[probe_a, probe_b])
    assert dlg.selected_penetration().probe_label == "ProbeA"

    from atlastrack.gui.widgets.ephys_discovery_dialog import EphysDiscoveryDialog

    for_b = EphysDiscoveryDialog(state, probe_label="ProbeB")
    qtbot.addWidget(for_b)
    for_b.set_penetrations([probe_a, probe_b])
    assert for_b.selected_penetration().probe_label == "ProbeB"


def test_shank_rows_carry_both_numbering_schemes(qtbot):
    """The notebooks number shanks 1-4; the app stores 0-3."""
    dlg = _dialog(qtbot, pens=[_lo07_probe_a()])

    assert dlg._shank_label(0).startswith("shank 0 · notes 1")
    assert dlg._shank_label(3).startswith("shank 3 · notes 4")


def test_the_end_shanks_are_tagged_anterior_and_posterior(qtbot):
    """'shank 1, most posterior' has to be checkable without opening the 3D view."""
    from atlastrack.gui.workflow import WorkflowState

    state = WorkflowState()
    # AP increases posteriorly, so shank 0 here is the posterior end (as on ProbeA).
    state.project.probes.append(ProbeSpec(
        label="ProbeA", type=ProbeType(name="NP2.0", n_shanks=4),
        shanks=[Shank(index=i,
                      tip_ccf_um=(11512.0 - 200.0 * i, 5500.0, 5700.0),
                      entry_ccf_um=(11512.0 - 200.0 * i, 5500.0, 1000.0))
                for i in range(4)],
    ))
    dlg = _dialog(qtbot, state=state, pens=[_lo07_probe_a()])

    assert dlg._shank_label(0).endswith("post")
    assert dlg._shank_label(3).endswith("ant")
    assert dlg._row_end(1) == "" and dlg._row_end(2) == ""


def test_an_unregistered_probe_still_gets_the_numbers(qtbot):
    from atlastrack.gui.workflow import WorkflowState

    state = WorkflowState()
    state.project.probes.append(ProbeSpec(
        label="ProbeA", type=ProbeType(name="NP2.0", n_shanks=4),
        shanks=[Shank(index=i) for i in range(4)],
    ))
    dlg = _dialog(qtbot, state=state, pens=[_lo07_probe_a()])

    assert dlg._shank_label(0) == "shank 0 · notes 1"
