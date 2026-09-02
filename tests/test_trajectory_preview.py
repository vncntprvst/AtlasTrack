"""The before/after preview, and what Apply records."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from atlastrack.project.schema import ProbeSpec, ProbeType, Shank

pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


class _StripedAtlas:
    """Flat slabs in DV, so the region columns have known, checkable edges."""

    structures: ClassVar[dict] = {}

    def structure_from_coords(self, coords, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        return "Outside atlas" if dv < 1000.0 else f"S{int((dv - 1000.0) // 400.0)}"

    def get_structure_ancestors(self, acronym):
        return ["root", "grey"]


def _state():
    from atlastrack.gui.workflow import WorkflowState

    state = WorkflowState()
    state.atlas = _StripedAtlas()
    state.project.probes.append(ProbeSpec(
        label="ProbeA", type=ProbeType(name="NP2.0", n_shanks=4),
        shanks=[Shank(index=i,
                      tip_ccf_um=(8000.0, 5000.0 + 250.0 * i, 5000.0),
                      entry_ccf_um=(8000.0, 5000.0 + 250.0 * i, 1000.0))
                for i in range(4)],
    ))
    return state


class _Scan:
    def __init__(self, ok):
        self._ok = ok

    def identifiable(self):
        return self._ok


class _Fit:
    """The parts of TrajectoryFit the dialog uses."""

    def __init__(self, *, offset=-180.0, roll=-10.0, tilt=0.0, ok=True):
        self.offset_um = offset
        self.roll_deg = roll
        self.tilt_deg = tilt
        self.score = type("S", (), {"explained": 0.53})()
        self.baseline = type("S", (), {"explained": 0.18})()
        self._ok = ok

    def identifiable(self):
        return {"offset_um": self._ok, "roll_deg": True, "tilt_deg": False}

    def summary(self):
        return "explains 53.3% of the detected-boundary weight (registered: 17.8%)"


def _dialog(qtbot, state=None, fit=None, evidence=None):
    from atlastrack.gui.widgets.trajectory_preview_dialog import (
        TrajectoryPreviewDialog,
    )

    dlg = TrajectoryPreviewDialog(state or _state(), 0, fit or _Fit(),
                                  evidence=evidence or {})
    qtbot.addWidget(dlg)
    return dlg


# ------------------------------------------------------------------- building


def test_the_preview_holds_both_placements(qtbot):
    dlg = _dialog(qtbot)

    assert dlg._tips.shape == (4, 3)
    assert dlg._after_tips.shape == (4, 3)
    # A roll turns the row without moving any tip along its own track.
    before = np.linalg.norm(dlg._tips - dlg._entries, axis=1)
    after = np.linalg.norm(dlg._after_tips - dlg._after_entries, axis=1)
    assert np.allclose(before, after)


def test_a_pure_roll_moves_the_tips_but_not_their_depth(qtbot):
    dlg = _dialog(qtbot, fit=_Fit(offset=0.0, roll=-10.0, tilt=0.0))

    assert not np.allclose(dlg._tips[:, 1], dlg._after_tips[:, 1])
    assert np.allclose(dlg._tips[:, 2], dlg._after_tips[:, 2], atol=1e-6)


def test_an_offset_moves_every_shank_along_the_track(qtbot):
    dlg = _dialog(qtbot, fit=_Fit(offset=200.0, roll=0.0, tilt=0.0))

    delta = dlg._after_tips - dlg._tips
    assert np.allclose(np.linalg.norm(delta, axis=1), 200.0)


def test_the_summary_shows_the_numbers_and_the_fit_quality(qtbot):
    dlg = _dialog(qtbot)

    text = dlg._summary.text()
    assert "-180" in text and "-10.0" in text
    assert "53.3%" in text


def test_the_shank_columns_are_drawn_one_per_registered_shank(qtbot):
    dlg = _dialog(qtbot)
    dlg.refresh()

    assert dlg._indices == [0, 1, 2, 3]
    assert len(dlg._shank_plot.ci.items) == 4


def test_both_placements_get_region_bands(qtbot):
    """The comparison is the point; one column filled and one empty is useless."""
    dlg = _dialog(qtbot)

    now = dlg._bands_for(dlg._tips[0], dlg._entries[0])
    proposed = dlg._bands_for(dlg._after_tips[0], dlg._after_entries[0])

    assert now and proposed
    assert all(hi > lo for lo, hi, _a, _c in now)


def test_no_atlas_means_no_bands_rather_than_a_crash(qtbot):
    state = _state()
    state.atlas = None
    dlg = _dialog(qtbot, state=state)

    assert dlg._bands_for(dlg._tips[0], dlg._entries[0]) == []
    dlg.refresh()


def test_the_probe_plane_projection_separates_the_two_rolls(qtbot):
    """Roll leaves every tip at the same depth, so it hides in anatomical views."""
    dlg = _dialog(qtbot, fit=_Fit(offset=0.0, roll=-10.0, tilt=0.0))
    project = dlg._probe_plane()

    before = project(dlg._tips)
    after = project(dlg._after_tips)

    assert not np.allclose(before[:, 0], after[:, 0])


def test_four_projection_panels_are_drawn(qtbot):
    dlg = _dialog(qtbot)
    dlg.refresh()

    assert len(dlg._proj_plot.ci.items) == 4


# -------------------------------------------------------------------- applying


def test_apply_records_the_adjustment_without_touching_the_registration(qtbot):
    state = _state()
    before = [tuple(s.tip_ccf_um) for s in state.project.probes[0].shanks]
    dlg = _dialog(qtbot, state=state)

    adj = dlg.apply_adjustment()

    probe = state.project.probes[0]
    assert probe.trajectory_adjustment is adj
    assert adj.offset_um == pytest.approx(-180.0)
    assert adj.roll_deg == pytest.approx(-10.0)
    assert [tuple(s.tip_ccf_um) for s in probe.shanks] == before


def test_the_adjustment_records_what_was_identifiable(qtbot):
    dlg = _dialog(qtbot, fit=_Fit(ok=False))

    adj = dlg.adjustment()

    assert adj.identifiable == {"offset_um": False, "roll_deg": True,
                                "tilt_deg": False}
    assert adj.explained == pytest.approx(0.53)
    assert adj.baseline_explained == pytest.approx(0.18)
    assert adj.created_at


def test_an_unidentifiable_fit_asks_before_applying(qtbot, monkeypatch):
    from atlastrack.gui.widgets import trajectory_preview_dialog as mod

    dlg = _dialog(qtbot, fit=_Fit(ok=False))
    asked = {}
    monkeypatch.setattr(mod.QMessageBox, "question",
                        lambda *a, **k: asked.setdefault("msg", a[2])
                        or mod.QMessageBox.No)

    dlg._on_apply()

    assert "not identifiable" in asked["msg"] or "could not establish" in asked["msg"]
    assert dlg.applied is False
    assert dlg.probe.trajectory_adjustment is None


def test_the_adjustment_survives_a_save_and_reload(qtbot, tmp_path):
    from atlastrack.project.io import load_project, save_project

    state = _state()
    dlg = _dialog(qtbot, state=state)
    dlg.apply_adjustment()

    path = tmp_path / "p.json"
    save_project(state.project, path)
    reloaded = load_project(path)

    adj = reloaded.probes[0].trajectory_adjustment
    assert adj is not None
    assert adj.offset_um == pytest.approx(-180.0)
    assert adj.identifiable["tilt_deg"] is False


def test_an_old_project_without_an_adjustment_still_loads(tmp_path):
    """Additive schema: nothing that predates this may break."""
    from atlastrack.project.io import load_project, save_project

    state = _state()
    path = tmp_path / "p.json"
    save_project(state.project, path)

    assert load_project(path).probes[0].trajectory_adjustment is None


# ------------------------------------------------------- the Ephys tab's button


def _panel(qtbot, state):
    import napari

    from atlastrack.gui.widgets.ephys_panel import EphysPanelWidget

    viewer = napari.Viewer(show=False)
    panel = EphysPanelWidget(state, viewer)
    qtbot.addWidget(panel)
    panel.refresh_probes()
    return panel, viewer


def _export(shank_index, n=60):
    from atlastrack.ephys.export import ShankFeatureExport

    rng = np.random.default_rng(shank_index)
    from_tip = np.arange(n, dtype=float) * 15.0 + 175.0
    psd = np.abs(rng.normal(size=(n, 24))) + 1.0
    psd[from_tip > 600.0] *= 6.0  # a step the detector can find
    return ShankFeatureExport(
        shank_index=shank_index, track_length_um=4000.0,
        lfp_psd=psd, lfp_freqs_hz=np.linspace(0.0, 300.0, 24),
        channel_depth_from_tip_um=from_tip,
        channel_depth_below_surface_um=4000.0 - from_tip,
    )


def test_the_fit_button_refuses_without_features(qtbot, monkeypatch):
    from atlastrack.gui.widgets import ephys_panel as mod

    state = _state()
    panel, viewer = _panel(qtbot, state)
    try:
        told = {}
        monkeypatch.setattr(mod.QMessageBox, "information",
                            lambda *a, **k: told.setdefault("msg", a[2]))
        monkeypatch.setattr(
            "atlastrack.gui.workers.trajectory_fit_worker",
            lambda *a, **k: pytest.fail("must not fit with nothing to fit to"),
        )
        panel._fit_trajectory()

        assert "nothing to fit to" in told["msg"]
    finally:
        viewer.close()


def test_the_fit_button_refuses_a_probe_with_one_registered_shank(qtbot, monkeypatch):
    """Roll is only identifiable from shanks disagreeing, so one shank cannot do it."""
    from atlastrack.gui.widgets import ephys_panel as mod

    state = _state()
    for shank in state.project.probes[0].shanks[1:]:
        shank.tip_ccf_um = None
        shank.entry_ccf_um = None
    panel, viewer = _panel(qtbot, state)
    try:
        warned = {}
        monkeypatch.setattr(mod.QMessageBox, "warning",
                            lambda *a, **k: warned.setdefault("msg", a[2]))
        panel._loaded_features = {i: _export(i) for i in range(4)}
        panel._fit_trajectory()

        assert "at least two shanks" in warned["msg"]
    finally:
        viewer.close()


def test_the_fit_button_passes_the_loaded_features_to_the_worker(qtbot, monkeypatch):
    state = _state()
    panel, viewer = _panel(qtbot, state)
    try:
        panel._loaded_features = {i: _export(i) for i in range(4)}
        seen = {}

        def fake(features, tips, entries, atlas):
            seen["shanks"] = sorted(features)
            seen["tips"] = np.asarray(tips)
            raise RuntimeError("stop here")

        monkeypatch.setattr("atlastrack.gui.workers.trajectory_fit_worker", fake)
        with pytest.raises(RuntimeError, match="stop here"):
            panel._fit_trajectory()

        assert seen["shanks"] == [0, 1, 2, 3]
        assert seen["tips"].shape == (4, 3)
    finally:
        viewer.close()


def test_a_fit_with_nothing_to_fit_reports_instead_of_opening_a_preview(qtbot):
    state = _state()
    panel, viewer = _panel(qtbot, state)
    try:
        dlg = panel._on_fit_done(0, {"fit": None, "evidence": {},
                                     "notes": "No shank produced a boundary"})

        assert dlg is None
        assert "No shank produced a boundary" in panel._status.text()
    finally:
        viewer.close()


def test_a_successful_fit_opens_the_preview(qtbot, monkeypatch):
    from atlastrack.gui.widgets import trajectory_preview_dialog as mod

    state = _state()
    panel, viewer = _panel(qtbot, state)
    try:
        monkeypatch.setattr(mod.TrajectoryPreviewDialog, "exec", lambda self: 0)
        dlg = panel._on_fit_done(0, {"fit": _Fit(), "evidence": {}, "notes": "note"})

        assert isinstance(dlg, mod.TrajectoryPreviewDialog)
        assert "-180" in panel._status.text()
        assert state.project.probes[0].trajectory_adjustment is None
    finally:
        viewer.close()


# ------------------------------------------------------ layout and the outline


def test_the_projections_are_one_row_of_four(qtbot):
    """A 2x2 grid stretched each panel to tens of mm across; the tracks are 5 mm tall
    and under 1 mm wide, so they came out a few pixels high."""
    dlg = _dialog(qtbot)
    dlg.refresh()

    assert len(dlg._proj_plot.ci.items) == 4
    assert sorted(dlg._proj_plot.ci.rows) == [0]


def test_the_legend_says_the_circles_are_tips(qtbot):
    dlg = _dialog(qtbot)

    title = dlg._proj_box.title()
    assert "TIPS" in title
    assert "dashed grey" in title and "solid blue" in title


def test_the_outline_is_off_by_default_and_toggles(qtbot):
    dlg = _dialog(qtbot)

    assert dlg._outline_check.isChecked() is False
    dlg._outline_check.setChecked(True)   # refresh is wired to the toggle
    assert dlg._outline_check.isChecked() is True


def test_an_atlas_without_an_annotation_yields_no_outline(qtbot):
    """The outline is orientation, never evidence; its absence must not break the view."""
    from atlastrack.gui.widgets.trajectory_preview_dialog import brain_outline

    assert brain_outline(_StripedAtlas(), 1, row_is_x=False) == []


def test_the_outline_is_computed_from_the_projected_silhouette(qtbot):
    """A silhouette, not a slice: the shanks are drawn through the whole volume."""
    from atlastrack.gui.widgets import trajectory_preview_dialog as mod

    class _Blob:
        atlas_name = "test-blob"
        resolution = (25.0, 25.0, 25.0)

        def __init__(self):
            self.annotation = np.zeros((40, 30, 50), dtype=int)
            self.annotation[10:30, 5:25, 15:35] = 1

    mod._OUTLINE_CACHE.clear()
    contours = mod.brain_outline(_Blob(), 1, row_is_x=False, min_points=4)

    assert contours
    xs, ys = contours[0]
    # Flattening DV leaves (AP, ML); with row_is_x False, x is ML and y is AP.
    assert 15 * 25 - 30 <= xs.min() <= 15 * 25 + 30
    assert 10 * 25 - 30 <= ys.min() <= 10 * 25 + 30
    mod._OUTLINE_CACHE.clear()


def test_the_outline_is_cached_per_projection(qtbot):
    from atlastrack.gui.widgets import trajectory_preview_dialog as mod

    class _Counting:
        atlas_name = "counting"
        resolution = (25.0, 25.0, 25.0)
        calls = 0

        @property
        def annotation(self):
            type(self).calls += 1
            a = np.zeros((20, 20, 20), dtype=int)
            a[5:15, 5:15, 5:15] = 1
            return a

    mod._OUTLINE_CACHE.clear()
    atlas = _Counting()
    mod.brain_outline(atlas, 1, row_is_x=False, min_points=4)
    mod.brain_outline(atlas, 1, row_is_x=False, min_points=4)

    assert _Counting.calls == 1
    mod._OUTLINE_CACHE.clear()


# ------------------------------------------------------------ saving the fit


def _real_fit(tmp_path):
    """A small but genuine TrajectoryFit over the striped atlas."""
    from atlastrack.probes.trajectory_fit import ShankEvidence, fit_trajectory

    atlas = _StripedAtlas()
    tips = np.array([[8000.0, 5000.0 + 250.0 * i, 5000.0] for i in range(4)])
    entries = tips - np.array([0.0, 0.0, 4000.0])
    ev = {i: ShankEvidence(i, np.array([500.0, 1400.0, 2300.0]),
                           np.array([4.0, 3.0, 5.0])) for i in range(4)}
    fit = fit_trajectory(tips, entries, ev, atlas,
                         offsets_um=np.arange(-100.0, 101.0, 50.0),
                         rolls_deg=[0.0], tilts_deg=[0.0])
    return fit, ev, tips, entries


def test_a_saved_fit_round_trips_with_its_scans_and_matches(tmp_path):
    from atlastrack.probes.trajectory_fit_io import load_fit, save_fit

    fit, ev, tips, entries = _real_fit(tmp_path)
    out = save_fit(tmp_path / "f.npz", fit, ev, probe_label="ProbeA",
                   notes="loo says roll is unstable", tips=tips, entries=entries)

    back, ev2, meta = load_fit(out)

    assert (back.offset_um, back.roll_deg, back.tilt_deg) == (
        fit.offset_um, fit.roll_deg, fit.tilt_deg)
    assert back.score.explained == pytest.approx(fit.score.explained)
    assert back.score.residual_spread_um == pytest.approx(
        fit.score.residual_spread_um)
    assert sorted(back.scans) == sorted(fit.scans)
    assert sorted(ev2) == sorted(ev)
    assert meta["probe_label"] == "ProbeA"
    assert "roll is unstable" in meta["notes"]


def test_a_saved_fit_knows_when_its_evidence_has_changed(tmp_path):
    """A cache that outlives its data would read as a fresh answer."""
    from atlastrack.probes.trajectory_fit import ShankEvidence
    from atlastrack.probes.trajectory_fit_io import matches_current, save_fit

    fit, ev, tips, entries = _real_fit(tmp_path)
    out = save_fit(tmp_path / "f.npz", fit, ev, tips=tips, entries=entries)

    assert matches_current(out, ev)

    moved = dict(ev)
    moved[0] = ShankEvidence(0, ev[0].depths_from_tip_um + 25.0, ev[0].weights)
    assert not matches_current(out, moved)
    assert not matches_current(out, {k: v for k, v in ev.items() if k != 3})


def test_a_missing_or_unreadable_cache_is_not_a_match(tmp_path):
    from atlastrack.probes.trajectory_fit_io import matches_current

    _fit, ev, _t, _e = _real_fit(tmp_path)
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not an npz")

    assert not matches_current(tmp_path / "absent.npz", ev)
    assert not matches_current(junk, ev)


def test_the_default_fit_path_sits_beside_the_features(tmp_path):
    from atlastrack.ephys.export import default_export_path
    from atlastrack.probes.trajectory_fit_io import default_fit_path

    project = tmp_path / "proj.json"

    assert default_fit_path(project, "ProbeA").parent == \
        default_export_path(project, "ProbeA").parent
    assert default_fit_path(project, "ProbeA").name.endswith("_fit.npz")


def test_the_dialog_can_write_a_fit(qtbot, tmp_path):
    from atlastrack.probes.trajectory_fit_io import load_fit

    fit, ev, _t, _e = _real_fit(tmp_path)
    dlg = _dialog(qtbot, fit=fit, evidence=ev)

    written = dlg.save_fit_to(tmp_path / "from_dialog.npz")
    back, ev2, meta = load_fit(written)

    assert meta["probe_label"] == "ProbeA"
    assert sorted(ev2) == sorted(ev)
    assert back.offset_um == pytest.approx(fit.offset_um)


def test_the_panel_reuses_a_matching_saved_fit_instead_of_refitting(qtbot, tmp_path):
    from atlastrack.probes.trajectory_fit import evidence_from_features
    from atlastrack.probes.trajectory_fit_io import default_fit_path, save_fit

    state = _state()
    state.project_path = tmp_path / "proj.json"
    panel, viewer = _panel(qtbot, state)
    try:
        features = {i: _export(i) for i in range(4)}
        panel._loaded_features = features
        ev = evidence_from_features(features)
        assert ev, "the synthetic features must produce boundaries"

        fit, _ev, _t, _e = _real_fit(tmp_path)
        save_fit(default_fit_path(state.project_path, "ProbeA"), fit, ev,
                 probe_label="ProbeA", notes="cached")

        reused = panel._reusable_fit(state.project.probes[0], features)

        assert reused is not None
        assert reused["from_cache"] is True
        assert reused["notes"] == "cached"
        assert "Reusing the saved fit" in panel._status.text()
    finally:
        viewer.close()


def test_the_panel_refits_when_the_cache_no_longer_matches(qtbot, tmp_path):
    from atlastrack.probes.trajectory_fit import ShankEvidence
    from atlastrack.probes.trajectory_fit_io import default_fit_path, save_fit

    state = _state()
    state.project_path = tmp_path / "proj.json"
    panel, viewer = _panel(qtbot, state)
    try:
        features = {i: _export(i) for i in range(4)}
        fit, _ev, _t, _e = _real_fit(tmp_path)
        stale = {0: ShankEvidence(0, np.array([10.0, 20.0]), np.array([1.0, 1.0]))}
        save_fit(default_fit_path(state.project_path, "ProbeA"), fit, stale,
                 probe_label="ProbeA")

        assert panel._reusable_fit(state.project.probes[0], features) is None
    finally:
        viewer.close()


def test_the_record_button_says_it_does_not_move_anything(qtbot):
    dlg = _dialog(qtbot)

    assert "does not move it" in dlg._apply_btn.text()
    assert "does NOT move the probe" in dlg._effect_note.text()


def test_the_preview_shows_the_registration_alternative(qtbot):
    """The other reading of the same fit belongs beside the probe move."""
    state = _state()
    for i, shank in enumerate(state.project.probes[0].shanks):
        shank.tip_section_idx = 3
        shank.entry_section_idx = 7 + i
    state.project.section_spacing_um = 100.0
    dlg = _dialog(qtbot, state=state, fit=_Fit(offset=200.0, roll=0.0, tilt=0.0))

    text = dlg.registration_suggestion_text()

    assert text.startswith("If this is a registration error")
    assert "Sections carrying this probe: 3, 7" in text
    assert dlg._suggestion.text() == text


def test_the_suggestion_is_empty_for_an_unregistered_probe(qtbot):
    state = _state()
    for shank in state.project.probes[0].shanks:
        shank.tip_ccf_um = None
        shank.entry_ccf_um = None
    dlg = _dialog(qtbot, state=state, fit=_Fit(offset=0.0, roll=0.0, tilt=0.0))

    assert dlg.registration_suggestion() is None
    assert dlg.registration_suggestion_text() == ""
