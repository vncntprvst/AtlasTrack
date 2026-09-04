"""The Export group: one button, a format selector, and a legible Paxinos control.

What it replaced: three buttons, two of them called "export ... CSV" and differing
only by a coordinate frame, plus a four-line paragraph of explanation sitting
permanently in the panel and a preset list labelled by citation ("Pinpoint / Qiu
2018", "Allen forum") rather than by what the preset does.
"""
from __future__ import annotations

import pytest

from atlastrack.io.ccf_coords import PAXINOS_ALIGNMENTS

pytestmark = pytest.mark.qt


def _panel(qtbot):
    import napari

    from atlastrack.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    panel, viz_panel = _build_panel(viewer)
    qtbot.addWidget(panel)
    qtbot.addWidget(viz_panel)
    return viz_panel, viewer


def _button_texts(widget):
    from qtpy.QtWidgets import QPushButton

    return [b.text() for b in widget.findChildren(QPushButton)]


# ---------------------------------------------------------------------------
# One button, one selector
# ---------------------------------------------------------------------------


def test_there_is_no_longer_a_second_csv_button(qtbot):
    """Two buttons both reading 'export CSV' made the frame the hidden variable."""
    viz, viewer = _panel(qtbot)
    try:
        csv_buttons = [t for t in _button_texts(viz) if "CSV" in t.upper()]

        assert csv_buttons == []
    finally:
        viewer.close()


def test_the_format_selector_offers_every_output(qtbot):
    """The 3D HTML is a file you share, so it belongs with the other exports; the
    live napari window stays in the 3D group because it is not a file."""
    viz, viewer = _panel(qtbot)
    try:
        combo = viz._format_combo
        keys = [combo.itemData(i) for i in range(combo.count())]

        assert keys == ["csv", "pkl", "html", "series"]
    finally:
        viewer.close()


def test_the_3d_group_keeps_the_live_view_and_loses_the_html_button(qtbot):
    viz, viewer = _panel(qtbot)
    try:
        texts = _button_texts(viz)

        assert "3D view" in texts
        assert not any("Plotly" in t for t in texts)
    finally:
        viewer.close()


def test_only_the_csv_offers_paxinos(qtbot):
    """The pkl and the HTML are both CCF; a tickable box would promise otherwise."""
    viz, viewer = _panel(qtbot)
    try:
        for i in range(viz._format_combo.count()):
            viz._format_combo.setCurrentIndex(i)
            expected = viz._format_combo.currentData() == "csv"
            assert viz._paxinos_check.isEnabled() is expected
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Paxinos control
# ---------------------------------------------------------------------------


def test_the_alignment_list_is_disabled_until_paxinos_is_asked_for(qtbot):
    """It only qualifies the checkbox, so it must not look like a live choice."""
    viz, viewer = _panel(qtbot)
    try:
        assert not viz._paxinos_check.isChecked()
        assert not viz._paxinos_combo.isEnabled()

        viz._paxinos_check.setChecked(True)

        assert viz._paxinos_combo.isEnabled()
    finally:
        viewer.close()


def test_the_pkl_format_disables_paxinos_and_says_why(qtbot):
    """Leaving the box tickable would promise a conversion the writer cannot do."""
    viz, viewer = _panel(qtbot)
    try:
        viz._format_combo.setCurrentIndex(1)  # pkl

        assert not viz._paxinos_check.isEnabled()
        assert not viz._paxinos_check.isChecked()
        assert "CCF" in viz._paxinos_check.toolTip()
    finally:
        viewer.close()


def test_the_paxinos_choice_survives_a_trip_through_pkl(qtbot):
    viz, viewer = _panel(qtbot)
    try:
        viz._paxinos_check.setChecked(True)
        viz._format_combo.setCurrentIndex(1)  # pkl - forced off
        viz._format_combo.setCurrentIndex(0)  # back to csv

        assert viz._paxinos_check.isChecked()
    finally:
        viewer.close()


def test_an_untouched_checkbox_stays_off_across_a_format_round_trip(qtbot):
    viz, viewer = _panel(qtbot)
    try:
        viz._format_combo.setCurrentIndex(1)
        viz._format_combo.setCurrentIndex(0)

        assert not viz._paxinos_check.isChecked()
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# The alignment presets
# ---------------------------------------------------------------------------


def test_every_offered_alignment_is_a_real_one(qtbot):
    """A typo here would be a ValueError only at export time, after the file dialog."""
    viz, viewer = _panel(qtbot)
    try:
        combo = viz._paxinos_combo
        keys = {combo.itemData(i) for i in range(combo.count())}

        assert keys == set(PAXINOS_ALIGNMENTS)
    finally:
        viewer.close()


def test_the_labels_lead_with_the_effect_not_the_citation(qtbot):
    """"Pinpoint" and "Allen forum" name where the numbers came from, not what
    they do; a user choosing between them learned nothing from the old labels."""
    viz, viewer = _panel(qtbot)
    try:
        combo = viz._paxinos_combo
        labels = [combo.itemText(i) for i in range(combo.count())]

        assert not any(t.startswith(("Pinpoint", "Allen forum")) for t in labels)
        assert all(
            t.startswith("Tilt") or t.startswith("No correction") for t in labels
        )
    finally:
        viewer.close()


def test_the_help_text_covers_every_preset_and_states_the_uncertainty(qtbot):
    from atlastrack.gui.widgets.viz_export_panel import _PAXINOS_HELP

    for name in ("Qiu 2018", "Dorr 2008", "Allen community", "No correction"):
        assert name in _PAXINOS_HELP
    assert "estimates" in _PAXINOS_HELP
    assert "not an atlas you register to" in _PAXINOS_HELP


def test_the_standing_paxinos_paragraph_is_gone_from_the_panel(qtbot):
    """It moved behind the "?"; a wall of text on every panel draw is not guidance."""
    from qtpy.QtWidgets import QLabel

    viz, viewer = _panel(qtbot)
    try:
        texts = [lbl.text() for lbl in viz.findChildren(QLabel)]

        assert not any("not an atlas you register to" in t for t in texts)
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# The button actually dispatches
# ---------------------------------------------------------------------------


_FAKE_ATLAS = object()


def _capture_export(qtbot, monkeypatch, tmp_path, *, fmt_index, paxinos):
    from qtpy.QtWidgets import QFileDialog

    from atlastrack.probes import channels

    viz, viewer = _panel(qtbot)
    # A loaded atlas makes the CCF export synchronous (no atlas worker) and is
    # what the export must hand on for the region columns.
    viz._state.atlas = _FAKE_ATLAS
    out = tmp_path / "out"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    calls = []
    monkeypatch.setattr(
        channels, "export_channel_csv",
        lambda project, path, **kw: calls.append(("ccf", kw)) or 3,
    )
    monkeypatch.setattr(
        channels, "export_paxinos_csv",
        lambda project, path, **kw: calls.append(("paxinos", kw)) or 3,
    )
    viz._format_combo.setCurrentIndex(fmt_index)
    viz._paxinos_check.setChecked(paxinos)
    viz._export()
    viewer.close()
    return calls


def test_exporting_without_paxinos_writes_ccf(qtbot, monkeypatch, tmp_path):
    calls = _capture_export(
        qtbot, monkeypatch, tmp_path, fmt_index=0, paxinos=False
    )

    assert [c[0] for c in calls] == ["ccf"]
    assert calls[0][1]["atlas"] is _FAKE_ATLAS


def test_exporting_with_paxinos_passes_the_selected_alignment(qtbot, monkeypatch,
                                                              tmp_path):
    calls = _capture_export(qtbot, monkeypatch, tmp_path, fmt_index=0, paxinos=True)

    assert [c[0] for c in calls] == ["paxinos"]
    assert calls[0][1]["alignment"] in PAXINOS_ALIGNMENTS


# ---------------------------------------------------------------------------
# The section-series format
# ---------------------------------------------------------------------------


def test_the_series_options_appear_only_for_the_series_format(qtbot):
    """Three checkboxes that apply to one of four formats must not sit greyed out."""
    viz, viewer = _panel(qtbot)
    try:
        shown = {}
        for i in range(viz._format_combo.count()):
            viz._format_combo.setCurrentIndex(i)
            shown[viz._format_combo.currentData()] = viz._series_box.isVisibleTo(viz)

        assert shown == {"csv": False, "pkl": False, "html": False, "series": True}
    finally:
        viewer.close()


def test_outlines_are_on_by_default_and_gate_the_burnt_in_copy(qtbot):
    """A burnt-in overlay without outlines is not a thing that can be produced."""
    viz, viewer = _panel(qtbot)
    try:
        assert viz._series_outlines.isChecked()

        viz._series_outlines.setChecked(False)

        assert not viz._series_overlays.isEnabled()
    finally:
        viewer.close()


def test_the_series_export_passes_the_checkboxes_through(qtbot, monkeypatch, tmp_path):
    from qtpy.QtWidgets import QFileDialog

    from atlastrack.io import series_export

    viz, viewer = _panel(qtbot)
    try:
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            staticmethod(lambda *a, **k: str(tmp_path)),
        )
        seen = {}

        def _fake(project, out_dir, **kwargs):
            seen.update(kwargs)
            return series_export.SeriesExportResult(out_dir=tmp_path, sections=2)

        monkeypatch.setattr(series_export, "export_section_series", _fake)
        # The real path resolves a region atlas first (outlines need one); that is
        # not what this test is about.
        monkeypatch.setattr(viz, "_ensure_display_atlas", lambda cb: cb())
        viz._format_combo.setCurrentIndex(3)  # series
        viz._series_overlays.setChecked(True)
        viz._export()

        assert seen["write_outlines"] is True
        assert seen["write_overlays"] is True
    finally:
        viewer.close()


def test_straightening_is_on_by_default_and_passed_through(qtbot, monkeypatch,
                                                           tmp_path):
    """The whole point of the series export: a set of sections you can flick through."""
    from qtpy.QtWidgets import QFileDialog

    from atlastrack.io import series_export

    viz, viewer = _panel(qtbot)
    try:
        assert viz._series_straighten.isChecked()
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            staticmethod(lambda *a, **k: str(tmp_path)),
        )
        monkeypatch.setattr(viz, "_ensure_display_atlas", lambda cb: cb())
        seen = {}

        def _fake(project, out_dir, **kwargs):
            seen.update(kwargs)
            return series_export.SeriesExportResult(out_dir=tmp_path, sections=1)

        monkeypatch.setattr(series_export, "export_section_series", _fake)
        viz._format_combo.setCurrentIndex(3)
        viz._export()

        assert seen["straighten"] is True
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Paxinos region labels, via the region atlas
# ---------------------------------------------------------------------------


def test_the_paxinos_labelled_atlas_is_offered_and_says_so(qtbot):
    """Chon/Kim is voxel-identical to Allen but carries Franklin-Paxinos names, so
    the label route needs no estimate at all - unlike the coordinate transform."""
    viz, viewer = _panel(qtbot)
    try:
        combo = viz._region_atlas_combo
        entries = {combo.itemData(i): combo.itemText(i) for i in range(combo.count())}

        assert "kim_mouse_25um" in entries
        assert "Paxinos" in entries["kim_mouse_25um"]
        assert "Paxinos" in combo.toolTip()
    finally:
        viewer.close()


def test_the_help_distinguishes_paxinos_labels_from_paxinos_coordinates(qtbot):
    """The coordinates are published estimates; the labels are just a relabelling."""
    from atlastrack.gui.widgets.viz_export_panel import _PAXINOS_HELP

    assert "Chon/Kim" in _PAXINOS_HELP
    assert "relabelling, not a transform" in _PAXINOS_HELP


def test_the_isotropic_v2_atlas_is_offered_with_its_offset_stated(qtbot):
    """It samples the same volume at 20 µm, so the export restates each anchoring
    on its grid; its annotation also sits ~102 µm posterior of the 25 µm release,
    which is the atlas's own property and is surfaced rather than corrected."""
    from atlastrack.gui.widgets.viz_export_panel import _REGION_ATLAS_CAVEATS

    viz, viewer = _panel(qtbot)
    try:
        combo = viz._region_atlas_combo
        ids = [combo.itemData(i) for i in range(combo.count())]

        assert "kim_mouse_isotropic_20um" in ids
        assert "102" in _REGION_ATLAS_CAVEATS["kim_mouse_isotropic_20um"]
        assert "102 µm posterior" in combo.toolTip()
    finally:
        viewer.close()
