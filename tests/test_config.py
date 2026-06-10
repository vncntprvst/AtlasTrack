"""Tests for config.py settings persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from histo_to_ccf.config import AppSettings, load_app_settings, save_app_settings


def test_defaults() -> None:
    s = AppSettings()
    assert s.last_atlas_id == "allen_mouse_25um"
    assert s.bspline_grid == 8
    assert s.max_iterations == 100
    assert s.section_spacing_um == 80.0
    assert s.reg_engine == "auto"
    assert s.bending_energy_weight == 20.0
    assert s.use_tissue_mask is True
    assert s.prealign_similarity is True


def test_engine_validator() -> None:
    assert AppSettings(reg_engine="elastix").reg_engine == "elastix"
    assert AppSettings(reg_engine="sitk").reg_engine == "sitk"
    # Unknown engine falls back to "auto".
    assert AppSettings(reg_engine="bogus").reg_engine == "auto"


def test_bending_weight_clamp() -> None:
    assert AppSettings(bending_energy_weight=1000.0).bending_energy_weight == 500.0
    assert AppSettings(bending_energy_weight=-5.0).bending_energy_weight == 0.0


def test_round_trip(tmp_path: Path, monkeypatch) -> None:
    import histo_to_ccf.config as cfg

    monkeypatch.setattr(cfg, "_PREFS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "_PREFS_FILE", tmp_path / "settings.json")

    settings = AppSettings(
        last_atlas_id="kim_mouse_25um",
        bspline_grid=6,
        max_iterations=50,
        section_spacing_um=150.0,
        reg_engine="elastix",
        bending_energy_weight=35.0,
        use_tissue_mask=False,
        prealign_similarity=False,
    )
    save_app_settings(settings)
    assert (tmp_path / "settings.json").exists()

    loaded = load_app_settings()
    assert loaded.last_atlas_id == "kim_mouse_25um"
    assert loaded.bspline_grid == 6
    assert loaded.max_iterations == 50
    assert loaded.section_spacing_um == pytest.approx(150.0)
    assert loaded.reg_engine == "elastix"
    assert loaded.bending_energy_weight == pytest.approx(35.0)
    assert loaded.use_tissue_mask is False
    assert loaded.prealign_similarity is False


def test_load_with_missing_file(tmp_path: Path, monkeypatch) -> None:
    import histo_to_ccf.config as cfg

    monkeypatch.setattr(cfg, "_PREFS_FILE", tmp_path / "nonexistent.json")
    settings = load_app_settings()
    assert settings.last_atlas_id == "allen_mouse_25um"


def test_load_with_corrupt_file(tmp_path: Path, monkeypatch) -> None:
    import histo_to_ccf.config as cfg

    prefs = tmp_path / "bad.json"
    prefs.write_text("{not valid json!!}", encoding="utf-8")
    monkeypatch.setattr(cfg, "_PREFS_FILE", prefs)
    settings = load_app_settings()
    assert settings.last_atlas_id == "allen_mouse_25um"  # falls back to defaults


def test_clamp_grid_validator() -> None:
    s = AppSettings(bspline_grid=100)
    assert s.bspline_grid == 24   # clamped to max

    s2 = AppSettings(bspline_grid=1)
    assert s2.bspline_grid == 4   # clamped to min


def test_clamp_iterations_validator() -> None:
    s = AppSettings(max_iterations=1000)
    assert s.max_iterations == 500


def test_version_command() -> None:
    """CLI `version` command prints the package version."""
    from typer.testing import CliRunner
    from histo_to_ccf.cli import app
    from histo_to_ccf import __version__

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_string() -> None:
    from histo_to_ccf import __version__
    assert __version__ == "0.2.14"


@pytest.mark.qt
def test_register_panel_with_settings(qtbot) -> None:
    """RegisterPanelWidget.apply_settings populates controls correctly."""
    import napari
    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget
    from histo_to_ccf.gui.workflow import WorkflowState

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        widget = RegisterPanelWidget(state, viewer)
        qtbot.addWidget(widget)

        settings = AppSettings(bspline_grid=6, max_iterations=50, boundary_snap=False)
        widget.apply_settings(settings)
        assert widget._grid_spin.value() == 6
        assert widget._iter_spin.value() == 50
        assert widget._boundary_snap.isChecked() is False

        out = AppSettings()
        widget.collect_settings(out)
        assert out.bspline_grid == 6
        assert out.max_iterations == 50
        assert out.boundary_snap is False

        # DeepSlice is on by default and the registration parameters are NOT shown
        # inline - they live in a dialog opened on demand.
        assert widget._use_deepslice.isChecked() is True
        assert widget._params_box.parent() is not widget
        widget.open_parameters_dialog()
        assert widget._params_dialog is not None
        assert widget._params_box.parent() is widget._params_dialog
    finally:
        viewer.close()
