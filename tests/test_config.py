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


def test_round_trip(tmp_path: Path, monkeypatch) -> None:
    import histo_to_ccf.config as cfg

    monkeypatch.setattr(cfg, "_PREFS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "_PREFS_FILE", tmp_path / "settings.json")

    settings = AppSettings(
        last_atlas_id="kim_mouse_25um",
        bspline_grid=6,
        max_iterations=50,
        section_spacing_um=150.0,
    )
    save_app_settings(settings)
    assert (tmp_path / "settings.json").exists()

    loaded = load_app_settings()
    assert loaded.last_atlas_id == "kim_mouse_25um"
    assert loaded.bspline_grid == 6
    assert loaded.max_iterations == 50
    assert loaded.section_spacing_um == pytest.approx(150.0)


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
    assert __version__ == "0.1.22"


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

        settings = AppSettings(bspline_grid=6, max_iterations=50)
        widget.apply_settings(settings)
        assert widget._grid_spin.value() == 6
        assert widget._iter_spin.value() == 50

        out = AppSettings()
        widget.collect_settings(out)
        assert out.bspline_grid == 6
        assert out.max_iterations == 50
    finally:
        viewer.close()
