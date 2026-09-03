"""Tests for config.py settings persistence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlastrack.config import AppSettings, load_app_settings, save_app_settings


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
    assert s.rigid_array_enforce is False
    assert s.rigid_array_tolerance == 0.25


def test_rigid_array_tolerance_clamp() -> None:
    assert AppSettings(rigid_array_tolerance=5.0).rigid_array_tolerance == 1.0
    assert AppSettings(rigid_array_tolerance=-1.0).rigid_array_tolerance == 0.0


def test_engine_validator() -> None:
    assert AppSettings(reg_engine="elastix").reg_engine == "elastix"
    assert AppSettings(reg_engine="sitk").reg_engine == "sitk"
    # Unknown engine falls back to "auto".
    assert AppSettings(reg_engine="bogus").reg_engine == "auto"


def test_bending_weight_clamp() -> None:
    assert AppSettings(bending_energy_weight=1000.0).bending_energy_weight == 500.0
    assert AppSettings(bending_energy_weight=-5.0).bending_energy_weight == 0.0


def test_round_trip(tmp_path: Path, monkeypatch) -> None:
    import atlastrack.config as cfg

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
    import atlastrack.config as cfg

    monkeypatch.setattr(cfg, "_PREFS_FILE", tmp_path / "nonexistent.json")
    settings = load_app_settings()
    assert settings.last_atlas_id == "allen_mouse_25um"


def test_load_with_corrupt_file(tmp_path: Path, monkeypatch) -> None:
    import atlastrack.config as cfg

    prefs = tmp_path / "bad.json"
    prefs.write_text("{not valid json!!}", encoding="utf-8")
    monkeypatch.setattr(cfg, "_PREFS_FILE", prefs)
    settings = load_app_settings()
    assert settings.last_atlas_id == "allen_mouse_25um"  # falls back to defaults


def test_remember_project(tmp_path: Path) -> None:
    proj = tmp_path / "a" / "one.json"
    proj.parent.mkdir()
    proj.write_text("{}", encoding="utf-8")

    s = AppSettings()
    s.remember_project(proj)
    assert s.last_project_dir == str(proj.parent)
    assert s.recent_projects[0] == str(proj)

    # A second project goes to the front; the dir follows it.
    other = tmp_path / "b" / "two.json"
    other.parent.mkdir()
    s.remember_project(other)
    assert s.recent_projects[:2] == [str(other), str(proj)]
    assert s.last_project_dir == str(other.parent)

    # Re-loading an earlier one moves it to the front without duplicating.
    s.remember_project(proj)
    assert s.recent_projects[0] == str(proj)
    assert s.recent_projects.count(str(proj)) == 1


def test_recent_projects_cap(tmp_path: Path) -> None:
    s = AppSettings()
    for i in range(20):
        s.remember_project(tmp_path / f"p{i}.json")
    from atlastrack.config import _MAX_RECENT_PROJECTS

    assert len(s.recent_projects) == _MAX_RECENT_PROJECTS
    assert s.recent_projects[0] == str(tmp_path / "p19.json")  # newest first


def test_project_start_dir(tmp_path: Path) -> None:
    s = AppSettings()
    assert s.project_start_dir() == ""  # unset
    s.last_project_dir = str(tmp_path)
    assert s.project_start_dir() == str(tmp_path)  # exists
    s.last_project_dir = str(tmp_path / "gone")
    assert s.project_start_dir() == ""  # missing dir -> empty


def test_recent_projects_round_trip(tmp_path: Path, monkeypatch) -> None:
    import atlastrack.config as cfg

    monkeypatch.setattr(cfg, "_PREFS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "_PREFS_FILE", tmp_path / "settings.json")

    s = AppSettings()
    s.remember_project(tmp_path / "proj.json")
    save_app_settings(s)
    loaded = load_app_settings()
    assert loaded.recent_projects == [str(tmp_path / "proj.json")]
    assert loaded.last_project_dir == str(tmp_path)


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
    from atlastrack.cli import app
    from atlastrack import __version__

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_string() -> None:
    from atlastrack import __version__
    assert __version__ == "0.7.2"


@pytest.mark.qt
def test_register_panel_with_settings(qtbot) -> None:
    """RegisterPanelWidget.apply_settings populates controls correctly."""
    import napari
    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget
    from atlastrack.gui.workflow import WorkflowState

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
