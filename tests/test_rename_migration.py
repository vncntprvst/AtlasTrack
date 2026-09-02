"""What must keep working across the rename from Histo-to-CCF to AtlasTrack.

A rename is only safe if the things a user already has on disk still work: their
preferences, their saved projects, and any script that calls the old command. Each
of those is a separate path, and each is easy to break silently - the settings file
in particular would simply reappear as defaults, which reads like "the app forgot my
atlas folder" rather than like an error.
"""
from __future__ import annotations

import json

import pytest

from atlastrack import config


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Point both the current and pre-rename preference paths into tmp_path."""
    new_dir = tmp_path / ".atlastrack"
    old_dir = tmp_path / ".histo2ccf"
    monkeypatch.setattr(config, "_PREFS_DIR", new_dir)
    monkeypatch.setattr(config, "_LEGACY_PREFS_DIR", old_dir)
    monkeypatch.setattr(config, "_PREFS_FILE", new_dir / "settings.json")
    monkeypatch.setattr(config, "_LEGACY_PREFS_FILE", old_dir / "settings.json")
    return new_dir, old_dir


def _write(directory, **values):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "settings.json").write_text(json.dumps(values), encoding="utf-8")


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def test_settings_are_read_from_the_old_directory_when_that_is_all_there_is(homes):
    """The upgrade case: the app has been renamed, the user has not run it yet."""
    _new, old = homes
    _write(old, last_atlas_id="kim_mouse_25um", section_spacing_um=120.0)

    settings = config.load_app_settings()

    assert settings.last_atlas_id == "kim_mouse_25um"
    assert settings.section_spacing_um == 120.0


def test_the_new_directory_wins_once_it_exists(homes):
    """After the first save the old file is stale and must not come back."""
    new, old = homes
    _write(old, last_atlas_id="kim_mouse_25um")
    _write(new, last_atlas_id="allen_mouse_25um")

    assert config.load_app_settings().last_atlas_id == "allen_mouse_25um"


def test_defaults_when_neither_exists(homes):
    assert config.load_app_settings().last_atlas_id is not None


def test_saving_writes_to_the_new_directory_only(homes):
    """Migration is one-way: nothing should keep updating the old location."""
    new, old = homes
    _write(old, last_atlas_id="kim_mouse_25um")

    settings = config.load_app_settings()
    config.save_app_settings(settings)

    assert (new / "settings.json").exists()
    saved = json.loads((old / "settings.json").read_text(encoding="utf-8"))
    assert saved == {"last_atlas_id": "kim_mouse_25um"}  # untouched


# ---------------------------------------------------------------------------
# Saved projects
# ---------------------------------------------------------------------------


def test_a_project_saved_before_the_rename_still_loads(tmp_path):
    """Nothing filters on the suffix, so the old extension keeps working."""
    from atlastrack.project.io import load_project, save_project
    from atlastrack.project.schema import Project

    old_style = tmp_path / "demo.histo2ccf.json"
    save_project(Project(), old_style)

    assert load_project(old_style) is not None


def test_new_projects_are_written_with_the_new_suffix(tmp_path):
    from atlastrack.project.io import save_project
    from atlastrack.project.schema import Project

    new_style = tmp_path / "demo.atlastrack.json"
    save_project(Project(), new_style)

    assert load_ok(new_style)


def load_ok(path):
    from atlastrack.project.io import load_project

    return load_project(path) is not None


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_both_command_names_point_at_the_same_app():
    """``histo2ccf`` is kept so existing scripts and habits keep working."""
    from pathlib import Path

    import tomllib

    pyproject = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = pyproject["project"]["scripts"]

    assert scripts["atlastrack"] == "atlastrack.cli:app"
    assert scripts["histo2ccf"] == scripts["atlastrack"]


def test_the_package_is_named_for_the_app():
    from pathlib import Path

    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "atlastrack"
    assert "atlastrack[" in "".join(
        pyproject["project"]["optional-dependencies"]["all"]
    )
