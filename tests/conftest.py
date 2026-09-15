"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LEGACY_DIR = Path(__file__).parent.parent / "legacy"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the bundled test-fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def legacy_dir() -> Path:
    """Path to the archived legacy code (HERBS scripts, old notebooks)."""
    return LEGACY_DIR


@pytest.fixture(autouse=True)
def _isolate_user_preferences(tmp_path_factory, monkeypatch):
    """Keep every test away from the real ``~/.atlastrack/settings.json``.

    Preferences are saved as a *side effect* of ordinary actions - loading a
    project records it in "Load recent" - so a test does not have to set out to
    write them. Constructing a save panel and calling ``load_path`` is enough, and
    :func:`save_app_settings` then writes to the developer's own home directory.

    That is not hypothetical: a test run replaced a real user's recent-projects
    list with a pytest ``tmp_path``, and their "Load recent" menu came back empty
    with a dead entry in it. It had been happening since before the rename - the
    pre-rename ``~/.histo2ccf/settings.json`` carries the same residue.

    Autouse and unconditional, because the tests that need these paths are not the
    ones that do the damage. Tests that patch these names themselves (the rename
    migration tests) still win: their fixture runs after this one.
    """
    from atlastrack import config

    prefs_dir = tmp_path_factory.mktemp("prefs")
    legacy_dir = prefs_dir / "legacy"
    monkeypatch.setattr(config, "_PREFS_DIR", prefs_dir)
    monkeypatch.setattr(config, "_PREFS_FILE", prefs_dir / "settings.json")
    monkeypatch.setattr(config, "_LEGACY_PREFS_DIR", legacy_dir)
    monkeypatch.setattr(config, "_LEGACY_PREFS_FILE", legacy_dir / "settings.json")
