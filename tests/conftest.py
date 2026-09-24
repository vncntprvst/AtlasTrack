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


@pytest.fixture(autouse=True)
def _keep_notices_off_the_desktop(monkeypatch):
    """No test may pop a dialog or message box onto the developer's screen.

    The suite runs on the real Qt platform, because napari needs OpenGL and the
    Windows ``offscreen`` platform has none. So a notice raised by the code under
    test appeared on the desktop of whoever ran the tests - and a message box, on
    Windows, may be the *native* dialog, which lingers. It was noticed when a
    test's fake 103.8 MP slide raised "Large image" in front of a user whose own
    project uses a 26 MP image; the help panel's pop-out pages did the same. Tests
    only need these windows to exist, so no dialog is shown.

    **Not** done for widgets in general, and this is measured, not assumed:
    showing every widget with ``WA_DontShowOnScreen`` denies OpenGL views a native
    surface, and Qt aborts the process (``Fatal Python error: Aborted`` part-way
    through test_trajectory_preview.py; the file passes without it).

    Teardown fails a test that still leaves a message box or dialog visible, and
    hides it, so a new leak is reported rather than just seen.
    """
    try:
        from qtpy.QtWidgets import QApplication, QDialog, QMessageBox
    except Exception:  # no Qt in this environment - nothing to guard
        yield
        return

    # Every dialog, message boxes included (QMessageBox is a QDialog). A no-op
    # show never creates a native window, so unlike WA_DontShowOnScreen it cannot
    # starve an OpenGL view of its surface.
    monkeypatch.setattr(QDialog, "show", lambda self: None, raising=False)
    monkeypatch.setattr(QMessageBox, "show", lambda self: None, raising=False)

    yield

    app = QApplication.instance()
    if app is None:
        return
    leaked = [w for w in app.topLevelWidgets()
              if w.isVisible() and isinstance(w, QMessageBox | QDialog)]
    names = [f"{type(w).__name__}({w.windowTitle()!r})" for w in leaked]
    for w in leaked:
        w.hide()  # do not leave it on the screen past this test either
    assert not names, f"test left dialog(s) on the desktop: {names}"
