"""Tests for the GUI crash log.

The point of this module is to record crashes that leave nothing behind, so the
test that matters is the subprocess one: it faults for real and checks the
Python stack lands in the file.
"""
from __future__ import annotations

import faulthandler
import subprocess
import sys
import textwrap
import threading

import pytest

from atlastrack.gui import crashlog


@pytest.fixture
def fresh_crashlog(tmp_path, monkeypatch):
    """Give each test its own log file and un-arm the module afterwards."""
    original_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    monkeypatch.setattr(crashlog, "_installed", False)
    monkeypatch.setattr(crashlog, "_log_stream", None)
    path = tmp_path / "crash.log"
    yield path
    stream = crashlog._log_stream
    if stream is not None:
        faulthandler.disable()
        stream.close()
    sys.excepthook = original_hook
    threading.excepthook = original_thread_hook


def test_install_writes_a_session_header(fresh_crashlog):
    path = crashlog.install(log_file=fresh_crashlog)

    assert path == fresh_crashlog
    text = path.read_text(encoding="utf-8")
    assert "session start" in text
    assert "atlastrack" in text
    # Without the interpreter and Qt binding a report is not actionable.
    assert "python" in text
    assert "qt " in text


def test_install_enables_faulthandler(fresh_crashlog):
    faulthandler.disable()
    crashlog.install(log_file=fresh_crashlog)

    assert faulthandler.is_enabled()


def test_install_is_idempotent(fresh_crashlog):
    crashlog.install(log_file=fresh_crashlog)
    first = fresh_crashlog.read_text(encoding="utf-8")
    crashlog.install(log_file=fresh_crashlog)

    assert fresh_crashlog.read_text(encoding="utf-8") == first


def test_install_creates_missing_directories(tmp_path, fresh_crashlog):
    nested = tmp_path / "a" / "b" / "crash.log"

    crashlog.install(log_file=nested)

    assert nested.exists()


def test_excepthook_logs_and_chains(fresh_crashlog):
    seen = []
    sys.excepthook = lambda *args: seen.append(args)
    crashlog.install(log_file=fresh_crashlog)

    try:
        raise ValueError("boom in the matcher")
    except ValueError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)

    text = fresh_crashlog.read_text(encoding="utf-8")
    assert "unhandled ValueError" in text
    assert "boom in the matcher" in text
    # The pre-existing hook (the Qt error dialog) must still run.
    assert len(seen) == 1


def test_thread_excepthook_is_logged(fresh_crashlog):
    crashlog.install(log_file=fresh_crashlog)

    def explode() -> None:
        raise RuntimeError("worker thread died")

    thread = threading.Thread(target=explode, name="deepslice")
    thread.start()
    thread.join()

    text = fresh_crashlog.read_text(encoding="utf-8")
    assert "unhandled RuntimeError in thread deepslice" in text
    assert "worker thread died" in text


def test_note_writes_a_breadcrumb(fresh_crashlog):
    crashlog.install(log_file=fresh_crashlog)

    crashlog.note("matcher: go to section 4")

    assert "matcher: go to section 4" in fresh_crashlog.read_text(encoding="utf-8")


def test_note_is_silent_when_not_installed(tmp_path, monkeypatch):
    """A breadcrumb in a non-GUI context (CLI, tests) must not raise."""
    monkeypatch.setattr(crashlog, "_log_stream", None)

    crashlog.note("no log armed")  # must not raise


def test_install_rolls_over_a_large_log(tmp_path, fresh_crashlog):
    fresh_crashlog.write_bytes(b"x" * (crashlog._MAX_LOG_BYTES + 1))

    crashlog.install(log_file=fresh_crashlog)

    assert fresh_crashlog.with_suffix(".log.1").exists()
    # The new session starts a fresh file rather than appending to a huge one.
    assert fresh_crashlog.stat().st_size < crashlog._MAX_LOG_BYTES


@pytest.mark.qt
def test_input_tracer_records_a_click(qtbot, fresh_crashlog):
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication, QPushButton

    crashlog.install(log_file=fresh_crashlog)
    try:
        assert crashlog.install_input_tracer()
        button = QPushButton("go")
        qtbot.addWidget(button)
        QApplication.instance().sendEvent(
            button,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress, QPointF(1, 1), QPointF(1, 1),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )

        assert "input: click on QPushButton" in fresh_crashlog.read_text(encoding="utf-8")
    finally:
        app = QApplication.instance()
        if app is not None and crashlog._input_tracer is not None:
            app.removeEventFilter(crashlog._input_tracer)
        crashlog._input_tracer = None


def test_native_crash_leaves_the_python_stack(tmp_path):
    """The case the log exists for: an access violation, which kills the process
    outright - no traceback, no exception, no Qt dialog."""
    log = tmp_path / "crash.log"
    script = textwrap.dedent(
        f"""
        import ctypes
        from atlastrack.gui import crashlog

        crashlog.install(log_file=r"{log}")

        def frame_that_should_be_recorded():
            ctypes.string_at(1)

        frame_that_should_be_recorded()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode != 0
    text = log.read_text(encoding="utf-8")
    assert "access violation" in text.lower() or "segmentation fault" in text.lower()
    assert "frame_that_should_be_recorded" in text
