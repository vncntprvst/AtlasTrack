"""The GL probe must diagnose the GPU, not become the fault it reports."""
from __future__ import annotations

import subprocess
import sys

import pytest

from histo_to_ccf.gui import gl_diagnostics

pytestmark = pytest.mark.qt


def test_probe_keeps_the_application_alive() -> None:
    """The regression that made the probe segfault on every machine.

    ``QApplication([])`` with the result discarded is collected immediately in
    PyQt, leaving ``instance()`` None; the next GL call then dereferences a
    destroyed application and takes the process down with an access violation.
    """
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    gl_diagnostics._probe_gl()

    assert QApplication.instance() is not None
    assert gl_diagnostics._probe_app is not None


def test_probe_subprocess_does_not_crash() -> None:
    """End to end: the child process the GUI actually runs must exit cleanly.

    A crash here is what produced 'No usable OpenGL context could be created at
    all' - a message that blames the user's driver - on healthy hardware.
    """
    result = subprocess.run(
        [sys.executable, "-m", "histo_to_ccf.gui.gl_diagnostics"],
        capture_output=True, text=True, timeout=120,
    )

    assert result.returncode == 0, (
        f"GL probe exited {result.returncode}\n{result.stdout}\n{result.stderr}"
    )


def test_report_never_raises_without_a_display(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise OSError("no display")

    monkeypatch.setattr(subprocess, "run", _boom)

    report = gl_diagnostics.gl_report()

    assert report["ok"] is False
    assert "could not launch GL probe" in report["error"]


def test_format_report_is_actionable_on_failure() -> None:
    text = gl_diagnostics.format_gl_report(
        {**gl_diagnostics._EMPTY_REPORT, "error": "no context"}
    )

    assert "no context" in text
    # A failure report without the remediation steps is not worth printing.
    assert "Reboot" in text
