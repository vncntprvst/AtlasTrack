"""OpenGL diagnostics - identify *why* the napari GUI can't render.

The GUI needs a working hardware OpenGL context (≥ 2.1, with framebuffer-object
support). When that's missing - a stale/updated GPU driver, a Remote Desktop
session (GDI / OpenGL 1.1, no FBOs), or the app landing on a disabled GPU - napari
spews ``QOpenGLFramebufferObject: Unsupported framebuffer format`` and renders
nothing usable.

This module creates a throwaway **offscreen** GL context, asks the driver what it
is, and turns that into a plain-language diagnosis plus the fix. Everything is
wrapped so probing never itself crashes.
"""
from __future__ import annotations

DRIVER_REMEDIATION = """\
How to restore hardware OpenGL (in order of likelihood):
  1. Reboot - transient driver/context faults usually clear.
  2. Remote Desktop? RDP only exposes GDI / OpenGL 1.1 (no framebuffer objects),
     which is exactly this error. Run on the physical console, or enable
     'Use hardware graphics adapters for all Remote Desktop Services sessions'
     (gpedit.msc -> Computer Config -> Admin Templates -> Windows Components ->
     Remote Desktop Services -> Remote Session Environment).
  3. GPU driver: update via the vendor app (NVIDIA/AMD/Intel). If a recent
     auto-update broke it, roll back: Device Manager -> Display adapters ->
     (your GPU) -> Properties -> Driver -> Roll Back Driver.
  4. Dual-GPU laptop: force Python/histo2ccf onto the discrete GPU
     (Windows Settings -> Display -> Graphics, or the vendor control panel)."""


_EMPTY_REPORT = {
    "ok": False, "vendor": None, "renderer": None,
    "version": None, "glsl": None, "error": None,
}


def gl_report() -> dict:
    """Probe the GL context in a **subprocess** and return the diagnostic dict.

    Creating a GL context on a severely broken driver can segfault natively
    (Python ``try/except`` can't catch that), so the probe runs in a child
    process. A crash there is turned into an actionable ``error`` rather than
    taking down the caller. Never raises.
    """
    import json
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "histo_to_ccf.gui.gl_diagnostics"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        return {**_EMPTY_REPORT, "error": f"could not launch GL probe: {exc}"}

    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:  # noqa: BLE001
                break
    return {
        **_EMPTY_REPORT,
        "error": (
            f"the GL probe process crashed (exit {result.returncode}) - a usable "
            "OpenGL context could not be created at all"
        ),
    }


def _probe_gl() -> dict:
    """Actually create an offscreen GL context and query it. Runs in the child."""
    report: dict = dict(_EMPTY_REPORT)
    try:
        from qtpy.QtGui import QOffscreenSurface, QOpenGLContext
        from qtpy.QtWidgets import QApplication

        if QApplication.instance() is None:
            QApplication([])  # noqa: F841 - kept alive by Qt

        ctx = QOpenGLContext()
        if not ctx.create():
            report["error"] = "QOpenGLContext.create() failed - no GL context available"
            return report
        surface = QOffscreenSurface()
        surface.create()
        if not ctx.makeCurrent(surface):
            report["error"] = "makeCurrent() failed - no usable GL surface"
            return report
        try:
            from OpenGL import GL

            def _s(name) -> str | None:
                try:
                    val = GL.glGetString(name)
                    if not val:
                        return None
                    return val.decode() if isinstance(val, bytes) else bytes(val).decode()
                except Exception as exc:  # noqa: BLE001
                    return f"<error: {exc}>"

            report["vendor"] = _s(GL.GL_VENDOR)
            report["renderer"] = _s(GL.GL_RENDERER)
            report["version"] = _s(GL.GL_VERSION)
            report["glsl"] = _s(GL.GL_SHADING_LANGUAGE_VERSION)
            report["ok"] = bool(report["renderer"])
        finally:
            ctx.doneCurrent()
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _interpret(report: dict) -> str:
    renderer = (report.get("renderer") or "").lower()
    if report.get("error") or not report.get("ok"):
        return (
            "-> No usable OpenGL context could be created at all. This is a GPU /"
            " driver / session problem, not a histo2ccf bug."
        )
    software_markers = ("basic render", "gdi generic", "llvmpipe", "software", "swiftshader")
    if any(m in renderer for m in software_markers):
        return (
            "-> A SOFTWARE / fallback renderer is active (no real GPU driver). On"
            " Windows this almost always means Remote Desktop or a missing/broken"
            " GPU driver - napari needs hardware OpenGL with framebuffer objects."
        )
    return (
        "-> A hardware GPU renderer is reported. If the GUI still fails with"
        " 'Unsupported framebuffer format', the driver is likely faulty/outdated"
        " - update or roll it back."
    )


def format_gl_report(report: dict | None = None) -> str:
    """Human-readable GL diagnostic + remediation text."""
    if report is None:
        report = gl_report()
    lines = [
        "OpenGL diagnostic",
        "-----------------",
        f"  vendor   : {report.get('vendor')}",
        f"  renderer : {report.get('renderer')}",
        f"  version  : {report.get('version')}",
        f"  GLSL     : {report.get('glsl')}",
    ]
    if report.get("error"):
        lines.append(f"  error    : {report['error']}")
    lines += ["", _interpret(report), "", DRIVER_REMEDIATION]
    return "\n".join(lines)


def report_launch_failure(exc: BaseException) -> None:
    """Print an actionable message when the napari Viewer can't be created."""
    print("\nhisto2ccf: the GUI failed to start - the GPU/OpenGL context is unusable.\n")
    print(f"Underlying error: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}\n")
    print(format_gl_report())


if __name__ == "__main__":
    # Child entry point used by gl_report(): emit the probe result as JSON.
    import json

    print(json.dumps(_probe_gl()))
