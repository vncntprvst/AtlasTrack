"""Crash logging, so a hard crash leaves evidence instead of nothing.

The GUI is a napari/Qt application on top of a stack of C extensions (Qt, vispy
and the OpenGL driver, numpy/scipy, SimpleITK). When one of those faults, the
process dies **instantly**: no traceback, no Qt dialog, the window simply
vanishes. :func:`install` arms three recorders that between them cover every way
this application can die:

* :mod:`faulthandler` - on a segfault / access violation, dumps the Python stack
  of **every thread** at the moment of the fault. This is the one that matters:
  a native crash otherwise leaves only a Windows Event Log entry naming a DLL,
  which does not say what the app was doing.
* a Qt message handler - Qt's own warnings and, critically, ``qFatal``, which
  PyQt calls when a Python exception escapes back into C++ and then aborts.
* ``sys.excepthook`` / ``threading.excepthook`` - ordinary Python exceptions,
  including ones raised in a worker thread where nothing else would show them.

Everything is appended to ``~/.atlastrack/logs/crash.log`` with a session header,
so a user who says "it just closed" has a file to send.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

LOG_DIR = Path.home() / ".atlastrack" / "logs"
LOG_FILE = LOG_DIR / "crash.log"

# Roll the log over past this size so a long session cannot grow it without
# bound. One previous file is kept - enough to still have the crash after the
# user relaunches and sends it.
_MAX_LOG_BYTES = 5 * 1024 * 1024

# The stream faulthandler writes to must stay open and at a stable fd for the
# whole session - faulthandler keeps the fd, not the Python object - so it is
# held here rather than left to the garbage collector.
_log_stream = None
_installed = False


def _write(text: str) -> None:
    if _log_stream is None:
        return
    _log_stream.write(text if text.endswith("\n") else text + "\n")
    _log_stream.flush()


def note(message: str) -> None:
    """Record what the app is about to do (a breadcrumb).

    :mod:`faulthandler` covers the case where the fault is delivered to CPython's
    exception handler. It is not always: a fault raised inside a graphics driver,
    or one that fails fast, can kill the process with the handler armed and
    nothing written - which is exactly what happened on 2026-08-06 (pid 68548,
    access violation in ntdll, empty log).

    Breadcrumbs are immune to that, because the line is already on disk before
    the operation runs. The **last line in the log names the operation that
    killed the process**. Keep them coarse - one per user-visible action or
    expensive step, never per repaint.
    """
    if _log_stream is None:
        return
    _write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {message}")


def _session_header() -> str:
    import platform

    lines = [
        "",
        "=" * 78,
        f"session start  {datetime.now().isoformat(timespec='seconds')}  pid {os.getpid()}",
        f"python   {sys.version.splitlines()[0]}",
        f"exe      {sys.executable}",
        f"platform {platform.platform()}",
    ]
    try:
        from atlastrack import __version__

        lines.append(f"atlastrack {__version__}")
    except Exception:  # pragma: no cover - version import cannot realistically fail
        pass
    for mod in ("napari", "qtpy", "numpy", "scipy", "vispy"):
        try:
            import importlib.metadata as md

            lines.append(f"{mod:9s}{md.version(mod)}")
        except Exception:
            pass
    try:
        import qtpy

        lines.append(f"qt       {qtpy.API_NAME} {qtpy.QT_VERSION}")
    except Exception:
        pass
    # Whether this session can detect heap misuse by a C extension. Set
    # PYTHONMALLOC=debug to turn CPython's guard bytes on: corruption of
    # Python-allocated memory is then reported where it is detected instead of
    # crashing somewhere unrelated later.
    lines.append(
        f"malloc   {os.environ.get('PYTHONMALLOC', 'default')}"
        f"{'  (dev mode)' if sys.flags.dev_mode else ''}"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def log_gl_info() -> None:
    """Append the OpenGL renderer/driver to the log.

    Kept separate from :func:`install` because probing GL needs a live context,
    which only exists once the viewer is up - and a crash *during* the probe
    would then take the log's session header with it.
    """
    try:
        from atlastrack.gui.gl_diagnostics import format_gl_report

        _write("OpenGL:\n" + format_gl_report())
    except Exception as exc:
        _write(f"OpenGL: probe failed ({exc!r})")


_input_tracer = None


def install_input_tracer() -> bool:
    """Breadcrumb every user input event, application-wide.

    Per-call-site breadcrumbs only cover code we thought to instrument, and the
    2026-08-06 crash landed in a **7 second gap** between the last instrumented
    action and the fault - so the log could not say what the user was doing.
    A filter on the QApplication has no gaps: mouse presses, key presses, wheel
    and window closes all leave a line naming the widget that received them.

    Mouse *moves* are deliberately excluded - they would flood the log and bury
    the events that matter. Returns False if there is no application to watch.
    """
    global _input_tracer

    try:
        from qtpy.QtCore import QEvent, QObject
        from qtpy.QtWidgets import QApplication
    except Exception:
        return False

    app = QApplication.instance()
    if app is None or _input_tracer is not None:
        return False

    watched = {
        QEvent.Type.MouseButtonPress: "click",
        QEvent.Type.MouseButtonDblClick: "double-click",
        QEvent.Type.KeyPress: "key",
        QEvent.Type.Wheel: "wheel",
        QEvent.Type.Close: "close",
    }

    class _Tracer(QObject):
        def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt signature)
            # An exception escaping eventFilter - a virtual - makes PyQt call
            # qFatal and abort, which would turn the crash recorder into a
            # second cause of crashes. Nothing in here may raise.
            try:
                name = watched.get(event.type())
                if name is not None:
                    note(f"input: {name} on {type(obj).__name__}")
            except Exception:
                pass
            return False

    _input_tracer = _Tracer()
    app.installEventFilter(_input_tracer)
    return True


def _install_qt_message_handler() -> None:
    try:
        from qtpy.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    names = {
        QtMsgType.QtDebugMsg: "DEBUG",
        QtMsgType.QtInfoMsg: "INFO",
        QtMsgType.QtWarningMsg: "WARNING",
        QtMsgType.QtCriticalMsg: "CRITICAL",
        QtMsgType.QtFatalMsg: "FATAL",
    }

    def _handler(msg_type, context, message) -> None:
        level = names.get(msg_type, str(msg_type))
        # Qt is chatty at debug/info level and that noise would bury the entry
        # that matters; warnings and worse are always worth keeping.
        if level in ("DEBUG", "INFO"):
            return
        where = ""
        fname = getattr(context, "file", None)
        if fname:
            where = f" ({fname}:{getattr(context, 'line', '?')})"
        _write(f"[Qt {level}] {message}{where}")
        if level == "FATAL":
            # qFatal aborts the process the moment this returns, so the Python
            # stack that led here has to be captured now or not at all.
            _write("Python stack at qFatal:")
            faulthandler.dump_traceback(file=_log_stream, all_threads=True)

    qInstallMessageHandler(_handler)


def install(*, log_file: Path | None = None) -> Path:
    """Arm the crash recorders. Returns the log path. Safe to call twice."""
    global _log_stream, _installed

    path = Path(log_file) if log_file is not None else LOG_FILE
    if _installed:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and path.stat().st_size > _MAX_LOG_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass  # a log we cannot roll is still a log we can append to
    # Line-buffered: a crash gives no chance to flush afterwards.
    _log_stream = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    _installed = True

    _write(_session_header())
    faulthandler.enable(file=_log_stream, all_threads=True)
    _install_qt_message_handler()

    previous = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        _write(
            f"[unhandled {exc_type.__name__}] "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
            + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )
        previous(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook

    def _thread_hook(args) -> None:
        _write(
            f"[unhandled {args.exc_type.__name__} in thread "
            f"{getattr(args.thread, 'name', '?')}]\n"
            + "".join(
                traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
            )
        )

    threading.excepthook = _thread_hook
    return path
