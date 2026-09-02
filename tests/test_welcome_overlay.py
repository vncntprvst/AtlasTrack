"""The figure that replaces napari's welcome screen on the empty canvas.

Two things here are easy to break without noticing. The overlay has to be parented to
the *canvas*, because ``QtViewer`` is a QSplitter and a child added to it silently
becomes a splitter pane laid out below the canvas instead of an overlay on top of it.
And the steps it draws have to stay in step with the actual tab order, or the first
thing a new user reads is a wrong instruction.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.gui.widgets.welcome_overlay import (
    APP_TITLE,
    STEPS,
    WelcomeOverlayWidget,
)

pytestmark = pytest.mark.qt


def _viewer():
    import napari

    return napari.Viewer(show=False)


# ---------------------------------------------------------------------------
# The figure itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1280, 720), (900, 560), (620, 480), (300, 240),
                                  (120, 90)])
def test_it_paints_at_any_size_without_raising(qtbot, size):
    """The layout is arithmetic on the widget size, so tiny sizes are the risk."""
    overlay = WelcomeOverlayWidget(theme="dark")
    qtbot.addWidget(overlay)
    overlay.resize(*size)

    pixmap = overlay.grab()

    assert not pixmap.isNull()
    assert (pixmap.width(), pixmap.height()) == size


def test_it_paints_an_opaque_background_over_the_canvas(qtbot):
    """It stands in for the canvas; a transparent one would show napari's black."""
    overlay = WelcomeOverlayWidget(theme="dark")
    qtbot.addWidget(overlay)
    overlay.resize(400, 300)

    corner = overlay.grab().toImage().pixelColor(2, 2)

    assert corner.alpha() == 255
    assert corner.name() == "#262930"  # napari dark theme background


def test_an_unknown_theme_still_paints(qtbot):
    """A themed figure is nice; a drawn one is required."""
    overlay = WelcomeOverlayWidget(theme="no-such-theme")
    qtbot.addWidget(overlay)
    overlay.resize(800, 500)

    assert not overlay.grab().isNull()


def test_it_does_not_swallow_canvas_clicks(qtbot):
    """It covers the canvas, so it must be invisible to the mouse."""
    from qtpy.QtCore import Qt

    overlay = WelcomeOverlayWidget(theme="dark")
    qtbot.addWidget(overlay)

    assert overlay.testAttribute(Qt.WA_TransparentForMouseEvents)


# ---------------------------------------------------------------------------
# Staying honest about the workflow
# ---------------------------------------------------------------------------


def test_the_steps_are_the_tabs_in_order(qtbot):
    """Move a tab and this fails - the figure would otherwise teach the wrong path."""
    from qtpy.QtWidgets import QTabWidget

    from atlastrack.gui.app import _build_panel

    viewer = _viewer()
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        tabs = panel.findChild(QTabWidget)
        labels = [tabs.tabText(i) for i in range(tabs.count())]

        assert labels == [name for name, _ in STEPS]
    finally:
        viewer.close()


def test_the_window_title_and_the_figure_use_one_name(qtbot):
    """The queued rename must not leave the title bar and the figure disagreeing."""
    viewer = _viewer()
    try:
        from atlastrack.gui.app import _install_welcome_overlay

        viewer.title = APP_TITLE
        assert _install_welcome_overlay(viewer) is not None
        assert viewer.title == APP_TITLE
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Installation into the viewer
# ---------------------------------------------------------------------------


def test_it_is_parented_to_the_canvas_not_the_splitter(qtbot):
    """``QtViewer`` is a QSplitter: a child of it becomes a pane, not an overlay."""
    from qtpy.QtWidgets import QSplitter

    from atlastrack.gui.app import _install_welcome_overlay

    viewer = _viewer()
    try:
        overlay = _install_welcome_overlay(viewer)
        canvas = viewer.window._qt_viewer.canvas.native

        assert isinstance(viewer.window._qt_viewer, QSplitter)  # the trap
        assert overlay.parent() is canvas
        assert overlay.geometry() == canvas.rect()
    finally:
        viewer.close()


def test_napari_own_welcome_screen_is_switched_off(qtbot):
    """Two welcome screens on one canvas is a redraw bug waiting to happen."""
    from atlastrack.gui.app import _install_welcome_overlay

    viewer = _viewer()
    try:
        _install_welcome_overlay(viewer)

        assert viewer.welcome_screen.visible is False
    finally:
        viewer.close()


def test_it_shows_only_while_the_canvas_is_empty(qtbot):
    """Same rule as napari's, so closing a project brings the figure back."""
    from atlastrack.gui.app import _install_welcome_overlay

    viewer = _viewer()
    try:
        overlay = _install_welcome_overlay(viewer)
        assert overlay.isVisibleTo(overlay.parent())

        viewer.add_image(np.zeros((8, 8), dtype=float), name="slide")
        assert not overlay.isVisibleTo(overlay.parent())

        viewer.layers.clear()
        assert overlay.isVisibleTo(overlay.parent())
    finally:
        viewer.close()


def test_it_tracks_the_canvas_size(qtbot):
    """No resize signal to connect to, so this goes through an event filter.

    The event is sent rather than left to ``resize()`` alone: Qt defers a hidden
    widget's resize event until it is shown, and this viewer is never shown.
    """
    from qtpy.QtGui import QResizeEvent
    from qtpy.QtWidgets import QApplication

    from atlastrack.gui.app import _install_welcome_overlay

    viewer = _viewer()
    try:
        overlay = _install_welcome_overlay(viewer)
        canvas = viewer.window._qt_viewer.canvas.native
        before = canvas.size()
        canvas.resize(640, 480)
        QApplication.sendEvent(canvas, QResizeEvent(canvas.size(), before))

        assert overlay.size() == canvas.size()
    finally:
        viewer.close()


def test_a_canvas_resized_before_it_is_shown_is_caught_on_show(qtbot):
    """Qt holds back a hidden widget's resize event, so Show has to sync too."""
    from qtpy.QtGui import QShowEvent
    from qtpy.QtWidgets import QApplication

    from atlastrack.gui.app import _install_welcome_overlay

    viewer = _viewer()
    try:
        overlay = _install_welcome_overlay(viewer)
        canvas = viewer.window._qt_viewer.canvas.native
        canvas.resize(512, 384)  # no resize event delivered: the canvas is hidden
        QApplication.sendEvent(canvas, QShowEvent())

        assert overlay.size() == canvas.size()
    finally:
        viewer.close()
