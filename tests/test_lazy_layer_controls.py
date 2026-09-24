"""napari's per-layer control panels are built only while their dock is shown.

The app hides that dock, and the atlas overlay adds a layer per section, so napari
was building a full Qt controls panel per section for nobody to see. The shim swaps
which of napari's own handlers answer the layer events - napari's source is not
changed - and these tests pin both halves of that: nothing is built while hidden,
and everything still works, including showing the dock later.

If a napari upgrade moves the internals this relies on, the first test fails
rather than the saving quietly disappearing.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.qt


@pytest.fixture
def viewer():
    import napari

    v = napari.Viewer(show=False)
    yield v
    v.close()


def _install(viewer) -> bool:
    from atlastrack.gui.app import _defer_layer_controls

    return _defer_layer_controls(viewer)


def _container(viewer):
    return viewer.window._qt_viewer.controls


def test_it_installs_on_this_napari(viewer) -> None:
    """A napari upgrade that moves these internals must fail here, loudly."""
    assert _install(viewer) is True


def test_no_controls_are_built_while_the_dock_is_hidden(viewer) -> None:
    _install(viewer)
    for i in range(5):
        viewer.add_labels(np.zeros((20, 20), np.uint8), name=f"overlay {i}")

    assert _container(viewer).widgets == {}, "panels were built for a hidden dock"


def test_showing_the_dock_builds_the_active_layers_controls(viewer) -> None:
    """napari's Window menu can reveal the dock; it must not come up empty."""
    _install(viewer)
    viewer.add_labels(np.zeros((20, 20), np.uint8), name="a")
    b = viewer.add_labels(np.zeros((20, 20), np.uint8), name="b")
    viewer.layers.selection.active = b
    dock = viewer.window._qt_viewer.dockLayerControls

    dock.visibilityChanged.emit(True)

    container = _container(viewer)
    assert b in container.widgets
    assert container.currentWidget() is container.widgets[b]


def test_removing_layers_is_safe_with_or_without_controls(viewer) -> None:
    """napari's own _remove indexes the dict and would KeyError on a skipped layer."""
    _install(viewer)
    a = viewer.add_labels(np.zeros((20, 20), np.uint8), name="a")
    b = viewer.add_labels(np.zeros((20, 20), np.uint8), name="b")
    viewer.layers.selection.active = b
    viewer.window._qt_viewer.dockLayerControls.visibilityChanged.emit(True)  # builds b's

    viewer.layers.remove(a)  # never had controls
    viewer.layers.remove(b)  # had controls

    assert _container(viewer).widgets == {}


def test_switching_the_active_layer_while_hidden_is_harmless(viewer) -> None:
    _install(viewer)
    layers = [viewer.add_labels(np.zeros((20, 20), np.uint8), name=f"l{i}") for i in range(3)]
    for layer in [*layers, None]:
        viewer.layers.selection.active = layer

    container = _container(viewer)
    assert container.currentWidget() is container.empty_widget


def test_it_leaves_napari_untouched_when_the_internals_are_missing(viewer, monkeypatch) -> None:
    """Degrade to napari's normal eager behaviour, never to a broken viewer."""
    # Something only the shim reads, so napari's own wiring cannot notice.
    def _gone(_self):
        raise AttributeError("dockLayerControls")

    monkeypatch.setattr(type(viewer.window._qt_viewer), "dockLayerControls", property(_gone))

    assert _install(viewer) is False
    viewer.add_labels(np.zeros((20, 20), np.uint8), name="still eager")
    assert len(_container(viewer).widgets) == 1, "napari's own wiring was disturbed"


def test_launch_installs_it() -> None:
    import inspect

    from atlastrack.gui import app

    assert "_defer_layer_controls(viewer)" in inspect.getsource(app.launch)
