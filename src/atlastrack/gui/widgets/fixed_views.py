"""The 3D window's layout: layer docks on the right, fixed views along the bottom.

The fixed views are the scene seen from the top, the bottom and the side, taken
from the 3D window's own canvas (see ``viz.napari3d.fixed_view_images``) and
refreshed each time the scene is.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

#: Height of the strip of fixed views, in pixels.
_STRIP_HEIGHT = 220


def _crop(rgba: np.ndarray, pad_frac: float = 0.04) -> np.ndarray:
    """The image cut down to what is drawn in it, plus a little margin.

    The canvas is shaped for the main view, so seen from the top or the side the
    brain fills only part of it; uncropped, the strip was mostly black.
    """
    drawn = rgba[..., :3].max(axis=-1) > 8
    if not drawn.any():
        return rgba
    rows, cols = np.nonzero(drawn.any(axis=1))[0], np.nonzero(drawn.any(axis=0))[0]
    pad = int(pad_frac * max(rgba.shape[:2]))
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, rgba.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, rgba.shape[1])
    return rgba[r0:r1, c0:c1]


def _pixmap(rgba: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(_crop(rgba)[..., :4].astype(np.uint8))
    h, w = arr.shape[:2]
    image = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


class _ViewLabel(QLabel):
    """A picture that keeps its aspect ratio as the strip is resized."""

    def __init__(self) -> None:
        super().__init__()
        self._full: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(80, 60)
        self.setStyleSheet("background-color: black;")

    def set_full(self, pixmap: QPixmap | None) -> None:
        self._full = pixmap
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._full is None or self._full.isNull():
            self.clear()
            return
        self.setPixmap(self._full.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))


class FixedViewsWidget(QWidget):
    """Top, bottom and side views of a 3D napari viewer, side by side."""

    def __init__(self, viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from atlastrack.viz.napari3d import FIXED_VIEWS

        self._viewer = viewer
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        self._labels: dict[str, _ViewLabel] = {}
        for name, _direction, _up in FIXED_VIEWS:
            column = QVBoxLayout()
            caption = QLabel(name)
            caption.setAlignment(Qt.AlignCenter)
            column.addWidget(caption)
            label = _ViewLabel()
            column.addWidget(label, 1)
            row.addLayout(column, 1)
            self._labels[name] = label
        self.setMinimumHeight(_STRIP_HEIGHT)

    def refresh_soon(self, delay_ms: int = 400) -> None:
        """Refresh once the window has drawn: a canvas not yet shown captures nothing."""
        QTimer.singleShot(delay_ms, self.refresh)

    def refresh(self) -> None:
        from atlastrack.gui import crashlog
        from atlastrack.viz.napari3d import fixed_view_images

        try:
            if not self._viewer.window._qt_window.isVisible():
                return
            crashlog.note("3D view: fixed views")
            images = fixed_view_images(self._viewer)
        except Exception as exc:  # noqa: BLE001 - the strip must never break the 3D view
            crashlog.note(f"3D view: fixed views failed: {exc}")
            return
        for name, image in images.items():
            self._labels[name].set_full(_pixmap(image))


def arrange_3d_window(viewer) -> FixedViewsWidget | None:
    """Layer list and layer controls on the right; the fixed views along the bottom."""
    try:
        window = viewer.window._qt_window
        qt_viewer = viewer.window._qt_viewer
    except Exception:  # noqa: BLE001 - no window (headless)
        return None
    for dock in (qt_viewer.dockLayerControls, qt_viewer.dockLayerList):
        try:
            window.addDockWidget(Qt.RightDockWidgetArea, dock)
        except Exception:  # noqa: BLE001 - keep napari's place for it
            pass
    strip = FixedViewsWidget(viewer)
    viewer.window.add_dock_widget(strip, area="bottom", name="Fixed views")
    return strip
