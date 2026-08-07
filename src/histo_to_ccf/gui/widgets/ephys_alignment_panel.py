"""Landmark alignment on top of the depth-resolved ephys feature panels.

The ephys panels stay fixed and the **anatomy moves against them**: each landmark
pins a depth on the histology track to the depth on the feature axis where its
signature actually appears, and the region column is redrawn through that warp. The
alternative - warping the ephys onto a fixed anatomy - looks equivalent but is not,
because it makes the measured data the thing that visibly bends.

The maths is entirely in :mod:`histo_to_ccf.ephys.landmarks` (headless, tested); this
widget is the handles, the buttons and the diagnostic plot around it.

The fit-quality plot below the panels is the honest summary: a straight diagonal is
"no correction", and any large departure from it means a stretch that the histology
does not support and that should be defensible before it is applied.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.ephys.landmarks import (
    AlignmentHistory,
    LandmarkCrossingError,
    Landmarks,
    segment_scales,
)
from histo_to_ccf.gui.widgets.ephys_features_view import EphysFeaturesView, pyqtgraph_available

if TYPE_CHECKING:
    from histo_to_ccf.ephys.penetration import PenetrationProfile

_LANDMARK_PEN = (255, 80, 80)
# A double-click within this fraction of the visible depth span of an existing
# landmark removes it instead of adding another on top of it.
_HIT_FRACTION = 0.015


class EphysAlignmentPanel(QWidget):
    """Feature panels, atlas region column, draggable landmarks and their history."""

    landmarksChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ok = pyqtgraph_available()
        self._view = EphysFeaturesView()
        self._history: AlignmentHistory | None = None
        self._lines: list = []
        self._fit_plot = None
        self._suspend_line_signals = False
        self._build_ui()

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.addWidget(self._view, 1)

        if self._ok:
            import pyqtgraph as pg

            self._fit_widget = pg.GraphicsLayoutWidget()
            self._fit_widget.setMaximumHeight(190)
            self._fit_plot = self._fit_widget.addPlot(title="Fit: track vs feature depth")
            self._fit_plot.setLabel("bottom", "feature depth", units="µm")
            self._fit_plot.setLabel("left", "track depth", units="µm")
            self._fit_plot.showGrid(x=True, y=True, alpha=0.2)
            self._fit_plot.addLegend(offset=(-10, 10))
            root.addWidget(self._fit_widget)

            scene = self._view.region_plot.scene()
            scene.sigMouseClicked.connect(self._on_scene_click)

        row = QHBoxLayout()
        self._add_btn = QPushButton("Add landmark")
        self._add_btn.setToolTip(
            "Add a landmark at mid-depth. Faster: double-click the region column "
            "where a boundary should sit, and double-click a landmark to remove it."
        )
        self._add_btn.clicked.connect(self._add_mid)
        self._clear_btn = QPushButton("Clear landmarks")
        self._clear_btn.clicked.connect(self.clear_landmarks)
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.setToolTip("Step back through the last 10 landmark states.")
        self._prev_btn.clicked.connect(self.undo)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self.redo)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Discard every landmark and the history with them.")
        self._reset_btn.clicked.connect(self.reset)
        for btn in (self._add_btn, self._clear_btn, self._prev_btn, self._next_btn,
                    self._reset_btn):
            row.addWidget(btn)

        row.addSpacing(12)
        row.addWidget(QLabel("Beyond the landmarks:"))
        self._mode_box = QComboBox()
        self._mode_box.addItem("shift only (uniform)", "uniform")
        self._mode_box.addItem("continue the trend (linear, needs 3+)", "linear")
        self._mode_box.setToolTip(
            "How the depths outside the outermost landmark are treated.\n"
            "'shift only' translates them by that landmark's offset and never "
            "stretches them - outside what you pinned there is no evidence for a "
            "scale change.\n'continue the trend' extends the regression through all "
            "your landmarks, for a track believed to be uniformly scaled."
        )
        self._mode_box.currentIndexChanged.connect(lambda _i: self._redraw())
        row.addWidget(self._mode_box)
        row.addStretch()
        root.addLayout(row)

        self._status = QLabel("No track loaded.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._status)
        self._update_buttons()

    # -- content ---------------------------------------------------------

    def view(self) -> EphysFeaturesView:
        """The feature panels, for callers that want to drive them directly."""
        return self._view

    def set_penetration(self, profile: PenetrationProfile, *,
                        track_length_um: float = 0.0) -> None:
        self._view.set_profile(profile, track_length_um=track_length_um)

    def set_track(self, atlas, tip_ccf_um, entry_ccf_um) -> None:
        """Load the anatomy and start a fresh landmark set spanning the track."""
        self._view.set_track(atlas, tip_ccf_um, entry_ccf_um)
        length = 0.0
        if tip_ccf_um is not None and entry_ccf_um is not None:
            length = float(
                np.linalg.norm(np.asarray(tip_ccf_um) - np.asarray(entry_ccf_um))
            )
        if length <= 0.0:
            # No registered track: fall back to whatever depth the recordings span, so
            # landmarks can still be placed and reviewed.
            bands = self._view.bands()
            length = max((b.bottom_um for b in bands), default=0.0)
        if length <= 0.0:
            self._history = None
            self._status.setText(
                "This shank has no registered tip/entry, so there is no track to align "
                "to. Register the sections first."
            )
            self._update_buttons()
            return
        self._history = AlignmentHistory(Landmarks.identity(0.0, length))
        self._redraw()

    def landmarks(self) -> Landmarks | None:
        """The current landmark state, or ``None`` before a track is loaded."""
        return self._history.current() if self._history is not None else None

    def restore_landmarks(self, feature_um, track_um) -> None:
        """Reload a stored landmark set, rebased onto the current track extent.

        Rebasing rather than restoring the stored end points matters when the shank
        has been re-registered since: the user's pairs are what they decided, the end
        points are bookkeeping, and keeping stale ones would shift the tails without
        anything on screen saying so.
        """
        feature_um = list(feature_um or [])
        track_um = list(track_um or [])
        if len(feature_um) < 2 or len(feature_um) != len(track_um):
            return
        stored = Landmarks(np.asarray(feature_um, dtype=float),
                           np.asarray(track_um, dtype=float))
        base = self.landmarks()
        state = Landmarks.identity(*base.track_extent_um) if base is not None else stored
        if base is not None:
            for feature, track in stored.user_pairs():
                state = state.added(feature, track)
        self._history = AlignmentHistory(state)
        self._redraw()

    def extremes_mode(self) -> str:
        return str(self._mode_box.currentData() or "uniform")

    def set_extremes_mode(self, mode: str) -> None:
        index = self._mode_box.findData(mode)
        if index < 0:
            raise ValueError(f"unknown extremes mode {mode!r}")
        self._mode_box.setCurrentIndex(index)

    # -- edits -----------------------------------------------------------

    def add_landmark_at(self, feature_um: float) -> None:
        """Pin whatever anatomy is currently drawn at ``feature_um`` to that depth."""
        current = self.landmarks()
        if current is None:
            return
        track = float(np.asarray(current.to_track(feature_um, self.extremes_mode())).ravel()[0])
        self._commit(current.added(float(feature_um), track))

    def remove_landmark(self, index: int) -> None:
        current = self.landmarks()
        if current is None:
            return
        self._commit(current.removed(index))

    def clear_landmarks(self) -> None:
        current = self.landmarks()
        if current is not None and current.n_user:
            self._commit(current.cleared())

    def move_landmark(self, index: int, feature_um: float) -> None:
        """Drag landmark ``index`` to a new depth on the feature axis."""
        current = self.landmarks()
        if current is None:
            return
        self._commit(current.moved(index, feature_um=float(feature_um)))

    def undo(self) -> None:
        if self._history is not None and self._history.previous() is not None:
            self._redraw()
            self.landmarksChanged.emit()

    def redo(self) -> None:
        if self._history is not None and self._history.next() is not None:
            self._redraw()
            self.landmarksChanged.emit()

    def reset(self) -> None:
        if self._history is None:
            return
        extent = self._history.current().track_extent_um
        self._history.reset(Landmarks.identity(*extent))
        self._redraw()
        self.landmarksChanged.emit()

    def _commit(self, state: Landmarks) -> None:
        if self._history is None:
            return
        self._history.push(state)
        self._redraw()
        self.landmarksChanged.emit()

    def _add_mid(self) -> None:
        current = self.landmarks()
        if current is None:
            return
        top, bottom = current.track_extent_um
        self.add_landmark_at(0.5 * (top + bottom))

    # -- interaction -----------------------------------------------------

    def _on_scene_click(self, event) -> None:
        """Double-click on the region column: add a landmark, or remove one under it."""
        if not event.double() or self._history is None:
            return
        plot = self._view.region_plot
        if not plot.sceneBoundingRect().contains(event.scenePos()):
            return
        y = float(plot.getViewBox().mapSceneToView(event.scenePos()).y())
        hit = self._landmark_near(y)
        if hit is None:
            self.add_landmark_at(y)
        else:
            self.remove_landmark(hit)
        event.accept()

    def _landmark_near(self, feature_um: float) -> int | None:
        current = self.landmarks()
        if current is None or current.n_user == 0:
            return None
        lo, hi = self._view.region_plot.getViewBox().viewRange()[1]
        tolerance = max(abs(hi - lo) * _HIT_FRACTION, 1.0)
        features = np.array([f for f, _ in current.user_pairs()])
        idx = int(np.argmin(np.abs(features - feature_um)))
        return idx if abs(features[idx] - feature_um) <= tolerance else None

    def _on_line_dragged(self, line) -> None:
        """A handle was released. Accept the move, or snap it back and say why.

        Neither branch rebuilds the handles. Destroying a ``QGraphicsItem`` from
        inside its own signal handler is a good way to earn a native crash, and it
        is not needed here: a legal drag can never reorder the landmarks, because
        reordering them in feature space while their track depths stay put *is* a
        crossing, and crossings are refused. So the indices stay valid.
        """
        if self._suspend_line_signals or self._history is None:
            return
        index = int(line.landmark_index)
        current = self.landmarks()
        if current is None or not 0 <= index < current.n_user:
            return
        try:
            moved = current.moved(index, feature_um=float(line.value()))
        except LandmarkCrossingError as exc:
            # Silently re-pairing the landmarks is the IBL behaviour we reject.
            self._status.setText(f"Landmark not moved: {exc}")
            self._set_line_value(line, current.user_pairs()[index][0])
            return
        self._history.push(moved)
        self._redraw(rebuild_lines=False)
        self.landmarksChanged.emit()

    def _set_line_value(self, line, feature_um: float) -> None:
        """Move a handle without it reporting the move back to us."""
        self._suspend_line_signals = True
        try:
            line.setValue(float(feature_um))
        finally:
            self._suspend_line_signals = False

    # -- drawing ---------------------------------------------------------

    def _redraw(self, *, rebuild_lines: bool = True) -> None:
        current = self.landmarks()
        self._view.set_landmarks(current, mode=self.extremes_mode())
        if rebuild_lines:
            self._draw_lines()
        self._draw_fit()
        self._update_status()
        self._update_buttons()

    def _draw_lines(self) -> None:
        if not self._ok:
            return
        import pyqtgraph as pg

        plot = self._view.region_plot
        self._suspend_line_signals = True
        try:
            for line in self._lines:
                plot.removeItem(line)
            self._lines = []
            current = self.landmarks()
            if current is None:
                return
            for i, (feature_um, _track_um) in enumerate(current.user_pairs()):
                line = pg.InfiniteLine(
                    pos=feature_um, angle=0, movable=True,
                    pen=pg.mkPen(_LANDMARK_PEN, width=2),
                    hoverPen=pg.mkPen(255, 200, 60, width=3),
                )
                line.landmark_index = i
                line.setZValue(20)
                line.setCursor(Qt.CursorShape.SizeVerCursor)
                line.sigPositionChangeFinished.connect(self._on_line_dragged)
                plot.addItem(line)
                self._lines.append(line)
        finally:
            self._suspend_line_signals = False

    def _draw_fit(self) -> None:
        if not self._ok or self._fit_plot is None:
            return
        import pyqtgraph as pg

        self._fit_plot.clear()
        current = self.landmarks()
        if current is None:
            return
        top, bottom = current.track_extent_um
        x = np.linspace(top, bottom, 200)
        self._fit_plot.plot(
            x, x, pen=pg.mkPen((130, 130, 130), width=1, style=Qt.PenStyle.DashLine),
            name="no correction",
        )
        self._fit_plot.plot(
            x, np.asarray(current.to_track(x, self.extremes_mode())),
            pen=pg.mkPen((90, 200, 250), width=2), name="alignment",
        )
        if current.n_user:
            features = np.array([f for f, _ in current.user_pairs()])
            tracks = np.array([t for _, t in current.user_pairs()])
            self._fit_plot.addItem(
                pg.ScatterPlotItem(
                    x=features, y=tracks, size=9, brush=pg.mkBrush(*_LANDMARK_PEN),
                    pen=None,
                )
            )

    def _update_status(self) -> None:
        current = self.landmarks()
        if current is None:
            self._status.setText("No track loaded.")
            return
        if current.n_user == 0:
            self._status.setText(
                "No landmarks: the region column shows the histology track unchanged. "
                "Double-click it where a boundary belongs."
            )
            return
        feature, track = current.fit(self.extremes_mode())
        _edges, scale = segment_scales(feature, track)
        interior = scale[1:-1] if scale.size > 2 else scale
        worst = float(np.nanmax(np.abs(interior - 1.0))) if interior.size else 0.0
        parts = [
            f"{current.n_user} landmark(s)",
            f"offset {current.offset_um():+.0f} µm",
            f"largest local stretch {1.0 + worst:.2f}x",
        ]
        if worst > 0.5:
            parts.append(
                "- that is a big local stretch; check the landmarks are not fighting "
                "each other before applying"
            )
        self._status.setText("  ·  ".join(parts))

    def _update_buttons(self) -> None:
        has_track = self._history is not None
        current = self.landmarks()
        self._add_btn.setEnabled(has_track)
        self._clear_btn.setEnabled(bool(current is not None and current.n_user))
        self._reset_btn.setEnabled(has_track)
        self._prev_btn.setEnabled(bool(has_track and self._history.can_undo))
        self._next_btn.setEnabled(bool(has_track and self._history.can_redo))

    # -- reporting -------------------------------------------------------

    def status_text(self) -> str:
        return self._status.text()


class EphysLandmarkDialog(QDialog):
    """The alignment panel for one shank, with load/store against the project.

    Deliberately usable before any recording has been registered: the atlas region
    column only needs a registered shank, so the anatomy along the track can be read
    (and landmarks placed against it) while the recording manager is still to come.
    """

    def __init__(self, state, probe_idx: int, shank_idx: int, *, profile=None,
                 on_applied=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ephys landmark alignment")
        self.resize(1180, 820)
        self._state = state
        self._on_applied = on_applied
        self._shank = state.project.probes[probe_idx].shanks[shank_idx]

        root = QVBoxLayout(self)
        self.panel = EphysAlignmentPanel()
        root.addWidget(self.panel, 1)

        if profile is None:
            from histo_to_ccf.ephys.penetration import PenetrationProfile

            profile = PenetrationProfile()
        track_length = 0.0
        if self._shank.tip_ccf_um is not None and self._shank.entry_ccf_um is not None:
            track_length = float(np.linalg.norm(
                np.asarray(self._shank.tip_ccf_um) - np.asarray(self._shank.entry_ccf_um)
            ))
        self.panel.set_penetration(profile, track_length_um=track_length)
        self.panel.set_track(state.atlas, self._shank.tip_ccf_um, self._shank.entry_ccf_um)
        if self._shank.ephys is not None:
            self.panel.restore_landmarks(
                self._shank.ephys.feature_um, self._shank.ephys.track_um
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(buttons)

    def apply(self) -> None:
        """Store the landmark arrays on the shank, leaving the rest of it alone."""
        from datetime import datetime

        from histo_to_ccf.project.schema import EphysAlignment

        landmarks = self.panel.landmarks()
        if landmarks is None:
            return
        if self._shank.ephys is None:
            self._shank.ephys = EphysAlignment()
        # Only the landmark arrays and the stamp: ``anchors`` and ``channel_ccf_um``
        # belong to the older tip-referenced alignment and writing depth-below-surface
        # numbers into them would silently flip the track.
        self._shank.ephys.feature_um = [float(v) for v in landmarks.feature_um]
        self._shank.ephys.track_um = [float(v) for v in landmarks.track_um]
        self._shank.ephys.created_at = datetime.now().isoformat(timespec="seconds")
        if self._on_applied is not None:
            self._on_applied()
        self.close()
