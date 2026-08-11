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

import contextlib
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
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
        self._shift_buttons: list = []
        # Handle positions being edited: [feature_depth, anatomy_depth] per landmark.
        # Held apart from the applied fit so the two can disagree until Align.
        self._pending: list[list[float]] = []
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
        row.addWidget(QLabel("Show:"))
        self._display_box = QComboBox()
        self._display_box.setToolTip(
            "Which ephys feature the left panel shows. Only displays with data behind "
            "them are offered - an empty panel would say 'no activity' when it means "
            "'not loaded'."
        )
        self._display_box.currentIndexChanged.connect(self._on_display_changed)
        row.addWidget(self._display_box)
        row.addSpacing(12)

        self._add_btn = QPushButton("Add landmark")
        self._add_btn.setToolTip(
            "Drop a landmark: a red bar on the ephys panel and a blue one on the "
            "region column, both at the same depth.\n"
            "Drag the red bar onto the feature you can see, and the blue bar onto the "
            "boundary it should be - they are allowed to differ. Then press Align.\n"
            "Faster: double-click either panel to drop one there; double-click a bar "
            "to remove it."
        )
        self._add_btn.clicked.connect(self._add_mid)
        self._align_btn = QPushButton("Align")
        self._align_btn.setToolTip(
            "Apply the landmarks: move the anatomy until each blue bar meets its red "
            "one. With one landmark that is a shift; with several, the intervals "
            "between them stretch."
        )
        self._align_btn.clicked.connect(self.align)
        self._clear_btn = QPushButton("Clear landmarks")
        self._clear_btn.clicked.connect(self.clear_landmarks)
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.setToolTip("Step back through the last 10 landmark states.")
        self._prev_btn.clicked.connect(self.undo)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self.redo)
        self._snap_btn = QPushButton("Snap to no correction")
        self._snap_btn.setToolTip(
            "Put every landmark back on the identity line, keeping the landmarks "
            "themselves. Unlike 'Clear', the depths you decided were interesting stay "
            "pinned - they just stop warping anything."
        )
        self._snap_btn.clicked.connect(self.snap_to_no_correction)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Discard every landmark and the history with them.")
        self._reset_btn.clicked.connect(self.reset)
        for btn in (self._add_btn, self._align_btn, self._clear_btn, self._snap_btn,
                    self._prev_btn, self._next_btn, self._reset_btn):
            row.addWidget(btn)

        # Shifting the whole track is the commonest correction by far - the insertion
        # zero is uncertain by a couple of hundred µm - and until now the only way to
        # express it was to drag landmarks, which stretches instead of translating.
        row.addSpacing(12)
        row.addWidget(QLabel("Shift all:"))
        for label, delta in (("↑ 100", -100.0), ("↓ 100", 100.0)):
            btn = QPushButton(label)
            btn.setToolTip(
                "Move the whole alignment 100 µm without stretching it: every landmark "
                "keeps its spacing, the anatomy just sits deeper or shallower."
            )
            btn.clicked.connect(lambda _checked=False, d=delta: self.shift(d))
            row.addWidget(btn)
            self._shift_buttons.append(btn)

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

        self._per_freq_check = QCheckBox("Normalise LFP per frequency")
        self._per_freq_check.setChecked(True)
        self._per_freq_check.setToolTip(
            "Scale each frequency column independently, so the 1/f gradient across "
            "frequencies stops dominating the image and the depth-dependent changes - "
            "the transitions you align to - stand out.\n"
            "Turn it off for the honest picture of absolute power."
        )
        self._per_freq_check.toggled.connect(
            lambda on: self._view.set_lfp_normalisation(per_freq=on)
        )
        row.addWidget(self._per_freq_check)
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
        self.refresh_display_modes()

    def refresh_display_modes(self) -> None:
        """Offer only the displays that have data behind them."""
        labels = {"lfp": "LFP power", "spikes": "Spikes", "rate": "Firing rate"}
        modes = self._view.available_modes()
        current = self._display_box.currentData()
        self._display_box.blockSignals(True)
        self._display_box.clear()
        for mode in modes:
            self._display_box.addItem(labels[mode], mode)
        if not modes:
            self._display_box.addItem("no ephys loaded", None)
        self._display_box.setEnabled(len(modes) > 1)
        index = self._display_box.findData(current)
        self._display_box.setCurrentIndex(max(index, 0))
        self._display_box.blockSignals(False)
        self._on_display_changed()

    def _on_display_changed(self, *_args) -> None:
        mode = self._display_box.currentData()
        if mode:
            self._view.set_display_mode(mode)
        self._per_freq_check.setEnabled(mode == "lfp")

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
        self._pending = []
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
        # A restored alignment is already applied, so both handles start together.
        self._pending = [[f, f] for f, _t in state.user_pairs()]
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
        """Drop a landmark: one handle on each panel, both at ``feature_um``.

        Nothing moves yet. The user then drags either handle to where it belongs and
        presses Align.
        """
        if self._history is None:
            return
        self._pending.append([float(feature_um), float(feature_um)])
        self._pending.sort(key=lambda p: p[0])
        self._draw_lines()
        self._update_status()
        self._update_buttons()
        self.landmarksChanged.emit()

    def remove_landmark(self, index: int) -> None:
        if not 0 <= index < len(self._pending):
            return
        del self._pending[index]
        self._draw_lines()
        self._update_status()
        self._update_buttons()
        self.landmarksChanged.emit()

    def clear_landmarks(self) -> None:
        current = self.landmarks()
        self._pending = []
        if current is not None and current.n_user:
            self._commit(current.cleared())
        else:
            self._draw_lines()
            self._update_status()
            self._update_buttons()
            self.landmarksChanged.emit()

    def move_landmark(self, index: int, feature_um: float, *, slot: int = 0) -> None:
        """Move one handle of landmark ``index``: slot 0 = feature, 1 = anatomy."""
        if not 0 <= index < len(self._pending):
            return
        self._pending[index][slot] = float(feature_um)
        self._draw_lines()
        self._update_status()
        self._update_buttons()
        self.landmarksChanged.emit()

    def shift(self, delta_um: float) -> None:
        """Move the whole alignment deeper (+) or shallower (-) without stretching."""
        current = self.landmarks()
        if current is None:
            return
        self._commit(current.shifted(float(delta_um)))

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
        """Double-click either panel: add a landmark there, or remove one under it."""
        if not event.double() or self._history is None:
            return
        for plot, slot in ((self._view.ephys_plot, 0), (self._view.region_plot, 1)):
            if plot is None or not plot.sceneBoundingRect().contains(event.scenePos()):
                continue
            y = float(plot.getViewBox().mapSceneToView(event.scenePos()).y())
            hit = self._landmark_near(y, slot)
            if hit is None:
                self.add_landmark_at(y)
            else:
                self.remove_landmark(hit)
            event.accept()
            return

    def _landmark_near(self, depth_um: float, slot: int = 0) -> int | None:
        if not self._pending:
            return None
        plot = self._view.ephys_plot if slot == 0 else self._view.region_plot
        lo, hi = plot.getViewBox().viewRange()[1]
        tolerance = max(abs(hi - lo) * _HIT_FRACTION, 1.0)
        depths = np.array([p[slot] for p in self._pending])
        idx = int(np.argmin(np.abs(depths - depth_um)))
        return idx if abs(depths[idx] - depth_um) <= tolerance else None

    def _on_line_dragged(self, line) -> None:
        """A handle was released: record its new depth, and leave its twin alone.

        Nothing is applied here. Moving a handle states an intention; **Align** is what
        acts on it. Keeping those separate is what lets the two handles disagree, which
        is the only way the user can express "this feature belongs to that boundary".

        Deliberately does not rebuild the handles - destroying a ``QGraphicsItem`` from
        inside its own signal handler is a good way to earn a native crash.
        """
        if self._suspend_line_signals or self._history is None:
            return
        index = int(line.landmark_index)
        slot = int(getattr(line, "landmark_slot", 0))
        if not 0 <= index < len(self._pending):
            return
        self._pending[index][slot] = float(line.value())
        self._update_status()
        self._update_buttons()
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
        """A landmark is **two** independent handles, one per panel.

        This is the whole interaction, and getting it wrong made the tool useless:

        * the handle on the **ephys** panel says *where the feature is* - you put it on
          the transition you can see in the LFP or the raster;
        * the handle on the **region column** says *which anatomical boundary that is* -
          you put it on the boundary it should be;
        * they are allowed to sit at **different depths**. The gap between them is the
          correction you are proposing. Dragging one must not drag the other.

        Pressing **Align** then applies them: the anatomy moves (or stretches, if other
        landmarks already pin it) until each pair meets.

        An earlier version tied the two handles to the same value, so the region handle
        could never say anything the ephys handle had not already said, and the whole
        point of a landmark was lost.
        """
        if not self._ok:
            return
        import pyqtgraph as pg

        self._suspend_line_signals = True
        try:
            for line in self._lines:
                line.parent_plot.removeItem(line)
            self._lines = []
            if self._history is None:
                return
            specs = (
                (self._view.ephys_plot, 0, "feature", (255, 80, 80)),
                (self._view.region_plot, 1, "anatomy", (90, 220, 255)),
            )
            for i, pair in enumerate(self._pending):
                for plot, slot, role, colour in specs:
                    if plot is None:
                        continue
                    line = pg.InfiniteLine(
                        pos=pair[slot], angle=0, movable=True,
                        pen=pg.mkPen(colour, width=3),
                        hoverPen=pg.mkPen(255, 220, 90, width=5),
                        label=f"{i} {role}",
                        labelOpts={"color": colour, "position": 0.03},
                    )
                    line.landmark_index = i
                    line.landmark_slot = slot
                    line.parent_plot = plot
                    line.setZValue(30)
                    line.setCursor(Qt.CursorShape.SizeVerCursor)
                    line.sigPositionChangeFinished.connect(self._on_line_dragged)
                    plot.addItem(line)
                    self._lines.append(line)
        finally:
            self._suspend_line_signals = False

    def pending_pairs(self) -> list[tuple[float, float]]:
        """Handle positions as ``(feature_depth, anatomy_depth)``, before aligning."""
        return [(float(a), float(b)) for a, b in self._pending]

    def align(self) -> None:
        """Apply the handles: move/stretch the anatomy until each pair meets.

        The anatomy handle sits in the *displayed* (already-warped) column, so it is
        mapped back through the current fit to get the track depth it is pointing at.
        With one landmark this is a shift; with several the interval between them
        stretches, which is exactly the difference the user should be able to see.
        """
        current = self.landmarks()
        if current is None or not self._pending:
            return
        mode = self.extremes_mode()
        try:
            state = Landmarks.identity(*current.track_extent_um)
            for feature_depth, anatomy_depth in self._pending:
                track = float(
                    np.asarray(current.to_track(anatomy_depth, mode)).ravel()[0]
                )
                state = state.added(float(feature_depth), track)
            self._commit(state)
        except LandmarkCrossingError as exc:
            # Building the state is where a crossing surfaces, so the guard has to
            # cover the loop, not just the commit.
            self._status.setText(f"Cannot align: {exc}")
            return
        # After the fit each pair coincides, so the handles come back together.
        self._pending = [[f, f] for f, _t in state.user_pairs()]
        self._draw_lines()

    def snap_to_no_correction(self) -> None:
        """Undo the warp, keeping the landmarks where they were placed.

        Distinct from Clear: the depths you decided were interesting stay pinned, they
        just stop warping anything, so you can restart the fit without hunting for the
        boundaries again.
        """
        current = self.landmarks()
        if current is None:
            return
        state = Landmarks.identity(*current.track_extent_um)
        for _feature, track in current.user_pairs():
            state = state.added(track, track)
        self._pending = [[t, t] for _f, t in current.user_pairs()]
        self._commit(state)

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
        gaps = [b - a for a, b in self._pending if abs(b - a) > 1.0]
        if gaps:
            worst = max(gaps, key=abs)
            self._status.setText(
                f"{len(self._pending)} landmark(s), {len(gaps)} not yet applied "
                f"(largest gap {worst:+.0f} µm between the feature bar and the "
                "anatomy bar). Press Align to move the anatomy onto them."
            )
            return
        if current.n_user == 0:
            self._status.setText(
                "No landmarks applied. Add one, drag the red bar onto a feature and "
                "the blue bar onto the boundary it should be, then press Align."
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
        self._align_btn.setEnabled(bool(has_track and self._pending))
        self._clear_btn.setEnabled(
            bool(self._pending or (current is not None and current.n_user))
        )
        self._snap_btn.setEnabled(bool(current is not None and current.n_user))
        self._reset_btn.setEnabled(has_track)
        for btn in self._shift_buttons:
            btn.setEnabled(has_track)
        self._prev_btn.setEnabled(bool(has_track and self._history.can_undo))
        self._next_btn.setEnabled(bool(has_track and self._history.can_redo))

    # -- reporting -------------------------------------------------------

    def status_text(self) -> str:
        return self._status.text()


class EphysProbeAlignmentDialog(QDialog):
    """Alignment for a **whole probe**: one tab per shank, one shared workflow.

    A recording carries every shank, so asking which shank to align before opening
    anything was the wrong question - the answer is nearly always "all of them". The
    tabs make each shank's full-width panels available without four windows, and the
    tab labels mark which shanks already carry landmarks.

    This replaces the two separate dialogs (LFP-only and landmark-only), which showed
    the same track through different halves of the evidence and left the user to
    reconcile them by eye.
    """

    def __init__(self, state, probe_idx: int, *, lfp_result=None, profile=None,
                 initial_shank: int = 0, on_applied=None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._probe = state.project.probes[probe_idx]
        self._on_applied = on_applied
        self.setWindowTitle(f"Ephys alignment - {self._probe.label}")

        root = QVBoxLayout(self)
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self.panels: list[EphysAlignmentPanel] = []
        self._shanks = list(self._probe.shanks)
        for shank in self._shanks:
            panel = EphysAlignmentPanel()
            self._load_shank(panel, shank, lfp_result, profile)
            self._tabs.addTab(panel, self._tab_label(shank))
            self.panels.append(panel)
            panel.landmarksChanged.connect(self._refresh_tab_labels)
        # The Ephys tab's Shank selector now only chooses which tab opens first -
        # every shank is present either way.
        if 0 <= initial_shank < self._tabs.count():
            self._tabs.setCurrentIndex(initial_shank)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close
        )
        save_btn = QPushButton("Save depth features…")
        save_btn.setToolTip(
            "Save every shank's depth-resolved features as one compressed .npz: the "
            "LFP power map, the firing-rate and amplitude profiles, the atlas regions "
            "along each track, and the landmarks placed against them.\n"
            "Defaults to a folder beside the project file."
        )
        save_btn.clicked.connect(self.save_features)
        buttons.addButton(save_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.button(QDialogButtonBox.StandardButton.Apply).setToolTip(
            "Store every shank's landmarks on the project. Shanks you have not touched "
            "keep whatever they had."
        )
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.close)
        root.addWidget(buttons)
        # Open big enough that the whole panel is visible without scrolling or
        # resizing - the panels are wide (five columns) and the fit plot sits below.
        self._size_to_screen()

    def _size_to_screen(self) -> None:
        """Tall, not wide. Two panels at 3:2 need far less width than five did."""
        from qtpy.QtWidgets import QApplication

        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1000, 950)
            return
        available = screen.availableGeometry()
        self.resize(
            min(1100, int(available.width() * 0.6)),
            int(available.height() * 0.92),
        )

    def _tab_label(self, shank) -> str:
        eph = shank.ephys
        n = max(len(eph.feature_um) - 2, 0) if eph and eph.feature_um else 0
        return f"Shank {shank.index}" + (f"  ✓{n}" if n else "")

    def _refresh_tab_labels(self) -> None:
        for i, (panel, shank) in enumerate(zip(self.panels, self._shanks, strict=True)):
            lm = panel.landmarks()
            n = lm.n_user if lm is not None else 0
            self._tabs.setTabText(i, f"Shank {shank.index}" + (f"  ✓{n}" if n else ""))

    def _load_shank(self, panel: EphysAlignmentPanel, shank, lfp_result, profile) -> None:
        if profile is None:
            from histo_to_ccf.ephys.penetration import PenetrationProfile

            profile = PenetrationProfile()
        track_length = 0.0
        if shank.tip_ccf_um is not None and shank.entry_ccf_um is not None:
            track_length = float(np.linalg.norm(
                np.asarray(shank.tip_ccf_um) - np.asarray(shank.entry_ccf_um)
            ))
        panel.set_penetration(profile, track_length_um=track_length)
        panel.set_track(self._state.atlas, shank.tip_ccf_um, shank.entry_ccf_um)
        panel.view().mark_track_ends(track_length)
        if lfp_result is not None and track_length > 0:
            self._load_shank_lfp(panel, shank, lfp_result, track_length)
        panel.refresh_display_modes()
        if shank.ephys is not None:
            panel.restore_landmarks(shank.ephys.feature_um, shank.ephys.track_um)

    @staticmethod
    def _load_shank_lfp(panel: EphysAlignmentPanel, shank, lfp_result,
                        track_length_um: float) -> None:
        """Feed this shank's slice of the recording's LFP into its panel.

        Split by the probe's **shank ids**, never by x: a NP2.0 shank has two
        electrode columns, so unique-x over-splits one shank into two and grabs a
        single column (the "48 instead of 96" bug).
        """
        depths_from_tip = np.asarray(lfp_result.get("depths_um", []), dtype=float)
        psd = np.asarray(lfp_result.get("psd", []), dtype=float)
        freqs = np.asarray(lfp_result.get("freqs", []), dtype=float)
        if depths_from_tip.size == 0 or psd.ndim != 2:
            return
        shank_ids = lfp_result.get("shank_ids")
        mask = np.ones(depths_from_tip.shape, dtype=bool)
        if shank_ids is not None:
            uniq = sorted({str(s) for s in np.asarray(shank_ids).tolist()})
            if len(uniq) > 1 and shank.index < len(uniq):
                mask = np.array([str(s) == uniq[shank.index] for s in shank_ids])
        if not mask.any():
            return
        # LFP depths are µm from the tip; the panels are depth below the surface.
        panel.view().set_lfp(track_length_um - depths_from_tip[mask], psd[mask], freqs)

    def feature_exports(self) -> list:
        """Assemble every shank's features for export. Headless-testable."""
        from histo_to_ccf.ephys.export import ShankFeatureExport
        from histo_to_ccf.ephys.features import depth_profiles

        out = []
        for panel, shank in zip(self.panels, self._shanks, strict=True):
            view = panel.view()
            landmarks = panel.landmarks()
            track_length = 0.0
            if shank.tip_ccf_um is not None and shank.entry_ccf_um is not None:
                track_length = float(np.linalg.norm(
                    np.asarray(shank.tip_ccf_um) - np.asarray(shank.entry_ccf_um)
                ))
            item = ShankFeatureExport(
                shank_index=shank.index,
                track_length_um=track_length,
                extremes_mode=panel.extremes_mode(),
            )
            lfp = view.lfp_data()
            if lfp is not None:
                depths_below, psd, freqs = lfp
                item.channel_depth_below_surface_um = np.asarray(depths_below)
                item.channel_depth_from_tip_um = track_length - np.asarray(depths_below)
                item.lfp_psd = np.asarray(psd)
                item.lfp_freqs_hz = np.asarray(freqs)
            profile = view._profile
            if profile is not None and profile.profiles:
                depth, amp, _t = profile.all_spikes()
                duration = sum(p.duration_s for p in profile.profiles)
                if depth.size and duration > 0:
                    centres, rate, mean_amp = depth_profiles(depth, amp, duration,
                                                             bin_um=25.0)
                    item.profile_depth_um = centres
                    item.firing_rate_hz = rate
                    item.mean_amplitude = mean_amp
            bands = view.bands()
            if bands:
                item.region_top_um = np.array([b.top_um for b in bands])
                item.region_bottom_um = np.array([b.bottom_um for b in bands])
                item.region_acronym = [b.acronym for b in bands]
            if landmarks is not None:
                item.landmark_feature_um = np.asarray(landmarks.feature_um)
                item.landmark_track_um = np.asarray(landmarks.track_um)
            out.append(item)
        return out

    def save_features(self) -> None:
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        from histo_to_ccf.ephys.export import default_export_path, save_feature_export

        suggested = default_export_path(
            getattr(self._state, "project_path", None), self._probe.label
        )
        # Create the folder first. Qt silently ignores a suggested path whose
        # directory does not exist and falls back to whatever directory was last
        # used - which is why the dialog opened in the raw-recording folder instead
        # of beside the project.
        with contextlib.suppress(OSError):
            suggested.parent.mkdir(parents=True, exist_ok=True)
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save depth features", str(suggested), "NumPy archive (*.npz)"
        )
        if not path:
            return
        try:
            written = save_feature_export(path, self._probe.label, self.feature_exports())
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc)[:2000])
            return
        size_kb = written.stat().st_size / 1024.0 if written.exists() else 0.0
        box = QMessageBox(
            QMessageBox.Icon.Information, "Saved",
            f"Depth features saved to\n{written}\n({size_kb:,.0f} kB)", parent=self,
        )
        # Non-modal: a blocking notice on a path a headless run can reach is how the
        # suite deadlocked for three days.
        box.setModal(False)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.show()
        self._save_box = box

    def apply(self) -> None:
        """Store every shank's landmarks, leaving untouched shanks as they were."""
        from datetime import datetime

        from histo_to_ccf.project.schema import EphysAlignment

        stamp = datetime.now().isoformat(timespec="seconds")
        # One insertion depth for the whole penetration when the recordings agree;
        # None means "assume it matches the histology track", the honest default
        # before any recording has pinned it.
        depths = {
            r.insertion_depth_um
            for r in (self._probe.recordings or [])
            if r.insertion_depth_um
        }
        insertion = float(next(iter(depths))) if len(depths) == 1 else None

        for panel, shank in zip(self.panels, self._shanks, strict=True):
            landmarks = panel.landmarks()
            if landmarks is None:
                continue
            if shank.ephys is None:
                shank.ephys = EphysAlignment()
            # Only the landmark arrays and the stamp: ``anchors`` and
            # ``channel_ccf_um`` belong to the older tip-referenced alignment, and
            # writing depth-below-surface numbers into them would flip the track.
            shank.ephys.feature_um = [float(v) for v in landmarks.feature_um]
            shank.ephys.track_um = [float(v) for v in landmarks.track_um]
            shank.ephys.extremes_mode = panel.extremes_mode()
            shank.ephys.insertion_depth_um = insertion
            shank.ephys.created_at = stamp
        if self._on_applied is not None:
            self._on_applied()
        self.close()
