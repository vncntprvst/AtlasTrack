"""Tip / entry-point click overlay widget.

Manages napari layers for tip/entry annotation and maps viewer clicks to
Section pixel coordinates stored in the project Shank objects.

Two ways to mark an entry point:

* **Marker** - click the brain surface directly (a cyan point).
* **Trajectory line** - draw the probe track as a line; the point where that
  line first crosses the tissue surface is taken as the entry. Useful when the
  surface itself is hard to click precisely.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.project.schema import Point2D

if TYPE_CHECKING:
    import napari


_LAYER_TIP = "Tips"
_LAYER_ENTRY = "Entries"
_LAYER_TRAJECTORY = "Trajectory"


class ClickOverlayWidget(QWidget):
    """Mode selector + point table for tip/entry annotation."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._tip_layer: "napari.layers.Points | None" = None
        self._entry_layer: "napari.layers.Points | None" = None
        self._traj_layer: "napari.layers.Shapes | None" = None
        # Suppress the data-changed handlers while we set marker data in bulk
        # (e.g. when restoring points after a project load) so the last point
        # isn't re-stored against the currently-selected probe/shank.
        self._suppress_store = False
        # Per-slide tissue-mask cache for trajectory→surface intersection.
        self._mask_cache: tuple[int, np.ndarray] | None = None
        self._build_ui()
        # NOTE: the Tips/Entries Points layers are created lazily (the first time
        # the user arms tip/entry), not here. Adding empty Points layers at
        # launch made vispy try to draw a Markers visual with no data, which on
        # some Windows GPUs triggers "Unsupported framebuffer format" / shader
        # errors before any slide is even loaded.

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_tip = QRadioButton("Tip")
        self._mode_tip.setChecked(True)
        self._mode_entry = QRadioButton("Entry")
        mode_grp = QButtonGroup(self)
        mode_grp.addButton(self._mode_tip)
        mode_grp.addButton(self._mode_entry)
        mode_row.addWidget(self._mode_tip)
        mode_row.addWidget(self._mode_entry)
        layout.addLayout(mode_row)

        # Selecting a mode immediately arms the matching viewer tool - no extra
        # button press needed. ``clicked`` fires even when the radio is already
        # checked, so re-arming after a discard/draw action still works.
        for btn in (self._mode_tip, self._mode_entry):
            btn.toggled.connect(self._activate_pick_mode)
            btn.clicked.connect(self._activate_pick_mode)

        entry_row = QHBoxLayout()
        entry_row.addWidget(QLabel("Entry via:"))
        self._entry_marker = QRadioButton("Marker")
        self._entry_marker.setChecked(True)
        self._entry_line = QRadioButton("Trajectory line")
        entry_grp = QButtonGroup(self)
        entry_grp.addButton(self._entry_marker)
        entry_grp.addButton(self._entry_line)
        self._entry_marker.setToolTip("Click the brain surface to drop the entry point.")
        self._entry_line.setToolTip(
            "Draw the probe trajectory as a line; the entry point is where it\n"
            "first crosses the tissue surface."
        )
        entry_row.addWidget(self._entry_marker)
        entry_row.addWidget(self._entry_line)
        layout.addLayout(entry_row)
        for btn in (self._entry_marker, self._entry_line):
            btn.toggled.connect(self._activate_pick_mode)
            btn.clicked.connect(self._activate_pick_mode)

        # Select probe + shank by label (consistent with the Ephys tab) so the
        # same probe is referred to the same way everywhere.
        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe:"))
        self._probe_combo = QComboBox()
        self._probe_combo.currentIndexChanged.connect(self._refresh_shank_combo)
        probe_row.addWidget(self._probe_combo, 1)
        layout.addLayout(probe_row)

        shank_row = QHBoxLayout()
        shank_row.addWidget(QLabel("Shank:"))
        self._shank_combo = QComboBox()
        shank_row.addWidget(self._shank_combo, 1)
        layout.addLayout(shank_row)
        self._refresh_probe_combo()

        clear_btn = QPushButton("Clear all points")
        clear_btn.clicked.connect(self._clear_points)
        layout.addWidget(clear_btn)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Probe", "Shank", "Type", "Coords (px)"])
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _ensure_points_layers(self) -> None:
        """Create the Tips/Entries Points layers on first use (and wire events)."""
        if self._tip_layer is None:
            if _LAYER_TIP in self._viewer.layers:
                self._tip_layer = self._viewer.layers[_LAYER_TIP]  # type: ignore[assignment]
            else:
                self._tip_layer = self._viewer.add_points(
                    name=_LAYER_TIP, face_color="red", size=12, ndim=2
                )
            self._tip_layer.events.data.connect(self._on_tip_data_changed)
        if self._entry_layer is None:
            if _LAYER_ENTRY in self._viewer.layers:
                self._entry_layer = self._viewer.layers[_LAYER_ENTRY]  # type: ignore[assignment]
            else:
                self._entry_layer = self._viewer.add_points(
                    name=_LAYER_ENTRY, face_color="cyan", size=12, ndim=2
                )
            self._entry_layer.events.data.connect(self._on_entry_data_changed)

    def _ensure_traj_layer(self) -> "napari.layers.Shapes":
        """Create (or fetch) the trajectory Shapes layer used for line drawing."""
        if _LAYER_TRAJECTORY in self._viewer.layers:
            self._traj_layer = self._viewer.layers[_LAYER_TRAJECTORY]  # type: ignore[assignment]
        elif self._traj_layer is None:
            self._traj_layer = self._viewer.add_shapes(
                name=_LAYER_TRAJECTORY, edge_color="yellow", face_color="transparent",
                edge_width=4,
            )
            self._traj_layer.events.data.connect(self._on_trajectory_changed)
        return self._traj_layer

    def _bring_to_front(self, layer) -> None:
        """Move ``layer`` to the top of the stack so its markers stay visible.

        Tip/Entry layers are created before the slide image and section
        overlays, so without this they would sit underneath and be hidden.
        """
        layers = self._viewer.layers
        try:
            src = layers.index(layer)
            if src != len(layers) - 1:
                layers.move(src, len(layers) - 1)
        except Exception:
            pass

    def arm_tip(self) -> None:
        """Select Tip + Marker mode and arm the viewer (e.g. after Add probe)."""
        # Point at the most recently added probe so the tip lands on it.
        self._refresh_probe_combo()
        n_probes = len(self._state.project.probes)
        if n_probes:
            self._probe_combo.setCurrentIndex(n_probes - 1)
        self._mode_tip.setChecked(True)
        self._entry_marker.setChecked(True)
        self._activate_pick_mode()

    # ------------------------------------------------------------------
    # Probe / shank selectors
    # ------------------------------------------------------------------

    def _refresh_probe_combo(self) -> None:
        """Repopulate the probe combo from the project, keeping the selection."""
        cur = self._probe_combo.currentIndex()
        self._probe_combo.blockSignals(True)
        self._probe_combo.clear()
        for probe in self._state.project.probes:
            self._probe_combo.addItem(probe.label)
        self._probe_combo.blockSignals(False)
        if 0 <= cur < self._probe_combo.count():
            self._probe_combo.setCurrentIndex(cur)
        self._refresh_shank_combo()

    def _refresh_shank_combo(self) -> None:
        self._shank_combo.clear()
        p_idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if 0 <= p_idx < len(probes):
            for shank in probes[p_idx].shanks:
                self._shank_combo.addItem(f"Shank {shank.index}")

    def _activate_pick_mode(self, *_args) -> None:
        """Arm the viewer tool that matches the current Tip/Entry selection."""
        if self._mode_entry.isChecked() and self._entry_line.isChecked():
            layer = self._ensure_traj_layer()
            self._viewer.layers.selection.active = layer
            self._bring_to_front(layer)
            try:
                layer.mode = "add_line"
            except Exception:
                layer.mode = "add_path"
            return

        self._ensure_points_layers()
        layer = self._tip_layer if self._mode_tip.isChecked() else self._entry_layer
        if layer is None:
            return
        self._viewer.layers.selection.active = layer
        self._bring_to_front(layer)
        layer.mode = "add"

    # ------------------------------------------------------------------
    # Point event handlers
    # ------------------------------------------------------------------

    def _on_tip_data_changed(self, event=None) -> None:
        if self._suppress_store:
            return
        if self._tip_layer is None or len(self._tip_layer.data) == 0:
            return
        last = self._tip_layer.data[-1]  # (row, col) image coords
        self._store_point(float(last[1]), float(last[0]), mode="tip")
        self._refresh_table()

    def _on_entry_data_changed(self, event=None) -> None:
        if self._suppress_store:
            return
        if self._entry_layer is None or len(self._entry_layer.data) == 0:
            return
        last = self._entry_layer.data[-1]
        self._store_point(float(last[1]), float(last[0]), mode="entry")
        self._refresh_table()

    def _on_trajectory_changed(self, event=None) -> None:
        """When a trajectory line is drawn, derive the surface entry point."""
        if self._traj_layer is None or len(self._traj_layer.data) == 0:
            return
        line = np.asarray(self._traj_layer.data[-1])  # (n_pts, 2) [row, col]
        if line.shape[0] < 2:
            return
        a = line[0]
        b = line[-1]
        entry = self._line_surface_crossing(a, b)
        if entry is None:
            return
        # Drop a marker on the Entries layer; its data event stores the point.
        ex, ey = entry
        self._ensure_points_layers()
        if self._entry_layer is not None:
            self._entry_layer.data = np.vstack([self._entry_layer.data, [[ey, ex]]])
            self._bring_to_front(self._entry_layer)

    # ------------------------------------------------------------------
    # Trajectory → tissue-surface intersection
    # ------------------------------------------------------------------

    def _tissue_mask(self) -> np.ndarray | None:
        """Binary tissue mask for the active slide (cached)."""
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            return None
        img = self._state.slide_images.get(slide_idx)
        if img is None:
            return None
        if self._mask_cache is not None and self._mask_cache[0] == slide_idx:
            return self._mask_cache[1]
        from histo_to_ccf.sectioning.split import _binarize, _to_gray

        mask = _binarize(_to_gray(img))
        self._mask_cache = (slide_idx, mask)
        return mask

    def _line_surface_crossing(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float] | None:
        """Return the (x, y) where segment a→b first enters tissue.

        ``a`` and ``b`` are ``[row, col]`` endpoints. Samples along the segment
        and finds the first background→tissue transition, scanning inward from
        whichever endpoint lies outside the tissue (the probe enters from the
        surface). Falls back to the outside endpoint, then the segment start.
        """
        mask = self._tissue_mask()
        if mask is None:
            return None
        h, w = mask.shape
        n = 256
        ts = np.linspace(0.0, 1.0, n)
        rows = a[0] + ts * (b[0] - a[0])
        cols = a[1] + ts * (b[1] - a[1])
        ri = np.clip(np.round(rows).astype(int), 0, h - 1)
        ci = np.clip(np.round(cols).astype(int), 0, w - 1)
        inside = mask[ri, ci]

        transitions = np.flatnonzero((~inside[:-1]) & inside[1:]) + 1
        if len(transitions):
            # If endpoint a is outside, the first crossing is the surface;
            # otherwise scan from b (reverse) and take the last crossing.
            idx = transitions[0] if not inside[0] else transitions[-1]
            return float(cols[idx]), float(rows[idx])
        # No clean crossing: return whichever endpoint is outside tissue.
        if not inside[0]:
            return float(a[1]), float(a[0])
        if not inside[-1]:
            return float(b[1]), float(b[0])
        return float(a[1]), float(a[0])

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_point(self, x_px: float, y_px: float, mode: str) -> None:
        """Store a clicked coordinate into the project schema."""
        probes = self._state.project.probes
        p_idx = self._probe_combo.currentIndex()
        s_idx = self._shank_combo.currentIndex()
        if p_idx < 0 or p_idx >= len(probes):
            return
        shanks = probes[p_idx].shanks
        if s_idx < 0 or s_idx >= len(shanks):
            return
        shank = shanks[s_idx]
        section_idx = self._find_section_for_point(x_px, y_px)
        pt = Point2D(x_px=x_px, y_px=y_px)
        if mode == "tip":
            shank.tip_px = pt
            shank.tip_section_idx = section_idx
        else:
            shank.entry_px = pt
            shank.entry_section_idx = section_idx

    def _find_section_for_point(self, x_px: float, y_px: float) -> int | None:
        """Return the index of the section containing - or nearest to - a pixel.

        Bounding boxes are often tight, so an entry point may land a little
        outside the box of the section it belongs to. We return the containing
        section if there is one, otherwise the section whose box is closest
        (squared distance to the box, 0 when inside).
        """
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or slide_idx >= len(self._state.project.slides):
            return None
        slide = self._state.project.slides[slide_idx]
        best_idx: int | None = None
        best_d = float("inf")
        for section in slide.sections:
            x0, y0, x1, y1 = section.bbox_px
            if x0 <= x_px < x1 and y0 <= y_px < y1:
                return section.index
            dx = max(x0 - x_px, 0.0, x_px - x1)
            dy = max(y0 - y_px, 0.0, y_px - y1)
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_idx = section.index
        return best_idx

    # ------------------------------------------------------------------
    # Table refresh
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        rows = []
        for p_idx, probe in enumerate(self._state.project.probes):
            for shank in probe.shanks:
                if shank.tip_px is not None:
                    rows.append((p_idx, shank.index, "tip", shank.tip_px))
                if shank.entry_px is not None:
                    rows.append((p_idx, shank.index, "entry", shank.entry_px))
        self._table.setRowCount(len(rows))
        for i, (p, s, t, pt) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(str(p)))
            self._table.setItem(i, 1, QTableWidgetItem(str(s)))
            self._table.setItem(i, 2, QTableWidgetItem(t))
            self._table.setItem(i, 3, QTableWidgetItem(f"{pt.x_px:.1f}, {pt.y_px:.1f}"))

    def refresh_after_load(self) -> None:
        """Restore tip/entry markers + the table from a freshly-loaded project."""
        self._mask_cache = None
        self._refresh_probe_combo()
        self._rebuild_markers()
        self._refresh_table()

    def _rebuild_markers(self) -> None:
        """Redraw the Tips/Entries point layers from the stored shank coords."""
        tips: list[list[float]] = []
        entries: list[list[float]] = []
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                if shank.tip_px is not None:
                    tips.append([shank.tip_px.y_px, shank.tip_px.x_px])
                if shank.entry_px is not None:
                    entries.append([shank.entry_px.y_px, shank.entry_px.x_px])
        if not tips and not entries:
            return  # nothing to draw - avoid creating empty layers
        self._ensure_points_layers()
        self._suppress_store = True
        try:
            if self._tip_layer is not None:
                self._tip_layer.data = np.array(tips, dtype=float) if tips else np.empty((0, 2))
            if self._entry_layer is not None:
                self._entry_layer.data = (
                    np.array(entries, dtype=float) if entries else np.empty((0, 2))
                )
        finally:
            self._suppress_store = False
        if self._tip_layer is not None:
            self._bring_to_front(self._tip_layer)
        if self._entry_layer is not None:
            self._bring_to_front(self._entry_layer)

    def _clear_points(self) -> None:
        if self._tip_layer is not None:
            self._tip_layer.data = np.empty((0, 2))
        if self._entry_layer is not None:
            self._entry_layer.data = np.empty((0, 2))
        if self._traj_layer is not None:
            self._traj_layer.data = []
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                shank.tip_px = None
                shank.tip_section_idx = None
                shank.entry_px = None
                shank.entry_section_idx = None
        self._refresh_table()
