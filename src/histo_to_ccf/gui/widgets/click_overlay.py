"""Tip / entry-point click overlay widget.

Manages napari layers for tip/entry annotation and maps viewer clicks to
Section pixel coordinates stored in the project Shank objects.

Markers are **colour-coded per shank**: a shank's tip and entry share one colour
(cycling as you select another shank/probe), and tip vs entry are told apart by
**symbol** (tip = disc, entry = triangle). Each point carries its ``(probe,
shank)`` in the layer ``features`` so identity survives moves / deletes, and the
layers stay in two-way sync with the project schema:

* **add** (Tip/Entry mode, click) - a point for the selected shank; a second
  click for a shank that already has that point *replaces* it (one per shank).
* **move / delete** ("Select / move" mode) - drag to reposition, Delete or
  "Clear selected" to remove just the selected points; "Clear all" wipes them.

Two ways to mark an entry point:

* **Marker** - click the brain surface directly.
* **Trajectory line** - draw the probe track as a line; the point where that
  line first crosses the tissue surface is taken as the entry.
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

# Distinct, colour-blind-friendlier cycle; a shank's global ordinal indexes it so
# the same shank gets the same colour in both the Tips and Entries layers.
_SHANK_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9a6324",
    "#800000", "#808000", "#000075", "#a9a9a9",
]


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
        # Suppress the data-changed handlers while we set marker data in bulk.
        self._suppress_store = False
        # Track point counts so a data event can tell an *add* from a move/delete.
        self._tip_count = 0
        self._entry_count = 0
        # Per-slide tissue-mask cache for trajectory→surface intersection.
        self._mask_cache: tuple[int, np.ndarray] | None = None
        self._build_ui()
        # NOTE: the Tips/Entries Points layers are created lazily (the first time
        # the user arms tip/entry), not here - adding empty Points layers at
        # launch made vispy try to draw a Markers visual with no data, which on
        # some Windows GPUs triggered shader / framebuffer errors before a slide
        # was even loaded.

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

        # Select probe + shank by label (consistent with the Ephys tab).
        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe:"))
        self._probe_combo = QComboBox()
        self._probe_combo.currentIndexChanged.connect(self._on_probe_changed)
        probe_row.addWidget(self._probe_combo, 1)
        layout.addLayout(probe_row)

        shank_row = QHBoxLayout()
        shank_row.addWidget(QLabel("Shank:"))
        self._shank_combo = QComboBox()
        # Shank change only re-targets new markers; it must NOT repopulate the
        # shank combo (that would recurse).
        self._shank_combo.currentIndexChanged.connect(self._apply_current_identity)
        shank_row.addWidget(self._shank_combo, 1)
        layout.addLayout(shank_row)
        self._refresh_probe_combo()

        # Edit / delete controls.
        edit_row = QHBoxLayout()
        self._select_btn = QPushButton("Select / move")
        self._select_btn.setCheckable(True)
        self._select_btn.setToolTip(
            "Enter select mode: drag a marker to reposition it, or click to select "
            "(Shift-click for several), then Delete or 'Clear selected' to remove. "
            "Turn off to go back to dropping new points."
        )
        self._select_btn.toggled.connect(self._on_select_toggled)
        edit_row.addWidget(self._select_btn)
        clear_sel_btn = QPushButton("Clear selected")
        clear_sel_btn.setToolTip("Remove only the currently selected marker(s).")
        clear_sel_btn.clicked.connect(self._clear_selected)
        edit_row.addWidget(clear_sel_btn)
        layout.addLayout(edit_row)

        clear_btn = QPushButton("Clear all points")
        clear_btn.clicked.connect(self._clear_points)
        layout.addWidget(clear_btn)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Probe", "Shank", "Type", "Coords (px)"])
        self._table.setMaximumHeight(200)
        layout.addWidget(self._table)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Colour / identity helpers
    # ------------------------------------------------------------------

    def _shank_ordinals(self) -> dict[tuple[int, int], int]:
        """Map ``(probe_pos, shank_pos)`` to a global ordinal for colour cycling."""
        out: dict[tuple[int, int], int] = {}
        k = 0
        for p_idx, probe in enumerate(self._state.project.probes):
            for s_idx in range(len(probe.shanks)):
                out[(p_idx, s_idx)] = k
                k += 1
        return out

    def _color_for(self, p_idx: int, s_idx: int) -> str:
        ordinal = self._shank_ordinals().get((p_idx, s_idx), 0)
        return _SHANK_COLORS[ordinal % len(_SHANK_COLORS)]

    def _current_ps(self) -> tuple[int, int]:
        return self._probe_combo.currentIndex(), self._shank_combo.currentIndex()

    @staticmethod
    def _feature_array(features, name: str, n: int) -> np.ndarray:
        """Read a feature column as a float array of length ``n`` (pad/truncate)."""
        arr = np.zeros(n, dtype=float)
        try:
            vals = np.asarray(features[name], dtype=float)
            m = min(len(vals), n)
            arr[:m] = vals[:m]
        except Exception:  # noqa: BLE001 - missing column / empty layer
            pass
        return arr

    def _apply_current_identity(self, *_args) -> None:
        """Default new markers to the selected shank (its colour is set on sync).

        Only ``feature_defaults`` is touched - the actual per-shank colour is
        applied by :meth:`_recolor` after each data change. (Setting
        ``current_face_color`` here would drive napari's colour-swatch control and
        can recurse, so we deliberately avoid it.)
        """
        p_idx, s_idx = self._current_ps()
        if p_idx < 0 or s_idx < 0:
            return
        for layer in (self._tip_layer, self._entry_layer):
            if layer is None:
                continue
            try:
                layer.feature_defaults = {"p": p_idx, "s": s_idx}
            except Exception:  # noqa: BLE001
                pass

    def _on_probe_changed(self, *_args) -> None:
        self._refresh_shank_combo()
        self._apply_current_identity()

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    def _drop_stale_layer_refs(self) -> None:
        """Forget layer references that are no longer in the viewer.

        A project close/clear empties ``viewer.layers`` but leaves these widget
        attributes pointing at the removed layers. Operating on such a detached
        layer means markers never draw and ``selection.active = layer`` warns
        "not in the list" - so reset the refs and let them be recreated.
        """
        layers = self._viewer.layers
        if self._tip_layer is not None and self._tip_layer not in layers:
            self._tip_layer = None
        if self._entry_layer is not None and self._entry_layer not in layers:
            self._entry_layer = None
        if self._traj_layer is not None and self._traj_layer not in layers:
            self._traj_layer = None

    def _ensure_points_layers(self) -> None:
        """Create the Tips/Entries Points layers on first use (and wire events)."""
        self._drop_stale_layer_refs()
        if self._tip_layer is None:
            if _LAYER_TIP in self._viewer.layers:
                self._tip_layer = self._viewer.layers[_LAYER_TIP]  # type: ignore[assignment]
            else:
                self._tip_layer = self._viewer.add_points(
                    name=_LAYER_TIP, face_color="red", size=12, ndim=2, symbol="disc",
                )
            self._tip_layer.events.data.connect(self._on_tip_data_changed)
        if self._entry_layer is None:
            if _LAYER_ENTRY in self._viewer.layers:
                self._entry_layer = self._viewer.layers[_LAYER_ENTRY]  # type: ignore[assignment]
            else:
                self._entry_layer = self._viewer.add_points(
                    name=_LAYER_ENTRY, face_color="cyan", size=12, ndim=2,
                    symbol="triangle_up",
                )
            self._entry_layer.events.data.connect(self._on_entry_data_changed)
        self._apply_current_identity()

    def _ensure_traj_layer(self) -> "napari.layers.Shapes":
        """Create (or fetch) the trajectory Shapes layer used for line drawing."""
        self._drop_stale_layer_refs()
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
        """Move ``layer`` to the top of the stack so its markers stay visible."""
        layers = self._viewer.layers
        try:
            src = layers.index(layer)
            if src != len(layers) - 1:
                layers.move(src, len(layers) - 1)
        except Exception:
            pass

    def arm_tip(self) -> None:
        """Select Tip + Marker mode and arm the viewer (e.g. after Add probe)."""
        self._refresh_probe_combo()
        n_probes = len(self._state.project.probes)
        if n_probes:
            self._probe_combo.setCurrentIndex(n_probes - 1)
        self._select_btn.setChecked(False)
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
        cur = self._shank_combo.currentIndex()
        self._shank_combo.blockSignals(True)
        self._shank_combo.clear()
        p_idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if 0 <= p_idx < len(probes):
            for shank in probes[p_idx].shanks:
                self._shank_combo.addItem(f"Shank {shank.index}")
        self._shank_combo.blockSignals(False)
        if 0 <= cur < self._shank_combo.count():
            self._shank_combo.setCurrentIndex(cur)

    def _activate_pick_mode(self, *_args) -> None:
        """Arm the viewer tool that matches the current selection / select toggle."""
        if self._select_btn.isChecked():
            self._set_points_mode("select")
            return
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

    def _set_points_mode(self, mode: str) -> None:
        self._ensure_points_layers()
        active = self._tip_layer if self._mode_tip.isChecked() else self._entry_layer
        for layer in (self._tip_layer, self._entry_layer):
            if layer is not None:
                try:
                    layer.mode = mode
                except Exception:  # noqa: BLE001
                    pass
        if active is not None:
            self._viewer.layers.selection.active = active
            self._bring_to_front(active)

    def _on_select_toggled(self, on: bool) -> None:
        if on:
            self._set_points_mode("select")
        else:
            self._activate_pick_mode()

    # ------------------------------------------------------------------
    # Point event handlers
    # ------------------------------------------------------------------

    def _on_tip_data_changed(self, event=None) -> None:
        if self._suppress_store or self._tip_layer is None:
            return
        n = len(self._tip_layer.data)
        added = n == self._tip_count + 1
        self._sync_layer(self._tip_layer, "tip", added)
        self._tip_count = len(self._tip_layer.data)

    def _on_entry_data_changed(self, event=None) -> None:
        if self._suppress_store or self._entry_layer is None:
            return
        n = len(self._entry_layer.data)
        added = n == self._entry_count + 1
        self._sync_layer(self._entry_layer, "entry", added)
        self._entry_count = len(self._entry_layer.data)

    def _sync_layer(self, layer, kind: str, added: bool) -> None:
        """Two-way sync a Points layer with the schema after add/move/delete.

        ``kind`` is ``"tip"`` or ``"entry"``. On *add* the new (last) point is
        assigned to the currently selected shank; points are then deduped to one
        per shank (newest wins), the schema is rewritten from the points, and the
        per-shank colours are reapplied.
        """
        data = np.asarray(layer.data, dtype=float)
        n = len(data)
        p_arr = self._feature_array(layer.features, "p", n)
        s_arr = self._feature_array(layer.features, "s", n)
        if added and n >= 1:
            cp, cs = self._current_ps()
            p_arr[-1], s_arr[-1] = float(cp), float(cs)

        # One marker of this kind per shank: keep the newest for each (p, s).
        keep: dict[tuple[int, int], int] = {}
        for i in range(n):
            keep[(int(p_arr[i]), int(s_arr[i]))] = i
        keep_idx = sorted(keep.values())

        # Rewrite the schema from the kept points.
        probes = self._state.project.probes
        for probe in probes:
            for shank in probe.shanks:
                if kind == "tip":
                    shank.tip_px, shank.tip_section_idx = None, None
                else:
                    shank.entry_px, shank.entry_section_idx = None, None
        for i in keep_idx:
            p, s = int(p_arr[i]), int(s_arr[i])
            if not (0 <= p < len(probes)) or not (0 <= s < len(probes[p].shanks)):
                continue
            y, x = float(data[i][0]), float(data[i][1])
            shank = probes[p].shanks[s]
            sec = self._find_section_for_point(x, y)
            if kind == "tip":
                shank.tip_px, shank.tip_section_idx = Point2D(x_px=x, y_px=y), sec
            else:
                shank.entry_px, shank.entry_section_idx = Point2D(x_px=x, y_px=y), sec

        # Push the cleaned points + colours back to the layer (suppressed).
        self._suppress_store = True
        try:
            if len(keep_idx) != n:
                layer.data = data[keep_idx]
            layer.features = {"p": p_arr[keep_idx], "s": s_arr[keep_idx]}
            self._recolor(layer)
        finally:
            self._suppress_store = False
        self._refresh_table()

    def _recolor(self, layer) -> None:
        """Colour each point by its shank's global ordinal (tip & entry match)."""
        n = len(layer.data)
        if n == 0:
            return
        p_arr = self._feature_array(layer.features, "p", n)
        s_arr = self._feature_array(layer.features, "s", n)
        ordinals = self._shank_ordinals()
        colors = [
            _SHANK_COLORS[ordinals.get((int(p_arr[i]), int(s_arr[i])), 0) % len(_SHANK_COLORS)]
            for i in range(n)
        ]
        try:
            layer.face_color = colors
            layer.border_color = colors
        except Exception:  # noqa: BLE001
            pass

    def _on_trajectory_changed(self, event=None) -> None:
        """When a trajectory line is drawn, derive the surface entry point."""
        if self._traj_layer is None or len(self._traj_layer.data) == 0:
            return
        line = np.asarray(self._traj_layer.data[-1])  # (n_pts, 2) [row, col]
        if line.shape[0] < 2:
            return
        entry = self._line_surface_crossing(line[0], line[-1])
        if entry is None:
            return
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
        """Return the (x, y) where segment a→b first enters tissue."""
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
            idx = transitions[0] if not inside[0] else transitions[-1]
            return float(cols[idx]), float(rows[idx])
        if not inside[0]:
            return float(a[1]), float(a[0])
        if not inside[-1]:
            return float(b[1]), float(b[0])
        return float(a[1]), float(a[0])

    # ------------------------------------------------------------------
    # Section lookup
    # ------------------------------------------------------------------

    def _find_section_for_point(self, x_px: float, y_px: float) -> int | None:
        """Return the index of the section containing - or nearest to - a pixel."""
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
        self._drop_stale_layer_refs()
        self._refresh_probe_combo()
        self._rebuild_markers()
        self._refresh_table()

    def _rebuild_markers(self) -> None:
        """Redraw the Tips/Entries point layers (with identity + colour) from schema."""
        tips: list[list[float]] = []
        tip_p: list[int] = []
        tip_s: list[int] = []
        entries: list[list[float]] = []
        ent_p: list[int] = []
        ent_s: list[int] = []
        for p_idx, probe in enumerate(self._state.project.probes):
            for s_idx, shank in enumerate(probe.shanks):
                if shank.tip_px is not None:
                    tips.append([shank.tip_px.y_px, shank.tip_px.x_px])
                    tip_p.append(p_idx)
                    tip_s.append(s_idx)
                if shank.entry_px is not None:
                    entries.append([shank.entry_px.y_px, shank.entry_px.x_px])
                    ent_p.append(p_idx)
                    ent_s.append(s_idx)
        if not tips and not entries:
            return  # nothing to draw - avoid creating empty layers
        self._ensure_points_layers()
        self._suppress_store = True
        try:
            self._set_layer(self._tip_layer, tips, tip_p, tip_s)
            self._set_layer(self._entry_layer, entries, ent_p, ent_s)
        finally:
            self._suppress_store = False
        self._tip_count = len(tips)
        self._entry_count = len(entries)
        for layer in (self._tip_layer, self._entry_layer):
            if layer is not None:
                self._bring_to_front(layer)

    def _set_layer(self, layer, pts, p_idx, s_idx) -> None:
        if layer is None:
            return
        layer.data = np.array(pts, dtype=float) if pts else np.empty((0, 2))
        layer.features = {"p": np.array(p_idx, dtype=float),
                          "s": np.array(s_idx, dtype=float)}
        self._recolor(layer)

    # ------------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------------

    def _clear_selected(self) -> None:
        """Remove only the selected marker(s); their shanks are cleared via sync."""
        for layer in (self._tip_layer, self._entry_layer):
            if layer is None or not getattr(layer, "selected_data", None):
                continue
            try:
                layer.remove_selected()  # fires data event -> _sync_layer
            except Exception:  # noqa: BLE001
                pass

    def _clear_points(self) -> None:
        self._suppress_store = True
        try:
            if self._tip_layer is not None:
                self._tip_layer.data = np.empty((0, 2))
            if self._entry_layer is not None:
                self._entry_layer.data = np.empty((0, 2))
            if self._traj_layer is not None:
                self._traj_layer.data = []
        finally:
            self._suppress_store = False
        self._tip_count = 0
        self._entry_count = 0
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                shank.tip_px = None
                shank.tip_section_idx = None
                shank.entry_px = None
                shank.entry_section_idx = None
        self._refresh_table()
