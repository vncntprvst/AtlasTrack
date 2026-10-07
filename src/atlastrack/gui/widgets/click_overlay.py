"""Probe tracks: place, select, move and remove each shank's track on the sections.

A shank's track is two markers - the **tip** (where the shank ends) and the
**entry** (where it went in) - and the line through them. Each shank has its own
colour, used by its markers, its line and its row in the list.

* **Add track** - pick the Probe and Shank, press Add track, click the tip, then
  the entry. While placing, the cursor is a probe and a dotted line follows it from
  the tip. Esc cancels. Adding a track for a shank that has one replaces it.
* **Select** - click a marker or a line (or a row in the list). The Probe and
  Shank selectors follow.
* **Move** - drag a marker. **Remove** - select the track, press Delete.

The line runs from the tip through the entry and on to the edge of the section's
box. When the tip and the entry are on different sections, each section gets its
own part: on the tip's section the line starts at the tip and points where the
entry would be if it sat at the same place in this section's box; on the entry's
section it starts at the entry and runs away from the tip to the box edge.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from atlastrack.gui.overlay_style import OVERLAY_BLENDING as _OVERLAY_BLENDING
from atlastrack.gui.widgets.separators import section_header
from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.schema import Point2D

if TYPE_CHECKING:
    import napari


_LAYER_TIP = "Tips"
_LAYER_ENTRY = "Entries"
_LAYER_LINES = "Probe tracks"
_LAYER_PREVIEW = "Track preview"

_TIP_PROMPT = "Place track end marker (shank tip)"
_ENTRY_PROMPT = "Place track origin marker (shank entrypoint)"

# Distinct, colour-blind-friendlier cycle; a shank's global ordinal indexes it so
# a shank keeps one colour on its markers, its line and its row in the list.
_SHANK_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9a6324",
    "#800000", "#808000", "#000075", "#a9a9a9",
]

# A press that moves less than this (screen pixels) before release is a click,
# not a pan.
_CLICK_SLOP_PX = 5


def _ray_to_box(start, direction, box) -> np.ndarray | None:
    """Where the ray from ``start`` along ``direction`` leaves ``box``.

    ``start`` and ``direction`` are (x, y); ``box`` is (x0, y0, x1, y1). None if
    the direction is zero or the ray never meets the box.
    """
    d = np.asarray(direction, dtype=float)
    p = np.asarray(start, dtype=float)
    if not np.any(d):
        return None
    x0, y0, x1, y1 = (float(v) for v in box)
    ts = []
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        if d[axis] > 0:
            ts.append((hi - p[axis]) / d[axis])
        elif d[axis] < 0:
            ts.append((lo - p[axis]) / d[axis])
    ts = [t for t in ts if t > 0]
    if not ts:
        return None
    return p + min(ts) * d


def track_segments(tip, tip_box, entry, entry_box) -> list[tuple[np.ndarray, np.ndarray]]:
    """The line segments drawn for one track, as ((x, y), (x, y)) pairs.

    ``tip``/``entry`` are (x, y) slide pixels (or None); ``tip_box``/``entry_box``
    the (x0, y0, x1, y1) boxes of the sections they are on (or None).

    * Same section: tip -> entry -> on to the box edge.
    * Different sections: on the tip's section, from the tip toward the entry as
      placed at the same spot in this box, to the box edge; on the entry's section,
      from the entry away from the tip (placed likewise) to the box edge.
    """
    if tip is None or entry is None:
        return []
    tip = np.asarray(tip, dtype=float)
    entry = np.asarray(entry, dtype=float)
    if tip_box is None or entry_box is None or tuple(tip_box) == tuple(entry_box):
        box = tip_box or entry_box
        end = _ray_to_box(entry, entry - tip, box) if box is not None else None
        return [(tip, end if end is not None else entry)]
    t0 = np.asarray(tip_box[:2], dtype=float)
    e0 = np.asarray(entry_box[:2], dtype=float)
    entry_here = t0 + (entry - e0)        # the entry, same place in the tip's box
    tip_there = e0 + (tip - t0)           # the tip, same place in the entry's box
    out = []
    end = _ray_to_box(tip, entry_here - tip, tip_box)
    if end is not None:
        out.append((tip, end))
    end = _ray_to_box(entry, entry - tip_there, entry_box)
    if end is not None:
        out.append((entry, end))
    return out


def _probe_cursor():
    """A probe-shaped mouse cursor: a thin shank ending in a point (the hot spot)."""
    from qtpy.QtCore import QPointF, Qt
    from qtpy.QtGui import QColor, QCursor, QPainter, QPen, QPixmap, QPolygonF

    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    shank = QPolygonF([QPointF(13, 1), QPointF(19, 1), QPointF(19, 22),
                       QPointF(16, 31), QPointF(13, 22)])
    painter.setPen(QPen(QColor("black"), 2))
    painter.setBrush(QColor("white"))
    painter.drawPolygon(shank)
    painter.setPen(QPen(QColor("black"), 1))
    for y in (6, 11, 16):                      # recording sites, so it reads as a probe
        painter.drawPoint(16, y)
    painter.end()
    return QCursor(pm, 16, 31)


class ClickOverlayWidget(QWidget):
    """Add / select / move / remove each shank's track (tip + entry + line)."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._tip_layer: napari.layers.Points | None = None
        self._entry_layer: napari.layers.Points | None = None
        self._line_layer: napari.layers.Shapes | None = None
        self._preview_layer: napari.layers.Points | None = None
        # None, or the marker being placed: "tip" then "entry".
        self._adding: str | None = None
        self._pending_tip: tuple[float, float] | None = None   # (x, y)
        self._selected: tuple[int, int] | None = None          # (probe, shank)
        self._build_ui()
        # The marker layers are created on first use, not here: empty Points
        # layers at launch made vispy draw a Markers visual with no data, which
        # on some Windows GPUs raised shader / framebuffer errors.
        drag = getattr(viewer, "mouse_drag_callbacks", None)
        if drag is not None:
            drag.append(self._on_mouse_drag)
        move = getattr(viewer, "mouse_move_callbacks", None)
        if move is not None:
            move.append(self._on_mouse_move)
        self._install_key_filter()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        # First thing in the panel, so no gap above it.
        layout.addWidget(section_header("Probe tracks", top_margin=0))

        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe:"))
        self._probe_combo = QComboBox()
        self._probe_combo.setToolTip("The probe the next track is added to.")
        self._probe_combo.currentIndexChanged.connect(self._on_probe_changed)
        probe_row.addWidget(self._probe_combo, 1)
        layout.addLayout(probe_row)

        shank_row = QHBoxLayout()
        shank_row.addWidget(QLabel("Shank:"))
        self._shank_combo = QComboBox()
        self._shank_combo.setToolTip("The shank the next track is added to.")
        shank_row.addWidget(self._shank_combo, 1)
        layout.addLayout(shank_row)
        self._refresh_probe_combo()

        self._add_btn = QPushButton("Add track")
        self._add_btn.setCheckable(True)
        self._add_btn.setToolTip(
            "Place a track for the Probe and Shank above: click the tip (where the "
            "shank ends), then the entry (where it went in). Esc cancels. A shank "
            "that already has a track gets the new one instead.\n"
            "To change a track later: click a marker or the line to select it, drag a "
            "marker to move it, press Delete to remove the track."
        )
        self._add_btn.toggled.connect(self._on_add_toggled)
        layout.addWidget(self._add_btn)

        clear_btn = QPushButton("Clear all tracks")
        clear_btn.setToolTip("Remove every track of every probe.")
        clear_btn.clicked.connect(self._clear_tracks)
        layout.addWidget(clear_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Probe", "Shank", "Tip (px)", "Entry (px)"])
        self._table.setMaximumHeight(200)
        header = self._table.horizontalHeader()
        for col in range(2):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for col in (2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(20)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.cellClicked.connect(self._on_row_clicked)
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

    def _on_probe_changed(self, *_args) -> None:
        self._refresh_shank_combo()

    # ------------------------------------------------------------------
    # Sizes (in slide pixels, so markers look alike at any slide resolution)
    # ------------------------------------------------------------------

    def _section_boxes(self) -> list:
        slide_idx = self._state.active_slide_idx
        slides = self._state.project.slides
        if slide_idx is None or not 0 <= slide_idx < len(slides):
            return []
        return [s.bbox_px for s in slides[slide_idx].sections]

    def _marker_size(self) -> float:
        boxes = self._section_boxes()
        if not boxes:
            return 14.0
        width = float(np.median([b[2] - b[0] for b in boxes]))
        return max(10.0, width / 28.0)

    def _line_width(self) -> float:
        return max(1.5, self._marker_size() / 9.0)

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def _drop_stale_layer_refs(self) -> None:
        """Forget layer references that are no longer in the viewer.

        A project close empties ``viewer.layers`` but leaves these attributes
        pointing at removed layers; drawing on those shows nothing.
        """
        layers = self._viewer.layers
        for attr in ("_tip_layer", "_entry_layer", "_line_layer", "_preview_layer"):
            layer = getattr(self, attr)
            if layer is not None and layer not in layers:
                setattr(self, attr, None)

    def _ensure_points_layers(self) -> None:
        """Create (or find) the Tips / Entries / Probe tracks layers."""
        self._drop_stale_layer_refs()
        layers = self._viewer.layers
        size = self._marker_size()
        if self._line_layer is None:
            if _LAYER_LINES in layers:
                self._line_layer = layers[_LAYER_LINES]  # type: ignore[assignment]
            else:
                self._line_layer = self._viewer.add_shapes(
                    blending=_OVERLAY_BLENDING, name=_LAYER_LINES, ndim=2,
                    edge_width=self._line_width(), face_color="transparent",
                )
        for attr, name, symbol in (("_tip_layer", _LAYER_TIP, "disc"),
                                   ("_entry_layer", _LAYER_ENTRY, "triangle_up")):
            if getattr(self, attr) is not None:
                continue
            if name in layers:
                setattr(self, attr, layers[name])
            else:
                # Empty inside with a bright outline in the shank's colour, so the
                # tissue under the marker stays visible.
                setattr(self, attr, self._viewer.add_points(
                    blending=_OVERLAY_BLENDING, name=name, ndim=2, symbol=symbol,
                    size=size, face_color="transparent", border_width=0.22,
                    border_width_is_relative=True,
                ))
        for layer in (self._line_layer, self._tip_layer, self._entry_layer):
            try:
                layer.mode = "pan_zoom"
            except Exception:  # noqa: BLE001
                pass

    def _bring_to_front(self, layer) -> None:
        """Move ``layer`` to the top of the stack so it stays visible."""
        layers = self._viewer.layers
        try:
            src = layers.index(layer)
            if src != len(layers) - 1:
                layers.move(src, len(layers) - 1)
        except Exception:  # noqa: BLE001
            pass

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

    def arm_tip(self) -> None:
        """Start adding a track for the newest probe's first shank (after Add probe)."""
        self._refresh_probe_combo()
        n_probes = len(self._state.project.probes)
        if n_probes:
            self._probe_combo.setCurrentIndex(n_probes - 1)
            self._shank_combo.setCurrentIndex(0)
        self._add_btn.setChecked(True)

    # ------------------------------------------------------------------
    # Adding a track
    # ------------------------------------------------------------------

    def _on_add_toggled(self, on: bool) -> None:
        if on:
            self._start_add()
        else:
            self._stop_add()

    def _start_add(self) -> None:
        p_idx, s_idx = self._current_ps()
        if p_idx < 0 or s_idx < 0:
            self._status.setText("Add a probe first (above).")
            self._add_btn.blockSignals(True)
            self._add_btn.setChecked(False)
            self._add_btn.blockSignals(False)
            return
        self._ensure_points_layers()
        self._adding = "tip"
        self._pending_tip = None
        if self._line_layer is not None:
            self._viewer.layers.selection.active = self._line_layer
        self._set_canvas_cursor(_probe_cursor())
        label = self._state.project.probes[p_idx].label
        self._status.setText(f"{_TIP_PROMPT} - {label}, shank {s_idx}. Esc cancels.")

    def _stop_add(self) -> None:
        self._adding = None
        self._pending_tip = None
        self._clear_preview()
        self._set_canvas_cursor(None)
        from qtpy.QtWidgets import QToolTip

        QToolTip.hideText()
        if self._add_btn.isChecked():
            self._add_btn.blockSignals(True)
            self._add_btn.setChecked(False)
            self._add_btn.blockSignals(False)

    def _cancel_add(self) -> None:
        self._stop_add()
        self._status.setText("Track not added.")

    def _place(self, x: float, y: float) -> None:
        """A click while adding: the tip first, then the entry (which completes it)."""
        if self._adding == "tip":
            self._pending_tip = (x, y)
            self._adding = "entry"
            self._status.setText(f"{_ENTRY_PROMPT}. Esc cancels.")
            self._update_preview(x, y)
            return
        if self._adding != "entry" or self._pending_tip is None:
            return
        p_idx, s_idx = self._current_ps()
        shank = self._state.project.probes[p_idx].shanks[s_idx]
        tx, ty = self._pending_tip
        shank.tip_px = Point2D(x_px=float(tx), y_px=float(ty))
        shank.tip_section_idx = self._find_section_for_point(tx, ty)
        shank.entry_px = Point2D(x_px=float(x), y_px=float(y))
        shank.entry_section_idx = self._find_section_for_point(x, y)
        self._selected = (p_idx, s_idx)
        self._stop_add()
        self._rebuild_markers()
        self._refresh_table()
        label = self._state.project.probes[p_idx].label
        self._status.setText(f"Track added: {label}, shank {s_idx}.")

    # ------------------------------------------------------------------
    # Preview (dotted line from the tip to the cursor) and the cursor
    # ------------------------------------------------------------------

    def _update_preview(self, x: float, y: float) -> None:
        if self._adding != "entry" or self._pending_tip is None:
            return
        tip = np.array(self._pending_tip, dtype=float)
        end = np.array([x, y], dtype=float)
        length = float(np.linalg.norm(end - tip))
        spacing = self._line_width() * 3.0
        n = max(2, int(length / spacing) + 1)
        pts = tip[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (end - tip)[None, :]
        data = pts[:, ::-1]                        # (y, x) for napari
        color = self._color_for(*self._current_ps())
        self._drop_stale_layer_refs()
        if self._preview_layer is None:
            self._preview_layer = self._viewer.add_points(
                data, blending=_OVERLAY_BLENDING, name=_LAYER_PREVIEW, ndim=2,
                size=self._line_width() * 1.4, face_color=color, border_width=0,
            )
        else:
            self._preview_layer.data = data
            self._preview_layer.face_color = color
        if self._line_layer is not None:
            self._viewer.layers.selection.active = self._line_layer

    def _clear_preview(self) -> None:
        self._drop_stale_layer_refs()
        if self._preview_layer is not None:
            try:
                self._viewer.layers.remove(self._preview_layer)
            except Exception:  # noqa: BLE001
                pass
            self._preview_layer = None

    def _canvas_widget(self):
        try:
            return self._viewer.window._qt_viewer.canvas.native
        except Exception:  # noqa: BLE001 - headless / fake viewer
            return None

    def _set_canvas_cursor(self, cursor) -> None:
        widget = self._canvas_widget()
        if widget is None:
            return
        if cursor is None:
            widget.unsetCursor()
            try:   # let napari put back the cursor its own mode wants
                style = self._viewer.cursor.style
                self._viewer.cursor.style = "standard"
                self._viewer.cursor.style = style
            except Exception:  # noqa: BLE001
                pass
        else:
            widget.setCursor(cursor)

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _interactive(self) -> bool:
        """Clicks on the canvas are ours while adding, or while the Probes tab shows."""
        return self._adding is not None or self.isVisible()

    @staticmethod
    def _event_xy(event) -> tuple[float, float]:
        y, x = (float(v) for v in event.position[-2:])
        return x, y

    def _on_mouse_move(self, _viewer, event) -> None:
        if self._adding is None:
            return
        x, y = self._event_xy(event)
        if self._adding == "entry":
            self._update_preview(x, y)
        from qtpy.QtGui import QCursor
        from qtpy.QtWidgets import QToolTip

        prompt = _TIP_PROMPT if self._adding == "tip" else _ENTRY_PROMPT
        QToolTip.showText(QCursor.pos(), prompt, self._canvas_widget())

    def _on_mouse_drag(self, _viewer, event):
        """Click to place (while adding) or select; drag a marker to move it."""
        if not self._interactive() or getattr(event, "button", 1) != 1:
            return
        start_screen = np.asarray(getattr(event, "pos", (0, 0)), dtype=float)
        x, y = self._event_xy(event)

        if self._adding is not None:
            # Drags still pan the view; only a click (press + release in place) places.
            moved = False
            yield
            while event.type == "mouse_move":
                if np.linalg.norm(np.asarray(event.pos, dtype=float) - start_screen) > _CLICK_SLOP_PX:
                    moved = True
                yield
            if not moved:
                self._place(x, y)
            return

        hit = self._hit_test(x, y)
        if hit is None:
            return
        p_idx, s_idx, kind = hit
        self._select_track(p_idx, s_idx)
        if kind not in ("tip", "entry"):
            return
        # Drag the marker: stop the view panning while it moves.
        try:
            self._viewer.camera.mouse_pan = False
        except Exception:  # noqa: BLE001
            pass
        try:
            yield
            while event.type == "mouse_move":
                mx, my = self._event_xy(event)
                self._move_marker(p_idx, s_idx, kind, mx, my, commit=False)
                yield
            mx, my = self._event_xy(event)
            if np.linalg.norm(np.asarray(event.pos, dtype=float) - start_screen) > _CLICK_SLOP_PX:
                self._move_marker(p_idx, s_idx, kind, mx, my, commit=True)
        finally:
            try:
                self._viewer.camera.mouse_pan = True
            except Exception:  # noqa: BLE001
                pass

    def _hit_test(self, x: float, y: float) -> tuple[int, int, str] | None:
        """The (probe, shank, 'tip'|'entry'|'line') under a click, or None."""
        radius = self._marker_size() * 0.7
        best = None
        for p_idx, probe in enumerate(self._state.project.probes):
            for s_idx, shank in enumerate(probe.shanks):
                for kind, pt in (("tip", shank.tip_px), ("entry", shank.entry_px)):
                    if pt is None:
                        continue
                    d = float(np.hypot(pt.x_px - x, pt.y_px - y))
                    if d <= radius and (best is None or d < best[0]):
                        best = (d, p_idx, s_idx, kind)
        if best is not None:
            return best[1], best[2], best[3]
        tol = max(self._line_width() * 2.0, self._marker_size() * 0.35)
        p = np.array([x, y], dtype=float)
        for p_idx, probe in enumerate(self._state.project.probes):
            for s_idx, _shank in enumerate(probe.shanks):
                for a, b in self._segments_for(p_idx, s_idx):
                    ab = b - a
                    denom = float(ab @ ab)
                    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0, 1))
                    d = float(np.linalg.norm(p - (a + t * ab)))
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, p_idx, s_idx, "line")
        return None if best is None else (best[1], best[2], best[3])

    def _move_marker(self, p_idx, s_idx, kind, x, y, *, commit: bool) -> None:
        shank = self._state.project.probes[p_idx].shanks[s_idx]
        pt = Point2D(x_px=float(x), y_px=float(y))
        if kind == "tip":
            shank.tip_px = pt
            if commit:
                shank.tip_section_idx = self._find_section_for_point(x, y)
        else:
            shank.entry_px = pt
            if commit:
                shank.entry_section_idx = self._find_section_for_point(x, y)
        self._rebuild_markers()
        if commit:
            self._refresh_table()

    # ------------------------------------------------------------------
    # Selection and removal
    # ------------------------------------------------------------------

    def _select_track(self, p_idx: int, s_idx: int) -> None:
        self._selected = (p_idx, s_idx)
        self._probe_combo.setCurrentIndex(p_idx)
        self._shank_combo.setCurrentIndex(s_idx)
        self._rebuild_markers()
        self._refresh_table()
        label = self._state.project.probes[p_idx].label
        self._status.setText(
            f"Selected: {label}, shank {s_idx}. Drag a marker to move it; Delete removes "
            "the track."
        )

    def _on_row_clicked(self, row: int, _col: int) -> None:
        item = self._table.item(row, 0)
        if item is None:
            return
        from qtpy.QtCore import Qt

        key = item.data(Qt.ItemDataRole.UserRole)
        if key:
            self._select_track(int(key[0]), int(key[1]))

    def _delete_selected(self) -> bool:
        if self._selected is None:
            return False
        p_idx, s_idx = self._selected
        probes = self._state.project.probes
        if not (0 <= p_idx < len(probes) and 0 <= s_idx < len(probes[p_idx].shanks)):
            self._selected = None
            return False
        shank = probes[p_idx].shanks[s_idx]
        shank.tip_px = shank.tip_section_idx = None
        shank.entry_px = shank.entry_section_idx = None
        self._selected = None
        self._rebuild_markers()
        self._refresh_table()
        self._status.setText(f"Track removed: {probes[p_idx].label}, shank {s_idx}.")
        return True

    def _install_key_filter(self) -> None:
        """Esc cancels adding; Delete / Backspace removes the selected track."""
        widget = self._canvas_widget()
        if widget is None:
            return
        from qtpy.QtCore import QEvent, QObject, Qt

        overlay = self

        class _Keys(QObject):
            def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt name
                if event.type() != QEvent.Type.KeyPress:
                    return False
                key = event.key()
                if key == Qt.Key.Key_Escape and overlay._adding is not None:
                    overlay._cancel_add()
                    return True
                if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and overlay._interactive():
                    return overlay._delete_selected()
                return False

        self._key_filter = _Keys(self)
        widget.installEventFilter(self._key_filter)

    def _clear_tracks(self) -> None:
        n = sum(1 for p in self._state.project.probes for s in p.shanks
                if s.tip_px is not None or s.entry_px is not None)
        if n and self.isVisible():
            from qtpy.QtWidgets import QMessageBox

            answer = QMessageBox.question(
                self, "Clear all tracks", f"Remove all {n} track(s) of every probe?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._stop_add()
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                shank.tip_px = shank.tip_section_idx = None
                shank.entry_px = shank.entry_section_idx = None
        self._selected = None
        self._rebuild_markers()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _box_of(self, section_idx: int | None):
        if section_idx is None:
            return None
        for slide in self._state.project.slides:
            for section in slide.sections:
                if section.index == section_idx:
                    return section.bbox_px
        return None

    def _segments_for(self, p_idx: int, s_idx: int) -> list:
        shank = self._state.project.probes[p_idx].shanks[s_idx]
        if shank.tip_px is None or shank.entry_px is None:
            return []
        tip_box = self._box_of(shank.tip_section_idx)
        entry_box = self._box_of(shank.entry_section_idx)
        return track_segments((shank.tip_px.x_px, shank.tip_px.y_px), tip_box,
                              (shank.entry_px.x_px, shank.entry_px.y_px), entry_box)

    def refresh_after_load(self) -> None:
        """Redraw the tracks and the list from a freshly-loaded project."""
        self._stop_add()
        self._selected = None
        self._drop_stale_layer_refs()
        self._refresh_probe_combo()
        self._rebuild_markers()
        self._refresh_table()

    def _rebuild_markers(self) -> None:
        """Redraw markers and lines (colour per shank; the selected track larger)."""
        tips, tip_c, tip_sel = [], [], []
        entries, ent_c, ent_sel = [], [], []
        lines, line_c, line_w = [], [], []
        size, width = self._marker_size(), self._line_width()
        for p_idx, probe in enumerate(self._state.project.probes):
            for s_idx, shank in enumerate(probe.shanks):
                color = self._color_for(p_idx, s_idx)
                selected = self._selected == (p_idx, s_idx)
                if shank.tip_px is not None:
                    tips.append([shank.tip_px.y_px, shank.tip_px.x_px])
                    tip_c.append(color)
                    tip_sel.append(selected)
                if shank.entry_px is not None:
                    entries.append([shank.entry_px.y_px, shank.entry_px.x_px])
                    ent_c.append(color)
                    ent_sel.append(selected)
                for a, b in self._segments_for(p_idx, s_idx):
                    lines.append(np.array([[a[1], a[0]], [b[1], b[0]]]))
                    line_c.append(color)
                    line_w.append(width * (2.0 if selected else 1.0))
        if not tips and not entries and self._tip_layer is None:
            return  # nothing to draw - avoid creating empty layers
        self._ensure_points_layers()
        for layer, pts, cols, sel in ((self._tip_layer, tips, tip_c, tip_sel),
                                      (self._entry_layer, entries, ent_c, ent_sel)):
            if layer is None:
                continue
            layer.data = np.array(pts, dtype=float) if pts else np.empty((0, 2))
            if pts:
                layer.face_color = "transparent"
                layer.border_color = cols
                layer.size = np.where(np.array(sel), size * 1.6, size)
        if self._line_layer is not None:
            self._line_layer.data = []
            if lines:
                self._line_layer.add_lines(lines, edge_color=line_c, edge_width=line_w)
        for layer in (self._line_layer, self._tip_layer, self._entry_layer):
            if layer is not None:
                self._bring_to_front(layer)

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
    # List
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor

        rows = []
        for p_idx, probe in enumerate(self._state.project.probes):
            for s_idx, shank in enumerate(probe.shanks):
                if shank.tip_px is None and shank.entry_px is None:
                    continue
                rows.append((p_idx, s_idx, probe.label, shank))
        self._table.setRowCount(len(rows))
        for i, (p_idx, s_idx, label, shank) in enumerate(rows):
            color = QColor(self._color_for(p_idx, s_idx))
            first = QTableWidgetItem(str(label))
            first.setData(Qt.ItemDataRole.DecorationRole, color)   # colour swatch
            first.setData(Qt.ItemDataRole.UserRole, (p_idx, s_idx))
            cells = [first, QTableWidgetItem(str(shank.index))]
            for pt in (shank.tip_px, shank.entry_px):
                cells.append(QTableWidgetItem("" if pt is None else f"{pt.x_px:.0f}, {pt.y_px:.0f}"))
            for col, item in enumerate(cells):
                if col:
                    item.setForeground(color)
                self._table.setItem(i, col, item)
            if self._selected == (p_idx, s_idx):
                self._table.selectRow(i)
