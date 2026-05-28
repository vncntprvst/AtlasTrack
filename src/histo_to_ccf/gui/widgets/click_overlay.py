"""Tip / entry-point click overlay widget.

Manages two napari Points layers (Tips, Entries) and maps viewer clicks to
Section pixel coordinates stored in the project Shank objects.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
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
        self._build_ui()
        self._init_layers()

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

        shank_row = QHBoxLayout()
        shank_row.addWidget(QLabel("Shank:"))
        self._shank_spin = QSpinBox()
        self._shank_spin.setRange(0, 7)
        shank_row.addWidget(self._shank_spin)
        layout.addLayout(shank_row)

        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe idx:"))
        self._probe_spin = QSpinBox()
        self._probe_spin.setRange(0, 31)
        probe_row.addWidget(self._probe_spin)
        layout.addLayout(probe_row)

        add_btn = QPushButton("Pick point (then click viewer)")
        add_btn.clicked.connect(self._activate_pick_mode)
        layout.addWidget(add_btn)

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

    def _init_layers(self) -> None:
        if _LAYER_TIP not in self._viewer.layers:
            self._tip_layer = self._viewer.add_points(
                name=_LAYER_TIP, face_color="red", size=12, ndim=2
            )
        else:
            self._tip_layer = self._viewer.layers[_LAYER_TIP]  # type: ignore[assignment]

        if _LAYER_ENTRY not in self._viewer.layers:
            self._entry_layer = self._viewer.add_points(
                name=_LAYER_ENTRY, face_color="cyan", size=12, ndim=2
            )
        else:
            self._entry_layer = self._viewer.layers[_LAYER_ENTRY]  # type: ignore[assignment]

        # Listen for new points on both layers
        self._tip_layer.events.data.connect(self._on_tip_data_changed)
        self._entry_layer.events.data.connect(self._on_entry_data_changed)

    def _activate_pick_mode(self) -> None:
        """Set the active layer to tip or entry and switch to add mode."""
        layer = self._tip_layer if self._mode_tip.isChecked() else self._entry_layer
        if layer is None:
            return
        self._viewer.layers.selection.active = layer
        layer.mode = "add"

    # ------------------------------------------------------------------
    # Point event handlers
    # ------------------------------------------------------------------

    def _on_tip_data_changed(self, event=None) -> None:
        if self._tip_layer is None:
            return
        pts = self._tip_layer.data
        if len(pts) == 0:
            return
        last = pts[-1]  # most recently added point (row, col in image coords)
        self._store_point(float(last[1]), float(last[0]), mode="tip")
        self._refresh_table()

    def _on_entry_data_changed(self, event=None) -> None:
        if self._entry_layer is None:
            return
        pts = self._entry_layer.data
        if len(pts) == 0:
            return
        last = pts[-1]
        self._store_point(float(last[1]), float(last[0]), mode="entry")
        self._refresh_table()

    def _store_point(self, x_px: float, y_px: float, mode: str) -> None:
        """Store a clicked coordinate into the project schema."""
        probes = self._state.project.probes
        p_idx = self._probe_spin.value()
        s_idx = self._shank_spin.value()
        if p_idx >= len(probes):
            return
        shanks = probes[p_idx].shanks
        if s_idx >= len(shanks):
            return
        shank = shanks[s_idx]
        # Determine the section index from the active slide
        section_idx = self._find_section_for_point(x_px, y_px)
        pt = Point2D(x_px=x_px, y_px=y_px)
        if mode == "tip":
            shank.tip_px = pt
            shank.tip_section_idx = section_idx
        else:
            shank.entry_px = pt
            shank.entry_section_idx = section_idx

    def _find_section_for_point(self, x_px: float, y_px: float) -> int | None:
        """Return the section index whose bbox contains the given slide pixel."""
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or slide_idx >= len(self._state.project.slides):
            return None
        slide = self._state.project.slides[slide_idx]
        for section in slide.sections:
            x0, y0, x1, y1 = section.bbox_px
            if x0 <= x_px < x1 and y0 <= y_px < y1:
                return section.index
        return None

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

    def _clear_points(self) -> None:
        if self._tip_layer is not None:
            self._tip_layer.data = np.empty((0, 2))
        if self._entry_layer is not None:
            self._entry_layer.data = np.empty((0, 2))
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                shank.tip_px = None
                shank.tip_section_idx = None
                shank.entry_px = None
                shank.entry_section_idx = None
        self._refresh_table()
