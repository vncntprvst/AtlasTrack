"""Section ordering and AP-spacing panel."""
from __future__ import annotations

from qtpy.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
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


class OrderingPanelWidget(QWidget):
    """Set section spacing, direction, and review AP assignments."""

    def __init__(
        self,
        state: WorkflowState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Section spacing (µm):"))
        self._spacing = QDoubleSpinBox()
        self._spacing.setRange(1.0, 10_000.0)
        self._spacing.setValue(200.0)
        self._spacing.setSingleStep(10.0)
        spacing_row.addWidget(self._spacing)
        layout.addLayout(spacing_row)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Direction:"))
        self._ant_post = QRadioButton("Anterior → Posterior")
        self._ant_post.setChecked(True)
        self._post_ant = QRadioButton("Posterior → Anterior")
        dir_grp = QButtonGroup(self)
        dir_grp.addButton(self._ant_post)
        dir_grp.addButton(self._post_ant)
        dir_row.addWidget(self._ant_post)
        dir_row.addWidget(self._post_ant)
        layout.addLayout(dir_row)

        anchor_row = QHBoxLayout()
        anchor_row.addWidget(QLabel("Anchor section idx:"))
        self._anchor_spin = QDoubleSpinBox()
        self._anchor_spin.setRange(0, 9999)
        self._anchor_spin.setDecimals(0)
        anchor_row.addWidget(self._anchor_spin)
        layout.addLayout(anchor_row)

        apply_btn = QPushButton("Apply spacing")
        apply_btn.clicked.connect(self._apply_spacing)
        layout.addWidget(apply_btn)

        refresh_btn = QPushButton("Refresh table")
        refresh_btn.clicked.connect(self._refresh_table)
        layout.addWidget(refresh_btn)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Section idx", "ap_order", "AP µm"])
        self._table.setMaximumHeight(220)
        layout.addWidget(self._table)

        self._status = QLabel("")
        layout.addWidget(self._status)
        layout.addStretch()

    def _apply_spacing(self) -> None:
        """Propagate evenly-spaced AP values from the anchor section outward."""
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            self._status.setText("No slide loaded.")
            return
        slide = self._state.project.slides[slide_idx]
        if not slide.sections:
            self._status.setText("No sections detected.")
            return

        spacing = self._spacing.value()
        anchor_idx = int(self._anchor_spin.value())
        forward = self._ant_post.isChecked()

        # Sort sections by ap_order
        sections = sorted(slide.sections, key=lambda s: s.ap_order)
        anchor_sec = next((s for s in sections if s.index == anchor_idx), sections[0])
        anchor_ap = anchor_sec.plane.ap_um if anchor_sec.plane is not None else 5400.0

        anchor_order = sections.index(anchor_sec)
        for i, section in enumerate(sections):
            delta = i - anchor_order
            ap = anchor_ap + delta * spacing * (1 if forward else -1)
            if section.plane is not None:
                section.plane = section.plane.model_copy(update={"ap_um": ap})
            else:
                from histo_to_ccf.project.schema import PlaneParams
                section.plane = PlaneParams(
                    ap_um=ap, midline_px=0.0, dorsal_surface_px=0.0, pixel_size_um=1.0
                )
        self._status.setText(f"Applied spacing={spacing:.0f} µm to {len(sections)} sections")
        self._refresh_table()

    def _refresh_table(self) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            return
        slide = self._state.project.slides[slide_idx]
        sections = sorted(slide.sections, key=lambda s: s.ap_order)
        self._table.setRowCount(len(sections))
        for i, s in enumerate(sections):
            ap_str = f"{s.plane.ap_um:.0f}" if s.plane else "—"
            self._table.setItem(i, 0, QTableWidgetItem(str(s.index)))
            self._table.setItem(i, 1, QTableWidgetItem(str(s.ap_order)))
            self._table.setItem(i, 2, QTableWidgetItem(ap_str))
