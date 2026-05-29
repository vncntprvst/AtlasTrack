"""Section ordering and AP-spacing panel."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM
from histo_to_ccf.gui.workflow import WorkflowState


class OrderingPanelWidget(QWidget):
    """Set section spacing, direction, and reorder sections by drag-and-drop."""

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
        self._spacing.setValue(80.0)
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

        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh_list)
        layout.addWidget(refresh_btn)

        layout.addWidget(QLabel("Sections (drag to reorder, top = first):"))
        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setToolTip(
            "Drag a section up or down to change its position in the AP sequence."
        )
        self._list.setMaximumHeight(240)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list)

        self._status = QLabel("")
        layout.addWidget(self._status)
        layout.addStretch()

    # ------------------------------------------------------------------

    def _active_slide(self):
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or slide_idx >= len(self._state.project.slides):
            return None
        return self._state.project.slides[slide_idx]

    @staticmethod
    def _item_text(section) -> str:
        if section.plane is not None:
            ap_bregma = BREGMA_AP_FROM_ORIGIN_UM - section.plane.ap_um
            ap_str = f"AP {ap_bregma:+.0f} µm"
        else:
            ap_str = "AP —"
        return f"Section {section.index}   ·   {ap_str}"

    def _apply_spacing(self) -> None:
        """Propagate evenly-spaced AP values from the anchor section outward."""
        slide = self._active_slide()
        if slide is None:
            self._status.setText("No slide loaded.")
            return
        if not slide.sections:
            self._status.setText("No sections detected.")
            return

        spacing = self._spacing.value()
        anchor_idx = int(self._anchor_spin.value())
        forward = self._ant_post.isChecked()

        sections = sorted(slide.sections, key=lambda s: s.ap_order)
        anchor_sec = next((s for s in sections if s.index == anchor_idx), sections[0])
        anchor_ap = (
            anchor_sec.plane.ap_um
            if anchor_sec.plane is not None
            else BREGMA_AP_FROM_ORIGIN_UM
        )

        anchor_order = sections.index(anchor_sec)
        for i, section in enumerate(sections):
            delta = i - anchor_order
            ap = anchor_ap + delta * spacing * (1 if forward else -1)
            if section.plane is not None:
                section.plane = section.plane.model_copy(update={"ap_um": ap})
            else:
                from histo_to_ccf.project.schema import PlaneParams
                section.plane = PlaneParams(ap_um=ap)
        self._status.setText(f"Applied spacing={spacing:.0f} µm to {len(sections)} sections")
        self._refresh_list()

    def _refresh_list(self) -> None:
        slide = self._active_slide()
        if slide is None:
            return
        sections = sorted(slide.sections, key=lambda s: s.ap_order)
        self._list.blockSignals(True)
        self._list.clear()
        for s in sections:
            item = QListWidgetItem(self._item_text(s))
            item.setData(Qt.UserRole, s.index)
            self._list.addItem(item)
        self._list.blockSignals(False)

    def _on_rows_moved(self, *_args) -> None:
        """Reassign ap_order to match the new top-to-bottom list order."""
        slide = self._active_slide()
        if slide is None:
            return
        by_index = {s.index: s for s in slide.sections}
        for pos in range(self._list.count()):
            item = self._list.item(pos)
            section = by_index.get(item.data(Qt.UserRole))
            if section is not None:
                section.ap_order = pos
                item.setText(self._item_text(section))
        self._status.setText("Reordered sections — apply spacing to update AP values.")
