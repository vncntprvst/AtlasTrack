"""Section ordering and AP-spacing panel."""
from __future__ import annotations

import numpy as np
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

from histo_to_ccf.gui.widgets.separators import section_header
from histo_to_ccf.io.ccf_coords import bregma_ap_for_display
from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.sectioning.ordering import geometric_order


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

        # This panel sits below the atlas browser in the Atlas tab, so it needs
        # its own heading to read as a separate job from AP assignment above.
        layout.addWidget(section_header("Section order and spacing", top_margin=22))

        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("Section spacing (µm):"))
        self._spacing = QDoubleSpinBox()
        self._spacing.setRange(1.0, 10_000.0)
        self._spacing.setValue(80.0)
        self._spacing.setSingleStep(10.0)
        # Persist the chosen spacing on the project so it reloads with it.
        self._spacing.valueChanged.connect(self._store_spacing)
        spacing_row.addWidget(self._spacing)
        layout.addLayout(spacing_row)

        dir_row = QHBoxLayout()
        dir_lbl = QLabel("Direction:")
        dir_tip = (
            "Which way the section sequence runs. With 'Anterior → Posterior' the "
            "FIRST section (top of the list below) is the most anterior; AP then "
            "steps posteriorly down the list. The list marks the anterior/posterior "
            "ends so you can check the numbering matches your slides."
        )
        dir_lbl.setToolTip(dir_tip)
        dir_row.addWidget(dir_lbl)
        self._ant_post = QRadioButton("Anterior → Posterior")
        self._ant_post.setChecked(True)
        self._post_ant = QRadioButton("Posterior → Anterior")
        for rb in (self._ant_post, self._post_ant):
            rb.setToolTip(dir_tip)
            rb.toggled.connect(self._refresh_list)  # re-label the end markers
        dir_grp = QButtonGroup(self)
        dir_grp.addButton(self._ant_post)
        dir_grp.addButton(self._post_ant)
        dir_row.addWidget(self._ant_post)
        dir_row.addWidget(self._post_ant)
        layout.addLayout(dir_row)

        # Slide layout order: how the on-slide grid maps to the AP sequence.
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Section order:"))
        self._col_first = QRadioButton("Column-first")
        self._col_first.setChecked(True)
        self._row_first = QRadioButton("Row-first")
        order_grp = QButtonGroup(self)
        order_grp.addButton(self._col_first)
        order_grp.addButton(self._row_first)
        self._col_first.setToolTip(
            "Number sections down column 0, then column 1,  (the lab default)."
        )
        self._row_first.setToolTip("Number sections across row 0, then row 1,  (reading order).")
        order_row.addWidget(self._col_first)
        order_row.addWidget(self._row_first)
        layout.addLayout(order_row)

        resort_btn = QPushButton("Re-sort by slide layout")
        resort_btn.setToolTip(
            "Recompute the section order from the boxes' positions on the slide,\n"
            "using the Column/Row choice above. Overrides manual drag ordering."
        )
        resort_btn.clicked.connect(self._resort_sections)
        layout.addWidget(resort_btn)

        anchor_row = QHBoxLayout()
        anchor_lbl = QLabel("Anchor section (1 = first):")
        anchor_lbl.setToolTip(
            "Which section in the AP order keeps its AP while the rest are\n"
            "spaced around it. Counted from 1, matching the list below."
        )
        anchor_row.addWidget(anchor_lbl)
        self._anchor_spin = QDoubleSpinBox()
        self._anchor_spin.setRange(1, 9999)
        self._anchor_spin.setDecimals(0)
        anchor_row.addWidget(self._anchor_spin)
        layout.addLayout(anchor_row)

        apply_btn = QPushButton("Apply spacing (all sections)")
        apply_btn.setToolTip(
            "Set every section's AP from the anchor section, stepping by the "
            "spacing above along the AP sequence.\n"
            "Overwrites any per-section AP you assigned in the Atlas tab - use "
            "this when sections are evenly spaced from one known level."
        )
        apply_btn.clicked.connect(self._apply_spacing)
        layout.addWidget(apply_btn)

        interp_btn = QPushButton("Interpolate AP (fill gaps)")
        interp_btn.setToolTip(
            "Keep the AP values you assigned to individual sections in the Atlas "
            "tab and fill in the rest by interpolating between them along the AP "
            "sequence. Use this when you assigned a few key levels by hand."
        )
        interp_btn.clicked.connect(self._interpolate_ap)
        layout.addWidget(interp_btn)

        refresh_btn = QPushButton("Refresh list")
        refresh_btn.clicked.connect(self._refresh_list)
        layout.addWidget(refresh_btn)

        layout.addWidget(QLabel("Sections in AP order (top = first; drag to reorder):"))
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
    # Public accessors (used by the Atlas matcher to stay in sync)
    # ------------------------------------------------------------------

    def current_spacing(self) -> float:
        return self._spacing.value()

    def set_spacing(self, value: float) -> None:
        self._spacing.setValue(value)

    def refresh(self) -> None:
        """Re-read the section list (e.g. after the matcher reorders/assigns AP)."""
        self._refresh_list()

    def refresh_after_load(self) -> None:
        """Repopulate spacing + the section list from a freshly-loaded project."""
        spacing = self._state.project.section_spacing_um
        if spacing is not None:
            self._spacing.blockSignals(True)
            self._spacing.setValue(spacing)
            self._spacing.blockSignals(False)
        self._refresh_list()

    def _store_spacing(self, value: float) -> None:
        """Mirror the spacing onto the project so a save captures it."""
        self._state.project.section_spacing_um = float(value)

    # ------------------------------------------------------------------

    def _active_slide(self):
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or slide_idx >= len(self._state.project.slides):
            return None
        return self._state.project.slides[slide_idx]

    def _item_text(self, section, position: int) -> str:
        """One list row. ``position`` is the 1-based place in the AP order.

        Counted from 1 for display; ``section.index`` stays the stored id that
        names the transform sidecar and that shank picks point at.

        An instance method rather than a static one because the bregma anchor the AP
        is shown against depends on the project's atlas.
        """
        if section.plane is not None:
            bregma_ap = bregma_ap_for_display(self._state.project.atlas.name)
            ap_bregma = bregma_ap - section.plane.ap_um
            ap_str = f"AP {ap_bregma:+.0f} µm"
        else:
            ap_str = "AP -"
        return f"Section {position}   ·   {ap_str}"

    def _apply_spacing(self) -> None:
        """Propagate evenly-spaced AP values from the anchor section outward."""
        slide = self._active_slide()
        if slide is None:
            self._status.setText("No slide loaded.")
            return
        if not slide.sections:
            self._status.setText("No sections detected.")
            return

        from histo_to_ccf.sectioning.ap_series import assign_section_ap

        spacing = self._spacing.value()
        # The spin counts from 1; assign_section_ap wants the stored section id.
        ordered = sorted(slide.sections, key=lambda s: s.ap_order)
        pos = max(0, min(int(self._anchor_spin.value()) - 1, len(ordered) - 1))
        n, mode = assign_section_ap(
            slide.sections,
            spacing_um=spacing,
            anchor_index=ordered[pos].index,
            forward=self._ant_post.isChecked(),
            bregma_ap_um=bregma_ap_for_display(self._state.project.atlas.name),
        )
        # With slide numbers the spacing is per slide-number step, so an unevenly
        # sampled series gets the AP gaps it actually has.
        per = "per slide" if mode == "slide_number" else "per section"
        self._status.setText(f"Applied spacing={spacing:.0f} µm {per} to {n} sections")
        self._refresh_list()

    def _resort_sections(self) -> None:
        """Recompute ap_order from box positions using the Column/Row choice."""
        slide = self._active_slide()
        if slide is None or not slide.sections:
            self._status.setText("No sections to sort.")
            return
        sections = list(slide.sections)
        ranks = geometric_order(
            [s.bbox_px for s in sections],
            column_first=self._col_first.isChecked(),
        )
        for section, rank in zip(sections, ranks):
            section.ap_order = rank
        mode = "column-first" if self._col_first.isChecked() else "row-first"
        self._status.setText(f"Re-sorted {len(sections)} sections ({mode}).")
        self._refresh_list()

    def _interpolate_ap(self) -> None:
        """Fill un-assigned section AP by interpolating between assigned ones."""
        slide = self._active_slide()
        if slide is None or not slide.sections:
            self._status.setText("No sections.")
            return
        from histo_to_ccf.project.schema import PlaneParams

        sections = sorted(slide.sections, key=lambda s: s.ap_order)
        known = [(i, s.plane.ap_um) for i, s in enumerate(sections) if s.plane is not None]
        if len(known) < 2:
            self._status.setText("Assign AP to at least two sections first (Atlas tab).")
            return

        positions = [i for i, _ in known]
        values = [v for _, v in known]
        for i, section in enumerate(sections):
            if section.plane is not None:
                continue  # keep hand-assigned values
            ap = float(np.interp(i, positions, values))  # extrapolates flat at ends
            section.plane = PlaneParams(ap_um=ap)
        self._status.setText(f"Interpolated AP for {len(sections) - len(known)} section(s).")
        self._refresh_list()

    def _refresh_list(self) -> None:
        slide = self._active_slide()
        if slide is None:
            return
        sections = sorted(slide.sections, key=lambda s: s.ap_order)
        forward = self._ant_post.isChecked()
        first_end = "anterior" if forward else "posterior"
        last_end = "posterior" if forward else "anterior"
        self._list.blockSignals(True)
        self._list.clear()
        for i, s in enumerate(sections):
            text = self._item_text(s, i + 1)
            if i == 0:
                text += f"   ◄ {first_end} end"
            elif i == len(sections) - 1:
                text += f"   ◄ {last_end} end"
            item = QListWidgetItem(text)
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
                item.setText(self._item_text(section, pos + 1))
        self._status.setText("Reordered sections - apply spacing to update AP values.")
