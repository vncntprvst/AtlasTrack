"""Image-prep tools: flip H/V and per-channel saturation/level adjustments."""
from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.widgets.separators import section_header
from histo_to_ccf.gui.workflow import WorkflowState

_CHANNELS = ("R", "G", "B")


class ImageToolsWidget(QWidget):
    """Flip H/V and per-channel level controls, scoped to slide or section."""

    def __init__(
        self,
        state: WorkflowState,
        on_display_changed: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._on_display_changed = on_display_changed
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Scope toggle
        scope_box = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_box)
        scope_row = QHBoxLayout()
        self._scope_whole = QRadioButton("Whole slide")
        self._scope_whole.setChecked(True)
        self._scope_section = QRadioButton("Selected section")
        scope_grp = QButtonGroup(self)
        scope_grp.addButton(self._scope_whole)
        scope_grp.addButton(self._scope_section)
        scope_row.addWidget(self._scope_whole)
        scope_row.addWidget(self._scope_section)
        scope_layout.addLayout(scope_row)

        # Which section the "Selected section" scope acts on. Without this the
        # active section was never set from the main UI, so section-scoped flips
        # and levels silently did nothing.
        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Section:"))
        self._section_combo = QComboBox()
        self._section_combo.setToolTip(
            "The section that 'Selected section' scope applies to. "
            "Choosing 'Selected section' refreshes this list."
        )
        self._section_combo.currentIndexChanged.connect(self._on_section_combo_changed)
        sec_row.addWidget(self._section_combo, 1)
        scope_layout.addLayout(sec_row)
        # Refresh the section list whenever section scope is chosen.
        self._scope_section.toggled.connect(self._on_scope_section_toggled)

        # Rotation is its own job, not a footnote to the scope selector: it gets a
        # box of its own, matching Flip below.
        rot_box = QGroupBox("Rotation")
        rot_box_layout = QVBoxLayout(rot_box)
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Angle (°):"))
        self._rotation_spin = QDoubleSpinBox()
        self._rotation_spin.setRange(-180.0, 180.0)
        self._rotation_spin.setSingleStep(0.5)
        self._rotation_spin.setDecimals(2)
        self._rotation_spin.setToolTip(
            "Rotate this section in the working image, the way Flip does. Because "
            "registration is computed on that image, rotating a section that is "
            "already registered invalidates its fit - re-register it.\n"
            "For a straight series you usually do not need this: the section-series "
            "export removes DeepSlice's measured tilt on its own."
        )
        self._rotation_spin.valueChanged.connect(self._on_rotation_changed)
        rot_row.addWidget(self._rotation_spin, 1)
        # DeepSlice knows the in-plane angle, but applying it automatically would
        # rotate every section the moment a pre-match ran - and so invalidate every
        # registration. Offer it; never take it.
        self._rotation_ds_btn = QPushButton("From DeepSlice")
        self._rotation_ds_btn.setToolTip(
            "Set this section's rotation to the in-plane angle of DeepSlice's "
            "predicted plane. Enabled only for sections that have a prediction."
        )
        self._rotation_ds_btn.clicked.connect(self._rotation_from_deepslice)
        rot_row.addWidget(self._rotation_ds_btn)
        rot_box_layout.addLayout(rot_row)

        self._rotation_warning = QLabel("")
        self._rotation_warning.setWordWrap(True)
        self._rotation_warning.setStyleSheet("color: #e3b617;")  # napari warning
        rot_box_layout.addWidget(self._rotation_warning)

        # "Adjustments" heads this block, set apart from the slide-loading
        # controls above it in this tab. Scope comes first: it says what Flip and
        # Levels below will act on.
        layout.addWidget(section_header("Adjustments", top_margin=4))
        layout.addWidget(scope_box)
        layout.addWidget(rot_box)

        # Flip controls
        flip_box = QGroupBox("Flip")
        flip_row = QHBoxLayout(flip_box)
        flip_h_btn = QPushButton("Flip H")
        flip_h_btn.clicked.connect(self._flip_h)
        flip_v_btn = QPushButton("Flip V")
        flip_v_btn.clicked.connect(self._flip_v)
        flip_row.addWidget(flip_h_btn)
        flip_row.addWidget(flip_v_btn)
        layout.addWidget(flip_box)

        # Per-channel level controls
        levels_box = QGroupBox("Levels (display)")
        levels_layout = QVBoxLayout(levels_box)
        self._low_spins: list[QDoubleSpinBox] = []
        self._high_spins: list[QDoubleSpinBox] = []
        for ch in _CHANNELS:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{ch}:"))
            lo = QDoubleSpinBox()
            lo.setRange(0.0, 1.0)
            lo.setSingleStep(0.01)
            lo.setValue(0.0)
            lo.setFixedWidth(60)
            hi = QDoubleSpinBox()
            hi.setRange(0.0, 1.0)
            hi.setSingleStep(0.01)
            hi.setValue(1.0)
            hi.setFixedWidth(60)
            row.addWidget(lo)
            row.addWidget(QLabel("–"))
            row.addWidget(hi)
            row.addStretch()
            levels_layout.addLayout(row)
            self._low_spins.append(lo)
            self._high_spins.append(hi)
            lo.valueChanged.connect(self._emit_display_changed)
            hi.valueChanged.connect(self._emit_display_changed)

        auto_btn = QPushButton("Auto")
        auto_btn.clicked.connect(self._auto_levels)
        levels_layout.addWidget(auto_btn)
        layout.addWidget(levels_box)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Section selection (drives the "Selected section" scope)
    # ------------------------------------------------------------------

    def _populate_sections(self) -> None:
        """Fill the section dropdown from the active slide's sections."""
        self._section_combo.blockSignals(True)
        self._section_combo.clear()
        slide_idx = self._state.active_slide_idx
        if slide_idx is not None and slide_idx < len(self._state.project.slides):
            slide = self._state.project.slides[slide_idx]
            for sec in sorted(slide.sections, key=lambda s: s.ap_order):
                self._section_combo.addItem(f"Section {sec.index}", sec.index)
        self._section_combo.blockSignals(False)
        # Keep the active section in sync with the (possibly new) selection.
        if self._section_combo.count():
            if self._state.active_section_idx is None:
                self._state.active_section_idx = self._section_combo.currentData()
            else:
                # Reselect the previously-active section if it still exists.
                pos = self._section_combo.findData(self._state.active_section_idx)
                if pos >= 0:
                    self._section_combo.setCurrentIndex(pos)
                else:
                    self._state.active_section_idx = self._section_combo.currentData()

    def select_section(self, section_index: int) -> bool:
        """Point the Section dropdown at ``section_index``; True if it exists.

        Called when a bounding box is selected in the viewer. The box *is* the
        section, so re-picking it in the dropdown is a step that adds nothing and
        can be got wrong - adjust the wrong section and the mistake is silent.

        The list is repopulated first because a box can be newer than it: draw a
        box, select it, and it would otherwise not be an option yet.
        """
        if self._section_combo.findData(section_index) < 0:
            self._populate_sections()
        pos = self._section_combo.findData(section_index)
        if pos < 0:
            return False
        self._section_combo.setCurrentIndex(pos)
        # setCurrentIndex is silent when the index is unchanged, so do not rely
        # on the signal to have updated the active section.
        self._state.active_section_idx = int(section_index)
        self._show_rotation_for_active_section()
        return True

    def _on_scope_section_toggled(self, checked: bool) -> None:
        if checked:
            self._populate_sections()

    def _on_section_combo_changed(self, _idx: int) -> None:
        data = self._section_combo.currentData()
        if data is not None:
            self._state.active_section_idx = int(data)
            self._show_rotation_for_active_section()

    def _active_section(self):
        """The section the 'Selected section' scope acts on, if there is one."""
        slide_idx = self._state.active_slide_idx
        section_idx = self._state.active_section_idx
        if slide_idx is None or section_idx is None:
            return None
        if slide_idx >= len(self._state.project.slides):
            return None
        for section in self._state.project.slides[slide_idx].sections:
            if section.index == section_idx:
                return section
        return None

    def _show_rotation_for_active_section(self) -> None:
        """Load the selected section's stored rotation without writing it back."""
        section = self._active_section()
        self._rotation_spin.blockSignals(True)
        self._rotation_spin.setValue(
            0.0 if section is None else float(getattr(section, "rotation_deg", 0.0))
        )
        self._rotation_spin.blockSignals(False)
        anchoring = None if section is None else getattr(
            section, "deepslice_anchoring", None
        )
        self._rotation_ds_btn.setEnabled(bool(anchoring) and len(anchoring) >= 6)
        self._refresh_rotation_warning()

    def _on_rotation_changed(self, value: float) -> None:
        section = self._active_section()
        if section is not None:
            section.rotation_deg = float(value)
        self._refresh_rotation_warning()

    def _refresh_rotation_warning(self) -> None:
        """Say so when a rotation has invalidated a fit that already exists.

        Rotation is baked into the image the registration is computed against, so
        rotating an already-registered section leaves its stored fit describing
        pixels that have moved. Silence here would be the expensive kind: the
        section still has a registration, it is just quietly wrong.
        """
        section = self._active_section()
        rotated = section is not None and abs(
            float(getattr(section, "rotation_deg", 0.0) or 0.0)
        ) > 1e-6
        registered = section is not None and (
            getattr(section, "registration", None) is not None
        )
        self._rotation_warning.setText(
            f"⚠ Section {section.index} was registered before this rotation. "
            "Its fit was computed on the un-rotated image - re-register it."
            if rotated and registered
            else ""
        )

    def _rotation_from_deepslice(self) -> None:
        section = self._active_section()
        anchoring = None if section is None else getattr(
            section, "deepslice_anchoring", None
        )
        if not anchoring or len(anchoring) < 6:
            return
        from histo_to_ccf.project.images import deepslice_rotation_deg

        self._rotation_spin.setValue(round(deepslice_rotation_deg(anchoring), 2))

    # ------------------------------------------------------------------
    # Flip helpers
    # ------------------------------------------------------------------

    def _flip_h(self) -> None:
        self._apply_flip(axis="h")

    def _flip_v(self) -> None:
        self._apply_flip(axis="v")

    def _apply_flip(self, axis: str) -> None:
        state = self._state
        if self._scope_section.isChecked():
            self._flip_section(axis)
        else:
            self._flip_slide(axis)
        self._emit_display_changed()

    def _flip_slide(self, axis: str) -> None:
        idx = self._state.active_slide_idx
        if idx is None:
            return
        img = self._state.slide_images.get(idx)
        if img is None:
            return
        self._state.slide_images[idx] = np.fliplr(img) if axis == "h" else np.flipud(img)
        slide = self._state.project.slides[idx]
        if axis == "h":
            slide.flip_h = not slide.flip_h
        else:
            slide.flip_v = not slide.flip_v

    def _flip_section(self, axis: str) -> None:
        s_idx = self._state.active_section_idx
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or s_idx is None:
            return
        slide = self._state.project.slides[slide_idx]
        # Find section by index
        section = next((s for s in slide.sections if s.index == s_idx), None)
        if section is None:
            return
        if axis == "h":
            section.flip_h = not section.flip_h
        else:
            section.flip_v = not section.flip_v
        # Also flip the cropped region in the slide image
        img = self._state.slide_images.get(slide_idx)
        if img is not None:
            x0, y0, x1, y1 = section.bbox_px
            crop = img[y0:y1, x0:x1].copy()
            flipped = np.fliplr(crop) if axis == "h" else np.flipud(crop)
            img[y0:y1, x0:x1] = flipped
            self._state.slide_images[slide_idx] = img

    # ------------------------------------------------------------------
    # Level helpers
    # ------------------------------------------------------------------

    def _auto_levels(self) -> None:
        img = self._get_active_image()
        if img is None:
            return
        if img.ndim == 2:
            lo = float(img.min()) / 255.0
            hi = float(img.max()) / 255.0
            for i in range(len(_CHANNELS)):
                self._low_spins[i].setValue(lo)
                self._high_spins[i].setValue(hi)
        else:
            rgb = img[..., :3].astype(float)
            for i in range(min(3, rgb.shape[2])):
                ch = rgb[..., i]
                lo = float(ch.min()) / 255.0
                hi = float(ch.max()) / 255.0
                self._low_spins[i].setValue(lo)
                self._high_spins[i].setValue(hi)
        self._save_levels()

    def _get_active_image(self) -> np.ndarray | None:
        idx = self._state.active_slide_idx
        if idx is None:
            return None
        return self._state.slide_images.get(idx)

    def _save_levels(self) -> None:
        from histo_to_ccf.project.schema import ChannelLevels

        low = [s.value() for s in self._low_spins]
        high = [s.value() for s in self._high_spins]
        levels = ChannelLevels(low=low, high=high)
        if self._scope_section.isChecked():
            s_idx = self._state.active_section_idx
            slide_idx = self._state.active_slide_idx
            if slide_idx is not None and s_idx is not None:
                slide = self._state.project.slides[slide_idx]
                section = next((s for s in slide.sections if s.index == s_idx), None)
                if section is not None:
                    section.levels = levels
        else:
            idx = self._state.active_slide_idx
            if idx is not None and idx < len(self._state.project.slides):
                self._state.project.slides[idx].levels = levels

    def current_levels(self) -> tuple[list[float], list[float]]:
        """Return (low, high) per-channel display cutoffs in [0, 1]."""
        return (
            [s.value() for s in self._low_spins],
            [s.value() for s in self._high_spins],
        )

    def _emit_display_changed(self) -> None:
        self._save_levels()
        if self._on_display_changed is not None:
            self._on_display_changed()
