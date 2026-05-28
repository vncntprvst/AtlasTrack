"""Image-prep tools: flip H/V and per-channel saturation/level adjustments."""
from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

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
        scope_row = QHBoxLayout(scope_box)
        self._scope_whole = QRadioButton("Whole slide")
        self._scope_whole.setChecked(True)
        self._scope_section = QRadioButton("Selected section")
        scope_grp = QButtonGroup(self)
        scope_grp.addButton(self._scope_whole)
        scope_grp.addButton(self._scope_section)
        scope_row.addWidget(self._scope_whole)
        scope_row.addWidget(self._scope_section)
        layout.addWidget(scope_box)

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
