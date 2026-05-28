"""Slide loader dock widget."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.workflow import WorkflowState


class SlideLoaderWidget(QWidget):
    """Load a slide image and run section detection."""

    def __init__(
        self,
        state: WorkflowState,
        on_slide_loaded: Callable[[int, np.ndarray], None] | None = None,
        on_sections_detected: Callable | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._on_slide_loaded = on_slide_loaded
        self._on_sections_detected = on_sections_detected
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # File chooser row
        row = QHBoxLayout()
        self._path_label = QLabel("No file loaded")
        self._path_label.setWordWrap(True)
        load_btn = QPushButton("Open slide…")
        load_btn.clicked.connect(self._open_file)
        row.addWidget(load_btn)
        layout.addLayout(row)
        layout.addWidget(self._path_label)

        # Section detection params
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("Min area px:"))
        self._min_area = QSpinBox()
        self._min_area.setRange(100, 1_000_000)
        self._min_area.setValue(5000)
        params_row.addWidget(self._min_area)
        params_row.addWidget(QLabel("Closing r:"))
        self._closing_r = QSpinBox()
        self._closing_r.setRange(0, 100)
        self._closing_r.setValue(0)
        params_row.addWidget(self._closing_r)
        layout.addLayout(params_row)

        detect_btn = QPushButton("Detect sections")
        detect_btn.clicked.connect(self._detect_sections)
        layout.addWidget(detect_btn)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._status)
        layout.addStretch()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open slide image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if not path:
            return
        self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        from histo_to_ccf.io.image import load_image

        img = load_image(path)
        slide_idx = self._state.add_slide(path, img)
        self._state.active_slide_idx = slide_idx
        self._path_label.setText(path.name)
        self._status.setText(f"Loaded {img.shape[1]}×{img.shape[0]} px")
        if self._on_slide_loaded is not None:
            self._on_slide_loaded(slide_idx, img)

    def _detect_sections(self) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or slide_idx not in self._state.slide_images:
            self._status.setText("Load a slide first.")
            return
        img = self._state.slide_images[slide_idx]
        self._status.setText("Detecting…")

        worker = self._run_detection(img)
        worker.returned.connect(self._on_detected)
        worker.errored.connect(lambda e: self._status.setText(f"Error: {e}"))
        worker.start()

    def _run_detection(self, img: np.ndarray):  # type: ignore[return]
        from histo_to_ccf.gui.workers import detect_sections_worker

        return detect_sections_worker(
            img,
            min_area_px=self._min_area.value(),
            closing_radius_px=self._closing_r.value(),
        )

    def _on_detected(self, sections) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            return
        from histo_to_ccf.project.schema import Section

        slide = self._state.project.slides[slide_idx]
        slide.sections.clear()
        for s in sections:
            bbox = (
                int(s.section.bbox_px[0]),
                int(s.section.bbox_px[1]),
                int(s.section.bbox_px[2]),
                int(s.section.bbox_px[3]),
            )
            slide.sections.append(
                Section(
                    index=s.ap_order,
                    slide_idx=slide_idx,
                    bbox_px=bbox,
                    ap_order=s.ap_order,
                )
            )
        self._status.setText(f"Detected {len(sections)} section(s)")
        if self._on_sections_detected is not None:
            self._on_sections_detected(sections)
