"""Slide loader dock widget — load image, auto-detect sections, edit results."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    import napari

# Name template for the section outline layer (must match app.py).
_SECTION_LAYER = "Sections {}"
# Temporary Shapes layer used only while the user draws a new rectangle.
_DRAW_LAYER = "_draw_section_temp"


class SlideLoaderWidget(QWidget):
    """Load a slide image, run section detection, and edit the results."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer | None" = None,
        on_slide_loaded: Callable[[int, np.ndarray], None] | None = None,
        on_sections_detected: Callable | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._on_slide_loaded = on_slide_loaded
        self._on_sections_detected = on_sections_detected
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Load -------------------------------------------------------
        load_box = QGroupBox("Slide image")
        load_layout = QVBoxLayout(load_box)
        load_btn = QPushButton("Open slide…")
        load_btn.setToolTip("Open a TIFF, PNG, or JPEG composite slide image.")
        load_btn.clicked.connect(self._open_file)
        self._path_label = QLabel("No file loaded")
        self._path_label.setWordWrap(True)
        load_layout.addWidget(load_btn)
        load_layout.addWidget(self._path_label)
        layout.addWidget(load_box)

        # --- Detection params -------------------------------------------
        params_box = QGroupBox("Section detection")
        params_layout = QVBoxLayout(params_box)

        area_row = QHBoxLayout()
        area_lbl = QLabel("Min area (px²):")
        area_lbl.setToolTip(
            "Minimum connected-component area to count as a brain section.\n"
            "Increase if slide text or debris is being picked up.\n"
            "Auto-estimated from the image after loading."
        )
        self._min_area = QSpinBox()
        self._min_area.setRange(100, 50_000_000)
        self._min_area.setValue(5_000)
        self._min_area.setSingleStep(1_000)
        self._min_area.setToolTip(area_lbl.toolTip())
        area_row.addWidget(area_lbl)
        area_row.addWidget(self._min_area)
        params_layout.addLayout(area_row)

        closing_row = QHBoxLayout()
        closing_lbl = QLabel("Closing radius (px):")
        closing_lbl.setToolTip(
            "Morphological closing radius applied before component detection.\n"
            "A non-zero value bridges small gaps within a section (e.g. cerebellum\n"
            "lobules, white-matter splits). Start at 0; try 10–40 if sections are\n"
            "fragmented. Larger values risk merging neighbouring sections."
        )
        self._closing_r = QSpinBox()
        self._closing_r.setRange(0, 200)
        self._closing_r.setValue(10)
        self._closing_r.setToolTip(closing_lbl.toolTip())
        closing_row.addWidget(closing_lbl)
        closing_row.addWidget(self._closing_r)
        params_layout.addLayout(closing_row)

        detect_btn = QPushButton("Detect sections")
        detect_btn.setToolTip("Run Otsu + connected-component detection with the parameters above.")
        detect_btn.clicked.connect(self._detect_sections)
        params_layout.addWidget(detect_btn)
        layout.addWidget(params_box)

        # --- Status -----------------------------------------------------
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # --- Edit detected sections ------------------------------------
        edit_box = QGroupBox("Edit sections")
        edit_layout = QVBoxLayout(edit_box)

        discard_btn = QPushButton("Click to discard a box…")
        discard_btn.setToolTip(
            "Click this button, then click anywhere near a coloured box in the\n"
            "viewer to remove that detection.  One click = one discard.\n"
            "Numbers shown in yellow are the box indices."
        )
        discard_btn.clicked.connect(self._arm_discard)
        edit_layout.addWidget(discard_btn)

        draw_btn = QPushButton("Draw new section…")
        draw_btn.setToolTip(
            "Activate rectangle-draw mode in the viewer.\n"
            "Draw a box around a missed or under-detected section,\n"
            "then click 'Add drawn section' to register it."
        )
        draw_btn.clicked.connect(self._start_draw)

        self._add_drawn_btn = QPushButton("Add drawn section")
        self._add_drawn_btn.setToolTip("Commit the rectangle you just drew as a new section.")
        self._add_drawn_btn.setEnabled(False)
        self._add_drawn_btn.clicked.connect(self._commit_drawn)

        edit_layout.addWidget(draw_btn)
        edit_layout.addWidget(self._add_drawn_btn)
        layout.addWidget(edit_box)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open slide image", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if path:
            self._load_path(path)

    def _load_path(self, path) -> None:
        from pathlib import Path
        from histo_to_ccf.io.image import load_image
        from histo_to_ccf.sectioning.split import estimate_min_area

        p = Path(path)
        img = load_image(p)
        slide_idx = self._state.add_slide(p, img)
        self._state.active_slide_idx = slide_idx
        self._path_label.setText(p.name)
        self._status.setText(f"Loaded {img.shape[1]}×{img.shape[0]} px — estimating min area…")

        # Auto-estimate in a worker so the UI stays responsive for large images.
        worker = self._estimate_worker(img)
        worker.returned.connect(lambda v: self._min_area.setValue(v))
        worker.returned.connect(lambda v: self._status.setText(
            f"Loaded {img.shape[1]}×{img.shape[0]} px  |  min area estimated: {v:,} px²"
        ))
        worker.start()

        if self._on_slide_loaded is not None:
            self._on_slide_loaded(slide_idx, img)

    def _estimate_worker(self, img: np.ndarray):
        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            from histo_to_ccf.sectioning.split import estimate_min_area
            return estimate_min_area(img)

        return _run()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_sections(self) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None or slide_idx not in self._state.slide_images:
            self._status.setText("Load a slide first.")
            return
        img = self._state.slide_images[slide_idx]
        self._status.setText("Detecting…")

        from histo_to_ccf.gui.workers import detect_sections_worker

        worker = detect_sections_worker(
            img,
            min_area_px=self._min_area.value(),
            closing_radius_px=self._closing_r.value(),
        )
        worker.returned.connect(self._on_detected)
        worker.errored.connect(lambda e: self._status.setText(f"Error: {e}"))
        worker.start()

    def _on_detected(self, sections) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            return
        from histo_to_ccf.project.schema import Section

        slide = self._state.project.slides[slide_idx]
        slide.sections.clear()
        for s in sections:
            bbox = tuple(int(v) for v in s.section.bbox_px)
            slide.sections.append(Section(
                index=s.ap_order,
                slide_idx=slide_idx,
                bbox_px=bbox,
                ap_order=s.ap_order,
            ))
        self._status.setText(f"Detected {len(sections)} section(s)  ·  use 'Discard' or 'Draw' to edit")
        if self._on_sections_detected is not None:
            self._on_sections_detected(sections)


    # ------------------------------------------------------------------
    # Discard (viewer-level one-shot click)
    # ------------------------------------------------------------------

    def _arm_discard(self) -> None:
        if self._viewer is None:
            self._status.setText("Viewer not available.")
            return
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            self._status.setText("Load a slide first.")
            return
        from histo_to_ccf.gui.app import install_discard_handler
        install_discard_handler(self._state, slide_idx, self._viewer)
        self._status.setText("Ready — click near a box in the viewer to discard it.")

    # ------------------------------------------------------------------
    # Manual draw
    # ------------------------------------------------------------------

    def _start_draw(self) -> None:
        if self._viewer is None:
            self._status.setText("Viewer not available — cannot draw.")
            return
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            self._status.setText("Load a slide first.")
            return

        # Remove any leftover temp layer.
        if _DRAW_LAYER in self._viewer.layers:
            self._viewer.layers.remove(_DRAW_LAYER)

        draw_layer = self._viewer.add_shapes(
            name=_DRAW_LAYER,
            shape_type="rectangle",
            edge_color="lime",
            face_color="transparent",
            edge_width=self._edge_width(),
        )
        draw_layer.mode = "add_rectangle"
        self._viewer.layers.selection.active = draw_layer
        self._add_drawn_btn.setEnabled(True)
        self._status.setText("Draw a rectangle in the viewer, then click 'Add drawn section'.")

    def _commit_drawn(self) -> None:
        if self._viewer is None or _DRAW_LAYER not in self._viewer.layers:
            self._status.setText("No drawn rectangle found.")
            return
        draw_layer = self._viewer.layers[_DRAW_LAYER]
        if len(draw_layer.data) == 0:
            self._status.setText("No rectangle drawn yet.")
            return

        shape = np.asarray(draw_layer.data[-1])   # (4, 2) [row, col]
        y0 = int(shape[:, 0].min())
        y1 = int(shape[:, 0].max())
        x0 = int(shape[:, 1].min())
        x1 = int(shape[:, 1].max())

        slide_idx = self._state.active_slide_idx
        slide = self._state.project.slides[slide_idx]
        next_idx = max((s.index for s in slide.sections), default=-1) + 1
        from histo_to_ccf.project.schema import Section

        slide.sections.append(Section(
            index=next_idx,
            slide_idx=slide_idx,
            bbox_px=(x0, y0, x1, y1),
            ap_order=len(slide.sections),
        ))
        self._viewer.layers.remove(_DRAW_LAYER)
        self._add_drawn_btn.setEnabled(False)
        self._status.setText(f"Added section {next_idx} ({x1 - x0}×{y1 - y0} px)")
        self._redraw_outlines()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _edge_width(self) -> int:
        """Return an outline thickness (px) that looks good at any zoom."""
        slide_idx = self._state.active_slide_idx
        if slide_idx is not None and slide_idx in self._state.slide_images:
            img = self._state.slide_images[slide_idx]
            return max(4, min(img.shape[0], img.shape[1]) // 150)
        return 8

    def _redraw_outlines(self) -> None:
        """Redraw the section-outline Labels layer and section numbers after an edit."""
        if self._viewer is None:
            return
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            return
        slide = self._state.project.slides[slide_idx]
        img = self._state.slide_images.get(slide_idx)
        if img is None:
            return

        from histo_to_ccf.gui.app import _update_section_numbers
        from histo_to_ccf.gui.section_display import sections_to_outline_labels
        labels = sections_to_outline_labels(img.shape[:2], slide.sections)
        layer_name = _SECTION_LAYER.format(slide_idx)
        if layer_name in self._viewer.layers:
            self._viewer.layers[layer_name].data = labels
        _update_section_numbers(self._viewer, self._state, slide_idx)
