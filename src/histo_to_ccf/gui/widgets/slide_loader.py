"""Slide loader dock widget - load image, auto-detect sections, edit results."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
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
# Name template for the section-number Points layer (must match app.py).
_NUMBERS_LAYER = "Section numbers {}"
# Temporary Shapes layer used only while the user draws a new rectangle.
_DRAW_LAYER = "_draw_section_temp"
# Editable rectangle layer for resize/move/add/delete of section boxes.
_BOX_LAYER = "Edit boxes {}"


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
        self._box_layer = None  # editable rectangle Shapes layer
        self._syncing_boxes = False  # re-entrancy guard for the data handler
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

        self._equalize_box = QCheckBox("Equalize under-sized boxes")
        self._equalize_box.setChecked(True)
        self._equalize_box.setToolTip(
            "Serial sections on one slide are about the same size, so boxes that "
            "come out noticeably smaller than the others (dim tissue falling below\n"
            "threshold) are grown to the median box size. Uncheck to keep raw boxes."
        )
        params_layout.addWidget(self._equalize_box)

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

        boxes_btn = QPushButton("Edit boxes (resize / move / add / delete)")
        boxes_btn.setToolTip(
            "Turn the detections into draggable rectangles:\n"
            "  • hover an edge or corner handle and drag to resize\n"
            "  • drag inside a box to move it\n"
            "  • press Delete to remove the selected box\n"
            "  • use the rectangle tool to add a missed section\n"
            "Edits are saved to the project as you go."
        )
        boxes_btn.clicked.connect(self._edit_boxes)
        edit_layout.addWidget(boxes_btn)

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
        # Allow selecting several images at once; they are merged into one slide.
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open slide image(s)", "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        for path in paths:
            self._load_path(path)

    def _load_path(self, path) -> None:
        from pathlib import Path

        p = Path(path)
        first = self._state.active_slide_idx is None or not self._state.project.slides
        if first:
            self._load_first_slide(p)
        else:
            self._merge_slide(p)

    def _load_first_slide(self, p) -> None:
        from histo_to_ccf.io.image import load_image

        img = load_image(p)
        slide_idx = self._state.add_slide(p, img)
        self._state.active_slide_idx = slide_idx
        self._state.project.slides[slide_idx].source_paths = [str(p)]
        self._path_label.setText(p.name)
        self._after_image_changed(img, slide_idx)

    def _merge_slide(self, p) -> None:
        """Merge a newly opened image into the existing single slide.

        Multiple slides share one coordinate space by living in one combined
        image, so probes can span sections from different source images. Merging
        changes pixel coordinates, so any existing detected sections are cleared
        and the user is told to re-detect.
        """
        from histo_to_ccf.io.image import load_image, merge_images

        slide_idx = self._state.active_slide_idx
        slide = self._state.project.slides[slide_idx]
        sources = list(slide.source_paths) if slide.source_paths else [slide.image_path]
        sources.append(str(p))
        # Deterministic, reproducible layout (also matches reload): alphabetical.
        sources = sorted(sources)

        from pathlib import Path

        try:
            images = [load_image(Path(s)) for s in sources]
            combined = merge_images(images)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Merge failed: {exc}")
            return

        had_sections = bool(slide.sections)
        slide.source_paths = sources
        slide.image_path = sources[0]
        slide.sections.clear()
        # The combined image is rebuilt from raw sources, so any prior flips /
        # levels no longer apply - clear them to stay consistent with reload.
        slide.flip_h = False
        slide.flip_v = False
        slide.levels = None
        self._state.slide_images[slide_idx] = combined
        self._path_label.setText(f"{len(sources)} images merged")
        note = (
            f"Merged {len(sources)} images into one combined slide "
            f"({combined.shape[1]}×{combined.shape[0]} px). "
        )
        if had_sections:
            note += "Existing sections were cleared - click 'Detect sections' again."
        self._info_merge(note)
        self._after_image_changed(combined, slide_idx)

    def _info_merge(self, message: str) -> None:
        """Tell the user that images were merged (best-effort dialog + status)."""
        self._status.setText(message)
        try:
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.information(self, "Slides merged", message)
        except Exception:
            pass

    def _after_image_changed(self, img, slide_idx: int) -> None:
        """Shared post-load work: estimate min area and refresh the viewer."""
        self._status.setText(
            f"Loaded {img.shape[1]}×{img.shape[0]} px - estimating min area…"
        )
        # Auto-estimate in a worker so the UI stays responsive for large images.
        worker = self._estimate_worker(img)
        worker.returned.connect(lambda v: self._min_area.setValue(v))
        worker.returned.connect(lambda v: self._status.setText(
            f"{img.shape[1]}×{img.shape[0]} px  |  min area estimated: {v:,} px²"
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
            equalize_boxes=self._equalize_box.isChecked(),
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
        self._status.setText("Ready - click near a box in the viewer to discard it.")

    # ------------------------------------------------------------------
    # Manual draw
    # ------------------------------------------------------------------

    def _start_draw(self) -> None:
        if self._viewer is None:
            self._status.setText("Viewer not available - cannot draw.")
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
    # Editable boxes (resize / move / add / delete via napari Shapes)
    # ------------------------------------------------------------------

    def _edit_boxes(self) -> None:
        if self._viewer is None:
            self._status.setText("Viewer not available - cannot edit boxes.")
            return
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            self._status.setText("Load a slide first.")
            return
        slide = self._state.project.slides[slide_idx]
        if not slide.sections:
            self._status.setText("Detect sections first, then edit the boxes.")
            return

        # Build one rectangle per section; carry the section index as a feature
        # so identity survives moves, additions and deletions.
        rects, idxs = [], []
        for s in sorted(slide.sections, key=lambda s: s.ap_order):
            x0, y0, x1, y1 = s.bbox_px
            rects.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float))
            idxs.append(int(s.index))

        name = _BOX_LAYER.format(slide_idx)
        if name in self._viewer.layers:
            self._viewer.layers.remove(name)
        layer = self._viewer.add_shapes(
            rects,
            name=name,
            shape_type="rectangle",
            edge_color="yellow",
            face_color="transparent",
            edge_width=self._edge_width(),
        )
        layer.features = {"idx": idxs}
        try:
            layer.text = {"string": "{idx}", "size": 14, "color": "yellow", "anchor": "center"}
        except Exception:
            pass
        try:
            layer.feature_defaults = {"idx": -1}  # new shapes flagged until synced
        except Exception:
            pass

        self._box_layer = layer
        layer.events.data.connect(self._sync_boxes_from_shapes)

        # Hide the static outline + numbers so the editable boxes are the only
        # representation while editing (avoids a confusing double display).
        for nm in (_SECTION_LAYER.format(slide_idx), _NUMBERS_LAYER.format(slide_idx)):
            if nm in self._viewer.layers:
                self._viewer.layers[nm].visible = False

        self._viewer.layers.selection.active = layer
        layer.mode = "select"
        self._status.setText(
            "Edit boxes: drag handles to resize, drag inside to move, "
            "Delete to remove, rectangle tool to add."
        )

    def _sync_boxes_from_shapes(self, event=None) -> None:
        """Write edited rectangles back to the project sections."""
        if self._syncing_boxes or self._box_layer is None:
            return
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            return
        self._syncing_boxes = True
        try:
            from histo_to_ccf.project.schema import Section

            slide = self._state.project.slides[slide_idx]
            data = list(self._box_layer.data)
            feats = self._box_layer.features
            idx_col = list(feats["idx"]) if "idx" in getattr(feats, "columns", []) else []

            by_index = {s.index: s for s in slide.sections}
            next_idx = max((s.index for s in slide.sections), default=-1) + 1
            stabilized: list[int] = []
            seen: set[int] = set()

            for i, poly in enumerate(data):
                poly = np.asarray(poly)
                y0, y1 = int(poly[:, 0].min()), int(poly[:, 0].max())
                x0, x1 = int(poly[:, 1].min()), int(poly[:, 1].max())
                bbox = (x0, y0, x1, y1)
                raw = idx_col[i] if i < len(idx_col) else -1
                sid = int(raw) if raw is not None and raw == raw and raw >= 0 else -1
                if sid in by_index:
                    by_index[sid].bbox_px = bbox
                else:
                    sid = next_idx
                    next_idx += 1
                    slide.sections.append(
                        Section(index=sid, slide_idx=slide_idx, bbox_px=bbox,
                                ap_order=len(slide.sections))
                    )
                stabilized.append(sid)
                seen.add(sid)

            # Drop sections whose rectangle was deleted.
            slide.sections[:] = [s for s in slide.sections if s.index in seen]

            # Persist stabilized indices so newly added shapes keep identity.
            if idx_col != stabilized:
                self._box_layer.features = {"idx": stabilized}
                try:
                    self._box_layer.refresh_text()
                except Exception:
                    pass
            self._status.setText(f"{len(seen)} section box(es).")
        finally:
            self._syncing_boxes = False

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
