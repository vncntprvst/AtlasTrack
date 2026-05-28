"""Register button + progress + results + 3D/export buttons."""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    import napari


class RegisterPanelWidget(QWidget):
    """Register button, progress display, residuals table, and export actions."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Registration parameters
        params_box = QGroupBox("Registration parameters")
        params_layout = QVBoxLayout(params_box)
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("B-spline grid (N×N):"))
        self._grid_spin = QSpinBox()
        self._grid_spin.setRange(4, 24)
        self._grid_spin.setValue(8)
        grid_row.addWidget(self._grid_spin)
        params_layout.addLayout(grid_row)
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max iterations:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(10, 500)
        self._iter_spin.setValue(100)
        iter_row.addWidget(self._iter_spin)
        params_layout.addLayout(iter_row)
        layout.addWidget(params_box)

        # Register button
        reg_btn = QPushButton("Register all sections")
        reg_btn.setFixedHeight(34)
        reg_btn.clicked.connect(self._run_registration)
        layout.addWidget(reg_btn)

        self._status = QLabel("Ready")
        layout.addWidget(self._status)

        # Residuals table
        layout.addWidget(QLabel("Per-section residuals:"))
        self._residuals_table = QTableWidget(0, 3)
        self._residuals_table.setHorizontalHeaderLabels(["Section", "AP µm", "Residual"])
        self._residuals_table.setMaximumHeight(160)
        layout.addWidget(self._residuals_table)

        # 3D viz
        viz_box = QGroupBox("3D Visualization")
        viz_layout = QVBoxLayout(viz_box)
        plotly_btn = QPushButton("Export Plotly HTML…")
        plotly_btn.clicked.connect(self._export_plotly)
        napari_btn = QPushButton("View in napari 3D")
        napari_btn.clicked.connect(self._view_napari3d)
        viz_layout.addWidget(plotly_btn)
        viz_layout.addWidget(napari_btn)
        layout.addWidget(viz_box)

        # HERBS export
        herbs_box = QGroupBox("HERBS Export")
        herbs_layout = QVBoxLayout(herbs_box)
        herbs_btn = QPushButton("Export HERBS pkl…")
        herbs_btn.clicked.connect(self._export_herbs)
        herbs_layout.addWidget(herbs_btn)
        ch_btn = QPushButton("Export per-channel CSV…")
        ch_btn.clicked.connect(self._export_channel_csv)
        herbs_layout.addWidget(ch_btn)
        layout.addWidget(herbs_box)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _run_registration(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            self._status.setText("Load an atlas first (Atlas tab).")
            return
        if not self._state.project.slides:
            self._status.setText("No slides loaded.")
            return

        # Build section_images: for each section with a plane, provide its crop.
        import numpy as np
        from histo_to_ccf.io.image import crop

        section_images: dict[int, np.ndarray] = {}
        for slide_idx, slide in enumerate(self._state.project.slides):
            img = self._state.slide_images.get(slide_idx)
            if img is None:
                continue
            for section in slide.sections:
                if section.plane is None:
                    continue
                x0, y0, x1, y1 = section.bbox_px
                crop_img = crop(img, (x0, y0, x1, y1))
                section_images[section.index] = crop_img.astype(np.float32)

        if not section_images:
            self._status.setText("No sections have an assigned AP plane. Use the Atlas tab first.")
            return

        # Transforms go next to the project, or in a temp dir.
        project_path = self._state.project_path
        if project_path is not None:
            transforms_dir = project_path.parent / "transforms"
        else:
            import tempfile
            transforms_dir = Path(tempfile.mkdtemp()) / "transforms"

        g = self._grid_spin.value()
        grid = (g, g)
        self._status.setText(f"Registering {len(section_images)} section(s)…")

        from histo_to_ccf.gui.workers import register_worker

        worker = register_worker(
            self._state.project,
            atlas,
            section_images,
            transforms_dir,
            bspline_grid=grid,
            max_iterations=self._iter_spin.value(),
        )
        worker.returned.connect(self._on_registration_done)
        worker.errored.connect(lambda e: self._status.setText(f"Error: {e}"))
        worker.start()

    def _on_registration_done(self, project) -> None:
        self._state.project = project
        n = sum(
            1 for slide in project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        self._status.setText(f"Done — {n} section(s) registered")
        self._refresh_residuals()

    def _refresh_residuals(self) -> None:
        rows = []
        for slide in self._state.project.slides:
            for sec in slide.sections:
                if sec.registration is not None:
                    ap = sec.plane.ap_um if sec.plane else float("nan")
                    res = sec.registration.residual
                    rows.append((sec.index, ap, res))
        self._residuals_table.setRowCount(len(rows))
        for i, (idx, ap, res) in enumerate(rows):
            ap_str = f"{ap:.0f}" if ap == ap else "—"
            res_str = f"{res:.4f}" if res is not None else "—"
            self._residuals_table.setItem(i, 0, QTableWidgetItem(str(idx)))
            self._residuals_table.setItem(i, 1, QTableWidgetItem(ap_str))
            self._residuals_table.setItem(i, 2, QTableWidgetItem(res_str))

    # ------------------------------------------------------------------
    # 3D viz
    # ------------------------------------------------------------------

    def _export_plotly(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plotly HTML", "", "HTML files (*.html);;All files (*)"
        )
        if not path:
            return
        from histo_to_ccf.viz.plotly3d import build_figure, save_html

        fig = build_figure(self._state.project, self._state.atlas)
        out = save_html(fig, path, open_browser=True)
        self._status.setText(f"Saved → {out.name}")

    def _view_napari3d(self) -> None:
        from histo_to_ccf.viz.napari3d import add_probe_layers, switch_to_3d

        add_probe_layers(self._viewer, self._state.project)
        switch_to_3d(self._viewer)
        self._status.setText("Probe layers added; viewer switched to 3D")

    # ------------------------------------------------------------------
    # HERBS export
    # ------------------------------------------------------------------

    def _export_herbs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HERBS pkl", "", "Pickle files (*.pkl);;All files (*)"
        )
        if not path:
            return
        import numpy as np
        from histo_to_ccf.io.herbs_writer import write_herbs_pkl

        all_ccf: list[np.ndarray] = []
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
                    continue
                ccf = np.linspace(
                    np.array(shank.entry_ccf_um, dtype=float),
                    np.array(shank.tip_ccf_um, dtype=float),
                    128,
                )
                all_ccf.append(ccf)

        if not all_ccf:
            self._status.setText("No registered shank coords to export.")
            return
        write_herbs_pkl(path, all_ccf)
        self._status.setText(f"HERBS pkl → {Path(path).name}")

    def _export_channel_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save per-channel CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        from histo_to_ccf.probes.channels import export_channel_csv

        n = export_channel_csv(self._state.project, path)
        if n == 0:
            self._status.setText("No registered shank coords to export.")
        else:
            self._status.setText(f"Per-channel CSV ({n} rows) → {Path(path).name}")
