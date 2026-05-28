"""Register button + progress bar + results + 3D/export buttons."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
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


def _error_dialog(parent: QWidget, title: str, message: str) -> None:
    """Show a modal error dialog."""
    QMessageBox.critical(parent, title, str(message)[:2000])


class RegisterPanelWidget(QWidget):
    """Register button, progress bar, residuals table, and export actions."""

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

        self._reg_btn = QPushButton("Register all sections")
        self._reg_btn.setFixedHeight(34)
        self._reg_btn.clicked.connect(self._run_registration)
        layout.addWidget(self._reg_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addWidget(QLabel("Per-section residuals:"))
        self._residuals_table = QTableWidget(0, 3)
        self._residuals_table.setHorizontalHeaderLabels(["Section", "AP µm", "Residual"])
        self._residuals_table.setMaximumHeight(160)
        layout.addWidget(self._residuals_table)

        viz_box = QGroupBox("3D Visualization")
        viz_layout = QVBoxLayout(viz_box)
        plotly_btn = QPushButton("Export Plotly HTML…")
        plotly_btn.clicked.connect(self._export_plotly)
        napari_btn = QPushButton("View in napari 3D")
        napari_btn.clicked.connect(self._view_napari3d)
        viz_layout.addWidget(plotly_btn)
        viz_layout.addWidget(napari_btn)
        layout.addWidget(viz_box)

        export_box = QGroupBox("Export")
        export_layout = QVBoxLayout(export_box)
        herbs_btn = QPushButton("Export HERBS pkl…")
        herbs_btn.clicked.connect(self._export_herbs)
        ch_btn = QPushButton("Export per-channel CSV…")
        ch_btn.clicked.connect(self._export_channel_csv)
        export_layout.addWidget(herbs_btn)
        export_layout.addWidget(ch_btn)
        layout.addWidget(export_box)

        layout.addStretch()

    def apply_settings(self, settings) -> None:
        """Populate controls from persisted AppSettings."""
        self._grid_spin.setValue(settings.bspline_grid)
        self._iter_spin.setValue(settings.max_iterations)

    def collect_settings(self, settings) -> None:
        """Write current control values back into settings."""
        settings.bspline_grid = self._grid_spin.value()
        settings.max_iterations = self._iter_spin.value()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _run_registration(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            _error_dialog(self, "Atlas not loaded", "Load an atlas in the Atlas tab first.")
            return
        if not self._state.project.slides:
            _error_dialog(self, "No slides", "Load at least one slide before registering.")
            return

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
                section_images[section.index] = crop(img, (x0, y0, x1, y1)).astype(np.float32)

        if not section_images:
            _error_dialog(
                self, "No AP planes assigned",
                "Assign AP positions to sections in the Atlas tab before registering."
            )
            return

        project_path = self._state.project_path
        transforms_dir = (
            project_path.parent / "transforms" if project_path is not None
            else Path(__import__("tempfile").mkdtemp()) / "transforms"
        )

        self._reg_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText(f"Starting registration of {len(section_images)} section(s)…")

        from histo_to_ccf.gui.workers import register_worker_progressive

        worker = register_worker_progressive(
            self._state.project,
            atlas,
            section_images,
            transforms_dir,
            bspline_grid=(self._grid_spin.value(),) * 2,
            max_iterations=self._iter_spin.value(),
        )
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._on_registration_done)
        worker.errored.connect(self._on_registration_error)
        worker.start()

    def _on_progress(self, info: dict) -> None:
        current = info.get("current", 0)
        total = info.get("total", 1) or 1
        msg = info.get("msg", "")
        pct = int(100 * current / total)
        self._progress.setValue(pct)
        self._progress.setFormat(f"{current}/{total} — {pct}%")
        self._status.setText(msg)

    def _on_registration_done(self, project) -> None:
        self._state.project = project
        n = sum(
            1 for slide in project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        self._progress.setValue(100)
        self._status.setText(f"Done — {n} section(s) registered")
        self._reg_btn.setEnabled(True)
        self._refresh_residuals()

    def _on_registration_error(self, exc: Exception) -> None:
        self._reg_btn.setEnabled(True)
        self._progress.setVisible(False)
        _error_dialog(self, "Registration failed", str(exc))
        self._status.setText(f"Error: {exc}")

    def _refresh_residuals(self) -> None:
        rows = []
        for slide in self._state.project.slides:
            for sec in slide.sections:
                if sec.registration is not None:
                    ap = sec.plane.ap_um if sec.plane else float("nan")
                    rows.append((sec.index, ap, sec.registration.residual))
        self._residuals_table.setRowCount(len(rows))
        for i, (idx, ap, res) in enumerate(rows):
            self._residuals_table.setItem(i, 0, QTableWidgetItem(str(idx)))
            self._residuals_table.setItem(i, 1, QTableWidgetItem(f"{ap:.0f}" if ap == ap else "—"))
            self._residuals_table.setItem(i, 2, QTableWidgetItem(f"{res:.4f}" if res is not None else "—"))

    # ------------------------------------------------------------------
    # 3D viz
    # ------------------------------------------------------------------

    def _export_plotly(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plotly HTML", "", "HTML files (*.html);;All files (*)"
        )
        if not path:
            return
        try:
            from histo_to_ccf.viz.plotly3d import build_figure, save_html
            fig = build_figure(self._state.project, self._state.atlas)
            out = save_html(fig, path, open_browser=True)
            self._status.setText(f"Saved → {out.name}")
        except Exception as exc:
            _error_dialog(self, "Export failed", str(exc))

    def _view_napari3d(self) -> None:
        try:
            from histo_to_ccf.viz.napari3d import add_probe_layers, switch_to_3d
            add_probe_layers(self._viewer, self._state.project)
            switch_to_3d(self._viewer)
            self._status.setText("Probe layers added; viewer in 3D mode")
        except Exception as exc:
            _error_dialog(self, "3D view failed", str(exc))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_herbs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HERBS pkl", "", "Pickle files (*.pkl);;All files (*)"
        )
        if not path:
            return
        try:
            import numpy as np
            from histo_to_ccf.io.herbs_writer import write_herbs_pkl

            all_ccf: list[np.ndarray] = []
            for probe in self._state.project.probes:
                for shank in probe.shanks:
                    if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
                        continue
                    all_ccf.append(np.linspace(
                        np.array(shank.entry_ccf_um, dtype=float),
                        np.array(shank.tip_ccf_um, dtype=float),
                        128,
                    ))
            if not all_ccf:
                _error_dialog(self, "Nothing to export", "No registered shank coordinates found.")
                return
            write_herbs_pkl(path, all_ccf)
            self._status.setText(f"HERBS pkl → {Path(path).name}")
        except Exception as exc:
            _error_dialog(self, "HERBS export failed", str(exc))

    def _export_channel_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save per-channel CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            from histo_to_ccf.probes.channels import export_channel_csv
            n = export_channel_csv(self._state.project, path)
            if n == 0:
                _error_dialog(self, "Nothing to export", "No registered shank coordinates found.")
            else:
                self._status.setText(f"Per-channel CSV ({n} rows) → {Path(path).name}")
        except Exception as exc:
            _error_dialog(self, "CSV export failed", str(exc))
