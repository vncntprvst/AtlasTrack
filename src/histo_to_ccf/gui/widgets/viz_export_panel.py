"""Permanent 3D-visualization + export panel (right dock).

Split out of the Register tab so 3D view and exports are always available, not
buried in one workflow step. Operates on the shared WorkflowState; lazily loads
the project's atlas when a 3D view / overlay needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    import napari


def _error_dialog(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, str(message)[:2000])


def _viewer_alive(viewer) -> bool:
    try:
        return bool(viewer.window._qt_window.isVisible())
    except Exception:
        return False


class VizExportPanelWidget(QWidget):
    """3D visualization + export actions, always docked (not a workflow tab)."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._viewer3d = None  # held so the separate 3D window isn't GC'd
        self._settings = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        viz_box = QGroupBox("3D Visualization")
        viz_layout = QVBoxLayout(viz_box)
        reg_row = QHBoxLayout()
        reg_row.addWidget(QLabel("Extra regions:"))
        self._extra_regions = QLineEdit()
        self._extra_regions.setPlaceholderText("acronyms, comma-sep (e.g. VII, XII)")
        self._extra_regions.setToolTip(
            "Atlas structure acronyms to also display in 3D, on top of the brain "
            "shell and the regions at each shank tip. Example: VII (facial nucleus), "
            "XII (hypoglossal nucleus)."
        )
        reg_row.addWidget(self._extra_regions)
        viz_layout.addLayout(reg_row)

        plotly_btn = QPushButton("Export Plotly HTML")
        plotly_btn.clicked.connect(self._export_plotly)
        napari_btn = QPushButton("View in napari 3D")
        napari_btn.clicked.connect(self._view_napari3d)
        viz_layout.addWidget(plotly_btn)
        viz_layout.addWidget(napari_btn)
        layout.addWidget(viz_box)

        export_box = QGroupBox("Export")
        export_layout = QVBoxLayout(export_box)
        herbs_btn = QPushButton("Export pkl file")
        herbs_btn.clicked.connect(self._export_herbs)
        ch_btn = QPushButton("Export per-channel CSV")
        ch_btn.clicked.connect(self._export_channel_csv)
        export_layout.addWidget(herbs_btn)
        export_layout.addWidget(ch_btn)
        layout.addWidget(export_box)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    def apply_settings(self, settings) -> None:
        """Store AppSettings (for the atlas storage folder used by lazy load)."""
        self._settings = settings

    # ------------------------------------------------------------------

    def _extra_region_list(self) -> list[str]:
        return [a.strip() for a in self._extra_regions.text().split(",") if a.strip()]

    def _ensure_atlas(self, on_ready) -> None:
        """Ensure ``state.atlas`` is loaded, then call ``on_ready()`` (lazy)."""
        if self._state.atlas is not None:
            on_ready()
            return
        atlas_id = self._state.project.atlas.name
        if not atlas_id:
            on_ready()
            return
        atlas_dir = None
        if self._settings is not None:
            atlas_dir = getattr(self._settings, "atlas_dir", "") or None
        self._status.setText(f"Loading atlas {atlas_id} for 3D view")
        from histo_to_ccf.gui.workers import load_atlas_worker

        worker = load_atlas_worker(atlas_id, brainglobe_dir=atlas_dir)

        def _loaded(atlas) -> None:
            self._state.atlas = atlas
            on_ready()

        worker.returned.connect(_loaded)
        worker.errored.connect(
            lambda exc: (self._status.setText(f"Atlas load failed: {exc}"), on_ready())
        )
        worker.start()

    def _export_plotly(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plotly HTML", "", "HTML files (*.html);;All files (*)"
        )
        if not path:
            return
        try:
            from histo_to_ccf.viz.plotly3d import build_figure, save_html

            fig = build_figure(
                self._state.project, self._state.atlas,
                extra_regions=self._extra_region_list(),
            )
            out = save_html(fig, path, open_browser=True)
            self._status.setText(f"Saved → {out.name}")
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "Export failed", str(exc))

    def _view_napari3d(self) -> None:
        self._ensure_atlas(self._render_napari3d)

    def _render_napari3d(self) -> None:
        try:
            import napari

            from histo_to_ccf.viz.napari3d import show_3d_scene

            if self._viewer3d is None or not _viewer_alive(self._viewer3d):
                self._viewer3d = napari.Viewer(title="Histo→CCF - 3D")
            else:
                self._viewer3d.layers.clear()

            added = show_3d_scene(
                self._viewer3d,
                self._state.project,
                self._state.atlas,
                extra_regions=self._extra_region_list(),
            )
            if self._state.atlas is None:
                self._status.setText(
                    "Opened 3D window: probe tracks only. Load an atlas to see the brain."
                )
            else:
                self._status.setText(f"Opened 3D window: brain + {len(added)} layer(s).")
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "3D view failed", str(exc))

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
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "CSV export failed", str(exc))
