"""Permanent 3D-visualization + export panel (right dock).

Split out of the Register tab so 3D view and exports are always available, not
buried in one workflow step. Operates on the shared WorkflowState; lazily loads
the project's atlas when a 3D view / overlay needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QComboBox,
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

# CCFv3-aligned atlases that share the Allen 25 µm voxel space, so probe
# coordinates are identical and only the region annotation/meshes/acronyms differ.
# (label, brainglobe id) - mirrors the Atlas tab's quick picks.
_COMPATIBLE_REGION_ATLASES = [
    ("Allen CCFv3 25 µm", "allen_mouse_25um"),
    ("CCFv3-BBP Augmented 25 µm", "ccfv3augmented_mouse_25um"),
    ("Chon/Kim Unified 25 µm", "kim_mouse_25um"),
]

# CCF→Paxinos stereotaxic alignment presets for the Paxinos export (label, key).
# All but "none" un-pitch CCFv3's ~5° nose-down tilt; see ccf_coords.PAXINOS_ALIGNMENTS.
_PAXINOS_ALIGNMENT_CHOICES = [
    ("Pinpoint / Qiu 2018 (5° pitch)", "qiu2018"),
    ("Pinpoint / Dorr 2008 (5° pitch)", "dorr2008"),
    ("Allen forum (5° pitch, DV ×0.943)", "allen_forum"),
    ("None - linear mirror, no tilt", "none"),
]

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
        # Region/display atlas (request: view/export regions in a compatible atlas
        # without re-registering). None until resolved by _ensure_display_atlas.
        self._display_atlas = None
        self._display_atlas_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Recompute probe/electrode CCF coordinates from the current registration.
        self._update_btn = QPushButton("Update coordinates")
        self._update_btn.setToolTip(
            "Re-map every probe tip / entry (and per-channel) from its pixel "
            "position through the current registration - including manual atlas "
            "corrections and any tip/entry points you moved - into CCF µm, then "
            "save. Moving points or correcting a section does NOT update the CCF "
            "coordinates (used by the 3D view + exports) until you click this; an "
            "open 3D window is refreshed too. Needs the atlas loaded."
        )
        self._update_btn.clicked.connect(self._update_coordinates)
        layout.addWidget(self._update_btn)

        viz_box = QGroupBox("3D Visualization")
        viz_layout = QVBoxLayout(viz_box)

        # Region/display atlas picker: render regions from a different but
        # coordinate-compatible CCFv3 atlas. Probe coordinates are unchanged.
        atlas_row = QHBoxLayout()
        atlas_row.addWidget(QLabel("Region atlas:"))
        self._region_atlas_combo = QComboBox()
        self._region_atlas_combo.addItem("Same as registration", "")
        for label, aid in _COMPATIBLE_REGION_ATLASES:
            self._region_atlas_combo.addItem(label, aid)
        self._region_atlas_combo.setToolTip(
            "Draw region meshes / acronyms from this atlas in the 3D views and Plotly "
            "export. These CCFv3-aligned 25 µm atlases share the registration atlas's "
            "voxel space, so probe coordinates are identical - only the region "
            "annotation differs. 'Same as registration' uses the project atlas."
        )
        atlas_row.addWidget(self._region_atlas_combo)
        viz_layout.addLayout(atlas_row)

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

        pax_row = QHBoxLayout()
        pax_row.addWidget(QLabel("Paxinos align:"))
        self._paxinos_combo = QComboBox()
        for label, key in _PAXINOS_ALIGNMENT_CHOICES:
            self._paxinos_combo.addItem(label, key)
        self._paxinos_combo.setToolTip(
            "CCF→Paxinos stereotaxic transform. CCFv3 is pitched ~5° nose-down vs a "
            "flat-skull frame, so all but 'None' un-pitch by 5° and apply published "
            "axis scaling (Qiu 2018 = Pinpoint's recommended default). These are "
            "ESTIMATES with real variance - validate against histology."
        )
        pax_row.addWidget(self._paxinos_combo)
        export_layout.addLayout(pax_row)

        pax_btn = QPushButton("Export per-channel Paxinos CSV")
        pax_btn.setToolTip(
            "Per-channel coordinates in Paxinos stereotaxic mm (bregma origin): "
            "AP anterior-positive, ML 0 at midline, DV depth below bregma."
        )
        pax_btn.clicked.connect(self._export_paxinos_csv)
        export_layout.addWidget(pax_btn)
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

    def _selected_region_atlas_id(self) -> str:
        idx = self._region_atlas_combo.currentIndex()
        return self._region_atlas_combo.itemData(idx) or ""

    def _ensure_display_atlas(self, on_ready) -> None:
        """Resolve the chosen region atlas into ``self._display_atlas``, then call
        ``on_ready()``. 'Same as registration' reuses ``state.atlas``; a different
        compatible atlas is loaded lazily and cached. The registration atlas is
        still ensured (probe remap needs it)."""
        chosen = self._selected_region_atlas_id()
        proj_id = self._state.project.atlas.name

        def _after_reg() -> None:
            if not chosen or chosen == proj_id:
                self._display_atlas = self._state.atlas
                self._display_atlas_id = proj_id or None
                on_ready()
                return
            if self._display_atlas is not None and self._display_atlas_id == chosen:
                on_ready()
                return
            atlas_dir = None
            if self._settings is not None:
                atlas_dir = getattr(self._settings, "atlas_dir", "") or None
            self._status.setText(f"Loading region atlas {chosen}…")
            from histo_to_ccf.gui.workers import load_atlas_worker

            worker = load_atlas_worker(chosen, brainglobe_dir=atlas_dir)

            def _loaded(atlas) -> None:
                self._display_atlas = atlas
                self._display_atlas_id = chosen
                on_ready()

            def _failed(exc) -> None:
                self._display_atlas = self._state.atlas
                self._status.setText(
                    f"Region atlas load failed ({exc}); using registration atlas."
                )
                on_ready()

            worker.returned.connect(_loaded)
            worker.errored.connect(_failed)
            worker.start()

        self._ensure_atlas(_after_reg)

    def _export_plotly(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plotly HTML", "", "HTML files (*.html);;All files (*)"
        )
        if not path:
            return
        self._ensure_display_atlas(lambda: self._do_export_plotly(path))

    def _do_export_plotly(self, path: str) -> None:
        try:
            from histo_to_ccf.viz.plotly3d import build_figure, save_html

            fig = build_figure(
                self._state.project, self._display_atlas,
                extra_regions=self._extra_region_list(),
            )
            out = save_html(fig, path, open_browser=True)
            self._status.setText(f"Saved → {out.name}")
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "Export failed", str(exc))

    def _remap_probes(self) -> int:
        """Re-project every shank tip/entry pixel through the current transforms.

        Reads the live registration (incl. manual affine / landmarks / reset-to-
        plane) and the current ``tip_px`` / ``entry_px`` (which moving a point in
        the Probes tab updates), and rewrites ``tip_ccf_um`` / ``entry_ccf_um``.
        Returns the number of shanks updated. Requires ``state.atlas``.
        """
        atlas = self._state.atlas
        if atlas is None or not self._state.project.probes:
            return 0
        from histo_to_ccf.registration.pipeline import (
            _apply_to_shank_registered,
            reload_registered_transforms,
        )

        base_dir = (
            self._state.project_path.parent if self._state.project_path else None
        )
        transforms = reload_registered_transforms(
            self._state.project, atlas, project_dir=base_dir
        )
        n = 0
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                _apply_to_shank_registered(shank, self._state.project, transforms)
                n += 1
        return n

    def _update_coordinates(self) -> None:
        self._ensure_atlas(self._do_update_coordinates)

    def _do_update_coordinates(self) -> None:
        if self._state.atlas is None:
            self._status.setText("Load an atlas first - it's needed to map pixels to CCF.")
            return
        if not self._state.project.probes:
            self._status.setText("No probes to update.")
            return
        try:
            n = self._remap_probes()
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "Update failed", str(exc))
            return
        msg = f"Updated {n} shank coordinate(s) from the current registration"
        if self._state.project_path is not None:
            try:
                from histo_to_ccf.project.io import save_project

                save_project(self._state.project, self._state.project_path)
                msg += f"  ·  saved → {self._state.project_path.name}"
            except Exception as exc:  # noqa: BLE001
                msg += f"  ·  save failed: {exc}"
        # Refresh an already-open 3D window so it reflects the new coordinates.
        if self._viewer3d is not None and _viewer_alive(self._viewer3d):
            self._render_napari3d()
        self._status.setText(msg)

    def _view_napari3d(self) -> None:
        # Re-map first so the 3D view always reflects the latest corrections; the
        # region atlas (possibly different from the registration atlas) is resolved
        # into self._display_atlas before rendering.
        self._ensure_display_atlas(self._remap_then_render)

    def _remap_then_render(self) -> None:
        try:
            self._remap_probes()
        except Exception:  # noqa: BLE001 - render with whatever coords exist
            pass
        self._render_napari3d()

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
                self._display_atlas,
                extra_regions=self._extra_region_list(),
            )
            if self._display_atlas is None:
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

    def _export_paxinos_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Paxinos CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            from histo_to_ccf.probes.channels import export_paxinos_csv

            align = self._paxinos_combo.currentData()
            n = export_paxinos_csv(self._state.project, path, alignment=align)
            if n == 0:
                _error_dialog(self, "Nothing to export", "No registered shank coordinates found.")
            else:
                self._status.setText(
                    f"Paxinos CSV ({n} rows, {align}) → {Path(path).name}"
                )
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "Paxinos export failed", str(exc))
