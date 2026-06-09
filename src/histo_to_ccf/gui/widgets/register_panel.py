"""Register button + progress bar + residuals + section overlay."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    """Register button, progress bar, residuals table, and section overlay."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        # Base dir that section.registration.bspline_transform_path resolves
        # against (set when registration runs); needed to reload transforms.
        self._reg_base_dir: Path | None = None
        # Persisted AppSettings (for the atlas storage folder); set via apply_settings.
        self._settings = None
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

        # Engine: elastix (masked + bending-energy regularized) vs plain SimpleITK.
        import importlib.util

        self._elastix_available = importlib.util.find_spec("itk") is not None

        self._use_elastix = QCheckBox("Regularized registration (elastix)")
        self._use_elastix.setChecked(self._elastix_available)
        self._use_elastix.setEnabled(self._elastix_available)
        self._use_elastix.setToolTip(
            "Use elastix (ABBA-style): a bending-energy penalty keeps the warp "
            "smooth and a tissue mask stops the fit chasing background/labels, so "
            "atlas boundaries stay on the tissue.\nUnchecked = plain SimpleITK "
            "B-spline."
            + ("" if self._elastix_available else "\nInstall the 'elastix' extra to enable.")
        )
        self._use_elastix.toggled.connect(self._on_elastix_toggled)
        params_layout.addWidget(self._use_elastix)

        bend_row = QHBoxLayout()
        bend_row.addWidget(QLabel("Smoothness (bending energy):"))
        self._bending_spin = QDoubleSpinBox()
        self._bending_spin.setRange(0.0, 500.0)
        self._bending_spin.setSingleStep(5.0)
        self._bending_spin.setValue(20.0)
        self._bending_spin.setToolTip(
            "Weight of the elastix bending-energy penalty. Higher = smoother / "
            "stiffer (closer to affine); lower = more local freedom."
        )
        bend_row.addWidget(self._bending_spin)
        params_layout.addLayout(bend_row)

        self._use_mask = QCheckBox("Restrict to tissue mask")
        self._use_mask.setChecked(True)
        self._use_mask.setToolTip(
            "Mask the metric to the section / atlas tissue so background and "
            "bright labels don't drive the registration (elastix only)."
        )
        params_layout.addWidget(self._use_mask)
        self._on_elastix_toggled(self._use_elastix.isChecked())

        method_lbl = QLabel(
            "Method: per-section 2D registration onto the atlas plane chosen by the "
            "AP you assigned. elastix adds a bending-energy penalty + tissue mask; "
            "each section is fit independently."
        )
        method_lbl.setWordWrap(True)
        method_lbl.setStyleSheet("color: gray; font-size: 11px;")
        method_lbl.setToolTip(
            "For each section: a coronal atlas plane is chosen (from DeepSlice or "
            "your assigned AP), the atlas reference is resampled at that plane and "
            "a 2D B-spline warps the histology onto it (mutual-information metric)."
        )
        params_layout.addWidget(method_lbl)

        self._use_deepslice = QCheckBox("Predict planes with DeepSlice")
        self._use_deepslice.setToolTip(
            "Run DeepSlice on all section images first to predict a consistent set "
            "of atlas planes across the series (with angle propagation and AP "
            "ordering), then refine each with the B-spline. No manual AP needed.\n"
            "First run downloads the DeepSlice model and is slow."
        )
        params_layout.addWidget(self._use_deepslice)
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

        self._overlay_btn = QPushButton("Show atlas overlay on sections")
        self._overlay_btn.setToolTip(
            "Warp the registered atlas region boundaries back onto each section "
            "image so you can see how well the fit matches the histology."
        )
        self._overlay_btn.clicked.connect(self._show_overlay)
        layout.addWidget(self._overlay_btn)

        # Manual per-section atlas correction (drag in the viewer).
        adjust_box = QGroupBox("Manual atlas adjustment")
        av = QVBoxLayout(adjust_box)
        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Section:"))
        self._adjust_combo = QComboBox()
        self._adjust_combo.setToolTip("Pick a registered section to nudge its atlas overlay.")
        sec_row.addWidget(self._adjust_combo, 1)
        av.addLayout(sec_row)
        self._adjust_btn = QPushButton("Adjust atlas (drag in viewer)")
        self._adjust_btn.setCheckable(True)
        self._adjust_btn.setToolTip(
            "Enter transform mode for this section's atlas overlay: drag the body to "
            "move it, drag the box handles to scale / stretch / rotate. Click again to "
            "apply (probes re-map and the project auto-saves)."
        )
        self._adjust_btn.toggled.connect(self._on_adjust_toggled)
        av.addWidget(self._adjust_btn)
        self._adjust_reset_btn = QPushButton("Reset adjustment")
        self._adjust_reset_btn.setToolTip("Clear the manual correction for this section.")
        self._adjust_reset_btn.clicked.connect(self._reset_adjustment)
        av.addWidget(self._adjust_reset_btn)
        layout.addWidget(adjust_box)

        layout.addStretch()

    def _on_elastix_toggled(self, on: bool) -> None:
        """Enable the elastix-only controls only when elastix is selected."""
        self._bending_spin.setEnabled(on)
        self._use_mask.setEnabled(on)

    def _engine(self) -> str:
        """Resolve the engine string from the checkbox."""
        return "elastix" if self._use_elastix.isChecked() else "sitk"

    def apply_settings(self, settings) -> None:
        """Populate controls from persisted AppSettings."""
        self._settings = settings
        self._grid_spin.setValue(settings.bspline_grid)
        self._iter_spin.setValue(settings.max_iterations)
        if self._elastix_available:
            # "auto" and "elastix" both mean "use elastix" in the UI.
            self._use_elastix.setChecked(settings.reg_engine != "sitk")
        self._bending_spin.setValue(settings.bending_energy_weight)
        self._use_mask.setChecked(settings.use_tissue_mask)
        self._on_elastix_toggled(self._use_elastix.isChecked())

    def collect_settings(self, settings) -> None:
        """Write current control values back into settings."""
        settings.bspline_grid = self._grid_spin.value()
        settings.max_iterations = self._iter_spin.value()
        settings.reg_engine = self._engine()
        settings.bending_energy_weight = self._bending_spin.value()
        settings.use_tissue_mask = self._use_mask.isChecked()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def refresh_after_load(self) -> None:
        """Repopulate the residuals table from a freshly-loaded project."""
        # Transform sidecars resolve against the loaded project's folder.
        if self._state.project_path is not None:
            self._reg_base_dir = self._state.project_path.parent
        self._refresh_residuals()
        self._populate_adjust_combo()
        n = sum(
            1 for slide in self._state.project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        if n:
            self._status.setText(f"Loaded project: {n} registered section(s).")

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

        use_deepslice = self._use_deepslice.isChecked()
        section_images: dict[int, np.ndarray] = {}
        for slide_idx, slide in enumerate(self._state.project.slides):
            img = self._state.slide_images.get(slide_idx)
            if img is None:
                continue
            for section in slide.sections:
                # With DeepSlice, planes are predicted — include every section.
                if not use_deepslice and section.plane is None:
                    continue
                x0, y0, x1, y1 = section.bbox_px
                section_images[section.index] = crop(img, (x0, y0, x1, y1)).astype(np.float32)

        if not section_images:
            _error_dialog(
                self, "Nothing to register",
                "Assign AP positions to sections in the Atlas tab first, or enable "
                "'Predict planes with DeepSlice'."
            )
            return

        # Ensure a persistent project location so the transforms (and the
        # auto-saved project) survive a reload, instead of a temp dir.
        project_path = self._ensure_project_path()
        transforms_dir = (
            project_path.parent / "transforms" if project_path is not None
            else Path(__import__("tempfile").mkdtemp()) / "transforms"
        )
        # Transform sidecar paths are stored relative to this base dir.
        self._reg_base_dir = transforms_dir.parent

        self._reg_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        if use_deepslice:
            self._status.setText("Running DeepSlice plane prediction (first run is slow)…")
            from histo_to_ccf.gui.workers import deepslice_worker

            ds_dir = transforms_dir.parent / "deepslice"
            ds = deepslice_worker(section_images, atlas, ds_dir)
            ds.returned.connect(
                lambda anch: self._start_register(section_images, transforms_dir, anch)
            )
            ds.errored.connect(self._on_registration_error)
            ds.start()
        else:
            self._start_register(section_images, transforms_dir, None)

    def _start_register(self, section_images, transforms_dir, anchorings) -> None:
        atlas = self._state.atlas
        n = len(anchorings) if anchorings else len(section_images)
        self._status.setText(f"Starting registration of {n} section(s)…")

        from histo_to_ccf.gui.workers import register_worker_progressive

        worker = register_worker_progressive(
            self._state.project,
            atlas,
            section_images,
            transforms_dir,
            bspline_grid=(self._grid_spin.value(),) * 2,
            max_iterations=self._iter_spin.value(),
            anchorings=anchorings,
            engine=self._engine(),
            bending_weight=self._bending_spin.value(),
            use_masks=self._use_mask.isChecked(),
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

    def _ensure_project_path(self) -> "Path | None":
        """Return the project path, defaulting to one next to the input data."""
        if self._state.project_path is not None:
            return self._state.project_path
        slides = self._state.project.slides
        if slides:
            self._state.project_path = Path(slides[0].image_path).with_suffix(
                ".histo2ccf.json"
            )
        return self._state.project_path

    def _on_registration_done(self, project) -> None:
        self._state.project = project
        n = sum(
            1 for slide in project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        self._progress.setValue(100)
        self._reg_btn.setEnabled(True)
        self._refresh_residuals()

        # Auto-save so the registration (results + transform sidecars) persists
        # and can be reloaded without re-running.
        msg = f"Done — {n} section(s) registered"
        path = self._ensure_project_path()
        if path is not None:
            try:
                from histo_to_ccf.project.io import save_project

                save_project(self._state.project, path)
                msg += f"  ·  auto-saved → {path.name}"
            except Exception as exc:  # noqa: BLE001
                msg += f"  ·  auto-save failed: {exc}"
        self._status.setText(msg)

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

    def _show_overlay(self) -> None:
        """Overlay registered atlas boundaries on each section in the viewer."""
        self._ensure_atlas(self._render_overlay)

    def _render_overlay(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            _error_dialog(self, "Atlas not loaded", "Load the atlas used for registration.")
            return
        import numpy as np
        from napari.utils.transforms import Affine

        from histo_to_ccf.registration.manual import section_to_world
        from histo_to_ccf.registration.transforms import (
            annotation_boundaries,
            warp_annotation_to_section,
        )

        registered = [
            section
            for slide in self._state.project.slides
            for section in slide.sections
            if section.registration is not None
        ]
        if not registered:
            self._status.setText("No registered sections yet — run registration first.")
            return

        # Transform sidecars resolve against the run's base dir, or — after a
        # project reload — against the loaded project's folder.
        base_dir = self._reg_base_dir
        if base_dir is None and self._state.project_path is not None:
            base_dir = self._state.project_path.parent

        count = 0
        first_error: Exception | None = None
        for section in registered:
            x0, y0, x1, y1 = section.bbox_px
            shape = (y1 - y0, x1 - x0)
            try:
                # Labels are already clipped to the warped atlas extent inside
                # warp_annotation_to_section (removes the inverse-extrapolation
                # stripes while keeping outlines over damaged tissue).
                labels = warp_annotation_to_section(
                    section.registration, atlas, shape, project_dir=base_dir
                )
                edges = annotation_boundaries(labels)
            except Exception as exc:  # noqa: BLE001 — surface to user below
                if first_error is None:
                    first_error = exc
                continue
            name = f"Atlas overlay {section.index}"
            edge_labels = edges.astype(np.uint8)
            if name in self._viewer.layers:
                layer = self._viewer.layers[name]
                layer.data = edge_labels
            else:
                layer = self._viewer.add_labels(
                    edge_labels, name=name, opacity=0.7, translate=(y0, x0)
                )
            # Re-apply any stored manual correction so it persists across reloads.
            if section.manual_affine is not None:
                world = section_to_world(np.asarray(section.manual_affine), (y0, x0))
                layer.affine = Affine(affine_matrix=world)
            else:
                layer.affine = Affine(affine_matrix=np.eye(3))
            count += 1

        if count:
            self._viewer.dims.ndisplay = 2
            self._populate_adjust_combo()
            self._status.setText(f"Overlaid atlas boundaries on {count} section(s).")
        else:
            _error_dialog(
                self, "Overlay failed",
                f"{len(registered)} section(s) are registered but the atlas could not "
                f"be warped onto them.\n\n{first_error}",
            )

    # ------------------------------------------------------------------
    # Manual atlas adjustment (drag the overlay in the viewer)
    # ------------------------------------------------------------------

    def _populate_adjust_combo(self) -> None:
        """List registered sections in the manual-adjust picker."""
        current = self._adjust_combo.currentData()
        self._adjust_combo.blockSignals(True)
        self._adjust_combo.clear()
        for slide in self._state.project.slides:
            for sec in slide.sections:
                if sec.registration is not None:
                    self._adjust_combo.addItem(f"Section {sec.index}", sec.index)
        if current is not None:
            i = self._adjust_combo.findData(current)
            if i >= 0:
                self._adjust_combo.setCurrentIndex(i)
        self._adjust_combo.blockSignals(False)

    def _adjust_section(self):
        """The Section currently chosen in the adjust picker, or None."""
        idx = self._adjust_combo.currentData()
        if idx is None:
            return None
        for slide in self._state.project.slides:
            for sec in slide.sections:
                if sec.index == idx:
                    return sec
        return None

    def _overlay_layer_for(self, section):
        name = f"Atlas overlay {section.index}"
        # napari LayerList has no .get(), so the SIM401 suggestion doesn't apply.
        if name in self._viewer.layers:  # noqa: SIM401
            return self._viewer.layers[name]
        return None

    def _on_adjust_toggled(self, on: bool) -> None:
        section = self._adjust_section()
        layer = self._overlay_layer_for(section) if section is not None else None
        if layer is None:
            if on:
                _error_dialog(
                    self, "Show the overlay first",
                    "Click 'Show atlas overlay on sections' before adjusting.",
                )
            self._adjust_btn.blockSignals(True)
            self._adjust_btn.setChecked(False)
            self._adjust_btn.blockSignals(False)
            self._adjust_btn.setText("Adjust atlas (drag in viewer)")
            return

        if on:
            self._viewer.layers.selection = {layer}
            layer.mode = "transform"
            self._adjust_btn.setText("Apply adjustment")
            self._status.setText(
                f"Adjusting section {section.index}: drag to move, box handles to "
                f"scale / stretch / rotate. Click 'Apply adjustment' when done."
            )
        else:
            self._commit_adjustment(section, layer)
            self._adjust_btn.setText("Adjust atlas (drag in viewer)")

    def _commit_adjustment(self, section, layer) -> None:
        """Read the layer's world affine, store it section-local, re-map + save."""
        import numpy as np

        from histo_to_ccf.registration.manual import is_identity, world_to_section

        try:
            layer.mode = "pan_zoom"
            x0, y0 = section.bbox_px[0], section.bbox_px[1]
            world = np.asarray(layer.affine.affine_matrix, dtype=float)
            section_affine = world_to_section(world, (y0, x0))
            section.manual_affine = (
                None if is_identity(section_affine) else section_affine.tolist()
            )
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "Adjustment failed", str(exc))
            return
        self._remap_and_save(section)

    def _reset_adjustment(self) -> None:
        import numpy as np
        from napari.utils.transforms import Affine

        section = self._adjust_section()
        if section is None:
            return
        section.manual_affine = None
        layer = self._overlay_layer_for(section)
        if layer is not None:
            layer.mode = "pan_zoom"
            layer.affine = Affine(affine_matrix=np.eye(3))
        if self._adjust_btn.isChecked():
            self._adjust_btn.blockSignals(True)
            self._adjust_btn.setChecked(False)
            self._adjust_btn.blockSignals(False)
            self._adjust_btn.setText("Adjust atlas (drag in viewer)")
        self._remap_and_save(section)

    def _remap_and_save(self, section) -> None:
        """Re-project probe coords through the manual-corrected transforms, save."""
        atlas = self._state.atlas
        base_dir = self._reg_base_dir
        if base_dir is None and self._state.project_path is not None:
            base_dir = self._state.project_path.parent
        remapped = False
        if atlas is not None:
            try:
                from histo_to_ccf.registration.pipeline import (
                    _apply_to_shank_registered,
                    reload_registered_transforms,
                )

                transforms = reload_registered_transforms(
                    self._state.project, atlas, project_dir=base_dir
                )
                for probe in self._state.project.probes:
                    for shank in probe.shanks:
                        _apply_to_shank_registered(shank, self._state.project, transforms)
                remapped = True
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Section {section.index}: probe re-map failed: {exc}")

        msg = f"Section {section.index} adjustment applied"
        msg += " (probes re-mapped)" if remapped else ""
        path = self._ensure_project_path()
        if path is not None:
            try:
                from histo_to_ccf.project.io import save_project

                save_project(self._state.project, path)
                msg += f"  ·  saved → {path.name}"
            except Exception as exc:  # noqa: BLE001
                msg += f"  ·  save failed: {exc}"
        self._status.setText(msg)

    # ------------------------------------------------------------------
    # Lazy atlas load (for the section overlay)
    # ------------------------------------------------------------------

    def _ensure_atlas(self, on_ready) -> None:
        """Ensure ``state.atlas`` is loaded, then call ``on_ready()``.

        After a project reload the atlas is not auto-loaded, so the section
        overlay would be empty. This lazily loads the atlas named in the project
        (from the saved atlas folder) in a background thread and invokes
        ``on_ready`` once available. If it is already loaded, or no atlas name is
        recorded, ``on_ready`` runs immediately.
        """
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
        self._status.setText(f"Loading atlas {atlas_id} for 3D view…")
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
