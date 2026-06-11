"""Register button + progress bar + residuals + section overlay."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
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

        # Registration parameters live in a dialog opened from the Registration
        # menu ("Parameters"), not inline - the defaults are good, so the panel
        # stays focused on "Register all sections" and the results below.
        self._params_dialog = None
        params_box = QGroupBox("Registration parameters")
        params_layout = QVBoxLayout(params_box)

        # DeepSlice plane prediction is the FIRST thing registration does (predict a
        # consistent set of atlas planes), so it sits at the top of the parameters
        # dialog; on by default since the user runs it almost every time.
        self._use_deepslice = QCheckBox("Predict planes with DeepSlice")
        self._use_deepslice.setChecked(True)
        self._use_deepslice.setToolTip(
            "Run DeepSlice on all section images first to predict a consistent set "
            "of atlas planes across the series (with angle propagation and AP "
            "ordering), then refine each with the B-spline. No manual AP needed.\n"
            "First run downloads the DeepSlice model and is slow."
        )
        params_layout.addWidget(self._use_deepslice)

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

        self._prealign = QCheckBox("Silhouette pre-align")
        self._prealign.setChecked(True)
        self._prealign.setToolTip(
            "Before the B-spline, snap the atlas onto each section's tissue with a "
            "closed-form translation + scale from the silhouettes (4-DOF, can't fold). "
            "Makes the outer-contour scale consistent across sections (elastix only)."
        )
        params_layout.addWidget(self._prealign)

        self._boundary_snap = QCheckBox("Snap atlas contour to tissue")
        self._boundary_snap.setChecked(True)
        self._boundary_snap.setToolTip(
            "After the fit, pull the atlas outer contour onto the section's tissue "
            "border with a smoothed, fold-proof landmark warp (the automatic "
            "version of the manual landmark drag). Large mismatches over damaged "
            "tissue are left alone. Works with any engine."
        )
        params_layout.addWidget(self._boundary_snap)
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
        # Held for the Parameters dialog; intentionally NOT added to the panel.
        self._params_box = params_box

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
        self._residuals_table.setHorizontalHeaderLabels(
            ["Section", "AP from bregma µm", "Residual"]
        )
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

        # Tool 1 - box transform, in its own outlined group.
        box_group = QGroupBox("Box transform")
        bg = QVBoxLayout(box_group)
        self._adjust_btn = QPushButton("Adjust atlas (drag in viewer)")
        self._adjust_btn.setCheckable(True)
        self._adjust_btn.setToolTip(
            "Enter transform mode for this section's atlas overlay: drag the body to "
            "move it, drag the box handles to scale / stretch / rotate. Click again to "
            "apply (probes re-map and the project auto-saves)."
        )
        self._adjust_btn.toggled.connect(self._on_adjust_toggled)
        bg.addWidget(self._adjust_btn)
        av.addWidget(box_group)

        # Tool 2 - landmark TPS warp, grouped: place, edit (move/add), apply.
        lm_group = QGroupBox("Landmarks")
        lg = QVBoxLayout(lm_group)
        self._reset_morph_btn = QPushButton("Reset morph to plane (keep AP/ML)")
        self._reset_morph_btn.setToolTip(
            "Drop the automatic B-spline warp for this section but keep its atlas "
            "plane (AP/ML). The overlay returns to the undistorted atlas slice, so "
            "for hard sections (torn / missing tissue) you can redo the fit by hand "
            "with landmarks instead of fighting a badly distorted outline. Clears "
            "any existing manual correction; re-maps probes and saves."
        )
        self._reset_morph_btn.clicked.connect(self._reset_morph)
        lg.addWidget(self._reset_morph_btn)
        self._place_lm_btn = QPushButton("Place landmarks")
        self._place_lm_btn.setToolTip(
            "Drop draggable correspondence points on the atlas overlay (6 around the "
            "border + 3 inside). Drag each onto the matching tissue feature for a "
            "thin-plate-spline warp that fixes LOCAL distortions a box transform can't.\n"
            "• drag = warp (pull the atlas to the tissue)\n"
            "• Ctrl+drag (or 'Move points' on) = relocate the landmark, no warp\n"
            "• 'Add points' on, then click = add; select a point + Delete = remove"
        )
        self._place_lm_btn.clicked.connect(self._place_landmarks)
        lg.addWidget(self._place_lm_btn)

        lm_row = QHBoxLayout()
        self._lm_move_btn = QPushButton("Move points")
        self._lm_move_btn.setCheckable(True)
        self._lm_move_btn.setToolTip(
            "Relocate landmarks (move the point + its atlas anchor together, no warp). "
            "Same as holding Ctrl while dragging."
        )
        self._lm_move_btn.toggled.connect(self._on_lm_move_toggled)
        lm_row.addWidget(self._lm_move_btn)
        self._lm_add_btn = QPushButton("Add points")
        self._lm_add_btn.setCheckable(True)
        self._lm_add_btn.setToolTip("Click in the viewer to add a landmark. Select a point + Delete removes it.")
        self._lm_add_btn.toggled.connect(self._on_lm_add_toggled)
        lm_row.addWidget(self._lm_add_btn)
        lg.addLayout(lm_row)

        self._apply_lm_btn = QPushButton("Apply landmark warp")
        self._apply_lm_btn.setToolTip("Warp the atlas through the dragged landmarks, re-map probes, save.")
        self._apply_lm_btn.clicked.connect(self._apply_landmarks)
        lg.addWidget(self._apply_lm_btn)
        av.addWidget(lm_group)

        # Reset applies to either tool - set it apart below a divider.
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        av.addWidget(divider)
        self._adjust_reset_btn = QPushButton("Reset adjustment")
        self._adjust_reset_btn.setToolTip("Clear the manual correction (box or landmarks) for this section.")
        self._adjust_reset_btn.clicked.connect(self._reset_adjustment)
        av.addWidget(self._adjust_reset_btn)
        layout.addWidget(adjust_box)
        # Active landmark-edit state. Source (atlas anchor) per point lives in the
        # Points layer `features` (travels through add/delete); `data` is the target.
        self._landmark_idx: int | None = None
        self._lm_prev_data = None  # for per-move delta tracking
        self._lm_ctrl_drag = False  # set while Ctrl is held during a drag

        layout.addStretch()

    def open_parameters_dialog(self) -> None:
        """Show the registration-parameters group in a (lazily-built) dialog.

        Wired to the menu-bar "Registration → Parameters" action. The parameter
        widgets are reparented into the dialog once and persist there, so the
        values the user sets are the same widgets the registration run reads.
        """
        if self._params_dialog is None:
            from qtpy.QtWidgets import QDialog, QVBoxLayout

            dlg = QDialog(self)
            dlg.setWindowTitle("Registration parameters")
            lay = QVBoxLayout(dlg)
            lay.addWidget(self._params_box)
            self._params_dialog = dlg
        self._params_dialog.show()
        self._params_dialog.raise_()
        self._params_dialog.activateWindow()

    def _on_elastix_toggled(self, on: bool) -> None:
        """Enable the elastix-only controls only when elastix is selected."""
        self._bending_spin.setEnabled(on)
        self._use_mask.setEnabled(on)
        self._prealign.setEnabled(on)

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
        self._prealign.setChecked(settings.prealign_similarity)
        self._boundary_snap.setChecked(settings.boundary_snap)
        self._on_elastix_toggled(self._use_elastix.isChecked())

    def collect_settings(self, settings) -> None:
        """Write current control values back into settings."""
        settings.bspline_grid = self._grid_spin.value()
        settings.max_iterations = self._iter_spin.value()
        settings.reg_engine = self._engine()
        settings.bending_energy_weight = self._bending_spin.value()
        settings.use_tissue_mask = self._use_mask.isChecked()
        settings.prealign_similarity = self._prealign.isChecked()
        settings.boundary_snap = self._boundary_snap.isChecked()

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
                # With DeepSlice, planes are predicted - include every section.
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
            # Number DeepSlice's input by the user's AP sequence (ap_order), so its
            # serial-section ordering follows the intended order, not the raw
            # detection index.
            ds = deepslice_worker(section_images, atlas, ds_dir,
                                  order=self._section_order())
            ds.returned.connect(
                lambda anch: self._start_register(section_images, transforms_dir, anch)
            )
            ds.errored.connect(self._on_registration_error)
            ds.start()
        else:
            self._start_register(section_images, transforms_dir, None)

    def _section_order(self) -> dict[int, int]:
        """Map ``section.index`` to its rank in the user's AP sequence (``ap_order``).

        DeepSlice orders the series by its input filename token; numbering by the
        ap_order rank makes it follow the order the user set in the ordering panel
        (which writes ``ap_order``) rather than the raw detection index.
        """
        order: dict[int, int] = {}
        for slide in self._state.project.slides:
            ranked = sorted(slide.sections, key=lambda s: s.ap_order)
            for rank, section in enumerate(ranked):
                order[section.index] = rank
        return order

    def _start_register(self, section_images, transforms_dir, anchorings) -> None:
        atlas = self._state.atlas
        # Guide DeepSlice with any user-assigned AP: shift its predicted planes so
        # they honour the hand-set AP values (keeping DeepSlice's tilt/spacing).
        if anchorings:
            from histo_to_ccf.registration.pipeline import guide_anchorings_with_planes

            anchorings = guide_anchorings_with_planes(
                anchorings, self._state.project, atlas
            )
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
            prealign=self._prealign.isChecked(),
            boundary_snap=self._boundary_snap.isChecked(),
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
        self._progress.setFormat(f"{current}/{total} - {pct}%")
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
        msg = f"Done - {n} section(s) registered"
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
        """Fill the residuals table - AP **from bregma** and in **AP order**, to
        match the Atlas/ordering tab, using the *actual registered* plane.

        The AP shown is the centre of each section's registered anchoring (what was
        really registered, incl. any DeepSlice guidance), converted to bregma µm -
        not ``plane.ap_um`` (the request), which can differ. Rows are sorted by
        ``ap_order`` so the sequence reads the same as the ordering list.
        """
        from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM

        ap_res = float(self._state.project.atlas.resolution_um or 25.0)
        rows = []
        for slide in self._state.project.slides:
            for sec in slide.sections:
                if sec.registration is None:
                    continue
                a = sec.registration.anchoring  # (ox,oy,oz,ux,uy,uz,vx,vy,vz) voxels
                ap_idx = a[0] + 0.5 * a[3] + 0.5 * a[6]  # AP of the plane centre
                ap_bregma = BREGMA_AP_FROM_ORIGIN_UM - ap_idx * ap_res
                rows.append((sec.ap_order, sec.index, ap_bregma, sec.registration.residual))
        rows.sort(key=lambda r: r[0])
        self._residuals_table.setRowCount(len(rows))
        for i, (_order, idx, ap, res) in enumerate(rows):
            self._residuals_table.setItem(i, 0, QTableWidgetItem(str(idx)))
            self._residuals_table.setItem(i, 1, QTableWidgetItem(f"{ap:+.0f}" if ap == ap else "-"))
            self._residuals_table.setItem(i, 2, QTableWidgetItem(f"{res:.4f}" if res is not None else "-"))

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
            self._status.setText("No registered sections yet - run registration first.")
            return

        # Transform sidecars resolve against the run's base dir, or - after a
        # project reload - against the loaded project's folder.
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
                # Landmark TPS warp is baked into the label image; the box-handle
                # affine rides on the layer's affine (live, free). Mutually exclusive.
                if section.manual_landmarks is not None:
                    from histo_to_ccf.registration.landmarks_warp import warp_label_image

                    labels = warp_label_image(
                        labels,
                        np.asarray(section.manual_landmarks.source, dtype=float),
                        np.asarray(section.manual_landmarks.target, dtype=float),
                    )
                edges = annotation_boundaries(labels)
            except Exception as exc:  # noqa: BLE001 - surface to user below
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
            if section.manual_landmarks is not None:
                layer.affine = Affine(affine_matrix=np.eye(3))  # warp is in the data
            elif section.manual_affine is not None:
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
        # Ordered by section index so the picker reads in sequence (0, 1, 2, …)
        # rather than the slide's detection order.
        indices = sorted(
            sec.index
            for slide in self._state.project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        for idx in indices:
            self._adjust_combo.addItem(f"Section {idx}", idx)
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
        if name in self._viewer.layers:
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
        section.manual_landmarks = None
        layer = self._overlay_layer_for(section)
        if layer is not None:
            layer.mode = "pan_zoom"
            layer.affine = Affine(affine_matrix=np.eye(3))
        # Drop any landmark points layer.
        lm_name = f"Atlas landmarks {section.index}"
        if lm_name in self._viewer.layers:
            self._viewer.layers.remove(lm_name)
        if self._adjust_btn.isChecked():
            self._adjust_btn.blockSignals(True)
            self._adjust_btn.setChecked(False)
            self._adjust_btn.blockSignals(False)
            self._adjust_btn.setText("Adjust atlas (drag in viewer)")
        # Restore the un-corrected overlay for this section.
        self._rerender_section_overlay(section)
        self._remap_and_save(section)

    def _reset_morph(self) -> None:
        """Discard the auto B-spline morph for a section, keep its plane (AP/ML).

        For sections the deformable fit mangles (torn / missing tissue), this
        reverts the overlay to the undistorted atlas slice so the user can redo it
        by hand with landmarks. Setting ``bspline_transform_path = None`` makes both
        the overlay (`warp_annotation_to_section` falls back to the plane resize)
        and the probe mapping (`bspline=None`) use only the anchoring. Any prior
        manual correction is cleared so the landmark fit starts from the raw plane.
        """
        section = self._adjust_section()
        if section is None or section.registration is None:
            _error_dialog(self, "No registered section", "Pick a registered section first.")
            return
        section.registration.bspline_transform_path = None
        section.manual_affine = None
        section.manual_landmarks = None
        # Drop any landmark/edit layer + reset the box-adjust toggle.
        lm_name = f"Atlas landmarks {section.index}"
        if lm_name in self._viewer.layers:
            self._viewer.layers.remove(lm_name)
        if self._adjust_btn.isChecked():
            self._adjust_btn.blockSignals(True)
            self._adjust_btn.setChecked(False)
            self._adjust_btn.blockSignals(False)
            self._adjust_btn.setText("Adjust atlas (drag in viewer)")
        self._rerender_section_overlay(section)
        self._remap_and_save(section)
        self._status.setText(
            f"Section {section.index}: morph dropped, atlas plane (AP/ML) kept. "
            "Click 'Place landmarks' to fit it by hand."
        )

    # ------------------------------------------------------------------
    # Landmark (thin-plate-spline) correction
    # ------------------------------------------------------------------

    def _warp_labels_for(self, section, *, apply_landmarks: bool):
        """Warped atlas label image for a section (optionally with the TPS)."""
        import numpy as np

        from histo_to_ccf.registration.transforms import warp_annotation_to_section

        if self._state.atlas is None:
            return None
        base_dir = self._reg_base_dir
        if base_dir is None and self._state.project_path is not None:
            base_dir = self._state.project_path.parent
        x0, y0, x1, y1 = section.bbox_px
        labels = warp_annotation_to_section(
            section.registration, self._state.atlas, (y1 - y0, x1 - x0), project_dir=base_dir
        )
        if apply_landmarks and section.manual_landmarks is not None:
            from histo_to_ccf.registration.landmarks_warp import warp_label_image

            labels = warp_label_image(
                labels,
                np.asarray(section.manual_landmarks.source, dtype=float),
                np.asarray(section.manual_landmarks.target, dtype=float),
            )
        return labels

    def _rerender_section_overlay(self, section) -> None:
        """Redraw one section's atlas-overlay layer from its current correction."""
        import numpy as np
        from napari.utils.transforms import Affine

        from histo_to_ccf.registration.transforms import annotation_boundaries

        labels = self._warp_labels_for(section, apply_landmarks=True)
        if labels is None:
            return
        edges = annotation_boundaries(labels).astype(np.uint8)
        x0, y0 = section.bbox_px[0], section.bbox_px[1]
        name = f"Atlas overlay {section.index}"
        if name in self._viewer.layers:
            layer = self._viewer.layers[name]
            layer.data = edges
            layer.translate = (y0, x0)
            layer.affine = Affine(affine_matrix=np.eye(3))
        else:
            self._viewer.add_labels(edges, name=name, opacity=0.7, translate=(y0, x0))

    def _landmark_layer(self):
        if self._landmark_idx is None:
            return None
        name = f"Atlas landmarks {self._landmark_idx}"
        if name in self._viewer.layers:
            return self._viewer.layers[name]
        return None

    def _place_landmarks(self) -> None:
        import numpy as np

        from histo_to_ccf.registration.landmarks_warp import auto_landmarks

        section = self._adjust_section()
        if section is None or section.registration is None:
            _error_dialog(self, "No registered section", "Pick a registered section first.")
            return
        labels = self._warp_labels_for(section, apply_landmarks=False)
        if labels is None:
            _error_dialog(self, "Atlas not loaded", "Click 'Show atlas overlay' first.")
            return
        # Continue from stored landmarks if present, else auto-place fresh ones.
        if section.manual_landmarks is not None:
            source = np.asarray(section.manual_landmarks.source, dtype=float)
            targets = np.asarray(section.manual_landmarks.target, dtype=float)
        else:
            source = auto_landmarks(labels > 0)
            targets = source.copy()
        self._landmark_idx = section.index

        x0, y0 = section.bbox_px[0], section.bbox_px[1]
        # napari (row, col) world coords. data = target; features carry source.
        data = np.column_stack([targets[:, 1] + y0, targets[:, 0] + x0])
        feats = {"sy": source[:, 1] + y0, "sx": source[:, 0] + x0}
        name = f"Atlas landmarks {section.index}"
        if name in self._viewer.layers:
            self._viewer.layers.remove(name)
        layer = self._viewer.add_points(
            data, name=name, size=16, face_color="red", border_color="white", features=feats
        )
        layer.mode = "select"
        self._viewer.layers.selection = {layer}
        self._viewer.dims.ndisplay = 2
        self._lm_prev_data = np.asarray(layer.data, dtype=float).copy()
        layer.events.data.connect(self._on_landmark_data)
        layer.mouse_drag_callbacks.append(self._landmark_drag_modifier)
        self._lm_move_btn.setChecked(False)
        self._lm_add_btn.setChecked(False)
        self._status.setText(
            f"Section {section.index}: drag landmarks onto the tissue (warp); Ctrl+drag "
            f"or 'Move points' to relocate; 'Add points' + click to add, Delete to remove. "
            f"Then 'Apply landmark warp'."
        )

    # --- landmark editing callbacks -----------------------------------

    def _landmark_drag_modifier(self, layer, event):
        """Record whether Ctrl is held for the duration of a drag (re-anchor)."""
        self._lm_ctrl_drag = "Control" in event.modifiers
        yield
        # keep the flag set through the drag; it's consumed by _on_landmark_data

    def _on_lm_move_toggled(self, on: bool) -> None:
        layer = self._landmark_layer()
        if layer is not None and not self._lm_add_btn.isChecked():
            layer.mode = "select"

    def _on_lm_add_toggled(self, on: bool) -> None:
        layer = self._landmark_layer()
        if layer is not None:
            layer.mode = "add" if on else "select"

    def _on_landmark_data(self, event=None) -> None:
        """Keep each point's source (atlas anchor, in `features`) in sync on edits."""
        import numpy as np

        layer = self._landmark_layer()
        if layer is None:
            return
        data = np.asarray(layer.data, dtype=float)
        prev = self._lm_prev_data
        sy = np.array(layer.features.get("sy", []), dtype=float)  # copy (Series is read-only)
        sx = np.array(layer.features.get("sx", []), dtype=float)

        if prev is None or len(data) > len(prev):
            # Added point(s) (appended at the end): anchor each where it was dropped.
            n_new = len(data) if prev is None else len(data) - len(prev)
            if sy.shape[0] != len(data):  # features not yet grown to match
                sy = np.resize(sy, len(data))
                sx = np.resize(sx, len(data))
            sy[-n_new:] = data[-n_new:, 0]
            sx[-n_new:] = data[-n_new:, 1]
            layer.features = {"sy": sy, "sx": sx}
        elif len(data) == len(prev):
            # A move. Relocate (Ctrl / 'Move points') shifts the anchor too, so the
            # displacement is unchanged; a plain drag moves only the target = warp.
            if (self._lm_ctrl_drag or self._lm_move_btn.isChecked()) and sy.shape[0] == len(data):
                delta = data - prev
                layer.features = {"sy": sy + delta[:, 0], "sx": sx + delta[:, 1]}
        # deletes keep features aligned automatically (napari drops the row).
        self._lm_prev_data = data.copy()

    def _apply_landmarks(self) -> None:
        import numpy as np

        from histo_to_ccf.project.schema import ManualLandmarks

        section = self._adjust_section()
        if section is None:
            return
        layer = self._landmark_layer()
        if layer is None or self._landmark_idx != section.index:
            _error_dialog(self, "No landmarks", "Click 'Place landmarks' first.")
            return
        x0, y0 = section.bbox_px[0], section.bbox_px[1]
        data = np.asarray(layer.data, dtype=float)  # (row, col) world = target
        sy = np.asarray(layer.features["sy"], dtype=float)
        sx = np.asarray(layer.features["sx"], dtype=float)
        if len(data) < 4:
            _error_dialog(self, "Too few landmarks", "Keep at least 4 landmark points.")
            return
        target = np.column_stack([data[:, 1] - x0, data[:, 0] - y0])  # (x, y) section-local
        source = np.column_stack([sx - x0, sy - y0])
        section.manual_landmarks = ManualLandmarks(
            source=source.tolist(), target=target.tolist()
        )
        section.manual_affine = None  # landmarks take precedence
        self._rerender_section_overlay(section)
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
