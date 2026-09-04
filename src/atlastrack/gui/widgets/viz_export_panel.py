"""Permanent 3D-visualization + export panel (right dock).

Split out of the Register tab so 3D view and exports are always available, not
buried in one workflow step. Operates on the shared WorkflowState; lazily loads
the project's atlas when a 3D view / overlay needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from atlastrack.gui.workflow import WorkflowState

#: What a single Export button can write (label, key). Both are the same registered
#: coordinates in different containers, which is why one button and a selector beat
#: one button per file type - the old panel had two "export CSV" buttons that differed
#: only by a coordinate frame.
_EXPORT_FORMATS = [
    ("Per-channel coordinates (CSV)", "csv"),
    ("Probe tracks for Python / HERBS (pkl)", "pkl"),
    ("3D view as interactive HTML", "html"),
    ("Registered section series (folder of images)", "series"),
]


def _muted(text: str) -> QLabel:
    """A wrapped, de-emphasised explanatory line."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(mid);")
    return label

# CCFv3-aligned atlases: all cover the same physical volume as the registration
# atlas, so probe coordinates are identical and only the naming/annotation differs.
# The isotropic Chon/Kim samples that volume at 20 µm rather than 25, which the
# section-series export handles by restating each anchoring on its grid (see
# atlas.planes.rescale_atlas_anchoring) - the 3D views work from meshes in µm and
# never cared. (label, brainglobe id).
_COMPATIBLE_REGION_ATLASES = [
    ("Allen CCFv3 25 µm", "allen_mouse_25um"),
    ("CCFv3-BBP Augmented 25 µm", "ccfv3augmented_mouse_25um"),
    ("Chon/Kim Unified 25 µm (Franklin-Paxinos labels)", "kim_mouse_25um"),
    (
        "Chon/Kim Unified v2, isotropic 20 µm (Franklin-Paxinos labels)",
        "kim_mouse_isotropic_20um",
    ),
]

#: Region atlases whose annotation does not sit exactly on the registration atlas's,
#: with the measured offset. Not corrected for: the shift belongs to the atlas
#: release, and silently moving v2's labels back onto v1 would throw away the
#: corrections that are the reason to use v2. Surfaced so labels within a section or
#: two of a boundary are not over-trusted.
_REGION_ATLAS_CAVEATS = {
    "kim_mouse_isotropic_20um": (
        "its annotation sits ~102 µm posterior of the 25 µm release (volume "
        "centroids over 811 structures: +101.8 +/- 26.6 µm, a pure translation)"
    ),
}

# CCF→Paxinos presets (label, key). The labels lead with what the preset *does*;
# the citation is secondary, because "Pinpoint" and "Allen forum" name the source of
# the numbers and tell a user nothing about the effect. Full detail lives behind the
# "?" button. All but "none" un-pitch CCFv3's ~5° nose-down tilt; the scale factors
# are in ccf_coords.PAXINOS_ALIGNMENTS.
_PAXINOS_ALIGNMENT_CHOICES = [
    ("Tilt + all-axis scaling - recommended (Qiu 2018)", "qiu2018"),
    ("Tilt + AP/DV scaling (Dorr 2008)", "dorr2008"),
    ("Tilt + DV scaling only (Allen community)", "allen_forum"),
    ("No correction - mirror onto bregma, no tilt", "none"),
]

#: Shown by the "?" next to the Paxinos checkbox. Long-form on purpose: this is the
#: one place the CCF/Paxinos distinction and the size of the guess are spelled out.
_PAXINOS_HELP = """\
<b>Paxinos is a conversion applied at export, not an atlas you register to.</b>

<p>Registration, the atlas overlay and every coordinate stored in your project are in
Allen CCF. There is no Paxinos volume to warp onto - it is a stereotaxic reference
frame, so it can only be reached by transforming finished CCF coordinates.</p>

<p>The conversion re-origins on bregma, un-pitches CCFv3's ~5° nose-down tilt relative
to a flat-skull frame, and applies published per-axis scale factors:</p>

<table cellpadding="4">
<tr><th align="left">Preset</th><th>Tilt</th><th>AP</th><th>ML</th><th>DV</th></tr>
<tr><td>Qiu 2018 (recommended)</td><td align="center">5°</td>
    <td align="center">1.031</td><td align="center">0.952</td>
    <td align="center">0.885</td></tr>
<tr><td>Dorr 2008</td><td align="center">5°</td>
    <td align="center">1.087</td><td align="center">1.000</td>
    <td align="center">0.952</td></tr>
<tr><td>Allen community</td><td align="center">5°</td>
    <td align="center">1.000</td><td align="center">1.000</td>
    <td align="center">0.943</td></tr>
<tr><td>No correction</td><td align="center">0°</td>
    <td align="center">1.000</td><td align="center">1.000</td>
    <td align="center">1.000</td></tr>
</table>

<p>Output is millimetres from bregma: AP anterior-positive, ML 0 at the midline, DV
depth below bregma.</p>

<p><b>Paxinos region <i>labels</i> are a separate thing, and they need no estimate at
all.</b> The Chon/Kim Unified atlases carry Franklin-Paxinos nomenclature over the
same CCF volume as Allen, so selecting one under <i>Region atlas</i> names regions
M1, S1BF, 4V rather than MOp, SSp-bfd, V4 - in the hover readout, the 3D views, and the
section series' outlines and region list. That is a relabelling, not a transform, so
nothing is approximated.</p>

<p><b>These presets are estimates and they disagree with each other</b> - by more than
a millimetre deep in the brain. None of them is ground truth for your animal. Treat
the choice as a stated assumption and validate against your own histology.</p>
"""

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

        # Both rows act on the probes, so they belong together and apart from the
        # visualisation and export groups below.
        probe_box = QGroupBox("Probe")
        probe_layout = QVBoxLayout(probe_box)

        # Recompute probe/electrode CCF coordinates from the current registration.
        self._update_btn = QPushButton("Update probe coordinates")
        self._update_btn.setToolTip(
            "Re-map every probe tip / entry (and per-channel) from its pixel "
            "position through the current registration - including manual atlas "
            "corrections and any tip/entry points you moved - into CCF µm, then "
            "save. Moving points or correcting a section does NOT update the CCF "
            "coordinates (used by the 3D view + exports) until you click this; an "
            "open 3D window is refreshed too. Needs the atlas loaded."
        )
        self._update_btn.clicked.connect(self._update_coordinates)
        probe_layout.addWidget(self._update_btn)

        # Enforce a rigid multi-shank array when re-mapping (physically, a
        # Neuropixels shank set is parallel + evenly spaced; independent per-section
        # picks are noisy). Applied on every "Update coordinates" / "View 3D" so it
        # survives re-mapping. Tolerance leaves a little slack for real deviations.
        rigid_row = QHBoxLayout()
        self._rigid_check = QCheckBox("Enforce rigid array")
        self._rigid_check.setToolTip(
            "Regularize each multi-shank probe (>=3 shanks) to a parallel, evenly-"
            "spaced array after mapping tip/entry from pixels. Spacing is estimated "
            "from your picks. The physical probe is rigid, so this removes picking "
            "noise (uneven spacing, an off-axis shank) while the tolerance keeps a "
            "little slack. Re-applied on every Update coordinates / View 3D."
        )
        rigid_row.addWidget(self._rigid_check)
        # Without the stretch the checkbox label and "tolerance" run together and
        # read as one sentence.
        rigid_row.addStretch(1)
        tol_label = QLabel("tolerance")
        tol_label.setToolTip("0 = strict even array; 1 = keep picks unchanged.")
        rigid_row.addWidget(tol_label)
        self._rigid_tol = QDoubleSpinBox()
        self._rigid_tol.setRange(0.0, 1.0)
        self._rigid_tol.setSingleStep(0.05)
        self._rigid_tol.setValue(0.25)
        self._rigid_tol.setFixedWidth(88)
        self._rigid_tol.setToolTip("0 = strict even array; 1 = keep picks unchanged.")
        rigid_row.addWidget(self._rigid_tol)
        probe_layout.addLayout(rigid_row)
        layout.addWidget(probe_box)
        layout.addSpacing(10)

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
            "Which atlas names the regions - in the 3D views, the hover readout and "
            "the section-series outlines and region list. These CCFv3-aligned "
            "atlases cover the same volume as the registration atlas, so probe "
            "coordinates are identical and only the naming differs.\n"
            "Chon/Kim carries Franklin-Paxinos labels (M1, S1BF, 4V) where Allen uses "
            "its own (MOp, SSp-bfd, V4), so pick it to get Paxinos region names out.\n"
            "'Same as registration' uses the project atlas.\n"
            "Note: kim_mouse_isotropic_20um samples the same volume at 20 µm and "
            "its annotation sits ~102 µm posterior of the 25 µm release, so a label "
            "within a section of a boundary may differ."
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

        napari_btn = QPushButton("3D view")
        napari_btn.setToolTip(
            "Open the probes and atlas in a separate napari 3D window. For a file "
            "you can share, use Export with the '3D view as interactive HTML' format."
        )
        napari_btn.clicked.connect(self._view_napari3d)
        viz_layout.addWidget(napari_btn)
        layout.addWidget(viz_box)
        layout.addSpacing(10)

        export_box = QGroupBox("Export")
        export_layout = QVBoxLayout(export_box)
        export_layout.addWidget(
            _muted(
                "Writes the registered result to a file - coordinates, or the 3D "
                "view as a shareable page. Pick the format, then Export."
            )
        )

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        #: Paxinos is disabled for pkl, so the tick is remembered across a
        #: format switch instead of being lost.
        self._paxinos_wanted = False
        self._format_combo = QComboBox()
        for label, key in _EXPORT_FORMATS:
            self._format_combo.addItem(label, key)
        self._format_combo.currentIndexChanged.connect(self._on_export_format_changed)
        fmt_row.addWidget(self._format_combo, 1)
        export_layout.addLayout(fmt_row)

        pax_row = QHBoxLayout()
        self._paxinos_check = QCheckBox("Convert to Paxinos stereotaxic coordinates")
        self._paxinos_check.toggled.connect(self._on_paxinos_toggled)
        pax_row.addWidget(self._paxinos_check, 1)
        help_btn = QToolButton()
        help_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))
        help_btn.setAutoRaise(True)
        help_btn.setToolTip("What Paxinos conversion means, and how much to trust it")
        help_btn.clicked.connect(self._show_paxinos_help)
        pax_row.addWidget(help_btn)
        export_layout.addLayout(pax_row)

        align_row = QHBoxLayout()
        align_row.addSpacing(18)  # indented: it only qualifies the checkbox above
        self._paxinos_label = QLabel("Alignment:")
        align_row.addWidget(self._paxinos_label)
        self._paxinos_combo = QComboBox()
        for label, key in _PAXINOS_ALIGNMENT_CHOICES:
            self._paxinos_combo.addItem(label, key)
        align_row.addWidget(self._paxinos_combo, 1)
        export_layout.addLayout(align_row)

        # Only meaningful for the section-series format, so shown only for it
        # rather than sitting there greyed out next to three other formats.
        self._series_box = QWidget()
        series_layout = QVBoxLayout(self._series_box)
        series_layout.setContentsMargins(18, 0, 0, 0)
        self._series_outlines = QCheckBox("Atlas outlines as sidecar images")
        self._series_outlines.setChecked(True)
        self._series_outlines.setToolTip(
            "One transparent PNG per section holding the registered region "
            "contours, in that section's own pixel frame."
        )
        self._series_overlays = QCheckBox("Also write sections with outlines burnt in")
        self._series_overlays.setToolTip(
            "A third image per section with the contours drawn on the histology - "
            "convenient to flick through, not editable."
        )
        self._series_svg = QCheckBox("Outlines also as SVG (editable)")
        self._series_svg.setToolTip(
            "One vector path per atlas region, tagged with its acronym and full "
            "name, so the outlines can be restyled and labelled in Illustrator or "
            "Inkscape instead of traced from a picture."
        )
        self._series_regions = QCheckBox("Region list (regions.csv)")
        self._series_regions.setToolTip(
            "Every atlas region appearing in each section, with the acronym and "
            "full name the canvas shows on hover, plus its area in pixels."
        )
        self._series_straighten = QCheckBox("Straighten sections (DeepSlice angle)")
        self._series_straighten.setChecked(True)
        self._series_straighten.setToolTip(
            "Rotate each exported section - and its outline - by the tilt DeepSlice "
            "measured, so the series is continuous to flick through. Presentation "
            "only: the project, the registration and every CCF coordinate are "
            "untouched. Any rotation you set in the Histology tab is already in the "
            "image, and is not applied twice."
        )
        self._series_outlines.toggled.connect(self._series_svg.setEnabled)
        self._series_outlines.toggled.connect(self._series_regions.setEnabled)
        for box in (
            self._series_outlines, self._series_overlays, self._series_svg,
            self._series_regions, self._series_straighten,
        ):
            series_layout.addWidget(box)
        self._series_outlines.toggled.connect(self._series_overlays.setEnabled)
        self._series_overlays.setEnabled(True)
        export_layout.addWidget(self._series_box)

        export_btn = QPushButton("Export\u2026")
        export_btn.clicked.connect(self._export)
        export_layout.addWidget(export_btn)
        self._on_export_format_changed()
        layout.addWidget(export_box)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    def apply_settings(self, settings) -> None:
        """Store AppSettings (atlas folder for lazy load) + rigid-array prefs."""
        self._settings = settings
        try:
            self._rigid_check.setChecked(bool(getattr(settings, "rigid_array_enforce", False)))
            self._rigid_tol.setValue(float(getattr(settings, "rigid_array_tolerance", 0.25)))
        except Exception:  # noqa: BLE001 - widgets always exist post-build; be safe
            pass

    def collect_settings(self, settings) -> None:
        """Write the rigid-array UI state back into ``settings`` for persistence."""
        settings.rigid_array_enforce = bool(self._rigid_check.isChecked())
        settings.rigid_array_tolerance = float(self._rigid_tol.value())

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
        self._status.setText(f"Loading atlas {atlas_id}")
        from atlastrack.gui.workers import load_atlas_worker

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
            self._status.setText(f"Loading region atlas {chosen}")
            from atlastrack.gui.workers import load_atlas_worker

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
            from atlastrack.viz.plotly3d import build_figure, save_html

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
        from atlastrack.registration.pipeline import (
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
        if self._rigid_check.isChecked():
            self._enforce_rigid_arrays()
        return n

    def _enforce_rigid_arrays(self) -> None:
        """Regularize every multi-shank probe to a parallel, evenly-spaced array.

        Applied after re-mapping so it isn't overwritten by the pixel→CCF pass. Only
        shanks with both tip and entry CCF participate; probes with <3 are skipped.
        """
        from atlastrack.probes.fitting import enforce_rigid_arrays

        enforce_rigid_arrays(
            self._state.project, tolerance=float(self._rigid_tol.value())
        )

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
                from atlastrack.project.io import save_project

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

            from atlastrack.viz.napari3d import show_3d_scene

            if self._viewer3d is None or not _viewer_alive(self._viewer3d):
                self._viewer3d = napari.Viewer(title="Registered histology and probe tracks - 3D view")
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

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_format_changed(self, _idx: int = 0) -> None:
        """Paxinos is a CSV-only option, so say so rather than exporting CCF quietly.

        The pkl carries CCF micrometres for downstream tools that expect them; there
        is no Paxinos variant of it, and silently ignoring a ticked box would be the
        worst of the options.
        """
        fmt = self._format_combo.currentData()
        self._series_box.setVisible(fmt == "series")
        csv = fmt == "csv"
        self._paxinos_check.setEnabled(csv)
        self._paxinos_check.setToolTip(
            ""
            if csv
            else "Only the CSV carries stereotaxic coordinates; the other formats "
            "are CCF."
        )
        if csv:
            # Restore what the user asked for: passing through pkl should not quietly
            # cost them the Paxinos choice they had already made.
            self._paxinos_check.setChecked(self._paxinos_wanted)
        else:
            self._paxinos_wanted = self._paxinos_check.isChecked()
            self._paxinos_check.setChecked(False)
        self._on_paxinos_toggled(self._paxinos_check.isChecked())

    def _on_paxinos_toggled(self, checked: bool) -> None:
        if self._paxinos_check.isEnabled():
            self._paxinos_wanted = checked
        self._paxinos_label.setEnabled(checked)
        self._paxinos_combo.setEnabled(checked)

    def _show_paxinos_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Paxinos stereotaxic coordinates")
        box.setTextFormat(Qt.RichText)
        box.setText(_PAXINOS_HELP)
        box.exec_() if hasattr(box, "exec_") else box.exec()

    def _export(self) -> None:
        """One button: pick a destination for the selected format, then write it."""
        fmt = self._format_combo.currentData()
        if fmt == "series":
            self._ensure_display_atlas(self._export_series)
            return
        if fmt == "html":
            self._export_plotly()
            return
        if fmt == "pkl":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export coordinates to pkl", "", "Pickle files (*.pkl);;All files (*)"
            )
            if path:
                self._write_herbs_pkl(path)
            return
        paxinos = self._paxinos_check.isChecked()
        title = "Export Paxinos CSV" if paxinos else "Export per-channel CSV"
        path, _ = QFileDialog.getSaveFileName(
            self, title, "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            self._write_channel_csv(path, paxinos=paxinos)

    def _export_series(self) -> None:
        """Write the section series into a folder the user picks.

        A folder, not a file: this export is a set of images plus a manifest, and
        asking for one filename would only invite a name that is then decorated.
        """
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the section series"
        )
        if not directory:
            return
        try:
            from atlastrack.io.series_export import export_section_series

            project_path = self._state.project_path
            result = export_section_series(
                self._state.project,
                directory,
                atlas=self._display_atlas,
                base_dir=None if project_path is None else Path(project_path).parent,
                write_outlines=self._series_outlines.isChecked(),
                write_overlays=(
                    self._series_outlines.isChecked()
                    and self._series_overlays.isChecked()
                ),
                write_svg=(
                    self._series_outlines.isChecked() and self._series_svg.isChecked()
                ),
                write_regions=(
                    self._series_outlines.isChecked()
                    and self._series_regions.isChecked()
                ),
                straighten=self._series_straighten.isChecked(),
                # The anchorings were measured on the registration atlas's grid;
                # the region atlas may sample the same volume differently.
                source_shape=getattr(
                    getattr(self._state.atlas, "reference", None), "shape", None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "Section series export failed", str(exc))
            return
        if result.sections == 0:
            _error_dialog(
                self, "Nothing to export",
                "No sections found. Detect sections on a slide first.",
            )
            return
        extras = "".join(
            part
            for part, n in (
                (f", {result.svgs} svg", result.svgs),
                (f", {result.regions} region row(s)", result.regions),
            )
            if n
        )
        message = (
            f"Section series: {result.sections} section(s), {result.outlines} "
            f"outline(s){extras} \u2192 {Path(directory).name}"
        )
        if result.skipped_outlines:
            # Naming the count is the difference between a gap and a silent omission.
            reasons = {reason for _idx, reason in result.skipped_outlines}
            message += (
                f"  \u00b7  no outline for {len(result.skipped_outlines)}: "
                + "; ".join(sorted(reasons))
            )
        self._status.setText(message)

    def _write_herbs_pkl(self, path: str) -> None:
        try:
            import numpy as np

            from atlastrack.io.herbs_writer import write_herbs_pkl

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
            self._status.setText(
                f"Coordinates (pkl, {len(all_ccf)} shank(s)) \u2192 {Path(path).name}"
            )
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "pkl export failed", str(exc))

    def _write_channel_csv(self, path: str, *, paxinos: bool) -> None:
        if paxinos:
            self._do_write_channel_csv(path, paxinos=True)
            return
        # The CCF CSV carries region columns, which need the atlas. Load it lazily
        # (as the 3D view does) rather than silently writing coordinates only.
        self._ensure_atlas(lambda: self._do_write_channel_csv(path, paxinos=False))

    def _do_write_channel_csv(self, path: str, *, paxinos: bool) -> None:
        try:
            from atlastrack.probes.channels import (
                export_channel_csv,
                export_paxinos_csv,
            )

            if paxinos:
                align = self._paxinos_combo.currentData()
                n = export_paxinos_csv(self._state.project, path, alignment=align)
                what = f"Paxinos CSV ({align})"
            else:
                atlas = self._state.atlas
                n = export_channel_csv(self._state.project, path, atlas=atlas)
                what = "Per-channel CSV (CCF)"
                if atlas is None:
                    what += ", no atlas so no region columns"
            if n == 0:
                _error_dialog(self, "Nothing to export", "No registered shank coordinates found.")
            else:
                self._status.setText(f"{what}, {n} rows \u2192 {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            _error_dialog(self, "CSV export failed", str(exc))
