"""Atlas browser: select atlas, preview coronal slices, assign AP to sections."""
from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.widgets.separators import section_header
from histo_to_ccf.io.ccf_coords import (
    BREGMA_AP_FROM_ORIGIN_UM,
    bregma_ap_from_origin_um,
)
from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    import napari

_QUICK_PICKS = [
    ("Allen CCFv3 25 µm", "allen_mouse_25um"),
    ("Allen CCFv3 100 µm", "allen_mouse_100um"),
    ("CCFv3-BBP Augmented 25 µm", "ccfv3augmented_mouse_25um"),
    ("Chon/Kim Unified 25 µm", "kim_mouse_25um"),
    ("Custom ID", ""),
]

class AtlasBrowserWidget(QWidget):
    """Browse coronal atlas slices and assign AP positions to sections."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer",
        settings=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._settings = settings
        self._matcher: "AtlasMatcherDialog | None" = None
        # Set by app.py so the matcher can sync the section-spacing value too.
        self.ordering_panel = None
        self._build_ui()
        if settings is not None:
            self._apply_settings(settings)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Atlas selector
        layout.addWidget(QLabel("Atlas:"))
        self._atlas_combo = QComboBox()
        for label, _ in _QUICK_PICKS:
            self._atlas_combo.addItem(label)
        self._atlas_combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self._atlas_combo)

        # Custom ID field (visible when "Custom ID" selected)
        self._custom_id = QLineEdit()
        self._custom_id.setPlaceholderText("brainglobe atlas id (e.g. allen_mouse_10um)")
        self._custom_id.setVisible(False)
        layout.addWidget(self._custom_id)

        # Atlas storage folder - atlases download once and are reused from here,
        # which is why a previously-fetched atlas loads almost instantly.
        layout.addWidget(QLabel("Atlas folder:"))
        dir_row = QHBoxLayout()
        self._atlas_dir = QLineEdit(self._default_atlas_dir())
        self._atlas_dir.setToolTip(
            "Where BrainGlobe atlases are downloaded to and loaded from.\n"
            "An atlas already present here is reused (no re-download)."
        )
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_atlas_dir)
        dir_row.addWidget(self._atlas_dir)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        self._load_btn = QPushButton("Load atlas")
        self._load_btn.clicked.connect(self._load_atlas)
        layout.addWidget(self._load_btn)

        self._atlas_status = QLabel("Atlas not loaded")
        self._atlas_status.setWordWrap(True)
        layout.addWidget(self._atlas_status)

        # Users look for Paxinos in this list. It is not here, and its absence on
        # its own reads as an omission rather than a design decision.
        frame_note = QLabel(
            "Registration works in CCF; only CCF-space atlases are listed. "
            "Paxinos coordinates are produced at export - see 3D & Export."
        )
        frame_note.setWordWrap(True)
        frame_note.setStyleSheet("color: palette(mid);")
        layout.addWidget(frame_note)

        # The AP controls are a distinct job from choosing and loading an atlas
        # above, so they get their own heading and a clear gap before it.
        layout.addWidget(section_header("AP assignment", top_margin=22))

        # AP position, shown relative to bregma (bregma = 0, anterior positive).
        ap_row = QHBoxLayout()
        ap_row.addWidget(QLabel("AP from bregma (µm):"))
        self._ap_spin = QDoubleSpinBox()
        self._ap_spin.setRange(-15000.0, BREGMA_AP_FROM_ORIGIN_UM)
        self._ap_spin.setValue(0.0)  # bregma
        self._ap_spin.setSingleStep(25.0)
        self._ap_spin.setToolTip(
            "Antero-posterior level relative to bregma.\n"
            "0 = bregma, negative = posterior, positive = anterior."
        )
        ap_row.addWidget(self._ap_spin)
        layout.addLayout(ap_row)

        # Section selector
        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Assign to section idx:"))
        self._sec_spin = QDoubleSpinBox()
        self._sec_spin.setRange(0, 9999)
        self._sec_spin.setDecimals(0)
        sec_row.addWidget(self._sec_spin)
        layout.addLayout(sec_row)

        assign_btn = QPushButton("Assign AP to section")
        assign_btn.clicked.connect(self._assign_ap)
        layout.addWidget(assign_btn)

        # Below the one-section-at-a-time controls it replaces: the matcher is
        # the fuller way to do the same job, not a prerequisite for it.
        matcher_btn = QPushButton("Open atlas matcher")
        matcher_btn.setToolTip(
            "Side-by-side / overlay tool to match each section to an atlas AP."
        )
        matcher_btn.clicked.connect(self._open_matcher)
        layout.addWidget(matcher_btn)

        self._assign_status = QLabel("")
        layout.addWidget(self._assign_status)
        layout.addStretch()

    # ------------------------------------------------------------------

    def _default_atlas_dir(self) -> str:
        """Initial atlas folder: persisted setting, else the BrainGlobe default."""
        if self._settings is not None:
            saved = getattr(self._settings, "atlas_dir", "")
            if saved:
                return saved
        from histo_to_ccf.config import get_settings

        return str(get_settings().atlas_cache_dir)

    def _browse_atlas_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose atlas storage folder", self._atlas_dir.text()
        )
        if path:
            self._atlas_dir.setText(path)

    # -- bregma ↔ absolute AP -------------------------------------------
    # The resampler indexes the volume with an absolute "distance from the
    # anterior edge" AP. The UI shows bregma-relative AP (anterior positive),
    # so convert on the way in and out.
    def _bregma_ap(self) -> float:
        """Bregma AP for the atlas in use.

        Falls back to the Allen anchor for an atlas we have no measurement for, so the
        spin box still works - but ``_on_atlas_loaded`` says so in the status line, and
        the CSV exports refuse outright rather than shipping a guessed frame.
        """
        return bregma_ap_from_origin_um(self._atlas_name()) or BREGMA_AP_FROM_ORIGIN_UM

    def _atlas_name(self) -> str | None:
        loaded = getattr(self._state.atlas, "atlas_name", None)
        return loaded or getattr(self._state.project.atlas, "name", None)

    def _bregma_to_absolute(self, ap_bregma: float) -> float:
        return self._bregma_ap() - ap_bregma

    def _absolute_to_bregma(self, ap_abs: float) -> float:
        return self._bregma_ap() - ap_abs

    def _on_combo_changed(self, idx: int) -> None:
        label, _ = _QUICK_PICKS[idx]
        self._custom_id.setVisible(label.startswith("Custom"))

    def _current_atlas_id(self) -> str:
        idx = self._atlas_combo.currentIndex()
        _, atlas_id = _QUICK_PICKS[idx]
        if not atlas_id:
            atlas_id = self._custom_id.text().strip()
        return atlas_id

    def _load_atlas(self) -> None:
        # Guard against re-entrancy: a slow load must not let repeated clicks
        # stack multiple atlas workers (each holds a thread; a hung one then
        # blocks process exit). Ignore clicks while one is already running.
        if getattr(self, "_atlas_loading", False):
            return
        atlas_id = self._current_atlas_id()
        if not atlas_id:
            self._atlas_status.setText("Enter an atlas id first.")
            return
        atlas_dir = self._atlas_dir.text().strip() or None
        self._atlas_status.setText(f"Loading {atlas_id}")
        from histo_to_ccf.gui.workers import load_atlas_worker

        self._atlas_loading = True
        self._load_btn.setEnabled(False)
        worker = load_atlas_worker(atlas_id, brainglobe_dir=atlas_dir)
        worker.returned.connect(self._on_atlas_loaded)
        worker.errored.connect(lambda e: self._atlas_status.setText(f"Error: {e}"))
        worker.finished.connect(self._on_atlas_load_finished)
        worker.start()

    def _on_atlas_load_finished(self) -> None:
        self._atlas_loading = False
        self._load_btn.setEnabled(True)

    def _atlas_switch_warning(self, previous: str | None, atlas_id: str) -> str:
        """Warn when the project's section APs were assigned under another atlas.

        Section APs are stored as absolute distance from the atlas's *anterior
        edge*, so they do not carry across atlases whose volumes begin at
        different places: Allen and the BBP-augmented CCFv3 put the same anatomy
        346 µm apart, and the augmented volume is 38 slices longer. Loading a
        different atlas recomputes nothing, so without this the change is silent
        and every stored AP quietly comes to mean something else.
        """
        if not previous or previous == atlas_id:
            return ""
        assigned = sum(
            1
            for slide in self._state.project.slides
            for sec in slide.sections
            if getattr(sec, "ap_source", None) is not None
        )
        if not assigned:
            return ""
        old_anchor = bregma_ap_from_origin_um(previous)
        new_anchor = bregma_ap_from_origin_um(atlas_id)
        shift = ""
        if old_anchor is not None and new_anchor is not None:
            shift = (
                f" The same anatomy sits {new_anchor - old_anchor:+.0f} µm along"
                " the AP axis between these two atlases."
            )
        return (
            f"\n⚠ {assigned} section(s) had their AP assigned under '{previous}'. "
            f"Those values are NOT converted.{shift} Re-assign the section APs "
            "and re-register before exporting against this atlas."
        )

    def _on_atlas_loaded(self, atlas) -> None:
        previous = getattr(self._state.project.atlas, "name", None)
        self._state.atlas = atlas
        atlas_id = self._current_atlas_id()
        self._state.project.atlas.name = atlas_id
        ap_max = atlas.reference.shape[0] * atlas.resolution[0]
        # Bregma-relative range: bregma (0) down to the posterior-most slice.
        self._ap_spin.setRange(
            self._absolute_to_bregma(float(ap_max)), self._bregma_ap()
        )
        location = getattr(atlas, "root_dir", None) or self._atlas_dir.text()
        caveat = ""
        if bregma_ap_from_origin_um(atlas_id) is None:
            caveat = (
                f"\n⚠ No bregma anchor known for {atlas_id}: AP is shown against the "
                "Allen anchor, and Paxinos export is unavailable."
            )
        self._atlas_status.setText(
            f"Loaded {atlas.atlas_name} ({atlas.resolution[0]:.0f} µm)\n"
            f"from {location}{caveat}"
            f"{self._atlas_switch_warning(previous, atlas_id)}"
        )
        if self._settings is not None:
            self._settings.last_atlas_id = atlas_id

    def _apply_settings(self, settings) -> None:
        """Pre-select the combo box to match the last-used atlas."""
        atlas_id = settings.last_atlas_id
        for i, (_, aid) in enumerate(_QUICK_PICKS):
            if aid == atlas_id:
                self._atlas_combo.setCurrentIndex(i)
                return
        # Not in quick-picks → select Custom and fill free-text field.
        last_idx = len(_QUICK_PICKS) - 1
        self._atlas_combo.setCurrentIndex(last_idx)
        self._custom_id.setText(atlas_id)
        self._custom_id.setVisible(True)

    def collect_settings(self, settings) -> None:
        """Write the current atlas selection back into settings."""
        atlas_id = self._current_atlas_id()
        if atlas_id:
            settings.last_atlas_id = atlas_id
        if hasattr(settings, "atlas_dir"):
            settings.atlas_dir = self._atlas_dir.text().strip()

    def _open_matcher(self) -> None:
        if self._state.atlas is None:
            self._assign_status.setText("Load an atlas first.")
            return
        if self._state.active_slide_idx is None:
            self._assign_status.setText("Load a slide with sections first.")
            return
        from histo_to_ccf.gui import crashlog
        from histo_to_ccf.gui.widgets.atlas_matcher import AtlasMatcherDialog

        crashlog.note("opening the Atlas matcher")
        # Keep a reference so the non-modal dialog is not garbage-collected.
        # Pass this browser so the matcher seeds its AP/spacing from the tab on
        # open and writes them back on close (sync with the Atlas tab).
        self._matcher = AtlasMatcherDialog(self._state, browser=self, parent=self)
        self._matcher.show()
        self._matcher.raise_()

    # -- matcher <-> tab sync accessors ---------------------------------

    def current_ap_bregma(self) -> float:
        return self._ap_spin.value()

    def set_ap_bregma(self, value: float) -> None:
        self._ap_spin.setValue(value)

    # -- reload ----------------------------------------------------------

    def _active_slide(self):
        idx = self._state.active_slide_idx
        if idx is None or idx >= len(self._state.project.slides):
            return None
        return self._state.project.slides[idx]

    def _select_atlas_id(self, atlas_id: str) -> None:
        """Point the combo (or Custom field) at ``atlas_id``."""
        for i, (_, aid) in enumerate(_QUICK_PICKS):
            if aid == atlas_id:
                self._atlas_combo.setCurrentIndex(i)
                self._custom_id.setVisible(False)
                return
        self._atlas_combo.setCurrentIndex(len(_QUICK_PICKS) - 1)  # Custom ID
        self._custom_id.setText(atlas_id)
        self._custom_id.setVisible(True)

    def auto_load_atlas(self) -> None:
        """Load the project's atlas in the background (after a project load).

        No-op if no atlas is recorded or the matching atlas is already loaded.
        Reuses :meth:`_load_atlas` (which reads the combo set by
        :meth:`refresh_after_load`), so the worker + status handling is shared.
        """
        atlas_id = self._state.project.atlas.name
        if not atlas_id:
            return
        if getattr(self._state.atlas, "atlas_name", None) == atlas_id:
            return  # already loaded
        self._select_atlas_id(atlas_id)
        self._load_atlas()

    def refresh_after_load(self) -> None:
        """Repopulate the atlas selection + AP fields from the loaded project."""
        atlas_id = self._state.project.atlas.name
        if atlas_id:
            self._select_atlas_id(atlas_id)
        slide = self._active_slide()
        if slide is None:
            return
        # Show the AP of the first assigned section (in AP order) as a starting point.
        sec = next(
            (s for s in sorted(slide.sections, key=lambda s: s.ap_order) if s.plane is not None),
            None,
        )
        if sec is not None:
            self._ap_spin.setValue(self._absolute_to_bregma(sec.plane.ap_um))
            self._sec_spin.setValue(sec.index)

    def _assign_ap(self) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            self._assign_status.setText("No slide loaded.")
            return
        sec_idx = int(self._sec_spin.value())
        slide = self._state.project.slides[slide_idx]
        section = next((s for s in slide.sections if s.index == sec_idx), None)
        if section is None:
            self._assign_status.setText(f"Section {sec_idx} not found.")
            return
        from histo_to_ccf.project.schema import PlaneParams

        ap_abs = self._bregma_to_absolute(self._ap_spin.value())
        if section.plane is not None:
            section.plane = section.plane.model_copy(update={"ap_um": ap_abs})
        else:
            section.plane = PlaneParams(ap_um=ap_abs)
        self._assign_status.setText(
            f"Assigned AP={self._ap_spin.value():.0f} µm (from bregma) to section {sec_idx}"
        )
