"""Ephys alignment tab: load an Open Ephys recording and refine shank depth.

Pick a probe + shank, point at the recording folder, compute its LFP power map,
then open the alignment dialog to drag anchors and store per-channel CCF on the
shank. SpikeInterface (the ``ephys`` extra) is only needed at compute time.
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

if TYPE_CHECKING:
    import napari


class EphysPanelWidget(QWidget):
    """Select a shank, load its Open Ephys LFP, and launch depth alignment."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._lfp_result: dict | None = None
        self._build_ui()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        sel_box = QGroupBox("Probe / shank")
        sel_layout = QVBoxLayout(sel_box)
        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe:"))
        self._probe_combo = QComboBox()
        self._probe_combo.currentIndexChanged.connect(self._refresh_shanks)
        probe_row.addWidget(self._probe_combo, 1)
        sel_layout.addLayout(probe_row)
        shank_row = QHBoxLayout()
        shank_row.addWidget(QLabel("Start on shank:"))
        self._shank_combo = QComboBox()
        # Optional. The recording carries every shank and the alignment dialog gives
        # each one a tab, so this only decides which tab opens first. It read as a
        # required choice when it was labelled just "Shank".
        self._shank_combo.setToolTip(
            "Optional. Every shank is aligned in the same dialog, one tab each - this "
            "only picks which tab opens first."
        )
        shank_row.addWidget(self._shank_combo, 1)
        sel_layout.addLayout(shank_row)
        hint = QLabel("(optional - all shanks are available in the alignment dialog)")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        sel_layout.addWidget(hint)
        refresh_btn = QPushButton("Refresh probe list")
        refresh_btn.clicked.connect(self.refresh_probes)
        sel_layout.addWidget(refresh_btn)
        layout.addWidget(sel_box)

        rec_box = QGroupBox("Recording (Open Ephys)")
        rec_layout = QVBoxLayout(rec_box)
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Record Node folder with Open Ephys binary data")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse_btn)
        rec_layout.addLayout(path_row)

        stream_row = QHBoxLayout()
        stream_row.addWidget(QLabel("Stream:"))
        self._stream_combo = QComboBox()
        self._stream_combo.addItem("Auto (LFP, else derive from AP)")
        stream_row.addWidget(self._stream_combo, 1)
        list_btn = QPushButton("List streams")
        list_btn.clicked.connect(self._list_streams)
        stream_row.addWidget(list_btn)
        rec_layout.addLayout(stream_row)

        # "Seconds to analyse" used to live here. It is gone: the excerpt reader picks
        # windows spread across the recording and rejects the ones dominated by
        # cross-channel artifact, so a single duration was both meaningless and
        # something no user could set well.
        layout.addWidget(rec_box)

        self._compute_btn = QPushButton("Load and compute LFP power")
        self._compute_btn.setFixedHeight(32)
        self._compute_btn.clicked.connect(self._compute)
        layout.addWidget(self._compute_btn)

        # One button, one dialog. There used to be two ("LFP alignment" and "landmark
        # alignment") showing the same track through different halves of the evidence,
        # which left the user to reconcile them by eye.
        self._align_btn = QPushButton("Open alignment…")
        self._align_btn.setFixedHeight(32)
        self._align_btn.setToolTip(
            "Align this probe's shanks to the atlas.\n\n"
            "One tab per shank, each with the LFP power map, spike raster, firing "
            "rate and the atlas region column on a single shared depth axis. Drag "
            "landmarks on the region column to pin a boundary to the feature "
            "transition where it really appears, or shift the whole track.\n\n"
            "Works with only the atlas and a registered probe - a recording adds the "
            "LFP and spike panels but is not required to read the anatomy."
        )
        self._align_btn.clicked.connect(self._open_alignment)
        layout.addWidget(self._align_btn)

        self._status = QLabel("Add and register probes first, then load a recording.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    # -- probe/shank population -----------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh_probes()

    def refresh_probes(self) -> None:
        cur = self._probe_combo.currentIndex()
        self._probe_combo.blockSignals(True)
        self._probe_combo.clear()
        for probe in self._state.project.probes:
            self._probe_combo.addItem(probe.label)
        self._probe_combo.blockSignals(False)
        if 0 <= cur < self._probe_combo.count():
            self._probe_combo.setCurrentIndex(cur)
        self._refresh_shanks()

    def refresh_after_load(self) -> None:
        """Repopulate the probe/shank combos + recording path from the project."""
        self.refresh_probes()
        # Restore the recording path from any shank that already has an alignment.
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                if shank.ephys is not None and shank.ephys.recording_path:
                    self._path_edit.setText(shank.ephys.recording_path)
                    return

    def _refresh_shanks(self) -> None:
        self._shank_combo.clear()
        idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if not (0 <= idx < len(probes)):
            return
        for shank in probes[idx].shanks:
            self._shank_combo.addItem(f"Shank {shank.index}")

    # -- recording -------------------------------------------------------

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Open Ephys recording folder")
        if path:
            self._path_edit.setText(path)
            self._list_streams()

    def _list_streams(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            return
        try:
            from histo_to_ccf.ephys.loader import list_streams

            streams = list_streams(path)
        except Exception as exc:
            self._status.setText(f"Could not list streams: {exc}")
            return
        self._stream_combo.clear()
        self._stream_combo.addItem("Auto (LFP, else derive from AP)")
        for s in streams:
            self._stream_combo.addItem(s)
        self._status.setText(f"Found {len(streams)} stream(s).")

    def _selected_stream(self) -> str | None:
        return None if self._stream_combo.currentIndex() == 0 else self._stream_combo.currentText()

    def _compute(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "No recording", "Select an Open Ephys recording folder.")
            return
        self._compute_btn.setEnabled(False)
        self._status.setText(
            "Reading + filtering the recording and computing the power map "
            "(nothing is cached - this re-reads the recording each time; deriving "
            "LFP from an AP stream is slower)."
        )
        from histo_to_ccf.gui.workers import lfp_power_worker

        # No user-set duration: the excerpt reader spreads windows across the
        # recording and rejects the artifact-dominated ones, which is strictly better
        # than a single number nobody could choose well.
        worker = lfp_power_worker(Path(path), self._selected_stream())
        worker.returned.connect(self._on_computed)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_computed(self, result: dict) -> None:
        self._lfp_result = result
        self._compute_btn.setEnabled(True)
        self._align_btn.setEnabled(True)
        n_ch = len(result.get("depths_um", []))
        derived = " (derived from AP)" if result.get("derived_from_ap") else ""
        self._status.setText(
            f"LFP power ready: {n_ch} channels, stream '{result.get('stream_name')}'"
            f"{derived}  ·  {result.get('epochs_kept', 0)}/{result.get('epochs_total', 0)}"
            f" windows kept ({result.get('seconds_used', 0.0):.0f} s)."
            + (f"  Rejected: {len(result.get('rejected') or [])} as artifact-dominated."
               if result.get("rejected") else "")
            + "  Click 'Open alignment…'."
        )

    def _on_error(self, exc: Exception) -> None:
        self._compute_btn.setEnabled(True)
        self._status.setText(f"Error: {exc}")
        QMessageBox.critical(self, "LFP computation failed", str(exc)[:2000])

    # -- alignment -------------------------------------------------------

    def _open_alignment(self) -> None:
        """Open the alignment for the whole probe - every shank, one dialog.

        Deliberately **not** gated on a computed recording: the atlas region column
        needs only a registered probe, and the LFP/spike panels fill in when one has
        been loaded. Requiring the recording first was what made the two old dialogs
        disagree about whether anything was loaded.
        """
        selection = self._selected_shank()
        if selection is None:
            return
        probe_idx, shank_idx = selection
        probe = self._state.project.probes[probe_idx]
        if not any(s.tip_ccf_um is not None and s.entry_ccf_um is not None
                   for s in probe.shanks):
            QMessageBox.warning(
                self, "Probe not registered",
                f"No shank of '{probe.label}' has a tip/entry in CCF yet, so there is "
                "no track to align to. Register the sections and place the probe first.",
            )
            return
        rec_path = self._path_edit.text().strip() or None

        from histo_to_ccf.gui.widgets.ephys_alignment_panel import (
            EphysProbeAlignmentDialog,
        )

        dlg = EphysProbeAlignmentDialog(
            self._state,
            probe_idx,
            lfp_result=self._lfp_result,
            initial_shank=shank_idx,
            on_applied=lambda: self._on_alignment_applied(probe_idx, rec_path),
            parent=self,
        )
        dlg.exec_()

    def _selected_shank(self) -> tuple[int, int] | None:
        """The chosen (probe, shank), or ``None`` after warning why there isn't one."""
        probe_idx = self._probe_combo.currentIndex()
        shank_idx = self._shank_combo.currentIndex()
        probes = self._state.project.probes
        if not (0 <= probe_idx < len(probes)) or not (
            0 <= shank_idx < len(probes[probe_idx].shanks)
        ):
            QMessageBox.warning(self, "No shank", "Select a probe and shank first.")
            return None
        if self._state.atlas is None:
            QMessageBox.warning(
                self, "Atlas not loaded",
                "Load the atlas (Atlas tab) so region boundaries can be shown.",
            )
            return None
        return probe_idx, shank_idx

    def _on_alignment_applied(self, probe_idx: int, rec_path: str | None) -> None:
        probe = self._state.project.probes[probe_idx]
        aligned = 0
        for shank in probe.shanks:
            if shank.ephys is None:
                continue
            if rec_path:
                shank.ephys.recording_path = rec_path
            if len(shank.ephys.feature_um) > 2:
                aligned += 1
        msg = (
            f"Alignment stored for '{probe.label}': {aligned} of {len(probe.shanks)} "
            "shanks carry landmarks."
        )
        # Auto-save so the alignment persists (mirrors the Register tab). If there's
        # no project path yet, tell the user to Save manually.
        path = self._state.project_path
        if path is not None:
            try:
                from histo_to_ccf.project.io import save_project

                save_project(self._state.project, path)
                msg += f"  ·  saved → {path.name}"
            except Exception as exc:
                msg += f"  ·  auto-save failed: {exc}"
        else:
            msg += "  ·  use Project ▸ Save Project to persist."
        self._status.setText(msg)
