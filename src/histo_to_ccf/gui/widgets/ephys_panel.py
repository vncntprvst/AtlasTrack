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
    QDoubleSpinBox,
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
        viewer: "napari.Viewer",
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
        shank_row.addWidget(QLabel("Shank:"))
        self._shank_combo = QComboBox()
        shank_row.addWidget(self._shank_combo, 1)
        sel_layout.addLayout(shank_row)
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

        secs_row = QHBoxLayout()
        secs_row.addWidget(QLabel("Seconds to analyse:"))
        self._secs_spin = QDoubleSpinBox()
        self._secs_spin.setRange(5.0, 600.0)
        self._secs_spin.setValue(60.0)
        self._secs_spin.setSingleStep(10.0)
        secs_row.addWidget(self._secs_spin)
        secs_row.addStretch()
        rec_layout.addLayout(secs_row)
        layout.addWidget(rec_box)

        self._compute_btn = QPushButton("Load and compute LFP power")
        self._compute_btn.setFixedHeight(32)
        self._compute_btn.clicked.connect(self._compute)
        layout.addWidget(self._compute_btn)

        self._align_btn = QPushButton("Open LFP alignment")
        self._align_btn.setEnabled(False)
        self._align_btn.clicked.connect(self._open_alignment)
        layout.addWidget(self._align_btn)

        self._landmark_btn = QPushButton("Open landmark alignment")
        self._landmark_btn.setToolTip(
            "Depth-resolved feature panels beside the atlas region column, on one "
            "shared depth axis, with draggable landmarks.\n"
            "Needs only the atlas and a registered shank - no recording - so the "
            "anatomy along the track can be read straight away."
        )
        self._landmark_btn.clicked.connect(self._open_landmark_alignment)
        layout.addWidget(self._landmark_btn)

        self._status = QLabel("Add and register probes first, then load a recording.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    # -- probe/shank population -----------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 (Qt signature)
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
        except Exception as exc:  # noqa: BLE001
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

        worker = lfp_power_worker(
            Path(path),
            self._selected_stream(),
            max_seconds=self._secs_spin.value(),
        )
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
            f"LFP power ready: {n_ch} channels, stream '{result.get('stream_name')}'{derived}. "
            "Click 'Open alignment'."
        )

    def _on_error(self, exc: Exception) -> None:
        self._compute_btn.setEnabled(True)
        self._status.setText(f"Error: {exc}")
        QMessageBox.critical(self, "LFP computation failed", str(exc)[:2000])

    # -- alignment -------------------------------------------------------

    def _open_alignment(self) -> None:
        if self._lfp_result is None:
            return
        selection = self._selected_shank()
        if selection is None:
            return
        probe_idx, shank_idx = selection
        # Persist which recording this shank's alignment came from.
        rec_path = self._path_edit.text().strip() or None

        from histo_to_ccf.gui.widgets.ephys_align_dialog import EphysAlignmentDialog

        dlg = EphysAlignmentDialog(
            self._state,
            probe_idx,
            shank_idx,
            self._lfp_result,
            on_applied=lambda: self._on_applied(probe_idx, shank_idx, rec_path),
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

    def _open_landmark_alignment(self) -> None:
        selection = self._selected_shank()
        if selection is None:
            return
        probe_idx, shank_idx = selection
        shank = self._state.project.probes[probe_idx].shanks[shank_idx]
        if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
            QMessageBox.warning(
                self, "Shank not registered",
                "This shank has no tip/entry in CCF yet, so there is no track to "
                "align to. Register the sections and place the probe first.",
            )
            return

        from histo_to_ccf.gui.widgets.ephys_alignment_panel import EphysLandmarkDialog

        dlg = EphysLandmarkDialog(
            self._state, probe_idx, shank_idx,
            on_applied=lambda: self._on_landmarks_applied(probe_idx, shank_idx),
            parent=self,
        )
        dlg.exec_()

    def _on_landmarks_applied(self, probe_idx: int, shank_idx: int) -> None:
        shank = self._state.project.probes[probe_idx].shanks[shank_idx]
        n = max(len(shank.ephys.feature_um) - 2, 0) if shank.ephys else 0
        self._status.setText(
            f"{n} landmark(s) stored on shank {shank_idx}. Save the project to keep them."
        )

    def _on_applied(self, probe_idx: int, shank_idx: int, rec_path: str | None) -> None:
        shank = self._state.project.probes[probe_idx].shanks[shank_idx]
        if shank.ephys is not None and rec_path:
            shank.ephys.recording_path = rec_path
        n = len(shank.ephys.channel_ccf_um) if shank.ephys else 0
        msg = f"Alignment applied: {n} channels mapped to CCF on shank {shank_idx}."
        # Auto-save so the per-channel CCF + anchors persist (mirrors the Register
        # tab). If there's no project path yet, tell the user to Save manually.
        path = self._state.project_path
        if path is not None:
            try:
                from histo_to_ccf.project.io import save_project

                save_project(self._state.project, path)
                msg += f"  ·  saved → {path.name}"
            except Exception as exc:  # noqa: BLE001
                msg += f"  ·  auto-save failed: {exc}"
        else:
            msg += "  ·  use Project ▸ Save Project to persist."
        msg += "  View in napari 3D to see the channels."
        self._status.setText(msg)
