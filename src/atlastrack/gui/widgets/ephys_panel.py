"""Ephys alignment tab: load an Open Ephys recording and refine shank depth.

Pick a probe + shank, point at the recording folder, compute its LFP power map,
then open the alignment dialog to drag anchors and store per-channel CCF on the
shank. SpikeInterface (the ``ephys`` extra) is only needed at compute time.
"""
from __future__ import annotations

import contextlib
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

from atlastrack.gui.workflow import WorkflowState

if TYPE_CHECKING:
    import napari

    from atlastrack.gui.widgets.ephys_discovery_dialog import EphysDiscoveryDialog


#: Combo entries that are not probe maps: no map, and the file browser.
MAP_FROM_RECORDING = "From the recording (Open Ephys / SpikeGLX)"
MAP_CHOOSE_FILE = "Choose a map file…"


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
        # Features reloaded from a saved .npz, keyed by shank index. Never carries
        # landmarks - see ephys.export.load_shank_features.
        self._loaded_features: dict | None = None
        # Per-shank stacks from every recording attached to the probe, keyed by shank
        # index. Set by the multi-recording compute; takes precedence over
        # ``_lfp_result``, which can only ever describe one recording.
        self._stacks: dict | None = None
        self._stack_recordings: list = []
        self._build_ui()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Probe only. There used to be a "Start on shank" selector here, but with
        # Shank 0 preselected there was no way to tell "chose shank 0" from "did not
        # care", and both produced the same result - the dialog's first tab. A control
        # whose two states are indistinguishable and equivalent is just noise; the
        # tabs in the alignment dialog are the shank selector.
        sel_box = QGroupBox("Probe")
        sel_layout = QVBoxLayout(sel_box)
        probe_row = QHBoxLayout()
        # probe_row.addWidget(QLabel("Probe:"))
        self._probe_combo = QComboBox()
        self._probe_combo.currentIndexChanged.connect(
            lambda *_: self.refresh_compute_button()
        )
        probe_row.addWidget(self._probe_combo, 1)
        sel_layout.addLayout(probe_row)
        refresh_btn = QPushButton("Refresh probe list")
        refresh_btn.clicked.connect(self.refresh_probes)
        sel_layout.addWidget(refresh_btn)
        layout.addWidget(sel_box)

        rec_box = QGroupBox("Recording (Open Ephys)")
        rec_layout = QVBoxLayout(rec_box)

        # Aligning a 4.5-5.4 mm track needs several recordings, because one bank spans
        # ~720 µm of shank. Finding them by hand - and remembering which share an
        # insertion - is where this stops being usable, so offer the scan first.
        self._discover_btn = QPushButton("Discover recordings in a session folder…")
        self._discover_btn.setToolTip(
            "Scan a session or subject folder for every Open Ephys recording, group "
            "them by penetration, and show what each shank ends up covered by.\n\n"
            "Everything the files know is read from them: which shanks carry sites, "
            "where those sites sit on the shank, which electrode bank. Insertion "
            "depth is the one thing that has to come from your notes - and only for "
            "penetrations recorded at more than one depth."
        )
        self._discover_btn.clicked.connect(self._open_discovery)
        rec_layout.addWidget(self._discover_btn)

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

        map_row = QHBoxLayout()
        map_row.addWidget(QLabel("Probe map:"))
        self._map_combo = QComboBox()
        self._map_combo.setToolTip(
            "Where channel depths come from. Open Ephys and SpikeGLX store the probe, "
            "so leave this on 'From the recording'. Intan stores none - the wiring "
            "lives in the adapter - so pick the matching probe + adapter, or the RHX "
            "'-probe.xml' for the rig. Without one, features are refused rather than "
            "plotted against channel indices."
        )
        self._last_map_index = 0
        self._fill_map_combo()
        self._map_combo.currentIndexChanged.connect(self._on_map_choice)
        map_row.addWidget(self._map_combo, 1)
        rec_layout.addLayout(map_row)


        # "Seconds to analyse" used to live here. It is gone: the excerpt reader picks
        # windows spread across the recording and rejects the ones dominated by
        # cross-channel artifact, so a single duration was both meaningless and
        # something no user could set well.
        layout.addWidget(rec_box)

        self._compute_btn = QPushButton("Compute features from recording")
        self._compute_btn.setFixedHeight(32)
        self._compute_btn.setToolTip(
            "Read screened excerpts from the recording above and compute the "
            "depth-resolved features. Nothing is cached, so this re-reads the "
            "recording each time; deriving LFP from an AP stream is slower."
        )
        self._compute_btn.clicked.connect(self._compute)
        layout.addWidget(self._compute_btn)

        io_row = QHBoxLayout()
        self._load_btn = QPushButton("Load saved features")
        self._load_btn.setFixedHeight(32)
        self._load_btn.setToolTip(
            "Reload a previously saved depth-features .npz instead of recomputing "
            "from the recording.\n\n"
            "Any landmarks in the file are deliberately NOT loaded here: they encode "
            "an alignment to one particular registration, and reloading them after "
            "you have changed the histology would silently overwrite your work. Load "
            "them from inside the alignment dialog, where you can see what changes."
        )
        self._load_btn.clicked.connect(self._load_features)
        io_row.addWidget(self._load_btn)

        self._save_btn = QPushButton("Save computed features")
        self._save_btn.setFixedHeight(32)
        self._save_btn.setToolTip(
            "Save the features just computed from the recording, without opening the "
            "alignment dialog.\n\n"
            "Measurements only - there are no landmarks at this stage, so reloading "
            "this file can never disturb an alignment."
        )
        self._save_btn.clicked.connect(self._save_features)
        io_row.addWidget(self._save_btn)
        layout.addLayout(io_row)

        layout.addSpacing(14)

        # One button, one dialog. There used to be two ("LFP alignment" and "landmark
        # alignment") showing the same track through different halves of the evidence,
        # which left the user to reconcile them by eye.
        self._align_btn = QPushButton("Open alignment")
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

        self._fit_btn = QPushButton("Fit probe trajectory to ephys…")
        self._fit_btn.setFixedHeight(32)
        self._fit_btn.setToolTip(
            "Search for a rigid move of the whole probe - along-track offset, array "
            "roll and pitch - that puts the atlas boundaries where the LFP says the "
            "transitions are.\n\n"
            "Moves the probe only; the registration is left alone. The result opens "
            "as a before/after preview and nothing is recorded until you accept it. "
            "Each parameter is reported with whether it is identifiable at all, and "
            "with a leave-one-shank-out check - on this dataset most of them are not, "
            "which is the point of showing it."
        )
        self._fit_btn.clicked.connect(self._fit_trajectory)
        layout.addWidget(self._fit_btn)

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
        self.refresh_compute_button()

    def refresh_compute_button(self) -> None:
        """Say which recordings the button will read, before it is pressed.

        Attached recordings take precedence over the single path in the box, and a
        button that silently means something different depending on hidden state is
        how the last mix-up happened - the label has to carry it.
        """
        refs = self._attached_recordings()
        if refs:
            self._compute_btn.setText(
                f"Compute features from {len(refs)} attached recording"
                f"{'s' if len(refs) != 1 else ''}"
            )
            self._compute_btn.setToolTip(
                "Reads every recording attached to this probe and stacks them onto one "
                "depth axis per shank: "
                + ", ".join(r.label or Path(r.path).name for r in refs)
                + ".\n\nThis is what covers shanks a single recording missed. The path "
                "box above is ignored while recordings are attached."
            )
        else:
            self._compute_btn.setText("Compute features from recording")
            self._compute_btn.setToolTip(
                "Read screened excerpts from the recording above and compute the "
                "depth-resolved features. Nothing is cached, so this re-reads the "
                "recording each time; deriving LFP from an AP stream is slower.\n\n"
                "Use 'Discover recordings…' to attach every recording of the "
                "penetration instead - one bank covers only ~720 µm of the track."
            )

    def refresh_after_load(self) -> None:
        """Repopulate the probe/shank combos + recording path from the project."""
        self.refresh_probes()
        # Restore the recording path from any shank that already has an alignment.
        for probe in self._state.project.probes:
            for shank in probe.shanks:
                if shank.ephys is not None and shank.ephys.recording_path:
                    self._path_edit.setText(shank.ephys.recording_path)
                    return

    # -- recording -------------------------------------------------------

    def _fill_map_combo(self) -> None:
        """Offer: nothing, a wired probe+adapter, a bare layout, or a file.

        The three are not interchangeable. A wired map includes the adapter, so it
        puts each site on the right channel; a catalog layout only says where the
        sites are, and still assumes the adapter does not permute them.
        """
        from atlastrack.ephys.probemap import BUILTIN_MAPS
        from atlastrack.probes.catalog import CATALOG

        self._map_combo.addItem(MAP_FROM_RECORDING, None)
        for name in sorted(BUILTIN_MAPS):
            self._map_combo.addItem(name, name)
        for name in sorted(CATALOG):
            self._map_combo.addItem(f"{name} - site layout only", name)
        self._map_combo.addItem(MAP_CHOOSE_FILE, MAP_CHOOSE_FILE)

    def _on_map_choice(self, index: int) -> None:
        if self._map_combo.itemData(index) != MAP_CHOOSE_FILE:
            self._last_map_index = index
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a probe map",
            filter="Probe maps (*.xml *.json *.prb *.imro *.csv);;All files (*)",
        )
        if not path:
            # Leave the previous choice in place rather than silently falling back to
            # "from the recording", which would produce a different result.
            self._map_combo.setCurrentIndex(self._last_map_index)
            return
        self._map_combo.insertItem(index, Path(path).name, path)
        self._map_combo.setCurrentIndex(index)

    def _selected_probe_map(self) -> str | None:
        data = self._map_combo.currentData()
        return None if data in (None, MAP_CHOOSE_FILE) else str(data)

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
            from atlastrack.ephys.loader import list_streams

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

    def _attached_recordings(self) -> list:
        """Recordings attached to the selected probe, if any.

        This is the join that makes discovery worth having. Without it the compute
        button reads the single path in the box, so a penetration whose shanks are
        covered by *different* recordings - LO_07 ProbeA: 005 on shank 0, 004 on the
        rest - silently produced features for one shank and blank panels for three.
        """
        idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if 0 <= idx < len(probes):
            return list(probes[idx].recordings or [])
        return []

    def _compute(self) -> None:
        if self._attached_recordings():
            self._compute_multi()
            return
        path = self._path_edit.text().strip()
        if not path:
            QMessageBox.warning(
                self, "No recording",
                "Select an Open Ephys recording folder, or use 'Discover recordings "
                "in a session folder…' to attach every recording of this penetration.",
            )
            return
        self._compute_btn.setEnabled(False)
        self._status.setText(
            "Reading + filtering the recording and computing the power map "
            "(nothing is cached - this re-reads the recording each time; deriving "
            "LFP from an AP stream is slower)."
        )
        from atlastrack.gui.workers import lfp_power_worker

        # No user-set duration: the excerpt reader spreads windows across the
        # recording and rejects the artifact-dominated ones, which is strictly better
        # than a single number nobody could choose well.
        worker = lfp_power_worker(
            Path(path), self._selected_stream(), probe_map=self._selected_probe_map()
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
            f"LFP power ready: {n_ch} channels, stream '{result.get('stream_name')}'"
            f"{derived}  ·  {result.get('epochs_kept', 0)}/{result.get('epochs_total', 0)}"
            f" windows kept ({result.get('seconds_used', 0.0):.0f} s)."
            + (f"  Rejected: {len(result.get('rejected') or [])} as artifact-dominated."
               if result.get("rejected") else "")
            + "  Click 'Open alignment'."
        )

    def _compute_multi(self) -> None:
        """Compute every attached recording and stack them onto one axis per shank."""
        idx = self._probe_combo.currentIndex()
        probe = self._state.project.probes[idx]
        refs = list(probe.recordings)
        missing = [r.label or r.path for r in refs if not r.insertion_depth_um]
        if missing and len(refs) > 1:
            depths = {r.insertion_depth_um for r in refs}
            if len(depths) > 1:
                # Recordings at different insertion depths cannot be placed relative to
                # each other without every depth. Refuse rather than stack them wrongly.
                QMessageBox.warning(
                    self, "Insertion depth missing",
                    "These recordings were taken at more than one insertion depth, so "
                    "each one needs its depth before they can share an axis. Missing "
                    "for: " + ", ".join(missing) + ".\n\nSet them in the discovery "
                    "dialog (Insertion depth column).",
                )
                return
        # The combo is the penetration's map; a recording that already carries its
        # own keeps it, so a per-recording choice is never overwritten from here.
        chosen = self._selected_probe_map()
        if chosen is not None:
            for ref in refs:
                if not getattr(ref, "probe_map", None):
                    ref.probe_map = chosen
        shanks = [s.index for s in probe.shanks] or [0]
        self._compute_btn.setEnabled(False)
        self._status.setText(
            f"Reading {len(refs)} recording(s) for {probe.label} - nothing is cached, "
            "so each one is read from disk."
        )
        from atlastrack.gui.workers import multi_lfp_power_worker

        worker = multi_lfp_power_worker(refs, shanks)
        worker.yielded.connect(
            lambda p: self._status.setText(str(p.get("msg", "")))
        )
        worker.returned.connect(self._on_multi_computed)
        worker.errored.connect(self._on_error)
        worker.start()

    def _on_multi_computed(self, result: dict) -> None:
        self._compute_btn.setEnabled(True)
        self._align_btn.setEnabled(True)
        self._stacks = result.get("stacks") or {}
        self._stack_recordings = result.get("recordings") or []
        self._lfp_result = None  # a single-recording result would now be misleading
        failed = result.get("failed") or []
        if not self._stacks:
            self._status.setText(
                "No shank got any data. "
                + ("; ".join(f"{lab}: {why}" for lab, why in failed)
                   if failed else "Check the attached recordings.")
            )
            return
        refs = {r.label for r in self._stack_recordings}
        lines = [
            f"{len(refs)} recording(s) stacked onto {len(self._stacks)} shank(s): "
            + ", ".join(sorted(refs))
        ]
        for index in sorted(self._stacks):
            stack = self._stacks[index]
            spans = stack.covered_spans_um()
            reach = f"{spans[0][0]:.0f}-{spans[-1][1]:.0f} µm" if spans else "nothing"
            gaps = stack.gaps_um()
            lines.append(
                f"  shank {index}: {reach} from tip, {stack.n_covered} bins"
                + (f", {len(gaps)} gap(s)" if gaps else "")
                + f"  [{stack.describe()}]"
            )
        if failed:
            lines.append("Failed: " + "; ".join(f"{lab}: {why}"
                                                for lab, why in failed))
        lines.append("Click 'Open alignment'.")
        self._status.setText("\n".join(lines))

    def _stack_features(self) -> dict:
        """Per-shank exports built from the stacks, keyed by shank index."""
        idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if not self._stacks or not (0 <= idx < len(probes)):
            return {}
        return {e.shank_index: e for e in self._stack_exports(probes[idx])}

    def _stack_exports(self, probe) -> list:
        """Turn the stacks into :class:`ShankFeatureExport` records.

        The depth-below-surface column is ``track_length - depth_from_tip``, the same
        convention the single-recording path uses, so an alignment made before this
        existed still lines up with the same features.
        """
        import numpy as np

        from atlastrack.ephys.export import ShankFeatureExport

        out = []
        for shank in probe.shanks:
            stack = (self._stacks or {}).get(shank.index)
            if stack is None:
                continue
            track = 0.0
            if shank.tip_ccf_um is not None and shank.entry_ccf_um is not None:
                track = float(np.linalg.norm(
                    np.asarray(shank.tip_ccf_um) - np.asarray(shank.entry_ccf_um)))
            from_tip = np.asarray(stack.depth_from_tip_um, dtype=float)
            out.append(ShankFeatureExport(
                shank_index=shank.index,
                track_length_um=track,
                lfp_psd=np.asarray(stack.psd, dtype=float),
                lfp_freqs_hz=np.asarray(stack.freqs_hz, dtype=float),
                channel_depth_from_tip_um=from_tip,
                channel_depth_below_surface_um=track - from_tip,
            ))
        return out

    def _computed_exports(self, probe) -> list:
        """Per-shank exports from the freshly computed LFP, split by shank id.

        By ``shank_ids``, never by x: a NP2.0 shank has two electrode columns, so
        splitting on unique x over-counts the shanks and takes a single column.
        """
        import numpy as np

        from atlastrack.ephys.export import ShankFeatureExport
        from atlastrack.probes.geometry import SHANK_TIP_LENGTH_UM

        if self._stacks:
            return self._stack_exports(probe)
        result = self._lfp_result or {}
        y = np.asarray(result.get("depths_um", []), dtype=float)
        psd = np.asarray(result.get("psd", []), dtype=float)
        freqs = np.asarray(result.get("freqs", []), dtype=float)
        if y.size == 0 or psd.ndim != 2:
            return []
        from atlastrack.ephys.recordings import channels_for_shank

        out = []
        for shank in probe.shanks:
            mask = channels_for_shank(
                shank.index, result.get("shank_ids"), result.get("x_um")
            )
            if mask is None:
                mask = (np.ones(y.shape, dtype=bool) if shank.index == 0
                        else np.zeros(y.shape, dtype=bool))
            if not mask.any():
                # This shank was not recorded. Skipping it is the honest outcome; the
                # alternative was giving it a copy of another shank's data.
                continue
            track = 0.0
            if shank.tip_ccf_um is not None and shank.entry_ccf_um is not None:
                track = float(np.linalg.norm(
                    np.asarray(shank.tip_ccf_um) - np.asarray(shank.entry_ccf_um)))
            from_tip = y[mask] - y[mask].min() + SHANK_TIP_LENGTH_UM
            out.append(ShankFeatureExport(
                shank_index=shank.index,
                track_length_um=track,
                lfp_psd=psd[mask],
                lfp_freqs_hz=freqs,
                channel_depth_from_tip_um=from_tip,
                channel_depth_below_surface_um=track - from_tip,
            ))
        return out

    def _save_features(self) -> None:
        from atlastrack.ephys.export import default_export_path, save_feature_export

        if self._lfp_result is None and not self._stacks:
            QMessageBox.information(
                self, "Nothing computed",
                "Compute features from a recording first, or load a saved file.",
            )
            return
        probe_idx = self._selected_probe()
        if probe_idx is None:
            return
        probe = self._state.project.probes[probe_idx]
        exports = self._computed_exports(probe)
        if not exports:
            QMessageBox.warning(self, "Nothing to save",
                                "No channels matched this probe's shanks.")
            return

        suggested = default_export_path(
            getattr(self._state, "project_path", None), probe.label
        )
        # Qt silently ignores a suggested path whose directory is missing and reverts
        # to the last-used one, so the folder has to exist before the dialog opens.
        with contextlib.suppress(OSError):
            suggested.parent.mkdir(parents=True, exist_ok=True)
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save computed features", str(suggested), "NumPy archive (*.npz)"
        )
        if not path:
            return
        try:
            written = save_feature_export(path, probe.label, exports)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc)[:2000])
            return
        self._status.setText(
            f"Saved features for {len(exports)} shank(s) → {written.name}"
        )

    def _load_features(self) -> None:
        """Reload a saved depth-features file - measurements only, never landmarks."""
        from atlastrack.ephys.export import default_export_path, load_shank_features

        probe_idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        label = probes[probe_idx].label if 0 <= probe_idx < len(probes) else "probe"
        start = default_export_path(
            getattr(self._state, "project_path", None), label
        ).parent
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load saved depth features", str(start), "NumPy archive (*.npz)"
        )
        if not path:
            return
        try:
            # include_landmarks stays False: see load_shank_features. This is the
            # whole reason the loader has that default.
            shanks, meta = load_shank_features(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc)[:2000])
            return
        self._loaded_features = shanks
        self._lfp_result = None
        n_landmarks = sum(s.get("n_landmarks", 0) for s in meta.get("shanks", []))
        msg = (
            f"Loaded features for {len(shanks)} shank(s) from "
            f"'{meta.get('probe', '?')}'."
        )
        if n_landmarks:
            msg += (
                f"  {n_landmarks} landmark(s) in the file were NOT loaded - load them "
                "from the alignment dialog if you want them."
            )
        self._status.setText(msg + "  Click 'Open alignment'.")

    def _on_error(self, exc: Exception) -> None:
        self._compute_btn.setEnabled(True)
        self._status.setText(f"Error: {exc}")
        QMessageBox.critical(self, "LFP computation failed", str(exc)[:2000])

    # -- discovery -------------------------------------------------------

    def _open_discovery(self) -> EphysDiscoveryDialog:
        from atlastrack.gui.widgets.ephys_discovery_dialog import EphysDiscoveryDialog

        start = self._path_edit.text().strip() or None
        idx0 = self._probe_combo.currentIndex()
        probes0 = self._state.project.probes
        label = probes0[idx0].label if 0 <= idx0 < len(probes0) else None
        dlg = EphysDiscoveryDialog(self._state, self, start_dir=start, probe_label=label)
        dlg.exec()
        # Deliberately not via _selected_probe(): that warns when no atlas is loaded,
        # and discovery reads probe geometry, not anatomy - it is useful before the
        # atlas is anywhere in sight. It also returns an index, not a ProbeSpec.
        idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if 0 <= idx < len(probes) and probes[idx].recordings:
            probe = probes[idx]
            self._status.setText(
                f"{probe.label}: {len(probe.recordings)} recording(s) attached. "
                "Compute features to stack them onto one depth axis per shank."
            )
        self.refresh_compute_button()
        return dlg

    # -- alignment -------------------------------------------------------

    def _open_alignment(self) -> None:
        """Open the alignment for the whole probe - every shank, one dialog.

        Deliberately **not** gated on a computed recording: the atlas region column
        needs only a registered probe, and the LFP/spike panels fill in when one has
        been loaded. Requiring the recording first was what made the two old dialogs
        disagree about whether anything was loaded.
        """
        probe_idx = self._selected_probe()
        if probe_idx is None:
            return
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

        from atlastrack.gui.widgets.ephys_alignment_panel import (
            EphysProbeAlignmentDialog,
        )

        dlg = EphysProbeAlignmentDialog(
            self._state,
            probe_idx,
            lfp_result=self._lfp_result,
            # Stacked features win: they are the only ones that can carry more than
            # one recording, and the alignment is the whole reason for stacking them.
            features=self._stack_features() or self._loaded_features,
            on_applied=lambda: self._on_alignment_applied(probe_idx, rec_path),
            parent=self,
        )
        dlg.exec_()

    def _fit_trajectory(self) -> None:
        """Fit a rigid probe adjustment to the detected LFP boundaries."""
        probe_idx = self._selected_probe()
        if probe_idx is None:
            return
        probe = self._state.project.probes[probe_idx]
        registered = [s for s in probe.shanks
                      if s.tip_ccf_um is not None and s.entry_ccf_um is not None]
        if len(registered) < 2:
            QMessageBox.warning(
                self, "Probe not registered",
                f"'{probe.label}' needs at least two shanks with a tip and entry in "
                "CCF. Roll and pitch are only identifiable from shanks disagreeing "
                "about anatomy, so one shank cannot constrain them.",
            )
            return
        features = self._stack_features() or self._loaded_features
        if not features:
            QMessageBox.information(
                self, "No features",
                "Compute features from the attached recordings, or load a saved "
                ".npz, before fitting. The fit is driven by the boundaries detected "
                "in the LFP - with no features there is nothing to fit to.",
            )
            return

        import numpy as np

        tips = np.array([s.tip_ccf_um for s in registered], dtype=float)
        entries = np.array([s.entry_ccf_um for s in registered], dtype=float)

        cached = self._reusable_fit(probe, features)
        if cached is not None:
            self._on_fit_done(probe_idx, cached)
            return

        self._fit_btn.setEnabled(False)
        self._status.setText("Fitting the probe trajectory to the LFP boundaries…")

        from atlastrack.gui.workers import trajectory_fit_worker

        self._fit_inputs = (tips, entries)
        worker = trajectory_fit_worker(features, tips, entries, self._state.atlas)
        worker.yielded.connect(lambda p: self._status.setText(str(p.get("msg", ""))))
        worker.returned.connect(lambda r: self._on_fit_done(probe_idx, r))
        worker.errored.connect(self._on_fit_failed)
        worker.start()

    def _reusable_fit(self, probe, features) -> dict | None:
        """A saved fit for this probe, if it was computed from these same boundaries.

        The fingerprint guard is the point. Features get recomputed with different
        recordings ticked, and reusing a cached fit that outlived its data would look
        like a fresh answer rather than a stale one - so a mismatch simply refits,
        silently and correctly.
        """
        from atlastrack.probes.trajectory_fit import evidence_from_features
        from atlastrack.probes.trajectory_fit_io import (
            default_fit_path,
            load_fit,
            matches_current,
        )

        path = default_fit_path(
            getattr(self._state, "project_path", None), probe.label
        )
        if not path.exists():
            return None
        try:
            if not matches_current(path, evidence_from_features(features)):
                return None
            fit, evidence, meta = load_fit(path)
        except Exception:
            return None  # an unreadable cache is a reason to refit, not to fail
        self._status.setText(
            f"Reusing the saved fit from {path.name} "
            f"(computed {meta.get('created_at', 'earlier')})."
        )
        return {"fit": fit, "evidence": evidence, "notes": meta.get("notes", ""),
                "from_cache": True}

    def _autosave_fit(self, probe, result: dict) -> None:
        """Write the fit beside the features so the next click is instant."""
        from atlastrack.probes.trajectory_fit_io import default_fit_path, save_fit

        tips, entries = getattr(self, "_fit_inputs", (None, None))
        try:
            path = default_fit_path(
                getattr(self._state, "project_path", None), probe.label
            )
            save_fit(path, result["fit"], result.get("evidence") or {},
                     probe_label=probe.label, notes=str(result.get("notes") or ""),
                     tips=tips, entries=entries)
        except Exception:
            # Not being able to cache must never cost the user the fit itself.
            pass

    def _on_fit_failed(self, exc: Exception) -> None:
        self._fit_btn.setEnabled(True)
        self._status.setText(f"Trajectory fit failed: {exc}")

    def _on_fit_done(self, probe_idx: int, result: dict):
        """Open the preview. Returns the dialog so a test can drive it."""
        self._fit_btn.setEnabled(True)
        fit = (result or {}).get("fit")
        if fit is None:
            self._status.setText(str((result or {}).get("notes", "Nothing to fit.")))
            return None
        self._status.setText(
            f"Fit: offset {fit.offset_um:+.0f} um, roll {fit.roll_deg:+.1f} deg, "
            f"tilt {fit.tilt_deg:+.1f} deg - "
            f"explains {fit.score.explained:.0%} vs {fit.baseline.explained:.0%}. "
            "Review it in the preview before accepting."
        )
        if not result.get("from_cache"):
            self._autosave_fit(self._state.project.probes[probe_idx], result)
        from atlastrack.gui.widgets.trajectory_preview_dialog import (
            TrajectoryPreviewDialog,
        )

        dlg = TrajectoryPreviewDialog(
            self._state, probe_idx, fit,
            evidence=result.get("evidence") or {},
            extra_notes=str(result.get("notes") or ""),
            parent=self,
        )
        dlg.exec()
        if dlg.applied:
            self._status.setText(
                f"Adjustment recorded on {self._state.project.probes[probe_idx].label}. "
                "Save the project to keep it."
            )
        return dlg

    def _selected_probe(self) -> int | None:
        """The chosen probe, or ``None`` after warning why there isn't one."""
        probe_idx = self._probe_combo.currentIndex()
        probes = self._state.project.probes
        if not (0 <= probe_idx < len(probes)):
            QMessageBox.warning(self, "No probe", "Add and select a probe first.")
            return None
        if self._state.atlas is None:
            QMessageBox.warning(
                self, "Atlas not loaded",
                "Load the atlas (Atlas tab) so region boundaries can be shown.",
            )
            return None
        return probe_idx

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
                from atlastrack.project.io import save_project

                save_project(self._state.project, path)
                msg += f"  ·  saved → {path.name}"
            except Exception as exc:
                msg += f"  ·  auto-save failed: {exc}"
        else:
            msg += "  ·  use Project ▸ Save Project to persist."
        self._status.setText(msg)
