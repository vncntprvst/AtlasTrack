"""Probe type selector widget."""
from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.schema import ProbeSpec, ProbeType, Shank

_PRESETS: dict[str, dict] = {
    "Neuropixels 1.0": {"n_shanks": 1, "shank_pitch_um": 250.0},
    "Neuropixels 2.0 (4-shank)": {"n_shanks": 4, "shank_pitch_um": 250.0},
    "NeuroNexus A1x32-Poly3-10mm-25s-177-OA32LP": {"n_shanks": 1, "shank_pitch_um": 250.0},
    "Custom": {"n_shanks": 1, "shank_pitch_um": 250.0},
}


class ProbePickerWidget(QWidget):
    """Select probe type, label, and number of shanks; adds ProbeSpec to project."""

    def __init__(
        self,
        state: WorkflowState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        # Optional callback fired after a probe is added (wired in app.py to arm
        # tip-marker mode so the user can immediately click a tip point).
        self.on_probe_added: Callable[[], None] | None = None
        # Fired after a probe is renamed, so other panels' probe combos (Probes
        # tip/entry, Ephys) refresh to the new label. Wired in app.py.
        self.on_probes_changed: Callable[[], None] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for name in _PRESETS:
            self._type_combo.addItem(name)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        row1.addWidget(self._type_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Label:"))
        self._label_edit = QLineEdit("probe1")
        row2.addWidget(self._label_edit)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Shanks:"))
        self._n_shanks = QSpinBox()
        self._n_shanks.setRange(1, 8)
        self._n_shanks.setValue(1)
        row3.addWidget(self._n_shanks)
        layout.addLayout(row3)

        add_btn = QPushButton("Add probe")
        add_btn.clicked.connect(self._add_probe)
        layout.addWidget(add_btn)

        # --- Rename an existing probe -------------------------------------
        rename_row = QHBoxLayout()
        rename_row.addWidget(QLabel("Rename:"))
        self._rename_combo = QComboBox()
        self._rename_combo.currentIndexChanged.connect(self._on_rename_selected)
        rename_row.addWidget(self._rename_combo)
        self._rename_edit = QLineEdit()
        self._rename_edit.setPlaceholderText("new label")
        self._rename_edit.returnPressed.connect(self._rename_probe)
        rename_row.addWidget(self._rename_edit)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_probe)
        rename_row.addWidget(rename_btn)
        layout.addLayout(rename_row)

        self._status = QLabel("")
        layout.addWidget(self._status)
        layout.addStretch()
        self._refresh_rename_combo()

    def _on_type_changed(self, name: str) -> None:
        preset = _PRESETS.get(name, {})
        self._n_shanks.setValue(preset.get("n_shanks", 1))

    def _add_probe(self) -> None:
        label = self._label_edit.text().strip() or "probe1"
        type_name = self._type_combo.currentText()
        n = self._n_shanks.value()
        probe_type = ProbeType(name=type_name, n_shanks=n)
        shanks = [Shank(index=i) for i in range(n)]
        probe = ProbeSpec(label=label, type=probe_type, shanks=shanks)
        self._state.project.probes.append(probe)
        self._refresh_rename_combo()
        self._status.setText(f"Added '{label}' ({n} shank{'s' if n > 1 else ''})")
        # Auto-select this probe in the annotation widget and arm tip mode.
        if self.on_probe_added is not None:
            self.on_probe_added()

    # ------------------------------------------------------------------
    # Rename
    # ------------------------------------------------------------------

    def _refresh_rename_combo(self) -> None:
        """Repopulate the rename combo from the project, keeping the selection."""
        cur = self._rename_combo.currentIndex()
        self._rename_combo.blockSignals(True)
        self._rename_combo.clear()
        for probe in self._state.project.probes:
            self._rename_combo.addItem(probe.label)
        self._rename_combo.blockSignals(False)
        if 0 <= cur < self._rename_combo.count():
            self._rename_combo.setCurrentIndex(cur)
        self._on_rename_selected(self._rename_combo.currentIndex())

    def _on_rename_selected(self, idx: int) -> None:
        """Prefill the edit with the selected probe's current label."""
        probes = self._state.project.probes
        if 0 <= idx < len(probes):
            self._rename_edit.setText(probes[idx].label)

    def _rename_probe(self) -> None:
        probes = self._state.project.probes
        idx = self._rename_combo.currentIndex()
        if not (0 <= idx < len(probes)):
            self._status.setText("No probe to rename.")
            return
        new = self._rename_edit.text().strip()
        if not new:
            self._status.setText("Enter a new label first.")
            return
        # Labels double as export keys (per-channel CSV / HERBS group by label),
        # so two probes sharing one would collide - reject duplicates.
        if any(i != idx and p.label == new for i, p in enumerate(probes)):
            self._status.setText(f"Label '{new}' is already used by another probe.")
            return
        old = probes[idx].label
        if new == old:
            return
        probes[idx].label = new
        self._refresh_rename_combo()
        self._rename_combo.setCurrentIndex(idx)
        # Refresh other panels' probe combos (Probes tip/entry, Ephys) to the
        # new label; do this BEFORE the status line so our message isn't clobbered.
        if self.on_probes_changed is not None:
            self.on_probes_changed()
        self._status.setText(f"Renamed '{old}' → '{new}'")

    def current_probe_index(self) -> int | None:
        probes = self._state.project.probes
        return len(probes) - 1 if probes else None

    def refresh_after_load(self) -> None:
        """Reflect the loaded project's probes in the status line."""
        self._refresh_rename_combo()
        probes = self._state.project.probes
        if not probes:
            self._status.setText("")
            return
        names = ", ".join(p.label for p in probes)
        self._status.setText(f"{len(probes)} probe(s) in project: {names}")
