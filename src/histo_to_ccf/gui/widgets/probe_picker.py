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

from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Shank

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

        self._status = QLabel("")
        layout.addWidget(self._status)
        layout.addStretch()

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
        self._status.setText(f"Added '{label}' ({n} shank{'s' if n > 1 else ''})")
        # Auto-select this probe in the annotation widget and arm tip mode.
        if self.on_probe_added is not None:
            self.on_probe_added()

    def current_probe_index(self) -> int | None:
        probes = self._state.project.probes
        return len(probes) - 1 if probes else None

    def refresh_after_load(self) -> None:
        """Reflect the loaded project's probes in the status line."""
        probes = self._state.project.probes
        if not probes:
            self._status.setText("")
            return
        names = ", ".join(p.label for p in probes)
        self._status.setText(f"{len(probes)} probe(s) in project: {names}")
