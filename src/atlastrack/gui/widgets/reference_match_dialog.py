"""Review the sections matched to a registered project before applying them."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from atlastrack.registration.reference_match import SectionMatch

#: Below this outline overlap a match is left unticked: worth a look first.
GOOD_OVERLAP = 0.93


class ReferenceMatchDialog(QDialog):
    """One row per matched section; the ticked ones are applied."""

    def __init__(self, matches: list[SectionMatch], reference_name: str,
                 n_sections: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Sections matched to {reference_name}")
        self._matches = matches
        layout = QVBoxLayout(self)
        unmatched = n_sections - len(matches)
        intro = QLabel(
            f"{len(matches)} section(s) found their counterpart in {reference_name}"
            + (f"; {unmatched} did not and are left as they are." if unmatched else ".")
            + "\n\nTicked sections take their counterpart's registration (plane, warp and "
            "corrections). Sections marked 'flipped' are flipped left-right to match. "
            "Any hand correction they had is cleared."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(len(matches), 5)
        self._table.setHorizontalHeaderLabels(
            ["Section", "Counterpart", "Outlines overlap", "Next best pairing", "Flipped"]
        )
        self._table.setToolTip(
            "Outlines overlap: how much of the two tissue outlines coincide once "
            "fitted (1 = all). Next best pairing: the outline overlap with any other "
            "counterpart before fitting - close to the first means the pairing was a "
            "near thing."
        )
        for row, m in enumerate(matches):
            item = QTableWidgetItem(str(m.target_index))
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            good = m.fit is not None and m.fit.overlap >= GOOD_OVERLAP
            item.setCheckState(Qt.Checked if good else Qt.Unchecked)
            self._table.setItem(row, 0, item)
            label = m.reference_label if m.reference_label is not None else m.reference_index
            values = (
                str(label),
                f"{m.fit.overlap:.3f}" if m.fit is not None else "-",
                f"{m.shape_overlap:.3f} vs {m.runner_up:.3f}",
                "yes" if m.mirrored else "",
            )
            for col, text in enumerate(values, start=1):
                cell = QTableWidgetItem(text)
                cell.setFlags(Qt.ItemIsEnabled)
                self._table.setItem(row, col, cell)
        self._table.resizeColumnsToContents()
        layout.addWidget(self._table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply ticked")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(620, 520)

    def chosen(self) -> list[SectionMatch]:
        return [
            m for row, m in enumerate(self._matches)
            if self._table.item(row, 0).checkState() == Qt.Checked
        ]
