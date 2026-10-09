"""Choose each probe's colour, used for its tracks in every view.

One row per probe: a swatch that opens a colour picker, and "Default" to go back to
the palette (see ``atlastrack.probes.colours``). A change applies at once; the
caller redraws the Probes tab and any open 3D window.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from atlastrack.probes.colours import probe_colour

if TYPE_CHECKING:
    from atlastrack.project.schema import Project


def _swatch_style(colour: str) -> str:
    return (
        f"QPushButton {{ background-color: {colour}; border: 1px solid palette(mid); "
        f"min-width: 48px; min-height: 18px; }}"
    )


class ProbeColoursDialog(QDialog):
    """A colour per probe; empty probes list says so."""

    def __init__(
        self,
        project: Project,
        on_changed: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._on_changed = on_changed
        self.setWindowTitle("Probe colours")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Each probe's tracks and electrodes are drawn in its colour in the Probes "
            "tab, the 3D window and the figures.\nBy default each shank takes the next "
            "colour of the palette; a colour chosen here is used for all of a probe's "
            "shanks."
        ))
        grid = QGridLayout()
        layout.addLayout(grid)
        self._swatches: list[QPushButton] = []
        if not project.probes:
            grid.addWidget(QLabel("No probes yet - add one in the Probes tab."), 0, 0)
        for row, probe in enumerate(project.probes):
            grid.addWidget(QLabel(probe.label), row, 0)
            swatch = QPushButton()
            swatch.setToolTip("Choose this probe's colour.")
            swatch.clicked.connect(lambda _=False, r=row: self._choose(r))
            grid.addWidget(swatch, row, 1)
            self._swatches.append(swatch)
            reset = QPushButton("Default")
            reset.setToolTip("Go back to the palette colours, one per shank.")
            reset.clicked.connect(lambda _=False, r=row: self.set_colour(r, None))
            grid.addWidget(reset, row, 2)
        self._show_swatches()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _show_swatches(self) -> None:
        for row, swatch in enumerate(self._swatches):
            colour = probe_colour(self._project, row)
            swatch.setStyleSheet(_swatch_style(colour))
            swatch.setText("" if self._project.probes[row].color else "palette")

    def _pick(self, initial: str) -> str | None:
        """Ask for a colour; None if cancelled. Tests replace this."""
        chosen = QColorDialog.getColor(QColor(initial), self, "Probe colour")
        return chosen.name() if chosen.isValid() else None

    def _choose(self, row: int) -> None:
        chosen = self._pick(probe_colour(self._project, row))
        if chosen is not None:
            self.set_colour(row, chosen)

    def set_colour(self, row: int, colour: str | None) -> None:
        """Give probe ``row`` its own colour (``#rrggbb``), or None for the palette."""
        self._project.probes[row].color = colour
        self._show_swatches()
        if self._on_changed is not None:
            self._on_changed()
