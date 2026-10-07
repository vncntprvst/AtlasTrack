"""A plot of every section's AP, pitch and yaw in series order.

A series cut from one block should step evenly front to back at one cutting
angle. Plotted together, a section out of order or tilted unlike the rest stands
out at a glance - which the per-section views cannot show.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qtpy.QtWidgets import QDialog, QLabel, QVBoxLayout

from atlastrack.project.schema import Project
from atlastrack.project.series import series_flags, series_rows


class SeriesCheckDialog(QDialog):
    def __init__(
        self,
        project: Project,
        bregma_ap_um: float,
        on_pick: Callable[[int], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Series check: AP and tilt per section")
        self._on_pick = on_pick
        rows = series_rows(project)
        flags = series_flags(rows)
        layout = QVBoxLayout(self)

        from atlastrack.gui.widgets.ephys_features_view import pyqtgraph_available

        if not rows:
            layout.addWidget(QLabel("No sections yet."))
            return
        if not pyqtgraph_available():
            layout.addWidget(QLabel(
                "The plot needs pyqtgraph, which comes with the ephys extra: "
                'pip install "atlastrack[ephys]".'
            ))
        else:
            layout.addWidget(self._plots(rows, flags, bregma_ap_um), 1)

        if flags:
            by_index = {r.index: r.label for r in rows}
            lines = [
                f"Section {by_index.get(idx, idx)}: " + "; ".join(reasons)
                for idx, reasons in flags.items()
            ]
            text = "Worth a look (red points):\n" + "\n".join(lines)
        else:
            text = "No section breaks the pattern of the series."
        note = QLabel(text + "\n\nClick a point to select that section in the Register tab.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.resize(820, 680)

    def _plots(self, rows, flags, bregma_ap_um: float):
        import pyqtgraph as pg

        x = np.arange(len(rows), dtype=float)
        nan = float("nan")
        ap = [r.ap_um - bregma_ap_um if r.ap_um is not None else nan for r in rows]
        pitch = [r.values.pitch_deg if r.values else nan for r in rows]
        yaw = [r.values.yaw_deg if r.values else nan for r in rows]
        flagged = [r.index in flags for r in rows]

        win = pg.GraphicsLayoutWidget()
        win.setBackground("w")
        ticks = [[(float(i), r.label) for i, r in enumerate(rows)]]
        first = None
        for k, (values, label) in enumerate((
            (ap, "AP from bregma (µm)"), (pitch, "Pitch (°)"), (yaw, "Yaw (°)"),
        )):
            plot = win.addPlot(row=k, col=0)
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setLabel("left", label)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.getAxis("bottom").setTicks(ticks)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            y = np.asarray(values, dtype=float)
            plot.plot(x, y, pen=pg.mkPen((150, 150, 150), width=1.5), connect="finite")
            brushes = [pg.mkBrush(214, 39, 40) if f else pg.mkBrush(31, 119, 180) for f in flagged]
            ok = np.isfinite(y)
            scatter = pg.ScatterPlotItem(
                x=x[ok], y=y[ok], size=9, brush=[b for b, o in zip(brushes, ok, strict=True) if o],
                data=[r.index for r, o in zip(rows, ok, strict=True) if o],
            )
            scatter.sigClicked.connect(self._clicked)
            plot.addItem(scatter)
            if k and ok.any():
                # Tilts: show at least 4° so a few tenths of jitter look as small as
                # they are, not like outliers.
                mid = float(np.median(y[ok]))
                half = max(2.0, float(np.nanmax(np.abs(y[ok] - mid))) * 1.15)
                plot.setYRange(mid - half, mid + half, padding=0)
            if k == 2:
                plot.setLabel("bottom", "Section, in series order")
        return win

    def _clicked(self, _scatter, points, *_args) -> None:
        if self._on_pick is not None and len(points):
            self._on_pick(int(points[0].data()))
