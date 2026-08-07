"""Depth-resolved ephys feature panels, on a shared, zoomable depth axis.

The panels all plot against **depth below the brain surface** on the y-axis, and
their y-ranges are linked, so a feature seen at one depth in the raster lines up
with the same depth in the LFP map and in the atlas region column. That shared axis
is the whole point: alignment is reading off where features and anatomy disagree.

Replaces the fixed 600 px `QGraphicsScene` of the old dialog, which could not be
zoomed - a problem when the interesting structure is a few tens of µm across a
5 mm track.

Depth increases **downwards** (the surface at the top, the tip at the bottom), which
is how the probe actually sits and how every ephys depth plot in the field is drawn.

pyqtgraph lives only under ``histo_to_ccf.gui``; the maths it draws is all in the
headless :mod:`histo_to_ccf.ephys` package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from histo_to_ccf.ephys.penetration import PenetrationProfile

# Grey wash over depths no recording covers, so a blind stretch never looks as
# well constrained as a measured one.
_GAP_BRUSH = (90, 90, 90, 90)
_OVERLAP_PEN = (255, 190, 60)
_RECORDING_COLOURS = [
    (77, 166, 255), (95, 211, 95), (224, 176, 64), (224, 108, 108),
    (180, 140, 255), (110, 210, 205),
]


def pyqtgraph_available() -> bool:
    """Is pyqtgraph importable? The panels degrade to a message if not."""
    try:
        import pyqtgraph  # noqa: F401
    except ImportError:
        return False
    return True


class EphysFeaturesView(QWidget):
    """Stacked feature panels sharing one depth axis.

    Read-only: this shows what the recordings say, and carries no alignment state.
    Landmark placement is layered on top of it separately, so the view stays usable
    (and testable) on its own.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._profile: PenetrationProfile | None = None
        self._plots: list = []
        self._raster = None
        self._rate_plot = None
        self._ok = pyqtgraph_available()
        self._build_ui()

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        if not self._ok:
            layout.addWidget(
                QLabel(
                    "pyqtgraph is not installed - the ephys feature panels need it.\n"
                    'Install with: pip install "histo-to-ccf[ephys]"'
                )
            )
            self._status = QLabel("")
            layout.addWidget(self._status)
            return

        import pyqtgraph as pg

        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self._layout_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._layout_widget, 1)

        self._raster = self._layout_widget.addPlot(row=0, col=0, title="Spikes")
        self._raster.setLabel("bottom", "time", units="s")
        self._raster.setLabel("left", "depth below surface", units="µm")

        self._rate_plot = self._layout_widget.addPlot(row=0, col=1, title="Firing rate")
        self._rate_plot.setLabel("bottom", "rate", units="Hz")

        self._coverage = self._layout_widget.addPlot(row=0, col=2, title="Recordings")
        self._coverage.setLabel("bottom", "")
        self._coverage.getAxis("bottom").setStyle(showValues=False)

        self._plots = [self._raster, self._rate_plot, self._coverage]
        for plot in self._plots:
            # Depth increases downwards, as the probe sits.
            plot.invertY(True)
            plot.showGrid(x=False, y=True, alpha=0.2)
        # One shared, zoomable depth axis across every panel.
        for plot in self._plots[1:]:
            plot.setYLink(self._raster)
            plot.getAxis("left").setStyle(showValues=False)

        self._status = QLabel("No recordings loaded.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- content ---------------------------------------------------------

    def set_profile(self, profile: PenetrationProfile, *, track_length_um: float = 0.0
                    ) -> None:
        """Draw a penetration's recordings."""
        self._profile = profile
        if not self._ok:
            self._status.setText(self._summary(profile, track_length_um))
            return

        self._draw_raster(profile)
        self._draw_rate(profile)
        self._draw_coverage(profile)
        self._mark_gaps(profile)
        self._status.setText(self._summary(profile, track_length_um))
        top, bottom = profile.depth_range_um()
        if bottom > top:
            margin = 0.02 * (bottom - top)
            self._raster.setYRange(top - margin, bottom + margin)

    def _draw_raster(self, profile: PenetrationProfile) -> None:
        import pyqtgraph as pg

        from histo_to_ccf.ephys.features import raster_points

        self._raster.clear()
        depth, amp, times = profile.all_spikes()
        if depth.size == 0:
            return
        t, d, a = raster_points(times, depth, amp)
        # Colour by amplitude, clipped at the 95th percentile so a few huge spikes
        # don't flatten the rest of the scale.
        hi = float(np.percentile(np.abs(a), 95)) or 1.0
        shade = np.clip(np.abs(a) / hi, 0.0, 1.0)
        brushes = [pg.mkBrush(60, int(60 + 195 * s), int(255 - 120 * s), 90) for s in shade]
        self._raster.addItem(
            pg.ScatterPlotItem(x=t, y=d, size=2, pen=None, brush=brushes)
        )

    def _draw_rate(self, profile: PenetrationProfile) -> None:
        import pyqtgraph as pg

        from histo_to_ccf.ephys.features import depth_profiles

        self._rate_plot.clear()
        for i, rec in enumerate(p for p in profile.profiles if p.has_spikes):
            centres, rate, _amp = depth_profiles(
                rec.spike_depth_um, rec.spike_amplitude, rec.duration_s, bin_um=20.0
            )
            keep = np.isfinite(rate)
            if not keep.any():
                continue
            colour = _RECORDING_COLOURS[i % len(_RECORDING_COLOURS)]
            self._rate_plot.plot(
                rate[keep], centres[keep], pen=pg.mkPen(colour, width=1.5),
                name=rec.label,
            )

    def _draw_coverage(self, profile: PenetrationProfile) -> None:
        import pyqtgraph as pg

        self._coverage.clear()
        for i, rec in enumerate(profile.profiles):
            colour = _RECORDING_COLOURS[i % len(_RECORDING_COLOURS)]
            x = 0.15 + 0.2 * (i % 4)
            self._coverage.addItem(
                pg.PlotDataItem(
                    x=[x, x], y=[rec.span.top_um, rec.span.bottom_um],
                    pen=pg.mkPen(colour, width=8),
                )
            )
            label = pg.TextItem(rec.label, color=colour, anchor=(0, 0.5))
            label.setPos(x + 0.04, 0.5 * (rec.span.top_um + rec.span.bottom_um))
            self._coverage.addItem(label)
        self._coverage.setXRange(0.0, 1.0)

    def _mark_gaps(self, profile: PenetrationProfile) -> None:
        import pyqtgraph as pg

        for lo, hi in profile.gaps_um():
            for plot in self._plots:
                region = pg.LinearRegionItem(
                    values=(lo, hi), orientation="horizontal",
                    brush=pg.mkBrush(*_GAP_BRUSH), movable=False,
                )
                region.setZValue(-10)
                plot.addItem(region)
        for lo, hi, _a, _b in profile.overlaps_um():
            for plot in self._plots:
                for y in (lo, hi):
                    # Qt.PenStyle.DotLine, not Qt.DotLine: pyqtgraph re-exports the
                    # raw binding, where PyQt6's enums are scoped. qtpy's promoted
                    # Qt is imported above, and the scoped form works on both.
                    line = pg.InfiniteLine(
                        pos=y, angle=0,
                        pen=pg.mkPen(_OVERLAP_PEN, width=1, style=Qt.PenStyle.DotLine),
                    )
                    line.setZValue(-5)
                    plot.addItem(line)

    # -- reporting -------------------------------------------------------

    def summary_text(self) -> str:
        """The status line, exposed so it can be asserted on."""
        return self._status.text()

    @staticmethod
    def _summary(profile: PenetrationProfile, track_length_um: float) -> str:
        if not profile.profiles:
            return "No recordings loaded."
        top, bottom = profile.depth_range_um()
        parts = [
            f"{len(profile.profiles)} recording(s), "
            f"{top:.0f}-{bottom:.0f} µm below surface"
        ]
        if track_length_um > 0:
            parts.append(
                f"covering {100 * profile.coverage_fraction(track_length_um):.0f}% "
                f"of the {track_length_um:.0f} µm track"
            )
        gaps = profile.gaps_um()
        if gaps:
            spans = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in gaps)
            parts.append(f"no coverage at {spans} µm (shaded)")
        overlaps = profile.overlaps_um()
        if overlaps:
            parts.append(
                f"{len(overlaps)} overlapping pair(s) - their features should agree "
                "where they meet"
            )
        unsorted_ = [p.label for p in profile.profiles if not p.has_spikes]
        if unsorted_:
            parts.append(f"no sorting for {', '.join(unsorted_)} (LFP only)")
        return "  ·  ".join(parts)
