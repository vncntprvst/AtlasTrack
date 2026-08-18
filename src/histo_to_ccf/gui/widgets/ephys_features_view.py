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

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from histo_to_ccf.ephys.regions import band_colours, white_matter_acronyms

if TYPE_CHECKING:
    from histo_to_ccf.ephys.landmarks import Landmarks
    from histo_to_ccf.ephys.penetration import PenetrationProfile
    from histo_to_ccf.ephys.regions import RegionBand

# Grey wash over depths no recording covers, so a blind stretch never looks as
# well constrained as a measured one.
_GAP_BRUSH = (90, 90, 90, 90)
_OVERLAP_PEN = (255, 190, 60)
_RECORDING_COLOURS = [
    (77, 166, 255), (95, 211, 95), (224, 176, 64), (224, 108, 108),
    (180, 140, 255), (110, 210, 205),
]

# How finely the track is sampled when looking regions up. The atlas is 25 µm, so
# 20 µm resolves every boundary it can actually represent; a 5 mm track is then
# ~250 lookups, fast enough to redo whenever the alignment changes.
_REGION_STEP_UM = 20.0
# A band thinner than this fraction of the visible depth span gets no label - it
# would only overprint its neighbours. Re-evaluated on every zoom, so zooming in
# reveals the slivers rather than hiding them for good.
_LABEL_MIN_FRACTION = 0.035
# Reserved height for both bottom axes, so their plot areas end at the same depth
# even though only one of them carries an axis label.
_AXIS_HEIGHT_PX = 46
# Regions are looked up this far beyond both ends of the track. Aligning shifts the
# anatomy along the column, so a region just outside the track can be pulled into
# view - and it must already have been sampled, or it would simply be missing.
# 1.2 mm covers far more shift than any plausible misregistration.
_REGION_MARGIN_UM = 1200.0


def _readable_on(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever stays legible on ``rgb``.

    The Allen palette runs from near-white yellows to saturated pinks, so a single
    text colour is unreadable on roughly half of it - white on yellow especially.
    Rec. 709 luma, with the threshold at mid-grey.
    """
    r, g, b = (float(c) for c in rgb)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (20, 20, 20) if luma > 140.0 else (245, 245, 245)


class _EmptyProfile:
    """Stand-in so the status line can be written before any profile is set."""

    profiles: ClassVar[list] = []


_EMPTY_PROFILE = _EmptyProfile()


def _with_coverage_alpha(psd, img):
    """Make depths that no recording reached transparent instead of dark.

    A stacked map spans the union of its recordings, so between banks - and above the
    top of the shallowest one - there are rows with no measurement. Drawn as a grey
    image those rows read as "the LFP is quiet here", which is a claim about tissue
    nothing recorded. Transparent shows the panel background through them, so the gap
    looks like the absence it is.

    Fully covered maps (every single-recording one) get no alpha channel at all, so
    the common path is unchanged.
    """
    from histo_to_ccf.ephys.features import covered_rows

    covered = covered_rows(psd)
    if covered.size != img.shape[0] or covered.all():
        return img
    rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
    for c in range(3):
        rgba[..., c] = img
    rgba[..., 3] = np.where(covered, 255, 0)[:, None]
    return rgba


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

    #: An end marker was dragged: ``(track_depth_it_stands_for, new_feature_depth)``.
    #: The view does not act on it - the panel turns it into a landmark.
    endMarkerDragged = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._track_length_um = 0.0
        self._profile: PenetrationProfile | None = None
        self._plots: list = []
        self._raster = None
        self._rate_plot = None
        # Atlas regions, held in *track* space (depth below surface along the
        # histology track) and drawn through the current warp, so the ephys panels
        # stay put and the anatomy stretches against them - which is the direction
        # that makes a misalignment visible.
        self._bands: list[RegionBand] = []
        self._band_colours: list[tuple[int, int, int]] = []
        self._white_matter: set[str] = set()
        # A probe-wide colour assignment, so a region looks the same on every shank
        # tab. None means "decide from this shank alone", which is fine standalone.
        self._shared_colours: dict | None = None
        self._region_names: dict[str, str] = {}
        self._landmarks: Landmarks | None = None
        self._extremes_mode = "uniform"
        self._region_items: list = []
        self._overlay_items: list = []
        self._ephys_items: list = []
        self._end_items: list = []
        # Kept so the normalisation can be toggled without re-reading the recording.
        self._lfp_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._lfp_per_freq = True
        self._mode = "lfp"
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

        # **One** ephys panel, toggled between LFP / spikes / firing rate, beside the
        # atlas column. There used to be four: LFP, Spikes, Firing rate and
        # "Recordings". Only the LFP has a data path in the GUI, so three of them were
        # guaranteed to render empty, and "Recordings" showed a multi-recording
        # concept nothing populates. Four panels' worth of width for one panel's worth
        # of data, and three of them silently claiming "no data" rather than "not
        # loaded".
        self._ephys_plot = self._layout_widget.addPlot(row=0, col=0, title="LFP power")
        self._ephys_plot.setLabel("left", "depth below surface", units="µm")
        # Kept as an alias: the ephys panel is the shared-axis master.
        self._raster = self._ephys_plot

        self._regions = self._layout_widget.addPlot(row=0, col=1, title="Atlas regions")
        self._regions.setLabel("bottom", "")
        self._regions.getAxis("bottom").setStyle(showValues=False)
        # No panning or zooming here at all. The column is y-linked to the ephys
        # panel, so nothing is lost - and a ViewBox that accepts drags competes with
        # the landmark handles the user is trying to grab, which is the likeliest
        # reason they would not move.
        self._regions.setMouseEnabled(x=False, y=False)
        self._regions.setMenuEnabled(False)
        self._regions.setXRange(0.0, 1.0, padding=0.0)
        # The ephys panel needs only a little more room than the region column - the
        # LFP map is 300 frequency bins, not a wide image, and the region labels need
        # real width. 3:2 keeps the whole dialog to a sane size.
        grid = self._layout_widget.ci.layout
        grid.setColumnStretchFactor(0, 3)
        grid.setColumnStretchFactor(1, 2)
        self._regions.setMinimumWidth(260)
        # Force both bottom axes to the same reserved height. Otherwise the ephys
        # panel's axis label ("frequency (Hz)") makes its axis taller, and the two
        # plot areas end at different depths - so the same y is a different place in
        # each panel, which is the one thing a shared depth axis must never allow.
        for plot in (self._ephys_plot, self._regions):
            plot.getAxis("bottom").setHeight(_AXIS_HEIGHT_PX)

        self._plots = [self._ephys_plot, self._regions]
        for plot in self._plots:
            # Depth increases downwards, as the probe sits.
            plot.invertY(True)
            plot.showGrid(x=False, y=True, alpha=0.2)
        self._regions.setYLink(self._ephys_plot)
        self._regions.getAxis("left").setStyle(showValues=False)
        # Which region bands are wide enough to label depends on the zoom, so the
        # labels are recomputed whenever the shared axis moves. Zooming in therefore
        # names the thin nuclei instead of hiding them permanently.
        self._ephys_plot.sigYRangeChanged.connect(lambda *_: self._relabel_regions())

        self._status = QLabel("No recordings loaded.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- content ---------------------------------------------------------

    def available_modes(self) -> list[str]:
        """Which displays actually have data, so the UI can offer only those.

        Offering "Spikes" with nothing behind it is what made three empty panels look
        like a broken recording rather than an unloaded one.
        """
        modes = []
        if self._lfp_data is not None:
            modes.append("lfp")
        profile = self._profile
        if profile is not None and any(p.has_spikes for p in profile.profiles):
            modes.extend(["spikes", "rate"])
        return modes

    def display_mode(self) -> str:
        return self._mode

    def set_display_mode(self, mode: str) -> None:
        """Show ``"lfp"``, ``"spikes"`` or ``"rate"`` in the single ephys panel."""
        if mode not in ("lfp", "spikes", "rate"):
            raise ValueError(f"unknown display mode {mode!r}")
        self._mode = mode
        self._redraw_ephys()

    def _redraw_ephys(self) -> None:
        if not self._ok:
            return
        for item in self._ephys_items:
            self._ephys_plot.removeItem(item)
        self._ephys_items = []
        titles = {"lfp": "LFP power", "spikes": "Spikes", "rate": "Firing rate"}
        units = {"lfp": ("frequency", "Hz"), "spikes": ("time", "s"),
                 "rate": ("rate", "Hz")}
        self._ephys_plot.setTitle(titles[self._mode])
        self._ephys_plot.setLabel("bottom", *units[self._mode][:1],
                                  units=units[self._mode][1])
        if self._mode == "lfp":
            self._draw_lfp_image()
        elif self._mode == "spikes":
            self._draw_raster()
        else:
            self._draw_rate()

    def set_profile(self, profile: PenetrationProfile, *, track_length_um: float = 0.0
                    ) -> None:
        """Load a penetration's spike data (may be empty - LFP alone is normal)."""
        self._profile = profile
        if not self._ok:
            self._status.setText(self._summary(profile, track_length_um))
            return
        self._mark_gaps(profile)
        self._status.setText(self._summary(profile, track_length_um))
        self._redraw_ephys()

    def _draw_raster(self) -> None:
        import pyqtgraph as pg

        from histo_to_ccf.ephys.features import raster_points

        profile = self._profile
        if profile is None:
            return
        depth, amp, times = profile.all_spikes()
        if depth.size == 0:
            return
        t, d, a = raster_points(times, depth, amp)
        # Colour by amplitude, clipped at the 95th percentile so a few huge spikes
        # don't flatten the rest of the scale.
        hi = float(np.percentile(np.abs(a), 95)) or 1.0
        shade = np.clip(np.abs(a) / hi, 0.0, 1.0)
        brushes = [pg.mkBrush(60, int(60 + 195 * s), int(255 - 120 * s), 90) for s in shade]
        item = pg.ScatterPlotItem(x=t, y=d, size=2, pen=None, brush=brushes)
        self._ephys_plot.addItem(item)
        self._ephys_items.append(item)

    def _draw_rate(self) -> None:
        import pyqtgraph as pg

        from histo_to_ccf.ephys.features import depth_profiles

        profile = self._profile
        if profile is None:
            return
        for i, rec in enumerate(p for p in profile.profiles if p.has_spikes):
            centres, rate, _amp = depth_profiles(
                rec.spike_depth_um, rec.spike_amplitude, rec.duration_s, bin_um=20.0
            )
            keep = np.isfinite(rate)
            if not keep.any():
                continue
            colour = _RECORDING_COLOURS[i % len(_RECORDING_COLOURS)]
            item = self._ephys_plot.plot(
                rate[keep], centres[keep], pen=pg.mkPen(colour, width=1.5),
                name=rec.label,
            )
            self._ephys_items.append(item)

    def _mark_gaps(self, profile: PenetrationProfile) -> None:
        import pyqtgraph as pg

        # Tracked and removed by hand rather than via plot.clear(): the region column
        # is redrawn on a different schedule from the features, and clearing a plot
        # would take the other's items with it.
        for plot, item in self._overlay_items:
            plot.removeItem(item)
        self._overlay_items = []

        def _add(plot, item) -> None:
            plot.addItem(item)
            self._overlay_items.append((plot, item))

        for lo, hi in profile.gaps_um():
            for plot in self._plots:
                region = pg.LinearRegionItem(
                    values=(lo, hi), orientation="horizontal",
                    brush=pg.mkBrush(*_GAP_BRUSH), movable=False,
                )
                region.setZValue(-10)
                _add(plot, region)
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
                    _add(plot, line)

    # -- LFP power -------------------------------------------------------

    def lfp_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The LFP the panel is showing: ``(depths_below_surface, psd, freqs)``."""
        return self._lfp_data

    def set_lfp_normalisation(self, *, per_freq: bool) -> None:
        """Re-render the stored LFP map with a different normalisation.

        Per-frequency scaling removes the 1/f gradient that otherwise dominates the
        image, which makes the depth-dependent changes - the thing being aligned to -
        far easier to see. Kept as a toggle because the un-normalised map is the
        honest picture of absolute power.
        """
        self._lfp_per_freq = bool(per_freq)
        if self._lfp_data is not None:
            depths, psd, freqs = self._lfp_data
            self.set_lfp(depths, psd, freqs, per_freq=self._lfp_per_freq)

    def set_lfp(self, depths_um, psd, freqs, *, per_freq: bool = True) -> None:
        """Store the depth x frequency LFP power map and show it.

        ``depths_um`` are depths **below the surface**, one per channel, in any order
        - they are sorted here so the image rows run shallow-to-deep whatever order
        the channels came in.
        """
        depths = np.asarray(depths_um, dtype=float).ravel()
        psd_arr = np.asarray(psd, dtype=float)
        freqs_arr = np.asarray(freqs, dtype=float).ravel()
        if depths.size == 0 or psd_arr.ndim != 2 or psd_arr.shape[0] != depths.size:
            return
        order = np.argsort(depths)
        self._lfp_data = (depths[order], psd_arr[order], freqs_arr)
        self._lfp_per_freq = bool(per_freq)
        self._mode = "lfp"
        self._redraw_ephys()
        if self._ok:
            self._status.setText(self._summary(
                self._profile if self._profile is not None else _EMPTY_PROFILE, 0.0
            ))

    def _draw_lfp_image(self) -> None:
        import pyqtgraph as pg

        from histo_to_ccf.ephys.features import power_image

        if self._lfp_data is None:
            return
        depths, psd, freqs = self._lfp_data
        img = power_image(psd, per_freq=self._lfp_per_freq)
        item = pg.ImageItem(_with_coverage_alpha(psd, img))
        top, bottom = float(depths[0]), float(depths[-1])
        width = float(freqs[-1]) if freqs.size else float(psd.shape[1])
        # Rows run shallow->deep and the view is y-inverted, so row 0 lands at the
        # top of the rect, which is the shallowest channel. Getting this backwards
        # flips the map against the region column.
        item.setRect(pg.QtCore.QRectF(0.0, top, width, max(bottom - top, 1.0)))
        self._ephys_plot.addItem(item)
        self._ephys_items.append(item)
        self._ephys_plot.setXRange(0.0, width, padding=0.0)

    def mark_track_ends(self, track_length_um: float) -> None:
        """Mark the brain surface and the shank tip - both **histology claims**.

        The old dialog labelled the top of its axis "surface" when that was really the
        topmost electrode - on LO_07 ProbeA, 921 µm above the actual surface, because
        the electrode column is longer than the insertion. Marking both ends explicitly
        is what stops that reading.

        These are track-space depths, so they are drawn through the current warp along
        with the regions: when the alignment moves the anatomy, the surface moves with
        it. And the surface line on the ephys panel is **draggable**, because "the
        brain starts here" is exactly the kind of claim the LFP can contradict.
        """
        if self._end_items and float(track_length_um) != self._track_length_um:
            # The track changed under us: drop the markers so they are rebuilt at the
            # new length rather than left pointing at the old tip.
            for plot, line in self._end_items:
                plot.removeItem(line)
            self._end_items = []
        self._track_length_um = float(track_length_um)
        self._draw_track_ends()

    def _draw_track_ends(self) -> None:
        """Create the end markers once, then only ever move them.

        Deliberately not recreated on each redraw. A pyqtgraph ``InfiniteLine`` with a
        ``label`` owns a child text item, and destroying/recreating that on every
        region redraw left the label's C++ object deleted while Python still held it -
        a "wrapped C/C++ object has been deleted" crash several test files later.
        """
        if not self._ok:
            return
        if self._end_items:
            self._position_track_ends()
            return
        import pyqtgraph as pg

        # Labels sit away from the left edge, where they used to collide with the
        # depth tick, and away from each other.
        # Both ends are draggable: each is a histology claim the ephys can contradict,
        # and the tip especially, since the dye marks the *physical tip* while the LFP
        # only reaches the lowest electrode above it.
        # Each end is draggable on exactly one panel, decided by what is a *claim* and
        # what is a fact:
        #   surface - draggable on the ephys panel, because "the brain starts here" is
        #             something the LFP can say and the histology can be wrong about;
        #   tip     - draggable on the region column only. Its position on the ephys
        #             panel is fixed geometry (the bottom channel plus the 175 µm
        #             chisel tip), so dragging it there would let the user contradict
        #             the probe's own dimensions.
        marks = [(0.0, "brain surface (drag me)", (120, 220, 255), "ephys", 0.55)]
        if self._track_length_um > 0:
            marks.append(
                (self._track_length_um, "shank tip", (255, 170, 90), "regions", 0.75)
            )
        for plot in self._plots:
            for depth, label, colour, drag_on, where in marks:
                on_ephys = plot is self._ephys_plot
                movable = drag_on == ("ephys" if on_ephys else "regions")
                line = pg.InfiniteLine(
                    pos=float(np.asarray(self._warp(depth)).ravel()[0]), angle=0,
                    movable=movable,
                    pen=pg.mkPen(colour, width=2 if movable else 1,
                                 style=Qt.PenStyle.DashLine),
                    hoverPen=pg.mkPen(255, 220, 90, width=4) if movable else None,
                    label=(label if on_ephys else None),
                    labelOpts={"color": colour, "position": where, "movable": False},
                )
                line.setZValue(8)
                if movable:
                    line.setCursor(Qt.CursorShape.SizeVerCursor)
                    # A bound method, not a lambda: a closure capturing ``self`` is
                    # owned by the line, which is owned by the plot, which is owned by
                    # this view - a reference cycle whose later collection tears down
                    # C++ objects in an order that crashes the interpreter.
                    line.sigPositionChangeFinished.connect(self._on_surface_line_moved)
                line.track_depth_um = float(depth)
                plot.addItem(line)
                self._end_items.append((plot, line))

    def _position_track_ends(self) -> None:
        """Put the markers where the current warp says their track depths land."""
        for _plot, line in self._end_items:
            depth = float(getattr(line, "track_depth_um", 0.0))
            line.setValue(float(np.asarray(self._warp(depth)).ravel()[0]))

    def _on_surface_line_moved(self, line) -> None:
        """Either end marker moved: report it with the track depth it stands for."""
        self.endMarkerDragged.emit(
            float(getattr(line, "track_depth_um", 0.0)), float(line.value())
        )

    # -- atlas regions ---------------------------------------------------

    @property
    def ephys_plot(self):
        """The single ephys panel - landmark handles are mirrored here too."""
        return getattr(self, "_ephys_plot", None)

    @property
    def region_plot(self):
        """The atlas-region panel, so landmark handles can be attached to it.

        The alignment controls live outside this widget (see
        :class:`~histo_to_ccf.gui.widgets.ephys_alignment_panel.EphysAlignmentPanel`),
        which keeps this view usable, and testable, with no alignment state at all.
        """
        return getattr(self, "_regions", None)

    def set_track(self, atlas, tip_ccf_um, entry_ccf_um, *,
                  step_um: float = _REGION_STEP_UM,
                  extra_um: float = _REGION_MARGIN_UM) -> None:
        """Look the atlas up along the shank and draw the region column.

        Sampling runs from the brain surface to the tip, extended by ``extra_um``
        past both when the recordings reach further than the histology track says
        they should - that disagreement is exactly what the alignment is for, so it
        must be visible rather than clipped away.
        """
        from histo_to_ccf.ephys.regions import region_bands, regions_along_track

        self._bands = []
        if atlas is None or tip_ccf_um is None or entry_ccf_um is None:
            self._draw_regions()
            return

        length = float(np.linalg.norm(np.asarray(tip_ccf_um) - np.asarray(entry_ccf_um)))
        top, bottom = -abs(extra_um), length + abs(extra_um)
        if self._profile is not None and self._profile.profiles:
            p_top, p_bottom = self._profile.depth_range_um()
            top, bottom = min(top, p_top), max(bottom, p_bottom)
        if not bottom > top:
            self._draw_regions()
            return

        depths = np.arange(top, bottom + step_um, max(step_um, 1.0))
        hits = regions_along_track(atlas, tip_ccf_um, entry_ccf_um, depths)
        self._bands = region_bands(hits, depths)
        self._white_matter = white_matter_acronyms(
            atlas, {b.acronym for b in self._bands}
        )
        self._band_colours = band_colours(
            self._bands, shared=self._shared_colours, white_matter=self._white_matter
        )
        self._region_names = self._lookup_names(atlas, self._bands)
        # Set the depth range explicitly. Without this the panels kept whatever
        # autorange they had (nothing, with no spike data), so the region column
        # rendered off-screen and only appeared once some later redraw moved the
        # view - which read as "regions don't show until you add a landmark".
        # Enough padding that the end markers - which sit exactly at 0 and at the
        # track length - are inside the view with their labels, not clipped at the edge.
        self._ephys_plot.setYRange(top, bottom, padding=0.06)
        self._draw_regions()

    @staticmethod
    def _lookup_names(atlas, bands) -> dict:
        """Acronym -> full structure name, for labels a reader can actually parse.

        Many brainstem and cerebellar acronyms are not guessable (PRP, PGRNd, chpl),
        so the column shows the name with the acronym after it.
        """
        names: dict[str, str] = {}
        for band in bands:
            if not band.acronym or band.acronym in names:
                continue
            try:
                names[band.acronym] = str(atlas.structures[band.acronym]["name"])
            except Exception:
                names[band.acronym] = ""
        return names

    def set_landmarks(self, landmarks: Landmarks | None, *, mode: str = "uniform") -> None:
        """Redraw the region column through a landmark warp (``None`` = unwarped)."""
        self._landmarks = landmarks
        self._extremes_mode = mode
        self._draw_regions()

    def set_shared_colours(self, mapping: dict | None) -> None:
        """Use a probe-wide region->colour map instead of deciding per shank."""
        self._shared_colours = mapping
        if self._bands:
            self._band_colours = band_colours(
                self._bands, shared=mapping, white_matter=self._white_matter
            )
            self._draw_regions()

    def bands(self) -> list[RegionBand]:
        """The region bands in track space, exposed for assertions and reporting."""
        return list(self._bands)

    def drawn_bands(self) -> list[tuple[str, float, float]]:
        """Where the bands are actually drawn: ``(acronym, top, bottom)`` after warping.

        Reading the rendered positions rather than recomputing them is what makes a
        test able to catch a warp that is applied to the maths but not to the picture.
        """
        out: list[tuple[str, float, float]] = []
        if not self._ok or not self._bands:
            return out
        import pyqtgraph as pg

        labelled = [b for b in self._bands if b.acronym]
        items = [it for it in self._region_items if isinstance(it, pg.LinearRegionItem)]
        for band, item in zip(labelled, items, strict=False):
            lo, hi = item.getRegion()
            out.append((band.acronym, float(lo), float(hi)))
        return out

    def _warp(self, track_um):
        """Track depth -> the depth it is drawn at, through the current landmarks."""
        if self._landmarks is None or self._landmarks.n_user == 0:
            return np.asarray(track_um, dtype=float)
        return self._landmarks.to_feature(track_um, self._extremes_mode)

    def _draw_regions(self) -> None:
        if not self._ok:
            return
        import pyqtgraph as pg

        for item in self._region_items:
            self._regions.removeItem(item)
        self._region_items = []
        if not self._bands:
            return

        edges = self._warp([b.top_um for b in self._bands] + [self._bands[-1].bottom_um])
        for band, colour, top, bottom in zip(
            self._bands, self._band_colours, edges[:-1], edges[1:], strict=False
        ):
            if band.acronym == "":
                continue  # outside the atlas: leave it as background, not a colour
            item = pg.LinearRegionItem(
                values=(float(top), float(bottom)), orientation="horizontal",
                brush=pg.mkBrush(*colour, 210), pen=pg.mkPen(None), movable=False,
            )
            item.setZValue(-20)
            self._regions.addItem(item)
            self._region_items.append(item)
        # The end markers are track-space claims too, so they follow the same warp.
        self._draw_track_ends()
        self._relabel_regions()

    def _relabel_regions(self) -> None:
        """Write acronyms on the bands wide enough to carry one at this zoom."""
        if not self._ok or not self._bands:
            return
        import pyqtgraph as pg

        labels = [it for it in self._region_items if isinstance(it, pg.TextItem)]
        for item in labels:
            self._regions.removeItem(item)
            self._region_items.remove(item)

        lo, hi = self._regions.getViewBox().viewRange()[1]
        span = abs(hi - lo)
        if span <= 0:
            return
        edges = self._warp([b.top_um for b in self._bands] + [self._bands[-1].bottom_um])
        for band, colour, top, bottom in zip(
            self._bands, self._band_colours, edges[:-1], edges[1:], strict=False
        ):
            if not band.acronym or abs(bottom - top) < _LABEL_MIN_FRACTION * span:
                continue
            mid = 0.5 * (top + bottom)
            if not lo <= mid <= hi:
                continue
            name = self._region_names.get(band.acronym, "")
            caption = f"{name} ({band.acronym})" if name else band.acronym
            text = pg.TextItem(caption, color=_readable_on(colour), anchor=(0, 0.5))
            text.setPos(0.04, mid)
            text.setZValue(5)
            self._regions.addItem(text)
            self._region_items.append(text)

    # -- reporting -------------------------------------------------------

    def summary_text(self) -> str:
        """The status line, exposed so it can be asserted on."""
        return self._status.text()

    def _summary(self, profile: PenetrationProfile, track_length_um: float) -> str:
        if not profile.profiles:
            # "No recordings loaded" was wrong and confusing when an LFP map was
            # plainly on screen: sorted spikes and LFP arrive by different routes, and
            # the status line was only reporting the former.
            if self._lfp_data is not None:
                n_ch = int(self._lfp_data[0].size)
                return (
                    f"LFP loaded: {n_ch} channels. No sorted spikes for this shank - "
                    "the Show selector lists only what has data."
                )
            return "Nothing loaded yet: the region column is the histology track alone."
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
