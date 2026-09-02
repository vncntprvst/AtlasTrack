"""Before / after preview of a proposed rigid probe adjustment.

An ephys fit produces three numbers. Numbers are the wrong thing to approve: the
question a user is actually being asked is "does this put the shanks in better
anatomy", and that is a spatial judgement. So this shows the move rather than
describing it, in the two views that can each catch a failure the other cannot.

* **Per-shank columns** - the atlas regions along each shank now and as proposed, side
  by side on one depth axis, with the detected LFP boundaries drawn across both. This
  is where you see whether a boundary the ephys is confident about actually lands on
  the region edge it was matched to.
* **Four projections** - top, side, back and the probe's own plane. A rigid move can
  improve the per-shank columns while walking the array somewhere anatomically absurd,
  and only a spatial view shows that. The probe-plane panel is the one that makes roll
  legible at all, since roll leaves every shank at the same depth and only turns the
  row.

Nothing is written until Apply, and Apply records a
:class:`~histo_to_ccf.project.schema.TrajectoryAdjustment` rather than overwriting the
registered tip and entry - the histology placement is a measurement and this is a
hypothesis about it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.widgets.tooltips import wrap_tooltips
from histo_to_ccf.probes.trajectory_refine import array_axes, transformed_array

if TYPE_CHECKING:
    from histo_to_ccf.gui.workflow import WorkflowState

#: Before is drawn muted, after at full strength - the eye should go to the proposal.
_BEFORE_RGB = (150, 150, 150)
_AFTER_RGB = (90, 200, 255)
_BOUNDARY_RGB = (255, 220, 90)
_OUTLINE_RGB = (120, 120, 130)

#: (title, x label, y label, CCF axis pair, annotation axis to flatten, row-is-x) for
#: the fixed anatomical projections. The probe plane is handled separately because its
#: axes are derived from the array rather than from anatomy.
#:
#: The annotation is ASR - (AP, DV, ML) - so flattening axis 1 leaves (AP, ML), axis 2
#: leaves (AP, DV) and axis 0 leaves (DV, ML). ``row_is_x`` says which way round the
#: resulting contour indices map onto the panel.
_PROJECTIONS = (
    ("Top (AP / ML)", "ML (µm)", "AP (µm)", (1, 0), 1, False),
    ("Side (AP / DV)", "AP (µm)", "DV (µm)", (0, 2), 2, True),
    ("Back (ML / DV)", "ML (µm)", "DV (µm)", (1, 2), 0, False),
)

#: Projected brain silhouettes, keyed by (atlas name, flatten axis). Computing one is
#: a few tens of ms, but the dialog redraws on every toggle and the atlas does not
#: change underneath it.
_OUTLINE_CACHE: dict = {}


def brain_outline(atlas, flatten_axis: int, *, row_is_x: bool,
                  min_points: int = 150) -> list:
    """Outline of the brain silhouette in one projection, as ``[(x, y), ...]`` in µm.

    A silhouette, not a slice: the shanks are drawn as full lines through the volume,
    so the honest backdrop is what the brain occupies anywhere along the viewing axis.
    A single mid-brain slice would show tissue the probe misses and omit tissue it
    passes through.

    Returns an empty list when the atlas has no annotation or scikit-image is absent -
    the outline is orientation, never evidence, so its absence must not stop the
    preview being usable.
    """
    key = (getattr(atlas, "atlas_name", id(atlas)), int(flatten_axis), bool(row_is_x))
    if key in _OUTLINE_CACHE:
        return _OUTLINE_CACHE[key]
    try:
        from skimage.measure import find_contours

        annotation = np.asarray(atlas.annotation)
        resolution = np.asarray(atlas.resolution, dtype=float)
    except Exception:
        _OUTLINE_CACHE[key] = []
        return []

    silhouette = (annotation > 0).any(axis=int(flatten_axis))
    kept_axes = [a for a in range(3) if a != int(flatten_axis)]
    row_um, col_um = resolution[kept_axes[0]], resolution[kept_axes[1]]
    out = []
    for contour in find_contours(silhouette.astype(float), 0.5):
        if contour.shape[0] < min_points:
            continue
        rows, cols = contour[:, 0] * row_um, contour[:, 1] * col_um
        out.append((rows, cols) if row_is_x else (cols, rows))
    _OUTLINE_CACHE[key] = out
    return out


class TrajectoryPreviewDialog(QDialog):
    """Show a proposed adjustment before it is recorded on the probe."""

    def __init__(
        self,
        state: WorkflowState,
        probe_idx: int,
        fit,
        *,
        evidence: dict | None = None,
        extra_notes: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preview trajectory adjustment")
        self.resize(1280, 880)
        self._state = state
        self._probe_idx = int(probe_idx)
        self._fit = fit
        self._evidence = evidence or {}
        self._extra_notes = extra_notes
        self._plot_ok = False
        self._applied = False

        probe = self.probe
        self._tips = np.array(
            [s.tip_ccf_um for s in probe.shanks if s.tip_ccf_um is not None], dtype=float
        )
        self._entries = np.array(
            [s.entry_ccf_um for s in probe.shanks if s.entry_ccf_um is not None],
            dtype=float,
        )
        self._indices = [
            s.index for s in probe.shanks
            if s.tip_ccf_um is not None and s.entry_ccf_um is not None
        ]
        if self._tips.ndim == 2 and len(self._tips) >= 2:
            self._after_tips, self._after_entries = transformed_array(
                self._tips, self._entries,
                offset_um=fit.offset_um, roll_deg=fit.roll_deg, tilt_deg=fit.tilt_deg,
            )
        else:
            # Nothing to move. array_axes needs two shanks to define the row axis, so
            # asking it here would raise rather than report the real problem.
            self._after_tips, self._after_entries = self._tips, self._entries
        self._build_ui()
        # Long explanatory tooltips would otherwise render as one screen-wide
        # line; see histo_to_ccf.gui.widgets.tooltips.
        wrap_tooltips(self)
        self.refresh()

    # -- state -----------------------------------------------------------

    @property
    def probe(self):
        return self._state.project.probes[self._probe_idx]

    @property
    def applied(self) -> bool:
        return self._applied

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._summary = QLabel(self._summary_text())
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._summary)

        split = QSplitter(Qt.Vertical)

        shank_box = QGroupBox("Regions along each shank - now vs proposed")
        shank_layout = QVBoxLayout(shank_box)
        self._shank_plot = self._make_plot()
        shank_layout.addWidget(
            self._shank_plot if self._shank_plot is not None
            else QLabel("Install the ephys extra for the preview plots.")
        )
        split.addWidget(shank_box)

        self._proj_box = QGroupBox(
            "Where the array moves - dashed grey = now, solid blue = proposed; "
            "circles mark the shank TIPS, the lines run up to the entry points"
        )
        proj_box = self._proj_box
        proj_layout = QVBoxLayout(proj_box)
        opts = QHBoxLayout()
        self._outline_check = QCheckBox("Show brain outline")
        self._outline_check.setToolTip(
            "Silhouette of the atlas projected along the viewing axis, for "
            "orientation only. Turning it on zooms out to fit the whole brain; off "
            "zooms in on the tips, where the difference between the two placements "
            "is. Not drawn on the probe-plane panel, whose axes come from the array "
            "rather than from anatomy."
        )
        self._outline_check.toggled.connect(lambda *_: self.refresh())
        opts.addWidget(self._outline_check)
        opts.addStretch()
        proj_layout.addLayout(opts)
        self._proj_plot = self._make_plot()
        proj_layout.addWidget(
            self._proj_plot if self._proj_plot is not None else QLabel("")
        )
        split.addWidget(proj_box)
        split.setSizes([420, 380])
        layout.addWidget(split, 1)

        # The button used to say "Apply adjustment to probe", which reads as "move the
        # probe". It does not move anything: nothing in the app consumes the stored
        # adjustment yet, so the shanks, the 3D view and every export still use the
        # histology placement. A tooltip cannot correct a wrong reading of a button
        # label, so the label itself says it, with a permanent line of text beside it.
        # The other reading of the same fit. Where the shank tips are visible in the
        # sections the probe is where the histology says, and the disagreement has to
        # be somewhere else - so the equivalent registration change belongs next to
        # the probe move, not in a separate place the user has to think to look.
        self._suggestion = QLabel(self.registration_suggestion_text())
        self._suggestion.setWordWrap(True)
        self._suggestion.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._suggestion)

        self._effect_note = QLabel(
            "Recording this does NOT move the probe. The registered tip and entry, the "
            "3D view and every export are unchanged - the fit is stored beside the "
            "histology placement as a hypothesis about it, so the two stay comparable."
        )
        self._effect_note.setWordWrap(True)
        layout.addWidget(self._effect_note)

        row = QHBoxLayout()
        self._save_btn = QPushButton("Save fit…")
        self._save_btn.setFixedHeight(30)
        self._save_btn.setToolTip(
            "Write the fit, its scans, its matched boundaries and the leave-one-out "
            "checks to a .npz, so this preview can be reopened without the ~20 s refit."
        )
        self._save_btn.clicked.connect(self._on_save)
        row.addWidget(self._save_btn)
        row.addStretch()
        self._apply_btn = QPushButton("Record fit on probe (does not move it)")
        self._apply_btn.setFixedHeight(30)
        self._apply_btn.setToolTip(
            "Stores the fitted offset, roll and tilt on the probe as a record, with "
            "the explained fraction and which parameters were identifiable. Nothing "
            "moves: the registered tip and entry are left untouched and no export "
            "reads this yet. Save the project to keep the record."
        )
        self._apply_btn.clicked.connect(self._on_apply)
        row.addWidget(self._apply_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        layout.addLayout(row)

    def _make_plot(self):
        try:
            import pyqtgraph as pg
        except ImportError:
            return None
        pg.setConfigOption("antialias", True)
        self._plot_ok = True
        return pg.GraphicsLayoutWidget()

    def _summary_text(self) -> str:
        fit = self._fit
        head = (
            f"Proposed: offset {fit.offset_um:+.0f} µm, roll {fit.roll_deg:+.1f}°, "
            f"tilt {fit.tilt_deg:+.1f}°"
        )
        body = fit.summary() if hasattr(fit, "summary") else ""
        parts = [head, body]
        if self._extra_notes:
            parts.append(self._extra_notes)
        return "\n".join(p for p in parts if p)

    # -- drawing ---------------------------------------------------------

    @property
    def has_geometry(self) -> bool:
        """Whether there is a registered array to draw at all.

        The Ephys tab refuses to fit a probe with fewer than two registered shanks, so
        this should not arise through the button - but the dialog indexes the tips as a
        (n, 3) array, and an unregistered probe makes that a 1-D empty one. Failing on
        a preview is a poor way to learn the probe was never placed.
        """
        return self._tips.ndim == 2 and len(self._tips) >= 1

    def refresh(self) -> None:
        if not self._plot_ok or not self.has_geometry:
            return
        self._draw_shanks()
        self._draw_projections()

    def _bands_for(self, tip, entry):
        """Region bands along one track, in µm from the tip, with colours."""
        from histo_to_ccf.ephys.regions import (
            band_colours,
            region_bands,
            regions_along_track,
            white_matter_acronyms,
        )

        atlas = getattr(self._state, "atlas", None)
        if atlas is None:
            return []
        track = float(np.linalg.norm(np.asarray(tip) - np.asarray(entry)))
        if track <= 0:
            return []
        below = np.arange(0.0, track, 15.0)
        bands = region_bands(regions_along_track(atlas, tip, entry, below), below)
        wm = white_matter_acronyms(atlas, {b.acronym for b in bands})
        colours = band_colours(bands, white_matter=wm)
        # Depth from the tip, so both columns share the axis the ephys is measured on.
        return [
            (track - b.bottom_um, track - b.top_um, b.acronym, c)
            for b, c in zip(bands, colours, strict=False)
        ]

    def _draw_shanks(self) -> None:
        import pyqtgraph as pg

        self._shank_plot.clear()
        for col, index in enumerate(self._indices):
            plot = self._shank_plot.addPlot(row=0, col=col)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.setMouseEnabled(x=False, y=True)
            plot.invertY(False)
            plot.setTitle(f"shank {index} · notes {index + 1}")
            plot.setXRange(0.0, 2.0, padding=0.02)
            plot.getAxis("bottom").setTicks([[(0.5, "now"), (1.5, "proposed")]])
            if col == 0:
                plot.setLabel("left", "µm from tip")

            for half, (tips, entries) in enumerate(
                ((self._tips, self._entries), (self._after_tips, self._after_entries))
            ):
                for lo, hi, acronym, colour in self._bands_for(
                    tips[col], entries[col]
                ):
                    item = pg.QtWidgets.QGraphicsRectItem(half, lo, 1.0, hi - lo)
                    item.setBrush(pg.mkBrush(*colour, 230))
                    item.setPen(pg.mkPen(None))
                    item.setZValue(-10)
                    plot.addItem(item)
                    if hi - lo > 180.0 and acronym:
                        text = pg.TextItem(acronym, color=(20, 20, 20), anchor=(0.5, 0.5))
                        text.setPos(half + 0.5, 0.5 * (lo + hi))
                        plot.addItem(text)

            ev = self._evidence.get(index)
            if ev is not None:
                for depth in np.asarray(ev.depths_from_tip_um, dtype=float):
                    line = pg.InfiniteLine(
                        pos=float(depth), angle=0, movable=False,
                        pen=pg.mkPen(*_BOUNDARY_RGB, 235, width=2,
                                     style=pg.QtCore.Qt.DashLine),
                    )
                    line.setZValue(20)
                    plot.addItem(line)

    def _probe_plane(self):
        """Coordinates in the array's own plane: along the row, along the track.

        Roll turns the shank row without moving any tip along its track, so it is
        invisible in every anatomical projection that happens to be near-parallel to
        the row. This panel is the one where it always shows.
        """
        u, r, centre = array_axes(self._tips, self._entries)

        def project(points):
            v = np.asarray(points, dtype=float) - centre
            return np.column_stack([v @ r, v @ u])

        return project

    def _pen_pair(self):
        """(before, after) pens. Before is dashed so an overlap stays readable."""
        import pyqtgraph as pg

        return (
            pg.mkPen(*_BEFORE_RGB, 235, width=2, style=pg.QtCore.Qt.DashLine),
            pg.mkPen(*_AFTER_RGB, 245, width=3),
        )

    def _frame_on_tips(self, plot, before_xy, after_xy) -> None:
        """Zoom to the tips, padded by the size of the move.

        Framing the whole 5 mm track instead - which is what the first version did -
        makes a 150 µm offset a sub-pixel change: the panels looked identical and the
        preview answered nothing. The tracks are still drawn and run off the edge; the
        view is put where the difference is.
        """
        pts = np.vstack([before_xy, after_xy])
        move = 0.0
        if before_xy.shape == after_xy.shape:
            move = float(np.max(np.linalg.norm(after_xy - before_xy, axis=1)))
        span = float(np.max(pts.max(0) - pts.min(0)))
        pad = max(250.0, 2.5 * move, 0.35 * span)
        centre = 0.5 * (pts.max(0) + pts.min(0))
        half_x = 0.5 * (pts[:, 0].max() - pts[:, 0].min()) + pad
        half_y = 0.5 * (pts[:, 1].max() - pts[:, 1].min()) + pad
        # Per-axis, not a forced square: the aspect lock already stops the geometry
        # being distorted, and squaring the range on top of it shrank the content into
        # a small patch in the middle of each panel.
        plot.setXRange(centre[0] - half_x, centre[0] + half_x, padding=0.0)
        plot.setYRange(centre[1] - half_y, centre[1] + half_y, padding=0.0)

    def _draw_pair(self, plot, before_t, before_e, after_t, after_e, *,
                   keep_outline: bool = False) -> None:
        import pyqtgraph as pg

        before_pen, after_pen = self._pen_pair()
        for tips, entries, pen, rgb in (
            (before_t, before_e, before_pen, _BEFORE_RGB),
            (after_t, after_e, after_pen, _AFTER_RGB),
        ):
            for tip, entry in zip(tips, entries, strict=False):
                plot.plot([entry[0], tip[0]], [entry[1], tip[1]], pen=pen)
            plot.plot(tips[:, 0], tips[:, 1], pen=None, symbol="o", symbolSize=7,
                      symbolBrush=pg.mkBrush(*rgb, 245))
        if not keep_outline:
            self._frame_on_tips(plot, before_t, after_t)
        else:
            # With the outline on, the whole brain is the point of reference; framing
            # tightly on the tips would crop it away and leave an unexplained arc.
            plot.autoRange()

    def _draw_outline(self, plot, flatten_axis: int, row_is_x: bool) -> None:
        import pyqtgraph as pg

        atlas = getattr(self._state, "atlas", None)
        if atlas is None:
            return
        for xs, ys in brain_outline(atlas, flatten_axis, row_is_x=row_is_x):
            curve = plot.plot(xs, ys, pen=pg.mkPen(*_OUTLINE_RGB, 150, width=1))
            curve.setZValue(-50)

    def _draw_projections(self) -> None:
        self._proj_plot.clear()
        show_outline = self._outline_check.isChecked()
        # One row of four: the tracks are several mm tall and under a millimetre wide,
        # so in a 2x2 grid the aspect lock stretched each panel to tens of mm across
        # and the arrays came out a few pixels high.
        for k, (title, xlabel, ylabel, (ax_x, ax_y), flat, row_is_x) in enumerate(
            _PROJECTIONS
        ):
            plot = self._proj_plot.addPlot(row=0, col=k)
            plot.setMenuEnabled(False)
            plot.setTitle(title)
            plot.setLabel("bottom", xlabel)
            plot.setLabel("left", ylabel)
            plot.setAspectLocked(True)
            if ax_y == 2:  # DV increases downwards in CCF
                plot.invertY(True)
            if show_outline:
                self._draw_outline(plot, flat, row_is_x)
            pick = [ax_x, ax_y]
            self._draw_pair(plot, self._tips[:, pick], self._entries[:, pick],
                            self._after_tips[:, pick], self._after_entries[:, pick],
                            keep_outline=show_outline)

        project = self._probe_plane()
        plot = self._proj_plot.addPlot(row=0, col=len(_PROJECTIONS))
        plot.setMenuEnabled(False)
        plot.setTitle("Probe plane (along row / along track)")
        plot.setLabel("bottom", "along the shank row (µm)")
        plot.setLabel("left", "along the insertion (µm)")
        plot.setAspectLocked(True)
        plot.invertY(True)
        # No outline here: this plane is defined by the array, so a brain silhouette
        # would have to be re-projected into it rather than read off an atlas axis.
        self._draw_pair(plot, project(self._tips), project(self._entries),
                        project(self._after_tips), project(self._after_entries))

    # -- applying --------------------------------------------------------

    def adjustment(self):
        """The record this dialog would write. Separated so it is testable."""
        from datetime import datetime, timezone

        from histo_to_ccf.project.schema import TrajectoryAdjustment

        fit = self._fit
        identifiable = fit.identifiable() if hasattr(fit, "identifiable") else {}
        return TrajectoryAdjustment(
            offset_um=float(fit.offset_um),
            roll_deg=float(fit.roll_deg),
            tilt_deg=float(fit.tilt_deg),
            explained=float(getattr(fit.score, "explained", 0.0)),
            baseline_explained=float(getattr(fit.baseline, "explained", 0.0)),
            identifiable={str(k): bool(v) for k, v in identifiable.items()},
            notes=self._extra_notes or None,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def apply_adjustment(self):
        """Record the adjustment on the probe. Returns it; headless-testable."""
        adj = self.adjustment()
        self.probe.trajectory_adjustment = adj
        self._applied = True
        return adj

    def _on_apply(self) -> None:
        unstable = [k for k, ok in (self._fit.identifiable() or {}).items() if not ok]
        if unstable:
            # Not a block - the user may have other reasons to accept it - but the
            # dialog must not let an unidentifiable number be applied silently.
            answer = QMessageBox.question(
                self, "Some parameters are not identifiable",
                "The fit could not establish " + ", ".join(unstable)
                + " from this data: the scan is flat, rough, or peaks at its own "
                "limit. Applying will record those values anyway.\n\nApply?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        adj = self.apply_adjustment()
        QMessageBox.information(
            self, "Fit recorded",
            f"Recorded on {self.probe.label}: offset {adj.offset_um:+.0f} µm, "
            f"roll {adj.roll_deg:+.1f}°, tilt {adj.tilt_deg:+.1f}°.\n\n"
            "Nothing has moved. The registered tip and entry are unchanged, and no "
            "view or export uses this yet - it is a record of what the ephys implies "
            "about the placement.\n\nSave the project to keep it.",
        )
        self.accept()

    def registration_suggestion(self):
        """The same fit expressed as a registration change. Headless-testable."""
        from histo_to_ccf.probes.registration_suggestion import (
            suggest_registration_change,
        )

        project = getattr(self._state, "project", None)
        return suggest_registration_change(
            self.probe,
            offset_um=float(self._fit.offset_um),
            roll_deg=float(self._fit.roll_deg),
            tilt_deg=float(self._fit.tilt_deg),
            section_spacing_um=getattr(project, "section_spacing_um", None),
            other_probes=getattr(project, "probes", ()) or (),
        )

    def registration_suggestion_text(self) -> str:
        suggestion = self.registration_suggestion()
        return "" if suggestion is None else suggestion.text()

    def save_fit_to(self, path):
        """Write the fit beside the features. Returns the path; headless-testable."""
        from histo_to_ccf.probes.trajectory_fit_io import save_fit

        return save_fit(path, self._fit, self._evidence,
                        probe_label=self.probe.label, notes=self._extra_notes,
                        tips=self._tips, entries=self._entries)

    def _on_save(self) -> None:
        import contextlib

        from histo_to_ccf.probes.trajectory_fit_io import default_fit_path

        suggested = default_fit_path(
            getattr(self._state, "project_path", None), self.probe.label
        )
        # Qt silently ignores a suggested path whose directory is missing.
        with contextlib.suppress(OSError):
            suggested.parent.mkdir(parents=True, exist_ok=True)
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save fit", str(suggested), "NumPy archive (*.npz)"
        )
        if not path:
            return
        try:
            written = self.save_fit_to(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc)[:2000])
            return
        QMessageBox.information(
            self, "Fit saved",
            f"Saved to\n{written}\n\nReopen it from the Ephys tab to skip the refit.",
        )
