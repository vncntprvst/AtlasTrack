"""Point at a session folder, see what each shank is actually covered by.

The decision this dialog exists to support is not "which files shall I load" but
**"do I have a landmark on this shank?"** - so a list of paths is the wrong shape. A
recording spanning the bottom 705 µm of a LO_07 shank looks identical to a useful one
in a file list, and is worthless: it sits entirely inside the gigantocellular
reticular formation, where there is no boundary to align to. Ticking a second
recording taken 400 µm shallower extends the span into the medial vestibular nucleus
and makes the shank alignable.

That difference is invisible in text and obvious in a picture, so the centre of this
dialog is a coverage plot: one row per shank, µm from the tip, the histology region
bands drawn behind and the recorded spans over them.

Everything derivable is derived (see :mod:`atlastrack.ephys.discovery`); the table
asks only for what is left, which in practice is an insertion depth for penetrations
recorded at more than one depth.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from atlastrack.ephys.discovery import (
    Penetration,
    coverage_from_tip,
    derive_electrode_range,
    grouping_warnings,
)
from atlastrack.gui.widgets.tooltips import wrap_tooltips
from atlastrack.gui.workflow import WorkflowState

if TYPE_CHECKING:
    from atlastrack.ephys.discovery import RecordingCandidate

_COLUMNS = ("Use", "Recording", "Coverage", "Insertion depth (µm)", "From", "Check")
_ROW_H_PX = 62
_SPAN_COLOUR = (250, 250, 250)


class EphysDiscoveryDialog(QDialog):
    """Scan a folder for recordings, review their coverage, add them to a probe."""

    def __init__(
        self,
        state: WorkflowState,
        parent: QWidget | None = None,
        *,
        start_dir: str | None = None,
        probe_label: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Discover ephys recordings")
        self.resize(1080, 680)
        self._state = state
        self._probe_label = probe_label
        self._penetrations: list[Penetration] = []
        self._worker = None
        self._plot_ok = False
        self._build_ui()
        # Long explanatory tooltips would otherwise render as one screen-wide
        # line; see atlastrack.gui.widgets.tooltips.
        wrap_tooltips(self)
        if start_dir:
            self._folder_edit.setText(start_dir)

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        src = QGroupBox("Where to look")
        src_form = QVBoxLayout(src)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Session folder:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(
            "A subject or session folder; every Open Ephys record node beneath it is read"
        )
        folder_row.addWidget(self._folder_edit, 1)
        folder_btn = QPushButton("Browse")
        folder_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(folder_btn)
        src_form.addLayout(folder_row)

        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Metadata table:"))
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText(
            "Optional CSV of insertion depths / dye - without it you type only the "
            "depths that are actually needed"
        )
        csv_row.addWidget(self._csv_edit, 1)
        csv_btn = QPushButton("Browse")
        csv_btn.clicked.connect(self._browse_csv)
        csv_row.addWidget(csv_btn)
        src_form.addLayout(csv_row)

        scan_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.setFixedHeight(30)
        self._scan_btn.setToolTip(
            "Read the probe geometry of every recording found. Traces are not "
            "touched, so this is fast even on a spinning disk."
        )
        self._scan_btn.clicked.connect(self.scan)
        scan_row.addWidget(self._scan_btn)
        scan_row.addWidget(QLabel("Penetration:"))
        self._pen_combo = QComboBox()
        self._pen_combo.currentIndexChanged.connect(self._on_penetration_changed)
        scan_row.addWidget(self._pen_combo, 1)
        src_form.addLayout(scan_row)
        layout.addWidget(src)

        split = QSplitter(Qt.Vertical)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.itemChanged.connect(self._on_item_changed)
        split.addWidget(self._table)

        cov_box = QGroupBox("Coverage per shank (µm from the tip)")
        cov_layout = QVBoxLayout(cov_box)
        self._coverage_host = QWidget()
        self._coverage_layout = QVBoxLayout(self._coverage_host)
        self._coverage_layout.setContentsMargins(0, 0, 0, 0)
        cov_layout.addWidget(self._coverage_host, 1)
        self._plot = self._make_plot()
        if self._plot is not None:
            self._coverage_layout.addWidget(self._plot)
        else:
            self._coverage_layout.addWidget(
                QLabel("Install the ephys extra for the coverage plot "
                       "(pip install 'atlastrack[ephys]').")
            )
        split.addWidget(cov_box)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 440])
        layout.addWidget(split, 1)

        self._status = QLabel("Choose a folder and press Scan.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._add_btn = QPushButton("Add ticked recordings to probe")
        self._add_btn.setFixedHeight(30)
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._apply_to_project)
        btn_row.addWidget(self._add_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _make_plot(self):
        try:
            import pyqtgraph as pg
        except ImportError:
            return None
        pg.setConfigOption("antialias", True)
        widget = pg.GraphicsLayoutWidget()
        self._plot_ok = True
        return widget

    # -- browsing --------------------------------------------------------

    def _browse_folder(self) -> None:
        start = self._folder_edit.text().strip() or ""
        path = QFileDialog.getExistingDirectory(self, "Session or subject folder", start)
        if path:
            self._folder_edit.setText(path)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Metadata table", self._csv_edit.text().strip() or "",
            "Table (*.csv);;All files (*)",
        )
        if path:
            self._csv_edit.setText(path)

    # -- scanning --------------------------------------------------------

    def scan(self) -> None:
        """Start the scan. Results arrive in :meth:`set_penetrations`."""
        root = self._folder_edit.text().strip()
        if not root:
            self._status.setText("Choose a folder first.")
            return
        sidecar = self._csv_edit.text().strip() or None
        self._scan_btn.setEnabled(False)
        self._status.setText(f"Scanning {root} …")

        from atlastrack.gui.workers import discover_recordings_worker

        worker = discover_recordings_worker(root, sidecar)
        worker.returned.connect(self._on_scan_done)
        worker.errored.connect(self._on_scan_failed)
        self._worker = worker
        worker.start()

    def _on_scan_done(self, penetrations: list) -> None:
        self._scan_btn.setEnabled(True)
        self._worker = None
        self.set_penetrations(penetrations)

    def _on_scan_failed(self, exc: Exception) -> None:
        self._scan_btn.setEnabled(True)
        self._worker = None
        self._status.setText(f"Scan failed: {exc}")

    def set_penetrations(self, penetrations: list[Penetration]) -> None:
        """Populate from a scan result. Separate from :meth:`scan` so it is testable."""
        self._penetrations = list(penetrations)
        self._pen_combo.blockSignals(True)
        self._pen_combo.clear()
        for pen in self._penetrations:
            n = len(pen.recordings)
            self._pen_combo.addItem(f"{pen.label} - {n} recording{'s' if n != 1 else ''}")
        self._pen_combo.blockSignals(False)
        if not self._penetrations:
            self._status.setText("No Open Ephys recordings found under that folder.")
            self._add_btn.setEnabled(False)
            self._table.setRowCount(0)
            return
        self._pen_combo.setCurrentIndex(self._preferred_index())
        self._on_penetration_changed()

    def _preferred_index(self) -> int:
        """Pre-select the penetration for the probe the user is working on.

        **Probe first, dye second.** One session has one dye per day but a probe per
        hemisphere, so LO_07 2026-05-08 has two green penetrations - ProbeA and
        ProbeB. Matching on dye alone always returned the first, so opening discovery
        with ProbeB selected showed ProbeA's coverage, which reads as the tool
        misplacing the single-shank recording rather than as the wrong penetration.
        """
        wanted = (self._probe_label or "").strip().lower()
        if wanted:
            for i, pen in enumerate(self._penetrations):
                if (pen.probe_label or "").strip().lower() == wanted:
                    return i
        # str() because project_path is a Path in the app and a plain string in
        # tests; calling .lower() on it directly worked in tests and threw in the GUI.
        name = str(getattr(self._state, "project_path", None) or "").lower()
        if not name:
            name = str(getattr(self._state.project, "name", "") or "").lower()
        for i, pen in enumerate(self._penetrations):
            if pen.dye and pen.dye in name:
                return i
        return 0

    # -- table -----------------------------------------------------------

    def selected_penetration(self) -> Penetration | None:
        i = self._pen_combo.currentIndex()
        if 0 <= i < len(self._penetrations):
            return self._penetrations[i]
        return None

    def _on_penetration_changed(self) -> None:
        self._refresh_table()
        self.refresh_coverage()

    def _refresh_table(self) -> None:
        pen = self.selected_penetration()
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        if pen is not None:
            self._table.setRowCount(len(pen.recordings))
            for row, rec in enumerate(pen.recordings):
                self._fill_row(row, rec)
        self._table.blockSignals(False)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._add_btn.setEnabled(bool(pen and pen.recordings))
        self._update_status()

    def _fill_row(self, row: int, rec: RecordingCandidate) -> None:
        check = QCheckBox()
        check.setChecked(True)
        check.stateChanged.connect(self._on_tick_changed)
        holder = QWidget()
        box = QHBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setAlignment(Qt.AlignCenter)
        box.addWidget(check)
        self._table.setCellWidget(row, 0, holder)

        label = rec.stream.recording_label or Path(rec.stream.recording_dir).name
        for col, text in ((1, label), (2, rec.stream.describe_config())):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, col, item)

        depth = QTableWidgetItem(
            "" if rec.insertion_depth_um is None else f"{rec.insertion_depth_um:.0f}"
        )
        depth.setFlags(depth.flags() | Qt.ItemIsEditable)
        if rec.insertion_depth_um is None:
            depth.setToolTip(
                "Needed only because this penetration was recorded at more than one "
                "depth. Recordings that differ only in electrode bank need no depth."
            )
        self._table.setItem(row, 3, depth)

        source = QTableWidgetItem(rec.depth_source)
        source.setFlags(source.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 4, source)

        mismatch = rec.config_mismatch()
        note = QTableWidgetItem(mismatch or "")
        note.setFlags(note.flags() & ~Qt.ItemIsEditable)
        if mismatch:
            note.setToolTip(
                "The notes and the probe geometry disagree about which electrodes "
                "were recorded. The geometry is used; check the notes."
            )
        self._table.setItem(row, 5, note)

    def _on_tick_changed(self, _state: int) -> None:
        self.refresh_coverage()
        self._update_status()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 3:
            return
        pen = self.selected_penetration()
        if pen is None or not (0 <= item.row() < len(pen.recordings)):
            return
        rec = pen.recordings[item.row()]
        text = item.text().strip()
        if not text:
            rec.insertion_depth_um, rec.depth_source = None, "unknown"
        else:
            try:
                rec.insertion_depth_um = float(text)
                rec.depth_source = "user"
            except ValueError:
                # Put the old value back rather than silently keeping a bad one.
                self._table.blockSignals(True)
                item.setText(
                    "" if rec.insertion_depth_um is None
                    else f"{rec.insertion_depth_um:.0f}"
                )
                self._table.blockSignals(False)
                self._status.setText(f"{text!r} is not a depth in µm.")
                return
        src = self._table.item(item.row(), 4)
        if src is not None:
            src.setText(rec.depth_source)
        self.refresh_coverage()
        self._update_status()

    def ticked(self) -> list[RecordingCandidate]:
        pen = self.selected_penetration()
        if pen is None:
            return []
        out = []
        for row, rec in enumerate(pen.recordings):
            holder = self._table.cellWidget(row, 0)
            box = holder.findChild(QCheckBox) if holder is not None else None
            if box is None or box.isChecked():
                out.append(rec)
        return out

    def _update_status(self) -> None:
        pen = self.selected_penetration()
        if pen is None:
            return
        chosen = self.ticked()
        missing = [r for r in chosen if r.insertion_depth_um is None]
        depths = sorted({r.insertion_depth_um for r in chosen
                         if r.insertion_depth_um is not None})
        if missing and len(depths) > 0:
            text = (
                f"{len(missing)} of {len(chosen)} ticked recordings need an insertion "
                "depth before they can be placed; they are left out of the coverage "
                "below."
            )
        elif missing:
            text = (
                f"No insertion depths known. The {len(missing)} recordings are shown "
                "as if taken at one depth - correct when they differ only in "
                "electrode bank, wrong if the probe was advanced between them."
            )
        else:
            text = (
                f"{len(chosen)} recordings at "
                + (", ".join(f"{d:.0f} µm" for d in depths) or "one depth")
                + "."
            )
        # How the grouping was arrived at, when it was not read from the path. This
        # belongs above the recording list rather than in a log: the user is about to
        # tick rows on the assumption that they share one insertion.
        warnings = grouping_warnings(self._penetrations)
        if warnings:
            text = "⚠ " + "\n⚠ ".join(warnings) + "\n" + text
        self._status.setText(text)
        self._status.setToolTip("\n\n".join(warnings) if warnings else "")

    # -- coverage plot ----------------------------------------------------

    def refresh_coverage(self) -> None:
        """Redraw one row per shank: region bands behind, recorded spans over them."""
        if not self._plot_ok:
            return
        import pyqtgraph as pg

        self._plot.clear()
        pen = self.selected_penetration()
        if pen is None:
            return
        chosen = self.ticked()
        view = Penetration(pen.subject, pen.session_date, pen.probe_label, pen.dye,
                           list(chosen))
        shanks = view.shanks or (0,)
        reference = max(view.depths) if view.depths else None
        bands_by_shank = self._region_bands(view, shanks, reference)
        extent = self._x_extent(view, shanks)

        first = None
        for r, shank in enumerate(shanks):
            plot = self._plot.addPlot(row=r, col=0)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.setMouseEnabled(x=True, y=False)
            plot.getAxis("left").setWidth(108)
            plot.getAxis("left").setTicks([[(0.5, self._shank_label(shank))]])
            plot.setYRange(0.0, 1.0, padding=0.0)
            plot.setXRange(0.0, extent, padding=0.01)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
            if r < len(shanks) - 1:
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setLabel("bottom", "µm from tip")
            spans = coverage_from_tip(view, shank, reference_depth_um=reference)
            self._draw_shank_row(plot, pg, bands_by_shank.get(shank, []), spans, extent)
        self._plot.setMinimumHeight(max(_ROW_H_PX * len(shanks) + 30, 120))

    def _shank_label(self, shank: int) -> str:
        """``shank 0 · notes 1 · post`` - both numbering schemes, plus which end.

        The lab's notebooks number shanks **1-4** and this app stores them **0-3**, so
        "single column, shank 1, most posterior" reads as a contradiction against a
        row labelled `shank 0`. It is not one - on LO_07 the registered geometry puts
        our shank 0 (ProbeA) and our shank 3 (ProbeB) at the posterior end, exactly
        where the notes say - but the reader has to do the translation, and that is
        the sort of thing that gets checked once and assumed thereafter.

        The anterior/posterior tag needs a registered probe; without one the row
        simply carries the two numbers.
        """
        text = f"shank {shank} · notes {shank + 1}"
        end = self._row_end(shank)
        return f"{text} · {end}" if end else text

    def _row_end(self, shank: int) -> str:
        """``post``/``ant`` for the end shanks of a registered array, else ``""``."""
        probe = self._probe_spec()
        if probe is None:
            return ""
        tips = [(s.index, s.tip_ccf_um) for s in probe.shanks if s.tip_ccf_um is not None]
        if len(tips) < 2:
            return ""
        # CCF AP increases posteriorly (atlastrack.io.ccf_coords).
        order = sorted(tips, key=lambda it: float(it[1][0]))
        if shank == order[-1][0]:
            return "post"
        if shank == order[0][0]:
            return "ant"
        return ""

    def _draw_shank_row(self, plot, pg, bands, spans, extent: float) -> None:
        """Anatomy at full strength where a recording reaches it, dimmed where not.

        Drawing the recorded spans *over* the regions was the obvious thing and it was
        nearly invisible - a pale bar on a busy background. Dimming the complement
        instead makes the question the plot exists to answer ("what does this shank
        actually see?") the brightest thing on it, and it degrades gracefully: with no
        coverage at all the whole row goes dark, which is exactly the message.
        """
        for band, colour in bands:
            if not band.acronym:
                continue
            item = pg.LinearRegionItem(
                values=(band.lo, band.hi), orientation="vertical",
                brush=pg.mkBrush(*colour, 235), pen=pg.mkPen(None), movable=False,
            )
            item.setZValue(-20)
            plot.addItem(item)

        for lo, hi in _complement(spans, extent):
            scrim = pg.LinearRegionItem(
                values=(lo, hi), orientation="vertical",
                brush=pg.mkBrush(0, 0, 0, 195), pen=pg.mkPen(None), movable=False,
            )
            scrim.setZValue(0)
            plot.addItem(scrim)

        for lo, hi in spans:
            bar = pg.LinearRegionItem(
                values=(lo, hi), orientation="vertical",
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen(*_SPAN_COLOUR, 230, width=2), movable=False,
            )
            bar.setZValue(20)
            plot.addItem(bar)
            width = pg.TextItem(f"{hi - lo:.0f} µm", color=_SPAN_COLOUR, anchor=(0.5, 1))
            width.setPos(0.5 * (lo + hi), 0.04)
            width.setZValue(25)
            plot.addItem(width)

        for band, _colour in bands:
            if not band.acronym or band.hi - band.lo <= 0.02 * extent:
                continue
            # Label the part of the band a recording actually reaches. Centring on the
            # whole band drops the text into the dimmed stretch, where it is both
            # unreadable and a claim about tissue nothing measured.
            overlaps = [(max(lo, band.lo), min(hi, band.hi)) for lo, hi in spans
                        if lo < band.hi and band.lo < hi]
            if not overlaps:
                continue
            lo, hi = max(overlaps, key=lambda ab: ab[1] - ab[0])
            if hi - lo <= 0.015 * extent:
                continue
            text = pg.TextItem(band.acronym, color=(25, 25, 25), anchor=(0.5, 0))
            text.setPos(0.5 * (lo + hi), 0.97)
            text.setZValue(25)
            plot.addItem(text)

    def _x_extent(self, pen: Penetration, shanks) -> float:
        reach = 0.0
        for s in shanks:
            for _, hi in coverage_from_tip(pen, s):
                reach = max(reach, hi)
        for rec in pen.recordings:
            for cov in rec.stream.coverage:
                reach = max(reach, cov.bottom_um)
        return max(reach, 1000.0) * 1.02

    def _region_bands(self, pen: Penetration, shanks, reference) -> dict:
        """Histology regions along each shank, in µm from the tip.

        Needs a registered probe, an atlas and an insertion depth. Any of those
        missing simply means no bands are drawn - the coverage spans are still the
        point, and a plain background is honest about what is not known.
        """
        atlas = getattr(self._state, "atlas", None)
        probe = self._probe_spec()
        if atlas is None or probe is None or reference is None:
            return {}
        import numpy as np

        from atlastrack.ephys.regions import (
            band_colours,
            region_bands,
            region_colour_map,
            regions_along_track,
            white_matter_acronyms,
        )

        raw: dict[int, list] = {}
        for shank in shanks:
            sh = next((s for s in probe.shanks if s.index == shank), None)
            if sh is None or sh.tip_ccf_um is None or sh.entry_ccf_um is None:
                continue
            from_tip = np.arange(0.0, self._x_extent(pen, shanks), 15.0)
            below = float(reference) - from_tip
            hits = regions_along_track(atlas, sh.tip_ccf_um, sh.entry_ccf_um, below)
            raw[shank] = region_bands(hits, from_tip)
        shared = region_colour_map(
            list(raw.values()),
            white_matter=white_matter_acronyms(
                atlas, {b.acronym for bands in raw.values() for b in bands}
            ),
        )
        out: dict[int, list] = {}
        for shank, bands in raw.items():
            colours = band_colours(bands, shared=shared)
            out[shank] = [
                (_Span(min(b.top_um, b.bottom_um), max(b.top_um, b.bottom_um),
                       b.acronym), c)
                for b, c in zip(bands, colours, strict=False)
            ]
        return out

    # -- applying ---------------------------------------------------------

    def _probe_spec(self):
        pen = self.selected_penetration()
        if pen is None:
            return None
        return next(
            (p for p in self._state.project.probes if p.label == pen.probe_label), None
        )

    def _apply_to_project(self) -> None:
        n = self.apply_to_project()
        if n < 0:
            return
        QMessageBox.information(
            self, "Recordings added",
            f"{n} recording{'s' if n != 1 else ''} added to "
            f"{self._probe_spec().label}.",
        )
        self.accept()

    def apply_to_project(self) -> int:
        """Write the ticked recordings onto the probe. Returns how many, or -1."""
        pen = self.selected_penetration()
        probe = self._probe_spec()
        if pen is None:
            return -1
        if probe is None:
            QMessageBox.warning(
                self, "No matching probe",
                f"This project has no probe labelled {pen.probe_label!r}. Add and "
                "register it first - the recordings attach to a probe, not to a shank.",
            )
            return -1

        from atlastrack.project.schema import EphysRecordingRef

        existing = {(r.path, r.stream_name) for r in probe.recordings}
        added = 0
        for rec in self.ticked():
            key = (rec.stream.recording_dir, rec.stream.stream_name)
            if key in existing:
                continue
            ranges = {
                derive_electrode_range(c) for c in rec.stream.coverage
            } - {None}
            probe.recordings.append(
                EphysRecordingRef(
                    path=rec.stream.recording_dir,
                    label=rec.stream.recording_label or "",
                    stream_name=rec.stream.stream_name,
                    insertion_depth_um=float(rec.insertion_depth_um or 0.0),
                    electrode_range=next(iter(ranges)) if len(ranges) == 1 else None,
                    # Site positions in these recordings are absolute on the shank,
                    # so the bank offset is already included; adding it again would
                    # push the recording a bank too shallow.
                    bank_offset_um=0.0,
                )
            )
            existing.add(key)
            added += 1
        return added


def _complement(spans: list[tuple[float, float]], extent: float
                ) -> list[tuple[float, float]]:
    """The stretches of ``0..extent`` no span covers."""
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for lo, hi in sorted(spans):
        if lo > cursor:
            out.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < extent:
        out.append((cursor, extent))
    return out


class _Span:
    """A region band reduced to what the coverage row draws."""

    __slots__ = ("acronym", "hi", "lo")

    def __init__(self, lo: float, hi: float, acronym: str) -> None:
        self.lo, self.hi, self.acronym = lo, hi, acronym
