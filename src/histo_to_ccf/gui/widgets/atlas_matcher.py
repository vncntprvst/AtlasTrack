"""Atlas matcher dialog: side-by-side / overlay section-to-atlas AP matching.

Replaces the old in-canvas "Atlas preview" vignette. The user scrolls through
their slide's sections on the left and coronal atlas slices on the right until
they match, then assigns the AP. With **Link** on, an anchored section drives the
atlas AP of every other section by an editable spacing (the visual twin of the
ordering panel's Interpolate AP). An **Overlay** view blends the atlas reference
(and region boundaries) on top of the section for a finer anatomical check.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM
from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    from histo_to_ccf.project.schema import ChannelLevels, Section

_EDGE_COLOR = (0, 255, 0)  # atlas region boundaries drawn in green


# ---------------------------------------------------------------------------
# numpy -> Qt helpers
# ---------------------------------------------------------------------------

def _to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert a uint8 array (HxW gray, HxWx3 RGB, or HxWx4 RGBA) to a QPixmap."""
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    if arr.ndim == 2:
        h, w = arr.shape
        fmt, bpl = QImage.Format_Grayscale8, w
    elif arr.shape[2] == 3:
        h, w = arr.shape[:2]
        fmt, bpl = QImage.Format_RGB888, 3 * w
    else:
        h, w = arr.shape[:2]
        fmt, bpl = QImage.Format_RGBA8888, 4 * w
    qimg = QImage(arr.data, w, h, bpl, fmt)
    # .copy() deep-copies before ``arr`` goes out of scope (QImage does not own it).
    return QPixmap.fromImage(qimg.copy())


def _stretch(channel: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    """Window a 2D channel to uint8. ``lo``/``hi`` are 0-1 fractions of full scale."""
    a = channel.astype(np.float32)
    full = 255.0 if a.max() <= 255.0 else float(a.max())
    if lo is None or hi is None:
        lo_v, hi_v = float(a.min()), float(a.max())
    else:
        lo_v, hi_v = lo * full, hi * full
    if hi_v <= lo_v:
        hi_v = lo_v + 1.0
    return (np.clip((a - lo_v) / (hi_v - lo_v), 0.0, 1.0) * 255.0).astype(np.uint8)


def _display_histology(crop: np.ndarray, levels: "ChannelLevels | None") -> np.ndarray:
    """Window a section crop to a displayable uint8 image (gray or RGB).

    Channels without an explicit level are windowed by a **shared** min/max taken
    across all three channels - NOT each channel's own min/max. Per-channel
    auto-stretch would blow an empty channel's background noise (e.g. green on a
    DAPI+red section) up to full saturation, painting the black background bright
    green; a shared window keeps signal-free channels dark and preserves colour.
    """
    if crop.ndim == 2:
        lo = levels.low[0] if levels else None
        hi = levels.high[0] if levels else None
        return _stretch(crop, lo, hi)
    rgb = crop[..., :3].astype(np.float32)
    full = 255.0 if rgb.max() <= 255.0 else float(rgb.max())
    shared_lo, shared_hi = float(rgb.min()) / full, float(rgb.max()) / full
    chans = []
    for i in range(3):
        src = crop[..., i] if crop.shape[2] > i else crop[..., -1]
        lo = levels.low[i] if levels and i < len(levels.low) else shared_lo
        hi = levels.high[i] if levels and i < len(levels.high) else shared_hi
        chans.append(_stretch(src, lo, hi))
    return np.stack(chans, axis=-1)


def _display_reference(ref: np.ndarray) -> np.ndarray:
    """Normalise a float atlas reference slice to uint8 grayscale."""
    return _stretch(ref, None, None)


def _edges_pixmap(annotation: np.ndarray) -> QPixmap:
    """Green RGBA edge map where atlas region labels meet (transparent elsewhere)."""
    from histo_to_ccf.registration.transforms import annotation_boundaries

    mask = annotation_boundaries(annotation)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask] = (*_EDGE_COLOR, 255)
    return _to_pixmap(rgba)


# ---------------------------------------------------------------------------
# Image pane
# ---------------------------------------------------------------------------

class _ImagePane(QGraphicsView):
    """A pan/zoom image view with a base image plus optional overlay + edges."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.black)
        self._base = QGraphicsPixmapItem()
        self._overlay = QGraphicsPixmapItem()
        self._edges = QGraphicsPixmapItem()
        for z, item in ((0, self._base), (1, self._overlay), (2, self._edges)):
            item.setZValue(z)
            self._scene.addItem(item)

    def set_base(self, pixmap: QPixmap, fit: bool = False) -> None:
        self._base.setPixmap(pixmap)
        self._scene.setSceneRect(self._base.boundingRect())
        if fit:
            self.fit()

    def set_overlay(self, pixmap: QPixmap | None, opacity: float = 1.0) -> None:
        self._overlay.setPixmap(pixmap or QPixmap())
        self._overlay.setOpacity(opacity)

    def set_edges(self, pixmap: QPixmap | None) -> None:
        self._edges.setPixmap(pixmap or QPixmap())

    def fit(self) -> None:
        if not self._base.pixmap().isNull():
            self.fitInView(self._base, Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class AtlasMatcherDialog(QDialog):
    """Side-by-side / overlay tool to match sections to atlas AP and assign it."""

    def __init__(
        self,
        state: WorkflowState,
        browser=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Atlas matcher")
        self.resize(1000, 640)
        self._state = state
        # The Atlas-tab browser (+ its ordering panel) to sync with on open/close.
        self._browser = browser
        self._pos = 0  # index into the AP-ordered section list
        self._anchor: tuple[int, float] | None = None  # (position, AP-from-bregma)
        self._updating = False
        self._build_ui()
        self._init_ap_range()
        self._sync_from_tab()
        self._refresh(fit=True)

    # -- sync with the Atlas tab ----------------------------------------

    def _ordering_panel(self):
        return getattr(self._browser, "ordering_panel", None) if self._browser else None

    def _sync_from_tab(self) -> None:
        """Seed AP + spacing from the Atlas-tab widgets when the dialog opens."""
        if self._browser is not None:
            self._set_ap_silent(self._browser.current_ap_bregma())
        panel = self._ordering_panel()
        if panel is not None:
            self._spacing_spin.setValue(panel.current_spacing())

    def _sync_to_tab(self) -> None:
        """Write AP + spacing back to the Atlas-tab widgets when closing."""
        if self._browser is not None:
            self._browser.set_ap_bregma(self._ap_spin.value())
        panel = self._ordering_panel()
        if panel is not None:
            panel.set_spacing(self._spacing_spin.value())
            panel.refresh()  # section ordering / AP may have changed

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._sync_to_tab()
        super().closeEvent(event)

    # -- ordered sections ------------------------------------------------

    def _ordered_sections(self) -> "list[Section]":
        idx = self._state.active_slide_idx
        if idx is None or idx >= len(self._state.project.slides):
            return []
        slide = self._state.project.slides[idx]
        return sorted(slide.sections, key=lambda s: s.ap_order)

    def _current_section(self) -> "Section | None":
        ordered = self._ordered_sections()
        if not ordered:
            return None
        self._pos = max(0, min(self._pos, len(ordered) - 1))
        return ordered[self._pos]

    # -- bregma <-> absolute AP -----------------------------------------

    @staticmethod
    def _bregma_to_absolute(ap_bregma: float) -> float:
        return BREGMA_AP_FROM_ORIGIN_UM - ap_bregma

    @staticmethod
    def _absolute_to_bregma(ap_abs: float) -> float:
        return BREGMA_AP_FROM_ORIGIN_UM - ap_abs

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Row 1: section navigation + AP.
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Section")
        self._prev_btn.clicked.connect(lambda: self._step_section(-1))
        self._next_btn = QPushButton("Section ▶")
        self._next_btn.clicked.connect(lambda: self._step_section(+1))
        self._sec_label = QLabel("-")
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._sec_label, 1)
        nav.addWidget(self._next_btn)
        nav.addSpacing(16)
        nav.addWidget(QLabel("AP from bregma (µm):"))
        self._ap_spin = QDoubleSpinBox()
        self._ap_spin.setRange(-15000.0, BREGMA_AP_FROM_ORIGIN_UM)
        self._ap_spin.setSingleStep(25.0)
        self._ap_spin.setValue(0.0)  # bregma
        self._ap_spin.valueChanged.connect(self._on_ap_changed)
        nav.addWidget(self._ap_spin)
        root.addLayout(nav)

        # Row 2: link / spacing / anchor.
        link_row = QHBoxLayout()
        self._link_check = QCheckBox("Link")
        self._link_check.setToolTip(
            "Drive the atlas AP of every section from one anchored section,\n"
            "stepping by the spacing below. Off = scroll each side freely."
        )
        self._link_check.toggled.connect(self._on_link_toggled)
        link_row.addWidget(self._link_check)
        link_row.addWidget(QLabel("spacing (µm):"))
        self._spacing_spin = QDoubleSpinBox()
        self._spacing_spin.setRange(-5000.0, 5000.0)
        self._spacing_spin.setSingleStep(25.0)
        self._spacing_spin.setValue(100.0)
        self._spacing_spin.setToolTip(
            "AP step between consecutive sections (section thickness x interval).\n"
            "Sign sets direction; flip it if your ordering runs the other way."
        )
        link_row.addWidget(self._spacing_spin)
        self._anchor_btn = QPushButton("Anchor here")
        self._anchor_btn.setToolTip("Pin the current section + AP as the link reference.")
        self._anchor_btn.clicked.connect(self._set_anchor)
        link_row.addWidget(self._anchor_btn)
        link_row.addStretch()
        root.addLayout(link_row)

        # Row 3: view mode / overlay controls.
        view_row = QHBoxLayout()
        self._split_radio = QRadioButton("Split")
        self._split_radio.setChecked(True)
        self._overlay_radio = QRadioButton("Overlay")
        mode_grp = QButtonGroup(self)
        mode_grp.addButton(self._split_radio)
        mode_grp.addButton(self._overlay_radio)
        self._split_radio.toggled.connect(self._on_mode_changed)
        view_row.addWidget(QLabel("View:"))
        view_row.addWidget(self._split_radio)
        view_row.addWidget(self._overlay_radio)
        view_row.addSpacing(16)
        view_row.addWidget(QLabel("opacity:"))
        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(50)
        self._opacity.setFixedWidth(120)
        self._opacity.valueChanged.connect(self._update_overlay_only)
        view_row.addWidget(self._opacity)
        self._edges_check = QCheckBox("Atlas edges")
        self._edges_check.setChecked(True)
        self._edges_check.toggled.connect(lambda: self._refresh())
        view_row.addWidget(self._edges_check)
        view_row.addStretch()
        root.addLayout(view_row)

        # Center: stacked split / overlay panes.
        self._hist_pane = _ImagePane()
        self._atlas_pane = _ImagePane()
        self._overlay_pane = _ImagePane()

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel("Histology section"))
        left_l.addWidget(self._hist_pane)
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.addWidget(QLabel("Atlas"))
        right_l.addWidget(self._atlas_pane)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([500, 500])

        self._stack = QStackedWidget()
        self._stack.addWidget(split)            # page 0: split
        self._stack.addWidget(self._overlay_pane)  # page 1: overlay
        root.addWidget(self._stack, 1)

        # Bottom: status + assign.
        bottom = QHBoxLayout()
        self._status = QLabel("")
        bottom.addWidget(self._status, 1)
        self._prematch_btn = QPushButton("Pre-match all (DeepSlice)")
        self._prematch_btn.setToolTip(
            "Run DeepSlice on every section of the active slide to fill a consistent\n"
            "set of AP positions in one pass, then fine-tune here. The full predicted\n"
            "planes (incl. tilt) are cached so Register can reuse them.\n"
            "First run downloads the DeepSlice model and is slow."
        )
        self._prematch_btn.clicked.connect(self._prematch_deepslice)
        bottom.addWidget(self._prematch_btn)
        assign_btn = QPushButton("Assign")
        assign_btn.setToolTip("Store this AP on the current section.")
        assign_btn.clicked.connect(self._assign_current)
        bottom.addWidget(assign_btn)
        assign_all_btn = QPushButton("Assign all (linked)")
        assign_all_btn.setToolTip(
            "Fill every section's AP from the anchor + spacing (like Interpolate AP)."
        )
        assign_all_btn.clicked.connect(self._assign_all)
        bottom.addWidget(assign_all_btn)
        root.addLayout(bottom)

    def _init_ap_range(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            return
        ap_max = atlas.reference.shape[0] * atlas.resolution[0]
        self._ap_spin.setRange(
            self._absolute_to_bregma(float(ap_max)), BREGMA_AP_FROM_ORIGIN_UM
        )

    # -- atlas sampling --------------------------------------------------

    def _atlas_slice(self, ap_abs: float, out_shape: tuple[int, int]):
        from histo_to_ccf.atlas.planes import coronal_anchoring, resample_atlas_at_plane

        atlas = self._state.atlas
        anchoring = coronal_anchoring(atlas, ap_abs)
        return resample_atlas_at_plane(atlas, anchoring, out_shape)

    def _section_crop(self, section: "Section") -> np.ndarray | None:
        img = self._state.slide_images.get(section.slide_idx)
        if img is None:
            return None
        h, w = img.shape[:2]
        x0, y0, x1, y1 = section.bbox_px
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1]

    # -- events ----------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().showEvent(event)
        for pane in (self._hist_pane, self._atlas_pane, self._overlay_pane):
            pane.fit()

    def _step_section(self, delta: int) -> None:
        ordered = self._ordered_sections()
        if not ordered:
            return
        self._pos = max(0, min(self._pos + delta, len(ordered) - 1))
        self._on_section_changed()

    def _on_section_changed(self) -> None:
        """Section navigation: update the AP shown, then redraw."""
        section = self._current_section()
        if section is None:
            self._refresh(fit=True)
            return
        if self._link_check.isChecked():
            self._set_ap_silent(self._linked_ap_bregma(self._pos))
        elif section.plane is not None:
            self._set_ap_silent(self._absolute_to_bregma(section.plane.ap_um))
        self._state.active_section_idx = section.index
        self._refresh(fit=True)

    def _on_ap_changed(self) -> None:
        if self._updating:
            return
        self._refresh()

    def _on_link_toggled(self, on: bool) -> None:
        if on and self._anchor is None:
            self._set_anchor()
        elif on:
            self._set_ap_silent(self._linked_ap_bregma(self._pos))
        self._refresh()

    def _on_mode_changed(self) -> None:
        self._stack.setCurrentIndex(0 if self._split_radio.isChecked() else 1)
        self._refresh(fit=True)

    def _set_ap_silent(self, ap_bregma: float) -> None:
        self._updating = True
        self._ap_spin.setValue(ap_bregma)
        self._updating = False

    # -- link math -------------------------------------------------------

    def _set_anchor(self) -> None:
        self._anchor = (self._pos, self._ap_spin.value())
        self._status.setText(
            f"Anchored section position {self._pos + 1} at AP "
            f"{self._ap_spin.value():.0f} µm (from bregma)."
        )

    def _linked_ap_bregma(self, pos: int) -> float:
        anchor_pos, anchor_ap = self._anchor or (self._pos, self._ap_spin.value())
        return anchor_ap - (pos - anchor_pos) * self._spacing_spin.value()

    # -- assignment ------------------------------------------------------

    def _assign_current(self) -> None:
        section = self._current_section()
        if section is None:
            self._status.setText("No section to assign.")
            return
        self._write_ap(section, self._bregma_to_absolute(self._ap_spin.value()))
        self._status.setText(
            f"Assigned AP={self._ap_spin.value():.0f} µm (from bregma) to "
            f"section {section.index}."
        )

    def _assign_all(self) -> None:
        ordered = self._ordered_sections()
        if not ordered:
            return
        if self._anchor is None:
            self._set_anchor()
        for pos, section in enumerate(ordered):
            self._write_ap(section, self._bregma_to_absolute(self._linked_ap_bregma(pos)))
        self._status.setText(f"Assigned AP to all {len(ordered)} sections from anchor + spacing.")

    @staticmethod
    def _write_ap(section: "Section", ap_abs: float) -> None:
        from histo_to_ccf.project.schema import PlaneParams

        if section.plane is not None:
            section.plane = section.plane.model_copy(update={"ap_um": ap_abs})
        else:
            section.plane = PlaneParams(ap_um=ap_abs)

    # -- DeepSlice pre-match ---------------------------------------------

    def _prematch_section_images(self) -> "dict[int, np.ndarray]":
        """Float32 crops of the active slide's sections, keyed by ``section.index``.

        Built with the same ``io.image.crop`` the Register step uses, so the cached
        crop fingerprints line up and Register can reuse this pre-match.
        """
        from histo_to_ccf.io.image import crop

        out: dict[int, np.ndarray] = {}
        for section in self._ordered_sections():
            img = self._state.slide_images.get(section.slide_idx)
            if img is None:
                continue
            x0, y0, x1, y1 = section.bbox_px
            out[section.index] = crop(img, (x0, y0, x1, y1)).astype(np.float32)
        return out

    def _prematch_deepslice(self) -> None:
        """Run DeepSlice on the active slide and fill every section's AP in one pass."""
        atlas = self._state.atlas
        if atlas is None:
            self._status.setText("Load an atlas in the Atlas tab first.")
            return
        section_images = self._prematch_section_images()
        if not section_images:
            self._status.setText("No section images on the active slide to pre-match.")
            return

        # DeepSlice orders the series by each section's ap_order rank and only
        # *enforces order*, not spacing. So a sane, de-duplicated section order +
        # a set spacing are what make its result trustworthy. Check both before
        # running, and let the user fix the order window first if it's ambiguous.
        ordered = self._ordered_sections()
        ranks = [s.ap_order for s in ordered]
        if len(set(ranks)) != len(ranks):
            QMessageBox.warning(
                self,
                "Section order is ambiguous",
                "Two or more sections share the same position in the AP order, so "
                "DeepSlice can't order the series reliably.\n\nFix the section order "
                "(and remove any gaps for missing sections) in the AP-order window "
                "first, then re-run the pre-match.",
            )
            return
        if abs(self._spacing_spin.value()) < 1e-6:
            resp = QMessageBox.question(
                self,
                "No spacing set",
                "Section spacing is 0 µm. Setting the expected spacing first lets the "
                "pre-match flag any sections that come back out of order or too close "
                "together.\n\nRun DeepSlice anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return

        # AP-sequence rank per section, so DeepSlice orders the series the way the
        # user did (ap_order), not by raw detection index.
        order = {s.index: rank for rank, s in enumerate(ordered)}

        pp = self._state.project_path
        ds_dir = (
            Path(pp).parent if pp is not None else Path(tempfile.mkdtemp())
        ) / "deepslice"

        from histo_to_ccf.gui.workers import deepslice_worker

        self._prematch_btn.setEnabled(False)
        self._status.setText(
            f"Running DeepSlice on {len(section_images)} section(s) (first run is slow)…"
        )
        worker = deepslice_worker(section_images, atlas, ds_dir, order=order)
        worker.returned.connect(lambda anch: self._apply_prematch(anch, section_images))
        worker.errored.connect(self._on_prematch_error)
        worker.start()

    def _on_prematch_error(self, exc: Exception) -> None:
        self._prematch_btn.setEnabled(True)
        self._status.setText(f"DeepSlice pre-match failed: {exc}")

    def _apply_prematch(
        self,
        anchorings: "dict[int, list[float]]",
        section_images: "dict[int, np.ndarray]",
    ) -> None:
        """Write DeepSlice's predicted AP onto each section and cache the full planes.

        Only the scalar centre-AP is stored on ``section.plane`` (the matcher displays
        and lets you fine-tune coronal AP). The full predicted plane - including the
        tilt DeepSlice is good at - is cached on the state so the Register step can
        reuse it without baking a possibly-misconverted tilt into the saved project.
        """
        from histo_to_ccf.gui.workflow import crop_fingerprint
        from histo_to_ccf.io.ccf_coords import atlas_resolution_um
        from histo_to_ccf.registration.pipeline import anchoring_center_ap_um

        self._prematch_btn.setEnabled(True)
        if not anchorings:
            self._status.setText("DeepSlice returned no planes.")
            return

        ap_res = atlas_resolution_um(self._state.atlas)[0]
        sec_by_idx = {s.index: s for s in self._ordered_sections()}
        n = 0
        for idx, anch in anchorings.items():
            section = sec_by_idx.get(idx)
            if section is None:
                continue
            self._write_ap(section, anchoring_center_ap_um(anch, ap_res))
            self._state.deepslice_anchorings[idx] = list(anch)
            if idx in section_images:
                self._state.deepslice_fingerprints[idx] = crop_fingerprint(section_images[idx])
            n += 1

        self._seed_spacing_from_planes()
        # Show the current section's freshly-assigned AP, then redraw.
        section = self._current_section()
        if section is not None and section.plane is not None:
            self._set_ap_silent(self._absolute_to_bregma(section.plane.ap_um))
        self._refresh()
        self._status.setText(
            f"Pre-matched {n} section(s) with DeepSlice. Fine-tune AP here, then "
            "Assign / register."
        )
        self._warn_if_prematch_disordered()

    def _warn_if_prematch_disordered(self) -> None:
        """Warn when DeepSlice's predicted APs aren't a clean monotonic series.

        DeepSlice only *enforces order*, never spacing, so it can return two
        sections almost on top of each other or (rarely) a local inversion. Flag
        AP steps that reverse direction or collapse to a small fraction of the
        median step, so the user fixes them before registering instead of finding
        out later (the exact failure that prompted this guard).
        """
        from histo_to_ccf.registration.pipeline import prematch_ap_order_issues

        ordered = [s for s in self._ordered_sections() if s.plane is not None]
        aps = [(s.index, s.plane.ap_um) for s in ordered]
        reversed_pairs, close_pairs = prematch_ap_order_issues(aps)
        if not reversed_pairs and not close_pairs:
            return

        def _fmt(pairs: "list[tuple[int, int]]") -> str:
            return ", ".join(f"{a}↔{b}" for a, b in pairs)

        lines = ["DeepSlice's predicted APs aren't a clean series:"]
        if reversed_pairs:
            lines.append(f"\n• Out of order (AP reverses): sections {_fmt(reversed_pairs)}")
        if close_pairs:
            lines.append(f"\n• Too close together: sections {_fmt(close_pairs)}")
        lines.append(
            "\n\nFix these before registering - set/confirm the spacing and use "
            "Link + 'Assign AP to all' to even them out, or correct individual APs "
            "here."
        )
        QMessageBox.warning(self, "Check pre-matched AP order", "".join(lines))

    def _seed_spacing_from_planes(self) -> None:
        """Seed the link spacing from the median |AP step| between consecutive sections."""
        from itertools import pairwise

        aps = [s.plane.ap_um for s in self._ordered_sections() if s.plane is not None]
        steps = [abs(b - a) for a, b in pairwise(aps) if b != a]
        if steps:
            self._spacing_spin.setValue(float(np.median(steps)))

    # -- rendering -------------------------------------------------------

    def _update_overlay_only(self) -> None:
        """Cheap path for the opacity slider - just retint the existing overlay."""
        self._overlay_pane.set_overlay(
            self._overlay_pane._overlay.pixmap(), self._opacity.value() / 100.0
        )

    def _refresh(self, fit: bool = False) -> None:
        atlas = self._state.atlas
        section = self._current_section()
        ordered = self._ordered_sections()
        if section is not None:
            self._sec_label.setText(
                f"Section {section.index}  ({self._pos + 1} / {len(ordered)})"
            )
        else:
            self._sec_label.setText("No sections")
        self._prev_btn.setEnabled(self._pos > 0)
        self._next_btn.setEnabled(bool(ordered) and self._pos < len(ordered) - 1)

        if atlas is None:
            self._status.setText("Load an atlas in the Atlas tab first.")
            return

        crop = self._section_crop(section) if section is not None else None

        # Histology pane.
        if crop is not None:
            section_levels = section.levels if section is not None else None
            hist_disp = _display_histology(crop, section_levels)
            self._hist_pane.set_base(_to_pixmap(hist_disp), fit=fit)

        ap_abs = self._bregma_to_absolute(self._ap_spin.value())

        # Atlas pane (native DV x ML aspect).
        dv, ml = atlas.reference.shape[1], atlas.reference.shape[2]
        ref, ann = self._atlas_slice(ap_abs, (dv, ml))
        self._atlas_pane.set_base(_to_pixmap(_display_reference(ref)), fit=fit)
        self._atlas_pane.set_edges(_edges_pixmap(ann) if self._edges_check.isChecked() else None)

        # Overlay pane (atlas resampled onto the section grid).
        if crop is not None:
            h, w = crop.shape[:2]
            o_ref, o_ann = self._atlas_slice(ap_abs, (h, w))
            self._overlay_pane.set_base(_to_pixmap(_display_histology(crop, section_levels)), fit=fit)
            self._overlay_pane.set_overlay(
                _to_pixmap(_display_reference(o_ref)), self._opacity.value() / 100.0
            )
            self._overlay_pane.set_edges(
                _edges_pixmap(o_ann) if self._edges_check.isChecked() else None
            )
