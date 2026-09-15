"""Place landmark correspondences with the section and the atlas side by side.

**Why a separate window rather than a mode in the napari canvas.** The canvas draws
the atlas overlay *on top of* the tissue in one coordinate space, so a click there
cannot say which of the two it meant, and nothing on screen could show what it had
been taken as. Two panes remove the ambiguity by construction: the pane you click
in *is* the answer.

**Both panes show the registration you already ran.** The atlas pane is the
registered atlas as it currently sits on this section - not a fresh coronal slice
of the raw atlas. That matters twice over: it is what makes the outline line up
with the tissue at all, and it is the frame ``ManualLandmarks`` is defined in -
``source`` is a position *on the registered overlay*, ``target`` where it should
have been. Drawing a raw plane here instead put every source point in the wrong
frame, which is why the outline sat visibly off the tissue and lurched once a
couple of pairs were applied.

The atlas pane deliberately shows the overlay **without** any stored landmark warp
(``apply_landmarks=False``): existing source points were recorded in that un-warped
frame, so new ones have to be picked in it too. The tissue pane's overlay applies
the warp, so it previews the result as pairs are placed.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Shared with the Atlas matcher on purpose: one way to turn an array into a pixmap
# and window a section crop, so the two windows cannot drift into showing the same
# slide differently.
from atlastrack.gui.widgets.atlas_matcher import (
    _display_histology,
    _ImagePane,
    _to_pixmap,
)

if TYPE_CHECKING:  # pragma: no cover
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.project.schema import Section

#: How far (view px) the pointer may travel between press and release and still
#: count as a click rather than a pan. Generous: placing a landmark is deliberate,
#: and both panes also pan with a left drag.
_CLICK_SLOP_PX = 4.0

#: Marker radius in scene (section) pixels, and the colours for each role.
_MARKER_R = 7.0
_ATLAS_COLOR = "#ff5f5f"
_TISSUE_COLOR = "#5fd35f"
_PENDING_COLOR = "#ffd23f"

#: Atlas outline colour, and the dim fill under it so the pane has a silhouette to
#: orient by rather than lines floating on black.
_EDGE_RGB = (90, 230, 120)
_EXTENT_GREY = 48

#: At least this many completed pairs before a thin-plate spline is worth solving.
#: Matches the check the Register panel applies to dragged landmarks.
_MIN_PAIRS = 4

#: Auto-placed atlas points, via the same helper "Place landmarks" uses, so both
#: routes seed the same features.
_AUTO_MAX_POINTS = 12

#: Below this many pairs a TPS preview is not meaningful, so the overlay is left
#: as the plain registered atlas.
_MIN_PREVIEW_PAIRS = 3


class _PickPane(_ImagePane):
    """An image pane that reports clicks in scene coordinates and draws markers.

    Left-drag still pans (inherited), so a click is distinguished from a drag by
    how far the pointer moved - otherwise every attempt to pan would drop a point.
    """

    clicked = Signal(float, float)  # scene x, y

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._press_pos = None
        self._markers: list = []

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton or self._press_pos is None:
            return
        moved = (event.pos() - self._press_pos).manhattanLength()
        self._press_pos = None
        if moved <= _CLICK_SLOP_PX:
            point = self.mapToScene(event.pos())
            self.clicked.emit(float(point.x()), float(point.y()))

    def clear_markers(self) -> None:
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers.clear()

    def add_marker(
        self, x: float, y: float, label: str, color: str, *, bold: bool = False
    ) -> None:
        """A ring plus its number, drawn above the image layers."""
        pen = QPen(QColor(color))
        pen.setWidthF(3.0 if bold else 2.0)
        pen.setCosmetic(True)  # constant on screen, so zoom does not fatten it
        radius = _MARKER_R * (1.5 if bold else 1.0)
        ring = self.scene().addEllipse(
            x - radius, y - radius, 2 * radius, 2 * radius, pen, QBrush(Qt.NoBrush)
        )
        ring.setZValue(10)
        self._markers.append(ring)

        font = QFont("", 9)
        font.setBold(bold)
        text = self.scene().addSimpleText(label, font)
        text.setBrush(QBrush(QColor(color)))
        text.setPos(x + radius, y - radius * 2)
        text.setFlag(text.GraphicsItemFlag.ItemIgnoresTransformations, True)
        text.setZValue(11)
        self._markers.append(text)


class PairPointsDialog(QDialog):
    """Pair atlas features with tissue features for one section.

    Owns nothing expensive: it writes ``Section.manual_landmarks`` and
    ``Section.plane``, then hands back to the Register panel through
    ``on_section_changed`` for the re-render / probe re-map / save, so there is
    exactly one implementation of that step. ``warp_labels`` is the panel's own
    ``_warp_labels_for``, reused so the overlay here and the overlay on the canvas
    cannot come from two different code paths.
    """

    def __init__(
        self,
        state: WorkflowState,
        section: Section,
        *,
        warp_labels: Callable[..., np.ndarray | None] | None = None,
        on_section_changed: Callable[[Section], None] | None = None,
        bregma_ap_um: float = 0.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._section = section
        self._warp_labels = warp_labels
        self._on_section_changed = on_section_changed
        self._bregma_ap_um = float(bregma_ap_um)

        # Each entry is ``[source, target]``; ``target`` is None while the atlas
        # side is placed but its tissue match is not - which is what auto-placing
        # leaves behind, and what the user then works through.
        self._pairs: list[list[tuple[float, float] | None]] = []
        self._pending_tissue: tuple[float, float] | None = None

        self._crop: np.ndarray | None = None
        self._base_labels: np.ndarray | None = None
        self._updating = False

        self.setWindowTitle(f"Pair points - section {section.index}")
        self.setModal(False)  # a modal dialog would block the viewer behind it
        self.resize(1150, 720)
        self._build_ui()
        self._load_existing()
        self._refresh_images(fit=True)
        self._refresh_markers()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        outer.addWidget(self._hint)

        panes = QHBoxLayout()
        self._hist_pane = _PickPane()
        self._atlas_pane = _PickPane()
        for title, pane in (
            ("Section (tissue)", self._hist_pane),
            ("Atlas as currently registered", self._atlas_pane),
        ):
            box = QVBoxLayout()
            box.addWidget(QLabel(title))
            box.addWidget(pane, stretch=1)
            wrap = QWidget()
            wrap.setLayout(box)
            panes.addWidget(wrap, stretch=1)
        outer.addLayout(panes, stretch=1)

        self._hist_pane.clicked.connect(lambda x, y: self._on_pane_clicked("tissue", x, y))
        self._atlas_pane.clicked.connect(lambda x, y: self._on_pane_clicked("atlas", x, y))

        outer.addWidget(self._build_display_box())
        outer.addWidget(self._build_plane_box())
        outer.addWidget(self._build_pairs_box())

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _build_display_box(self) -> QGroupBox:
        box = QGroupBox("Display")
        row = QHBoxLayout(box)

        self._overlay_check = QCheckBox("Atlas over tissue")
        self._overlay_check.setToolTip(
            "Draw the registered atlas on the section, so the fit can be judged "
            "directly.\nOff by default: it hides the tissue you are trying to click."
        )
        self._overlay_check.toggled.connect(self._on_overlay_toggled)
        row.addWidget(self._overlay_check)

        row.addWidget(QLabel("Opacity:"))
        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(55)
        self._opacity.setEnabled(False)
        self._opacity.setFixedWidth(130)
        self._opacity.valueChanged.connect(lambda _: self._refresh_images())
        row.addWidget(self._opacity)

        self._preview_check = QCheckBox("Preview warp")
        self._preview_check.setChecked(True)
        self._preview_check.setToolTip(
            "Bend the overlay through the pairs placed so far, as they are placed, "
            "instead of waiting for Apply."
        )
        self._preview_check.toggled.connect(lambda _: self._refresh_images())
        row.addWidget(self._preview_check)
        row.addStretch()
        return box

    def _build_plane_box(self) -> QGroupBox:
        box = QGroupBox("Atlas plane (fine-tune)")
        box.setToolTip(
            "The plane first comes from the Atlas tab; this is where to nudge it "
            "against the tissue you are pairing against.\n"
            "These values describe the slice the NEXT registration will fit - the "
            "panes above show the registration you already have, so they follow a "
            "re-register, not a spin box."
        )
        grid = QGridLayout(box)

        self._ap_spin = QDoubleSpinBox()
        self._ap_spin.setRange(-20000.0, 20000.0)
        self._ap_spin.setSingleStep(25.0)
        self._ap_spin.setSuffix(" um")
        grid.addWidget(QLabel("AP from bregma:"), 0, 0)
        grid.addWidget(self._ap_spin, 0, 1)

        self._ml_spin = QDoubleSpinBox()
        self._ml_spin.setRange(-30.0, 30.0)
        self._ml_spin.setSingleStep(0.5)
        self._ml_spin.setSuffix(" deg")
        self._ml_spin.setToolTip("Tilt about the DV axis: the medial edge moves anterior.")
        grid.addWidget(QLabel("ML tilt:"), 0, 2)
        grid.addWidget(self._ml_spin, 0, 3)

        self._dv_spin = QDoubleSpinBox()
        self._dv_spin.setRange(-30.0, 30.0)
        self._dv_spin.setSingleStep(0.5)
        self._dv_spin.setSuffix(" deg")
        self._dv_spin.setToolTip("Tilt about the ML axis: the dorsal edge moves anterior.")
        grid.addWidget(QLabel("DV tilt:"), 0, 4)
        grid.addWidget(self._dv_spin, 0, 5)

        apply_plane = QPushButton("Apply plane")
        apply_plane.setToolTip(
            "Write these values to the section as a manual AP.\n"
            "Then re-run 'Register all sections' for the overlay to follow."
        )
        apply_plane.clicked.connect(self._apply_plane)
        grid.addWidget(apply_plane, 0, 6)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        grid.addWidget(self._status, 1, 0, 1, 7)
        return box

    def _build_pairs_box(self) -> QGroupBox:
        box = QGroupBox("Landmark pairs")
        row = QHBoxLayout(box)

        self._pairs_label = QLabel("")
        row.addWidget(self._pairs_label)
        row.addStretch()

        auto = QPushButton("Auto-place atlas points")
        auto.setToolTip(
            "Drop points on distinctive atlas features - outline tips, region "
            "junctions, corners - so there is something to work from.\n"
            "Each then waits for you to click where it belongs on the tissue."
        )
        auto.clicked.connect(self._auto_place)
        row.addWidget(auto)

        undo = QPushButton("Undo")
        undo.clicked.connect(self._undo)
        row.addWidget(undo)

        clear = QPushButton("Clear landmarks")
        clear.setToolTip("Remove every point on this section, including saved ones.")
        clear.clicked.connect(self._clear_landmarks)
        row.addWidget(clear)

        reset_tf = QPushButton("Reset transform")
        reset_tf.setToolTip(
            "Drop the box transform / landmark warp on this section and go back to "
            "the registered atlas overlay."
        )
        reset_tf.clicked.connect(self._reset_transform)
        row.addWidget(reset_tf)

        self._apply_btn = QPushButton("Apply landmark warp")
        self._apply_btn.setToolTip(
            f"Warp the atlas through these pairs, re-map probes and save. "
            f"Needs at least {_MIN_PAIRS} completed pairs."
        )
        self._apply_btn.clicked.connect(self._apply_landmarks)
        row.addWidget(self._apply_btn)
        return box

    # ------------------------------------------------------- data / drawing

    def _section_crop(self) -> np.ndarray | None:
        img = self._state.slide_images.get(self._section.slide_idx)
        if img is None:
            return None
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self._section.bbox_px
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        y0, y1 = max(0, int(y0)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return img[y0:y1, x0:x1]

    def _registered_labels(self) -> np.ndarray | None:
        """The registered atlas in section space, **without** any stored TPS.

        Without the TPS on purpose - see the module docstring: stored source points
        live in this frame, so new ones must be picked in it.
        """
        if self._warp_labels is None:
            return None
        try:
            return self._warp_labels(self._section, apply_landmarks=False)
        except Exception:
            return None

    def _complete_pairs(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return [(s, t) for s, t in self._pairs if s is not None and t is not None]

    def _atlas_rgba(self, labels: np.ndarray) -> np.ndarray:
        """Outlines over a dim silhouette, as an RGBA image."""
        from atlastrack.registration.transforms import annotation_boundaries

        extent = np.asarray(labels) > 0
        edges = annotation_boundaries(labels)
        h, w = extent.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[extent] = (_EXTENT_GREY, _EXTENT_GREY, _EXTENT_GREY, 255)
        rgba[edges] = (*_EDGE_RGB, 255)
        return rgba

    def _preview_labels(self, base: np.ndarray) -> np.ndarray:
        """``base`` bent through the pairs placed so far, for the live overlay."""
        pairs = self._complete_pairs()
        if not self._preview_check.isChecked() or len(pairs) < _MIN_PREVIEW_PAIRS:
            return base
        try:
            from atlastrack.registration.landmarks_warp import warp_label_image

            source = np.array([s for s, _ in pairs], dtype=float)
            target = np.array([t for _, t in pairs], dtype=float)
            return warp_label_image(base, source, target)
        except Exception:
            return base

    def _refresh_images(self, *, fit: bool = False) -> None:
        crop = self._section_crop()
        if crop is None:
            self._status.setText("This section's slide image is not loaded.")
            return
        self._crop = crop
        levels = self._section.levels or self._slide_levels()
        self._hist_pane.set_base(_to_pixmap(_display_histology(crop, levels)), fit=fit)

        labels = self._registered_labels()
        if labels is None:
            self._base_labels = None
            self._atlas_pane.set_base(_to_pixmap(np.zeros(crop.shape[:2], np.uint8)), fit=fit)
            self._hist_pane.set_overlay(None)
            self._status.setText(
                "No registered atlas for this section yet - run 'Register all "
                "sections' first, so there is an overlay to correct."
            )
            self._update_hint()
            return

        self._base_labels = labels
        self._atlas_pane.set_base(_to_pixmap(self._atlas_rgba(labels)), fit=fit)

        if self._overlay_check.isChecked():
            shown = self._preview_labels(labels)
            self._hist_pane.set_overlay(
                _to_pixmap(self._atlas_rgba(shown)), self._opacity.value() / 100.0
            )
        else:
            self._hist_pane.set_overlay(None)

        done = len(self._complete_pairs())
        previewing = self._preview_check.isChecked() and done >= _MIN_PREVIEW_PAIRS
        self._status.setText(
            f"Showing the registration already computed for section "
            f"{self._section.index}."
            + (" Overlay previews the warp." if previewing else "")
        )
        self._update_hint()

    def _slide_levels(self):
        try:
            return self._state.project.slides[self._section.slide_idx].levels
        except Exception:
            return None

    def _next_unmatched(self) -> int | None:
        """Index of the first atlas point still waiting for its tissue match."""
        for i, (source, target) in enumerate(self._pairs):
            if source is not None and target is None:
                return i
        return None

    def _refresh_markers(self) -> None:
        self._hist_pane.clear_markers()
        self._atlas_pane.clear_markers()
        nxt = self._next_unmatched()
        for i, (source, target) in enumerate(self._pairs):
            label = str(i + 1)
            if source is not None:
                color = _PENDING_COLOR if target is None else _ATLAS_COLOR
                self._atlas_pane.add_marker(
                    source[0], source[1], label, color, bold=(i == nxt)
                )
            if target is not None:
                self._hist_pane.add_marker(target[0], target[1], label, _TISSUE_COLOR)
        if self._pending_tissue is not None:
            self._hist_pane.add_marker(
                self._pending_tissue[0], self._pending_tissue[1], "?", _PENDING_COLOR, bold=True
            )

        done = len(self._complete_pairs())
        waiting = sum(1 for s, t in self._pairs if s is not None and t is None)
        parts = [f"{done} complete"]
        if waiting:
            parts.append(f"{waiting} awaiting a tissue click")
        self._pairs_label.setText("   -   ".join(parts))
        self._apply_btn.setEnabled(done >= _MIN_PAIRS)
        self._update_hint()

    def _update_hint(self) -> None:
        nxt = self._next_unmatched()
        if nxt is not None:
            self._hint.setText(
                f"Point {nxt + 1} is marked on the atlas (bold). Click the same "
                f"feature on the tissue to complete it."
            )
        elif self._pending_tissue is not None:
            self._hint.setText("Now click the matching feature on the atlas.")
        else:
            self._hint.setText(
                "Click a feature in one pane, then the same feature in the other - "
                "the pane you click in says which side it is. Or press "
                "'Auto-place atlas points' and just click the tissue side."
            )

    # ------------------------------------------------------------- actions

    def _on_pane_clicked(self, side: str, x: float, y: float) -> None:
        if self._crop is None:
            return
        h, w = self._crop.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return  # outside the image: a stray click on the black surround

        if side == "tissue":
            nxt = self._next_unmatched()
            if nxt is not None:
                self._pairs[nxt][1] = (x, y)  # completes the highlighted atlas point
            else:
                self._pending_tissue = (x, y)
        elif self._pending_tissue is not None:
            self._pairs.append([(x, y), self._pending_tissue])
            self._pending_tissue = None
        else:
            self._pairs.append([(x, y), None])

        self._refresh_markers()
        self._refresh_images()

    def _auto_place(self) -> None:
        """Seed atlas-side points on salient features, each awaiting a tissue click."""
        if self._base_labels is None:
            self._status.setText("No registered atlas to place points on.")
            return
        from atlastrack.registration.landmarks_warp import salient_landmarks

        points = salient_landmarks(self._base_labels, max_points=_AUTO_MAX_POINTS)
        for x, y in np.asarray(points, dtype=float).reshape(-1, 2):
            self._pairs.append([(float(x), float(y)), None])
        self._status.setText(
            f"Placed {len(points)} atlas points. Click each one's match on the "
            f"tissue; the bold marker is the one being waited on."
        )
        self._refresh_markers()

    def _undo(self) -> None:
        if self._pending_tissue is not None:
            self._pending_tissue = None
        elif self._pairs:
            self._pairs.pop()
        self._refresh_markers()
        self._refresh_images()

    def _load_existing(self) -> None:
        self._updating = True
        try:
            plane = self._section.plane
            if plane is not None:
                self._ap_spin.setValue(self._bregma_ap_um - plane.ap_um)
                self._ml_spin.setValue(plane.ml_tilt_deg)
                self._dv_spin.setValue(plane.dv_tilt_deg)
        finally:
            self._updating = False

        landmarks = self._section.manual_landmarks
        if landmarks is None:
            return
        source = np.asarray(landmarks.source, dtype=float).reshape(-1, 2)
        target = np.asarray(landmarks.target, dtype=float).reshape(-1, 2)
        for s, t in zip(source, target, strict=False):
            self._pairs.append([(float(s[0]), float(s[1])), (float(t[0]), float(t[1]))])

    def _apply_plane(self) -> None:
        from atlastrack.project.schema import PlaneParams

        base = self._section.plane
        update = {
            "ap_um": self._bregma_ap_um - float(self._ap_spin.value()),
            "ml_tilt_deg": float(self._ml_spin.value()),
            "dv_tilt_deg": float(self._dv_spin.value()),
        }
        self._section.plane = (
            PlaneParams(**update) if base is None else base.model_copy(update=update)
        )
        self._section.ap_source = "manual"
        self._notify()
        self._status.setText(
            f"Plane written to section {self._section.index}. Re-run 'Register all "
            f"sections' for the overlay to follow it."
        )

    def _clear_landmarks(self) -> None:
        if self._pairs and not self._confirm(
            "Clear landmarks",
            f"Remove all {len(self._pairs)} point(s) from section {self._section.index}?",
        ):
            return
        self._pairs.clear()
        self._pending_tissue = None
        self._section.manual_landmarks = None
        self._notify()
        self._refresh_markers()
        self._refresh_images()

    def _reset_transform(self) -> None:
        if not self._confirm(
            "Reset transform",
            f"Drop the manual transform on section {self._section.index} and go back "
            f"to the registered overlay?",
        ):
            return
        self._section.manual_affine = None
        self._section.manual_landmarks = None
        self._pairs.clear()
        self._pending_tissue = None
        self._notify()
        self._refresh_markers()
        self._refresh_images()

    def _apply_landmarks(self) -> None:
        from atlastrack.project.schema import ManualLandmarks

        pairs = self._complete_pairs()
        if len(pairs) < _MIN_PAIRS:
            # Status line, not a message box: the Apply button is already disabled
            # below this count, so this is a backstop, and a modal box with no one
            # to click it is what hung the GUI tests once before (see _info_merge).
            self._status.setText(
                f"Complete at least {_MIN_PAIRS} pairs before warping "
                f"({len(pairs)} so far)."
            )
            return
        self._section.manual_landmarks = ManualLandmarks(
            source=[[s[0], s[1]] for s, _ in pairs],
            target=[[t[0], t[1]] for _, t in pairs],
        )
        self._section.manual_affine = None  # landmarks take precedence
        self._notify()
        self._status.setText(
            f"Warped section {self._section.index} through {len(pairs)} pair(s)."
        )
        self._refresh_images()

    def _notify(self) -> None:
        """Hand the expensive part back to the Register panel."""
        if self._on_section_changed is not None:
            try:
                self._on_section_changed(self._section)
            except Exception as exc:  # a redraw failure must not close the dialog
                self._status.setText(f"Applied, but the redraw failed: {exc}")

    # ------------------------------------------------------------- helpers

    def _confirm(self, title: str, text: str) -> bool:
        """Ask before destroying work.

        Modal on purpose, and the one blocking call here - tests patch this method
        rather than clicking it.
        """
        reply = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def _on_overlay_toggled(self, on: bool) -> None:
        self._opacity.setEnabled(on)
        self._refresh_images()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        for pane in (self._hist_pane, self._atlas_pane):
            pane.fit()
