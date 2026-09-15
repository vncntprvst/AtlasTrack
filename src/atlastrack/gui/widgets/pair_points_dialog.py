"""Place landmark correspondences with the section and the atlas side by side.

**Why a separate window rather than a mode in the napari canvas.** The canvas draws
the atlas overlay *on top of* the tissue in one coordinate space, so a click there
cannot say which of the two it meant. The first attempt asked the user to click
"the atlas, then the tissue" in that shared space, which is not something the
interface could show or the user could verify - there was no way to tell what a
click had been taken as. Two panes remove the ambiguity by construction: the pane
you click in *is* the answer.

Both panes are drawn at the **section crop's** resolution, and the atlas slice is
resampled to that same shape. Scene coordinates in either pane are therefore
section-local pixels already - the exact frame ``ManualLandmarks`` stores - so no
mapping is needed between what is clicked and what is saved.
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

# Shared with the Atlas matcher on purpose: one way to turn an array into a
# pixmap, window a section crop and draw region edges, so the two windows cannot
# drift into showing the same slide differently.
from atlastrack.gui.widgets.atlas_matcher import (
    _display_histology,
    _display_reference,
    _edges_pixmap,
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

#: Marker radius in scene (section) pixels, and the colours for each side.
_MARKER_R = 7.0
_ATLAS_COLOR = "#ff5f5f"
_TISSUE_COLOR = "#5fd35f"
_PENDING_COLOR = "#ffd23f"

#: At least this many pairs before a thin-plate spline is worth solving. Matches
#: the check the Register panel applies to dragged landmarks.
_MIN_PAIRS = 4


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

    def add_marker(self, x: float, y: float, label: str, color: str) -> None:
        """A ring plus its pair number, drawn above the image layers."""
        pen = QPen(QColor(color))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)  # constant on screen, so zoom does not fatten it
        ring = self.scene().addEllipse(
            x - _MARKER_R, y - _MARKER_R, 2 * _MARKER_R, 2 * _MARKER_R, pen, QBrush(Qt.NoBrush)
        )
        ring.setZValue(10)
        self._markers.append(ring)

        text = self.scene().addSimpleText(label, QFont("", 9))
        text.setBrush(QBrush(QColor(color)))
        text.setPos(x + _MARKER_R, y - _MARKER_R * 2)
        text.setFlag(text.GraphicsItemFlag.ItemIgnoresTransformations, True)
        text.setZValue(11)
        self._markers.append(text)


class PairPointsDialog(QDialog):
    """Pair atlas features with tissue features for one section.

    Owns nothing expensive: it writes ``Section.manual_landmarks`` and
    ``Section.plane`` and then hands back to the Register panel through
    ``on_section_changed`` for the re-render / probe re-map / save, so there is
    exactly one implementation of that step.
    """

    def __init__(
        self,
        state: WorkflowState,
        section: Section,
        *,
        on_section_changed: Callable[[Section], None] | None = None,
        bregma_ap_um: float = 0.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._section = section
        self._on_section_changed = on_section_changed
        self._bregma_ap_um = float(bregma_ap_um)

        # Pairs held as section-local (x, y). ``_pending`` is the half-finished
        # one: which pane it came from decides which half of the pair it fills.
        self._pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._pending: tuple[str, tuple[float, float]] | None = None

        self._crop: np.ndarray | None = None
        self._annotation: np.ndarray | None = None
        self._updating = False

        self.setWindowTitle(f"Pair points - section {section.index}")
        self.setModal(False)  # a modal dialog would block the viewer behind it
        self.resize(1100, 680)
        self._build_ui()
        self._load_existing_landmarks()
        self._refresh_images(fit=True)
        self._refresh_markers()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        hint = QLabel(
            "Click a feature in one pane, then the same feature in the other. "
            "The pane you click in says which side it is - order does not matter."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        panes = QHBoxLayout()
        self._hist_pane = _PickPane()
        self._atlas_pane = _PickPane()
        for title, pane in (("Section (tissue)", self._hist_pane), ("Atlas", self._atlas_pane)):
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

        self._atlas_outlines = QCheckBox("Atlas outlines")
        self._atlas_outlines.setChecked(True)
        self._atlas_outlines.setToolTip("Region boundaries drawn on the atlas pane.")
        self._atlas_outlines.toggled.connect(lambda _: self._refresh_images())
        row.addWidget(self._atlas_outlines)

        self._overlay_check = QCheckBox("Atlas over tissue")
        self._overlay_check.setToolTip(
            "Draw the atlas slice on top of the section, to judge the fit directly.\n"
            "Off by default: it hides the tissue you are trying to click."
        )
        self._overlay_check.toggled.connect(self._on_overlay_toggled)
        row.addWidget(self._overlay_check)

        self._overlay_outlines = QCheckBox("with outlines")
        self._overlay_outlines.setChecked(True)
        self._overlay_outlines.setEnabled(False)
        self._overlay_outlines.toggled.connect(lambda _: self._refresh_images())
        row.addWidget(self._overlay_outlines)

        row.addWidget(QLabel("Opacity:"))
        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.setValue(45)
        self._opacity.setEnabled(False)
        self._opacity.setFixedWidth(120)
        self._opacity.setToolTip("How strongly the atlas is painted over the tissue.")
        self._opacity.valueChanged.connect(lambda _: self._refresh_images())
        row.addWidget(self._opacity)
        row.addStretch()
        return box

    def _build_plane_box(self) -> QGroupBox:
        box = QGroupBox("Atlas plane (fine-tune)")
        box.setToolTip(
            "The plane first comes from the Atlas tab; this is where to nudge it "
            "against the tissue you are pairing against.\n"
            "Moving these re-slices the atlas immediately - it does not re-register "
            "the section until you press 'Apply plane'."
        )
        grid = QGridLayout(box)

        self._ap_spin = QDoubleSpinBox()
        self._ap_spin.setRange(-20000.0, 20000.0)
        self._ap_spin.setSingleStep(25.0)
        self._ap_spin.setSuffix(" µm")
        self._ap_spin.setToolTip("AP position of the section centre, relative to bregma.")
        grid.addWidget(QLabel("AP from bregma:"), 0, 0)
        grid.addWidget(self._ap_spin, 0, 1)

        self._ml_spin = QDoubleSpinBox()
        self._ml_spin.setRange(-30.0, 30.0)
        self._ml_spin.setSingleStep(0.5)
        self._ml_spin.setSuffix(" °")
        self._ml_spin.setToolTip("Tilt about the DV axis: the medial edge moves anterior.")
        grid.addWidget(QLabel("ML tilt:"), 0, 2)
        grid.addWidget(self._ml_spin, 0, 3)

        self._dv_spin = QDoubleSpinBox()
        self._dv_spin.setRange(-30.0, 30.0)
        self._dv_spin.setSingleStep(0.5)
        self._dv_spin.setSuffix(" °")
        self._dv_spin.setToolTip("Tilt about the ML axis: the dorsal edge moves anterior.")
        grid.addWidget(QLabel("DV tilt:"), 0, 4)
        grid.addWidget(self._dv_spin, 0, 5)

        for spin in (self._ap_spin, self._ml_spin, self._dv_spin):
            spin.valueChanged.connect(self._on_plane_spin_changed)

        apply_plane = QPushButton("Apply plane")
        apply_plane.setToolTip(
            "Write these values to the section (marked as a manual AP).\n"
            "Re-running the registration itself is still 'Register all sections'."
        )
        apply_plane.clicked.connect(self._apply_plane)
        grid.addWidget(apply_plane, 0, 6)

        self._plane_status = QLabel("")
        self._plane_status.setWordWrap(True)
        grid.addWidget(self._plane_status, 1, 0, 1, 7)
        return box

    def _build_pairs_box(self) -> QGroupBox:
        box = QGroupBox("Landmark pairs")
        row = QHBoxLayout(box)

        self._pairs_label = QLabel("")
        row.addWidget(self._pairs_label)
        row.addStretch()

        undo = QPushButton("Undo last pair")
        undo.clicked.connect(self._undo_pair)
        row.addWidget(undo)

        clear = QPushButton("Clear landmarks")
        clear.setToolTip("Remove every pair on this section, including saved ones.")
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
            f"Needs at least {_MIN_PAIRS}."
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

    def _current_plane(self):
        """PlaneParams reflecting the spin boxes, without writing to the section."""
        from atlastrack.project.schema import PlaneParams

        base = self._section.plane
        ap_abs = self._bregma_ap_um - float(self._ap_spin.value())
        if base is None:
            return PlaneParams(
                ap_um=ap_abs,
                ml_tilt_deg=float(self._ml_spin.value()),
                dv_tilt_deg=float(self._dv_spin.value()),
            )
        return base.model_copy(
            update={
                "ap_um": ap_abs,
                "ml_tilt_deg": float(self._ml_spin.value()),
                "dv_tilt_deg": float(self._dv_spin.value()),
            }
        )

    def _atlas_slice(self, out_shape: tuple[int, int]):
        """``(reference, annotation)`` at the plane the spin boxes describe."""
        from atlastrack.atlas.planes import anchoring_from_plane_params, resample_atlas_at_plane

        atlas = self._state.atlas
        if atlas is None:
            return None, None
        anchoring = anchoring_from_plane_params(atlas, self._current_plane())
        return resample_atlas_at_plane(atlas, anchoring, out_shape)

    def _refresh_images(self, *, fit: bool = False) -> None:
        crop = self._section_crop()
        if crop is None:
            self._plane_status.setText("This section's slide image is not loaded.")
            return
        self._crop = crop
        levels = self._section.levels or self._slide_levels()
        self._hist_pane.set_base(_to_pixmap(_display_histology(crop, levels)), fit=fit)

        reference, annotation = self._atlas_slice(crop.shape[:2])
        if reference is None:
            self._plane_status.setText("Load an atlas to see the atlas pane.")
            self._atlas_pane.set_base(_to_pixmap(np.zeros(crop.shape[:2], np.uint8)), fit=fit)
            self._atlas_pane.set_edges(None)
            self._hist_pane.set_overlay(None)
            return

        self._annotation = annotation
        self._atlas_pane.set_base(_to_pixmap(_display_reference(reference)), fit=fit)
        self._atlas_pane.set_edges(
            _edges_pixmap(annotation) if self._atlas_outlines.isChecked() else None
        )

        if self._overlay_check.isChecked():
            self._hist_pane.set_overlay(
                _to_pixmap(_display_reference(reference)), self._opacity.value() / 100.0
            )
            self._hist_pane.set_edges(
                _edges_pixmap(annotation) if self._overlay_outlines.isChecked() else None
            )
        else:
            self._hist_pane.set_overlay(None)
            self._hist_pane.set_edges(None)

        ap_bregma = self._ap_spin.value()
        self._plane_status.setText(
            f"Atlas sliced at AP {ap_bregma:+.0f} µm from bregma, "
            f"ML tilt {self._ml_spin.value():+.1f}°, DV tilt {self._dv_spin.value():+.1f}°."
        )

    def _slide_levels(self):
        try:
            return self._state.project.slides[self._section.slide_idx].levels
        except Exception:
            return None

    def _refresh_markers(self) -> None:
        self._hist_pane.clear_markers()
        self._atlas_pane.clear_markers()
        for n, (source, target) in enumerate(self._pairs, start=1):
            self._atlas_pane.add_marker(source[0], source[1], str(n), _ATLAS_COLOR)
            self._hist_pane.add_marker(target[0], target[1], str(n), _TISSUE_COLOR)
        if self._pending is not None:
            side, (x, y) = self._pending
            pane = self._atlas_pane if side == "atlas" else self._hist_pane
            pane.add_marker(x, y, "?", _PENDING_COLOR)

        n = len(self._pairs)
        waiting = ""
        if self._pending is not None:
            other = "tissue" if self._pending[0] == "atlas" else "atlas"
            waiting = f"   ·   waiting for the matching point on the {other}"
        self._pairs_label.setText(f"{n} pair(s){waiting}")
        self._apply_btn.setEnabled(n >= _MIN_PAIRS)

    # ------------------------------------------------------------- actions

    def _on_pane_clicked(self, side: str, x: float, y: float) -> None:
        """First click arms a side; the second, from the other pane, completes it."""
        if self._crop is None:
            return
        h, w = self._crop.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return  # outside the image: a stray click on the black surround
        if self._pending is None:
            self._pending = (side, (x, y))
        elif self._pending[0] == side:
            # Same pane twice: treat it as moving the pending point rather than
            # silently pairing a feature with itself.
            self._pending = (side, (x, y))
        else:
            first_side, first = self._pending
            source, target = (first, (x, y)) if first_side == "atlas" else ((x, y), first)
            self._pairs.append((source, target))
            self._pending = None
        self._refresh_markers()

    def _undo_pair(self) -> None:
        if self._pending is not None:
            self._pending = None
        elif self._pairs:
            self._pairs.pop()
        self._refresh_markers()

    def _load_existing_landmarks(self) -> None:
        """Show pairs already stored on the section, and the plane it already has."""
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
            self._pairs.append(((float(s[0]), float(s[1])), (float(t[0]), float(t[1]))))

    def _on_plane_spin_changed(self) -> None:
        if self._updating:
            return
        self._refresh_images()

    def _apply_plane(self) -> None:
        self._section.plane = self._current_plane()
        self._section.ap_source = "manual"
        self._notify()
        self._plane_status.setText(
            f"Plane written to section {self._section.index}. "
            f"Re-run 'Register all sections' to refit the atlas to the tissue."
        )

    def _clear_landmarks(self) -> None:
        if self._pairs and not self._confirm(
            "Clear landmarks",
            f"Remove all {len(self._pairs)} pair(s) from section {self._section.index}?",
        ):
            return
        self._pairs.clear()
        self._pending = None
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
        self._pending = None
        self._notify()
        self._refresh_markers()
        self._refresh_images()

    def _apply_landmarks(self) -> None:
        from atlastrack.project.schema import ManualLandmarks

        if len(self._pairs) < _MIN_PAIRS:
            # Status line, not a message box: the Apply button is already disabled
            # below this count, so this is a backstop, and a modal box with no one
            # to click it is what hung the GUI tests once before (see _info_merge).
            self._plane_status.setText(
                f"Place at least {_MIN_PAIRS} pairs before warping "
                f"({len(self._pairs)} so far)."
            )
            return
        self._section.manual_landmarks = ManualLandmarks(
            source=[[s[0], s[1]] for s, _ in self._pairs],
            target=[[t[0], t[1]] for _, t in self._pairs],
        )
        self._section.manual_affine = None  # landmarks take precedence
        self._notify()
        self._plane_status.setText(
            f"Warped section {self._section.index} through {len(self._pairs)} pair(s)."
        )

    def _notify(self) -> None:
        """Hand the expensive part back to the Register panel."""
        if self._on_section_changed is not None:
            try:
                self._on_section_changed(self._section)
            except Exception as exc:  # a redraw failure must not close the dialog
                self._plane_status.setText(f"Applied, but the redraw failed: {exc}")

    # ------------------------------------------------------------- helpers

    def _confirm(self, title: str, text: str) -> bool:
        """Ask before destroying work. Modal on purpose, and the one blocking call
        here - tests patch this method rather than clicking it."""
        reply = QMessageBox.question(
            self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def _on_overlay_toggled(self, on: bool) -> None:
        self._overlay_outlines.setEnabled(on)
        self._opacity.setEnabled(on)
        self._refresh_images()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        for pane in (self._hist_pane, self._atlas_pane):
            pane.fit()
