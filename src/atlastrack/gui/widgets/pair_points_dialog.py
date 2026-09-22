"""Place landmark correspondences with the section and the atlas side by side.

**Why a separate window rather than a mode in the napari canvas.** The canvas draws
the atlas overlay *on top of* the tissue in one coordinate space, so a click there
cannot say which of the two it meant, and nothing on screen could show what it had
been taken as. Two panes remove the ambiguity by construction: the pane a marker
lives in *is* the side it belongs to.

**Both panes show the registration you already ran.** The atlas pane is the
registered atlas as it currently sits on this section - not a fresh coronal slice
of the raw atlas. That matters twice over: it is what makes the outline line up
with the tissue at all, and it is the frame ``ManualLandmarks`` is defined in -
``source`` is a position *on the registered overlay*, ``target`` where it should
have been.

**Every pair is complete from the moment it exists.** Auto-placement drops each
atlas point and its tissue counterpart at the *same* coordinate - a zero
displacement, which warps nothing - and the user drags the tissue dot onto the real
feature. The earlier design left the tissue side empty and made the user answer the
points one at a time in a fixed order; a single mis-click then had nowhere to go but
into the warp, and the outline tore. Anything can be dragged at any time, in either
pane, in any order.

Numbering follows the section rather than the order the algorithm happened to emit:
outer ring first, counter-clockwise, then inward. Working through "1, 2, 3..." then
walks steadily round the tissue instead of hopping across it.
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
#: count as a click rather than a pan or a drag.
_CLICK_SLOP_PX = 4.0

#: Grab radius for a marker, in **view** pixels, so a dot is equally easy to catch
#: at any zoom.
_GRAB_PX = 12.0

#: Marker radius in scene (section) pixels, and the colours for each role.
#:
#: **Filled discs with outlined numbers, not hollow rings with plain text.** A thin
#: ring over a busy green outline and a same-coloured number beside it disappears
#: into the anatomy at working zoom - which is exactly what happened. A solid disc
#: reads as a marker at any size, and white text carrying a black outline stays
#: legible over white matter, black background and green boundaries alike, so it
#: never has to be guessed at from context.
_MARKER_R = 6.0
_SELECTED_R = 9.0
_ATLAS_COLOR = "#ff5252"
_TISSUE_COLOR = "#2fe36a"
_MOVED_COLOR = "#ffc400"
_SELECT_COLOR = "#ffffff"

#: Number size, in points. Screen-constant (the item ignores view transforms), so
#: this is a real on-screen size rather than something that shrinks as you zoom out.
_NUMBER_PT = 11

#: Atlas outline colour, and the dim fill under it so the pane has a silhouette to
#: orient by rather than lines floating on black.
_EDGE_RGB = (90, 230, 120)
_EXTENT_GREY = 48

#: At least this many pairs before a thin-plate spline is worth solving.
_MIN_PAIRS = 4

#: Auto-placed atlas points, via the same helper "Place landmarks" uses.
_AUTO_MAX_POINTS = 12

#: Below this many *moved* pairs a warp preview is not meaningful.
_MIN_PREVIEW_PAIRS = 3

#: A tissue dot still sitting on its atlas coordinate contributes nothing to the
#: warp; this is how far it must move before it counts as answered.
_MOVED_EPS_PX = 0.75

#: Points beyond this fraction of the largest radius count as the outer ring, for
#: numbering. Loose on purpose - it only decides the order dots are labelled in.
_OUTER_RING_FRAC = 0.62


def spatial_order(points: np.ndarray) -> list[int]:
    """Indices of ``points`` ordered outer-ring-first, counter-clockwise, then in.

    ``salient_landmarks`` emits in priority order - silhouette tips, then whichever
    region junctions survived the spread - which is arbitrary on the page. Numbering
    that order makes the user hop from one side of the section to the other and back.
    Walking the outer ring first and then the interior keeps consecutive numbers
    close together, which is the whole value of numbering them.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return list(range(len(pts)))
    centre = pts.mean(axis=0)
    delta = pts - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    # Screen y grows downward; negate it so "counter-clockwise" means what it looks
    # like on screen rather than what it is in matrix coordinates.
    angle = np.arctan2(-delta[:, 1], delta[:, 0])
    largest = float(radius.max()) or 1.0
    outer = radius >= _OUTER_RING_FRAC * largest

    def ring(mask):
        idx = np.nonzero(mask)[0]
        return sorted(idx.tolist(), key=lambda i: float(angle[i]))

    return ring(outer) + ring(~outer)


class _PickPane(_ImagePane):
    """An image pane whose markers can be grabbed and dragged.

    Left-drag still pans the view, so the pane has to decide on press whether the
    pointer landed on a marker: if it did, panning is switched off for the duration
    of that drag and re-armed on release.
    """

    clicked = Signal(float, float)  # scene x, y, on empty space
    picked = Signal(int)  # a marker was clicked without being dragged
    drag_started = Signal(int)  # pair index
    dragged = Signal(int, float, float)  # pair index, scene x, y
    drag_finished = Signal()
    delete_requested = Signal(int)  # right-click on a marker
    delete_pressed = Signal()  # Delete / Backspace while this pane has focus

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._press_pos = None
        self._markers: list = []
        self._handles: list[tuple[int, float, float]] = []
        self._drag_index: int | None = None
        self._drag_moved = False
        self.setFocusPolicy(Qt.StrongFocus)  # so Delete reaches keyPressEvent

    # -- geometry ------------------------------------------------------

    def set_handles(self, handles: list[tuple[int, float, float]]) -> None:
        """Grabbable positions, as ``(pair index, x, y)`` in scene coordinates."""
        self._handles = list(handles)

    def _handle_at(self, scene_x: float, scene_y: float) -> int | None:
        """The pair whose marker is under the pointer, or None."""
        if not self._handles:
            return None
        scale = abs(self.transform().m11()) or 1.0
        reach = (_GRAB_PX / scale) ** 2
        best, best_d2 = None, reach
        for index, hx, hy in self._handles:
            d2 = (hx - scene_x) ** 2 + (hy - scene_y) ** 2
            if d2 <= best_d2:
                best, best_d2 = index, d2
        return best

    # -- events --------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        point = self.mapToScene(event.pos())
        index = self._handle_at(point.x(), point.y())
        if event.button() == Qt.RightButton and index is not None:
            self.delete_requested.emit(index)
            return
        if event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            self._press_pos = event.pos()
            if index is not None:
                self._drag_index = index
                self._drag_moved = False
                self.setDragMode(_ImagePane.NoDrag)
                self.drag_started.emit(index)
                return  # swallow it: this is a grab, not a pan
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_index is not None:
            self._drag_moved = True
            point = self.mapToScene(event.pos())
            self.dragged.emit(self._drag_index, float(point.x()), float(point.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_index is not None and event.button() == Qt.LeftButton:
            index, moved = self._drag_index, self._drag_moved
            self._drag_index = None
            self._press_pos = None
            self.setDragMode(_ImagePane.ScrollHandDrag)
            if moved:
                self.drag_finished.emit()
            else:
                # Pressed and released on a marker without moving it: that is a
                # selection, so Delete has something to act on.
                self.picked.emit(index)
            return
        super().mouseReleaseEvent(event)
        if event.button() != Qt.LeftButton or self._press_pos is None:
            return
        travelled = (event.pos() - self._press_pos).manhattanLength()
        self._press_pos = None
        if travelled <= _CLICK_SLOP_PX:
            point = self.mapToScene(event.pos())
            self.clicked.emit(float(point.x()), float(point.y()))

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_pressed.emit()
            return
        super().keyPressEvent(event)

    # -- markers -------------------------------------------------------

    def clear_markers(self) -> None:
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers.clear()

    def add_marker(
        self, x: float, y: float, label: str, color: str, *, selected: bool = False
    ) -> None:
        """A filled disc and its number, drawn above the image layers."""
        radius = _SELECTED_R if selected else _MARKER_R

        if selected:
            # A white collar, so the selected dot is obvious against any colour.
            collar = QPen(QColor(_SELECT_COLOR))
            collar.setWidthF(2.5)
            collar.setCosmetic(True)
            halo = self.scene().addEllipse(
                x - radius - 3, y - radius - 3, 2 * (radius + 3), 2 * (radius + 3),
                collar, QBrush(Qt.NoBrush),
            )
            halo.setZValue(9)
            self._markers.append(halo)

        edge = QPen(QColor(0, 0, 0, 200))
        edge.setWidthF(1.2)
        edge.setCosmetic(True)  # constant on screen, so zoom does not fatten it
        disc = self.scene().addEllipse(
            x - radius, y - radius, 2 * radius, 2 * radius, edge, QBrush(QColor(color))
        )
        disc.setZValue(10)
        self._markers.append(disc)

        font = QFont("", _NUMBER_PT)
        font.setBold(True)
        pos_x, pos_y = x + radius, y - radius * 2.2

        # The halo is a **separate, fatter copy drawn underneath**, not an outline
        # pen on the glyph. A pen is centred on the glyph path, so at this size it
        # eats the fill and the number comes out dark with a pale rim - the very
        # thing that made these hard to read. Two passes keep the fill pure white.
        halo = self.scene().addSimpleText(label, font)
        halo_pen = QPen(QColor(0, 0, 0, 235))
        halo_pen.setWidthF(3.5)
        halo_pen.setCosmetic(True)
        halo.setPen(halo_pen)
        halo.setBrush(QBrush(QColor(0, 0, 0, 235)))
        halo.setPos(pos_x, pos_y)
        halo.setFlag(halo.GraphicsItemFlag.ItemIgnoresTransformations, True)
        halo.setZValue(11)
        self._markers.append(halo)

        text = self.scene().addSimpleText(label, font)
        text.setPen(QPen(Qt.NoPen))
        text.setBrush(QBrush(QColor(_SELECT_COLOR)))
        text.setPos(pos_x, pos_y)
        text.setFlag(text.GraphicsItemFlag.ItemIgnoresTransformations, True)
        text.setZValue(12)
        self._markers.append(text)

    def add_link(self, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
        """Faint line from where a dot started to where it has been dragged."""
        pen = QPen(QColor(color))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        pen.setStyle(Qt.DashLine)
        line = self.scene().addLine(x0, y0, x1, y1, pen)
        line.setZValue(9)
        self._markers.append(line)


class PairPointsDialog(QDialog):
    """Pair atlas features with tissue features for one section.

    Owns nothing expensive: it writes ``Section.manual_landmarks`` and
    ``Section.plane``, then hands back to the Register panel through
    ``on_section_changed`` for the re-render / probe re-map / save. ``warp_labels``
    is the panel's own ``_warp_labels_for``, reused so the overlay here and the
    overlay on the canvas cannot come from two different code paths.
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

        # Each entry is ``[source, target]``, both always set. A freshly placed pair
        # has them equal - a zero displacement, which warps nothing and doubles as an
        # anchor holding that part of the atlas still while other points are moved.
        self._pairs: list[list[tuple[float, float]]] = []
        # Whole-list snapshots, because "undo" has to reverse a drag as readily as an
        # insertion, and a drag is a continuous stream of positions rather than one
        # invertible step.
        self._history: list[list[list[tuple[float, float]]]] = []

        self._crop: np.ndarray | None = None
        self._base_labels: np.ndarray | None = None
        self._selected: int | None = None
        self._dragging = False
        self._updating = False

        self.setWindowTitle(f"Pair points - section {section.index}")
        self.setModal(False)  # a modal dialog would block the viewer behind it
        self.resize(1150, 760)
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
            ("Section (tissue) - drag each dot onto its feature",
             self._hist_pane),
            ("Atlas as currently registered", self._atlas_pane),
        ):
            box = QVBoxLayout()
            box.addWidget(QLabel(title))
            box.addWidget(pane, stretch=1)
            wrap = QWidget()
            wrap.setLayout(box)
            panes.addWidget(wrap, stretch=1)
        outer.addLayout(panes, stretch=1)

        for pane, role in ((self._hist_pane, "target"), (self._atlas_pane, "source")):
            pane.clicked.connect(lambda x, y: self._add_pair_at(x, y))
            pane.picked.connect(self._select_pair)
            pane.drag_started.connect(lambda _i: self._snapshot())
            pane.dragged.connect(
                lambda i, x, y, role=role: self._move_point(role, i, x, y)
            )
            pane.drag_finished.connect(self._on_drag_finished)
            pane.delete_requested.connect(self._delete_pair)
            pane.delete_pressed.connect(self._delete_selected)

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
            "Bend the overlay through the pairs as they are dragged, instead of "
            "waiting for Apply."
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

        auto = QPushButton("Auto-place points")
        auto.setToolTip(
            "Drop points on distinctive atlas features - outline tips, region "
            "junctions, corners - with each tissue dot starting on top of its atlas "
            "twin.\nDrag the tissue dots onto the real features, in any order."
        )
        auto.clicked.connect(self._auto_place)
        row.addWidget(auto)

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setToolTip("Step back one change, including a drag.")
        self._undo_btn.clicked.connect(self._undo)
        row.addWidget(self._undo_btn)

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
            f"Needs at least {_MIN_PAIRS} pairs."
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

        Without the TPS on purpose: stored source points live in that frame, so new
        ones have to be picked in it too.
        """
        if self._warp_labels is None:
            return None
        try:
            return self._warp_labels(self._section, apply_landmarks=False)
        except Exception:
            return None

    def _moved(self, pair) -> bool:
        (sx, sy), (tx, ty) = pair
        return abs(tx - sx) > _MOVED_EPS_PX or abs(ty - sy) > _MOVED_EPS_PX

    def _moved_count(self) -> int:
        return sum(1 for p in self._pairs if self._moved(p))

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
        """``base`` bent through every pair, for the live overlay.

        Every pair, not just the dragged ones: a pair still sitting on its atlas
        coordinate pins that spot, which is what stops one mis-dragged dot dragging
        the whole outline with it.
        """
        if not self._preview_check.isChecked():
            return base
        if len(self._pairs) < _MIN_PAIRS or self._moved_count() < _MIN_PREVIEW_PAIRS:
            return base
        try:
            from atlastrack.registration.landmarks_warp import warp_label_image

            source = np.array([s for s, _ in self._pairs], dtype=float)
            target = np.array([t for _, t in self._pairs], dtype=float)
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

        moved = self._moved_count()
        previewing = (
            self._preview_check.isChecked()
            and len(self._pairs) >= _MIN_PAIRS
            and moved >= _MIN_PREVIEW_PAIRS
        )
        self._status.setText(
            f"Showing the registration already computed for section "
            f"{self._section.index}."
            + (" Overlay previews the warp." if previewing else "")
        )

    def _slide_levels(self):
        try:
            return self._state.project.slides[self._section.slide_idx].levels
        except Exception:
            return None

    def _refresh_markers(self) -> None:
        for pane in (self._hist_pane, self._atlas_pane):
            pane.clear_markers()

        for i, (source, target) in enumerate(self._pairs):
            label = str(i + 1)
            moved = self._moved((source, target))
            chosen = i == self._selected
            self._atlas_pane.add_marker(
                source[0], source[1], label, _ATLAS_COLOR, selected=chosen
            )
            self._hist_pane.add_marker(
                target[0], target[1], label,
                _TISSUE_COLOR if moved else _MOVED_COLOR, selected=chosen,
            )
            if moved:
                # Where it started and where it is now, so the displacement each
                # pair contributes is visible rather than inferred.
                self._hist_pane.add_link(source[0], source[1], target[0], target[1],
                                         _TISSUE_COLOR)

        self._atlas_pane.set_handles([(i, s[0], s[1]) for i, (s, _t) in enumerate(self._pairs)])
        self._hist_pane.set_handles([(i, t[0], t[1]) for i, (_s, t) in enumerate(self._pairs)])

        total, moved = len(self._pairs), self._moved_count()
        self._pairs_label.setText(f"{total} pair(s)   -   {moved} moved")
        self._apply_btn.setEnabled(total >= _MIN_PAIRS)
        self._undo_btn.setEnabled(bool(self._history))
        self._update_hint(total, moved)

    #: One line, always on screen, saying what the mouse and keyboard do here.
    _CONTROLS = (
        "Click empty space to add a pair  ·  drag a dot to move it  ·  "
        "click a dot then Delete, or right-click it, to remove it."
    )

    def _update_hint(self, total: int, moved: int) -> None:
        if total == 0:
            lead = (
                "Press 'Auto-place points', then drag each dot on the LEFT onto the "
                "feature it marks on the RIGHT."
            )
        elif moved < total:
            lead = (
                f"{total - moved} dot(s) still sit on their atlas position (amber) "
                f"and hold the atlas still there. Drag the ones you can identify, in "
                f"any order."
            )
        else:
            lead = "All pairs moved. Drag any dot to adjust, or 'Apply landmark warp'."
        self._hint.setText(f"{lead}\n{self._CONTROLS}")

    # ------------------------------------------------------------- editing

    def _snapshot(self) -> None:
        """Remember the current pairs so the next change can be stepped back."""
        self._history.append([[tuple(s), tuple(t)] for s, t in self._pairs])

    def _add_pair_at(self, x: float, y: float) -> None:
        """A click on empty canvas adds a pair, both halves on that coordinate."""
        if self._crop is None:
            return
        h, w = self._crop.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return  # outside the image: a stray click on the black surround
        self._snapshot()
        self._pairs.append([(x, y), (x, y)])
        self._reorder_pairs()
        self._selected = None  # renumbering invalidates an index-based selection
        self._refresh_markers()
        self._refresh_images()

    def _select_pair(self, index: int) -> None:
        """Clicking a dot selects the pair, so Delete knows what to remove."""
        self._selected = index if 0 <= index < len(self._pairs) else None
        self._refresh_markers()

    def _delete_selected(self) -> None:
        if self._selected is not None:
            self._delete_pair(self._selected)

    def _delete_pair(self, index: int) -> None:
        """Remove one pair. Both halves go: a correspondence is not half a thing."""
        if not (0 <= index < len(self._pairs)):
            return
        self._snapshot()
        del self._pairs[index]
        self._selected = None
        self._refresh_markers()
        self._refresh_images()

    def _move_point(self, role: str, index: int, x: float, y: float) -> None:
        if not (0 <= index < len(self._pairs)) or self._crop is None:
            return
        h, w = self._crop.shape[:2]
        x = float(min(max(x, 0.0), w - 1))
        y = float(min(max(y, 0.0), h - 1))
        self._dragging = True
        self._pairs[index][0 if role == "source" else 1] = (x, y)
        self._refresh_markers()

    def _on_drag_finished(self) -> None:
        """Recompute the (expensive) warp preview once, at the end of a drag."""
        self._dragging = False
        self._refresh_images()

    def _reorder_pairs(self) -> None:
        """Renumber outer-ring-first, counter-clockwise, by the atlas positions."""
        if len(self._pairs) < 2:
            return
        order = spatial_order(np.array([s for s, _t in self._pairs], dtype=float))
        self._pairs = [self._pairs[i] for i in order]

    def _auto_place(self) -> None:
        if self._base_labels is None:
            self._status.setText("No registered atlas to place points on.")
            return
        from atlastrack.registration.landmarks_warp import salient_landmarks

        points = np.asarray(
            salient_landmarks(self._base_labels, max_points=_AUTO_MAX_POINTS), dtype=float
        ).reshape(-1, 2)
        self._snapshot()
        for x, y in points[spatial_order(points)]:
            self._pairs.append([(float(x), float(y)), (float(x), float(y))])
        self._status.setText(
            f"Placed {len(points)} pairs. Each tissue dot starts on its atlas "
            f"position, so nothing is warped until you drag it."
        )
        self._refresh_markers()
        self._refresh_images()

    def _undo(self) -> None:
        if not self._history:
            return
        self._pairs = self._history.pop()
        self._selected = None
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
        self._reorder_pairs()

    # ------------------------------------------------------------- actions

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
            f"Remove all {len(self._pairs)} pair(s) from section {self._section.index}?",
        ):
            return
        self._snapshot()
        self._pairs.clear()
        self._selected = None
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
        self._snapshot()
        self._section.manual_affine = None
        self._section.manual_landmarks = None
        self._pairs.clear()
        self._selected = None
        self._notify()
        self._refresh_markers()
        self._refresh_images()

    def _apply_landmarks(self) -> None:
        from atlastrack.project.schema import ManualLandmarks

        if len(self._pairs) < _MIN_PAIRS:
            # Status line, not a message box: the Apply button is already disabled
            # below this count, so this is a backstop, and a modal box with no one
            # to click it is what hung the GUI tests once before (see _info_merge).
            self._status.setText(
                f"Place at least {_MIN_PAIRS} pairs before warping "
                f"({len(self._pairs)} so far)."
            )
            return
        if self._moved_count() == 0:
            self._status.setText(
                "Every dot is still on its atlas position, so this warp would do "
                "nothing. Drag the tissue dots onto their features first."
            )
            return
        self._section.manual_landmarks = ManualLandmarks(
            source=[[s[0], s[1]] for s, _ in self._pairs],
            target=[[t[0], t[1]] for _, t in self._pairs],
        )
        self._section.manual_affine = None  # landmarks take precedence
        self._notify()
        self._status.setText(
            f"Warped section {self._section.index} through {len(self._pairs)} pair(s), "
            f"{self._moved_count()} of them moved."
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

    def keyPressEvent(self, event) -> None:
        from qtpy.QtCore import Qt as _Qt

        if event.key() in (_Qt.Key_Delete, _Qt.Key_Backspace):
            self._delete_selected()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        for pane in (self._hist_pane, self._atlas_pane):
            pane.fit()
