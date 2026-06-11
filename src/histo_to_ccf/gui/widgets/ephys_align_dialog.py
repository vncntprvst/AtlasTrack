"""Ephys alignment dialog: warp the LFP power map onto the histology track.

Layout mirrors the IBL ephys-alignment GUI: the LFP depth x frequency power map
on the left, the atlas region colour strip (with labels) on the right, sharing a
single vertical *track depth* axis (0 = shank tip at the bottom). The LFP map is
shown warped into track space by the current anchor set, so when the alignment is
right its power transitions line up with the region boundaries on the strip.

Anchors are draggable horizontal handles. Each handle pins one LFP *feature
depth* to a *track depth* (its current y); dragging it re-warps the LFP map.
"Apply" places every channel on the tip->entry line and stores the per-channel
CCF coordinates on the shank.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.ephys.alignment import apply_depth_alignment, channel_ccf_um, invert_anchors
from histo_to_ccf.ephys.features import power_image
from histo_to_ccf.ephys.regions import region_strip_image, regions_at_ccf
from histo_to_ccf.gui.widgets.atlas_matcher import _to_pixmap
from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    from histo_to_ccf.project.schema import Shank

_DISPLAY_H = 600  # scene height in pixels (track-depth axis)
_IMG_W = 320  # LFP map width in pixels
_GAP = 10
_STRIP_W = 28
_LEFT = 86  # left margin for the depth / channel axis labels
_LABEL_W = 168  # right margin for region acronym labels beside the strip


def _cluster_x_into_shanks(x: np.ndarray, n_shanks: int) -> np.ndarray | None:
    """Group channels into ``n_shanks`` by their x position (gap-based clustering).

    Fallback when the probe carries no shank ids. The between-shank x gap (≈250 µm
    on Neuropixels 2.0) is much larger than the within-shank column gap (≈32 µm),
    so the ``n_shanks - 1`` largest gaps in the sorted unique x mark the shank
    boundaries. Returns a per-channel group id in ``[0, n_shanks)``, or ``None`` if
    it can't form that many groups.
    """
    if n_shanks <= 1:
        return None
    ux = np.unique(np.round(np.asarray(x, dtype=float), 1))
    if ux.size < n_shanks:
        return None
    gaps = np.diff(ux)
    cut = np.sort(gaps)[-(n_shanks - 1)]  # threshold = (n_shanks-1)-th largest gap
    # Boundaries sit at the midpoints of the large gaps.
    bounds = [(ux[i] + ux[i + 1]) / 2.0 for i in range(gaps.size) if gaps[i] >= cut]
    return np.searchsorted(np.asarray(bounds), np.asarray(x, dtype=float), side="right")


class _AnchorLine(QGraphicsLineItem):
    """A draggable horizontal handle pinning a feature depth to a track depth."""

    def __init__(self, dialog: "EphysAlignmentDialog", feature_depth: float, width: float):
        super().__init__(0, 0, width, 0)
        self._dialog = dialog
        self.feature_depth = feature_depth
        pen = QPen(QColor(255, 80, 80), 2)
        self.setPen(pen)
        self.setFlag(QGraphicsLineItem.ItemIsMovable, True)
        self.setFlag(QGraphicsLineItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsLineItem.ItemIsSelectable, True)
        self.setZValue(10)
        self.setCursor(Qt.SizeVerCursor)

    def itemChange(self, change, value):  # noqa: N802 (Qt signature)
        if change == QGraphicsLineItem.ItemPositionChange:
            # Constrain to vertical movement (x pinned to the image left edge),
            # clamped to the scene height.
            y = max(0.0, min(float(value.y()), float(_DISPLAY_H)))
            value.setX(float(_LEFT))
            value.setY(y)
            return value
        if change == QGraphicsLineItem.ItemPositionHasChanged:
            self._dialog._on_anchor_moved()
        return super().itemChange(change, value)


class _AlignView(QGraphicsView):
    """Graphics view that adds an anchor where the user double-clicks."""

    def __init__(self, scene, dialog: "EphysAlignmentDialog"):
        super().__init__(scene)
        self._dialog = dialog

    def mouseDoubleClickEvent(self, event):  # noqa: N802 (Qt signature)
        pt = self.mapToScene(event.pos())
        self._dialog._add_anchor_at_scene_y(pt.y())


class EphysAlignmentDialog(QDialog):
    """Warp the LFP power map onto a shank's histology track and store CCF."""

    def __init__(
        self,
        state: WorkflowState,
        probe_idx: int,
        shank_idx: int,
        lfp_result: dict,
        on_applied=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ephys alignment")
        self.resize(720, 720)
        self._state = state
        self._probe_idx = probe_idx
        self._shank_idx = shank_idx
        self._on_applied = on_applied
        self._handles: list[_AnchorLine] = []

        probe = state.project.probes[probe_idx]
        self._shank: "Shank" = probe.shanks[shank_idx]
        self._tip = self._shank.tip_ccf_um
        self._entry = self._shank.entry_ccf_um

        self._prepare_channels(lfp_result)
        self._build_ui()
        self._restore_anchors()
        self._render()

    # -- data prep -------------------------------------------------------

    def _prepare_channels(self, lfp_result: dict) -> None:
        """Select this shank's channels and build the feature-space power map."""
        depths = np.asarray(lfp_result["depths_um"], dtype=float)
        x = np.asarray(lfp_result["x_um"], dtype=float)
        img = np.asarray(lfp_result["image"])  # (n_channels, n_freq), uint8
        psd = np.asarray(lfp_result.get("psd", img), dtype=float)  # raw PSD for re-norm
        ids = list(lfp_result.get("channel_ids", list(range(len(depths)))))
        shank_ids = lfp_result.get("shank_ids")
        self._freqs = np.asarray(lfp_result.get("freqs", []), dtype=float)

        mask = self._shank_mask(depths, x, shank_ids)

        d = depths[mask]
        order = np.argsort(d)
        # ABSOLUTE distance along the shank from the tip (NOT re-zeroed): the lowest
        # recorded electrode usually sits above the physical tip, and the alignment
        # places electrodes at these depths above the histology tip.
        self._depths = d[order]
        self._psd = psd[mask][order]  # rows ascending feature depth (raw power)
        self._img_feat = img[mask][order]  # current display map (re-derived on toggle)
        masked_ids = [i for i, m in zip(ids, mask, strict=False) if m]
        self._channel_ids = [masked_ids[i] for i in order] if masked_ids else []
        self._stream = lfp_result.get("stream_name", "")

        # Channel / row structure - derived from the data, never hard-coded. A
        # Neuropixels 2.0 row holds 2 sites, so n_channels can be 2 x n_rows.
        self._n_channels = int(self._depths.size)
        uniq = np.unique(np.round(self._depths, 1)) if self._depths.size else np.array([])
        self._n_rows = int(uniq.size)
        self._sites_per_row = (self._n_channels / self._n_rows) if self._n_rows else 1.0
        self._row_pitch = float(np.median(np.diff(uniq))) if uniq.size > 1 else 0.0
        # Recorded electrode extent along the shank (µm from tip).
        self._rec_bottom = float(self._depths.min()) if self._depths.size else 0.0
        self._rec_top = float(self._depths.max()) if self._depths.size else 0.0

        self._insertion = 0.0
        if self._tip is not None and self._entry is not None:
            self._insertion = float(np.linalg.norm(np.array(self._entry) - np.array(self._tip)))
        # Track axis spans 0 (tip) .. track_max (surface); electrodes occupy only
        # [rec_bottom, rec_top], which is usually shorter than the histology track.
        self._track_max = max(self._rec_top, self._insertion, 1.0)

    def _shank_mask(self, depths: np.ndarray, x: np.ndarray, shank_ids) -> np.ndarray:
        """Boolean mask of this shank's channels.

        Prefers the probe's **shank ids** (correct for NP2.0, whose shank has two
        electrode columns - so splitting by unique x over-counts shanks and grabs a
        single column). Falls back to clustering x into the probe's shank count by
        the large between-shank gaps.
        """
        self._shank_x = None
        if shank_ids is not None:
            sids = np.asarray(shank_ids)
            uniq = sorted({str(s) for s in sids.tolist()})
            if len(uniq) > 1 and self._shank_idx < len(uniq):
                return np.array([str(s) == uniq[self._shank_idx] for s in sids])
            return np.ones(depths.shape, dtype=bool)
        n_shanks = len(self._state.project.probes[self._probe_idx].shanks)
        groups = _cluster_x_into_shanks(x, n_shanks)
        if groups is not None and self._shank_idx <= int(groups.max()):
            sel = groups == self._shank_idx
            if sel.any():
                self._shank_x = float(np.mean(x[sel]))
            return sel
        return np.ones(depths.shape, dtype=bool)

    def _recompute_image(self) -> None:
        """Rebuild the displayed power map from the raw PSD honouring the toggle."""
        per_freq = bool(self._per_freq_check.isChecked())
        self._img_feat = power_image(self._psd, per_freq=per_freq)
        self._render_lfp_only()

    # -- anchors <-> handles --------------------------------------------

    def _y_to_track(self, y: float) -> float:
        """Scene y (0 top) -> track depth µm (0 at tip, bottom of view)."""
        return float(self._track_max * (1.0 - y / _DISPLAY_H))

    def _track_to_y(self, track: float) -> float:
        return float(_DISPLAY_H * (1.0 - track / self._track_max))

    def anchors(self) -> list[tuple[float, float]]:
        """Current (feature_depth, track_depth) pairs from the handle positions."""
        return [(h.feature_depth, self._y_to_track(h.pos().y())) for h in self._handles]

    def add_anchor(self, feature_depth: float, track_depth: float) -> None:
        line = _AnchorLine(self, feature_depth, _IMG_W + _GAP + _STRIP_W)
        self._scene.addItem(line)
        line.setPos(float(_LEFT), self._track_to_y(track_depth))
        self._handles.append(line)
        self._render_lfp_only()

    def add_anchor_at_track(self, track_depth: float) -> None:
        """Add an anchor at a track depth, pinning the LFP feature shown there."""
        inv = invert_anchors(self.anchors())  # track -> feature
        feature = float(apply_depth_alignment(np.array([track_depth]), inv)[0])
        self.add_anchor(feature, track_depth)

    def _add_anchor_at_scene_y(self, y: float) -> None:
        y = max(0.0, min(float(y), float(_DISPLAY_H)))
        self.add_anchor_at_track(self._y_to_track(y))

    def clear_anchors(self) -> None:
        for h in self._handles:
            self._scene.removeItem(h)
        self._handles.clear()
        self._render_lfp_only()

    def _remove_selected(self) -> None:
        keep = []
        for h in self._handles:
            if h.isSelected():
                self._scene.removeItem(h)
            else:
                keep.append(h)
        self._handles = keep
        self._render_lfp_only()

    def _restore_anchors(self) -> None:
        if self._shank.ephys is not None and self._shank.ephys.anchors:
            for f, t in self._shank.ephys.anchors:
                self.add_anchor(float(f), float(t))
        elif self._depths.size:
            # Fresh shank: pre-set two anchors at the recorded electrode block
            # edges (identity), as draggable starting handles for the user.
            self.add_anchor(self._rec_bottom, self._rec_bottom)
            self.add_anchor(self._rec_top, self._rec_top)

    # -- rendering -------------------------------------------------------

    def _warp_lfp(self) -> np.ndarray:
        """LFP map warped into track space: (_DISPLAY_H, n_freq) uint8."""
        if self._img_feat.size == 0:
            return np.zeros((_DISPLAY_H, 1), dtype=np.uint8)
        n_ch = self._img_feat.shape[0]
        inv = invert_anchors(self.anchors())  # track -> feature
        rows = np.arange(_DISPLAY_H)
        track = self._track_max * (1.0 - rows / _DISPLAY_H)
        feat = apply_depth_alignment(track, inv)
        # feature depth -> source row index in the ascending-depth LFP map.
        src = np.interp(feat, self._depths, np.arange(n_ch))
        src = np.clip(np.round(src).astype(int), 0, n_ch - 1)
        return self._img_feat[src]

    def _region_strip(self):
        rows = np.arange(_DISPLAY_H)
        track = self._track_max * (1.0 - rows / _DISPLAY_H)
        atlas = self._state.atlas
        if atlas is None or self._tip is None or self._entry is None:
            return np.zeros((_DISPLAY_H, _STRIP_W, 3), dtype=np.uint8), []
        ccf = channel_ccf_um(self._tip, self._entry, track, [])  # track == physical depth
        hits = regions_at_ccf(atlas, ccf)
        return region_strip_image(hits, _DISPLAY_H, _STRIP_W), hits

    def _render_lfp_only(self) -> None:
        warped = self._warp_lfp()
        # Stretch the n_freq-wide map to _IMG_W for display.
        self._lfp_item.setPixmap(
            _to_pixmap(warped).scaled(_IMG_W, _DISPLAY_H)
        )

    def _render(self) -> None:
        self._render_lfp_only()
        strip, hits = self._region_strip()
        self._strip_item.setPixmap(_to_pixmap(strip))
        self._draw_region_labels(hits)      # acronyms beside the strip (track-space, fixed)
        self._draw_recorded_extent()        # bracket lines (warp-dependent)
        # Region label summary (distinct regions top->bottom).
        labels: list[str] = []
        for acr, _ in hits:
            if acr and (not labels or labels[-1] != acr):
                labels.append(acr)
        self._regions_label.setText("Regions (surface→tip): " + " · ".join(labels[:24]))

    def _draw_region_labels(self, hits: list) -> None:
        """Write each region's acronym (+ short name) beside its band on the strip."""
        for it in self._region_label_items:
            self._scene.removeItem(it)
        self._region_label_items = []
        if not hits:
            return
        x0 = _LEFT + _IMG_W + _GAP + _STRIP_W + 4
        start = 0
        for i in range(1, len(hits) + 1):
            if i == len(hits) or hits[i][0] != hits[start][0]:
                acr = hits[start][0]
                if acr and (i - start) >= 7:  # skip slivers too thin to label
                    mid_y = (start + i - 1) / 2.0
                    item = QGraphicsTextItem(self._region_caption(acr))
                    item.setDefaultTextColor(QColor(215, 215, 215))
                    item.setPos(x0, mid_y - 8)
                    item.setZValue(6)
                    self._scene.addItem(item)
                    self._region_label_items.append(item)
                start = i

    def _region_caption(self, acr: str) -> str:
        """``ACR — full name`` when the atlas knows the structure, else the acronym."""
        try:
            name = str(self._state.atlas.structures[acr]["name"])
        except Exception:
            name = ""
        if name and name.lower() != acr.lower():
            return f"{acr} - {name[:24]}"
        return acr

    def _draw_recorded_extent(self) -> None:
        """Dashed bracket lines where the recorded electrode block lands on the track."""
        for it in self._extent_items:
            self._scene.removeItem(it)
        self._extent_items = []
        if not self._depths.size:
            return
        edges = apply_depth_alignment(
            np.array([self._rec_bottom, self._rec_top]), self.anchors()
        )  # feature depth -> track depth, through the current anchors
        x_right = _LEFT + _IMG_W + _GAP + _STRIP_W
        pen = QPen(QColor(60, 220, 120), 1, Qt.DashLine)
        for trackd, tag in zip(edges, ("rec. bottom", "rec. top"), strict=True):
            y = self._track_to_y(float(trackd))
            line = QGraphicsLineItem(float(_LEFT), y, float(x_right), y)
            line.setPen(pen)
            line.setZValue(7)
            self._scene.addItem(line)
            self._extent_items.append(line)
            lab = QGraphicsTextItem(tag)
            lab.setDefaultTextColor(QColor(60, 220, 120))
            lab.setPos(float(x_right + 4), y - 8)
            lab.setZValue(7)
            self._scene.addItem(lab)
            self._extent_items.append(lab)

    def _on_anchor_moved(self) -> None:
        self._render_lfp_only()
        self._draw_recorded_extent()

    # -- UI --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        rows_note = ""
        if self._n_rows and self._sites_per_row > 1.4:
            rows_note = f" in {self._n_rows} rows (~{self._sites_per_row:.0f} sites/row)"
        elif self._n_rows:
            rows_note = f" in {self._n_rows} rows"
        info = (
            f"Probe '{self._state.project.probes[self._probe_idx].label}', "
            f"shank {self._shank_idx}  ·  stream {self._stream}\n"
            f"{self._n_channels} recorded channels{rows_note}  ·  electrodes "
            f"{self._rec_bottom:.0f}-{self._rec_top:.0f} µm from tip "
            f"(block {self._rec_top - self._rec_bottom:.0f} µm)  ·  "
            f"histology track {self._insertion:.0f} µm (≥ electrode span)"
        )
        if self._tip is not None and self._entry is not None:
            t, e = self._tip, self._entry
            info += (
                f"\nTip CCF (AP,ML,DV): {t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f} µm   ·   "
                f"Entry CCF: {e[0]:.0f}, {e[1]:.0f}, {e[2]:.0f} µm"
            )
        else:
            info += "  ·  WARNING: shank not registered (no tip/entry CCF)"
        root.addWidget(QLabel(info))

        self._regions_label = QLabel("")
        self._regions_label.setWordWrap(True)
        self._regions_label.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._regions_label)

        self._scene = QGraphicsScene(self)
        self._view = _AlignView(self._scene, self)
        self._view.setBackgroundBrush(Qt.black)
        self._lfp_item = QGraphicsPixmapItem()
        self._lfp_item.setPos(_LEFT, 0)
        self._strip_item = QGraphicsPixmapItem()
        self._strip_item.setPos(_LEFT + _IMG_W + _GAP, 0)
        self._scene.addItem(self._lfp_item)
        self._scene.addItem(self._strip_item)
        self._scene.setSceneRect(
            0, 0, _LEFT + _IMG_W + _GAP + _STRIP_W + _LABEL_W, _DISPLAY_H
        )
        self._region_label_items: list = []  # acronym text beside the strip
        self._extent_items: list = []        # electrode-extent bracket lines
        self._add_axis_labels()
        root.addWidget(self._view, 1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add anchor (mid)")
        add_btn.setToolTip(
            "Add an anchor at mid-depth. Tip: double-click anywhere on the LFP map "
            "to drop an anchor right there instead."
        )
        add_btn.clicked.connect(self._add_mid_anchor)
        rm_btn = QPushButton("Remove selected")
        rm_btn.clicked.connect(self._remove_selected)
        clear_btn = QPushButton("Clear anchors")
        clear_btn.clicked.connect(self.clear_anchors)
        save_btn = QPushButton("Save LFP power")
        save_btn.setToolTip(
            "Export the computed depth x frequency LFP power for this shank "
            "(per-channel PSD + depths + frequencies) as .npz or .csv."
        )
        save_btn.clicked.connect(self._save_lfp_power)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        self._per_freq_check = QCheckBox("Normalize per frequency")
        self._per_freq_check.setToolTip(
            "Scale each frequency column independently so depth-dependent power "
            "changes (the features you align to regions) stand out, instead of the "
            "overall 1/f gradient dominating the image."
        )
        self._per_freq_check.toggled.connect(self._recompute_image)
        btn_row.addWidget(self._per_freq_check)
        root.addLayout(btn_row)

        hint = QLabel(
            "Double-click the LFP map to drop a red anchor, then drag it to align an "
            "LFP power transition with a region boundary on the right. Horizontal axis "
            "= frequency (0–300 Hz); vertical = depth (tip at the bottom)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        root.addWidget(buttons)

    def _add_axis_labels(self) -> None:
        """Depth ticks down the left margin. The axis is the *histology track*
        (0 = tip at the bottom, surface at the top); the recorded electrodes occupy
        only the green-bracketed sub-range, so the labels mark track endpoints, not
        channels."""
        top = QGraphicsTextItem(f"surface\n{self._track_max:.0f} µm")
        top.setDefaultTextColor(QColor(220, 220, 220))
        top.setPos(2, 2)
        bottom = QGraphicsTextItem("tip\n0 µm")
        bottom.setDefaultTextColor(QColor(220, 220, 220))
        bottom.setPos(2, _DISPLAY_H - 34)
        for it in (top, bottom):
            it.setZValue(5)
            self._scene.addItem(it)

    def _add_mid_anchor(self) -> None:
        self.add_anchor_at_track(self._track_max / 2.0)

    def _save_lfp_power(self) -> None:
        """Export this shank's depth x frequency LFP power (.npz or .csv)."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        if not self._depths.size:
            QMessageBox.information(self, "Nothing to save", "No channels for this shank.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save LFP power", "lfp_power.npz",
            "NumPy archive (*.npz);;CSV (*.csv)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                import csv

                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["channel_id", "depth_um_from_tip",
                                *[f"{fr:.2f}Hz" for fr in self._freqs]])
                    for cid, d, row in zip(self._channel_ids, self._depths, self._psd, strict=False):
                        w.writerow([cid, f"{d:.2f}", *[f"{v:.6g}" for v in row]])
            else:
                np.savez(
                    path,
                    psd=self._psd, depths_um_from_tip=self._depths, freqs_hz=self._freqs,
                    channel_ids=np.array([str(c) for c in self._channel_ids]),
                    shank_index=self._shank_idx, stream_name=self._stream,
                )
            QMessageBox.information(self, "Saved", f"LFP power saved to\n{path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    # -- apply -----------------------------------------------------------

    def _apply(self) -> None:
        from histo_to_ccf.project.schema import EphysAlignment

        anchors = self.anchors()
        ccf = (
            channel_ccf_um(self._tip, self._entry, self._depths, anchors)
            if self._tip is not None and self._entry is not None and self._depths.size
            else np.zeros((0, 3))
        )
        self._shank.ephys = EphysAlignment(
            recording_path=(self._shank.ephys.recording_path if self._shank.ephys else None),
            stream_name=self._stream or None,
            shank_x_um=self._shank_x,
            channel_depths_um=[float(d) for d in self._depths],
            anchors=[(float(f), float(t)) for f, t in anchors],
            channel_ccf_um=[tuple(float(v) for v in row) for row in ccf],
        )
        if self._on_applied is not None:
            self._on_applied()
        self.close()
