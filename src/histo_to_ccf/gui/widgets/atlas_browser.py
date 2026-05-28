"""Atlas browser: select atlas, preview coronal slices, assign AP to sections."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from histo_to_ccf.gui.workflow import WorkflowState

if TYPE_CHECKING:
    import napari

_QUICK_PICKS = [
    ("Allen CCFv3 25 µm", "allen_mouse_25um"),
    ("Allen CCFv3 100 µm", "allen_mouse_100um"),
    ("CCFv3-BBP Augmented 25 µm", "ccfv3augmented_mouse_25um"),
    ("Chon/Kim Unified 25 µm", "kim_mouse_25um"),
    ("Custom ID…", ""),
]

_LAYER_ATLAS = "Atlas preview"


class AtlasBrowserWidget(QWidget):
    """Browse coronal atlas slices and assign AP positions to sections."""

    def __init__(
        self,
        state: WorkflowState,
        viewer: "napari.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._viewer = viewer
        self._atlas_layer: "napari.layers.Image | None" = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Atlas selector
        layout.addWidget(QLabel("Atlas:"))
        self._atlas_combo = QComboBox()
        for label, _ in _QUICK_PICKS:
            self._atlas_combo.addItem(label)
        self._atlas_combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self._atlas_combo)

        # Custom ID field (visible when "Custom ID…" selected)
        self._custom_id = QLineEdit()
        self._custom_id.setPlaceholderText("brainglobe atlas id (e.g. allen_mouse_10um)")
        self._custom_id.setVisible(False)
        layout.addWidget(self._custom_id)

        load_btn = QPushButton("Load atlas")
        load_btn.clicked.connect(self._load_atlas)
        layout.addWidget(load_btn)

        self._atlas_status = QLabel("Atlas not loaded")
        layout.addWidget(self._atlas_status)

        # AP slider (SpinBox for precision)
        ap_row = QHBoxLayout()
        ap_row.addWidget(QLabel("AP µm:"))
        self._ap_spin = QDoubleSpinBox()
        self._ap_spin.setRange(0.0, 15000.0)
        self._ap_spin.setValue(5400.0)
        self._ap_spin.setSingleStep(25.0)
        self._ap_spin.valueChanged.connect(self._preview_slice)
        ap_row.addWidget(self._ap_spin)
        layout.addLayout(ap_row)

        preview_btn = QPushButton("Preview slice")
        preview_btn.clicked.connect(self._preview_slice)
        layout.addWidget(preview_btn)

        # Section selector
        sec_row = QHBoxLayout()
        sec_row.addWidget(QLabel("Assign to section idx:"))
        self._sec_spin = QDoubleSpinBox()
        self._sec_spin.setRange(0, 9999)
        self._sec_spin.setDecimals(0)
        sec_row.addWidget(self._sec_spin)
        layout.addLayout(sec_row)

        # Midline and dorsal-surface px (needed for PlaneParams)
        mid_row = QHBoxLayout()
        mid_row.addWidget(QLabel("Midline px:"))
        self._midline_spin = QDoubleSpinBox()
        self._midline_spin.setRange(0, 100000)
        self._midline_spin.setValue(0.0)
        mid_row.addWidget(self._midline_spin)
        layout.addLayout(mid_row)

        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("Dorsal surf px:"))
        self._dorsal_spin = QDoubleSpinBox()
        self._dorsal_spin.setRange(0, 100000)
        self._dorsal_spin.setValue(0.0)
        ds_row.addWidget(self._dorsal_spin)
        layout.addLayout(ds_row)

        px_row = QHBoxLayout()
        px_row.addWidget(QLabel("µm/px:"))
        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.01, 1000.0)
        self._px_spin.setValue(1.0)
        self._px_spin.setSingleStep(0.1)
        px_row.addWidget(self._px_spin)
        layout.addLayout(px_row)

        assign_btn = QPushButton("Assign AP to section")
        assign_btn.clicked.connect(self._assign_ap)
        layout.addWidget(assign_btn)

        self._assign_status = QLabel("")
        layout.addWidget(self._assign_status)
        layout.addStretch()

    # ------------------------------------------------------------------

    def _on_combo_changed(self, idx: int) -> None:
        label, _ = _QUICK_PICKS[idx]
        self._custom_id.setVisible(label.startswith("Custom"))

    def _current_atlas_id(self) -> str:
        idx = self._atlas_combo.currentIndex()
        _, atlas_id = _QUICK_PICKS[idx]
        if not atlas_id:
            atlas_id = self._custom_id.text().strip()
        return atlas_id

    def _load_atlas(self) -> None:
        atlas_id = self._current_atlas_id()
        if not atlas_id:
            self._atlas_status.setText("Enter an atlas id first.")
            return
        self._atlas_status.setText(f"Loading {atlas_id}…")
        from histo_to_ccf.gui.workers import load_atlas_worker

        worker = load_atlas_worker(atlas_id)
        worker.returned.connect(self._on_atlas_loaded)
        worker.errored.connect(lambda e: self._atlas_status.setText(f"Error: {e}"))
        worker.start()

    def _on_atlas_loaded(self, atlas) -> None:
        self._state.atlas = atlas
        self._state.project.atlas.name = self._current_atlas_id()
        ap_max = atlas.reference.shape[0] * atlas.resolution[0]
        self._ap_spin.setRange(0.0, float(ap_max))
        self._atlas_status.setText(
            f"Loaded {atlas.atlas_name}  {atlas.resolution[0]:.0f} µm"
        )

    def _preview_slice(self) -> None:
        atlas = self._state.atlas
        if atlas is None:
            return
        from histo_to_ccf.atlas.planes import coronal_anchoring, resample_atlas_at_plane

        ap_um = self._ap_spin.value()
        anchoring = coronal_anchoring(atlas, ap_um)
        dv, ml = atlas.reference.shape[1], atlas.reference.shape[2]
        ref, _ = resample_atlas_at_plane(atlas, anchoring, (dv, ml))

        if _LAYER_ATLAS in self._viewer.layers:
            self._viewer.layers[_LAYER_ATLAS].data = ref
        else:
            self._atlas_layer = self._viewer.add_image(
                ref, name=_LAYER_ATLAS, colormap="gray", opacity=0.5
            )

    def _assign_ap(self) -> None:
        slide_idx = self._state.active_slide_idx
        if slide_idx is None:
            self._assign_status.setText("No slide loaded.")
            return
        sec_idx = int(self._sec_spin.value())
        slide = self._state.project.slides[slide_idx]
        section = next((s for s in slide.sections if s.index == sec_idx), None)
        if section is None:
            self._assign_status.setText(f"Section {sec_idx} not found.")
            return
        from histo_to_ccf.project.schema import PlaneParams

        section.plane = PlaneParams(
            ap_um=self._ap_spin.value(),
            midline_px=self._midline_spin.value(),
            dorsal_surface_px=self._dorsal_spin.value(),
            pixel_size_um=self._px_spin.value(),
        )
        self._assign_status.setText(f"Assigned AP={self._ap_spin.value():.0f} µm to section {sec_idx}")
