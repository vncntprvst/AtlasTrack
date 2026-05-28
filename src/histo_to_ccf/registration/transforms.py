"""Section transforms: map a pixel inside a section to Allen CCF µm.

For M1 we implement the simplest plane mapping with no B-spline. The pipeline:

    (x_px, y_px) within the section
        → physical offsets from (midline_px, dorsal_surface_px) using
          ``pixel_size_um``
        → CCF µm via the PlaneParams: AP from ``ap_um``,
          ML from MIDLINE_ML_UM ± Δml, DV from Δdv.

Later milestones extend this to compose a 2D B-spline displacement field
(SimpleITK) and oblique-plane tilts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histo_to_ccf.io.ccf_coords import MIDLINE_ML_UM
from histo_to_ccf.project.schema import PlaneParams


@dataclass(frozen=True)
class SectionTransform:
    """Composed pixel → CCF µm transform for one section."""

    plane: PlaneParams

    def apply(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        """Map a section pixel to CCF (AP, ML, DV) in µm."""
        p = self.plane
        # Signed pixel offsets from anchor points.
        dx_px = x_px - p.midline_px
        dy_px = y_px - p.dorsal_surface_px

        # Physical offsets (µm). Image right vs. anatomical right sets the sign.
        ml_offset_um = dx_px * p.pixel_size_um
        if not p.image_right_is_anatomical_right:
            ml_offset_um = -ml_offset_um
        # Image y increases downward = ventral; CCF DV increases ventrally too.
        dv_um = dy_px * p.pixel_size_um

        # CCF ML is measured from the lateral edge; midline of the brain sits at
        # MIDLINE_ML_UM (≈ 5700 µm in the 25 µm atlas). A positive ml_offset_um
        # (right-of-midline anatomically) lands at MIDLINE_ML_UM + offset.
        ml_ccf = MIDLINE_ML_UM + ml_offset_um

        # AP comes from the PlaneParams. Tilts are ignored at M1.
        ap_ccf = p.ap_um

        return ap_ccf, ml_ccf, dv_um

    def apply_many(self, pts_px: np.ndarray) -> np.ndarray:
        """Vectorized variant. ``pts_px`` is ``(N, 2)``; returns ``(N, 3)`` µm."""
        pts_px = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        out = np.empty((len(pts_px), 3), dtype=float)
        for i, (x, y) in enumerate(pts_px):
            out[i] = self.apply(float(x), float(y))
        return out
