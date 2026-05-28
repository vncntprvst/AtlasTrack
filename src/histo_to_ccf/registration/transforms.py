"""Section transforms: map a pixel inside a section to Allen CCF µm.

Two transform modes:

- **Manual** (M1): a :class:`PlaneParams` carries midline/dorsal-surface anchors
  and an AP/tilt-free pixel→CCF computation. Used when DeepSlice or the
  B-spline refinement has not been run.

- **Registered** (M3): a :class:`RegistrationResult` carries a QuickNII
  ``anchoring`` (3D atlas plane) and an optional SimpleITK B-spline transform
  for in-plane refinement. The histology pixel is first mapped through the
  B-spline into the atlas-slice coordinate system, then through the anchoring
  into atlas voxel coordinates, then to CCF µm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from histo_to_ccf.atlas.planes import Anchoring
from histo_to_ccf.io.ccf_coords import MIDLINE_ML_UM
from histo_to_ccf.project.schema import PlaneParams, RegistrationResult

if TYPE_CHECKING:
    import SimpleITK as sitk

    from brainglobe_atlasapi import BrainGlobeAtlas


@dataclass(frozen=True)
class ManualSectionTransform:
    """The simple M1 pixel→CCF mapping (no atlas resampling)."""

    plane: PlaneParams

    def apply(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        p = self.plane
        dx_px = x_px - p.midline_px
        dy_px = y_px - p.dorsal_surface_px

        ml_offset_um = dx_px * p.pixel_size_um
        if not p.image_right_is_anatomical_right:
            ml_offset_um = -ml_offset_um
        dv_um = dy_px * p.pixel_size_um
        ml_ccf = MIDLINE_ML_UM + ml_offset_um
        ap_ccf = p.ap_um
        return ap_ccf, ml_ccf, dv_um

    def apply_many(self, pts_px: np.ndarray) -> np.ndarray:
        pts_px = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        out = np.empty((len(pts_px), 3), dtype=float)
        for i, (x, y) in enumerate(pts_px):
            out[i] = self.apply(float(x), float(y))
        return out


# Back-compat alias — older code imports SectionTransform.
SectionTransform = ManualSectionTransform


@dataclass(frozen=True)
class RegisteredSectionTransform:
    """The M3 pixel→CCF mapping: optional B-spline ∘ anchoring."""

    anchoring: Anchoring
    output_size_px: tuple[int, int]
    bspline: "sitk.Transform | None"
    atlas_resolution_um: tuple[float, float, float]  # (ap_res, dv_res, ml_res)

    def _section_px_to_slice_px(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Map a histology pixel through the inverse B-spline to slice-px coords.

        The B-spline was trained mapping FIXED (atlas slice) → MOVING
        (histology). To go in the other direction we need the inverse. For a
        ``CompositeTransform(Affine, BSpline)`` SimpleITK's :func:`GetInverse`
        is exact for the affine; the B-spline inverse is iterative.
        """
        if self.bspline is None:
            return x_px, y_px
        import SimpleITK as sitk

        # SITK TransformPoint maps fixed→moving; we want the inverse.
        try:
            inv = self.bspline.GetInverse()
        except RuntimeError:
            # Some B-splines need an iterative inverse displacement field.
            inv = _invert_displacement(self.bspline, self.output_size_px)
        slice_x, slice_y = inv.TransformPoint((float(x_px), float(y_px)))
        return float(slice_x), float(slice_y)

    def apply(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        sx, sy = self._section_px_to_slice_px(x_px, y_px)
        h, w = self.output_size_px
        su = sx / max(w, 1)
        sv = sy / max(h, 1)
        a = self.anchoring
        ap_idx = a.ox + su * a.ux + sv * a.vx
        dv_idx = a.oy + su * a.uy + sv * a.vy
        ml_idx = a.oz + su * a.uz + sv * a.vz
        ap_res, dv_res, ml_res = self.atlas_resolution_um
        return ap_idx * ap_res, ml_idx * ml_res, dv_idx * dv_res

    def apply_many(self, pts_px: np.ndarray) -> np.ndarray:
        pts_px = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        out = np.empty((len(pts_px), 3), dtype=float)
        for i, (x, y) in enumerate(pts_px):
            out[i] = self.apply(float(x), float(y))
        return out


def _invert_displacement(
    transform: "sitk.Transform", output_size_px: tuple[int, int]
) -> "sitk.Transform":
    """Fall-back B-spline inverse via a sampled displacement field."""
    import SimpleITK as sitk

    h, w = output_size_px
    ref = sitk.Image(int(w), int(h), sitk.sitkFloat32)
    disp = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        ref.GetSize(),
        ref.GetOrigin(),
        ref.GetSpacing(),
        ref.GetDirection(),
    )
    inverse_disp = sitk.InvertDisplacementField(
        disp,
        maximumNumberOfIterations=20,
        meanErrorToleranceThreshold=1e-3,
        maxErrorToleranceThreshold=1e-2,
        enforceBoundaryCondition=True,
    )
    return sitk.DisplacementFieldTransform(inverse_disp)


def build_registered_transform(
    result: RegistrationResult,
    atlas: "BrainGlobeAtlas",
    *,
    project_dir: Path | None = None,
) -> RegisteredSectionTransform:
    """Construct a :class:`RegisteredSectionTransform` from persisted state."""
    from histo_to_ccf.io.ccf_coords import atlas_resolution_um

    anchoring = Anchoring.from_iterable(result.anchoring)
    bspline = None
    if result.bspline_transform_path is not None:
        import SimpleITK as sitk

        path = Path(result.bspline_transform_path)
        if project_dir is not None and not path.is_absolute():
            path = project_dir / path
        bspline = sitk.ReadTransform(str(path))
    return RegisteredSectionTransform(
        anchoring=anchoring,
        output_size_px=tuple(result.output_size_px),  # type: ignore[arg-type]
        bspline=bspline,
        atlas_resolution_um=atlas_resolution_um(atlas),
    )
