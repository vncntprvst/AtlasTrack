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


def warp_annotation_to_section(
    result: RegistrationResult,
    atlas: "BrainGlobeAtlas",
    section_shape: tuple[int, int],
) -> np.ndarray:
    """Render the registered atlas annotation in a section's pixel grid.

    Resamples the atlas annotation at the section's plane (``anchoring``), then
    warps that slice through the fitted B-spline into the histology section's
    pixel space, so the result can be overlaid directly on the section image.

    Returns an integer label image of shape ``section_shape`` (H, W). If no
    B-spline was stored the plane annotation is returned resized to the section
    (i.e. the un-refined plane), which is still a useful sanity overlay.
    """
    from histo_to_ccf.atlas.planes import resample_atlas_at_plane

    anchoring = Anchoring.from_iterable(result.anchoring)
    h_slice, w_slice = result.output_size_px
    _, ann_slice = resample_atlas_at_plane(atlas, anchoring, (int(h_slice), int(w_slice)))

    h_sec, w_sec = int(section_shape[0]), int(section_shape[1])
    if result.bspline_transform_path is None:
        # No refinement available: nearest-neighbour resize of the plane.
        ys = (np.linspace(0, h_slice - 1, h_sec)).round().astype(int)
        xs = (np.linspace(0, w_slice - 1, w_sec)).round().astype(int)
        return ann_slice[np.ix_(ys, xs)]

    import SimpleITK as sitk

    transform = build_registered_transform(result, atlas).bspline
    # The B-spline maps fixed (atlas slice) → moving (section). Resampling the
    # annotation into the section grid needs the section → slice map (inverse).
    try:
        inverse = transform.GetInverse()
    except RuntimeError:
        inverse = _invert_displacement(transform, (h_slice, w_slice))

    ann_img = sitk.GetImageFromArray(ann_slice.astype(np.int32))
    reference = sitk.Image(w_sec, h_sec, sitk.sitkInt32)
    warped = sitk.Resample(
        ann_img, reference, inverse, sitk.sitkNearestNeighbor, 0.0
    )
    return sitk.GetArrayFromImage(warped)


def annotation_boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean edge map: True where a label differs from a 4-neighbour."""
    labels = np.asarray(labels)
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[1:, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edges


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
