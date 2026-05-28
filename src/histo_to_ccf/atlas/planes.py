"""Resample a 3D atlas at an arbitrary oblique plane.

The plane is expressed in **QuickNII anchoring** format — a 9-vector
``(ox, oy, oz, ux, uy, uz, vx, vy, vz)`` interpreted in atlas voxel coordinates
(ASR order: axis0=AP, axis1=DV, axis2=ML):

    sample(x_px, y_px) = (ox, oy, oz) + (x_px / W) * (ux, uy, uz)
                                       + (y_px / H) * (vx, vy, vz)

where ``W`` and ``H`` are the requested output width and height in pixels.

This matches what DeepSlice writes and what VisuAlign reads, so we can
interoperate without an additional file format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from histo_to_ccf.io.ccf_coords import atlas_resolution_um
from histo_to_ccf.project.schema import PlaneParams

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


@dataclass(frozen=True)
class Anchoring:
    """QuickNII-style 9-vector anchoring (voxel coords, ASR order)."""

    ox: float
    oy: float
    oz: float
    ux: float
    uy: float
    uz: float
    vx: float
    vy: float
    vz: float

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.ox, self.oy, self.oz,
            self.ux, self.uy, self.uz,
            self.vx, self.vy, self.vz,
        )

    @classmethod
    def from_iterable(cls, values: "list[float] | tuple[float, ...]") -> "Anchoring":
        if len(values) != 9:
            raise ValueError(f"anchoring expects 9 floats, got {len(values)}")
        return cls(*[float(v) for v in values])


def coronal_anchoring(
    atlas: "BrainGlobeAtlas",
    ap_um: float,
    *,
    ml_tilt_deg: float = 0.0,
    dv_tilt_deg: float = 0.0,
) -> Anchoring:
    """Build a coronal-plane anchoring at ``ap_um``, optionally tilted.

    The default (zero tilts) coronal plane has:
        origin at (ap_idx, 0, 0) — top-left of the AP slab
        u along +ML (image x increases → lateral right)
        v along +DV (image y increases → ventral)

    ``ml_tilt_deg`` rotates the plane about the DV axis (tilts the slice so
    its medial edge moves anterior / posterior).
    ``dv_tilt_deg`` rotates about the ML axis (tilts so the dorsal edge moves
    anterior / posterior).
    """
    ap_res, dv_res, ml_res = atlas_resolution_um(atlas)
    ap_idx = ap_um / ap_res
    ap_size, dv_size, ml_size = atlas.annotation.shape

    # Untilted basis (atlas-voxel coords, ASR order).
    u = np.array([0.0, 0.0, float(ml_size)])
    v = np.array([0.0, float(dv_size), 0.0])
    origin = np.array([ap_idx, 0.0, 0.0])

    # ml_tilt: rotation about the DV axis (axis 1). Positive tilt → medial-right
    # edge of the slice moves anterior (smaller AP index).
    if ml_tilt_deg != 0.0:
        a = np.deg2rad(ml_tilt_deg)
        R = np.array(
            [
                [np.cos(a), 0.0, -np.sin(a)],
                [0.0, 1.0, 0.0],
                [np.sin(a), 0.0, np.cos(a)],
            ]
        )
        u = R @ u
        v = R @ v

    # dv_tilt: rotation about the ML axis (axis 2). Positive → dorsal edge moves
    # anterior.
    if dv_tilt_deg != 0.0:
        a = np.deg2rad(dv_tilt_deg)
        R = np.array(
            [
                [np.cos(a), -np.sin(a), 0.0],
                [np.sin(a), np.cos(a), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        u = R @ u
        v = R @ v

    return Anchoring(
        ox=float(origin[0]), oy=float(origin[1]), oz=float(origin[2]),
        ux=float(u[0]), uy=float(u[1]), uz=float(u[2]),
        vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
    )


def anchoring_from_plane_params(
    atlas: "BrainGlobeAtlas", plane: PlaneParams
) -> Anchoring:
    """Build an anchoring from the simpler PlaneParams used by M1."""
    return coronal_anchoring(
        atlas,
        ap_um=plane.ap_um,
        ml_tilt_deg=plane.ml_tilt_deg,
        dv_tilt_deg=plane.dv_tilt_deg,
    )


def sample_plane(
    volume: np.ndarray,
    anchoring: Anchoring,
    out_shape: tuple[int, int],
    *,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Sample ``volume`` on the 2D grid defined by ``anchoring``.

    Parameters
    ----------
    volume
        3D atlas array, axis order ASR (AP, DV, ML).
    anchoring
        QuickNII 9-vector.
    out_shape
        ``(height, width)`` of the output slice in pixels.
    order
        Interpolation order. 1 = linear (use for reference Nissl);
        0 = nearest (use for integer annotation labels).
    cval
        Value for samples outside the volume.
    """
    h, w = out_shape
    # 1-D parametric coords along u and v in [0, 1].
    s_u = np.linspace(0.0, 1.0, w, endpoint=False, dtype=float)
    s_v = np.linspace(0.0, 1.0, h, endpoint=False, dtype=float)
    su, sv = np.meshgrid(s_u, s_v, indexing="xy")  # (h, w)

    ox, oy, oz = anchoring.ox, anchoring.oy, anchoring.oz
    ux, uy, uz = anchoring.ux, anchoring.uy, anchoring.uz
    vx, vy, vz = anchoring.vx, anchoring.vy, anchoring.vz

    coords_ap = ox + su * ux + sv * vx
    coords_dv = oy + su * uy + sv * vy
    coords_ml = oz + su * uz + sv * vz
    coords = np.stack([coords_ap, coords_dv, coords_ml], axis=0)  # (3, h, w)

    return ndimage.map_coordinates(
        volume, coords, order=order, mode="constant", cval=cval, prefilter=False
    )


def resample_atlas_at_plane(
    atlas: "BrainGlobeAtlas",
    anchoring: Anchoring,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reference_slice, annotation_slice)`` resampled at ``anchoring``.

    ``reference_slice`` (float, linear interp) is suitable as the FIXED image
    for B-spline registration. ``annotation_slice`` (uint, nearest-neighbor)
    carries integer region labels.
    """
    reference = sample_plane(atlas.reference.astype(np.float32), anchoring, out_shape, order=1)
    annotation = sample_plane(
        atlas.annotation.astype(np.int32), anchoring, out_shape, order=0
    ).astype(atlas.annotation.dtype)
    return reference, annotation
