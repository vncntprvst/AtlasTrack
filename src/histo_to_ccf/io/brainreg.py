"""Read a brainreg registration and map sample-volume points into CCF micrometres.

Lightsheet subjects don't go through this app's 2D section pipeline at all: the
whole cleared brain is registered to the atlas in one shot by `brainreg`. What is
still needed is the last step - turning a point picked in the sample volume (a
probe tip, a shank track) into a CCF coordinate the rest of the app understands.

`brainreg` writes three ``deformation_field_{0,1,2}.tiff`` on the **sample** grid.
Voxel ``(i, j, k)`` of the sample volume holds the atlas position it maps to, in
**millimetres**, with the field index matching the atlas array axes. For
``allen_mouse_25um`` (orientation ``asr``) those axes are (AP, DV, ML), so

    ccf_um = (df0, df1, df2) * 1000        -> (AP, DV, ML)

and this module reorders to the app's (AP, ML, DV) convention. Verified against
``registered_atlas.tiff`` on real data: sampling the fields and looking the result
up in the atlas annotation reproduces the structure id brainreg assigned to the
same voxel for 100% of test voxels (see ``tests/test_brainreg_io.py``).

Nothing here re-runs a registration - it only reads what brainreg produced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFORMATION_FIELDS = ("deformation_field_0.tiff", "deformation_field_1.tiff",
                      "deformation_field_2.tiff")


class BrainregRegistration:
    """A completed brainreg run, opened lazily from its output directory.

    The deformation fields are memory-mapped (they run to hundreds of MB each),
    so constructing this is cheap and only the sampled voxels are read.
    """

    def __init__(self, directory: str | Path):
        import tifffile

        self.directory = Path(directory)
        missing = [f for f in DEFORMATION_FIELDS if not (self.directory / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self.directory} is not a brainreg output directory - missing "
                + ", ".join(missing)
            )
        self._fields = [
            tifffile.memmap(str(self.directory / f), mode="r") for f in DEFORMATION_FIELDS
        ]
        self.shape = tuple(int(v) for v in self._fields[0].shape)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BrainregRegistration({self.directory}, shape={self.shape})"

    # ------------------------------------------------------------------
    # Point mapping
    # ------------------------------------------------------------------

    def sample_voxels_to_ccf_um(self, voxels: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` sample-volume voxel indices to CCF ``(AP, ML, DV)`` µm.

        Indices are rounded to the nearest voxel and clamped to the volume, so a
        point just outside the grid maps to the nearest edge rather than raising.
        Voxels that brainreg left unmapped come back as ``NaN``.
        """
        pts = np.atleast_2d(np.asarray(voxels, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"voxels must be (N, 3), got {np.shape(voxels)}")

        idx = np.rint(pts).astype(int)
        for axis in range(3):
            idx[:, axis] = np.clip(idx[:, axis], 0, self.shape[axis] - 1)

        ap_dv_ml = np.stack(
            [np.asarray(f[idx[:, 0], idx[:, 1], idx[:, 2]], dtype=float) for f in self._fields],
            axis=1,
        ) * 1000.0
        # brainreg leaves unmapped voxels at 0 in every field; a real coordinate is
        # never exactly the atlas origin in all three axes.
        unmapped = np.all(ap_dv_ml == 0.0, axis=1)
        out = ap_dv_ml[:, [0, 2, 1]]  # (AP, DV, ML) -> (AP, ML, DV)
        out[unmapped] = np.nan
        return out

    def sample_um_to_voxels(
        self, points_um: np.ndarray, *, voxel_size_um: float | tuple[float, float, float]
    ) -> np.ndarray:
        """Convert micrometre positions in the sample volume to voxel indices."""
        pts = np.atleast_2d(np.asarray(points_um, dtype=float))
        size = np.broadcast_to(np.asarray(voxel_size_um, dtype=float), (3,))
        if np.any(size <= 0):
            raise ValueError(f"voxel_size_um must be positive, got {voxel_size_um}")
        return pts / size

    # ------------------------------------------------------------------
    # Sanity checks - a registration can complete and still be unusable
    # ------------------------------------------------------------------

    def dv_extent_um(self, *, resolution_um: float = 25.0) -> float:
        """Dorsoventral extent of the sample grid, in µm.

        A whole mouse brain is roughly 8000 µm deep. A value far from that means
        the run was given wrong voxel sizes, which silently distorts every
        coordinate derived from it - see :func:`check_geometry`.
        """
        return float(self.shape[1] * resolution_um)

    def check_geometry(
        self,
        *,
        resolution_um: float = 25.0,
        expected_dv_um: float = 8000.0,
        tolerance: float = 0.25,
    ) -> tuple[bool, str]:
        """Is the sample grid anatomically plausible? Returns ``(ok, message)``.

        Cheap guard against the most damaging brainreg mistake: passing voxel
        sizes in the wrong axis order, which scales one axis and leaves the
        registration to absorb it (sometimes it can, sometimes it cannot).
        """
        dv = self.dv_extent_um(resolution_um=resolution_um)
        ratio = dv / expected_dv_um
        if abs(ratio - 1.0) <= tolerance:
            return True, f"sample DV extent {dv:.0f} µm ({ratio:.2f}x expected)"
        return False, (
            f"sample DV extent {dv:.0f} µm is {ratio:.2f}x the ~{expected_dv_um:.0f} µm "
            "of a mouse brain - check the voxel sizes passed to brainreg "
            "(-v takes them in image axis order, z first)"
        )
