"""Manual per-section atlas correction: affine bookkeeping.

The user drags the atlas overlay in napari's ``transform`` mode, which writes a
3x3 **world** affine to the layer (napari (row, col) order). The overlay layer is
placed at the section's bbox origin via ``translate=(y0, x0)``, so the world
affine ``W`` and the bbox-independent **section-local** affine ``A`` (what we
persist) are related by conjugation with that origin::

    A = T(-origin) @ W @ T(origin)
    W = T(origin)  @ A @ T(-origin)

``A`` maps a *registered* atlas position (section-local row, col) to where the
user dragged it. For the probe -> CCF mapping we need the opposite: a tissue
point is fixed, the atlas moved, so to read the atlas coordinate under a clicked
point we apply ``A^-1`` before the registration inverse (see
:meth:`RegisteredSectionTransform.apply`).

All matrices are 3x3 numpy arrays in homogeneous (row, col) order. Headless.
"""
from __future__ import annotations

import numpy as np

IDENTITY = np.eye(3)


def _translation(origin_yx: tuple[float, float], sign: float = 1.0) -> np.ndarray:
    m = np.eye(3)
    m[0, 2] = sign * float(origin_yx[0])
    m[1, 2] = sign * float(origin_yx[1])
    return m


def world_to_section(world: np.ndarray, origin_yx: tuple[float, float]) -> np.ndarray:
    """napari world affine -> bbox-independent section-local affine."""
    w = np.asarray(world, dtype=float).reshape(3, 3)
    return _translation(origin_yx, -1.0) @ w @ _translation(origin_yx, 1.0)


def section_to_world(section: np.ndarray, origin_yx: tuple[float, float]) -> np.ndarray:
    """Section-local affine -> napari world affine (for layer.affine)."""
    a = np.asarray(section, dtype=float).reshape(3, 3)
    return _translation(origin_yx, 1.0) @ a @ _translation(origin_yx, -1.0)


def is_identity(affine: np.ndarray | None, *, atol: float = 1e-6) -> bool:
    if affine is None:
        return True
    return bool(np.allclose(np.asarray(affine, dtype=float).reshape(3, 3), IDENTITY, atol=atol))


def invert_apply(section_affine: np.ndarray, x_px: float, y_px: float) -> tuple[float, float]:
    """Map a section-local point (x=col, y=row) through ``A^-1``.

    Used to pull a clicked (corrected-frame) point back into the registered
    frame before the registration inverse runs.
    """
    a = np.asarray(section_affine, dtype=float).reshape(3, 3)
    inv = np.linalg.inv(a)
    row, col, _ = inv @ np.array([float(y_px), float(x_px), 1.0])
    return float(col), float(row)
