"""Image loaders for histology slides.

Thin wrapper over :mod:`tifffile` / :mod:`imageio` so the rest of the code only
sees a numpy array. Pyramid handling will arrive with M2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load a 2D or 3D (H, W[, C]) image as a numpy array."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))

    # Pillow / imageio handles png/jpg/etc.
    import imageio.v3 as iio

    return np.asarray(iio.imread(str(path)))


def crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``image`` to ``(x0, y0, x1, y1)`` in pixel coords (right/bottom exclusive)."""
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce a 2D/RGB/RGBA array to a 3-channel (H, W, 3) array."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[2] >= 3:
        return arr[..., :3]
    # Single-channel-in-3D or 2-channel: replicate the first channel.
    return np.stack([arr[..., 0]] * 3, axis=-1)


def merge_images(images: list[np.ndarray], *, gap_px: int = 40) -> np.ndarray:
    """Merge several slide images into one canvas, stacked vertically with a gap.

    Each image is placed top-left in a row; rows are padded to the widest image
    and separated by ``gap_px`` of background. Grayscale and colour inputs are
    mixed by promoting everything to RGB when any input is colour. The layout is
    a pure function of the input order, so a project that records its source
    paths (sorted) can reproduce the exact same combined image on reload — which
    keeps section bounding boxes valid.

    This is how the app supports "multiple slides": rather than tracking per-slide
    offsets through clicks / registration / 3D, all slides live in one image and
    therefore one coordinate space (a probe may enter on a section from one slide
    and have its tip on a section from another).
    """
    arrays = [np.asarray(im) for im in images]
    if not arrays:
        raise ValueError("merge_images requires at least one image")
    if len(arrays) == 1:
        return arrays[0]

    any_rgb = any(a.ndim == 3 for a in arrays)
    if any_rgb:
        arrays = [_to_rgb(a) for a in arrays]

    dtype = np.result_type(*[a.dtype for a in arrays])
    arrays = [a.astype(dtype, copy=False) for a in arrays]

    max_w = max(a.shape[1] for a in arrays)
    n = len(arrays)
    total_h = sum(a.shape[0] for a in arrays) + gap_px * (n - 1)
    shape = (total_h, max_w, 3) if any_rgb else (total_h, max_w)
    canvas = np.zeros(shape, dtype=dtype)

    y = 0
    for a in arrays:
        h, w = a.shape[:2]
        canvas[y:y + h, :w] = a
        y += h + gap_px
    return canvas
