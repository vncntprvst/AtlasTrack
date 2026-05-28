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
