"""Utility: render section bounding boxes as a raster outline Labels array.

Using a Labels layer (integer raster) instead of a Shapes layer for the
outline display avoids the zoom-dependent rendering artefacts that affect
napari's vector Shapes layer at low magnification.  Each section gets a
unique label id; outline thickness scales with the image diagonal so boxes
remain visible at any zoom level.
"""
from __future__ import annotations

import numpy as np


def sections_to_outline_labels(
    image_shape: tuple[int, int],
    sections,
    *,
    thickness: int | None = None,
) -> np.ndarray:
    """Return a uint16 Labels array with a coloured border for each section.

    Parameters
    ----------
    image_shape
        ``(height, width)`` of the slide image.
    sections
        Iterable of objects with ``.index`` and ``.bbox_px`` (x0,y0,x1,y1).
    thickness
        Border thickness in pixels.  Auto-computed from the image diagonal
        when ``None`` (typically 6–40 px depending on image size).
    """
    h, w = image_shape[:2]
    if thickness is None:
        diag = (h ** 2 + w ** 2) ** 0.5
        thickness = max(4, int(diag / 300))

    labels = np.zeros((h, w), dtype=np.uint16)
    for sec in sections:
        x0, y0, x1, y1 = sec.bbox_px
        # Clamp to image bounds.
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        lbl = sec.index + 1  # label 0 = background
        t = min(thickness, (y1 - y0) // 2, (x1 - x0) // 2)
        t = max(1, t)
        labels[y0:y0 + t, x0:x1] = lbl          # top edge
        labels[y1 - t:y1, x0:x1] = lbl          # bottom edge
        labels[y0:y1, x0:x0 + t] = lbl          # left edge
        labels[y0:y1, x1 - t:x1] = lbl          # right edge

    return labels
