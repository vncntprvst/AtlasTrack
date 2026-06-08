"""Atlas region lookup along a probe shank (per channel / per depth).

Unlike :func:`histo_to_ccf.atlas.meshes.region_acronyms_at_points` (which
de-duplicates for 3D mesh selection), the ephys alignment view needs the region
at *every* sample point along the track, in order, to draw a depth-resolved
colour strip beside the LFP features. Pure-core (no Qt).
"""
from __future__ import annotations

import numpy as np

from histo_to_ccf.atlas.meshes import structure_rgb

RegionHit = tuple[str, tuple[int, int, int]]


def regions_at_ccf(atlas, points_ap_ml_dv_um) -> list[RegionHit]:
    """Region ``(acronym, rgb)`` at each CCF ``(AP, ML, DV)`` µm point, in order.

    Points outside the atlas yield ``("", (0, 0, 0))``. The atlas indexes ASR
    order ``(AP, DV, ML)`` so each point is reordered before lookup.
    """
    out: list[RegionHit] = []
    for p in np.asarray(points_ap_ml_dv_um, dtype=float):
        ap, ml, dv = float(p[0]), float(p[1]), float(p[2])
        try:
            acr = atlas.structure_from_coords((ap, dv, ml), microns=True, as_acronym=True)
        except Exception:
            acr = ""
        if not acr or acr == "Outside atlas":
            out.append(("", (0, 0, 0)))
        else:
            out.append((acr, structure_rgb(atlas, acr)))
    return out


def region_strip_image(hits: list[RegionHit], height: int, width: int = 24) -> np.ndarray:
    """Render a vertical ``(height, width, 3)`` RGB strip from ordered region hits.

    ``hits[0]`` is drawn at the top row, ``hits[-1]`` at the bottom; rows are
    nearest-neighbour sampled so the strip works for any number of channels.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(hits)
    if n == 0:
        return img
    for row in range(height):
        idx = min(n - 1, int(row / max(1, height) * n))
        img[row, :, :] = hits[idx][1]
    return img
