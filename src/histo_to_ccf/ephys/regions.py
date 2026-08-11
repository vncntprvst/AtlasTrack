"""Atlas region lookup along a probe shank (per channel / per depth).

Unlike :func:`histo_to_ccf.atlas.meshes.region_acronyms_at_points` (which
de-duplicates for 3D mesh selection), the ephys alignment view needs the region
at *every* sample point along the track, in order, to draw a depth-resolved
colour strip beside the LFP features. Pure-core (no Qt).
"""
from __future__ import annotations

from dataclasses import dataclass

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


# -- depth-below-surface flavour, for the feature panels -------------------


@dataclass(frozen=True)
class RegionBand:
    """One contiguous run of a single region along the track.

    Depths are **below the brain surface** (0 at the entry point, increasing to the
    tip), the axis every recording is put on by
    :mod:`histo_to_ccf.ephys.penetration`, so a band can be drawn straight onto the
    feature panels without a further flip.
    """

    top_um: float
    bottom_um: float
    acronym: str
    rgb: tuple[int, int, int]

    @property
    def thickness_um(self) -> float:
        return self.bottom_um - self.top_um

    @property
    def mid_um(self) -> float:
        return 0.5 * (self.top_um + self.bottom_um)


def track_points_ccf_um(tip_ccf_um, entry_ccf_um, depths_below_surface_um) -> np.ndarray:
    """CCF ``(AP, ML, DV)`` µm at each depth below the surface along the shank.

    The entry point *is* the surface, so depth 0 sits at ``entry`` and the insertion
    length sits at ``tip``. Depths beyond either end are extrapolated along the same
    line rather than clipped - the caller can see a channel sitting outside the brain
    and should, because that is a real disagreement between the ephys and the
    histology, not something to hide.
    """
    entry = np.asarray(entry_ccf_um, dtype=float)
    tip = np.asarray(tip_ccf_um, dtype=float)
    depths = np.atleast_1d(np.asarray(depths_below_surface_um, dtype=float))
    vec = tip - entry
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        return np.repeat(entry[None, :], depths.size, axis=0)
    direction = vec / length
    return entry[None, :] + depths[:, None] * direction[None, :]


def regions_along_track(atlas, tip_ccf_um, entry_ccf_um, depths_below_surface_um
                        ) -> list[RegionHit]:
    """Region at each depth below the surface along the tip-entry line."""
    return regions_at_ccf(
        atlas, track_points_ccf_um(tip_ccf_um, entry_ccf_um, depths_below_surface_um)
    )


def band_colours(bands: list[RegionBand]) -> list[tuple[int, int, int]]:
    """Colour each band from the project palette, keeping neighbours distinguishable.

    The Allen ``rgb_triplet`` values are near-useless for a depth column: whole
    cerebellar cortices come back as the same wash of yellow, so a boundary between
    two lobules is invisible - which defeats the point of drawing the column.

    Uses the palette the 3D views already use
    (:data:`histo_to_ccf.viz.plotly3d.REGION_STYLE` plus its qualitative fallbacks),
    so a region is the same colour wherever it appears in the app. One acronym always
    gets one colour; when a region needs a fallback, the next one that differs from the
    band immediately above is chosen, so adjacent bands never collide even past the end
    of the palette.
    """
    from histo_to_ccf.viz.plotly3d import REGION_STYLE, hex_to_rgb, region_style

    assigned: dict[str, tuple[int, int, int]] = {}
    out: list[tuple[int, int, int]] = []
    previous: tuple[int, int, int] | None = None
    fallback = 0
    for band in bands:
        acronym = band.acronym
        if not acronym:
            out.append((0, 0, 0))
            previous = None
            continue
        if acronym not in assigned:
            if acronym in REGION_STYLE:
                assigned[acronym] = hex_to_rgb(region_style(acronym)[0])
            else:
                for _attempt in range(16):
                    rgb = hex_to_rgb(region_style(acronym, fallback)[0])
                    fallback += 1
                    if rgb != previous:
                        break
                assigned[acronym] = rgb
        out.append(assigned[acronym])
        previous = assigned[acronym]
    return out


def region_bands(hits: list[RegionHit], depths_um) -> list[RegionBand]:
    """Merge ordered per-depth hits into contiguous bands.

    Boundaries land at the midpoint between the two samples that straddle them, so a
    band's reported thickness does not depend on which side of the change the sample
    grid happened to fall. Unlabelled stretches (outside the atlas) are kept as bands
    with an empty acronym - a probe leaving the atlas is information.
    """
    depths = np.asarray(depths_um, dtype=float).ravel()
    n = min(len(hits), depths.size)
    if n == 0:
        return []
    bands: list[RegionBand] = []
    start = 0
    for i in range(1, n + 1):
        if i < n and hits[i][0] == hits[start][0]:
            continue
        top = float(depths[0]) if start == 0 else float(
            0.5 * (depths[start - 1] + depths[start])
        )
        bottom = float(depths[n - 1]) if i == n else float(
            0.5 * (depths[i - 1] + depths[i])
        )
        acr, rgb = hits[start]
        bands.append(RegionBand(top_um=top, bottom_um=bottom, acronym=acr, rgb=rgb))
        start = i
    return bands
