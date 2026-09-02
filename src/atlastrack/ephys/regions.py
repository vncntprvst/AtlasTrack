"""Atlas region lookup along a probe shank (per channel / per depth).

Unlike :func:`atlastrack.atlas.meshes.region_acronyms_at_points` (which
de-duplicates for 3D mesh selection), the ephys alignment view needs the region
at *every* sample point along the track, in order, to draw a depth-resolved
colour strip beside the LFP features. Pure-core (no Qt).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from atlastrack.atlas.meshes import structure_rgb

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
    :mod:`atlastrack.ephys.penetration`, so a band can be drawn straight onto the
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


#: Reserved for fibre tracts, and for nothing else. White matter is the most useful
#: landmark on an LFP panel - power drops in it - but the tracts are often thin enough
#: that the region column has no room for a label, leaving the user unable to tell a
#: tract from any other unlabelled sliver. One colour, used exclusively, answers that
#: without a label.
WHITE_MATTER_RGB = (255, 255, 255)

#: The Allen structure tree's fibre-tract root. Everything under it is white matter.
FIBER_TRACTS_ACRONYM = "fiber tracts"


def white_matter_acronyms(atlas, acronyms) -> set[str]:
    """Which of ``acronyms`` are fibre tracts, per the atlas structure tree.

    Ancestry, not a hard-coded list: ``arb``, ``cbc``, ``py``, ``ml`` and the rest all
    hang off ``fiber tracts``, and any list of them written out here would be wrong
    for the next atlas. Atlases without a ``fiber tracts`` node simply yield nothing,
    so the colouring degrades to what it was.
    """
    if atlas is None:
        return set()
    out: set[str] = set()
    for acr in {a for a in acronyms if a}:
        if acr == FIBER_TRACTS_ACRONYM:
            out.add(acr)
            continue
        try:
            ancestors = atlas.get_structure_ancestors(acr)
        except Exception:
            continue
        if FIBER_TRACTS_ACRONYM in ancestors:
            out.add(acr)
    return out


#: How far a non-tract colour must sit from white, as an RGB Euclidean distance.
#:
#: Distance rather than a per-channel threshold, because the first attempt used
#: "every channel >= 230" and passed the real case it was written for: on LO_07
#: ProbeA shank 0 the cerebellar cortex ``CUL4, 5`` drew as cream (255, 250, 200) in
#: 15-60 µm slivers *alternating with* ``arb`` at pure white. Per-channel it is not
#: white; on screen, in a 15 µm band, it is indistinguishable from one.
#:
#: 110 is set from the palette: it rejects the five pastels (55-104) and keeps the
#: curated colours for CB, Isocortex and OLF (111.5), which are large regions that
#: always carry a label anyway. 19 of the 24 fallbacks remain usable.
WHITE_CLEARANCE = 110.0


def _too_close_to_white(rgb: tuple[int, int, int],
                        *, clearance: float = WHITE_CLEARANCE) -> bool:
    """Close enough to white that it could be read as a fibre tract."""
    return math.dist([float(c) for c in rgb], [255.0, 255.0, 255.0]) < clearance


def _non_white(acronym: str) -> tuple[int, int, int]:
    """This region's colour, nudged clear of white so only tracts read as white."""
    colour = region_colour(acronym)
    if not _too_close_to_white(colour):
        return colour
    for step in range(1, _n_fallbacks() + 1):
        candidate = region_colour(acronym, offset=step)
        if not _too_close_to_white(candidate):
            return candidate
    return (128, 128, 128)


def band_colours(bands: list[RegionBand], *, shared: dict | None = None,
                 white_matter=()) -> list[tuple[int, int, int]]:
    """Colour each band from the project palette, keeping neighbours distinguishable.

    The Allen ``rgb_triplet`` values are near-useless for a depth column: whole
    cerebellar cortices come back as the same wash of yellow, so a boundary between
    two lobules is invisible - which defeats the point of drawing the column.

    Uses the palette the 3D views already use
    (:data:`atlastrack.viz.plotly3d.REGION_STYLE` plus its qualitative fallbacks),
    so a region is the same colour wherever it appears in the app. One acronym always
    gets one colour; when a region needs a fallback, the next one that differs from the
    band immediately above is chosen, so adjacent bands never collide even past the end
    of the palette.
    """
    assigned = (
        shared if shared is not None
        else region_colour_map([bands], white_matter=white_matter)
    )
    return [
        assigned.get(b.acronym, (0, 0, 0)) if b.acronym else (0, 0, 0) for b in bands
    ]


def region_colour_map(band_lists, *, white_matter=()) -> dict[str, tuple[int, int, int]]:
    """One colour per region across **every** list given, collisions resolved once.

    Pass all of a probe's shanks together. Resolving per shank instead means a
    collision that only happens on one shank shifts that region's colour on that tab
    alone - and comparing tabs is the whole reason the colours need to be stable.

    ``white_matter`` (from :func:`white_matter_acronyms`) all get
    :data:`WHITE_MATTER_RGB`, and every other region is kept off it. Two adjacent
    tracts therefore share a colour, which is deliberate: "this is white matter" is
    the reading being supported, and telling ``cbc`` from ``arb`` matters less than
    telling either from a thin unlabelled nucleus.
    """
    thickness: dict[str, float] = {}
    adjacency: set[tuple[str, str]] = set()
    for bands in band_lists:
        for band in bands:
            if band.acronym:
                thickness[band.acronym] = (
                    thickness.get(band.acronym, 0.0) + band.thickness_um
                )
        for above, below in itertools.pairwise(bands):
            a, b = above.acronym, below.acronym
            if a and b and a != b:
                adjacency.add((a, b) if a < b else (b, a))

    wm = {a for a in white_matter if a}
    assigned = {
        a: (WHITE_MATTER_RGB if a in wm else _non_white(a)) for a in thickness
    }
    # The thinner region gives way: the big landmark structures are the ones worth
    # keeping recognisable from tab to tab.
    for _sweep in range(12):
        clash = next(
            (
                (a, b) for a, b in sorted(adjacency)
                if assigned[a] == assigned[b] and not (a in wm and b in wm)
            ),
            None,
        )
        if clash is None:
            break
        a, b = clash
        # A tract keeps white whatever it is next to; the other side moves.
        if a in wm:
            loser = b
        elif b in wm:
            loser = a
        else:
            loser = a if (thickness[a], a) <= (thickness[b], b) else b
        taken = {c for k, c in assigned.items() if k != loser}
        for step in range(1, _n_fallbacks() + 1):
            candidate = region_colour(loser, offset=step)
            if candidate not in taken and not _too_close_to_white(candidate):
                assigned[loser] = candidate
                break
    return assigned


# A wider qualitative palette than the 3D views' 12, used only for regions that have
# no curated colour. Width is the point: a shank crosses ~10-15 structures, so with 12
# colours adjacent collisions are common, and every collision forces a region to change
# colour between tabs - which is exactly what makes the tabs hard to compare. 24
# well-separated colours make collisions rare enough that stability effectively holds.
_DEPTH_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#a9a9a9", "#00a5a5", "#c71585", "#7f8c00", "#5d8aa8",
]


def _n_fallbacks() -> int:
    return len(_DEPTH_PALETTE)


def region_colour(acronym: str, *, offset: int = 0) -> tuple[int, int, int]:
    """The colour for a region, **stable everywhere it appears**.

    Curated regions take their fixed colour from
    :data:`atlastrack.viz.plotly3d.REGION_STYLE`; everything else is hashed onto the
    fallback palette. Hashing rather than cycling in encounter order is the point: a
    counter gives a region a different colour on every shank, because each shank
    crosses a different set of structures - so FN came out blue on one tab and orange
    on the next, which makes the tabs impossible to compare.

    ``zlib.crc32``, not the builtin ``hash``: Python salts string hashing per process,
    so colours would change between runs of the same app.
    """
    from zlib import crc32

    from atlastrack.viz.plotly3d import REGION_STYLE, hex_to_rgb

    if acronym in REGION_STYLE:
        return hex_to_rgb(REGION_STYLE[acronym][0])
    index = (crc32(acronym.encode("utf-8")) + int(offset)) % len(_DEPTH_PALETTE)
    return hex_to_rgb(_DEPTH_PALETTE[index])


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
