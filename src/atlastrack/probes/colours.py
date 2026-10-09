"""One colour per probe shank, the same in every view.

The Probes tab, the 3D window, the interactive HTML and the figures all take a
shank's colour from here, so a track is the same colour wherever it is drawn. By
default each shank gets the next colour of :data:`PALETTE`, counted over the
project's shanks in order. A probe given its own colour (``ProbeSpec.color``) uses
it for all its shanks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlastrack.project.schema import Project

#: Distinct, colour-blind-friendlier colours, in the order shanks take them.
PALETTE = (
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9a6324",
    "#800000", "#808000", "#000075", "#a9a9a9",
)


def default_shank_colour(project: Project, probe_idx: int, shank_idx: int) -> str:
    """The palette colour of a shank, ignoring any colour set on its probe."""
    ordinal = 0
    for p, probe in enumerate(project.probes):
        if p == probe_idx:
            ordinal += min(shank_idx, max(len(probe.shanks) - 1, 0))
            break
        ordinal += len(probe.shanks)
    return PALETTE[ordinal % len(PALETTE)]


def shank_colour(project: Project, probe_idx: int, shank_idx: int) -> str:
    """A shank's colour as ``#rrggbb``: its probe's own colour, else the palette's."""
    try:
        own = project.probes[probe_idx].color
    except (IndexError, AttributeError):
        own = None
    return own or default_shank_colour(project, probe_idx, shank_idx)


def probe_colour(project: Project, probe_idx: int) -> str:
    """The colour that stands for a whole probe: its own, else its first shank's."""
    return shank_colour(project, probe_idx, 0)


def hex_to_rgba(colour: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``#rrggbb`` -> RGBA floats in 0-1."""
    c = colour.lstrip("#")
    return (int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0,
            float(alpha))
