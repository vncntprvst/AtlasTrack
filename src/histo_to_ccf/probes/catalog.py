"""Probe type catalog: channel layouts for common Neuropixels models.

Depths are measured from the physical tip of the probe (tip = 0 µm).
Lateral offsets are measured from the shank centreline (positive = right
when viewed from the front face).

References
----------
- NP 1.0: https://www.neuropixels.org/probe10a (imec)
- NP 2.0: https://www.neuropixels.org/probe20 (imec)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProbeLayout:
    """Physical recording-site layout for one probe model.

    Most Neuropixels models are a regular interleaved grid, fully described by
    the parametric fields (``tip_to_first_site_um`` … ``col_pitch_um``).
    Irregular layouts (e.g. the NeuroNexus Poly3, whose centre column is longer
    than its flanking columns) instead supply ``explicit_depths_um`` and
    ``explicit_offsets_um`` - per-site arrays in channel order (tip → base) that
    override the parametric computation.
    """

    name: str
    n_channels: int
    tip_to_first_site_um: float = 0.0
    site_row_pitch_um: float = 0.0
    n_columns: int = 1
    col_pitch_um: float = 0.0
    # Optional explicit per-site geometry (length == n_channels, tip → base).
    explicit_depths_um: tuple[float, ...] | None = None
    explicit_offsets_um: tuple[float, ...] | None = None
    # Informational: optical-fibre offset above the top-most site (optetrodes).
    fiber_offset_above_top_site_um: float | None = None

    def site_depths_from_tip_um(self) -> np.ndarray:
        """Depth of each recording site from the probe tip (µm).

        Sites are assigned in column-interleaved order (col 0 row 0,
        col 1 row 0, col 0 row 1, …) matching how channels are typically
        numbered from tip to base.
        """
        if self.explicit_depths_um is not None:
            return np.array(self.explicit_depths_um, dtype=float)
        n_per_col = self.n_channels // self.n_columns
        row_pitch = self.site_row_pitch_um * self.n_columns  # distance between same-col rows
        depths = []
        for row in range(n_per_col):
            for col in range(self.n_columns):
                depth = self.tip_to_first_site_um + col * self.site_row_pitch_um + row * row_pitch
                depths.append(depth)
        return np.array(depths[: self.n_channels], dtype=float)

    def site_lateral_offsets_um(self) -> np.ndarray:
        """Lateral offset of each site from the shank centreline (µm).

        Returns an array of length ``n_channels`` in the same channel order
        as :meth:`site_depths_from_tip_um`.
        """
        if self.explicit_offsets_um is not None:
            return np.array(self.explicit_offsets_um, dtype=float)
        if self.n_columns == 1:
            return np.zeros(self.n_channels, dtype=float)
        half = (self.n_columns - 1) * self.col_pitch_um / 2.0
        col_offsets = np.arange(self.n_columns) * self.col_pitch_um - half
        n_per_col = self.n_channels // self.n_columns
        offsets = []
        for _ in range(n_per_col):
            for col in range(self.n_columns):
                offsets.append(col_offsets[col])
        return np.array(offsets[: self.n_channels], dtype=float)


# ---------------------------------------------------------------------------
# Catalog of known probe models
# ---------------------------------------------------------------------------

NEURONEXUS_A1X32_POLY3 = "NeuroNexus A1x32-Poly3-10mm-25s-177-OA32LP"


def _neuronexus_a1x32_poly3() -> ProbeLayout:
    """Build the NeuroNexus A1x32-Poly3-10mm-25s-177(-OA32LP) site layout.

    The Poly3 topology is taken from the catalogued ProbeInterface entry
    ``neuronexus / A1x32-Poly3-10mm-50-177`` (verified geometry: 3 columns, the
    centre column carrying 12 sites and each side column 10, for 32 total) with
    the site grid rescaled from the 50 µm to the 25 µm pitch of this model:

      * 25 µm vertical pitch within a column; 25 µm lateral column pitch
      * the physical shank tip sits 100 µm below the lowest site (the taper is a
        property of the 10 mm shank, so it is *not* rescaled with the pitch)

    The OA32LP optical assembly carries a fibre 50 µm above the top-most site;
    that offset is recorded as metadata and does not affect the site
    coordinates.  Sites are ordered tip → base (ascending depth), ties broken
    left → right, matching the channel convention used elsewhere in the catalog.
    """
    pitch = 25.0
    tip_to_lowest_site = 100.0
    left, centre, right = -pitch, 0.0, pitch

    sites: list[tuple[float, float]] = []  # (depth_from_tip, lateral_offset)
    # Centre column spans rows 0..11; side columns the inner rows 1..10.
    for row in range(12):
        sites.append((tip_to_lowest_site + row * pitch, centre))
    for row in range(1, 11):
        sites.append((tip_to_lowest_site + row * pitch, left))
        sites.append((tip_to_lowest_site + row * pitch, right))
    sites.sort(key=lambda s: (s[0], s[1]))

    return ProbeLayout(
        name=NEURONEXUS_A1X32_POLY3,
        n_channels=32,
        explicit_depths_um=tuple(d for d, _ in sites),
        explicit_offsets_um=tuple(o for _, o in sites),
        fiber_offset_above_top_site_um=50.0,
    )


CATALOG: dict[str, ProbeLayout] = {
    "Neuropixels 1.0": ProbeLayout(
        name="Neuropixels 1.0",
        n_channels=384,
        tip_to_first_site_um=175.0,  # base of the taper
        site_row_pitch_um=10.0,       # 10 µm between alternating rows (20 µm same-col)
        n_columns=2,
        col_pitch_um=32.0,            # ±16 µm from centreline
    ),
    "Neuropixels 2.0 (4-shank)": ProbeLayout(
        name="Neuropixels 2.0 (4-shank)",
        n_channels=384,               # per shank
        tip_to_first_site_um=0.0,
        site_row_pitch_um=7.5,
        n_columns=2,
        col_pitch_um=32.0,
    ),
    NEURONEXUS_A1X32_POLY3: _neuronexus_a1x32_poly3(),
}


def get_layout(probe_name: str) -> ProbeLayout:
    """Return a :class:`ProbeLayout` by name; falls back to NP 1.0 if unknown."""
    # Exact match first.
    if probe_name in CATALOG:
        return CATALOG[probe_name]
    # Case-insensitive prefix match.
    lower = probe_name.lower()
    for key, layout in CATALOG.items():
        if key.lower().startswith(lower[:8]):
            return layout
    return CATALOG["Neuropixels 1.0"]
