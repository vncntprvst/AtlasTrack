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
    """Physical recording-site layout for one probe model."""

    name: str
    n_channels: int
    tip_to_first_site_um: float
    site_row_pitch_um: float
    n_columns: int
    col_pitch_um: float

    def site_depths_from_tip_um(self) -> np.ndarray:
        """Depth of each recording site from the probe tip (µm).

        Sites are assigned in column-interleaved order (col 0 row 0,
        col 1 row 0, col 0 row 1, …) matching how channels are typically
        numbered from tip to base.
        """
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

CATALOG: dict[str, ProbeLayout] = {
    "Neuropixels 1.0": ProbeLayout(
        name="Neuropixels 1.0",
        n_channels=384,
        tip_to_first_site_um=175.0,  # base of the taper
        site_row_pitch_um=10.0,       # 10 µm between alternating rows (20 µm same-col)
        n_columns=2,
        col_pitch_um=32.0,            # ±16 µm from centreline
    ),
    "Neuropixels 2.0 (1-shank)": ProbeLayout(
        name="Neuropixels 2.0 (1-shank)",
        n_channels=384,
        tip_to_first_site_um=0.0,     # first site very close to tip
        site_row_pitch_um=7.5,        # 15 µm same-col pitch
        n_columns=2,
        col_pitch_um=32.0,
    ),
    "Neuropixels 2.0 (4-shank)": ProbeLayout(
        name="Neuropixels 2.0 (4-shank)",
        n_channels=384,               # per shank
        tip_to_first_site_um=0.0,
        site_row_pitch_um=7.5,
        n_columns=2,
        col_pitch_um=32.0,
    ),
    "Neuropixels Ultra": ProbeLayout(
        name="Neuropixels Ultra",
        n_channels=384,
        tip_to_first_site_um=0.0,
        site_row_pitch_um=3.0,        # 6 µm same-col pitch (dense packing)
        n_columns=2,
        col_pitch_um=6.0,
    ),
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
