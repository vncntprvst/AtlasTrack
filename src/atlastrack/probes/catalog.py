"""Probe type catalog: channel layouts for common Neuropixels models.

Depths are measured from the physical tip of the probe (tip = 0 µm).
Lateral offsets are measured from the shank centreline (positive = right
when viewed from the front face).

References
----------
- NP 1.0: https://www.neuropixels.org/probe10a (imec)
- NP 2.0: https://www.neuropixels.org/probe20 (imec)
- Neuropixels geometry: imec's ProbeTable (https://github.com/billkarsh/ProbeTable),
  as copied into probeinterface's ``neuropixels_probe_features.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ProbeLayout:
    """Physical recording-site layout for one probe model.

    Most Neuropixels models are a regular interleaved grid, fully described by
    the parametric fields (``tip_to_first_site_um``  ``col_pitch_um``).
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
        col 1 row 0, col 0 row 1, ) matching how channels are typically
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

#: From the top of a Neuropixels shank's taper to the centre of its lowest electrode.
#: ProbeTable does not give it; probeinterface uses this value to draw the probe.
_NP_TAPER_TOP_TO_LOWEST_SITE_UM = 11.0


def _neuropixels(
    name: str,
    *,
    n_channels: int,
    tip_length_um: float,
    row_pitch_um: float,
    even_row_x_um: float,
    odd_row_x_um: float,
    col_pitch_um: float = 32.0,
    shank_width_um: float = 70.0,
) -> ProbeLayout:
    """The lowest ``n_channels`` electrodes of one Neuropixels shank.

    Values are ProbeTable's: ``tip_length_um`` is the taper, and ``*_row_x_um`` is
    the distance from the shank's left edge to the left electrode of a row (row 0
    is an even row). Both electrodes of a row are at the same depth; on NP 1.0 the
    rows alternate between two x positions, on NP 2.0 they do not.

    The lowest electrode sits ``tip_length_um + 11`` µm above the physical tip
    (220 µm on NP 1.0, 217 µm on NP 2.0). This used to be 175 µm on NP 1.0 and 0 on
    NP 2.0, which put every NP 2.0 electrode about 0.2 mm too deep.
    """
    tip_to_lowest = tip_length_um + _NP_TAPER_TOP_TO_LOWEST_SITE_UM
    depths, offsets = [], []
    for k in range(n_channels):
        row, col = divmod(k, 2)
        left = even_row_x_um if row % 2 == 0 else odd_row_x_um
        depths.append(tip_to_lowest + row * row_pitch_um)
        offsets.append(left + col * col_pitch_um - shank_width_um / 2.0)
    return ProbeLayout(
        name=name,
        n_channels=n_channels,
        tip_to_first_site_um=tip_to_lowest,
        n_columns=2,
        col_pitch_um=col_pitch_um,
        explicit_depths_um=tuple(depths),
        explicit_offsets_um=tuple(offsets),
    )


def _neuronexus_a1x32_poly3() -> ProbeLayout:
    """Build the NeuroNexus A1x32-Poly3-10mm-25s-177(-OA32LP) site layout.

    Geometry from the NeuroNexus catalog drawing for this exact part number
    (*Penetrating Probes*, p. 52; the 5 mm sibling on p. 49 is the same site layout),
    which agrees with the rig's own RHX ``A1x32-Poly3-A32-RHD2132-probe.xml``:

      * three columns at -18 / 0 / +18 µm from the centreline
      * the centre column carries 12 sites at 25 µm pitch; each side column carries
        10, **interleaved** by half a pitch rather than aligned with the centre rows
      * the shank tapers to a point 62 µm below the lowest site

    The drawing quotes 18 µm, 25 µm and 22 µm: the last is the centre-to-side site
    distance, and sqrt(18^2 + 12.5^2) = 21.9 µm is what fixes the interleave at half
    a pitch. Total site span 275 µm.

    These values previously came from rescaling the ProbeInterface entry
    ``neuronexus / A1x32-Poly3-10mm-50-177`` to a 25 µm pitch, which put every site
    38 µm too deep and left the side columns level with the centre rows. That entry is
    correct for *its own* part - the catalog drawing on p. 53 gives 50 µm columns, a
    550 µm span and a 100 µm tip, exactly as ProbeInterface has it. The 25 µm probe is
    a different design, not a scaled one, and ProbeInterface has no entry for it.

    The OA32LP optical assembly carries a fibre 50 µm above the top-most site;
    that offset is recorded as metadata and does not affect the site
    coordinates.  Sites are ordered tip -> base (ascending depth), ties broken
    left -> right, matching the channel convention used elsewhere in the catalog.

    This is the *site layout* only. Which recording channel each site lands on
    depends on the adapter; see :mod:`atlastrack.ephys.probemap`.
    """
    pitch = 25.0
    tip_to_lowest_site = 62.0
    left, centre, right = -18.0, 0.0, 18.0

    sites: list[tuple[float, float]] = []  # (depth_from_tip, lateral_offset)
    for row in range(12):
        sites.append((tip_to_lowest_site + row * pitch, centre))
    for row in range(10):
        depth = tip_to_lowest_site + pitch / 2.0 + row * pitch
        sites.append((depth, left))
        sites.append((depth, right))
    sites.sort(key=lambda s: (s[0], s[1]))

    return ProbeLayout(
        name=NEURONEXUS_A1X32_POLY3,
        n_channels=32,
        explicit_depths_um=tuple(d for d, _ in sites),
        explicit_offsets_um=tuple(o for _, o in sites),
        fiber_offset_above_top_site_um=50.0,
    )


CATALOG: dict[str, ProbeLayout] = {
    # ProbeTable NP1000.
    "Neuropixels 1.0": _neuropixels(
        "Neuropixels 1.0", n_channels=384, tip_length_um=209.0, row_pitch_um=20.0,
        even_row_x_um=27.0, odd_row_x_um=11.0,
    ),
    # ProbeTable NP2013 / NP2014 / NP2020 / NP2021 (all the same shank). The lowest
    # 384 electrodes of each shank: which electrodes were recorded is in the
    # recording's own channel map, not here.
    "Neuropixels 2.0 (4-shank)": _neuropixels(
        "Neuropixels 2.0 (4-shank)", n_channels=384, tip_length_um=206.0,
        row_pitch_um=15.0, even_row_x_um=27.0, odd_row_x_um=27.0,
    ),
    NEURONEXUS_A1X32_POLY3: _neuronexus_a1x32_poly3(),
}


#: Physical tip to the lowest electrode of a Neuropixels 2.0 shank (217 µm).
NP2_TIP_TO_LOWEST_SITE_UM: float = CATALOG["Neuropixels 2.0 (4-shank)"].tip_to_first_site_um


def tip_to_lowest_site_um(probe_name: str) -> float:
    """Distance from the physical tip to the lowest electrode of ``probe_name``."""
    return float(get_layout(probe_name).site_depths_from_tip_um().min())


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
