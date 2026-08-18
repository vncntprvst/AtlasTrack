"""Placing several recordings from one penetration on a shared depth axis.

A Neuropixels 2.0 bank reads 96 sites per shank. With 2 columns at a 15 µm row
pitch that is 48 rows = **720 µm of shank** - against insertion depths of 4.5-5.4 mm
in this dataset. One recording therefore constrains only the bottom ~15 % of a
track, which is why a single-recording alignment is so weakly determined.

The dataset is collected to fix that. Recordings on one insertion differ in two
ways, and both have to be undone before their features can be compared:

* **Different banks.** LO_06 2026-02-07 has banks 1-96, 97-192, 385-480 and
  1153-1248 on one insertion. Same probe position, sites further up the shank.
* **Different insertion depths.** LO_07 advances the probe between recordings
  (ProbeA 4532 -> 4520 -> 4576 -> 4976 µm), so the *same* electrode sits at a
  different brain depth in each.

Hence :func:`depth_below_surface_um`: depth from the tip is a property of the
probe, but only depth below the brain surface is comparable **across** recordings
taken at different insertion depths. Everything downstream works on that axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Neuropixels 2.0: two site columns per shank, one row every 15 µm, shanks 250 µm apart.
NP2_COLUMNS = 2
NP2_ROW_PITCH_UM = 15.0
SHANK_PITCH_UM = 250.0


def bank_offset_um(
    electrode_range: tuple[int, int] | None,
    *,
    columns: int = NP2_COLUMNS,
    row_pitch_um: float = NP2_ROW_PITCH_UM,
) -> float:
    """How far up the shank a bank's first site sits, in µm from the tip.

    ``electrode_range`` is 1-based and inclusive, the way the lab records it
    ("all shanks 97-192"). Electrodes fill the shank row by row, ``columns`` per
    row, so electrode *n* is in row ``(n - 1) // columns``.

    >>> bank_offset_um((1, 96))
    0.0
    >>> bank_offset_um((97, 192))
    720.0
    """
    if electrode_range is None:
        return 0.0
    first = int(electrode_range[0])
    if first < 1:
        raise ValueError(f"electrode numbers are 1-based; got {electrode_range!r}")
    return float((first - 1) // columns) * row_pitch_um


def resolve_bank_offset(
    axial_um: np.ndarray,
    electrode_range: tuple[int, int] | None,
    *,
    tolerance_um: float = NP2_ROW_PITCH_UM * 2,
) -> float:
    """How much to *add* to ``axial_um`` to get depth from the tip.

    The catch this exists for: when the probe map covers the whole shank -
    which it does for these recordings - SpikeInterface already reports the site's
    **absolute** position on the shank, so the bank offset is baked in. Measured on
    LO_06 2026-02-07: recording 001 (bank 1-96) reports y = 0-705 µm, and recording
    002 (bank 97-192) reports y = **720-1410 µm**, not 0-690.

    Adding :func:`bank_offset_um` to those would double-count it and push the
    recording 720 µm too shallow, inventing a gap between two banks that in fact
    abut. So: compare where the sites actually start against where the bank says
    they should, and return 0 when the geometry already accounts for it.

    Recordings whose probe map *was* rebuilt per bank (positions restarting at 0)
    still get the offset, so both conventions work.
    """
    axial = np.asarray(axial_um, dtype=float)
    expected = bank_offset_um(electrode_range)
    if expected == 0.0 or axial.size == 0:
        return expected
    if abs(float(axial.min()) - expected) <= tolerance_um:
        return 0.0  # already absolute
    return expected


def depth_from_tip_um(local_axial_um: np.ndarray, bank_offset: float) -> np.ndarray:
    """Depth along the shank from the tip, given the offset to add.

    Pass the result of :func:`resolve_bank_offset` rather than
    :func:`bank_offset_um` unless you know the positions are bank-local - see that
    function for why the difference bites.
    """
    return np.asarray(local_axial_um, dtype=float) + float(bank_offset)


def channels_for_shank(shank_index: int, shank_ids=None, x_um=None,
                       *, pitch_um: float = 250.0) -> np.ndarray | None:
    """Boolean mask of the channels belonging to one shank, or ``None`` if unknowable.

    **Returns an empty mask, not everything, when a shank was not recorded.** A
    recording need not cover every shank - LO_07_005 is a single column on one shank of
    a four-shank probe - and an earlier version fell back to "all channels" whenever it
    could not find one distinct id per shank. Every shank tab then showed the *same*
    LFP map: four identical panels look like four measurements, and they were one,
    copied.

    **x position wins over ``shank_ids``**, because the two answer different
    questions. x is absolute on the probe, so the rounded group *is* the physical
    shank. ``shank_ids`` (SpikeInterface's ``group``) numbers the groups *present in
    this recording* from zero: LO_07_005 ProbeB records one shank at x = 750/782 -
    physically shank 3 - and reports ``group = 0`` for every channel. Trusting the id
    there puts probe B's only data on the shank-0 tab and leaves shank 3 empty, which
    is worse than showing nothing because it looks like an answer.

    ``shank_ids`` is used only when there is no geometry to go on.
    """
    if x_um is not None:
        x = np.asarray(x_um, dtype=float).ravel()
        if x.size:
            return np.rint(x / float(pitch_um)).astype(int) == int(shank_index)
    if shank_ids is not None:
        ids = np.asarray([str(s).strip() for s in np.asarray(shank_ids).ravel()])
        if ids.size and all(i.isdigit() for i in ids):
            # Numeric ids we understand: an empty result means "this shank was not
            # recorded", which is a real answer, not a reason to fall back.
            return ids == str(int(shank_index))
    return None


def shank_index_from_x(x_um, *, pitch_um: float = 250.0) -> int | None:
    """Which shank a block of channels sits on, from x alone; ``None`` if it spans more.

    Columns within one shank are tens of µm apart while shanks are ``pitch_um`` apart,
    so a single rounded group means a single shank.
    """
    x = np.asarray(x_um, dtype=float).ravel()
    if x.size == 0:
        return None
    groups = np.unique(np.rint(x / float(pitch_um)).astype(int))
    return int(groups[0]) if groups.size == 1 else None


def depth_below_surface_um(
    depth_from_tip: np.ndarray, insertion_depth_um: float
) -> np.ndarray:
    """Convert depth-from-tip to depth below the brain surface.

    The tip is the deepest point, so a site ``d`` µm up the shank sits
    ``insertion_depth - d`` below the surface. Values below 0 are sites that were
    **above the brain surface** - which is not an error but a useful landmark: LFP
    power collapses there, pinning the surface. LO_06's bank 1153-1248 recording is
    exactly this ("reference/positioning only").
    """
    return float(insertion_depth_um) - np.asarray(depth_from_tip, dtype=float)


@dataclass(frozen=True)
class RecordingSpan:
    """Where one recording's sites land on the shared depth-below-surface axis."""

    label: str
    top_um: float  # smallest depth below surface (shallowest site)
    bottom_um: float  # largest depth below surface (deepest site)
    n_channels: int
    above_surface: bool  # any site above the brain surface

    @property
    def extent_um(self) -> float:
        return self.bottom_um - self.top_um


def recording_span(
    local_axial_um: np.ndarray,
    *,
    label: str,
    insertion_depth_um: float,
    electrode_range: tuple[int, int] | None = None,
    bank_offset: float | None = None,
) -> RecordingSpan:
    """Summarise one recording's coverage on the shared axis."""
    axial = np.asarray(local_axial_um, dtype=float)
    if axial.size == 0:
        raise ValueError(f"recording {label!r} has no channel positions")
    offset = (
        resolve_bank_offset(axial, electrode_range)
        if bank_offset is None
        else float(bank_offset)
    )
    depths = depth_below_surface_um(depth_from_tip_um(axial, offset), insertion_depth_um)
    return RecordingSpan(
        label=label,
        top_um=float(depths.min()),
        bottom_um=float(depths.max()),
        n_channels=int(axial.size),
        above_surface=bool(depths.min() < 0.0),
    )


def coverage_gaps_um(
    spans: list[RecordingSpan], *, min_gap_um: float = NP2_ROW_PITCH_UM
) -> list[tuple[float, float]]:
    """Depth ranges no recording covers, between the shallowest and deepest site.

    Reported so the alignment can say where it is interpolating blind rather than
    letting a landmark-free stretch look as well constrained as the rest.

    Gaps of at most ``min_gap_um`` are ignored. Two abutting banks are separated by
    exactly one row pitch - the deepest site of the upper bank sits 15 µm above the
    shallowest site of the lower one - and calling that a blind spot would bury the
    real gaps in noise. It is the site spacing, which every recording has anyway.
    """
    if not spans:
        return []
    intervals = sorted(((s.top_um, s.bottom_um) for s in spans), key=lambda ab: ab[0])
    gaps: list[tuple[float, float]] = []
    reach = intervals[0][1]
    for top, bottom in intervals[1:]:
        if top - reach > min_gap_um:
            gaps.append((reach, top))
        reach = max(reach, bottom)
    return gaps
