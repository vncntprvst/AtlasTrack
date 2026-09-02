"""Stitching several recordings of one penetration into one map per shank.

This is the step that makes a multi-recording penetration worth collecting. LO_07
ProbeA on 2026-05-08 is the case it was written for:

* ``LO_07_004`` reads electrodes 1-96 on **all four shanks** - 705 µm each, the
  bottom 15 % of a 4976 µm insertion.
* ``LO_07_005`` reads a **single column of one shank** over 5745 µm, most of the
  track, but on shank 0 only (ProbeB's equivalent is on shank 3).

Neither alone is enough: 005 leaves shanks 1-3 blank, and 004 covers 705 µm of
homogeneous reticular formation with no boundary in it to align to. Together shank 0
gets the whole track and the rest get their bottom bank, which is what the alignment
needs and what the discovery dialog exists to attach.

Three things have to be undone before two recordings can share an axis, and getting
any of them wrong is silent rather than loud:

**Bank offset.** Where a site sits on the shank. :func:`resolve_bank_offset` handles
the trap that the probe map may already report absolute positions.

**Insertion depth.** The probe is advanced between recordings, so the *same*
electrode is at a different brain depth in each. Spans are placed by depth below the
surface and then expressed in the frame of the deepest insertion - the position the
tip actually reached, and the one the histology track is measured in.

**Reference level.** Two recordings that referenced over different sets of channels
report different absolute power for the same tissue. The per-shank reference in
:func:`~histo_to_ccf.ephys.epochs.common_median_reference` removes most of it; what
is left is measured **in the depth range where the recordings overlap** and removed,
which is only possible because they overlap. :attr:`Contribution.level_offset_dec`
reports it, and a large value means the recordings disagree about tissue they both
saw - worth seeing rather than hiding.

Output is on a uniform depth grid because the display stretches the power map between
its first and last depth: a ragged axis would have the image silently misplaced
against the region column. Bins nothing covers are **NaN**, not zero - "no recording
reached here" and "the LFP is quiet here" must not look the same.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from histo_to_ccf.ephys.recordings import (
    NP2_ROW_PITCH_UM,
    channels_for_shank,
    resolve_bank_offset,
)
from histo_to_ccf.probes.geometry import SHANK_TIP_LENGTH_UM


@dataclass
class RecordingFeatures:
    """One recording's computed LFP, with everything needed to place it on a shank.

    ``axial_um`` is the site position as the probe map reports it - not yet depth
    from the tip, because whether the bank offset is already included is a property
    of the map, decided by :func:`resolve_bank_offset`.
    """

    label: str
    stream_name: str
    insertion_depth_um: float
    freqs_hz: np.ndarray
    psd: np.ndarray  # (n_channels, n_freq)
    axial_um: np.ndarray  # (n_channels,) probe y
    x_um: np.ndarray  # (n_channels,) probe x
    shank_ids: np.ndarray | None = None
    electrode_range: tuple[int, int] | None = None
    channel_ids: list = field(default_factory=list)
    reference_groups: int = 0

    @property
    def n_channels(self) -> int:
        return int(np.asarray(self.axial_um).size)


@dataclass
class Contribution:
    """What one recording contributed to one shank, and how well it agreed."""

    label: str
    stream_name: str
    n_channels: int
    top_um: float  # shallowest depth from the tip it reached
    bottom_um: float
    is_level_reference: bool = False
    #: log10 power offset removed to match the reference recording, in decades.
    #: ``None`` when there was no depth overlap to measure it in.
    level_offset_dec: float | None = None
    overlap_um: float = 0.0

    @property
    def extent_um(self) -> float:
        return self.bottom_um - self.top_um


@dataclass
class ShankStack:
    """One shank's LFP map, assembled from every recording that reached it.

    ``depth_from_tip_um`` is a uniform grid in the reference insertion's frame and
    **includes the 175 µm chisel tip**, so it is directly comparable with the
    histology track, whose tip is the physical tip and not the lowest electrode.
    """

    shank_index: int
    depth_from_tip_um: np.ndarray
    psd: np.ndarray  # (n_bins, n_freq), NaN where nothing was recorded
    freqs_hz: np.ndarray
    contributions: list[Contribution] = field(default_factory=list)
    reference_depth_um: float = 0.0
    bin_um: float = NP2_ROW_PITCH_UM

    @property
    def covered(self) -> np.ndarray:
        return np.isfinite(self.psd).any(axis=1)

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())

    def covered_spans_um(self) -> list[tuple[float, float]]:
        """Contiguous stretches of the grid that hold data, in µm from the tip."""
        return _runs(self.depth_from_tip_um, self.covered, self.bin_um)

    def gaps_um(self) -> list[tuple[float, float]]:
        """Stretches inside the recorded range that nothing covers.

        Reported so the alignment can say where it is interpolating blind instead of
        letting an unmeasured stretch look as well constrained as the rest.
        """
        return [(a[1], b[0]) for a, b in pairwise(self.covered_spans_um())]

    def describe(self) -> str:
        parts = []
        for c in self.contributions:
            bit = f"{c.label} {c.top_um:.0f}-{c.bottom_um:.0f} µm ({c.n_channels} ch)"
            if c.level_offset_dec is not None:
                bit += f", level {c.level_offset_dec:+.2f} dec"
            parts.append(bit)
        return "; ".join(parts)


def _runs(grid: np.ndarray, flag: np.ndarray, bin_um: float
          ) -> list[tuple[float, float]]:
    """Contiguous ``True`` runs of ``flag`` as (start, end) grid ranges."""
    grid = np.asarray(grid, dtype=float)
    flag = np.asarray(flag, dtype=bool)
    if grid.size == 0 or not flag.any():
        return []
    out: list[tuple[float, float]] = []
    start = None
    for i, on in enumerate(flag):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((float(grid[start]), float(grid[i - 1]) + bin_um))
            start = None
    if start is not None:
        out.append((float(grid[start]), float(grid[-1]) + bin_um))
    return out


def depths_from_tip(rec: RecordingFeatures, mask: np.ndarray) -> np.ndarray:
    """Depth from the physical tip for the masked channels of one recording.

    Adds the chisel tip length: the histology track ends at the physical tip, which
    is 175 µm below the lowest electrode (Neuropixels 2.0 spec). Leaving it out puts
    every channel 175 µm too deep - small, but the same size as the nuclei being
    aligned to.
    """
    axial = np.asarray(rec.axial_um, dtype=float)[mask]
    offset = resolve_bank_offset(axial, rec.electrode_range)
    return axial + offset + SHANK_TIP_LENGTH_UM


def _interp_to(freqs_from: np.ndarray, psd: np.ndarray,
               freqs_to: np.ndarray) -> np.ndarray:
    """Put one recording's PSD on another's frequency grid.

    Normally a no-op - the same reader settings give the same grid - but two
    recordings with different sampling rates would otherwise be silently misaligned
    in frequency, and interpolating is cheap next to detecting it and giving up.
    """
    src = np.asarray(freqs_from, dtype=float).ravel()
    dst = np.asarray(freqs_to, dtype=float).ravel()
    if src.size == dst.size and np.allclose(src, dst):
        return np.asarray(psd, dtype=float)
    out = np.empty((psd.shape[0], dst.size), dtype=float)
    for i, row in enumerate(np.asarray(psd, dtype=float)):
        out[i] = np.interp(dst, src, row, left=np.nan, right=np.nan)
    return out


def _bin_index(depths: np.ndarray, origin: float, bin_um: float) -> np.ndarray:
    return np.floor((np.asarray(depths, dtype=float) - origin) / bin_um).astype(int)


def _mean_per_bin(idx: np.ndarray, psd: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean PSD per depth bin; NaN for bins with no channel."""
    out = np.full((n_bins, psd.shape[1]), np.nan, dtype=float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # An all-NaN bin is normal (a frequency outside a recording's grid) and
        # nanmean warns about it; the NaN it returns is the right answer.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN")
        for b in np.unique(idx):
            if 0 <= b < n_bins:
                out[b] = np.nanmean(psd[idx == b], axis=0)
    return out


def _level_offset(ref: np.ndarray, other: np.ndarray) -> float | None:
    """Median log10 offset between two binned maps, over bins both cover.

    In decades of power, on the log scale the display uses, and by median so a few
    bad channels do not drag it. ``None`` when the two never overlap - in which case
    nothing can be said about their relative level and nothing should be invented.
    """
    both = np.isfinite(ref) & np.isfinite(other) & (ref > 0) & (other > 0)
    if not both.any():
        return None
    diff = np.log10(other[both]) - np.log10(ref[both])
    return float(np.median(diff))


def stack_shank(
    recordings: list[RecordingFeatures],
    shank_index: int,
    *,
    reference_depth_um: float | None = None,
    bin_um: float = NP2_ROW_PITCH_UM,
    match_levels: bool = True,
) -> ShankStack | None:
    """Combine every recording that reached ``shank_index`` into one map.

    ``None`` when no recording covered this shank - which is the honest outcome and
    the reason shanks 1-3 were blank when only ``LO_07_005`` had been computed.

    ``reference_depth_um`` is the insertion depth whose frame the result is
    expressed in; the deepest of the recordings by default, since that is where the
    tip actually ended up.
    """
    usable: list[tuple[RecordingFeatures, np.ndarray, np.ndarray]] = []
    for rec in recordings:
        mask = channels_for_shank(shank_index, rec.shank_ids, rec.x_um)
        if mask is None:
            # Nothing identifies the shanks. Attribute the recording to shank 0
            # rather than copying it onto all four, which is what once made four
            # tabs show one measurement.
            mask = np.zeros(rec.n_channels, dtype=bool)
            if shank_index == 0:
                mask[:] = True
        if not mask.any() or np.asarray(rec.psd).ndim != 2:
            continue
        usable.append((rec, mask, depths_from_tip(rec, mask)))
    if not usable:
        return None

    if reference_depth_um is None:
        reference_depth_um = max(float(r.insertion_depth_um) for r, _, _ in usable)
    freqs = np.asarray(usable[0][0].freqs_hz, dtype=float).ravel()

    # Place each recording in the reference insertion's frame. A site's depth below
    # the surface is insertion - from_tip, so in the reference frame its distance
    # from the tip becomes from_tip + (reference - insertion): a shallower insertion
    # had not pushed the tip as far, so its sites sit further up the final track.
    placed = []
    for rec, mask, from_tip in usable:
        shift = float(reference_depth_um) - float(rec.insertion_depth_um)
        placed.append((rec, mask, from_tip + shift))

    lo = min(float(d.min()) for _, _, d in placed)
    hi = max(float(d.max()) for _, _, d in placed)
    origin = np.floor(lo / bin_um) * bin_um
    n_bins = int(np.floor((hi - origin) / bin_um)) + 1
    grid = origin + np.arange(n_bins, dtype=float) * bin_um

    per_rec: list[np.ndarray] = []
    for rec, mask, depths in placed:
        psd = _interp_to(rec.freqs_hz, np.asarray(rec.psd, dtype=float)[mask], freqs)
        per_rec.append(_mean_per_bin(_bin_index(depths, origin, bin_um), psd, n_bins))

    # The widest recording is the level reference: it is the one whose absolute level
    # sets the map over most of the track, so matching the others to it moves the
    # least data.
    widest = int(np.argmax([float(d.max() - d.min()) for _, _, d in placed]))
    contributions: list[Contribution] = []
    for i, (rec, mask, depths) in enumerate(placed):
        offset = None
        both = np.isfinite(per_rec[widest]).any(1) & np.isfinite(per_rec[i]).any(1)
        overlap = float(both.sum()) * bin_um
        if i != widest and match_levels:
            offset = _level_offset(per_rec[widest], per_rec[i])
            if offset is not None:
                per_rec[i] = per_rec[i] / (10.0**offset)
        contributions.append(Contribution(
            label=rec.label,
            stream_name=rec.stream_name,
            n_channels=int(mask.sum()),
            top_um=float(depths.min()),
            bottom_um=float(depths.max()),
            is_level_reference=(i == widest),
            level_offset_dec=offset,
            overlap_um=0.0 if i == widest else overlap,
        ))

    stacked = np.stack(per_rec, axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN")
        combined = np.nanmean(stacked, axis=0)
    return ShankStack(
        shank_index=shank_index,
        depth_from_tip_um=grid,
        psd=combined,
        freqs_hz=freqs,
        contributions=contributions,
        reference_depth_um=float(reference_depth_um),
        bin_um=float(bin_um),
    )


def stack_penetration(
    recordings: list[RecordingFeatures],
    shank_indices,
    *,
    bin_um: float = NP2_ROW_PITCH_UM,
    match_levels: bool = True,
) -> dict[int, ShankStack]:
    """Stack every shank of one penetration in a single reference frame.

    The frame is shared across shanks deliberately: it belongs to the penetration,
    not to a shank, so a shank covered only by the shallower recording still lands
    where the deepest insertion put it.
    """
    if not recordings:
        return {}
    reference = max(float(r.insertion_depth_um) for r in recordings)
    out: dict[int, ShankStack] = {}
    for shank in shank_indices:
        stack = stack_shank(
            recordings, int(shank), reference_depth_um=reference,
            bin_um=bin_um, match_levels=match_levels,
        )
        if stack is not None:
            out[int(shank)] = stack
    return out
