"""Combining several recordings of one penetration into one depth profile.

Each recording contributes a window of the track (see
:mod:`atlastrack.ephys.recordings`); this module puts them on the common
depth-below-surface axis and merges them into the arrays the viewer plots.

Two things it deliberately does **not** do:

* **It does not blend overlapping recordings into an average.** Where two
  recordings cover the same depth they are kept separate and the overlap is
  reported. Averaging would hide a disagreement between them, and a disagreement
  is exactly the evidence that an insertion depth or bank label is wrong - as the
  LO_04 2025-08-26 bank discrepancy in ``docs/dataset.md`` shows.
* **It does not interpolate across gaps.** A stretch no recording covers stays
  empty, so the viewer can grey it out rather than letting a landmark-free region
  look as well constrained as the rest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atlastrack.ephys.recordings import (
    RecordingSpan,
    coverage_gaps_um,
    depth_below_surface_um,
    depth_from_tip_um,
    resolve_bank_offset,
)


@dataclass
class RecordingProfile:
    """One recording's contribution to a penetration, on the shared axis."""

    label: str
    span: RecordingSpan
    # Spike raster, already on the shared axis. Empty when the recording has no
    # sorting - which is normal (LO_06 2026-02-07/003 and /004 were never sorted).
    spike_times_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    spike_depth_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    spike_amplitude: np.ndarray = field(default_factory=lambda: np.empty(0))
    duration_s: float = 0.0
    # Per-channel LFP power, if raw excerpts were read. Empty otherwise.
    channel_depth_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    lfp_psd: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    lfp_freqs_hz: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def has_spikes(self) -> bool:
        return self.spike_times_s.size > 0

    @property
    def has_lfp(self) -> bool:
        return self.lfp_psd.size > 0


def to_shared_axis(
    values_axial_um: np.ndarray,
    *,
    insertion_depth_um: float,
    electrode_range: tuple[int, int] | None = None,
    bank_offset: float | None = None,
    reference_axial_um: np.ndarray | None = None,
) -> np.ndarray:
    """Map probe-frame axial positions onto depth below the brain surface.

    ``reference_axial_um`` is the recording's **channel** positions, used to decide
    whether the bank offset is already baked into the geometry. Pass it whenever the
    values being mapped are spike depths: spikes localise slightly beyond the site
    span (measured -31 to 734 µm for a 0-705 µm bank), so deciding from the spikes
    themselves could misjudge the convention by a row.
    """
    axial = np.asarray(values_axial_um, dtype=float)
    reference = reference_axial_um if reference_axial_um is not None else axial
    offset = (
        resolve_bank_offset(np.asarray(reference, dtype=float), electrode_range)
        if bank_offset is None
        else float(bank_offset)
    )
    return depth_below_surface_um(depth_from_tip_um(axial, offset), insertion_depth_um)


@dataclass
class PenetrationProfile:
    """Every recording on one penetration, merged onto the shared depth axis."""

    profiles: list[RecordingProfile] = field(default_factory=list)

    @property
    def spans(self) -> list[RecordingSpan]:
        return [p.span for p in self.profiles]

    def depth_range_um(self) -> tuple[float, float]:
        """Shallowest and deepest site across all recordings."""
        if not self.profiles:
            return (0.0, 0.0)
        return (
            min(p.span.top_um for p in self.profiles),
            max(p.span.bottom_um for p in self.profiles),
        )

    def gaps_um(self) -> list[tuple[float, float]]:
        """Depth stretches no recording covers."""
        return coverage_gaps_um(self.spans)

    def overlaps_um(self) -> list[tuple[float, float, str, str]]:
        """Depth stretches two recordings both cover, with their labels.

        Reported rather than merged: where two recordings overlap, their features
        should agree, and comparing them there is the only independent check on the
        insertion depths and bank labels that placed them.
        """
        out: list[tuple[float, float, str, str]] = []
        ordered = sorted(self.profiles, key=lambda p: p.span.top_um)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                lo = max(a.span.top_um, b.span.top_um)
                hi = min(a.span.bottom_um, b.span.bottom_um)
                if hi > lo:
                    out.append((lo, hi, a.span.label, b.span.label))
        return out

    def coverage_fraction(self, track_length_um: float) -> float:
        """Fraction of the track any recording covers, overlaps counted once."""
        if track_length_um <= 0 or not self.profiles:
            return 0.0
        top, bottom = self.depth_range_um()
        covered = (bottom - top) - sum(hi - lo for lo, hi in self.gaps_um())
        return float(np.clip(covered / track_length_um, 0.0, 1.0))

    def all_spikes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every recording's spikes concatenated: (depth_um, amplitude, time_s).

        Times are *not* made continuous across recordings - each keeps its own
        clock, because they are separate acquisitions and a merged time axis would
        imply a continuity that does not exist. Depth is the axis that is shared.
        """
        if not self.profiles:
            return (np.empty(0), np.empty(0), np.empty(0))
        with_spikes = [p for p in self.profiles if p.has_spikes]
        if not with_spikes:
            return (np.empty(0), np.empty(0), np.empty(0))
        return (
            np.concatenate([p.spike_depth_um for p in with_spikes]),
            np.concatenate([p.spike_amplitude for p in with_spikes]),
            np.concatenate([p.spike_times_s for p in with_spikes]),
        )
