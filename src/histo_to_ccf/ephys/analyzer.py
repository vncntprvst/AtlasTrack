"""Reading spike features from a SpikeInterface ``SortingAnalyzer``.

This is the cheap half of the feature pipeline, and the reason most of the ephys
work needs no raw data at all. The analyzers written by the AIND pipeline
(``spike_sorting_output/postprocessed/*_group<N>.zarr``) and by SpikeInterface
(``analyzer_cache_*``) already carry ``spike_locations``, ``spike_amplitudes`` and
``unit_locations``. Measured on LO_06 2026-02-07/001 an analyzer set is **3.5 GB**
against **52 GB** of raw for the same recording, and it holds exactly the spike
depth and amplitude the raster and depth profiles need.

One analyzer per shank: the AIND stores are split ``group0``..``group3``, which for
a 4-shank NP2.0 probe are the four shanks.

SpikeInterface is imported lazily, as in :mod:`histo_to_ccf.ephys.loader`, so the
module can be imported without the ``ephys`` extra installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _require_si():
    try:
        import spikeinterface.full as si
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "SpikeInterface is required to read sorting analyzers; install it with "
            'pip install "histo-to-ccf[ephys]"'
        ) from exc
    return si


@dataclass
class SpikeFeatures:
    """Per-spike quantities for one shank, on the shank's own depth axis.

    ``depth_um`` is the localised depth of each spike in the probe frame (µm from
    the bottom of the recorded bank, *not* from the probe tip) - add the bank offset
    from :mod:`histo_to_ccf.ephys.recordings` to place it on the shank.
    """

    times_s: np.ndarray  # (n_spikes,)
    depth_um: np.ndarray  # (n_spikes,)
    amplitude: np.ndarray  # (n_spikes,)
    unit_ids: np.ndarray  # (n_spikes,) cluster id per spike
    channel_depth_um: np.ndarray  # (n_channels,) site axial position
    channel_x_um: np.ndarray  # (n_channels,) site column
    duration_s: float
    n_units: int
    source: str = ""

    def __post_init__(self) -> None:
        n = len(self.times_s)
        for name in ("depth_um", "amplitude", "unit_ids"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries but there are "
                    f"{n} spikes"
                )


def is_analyzer_dir(path: str | Path) -> bool:
    """Does ``path`` look like a saved SortingAnalyzer (zarr or binary folder)?"""
    p = Path(path)
    if not p.exists():
        return False
    # Both the zarr and binary_folder formats carry these two members.
    return (p / "sorting").exists() and (
        (p / "extensions").exists() or (p / "extensions.zarr").exists()
    )


def find_shank_analyzers(root: str | Path) -> list[Path]:
    """Find the per-shank analyzers under a processed-data folder, in group order.

    Handles both layouts in this dataset: AIND's
    ``spike_sorting_output/postprocessed/*_group<N>.zarr`` and SpikeInterface's
    ``analyzer_cache_*`` directories.
    """
    root = Path(root)
    if is_analyzer_dir(root):
        return [root]
    candidates: list[Path] = []
    for pattern in ("postprocessed/*.zarr", "*.zarr", "analyzer_cache_*"):
        candidates.extend(p for p in sorted(root.glob(pattern)) if is_analyzer_dir(p))
        if candidates:
            break

    def group_key(p: Path) -> tuple[int, str]:
        # "..._group3.zarr" -> 3, so shank order is numeric rather than "10" < "2".
        stem = p.name
        marker = "group"
        if marker in stem:
            tail = stem.split(marker)[-1]
            digits = "".join(c for c in tail if c.isdigit())
            if digits:
                return (int(digits), stem)
        return (10**6, stem)

    return sorted(dict.fromkeys(candidates), key=group_key)


def load_spike_features(path: str | Path) -> SpikeFeatures:
    """Read one shank's spike times, depths and amplitudes from an analyzer.

    Requires the ``spike_locations`` extension, which is what gives each spike a
    depth. Without it a raster cannot be drawn and the caller should say so rather
    than fall back to per-unit positions, which would silently coarsen the plot.
    """
    si = _require_si()
    analyzer = si.load_sorting_analyzer(str(path))

    locations = analyzer.get_extension("spike_locations")
    if locations is None:
        raise ValueError(
            f"{Path(path).name} has no 'spike_locations' extension, so spikes have "
            "no depth. Recompute it in SpikeInterface, or use the LFP features."
        )
    loc = locations.get_data()
    depth = np.asarray(loc["y"], dtype=float)

    amps_ext = analyzer.get_extension("spike_amplitudes")
    amplitude = (
        np.asarray(amps_ext.get_data(), dtype=float)
        if amps_ext is not None
        else np.zeros(depth.shape, dtype=float)
    )

    sorting = analyzer.sorting
    fs = float(sorting.get_sampling_frequency())
    # to_spike_vector() gives one record per spike with its frame and unit index,
    # in the same order as the spike_locations rows.
    spikes = sorting.to_spike_vector()
    times_s = np.asarray(spikes["sample_index"], dtype=float) / fs
    unit_index = np.asarray(spikes["unit_index"], dtype=int)
    unit_ids = np.asarray(sorting.unit_ids)[unit_index]

    channel_locations = np.asarray(analyzer.get_channel_locations(), dtype=float)

    return SpikeFeatures(
        times_s=times_s,
        depth_um=depth,
        amplitude=amplitude,
        unit_ids=unit_ids,
        channel_depth_um=channel_locations[:, 1],
        channel_x_um=channel_locations[:, 0],
        duration_s=float(times_s.max()) if times_s.size else 0.0,
        n_units=len(sorting.unit_ids),
        source=str(path),
    )


def restrict_to_epochs(
    features: SpikeFeatures, epochs: list[tuple[float, float]]
) -> SpikeFeatures:
    """Keep only spikes inside ``epochs`` (seconds), preserving everything else.

    An empty ``epochs`` means "no restriction" rather than "no spikes" - the caller
    that has not chosen excerpts yet should still see the whole recording.
    """
    if not epochs:
        return features
    keep = np.zeros(features.times_s.shape, dtype=bool)
    for t0, t1 in epochs:
        keep |= (features.times_s >= t0) & (features.times_s < t1)
    kept_duration = float(sum(max(0.0, t1 - t0) for t0, t1 in epochs))
    return SpikeFeatures(
        times_s=features.times_s[keep],
        depth_um=features.depth_um[keep],
        amplitude=features.amplitude[keep],
        unit_ids=features.unit_ids[keep],
        channel_depth_um=features.channel_depth_um,
        channel_x_um=features.channel_x_um,
        duration_s=kept_duration,
        n_units=features.n_units,
        source=features.source,
    )
