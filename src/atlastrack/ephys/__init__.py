"""Ephys alignment: refine probe shank locations from LFP features.

Pure-core package (no napari / Qt). The SpikeInterface-dependent loader lives in
:mod:`atlastrack.ephys.loader` and gates its import so the base install stays
light; the depth-warping math (:mod:`alignment`) and LFP feature computation
(:mod:`features`) are plain numpy/scipy and fully testable on their own.
"""
from __future__ import annotations

from atlastrack.ephys.alignment import (
    apply_depth_alignment,
    channel_ccf_um,
    invert_anchors,
)
from atlastrack.ephys.epochs import (
    activity_score,
    artifact_score,
    candidate_windows,
    common_median_reference,
    rank_epochs,
    screen_window,
)
from atlastrack.ephys.features import (
    depth_profiles,
    lfp_band_power,
    lfp_psd,
    power_image,
    raster_points,
)
from atlastrack.ephys.recordings import (
    bank_offset_um,
    coverage_gaps_um,
    depth_below_surface_um,
    depth_from_tip_um,
    recording_span,
    resolve_bank_offset,
)

__all__ = [
    "activity_score",
    "apply_depth_alignment",
    "artifact_score",
    "bank_offset_um",
    "candidate_windows",
    "channel_ccf_um",
    "common_median_reference",
    "coverage_gaps_um",
    "depth_below_surface_um",
    "depth_from_tip_um",
    "depth_profiles",
    "invert_anchors",
    "lfp_band_power",
    "lfp_psd",
    "power_image",
    "rank_epochs",
    "raster_points",
    "recording_span",
    "resolve_bank_offset",
    "screen_window",
]
