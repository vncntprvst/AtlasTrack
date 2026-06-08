"""Ephys alignment: refine probe shank locations from LFP features.

Pure-core package (no napari / Qt). The SpikeInterface-dependent loader lives in
:mod:`histo_to_ccf.ephys.loader` and gates its import so the base install stays
light; the depth-warping math (:mod:`alignment`) and LFP feature computation
(:mod:`features`) are plain numpy/scipy and fully testable on their own.
"""
from __future__ import annotations

from histo_to_ccf.ephys.alignment import (
    apply_depth_alignment,
    channel_ccf_um,
    invert_anchors,
)
from histo_to_ccf.ephys.features import lfp_psd, power_image

__all__ = [
    "apply_depth_alignment",
    "channel_ccf_um",
    "invert_anchors",
    "lfp_psd",
    "power_image",
]
