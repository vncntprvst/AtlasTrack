"""Lazy wrapper around :mod:`brainglobe_atlasapi`.

Keeps the rest of the codebase from importing BrainGlobe directly so we can
mock the atlas in tests without a network round-trip.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


@lru_cache(maxsize=4)
def get_atlas(name: str = "allen_mouse_25um") -> "BrainGlobeAtlas":
    """Return a cached BrainGlobe atlas instance. Downloads on first use."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    return BrainGlobeAtlas(name, check_latest=False)
