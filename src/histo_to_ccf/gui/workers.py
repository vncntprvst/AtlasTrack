"""Thread workers for expensive operations."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from napari.qt.threading import thread_worker

from histo_to_ccf.sectioning.ordering import OrderedSection, order_sections
from histo_to_ccf.sectioning.split import detect_sections

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


@thread_worker
def detect_sections_worker(
    image: np.ndarray,
    *,
    min_area_px: int = 5000,
    closing_radius_px: int = 0,
) -> list[OrderedSection]:
    """Detect and order sections in a slide image."""
    sections = detect_sections(image, min_area_px=min_area_px, closing_radius_px=closing_radius_px)
    return order_sections(sections)


@thread_worker
def load_atlas_worker(atlas_id: str) -> "BrainGlobeAtlas":
    """Load a BrainGlobe atlas by ID."""
    from brainglobe_atlasapi import BrainGlobeAtlas
    return BrainGlobeAtlas(atlas_id)
