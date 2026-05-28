"""Thread workers for expensive operations."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.qt.threading import thread_worker

from histo_to_ccf.sectioning.ordering import OrderedSection, order_sections
from histo_to_ccf.sectioning.split import detect_sections

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from histo_to_ccf.project.schema import Project


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


@thread_worker
def register_worker(
    project: "Project",
    atlas: "BrainGlobeAtlas",
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    *,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
) -> "Project":
    """Run the M3 registration pipeline in a background thread."""
    from histo_to_ccf.registration.pipeline import register_project_with_atlas
    return register_project_with_atlas(
        project,
        atlas,
        section_images=section_images,
        transforms_dir=transforms_dir,
        bspline_grid=bspline_grid,
        max_iterations=max_iterations,
    )
