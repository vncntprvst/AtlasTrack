"""Thread workers for expensive operations."""
from __future__ import annotations

import time
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
def load_atlas_worker(atlas_id: str, *, max_retries: int = 3) -> "BrainGlobeAtlas":
    """Load a BrainGlobe atlas, retrying up to ``max_retries`` times on failure."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    last_exc: Exception = RuntimeError("unknown error")
    for attempt in range(max_retries):
        try:
            return BrainGlobeAtlas(atlas_id)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


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
    """Run the full registration pipeline in a background thread (no progress)."""
    from histo_to_ccf.registration.pipeline import register_project_with_atlas

    return register_project_with_atlas(
        project,
        atlas,
        section_images=section_images,
        transforms_dir=transforms_dir,
        bspline_grid=bspline_grid,
        max_iterations=max_iterations,
    )


@thread_worker
def register_worker_progressive(
    project: "Project",
    atlas: "BrainGlobeAtlas",
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    *,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
):
    """Registration pipeline that yields per-section progress dicts.

    Each yielded value is a dict with keys:
        ``current``  int — sections completed so far
        ``total``    int — total sections to process
        ``msg``      str — human-readable status line

    The final ``return`` value is the updated :class:`Project`.
    """
    import SimpleITK as sitk

    from histo_to_ccf.atlas.planes import anchoring_from_plane_params
    from histo_to_ccf.io.ccf_coords import atlas_resolution_um
    from histo_to_ccf.registration.pipeline import (
        _apply_to_shank_registered,
        register_section_image,
    )
    from histo_to_ccf.registration.transforms import RegisteredSectionTransform

    # Collect work items: only sections with a plane AND a provided image.
    tasks: list[tuple[int, object, object]] = []
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            if section.plane is not None and section.index in section_images:
                tasks.append((slide_idx, slide, section))

    n_total = len(tasks)
    if n_total == 0:
        yield {"current": 0, "total": 0, "msg": "No sections to register."}
        return project

    tfm_dir = Path(transforms_dir)
    tfm_dir.mkdir(parents=True, exist_ok=True)
    res_um = atlas_resolution_um(atlas)
    registered: dict[tuple[int, int], RegisteredSectionTransform] = {}

    for i, (slide_idx, slide, section) in enumerate(tasks):
        yield {"current": i, "total": n_total, "msg": f"Registering section {section.index} ({i + 1}/{n_total})…"}

        img = section_images[section.index]
        anchoring = anchoring_from_plane_params(atlas, section.plane)
        reg, sitk_tf = register_section_image(
            img,
            atlas,
            anchoring=anchoring,
            bspline_grid=bspline_grid,
            max_iterations=max_iterations,
        )

        tfm_path = tfm_dir / f"section_{section.index:03d}.h5"
        sitk.WriteTransform(sitk_tf, str(tfm_path))
        reg.bspline_transform_path = str(tfm_path.relative_to(tfm_dir.parent))
        section.registration = reg

        registered[(slide_idx, section.index)] = RegisteredSectionTransform(
            anchoring=anchoring,
            output_size_px=reg.output_size_px,
            bspline=sitk_tf,
            atlas_resolution_um=res_um,
        )
        res_str = f"{reg.residual:.4f}" if reg.residual is not None else "n/a"
        yield {"current": i + 1, "total": n_total, "msg": f"Section {section.index} done (residual={res_str})"}

    # Project CCF positions onto shanks.
    for probe in project.probes:
        for shank in probe.shanks:
            _apply_to_shank_registered(shank, project, registered)

    return project
