"""Orchestrate the per-section registration pipeline.

For M1 this is intentionally minimal: it takes a Project, a PlanePredictor, and
fills each shank's tip/entry CCF coordinates by composing the per-section
:class:`SectionTransform`. There is no B-spline refinement yet.
"""
from __future__ import annotations

from loguru import logger

from histo_to_ccf.project.schema import Project, Section, Shank
from histo_to_ccf.registration.predictor import PlanePredictor
from histo_to_ccf.registration.transforms import SectionTransform


def _section_transform_for(section: Section, predictor: PlanePredictor) -> SectionTransform:
    if section.plane is None:
        # The predictor would normally see the cropped section image here.
        # For M1 the ManualPredictor ignores the image, so we pass None safely.
        plane = predictor.predict(None, section_index=section.index)  # type: ignore[arg-type]
    else:
        plane = section.plane
    return SectionTransform(plane=plane)


def register_project(project: Project, predictor: PlanePredictor) -> Project:
    """Fill in tip/entry CCF coordinates for every shank in every probe.

    Mutates and returns the project in place.
    """
    # Build a transform per (slide_idx, section_idx).
    transforms: dict[tuple[int, int], SectionTransform] = {}
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            transforms[(slide_idx, section.index)] = _section_transform_for(
                section, predictor
            )

    for probe in project.probes:
        for shank in probe.shanks:
            _apply_to_shank(shank, project, transforms)
    return project


def _apply_to_shank(
    shank: Shank,
    project: Project,
    transforms: dict[tuple[int, int], SectionTransform],
) -> None:
    if shank.tip_px is not None and shank.tip_section_idx is not None:
        tx = _lookup_transform(project, shank.tip_section_idx, transforms)
        if tx is None:
            logger.warning("no transform for tip section idx {}", shank.tip_section_idx)
        else:
            shank.tip_ccf_um = tx.apply(shank.tip_px.x_px, shank.tip_px.y_px)

    if shank.entry_px is not None and shank.entry_section_idx is not None:
        tx = _lookup_transform(project, shank.entry_section_idx, transforms)
        if tx is None:
            logger.warning("no transform for entry section idx {}", shank.entry_section_idx)
        else:
            shank.entry_ccf_um = tx.apply(shank.entry_px.x_px, shank.entry_px.y_px)


def _lookup_transform(
    project: Project,
    section_idx: int,
    transforms: dict[tuple[int, int], SectionTransform],
) -> SectionTransform | None:
    """Find a transform by section index, scanning slides in order."""
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            if section.index == section_idx:
                return transforms.get((slide_idx, section.index))
    return None
