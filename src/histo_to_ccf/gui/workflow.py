"""Shared mutable state for the GUI session."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from histo_to_ccf.project.schema import Project, Slide

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


def crop_fingerprint(arr: np.ndarray) -> tuple:
    """Cheap content signature of a section crop (shape + pixel sum).

    Used to detect when a cached DeepSlice prediction has gone stale - e.g. after
    swapping the dye image on the *same* section (same bbox/index, new pixels). A
    fingerprint mismatch makes the caller re-run DeepSlice rather than silently
    reuse a prediction for the wrong image.
    """
    a = np.asarray(arr)
    return (tuple(a.shape), float(a.astype(np.float64).sum()))


@dataclass
class WorkflowState:
    """Holds all live state shared across GUI widgets."""

    project: Project = field(default_factory=Project)
    project_path: Path | None = None
    slide_images: dict[int, np.ndarray] = field(default_factory=dict)
    # Per-slide vertical (y_start, y_end) bands of each merged source image, so
    # section detection can order columns/rows per source slide (slide-aware).
    # A single-source slide has one band covering the whole image.
    slide_bands: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    _atlas: object = field(default=None, repr=False)
    active_slide_idx: int | None = None
    active_section_idx: int | None = None
    # DeepSlice "pre-match" cache: section.index -> atlas-frame anchoring9 (the full
    # predicted plane, incl. tilt), with a parallel per-crop fingerprint so a stale
    # entry is re-run instead of reused. Lets the Register step skip a second
    # DeepSlice pass when the user already pre-matched in the Atlas matcher.
    deepslice_anchorings: dict[int, list[float]] = field(default_factory=dict)
    deepslice_fingerprints: dict[int, tuple] = field(default_factory=dict)

    @property
    def atlas(self) -> "BrainGlobeAtlas | None":
        return self._atlas  # type: ignore[return-value]

    @atlas.setter
    def atlas(self, value: "BrainGlobeAtlas | None") -> None:
        self._atlas = value

    def reset(self) -> None:
        """Clear all project state for a fresh start (keeps the loaded atlas).

        Wipes the project, its images/bands, the active selection and the saved
        path so the GUI returns to its just-launched state without restarting.
        The loaded atlas object is intentionally kept in memory (it is expensive
        to reload and a new project usually targets the same atlas).
        """
        self.project = Project()
        self.project_path = None
        self.slide_images.clear()
        self.slide_bands.clear()
        self.deepslice_anchorings.clear()
        self.deepslice_fingerprints.clear()
        self.active_slide_idx = None
        self.active_section_idx = None

    def add_slide(self, image_path: str | Path, img: np.ndarray) -> int:
        """Append a slide to the project, store its image, return slide_idx."""
        slide_idx = len(self.project.slides)
        self.project.slides.append(Slide(image_path=str(image_path)))
        self.slide_images[slide_idx] = img
        return slide_idx
