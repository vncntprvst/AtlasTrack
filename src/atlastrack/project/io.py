"""Load / save the persisted project JSON."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from atlastrack.project.schema import Project


def load_project(path: str | Path) -> Project:
    return Project.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_project(
    project: Project,
    path: str | Path,
    *,
    slide_images: Mapping[int, np.ndarray] | None = None,
    section_folders: bool = True,
) -> Path:
    """Write the project JSON, and the per-section folders next to it.

    ``slide_images`` (the slides' finished pixels, as the app holds them) spares
    reloading a slide when a section image has to be cut. See
    :mod:`atlastrack.project.section_files`.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if section_folders:
        from atlastrack.project.section_files import write_section_folders

        # First, so the JSON records each section's image path.
        write_section_folders(project, p, slide_images=slide_images)
    p.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    return p
