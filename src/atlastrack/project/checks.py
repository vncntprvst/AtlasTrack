"""Checks run on a loaded project, reported to the user as plain notes."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlastrack.project.schema import Project


def mirrored_sections(project: Project) -> list[int]:
    """Sections whose plane was placed for the other face than the project's.

    Such a plane mirrors ML (and swaps the hemispheres) on that section while its
    outlines, the atlas being symmetric, still look right. See
    :mod:`atlastrack.project.orientation`.
    """
    from atlastrack.project.orientation import disagreeing_sections

    return disagreeing_sections(project)


def mirrored_sections_note(project: Project) -> str | None:
    """One-line warning about :func:`mirrored_sections`, or ``None``."""
    from atlastrack.project.orientation import project_view

    idx = mirrored_sections(project)
    if not idx:
        return None
    view = project_view(project)
    other = "front" if view == "back" else "back"
    shown = ", ".join(str(i) for i in idx[:12]) + (", ..." if len(idx) > 12 else "")
    matched = {s.index for sl in project.slides for s in sl.sections if s.reference is not None}
    fix = (
        "Match them again (Register > Match to a registered project)."
        if set(idx) <= matched
        else "Histology > Sections seen from can mirror them."
    )
    return (
        f"{len(idx)} section(s) have planes placed as if seen from the {other} ({shown}), "
        f"but this project's sections are seen from the {view}: their ML is mirrored. {fix}"
    )
