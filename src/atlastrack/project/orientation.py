"""Which face of the sections the images show: seen from the front or the back.

A coronal section can be imaged from either face. Seen from the front (as atlas
plates are drawn) the animal's right is on the image's left; seen from the back
it is on the image's right. The tissue alone cannot tell the two apart - the
brain is almost symmetric - so a plane placed for the wrong one mirrors every ML
coordinate about the midline while its outlines still look right.

A project therefore says which it is (``Project.seen_from``), and everything that
places a plane follows it: DeepSlice and hand-placed planes are produced for the
front and mirrored for a back-view project. Older projects did not say; their
planes show which view they were placed for (see :func:`project_view`).

In an anchoring, ``u`` runs along the image's width: towards the animal's left
(ML grows) when seen from the front, towards its right when seen from the back.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from atlastrack.project.schema import Project, Section

View = Literal["front", "back"]

#: CCF ML extent in µm (every CCF-space atlas shares it).
_CCF_ML_EXTENT_UM = 11400.0


def anchoring_view(anchoring) -> View:
    """The view an anchoring was placed for."""
    return "front" if float(anchoring[5]) >= 0 else "back"


def mirror_anchoring(anchoring, ml_size: float) -> list[float]:
    """The same plane mirrored about the atlas midline (ML axis of ``ml_size`` voxels).

    Every image pixel keeps its AP and DV and gets the mirror-image ML.
    """
    a = [float(v) for v in anchoring]
    a[2] = ml_size - a[2]
    a[5] = -a[5]
    a[8] = -a[8]
    return a


def ml_voxels(project: Project) -> float:
    """The atlas's ML size in voxels, for :func:`mirror_anchoring`."""
    return _CCF_ML_EXTENT_UM / float(project.atlas.resolution_um)


def _plane(section: Section):
    if section.registration is not None:
        return section.registration.anchoring
    return section.deepslice_anchoring


def implied_view(project: Project) -> View | None:
    """The view most of the project's planes were placed for, or None if no planes."""
    views = [anchoring_view(a) for s in _sections(project) if (a := _plane(s)) is not None]
    if not views:
        return None
    return "back" if views.count("back") > views.count("front") else "front"


def project_view(project: Project, default: View = "front") -> View:
    """The project's view: as set, else as its planes imply, else ``default``."""
    return project.seen_from or implied_view(project) or default


def disagreeing_sections(project: Project) -> list[int]:
    """Sections whose plane was placed for the other view than the project's."""
    view = project_view(project)
    return [
        s.index for s in _sections(project)
        if (a := _plane(s)) is not None and anchoring_view(a) != view
    ]


def set_view(project: Project, view: View, *, mirror_planes: bool) -> tuple[list[int], list[int]]:
    """Set the project's view; with ``mirror_planes``, mirror the planes placed for the other.

    Returns ``(mirrored, skipped)`` section indices. A section matched to another
    project is skipped: its plane is fixed by its counterpart, not by a view, and
    matching again (which flips the section as the view asks) is what turns it.
    Warps and landmarks stay as they are - they live in image pixels.
    """
    project.seen_from = view
    mirrored, skipped = [], []
    if not mirror_planes:
        return mirrored, skipped
    size = ml_voxels(project)
    for s in _sections(project):
        plane = _plane(s)
        if plane is None or anchoring_view(plane) == view:
            continue
        if s.reference is not None:
            skipped.append(s.index)
            continue
        if s.registration is not None:
            s.registration.anchoring = mirror_anchoring(s.registration.anchoring, size)
        if s.deepslice_anchoring is not None:
            s.deepslice_anchoring = mirror_anchoring(s.deepslice_anchoring, size)
        mirrored.append(s.index)
    return mirrored, skipped


def _sections(project: Project):
    return [s for slide in project.slides for s in slide.sections]
