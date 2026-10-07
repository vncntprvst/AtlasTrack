"""Shared steps for importing a registration made by another tool.

An imported series arrives as one image per section, each with its own plane and,
often, its own non-linear correction. It becomes a project whose slide exists only
as those section images (see :mod:`atlastrack.project.section_files`): each image
is written into its section folder, the sections are laid out on a grid, and the
slide is put together from them when the project is opened.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from atlastrack.project.schema import (
    AtlasRef,
    ChannelColour,
    ManualLandmarks,
    PlaneParams,
    Project,
    RegistrationResult,
    Section,
    Slide,
)

#: Background gap between sections on the slide, in pixels.
GRID_GAP_PX = 40


@dataclass
class ImportedSection:
    """One section as an importer hands it over."""

    image: np.ndarray  # (H, W) or (H, W, 3), the pixels to register and show
    anchoring: list[float]  # (AP, DV, ML) atlas voxels over the image
    label: int  # the section's number in the source tool
    grid_cell: tuple[int, int] | None = None  # (row, column) on the slide
    landmarks: ManualLandmarks | None = None
    # Displacement field on the image grid, (H, W, 2) in (dx, dy): the atlas point
    # the plane puts at pixel p is drawn at p + field[p]. Becomes the section's warp.
    warp: np.ndarray | None = None
    # Its exact inverse on the same grid, when the source tool defines one:
    # section pixel q -> q + warp_inverse[q] on the plane. Used to map points.
    warp_inverse: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def rescale(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize by ``factor``, keeping the dtype; pixel centres scale about the corner."""
    if abs(factor - 1.0) < 1e-6:
        return image
    from skimage.transform import resize

    h, w = image.shape[:2]
    shape = (round(h * factor), round(w * factor), *image.shape[2:])
    out = resize(image, shape, order=1, anti_aliasing=factor < 1, preserve_range=True)
    return out.astype(image.dtype)


def rescale_points(xy: np.ndarray, factor: float) -> np.ndarray:
    """Pixel coordinates after :func:`rescale` by ``factor``."""
    return (np.asarray(xy, dtype=float) + 0.5) * factor - 0.5


def grid_cells(n: int) -> list[tuple[int, int]]:
    """A near-square grid, filled row by row."""
    n_cols = max(1, math.ceil(math.sqrt(n)))
    return [divmod(i, n_cols) for i in range(n)]


def layout(shapes: list[tuple[int, int]], cells: list[tuple[int, int]]):
    """Boxes ``(x0, y0, x1, y1)`` for images of ``(h, w)`` placed on a grid."""
    n_rows = max(r for r, _ in cells) + 1
    n_cols = max(c for _, c in cells) + 1
    row_h = [0] * n_rows
    col_w = [0] * n_cols
    for (h, w), (r, c) in zip(shapes, cells, strict=True):
        row_h[r] = max(row_h[r], h)
        col_w[c] = max(col_w[c], w)
    y0 = np.cumsum([0] + [h + GRID_GAP_PX for h in row_h])
    x0 = np.cumsum([0] + [w + GRID_GAP_PX for w in col_w])
    return [
        (int(x0[c]), int(y0[r]), int(x0[c]) + w, int(y0[r]) + h)
        for (h, w), (r, c) in zip(shapes, cells, strict=True)
    ]


def write_warp(
    path: Path,
    warp: np.ndarray,
    inverse: np.ndarray | None = None,
    *,
    spacing_px: float = 1.0,
) -> None:
    """Write a displacement field as the SimpleITK transform a registration stores.

    An exact ``inverse`` goes beside it (see
    :func:`atlastrack.registration.transforms.inverse_warp_path`). With
    ``spacing_px`` > 1 the fields hold one value every that many pixels, and are
    interpolated between - plenty for a smooth warp, and far smaller on disk.
    """
    import SimpleITK as sitk

    from atlastrack.registration.transforms import inverse_warp_path

    path.parent.mkdir(parents=True, exist_ok=True)
    for field_, target in ((warp, path), (inverse, inverse_warp_path(path))):
        if field_ is None:
            continue
        image = sitk.GetImageFromArray(np.asarray(field_, dtype=np.float64), isVector=True)
        image.SetSpacing((float(spacing_px), float(spacing_px)))
        sitk.WriteTransform(sitk.DisplacementFieldTransform(image), str(target))


def write_imported_project(
    sections: list[ImportedSection],
    out_dir: Path,
    name: str,
    *,
    atlas_name: str,
    resolution_um: float,
    plane_for: dict[int, PlaneParams] | None = None,
    channel_names: dict[ChannelColour, str] | None = None,
    align_channel: ChannelColour | None = None,
    pixel_um: float | None = None,
    section_spacing_um: float | None = None,
    finish=None,
) -> Path:
    """Write section images, warps and ``<name>.atlastrack.json`` into ``out_dir``.

    ``finish(project)``, if given, is called on the new project just before it is
    saved (to add what only the importer knows, such as counted cells).

    Sections keep the order given, which is their series order. Returns the
    project path.
    """
    from atlastrack.project.io import save_project
    from atlastrack.project.section_files import transforms_folder, write_section_image
    from atlastrack.project.series import plane_values

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    project_path = out_dir / f"{name}.atlastrack.json"
    if all(s.grid_cell is not None for s in sections):
        cells = [s.grid_cell for s in sections]
    else:
        cells = grid_cells(len(sections))
    boxes = layout([s.image.shape[:2] for s in sections], cells)  # type: ignore[arg-type]

    project_sections = []
    for i, (imp, box) in enumerate(zip(sections, boxes, strict=True)):
        x0, y0, x1, y1 = box
        h, w = y1 - y0, x1 - x0
        reg = RegistrationResult(anchoring=list(imp.anchoring), output_size_px=(h, w))
        if imp.warp is not None:
            rel = f"{transforms_folder(project_path)}/section_{i:03d}.h5"
            write_warp(out_dir / rel, imp.warp, imp.warp_inverse)
            reg.bspline_transform_path = rel
        plane = (plane_for or {}).get(i)
        if plane is None:
            plane = PlaneParams(
                ap_um=plane_values(imp.anchoring, (h, w), resolution_um).ap_um
            )
        section = Section(
            index=i,
            slide_idx=0,
            bbox_px=box,
            ap_order=i,
            slide_number=imp.label,
            plane=plane,
            ap_source="manual",
            registration=reg,
            manual_landmarks=imp.landmarks,
        )
        write_section_image(project_path, section, imp.image)
        project_sections.append(section)

    project = Project(
        atlas=AtlasRef(name=atlas_name, resolution_um=resolution_um),
        slides=[Slide(
            image_path="",
            sections=project_sections,
            channel_names=channel_names or {},
            align_channel=align_channel,
            pixel_um=pixel_um,
        )],
        section_spacing_um=section_spacing_um,
        # slicereg and the QUINT tools show sections as atlas plates do: from the front.
        seen_from="front",
    )
    if finish is not None:
        finish(project)
    # The section images are already in place; this writes each section.json.
    save_project(project, project_path)
    return project_path
