"""One folder per section, written next to the project on save.

::

    <name>.atlastrack.json
    <name>_sections/section_003/
        image.tif       the section's box cut from the slide, flips and rotation applied
        section.json    the section in readable numbers (written for people, not read back)
    <name>_transforms/section_003.h5    its fitted warp, when it has one

Folders carry the project's name, so several projects can share a folder without
writing over each other's files (a plain ``transforms/`` used to be shared). A
project keeps the folders it already has: ``section.json`` records which project
wrote it, and a project saved under a new name gets folders of its own.

The slide stays the normal way to work: sections are found on it, and their
images are cut from it when the project is saved. An image is cut again only when
its box, flips or rotation changed since. If the slide image is missing when a
project is opened, the slide is put back together from these images (see
:func:`atlastrack.project.images.rebuild_slide_image`), so a project can be shared
without its slide. An importer can also write a project that has only these.

``section.json`` repeats what the project file says about the section in plain
numbers - the plane as AP, pitch, yaw, rotation and scale, the landmarks as image
pixels paired with atlas µm. The project file stays the only one AtlasTrack reads.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from atlastrack.project.schema import Project, Section, Slide

def project_name(project_path: str | Path) -> str:
    """A project's name: its file name without ``.atlastrack.json`` / ``.json``."""
    name = Path(project_path).name
    for suffix in (".atlastrack.json", ".json"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def section_folder(project_path: str | Path, index: int) -> str:
    """Project-relative folder for section ``index`` of this project."""
    return f"{project_name(project_path)}_sections/section_{index:03d}"


def transforms_folder(project_path: str | Path) -> str:
    """Project-relative folder for this project's warp files."""
    return f"{project_name(project_path)}_transforms"


def folder_for(project_path: str | Path, section: Section) -> str:
    """The folder this section's files go in: the one it has, if this project's."""
    if section.image_path:
        current = Path(section.image_path).parent
        summary = _read_json(Path(project_path).parent / current / "section.json")
        name = Path(project_path).name
        # Files from before "project" was recorded name it in their note.
        owner = summary.get("project")
        if owner == name or (owner is None and f"when {name} was saved" in summary.get("note", "")):
            return current.as_posix()
    return section_folder(project_path, section.index)


def crop_key(slide: Slide, section: Section) -> dict:
    """Everything that decides a section image's pixels; a change means re-cut."""
    return {
        "slide_image": list(slide.source_paths) or [slide.image_path],
        "slide_flip_h": slide.flip_h,
        "slide_flip_v": slide.flip_v,
        "bbox_px": list(section.bbox_px),
        "flip_h": section.flip_h,
        "flip_v": section.flip_v,
        "rotation_deg": float(section.rotation_deg or 0.0),
    }


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_section_image(
    project_path: str | Path, section: Section, pixels: np.ndarray, folder: str | None = None
) -> str:
    """Write a section's image; returns and records its project-relative path."""
    import tifffile

    rel = f"{folder or section_folder(project_path, section.index)}/image.tif"
    path = Path(project_path).parent / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = pixels.ndim == 3 and pixels.shape[-1] in (3, 4)
    tifffile.imwrite(
        str(path), np.ascontiguousarray(pixels),
        photometric="rgb" if rgb else "minisblack", compression="zlib",
    )
    section.image_path = rel
    return rel


def _landmarks_in_atlas(
    project: Project, section: Section, project_dir: Path
) -> list[dict] | None:
    """Each landmark as its image pixel and the atlas point placed there (µm)."""
    lm = section.manual_landmarks
    if lm is None or section.registration is None:
        return None
    from atlastrack.atlas.planes import Anchoring
    from atlastrack.registration.transforms import RegisteredSectionTransform

    reg = section.registration
    bspline = None
    if reg.bspline_transform_path:
        import SimpleITK as sitk

        bspline = sitk.ReadTransform(str(project_dir / reg.bspline_transform_path))
    res = float(project.atlas.resolution_um)
    # The source points sit on the registered overlay, so they are mapped
    # without the landmark correction: that gives the atlas point placed there.
    transform = RegisteredSectionTransform(
        anchoring=Anchoring.from_iterable(reg.anchoring),
        output_size_px=tuple(reg.output_size_px),
        bspline=bspline,
        atlas_resolution_um=(res, res, res),
    )
    ccf = transform.apply_many(np.asarray(lm.source, dtype=float))
    return [
        {
            "image_px": [round(float(tx), 2), round(float(ty), 2)],
            "atlas_um": {"ap": round(ap, 1), "ml": round(ml, 1), "dv": round(dv, 1)},
        }
        for (tx, ty), (ap, ml, dv) in zip(lm.target, ccf.tolist(), strict=True)
    ]


def section_summary(project: Project, slide: Slide, section: Section,
                    project_dir: Path, project_file: str) -> dict:
    """The readable ``section.json`` content for one section."""
    from atlastrack.project.series import section_plane_values

    x0, y0, x1, y1 = section.bbox_px
    out: dict = {
        "note": (
            f"Written by AtlasTrack when {project_file} was saved, for reading. "
            "Editing this file does not change the project."
        ),
        "project": project_file,
        "index": section.index,
        "slide_number": section.slide_number,
        "series_position": section.ap_order,
        "image": section.image_path,
        "size_px": [x1 - x0, y1 - y0],
        "pixel_um": slide.pixel_um,
        "crop": crop_key(slide, section),
        "atlas": project.atlas.name,
    }
    values = section_plane_values(section, float(project.atlas.resolution_um))
    if values is not None:
        out["plane"] = values.as_dict()
        out["plane"]["source"] = "registration" if section.registration else "deepslice"
    elif section.plane is not None:
        out["plane"] = {"ap_um": section.plane.ap_um, "source": section.ap_source}
    if section.registration is not None:
        reg = section.registration
        out["registration"] = {
            "anchoring_voxels_ap_dv_ml": [round(v, 4) for v in reg.anchoring],
            "warp_file": reg.bspline_transform_path,
            "residual": reg.residual,
        }
    try:
        landmarks = _landmarks_in_atlas(project, section, project_dir)
    except Exception as exc:
        logger.warning("section {}: landmarks not written to section.json ({})", section.index, exc)
        landmarks = None
    if landmarks is not None:
        out["landmarks"] = landmarks
    return out


def write_section_folders(
    project: Project,
    project_path: str | Path,
    *,
    slide_images: Mapping[int, np.ndarray] | None = None,
) -> int:
    """Cut missing or outdated section images and write every ``section.json``.

    ``slide_images`` are the slides' finished pixels (flips and rotation applied),
    as the app holds them; any slide not given is rebuilt from disk, and only if
    one of its sections needs a new image. Returns how many images were written.
    Never raises: these files are a by-product of saving, not part of it.
    """
    project_path = Path(project_path)
    project_dir = project_path.parent
    written = 0
    for slide_idx, slide in enumerate(project.slides):
        image = None if slide_images is None else slide_images.get(slide_idx)
        unavailable = False  # the slide could not be rebuilt; don't retry per section
        for section in slide.sections:
            rel_folder = folder_for(project_path, section)
            folder = project_dir / rel_folder
            summary_path = folder / "section.json"
            key = crop_key(slide, section)
            in_place = section.image_path == f"{rel_folder}/image.tif"
            has_image = in_place and (project_dir / f"{rel_folder}/image.tif").is_file()
            # A slide with no image of its own is made of these images: they are
            # the source, never cut again (only copied, into a new project's folder).
            sections_only = not slide.image_path and not slide.source_paths
            current = has_image and (
                sections_only or _read_json(summary_path).get("crop") == key
            )
            if not current and image is None and not unavailable:
                from atlastrack.project.images import rebuild_slide_image

                try:
                    image, _ = rebuild_slide_image(slide, base_dir=project_dir)
                except Exception as exc:
                    logger.warning("slide {}: section images not cut ({})", slide_idx, exc)
                    unavailable = True
            try:
                if not current and image is not None:
                    x0, y0, x1, y1 = section.bbox_px
                    write_section_image(
                        project_path, section, np.asarray(image)[y0:y1, x0:x1], rel_folder
                    )
                    written += 1
                folder.mkdir(parents=True, exist_ok=True)
                summary = section_summary(project, slide, section, project_dir, project_path.name)
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except Exception as exc:
                logger.warning("section {}: folder not written ({})", section.index, exc)
    return written
