"""Rebuild the pixel data a project's section bboxes were defined against.

A project stores *paths and flags*, not pixels. Reproducing the exact image a
section's ``bbox_px`` refers to takes three steps that are easy to forget:

1. a multi-source slide must be **merged** in the stored order (``source_paths``),
   because the bboxes live in the combined image's coordinate space;
2. whole-slide flips must be re-applied (only the flags are persisted);
3. per-section flips must be re-applied inside each bbox.

The GUI did all three on load while the CLI did none of them, so a headless
``register`` run silently registered against the wrong pixels - the first source
image only, un-flipped. Both now go through this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from atlastrack.io.image import crop, load_image, merge_images, slide_bands

if TYPE_CHECKING:
    from atlastrack.project.schema import Project, Slide


_CHANNEL_INDEX = {"red": 0, "green": 1, "blue": 2}


def align_channel_image(patch: np.ndarray, align_channel: str | None) -> np.ndarray:
    """The channel registration should compare with the atlas, or ``patch`` as is.

    With no channel named, or a grayscale patch, nothing changes and the caller's
    usual handling (luminance, with red/green labels masked out) applies.
    """
    if not align_channel or patch.ndim != 3 or patch.shape[-1] < 3:
        return patch
    return patch[..., _CHANNEL_INDEX[align_channel]]


def deepslice_rotation_deg(anchoring: list[float] | tuple[float, ...]) -> float:
    """In-plane rotation of a section, degrees, from its stored anchoring.

    A stored anchoring is ``[o_ap, o_dv, o_ml, u_ap, u_dv, u_ml, v_ap, v_dv, v_ml]``
    in atlas voxels: ``u`` runs along the image width. For a section lying square on
    the slide ``u`` is purely ML, so any DV component is the angle it was mounted at
    - the thing that makes a flick-through series wobble.

    This is a *suggestion*, never applied on its own: rotation changes the image
    registration runs on, so applying a prediction automatically would invalidate
    every fit the moment a pre-match ran.
    """
    import math

    # |u_ml|: a plane for the sections' back face runs the other way along ML, and
    # must read as turned by a few degrees, not by about 180.
    return math.degrees(math.atan2(float(anchoring[4]), abs(float(anchoring[5]))))


def rotate_in_bbox(patch: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a section patch about its centre, keeping its exact shape.

    The shape has to survive: ``bbox_px`` is the section's frame everywhere else -
    overlay placement, landmarks, per-channel coordinates - so a rotation that grew
    the canvas would silently desynchronise all of them. A detection box is
    axis-aligned around tilted tissue, so it normally has the slack to absorb the
    few degrees this is used for; a large angle will clip the corners.
    """
    if abs(degrees) < 1e-6:
        return patch
    from scipy.ndimage import rotate as _rotate

    out = _rotate(
        patch, degrees, axes=(1, 0), reshape=False, order=1, mode="constant", cval=0
    )
    return out.astype(patch.dtype, copy=False)


def assemble_from_sections(
    slide: Slide, *, base_dir: Path | None = None
) -> np.ndarray | None:
    """The slide put back together from its sections' own images, or ``None``.

    Each image goes back into its box on a blank canvas. The images already carry
    their flips and rotation, so nothing is re-applied. ``None`` unless every
    section has an image on disk that fits its box.
    """
    if not slide.sections:
        return None
    patches = []
    for section in slide.sections:
        if not section.image_path:
            return None
        path = Path(section.image_path)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if not path.is_file():
            return None
        patch = load_image(path)
        x0, y0, x1, y1 = section.bbox_px
        if patch.shape[:2] != (y1 - y0, x1 - x0):
            return None
        patches.append((section.bbox_px, patch))
    height = max(b[3] for b, _ in patches)
    width = max(b[2] for b, _ in patches)
    first = patches[0][1]
    canvas = np.zeros((height, width, *first.shape[2:]), dtype=first.dtype)
    for (x0, y0, x1, y1), patch in patches:
        canvas[y0:y1, x0:x1] = patch
    return canvas


def rebuild_slide_image(
    slide: "Slide",
    *,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Reload one slide's pixels exactly as the stored bboxes expect them.

    Merges ``source_paths`` (when there is more than one), re-applies the slide's
    own flips, then re-applies each section's flip inside its bbox. Returns the
    image and the per-source row bands (``[(y0, y1), ...]``), which a re-detect
    needs to stay slide-aware. When the slide image is missing (or the slide never
    had one), the slide is put together from its sections' own images instead.

    ``base_dir`` resolves relative paths; absolute stored paths are used as-is.
    """
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() or base_dir is None else base_dir / path

    sources = list(slide.source_paths) or ([slide.image_path] if slide.image_path else [])
    if not sources or not all(_resolve(s).is_file() for s in sources):
        assembled = assemble_from_sections(slide, base_dir=base_dir)
        if assembled is not None:
            return assembled, [(0, int(assembled.shape[0]))]
        if not sources:
            raise FileNotFoundError("slide has no image and its sections have none either")

    if slide.source_paths and len(slide.source_paths) > 1:
        sources = [load_image(_resolve(s)) for s in slide.source_paths]
        img = merge_images(sources)
        bands = slide_bands([s.shape[0] for s in sources])
    else:
        img = load_image(_resolve(slide.image_path))
        bands = [(0, int(img.shape[0]))]

    if slide.flip_h:
        img = np.fliplr(img)
    if slide.flip_v:
        img = np.flipud(img)
    # np.fliplr/flipud return views; the per-section writes below need a real array.
    img = np.ascontiguousarray(img)

    for section in slide.sections:
        if not (section.flip_h or section.flip_v):
            continue
        x0, y0, x1, y1 = section.bbox_px
        patch = img[y0:y1, x0:x1]
        if section.flip_h:
            patch = np.fliplr(patch)
        if section.flip_v:
            patch = np.flipud(patch)
        img[y0:y1, x0:x1] = patch

    # Rotation is baked in for the same reason flips are: it changes the pixels the
    # registration is computed against, so it has to be part of the working image
    # rather than something applied later at export time. Rotating a section that is
    # already registered invalidates that section's fit - the GUI warns about it.
    for section in slide.sections:
        angle = float(getattr(section, "rotation_deg", 0.0) or 0.0)
        if abs(angle) < 1e-6:
            continue
        x0, y0, x1, y1 = section.bbox_px
        img[y0:y1, x0:x1] = rotate_in_bbox(img[y0:y1, x0:x1], angle)

    return img, bands


def section_images(
    project: "Project",
    *,
    base_dir: Path | None = None,
    grayscale: bool = True,
) -> dict[int, np.ndarray]:
    """Map ``section.index`` -> the cropped image the registration should use.

    Slides that fail to load are skipped with their sections omitted, so a
    partially-available project still registers what it can.
    """
    out: dict[int, np.ndarray] = {}
    for slide in project.slides:
        try:
            img, _ = rebuild_slide_image(slide, base_dir=base_dir)
        except Exception:  # noqa: BLE001 - a missing/corrupt source shouldn't abort the run
            continue
        for section in slide.sections:
            patch = crop(img, section.bbox_px)
            if grayscale:
                patch = align_channel_image(patch, slide.align_channel)
            if grayscale and patch.ndim == 3:
                patch = patch[..., :3].astype(np.float32).mean(axis=-1)
            out[section.index] = patch.astype(np.float32)
    return out
