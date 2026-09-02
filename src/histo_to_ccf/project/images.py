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

from histo_to_ccf.io.image import crop, load_image, merge_images, slide_bands

if TYPE_CHECKING:
    from histo_to_ccf.project.schema import Project, Slide


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

    return math.degrees(math.atan2(float(anchoring[4]), float(anchoring[5])))


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


def rebuild_slide_image(
    slide: "Slide",
    *,
    base_dir: Path | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Reload one slide's pixels exactly as the stored bboxes expect them.

    Merges ``source_paths`` (when there is more than one), re-applies the slide's
    own flips, then re-applies each section's flip inside its bbox. Returns the
    image and the per-source row bands (``[(y0, y1), ...]``), which a re-detect
    needs to stay slide-aware.

    ``base_dir`` resolves relative paths; absolute stored paths are used as-is.
    """
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() or base_dir is None else base_dir / path

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
            if grayscale and patch.ndim == 3:
                patch = patch[..., :3].astype(np.float32).mean(axis=-1)
            out[section.index] = patch.astype(np.float32)
    return out
