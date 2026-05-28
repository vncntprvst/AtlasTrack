"""Split a composite slide image into individual brain sections.

The algorithm assumes tissue is brighter than background (fluorescence) or
darker (brightfield); polarity is auto-detected. Steps:

    1. Grayscale conversion (max-of-channels for fluorescence color)
    2. Light Gaussian blur (denoise)
    3. Otsu threshold; invert if background is brighter than tissue
    4. Morphological opening to remove specks
    5. Connected-component labeling
    6. Filter components by area and aspect ratio (drops slide labels)
    7. Return bounding boxes + per-section binary masks

Manual override happens at the project layer: the GUI exposes the resulting
masks as a napari ``Labels`` layer the user can edit, and the section list is
just whatever the user confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import filters, measure, morphology
from skimage.color import rgb2gray


@dataclass(frozen=True)
class DetectedSection:
    """One auto-detected section: bbox + mask + simple shape stats."""

    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1) in slide coords
    mask: np.ndarray  # bool, full-slide-sized
    area_px: int
    centroid_px: tuple[float, float]  # (cx, cy)
    aspect_ratio: float  # width / height (always >= 1)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert to a float grayscale image in [0, 1]."""
    if image.ndim == 2:
        return image.astype(float) / (255.0 if image.dtype == np.uint8 else 1.0)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        # For fluorescence (predominantly blue/green channels), max-of-channels
        # tends to preserve tissue better than luminance-based RGB→gray.
        rgb = image[..., :3].astype(float)
        if rgb.max() > 1.5:
            rgb /= 255.0
        return np.maximum(rgb2gray(rgb), rgb.max(axis=-1))
    raise ValueError(f"unsupported image shape {image.shape}")


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold and orient so tissue=True."""
    blurred = filters.gaussian(gray, sigma=2.0, preserve_range=True)
    thresh = filters.threshold_otsu(blurred)
    fg = blurred > thresh
    # If "foreground" covers >half the image, we picked the background. Flip.
    if fg.mean() > 0.5:
        fg = ~fg
    return fg


def detect_sections(
    image: np.ndarray,
    *,
    min_area_px: int = 5000,
    max_area_frac: float = 0.4,
    opening_radius_px: int = 3,
    closing_radius_px: int = 0,
    aspect_ratio_max: float = 4.0,
    expected_count: int | None = None,
) -> list[DetectedSection]:
    """Find brain sections in a composite slide image.

    Parameters
    ----------
    image
        2D grayscale or HxWx{3,4} color image.
    min_area_px
        Drop components below this size (specks / dust).
    max_area_frac
        Drop components larger than this fraction of the image (touching
        artifacts / whole-frame bleeds).
    opening_radius_px
        Disk radius for morphological opening (separates touching sections,
        removes specks).
    closing_radius_px
        Disk radius for morphological closing run AFTER opening. Bridges
        anatomical gaps within a section (e.g., cerebellum/brainstem split
        by the 4th ventricle in caudal coronal slices). 0 disables.
    aspect_ratio_max
        Drop very elongated components (slide-edge labels like "MAS-CP").
        Section aspect ratios are typically < 2.
    expected_count
        If given, keep only the top-``expected_count`` components by area
        among those that pass the filters. Useful when the user knows there
        should be 15 sections on the slide.
    """
    gray = _to_gray(image)
    fg = _binarize(gray)

    if opening_radius_px > 0:
        fg = morphology.opening(fg, morphology.disk(opening_radius_px))
    if closing_radius_px > 0:
        fg = morphology.closing(fg, morphology.disk(closing_radius_px))

    labeled = measure.label(fg, connectivity=2)
    h, w = fg.shape
    max_area_px = int(max_area_frac * h * w)

    sections: list[DetectedSection] = []
    for region in measure.regionprops(labeled):
        if region.area < min_area_px or region.area > max_area_px:
            continue
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        if height == 0 or width == 0:
            continue
        ar = max(width / height, height / width)
        if ar > aspect_ratio_max:
            continue
        full_mask = labeled == region.label
        cy, cx = region.centroid
        sections.append(
            DetectedSection(
                bbox_px=(int(minc), int(minr), int(maxc), int(maxr)),
                mask=full_mask,
                area_px=int(region.area),
                centroid_px=(float(cx), float(cy)),
                aspect_ratio=float(ar),
            )
        )

    if expected_count is not None and len(sections) > expected_count:
        sections = sorted(sections, key=lambda s: -s.area_px)[:expected_count]

    return sections


def section_mask_crop(section: DetectedSection) -> np.ndarray:
    """Return the per-section mask cropped to its bbox."""
    x0, y0, x1, y1 = section.bbox_px
    return section.mask[y0:y1, x0:x1]
