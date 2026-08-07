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

from dataclasses import dataclass, replace

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
    equalize_boxes: bool = True,
    box_min_frac: float = 0.85,
    margin_frac: float = 0.06,
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
    equalize_boxes
        Serial sections on one slide are roughly the same size. When True,
        boxes that are notably smaller than the slide's median box (because
        dimmer tissue fell below threshold) are expanded to the median size,
        centred on the box. Prevents recurrent under-fitting of a few sections.
    box_min_frac
        A box is expanded when its width or height is below this fraction of
        the median box width/height.
    margin_frac
        Background kept around the tissue, as a fraction of the box's own size,
        added to every side. A box that ends flush against the tissue leaves the
        registration nothing to work with on that side: the mask and the boundary
        snap then take the *image border* for the tissue edge, which flattens the
        atlas contour there (most visibly along the bottom). Growth is capped at
        half the gap to the nearest other section so the margin never eats into a
        neighbour, and clipped to the image. 0 disables.
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

    if equalize_boxes:
        sections = _equalize_box_sizes(sections, (h, w), min_frac=box_min_frac)

    if margin_frac > 0:
        sections = _add_box_margin(sections, (h, w), margin_frac=margin_frac)

    return sections


# Past this many boxes the pairwise neighbour cap would need an n^2 array; skip it.
_MARGIN_PAIRWISE_MAX = 2000


def _add_box_margin(
    sections: list[DetectedSection],
    image_shape: tuple[int, int],
    *,
    margin_frac: float,
) -> list[DetectedSection]:
    """Grow every box outwards so tissue is not flush against its edge.

    Each box grows by ``margin_frac`` of its own width/height on each side, but
    never by more than half the clear gap to the nearest other box - otherwise a
    dense slide grid would have neighbouring sections bleeding into each other's
    crops. Growth is clipped to the image. Masks are left untouched: the margin is
    background by definition, so the mask stays exactly the detected tissue.
    """
    if not sections:
        return []
    h_img, w_img = image_shape
    boxes = np.array([s.bbox_px for s in sections], dtype=float)
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    want = margin_frac * np.minimum(x1 - x0, y1 - y0)

    # Cap by the gap to any other box, measured only where the two actually face
    # each other (their spans overlap on the other axis). Vectorised, because a
    # Python-level pairwise loop is far too slow once detection returns a lot of
    # components - but the vectorised form is O(n^2) in *memory*, so past a few
    # thousand boxes the cap is skipped rather than allocating gigabytes. A slide
    # with that many "sections" is noise, where neighbour spacing is meaningless.
    if len(sections) > _MARGIN_PAIRWISE_MAX:
        margins = np.maximum(0.0, want).astype(int)
        return _apply_margins(sections, boxes, margins, image_shape)

    big = np.float64(np.inf)
    share_rows = ~((y1[:, None] <= y0[None, :]) | (y0[:, None] >= y1[None, :]))
    share_cols = ~((x1[:, None] <= x0[None, :]) | (x0[:, None] >= x1[None, :]))
    np.fill_diagonal(share_rows, False)
    np.fill_diagonal(share_cols, False)

    right = np.where(share_rows & (x0[None, :] >= x1[:, None]),
                     x0[None, :] - x1[:, None], big)
    left = np.where(share_rows & (x1[None, :] <= x0[:, None]),
                    x0[:, None] - x1[None, :], big)
    below = np.where(share_cols & (y0[None, :] >= y1[:, None]),
                     y0[None, :] - y1[:, None], big)
    above = np.where(share_cols & (y1[None, :] <= y0[:, None]),
                     y0[:, None] - y1[None, :], big)
    nearest = np.minimum.reduce([
        right.min(axis=1), left.min(axis=1), below.min(axis=1), above.min(axis=1)
    ])
    want = np.minimum(want, nearest / 2.0)
    margins = np.maximum(0.0, want).astype(int)
    return _apply_margins(sections, boxes, margins, image_shape)


def _apply_margins(
    sections: list[DetectedSection],
    boxes: np.ndarray,
    margins: np.ndarray,
    image_shape: tuple[int, int],
) -> list[DetectedSection]:
    """Grow each box by its margin, clipped to the image. Masks are untouched."""
    h_img, w_img = image_shape
    return [
        DetectedSection(
            bbox_px=(
                int(max(0, boxes[i, 0] - m)),
                int(max(0, boxes[i, 1] - m)),
                int(min(w_img, boxes[i, 2] + m)),
                int(min(h_img, boxes[i, 3] + m)),
            ),
            mask=sec.mask,
            area_px=sec.area_px,
            centroid_px=sec.centroid_px,
            aspect_ratio=sec.aspect_ratio,
        )
        for i, (sec, m) in enumerate(zip(sections, margins, strict=True))
    ]


def _equalize_box_sizes(
    sections: list[DetectedSection],
    image_shape: tuple[int, int],
    *,
    min_frac: float = 0.85,
) -> list[DetectedSection]:
    """Expand under-sized boxes to the slide's median box size.

    Only boxes whose width/height is below ``min_frac`` × the median are grown
    (to the median, centred on the existing box and clamped to the image). Boxes
    at or above the median are left untouched, so genuinely typical sections are
    not enlarged. Needs at least 3 sections to estimate a stable median.
    """
    if len(sections) < 3:
        return sections
    h_img, w_img = image_shape
    widths = np.array([s.bbox_px[2] - s.bbox_px[0] for s in sections], dtype=float)
    heights = np.array([s.bbox_px[3] - s.bbox_px[1] for s in sections], dtype=float)
    med_w = float(np.median(widths))
    med_h = float(np.median(heights))

    out: list[DetectedSection] = []
    for s in sections:
        x0, y0, x1, y1 = s.bbox_px
        w = x1 - x0
        h = y1 - y0
        target_w = med_w if w < min_frac * med_w else w
        target_h = med_h if h < min_frac * med_h else h
        if target_w == w and target_h == h:
            out.append(s)
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        nx0 = max(0, int(round(cx - target_w / 2.0)))
        nx1 = min(w_img, int(round(cx + target_w / 2.0)))
        ny0 = max(0, int(round(cy - target_h / 2.0)))
        ny1 = min(h_img, int(round(cy + target_h / 2.0)))
        out.append(replace(s, bbox_px=(nx0, ny0, nx1, ny1)))
    return out


def section_mask_crop(section: DetectedSection) -> np.ndarray:
    """Return the per-section mask cropped to its bbox."""
    x0, y0, x1, y1 = section.bbox_px
    return section.mask[y0:y1, x0:x1]


def estimate_min_area(image: np.ndarray) -> int:
    """Estimate a reasonable ``min_area_px`` for this slide.

    Strategy
    --------
    1. Drop components larger than 25 % of the slide (background blob).
    2. Drop the bottom 80 % of components by count - these are tiny noise
       specks that would otherwise dominate the Otsu histogram and push the
       threshold into the wrong gap (noise vs debris instead of debris vs
       sections).
    3. Run log-space Otsu on the surviving "significant" components to split
       debris/text from real sections.
    4. Return 30 % of the *largest* component above the threshold.
       That value sits well below the biggest section (so nothing is excluded)
       while being large enough to reject most debris.
    """
    gray = _to_gray(image)
    fg = _binarize(gray)
    labeled = measure.label(fg, connectivity=2)
    h, w = fg.shape
    image_area = h * w

    # Collect areas AND aspect-ratios in one regionprops pass.
    areas, aspect_ratios = [], []
    for r in measure.regionprops(labeled):
        minr, minc, maxr, maxc = r.bbox
        rh = maxr - minr
        rw = maxc - minc
        if rh == 0 or rw == 0:
            continue
        ar = max(rh / rw, rw / rh)
        areas.append(float(r.area))
        aspect_ratios.append(ar)

    if not areas:
        return 5_000

    areas_arr = np.array(areas, dtype=float)
    ar_arr = np.array(aspect_ratios, dtype=float)

    # Step 1: drop background-sized blobs AND elongated slide labels/text.
    # Brain sections are roughly equant (aspect ratio < 3); text labels,
    # slide barcodes and scale bars are elongated.
    keep = (areas_arr <= 0.25 * image_area) & (ar_arr < 3.5)
    all_areas = areas_arr[keep]
    if len(all_areas) < 2:
        return max(1_000, int(0.001 * image_area))

    # Step 2: keep only the top 20 % by size - removes the huge pool of tiny
    # noise components so Otsu can find the debris/section boundary.
    n_keep = max(2, len(all_areas) // 5)
    significant = np.sort(all_areas)[::-1][:n_keep]

    # Step 3: log-space Otsu on the significant components.
    log_a = np.log10(significant + 1.0)
    try:
        thresh_log = filters.threshold_otsu(log_a)
        large = significant[log_a >= thresh_log]
    except Exception:
        large = significant  # if Otsu fails, treat all significant as large

    if len(large) == 0:
        large = significant

    # Step 4: 30 % of the biggest "section-sized" component.
    estimate = int(np.max(large) * 0.30)

    floor = max(1_000, int(0.0005 * image_area))
    ceiling = int(0.05 * image_area)
    return max(floor, min(estimate, ceiling))


def group_fragmented_sections(
    sections: list[DetectedSection],
    *,
    x_overlap_frac: float = 0.5,
    max_passes: int = 4,
) -> list[DetectedSection]:
    """Re-join pieces of one section that got detected as separate components.

    A coronal brainstem section often arrives on the slide in pieces - cerebellum
    detached from brainstem, or a slab broken along the 4th ventricle. Connected-
    component detection then returns each piece separately, and anything that
    keeps only the largest component silently discards the rest (a probe track in
    the dropped brainstem piece is simply lost).

    The pieces of one section sit **stacked**: they share the section's horizontal
    span. Debris carried over from a neighbouring section sits *beside* it, at the
    left or right edge of the frame. So a piece is absorbed when at least
    ``x_overlap_frac`` of its own width lies within a larger piece's horizontal
    span, and left alone otherwise.

    Returns one :class:`DetectedSection` per group, with the union bounding box
    and mask; input order is not preserved (largest first). Merging is repeated
    up to ``max_passes`` times so a chain of pieces collapses into one group.
    """
    if len(sections) < 2:
        return list(sections)

    def _x_overlap(a: DetectedSection, b: DetectedSection) -> float:
        """Fraction of the narrower box's width that lies inside the other's span."""
        lo = max(a.bbox_px[0], b.bbox_px[0])
        hi = min(a.bbox_px[2], b.bbox_px[2])
        overlap = max(0, hi - lo)
        narrower = min(a.bbox_px[2] - a.bbox_px[0], b.bbox_px[2] - b.bbox_px[0])
        return overlap / narrower if narrower > 0 else 0.0

    def _merge(a: DetectedSection, b: DetectedSection) -> DetectedSection:
        x0 = min(a.bbox_px[0], b.bbox_px[0])
        y0 = min(a.bbox_px[1], b.bbox_px[1])
        x1 = max(a.bbox_px[2], b.bbox_px[2])
        y1 = max(a.bbox_px[3], b.bbox_px[3])
        mask = a.mask | b.mask
        area = int(a.area_px + b.area_px)
        # Area-weighted centroid of the combined piece.
        cx = (a.centroid_px[0] * a.area_px + b.centroid_px[0] * b.area_px) / area
        cy = (a.centroid_px[1] * a.area_px + b.centroid_px[1] * b.area_px) / area
        width, height = x1 - x0, y1 - y0
        ar = max(width / height, height / width) if width and height else 1.0
        return DetectedSection(
            bbox_px=(int(x0), int(y0), int(x1), int(y1)),
            mask=mask,
            area_px=area,
            centroid_px=(float(cx), float(cy)),
            aspect_ratio=float(ar),
        )

    groups = sorted(sections, key=lambda s: -s.area_px)
    for _ in range(max_passes):
        merged_any = False
        out: list[DetectedSection] = []
        for piece in groups:
            for i, group in enumerate(out):
                if _x_overlap(group, piece) >= x_overlap_frac:
                    out[i] = _merge(group, piece)
                    merged_any = True
                    break
            else:
                out.append(piece)
        groups = sorted(out, key=lambda s: -s.area_px)
        if not merged_any:
            break
    return groups
