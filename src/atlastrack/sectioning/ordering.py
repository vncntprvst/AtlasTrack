"""Order detected sections into a linear AP sequence.

A composite slide typically has 3–4 rows of sections, ordered top-to-bottom
then left-to-right within each row. The user can override the linear order at
the GUI layer; this module just produces a reasonable default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atlastrack.sectioning.split import DetectedSection


@dataclass(frozen=True)
class OrderedSection:
    """A section with its assigned row/column and linear AP order."""

    section: DetectedSection
    row: int
    col: int
    ap_order: int


def _cluster_1d(values: np.ndarray, min_gap: float) -> np.ndarray:
    """Cluster 1D centroid values: a new group starts when a sorted gap > ``min_gap``."""
    order = np.argsort(values)
    sorted_vals = values[order]
    diffs = np.diff(sorted_vals)
    if len(diffs) == 0:
        return np.zeros(len(values), dtype=int)
    group_of_sorted = np.zeros(len(values), dtype=int)
    current = 0
    for i, d in enumerate(diffs, start=1):
        if d > min_gap:
            current += 1
        group_of_sorted[i] = current
    group_of_original = np.empty_like(group_of_sorted)
    group_of_original[order] = group_of_sorted
    return group_of_original


def _grid_positions(
    cxs: np.ndarray,
    cys: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    *,
    column_first: bool,
    left_to_right: bool,
    top_to_bottom: bool,
    gap_factor: float,
) -> list[tuple[int, int, int]]:
    """Assign each section a ``(orig_idx, row, col)`` grid position.

    **Which axis is clustered follows ``column_first``, and that is the whole
    point.** Finding rows as gaps in a global sort of y assumes every column's
    sections line up horizontally. On a real slide they frequently do not - the
    sections are laid down column by column and drift vertically, so no global y
    gap exists. Every section then lands in a single row, and column-first
    numbering collapses to "sort by x" - which, *within* one column, is noise: on
    a real 5-column slide the left column's centroids spanned only 2213..2335 px,
    so its six sections came out numbered 0, 2, 3, 1, 5, 4 from top to bottom.

    Columns are separated by roughly the width of a section, which is exactly the
    gap this clustering can see. So column-first clusters **x** into columns and
    ranks by y inside each; row-first clusters **y** into rows and ranks by x
    inside each. On a cleanly aligned grid the two agree, which is why the
    synthetic-grid tests passed throughout.
    """
    if column_first:
        spread, inner_vals = cxs, cys
        extent, inner_forward = widths, top_to_bottom
        outer_forward = left_to_right
    else:
        spread, inner_vals = cys, cxs
        extent, inner_forward = heights, left_to_right
        outer_forward = top_to_bottom

    median_extent = float(np.median(extent)) if len(extent) else 1.0
    groups = _cluster_1d(spread, min_gap=max(median_extent * gap_factor, 1.0))

    means = {g: float(spread[groups == g].mean()) for g in np.unique(groups)}
    group_order = sorted(means, key=lambda g: means[g], reverse=not outer_forward)

    entries: list[tuple[int, int, int]] = []
    for outer, g in enumerate(group_order):
        idxs = np.where(groups == g)[0]
        inner_sort = np.argsort(inner_vals[idxs])
        if not inner_forward:
            inner_sort = inner_sort[::-1]
        for inner, idx in enumerate(idxs[inner_sort]):
            row, col = (inner, outer) if column_first else (outer, inner)
            entries.append((int(idx), row, col))
    return entries


def _band_of(cy: float, bands: list[tuple[int, int]]) -> int:
    """Index of the band whose ``(y_start, y_end)`` contains ``cy`` (nearest if gap)."""
    for i, (y0, y1) in enumerate(bands):
        if y0 <= cy < y1:
            return i
    centers = [0.5 * (y0 + y1) for y0, y1 in bands]
    return int(np.argmin([abs(cy - c) for c in centers]))


def order_sections(
    sections: list[DetectedSection],
    *,
    column_first: bool = True,
    left_to_right: bool = True,
    top_to_bottom: bool = True,
    row_gap_factor: float = 0.6,
    band_bounds: list[tuple[int, int]] | None = None,
) -> list[OrderedSection]:
    """Order sections into a linear AP sequence and tag each with row/col.

    Sections are clustered along whichever axis the walk order depends on, using
    ``row_gap_factor`` × the median section extent on that axis as the gap that
    separates one group from the next (see :func:`_grid_positions`). The ``row``
    and ``col`` tags follow that grid, and ``ap_order`` walks it either:

    * **column-first** (default) - down column 0 (top→bottom), then column 1,
      etc. This matches how sections are usually laid out on the lab's slides.
    * **row-first** - across row 0 (left→right), then row 1, etc. (reading
      order).

    When ``band_bounds`` is given (the per-source vertical bands from
    :func:`atlastrack.io.image.slide_bands` for a merged multi-slide canvas),
    ordering is **slide-aware**: each source's sections are gridded and numbered
    independently, then concatenated top band first - so a column never runs
    across two stacked slides. Row/col tags are local to each band.
    """
    if not sections:
        return []

    # Slide-aware: order each source's band independently, then renumber the
    # ap_order sequentially across bands (top band first).
    if band_bounds and len(band_bounds) > 1:
        groups: dict[int, list[DetectedSection]] = {}
        for s in sections:
            b = _band_of(float(s.centroid_px[1]), band_bounds)
            groups.setdefault(b, []).append(s)
        out: list[OrderedSection] = []
        offset = 0
        for b in sorted(groups):
            sub = order_sections(
                groups[b], column_first=column_first, left_to_right=left_to_right,
                top_to_bottom=top_to_bottom, row_gap_factor=row_gap_factor,
            )
            out.extend(
                OrderedSection(section=o.section, row=o.row, col=o.col,
                               ap_order=o.ap_order + offset)
                for o in sub
            )
            offset += len(sub)
        return out

    cys = np.array([s.centroid_px[1] for s in sections])
    cxs = np.array([s.centroid_px[0] for s in sections])
    heights = np.array([s.bbox_px[3] - s.bbox_px[1] for s in sections], dtype=float)
    widths = np.array([s.bbox_px[2] - s.bbox_px[0] for s in sections], dtype=float)

    entries = _grid_positions(
        cxs, cys, widths, heights, column_first=column_first,
        left_to_right=left_to_right, top_to_bottom=top_to_bottom,
        gap_factor=row_gap_factor,
    )

    # Number ap_order by walking the grid in the requested order.
    key = (lambda e: (e[2], e[1])) if column_first else (lambda e: (e[1], e[2]))
    ap_rank = {e[0]: rank for rank, e in enumerate(sorted(entries, key=key))}

    return [
        OrderedSection(section=sections[idx], row=row, col=col, ap_order=ap_rank[idx])
        for (idx, row, col) in entries
    ]


def geometric_order(
    bboxes: list[tuple[int, int, int, int]],
    *,
    column_first: bool = True,
    left_to_right: bool = True,
    top_to_bottom: bool = True,
    row_gap_factor: float = 0.6,
) -> list[int]:
    """Return the AP-order rank for each bbox, in the order they were given.

    Operates on bounding boxes ``(x0, y0, x1, y1)`` alone (centroids derived
    from them), so the GUI can re-sort the project's sections after the user has
    added/removed boxes without needing the original masks. ``rank[i]`` is the
    AP position of ``bboxes[i]``; smaller = earlier in the sequence.
    """
    if not bboxes:
        return []
    cxs = np.array([(b[0] + b[2]) / 2.0 for b in bboxes])
    cys = np.array([(b[1] + b[3]) / 2.0 for b in bboxes])
    heights = np.array([b[3] - b[1] for b in bboxes], dtype=float)
    widths = np.array([b[2] - b[0] for b in bboxes], dtype=float)

    entries = _grid_positions(
        cxs, cys, widths, heights, column_first=column_first,
        left_to_right=left_to_right, top_to_bottom=top_to_bottom,
        gap_factor=row_gap_factor,
    )

    key = (lambda e: (e[2], e[1])) if column_first else (lambda e: (e[1], e[2]))
    ordered = sorted(entries, key=key)
    rank = [0] * len(bboxes)
    for r, (idx, _row, _col) in enumerate(ordered):
        rank[idx] = r
    return rank


def apply_missing_flags(
    ordered: list[OrderedSection],
    missing_after_indices: list[int],
) -> list[int | None]:
    """Insert ``None`` placeholders into the AP sequence after given indices.

    Used when the user reports that some sections are missing from the slide
    (lost during sectioning). Returns a list whose entries are either the
    ``ap_order`` of an OrderedSection or ``None`` for a gap.
    """
    sequence: list[int | None] = []
    missing_set = set(missing_after_indices)
    for s in ordered:
        sequence.append(s.ap_order)
        if s.ap_order in missing_set:
            sequence.append(None)
    return sequence
