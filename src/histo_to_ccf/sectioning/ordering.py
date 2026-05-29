"""Order detected sections into a linear AP sequence.

A composite slide typically has 3–4 rows of sections, ordered top-to-bottom
then left-to-right within each row. The user can override the linear order at
the GUI layer; this module just produces a reasonable default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histo_to_ccf.sectioning.split import DetectedSection


@dataclass(frozen=True)
class OrderedSection:
    """A section with its assigned row/column and linear AP order."""

    section: DetectedSection
    row: int
    col: int
    ap_order: int


def _cluster_rows(ys: np.ndarray, min_gap: float) -> np.ndarray:
    """Cluster centroid y-values into rows: new row when diff > ``min_gap``."""
    order = np.argsort(ys)
    ys_sorted = ys[order]
    diffs = np.diff(ys_sorted)
    if len(diffs) == 0:
        return np.zeros(len(ys), dtype=int)
    row_of_sorted = np.zeros(len(ys), dtype=int)
    current_row = 0
    for i, d in enumerate(diffs, start=1):
        if d > min_gap:
            current_row += 1
        row_of_sorted[i] = current_row
    row_of_original = np.empty_like(row_of_sorted)
    row_of_original[order] = row_of_sorted
    return row_of_original


def order_sections(
    sections: list[DetectedSection],
    *,
    column_first: bool = True,
    left_to_right: bool = True,
    top_to_bottom: bool = True,
    row_gap_factor: float = 0.6,
) -> list[OrderedSection]:
    """Order sections into a linear AP sequence and tag each with row/col.

    Sections are clustered into rows (centroids differing in y by more than
    ``row_gap_factor`` × median section height start a new row). The ``row`` and
    ``col`` tags follow that grid. The ``ap_order`` numbering then walks the grid
    either:

    * **column-first** (default) — down column 0 (top→bottom), then column 1,
      etc. This matches how sections are usually laid out on the lab's slides.
    * **row-first** — across row 0 (left→right), then row 1, etc. (reading
      order).
    """
    if not sections:
        return []

    cys = np.array([s.centroid_px[1] for s in sections])
    cxs = np.array([s.centroid_px[0] for s in sections])
    heights = np.array([s.bbox_px[3] - s.bbox_px[1] for s in sections], dtype=float)
    median_h = float(np.median(heights)) if len(heights) else 1.0
    min_gap = max(median_h * row_gap_factor, 1.0)
    rows = _cluster_rows(cys, min_gap=min_gap)

    # Sort rows top-to-bottom (or bottom-to-top) by mean y.
    row_means = {r: float(cys[rows == r].mean()) for r in np.unique(rows)}
    row_order = sorted(row_means, key=lambda r: row_means[r], reverse=not top_to_bottom)
    new_row_idx = {old: new for new, old in enumerate(row_order)}

    # First pass: assign each section its (row, col) grid position.
    entries: list[tuple[int, int, int]] = []  # (orig_idx, row, col)
    for old_row in row_order:
        indices = np.where(rows == old_row)[0]
        col_sort = np.argsort(cxs[indices])
        if not left_to_right:
            col_sort = col_sort[::-1]
        for col, idx in enumerate(indices[col_sort]):
            entries.append((int(idx), new_row_idx[old_row], col))

    # Second pass: number ap_order by walking the grid in the requested order.
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
    median_h = float(np.median(heights)) if len(heights) else 1.0
    rows = _cluster_rows(cys, min_gap=max(median_h * row_gap_factor, 1.0))

    row_means = {r: float(cys[rows == r].mean()) for r in np.unique(rows)}
    row_order = sorted(row_means, key=lambda r: row_means[r], reverse=not top_to_bottom)
    new_row_idx = {old: new for new, old in enumerate(row_order)}

    entries: list[tuple[int, int, int]] = []
    for old_row in row_order:
        indices = np.where(rows == old_row)[0]
        col_sort = np.argsort(cxs[indices])
        if not left_to_right:
            col_sort = col_sort[::-1]
        for col, idx in enumerate(indices[col_sort]):
            entries.append((int(idx), new_row_idx[old_row], col))

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
