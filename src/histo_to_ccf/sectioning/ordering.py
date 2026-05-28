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
    left_to_right: bool = True,
    top_to_bottom: bool = True,
    row_gap_factor: float = 0.6,
) -> list[OrderedSection]:
    """Order sections row-by-row then within-row.

    Two centroids belong to different rows when their y differ by more than
    ``row_gap_factor`` × median section bbox height. Defaults match the
    convention used in the example fluorescence slides: rows top→bottom,
    columns left→right, AP order = reading order.
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

    out: list[OrderedSection] = []
    ap = 0
    for new_row in range(len(row_order)):
        old_row = row_order[new_row]
        indices = np.where(rows == old_row)[0]
        xs_in_row = cxs[indices]
        col_sort = np.argsort(xs_in_row)
        if not left_to_right:
            col_sort = col_sort[::-1]
        for col, idx in enumerate(indices[col_sort]):
            out.append(
                OrderedSection(
                    section=sections[idx],
                    row=new_row_idx[old_row],
                    col=col,
                    ap_order=ap,
                )
            )
            ap += 1
    return out


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
