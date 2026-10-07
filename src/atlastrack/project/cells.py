"""Counted cells: reading them in, and placing them for display.

Cells come from slicereg's ``slices/slice_NN/cells.csv`` (one row per cell, with
its pixel on that slice's ``image.tif`` and its atlas position). They are kept
as a :class:`~atlastrack.project.schema.CellSet` in the project: atlas positions
for the 3-D view, and section pixels to draw them on the sections. Matching a
project to one that has cells copies them across (see
:mod:`atlastrack.registration.reference_match`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atlastrack.project.schema import CellSet, Project

#: Colours for cell types, in order of first appearance (then repeated). Yellow
#: first: labelled cells are mostly imaged red or green, and yellow stands out on both.
CELL_COLOURS = ("#ffe14d", "#3fd2ff", "#ff8cf5", "#7dff6b", "#ffffff", "#ff4d4d")


def slicereg_cells(
    slicereg_dir: str | Path,
    placement: dict[int, tuple[int, float]],
    *,
    name: str = "cells",
) -> CellSet:
    """The cells of a slicereg project as a cell set.

    ``placement`` maps a slicereg slice id to ``(section index, factor)``: the
    section the slice became, and its pixel scale relative to slicereg's
    ``image.tif`` (new pixels per old pixel). Slices missing from it keep their
    atlas position only.
    """
    from atlastrack.project.schema import CellPoint, CellSet

    slicereg_dir = Path(slicereg_dir)
    meta = json.loads((slicereg_dir / "project.json").read_text())
    cells = []
    for s in sorted(meta.get("slices") or [], key=lambda s: s["id"]):
        path = slicereg_dir / "slices" / f"slice_{s['id']:02d}" / "cells.csv"
        if not path.is_file():
            continue
        index, factor = placement.get(int(s["id"]), (None, None))
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                x = y = None
                if index is not None and row.get("x_px") and row.get("y_px"):
                    x = (float(row["x_px"]) + 0.5) * factor - 0.5
                    y = (float(row["y_px"]) + 0.5) * factor - 0.5
                cells.append(CellPoint(
                    ap_um=float(row["ap_um"]), ml_um=float(row["ml_um"]),
                    dv_um=float(row["dv_um"]),
                    section_index=index, x_px=x, y_px=y,
                    cell_type=row.get("cell_type") or "cell",
                    region=row.get("region_acronym") or None,
                ))
    return CellSet(name=name, source=str(slicereg_dir.resolve()), cells=cells)


def add_slicereg_cells(project: Project, slicereg_dir: str | Path) -> CellSet:
    """Add a slicereg project's cells to ``project``, an import of that project.

    Sections are found by their slicereg slice number (``slide_number``); the
    pixel scale comes from the slide's ``pixel_um`` and slicereg's own.
    """
    meta = json.loads((Path(slicereg_dir) / "project.json").read_text())
    pixel = {int(s["id"]): float(s["pixel_um"]) for s in meta.get("slices") or []}
    placement = {}
    for slide in project.slides:
        for section in slide.sections:
            sid = section.slide_number
            if sid in pixel and slide.pixel_um:
                placement[sid] = (section.index, pixel[sid] / slide.pixel_um)
    cell_set = slicereg_cells(slicereg_dir, placement)
    project.cell_sets = [c for c in project.cell_sets if c.name != cell_set.name]
    project.cell_sets.append(cell_set)
    return cell_set


def cell_types(project: Project) -> list[str]:
    """Every cell type in the project, in order of first appearance."""
    seen: dict[str, None] = {}
    for cs in project.cell_sets:
        for c in cs.cells:
            seen.setdefault(c.cell_type, None)
    return list(seen)


def cell_colour(project: Project, cell_type: str) -> str:
    types = cell_types(project)
    k = types.index(cell_type) if cell_type in types else 0
    return CELL_COLOURS[k % len(CELL_COLOURS)]


def cells_on_slide(project: Project, slide_idx: int) -> tuple[np.ndarray, list[str]]:
    """Cells drawn on a slide: ``(N, 2)`` (row, col) slide pixels, and their types."""
    boxes = {s.index: s.bbox_px for s in project.slides[slide_idx].sections}
    pts, types = [], []
    for cs in project.cell_sets:
        for c in cs.cells:
            box = boxes.get(c.section_index) if c.section_index is not None else None
            if box is None or c.x_px is None or c.y_px is None:
                continue
            pts.append((box[1] + c.y_px, box[0] + c.x_px))
            types.append(c.cell_type)
    return np.asarray(pts, dtype=float).reshape(-1, 2), types


def cells_in_atlas(project: Project) -> dict[str, np.ndarray]:
    """Every cell's atlas position, (N, 3) (AP, ML, DV) µm, by cell type."""
    out: dict[str, list] = {}
    for cs in project.cell_sets:
        for c in cs.cells:
            out.setdefault(c.cell_type, []).append((c.ap_um, c.ml_um, c.dv_um))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}
