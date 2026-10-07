"""Recognise a registration made by another tool, and import it.

Two kinds are read:

* **slicereg** (github.com/mvdokh/cell-counting): a folder with ``project.json``
  and ``slices/slice_NN/alignment.json`` - see :mod:`atlastrack.io.slicereg`.
* **QUINT** (QuickNII, DeepSlice, VisuAlign): one JSON file listing the sections
  with their ``anchoring`` (``.waln``: ``ouv``) - see :mod:`atlastrack.io.quint`.
  A folder holding one such file works too.
"""
from __future__ import annotations

import json
from pathlib import Path


def _is_quint_file(path: Path) -> bool:
    if path.suffix.lower() not in (".json", ".waln", ".wwrp") or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    entries = data.get("slices") or data.get("sections") or []
    return any(isinstance(e, dict) and (e.get("anchoring") or e.get("ouv")) for e in entries)


def find_quint_file(path: Path) -> Path | None:
    """The QUINT series file at ``path`` (a file, or a folder holding exactly one)."""
    if path.is_file():
        return path if _is_quint_file(path) else None
    candidates = [p for p in sorted(path.iterdir()) if _is_quint_file(p)]
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"{path} holds several QUINT files ({names}); name the one to import")
    return candidates[0] if candidates else None


def detect(path: str | Path) -> str | None:
    """``"slicereg"``, ``"quint"``, or ``None`` if ``path`` is neither."""
    from atlastrack.io.slicereg import is_slicereg_project

    path = Path(path)
    if path.is_dir() and is_slicereg_project(path):
        return "slicereg"
    if path.exists() and find_quint_file(path) is not None:
        return "quint"
    return None


def import_registration(
    path: str | Path,
    out_dir: str | Path | None = None,
    *,
    pixel_um: float | None = None,
) -> Path:
    """Import ``path`` as whichever kind it is; returns the new project's path.

    ``pixel_um`` applies to slicereg projects only (QUINT images carry no pixel
    size, so they are kept as they are).
    """
    path = Path(path)
    kind = detect(path)
    if kind == "slicereg":
        from atlastrack.io.slicereg import import_slicereg

        return import_slicereg(path, out_dir, pixel_um=pixel_um)
    if kind == "quint":
        from atlastrack.io.quint import import_quint

        series = find_quint_file(path)
        assert series is not None
        return import_quint(series, out_dir)
    raise ValueError(
        f"{path} is neither a slicereg project (project.json + slices/) nor a "
        "QUINT series file (QuickNII / DeepSlice / VisuAlign JSON)"
    )
