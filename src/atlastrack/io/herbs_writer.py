"""Write HERBS-compatible pickle files from CCF µm coordinates.

The reverse of :func:`atlastrack.io.herbs.load_herbs_pkl`. The resulting
pickle can be consumed by the legacy ``herbs_probe_mapping.py`` script as well
as any other HERBS-aware tool that reads ``sites_vox`` / ``region_sites`` /
``label_acronym``.
"""
from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from atlastrack.io.herbs import DEFAULT_HERBS_GRID_UM, ccf_um_to_herbs_voxel


def _rle_labels(labels: Sequence[str]) -> tuple[list[str], list[int]]:
    """Run-length-encode a per-site label list into HERBS' (acronym, count) pairs."""
    if not labels:
        return [], []
    acronyms: list[str] = []
    counts: list[int] = []
    current = labels[0]
    count = 0
    for lab in labels:
        if lab == current:
            count += 1
        else:
            acronyms.append(current)
            counts.append(count)
            current = lab
            count = 1
    acronyms.append(current)
    counts.append(count)
    return acronyms, counts


def write_herbs_pkl(
    out_path: str | Path,
    shanks_ccf_um: Sequence[np.ndarray],
    *,
    regions_per_shank: Sequence[Sequence[str]] | None = None,
    grid_um: int = DEFAULT_HERBS_GRID_UM,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a HERBS-compatible pkl from per-shank CCF µm coordinates.

    Parameters
    ----------
    out_path
        Destination .pkl path.
    shanks_ccf_um
        One ``(N_i, 3)`` array per shank, columns (AP, ML, DV) in µm.
    regions_per_shank
        Optional per-site region acronyms (one list per shank, length ``N_i``).
        Defaults to ``""`` per site.
    grid_um
        HERBS voxel grid. Defaults to 10 µm (the HERBS internal default).
    extra
        Optional extra keys to merge into the ``data`` dict (for forward-compat
        with future HERBS fields).
    """
    sites_vox: list[np.ndarray] = []
    all_acronyms: list[str] = []
    all_counts: list[int] = []

    for i, shank_um in enumerate(shanks_ccf_um):
        sv = ccf_um_to_herbs_voxel(np.asarray(shank_um, dtype=float), grid_um=grid_um)
        sites_vox.append(sv)
        n = len(sv)
        regions = (
            list(regions_per_shank[i]) if regions_per_shank is not None else [""] * n
        )
        if len(regions) != n:
            raise ValueError(
                f"shank {i}: regions length {len(regions)} != sites length {n}"
            )
        a, c = _rle_labels(regions)
        all_acronyms.extend(a)
        all_counts.extend(c)

    data: dict[str, Any] = {
        "sites_vox": sites_vox,
        "region_sites": all_counts,
        "label_acronym": all_acronyms,
    }
    if extra:
        data.update(extra)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"data": data}, f)
    return out_path
