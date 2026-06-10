"""Per-channel CCF coordinate computation.

Given a probe's tip and entry positions in CCF µm and a :class:`ProbeLayout`,
this module computes the 3D CCF position of every recording channel.

Coordinate convention throughout: (AP, ML, DV) µm, matching the project
schema.  DV increases ventrally (deeper = larger DV value).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from histo_to_ccf.probes.catalog import ProbeLayout, get_layout
from histo_to_ccf.probes.geometry import ELECTRODE_COLUMN_CENTER_UM

if TYPE_CHECKING:
    from histo_to_ccf.project.schema import Project, Shank


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def channel_ccf_coords(
    entry_ccf: np.ndarray | tuple[float, float, float],
    tip_ccf: np.ndarray | tuple[float, float, float],
    site_depths_from_tip_um: np.ndarray,
    *,
    site_lateral_offsets_um: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the CCF position of every recording site.

    Parameters
    ----------
    entry_ccf
        Where the probe enters the brain surface - (AP, ML, DV) µm.
    tip_ccf
        Physical probe tip - (AP, ML, DV) µm.
    site_depths_from_tip_um
        Distance from the tip to each recording site, shape (n_channels,).
        Larger values = further from tip = closer to entry.
    site_lateral_offsets_um
        Optional lateral displacement of each site from the shank centreline,
        shape (n_channels,).  Positive = right when facing probe front face.

    Returns
    -------
    coords
        Shape (n_channels, 3) in (AP, ML, DV) µm.
    """
    entry = np.asarray(entry_ccf, dtype=float)
    tip = np.asarray(tip_ccf, dtype=float)
    trajectory = tip - entry
    length = float(np.linalg.norm(trajectory))

    if length < 1.0:
        return np.tile(entry, (len(site_depths_from_tip_um), 1))

    axis_hat = trajectory / length

    # Each site sits at (length - depth_from_tip) from the entry point.
    depths_from_entry = length - np.asarray(site_depths_from_tip_um, dtype=float)
    coords = entry[np.newaxis, :] + depths_from_entry[:, np.newaxis] * axis_hat[np.newaxis, :]

    if site_lateral_offsets_um is not None:
        lat = np.asarray(site_lateral_offsets_um, dtype=float)
        # Width vector: perpendicular to axis in the ML-DV plane.
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(axis_hat @ ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        width_hat = np.cross(axis_hat, ref)
        w_norm = np.linalg.norm(width_hat)
        if w_norm > 1e-9:
            width_hat = width_hat / w_norm
            coords += (lat - ELECTRODE_COLUMN_CENTER_UM)[:, np.newaxis] * width_hat[np.newaxis, :]

    return coords


# ---------------------------------------------------------------------------
# High-level shank / project helpers
# ---------------------------------------------------------------------------

def shank_channel_coords(
    shank: "Shank",
    layout: ProbeLayout,
) -> np.ndarray | None:
    """Return per-channel CCF coords for ``shank`` using ``layout``.

    Returns ``None`` if the shank has no registered tip/entry coordinates.
    """
    if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
        return None
    depths = layout.site_depths_from_tip_um()
    laterals = layout.site_lateral_offsets_um()
    return channel_ccf_coords(
        shank.entry_ccf_um,
        shank.tip_ccf_um,
        depths,
        site_lateral_offsets_um=laterals,
    )


def project_channel_coords(
    project: "Project",
) -> dict[tuple[str, int], np.ndarray]:
    """Compute per-channel CCF coords for every registered shank in ``project``.

    Returns a dict mapping ``(probe_label, shank_index)`` →
    ``np.ndarray`` of shape ``(n_channels, 3)``.
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for probe in project.probes:
        layout = get_layout(probe.type.name)
        for shank in probe.shanks:
            coords = shank_channel_coords(shank, layout)
            if coords is not None:
                out[(probe.label, shank.index)] = coords
    return out


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_channel_csv(
    project: "Project",
    output_path: str | Path,
    *,
    probe_label: str | None = None,
) -> int:
    """Export per-channel CCF coordinates to a CSV file.

    Columns: ``probe, shank, channel, ap_um, ml_um, dv_um``.

    Parameters
    ----------
    project
        The registered project.
    output_path
        Destination CSV path.
    probe_label
        If given, only export this probe; otherwise export all probes.

    Returns
    -------
    n_rows
        Number of data rows written.
    """
    coords_map = project_channel_coords(project)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["probe", "shank", "channel", "ap_um", "ml_um", "dv_um"])
        for (label, shank_idx), coords in sorted(coords_map.items()):
            if probe_label is not None and label != probe_label:
                continue
            for ch_idx, (ap, ml, dv) in enumerate(coords):
                writer.writerow([label, shank_idx, ch_idx, f"{ap:.2f}", f"{ml:.2f}", f"{dv:.2f}"])
                n_rows += 1

    return n_rows
