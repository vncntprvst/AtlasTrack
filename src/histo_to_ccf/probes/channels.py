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

def aligned_site_depths_from_tip(
    shank: Shank,
    site_depths_from_tip_um: np.ndarray,
    track_length_um: float,
) -> tuple[np.ndarray, bool]:
    """Warp geometric site depths through the shank's ephys landmark alignment.

    Returns ``(depths_from_tip, used_alignment)``. Without a usable alignment the
    depths come back untouched, so callers get the geometric placement and can say so.

    The landmark arrays live on the **depth-below-surface** axis (see
    :mod:`histo_to_ccf.ephys.landmarks`) while probe geometry is µm **from the tip**,
    so the conversion happens here and only here:

    * feature depth below surface = ``insertion_depth - depth_from_tip``, where the
      insertion depth is the manipulator's, falling back to the histology track
      length when no recording pinned it;
    * the landmarks map that onto a track depth below surface;
    * back to µm from the tip against the **histology** track, which is the line the
      channels are actually placed on.

    Getting either subtraction backwards flips the shank end for end, which is why it
    is written out rather than folded into the caller.
    """
    depths = np.asarray(site_depths_from_tip_um, dtype=float)
    eph = getattr(shank, "ephys", None)
    if eph is None:
        return depths, False
    feature = list(eph.feature_um or [])
    track = list(eph.track_um or [])
    if len(feature) < 2 or len(feature) != len(track) or len(feature) == 2:
        # Two entries are the bare track end points: an alignment with no user
        # landmarks, which is the identity. Nothing to apply.
        return depths, False

    from histo_to_ccf.ephys.landmarks import Landmarks

    reference = eph.insertion_depth_um
    if reference is None or reference <= 0:
        reference = track_length_um
    landmarks = Landmarks(np.asarray(feature, dtype=float), np.asarray(track, dtype=float))
    feature_below = reference - depths
    track_below = np.asarray(
        landmarks.to_track(feature_below, getattr(eph, "extremes_mode", "uniform") or "uniform")
    )
    return track_length_um - track_below, True


def shank_channel_coords(
    shank: Shank,
    layout: ProbeLayout,
    *,
    use_ephys: bool = True,
) -> np.ndarray | None:
    """Return per-channel CCF coords for ``shank`` using ``layout``.

    Applies the shank's ephys landmark alignment when it has one, so an alignment the
    user placed actually reaches the exports. Pass ``use_ephys=False`` for the raw
    geometric placement. Returns ``None`` if the shank has no registered tip/entry.
    """
    coords, _used = shank_channel_coords_with_source(shank, layout, use_ephys=use_ephys)
    return coords


def shank_channel_coords_with_source(
    shank: Shank,
    layout: ProbeLayout,
    *,
    use_ephys: bool = True,
) -> tuple[np.ndarray | None, bool]:
    """As :func:`shank_channel_coords`, but also say whether the alignment was used."""
    if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
        return None, False
    depths = layout.site_depths_from_tip_um()
    laterals = layout.site_lateral_offsets_um()
    waypoints = list(getattr(shank, "track_points_ccf_um", None) or [])
    used = False
    if use_ephys:
        # Alignment depths are referenced to the *track*, which is the path length
        # when the shank curves - a curved shank is longer than tip-to-entry.
        if waypoints:
            from histo_to_ccf.probes.track_path import path_length_um, track_polyline

            track_length_um = path_length_um(
                track_polyline(shank.tip_ccf_um, shank.entry_ccf_um, waypoints)
            )
        else:
            track_length_um = float(
                np.linalg.norm(np.asarray(shank.tip_ccf_um) - np.asarray(shank.entry_ccf_um))
            )
        depths, used = aligned_site_depths_from_tip(shank, depths, track_length_um)

    if waypoints:
        return curved_channel_ccf_coords(
            shank.tip_ccf_um, shank.entry_ccf_um, waypoints, depths,
            site_lateral_offsets_um=laterals,
        ), used
    return (
        channel_ccf_coords(
            shank.entry_ccf_um,
            shank.tip_ccf_um,
            depths,
            site_lateral_offsets_um=laterals,
        ),
        used,
    )


def curved_channel_ccf_coords(
    tip_ccf,
    entry_ccf,
    waypoints,
    site_depths_from_tip_um: np.ndarray,
    *,
    site_lateral_offsets_um: np.ndarray | None = None,
) -> np.ndarray:
    """Place sites along a **curved** shank track, by arc length from the tip.

    ``site_depths_from_tip_um`` is distance along the shank, so it is arc length along
    the path - not distance along the straight tip→entry chord, which a curved shank
    exceeds. Lateral offsets use the **local** tangent, so the electrode columns stay
    perpendicular to the shank as it bends.

    With no waypoints this reduces exactly to :func:`channel_ccf_coords`; a test pins
    that, because a curvature feature that quietly perturbs every existing straight
    track would be worse than no feature.
    """
    from histo_to_ccf.probes.track_path import (
        points_at_distance,
        tangents_at_distance,
        track_polyline,
    )

    path = track_polyline(tip_ccf, entry_ccf, waypoints)
    depths = np.asarray(site_depths_from_tip_um, dtype=float)
    coords = points_at_distance(path, depths)
    if site_lateral_offsets_um is None:
        return coords

    lat = np.asarray(site_lateral_offsets_um, dtype=float)
    # The path tangent runs tip->entry; channel_ccf_coords builds its width vector
    # from the entry->tip axis. Negating here is what makes the two agree instead of
    # mirroring every electrode column, which is a 32 µm error that looks like noise.
    tangents = -tangents_at_distance(path, depths)
    offsets = lat - ELECTRODE_COLUMN_CENTER_UM
    for i, axis_hat in enumerate(tangents):
        # Same reference choice as the straight-line path, so the two agree exactly
        # when the track happens to be straight.
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(axis_hat @ ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        width_hat = np.cross(axis_hat, ref)
        norm = np.linalg.norm(width_hat)
        if norm > 1e-9:
            coords[i] += offsets[i] * (width_hat / norm)
    return coords


def project_channel_coords(
    project: Project,
) -> dict[tuple[str, int], np.ndarray]:
    """Compute per-channel CCF coords for every registered shank in ``project``.

    Returns a dict mapping ``(probe_label, shank_index)`` →
    ``np.ndarray`` of shape ``(n_channels, 3)``.
    """
    return {k: v for k, (v, _used) in project_channel_coords_with_source(project).items()}


def project_channel_coords_with_source(
    project: Project,
    *,
    use_ephys: bool = True,
) -> dict[tuple[str, int], tuple[np.ndarray, bool]]:
    """As :func:`project_channel_coords`, plus whether each shank used its alignment.

    The flag is carried through to the CSV rather than dropped: a reader must be able
    to tell an ephys-corrected depth from a purely geometric one, because they are not
    the same claim.
    """
    out: dict[tuple[str, int], tuple[np.ndarray, bool]] = {}
    for probe in project.probes:
        layout = get_layout(probe.type.name)
        for shank in probe.shanks:
            coords, used = shank_channel_coords_with_source(shank, layout, use_ephys=use_ephys)
            if coords is not None:
                out[(probe.label, shank.index)] = (coords, used)
    return out


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_channel_csv(
    project: Project,
    output_path: str | Path,
    *,
    probe_label: str | None = None,
    atlas=None,
) -> int:
    """Export per-channel CCF coordinates to a CSV file.

    Columns: ``probe, shank, channel, ap_um, ml_um, dv_um, depth_source`` - plus
    ``region`` when an ``atlas`` is given.

    ``depth_source`` says whether each shank's depths came from the ephys landmark
    alignment or from probe geometry alone. Two shanks of one probe can legitimately
    differ, and a reader who cannot tell them apart will over-trust the geometric ones.

    Parameters
    ----------
    project
        The registered project.
    output_path
        Destination CSV path.
    probe_label
        If given, only export this probe; otherwise export all probes.
    atlas
        Optional BrainGlobe atlas; when given, each channel gets its region acronym.

    Returns
    -------
    n_rows
        Number of data rows written.
    """
    coords_map = project_channel_coords_with_source(project)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["probe", "shank", "channel", "ap_um", "ml_um", "dv_um", "depth_source"]
    if atlas is not None:
        header.append("region")

    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for (label, shank_idx), (coords, used) in sorted(coords_map.items()):
            if probe_label is not None and label != probe_label:
                continue
            source = "ephys_alignment" if used else "geometry"
            regions = _region_acronyms(atlas, coords) if atlas is not None else None
            for ch_idx, (ap, ml, dv) in enumerate(coords):
                row = [label, shank_idx, ch_idx, f"{ap:.2f}", f"{ml:.2f}", f"{dv:.2f}", source]
                if regions is not None:
                    row.append(regions[ch_idx])
                writer.writerow(row)
                n_rows += 1

    return n_rows


def _region_acronyms(atlas, coords: np.ndarray) -> list[str]:
    """Atlas acronym at each ``(AP, ML, DV)`` µm point, ``""`` outside the atlas."""
    from histo_to_ccf.ephys.regions import regions_at_ccf

    return [acr for acr, _rgb in regions_at_ccf(atlas, coords)]


def export_ibl_channel_locations(
    project: Project,
    output_dir: str | Path,
    *,
    probe_label: str | None = None,
    atlas=None,
) -> list[Path]:
    """Write IBL-dialect ``channel_locations.json`` + ``prev_alignments.json``.

    One folder per shank (``<probe>_shank<N>/``), matching the layout the IBL / AIND
    alignment GUI reads. Field names follow their ``create_channel_dict``: ``x``,
    ``y``, ``z``, ``axial``, ``lateral``, ``brain_region_id``, ``brain_region``, keyed
    ``channel_0``, ``channel_1``, ...; ``prev_alignments.json`` is
    ``{iso_timestamp: [feature, track]}``.

    **The coordinates are Allen CCF µm, not IBL's bregma-referenced frame.** Only the
    *axis naming* is theirs (x=ML, y=AP, z=DV); the origin is the CCF corner. An
    ``origin`` entry records that in the file, because silently shipping CCF numbers
    in a field another tool will read as bregma-referenced would be a wrong answer
    that looks right. Convert deliberately downstream if the consumer needs bregma.

    Returns the paths written.
    """
    import json
    from datetime import datetime

    out_root = Path(output_dir)
    written: list[Path] = []
    for probe in project.probes:
        if probe_label is not None and probe.label != probe_label:
            continue
        layout = get_layout(probe.type.name)
        axial = np.asarray(layout.site_depths_from_tip_um(), dtype=float)
        lateral_raw = layout.site_lateral_offsets_um()
        lateral = (
            np.zeros_like(axial) if lateral_raw is None
            else np.asarray(lateral_raw, dtype=float)
        )
        for shank in probe.shanks:
            coords, used = shank_channel_coords_with_source(shank, layout)
            if coords is None:
                continue
            regions = _region_acronyms(atlas, coords) if atlas is not None else None
            folder = out_root / f"{probe.label}_shank{shank.index}"
            folder.mkdir(parents=True, exist_ok=True)

            payload: dict = {
                "origin": {
                    "frame": "allen_ccf_um",
                    "axes": "x=ML, y=AP, z=DV (Allen CCF corner origin, NOT bregma)",
                    "depth_source": "ephys_alignment" if used else "geometry",
                }
            }
            for i, (ap, ml, dv) in enumerate(coords):
                entry = {
                    "x": float(ml),
                    "y": float(ap),
                    "z": float(dv),
                    "axial": float(axial[i]) if i < axial.size else 0.0,
                    "lateral": float(lateral[i]) if i < lateral.size else 0.0,
                }
                if regions is not None:
                    acr = regions[i]
                    entry["brain_region"] = acr
                    entry["brain_region_id"] = _structure_id(atlas, acr)
                payload[f"channel_{i}"] = entry

            path = folder / "channel_locations.json"
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            written.append(path)

            eph = shank.ephys
            if eph is not None and len(eph.feature_um or []) >= 2:
                stamp = eph.created_at or datetime.now().isoformat(timespec="seconds")
                prev = folder / "prev_alignments.json"
                # Merge rather than overwrite: prev_alignments is a history, and
                # dropping earlier attempts would discard the record of what changed.
                existing = {}
                if prev.exists():
                    try:
                        existing = json.loads(prev.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        existing = {}
                existing[stamp] = [list(eph.feature_um), list(eph.track_um)]
                prev.write_text(json.dumps(existing, indent=1), encoding="utf-8")
                written.append(prev)
    return written


def _structure_id(atlas, acronym: str) -> int:
    """Allen structure id for an acronym, 0 when unknown (IBL's "void")."""
    if not acronym:
        return 0
    try:
        return int(atlas.structures[acronym]["id"])
    except Exception:
        return 0


def export_paxinos_csv(
    project: Project,
    output_path: str | Path,
    *,
    probe_label: str | None = None,
    alignment: str | None = None,
) -> int:
    """Export per-channel coordinates in **Paxinos** stereotaxic mm (bregma origin).

    Columns: ``probe, shank, channel, ap_mm, ml_mm, dv_mm`` where AP is
    anterior-positive, ML 0 at the midline, DV depth below bregma. ``alignment``
    selects the CCF→stereotaxic transform (5° pitch + scaling); see
    :data:`histo_to_ccf.io.ccf_coords.PAXINOS_ALIGNMENTS`. Returns the row count.
    """
    from histo_to_ccf.io.ccf_coords import (
        DEFAULT_PAXINOS_ALIGNMENT,
        ccf_um_to_paxinos_mm,
    )

    align = alignment or DEFAULT_PAXINOS_ALIGNMENT
    coords_map = project_channel_coords(project)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["probe", "shank", "channel", "ap_mm", "ml_mm", "dv_mm"])
        for (label, shank_idx), coords in sorted(coords_map.items()):
            if probe_label is not None and label != probe_label:
                continue
            ap_mm, ml_mm, dv_mm = ccf_um_to_paxinos_mm(
                coords[:, 0], coords[:, 1], coords[:, 2], alignment=align
            )
            for ch_idx in range(len(coords)):
                writer.writerow([
                    label, shank_idx, ch_idx,
                    f"{ap_mm[ch_idx]:.3f}", f"{ml_mm[ch_idx]:.3f}", f"{dv_mm[ch_idx]:.3f}",
                ])
                n_rows += 1

    return n_rows
