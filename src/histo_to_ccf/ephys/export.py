"""Exporting the depth-resolved features an alignment was read from.

Not "LFP power" any more: a shank's panel carries the LFP power map, the spike
firing-rate and amplitude profiles, the atlas regions along the track and the
landmarks placed against them. Saving only one of those would leave a figure that
cannot be regenerated, so the export takes all of it.

**Format: a single compressed ``.npz`` per probe.** It is the lightest thing that
holds ragged per-shank arrays without inventing a schema - a CSV would need one file
per array (or a lot of padding), and the LFP block alone is 384 x 301 floats per
shank. Arrays are keyed ``shank<N>_<name>``, with a ``meta`` JSON string carrying the
scalars, so a reader needs nothing but numpy and ``json``.

Depths are recorded in **both** conventions - µm from the tip and µm below the
surface - because every mix-up in this codebase has come from one being read as the
other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ShankFeatureExport:
    """Everything one shank's alignment was read from."""

    shank_index: int
    track_length_um: float = 0.0
    # LFP power map.
    lfp_freqs_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    lfp_psd: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    channel_depth_from_tip_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    channel_depth_below_surface_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    channel_ids: list = field(default_factory=list)
    # Spike depth profiles.
    profile_depth_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    firing_rate_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_amplitude: np.ndarray = field(default_factory=lambda: np.empty(0))
    # Atlas regions along the track (in track space, not warped).
    region_top_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    region_bottom_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    region_acronym: list = field(default_factory=list)
    # The alignment itself.
    landmark_feature_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    landmark_track_um: np.ndarray = field(default_factory=lambda: np.empty(0))
    extremes_mode: str = "uniform"


def default_export_path(project_path: str | Path | None, probe_label: str) -> Path:
    """Where the export should land: a folder beside the project file.

    Outputs belong next to the data they describe, never the working directory. With
    no project path yet, fall back to the current directory so the caller still gets a
    usable suggestion rather than an exception.
    """
    name = f"{_slug(probe_label)}_depth_features.npz"
    if project_path is None:
        return Path.cwd() / name
    project = Path(project_path)
    return project.parent / f"{project.stem}_ephys_features" / name


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text)]
    return "".join(keep).strip("_") or "probe"


def build_payload(probe_label: str, shanks: list[ShankFeatureExport]) -> dict:
    """Flatten per-shank exports into the ``.npz`` key/array mapping."""
    payload: dict = {}
    meta: dict = {"probe": probe_label, "shanks": []}
    for shank in shanks:
        prefix = f"shank{shank.shank_index}_"
        arrays = {
            "lfp_freqs_hz": shank.lfp_freqs_hz,
            "lfp_psd": shank.lfp_psd,
            "channel_depth_from_tip_um": shank.channel_depth_from_tip_um,
            "channel_depth_below_surface_um": shank.channel_depth_below_surface_um,
            "profile_depth_um": shank.profile_depth_um,
            "firing_rate_hz": shank.firing_rate_hz,
            "mean_amplitude": shank.mean_amplitude,
            "region_top_um": shank.region_top_um,
            "region_bottom_um": shank.region_bottom_um,
            "landmark_feature_um": shank.landmark_feature_um,
            "landmark_track_um": shank.landmark_track_um,
        }
        for name, value in arrays.items():
            arr = np.asarray(value)
            if arr.size:
                payload[prefix + name] = arr
        if shank.region_acronym:
            payload[prefix + "region_acronym"] = np.array(
                [str(a) for a in shank.region_acronym]
            )
        if shank.channel_ids:
            payload[prefix + "channel_ids"] = np.array(
                [str(c) for c in shank.channel_ids]
            )
        meta["shanks"].append({
            "index": shank.shank_index,
            "track_length_um": float(shank.track_length_um),
            "extremes_mode": shank.extremes_mode,
            "n_landmarks": max(int(np.asarray(shank.landmark_feature_um).size) - 2, 0),
            "n_channels": int(np.asarray(shank.channel_depth_from_tip_um).size),
        })
    payload["meta"] = np.array(json.dumps(meta, indent=1))
    return payload


def save_feature_export(path: str | Path, probe_label: str,
                        shanks: list[ShankFeatureExport]) -> Path:
    """Write the compressed ``.npz``, creating the folder if needed. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **build_payload(probe_label, shanks))
    # numpy appends .npz when the name lacks it; report what was actually written.
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def load_feature_export(path: str | Path) -> tuple[dict, dict]:
    """Read one back: ``(arrays, meta)``. Mirrors :func:`save_feature_export`."""
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return arrays, meta
