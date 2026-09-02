"""Saving a fitted trajectory so the preview can be reopened without refitting.

A fit costs ~20 s: detecting boundaries in every shank, a coarse then fine grid over
three parameters, and a leave-one-out refit per shank. None of that changes unless the
features do, so recomputing it to look at the same picture twice is pure waste.

**What is stored is the whole argument, not the conclusion.** The three fitted numbers
alone would be worthless to reopen - the preview exists precisely because three numbers
are the wrong thing to judge - so the scans, the matched boundaries and the
leave-one-out notes travel with them.

A ``fingerprint`` of the evidence is stored alongside. Features get recomputed with
different recordings ticked, and a cached fit that silently outlived the data it was
fitted to would be worse than no cache: it would look like a fresh answer. The loader
reports the mismatch and lets the caller decide.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

from atlastrack.probes.trajectory_fit import (
    Match,
    ParameterScan,
    PlacementScore,
    ShankEvidence,
    TrajectoryFit,
)

#: Bumped when the stored layout changes in a way an older reader cannot handle.
FORMAT_VERSION = 1


def default_fit_path(project_path, probe_label: str) -> Path:
    """Where a probe's fit belongs: beside its depth features.

    Outputs go next to the data they describe, never the working directory.
    """
    from atlastrack.ephys.export import default_export_path

    features = default_export_path(project_path, probe_label)
    return features.with_name(features.stem + "_fit.npz")


def evidence_fingerprint(evidence: dict) -> str:
    """A short digest of the boundaries a fit was computed from.

    Over the depths and weights only: those are what the fit consumed, so a change in
    either means the cached answer no longer describes the current features, while a
    change in anything else does not.
    """
    hasher = sha256()
    for index in sorted(evidence):
        ev = evidence[index]
        hasher.update(str(int(index)).encode())
        hasher.update(np.asarray(ev.depths_from_tip_um, dtype=float).round(3).tobytes())
        hasher.update(np.asarray(ev.weights, dtype=float).round(6).tobytes())
    return hasher.hexdigest()[:16]


def save_fit(path, fit: TrajectoryFit, evidence: dict, *, probe_label: str = "",
             notes: str = "", tips=None, entries=None) -> Path:
    """Write a fit, its evidence and its scans to a compressed ``.npz``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    meta: dict = {
        "format_version": FORMAT_VERSION,
        "probe_label": probe_label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": evidence_fingerprint(evidence),
        "notes": notes,
        "offset_um": float(fit.offset_um),
        "roll_deg": float(fit.roll_deg),
        "tilt_deg": float(fit.tilt_deg),
        "shanks": sorted(int(k) for k in evidence),
        "scans": sorted(fit.scans),
        "score": _score_meta(fit.score),
        "baseline": _score_meta(fit.baseline),
    }
    for index, ev in evidence.items():
        payload[f"ev{int(index)}_depths"] = np.asarray(ev.depths_from_tip_um, float)
        payload[f"ev{int(index)}_weights"] = np.asarray(ev.weights, float)
    for name, scan in fit.scans.items():
        payload[f"scan_{name}_values"] = np.asarray(scan.values, float)
        payload[f"scan_{name}_explained"] = np.asarray(scan.explained, float)
    for tag, score in (("score", fit.score), ("baseline", fit.baseline)):
        payload.update(_match_arrays(tag, score))
    if tips is not None:
        payload["tips_ccf_um"] = np.asarray(tips, dtype=float)
    if entries is not None:
        payload["entries_ccf_um"] = np.asarray(entries, dtype=float)
    payload["meta"] = np.array(json.dumps(meta, indent=1))
    np.savez_compressed(out, **payload)
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def _score_meta(score: PlacementScore) -> dict:
    return {
        "explained": float(score.explained),
        "matched": int(score.matched),
        "available": int(score.available),
        "total_weight": float(score.total_weight),
        "offset_um": float(score.offset_um),
        "roll_deg": float(score.roll_deg),
        "tilt_deg": float(score.tilt_deg),
    }


def _match_arrays(tag: str, score: PlacementScore) -> dict:
    if not score.matches:
        return {}
    return {
        f"{tag}_match_shank": np.array([m.shank_index for m in score.matches]),
        f"{tag}_match_feature": np.array([m.feature_um for m in score.matches]),
        f"{tag}_match_atlas": np.array([m.atlas_um for m in score.matches]),
        f"{tag}_match_weight": np.array([m.weight for m in score.matches]),
        f"{tag}_match_gain": np.array([m.gain for m in score.matches]),
        f"{tag}_match_label": np.array([str(m.label) for m in score.matches]),
    }


def _matches_from(arrays: dict, tag: str) -> list:
    key = f"{tag}_match_shank"
    if key not in arrays:
        return []
    return [
        Match(shank_index=int(s), feature_um=float(f), atlas_um=float(a),
              weight=float(w), gain=float(g), label=str(lbl))
        for s, f, a, w, g, lbl in zip(
            arrays[key], arrays[f"{tag}_match_feature"], arrays[f"{tag}_match_atlas"],
            arrays[f"{tag}_match_weight"], arrays[f"{tag}_match_gain"],
            arrays[f"{tag}_match_label"], strict=False,
        )
    ]


def _score_from(arrays: dict, meta: dict, tag: str) -> PlacementScore:
    block = meta.get(tag, {})
    return PlacementScore(
        explained=float(block.get("explained", 0.0)),
        matched=int(block.get("matched", 0)),
        available=int(block.get("available", 0)),
        total_weight=float(block.get("total_weight", 0.0)),
        matches=_matches_from(arrays, tag),
        offset_um=float(block.get("offset_um", 0.0)),
        roll_deg=float(block.get("roll_deg", 0.0)),
        tilt_deg=float(block.get("tilt_deg", 0.0)),
    )


def load_fit(path) -> tuple[TrajectoryFit, dict, dict]:
    """Read one back: ``(fit, evidence, meta)``.

    ``meta`` carries the fingerprint and the notes. Compare the fingerprint against
    :func:`evidence_fingerprint` of the current features before trusting it - a cached
    fit that no longer matches its data is the failure this guard exists for.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}

    evidence = {
        int(index): ShankEvidence(
            shank_index=int(index),
            depths_from_tip_um=np.asarray(arrays[f"ev{int(index)}_depths"], float),
            weights=np.asarray(arrays[f"ev{int(index)}_weights"], float),
        )
        for index in meta.get("shanks", [])
        if f"ev{int(index)}_depths" in arrays
    }
    scans = {
        name: ParameterScan(
            name=name,
            values=np.asarray(arrays[f"scan_{name}_values"], float),
            explained=np.asarray(arrays[f"scan_{name}_explained"], float),
        )
        for name in meta.get("scans", [])
        if f"scan_{name}_values" in arrays
    }
    fit = TrajectoryFit(
        offset_um=float(meta.get("offset_um", 0.0)),
        roll_deg=float(meta.get("roll_deg", 0.0)),
        tilt_deg=float(meta.get("tilt_deg", 0.0)),
        score=_score_from(arrays, meta, "score"),
        baseline=_score_from(arrays, meta, "baseline"),
        scans=scans,
    )
    return fit, evidence, meta


def matches_current(path, evidence: dict) -> bool:
    """Whether a saved fit was computed from exactly these boundaries."""
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
    except Exception:
        return False
    return bool(meta.get("fingerprint")) and \
        meta["fingerprint"] == evidence_fingerprint(evidence)
