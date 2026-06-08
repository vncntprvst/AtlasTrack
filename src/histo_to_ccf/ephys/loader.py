"""Load Open Ephys LFP via SpikeInterface (optional dependency).

SpikeInterface is heavy (and pulls neo / probeinterface), so it lives behind the
``ephys`` extra and is imported lazily here. Everything that touches it stays in
this module; the rest of the ephys package is pure numpy.

Neuropixels 1.0 records a dedicated LFP stream (``...-LFP``); NP 2.0 does not, so
when no LFP stream is present we derive a band-limited, decimated LFP from the AP
stream (low-pass + resample to ~2.5 kHz).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_INSTALL_HINT = (
    "SpikeInterface is required for ephys alignment. Install the extra:\n"
    "    pip install \"histo-to-ccf[ephys]\"\n"
    "(or: pip install spikeinterface)"
)


def _require_si():
    try:
        import spikeinterface.full as si  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return si


@dataclass
class LfpData:
    """LFP traces + channel geometry for one recording/stream."""

    traces: np.ndarray  # (n_samples, n_channels), microvolts
    fs: float
    channel_depths_um: np.ndarray  # (n_channels,) y location = distance along shank
    channel_x_um: np.ndarray  # (n_channels,) x location = shank column
    channel_ids: list
    stream_name: str
    derived_from_ap: bool = False


def list_streams(recording_dir: "str | Path") -> list[str]:
    """All Open Ephys binary stream names in a recording folder."""
    si = _require_si()
    names, _ = si.get_neo_streams("openephysbinary", str(recording_dir))
    return list(names)


def _select_lfp_stream(streams: list[str]) -> str | None:
    for s in streams:
        u = s.upper()
        if "LFP" in u or u.endswith("-LF") or u.endswith(".LF"):
            return s
    return None


def _select_ap_stream(streams: list[str]) -> str | None:
    cands = [s for s in streams if "Neuropix" in s and ("AP" in s.upper())]
    if cands:
        return cands[0]
    cands = [s for s in streams if "Neuropix" in s]
    return cands[0] if cands else (streams[0] if streams else None)


def load_lfp(
    recording_dir: "str | Path",
    stream_name: str | None = None,
    *,
    max_seconds: float = 60.0,
    lfp_fs: float = 2500.0,
) -> LfpData:
    """Load an LFP segment + channel geometry from an Open Ephys recording.

    Picks an LFP stream automatically when ``stream_name`` is omitted; if none
    exists it derives LFP from the AP stream (low-pass + resample to ``lfp_fs``).
    A central ``max_seconds`` window is read to keep memory bounded.
    """
    si = _require_si()
    streams = list_streams(recording_dir)
    if not streams:
        raise RuntimeError(f"No Open Ephys streams found in {recording_dir}")

    derived = False
    if stream_name is None:
        stream_name = _select_lfp_stream(streams)
    if stream_name is None:
        ap = _select_ap_stream(streams)
        if ap is None:
            raise RuntimeError(f"No LFP or AP Neuropixels stream in {recording_dir}")
        rec = si.read_openephys(str(recording_dir), stream_name=ap)
        rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=300.0)
        if rec.get_sampling_frequency() > lfp_fs:
            rec = si.resample(rec, int(lfp_fs))
        stream_name = ap
        derived = True
    else:
        rec = si.read_openephys(str(recording_dir), stream_name=stream_name)

    fs = float(rec.get_sampling_frequency())
    n_total = rec.get_num_samples()
    n_keep = min(n_total, int(max_seconds * fs))
    start = max(0, (n_total - n_keep) // 2)  # central window
    traces = rec.get_traces(start_frame=start, end_frame=start + n_keep, return_scaled=True)

    try:
        locs = np.asarray(rec.get_channel_locations(), dtype=float)
        x_um, depth_um = locs[:, 0], locs[:, 1]
    except Exception:
        n_ch = traces.shape[1]
        x_um = np.zeros(n_ch)
        depth_um = np.arange(n_ch, dtype=float)

    return LfpData(
        traces=np.asarray(traces, dtype=float),
        fs=fs,
        channel_depths_um=depth_um,
        channel_x_um=x_um,
        channel_ids=list(rec.channel_ids),
        stream_name=stream_name,
        derived_from_ap=derived,
    )
