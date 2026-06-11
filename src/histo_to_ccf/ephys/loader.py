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
    # Per-channel shank id from the probe (probeinterface). ``None`` when the probe
    # doesn't define shanks. This is the *correct* way to split multi-shank probes -
    # NOT the x location, since a Neuropixels 2.0 shank has TWO electrode columns,
    # so unique-x over-splits one shank into two.
    channel_shank_ids: np.ndarray | None = None


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
        # Decimate the wideband AP stream to ~LFP rate first (resample applies its
        # own anti-alias low-pass), then band-limit. Doing the filter on the
        # decimated stream is far cheaper than filtering at 30 kHz.
        if rec.get_sampling_frequency() > lfp_fs:
            rec = si.resample(rec, int(lfp_fs))
        # ignore_low_freq_error: SpikeInterface rejects freq_min < ~1 Hz by default
        # (chunk-edge artifacts); we read one central window with a wide margin, so
        # the bypass is safe here and keeps the LFP drift band.
        rec = si.bandpass_filter(
            rec, freq_min=0.5, freq_max=300.0, ignore_low_freq_error=True
        )
        stream_name = ap
        derived = True
    else:
        rec = si.read_openephys(str(recording_dir), stream_name=stream_name)

    fs = float(rec.get_sampling_frequency())
    n_total = rec.get_num_samples()
    n_keep = min(n_total, int(max_seconds * fs))
    start = max(0, (n_total - n_keep) // 2)  # central window
    # return_in_uV is the new name; older SpikeInterface uses return_scaled.
    try:
        traces = rec.get_traces(start_frame=start, end_frame=start + n_keep, return_in_uV=True)
    except TypeError:
        traces = rec.get_traces(start_frame=start, end_frame=start + n_keep, return_scaled=True)

    try:
        locs = np.asarray(rec.get_channel_locations(), dtype=float)
        x_um, depth_um = locs[:, 0], locs[:, 1]
    except Exception:
        n_ch = traces.shape[1]
        x_um = np.zeros(n_ch)
        depth_um = np.arange(n_ch, dtype=float)

    # Per-channel shank id, in recording-channel order (the probe attached to the
    # recording is already aligned to the channels). Lets the alignment view split
    # a multi-shank probe correctly instead of guessing from x.
    shank_ids = None
    try:
        sids = rec.get_probe().shank_ids
        if sids is not None and len(sids) == traces.shape[1]:
            shank_ids = np.asarray(sids)
    except Exception:
        shank_ids = None

    return LfpData(
        traces=np.asarray(traces, dtype=float),
        fs=fs,
        channel_depths_um=depth_um,
        channel_x_um=x_um,
        channel_ids=list(rec.channel_ids),
        stream_name=stream_name,
        derived_from_ap=derived,
        channel_shank_ids=shank_ids,
    )
