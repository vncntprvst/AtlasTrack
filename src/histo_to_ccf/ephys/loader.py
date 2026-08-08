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
        import spikeinterface.full as si
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


@dataclass
class LfpExcerpts:
    """LFP from several short, screened windows of one recording.

    The windows are kept **separate, not concatenated**. Joining them would put a
    step discontinuity at every seam, and a step is broadband - it would show up in
    the PSD as power at every frequency and contaminate exactly the depth-resolved
    band structure the excerpts were read for. Callers average per-window PSDs
    instead (see :func:`excerpt_psd`).
    """

    windows: list[np.ndarray]  # each (n_samples, n_channels), microvolts
    fs: float
    channel_depths_um: np.ndarray
    channel_x_um: np.ndarray
    channel_ids: list
    stream_name: str
    derived_from_ap: bool = False
    channel_shank_ids: np.ndarray | None = None
    # Every candidate window with its scores and, if rejected, the reason. Rejected
    # windows are reported rather than dropped so a bad recording is diagnosable and
    # a questionable rejection can be reviewed.
    verdicts: list = None  # list[EpochVerdict]
    referenced: bool = False

    def __post_init__(self) -> None:
        if self.verdicts is None:
            self.verdicts = []

    @property
    def kept_epochs(self) -> list[tuple[float, float]]:
        return [(v.t_start_s, v.t_end_s) for v in self.verdicts if v.kept]

    @property
    def total_seconds(self) -> float:
        return float(sum(w.shape[0] for w in self.windows)) / self.fs if self.fs else 0.0


def excerpt_psd(excerpts: LfpExcerpts, *, fmin: float = 0.0, fmax: float = 300.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Average the per-window PSDs. Returns ``(freqs, psd)`` as :func:`lfp_psd` does.

    Averaging power across windows (rather than taking the spectrum of concatenated
    traces) is what keeps the seams out of the result, and is also the right
    estimator: each window is an independent sample of the same spectrum.
    """
    from histo_to_ccf.ephys.features import lfp_psd

    if not excerpts.windows:
        return np.empty(0), np.empty((0, 0))
    total, freqs = None, np.empty(0)
    for window in excerpts.windows:
        freqs, psd = lfp_psd(window, excerpts.fs, fmin=fmin, fmax=fmax)
        total = psd if total is None else total + psd
    return freqs, total / len(excerpts.windows)


def list_streams(recording_dir: str | Path) -> list[str]:
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


def _open_lfp_recording(recording_dir, stream_name, lfp_fs: float):
    """Build the (lazy) LFP recording and say whether it was derived from AP."""
    si = _require_si()
    streams = list_streams(recording_dir)
    if not streams:
        raise RuntimeError(f"No Open Ephys streams found in {recording_dir}")

    if stream_name is None:
        stream_name = _select_lfp_stream(streams)
    if stream_name is not None:
        return si.read_openephys(str(recording_dir), stream_name=stream_name), stream_name, False

    ap = _select_ap_stream(streams)
    if ap is None:
        raise RuntimeError(f"No LFP or AP Neuropixels stream in {recording_dir}")
    rec = si.read_openephys(str(recording_dir), stream_name=ap)
    # Decimate the wideband AP stream to ~LFP rate first (resample applies its own
    # anti-alias low-pass), then band-limit: filtering at 30 kHz would cost ~12x more.
    if rec.get_sampling_frequency() > lfp_fs:
        rec = si.resample(rec, int(lfp_fs))
    rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=300.0, ignore_low_freq_error=True)
    return rec, ap, True


def _read_window(rec, start: int, stop: int) -> np.ndarray:
    # return_in_uV is the new name; older SpikeInterface uses return_scaled.
    try:
        traces = rec.get_traces(start_frame=start, end_frame=stop, return_in_uV=True)
    except TypeError:
        traces = rec.get_traces(start_frame=start, end_frame=stop, return_scaled=True)
    return np.asarray(traces, dtype=float)


def load_lfp_excerpts(
    recording_dir: str | Path,
    stream_name: str | None = None,
    *,
    window_s: float = 10.0,
    n_windows: int = 6,
    keep: int | None = None,
    lfp_fs: float = 2500.0,
    reference: bool = True,
    artifact_tolerance: float | None = None,
) -> LfpExcerpts:
    """Read a handful of screened LFP windows spread across a recording.

    This is the only part of the feature pipeline that touches raw data, and it is
    deliberately small: six 10 s windows at 384 channels and 2.5 kHz is ~115 MB read
    once, against 52-61 GB for the whole recording. SpikeInterface's preprocessing is
    lazy, so the decimation and filtering are paid only on the windows actually read.

    Windows are screened as they are read (:mod:`histo_to_ccf.ephys.epochs`): those
    dominated by cross-channel transients - which is what a lick artifact is - are
    rejected and reported, never silently averaged in. With ``reference`` the kept
    windows get a common median reference, the standard defence against whatever is
    left that appears on every channel at once.

    ``keep`` trims to the most informative N windows; ``None`` keeps all that pass.
    """
    from histo_to_ccf.ephys.epochs import (
        candidate_windows,
        common_median_reference,
        rank_epochs,
        screen_window,
    )

    rec, stream_name, derived = _open_lfp_recording(recording_dir, stream_name, lfp_fs)
    fs = float(rec.get_sampling_frequency())
    n_total = rec.get_num_samples()

    screen_kwargs = (
        {} if artifact_tolerance is None else {"artifact_tolerance": artifact_tolerance}
    )
    verdicts, traces_by_epoch = [], {}
    for t0, t1 in candidate_windows(n_total, fs, window_s=window_s, n_windows=n_windows):
        start, stop = round(t0 * fs), min(n_total, round(t1 * fs))
        if stop <= start:
            continue
        traces = _read_window(rec, start, stop)
        # Reference *before* screening. Common median referencing exists to remove
        # exactly the cross-channel transients the artifact score measures, so
        # screening the raw traces would reject windows that are perfectly usable
        # once referenced - and in a licking task that is most of them. What must
        # disqualify a window is what survives the standard denoising.
        if reference:
            traces = common_median_reference(traces)
        verdict = screen_window(traces, t0, t1, **screen_kwargs)
        verdicts.append(verdict)
        if verdict.kept:
            traces_by_epoch[(verdict.t_start_s, verdict.t_end_s)] = traces

    verdicts = rank_epochs(verdicts, keep=keep)
    windows = [
        traces_by_epoch[(v.t_start_s, v.t_end_s)] for v in verdicts if v.kept
    ]

    try:
        locs = np.asarray(rec.get_channel_locations(), dtype=float)
        x_um, depth_um = locs[:, 0], locs[:, 1]
    except Exception:
        n_ch = windows[0].shape[1] if windows else 0
        x_um, depth_um = np.zeros(n_ch), np.arange(n_ch, dtype=float)

    shank_ids = None
    try:
        sids = rec.get_probe().shank_ids
        if sids is not None and len(sids) == len(rec.channel_ids):
            shank_ids = np.asarray(sids)
    except Exception:
        shank_ids = None

    return LfpExcerpts(
        windows=windows,
        fs=fs,
        channel_depths_um=depth_um,
        channel_x_um=x_um,
        channel_ids=list(rec.channel_ids),
        stream_name=stream_name,
        derived_from_ap=derived,
        channel_shank_ids=shank_ids,
        verdicts=verdicts,
        referenced=bool(reference and windows),
    )


def load_lfp(
    recording_dir: str | Path,
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
