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
    #: See :attr:`LfpExcerpts.geometry_source`.
    geometry_source: str = "recording"


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
    #: Where ``channel_depths_um`` came from. ``channel_index`` means the recording
    #: carried no geometry and none was supplied, so the depths are ordinals rather
    #: than micrometres - see :mod:`histo_to_ccf.ephys.probemap`.
    geometry_source: str = "recording"
    #: How many shanks the common median reference was taken within. 1 for a
    #: single-shank recording, 4 for a full bank. Recorded because two recordings are
    #: only comparable if this matched, and a silent mismatch shifts whole decades of
    #: power - see :func:`histo_to_ccf.ephys.epochs.common_median_reference`.
    reference_groups: int = 0

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


def _detect(recording_dir: str | Path):
    """Identify the recording, or raise naming what was actually there."""
    from histo_to_ccf.ephys.formats import detect_format

    detected = detect_format(recording_dir)
    if detected is None:
        raise RuntimeError(
            f"{recording_dir} is not a recording this app can read. Expected an Open "
            "Ephys record node (experimentN/recordingM), an Intan info.rhd/.rhs, or a "
            "SpikeGLX run folder of .bin/.meta pairs."
        )
    return detected


def list_streams(recording_dir: str | Path) -> list[str]:
    """All stream names in a recording folder, whatever wrote it."""
    from histo_to_ccf.ephys.formats import list_streams as _list

    return _list(_detect(recording_dir))


def _select_lfp_stream(streams: list[str], detected=None) -> str | None:
    from histo_to_ccf.ephys.formats import OPEN_EPHYS, DetectedRecording
    from histo_to_ccf.ephys.formats import select_lfp_stream as _sel

    if detected is None:  # back-compat: bare stream lists are Open Ephys
        detected = DetectedRecording(OPEN_EPHYS, Path("."), Path("."))
    return _sel(detected, streams)


def _select_ap_stream(streams: list[str], detected=None) -> str | None:
    from histo_to_ccf.ephys.formats import OPEN_EPHYS, DetectedRecording
    from histo_to_ccf.ephys.formats import select_wideband_stream as _sel

    if detected is None:
        detected = DetectedRecording(OPEN_EPHYS, Path("."), Path("."))
    return _sel(detected, streams)


def _open_lfp_recording(recording_dir, stream_name, lfp_fs: float):
    """Build the (lazy) LFP recording and say whether it was derived from wideband.

    Neuropixels 1.0 on Open Ephys and SpikeGLX records a real LFP stream, which is
    used as-is. Neuropixels 2.0 and every Intan recording have only a wideband
    stream, so LFP is derived from it: decimate to ``lfp_fs`` first (resampling
    applies its own anti-alias low-pass) and band-limit after, because filtering at
    30 kHz would cost about 12x more for the same result.
    """
    si = _require_si()
    from histo_to_ccf.ephys.formats import list_streams as _list
    from histo_to_ccf.ephys.formats import open_stream, select_lfp_stream, select_wideband_stream

    detected = _detect(recording_dir)
    streams = _list(detected)
    if not streams:
        raise RuntimeError(f"No {detected.format.label} streams found in {recording_dir}")

    if stream_name is None:
        stream_name = select_lfp_stream(detected, streams)
    if stream_name is not None:
        return open_stream(detected, stream_name), stream_name, False

    wideband = select_wideband_stream(detected, streams)
    if wideband is None:
        raise RuntimeError(
            f"No electrode stream in {recording_dir}; found {streams}. "
            "Auxiliary and digital streams carry no channel geometry, so they are "
            "never selected automatically - name the stream explicitly to use one."
        )
    rec = open_stream(detected, wideband)
    if rec.get_sampling_frequency() > lfp_fs:
        rec = si.resample(rec, int(lfp_fs))
    rec = si.bandpass_filter(rec, freq_min=0.5, freq_max=300.0, ignore_low_freq_error=True)
    return rec, wideband, True


def _channel_geometry(rec, probe_map, n_channels: int | None = None):
    """Channel depths/x/shank ids and where they came from.

    Order of preference: geometry stored in the recording, then an explicitly
    supplied probe map, then channel indices. The last is not a silent fallback -
    it is reported via the returned :class:`~histo_to_ccf.ephys.probemap.GeometrySource`
    so a caller computing depth-referenced features can refuse rather than treat an
    ordinal as a micrometre.

    A supplied map wins over the recording's own geometry only when the recording has
    none; overriding real stored geometry would be a silent way to break a working
    Neuropixels recording.
    """
    from histo_to_ccf.ephys.probemap import GeometrySource, resolve_probe_map

    try:
        locs = np.asarray(rec.get_channel_locations(), dtype=float)
        if locs.ndim == 2 and locs.shape[1] >= 2 and locs.shape[0] > 0:
            shank_ids = None
            try:
                sids = rec.get_probe().shank_ids
                if sids is not None and len(sids) == locs.shape[0]:
                    shank_ids = np.asarray(sids)
            except Exception:
                shank_ids = None
            return (
                locs[:, 1], locs[:, 0], shank_ids, GeometrySource.RECORDING,
            )
    except Exception:
        pass

    n_ch = n_channels if n_channels is not None else len(rec.channel_ids)
    resolved = resolve_probe_map(probe_map, n_channels=n_ch)
    if resolved is not None:
        return (
            resolved.depth_um, resolved.x_um, resolved.shank_ids, resolved.source,
        )
    return (
        np.arange(n_ch, dtype=float),
        np.zeros(n_ch, dtype=float),
        None,
        GeometrySource.CHANNEL_INDEX,
    )


def _read_window(rec, start: int, stop: int) -> np.ndarray:
    # return_in_uV is the new name; older SpikeInterface uses return_scaled.
    try:
        traces = rec.get_traces(start_frame=start, end_frame=stop, return_in_uV=True)
    except TypeError:
        traces = rec.get_traces(start_frame=start, end_frame=stop, return_scaled=True)
    return np.asarray(traces, dtype=float)


def _reference_groups(x_um: np.ndarray) -> np.ndarray | None:
    """Per-channel shank label for referencing, from x alone.

    x, not the probe's ``group`` property: ``group`` numbers the shanks *present in
    this recording* from zero, so it cannot say which physical shank a single-shank
    recording is on - but for referencing all that matters is that channels on one
    shank get one label, which the rounded x gives directly and unambiguously.
    """
    from histo_to_ccf.ephys.recordings import SHANK_PITCH_UM

    x = np.asarray(x_um, dtype=float).ravel()
    if x.size == 0:
        return None
    return np.rint(x / SHANK_PITCH_UM).astype(int)


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
    probe_map: object = None,
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

    # Geometry first: the reference is taken per shank, so the shank of each channel
    # has to be known before the first window is referenced.
    depth_um, x_um, shank_ids, geometry_source = _channel_geometry(rec, probe_map)
    physical = geometry_source.is_physical
    ref_groups = _reference_groups(x_um) if (reference and physical) else None

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
            groups = ref_groups if (
                ref_groups is not None and ref_groups.size == traces.shape[1]
            ) else None
            traces = common_median_reference(traces, groups)
        verdict = screen_window(traces, t0, t1, **screen_kwargs)
        verdicts.append(verdict)
        if verdict.kept:
            traces_by_epoch[(verdict.t_start_s, verdict.t_end_s)] = traces

    verdicts = rank_epochs(verdicts, keep=keep)
    windows = [
        traces_by_epoch[(v.t_start_s, v.t_end_s)] for v in verdicts if v.kept
    ]

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
        reference_groups=(
            int(np.unique(ref_groups).size) if ref_groups is not None else 0
        ),
        geometry_source=str(geometry_source.value),
    )


def load_lfp(
    recording_dir: str | Path,
    stream_name: str | None = None,
    *,
    max_seconds: float = 60.0,
    lfp_fs: float = 2500.0,
    probe_map: object = None,
) -> LfpData:
    """Load an LFP segment + channel geometry from one recording.

    Reads Open Ephys, Intan and SpikeGLX (see :mod:`histo_to_ccf.ephys.formats`).
    Picks a dedicated LFP stream automatically when ``stream_name`` is omitted; where
    the format has none - Neuropixels 2.0, and every Intan recording - LFP is derived
    from the wideband stream. A central ``max_seconds`` window is read to keep memory
    bounded.
    """
    rec, stream_name, derived = _open_lfp_recording(
        recording_dir, stream_name, lfp_fs
    )
    fs = float(rec.get_sampling_frequency())
    n_total = rec.get_num_samples()
    n_keep = min(n_total, int(max_seconds * fs))
    start = max(0, (n_total - n_keep) // 2)  # central window
    # return_in_uV is the new name; older SpikeInterface uses return_scaled.
    try:
        traces = rec.get_traces(start_frame=start, end_frame=start + n_keep, return_in_uV=True)
    except TypeError:
        traces = rec.get_traces(start_frame=start, end_frame=start + n_keep, return_scaled=True)

    # Per-channel geometry, in recording-channel order. Lets the alignment view split
    # a multi-shank probe correctly instead of guessing from x.
    depth_um, x_um, shank_ids, geometry_source = _channel_geometry(
        rec, probe_map, traces.shape[1]
    )

    return LfpData(
        traces=np.asarray(traces, dtype=float),
        fs=fs,
        channel_depths_um=depth_um,
        channel_x_um=x_um,
        channel_ids=list(rec.channel_ids),
        stream_name=stream_name,
        derived_from_ap=derived,
        channel_shank_ids=shank_ids,
        geometry_source=str(geometry_source.value),
    )
