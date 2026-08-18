"""Finding the recordings that belong to one penetration, and what they cover.

Aligning a 4.5-5.4 mm track needs several recordings, because one Neuropixels 2.0
bank spans ~720 µm of shank. Asking the user to hunt those down by hand, type an
insertion depth for each and remember which ones share an insertion is where the
ergonomics collapse. So this module derives everything the data already knows and
asks only for what it cannot.

The split, which is the whole design:

* **From the recording itself** - probes, streams, which shanks carry sites, where
  those sites sit on the shank, sampling rate, duration. Read from the probe
  geometry only; no traces are touched, so scanning a session is fast even on the
  spinning disk.
* **From an optional sidecar table** - insertion depth, dye, orientation. These live
  in the experimenter's notes and nowhere in the files.
* **From the user** - only the fields still missing after the first two.

**Everything degrades when there is no sidecar, and that is the normal case.** Only
one field is truly load-bearing: ``insertion_depth_um``. Without it, recordings that
differ only in *bank* are still placed correctly - the bank offset is derivable from
the site positions - and only recordings taken at different depths need a number
typed in. :func:`missing_depths` says exactly which ones, so the ask is as small as
the data allows. Nothing here guesses a depth, and nothing assumes two recordings
share one: an unknown depth stays ``None`` all the way through.

Grouping is by ``(subject, date, probe, dye)``. Dye earns its place in the key
because it is what ties a penetration to a histology track - one dye per
penetration, and the registration projects are named for it - so an open project
identifies its own recordings without the user selecting anything.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from histo_to_ccf.ephys.recordings import (
    NP2_ROW_PITCH_UM,
    SHANK_PITCH_UM,
    channels_for_shank,
)

# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class ShankCoverage:
    """Which part of one shank a recording sampled, in µm from the tip.

    ``step_um`` is the median spacing between occupied rows. It separates the two
    layouts in this dataset at a glance: a full bank reads both columns of 48
    consecutive rows (step 15 µm over 720 µm), while the single-column depth
    recordings read one site per row over 5745 µm at the same 15 µm step but four
    times the reach.
    """

    shank: int
    n_sites: int
    top_um: float  # closest to the tip
    bottom_um: float  # furthest up the shank
    step_um: float
    n_columns: int

    @property
    def extent_um(self) -> float:
        return self.bottom_um - self.top_um


@dataclass(frozen=True)
class StreamInfo:
    """One probe's channels in one recording, derived from the data alone.

    Nothing here comes from notes or filenames except the labels, so this is what
    remains trustworthy when a project has no metadata table at all.
    """

    recording_dir: str
    stream_name: str
    probe_label: str
    n_channels: int
    sampling_rate_hz: float
    duration_s: float
    coverage: tuple[ShankCoverage, ...]
    subject: str | None = None
    session_date: date | None = None
    recording_label: str | None = None

    @property
    def shanks(self) -> tuple[int, ...]:
        return tuple(c.shank for c in self.coverage)

    @property
    def is_absolute_geometry(self) -> bool:
        """Whether site positions are absolute on the shank rather than bank-local.

        Absolute positions already include the bank offset; adding it again pushes
        the recording a bank too shallow. Measured on LO_06 2026-02-07: the bank
        97-192 recording reports y = 720-1410 µm, not 0-690. A recording whose
        sites start well above the tip is reporting absolute positions.
        """
        return any(c.top_um > NP2_ROW_PITCH_UM * 2 for c in self.coverage)

    def describe_config(self) -> str:
        """A short human label like ``all shanks 0-720 µm`` or ``shank 0, 0-5745 µm``."""
        if not self.coverage:
            return "no sites"
        lo = min(c.top_um for c in self.coverage)
        hi = max(c.bottom_um for c in self.coverage)
        which = (
            f"all {len(self.coverage)} shanks"
            if len(self.coverage) > 1
            else f"shank {self.coverage[0].shank}"
        )
        cols = self.coverage[0].n_columns
        col = "" if cols != 1 else ", single column"
        return f"{which}, {lo:.0f}-{hi:.0f} µm from tip{col}"


@dataclass
class RecordingCandidate:
    """A stream plus whatever metadata could be attached to it.

    ``depth_source`` records where ``insertion_depth_um`` came from - ``"sidecar"``,
    ``"user"``, or ``"unknown"`` when it is still missing. It is kept because a depth
    the user typed and a depth read from a table fail in different ways, and a
    review screen that cannot tell them apart hides the one worth checking.
    """

    stream: StreamInfo
    insertion_depth_um: float | None = None
    dye: str | None = None
    orientation: str | None = None
    notes: str | None = None
    depth_source: str = "unknown"
    stated_config: str | None = None

    @property
    def probe_label(self) -> str:
        return self.stream.probe_label

    def config_mismatch(self) -> str | None:
        """Whether the stated electrode config disagrees with the measured geometry.

        A cheap guard on the class of error that is otherwise invisible until the
        depths come out wrong: the notes say one bank and the probe map says
        another. Only clear disagreements are reported - the stated strings are free
        text and a missing one is not a mismatch.
        """
        stated = (self.stated_config or "").lower()
        if not stated or not self.stream.coverage:
            return None
        n_shanks = len(self.stream.coverage)
        says_all = "all shank" in stated
        says_one = bool(re.search(r"\bshank\s*\d+\b", stated)) and not says_all
        if says_all and n_shanks == 1:
            return f"notes say all shanks, geometry has {n_shanks}"
        if says_one and n_shanks > 1:
            return f"notes say a single shank, geometry has {n_shanks}"
        m = re.search(r"(\d+)\s*-\s*(\d+)", stated)
        if m and self.stream.is_absolute_geometry:
            # "97-192" -> sites should start (97-1)//2 * 15 = 720 µm up the shank.
            expected = ((int(m.group(1)) - 1) // 2) * NP2_ROW_PITCH_UM
            got = min(c.top_um for c in self.stream.coverage)
            if abs(got - expected) > NP2_ROW_PITCH_UM * 4:
                return f"notes say electrodes {m.group(0)} (≈{expected:.0f} µm), geometry starts at {got:.0f} µm"
        return None


@dataclass
class Penetration:
    """Recordings sharing one insertion, and what they jointly cover."""

    subject: str | None
    session_date: date | None
    probe_label: str
    dye: str | None
    recordings: list[RecordingCandidate] = field(default_factory=list)
    #: Folder the recordings were grouped by when the path gave no subject or date.
    #: ``None`` when the identity came from the path, which is the normal case.
    folder_hint: str | None = None

    @property
    def key(self) -> tuple:
        return (self.subject, self.session_date, self.probe_label, self.dye,
                self.folder_hint)

    @property
    def identified_from_path(self) -> bool:
        """Whether the subject and date were readable from the directory layout."""
        return self.subject is not None and self.session_date is not None

    @property
    def label(self) -> str:
        if not self.identified_from_path and self.folder_hint:
            return f"{Path(self.folder_hint).name} {self.probe_label}" + (
                f" ({self.dye})" if self.dye else ""
            )
        d = self.session_date.isoformat() if self.session_date else "?"
        return f"{self.subject or '?'} {d} {self.probe_label}" + (
            f" ({self.dye})" if self.dye else ""
        )

    @property
    def shanks(self) -> tuple[int, ...]:
        return tuple(sorted({s for r in self.recordings for s in r.stream.shanks}))

    @property
    def depths(self) -> list[float]:
        return sorted({r.insertion_depth_um for r in self.recordings
                       if r.insertion_depth_um is not None})

    def missing_depths(self) -> list[RecordingCandidate]:
        """Recordings that still need an insertion depth typed in.

        Empty when every recording has one **or** when they are all at one depth and
        only the banks differ - in that case the shared axis is recoverable from the
        site positions alone and no number is needed to compare them.
        """
        return [r for r in self.recordings if r.insertion_depth_um is None]


# ----------------------------------------------------------------- reading the data


def _require_si():
    try:
        import spikeinterface.full as si
    except ImportError as exc:  # pragma: no cover - exercised by the extras gate
        raise ImportError(
            "Reading recordings needs SpikeInterface: pip install 'histo-to-ccf[ephys]'"
        ) from exc
    return si


def find_record_nodes(root: str | Path, *, max_depth: int = 6) -> list[Path]:
    """Open Ephys record node directories at or under ``root``.

    ``root`` itself counts: pointing the file dialog straight at
    ``.../raw_ephys_data/Record Node 104`` is the obvious thing to do, and finding
    nothing there reads as "no data" rather than "wrong level".

    A record node is recognised by the ``experimentN/recordingM`` children Open Ephys
    writes inside it, with the folder name only as a shortcut. Structure over name
    because the name is the part people change: a copied or renamed node is still a
    record node, and its experiments are what the reader actually opens.

    Bounded in depth because these trees also contain sorting outputs and zarr
    stores with thousands of entries, and an unbounded walk over the reference disk
    is slow enough to look like a hang.
    """
    root = Path(root)
    if not root.exists():
        return []
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = [e for e in d.iterdir() if e.is_dir()]
        except (PermissionError, OSError):
            continue
        if d.name.lower().startswith("record node") or _holds_experiments(entries):
            found.append(d)
            continue  # a record node's children are experiments, not more nodes
        if depth >= max_depth:
            continue
        for e in entries:
            if not e.name.endswith(".zarr"):
                stack.append((e, depth + 1))
    return sorted(found)


_DATE_DIR = re.compile(r"^(\d{4})[_-](\d{2})[_-](\d{2})$")

#: Directories that sit between a session and its record nodes and name nothing.
#: Used only for the no-date fallback, where the goal is to land on the folder a user
#: would call "the session".
_CONTAINER_DIRS = frozenset({
    "raw ephys data", "raw ephys", "ephys", "raw", "raw data", "data",
    "openephys", "open ephys", "recording", "recordings",
})
_EXPERIMENT_DIR = re.compile(r"^experiment\d+$", re.IGNORECASE)
_RECORDING_DIR = re.compile(r"^recording\d+$", re.IGNORECASE)


def _holds_experiments(entries: list[Path]) -> bool:
    """Whether these children are Open Ephys ``experimentN/recordingM`` folders.

    Both halves are required. The AIND sorting outputs put zarr stores named
    ``experiment1_Record Node 104#...ProbeA-AP_recording1_group0.zarr`` beside the raw
    data, and matching on ``experiment*`` alone claimed all six of those folders as
    record nodes - harmless in the result, but it cost a failed reader call each on
    the reference disk and stopped the walk descending through them.
    """
    for e in entries:
        if not _EXPERIMENT_DIR.match(e.name):
            continue
        try:
            if any(_RECORDING_DIR.match(c.name) and c.is_dir() for c in e.iterdir()):
                return True
        except (PermissionError, OSError):
            continue
    return False


def infer_path_ids(record_node: Path) -> tuple[str | None, date | None, str | None]:
    """Guess ``(subject, date, recording)`` from the directory layout.

    Anchored on a ``YYYY_MM_DD`` component, which is the one part of these trees
    that is unambiguous: the subject is the directory above it and the recording the
    one below. Returns ``None`` for anything that cannot be read that way rather
    than inventing a label - a wrong subject silently splits one penetration in two.
    """
    parts = list(record_node.parts)
    for i, p in enumerate(parts):
        m = _DATE_DIR.match(p)
        if not m:
            continue
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        subject = parts[i - 1] if i >= 1 else None
        recording = parts[i + 1] if i + 1 < len(parts) else None
        return subject, d, recording
    return None, None, None


def _probe_label(stream_name: str) -> str:
    """``...Neuropix-PXI-100.ProbeA`` -> ``ProbeA``; falls back to the stream name."""
    tail = stream_name.rsplit(".", 1)[-1].strip()
    return tail or stream_name


def _shank_coverage(x_um, y_um, shank_ids) -> tuple[ShankCoverage, ...]:
    x = np.asarray(x_um, dtype=float).ravel()
    y = np.asarray(y_um, dtype=float).ravel()
    out: list[ShankCoverage] = []
    groups = sorted({int(g) for g in np.rint(x / SHANK_PITCH_UM).astype(int)})
    for s in groups:
        m = channels_for_shank(s, shank_ids, x)
        if m is None or not m.any():
            continue
        ys = np.sort(np.unique(y[m]))
        step = float(np.median(np.diff(ys))) if ys.size > 1 else 0.0
        out.append(
            ShankCoverage(
                shank=s,
                n_sites=int(m.sum()),
                top_um=float(ys.min()),
                bottom_um=float(ys.max()),
                step_um=step,
                n_columns=int(np.unique(x[m]).size),
            )
        )
    return tuple(out)


def describe_stream(recording_dir: str | Path, stream_name: str) -> StreamInfo:
    """Read one stream's geometry and timing. Does not touch the traces."""
    si = _require_si()
    recording_dir = Path(recording_dir)
    rec = si.read_openephys(str(recording_dir), stream_name=stream_name)
    loc = rec.get_channel_locations()
    try:
        shank_ids = np.asarray(rec.get_property("group"))
    except Exception:
        shank_ids = None
    subject, session_date, label = infer_path_ids(recording_dir)
    return StreamInfo(
        recording_dir=str(recording_dir),
        stream_name=stream_name,
        probe_label=_probe_label(stream_name),
        n_channels=int(rec.get_num_channels()),
        sampling_rate_hz=float(rec.get_sampling_frequency()),
        duration_s=float(rec.get_total_duration()),
        coverage=_shank_coverage(loc[:, 0], loc[:, 1], shank_ids),
        subject=subject,
        session_date=session_date,
        recording_label=label,
    )


def scan_streams(root: str | Path, *, probe_only: bool = True) -> list[StreamInfo]:
    """Describe every Neuropixels stream under ``root``.

    Streams that fail to open are skipped rather than aborting the scan: one
    unreadable recording in a session should not cost the user the other nine.
    """
    from histo_to_ccf.ephys.loader import list_streams

    out: list[StreamInfo] = []
    for node in find_record_nodes(root):
        try:
            names = list_streams(node)
        except Exception:
            continue
        for name in names:
            if probe_only and "Neuropix" not in name:
                continue
            try:
                out.append(describe_stream(node, name))
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------- the sidecar

#: Default column names, matching the lab's ``session_status.csv``. Every project
#: spells these differently, so the mapping is data rather than hard-coded lookups.
DEFAULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "subject": ("subject", "animal", "mouse", "subject_id"),
    "date": ("date", "session_date", "session"),
    "recording": ("recording", "session", "run", "recording_label"),
    "probe": ("probe", "probe_label"),
    "insertion_depth_um": ("insertion depth", "insertion_depth", "depth", "depth_um"),
    "dye": ("dye", "tracer", "label"),
    "orientation": ("orientation", "angles", "pitch_roll"),
    "electrodes_config": ("electrodes config", "electrodes", "bank", "config"),
    "notes": ("notes", "comment", "comments"),
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def parse_date(value) -> date | None:
    """Parse the date spellings these tables use; ``None`` when unreadable.

    Accepts ``YYYY-MM-DD`` / ``YYYY_MM_DD`` and the US ``M/D/YYYY`` the lab's sheet
    uses. Two-digit years are refused rather than guessed at a century.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    m = _DATE_DIR.match(s.replace("-", "_"))
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:  # M/D/YYYY
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    return None


def normalise_dye(value: str | None) -> str | None:
    """Reduce a dye name to the colour, which is what links ephys to histology.

    ``DiI (red)``, ``CM-DiI (red)`` and a bare ``red`` all name the same track, and
    the registration projects are named for the colour alone
    (``LO_07_far red_whole.json``). Anything unrecognised is passed through
    lowercased rather than dropped.
    """
    if not value:
        return None
    s = str(value).strip().lower()
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        s = m.group(1).strip()
    for colour in ("far red", "deep red", "green", "red", "blue"):
        if colour in s:
            return "far red" if colour == "deep red" else colour
    return s or None


def read_sidecar(path: str | Path, *, columns: dict | None = None) -> list[dict]:
    """Read a metadata table into normalised rows.

    Any CSV with recognisable headers works; ``columns`` overrides the mapping for
    tables that spell things differently. Unmapped columns are kept verbatim so
    nothing is lost, and a table missing every recognised column raises rather than
    returning rows that would silently match nothing.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raise ValueError(
            f"{path.name} is a spreadsheet; export it to CSV first "
            "(no Excel reader is installed and the base install stays light)"
        )
    mapping = {**DEFAULT_COLUMNS, **(columns or {})}
    lookup = {_norm(alias): canon for canon, aliases in mapping.items() for alias in aliases}

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        resolved = {h: lookup.get(_norm(h), h) for h in headers}
        if not any(v in mapping for v in resolved.values()):
            raise ValueError(
                f"{path.name} has no recognisable columns (found {headers!r}); "
                "pass `columns=` to map them"
            )
        rows: list[dict] = []
        for raw in reader:
            row = {resolved[k]: v for k, v in raw.items() if k is not None}
            row["date"] = parse_date(row.get("date"))
            row["dye"] = normalise_dye(row.get("dye"))
            depth = str(row.get("insertion_depth_um") or "").strip()
            try:
                row["insertion_depth_um"] = float(depth) if depth else None
            except ValueError:
                row["insertion_depth_um"] = None
            rows.append(row)
    return rows


def _probe_key(value) -> str:
    """``A`` / ``probeA`` / ``ProbeA`` all mean the same probe."""
    s = _norm(value).replace("probe", "").strip()
    return s or _norm(value)


def attach_metadata(
    streams: list[StreamInfo], rows: list[dict] | None
) -> list[RecordingCandidate]:
    """Match sidecar rows onto streams; leave the rest unknown.

    Matching is on ``(subject, date, recording, probe)`` and falls back to
    ``(subject, date, probe)`` when the recording labels do not line up. A stream
    with no matching row is still returned - with ``depth_source="unknown"`` - so a
    project with no table at all yields the same structure, just emptier.
    """
    by_full: dict[tuple, dict] = {}
    by_session: dict[tuple, list[dict]] = {}
    for r in rows or []:
        subj, d, probe = _norm(r.get("subject")), r.get("date"), _probe_key(r.get("probe"))
        by_full[(subj, d, _norm(r.get("recording")), probe)] = r
        by_session.setdefault((subj, d, probe), []).append(r)

    out: list[RecordingCandidate] = []
    for s in streams:
        key_full = (
            _norm(s.subject), s.session_date, _norm(s.recording_label),
            _probe_key(s.probe_label),
        )
        row = by_full.get(key_full)
        if row is None:
            same = by_session.get((_norm(s.subject), s.session_date,
                                   _probe_key(s.probe_label)), [])
            # Only usable when the session is unambiguous about the depth.
            depths = {r.get("insertion_depth_um") for r in same}
            row = same[0] if len(same) == 1 or len(depths) == 1 else None
        if row is None:
            out.append(RecordingCandidate(stream=s))
            continue
        depth = row.get("insertion_depth_um")
        out.append(
            RecordingCandidate(
                stream=s,
                insertion_depth_um=depth,
                dye=row.get("dye"),
                orientation=row.get("orientation") or None,
                notes=row.get("notes") or None,
                depth_source="sidecar" if depth is not None else "unknown",
                stated_config=row.get("electrodes_config") or None,
            )
        )
    return out


# ----------------------------------------------------------------------- grouping


def session_folder(record_node: str | Path) -> str:
    """The folder a record node belongs to, for grouping when the path says nothing.

    One level above the record node, skipping the container directories acquisition
    software and labs habitually insert (``raw_ephys_data``, ``ephys``, ``raw``...).
    On ``.../LO_07_005/raw_ephys_data/Record Node 104`` that is ``LO_07_005``, which is
    the session; on ``.../session_final/Record Node 104`` it is ``session_final``.
    """
    node = Path(record_node)
    parent = node.parent
    if _norm(parent.name) in _CONTAINER_DIRS and parent.parent != parent:
        parent = parent.parent
    return str(parent)


def group_penetrations(candidates: list[RecordingCandidate]) -> list[Penetration]:
    """Group recordings by ``(subject, date, probe, dye)``.

    Dye is part of the key because it is the link to a histology track, but a
    missing dye does not split a group off on its own: with no table there is no dye
    at all, and ``(subject, date, probe)`` is still the right grouping for a day's
    recordings on one probe.

    **When the path yields neither subject nor date, the containing folder joins the
    key.** Those fields are read from a ``YYYY_MM_DD`` component, which is this lab's
    convention and not anyone else's; without one every recording of a probe anywhere
    under the scanned root previously collapsed into a single "penetration", and the
    compute step would then stack recordings from *different insertions* onto one
    depth axis. Splitting can be undone by the user in a moment; a silent merge is
    read as fact and produces a confident, wrong alignment. See
    :func:`grouping_warnings`, which says so out loud.
    """
    groups: dict[tuple, Penetration] = {}
    for c in candidates:
        s = c.stream
        hint = (None if (s.subject is not None and s.session_date is not None)
                else session_folder(s.recording_dir))
        key = (s.subject, s.session_date, c.probe_label, c.dye, hint)
        pen = groups.get(key)
        if pen is None:
            pen = Penetration(
                subject=s.subject, session_date=s.session_date,
                probe_label=c.probe_label, dye=c.dye, folder_hint=hint,
            )
            groups[key] = pen
        pen.recordings.append(c)
    for pen in groups.values():
        pen.recordings.sort(key=lambda r: (r.stream.recording_label or "",
                                           r.stream.stream_name))
    return sorted(
        groups.values(),
        key=lambda p: (str(p.subject), p.session_date or date.min, p.probe_label,
                       p.folder_hint or ""),
    )


def grouping_warnings(penetrations: list[Penetration]) -> list[str]:
    """What the user needs to check about how these recordings were grouped.

    Returns plain sentences, empty when the path identified everything. Two cases are
    worth saying out loud, and both are silent otherwise:

    * The subject and date could not be read, so grouping fell back to folders. If two
      folders hold the *same* insertion they are now two penetrations and have to be
      scanned or added separately.
    * A folder-grouped penetration holds recordings whose insertion depth is unknown
      *and* whose coverage differs, which is what a stack of different insertions looks
      like from the outside.
    """
    out: list[str] = []
    unidentified = [p for p in penetrations if not p.identified_from_path]
    if unidentified:
        folders = sorted({p.folder_hint or "?" for p in unidentified})
        out.append(
            f"No YYYY_MM_DD folder in these paths, so the session could not be read. "
            f"{len(unidentified)} penetration(s) were grouped by folder instead "
            f"({len(folders)} folder(s)). If one insertion spans several folders they "
            "will appear separately; if one folder holds several insertions, scan one "
            "session folder at a time."
        )
    for pen in unidentified:
        configs = {r.stream.describe_config() for r in pen.recordings}
        if len(pen.recordings) > 1 and pen.missing_depths() and len(configs) > 1:
            out.append(
                f"{pen.label}: {len(pen.recordings)} recordings with different "
                "electrode configurations and no insertion depth. Set the depths, or "
                "untick all but one - recordings from different insertions cannot "
                "share a depth axis."
            )
    return out


def coverage_from_tip(
    pen: Penetration, shank: int, *, reference_depth_um: float | None = None
) -> list[tuple[float, float]]:
    """Merged coverage of one shank, in µm from the tip at ``reference_depth_um``.

    Recordings taken at different insertion depths sample different tissue with the
    same electrodes, so their spans are placed by depth below the surface and then
    expressed in the frame of the deepest insertion - the position the tip actually
    reached, and the frame the histology track is measured in.

    Recordings with no known depth are placed only when every depth is unknown, in
    which case they are all assumed to be at one insertion; mixing a known and an
    unknown depth would be a guess, so the unknown ones are left out. Use
    :meth:`Penetration.missing_depths` to find out before trusting the result.
    """
    depths = pen.depths
    if reference_depth_um is None:
        reference_depth_um = max(depths) if depths else None
    spans: list[tuple[float, float]] = []
    for r in pen.recordings:
        cov = next((c for c in r.stream.coverage if c.shank == shank), None)
        if cov is None:
            continue
        if reference_depth_um is None:
            shift = 0.0  # no depths at all: one insertion assumed
        elif r.insertion_depth_um is None:
            continue
        else:
            shift = float(reference_depth_um) - float(r.insertion_depth_um)
        spans.append((cov.top_um + shift, cov.bottom_um + shift))
    return merge_spans(spans)


def derive_electrode_range(
    cov: ShankCoverage, *, row_pitch_um: float = NP2_ROW_PITCH_UM, columns: int = 2
) -> tuple[int, int] | None:
    """The 1-based electrode numbers a full-bank coverage corresponds to.

    Inverts :func:`histo_to_ccf.ephys.recordings.bank_offset_um`, so the notes and the
    geometry can be checked against each other in both directions: LO_06's bank
    starting 720 µm up the shank comes back as ``(97, 192)``, which is exactly what
    the notes say.

    ``None`` when the coverage is not a contiguous both-column block - the
    single-column depth recordings read one site per row over the whole shank, and
    calling that "electrodes 1-768" would claim twice the sites that were recorded.
    """
    if cov.step_um <= 0 or cov.n_columns != columns:
        return None
    rows = round(cov.extent_um / row_pitch_um) + 1
    if cov.n_sites != rows * columns:
        return None
    first = round(cov.top_um / row_pitch_um) * columns + 1
    return (first, first + cov.n_sites - 1)


def merge_spans(spans: list[tuple[float, float]], *, gap_um: float = NP2_ROW_PITCH_UM
                ) -> list[tuple[float, float]]:
    """Merge overlapping or abutting spans.

    Two banks that abut are separated by exactly one row pitch - the top site of one
    and the bottom site of the next - and reporting that as a hole would bury the
    real gaps in noise.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    out = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo - out[-1][1] <= gap_um:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def discover(
    root: str | Path, *, sidecar: str | Path | None = None, columns: dict | None = None
) -> list[Penetration]:
    """Scan ``root`` and return its penetrations, enriched by ``sidecar`` if given."""
    rows = read_sidecar(sidecar, columns=columns) if sidecar else None
    return group_penetrations(attach_metadata(scan_streams(root), rows))
