"""Which acquisition system wrote a recording, and how to open it.

The feature pipeline was written against Open Ephys because that is what the
Neuropixels rig produces, and the coupling had spread into stream listing, stream
selection and two separate open paths. This module isolates it, so that supporting
Intan and SpikeGLX is a matter of naming a reader rather than editing the LFP code.

Three formats, distinguished by what they leave on disk:

``openephys``
    A record node holding ``experimentN/recordingM``. Multi-stream: Neuropixels 1.0
    writes a dedicated ``...-LFP`` stream, 2.0 does not.
``intan``
    An ``info.rhd`` (or ``.rhs``) beside ``amplifier.dat`` and friends, or a single
    ``.rhd`` in the traditional layout. One wideband amplifier stream at 20-30 kHz -
    there is no LFP stream, so LFP is always derived. **Carries no geometry**: see
    :mod:`histo_to_ccf.ephys.probemap`.
``spikeglx``
    A run folder of ``*.imec0.ap.bin`` / ``.meta`` pairs, optionally with ``.lf``
    (Neuropixels 1.0) and a ``.nidq`` sync stream. The ``.meta`` carries the imro
    table, so geometry comes for free.

Everything here is headless; SpikeInterface is imported lazily so the module can be
used to *identify* a recording without the ``ephys`` extra installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Intan and Open Ephys both write a ``settings.xml``, so the file name alone cannot
# tell them apart. Intan's root element is ``<IntanRHX>``; Open Ephys's is
# ``<SETTINGS>``. Detection keys off the .rhd/experiment structure instead and never
# has to open the XML, but this is why "has settings.xml" is not a rule below.
_INTAN_SUFFIXES = (".rhd", ".rhs")
_SPIKEGLX_TOKENS = (".imec", ".nidq")
# Exact, not prefix: see _is_open_ephys for what prefix matching claimed.
_EXPERIMENT_DIR = re.compile(r"^experiment\d+$", re.IGNORECASE)
_RECORDING_DIR = re.compile(r"^recording\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class RecordingFormat:
    """One acquisition format and the SpikeInterface entry points for it."""

    key: str
    label: str
    #: Name neo/SpikeInterface use for :func:`spikeinterface.get_neo_streams`.
    neo_name: str
    #: Whether the reader is handed a file (Intan) or a directory (the others).
    wants_file: bool = False
    #: Whether the format stores probe geometry. When False, a probe map has to be
    #: supplied before channel depths mean anything - see
    #: :func:`histo_to_ccf.ephys.probemap.resolve_probe`.
    carries_geometry: bool = True

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


OPEN_EPHYS = RecordingFormat("openephys", "Open Ephys", "openephysbinary")
INTAN = RecordingFormat(
    "intan", "Intan RHD/RHS", "intan", wants_file=True, carries_geometry=False
)
SPIKEGLX = RecordingFormat("spikeglx", "SpikeGLX", "spikeglx")

FORMATS: tuple[RecordingFormat, ...] = (OPEN_EPHYS, INTAN, SPIKEGLX)
FORMATS_BY_KEY: dict[str, RecordingFormat] = {f.key: f for f in FORMATS}


@dataclass(frozen=True)
class DetectedRecording:
    """A recording, its format, and the exact path its reader wants.

    ``root`` is what the user pointed at; ``path`` is what SpikeInterface is given.
    They differ for Intan, where the reader takes the ``info.rhd`` file rather than
    the folder holding it.
    """

    format: RecordingFormat
    path: Path
    root: Path

    @property
    def key(self) -> str:
        return self.format.key

    @property
    def carries_geometry(self) -> bool:
        return self.format.carries_geometry

    def describe(self) -> str:
        where = self.path.name if self.path != self.root else self.root.name
        return f"{self.format.label} ({where})"


def _intan_file(directory: Path) -> Path | None:
    """The ``.rhd``/``.rhs`` a reader should be pointed at, if this is an Intan set."""
    for stem in ("info", "settings"):
        for suffix in _INTAN_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    # Traditional layout: one self-contained .rhd. Only unambiguous when there is
    # exactly one, otherwise we would be picking a recording on the user's behalf.
    loose = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _INTAN_SUFFIXES
    )
    return loose[0] if len(loose) == 1 else None


def _is_spikeglx(directory: Path) -> bool:
    """Whether this folder (or its ``*_imec*`` children) holds SpikeGLX binaries.

    Requires a ``.meta`` **and** its ``.bin``: a stray meta left behind by a copy is
    not a readable recording, and reporting it as one produces a failure much later
    and much further away.
    """
    for depth in (directory, *(d for d in directory.iterdir() if d.is_dir())):
        try:
            metas = [
                p for p in depth.iterdir()
                if p.suffix.lower() == ".meta"
                and any(t in p.name.lower() for t in _SPIKEGLX_TOKENS)
            ]
        except (PermissionError, OSError):
            continue
        if any(m.with_suffix(".bin").is_file() for m in metas):
            return True
    return False


def _is_open_ephys(directory: Path) -> bool:
    """Whether this folder is an Open Ephys record node.

    Both halves of ``experimentN/recordingM`` must match *exactly*, and prefix
    matching is not good enough for either. The AIND pipeline writes analyzer stores
    named ``experiment1_Record Node 104#...ProbeA_recording1_group0.zarr``, and each
    of those contains a child directory called plainly ``recording`` - so globbing
    ``experiment*`` and accepting any child starting with "recording" claims every
    sorting output as a raw recording.
    """
    if (directory / "structure.oebin").is_file():
        return True
    try:
        entries = [e for e in directory.iterdir() if e.is_dir()]
    except (PermissionError, OSError):
        return False
    for experiment in entries:
        if experiment.name.endswith(".zarr") or not _EXPERIMENT_DIR.match(experiment.name):
            continue
        try:
            children = list(experiment.iterdir())
        except (PermissionError, OSError):
            continue
        if any(c.is_dir() and _RECORDING_DIR.match(c.name) for c in children):
            return True
    return directory.name.lower().startswith("record node")


def detect_format(path: str | Path) -> DetectedRecording | None:
    """Identify the recording at ``path``, or ``None`` if it is not one.

    Accepts either a directory or, for Intan, the ``.rhd``/``.rhs`` file itself.
    Checks the formats with distinctive files (Intan, SpikeGLX) before Open Ephys,
    whose signature is structural and therefore the loosest.
    """
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in _INTAN_SUFFIXES:
            return DetectedRecording(INTAN, p, p.parent)
        if p.suffix.lower() == ".meta" and any(
            t in p.name.lower() for t in _SPIKEGLX_TOKENS
        ):
            return DetectedRecording(SPIKEGLX, p.parent, p.parent)
        return None
    if not p.is_dir():
        return None

    try:
        intan = _intan_file(p)
    except (PermissionError, OSError):
        intan = None
    if intan is not None:
        return DetectedRecording(INTAN, intan, p)

    try:
        if _is_spikeglx(p):
            return DetectedRecording(SPIKEGLX, p, p)
        if _is_open_ephys(p):
            return DetectedRecording(OPEN_EPHYS, p, p)
    except (PermissionError, OSError):
        return None
    return None


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

_INSTALL_HINT = (
    "SpikeInterface is required for ephys alignment. Install the extra:\n"
    '    pip install "histo-to-ccf[ephys]"\n'
    "(or: pip install spikeinterface)"
)


def _require_si():
    try:
        import spikeinterface.full as si
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return si


def list_streams(detected: DetectedRecording) -> list[str]:
    """Stream names in a recording, in the acquisition system's own vocabulary."""
    si = _require_si()
    names, _ids = si.get_neo_streams(detected.format.neo_name, str(detected.path))
    return list(names)


def open_stream(detected: DetectedRecording, stream_name: str | None = None):
    """Open one stream as a lazy SpikeInterface recording."""
    si = _require_si()
    key = detected.key
    if key == OPEN_EPHYS.key:
        if stream_name is None:
            return si.read_openephys(str(detected.path))
        return si.read_openephys(str(detected.path), stream_name=stream_name)
    if key == INTAN.key:
        if stream_name is None:
            stream_name = select_wideband_stream(detected, list_streams(detected))
        return si.read_intan(detected.path, stream_name=stream_name)
    if key == SPIKEGLX.key:
        if stream_name is None:
            return si.read_spikeglx(str(detected.path))
        return si.read_spikeglx(str(detected.path), stream_name=stream_name)
    raise ValueError(f"no reader for format {key!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Stream selection
# ---------------------------------------------------------------------------


def select_lfp_stream(
    detected: DetectedRecording, streams: list[str]
) -> str | None:
    """A dedicated LFP stream, or ``None`` when the format has none.

    Only Neuropixels 1.0 records one: on Open Ephys it is the ``...-LFP`` stream, on
    SpikeGLX the ``imec*.lf`` stream. Intan has a single wideband amplifier stream and
    always returns ``None``, so the caller derives LFP from it.
    """
    if detected.key == INTAN.key:
        return None
    for s in streams:
        u = s.upper()
        if "SYNC" in u:
            continue
        if detected.key == SPIKEGLX.key:
            if ".LF" in u or u.endswith("LF"):
                return s
        elif "LFP" in u or u.endswith("-LF") or u.endswith(".LF"):
            return s
    return None


def select_wideband_stream(
    detected: DetectedRecording, streams: list[str]
) -> str | None:
    """The broadband/spike-band stream LFP can be derived from.

    Deliberately skips the auxiliary and digital streams Intan and SpikeGLX carry
    alongside the electrode data: those have no channel geometry, and averaging a
    digital line into an LFP profile is silent nonsense rather than an error.
    """
    if not streams:
        return None
    if detected.key == INTAN.key:
        for s in streams:
            if "amplifier" in s.lower():
                return s
        # Never fall through to "the first stream": on Intan that is as likely to be
        # the auxiliary or digital-input stream as the electrodes.
        return None
    if detected.key == SPIKEGLX.key:
        # "imec0.ap-SYNC" contains ".ap" but is the single square-wave sync line, and
        # the NI stream carries no electrodes at all. Drop both before matching.
        electrode = [
            s for s in streams
            if "nidq" not in s.lower() and "sync" not in s.lower()
        ]
        for s in electrode:
            if ".ap" in s.lower() or s.lower().endswith("ap"):
                return s
        return electrode[0] if electrode else None
    cands = [s for s in streams if "Neuropix" in s and "AP" in s.upper()]
    if cands:
        return cands[0]
    cands = [s for s in streams if "Neuropix" in s]
    return cands[0] if cands else streams[0]
