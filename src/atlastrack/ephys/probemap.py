"""Supply channel geometry for recordings that do not carry any.

Open Ephys and SpikeGLX write the probe into the recording, so a channel's position
along the shank is a fact you can read back. **Intan does not.** An ``.rhd`` names its
channels ``A-000 … A-031`` in headstage order and says nothing about where those sites
sit, because the mapping lives in the adapter wiring between probe and headstage.

That mapping is not recoverable from the data, and this module exists because the
alternative - guessing - fails silently. Two candidates were tested against a real
32-channel NeuroNexus Poly3 recording (TJO_optotag_07, 2026-05-11, 5028 µm):

* The ``UserOrder`` permutation in Intan's ``settings.xml`` is a *display* order, not
  a depth order. Sorting by it made the LFP-power depth profile **rougher** than
  native order (70th vs 45th percentile against random permutations).
* LFP correlation carries no geometry here either: it is flat at 0.79-0.83 across
  every channel-index lag, because a Poly3 spans ~275 µm and sits inside one
  coherence length.

So the map is an input, not an inference. Without one, depths stay in channel-index
units and :class:`GeometrySource` records that, so nothing downstream can mistake an
index for a micrometre.
"""
from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class GeometrySource(str, Enum):
    """Where a recording's channel geometry came from.

    ``CHANNEL_INDEX`` is the honest no-geometry case: positions are ordinals, not
    micrometres, and any depth-referenced result computed from them is meaningless.
    Callers must check this rather than assume the numbers are physical.
    """

    RECORDING = "recording"
    PROBE_MAP = "probe_map"
    CATALOG = "catalog"
    CHANNEL_INDEX = "channel_index"

    @property
    def is_physical(self) -> bool:
        return self is not GeometrySource.CHANNEL_INDEX


@dataclass(frozen=True)
class ProbeMap:
    """Per-channel site geometry, in recording-channel order.

    ``depth_um`` is distance along the shank. The sign convention is the one the rest
    of the ephys code uses: larger = further from the tip.
    """

    depth_um: np.ndarray
    x_um: np.ndarray
    shank_ids: np.ndarray | None = None
    name: str = ""
    source: GeometrySource = GeometrySource.PROBE_MAP

    def __post_init__(self) -> None:
        if self.depth_um.shape != self.x_um.shape:
            raise ValueError(
                f"depth_um {self.depth_um.shape} and x_um {self.x_um.shape} differ"
            )
        if self.shank_ids is not None and self.shank_ids.shape != self.depth_um.shape:
            raise ValueError("shank_ids must have one entry per channel")

    @property
    def n_channels(self) -> int:
        return int(self.depth_um.size)

    @property
    def extent_um(self) -> float:
        """Tip-to-top span of the sites. The number that makes a map obviously wrong."""
        if self.depth_um.size == 0:
            return 0.0
        return float(np.ptp(self.depth_um))

    def check_matches(self, n_channels: int) -> None:
        """Refuse a map that does not fit the recording, saying by how much.

        A silently-truncated or recycled map is worse than none: it produces depths
        for every channel, all of them wrong.
        """
        if self.n_channels != n_channels:
            raise ValueError(
                f"probe map {self.name or '<unnamed>'} has {self.n_channels} channels "
                f"but the recording has {n_channels}. A map must cover the recording "
                "exactly - check you picked the map for this probe and headstage."
            )


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

#: Column aliases accepted in a CSV map, lowercased. Depth is required; the rest
#: default (x to 0, shank to a single shank).
_DEPTH_KEYS = ("depth_um", "depth", "y_um", "y")
_X_KEYS = ("x_um", "x", "lateral_um", "lateral")
_SHANK_KEYS = ("shank", "shank_id", "shank_ids", "group")
_CHANNEL_KEYS = ("channel", "channel_id", "chan", "index")


def _pick(row: dict, keys) -> str | None:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return str(row[k]).strip()
    return None


def read_csv_map(path: str | Path) -> ProbeMap:
    """Read a plain CSV channel map.

    One row per channel, with a header. ``depth_um`` (aliases: ``depth``, ``y_um``,
    ``y``) is required; ``x_um`` and ``shank`` are optional. An optional ``channel``
    column sets the row order explicitly - without it, file order is channel order,
    which is the assumption most likely to be silently wrong, so state it when you can.
    """
    p = Path(path)
    with p.open(newline="", encoding="utf-8-sig") as fh:
        rows = [
            {(k or "").strip().lower(): v for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]
    if not rows:
        raise ValueError(f"{p} has no rows")

    depths, xs, shanks, channels = [], [], [], []
    for i, row in enumerate(rows):
        d = _pick(row, _DEPTH_KEYS)
        if d is None:
            raise ValueError(
                f"{p} row {i + 1} has no depth column; expected one of {_DEPTH_KEYS}"
            )
        depths.append(float(d))
        xs.append(float(_pick(row, _X_KEYS) or 0.0))
        s = _pick(row, _SHANK_KEYS)
        shanks.append(None if s is None else int(float(s)))
        c = _pick(row, _CHANNEL_KEYS)
        channels.append(None if c is None else int(float(c)))

    order = np.arange(len(rows))
    if all(c is not None for c in channels):
        order = np.argsort(np.asarray(channels, dtype=int), kind="stable")

    depth = np.asarray(depths, dtype=float)[order]
    x = np.asarray(xs, dtype=float)[order]
    shank = (
        None if any(s is None for s in shanks)
        else np.asarray(shanks, dtype=int)[order]
    )
    return ProbeMap(depth_um=depth, x_um=x, shank_ids=shank, name=p.name)


def read_rhx_probe_map(path: str | Path) -> ProbeMap:
    """Read an Intan RHX *probe map* XML (``<IntanRHX><ProbeMapSettings>``).

    This is the file the rig already uses to draw the probe in RHX, and it is the one
    place the whole signal chain is written down: each ``<ElectrodeSite>`` gives an
    Intan ``channelNumber`` and the ``x``/``y`` of the site it lands on, so probe,
    adapter and headstage wiring are already composed. That makes it the preferred
    map for an Intan recording - nothing has to be assumed about the adapter.

    Each ``<Page>`` is one shank. ``y`` in the file is measured from the lowest site,
    so it is re-referenced to the tip using the shank outline (``<Line>``), keeping
    ``depth_um`` on the same from-tip convention as :func:`map_from_catalog`.
    """
    import xml.etree.ElementTree as ET

    p = Path(path)
    root = ET.parse(p).getroot()
    settings = root if root.tag == "ProbeMapSettings" else root.find("ProbeMapSettings")
    if settings is None:
        hint = (
            " This is an RHX *settings* file; the probe map is the separate "
            "'-probe.xml'."
            if root.tag == "IntanRHX"
            else ""
        )
        raise ValueError(f"{p} has no <ProbeMapSettings> element.{hint}")

    pages = settings.findall("Page") or [settings]
    sites: dict[int, tuple[float, float, int]] = {}
    seen_in: dict[int, str] = {}
    for shank, page in enumerate(pages):
        # The outline says where the silicon ends. Sites are quoted from the lowest
        # site, not from the tip, so the difference is a real offset to undo.
        outline = [
            float(line.get(attr, 0.0))
            for line in page.findall("Line")
            for attr in ("y1", "y2")
        ]
        tip_y = min(outline) if outline else 0.0
        for port in page.findall("Port") or [page]:
            where = f"page {page.get('name', shank)!r} port {port.get('name', '?')!r}"
            for site in port.findall("ElectrodeSite"):
                number = site.get("channelNumber")
                if number is None:
                    raise ValueError(
                        f"{p}: an <ElectrodeSite> in {where} has no channelNumber"
                    )
                channel = int(number)
                if channel in sites:
                    raise ValueError(
                        f"{p}: channel {channel} appears in both {seen_in[channel]} "
                        f"and {where}. A multi-port map cannot be read as one "
                        "recording channel axis without knowing the port order - "
                        "export one port per file."
                    )
                seen_in[channel] = where
                sites[channel] = (
                    float(site.get("x", 0.0)),
                    float(site.get("y", 0.0)) - tip_y,
                    shank,
                )

    if not sites:
        raise ValueError(f"{p} has no <ElectrodeSite> entries")
    # Size the run from the highest channel present: a map missing channel 7 of 32
    # should say so, not claim it was a 31-channel map all along.
    top = max(sites)
    missing = sorted(set(range(top + 1)) - set(sites))
    if missing:
        raise ValueError(
            f"{p} numbers channels up to {top} but is missing {missing[:4]}"
            f"{'...' if len(missing) > 4 else ''}. A partial map cannot be lined up "
            "with the recording's channels."
        )

    order = sorted(sites)
    x = np.array([sites[c][0] for c in order], dtype=float)
    depth = np.array([sites[c][1] for c in order], dtype=float)
    shanks = (
        np.array([sites[c][2] for c in order], dtype=int) if len(pages) > 1 else None
    )
    return ProbeMap(depth_um=depth, x_um=x, shank_ids=shanks, name=p.name)


def _from_probeinterface(probe) -> ProbeMap:
    """Convert a probeinterface ``Probe`` to our per-channel arrays.

    ``device_channel_indices`` is the wiring: it says which recording channel each
    contact is on. Honouring it is the whole point of loading a map for an Intan
    recording, so a probe that declares one is reordered by it rather than trusted to
    already be in channel order.
    """
    positions = np.asarray(probe.contact_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("probe has no 2-D contact positions")
    x, depth = positions[:, 0], positions[:, 1]

    shank_ids = getattr(probe, "shank_ids", None)
    shanks = None
    if shank_ids is not None and len(shank_ids) == depth.size:
        # probeinterface stores shank ids as strings; map them to stable integers.
        labels = [str(s) for s in shank_ids]
        uniq = {name: i for i, name in enumerate(sorted(set(labels)))}
        shanks = np.array([uniq[name] for name in labels], dtype=int)

    device = getattr(probe, "device_channel_indices", None)
    if device is not None and len(device) == depth.size:
        device = np.asarray(device, dtype=int)
        if (device >= 0).all():
            order = np.argsort(device, kind="stable")
            x, depth = x[order], depth[order]
            shanks = None if shanks is None else shanks[order]

    name = getattr(probe, "annotations", {}).get("name", "") or ""
    return ProbeMap(depth_um=depth, x_um=x, shank_ids=shanks, name=str(name))


def load_probe_map(path: str | Path) -> ProbeMap:
    """Load a channel map, dispatching on the file extension.

    Supported: ``.xml`` (an Intan RHX probe map - the best source for an Intan
    recording, see :func:`read_rhx_probe_map`), ``.json`` (probeinterface), ``.prb``,
    ``.imro`` (Neuropixels), and ``.csv`` (documented in :func:`read_csv_map`).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no probe map at {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return read_csv_map(p)
    if suffix == ".xml":
        return read_rhx_probe_map(p)

    try:
        import probeinterface as pi
    except ImportError as exc:  # pragma: no cover - needs the ephys extra
        raise ImportError(
            "probeinterface is required to read .json/.prb/.imro probe maps. "
            'Install the extra: pip install "atlastrack[ephys]"'
        ) from exc

    if suffix == ".json":
        group = pi.read_probeinterface(p)
        probes = list(group.probes)
        if len(probes) != 1:
            raise ValueError(
                f"{p} holds {len(probes)} probes; this expects exactly one. Split the "
                "file, or export the single probe you mean to use."
            )
        return _from_probeinterface(probes[0])
    if suffix == ".prb":
        group = pi.read_prb(p)
        probes = list(group.probes)
        if len(probes) != 1:
            raise ValueError(f"{p} holds {len(probes)} probes; this expects exactly one")
        return _from_probeinterface(probes[0])
    if suffix == ".imro":
        return _from_probeinterface(pi.read_imro(p))
    raise ValueError(
        f"unsupported probe map {p.suffix!r}; expected .xml, .json, .prb, .imro "
        "or .csv"
    )


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def map_from_catalog(probe_name: str) -> ProbeMap:
    """Build a map from a probe model in :mod:`atlastrack.probes.catalog`.

    This covers the *site layout* only. For a recording system that stores no wiring
    - Intan - a catalog layout still assumes the headstage channel order matches the
    site order tip-to-base, which is a property of the adapter, not of the probe. Load
    an explicit map when the adapter reorders channels.
    """
    from atlastrack.probes.catalog import CATALOG

    if probe_name not in CATALOG:
        raise KeyError(
            f"unknown probe {probe_name!r}; catalog has {sorted(CATALOG)} and "
            f"the wired maps are {sorted(BUILTIN_MAPS)}"
        )
    layout = CATALOG[probe_name]
    return ProbeMap(
        depth_um=np.asarray(layout.site_depths_from_tip_um(), dtype=float),
        x_um=np.asarray(layout.site_lateral_offsets_um(), dtype=float),
        shank_ids=None,
        name=layout.name,
        source=GeometrySource.CATALOG,
    )


# ---------------------------------------------------------------------------
# Built-in maps for known probe + adapter + headstage combinations
# ---------------------------------------------------------------------------

#: The NeuroNexus A32-OM32 adapter wired into an Intan RHD2132 headstage: entry *i*
#: is the Intan channel that A32 probe-side pin ``i + 1`` lands on.
#:
#: The probe-side and adapter-side A32 pinouts are *not* the same even though both
#: are NeuroNexus parts, which is why this cannot be reconstructed from the probe
#: alone. Taken from ``A32_Neuronexus_to_RHD2132_Mapping.xlsx`` and cross-checked
#: against the lab's probeinterface notebook and RHX probe-map XML, which agree.
#: probeinterface does not ship this pathway - it has ``H32>RHD2132`` and
#: ``ASSY-116>RHD2132``, but not ``A32>RHD2132`` - so there is nothing upstream to
#: defer to here.
A32_TO_RHD2132: tuple[int, ...] = (
    30, 26, 21, 17, 27, 22, 20, 25, 28, 23, 19, 24, 29, 18, 31, 16,
    0, 15, 2, 13, 8, 9, 7, 1, 6, 14, 10, 11, 5, 12, 4, 3,
)

#: A1x32-Poly3-10mm-25s-177 sites in A32 probe-side pin order, as ``(x, y)`` from the
#: lowest site. Three columns at x = -18 / 0 / +18 µm; the centre column carries 12
#: sites at 25 µm pitch and each side column 10, interleaved by 12.5 µm.
_POLY3_SITES_BY_PIN: tuple[tuple[float, float], ...] = (
    *((-18.0, 12.5 + 25.0 * i) for i in range(10)),   # pins 1-10, left, tip -> base
    *((0.0, 50.0 * i) for i in range(6)),             # pins 11-16, centre, even rows
    *((0.0, 275.0 - 50.0 * i) for i in range(6)),     # pins 17-22, centre, odd rows
    *((18.0, 237.5 - 25.0 * i) for i in range(10)),   # pins 23-32, right, base -> tip
)

#: Tip-to-lowest-site distance for the 10 mm Poly3 shank, from the outline in the
#: RHX probe map (the shank tapers to a point 62 µm below the deepest site).
_POLY3_TIP_TO_LOWEST_SITE_UM = 62.0

NEURONEXUS_POLY3_A32_RHD2132 = "NeuroNexus A1x32-Poly3 + A32>RHD2132"


def _poly3_a32_rhd2132() -> ProbeMap:
    """The lab's acute Intan rig: Poly3 -> A32-OM32 adapter -> RHD2132 headstage.

    Equivalent to loading the RHX probe map for this probe, and kept as a named map
    so a recording can be set up without hunting for that file. Prefer the file when
    the rig changes: it is generated from the wiring, this is a copy of it.
    """
    depth = np.zeros(len(A32_TO_RHD2132), dtype=float)
    x = np.zeros_like(depth)
    for pin, channel in enumerate(A32_TO_RHD2132):
        site_x, site_y = _POLY3_SITES_BY_PIN[pin]
        x[channel] = site_x
        depth[channel] = site_y + _POLY3_TIP_TO_LOWEST_SITE_UM
    return ProbeMap(depth_um=depth, x_um=x, name=NEURONEXUS_POLY3_A32_RHD2132)


#: Named maps that already include the adapter wiring, unlike :func:`map_from_catalog`.
BUILTIN_MAPS: dict[str, Callable[[], ProbeMap]] = {
    NEURONEXUS_POLY3_A32_RHD2132: _poly3_a32_rhd2132,
}


def resolve_probe_map(
    probe_map: str | Path | ProbeMap | None,
    *,
    n_channels: int | None = None,
) -> ProbeMap | None:
    """Normalise the ``probe_map`` argument callers pass around.

    Accepts a :class:`ProbeMap`, the name of a map in :data:`BUILTIN_MAPS`, a path to
    a map file, or the name of a catalog probe.
    Returns ``None`` for ``None`` - the caller decides whether missing geometry is
    fatal, because it is for a depth-referenced feature and harmless for a raw trace.
    """
    if probe_map is None:
        return None
    if isinstance(probe_map, ProbeMap):
        resolved = probe_map
    elif str(probe_map) in BUILTIN_MAPS:
        resolved = BUILTIN_MAPS[str(probe_map)]()
    else:
        text = str(probe_map)
        candidate = Path(text)
        if candidate.suffix and (candidate.is_file() or candidate.suffix == ".csv"):
            resolved = load_probe_map(candidate)
        else:
            resolved = map_from_catalog(text)
    if n_channels is not None:
        resolved.check_matches(n_channels)
    return resolved
