"""Plain-number plane values per section, and checks across the series.

A section's plane is stored as a QuickNII-style anchoring: nine numbers in atlas
voxels that are hard to read. :func:`plane_values` turns one into the numbers a
person thinks in - where the section is front to back, how the cut is tilted, how
the image is turned and scaled. :func:`series_rows` lists them for every section
in order and :func:`series_flags` points out the ones that break the pattern.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atlastrack.project.schema import Project, Section


@dataclass
class PlaneValues:
    """A section plane in readable numbers.

    ``ap_um``: AP (µm from the atlas front) at the image centre. ``pitch_deg``:
    tilt of the cut about the ML axis - positive means deeper parts of the section
    lie further back. ``yaw_deg``: tilt about the DV axis - positive means the
    animal's left side lies further back. ``rotation_deg``: how far the image's
    x axis is turned from the plane's ML direction (positive towards ventral).
    ``scale_x/y_um_per_px``: atlas µm per image pixel along x and y.
    ``mirrored``: the left hemisphere is on the image's left.
    """

    ap_um: float
    pitch_deg: float
    yaw_deg: float
    rotation_deg: float
    scale_x_um_per_px: float
    scale_y_um_per_px: float
    mirrored: bool

    def as_dict(self, digits: int = 2) -> dict:
        return {k: (round(v, digits) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def plane_values(
    anchoring: list[float], output_size_px: tuple[int, int], resolution_um: float
) -> PlaneValues:
    """Readable values for an anchoring ``(AP, DV, ML)`` over an ``(h, w)`` image."""
    a = np.asarray(anchoring, dtype=float) * resolution_um
    o, u, v = a[:3], a[3:6], a[6:]
    h, w = output_size_px
    ax, ay = u / max(w, 1), v / max(h, 1)  # µm per pixel along image x and y
    n = np.cross(ax, ay)
    if n[0] < 0:
        n = -n
    centre = o + u / 2 + v / 2
    # On the plane n . d = 0, so dAP/dDV = -n_dv / n_ap and dAP/dML = -n_ml / n_ap.
    pitch = math.degrees(math.atan2(-n[1], n[0]))
    yaw = math.degrees(math.atan2(-n[2], n[0]))
    return PlaneValues(
        ap_um=float(centre[0]),
        pitch_deg=pitch,
        yaw_deg=yaw,
        # Measured from whichever ML direction the image x axis runs along, so a
        # mirrored plane reads as turned, not as turned by ~180°.
        rotation_deg=math.degrees(math.atan2(ax[1], abs(ax[2]))),
        scale_x_um_per_px=float(np.linalg.norm(ax)),
        scale_y_um_per_px=float(np.linalg.norm(ay)),
        mirrored=bool(u[2] < 0),
    )


def section_plane_values(section: Section, resolution_um: float) -> PlaneValues | None:
    """Plane values from a section's registration, else its DeepSlice placement."""
    x0, y0, x1, y1 = section.bbox_px
    if section.registration is not None:
        return plane_values(
            section.registration.anchoring, section.registration.output_size_px, resolution_um
        )
    if section.deepslice_anchoring is not None:
        return plane_values(section.deepslice_anchoring, (y1 - y0, x1 - x0), resolution_um)
    return None


@dataclass
class SeriesRow:
    index: int
    label: str
    values: PlaneValues | None
    # AP the user set (or even spacing gave), when there is no plane yet.
    assigned_ap_um: float | None = None

    @property
    def ap_um(self) -> float | None:
        return self.values.ap_um if self.values is not None else self.assigned_ap_um


def series_rows(project: Project) -> list[SeriesRow]:
    """One row per section, in the series (AP) order."""
    res = float(project.atlas.resolution_um)
    sections = [s for slide in project.slides for s in slide.sections]
    sections.sort(key=lambda s: (s.ap_order, s.index))
    rows = []
    for s in sections:
        label = str(s.slide_number) if s.slide_number is not None else str(s.index)
        rows.append(SeriesRow(
            index=s.index,
            label=label,
            values=section_plane_values(s, res),
            assigned_ap_um=s.plane.ap_um if s.plane is not None else None,
        ))
    return rows


def series_flags(
    rows: list[SeriesRow], *, tilt_tolerance_deg: float = 2.0
) -> dict[int, list[str]]:
    """Sections that break the pattern of the series, with the reason in words.

    Flags an AP step against the direction of the series, and a pitch or yaw more
    than ``tilt_tolerance_deg`` from the series median - all sections of one block
    are cut at the same angle, so one that differs is more likely misplaced.
    """
    flags: dict[int, list[str]] = {}
    aps = [(r.index, r.ap_um) for r in rows if r.ap_um is not None]
    if len(aps) >= 3:
        # The series' trend: its median step, through the median offset. Of two
        # sections out of order, the one further from the trend is the odd one.
        values = np.array([ap for _, ap in aps], dtype=float)
        pos = np.arange(len(values))
        step = float(np.median(np.diff(values)))
        off_trend = np.abs(values - (np.median(values - step * pos) + step * pos))
        direction = 1.0 if values[-1] >= values[0] else -1.0
        for k, ((_, a), (_, b)) in enumerate(pairwise(aps)):
            if (b - a) * direction <= 0:
                odd = k if off_trend[k] > off_trend[k + 1] else k + 1
                flags.setdefault(aps[odd][0], []).append("AP out of order in the series")
    planes = [r for r in rows if r.values is not None]
    if len(planes) >= 3:
        for name in ("pitch_deg", "yaw_deg"):
            med = float(np.median([getattr(r.values, name) for r in planes]))
            for r in planes:
                if abs(getattr(r.values, name) - med) > tilt_tolerance_deg:
                    word = name.split("_")[0]
                    flags.setdefault(r.index, []).append(
                        f"{word} {getattr(r.values, name):+.1f}° vs {med:+.1f}° for the series"
                    )
    for r in rows:
        if r.values is not None and r.values.mirrored:
            flags.setdefault(r.index, []).append("left hemisphere on the image's left")
    return flags
