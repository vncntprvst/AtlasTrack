"""A shank's track as a polyline, so a curved shank can be represented.

Shanks are flexible: they follow their own trajectories and occasionally curve or
diverge. A straight tip→entry line cannot express that, and the error is not small -
measured on LO_06/LO_07, inner shanks sit 100-155 µm off an evenly spaced row, which
is the same scale as the region boundaries an alignment is trying to hit. A site
half-way down a shank bowing 100 µm is misplaced by about that much, silently.

**Designed around what the histology actually gives you**, which is usually not a
clean set of points per shank:

* **The tip is the reliable anchor.** Arc length is measured *from the tip*, so
  everything is expressed relative to the one point that is normally clear, and
  uncertainty about the far end never shifts sites near the tip.
* **The entry is often guesswork.** It stays the far end of the path but carries no
  special weight beyond that; adding waypoints progressively takes the path away from
  depending on it.
* **Waypoints are optional and additive.** With none, the path is exactly
  ``[tip, entry]`` and every result is identical to the straight-line placement it
  replaces - so "tip + entry" and "tip + a straight trajectory" remain first-class,
  not a degraded mode.
* **Order is derived, not demanded.** Points are sorted by their projection onto the
  tip→entry axis, so they can be picked in any order and from any section.

Points that cannot be attributed to a particular shank do not belong here - they are
probe-level observations (``ProbeSpec.unassigned_track_points_ccf_um``) and are kept
without being forced onto a shank they may not belong to.

Pure numpy.
"""
from __future__ import annotations

import numpy as np


def track_polyline(tip_ccf_um, entry_ccf_um, waypoints=None) -> np.ndarray:
    """Ordered ``(n_points, 3)`` path from the tip to the entry.

    Waypoints are sorted by their projection onto the tip→entry axis and any that
    coincide with a neighbour are dropped, so picks made out of order, or twice, do
    not fold the path back on itself.
    """
    tip = np.asarray(tip_ccf_um, dtype=float).reshape(3)
    entry = np.asarray(entry_ccf_um, dtype=float).reshape(3)
    if waypoints is None or len(waypoints) == 0:
        return np.vstack([tip, entry])

    pts = np.asarray(waypoints, dtype=float).reshape(-1, 3)
    axis = entry - tip
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return np.vstack([tip, entry])
    axis = axis / length
    order = np.argsort((pts - tip) @ axis)
    path = np.vstack([tip, pts[order], entry])

    keep = [0]
    for i in range(1, len(path)):
        if np.linalg.norm(path[i] - path[keep[-1]]) > 1e-6:
            keep.append(i)
    return path[keep]


def arc_lengths(path: np.ndarray) -> np.ndarray:
    """Cumulative distance along the path from the tip, one per point."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return np.zeros(len(path))
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def path_length_um(path: np.ndarray) -> float:
    """Total length along the path - longer than tip-to-entry when it curves."""
    return float(arc_lengths(path)[-1]) if len(path) >= 2 else 0.0


def points_at_distance(path: np.ndarray, distances_um) -> np.ndarray:
    """Positions at given arc lengths from the tip, ``(n, 3)``.

    Distances beyond either end are extrapolated along that end's segment rather than
    clipped: a site sitting past the tip, or above the entry, is a real disagreement
    between the probe geometry and the picked track and must stay visible.
    """
    path = np.asarray(path, dtype=float)
    d = np.atleast_1d(np.asarray(distances_um, dtype=float))
    if len(path) == 1:
        return np.repeat(path[0][None, :], d.size, axis=0)
    s = arc_lengths(path)
    out = np.empty((d.size, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(d, s, path[:, axis])

    below, above = d < s[0], d > s[-1]
    if below.any():
        direction = _segment_unit(path[0], path[1])
        out[below] = path[0][None, :] + (d[below] - s[0])[:, None] * direction[None, :]
    if above.any():
        direction = _segment_unit(path[-2], path[-1])
        out[above] = path[-1][None, :] + (d[above] - s[-1])[:, None] * direction[None, :]
    return out


def tangents_at_distance(path: np.ndarray, distances_um) -> np.ndarray:
    """Unit direction of travel (tip→entry) at each arc length, ``(n, 3)``."""
    path = np.asarray(path, dtype=float)
    d = np.atleast_1d(np.asarray(distances_um, dtype=float))
    if len(path) < 2:
        return np.repeat(np.array([0.0, 0.0, 1.0])[None, :], d.size, axis=0)
    s = arc_lengths(path)
    # Segment index for each distance, clamped so the ends use their own segment.
    idx = np.clip(np.searchsorted(s, d, side="right") - 1, 0, len(path) - 2)
    return np.array([_segment_unit(path[i], path[i + 1]) for i in idx])


def _segment_unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def max_deviation_um(path: np.ndarray) -> float:
    """Largest perpendicular departure of the path from the straight tip→entry line.

    The number to report when deciding whether a curve is worth modelling: it is the
    distance by which a straight-line placement would misplace the worst site.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 3:
        return 0.0
    tip, entry = path[0], path[-1]
    axis = entry - tip
    length = float(np.linalg.norm(axis))
    if length <= 0:
        return float(np.max(np.linalg.norm(path - tip, axis=1)))
    axis = axis / length
    rel = path - tip
    perp = rel - (rel @ axis)[:, None] * axis
    return float(np.max(np.linalg.norm(perp, axis=1)))
