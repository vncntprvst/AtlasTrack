"""Depth alignment math: warp channel depths onto the histology track.

The histology gives a shank's tip and entry in CCF µm. A channel at *feature
depth* ``d`` (distance from the tip along the shank, as read from the probe
geometry) maps, with no alignment, to the point ``tip + (entry - tip) * d / L``
where ``L`` is the insertion length. Ephys features (LFP power transitions) let
the user pin a feature depth to a *track depth* — an anchor point. The set of
anchors defines a piecewise-linear warp ``feature -> track`` (IBL-style), with
linear extrapolation beyond the extreme anchors so channels above/below the
anchored span still get a sensible position.

This module is pure numpy — no atlas, no Qt — so it is trivially testable.
"""
from __future__ import annotations

import numpy as np

# An anchor pairs a feature depth (µm from tip) with a track depth (µm from tip).
Anchor = tuple[float, float]


def _sorted_unique_anchors(anchors: "list[Anchor]") -> list[Anchor]:
    """Sort anchors by feature depth, dropping duplicates on the feature axis."""
    out: dict[float, float] = {}
    for f, t in anchors:
        out[float(f)] = float(t)
    return sorted(out.items())


def apply_depth_alignment(
    feature_depths: "np.ndarray | list[float]",
    anchors: "list[Anchor]",
) -> np.ndarray:
    """Map feature depths (µm from tip) to track depths (µm from tip).

    With no anchors the mapping is the identity; with a single anchor it is a
    pure shift; with two or more it is piecewise-linear through the anchors with
    linear extrapolation beyond the first/last anchor using the end segments'
    slopes.
    """
    f = np.asarray(feature_depths, dtype=float)
    pairs = _sorted_unique_anchors(anchors)
    if not pairs:
        return f.copy()
    if len(pairs) == 1:
        shift = pairs[0][1] - pairs[0][0]
        return f + shift

    xs = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    out = np.interp(f, xs, ys)  # clamps outside [xs[0], xs[-1]]

    # Linear extrapolation beyond the extreme anchors (np.interp clamps instead).
    left_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    right_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    below = f < xs[0]
    above = f > xs[-1]
    out[below] = ys[0] + left_slope * (f[below] - xs[0])
    out[above] = ys[-1] + right_slope * (f[above] - xs[-1])
    return out


def invert_anchors(anchors: "list[Anchor]") -> list[Anchor]:
    """Swap each anchor's feature/track depth to map track -> feature.

    Used for display: the LFP image lives in feature space but is shown warped
    into track space, which needs the inverse mapping.
    """
    return [(t, f) for f, t in anchors]


def channel_ccf_um(
    tip_ccf_um: "tuple[float, float, float]",
    entry_ccf_um: "tuple[float, float, float]",
    feature_depths: "np.ndarray | list[float]",
    anchors: "list[Anchor]",
) -> np.ndarray:
    """Per-channel CCF ``(AP, ML, DV)`` µm along the tip->entry line.

    ``feature_depths`` are µm from the tip. They are warped by ``anchors`` into
    track depths, then placed on the straight line from ``tip`` toward ``entry``.
    Returns an ``(n_channels, 3)`` array.
    """
    tip = np.asarray(tip_ccf_um, dtype=float)
    entry = np.asarray(entry_ccf_um, dtype=float)
    track = apply_depth_alignment(feature_depths, anchors)

    vec = entry - tip
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        return np.repeat(tip[None, :], len(track), axis=0)
    direction = vec / length
    return tip[None, :] + track[:, None] * direction[None, :]
