"""Landmark alignment between an ephys feature axis and the histology track.

This is the IBL alignment model, reimplemented against the reference source
(``ephys_alignment_gui/ephys_alignment.py``) so that our results interoperate with
theirs.

Two parallel arrays describe the alignment: ``feature_um[i]`` is a depth on the
**ephys feature axis** (where a transition is seen in the LFP/spike panels) and
``track_um[i]`` is the depth on the **histology track** it belongs to. Between
landmarks the map is piecewise linear; beyond the outermost pair it continues on
the end segment's slope.

Following IBL, the arrays carry **two extra points beyond the user's landmarks**,
one at each end, holding the track extent. They are not user landmarks; their only
job is to define what happens in the tails, and the two ``adjust_extremes_*``
functions set them. ``n_user`` is therefore ``len(feature_um) - 2``.

Both axes here are **depth below the brain surface, increasing downwards** - the
axis :mod:`histo_to_ccf.ephys.penetration` puts every recording on. That differs
from the older :mod:`histo_to_ccf.ephys.alignment` anchors, which are µm *from the
tip*; the maths is identical either way, but mixing the two silently flips the
track, so keep a whole alignment in one convention.

Where we deviate from IBL, deliberately:

* **Crossed landmarks are refused, not silently re-paired.** IBL sorts the feature
  and track arrays independently, so dragging one landmark past its neighbour
  re-pairs them without complaint and the fit quietly means something else.
  :func:`check_monotonic` detects the crossing and names the pair.
* **The linear-extremes mode falls back to uniform below 3 landmarks** instead of
  collapsing the tails to zero, which is what IBL's ``lin_fit = 0`` branch does.

Pure numpy - no atlas, no Qt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# How far past the track ends the outer reference points are pushed before the
# tails are fitted on the global regression. IBL uses 1 (metre) on a track a few
# millimetres long, i.e. "far enough out that the end segment carries the global
# slope and nothing else". 1e6 µm is the same distance.
EXTEND_UM = 1.0e6

# Below this many user landmarks a global regression through the interior is not
# meaningful, so ``linear`` extremes degrade to ``uniform``. IBL's equivalent test
# is ``feature.size >= 5`` on the array that includes the two end points.
MIN_LANDMARKS_FOR_LINEAR = 3

ExtremesMode = str  # "uniform" | "linear" | "none"


class LandmarkCrossingError(ValueError):
    """Two landmarks disagree about their order in feature space vs track space.

    Raised instead of silently re-pairing them, which is what an independent sort
    of the two arrays would do.
    """


# -- the map ---------------------------------------------------------------


def _slope(x0: float, y0: float, x1: float, y1: float) -> float:
    """Slope of the segment, or 1.0 where the two x values coincide."""
    dx = x1 - x0
    if dx == 0.0:
        return 1.0
    return (y1 - y0) / dx


def _interp_extrap(x, xp, fp) -> np.ndarray:
    """Piecewise-linear interpolation with linear extrapolation past both ends.

    Equivalent to ``scipy.interpolate.interp1d(xp, fp, fill_value="extrapolate")``
    for a linear fit, which is what IBL uses, without the scipy call.
    """
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    x_arr = np.atleast_1d(x_arr)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    if xp.size == 0:
        out = x_arr.copy()
    elif xp.size == 1:
        out = x_arr + (fp[0] - xp[0])
    else:
        order = np.argsort(xp, kind="stable")
        xs, ys = xp[order], fp[order]
        out = np.interp(x_arr, xs, ys)  # clamps outside [xs[0], xs[-1]]
        below = x_arr < xs[0]
        above = x_arr > xs[-1]
        if below.any():
            m = _slope(xs[0], ys[0], xs[1], ys[1])
            out[below] = ys[0] + m * (x_arr[below] - xs[0])
        if above.any():
            m = _slope(xs[-2], ys[-2], xs[-1], ys[-1])
            out[above] = ys[-1] + m * (x_arr[above] - xs[-1])
    return float(out[0]) if scalar else out


def feature2track(feature_new, feature_ref, track_ref) -> np.ndarray:
    """Map depths on the ephys feature axis onto the histology track."""
    return _interp_extrap(feature_new, feature_ref, track_ref)


def track2feature(track_new, feature_ref, track_ref) -> np.ndarray:
    """Map depths on the histology track onto the ephys feature axis.

    This is the direction the region column is drawn through: the ephys panels stay
    put and the anatomy stretches against them.
    """
    return _interp_extrap(track_new, track_ref, feature_ref)


def segment_scales(feature_ref, track_ref) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment stretch factor ``d(track)/d(feature)``, with segment edges.

    A scale far from 1 means that stretch of track was compressed or expanded a lot
    to make the features line up, which is the thing to be sceptical about. Returns
    ``(edges_feature_um, scale)`` where ``scale`` has one fewer element than
    ``edges``.
    """
    f = np.asarray(feature_ref, dtype=float)
    t = np.asarray(track_ref, dtype=float)
    order = np.argsort(f, kind="stable")
    f, t = f[order], t[order]
    if f.size < 2:
        return f, np.empty(0)
    df = np.diff(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(df != 0, np.diff(t) / df, np.nan)
    return f, scale


# -- the tails -------------------------------------------------------------


def adjust_extremes_uniform(feature, track) -> tuple[np.ndarray, np.ndarray]:
    """Force the two outer segments to slope 1, i.e. pure translation in the tails.

    Depths beyond the outermost landmark are then shifted by exactly the offset that
    landmark carries, and never stretched. This is IBL's default and ours: outside
    what the user actually pinned, claiming a scale change is claiming evidence that
    is not there.

    Operates on the arrays including the two end points, and returns copies.
    """
    feature = np.asarray(feature, dtype=float).copy()
    track = np.asarray(track, dtype=float).copy()
    if feature.size < 2:
        return feature, track
    diff = np.diff(feature - track)
    track[0] -= diff[0]
    track[-1] += diff[-1]
    return feature, track


def adjust_extremes_linear(feature, track, *, extend_um: float = EXTEND_UM
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Continue the tails on the global regression through the interior landmarks.

    Use when the whole track is believed to be scaled (shrinkage, or a wrong
    insertion depth) rather than merely shifted. Needs at least
    ``MIN_LANDMARKS_FOR_LINEAR`` user landmarks for the regression to mean anything;
    below that it falls back to :func:`adjust_extremes_uniform` rather than IBL's
    ``lin_fit = 0``, which sends both tail points to zero.

    ``track[0]`` / ``track[-1]`` must still hold the track extent on entry (they do
    for a :class:`Landmarks` state, which never stores adjusted arrays).
    """
    feature = np.asarray(feature, dtype=float).copy()
    track = np.asarray(track, dtype=float).copy()
    if feature.size < MIN_LANDMARKS_FOR_LINEAR + 2:
        return adjust_extremes_uniform(feature, track)
    slope, intercept = np.polyfit(feature[1:-1], track[1:-1], 1)
    feature[0] = track[0] - extend_um
    feature[-1] = track[-1] + extend_um
    track[0] = slope * feature[0] + intercept
    track[-1] = slope * feature[-1] + intercept
    return feature, track


def check_monotonic(feature, track) -> None:
    """Raise :class:`LandmarkCrossingError` if the landmarks are not consistently ordered.

    Sorting by feature depth must leave the track depths ascending too. If it does
    not, two landmarks have crossed: the user has said that a deeper feature belongs
    to a shallower piece of track than a shallower feature does, which no monotonic
    warp can honour.
    """
    f = np.asarray(feature, dtype=float)
    t = np.asarray(track, dtype=float)
    if f.size < 2:
        return
    order = np.argsort(f, kind="stable")
    fs, ts = f[order], t[order]
    bad_feature = np.nonzero(np.diff(fs) <= 0)[0]
    if bad_feature.size:
        i = int(bad_feature[0])
        raise LandmarkCrossingError(
            f"two landmarks share the feature depth {fs[i]:.1f} µm "
            f"(track {ts[i]:.1f} and {ts[i + 1]:.1f} µm) - move one of them"
        )
    bad_track = np.nonzero(np.diff(ts) <= 0)[0]
    if bad_track.size:
        i = int(bad_track[0])
        raise LandmarkCrossingError(
            f"landmarks crossed: feature {fs[i]:.1f} µm -> track {ts[i]:.1f} µm but the "
            f"deeper feature {fs[i + 1]:.1f} µm -> shallower track {ts[i + 1]:.1f} µm"
        )


# -- state -----------------------------------------------------------------


@dataclass(frozen=True)
class Landmarks:
    """A landmark set: the user's pairs plus the two track-extent end points.

    Immutable - every edit returns a new instance, which is what makes the undo
    history a list of these and nothing more. The stored arrays are always the
    *unadjusted* ones (``track_um[0]`` and ``track_um[-1]`` hold the track extent);
    the tails are adjusted only when :meth:`fit` is called, so repeated fitting can
    never drift.
    """

    feature_um: np.ndarray
    track_um: np.ndarray

    def __post_init__(self) -> None:
        f = np.asarray(self.feature_um, dtype=float).ravel()
        t = np.asarray(self.track_um, dtype=float).ravel()
        if f.size != t.size:
            raise ValueError(f"feature/track length mismatch: {f.size} vs {t.size}")
        if f.size < 2:
            raise ValueError("a landmark set needs at least the two track end points")
        object.__setattr__(self, "feature_um", f)
        object.__setattr__(self, "track_um", t)

    # -- construction --

    @classmethod
    def identity(cls, top_um: float, bottom_um: float) -> Landmarks:
        """No user landmarks: the feature axis and the track are the same thing."""
        if not bottom_um > top_um:
            raise ValueError(f"track extent must be top < bottom, got {top_um}, {bottom_um}")
        ends = np.array([float(top_um), float(bottom_um)])
        return cls(ends.copy(), ends.copy())

    @property
    def n_user(self) -> int:
        """How many landmarks the user actually placed."""
        return int(self.feature_um.size - 2)

    @property
    def track_extent_um(self) -> tuple[float, float]:
        return (float(self.track_um[0]), float(self.track_um[-1]))

    def user_pairs(self) -> list[tuple[float, float]]:
        """The user's ``(feature_um, track_um)`` pairs, shallowest first."""
        return [
            (float(f), float(t))
            for f, t in zip(self.feature_um[1:-1], self.track_um[1:-1], strict=True)
        ]

    # -- edits --

    def added(self, feature_um: float, track_um: float) -> Landmarks:
        """A copy with one more landmark, inserted in feature order."""
        pairs = [*self.user_pairs(), (float(feature_um), float(track_um))]
        return self._rebuilt(pairs)

    def removed(self, index: int) -> Landmarks:
        """A copy without user landmark ``index`` (0 = shallowest)."""
        pairs = self.user_pairs()
        if not 0 <= index < len(pairs):
            raise IndexError(f"no user landmark {index} (have {len(pairs)})")
        del pairs[index]
        return self._rebuilt(pairs)

    def moved(self, index: int, *, feature_um: float | None = None,
              track_um: float | None = None) -> Landmarks:
        """A copy with user landmark ``index`` moved on one or both axes.

        Dragging a line in the (feature-space) display moves ``feature_um`` and
        leaves ``track_um`` alone: the anatomy pinned there follows the line.
        """
        pairs = self.user_pairs()
        if not 0 <= index < len(pairs):
            raise IndexError(f"no user landmark {index} (have {len(pairs)})")
        f, t = pairs[index]
        pairs[index] = (
            f if feature_um is None else float(feature_um),
            t if track_um is None else float(track_um),
        )
        return self._rebuilt(pairs)

    def cleared(self) -> Landmarks:
        """A copy with every user landmark removed (back to the identity map)."""
        return self._rebuilt([])

    def _rebuilt(self, pairs: list[tuple[float, float]]) -> Landmarks:
        top, bottom = self.track_extent_um
        pairs = sorted(pairs, key=lambda p: p[0])
        feature = np.array([top, *[p[0] for p in pairs], bottom], dtype=float)
        track = np.array([top, *[p[1] for p in pairs], bottom], dtype=float)
        check_monotonic(feature[1:-1], track[1:-1])
        return Landmarks(feature, track)

    # -- use --

    def fit(self, mode: ExtremesMode = "uniform") -> tuple[np.ndarray, np.ndarray]:
        """The ``(feature, track)`` arrays to map through, tails adjusted per ``mode``."""
        feature = self.feature_um.copy()
        track = self.track_um.copy()
        if mode == "none" or self.n_user == 0:
            return feature, track
        if mode == "uniform":
            return adjust_extremes_uniform(feature, track)
        if mode == "linear":
            return adjust_extremes_linear(feature, track)
        raise ValueError(f"unknown extremes mode {mode!r}; use uniform, linear or none")

    def to_track(self, feature_depths, mode: ExtremesMode = "uniform") -> np.ndarray:
        f, t = self.fit(mode)
        return feature2track(feature_depths, f, t)

    def to_feature(self, track_depths, mode: ExtremesMode = "uniform") -> np.ndarray:
        f, t = self.fit(mode)
        return track2feature(track_depths, f, t)

    def offset_um(self, mode: ExtremesMode = "uniform") -> float:
        """Mean track-minus-feature shift over the user landmarks (0 with none).

        This is the along-track offset the alignment implies, which Phase 3 turns
        into a trajectory correction.
        """
        if self.n_user == 0:
            return 0.0
        return float(np.mean(self.track_um[1:-1] - self.feature_um[1:-1]))

    def shifted(self, delta_um: float) -> Landmarks:
        """A copy with every landmark's track depth moved by ``delta_um``.

        A whole-track nudge for when the features are right relative to each other
        but the whole penetration sits too deep or too shallow.
        """
        if self.n_user == 0:
            # Nothing pinned: shift the map itself by pinning the two ends.
            top, bottom = self.track_extent_um
            return Landmarks(
                np.array([top, bottom]), np.array([top + delta_um, bottom + delta_um])
            )
        pairs = [(f, t + float(delta_um)) for f, t in self.user_pairs()]
        return self._rebuilt(pairs)


class AlignmentHistory:
    """Bounded undo/redo over landmark states, 10 deep as in the IBL GUI.

    Unlike IBL's ring, stepping back past the oldest kept state stops there rather
    than wrapping round to the newest - a wrap looks like an undo but is a jump
    forward, and there is no way to tell from the display which happened.
    """

    def __init__(self, initial: Landmarks, *, depth: int = 10) -> None:
        if depth < 1:
            raise ValueError("history depth must be >= 1")
        self._depth = int(depth)
        self._states: list[Landmarks] = [initial]
        self._cursor = 0

    def push(self, state: Landmarks) -> None:
        """Record a new state, discarding anything the cursor had stepped back past."""
        del self._states[self._cursor + 1 :]
        self._states.append(state)
        if len(self._states) > self._depth:
            self._states = self._states[-self._depth :]
        self._cursor = len(self._states) - 1

    def current(self) -> Landmarks:
        return self._states[self._cursor]

    def previous(self) -> Landmarks | None:
        """Step back one state, or ``None`` if already at the oldest kept."""
        if self._cursor == 0:
            return None
        self._cursor -= 1
        return self.current()

    def next(self) -> Landmarks | None:
        """Step forward one state, or ``None`` if already at the newest."""
        if self._cursor >= len(self._states) - 1:
            return None
        self._cursor += 1
        return self.current()

    def reset(self, state: Landmarks) -> None:
        """Drop the whole history and start again from ``state``."""
        self._states = [state]
        self._cursor = 0

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._states) - 1

    @property
    def n_states(self) -> int:
        return len(self._states)
