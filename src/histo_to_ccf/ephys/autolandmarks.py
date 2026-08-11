"""Proposing landmarks automatically: atlas boundaries matched to feature steps.

The manual workflow is: pick a region boundary that is thick enough to be real, find
the depth where the ephys steps, and pin them together. This does the same thing.

Two parts, deliberately separable so the matching can be judged without the atlas:

* :func:`step_profile` scores, at every depth, how much the features change *across*
  that depth - a level contrast, not a sample-to-sample difference. That distinction
  is the whole reason an earlier attempt at this measured nothing: an anatomical
  transition is spread over 100-300 µm, so its neighbour-to-neighbour steps are
  unremarkable even when the total change is large.
* :func:`propose_landmarks` assigns each atlas boundary the feature depth that scores
  best, **subject to keeping their order**. The ordering constraint is not a detail:
  greedily taking each boundary's best peak routinely produces crossed pairs, which no
  monotonic warp can honour, so the assignment is solved jointly by dynamic
  programming instead.

Pure numpy. Nothing here decides to *apply* anything - it returns proposals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A boundary between two regions thinner than this is not worth pinning to: the atlas
# cannot place it better than the sample grid, and the ephys certainly cannot.
MIN_BAND_UM = 150.0
# How far a boundary may be moved to find its feature. Wider than the ~150-330 µm
# offsets seen on LO_06/LO_07, narrow enough that a boundary cannot be matched to a
# feature belonging to a different structure.
MAX_SHIFT_UM = 180.0
# How far a boundary may deviate from the anchored common shift. Kept small on
# purpose: past the anchor, a large local move is far more likely to be a different
# structure's feature than a real local stretch.
LOCAL_TOLERANCE_UM = 120.0
# Two proposals closer than this describe the same transition; keep the stronger.
MIN_SEPARATION_UM = 120.0
# Penalty per µm of *change* in shift between consecutive boundaries. A real
# misregistration is smooth; without this the matcher happily assigns +239 µm to one
# boundary and -441 µm to the next, which is not a warp anyone would draw.
SMOOTHNESS = 0.004


def estimate_global_shift(
    boundaries: list[tuple[float, str]],
    grid_um,
    score,
    *,
    max_shift_um: float = 700.0,
    step_um: float = 10.0,
) -> tuple[float, float]:
    """The single shift that best lines every boundary up at once: ``(shift, score)``.

    **Anchor first, then refine.** Letting each boundary find its own nearest peak
    fails badly in practice: measured on LO_07 shank 0 it matched 2 of 6 and missed the
    rest by 330-430 µm, because a strong feature ~400 µm away always outbids a modest
    one in the right place, and the search is symmetric so "everything up 400" scores
    as well as "everything down 230". A whole-track shift has to explain *all* the
    boundaries simultaneously, which breaks that symmetry before any local matching
    starts.
    """
    grid = np.asarray(grid_um, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    if grid.size == 0 or not boundaries:
        return 0.0, 0.0
    tracks = np.array([b[0] for b in boundaries], dtype=float)
    shifts = np.arange(-max_shift_um, max_shift_um + step_um, step_um)
    totals = np.array([
        float(np.sum(np.interp(tracks + s, grid, score, left=0.0, right=0.0)))
        for s in shifts
    ])
    best = int(np.argmax(totals))
    n = max(len(boundaries), 1)
    return float(shifts[best]), float(totals[best] / n)


@dataclass(frozen=True)
class LandmarkProposal:
    """One suggested pairing, with the evidence for it."""

    feature_um: float
    track_um: float
    score: float
    label: str = ""

    @property
    def shift_um(self) -> float:
        return self.feature_um - self.track_um


def bad_channel_mask(values, *, k: float = 5.0, neighbours: int = 9) -> np.ndarray:
    """Flag channels whose level is wildly out of line with their neighbours.

    Dead, saturated and reference channels are the single biggest false-positive
    source for step detection: one bad row is a huge, perfectly sharp change in level,
    so the contrast measure scores it far above any real anatomical transition. On
    LO_07 ProbeA a landmark was proposed squarely on one - and the same rows are almost
    certainly the dark band visible across the LFP map.

    Compared against a local median rather than the whole shank, so a genuine
    depth-dependent gradient is not mistaken for a run of bad channels. ``values`` is
    ``(n_depths,)`` or ``(n_depths, n_features)``, already depth-sorted.
    """
    mat = np.asarray(values, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    level = np.nanmean(mat, axis=1)
    n = level.size
    if n < 5:
        return np.zeros(n, dtype=bool)
    half = max(1, int(neighbours) // 2)
    local = np.array([
        np.nanmedian(level[max(0, i - half):min(n, i + half + 1)]) for i in range(n)
    ])
    resid = level - local
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return ~np.isfinite(level)
    return (np.abs(resid) > k * scale) | ~np.isfinite(level)


def _interpolate_bad(depths: np.ndarray, mat: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace flagged rows by interpolating their neighbours, per feature."""
    if not bad.any() or bad.all():
        return mat
    good = ~bad
    out = mat.copy()
    for k in range(mat.shape[1]):
        out[bad, k] = np.interp(depths[bad], depths[good], mat[good, k])
    return out


def step_profile(depths_um, values, *, window_um: float = 250.0,
                 step_um: float = 20.0, reject_bad: bool = True
                 ) -> tuple[np.ndarray, np.ndarray]:
    """How strongly the features step across each depth: ``(grid, score)``.

    The 250 µm window is deliberately wide. A narrower one resolves every local
    wobble in the LFP and the matcher then has dozens of near-equal peaks to choose
    between; at this scale only transitions on the order of a real structure survive,
    which is the scale the atlas boundaries are on too.

    ``values`` is ``(n_depths,)`` or ``(n_depths, n_features)`` - several LFP bands,
    say. Each feature is standardised first so a loud band cannot outvote a quiet one,
    and the per-feature contrasts are averaged.
    """
    from histo_to_ccf.ephys.features import boundary_contrast

    depths = np.asarray(depths_um, dtype=float).ravel()
    mat = np.asarray(values, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    if depths.size < 4 or mat.shape[0] != depths.size:
        return np.empty(0), np.empty(0)

    order = np.argsort(depths)
    depths, mat = depths[order], mat[order]
    if reject_bad:
        # Before anything else: a dead channel is a perfect step, and it is not one.
        mat = _interpolate_bad(depths, mat, bad_channel_mask(mat))
    # Standardise per feature: the contrast is already scale-free, but this keeps a
    # NaN-heavy band from dominating through its variance.
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(mat, axis=0)
        spread[~np.isfinite(spread) | (spread <= 0)] = 1.0
        mat = (mat - np.nanmean(mat, axis=0)) / spread

    grid = np.arange(depths[0] + window_um, depths[-1] - window_um, float(step_um))
    if grid.size == 0:
        return np.empty(0), np.empty(0)
    score = np.zeros(grid.size, dtype=float)
    for i, at in enumerate(grid):
        per_feature = [
            boundary_contrast(depths, mat[:, k], at, window_um=window_um)
            for k in range(mat.shape[1])
        ]
        finite = [v for v in per_feature if np.isfinite(v)]
        score[i] = float(np.mean(finite)) if finite else 0.0
    return grid, score


def candidate_boundaries(bands, *, min_band_um: float = MIN_BAND_UM
                         ) -> list[tuple[float, str]]:
    """Atlas boundaries worth pinning: ``(track_depth, "ABOVE|BELOW")``.

    Only boundaries where **both** neighbours are at least ``min_band_um`` thick. A
    boundary against a 40 µm sliver is a boundary the atlas itself cannot place.
    """
    import itertools

    out: list[tuple[float, str]] = []
    for above, below in itertools.pairwise(bands):
        if above.thickness_um < min_band_um or below.thickness_um < min_band_um:
            continue
        if not above.acronym or not below.acronym:
            continue
        out.append((float(above.bottom_um), f"{above.acronym}|{below.acronym}"))
    return out


def propose_landmarks(
    boundaries: list[tuple[float, str]],
    grid_um,
    score,
    *,
    max_shift_um: float = MAX_SHIFT_UM,
    min_score: float = 0.5,
    min_separation_um: float = LOCAL_TOLERANCE_UM,
    smoothness: float = SMOOTHNESS,
    anchor: bool = True,
) -> list[LandmarkProposal]:
    """Match each atlas boundary to a feature step, keeping the boundaries in order.

    Solved as a shortest path over (boundary, candidate depth): each boundary picks a
    feature depth within ``max_shift_um``, the assignment stays strictly increasing,
    and the total step score is maximised.

    **The score alone is not enough**, and this is what a first attempt got wrong:
    picking each boundary's strongest nearby peak matched 2 of 6 boundaries on LO_07
    shank 0 and missed the rest by 330-430 µm, because a big peak 400 µm away always
    beats a modest one at the right place. A real misregistration is *smooth* - the
    whole track is shifted, and neighbouring boundaries need nearly the same
    correction. ``smoothness`` penalises the change in shift between consecutive
    boundaries, which is what turns the independent peak-picks into one coherent warp.
    Set it to 0 to recover the unconstrained behaviour.
    """
    grid = np.asarray(grid_um, dtype=float).ravel()
    score = np.asarray(score, dtype=float).ravel()
    ordered = sorted(boundaries, key=lambda b: b[0])
    if grid.size == 0 or not ordered:
        return []

    # Anchor the whole track first, then let each boundary deviate only a little from
    # it. ``max_shift_um`` is now a *local* tolerance around the common shift, not a
    # free search range - which is what stops one loud feature dragging a boundary
    # several hundred µm away from where the rest of the track says it belongs.
    if anchor:
        global_shift, _s = estimate_global_shift(ordered, grid, score)
    else:
        global_shift = 0.0

    # Candidate feature depths per boundary, and their scores. A boundary whose
    # window falls off the end of the measured span simply has no candidates - drop
    # that boundary, do not abandon the whole match, which is what an earlier version
    # did and why a single out-of-range boundary produced zero proposals.
    kept: list[tuple[float, str]] = []
    cands: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for track, label in ordered:
        mask = np.abs(grid - (track + global_shift)) <= max_shift_um
        if not mask.any():
            continue
        kept.append((track, label))
        cands.append(grid[mask])
        vals.append(score[mask])
    ordered = kept
    if not ordered:
        return []

    shifts = [c - ordered[i][0] for i, c in enumerate(cands)]
    n = len(ordered)
    best = [np.full(c.size, -np.inf) for c in cands]
    back = [np.full(c.size, -1, dtype=int) for c in cands]
    best[0] = vals[0].copy()
    for i in range(1, n):
        for j, depth in enumerate(cands[i]):
            # Previous boundary must sit strictly above, by at least the separation.
            allowed = np.nonzero(cands[i - 1] <= depth - min_separation_um)[0]
            if allowed.size == 0:
                continue
            # Full sweep, not just the previous best: with a smoothness penalty the
            # optimal predecessor is no longer the highest-scoring one.
            total = (best[i - 1][allowed] + vals[i][j]
                     - smoothness * np.abs(shifts[i][j] - shifts[i - 1][allowed]))
            k = allowed[int(np.argmax(total))]
            if np.isfinite(best[i - 1][k]):
                best[i][j] = float(np.max(total))
                back[i][j] = int(k)

    if not np.isfinite(best[-1]).any():
        return []
    path = [int(np.argmax(best[-1]))]
    for i in range(n - 1, 0, -1):
        prev = back[i][path[-1]]
        if prev < 0:
            return []
        path.append(int(prev))
    path.reverse()

    out = []
    for i, (track, label) in enumerate(ordered):
        j = path[i]
        if vals[i][j] < min_score:
            continue
        out.append(LandmarkProposal(feature_um=float(cands[i][j]), track_um=float(track),
                                    score=float(vals[i][j]), label=label))
    return out
