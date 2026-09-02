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

import warnings
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


# --------------------------------------------------------------- multi-scale steps

#: Window widths tried when looking for a step, in µm.
#:
#: A single width cannot work, and this is measured rather than assumed. On LO_07
#: ProbeA shank 1 the two atlas boundaries in range are found at *different* scales:
#: ``MY|PGRNd`` (4245 µm) peaks at an 80 µm window and is hit to within 2 µm, while
#: ``PGRNd|GRN`` (4590 µm) only emerges at ~180 µm. The old fixed 250 µm default
#: missed both by 132-158 µm - it was tuned on the 5745 µm single-column recording and
#: had never been run against a 705 µm bank.
WINDOW_LADDER_UM: tuple[float, ...] = (60.0, 100.0, 150.0, 220.0)

#: The widest window worth using, as a fraction of the recorded extent.
#:
#: :func:`step_profile` can only score depths at least one window from either end, so
#: a 250 µm window on a 705 µm bank leaves 205 µm of usable grid - and the answer it
#: gives there is dominated by the edges. A third of the extent keeps most of the
#: coverage scorable at every scale used.
MAX_WINDOW_FRACTION = 1.0 / 3.0

#: Fallback floor when no null has been computed. A step profile is rough everywhere,
#: so the question is never "is there a peak" but "is it more than this signal
#: produces by itself" - which :func:`null_threshold` answers properly.
MIN_BOUNDARY_Z = 2.0

#: Surrogate datasets used to calibrate the threshold, and the percentile taken.
#: 16 is enough to place a 90th percentile without the calibration costing more than
#: the detection it guards.
N_SURROGATES = 16
NULL_PERCENTILE = 90.0
#: Depth step for the surrogate profiles. Coarser than the real one on purpose: the
#: null only needs the distribution of contrast values and the height of its peaks,
#: neither of which needs 5 µm resolution, and the fine grid made calibrating the
#: 5745 µm column take 28 s.
NULL_STEP_UM = 20.0


def adaptive_windows(extent_um: float, *, ladder=WINDOW_LADDER_UM,
                     max_fraction: float = MAX_WINDOW_FRACTION) -> tuple[float, ...]:
    """Which windows of ``ladder`` are usable over a recording of this extent.

    Always returns at least the narrowest, even on a very short recording: a bad
    estimate the caller can see is better than an empty profile that reads as "no
    boundaries here".
    """
    extent = float(extent_um)
    usable = tuple(w for w in ladder if w <= extent * float(max_fraction))
    return usable or (min(ladder),)


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Deviations from the median in robust SDs; zeros when there is no spread."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    centre = float(np.median(v[finite]))
    mad = float(np.median(np.abs(v[finite] - centre)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        spread = float(np.std(v[finite]))
        scale = spread if spread > 0 else 1.0
    out = np.full_like(v, np.nan)
    out[finite] = (v[finite] - centre) / scale
    return out


def _phase_randomised(mat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A surrogate with the same depth-smoothness but no localised steps.

    **Why a surrogate and not plain noise.** The threshold has to answer "is this step
    bigger than what this signal produces on its own", and that depends on how smooth
    the signal is: measured on a 705 µm bank, white noise alone reaches a contrast of
    d = 2.9-4.1 while the real detected boundaries sit at d = 2.4-7.6. The two overlap,
    so no fixed number separates them and the threshold must be calibrated per signal.

    Randomising the Fourier phases keeps each feature's power spectrum - hence its
    autocorrelation, hence its roughness - and destroys the localisation that makes a
    step a step. Shuffling the depths instead would whiten the signal and set the bar
    far too low; rotating it would carry the step along and set it too high.
    """
    out = np.empty_like(mat, dtype=float)
    n = mat.shape[0]
    for k in range(mat.shape[1]):
        col = np.asarray(mat[:, k], dtype=float)
        finite = np.isfinite(col)
        if not finite.all():
            col = np.interp(np.arange(n), np.flatnonzero(finite), col[finite]) \
                if finite.any() else np.zeros(n)
        spectrum = np.fft.rfft(col)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.size)
        phases[0] = 0.0
        if n % 2 == 0:
            phases[-1] = 0.0
        out[:, k] = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=n)
    return out


def calibrate_scales(
    depths_um, values, windows, *, step_um: float = NULL_STEP_UM,
    n_surrogates: int = N_SURROGATES, percentile: float = NULL_PERCENTILE, seed: int = 0,
) -> tuple[dict, float]:
    """Per-window ``(median, spread)`` of the null contrast, and the peak it reaches.

    Returns ``({window: (median, spread)}, null_z)``. The stats turn a raw Cohen's d
    into "how unusual is this for *this* signal at *this* scale", which is what makes
    scales comparable; ``null_z`` is the ``percentile``-th surrogate peak on that same
    calibrated axis, i.e. the height to beat.

    **Calibrating against the profile's own spread instead does the opposite of what
    is wanted, and it was tried first.** A surrogate is featureless, so its profile has
    a small spread and its largest wobble scores a big self-referential z; real data is
    full of structure, so its spread is large and a genuine boundary scores a modest
    one. Measured on LO_07 ProbeA the self-normalised threshold came out at 5.0-5.3
    while the real peaks sat at 4.3-4.5 - it rejected every true boundary and still
    passed 2 of 10 pure-noise traces. Against the surrogate distribution the same
    shanks give real peaks of 4.4-19.2 against surrogate maxima of 3.4-4.7.
    """
    depths = np.asarray(depths_um, dtype=float).ravel()
    mat = np.asarray(values, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    if depths.size < 8 or n_surrogates < 1 or not windows:
        return {}, float("nan")

    rng = np.random.default_rng(seed)
    per_surrogate: list[dict] = []
    for _ in range(int(n_surrogates)):
        surrogate = _phase_randomised(mat, rng)
        one: dict[float, tuple] = {}
        for w in windows:
            g, d = step_profile(depths, surrogate, window_um=float(w),
                                step_um=float(step_um), reject_bad=False)
            if g.size:
                one[float(w)] = (g, d)
        if one:
            per_surrogate.append(one)
    if not per_surrogate:
        return {}, float("nan")

    stats: dict[float, tuple[float, float]] = {}
    for w in {k for one in per_surrogate for k in one}:
        pool = np.concatenate([one[w][1] for one in per_surrogate if w in one])
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            continue
        centre = float(np.median(pool))
        spread = 1.4826 * float(np.median(np.abs(pool - centre)))
        stats[w] = (centre, spread if spread > 0 else 1.0)

    # The same surrogates, scored on the axis they just defined: what the null peaks at.
    peaks: list[float] = []
    for one in per_surrogate:
        best = -np.inf
        for w, (_g, d) in one.items():
            if w not in stats:
                continue
            centre, spread = stats[w]
            z = (d - centre) / spread
            if np.isfinite(z).any():
                best = max(best, float(np.nanmax(z)))
        if np.isfinite(best):
            peaks.append(best)
    null_z = float(np.percentile(peaks, float(percentile))) if peaks else float("nan")
    return stats, null_z


@dataclass(frozen=True)
class MultiscaleProfile:
    """Step evidence at every depth, and the scale that supplied it.

    ``score`` is a **robust z per scale, maximised over scales** - not an average.
    Averaging is what a single wide window already does, and it is why a boundary that
    is sharp at 80 µm and invisible at 220 µm came out mediocre at both: the scales
    disagree because the structures differ in size, so the right combination keeps the
    best evidence rather than diluting it. ``scale_um`` records which window won at
    each depth, which is itself informative - a boundary found only at 220 µm is a
    gradual transition, and should not be pinned as precisely as one found at 60.
    """

    grid_um: np.ndarray
    score: np.ndarray
    scale_um: np.ndarray
    per_window: dict = None  # {window_um: z on grid_um}
    #: What this signal's own roughness reaches, from :func:`null_threshold`.
    #: NaN when no null was computed.
    null_z: float = float("nan")

    @property
    def windows_um(self) -> tuple[float, ...]:
        return tuple(sorted(self.per_window or {}))


def multiscale_step_profile(
    depths_um, values, *, windows=None, step_um: float = 5.0, reject_bad: bool = True,
    n_surrogates: int = N_SURROGATES, seed: int = 0,
) -> MultiscaleProfile:
    """:func:`step_profile` at several window widths, combined by taking the best.

    ``windows`` defaults to :func:`adaptive_windows` for the extent of ``depths_um``,
    so a 705 µm bank and a 5745 µm column each get scales they can actually support
    instead of one number chosen for the longer of the two.

    ``n_surrogates`` calibrates :attr:`MultiscaleProfile.null_z` so the caller can tell
    a real step from this signal's own roughness; pass 0 to skip it when the threshold
    is being supplied some other way.
    """
    depths = np.asarray(depths_um, dtype=float).ravel()
    if depths.size < 4:
        return MultiscaleProfile(np.empty(0), np.empty(0), np.empty(0), {})
    extent = float(depths.max() - depths.min())
    chosen = tuple(adaptive_windows(extent) if windows is None else windows)
    stats, null = (
        calibrate_scales(depths, values, chosen, n_surrogates=n_surrogates, seed=seed)
        if n_surrogates else ({}, float("nan"))
    )

    grid = np.arange(depths.min(), depths.max() + step_um, float(step_um))
    per_window: dict[float, np.ndarray] = {}
    for w in chosen:
        g, s = step_profile(depths, values, window_um=float(w), step_um=float(step_um),
                            reject_bad=reject_bad)
        if g.size == 0:
            continue
        if float(w) in stats:
            centre, spread = stats[float(w)]
            z = (s - centre) / spread
        else:
            # No null available: fall back to the profile's own spread. Comparable
            # across scales, but see calibrate_scales for why it is a poor threshold.
            z = _robust_z(s)
        # Outside a window's usable span it has no opinion, which is not the same as
        # scoring zero - NaN keeps it out of the maximum instead of holding it down.
        per_window[float(w)] = np.interp(grid, g, z, left=np.nan, right=np.nan)
    if not per_window:
        return MultiscaleProfile(np.empty(0), np.empty(0), np.empty(0), {})

    order = sorted(per_window)
    stack = np.vstack([per_window[w] for w in order])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN")
        score = np.nanmax(stack, axis=0)
        winner = np.nanargmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    valid = np.isfinite(score)
    scale = np.where(valid, np.asarray(order, dtype=float)[winner], np.nan)
    return MultiscaleProfile(grid_um=grid, score=np.where(valid, score, np.nan),
                             scale_um=scale, per_window=per_window, null_z=null)


@dataclass(frozen=True)
class DetectedBoundary:
    """A depth the features step across, with how strongly and at what scale.

    ``z_score`` is robust standard deviations above what this profile does at a
    typical depth, and ``prominence`` how far the peak stands above its own
    neighbourhood. Neither is a probability and neither is presented as one - they are
    weights, for ranking boundaries and for letting a trajectory fit trust a sharp
    step more than a vague one.
    """

    depth_um: float
    z_score: float
    prominence: float
    scale_um: float

    @property
    def weight(self) -> float:
        """Non-negative weight for a fit: the smaller of strength and prominence.

        The minimum of the two, because either alone is fooled: a broad plateau scores
        a high z everywhere across it, and a spike on an otherwise noisy profile is
        prominent without being strong.
        """
        return max(0.0, min(float(self.z_score), float(self.prominence)))


def detect_boundaries(
    profile: MultiscaleProfile,
    *,
    min_z: float | None = None,
    min_separation_um: float = MIN_SEPARATION_UM,
    max_n: int | None = None,
) -> list[DetectedBoundary]:
    """Peaks of a multi-scale profile, strongest first then thinned by separation.

    ``min_z`` defaults to the profile's own :attr:`~MultiscaleProfile.null_z` - what
    this signal reaches with its structure destroyed. **Returning nothing is a real
    answer**: a peak-picker always finds a maximum, so without a calibrated floor this
    reported a confident boundary in pure noise.

    Thinning is by strength rather than by depth order so that when two peaks are
    within ``min_separation_um`` - the same transition seen at two scales - the better
    evidenced one survives. Returned sorted by depth, which is the order a landmark
    list has to be in.
    """
    if min_z is None:
        min_z = (float(profile.null_z) if np.isfinite(profile.null_z)
                 else float(MIN_BOUNDARY_Z))
    grid = np.asarray(profile.grid_um, dtype=float)
    score = np.asarray(profile.score, dtype=float)
    if grid.size < 3:
        return []
    peaks: list[int] = []
    for i in range(1, grid.size - 1):
        s = score[i]
        if not np.isfinite(s) or s < min_z:
            continue
        left, right = score[i - 1], score[i + 1]
        if (np.isnan(left) or s >= left) and (np.isnan(right) or s > right):
            peaks.append(i)
    if not peaks:
        return []

    span = max(float(min_separation_um), 1.0)
    kept: list[int] = []
    for i in sorted(peaks, key=lambda j: -score[j]):
        if all(abs(grid[i] - grid[j]) >= span for j in kept):
            kept.append(i)
        if max_n is not None and len(kept) >= max_n:
            break

    out: list[DetectedBoundary] = []
    for i in sorted(kept):
        near = (grid >= grid[i] - span) & (grid <= grid[i] + span)
        neighbourhood = score[near]
        floor = float(np.nanmin(neighbourhood)) if np.isfinite(neighbourhood).any() else 0.0
        out.append(DetectedBoundary(
            depth_um=float(grid[i]),
            z_score=float(score[i]),
            prominence=float(score[i] - floor),
            scale_um=float(profile.scale_um[i]),
        ))
    return out


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
