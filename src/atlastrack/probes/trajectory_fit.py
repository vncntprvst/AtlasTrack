"""Fitting a probe placement to detected LFP boundaries, by matched assignment.

**Why this exists alongside**
:func:`~atlastrack.probes.trajectory_refine.score_trajectory`. That scorer sums the
continuous step profile at every atlas boundary. Run on real LO_07 data it does not
converge, and the reasons are structural rather than a matter of tuning:

* Moving the probe changes *which* atlas boundaries it crosses - measured 22 to 26
  over a +/-300 µm offset and 18 to 31 across a roll scan - so the total and the
  per-boundary mean are scoring a different question at every candidate placement.
* Nothing requires the same boundaries to stay matched, so a placement that happens to
  drop several boundaries onto any profile peaks scores well for no anatomical reason.
* The result is a landscape with no basin: at a 20 µm offset step the score jumps by a
  median of 0.374 against a total range of 2.617, and grid search on that lands on the
  edge of whatever range it is given (ProbeA tilt at +20 deg, ProbeB offset at +700 µm).

This module fixes the comparison by turning it around. **The ephys observations are
the fixed set**; a placement is judged by how much of *them* it explains. The detected
boundaries live in µm from the tip, which is welded to the shank and does not move when
the probe does, so the denominator is constant across every placement being compared
and the score is a fraction in ``[0, 1]``.

Matching is an order-preserving assignment, not nearest-neighbour: boundaries that
cross cannot both be right, and a greedy pairing produces exactly that. The same
argument is made at length in :mod:`atlastrack.ephys.autolandmarks`, whose dynamic
programme this mirrors.

Pure numpy plus an atlas lookup. No Qt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atlastrack.probes.trajectory_refine import shift_along_track, transformed_array

#: How far a detected boundary may sit from an atlas boundary and still be called the
#: same transition. Beyond this the pairing is refused outright rather than scored low:
#: past a few hundred µm the nearer explanation is that they are different structures.
MAX_MATCH_UM = 350.0

#: Width of the agreement kernel. A boundary matched to within this scores about 0.6 of
#: its weight. Chosen to match the resolution the detector actually has - on LO_07 the
#: sharp boundaries land within 2-40 µm, the vague ones within 100-200.
MATCH_SIGMA_UM = 120.0

#: Atlas bands thinner than this are not scored: the atlas cannot place their edges
#: better than the sample grid, and the ephys certainly cannot.
MIN_BAND_UM = 150.0


@dataclass(frozen=True)
class ShankEvidence:
    """What one shank's ephys says, in µm from the tip.

    From the tip because that is the one axis fixed to the hardware: the electrodes do
    not move when the placement hypothesis does, so this is the part of the problem
    that stays constant while the probe is being moved around.
    """

    shank_index: int
    depths_from_tip_um: np.ndarray
    weights: np.ndarray

    @property
    def total_weight(self) -> float:
        w = np.asarray(self.weights, dtype=float)
        return float(np.sum(w[np.isfinite(w)]))

    @classmethod
    def from_boundaries(cls, shank_index: int, boundaries) -> ShankEvidence:
        """Build from :class:`~atlastrack.ephys.autolandmarks.DetectedBoundary`."""
        ordered = sorted(boundaries, key=lambda b: float(b.depth_um))
        return cls(
            shank_index=int(shank_index),
            depths_from_tip_um=np.array([float(b.depth_um) for b in ordered]),
            weights=np.array([float(b.weight) for b in ordered]),
        )


def evidence_from_features(features) -> dict:
    """Detected boundaries per shank from saved/computed depth features.

    ``features`` maps shank index to
    :class:`~atlastrack.ephys.export.ShankFeatureExport`, which is what both the
    compute step and the saved ``.npz`` produce. Shanks whose LFP yields no boundary
    above its own null are **left out entirely** rather than contributing an empty
    entry: a shank that saw nothing must not dilute the denominator, or a placement
    would be rewarded for the shanks that had nothing to say.

    Band powers rather than the raw PSD, because the detector standardises per feature
    and five bands give it something to agree across.
    """
    from atlastrack.ephys.autolandmarks import (
        detect_boundaries,
        multiscale_step_profile,
    )
    from atlastrack.ephys.features import lfp_band_power

    out: dict = {}
    for index, export in sorted((features or {}).items()):
        psd = np.asarray(getattr(export, "lfp_psd", []), dtype=float)
        from_tip = np.asarray(
            getattr(export, "channel_depth_from_tip_um", []), dtype=float
        )
        freqs = np.asarray(getattr(export, "lfp_freqs_hz", []), dtype=float)
        if psd.ndim != 2 or psd.size == 0 or from_tip.size != psd.shape[0]:
            continue
        keep = np.isfinite(psd).any(axis=1)
        if keep.sum() < 8:
            continue
        order = np.argsort(from_tip[keep])
        depths = from_tip[keep][order]
        bands = lfp_band_power(psd[keep][order], freqs)
        found = detect_boundaries(multiscale_step_profile(depths, bands))
        if found:
            out[int(index)] = ShankEvidence.from_boundaries(int(index), found)
    return out


@dataclass(frozen=True)
class Match:
    """One detected boundary paired with one atlas boundary."""

    shank_index: int
    feature_um: float  # detected, µm from the tip
    atlas_um: float  # atlas boundary, µm from the tip at this placement
    weight: float
    gain: float
    label: str = ""

    @property
    def residual_um(self) -> float:
        return self.feature_um - self.atlas_um


@dataclass(frozen=True)
class PlacementScore:
    """How much of the ephys a candidate placement explains.

    ``explained`` is the headline: gain over the **total available weight**, so 0 means
    the placement accounts for nothing and 1 would mean every detected boundary sits
    exactly on an atlas boundary. It is comparable between placements, between shanks
    and between probes, which the old total-score was not.
    """

    explained: float
    matched: int
    available: int
    total_weight: float
    matches: list = field(default_factory=list)
    offset_um: float = 0.0
    roll_deg: float = 0.0
    tilt_deg: float = 0.0

    @property
    def residuals_um(self) -> np.ndarray:
        return np.array([m.residual_um for m in self.matches], dtype=float)

    @property
    def mean_residual_um(self) -> float:
        r = self.residuals_um
        return float(np.mean(r)) if r.size else 0.0

    @property
    def residual_spread_um(self) -> float:
        """Robust spread of the residuals - how consistent the matches are.

        A rigid move that is genuinely right leaves small, unstructured residuals. A
        large spread with a good score means the placement satisfied a few boundaries
        by luck and the rest are anywhere.
        """
        r = self.residuals_um
        if r.size < 2:
            return 0.0
        return float(1.4826 * np.median(np.abs(r - np.median(r))))


def atlas_boundaries_from_tip(
    atlas, tip_ccf_um, entry_ccf_um, *, min_band_um: float = MIN_BAND_UM,
    margin_um: float = 600.0, step_um: float = 15.0,
) -> list[tuple[float, str]]:
    """Atlas boundaries along one shank as ``(µm from tip, label)``.

    Sampled a little past both ends so a boundary just outside the modelled track can
    still claim a feature - which is the situation whenever the placement is wrong,
    i.e. exactly when this is being used.
    """
    from atlastrack.ephys.autolandmarks import candidate_boundaries
    from atlastrack.ephys.regions import region_bands, regions_along_track

    tip = np.asarray(tip_ccf_um, dtype=float)
    entry = np.asarray(entry_ccf_um, dtype=float)
    track = float(np.linalg.norm(tip - entry))
    if track <= 0:
        return []
    depths = np.arange(-margin_um, track + margin_um + step_um, step_um)
    bands = region_bands(regions_along_track(atlas, tip, entry, depths), depths)
    return [
        (track - below, label)
        for below, label in candidate_boundaries(bands, min_band_um=min_band_um)
    ]


def match_ordered(
    feature_um, weights, atlas_um, *,
    sigma_um: float = MATCH_SIGMA_UM, max_match_um: float = MAX_MATCH_UM,
) -> list[tuple[int, int, float]]:
    """Order-preserving assignment maximising total gain: ``[(i_feature, j_atlas, gain)]``.

    Both inputs must be sorted ascending. Either side may be skipped - a detected
    boundary with no atlas counterpart contributes nothing rather than being forced
    onto the nearest one, and an atlas boundary the ephys did not see costs nothing
    either, because plenty of real boundaries are LFP-silent.

    Dynamic programming rather than greedy nearest-neighbour: greedy routinely pairs
    boundaries in crossed order, and a crossed pair asserts that two transitions swapped
    places along the shank, which no rigid move can produce.
    """
    f = np.asarray(feature_um, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    a = np.asarray(atlas_um, dtype=float).ravel()
    n, m = f.size, a.size
    if n == 0 or m == 0:
        return []

    gain = np.zeros((n, m), dtype=float)
    for i in range(n):
        d = np.abs(f[i] - a)
        g = float(w[i]) * np.exp(-0.5 * (d / float(sigma_um)) ** 2)
        gain[i] = np.where(d <= float(max_match_um), g, -1.0)

    dp = np.zeros((n + 1, m + 1), dtype=float)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 skip-f, 1 skip-a, 2 pair
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, choice = dp[i - 1, j], 0
            if dp[i, j - 1] > best:
                best, choice = dp[i, j - 1], 1
            g = gain[i - 1, j - 1]
            if g > 0 and dp[i - 1, j - 1] + g > best:
                best, choice = dp[i - 1, j - 1] + g, 2
            dp[i, j], back[i, j] = best, choice

    out: list[tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 and j > 0:
        move = back[i, j]
        if move == 2:
            out.append((i - 1, j - 1, float(gain[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif move == 1:
            j -= 1
        else:
            i -= 1
    return list(reversed(out))


def score_placement(
    tips, entries, evidence, atlas, *,
    sigma_um: float = MATCH_SIGMA_UM, max_match_um: float = MAX_MATCH_UM,
    min_band_um: float = MIN_BAND_UM, offset_um: float = 0.0,
    roll_deg: float = 0.0, tilt_deg: float = 0.0,
) -> PlacementScore:
    """Fraction of the detected-boundary weight this placement explains.

    ``evidence`` maps shank index to :class:`ShankEvidence`. Every shank is scored
    against the same placement and the results summed, which is what makes roll and
    pitch identifiable at all: they barely move any single shank but change the
    *relative* anatomy of all four.
    """
    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    moved_t, moved_e = transformed_array(
        tips, entries, offset_um=offset_um, roll_deg=roll_deg, tilt_deg=tilt_deg
    )

    total_weight = 0.0
    available = 0
    matches: list[Match] = []
    for index, ev in sorted(evidence.items()):
        if index >= len(moved_t):
            continue
        total_weight += ev.total_weight
        available += int(np.asarray(ev.depths_from_tip_um).size)
        edges = atlas_boundaries_from_tip(
            atlas, moved_t[index], moved_e[index], min_band_um=min_band_um
        )
        if not edges:
            continue
        edges.sort(key=lambda e: e[0])
        a_um = np.array([e[0] for e in edges], dtype=float)
        labels = [e[1] for e in edges]
        for i, j, g in match_ordered(ev.depths_from_tip_um, ev.weights, a_um,
                                     sigma_um=sigma_um, max_match_um=max_match_um):
            matches.append(Match(
                shank_index=int(index),
                feature_um=float(ev.depths_from_tip_um[i]),
                atlas_um=float(a_um[j]),
                weight=float(ev.weights[i]),
                gain=float(g),
                label=labels[j],
            ))

    gained = float(sum(m.gain for m in matches))
    return PlacementScore(
        explained=(gained / total_weight) if total_weight > 0 else 0.0,
        matched=len(matches), available=available, total_weight=total_weight,
        matches=matches, offset_um=float(offset_um), roll_deg=float(roll_deg),
        tilt_deg=float(tilt_deg),
    )


@dataclass(frozen=True)
class ParameterScan:
    """One parameter swept with the others held, and whether it is identifiable."""

    name: str
    values: np.ndarray
    explained: np.ndarray

    @property
    def best_value(self) -> float:
        return float(self.values[int(np.argmax(self.explained))])

    @property
    def at_edge(self) -> bool:
        """Whether the optimum sits at either end - i.e. the scan was too narrow."""
        i = int(np.argmax(self.explained))
        return i in (0, self.explained.size - 1)

    @property
    def contrast(self) -> float:
        """Peak height above the median, relative to the peak. 0 = flat, 1 = a spike."""
        peak = float(np.max(self.explained))
        if peak <= 0:
            return 0.0
        return float((peak - np.median(self.explained)) / peak)

    @property
    def roughness(self) -> float:
        """Median step-to-step change over the peak-to-trough range.

        Small means a smooth hill worth optimising on; large means the landscape is
        noise and the argmax is a lottery. This is the number that condemned the old
        objective at 0.14.
        """
        span = float(np.max(self.explained) - np.min(self.explained))
        if self.explained.size < 3 or span <= 0:
            return 0.0
        return float(np.median(np.abs(np.diff(self.explained))) / span)

    def identifiable(self, *, min_contrast: float = 0.25,
                     max_roughness: float = 0.12) -> bool:
        """Interior peak, clearly above the rest, on a smooth landscape - all three."""
        return (not self.at_edge and self.contrast >= min_contrast
                and self.roughness <= max_roughness)


def scan_parameter(tips, entries, evidence, atlas, name: str, values, **fixed
                   ) -> ParameterScan:
    """Sweep one of ``offset_um`` / ``roll_deg`` / ``tilt_deg``, holding the others."""
    vals = np.asarray(values, dtype=float)
    out = np.array([
        score_placement(tips, entries, evidence, atlas, **{**fixed, name: float(v)}
                        ).explained
        for v in vals
    ])
    return ParameterScan(name=name, values=vals, explained=out)


@dataclass(frozen=True)
class LeaveOneOut:
    """Refits with each shank held out, to see whether one shank is carrying it.

    Roll is the parameter this exists for. It barely moves any single shank along its
    own track - rotating the comb about the insertion axis leaves every tip at the same
    axial position - so it is identifiable *only* from the four shanks disagreeing about
    anatomy in a coordinated way. That makes it exactly the parameter that a single
    dominant shank can fake: on LO_07 ProbeA one shank carries a 5745 µm column and the
    other three have 705 µm each, so an estimate driven entirely by the long one would
    look confident and mean nothing about the array.

    Holding each shank out in turn separates the two. Agreement means the geometry is
    constrained; a large ``spread``, or one shank whose removal moves the answer, means
    it is not.
    """

    name: str
    full: float
    per_shank: dict
    weight_share: dict = field(default_factory=dict)

    @property
    def estimates(self) -> np.ndarray:
        return np.array(sorted(self.per_shank.values()), dtype=float)

    @property
    def spread(self) -> float:
        """Full range of the held-out estimates - the honest measure with n = 4."""
        e = self.estimates
        return float(e.max() - e.min()) if e.size > 1 else 0.0

    @property
    def dominant_shank(self):
        """The shank whose removal moves the estimate most, or ``None``."""
        if not self.per_shank:
            return None
        return max(self.per_shank, key=lambda k: abs(self.per_shank[k] - self.full))

    @property
    def max_influence(self) -> float:
        s = self.dominant_shank
        return 0.0 if s is None else abs(self.per_shank[s] - self.full)

    def stable(self, tolerance: float) -> bool:
        """Every subset agrees within ``tolerance``, and at least three were usable."""
        return (len(self.per_shank) >= 3 and self.spread <= tolerance
                and self.max_influence <= tolerance)

    def summary(self, tolerance: float, unit: str = "deg") -> str:
        lines = [
            f"{self.name}: {self.full:+.1f} {unit} with all shanks; "
            f"leave-one-out spread {self.spread:.1f} {unit} "
            f"({'stable' if self.stable(tolerance) else 'NOT stable'} "
            f"at +/-{tolerance:g})"
        ]
        for shank in sorted(self.per_shank):
            share = self.weight_share.get(shank, 0.0)
            lines.append(
                f"   without shank {shank} (notes {shank + 1}, {share:.0%} of the "
                f"evidence): {self.per_shank[shank]:+.1f} {unit}"
            )
        dom = self.dominant_shank
        if dom is not None and self.max_influence > tolerance:
            lines.append(
                f"   shank {dom} alone moves it by {self.max_influence:.1f} {unit} - "
                "the estimate rests on that shank, not on the array"
            )
        return chr(10).join(lines)


def leave_one_out(
    tips, entries, evidence, atlas, *, name: str = "roll_deg", values=None, **fixed
) -> LeaveOneOut:
    """Scan ``name`` with all shanks, then again with each one held out."""
    vals = (np.arange(-15.0, 15.1, 2.5) if values is None
            else np.asarray(values, dtype=float))
    full = scan_parameter(tips, entries, evidence, atlas, name, vals, **fixed)

    total = sum(ev.total_weight for ev in evidence.values()) or 1.0
    share = {k: ev.total_weight / total for k, ev in evidence.items()}
    per_shank: dict = {}
    for held in sorted(evidence):
        rest = {k: v for k, v in evidence.items() if k != held}
        if len(rest) < 2:
            continue
        per_shank[held] = scan_parameter(
            tips, entries, rest, atlas, name, vals, **fixed
        ).best_value
    return LeaveOneOut(name=name, full=full.best_value, per_shank=per_shank,
                       weight_share=share)


@dataclass(frozen=True)
class TrajectoryFit:
    """The best placement found, with the evidence for believing each parameter."""

    offset_um: float
    roll_deg: float
    tilt_deg: float
    score: PlacementScore
    baseline: PlacementScore
    scans: dict = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return self.score.explained - self.baseline.explained

    def identifiable(self) -> dict:
        return {k: s.identifiable() for k, s in self.scans.items()}

    def summary(self) -> str:
        bits = [
            f"explains {self.score.explained:.1%} of the detected-boundary weight "
            f"(registered: {self.baseline.explained:.1%})",
            f"{self.score.matched}/{self.score.available} boundaries matched, "
            f"residual spread {self.score.residual_spread_um:.0f} um",
        ]
        for name, scan in self.scans.items():
            verdict = "identifiable" if scan.identifiable() else "NOT identifiable"
            why = []
            if scan.at_edge:
                why.append("optimum at the edge of the scan")
            if scan.contrast < 0.25:
                why.append(f"contrast {scan.contrast:.2f}")
            if scan.roughness > 0.12:
                why.append(f"roughness {scan.roughness:.2f}")
            bits.append(f"{name}: {scan.best_value:+.1f} - {verdict}"
                        + (f" ({'; '.join(why)})" if why else ""))
        return "\n".join(bits)


def fit_trajectory(
    tips, entries, evidence, atlas, *,
    offsets_um=None, rolls_deg=None, tilts_deg=None, **score_kwargs
) -> TrajectoryFit:
    """Grid search, then a 1-D scan per parameter so identifiability is reported.

    The scans are not decoration. A grid search always returns an argmax; whether that
    argmax means anything is a separate question, and one this codebase has already got
    wrong once by not asking it.
    """
    offsets = (np.arange(-400.0, 401.0, 25.0) if offsets_um is None
               else np.asarray(offsets_um, dtype=float))
    rolls = (np.arange(-15.0, 15.1, 2.5) if rolls_deg is None
             else np.asarray(rolls_deg, dtype=float))
    tilts = (np.arange(-10.0, 10.1, 2.5) if tilts_deg is None
             else np.asarray(tilts_deg, dtype=float))

    baseline = score_placement(tips, entries, evidence, atlas, **score_kwargs)
    best = baseline
    for tilt in tilts:
        for roll in rolls:
            moved_t, moved_e = transformed_array(tips, entries, roll_deg=float(roll),
                                                 tilt_deg=float(tilt))
            for off in offsets:
                cand_t, cand_e = shift_along_track(moved_t, moved_e, float(off))
                s = score_placement(cand_t, cand_e, evidence, atlas, **score_kwargs)
                if s.explained > best.explained:
                    best = PlacementScore(
                        explained=s.explained, matched=s.matched,
                        available=s.available, total_weight=s.total_weight,
                        matches=s.matches, offset_um=float(off),
                        roll_deg=float(roll), tilt_deg=float(tilt),
                    )

    fixed = {"offset_um": best.offset_um, "roll_deg": best.roll_deg,
             "tilt_deg": best.tilt_deg, **score_kwargs}
    scans = {}
    for name, values in (("offset_um", offsets), ("roll_deg", rolls),
                         ("tilt_deg", tilts)):
        held = {k: v for k, v in fixed.items() if k != name}
        scans[name] = scan_parameter(tips, entries, evidence, atlas, name, values,
                                     **held)
    return TrajectoryFit(offset_um=best.offset_um, roll_deg=best.roll_deg,
                         tilt_deg=best.tilt_deg, score=best, baseline=baseline,
                         scans=scans)
