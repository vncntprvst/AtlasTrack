"""Refining a multi-shank probe's trajectory: along-track offset and array roll.

**What the ephys can and cannot constrain, stated up front, because the original plan
had one of them wrong.**

A Neuropixels 2.0 4-shank array is a rigid comb: parallel shanks, 250 µm pitch, and
**tips coplanar** - all four sit at the same position along the insertion axis. Roll is
the rotation of the shank row about that axis. So:

* **Roll does not move any shank along the track.** Rotating the comb about its own
  long axis sweeps the row through the plane *perpendicular* to the insertion; every
  tip stays at the same axial position. The plan's proposal to "fit roll from the
  linear trend of per-shank depth offsets across shank index" therefore cannot work -
  those offsets measure how the brain surface and the anatomy vary across the 750 µm
  the row spans, which is tissue geometry, not roll.
* **Roll changes which anatomy each shank passes through**, because it points the
  750 µm row along AP rather than ML (or anywhere between). *That* is identifiable
  from ephys: score a candidate roll by how well each shank's measured feature
  transitions line up with the atlas boundaries predicted for it, and scan.
* **Along-track offset is well constrained by the ephys** - it is the shift the
  landmark alignment already produces. But the *common* part of it is not evidence
  about the tissue: the insertion zero is itself uncertain by a couple of hundred µm
  (dura, brain swelling, breathing), so a shared offset absorbs manipulator error as
  readily as shrinkage. Only per-shank *differences* carry geometry.
* **Perpendicular offset is weakly constrained.** Not fitted here; a manual nudge with
  a visible deviation readout is the honest treatment.

Pure numpy - no atlas, no Qt. The scoring that needs an atlas composes these with
:mod:`histo_to_ccf.ephys.regions`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Lab convention (Wang lab notebooks), which this module follows exactly:
#
#   pitch  angle of the probe away from vertical. "Vertical pitch" = straight down,
#          roughly orthogonal to the brain surface. 10° is a small tilt, 20° more.
#   roll   rotation of the probe about its own holder axis, **0 = all shanks in line
#          with the AP axis**. Positive roll swings the *anterior* shank laterally
#          and the posterior shank medially, so "vertical pitch + 45° roll" is a row
#          running anterolateral-to-posteromedial.
#
# The lab uses the same numbers on both hemispheres, so "lateral" is away from the
# midline rather than a fixed CCF direction. That is why the hemisphere is read from
# the array's own position instead of assumed - the same +45° describes mirror-image
# arrangements on the two sides, and hard-coding one would silently mirror the other.
#
# CCF axis directions, from histo_to_ccf.io.ccf_coords: AP *increases posteriorly*,
# so anterior is -AP; the ML midline sits at 5700 µm.
_AP_AXIS = np.array([1.0, 0.0, 0.0])
_ML_AXIS = np.array([0.0, 1.0, 0.0])
_ANTERIOR = -_AP_AXIS
_MIDLINE_ML_UM = 5700.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def array_axes(tips, entries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insertion direction ``u``, row axis ``r`` (perpendicular to ``u``), and centre.

    ``r`` is the principal axis of the tips *after projecting out* ``u``, matching
    :func:`histo_to_ccf.probes.fitting.fit_rigid_array`: without that projection the
    millimetre-scale spread in insertion depth would dominate the sub-millimetre row
    and hijack the axis.
    """
    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    if tips.shape != entries.shape or tips.ndim != 2 or len(tips) < 2:
        raise ValueError("tips and entries must both be (n_shanks >= 2, 3)")
    u = _unit((tips - entries).mean(0))
    centred = tips - tips.mean(0)
    perp = centred - (centred @ u)[:, None] * u
    if not np.any(np.abs(perp) > 1e-9):
        # Every shank at the same place: no row to speak of. Pick any perpendicular.
        ref = _AP_AXIS if abs(float(u @ _ML_AXIS)) > 0.9 else _ML_AXIS
        return u, _unit(ref - (ref @ u) * u), tips.mean(0)
    _s, _v, vt = np.linalg.svd(perp, full_matrices=False)
    r = _unit(vt[0] - (vt[0] @ u) * u)
    return u, r, tips.mean(0)


def lateral_sign(tips, midline_ml_um: float = _MIDLINE_ML_UM) -> float:
    """``+1`` if the array sits on the ML-greater side of the midline, else ``-1``.

    "Lateral" means away from the midline, because the lab's roll numbers are used
    unchanged on both hemispheres. Read from the data rather than assumed.
    """
    centre_ml = float(np.asarray(tips, dtype=float)[:, 1].mean())
    return 1.0 if centre_ml >= midline_ml_um else -1.0


def pitch_deg(tips, entries) -> float:
    """Angle of the insertion away from vertical, in degrees (0 = straight down).

    Matches the lab's "vertical pitch / 10° pitch / 20° pitch": how far the probe was
    tipped in the manipulator, not a signed direction.
    """
    u, _r, _centre = array_axes(tips, entries)
    return float(np.degrees(np.arccos(np.clip(abs(float(u[2])), 0.0, 1.0))))


def roll_deg(tips, entries, *, midline_ml_um: float = _MIDLINE_ML_UM) -> float:
    """Roll in the **lab's convention**: 0° = shank row along AP, + = anterior lateral.

    So ``+45`` means the most anterior shank sits more laterally and the most
    posterior shank more medially - and it means that on either hemisphere, because
    "lateral" is resolved against the midline from the array's own position.

    The row is an undirected line, so the value is folded into ``[-90, 90]``:
    "shank 0 at +40°" and "shank 3 at -140°" describe the same physical array, and
    reporting them as different numbers would invite a spurious correction. Note that
    ``+45`` and ``-45`` are *not* the same - they are anterior-lateral versus
    anterior-medial - so the fold keeps the distinction that matters.
    """
    u, r, _centre = array_axes(tips, entries)
    anterior = _unit(_ANTERIOR - (_ANTERIOR @ u) * u)
    lateral = lateral_sign(tips, midline_ml_um) * _ML_AXIS
    lateral = _unit(lateral - (lateral @ u) * u)
    angle = np.degrees(np.arctan2(float(r @ lateral), float(r @ anterior)))
    return float((angle + 90.0) % 180.0 - 90.0)


def row_direction(roll_degrees: float, insertion_dir, lateral: float = 1.0) -> np.ndarray:
    """The unit row axis for a lab-convention roll, given the insertion direction.

    The inverse of :func:`roll_deg`, so a roll read off the notebook can be turned
    into a trajectory. ``lateral`` is :func:`lateral_sign` for the hemisphere.
    """
    u = _unit(np.asarray(insertion_dir, dtype=float))
    anterior = _unit(_ANTERIOR - (_ANTERIOR @ u) * u)
    lat = _unit(float(lateral) * _ML_AXIS - (float(lateral) * _ML_AXIS @ u) * u)
    theta = np.radians(float(roll_degrees))
    return _unit(np.cos(theta) * anterior + np.sin(theta) * lat)


def rolled_array(tips, entries, delta_deg: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the shank row about the insertion axis by ``delta_deg``.

    Spacing, the array centre and the insertion direction are all preserved - only the
    direction the row points changes, which is exactly what roll is. Each shank's
    component *along* the track is untouched, which is the geometric fact that rules
    out reading roll from per-shank depth offsets.
    """
    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    u, _r, _centre = array_axes(tips, entries)
    theta = np.radians(float(delta_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _rotate(points: np.ndarray) -> np.ndarray:
        centre = points.mean(0)
        v = points - centre
        along = (v @ u)[:, None] * u
        perp = v - along
        # Rodrigues about u; perp is already perpendicular so the (u·perp) term drops.
        spun = perp * cos_t + np.cross(u, perp) * sin_t
        return centre + along + spun

    return _rotate(tips), _rotate(entries)


def shift_along_track(tips, entries, delta_um: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Slide the whole array along the insertion axis by ``delta_um`` (+ = deeper).

    Applied to tips and entries together, so the track length is unchanged: this moves
    where the probe sits, it does not stretch it.
    """
    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    u, _r, _centre = array_axes(tips, entries)
    step = float(delta_um) * u
    return tips + step, entries + step


def tilted_array(tips, entries, delta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Change the insertion **pitch** by ``delta_deg``, pivoting at the entry.

    Rotation is about the row axis and about the *entry* centroid, because that is
    what changing the manipulator angle does: the probe stays where it went into the
    brain and the tips swing. Pivoting at the tips instead would drag the entry points
    across the surface, which no manipulator adjustment can do.
    """
    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    _u, r, _centre = array_axes(tips, entries)
    pivot = entries.mean(0)
    theta = np.radians(float(delta_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _rotate(points: np.ndarray) -> np.ndarray:
        v = points - pivot
        along = (v @ r)[:, None] * r
        perp = v - along
        spun = perp * cos_t + np.cross(r, perp) * sin_t
        return pivot + along + spun

    return _rotate(tips), _rotate(entries)


def transformed_array(tips, entries, *, offset_um: float = 0.0, roll_deg: float = 0.0,
                      tilt_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Apply a rigid probe adjustment: along-track offset, roll, and pitch.

    **Moves the probe, never the anatomy.** The histology registration is taken as
    given here; this asks only "where must the probe have been, for its recordings to
    sit where the atlas says these structures are".

    Order matters and is fixed: tilt, then roll, then offset. Roll is defined about
    the insertion axis, so it has to be applied after the axis is final, and the
    offset is along that same final axis.
    """
    out_tips, out_entries = np.asarray(tips, float), np.asarray(entries, float)
    if tilt_deg:
        out_tips, out_entries = tilted_array(out_tips, out_entries, tilt_deg)
    if roll_deg:
        out_tips, out_entries = rolled_array(out_tips, out_entries, roll_deg)
    if offset_um:
        out_tips, out_entries = shift_along_track(out_tips, out_entries, offset_um)
    return out_tips, out_entries


@dataclass(frozen=True)
class TrajectoryFit:
    """A candidate probe adjustment and how well it explains the ephys."""

    offset_um: float = 0.0
    roll_deg: float = 0.0
    tilt_deg: float = 0.0
    score: float = 0.0
    n_boundaries: int = 0

    @property
    def mean_score(self) -> float:
        return self.score / self.n_boundaries if self.n_boundaries else 0.0


def score_trajectory(tips, entries, shank_profiles, atlas, *, min_band_um: float = 150.0,
                     margin_um: float = 1200.0, step_um: float = 20.0) -> tuple[float, int]:
    """How well a probe placement's atlas boundaries land on measured feature steps.

    ``shank_profiles`` maps shank index -> ``(depth_from_tip_grid, step_score)``, the
    output of :func:`histo_to_ccf.ephys.autolandmarks.step_profile` expressed in µm
    **from the tip** - the one axis that does not move when the probe does, since the
    electrodes are fixed to the shank.

    Returns ``(total_score, n_boundaries)``. Summed across shanks: a placement has to
    explain all four at once, which is what makes roll and pitch identifiable at all.
    """
    from histo_to_ccf.ephys.autolandmarks import candidate_boundaries
    from histo_to_ccf.ephys.regions import region_bands, regions_along_track

    tips = np.asarray(tips, dtype=float)
    entries = np.asarray(entries, dtype=float)
    total, count = 0.0, 0
    for index, (grid, score) in shank_profiles.items():
        if index >= len(tips) or np.size(grid) == 0:
            continue
        tip, entry = tips[index], entries[index]
        track = float(np.linalg.norm(tip - entry))
        if track <= 0:
            continue
        depths = np.arange(-margin_um, track + margin_um + step_um, step_um)
        bands = region_bands(
            regions_along_track(atlas, tip, entry, depths), depths
        )
        for below_surface, _label in candidate_boundaries(bands, min_band_um=min_band_um):
            from_tip = track - below_surface
            total += float(np.interp(from_tip, grid, score, left=0.0, right=0.0))
            count += 1
    return total, count


def refine_trajectory(
    tips, entries, shank_profiles, atlas, *,
    offsets_um=None, rolls_deg=None, tilts_deg=None, **score_kwargs
) -> TrajectoryFit:
    """Coarse grid search over offset / roll / pitch. Returns the best placement.

    A grid, not a gradient method, and deliberately: the score is built from atlas
    boundaries appearing and disappearing as the probe moves, so it is piecewise
    constant in places and has no useful derivative. The grid also makes the result
    reproducible, which matters more here than the last few µm.

    Defaults scan ±500 µm of offset and ±10° of roll and pitch - the range over which
    a histology placement is plausibly wrong, not the range the parameters can take.
    """
    offsets = np.arange(-500.0, 501.0, 50.0) if offsets_um is None else np.asarray(offsets_um)
    rolls = np.arange(-10.0, 10.1, 2.5) if rolls_deg is None else np.asarray(rolls_deg)
    tilts = np.arange(-10.0, 10.1, 2.5) if tilts_deg is None else np.asarray(tilts_deg)

    best = TrajectoryFit()
    for tilt in tilts:
        for roll in rolls:
            moved_t, moved_e = transformed_array(tips, entries, roll_deg=float(roll),
                                                 tilt_deg=float(tilt))
            for offset in offsets:
                cand_t, cand_e = shift_along_track(moved_t, moved_e, float(offset))
                total, n = score_trajectory(cand_t, cand_e, shank_profiles, atlas,
                                            **score_kwargs)
                if n and total > best.score:
                    best = TrajectoryFit(offset_um=float(offset), roll_deg=float(roll),
                                         tilt_deg=float(tilt), score=total,
                                         n_boundaries=n)
    return best


def shank_row_positions(tips, entries) -> np.ndarray:
    """Each shank's signed position along the row axis, µm from the array centre.

    Ordering these gives the physical shank order, which is what decides whether
    "shank 4 anterior" is satisfied.
    """
    u, r, _centre = array_axes(tips, entries)
    tips = np.asarray(tips, dtype=float)
    centred = tips - tips.mean(0)
    perp = centred - (centred @ u)[:, None] * u
    return perp @ r
