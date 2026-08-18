"""Restating a fitted probe move as the registration change that would do the same.

The ephys fit answers "where must the probe have been". It cannot answer "why is it not
there", and there are three candidates: the probe was placed differently than the
histology says, the sections were registered to the wrong atlas planes, or the tissue
shrank. All three produce the same disagreement.

One of them can often be ruled out from the bench: **the shank tips are visible in the
sections**. Where the dye is clear, the probe is where the histology puts it, and moving
it away to fix an error elsewhere makes the model disagree with the evidence in order to
agree with a hypothesis. In that case the same improvement should be sought by changing
the registration instead.

So this module inverts the fit. A probe displacement ``d`` in CCF is equivalent to the
tissue being registered ``d`` away from where it was: keep the click, change what the
section maps to. The output is in the units the registration workflow actually has -
``Section.plane.ap_um`` in µm and in whole sections, plus the in-plane components - not
in the fit's own parameters.

**Nothing here is applied.** It is a suggestion with its caveats attached, because a
registration change moves every structure on those sections and every other probe with
them, which is exactly why it is also the strongest test: a real registration error
should want the *same* change for every probe in the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from histo_to_ccf.probes.trajectory_refine import transformed_array

#: Below this, a component is reported as "no meaningful change" rather than as a
#: number. The registration cannot be steered finer than roughly half a section, and
#: quoting 12 µm of AP invites a precision the method does not have.
NEGLIGIBLE_UM = 25.0

#: Above this spread between shanks, a single per-section shift cannot reproduce the
#: fit - the move has a rotational part, and rotations are not per-section shifts.
ROTATION_SPREAD_UM = 50.0


@dataclass(frozen=True)
class ShankShift:
    """Where one shank's tip and entry would have to move, in CCF µm."""

    shank_index: int
    tip_delta_um: tuple[float, float, float]
    entry_delta_um: tuple[float, float, float]
    tip_section_idx: int | None = None
    entry_section_idx: int | None = None

    @property
    def magnitude_um(self) -> float:
        return float(np.linalg.norm(self.tip_delta_um))


@dataclass(frozen=True)
class RegistrationSuggestion:
    """What the registration would have to change to explain the same ephys."""

    probe_label: str
    ap_um: float
    ml_um: float
    dv_um: float
    per_shank: list = field(default_factory=list)
    sections: tuple = ()
    section_spacing_um: float | None = None
    shank_spread_um: float = 0.0
    has_rotation: bool = False
    #: Other probes sharing at least one section with this one, and those that do not.
    shares_sections_with: tuple = ()
    disjoint_from: tuple = ()

    @property
    def ap_in_sections(self) -> float | None:
        """The AP change in whole sections, the unit the AP is actually set in."""
        if not self.section_spacing_um:
            return None
        return float(self.ap_um / float(self.section_spacing_um))

    def text(self) -> str:
        """A human-readable suggestion with its caveats. Never a recommendation."""
        lines = [
            f"If this is a registration error rather than a probe placement error, "
            f"the equivalent change for {self.probe_label} would be:"
        ]
        lines.extend(self._component_lines())
        if self.sections:
            shown = ", ".join(str(s) for s in self.sections[:10])
            more = "" if len(self.sections) <= 10 else f" (+{len(self.sections) - 10} more)"
            lines.append(f"  Sections carrying this probe: {shown}{more}")
        lines.extend(self._caveat_lines())
        return "\n".join(lines)

    def _component_lines(self) -> list[str]:
        out = []
        if abs(self.ap_um) >= NEGLIGIBLE_UM:
            piece = (f"  AP: move these sections {abs(self.ap_um):.0f} µm "
                     f"{'posterior' if self.ap_um > 0 else 'anterior'} "
                     f"(Section.plane.ap_um {self.ap_um:+.0f} µm)")
            in_sections = self.ap_in_sections
            if in_sections is not None:
                piece += f" = {abs(in_sections):.1f} section(s) at " \
                         f"{self.section_spacing_um:.0f} µm spacing"
            out.append(piece)
        else:
            out.append(f"  AP: no meaningful change ({self.ap_um:+.0f} µm)")
        for name, value in (("ML", self.ml_um), ("DV", self.dv_um)):
            if abs(value) >= NEGLIGIBLE_UM:
                out.append(
                    f"  {name}: the atlas overlay on those sections sits "
                    f"{abs(value):.0f} µm off; nudge it {value:+.0f} µm in {name}"
                )
            else:
                out.append(f"  {name}: no meaningful change ({value:+.0f} µm)")
        return out

    def _caveat_lines(self) -> list[str]:
        out = ["  ---"]
        if self.has_rotation:
            out.append(
                f"  The fit also rotates the array (shank displacements differ by "
                f"{self.shank_spread_um:.0f} µm). A per-section shift cannot reproduce "
                "that - rotation about the insertion axis is not a registration "
                "parameter, so only the translation above is expressible here."
            )
        if self.shares_sections_with:
            out.append(
                "  Shares sections with " + ", ".join(self.shares_sections_with)
                + " - changing them moves those probes too, so the change has to suit "
                "all of them at once."
            )
        if self.disjoint_from:
            verb = "sits" if len(self.disjoint_from) == 1 else "sit"
            out.append(
                "  " + ", ".join(self.disjoint_from) + f" {verb} on different "
                "sections, so this change would leave them alone. That also weakens the "
                "cross-probe test: probes on disjoint sections can disagree without "
                "either being wrong, because a per-section AP error need not be the "
                "same everywhere in the series."
            )
        out.append(
            "  A registration change moves every structure on those sections. A real "
            "registration error should want a consistent change across the series; a "
            "placement error only for this probe."
        )
        out.append(
            "  Nothing here is applied, and the fit cannot tell placement error from "
            "registration error or from shrinkage - only that the two disagree."
        )
        return out


def _sections_for(probe) -> tuple:
    found: list[int] = []
    for shank in getattr(probe, "shanks", []):
        for idx in (getattr(shank, "tip_section_idx", None),
                    getattr(shank, "entry_section_idx", None)):
            if idx is not None and int(idx) not in found:
                found.append(int(idx))
    return tuple(sorted(found))


def suggest_registration_change(
    probe, *, offset_um: float = 0.0, roll_deg: float = 0.0, tilt_deg: float = 0.0,
    section_spacing_um: float | None = None, other_probes=(),
) -> RegistrationSuggestion | None:
    """The registration change equivalent to moving this probe by the fitted amount.

    ``None`` when the probe has no registered shanks - there is then no displacement
    to express and nothing to suggest.
    """
    registered = [
        s for s in getattr(probe, "shanks", [])
        if getattr(s, "tip_ccf_um", None) is not None
        and getattr(s, "entry_ccf_um", None) is not None
    ]
    if not registered:
        return None

    tips = np.array([s.tip_ccf_um for s in registered], dtype=float)
    entries = np.array([s.entry_ccf_um for s in registered], dtype=float)
    moved_t, moved_e = transformed_array(
        tips, entries, offset_um=offset_um, roll_deg=roll_deg, tilt_deg=tilt_deg
    )
    tip_deltas = moved_t - tips
    entry_deltas = moved_e - entries

    per_shank = [
        ShankShift(
            shank_index=int(s.index),
            tip_delta_um=tuple(float(v) for v in tip_deltas[i]),
            entry_delta_um=tuple(float(v) for v in entry_deltas[i]),
            tip_section_idx=getattr(s, "tip_section_idx", None),
            entry_section_idx=getattr(s, "entry_section_idx", None),
        )
        for i, s in enumerate(registered)
    ]

    # The common part is what a per-section change can express; the spread around it is
    # the rotational remainder, which one cannot.
    mean_delta = tip_deltas.mean(axis=0)
    spread = float(np.max(np.linalg.norm(tip_deltas - mean_delta, axis=1)))

    mine = set(_sections_for(probe))
    shared, disjoint = [], []
    for other in other_probes or ():
        if getattr(other, "label", None) == getattr(probe, "label", None):
            continue
        theirs = set(_sections_for(other))
        if not theirs:
            continue
        (shared if mine & theirs else disjoint).append(str(other.label))

    return RegistrationSuggestion(
        probe_label=str(getattr(probe, "label", "")),
        ap_um=float(mean_delta[0]),
        ml_um=float(mean_delta[1]),
        dv_um=float(mean_delta[2]),
        per_shank=per_shank,
        sections=_sections_for(probe),
        section_spacing_um=section_spacing_um,
        shank_spread_um=spread,
        has_rotation=spread >= ROTATION_SPREAD_UM,
        shares_sections_with=tuple(sorted(shared)),
        disjoint_from=tuple(sorted(disjoint)),
    )
