"""Assign each section an AP coordinate along the cutting series.

Registering a brainstem series taught us that getting the *AP progression* right
matters more than any registration knob: DeepSlice compresses AP on look-alike
sections, so structures land on the wrong section entirely. The fix is to impose
a known progression and anchor it to one section the user is confident about.

Two progressions are supported:

``ordinal``
    Sections are evenly sampled, so AP steps by a constant per position in
    ``ap_order``. This is the original behaviour.

``slide_number``
    Sections were sampled unevenly - the usual case when a series is picked from
    a slide box, keeping some sections and skipping others. Each section carries
    the physical slide number it came from and AP steps by the *gap* between
    slide numbers, so a jump from slide 72 to slide 59 spans 13 steps, not one.

``ordinal`` is what you get when slide numbers are absent or all equal; the two
agree exactly when the slide numbers are consecutive.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from histo_to_ccf.project.schema import Section

Mode = Literal["ordinal", "slide_number"]


def _sign(value: float) -> int:
    return 1 if value >= 0 else -1


def ap_offsets(
    slide_numbers: list[int | None],
    *,
    anchor_pos: int,
    spacing_um: float,
    forward: bool = True,
) -> tuple[list[float], Mode]:
    """AP offset (µm) of each section relative to the anchor, in ``ap_order``.

    ``slide_numbers`` is in ``ap_order`` and may contain ``None``. When every
    entry is present and they are not all identical, offsets follow the
    slide-number gaps; otherwise they follow list position. ``forward`` means
    ``ap_order`` runs anterior→posterior, i.e. AP increases along the list.

    Slide numbers may count either direction (a series can be numbered from the
    anterior or the posterior end); the overall trend along ``ap_order`` is used
    to orient them, so ``forward`` keeps its meaning either way.
    """
    n = len(slide_numbers)
    if n == 0:
        return [], "ordinal"
    if not (0 <= anchor_pos < n):
        raise ValueError(f"anchor_pos {anchor_pos} out of range for {n} section(s)")

    direction = 1.0 if forward else -1.0
    usable = [s for s in slide_numbers if s is not None]
    use_slides = len(usable) == n and len(set(usable)) > 1

    if not use_slides:
        return [(i - anchor_pos) * spacing_um * direction for i in range(n)], "ordinal"

    numbers = [int(s) for s in slide_numbers]  # type: ignore[arg-type]
    # Orient the numbering so it increases along ap_order, whichever end it
    # was counted from.
    trend = _sign(numbers[-1] - numbers[0])
    anchor_number = numbers[anchor_pos]
    offsets = [
        (num - anchor_number) * trend * spacing_um * direction for num in numbers
    ]
    return offsets, "slide_number"


def assign_section_ap(
    sections: list["Section"],
    *,
    spacing_um: float,
    anchor_index: int | None = None,
    anchor_ap_um: float | None = None,
    forward: bool = True,
    bregma_ap_um: float | None = None,
) -> tuple[int, Mode]:
    """Write ``plane.ap_um`` across ``sections``, spaced from one anchor.

    ``sections`` need not be sorted - they are processed in ``ap_order``.
    ``anchor_index`` names the section (by ``Section.index``) whose AP is held
    fixed; it defaults to the first in ``ap_order``. ``anchor_ap_um`` overrides
    that section's current AP, which is otherwise kept (or taken as bregma when
    it has no plane yet).

    ``bregma_ap_um`` is that fallback bregma position, which differs between atlases
    (see :data:`histo_to_ccf.io.ccf_coords.BREGMA_AP_BY_ATLAS`). It defaults to the
    Allen anchor, which is right for Allen and Kim but 346 µm off for the BBP
    augmented CCFv3.

    Returns ``(n_sections_updated, mode)`` - see the module docstring for modes.
    """
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_FROM_ORIGIN_UM
    from histo_to_ccf.project.schema import PlaneParams

    if not sections:
        return 0, "ordinal"

    ordered = sorted(sections, key=lambda s: s.ap_order)
    anchor_pos = 0
    if anchor_index is not None:
        anchor_pos = next(
            (i for i, s in enumerate(ordered) if s.index == anchor_index), 0
        )

    anchor_sec = ordered[anchor_pos]
    if anchor_ap_um is not None:
        anchor_ap = float(anchor_ap_um)
    elif anchor_sec.plane is not None:
        anchor_ap = float(anchor_sec.plane.ap_um)
    else:
        anchor_ap = (
            BREGMA_AP_FROM_ORIGIN_UM if bregma_ap_um is None else float(bregma_ap_um)
        )

    offsets, mode = ap_offsets(
        [s.slide_number for s in ordered],
        anchor_pos=anchor_pos,
        spacing_um=spacing_um,
        forward=forward,
    )

    for section, offset in zip(ordered, offsets, strict=True):
        ap = anchor_ap + offset
        if section.plane is not None:
            section.plane = section.plane.model_copy(update={"ap_um": ap})
        else:
            section.plane = PlaneParams(ap_um=ap)
    return len(ordered), mode
