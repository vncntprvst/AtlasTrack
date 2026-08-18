"""Restating a probe fit as the equivalent registration change.

Written because the fit cannot tell placement error from registration error, and one
of those can often be ruled out at the bench: where the shank tips are visible in the
sections, moving the probe away from its own dye fixes the wrong thing.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.probes.registration_suggestion import (
    NEGLIGIBLE_UM,
    suggest_registration_change,
)
from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Shank


def _probe(label="ProbeA", *, n=4, sections=(1, 4), tip_dv=5000.0, ap=8000.0):
    return ProbeSpec(
        label=label, type=ProbeType(name="NP2.0", n_shanks=n),
        shanks=[Shank(index=i,
                      tip_ccf_um=(ap, 5000.0 + 250.0 * i, tip_dv),
                      entry_ccf_um=(ap, 5000.0 + 250.0 * i, tip_dv - 4000.0),
                      tip_section_idx=sections[0],
                      entry_section_idx=sections[-1])
                for i in range(n)],
    )


def test_a_pure_along_track_offset_on_a_vertical_probe_is_a_dv_change():
    """The probe goes straight down, so 'deeper' is DV and nothing else."""
    s = suggest_registration_change(_probe(), offset_um=200.0)

    assert abs(s.ap_um) < 1.0
    assert abs(s.ml_um) < 1.0
    assert s.dv_um == pytest.approx(200.0, abs=1.0)
    assert "DV" in s.text()


def test_the_ap_change_is_reported_in_whole_sections():
    """AP is set per section, so µm alone is not an actionable unit."""
    tilted = _probe()
    # Lean the probe so an along-track move has a real AP component.
    for shank in tilted.shanks:
        ap, ml, dv = shank.tip_ccf_um
        shank.entry_ccf_um = (ap - 4000.0, ml, dv - 1000.0)

    s = suggest_registration_change(tilted, offset_um=400.0, section_spacing_um=100.0)

    assert s.ap_in_sections == pytest.approx(s.ap_um / 100.0)
    assert "section(s) at 100 µm spacing" in s.text()


def test_no_spacing_means_no_section_count_rather_than_a_guess():
    s = suggest_registration_change(_probe(), offset_um=200.0, section_spacing_um=None)

    assert s.ap_in_sections is None
    assert "section(s) at" not in s.text()


def test_a_negligible_component_is_not_quoted_as_a_number_to_act_on():
    s = suggest_registration_change(_probe(), offset_um=float(NEGLIGIBLE_UM) / 2.0)

    assert "no meaningful change" in s.text()


def test_a_rotation_is_reported_as_not_expressible_per_section():
    """Roll is not a registration parameter; a per-section shift cannot reproduce it."""
    s = suggest_registration_change(_probe(), roll_deg=10.0)

    assert s.has_rotation
    assert s.shank_spread_um > 50.0
    assert "cannot reproduce that" in s.text()


def test_a_pure_translation_reports_no_rotation():
    s = suggest_registration_change(_probe(), offset_um=200.0)

    assert not s.has_rotation
    assert "cannot reproduce that" not in s.text()


def test_the_sections_carrying_the_probe_are_named():
    s = suggest_registration_change(_probe(sections=(3, 7)), offset_um=200.0)

    assert s.sections == (3, 7)
    assert "3, 7" in s.text()


def test_probes_sharing_sections_are_flagged_as_affected():
    mine = _probe("ProbeA", sections=(1, 4))
    other = _probe("ProbeB", sections=(4, 9))

    s = suggest_registration_change(mine, offset_um=200.0, other_probes=[mine, other])

    assert s.shares_sections_with == ("ProbeB",)
    assert "moves those probes too" in s.text()


def test_probes_on_disjoint_sections_weaken_the_cross_probe_test():
    """Two probes on different sections can disagree without either being wrong."""
    mine = _probe("ProbeA", sections=(1, 4))
    other = _probe("ProbeB", sections=(15, 20))

    s = suggest_registration_change(mine, offset_um=200.0, other_probes=[mine, other])

    assert s.disjoint_from == ("ProbeB",)
    assert "ProbeB sits on different sections" in s.text()
    assert "weakens the cross-probe test" in s.text()


def test_the_suggestion_never_reads_as_a_recommendation():
    s = suggest_registration_change(_probe(), offset_um=200.0)
    text = s.text()

    assert text.startswith("If this is a registration error")
    assert "Nothing here is applied" in text
    assert "cannot tell placement error from registration error" in text


def test_an_unregistered_probe_yields_no_suggestion():
    bare = ProbeSpec(label="P", type=ProbeType(name="NP2.0", n_shanks=4),
                     shanks=[Shank(index=i) for i in range(4)])

    assert suggest_registration_change(bare, offset_um=200.0) is None


def test_the_displacement_is_recorded_per_shank():
    s = suggest_registration_change(_probe(), offset_um=200.0)

    assert [sh.shank_index for sh in s.per_shank] == [0, 1, 2, 3]
    assert all(sh.magnitude_um == pytest.approx(200.0, abs=1.0) for sh in s.per_shank)
    assert all(sh.tip_section_idx == 1 for sh in s.per_shank)


def test_the_sign_moves_the_anatomy_the_same_way_the_probe_went():
    """Keep the click, change what the section maps to: the atlas follows the probe."""
    deeper = suggest_registration_change(_probe(), offset_um=200.0)
    shallower = suggest_registration_change(_probe(), offset_um=-200.0)

    assert deeper.dv_um > 0 and shallower.dv_um < 0
    assert np.isclose(deeper.dv_um, -shallower.dv_um, atol=1.0)
