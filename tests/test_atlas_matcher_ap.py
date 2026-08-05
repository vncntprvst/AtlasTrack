"""Atlas matcher AP handling: source badge, live order check, irregular spacing.

These cover the ways the matcher used to mislead: it reported a broken AP series
only once (right after a pre-match, so later hand-edits went unflagged), gave no
sign of which APs were predicted versus set by hand, forced uniform spacing on
unevenly sampled series, and kept DeepSlice's planes in memory only.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.gui.workflow import WorkflowState, crop_fingerprint
from histo_to_ccf.project.schema import PlaneParams, Section

pytest.importorskip("qtpy")


class _FakeAtlas:
    resolution = (25.0, 25.0, 25.0)
    reference = np.zeros((600, 80, 90), dtype=np.uint16)
    annotation = np.zeros((600, 80, 90), dtype=np.uint32)


def _state(aps=None, sources=None, slide_numbers=None, n=8) -> WorkflowState:
    state = WorkflowState()
    state.add_slide("s.png", np.zeros((60, 700), dtype=np.uint8))
    state.active_slide_idx = 0
    slide = state.project.slides[0]
    for i in range(n):
        slide.sections.append(
            Section(
                index=i,
                slide_idx=0,
                bbox_px=(i * 80, 0, i * 80 + 78, 50),
                ap_order=i,
                slide_number=None if slide_numbers is None else slide_numbers[i],
                plane=None if aps is None else PlaneParams(ap_um=aps[i]),
                ap_source=None if sources is None else sources[i],
            )
        )
    state.atlas = _FakeAtlas()
    return state


def _dialog(qtbot, state):
    from histo_to_ccf.gui.widgets.atlas_matcher import AtlasMatcherDialog

    dlg = AtlasMatcherDialog(state)
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# AP source badge
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_badge_reports_each_section_ap_source(qtbot) -> None:
    state = _state(
        aps=[10000.0, 10100.0], sources=["deepslice", "manual"], n=2
    )
    dlg = _dialog(qtbot, state)

    dlg._refresh()
    assert "DeepSlice" in dlg._sec_label.text()

    dlg._step_section(1)
    assert "set by hand" in dlg._sec_label.text()


@pytest.mark.qt
def test_badge_says_not_set_without_a_plane(qtbot) -> None:
    dlg = _dialog(qtbot, _state(n=2))
    dlg._refresh()
    assert "not set" in dlg._sec_label.text()


@pytest.mark.qt
def test_assigning_records_the_source(qtbot) -> None:
    state = _state(n=3)
    dlg = _dialog(qtbot, state)
    sections = state.project.slides[0].sections

    dlg._ap_spin.setValue(-2000.0)
    dlg._assign_current()
    assert sections[0].ap_source == "manual"

    dlg._spacing_spin.setValue(100.0)
    dlg._assign_all()
    assert [s.ap_source for s in sections] == ["even_spacing"] * 3


# ---------------------------------------------------------------------------
# Live order check
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_order_strip_flags_a_reversal_created_by_hand_edits(qtbot) -> None:
    """The real LO_03 case: a clean pre-match, then hand-edits break the order."""
    aps = [10782.3, 10783.2, 11056.1, 11058.9, 11357.6]
    state = _state(aps=aps, sources=["deepslice"] * 5, n=5)
    dlg = _dialog(qtbot, state)
    dlg._refresh()
    assert "nearly identical" in dlg._order_label.text()
    assert "reverses" not in dlg._order_label.text()

    # Hand-set the first section far anterior, creating a reversal.
    sections = state.project.slides[0].sections
    sections[0].plane = PlaneParams(ap_um=11500.0)
    dlg._refresh()

    assert dlg._order_label.isVisibleTo(dlg)
    assert "reverses" in dlg._order_label.text()


@pytest.mark.qt
def test_order_strip_reports_a_clean_series(qtbot) -> None:
    state = _state(aps=[10000.0 + 100 * i for i in range(5)], n=5)
    dlg = _dialog(qtbot, state)
    dlg._refresh()
    assert "looks consistent" in dlg._order_label.text()


@pytest.mark.qt
def test_order_strip_hidden_when_too_few_sections_to_judge(qtbot) -> None:
    state = _state(aps=[10000.0, 10100.0], n=2)
    dlg = _dialog(qtbot, state)
    dlg._refresh()
    assert not dlg._order_label.isVisibleTo(dlg)


# ---------------------------------------------------------------------------
# Irregular spacing via slide_number
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_assign_all_spaces_by_slide_number_gaps(qtbot) -> None:
    # Sections cut every slide, then a jump: 1, 2, 3, then slide 8.
    state = _state(slide_numbers=[1, 2, 3, 8], n=4)
    dlg = _dialog(qtbot, state)
    dlg._ap_spin.setValue(0.0)
    dlg._set_anchor()
    dlg._spacing_spin.setValue(100.0)

    dlg._assign_all()

    aps = [s.plane.ap_um for s in state.project.slides[0].sections]
    steps = [b - a for a, b in zip(aps, aps[1:])]
    assert steps == pytest.approx([100.0, 100.0, 500.0]), "gap must span 5 slides"


@pytest.mark.qt
def test_assign_all_without_slide_numbers_is_evenly_spaced(qtbot) -> None:
    state = _state(n=4)
    dlg = _dialog(qtbot, state)
    dlg._ap_spin.setValue(0.0)
    dlg._set_anchor()
    dlg._spacing_spin.setValue(100.0)

    dlg._assign_all()

    aps = [s.plane.ap_um for s in state.project.slides[0].sections]
    steps = [b - a for a, b in zip(aps, aps[1:])]
    assert steps == pytest.approx([100.0, 100.0, 100.0])


@pytest.mark.qt
def test_reference_label_shows_the_anchor(qtbot) -> None:
    state = _state(n=3)
    dlg = _dialog(qtbot, state)
    assert "none set" in dlg._anchor_label.text()

    dlg._step_section(1)
    dlg._ap_spin.setValue(-1500.0)
    dlg._set_anchor()

    assert "section 1" in dlg._anchor_label.text()
    assert "-1500" in dlg._anchor_label.text()


# ---------------------------------------------------------------------------
# DeepSlice planes survive a reload
# ---------------------------------------------------------------------------

def test_seed_deepslice_cache_restores_planes_and_fingerprints() -> None:
    state = _state(n=3)
    sections = state.project.slides[0].sections
    anchoring = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    fp = crop_fingerprint(np.ones((4, 5), dtype=np.uint8))
    sections[1].deepslice_anchoring = list(anchoring)
    sections[1].deepslice_fingerprint = list(fp)

    n = state.seed_deepslice_cache_from_project()

    assert n == 1
    assert state.deepslice_anchorings[1] == anchoring
    assert state.deepslice_fingerprints[1] == fp


def test_seed_deepslice_cache_is_a_noop_without_stored_planes() -> None:
    state = _state(n=3)
    assert state.seed_deepslice_cache_from_project() == 0
    assert state.deepslice_anchorings == {}


def test_deepslice_fields_survive_a_project_round_trip(tmp_path) -> None:
    from histo_to_ccf.project.io import load_project, save_project

    state = _state(aps=[10000.0], sources=["deepslice"], n=1)
    section = state.project.slides[0].sections[0]
    section.deepslice_anchoring = [float(i) for i in range(9)]
    section.deepslice_fingerprint = crop_fingerprint(np.ones((3, 4), dtype=np.uint8))

    path = tmp_path / "proj.json"
    save_project(state.project, path)
    reloaded = load_project(path)

    got = reloaded.slides[0].sections[0]
    assert got.ap_source == "deepslice"
    assert got.deepslice_anchoring == section.deepslice_anchoring
    assert got.deepslice_fingerprint == section.deepslice_fingerprint


def test_crop_fingerprint_is_json_safe_and_still_discriminates() -> None:
    import json

    a = np.ones((4, 5), dtype=np.uint8)
    b = np.ones((5, 4), dtype=np.uint8)   # same sum, different shape
    c = a.copy()
    c[0, 0] = 9                           # same shape, different content

    assert json.loads(json.dumps(crop_fingerprint(a))) == crop_fingerprint(a)
    assert crop_fingerprint(a) == crop_fingerprint(a.copy())
    assert crop_fingerprint(a) != crop_fingerprint(b)
    assert crop_fingerprint(a) != crop_fingerprint(c)
