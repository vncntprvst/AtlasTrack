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

    assert "section 2" in dlg._anchor_label.text()  # 1-based display of position 1
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


# ---------------------------------------------------------------------------
# Stack view: the series drawn as sheets spaced by AP
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_stack_view_is_a_third_page_with_its_own_spread_control(qtbot) -> None:
    state = _state(aps=[10000.0, 10100.0, 10200.0], n=3)
    dlg = _dialog(qtbot, state)

    assert not dlg._spread.isEnabled(), "spread means nothing outside the stack view"
    dlg._stack_radio.setChecked(True)
    assert dlg._stack.currentIndex() == 2
    assert dlg._spread.isEnabled()

    dlg._overlay_radio.setChecked(True)
    assert dlg._stack.currentIndex() == 1
    assert not dlg._spread.isEnabled()


@pytest.mark.qt
def test_sheets_are_positioned_by_ap_not_by_index(qtbot) -> None:
    # Sections 1 and 2 are 1000 um apart; 0 and 1 only 100 um.
    state = _state(aps=[10000.0, 10100.0, 11100.0], n=3)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)
    dlg._spread.setValue(100)  # 100 px per 100 um
    dlg._refresh_stack()

    xs = {pos: x0 for x0, _, pos in dlg._stack_pane._hit}
    # Bregma AP runs opposite to absolute AP, so the most posterior sits leftmost.
    assert xs[2] == pytest.approx(0.0)
    assert xs[1] - xs[2] == pytest.approx(1000.0)
    assert xs[0] - xs[1] == pytest.approx(100.0)


@pytest.mark.qt
def test_near_duplicate_aps_land_on_top_of_each_other(qtbot) -> None:
    """The LO_03 pair DeepSlice put 0.9 um apart must visibly coincide."""
    state = _state(aps=[10782.3, 10783.2, 11500.0], n=3)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)
    dlg._spread.setValue(60)
    dlg._refresh_stack()

    xs = {pos: x0 for x0, _, pos in dlg._stack_pane._hit}
    assert abs(xs[0] - xs[1]) < 1.0
    assert abs(xs[2] - xs[0]) > 100.0


@pytest.mark.qt
def test_spread_slider_scales_the_separation(qtbot) -> None:
    state = _state(aps=[10000.0, 10500.0], n=2)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)

    dlg._spread.setValue(50)
    dlg._refresh_stack()
    narrow = {p: x for x, _, p in dlg._stack_pane._hit}
    dlg._spread.setValue(200)
    dlg._refresh_stack()
    wide = {p: x for x, _, p in dlg._stack_pane._hit}

    assert wide[0] - wide[1] == pytest.approx(4 * (narrow[0] - narrow[1]))


@pytest.mark.qt
def test_section_without_an_ap_is_shown_beside_its_predecessor(qtbot) -> None:
    state = _state(n=3)
    state.project.slides[0].sections[0].plane = PlaneParams(ap_um=10000.0)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)
    dlg._refresh_stack()

    # All three sheets exist; the two unplaced ones are parked, not dropped.
    assert len(dlg._stack_pane._hit) == 3


@pytest.mark.qt
def test_double_clicking_a_sheet_navigates_to_that_section(qtbot) -> None:
    state = _state(aps=[10000.0, 10100.0, 10200.0], n=3)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)
    dlg._refresh_stack()

    dlg._on_stack_clicked(2)
    assert dlg._pos == 2
    # Displayed 1-based: position 2 is "Section 3 of 3".
    assert "Section 3 of 3" in dlg._sec_label.text()


@pytest.mark.qt
def test_stack_renders_without_an_atlas(qtbot) -> None:
    """The stack maps the sections' own APs, so it must not need an atlas."""
    state = _state(aps=[10000.0, 10100.0], n=2)
    state.atlas = None
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)
    dlg._refresh(fit=True)
    assert len(dlg._stack_pane._hit) == 2


@pytest.mark.qt
def test_thumbnails_are_cached_across_redraws(qtbot) -> None:
    state = _state(aps=[10000.0, 10100.0], n=2)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)

    dlg._refresh_stack()
    assert len(dlg._thumb_cache) == 2
    cached = dict(dlg._thumb_cache)
    dlg._refresh_stack()
    assert all(dlg._thumb_cache[k] is v for k, v in cached.items())


@pytest.mark.qt
def test_stack_labels_never_overlap_on_a_row(qtbot) -> None:
    """Coincident sheets are the signal; their labels must still be readable."""
    # The user's real series: three pairs only a few um apart.
    bregma = [-5475, -5550, -5600, -5607, -5908, -5920, -6025, -6033]
    state = _state(aps=[5400.0 - b for b in bregma], n=8)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)
    dlg._spread.setValue(190)
    dlg._refresh_stack()

    boxes = dlg._stack_pane._label_boxes
    assert len(boxes) == 8
    by_row: dict[int, list[tuple[float, float]]] = {}
    for x0, x1, row, _ in boxes:
        by_row.setdefault(row, []).append((x0, x1))
    for row, spans in by_row.items():
        spans.sort()
        for (_, a1), (b0, _) in zip(spans, spans[1:]):
            assert a1 <= b0, f"labels overlap on row {row}"

    # The near-coincident pairs must have been pushed apart, not left stacked.
    assert max(row for _, _, row, _ in boxes) >= 1


@pytest.mark.qt
def test_stack_label_rows_collapse_when_spread_is_wide(qtbot) -> None:
    state = _state(aps=[10000.0, 10400.0, 10800.0], n=3)
    dlg = _dialog(qtbot, state)
    dlg._stack_radio.setChecked(True)

    dlg._spread.setValue(400)
    dlg._refresh_stack()
    assert {row for _, _, row, _ in dlg._stack_pane._label_boxes} == {0}


@pytest.mark.qt
def test_view_controls_are_enabled_only_where_they_act(qtbot) -> None:
    """opacity and Atlas edges have nothing to act on in the Stack view."""
    dlg = _dialog(qtbot, _state(aps=[10000.0, 10100.0], n=2))

    # Split: the atlas pane can draw edges, but there is no blend to adjust.
    assert not dlg._spread.isEnabled()
    assert not dlg._opacity.isEnabled()
    assert dlg._edges_check.isEnabled()

    dlg._overlay_radio.setChecked(True)
    assert not dlg._spread.isEnabled()
    assert dlg._opacity.isEnabled()
    assert dlg._edges_check.isEnabled()

    # Stack shows no atlas at all - only the spread applies.
    dlg._stack_radio.setChecked(True)
    assert dlg._spread.isEnabled()
    assert not dlg._opacity.isEnabled()
    assert not dlg._edges_check.isEnabled()


# ---------------------------------------------------------------------------
# Pre-match must not silently discard hand-set APs
# ---------------------------------------------------------------------------


def _prematch_with_dialog(qtbot, monkeypatch, sources, answer):
    """Click Pre-match with ``answer`` given to the overwrite prompt.

    The DeepSlice worker is replaced: this is about the guard in front of it, and
    the real thing would load TensorFlow and run a model.
    """
    from qtpy.QtWidgets import QMessageBox

    from histo_to_ccf.gui import workers

    state = _state(aps=[1000.0 * i for i in range(8)], sources=sources)
    dlg = _dialog(qtbot, state)
    dlg._spacing_spin.setValue(500.0)  # so the "no spacing" prompt stays out of it

    asked = []

    def _question(_parent, title, text, *_args, **_kwargs):
        asked.append((title, text))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    started = []
    monkeypatch.setattr(
        workers, "deepslice_worker", lambda *a, **k: started.append(a) or _NoWorker()
    )
    dlg._prematch_deepslice()
    return dlg, asked, started


class _NoWorker:
    """A worker that is never going to run; only its wiring is exercised."""

    class _Sig:
        def connect(self, *_a, **_k):
            return None

    returned = _Sig()
    errored = _Sig()

    def start(self):
        return None


@pytest.mark.qt
def test_prematch_warns_before_overwriting_hand_set_aps(qtbot, monkeypatch) -> None:
    from qtpy.QtWidgets import QMessageBox

    sources = ["deepslice"] * 8
    sources[2] = sources[5] = "manual"

    _dlg, asked, started = _prematch_with_dialog(
        qtbot, monkeypatch, sources, QMessageBox.No
    )

    assert len(asked) == 1
    title, text = asked[0]
    assert "Overwrite" in title
    assert "2 section(s)" in text
    assert "2, 5" in text  # names which ones, so the user can judge the cost
    assert not started  # declining must not run DeepSlice


@pytest.mark.qt
def test_prematch_proceeds_when_the_overwrite_is_accepted(qtbot, monkeypatch) -> None:
    from qtpy.QtWidgets import QMessageBox

    sources = ["manual"] + ["deepslice"] * 7

    _dlg, asked, started = _prematch_with_dialog(
        qtbot, monkeypatch, sources, QMessageBox.Yes
    )

    assert len(asked) == 1
    assert started


@pytest.mark.qt
def test_prematch_does_not_prompt_when_nothing_was_set_by_hand(qtbot, monkeypatch):
    """A prompt on every run is one people learn to click through."""
    from qtpy.QtWidgets import QMessageBox

    _dlg, asked, started = _prematch_with_dialog(
        qtbot, monkeypatch, ["deepslice"] * 8, QMessageBox.No
    )

    assert asked == []
    assert started
