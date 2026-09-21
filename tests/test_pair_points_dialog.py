"""The side-by-side landmark pairing window.

Three earlier designs failed here and each failure is pinned below. A click in the
napari canvas could not say whether it meant atlas or tissue. Slicing the raw atlas
recorded source points in a frame the warp is not defined in. And answering
auto-placed points one at a time, in a fixed order, meant a single mis-click went
straight into the warp with nowhere to put it - the outline tore, and Undo stepped
back the wrong thing because completing a pair edited it in place instead of
appending.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.schema import ManualLandmarks, Section, Slide

pytestmark = pytest.mark.qt


def _labels() -> np.ndarray:
    """A small label image standing in for a registered atlas slice."""
    labels = np.zeros((150, 200), dtype=np.int32)
    labels[30:120, 40:160] = 1
    labels[50:100, 70:130] = 2
    return labels


def _state(with_landmarks: bool = False) -> WorkflowState:
    section = Section(index=3, slide_idx=0, bbox_px=(0, 0, 200, 150), ap_order=3)
    if with_landmarks:
        section.manual_landmarks = ManualLandmarks(
            source=[[10.0, 20.0], [30.0, 40.0]],
            target=[[12.0, 22.0], [33.0, 44.0]],
        )
    state = WorkflowState()
    state.project.slides.append(Slide(image_path="s.png", sections=[section]))
    state.slide_images[0] = np.zeros((150, 200, 3), dtype=np.uint8)
    state.active_slide_idx = 0
    return state


def _dialog(qtbot, state, *, warp=True, **kwargs):
    from atlastrack.gui.widgets.pair_points_dialog import PairPointsDialog

    calls: list[dict] = []

    def _warp_labels(section, *, apply_landmarks):
        calls.append({"section": section.index, "apply_landmarks": apply_landmarks})
        return _labels()

    section = state.project.slides[0].sections[0]
    dialog = PairPointsDialog(
        state, section, warp_labels=_warp_labels if warp else None, **kwargs
    )
    dialog._confirm = lambda *a, **k: True  # never block on a modal prompt
    qtbot.addWidget(dialog)
    return dialog, section, calls


# --------------------------------------------------------------- the frame


def test_the_atlas_pane_uses_the_registration_already_computed(qtbot) -> None:
    _d, _s, calls = _dialog(qtbot, _state())
    assert calls, "the dialog must ask for the registered atlas"
    assert all(c["section"] == 3 for c in calls)


def test_the_atlas_pane_excludes_any_stored_landmark_warp(qtbot) -> None:
    """Stored source points were recorded pre-TPS, so new ones must be picked there."""
    _d, _s, calls = _dialog(qtbot, _state(with_landmarks=True))
    assert calls
    assert all(c["apply_landmarks"] is False for c in calls), calls


def test_without_a_registration_it_says_to_register_first(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state(), warp=False)
    assert "register" in dialog._status.text().lower()
    assert dialog._base_labels is None


# ------------------------------------------------- every pair starts complete


def test_auto_placed_pairs_start_on_top_of_their_atlas_point(qtbot) -> None:
    """A zero displacement warps nothing, and anchors that spot while others move.

    The previous design left the tissue half empty, so the user had to answer the
    points one at a time in a fixed order and a mis-click had nowhere to go but the
    warp.
    """
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._auto_place()

    assert len(dialog._pairs) >= 4
    for source, target in dialog._pairs:
        assert source == target, "the tissue dot must start on its atlas twin"
    assert dialog._moved_count() == 0
    assert dialog._apply_btn.isEnabled(), "enough pairs to apply, once some move"


def test_both_panes_expose_every_pair_as_a_grabbable_handle(qtbot) -> None:
    """Any dot, either side, at any time - not a fixed sequence."""
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._auto_place()

    n = len(dialog._pairs)
    assert len(dialog._atlas_pane._handles) == n
    assert len(dialog._hist_pane._handles) == n
    assert {i for i, _x, _y in dialog._hist_pane._handles} == set(range(n))


def test_dragging_a_tissue_dot_moves_only_that_target(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._auto_place()
    before = [tuple(s) for s, _t in dialog._pairs]

    dialog._move_point("target", 2, 111.0, 77.0)

    assert dialog._pairs[2][1] == (111.0, 77.0)
    assert [tuple(s) for s, _t in dialog._pairs] == before, "sources must not shift"
    assert dialog._moved_count() == 1


def test_dragging_is_clamped_to_the_image(qtbot) -> None:
    """A dot dragged off the pane would put a landmark outside the section."""
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._auto_place()

    dialog._move_point("target", 0, -50.0, 9999.0)

    x, y = dialog._pairs[0][1]
    assert 0 <= x < 200 and 0 <= y < 150, (x, y)


# ------------------------------------------------------------------- undo


def test_undo_reverses_a_drag(qtbot) -> None:
    """The old Undo popped the last list entry, which a drag never touched.

    It changed the count and nothing else on screen.
    """
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._auto_place()
    original = tuple(dialog._pairs[1][1])

    dialog._snapshot()  # what drag_started does
    dialog._move_point("target", 1, 150.0, 120.0)
    assert dialog._pairs[1][1] == (150.0, 120.0)

    dialog._undo()

    assert tuple(dialog._pairs[1][1]) == original, "undo must put the dot back"
    assert dialog._moved_count() == 0


def test_undo_reverses_auto_placement(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._auto_place()
    assert dialog._pairs

    dialog._undo()

    assert dialog._pairs == []
    assert dialog._undo_btn.isEnabled() is False, "nothing left to step back"


def test_undo_on_an_untouched_dialog_does_nothing(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._undo()
    assert dialog._pairs == []


# ------------------------------------------------------------- numbering


def test_numbering_walks_the_outer_ring_counter_clockwise_then_inward() -> None:
    """Consecutive numbers should sit next to each other on the section."""
    from atlastrack.gui.widgets.pair_points_dialog import spatial_order

    # A ring of eight at radius 100 plus two near the centre, deliberately shuffled.
    ring = [
        (100.0, 0.0), (71.0, -71.0), (0.0, -100.0), (-71.0, -71.0),
        (-100.0, 0.0), (-71.0, 71.0), (0.0, 100.0), (71.0, 71.0),
    ]
    inner = [(20.0, 0.0), (-20.0, 0.0)]
    pts = np.array([(x + 200, y + 200) for x, y in ring + inner], dtype=float)
    shuffled = pts[[9, 3, 7, 0, 5, 2, 8, 6, 1, 4]]

    order = spatial_order(shuffled)
    ordered = shuffled[order]

    # The outer ring comes first, then the interior.
    centre = shuffled.mean(axis=0)
    radii = np.hypot(*(ordered - centre).T)
    assert radii[:8].min() > radii[8:].max(), "outer ring must precede the interior"

    # And the ring itself is walked in one rotational direction, not criss-crossed.
    ring_pts = ordered[:8] - centre
    angles = np.arctan2(-ring_pts[:, 1], ring_pts[:, 0])
    assert list(angles) == sorted(angles), "the outer ring must be monotonic in angle"


def test_auto_placed_points_are_numbered_spatially(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._auto_place()

    from atlastrack.gui.widgets.pair_points_dialog import spatial_order

    sources = np.array([s for s, _t in dialog._pairs], dtype=float)
    assert spatial_order(sources) == list(range(len(sources))), (
        "the stored order must already be the spatial one"
    )


# ------------------------------------------------------------------ apply


def test_apply_writes_every_pair_including_the_unmoved_anchors(qtbot) -> None:
    """Unmoved pairs pin their part of the atlas; dropping them frees it to drift."""
    fired: list[int] = []
    state = _state()
    dialog, section, _c = _dialog(
        qtbot, state, on_section_changed=lambda s: fired.append(s.index)
    )
    section.manual_affine = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]
    dialog._auto_place()
    total = len(dialog._pairs)
    for i in range(3):
        dialog._move_point("target", i, 100.0 + i, 90.0 + i)

    dialog._apply_landmarks()

    assert section.manual_landmarks is not None
    assert len(section.manual_landmarks.source) == total, "anchors must be kept"
    assert section.manual_affine is None, "a box transform must not survive a warp"
    assert fired == [3]


def test_apply_refuses_when_nothing_has_been_dragged(qtbot) -> None:
    """Every dot on its atlas position is an identity warp - say so, do not write it."""
    state = _state()
    dialog, section, _c = _dialog(qtbot, state)
    dialog._auto_place()

    dialog._apply_landmarks()

    assert section.manual_landmarks is None
    assert "still on its atlas position" in dialog._status.text()


def test_apply_below_the_minimum_reports_without_a_modal_box(qtbot) -> None:
    state = _state()
    dialog, section, _c = _dialog(qtbot, state)
    dialog._add_pair_at(10.0, 20.0)
    dialog._move_point("target", 0, 50.0, 60.0)

    dialog._apply_landmarks()  # must return, not block

    assert section.manual_landmarks is None
    assert "at least" in dialog._status.text()


# ------------------------------------------------------------ adding pairs


def test_clicking_empty_canvas_adds_a_pair_on_both_sides(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._add_pair_at(60.0, 70.0)

    assert len(dialog._pairs) == 1
    assert dialog._pairs[0] == [(60.0, 70.0), (60.0, 70.0)]
    assert len(dialog._hist_pane._handles) == 1


def test_clicks_outside_the_image_are_ignored(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._add_pair_at(-5.0, 10.0)
    dialog._add_pair_at(10.0, 999.0)
    assert dialog._pairs == []


def test_landmarks_already_on_the_section_are_shown(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state(with_landmarks=True))

    stored = {(tuple(s), tuple(t)) for s, t in dialog._pairs}
    assert stored == {((10.0, 20.0), (12.0, 22.0)), ((30.0, 40.0), (33.0, 44.0))}
    assert dialog._moved_count() == 2


def test_clear_removes_saved_landmarks_too(qtbot) -> None:
    fired: list[int] = []
    state = _state(with_landmarks=True)
    dialog, section, _c = _dialog(
        qtbot, state, on_section_changed=lambda s: fired.append(s.index)
    )

    dialog._clear_landmarks()

    assert dialog._pairs == []
    assert section.manual_landmarks is None
    assert fired == [3]


def test_reset_transform_clears_both_kinds_of_correction(qtbot) -> None:
    state = _state(with_landmarks=True)
    dialog, section, _c = _dialog(qtbot, state)
    section.manual_affine = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]

    dialog._reset_transform()

    assert section.manual_affine is None
    assert section.manual_landmarks is None
    assert dialog._pairs == []
