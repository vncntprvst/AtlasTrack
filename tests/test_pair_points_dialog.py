"""The side-by-side landmark pairing window.

The feature it replaces failed for two reasons, both pinned here. In the napari
canvas a click could not say whether it meant the atlas or the tissue. And the
first version of this window sliced the **raw** atlas at the section's plane, so
every source point was recorded in a frame the warp is not defined in - the
outline sat visibly off the tissue and lurched once a few pairs were applied.
``ManualLandmarks.source`` is a position on the *registered* overlay, so that is
what the atlas pane must draw.
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


def test_the_atlas_pane_uses_the_registration_already_computed(qtbot) -> None:
    """It must ask the panel for the warped atlas, not slice the raw one."""
    _dialog_, _section, calls = _dialog(qtbot, _state())
    assert calls, "the dialog must ask for the registered atlas"
    assert all(c["section"] == 3 for c in calls)


def test_the_atlas_pane_excludes_any_stored_landmark_warp(qtbot) -> None:
    """Stored source points were recorded pre-TPS, so new ones must be picked there.

    Asking with ``apply_landmarks=True`` would show an already-bent atlas and
    silently record source points in a frame the stored ones are not in.
    """
    _dialog_, _section, calls = _dialog(qtbot, _state(with_landmarks=True))
    assert calls
    assert all(c["apply_landmarks"] is False for c in calls), calls


def test_a_pair_needs_one_click_in_each_pane(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._on_pane_clicked("atlas", 100.0, 120.0)
    assert dialog._complete_pairs() == [], "one click is not a pair"
    assert dialog._next_unmatched() == 0, "the atlas point waits for its tissue match"

    dialog._on_pane_clicked("tissue", 110.0, 130.0)
    assert dialog._complete_pairs() == [((100.0, 120.0), (110.0, 130.0))]


def test_the_pane_decides_the_side_so_either_order_works(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._on_pane_clicked("tissue", 50.0, 60.0)
    dialog._on_pane_clicked("atlas", 40.0, 55.0)

    source, target = dialog._complete_pairs()[0]
    assert source == (40.0, 55.0), "the atlas click is the source"
    assert target == (50.0, 60.0), "the tissue click is the target"


def test_several_atlas_points_queue_up_and_are_matched_in_order(qtbot) -> None:
    """This is the auto-place workflow, done by hand: place, then match each."""
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._on_pane_clicked("atlas", 10.0, 10.0)
    dialog._on_pane_clicked("atlas", 80.0, 90.0)
    assert dialog._complete_pairs() == []
    assert dialog._next_unmatched() == 0

    dialog._on_pane_clicked("tissue", 12.0, 12.0)
    assert dialog._next_unmatched() == 1, "the queue advances"
    dialog._on_pane_clicked("tissue", 82.0, 92.0)

    assert dialog._complete_pairs() == [
        ((10.0, 10.0), (12.0, 12.0)),
        ((80.0, 90.0), (82.0, 92.0)),
    ]


def test_auto_place_seeds_atlas_points_awaiting_a_tissue_click(qtbot) -> None:
    """'Why not apply some automatically' - so there is something to work from."""
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._auto_place()

    assert len(dialog._pairs) >= 4, "auto-placement must produce a usable set"
    assert dialog._complete_pairs() == [], "each still needs its tissue match"
    assert dialog._next_unmatched() == 0
    assert "tissue" in dialog._hint.text().lower()


def test_clicks_outside_the_image_are_ignored(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())

    dialog._on_pane_clicked("atlas", -5.0, 10.0)
    dialog._on_pane_clicked("atlas", 10.0, 999.0)

    assert dialog._pairs == []
    assert dialog._pending_tissue is None


def test_landmarks_already_on_the_section_are_shown(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state(with_landmarks=True))

    assert dialog._complete_pairs() == [
        ((10.0, 20.0), (12.0, 22.0)),
        ((30.0, 40.0), (33.0, 44.0)),
    ]
    assert "2 complete" in dialog._pairs_label.text()


def test_apply_writes_the_pairs_and_drops_the_box_transform(qtbot) -> None:
    fired: list[int] = []
    state = _state()
    dialog, section, _c = _dialog(
        qtbot, state, on_section_changed=lambda s: fired.append(s.index)
    )
    section.manual_affine = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]

    for n in range(4):
        dialog._on_pane_clicked("atlas", 10.0 + n, 20.0 + n)
        dialog._on_pane_clicked("tissue", 11.0 + n, 21.0 + n)
    dialog._apply_landmarks()

    assert section.manual_landmarks is not None
    assert len(section.manual_landmarks.source) == 4
    assert section.manual_landmarks.source[0] == [10.0, 20.0]
    assert section.manual_landmarks.target[0] == [11.0, 21.0]
    assert section.manual_affine is None, "a box transform must not survive a warp"
    assert fired == [3], "the panel must be told to re-render and save"


def test_half_finished_points_are_not_written(qtbot) -> None:
    """Auto-placed points that were never matched must not become landmarks."""
    state = _state()
    dialog, section, _c = _dialog(qtbot, state)

    for n in range(4):
        dialog._on_pane_clicked("atlas", 10.0 + n, 20.0 + n)
        dialog._on_pane_clicked("tissue", 11.0 + n, 21.0 + n)
    dialog._on_pane_clicked("atlas", 99.0, 99.0)  # left unmatched
    dialog._apply_landmarks()

    assert len(section.manual_landmarks.source) == 4


def test_apply_is_disabled_until_there_are_enough_pairs(qtbot) -> None:
    from atlastrack.gui.widgets.pair_points_dialog import _MIN_PAIRS

    dialog, _s, _c = _dialog(qtbot, _state())
    assert dialog._apply_btn.isEnabled() is False

    for n in range(_MIN_PAIRS):
        dialog._on_pane_clicked("atlas", 10.0 + n, 20.0 + n)
        dialog._on_pane_clicked("tissue", 11.0 + n, 21.0 + n)

    assert dialog._apply_btn.isEnabled() is True


def test_apply_below_the_minimum_reports_without_a_modal_box(qtbot) -> None:
    """A message box with no one to click it is what hung the GUI suite before."""
    state = _state()
    dialog, section, _c = _dialog(qtbot, state)

    dialog._on_pane_clicked("atlas", 10.0, 20.0)
    dialog._on_pane_clicked("tissue", 11.0, 21.0)
    dialog._apply_landmarks()  # must return, not block

    assert section.manual_landmarks is None
    assert "at least" in dialog._status.text()


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


def test_without_a_registration_it_says_to_register_first(qtbot) -> None:
    """Rather than drawing a raw atlas plane that would not line up with anything."""
    dialog, _s, _c = _dialog(qtbot, _state(), warp=False)

    assert "register" in dialog._status.text().lower()
    assert dialog._base_labels is None


def test_undo_takes_back_the_pending_click_first(qtbot) -> None:
    dialog, _s, _c = _dialog(qtbot, _state())
    dialog._on_pane_clicked("atlas", 10.0, 20.0)
    dialog._on_pane_clicked("tissue", 11.0, 21.0)
    dialog._on_pane_clicked("tissue", 50.0, 50.0)  # no queue left, so it pends

    assert dialog._pending_tissue is not None
    dialog._undo()
    assert dialog._pending_tissue is None
    assert len(dialog._complete_pairs()) == 1, "the finished pair must survive"

    dialog._undo()
    assert dialog._pairs == []
