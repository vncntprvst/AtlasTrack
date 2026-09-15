"""The side-by-side landmark pairing window.

The feature it replaces failed for one reason: in the napari canvas the atlas is
drawn on top of the tissue, so a click could not say which of the two it meant and
nothing on screen could show what it had been taken as. These tests pin the part
that fixes it - the pane a click lands in *is* the answer - plus the destructive
actions, which touch saved work.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.schema import ManualLandmarks, Section, Slide

pytestmark = pytest.mark.qt


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


def _dialog(qtbot, state, **kwargs):
    from atlastrack.gui.widgets.pair_points_dialog import PairPointsDialog

    section = state.project.slides[0].sections[0]
    dialog = PairPointsDialog(state, section, **kwargs)
    dialog._confirm = lambda *a, **k: True  # never block on a modal prompt
    qtbot.addWidget(dialog)
    return dialog, section


def test_a_pair_needs_one_click_in_each_pane(qtbot) -> None:
    """One click arms; the second, from the *other* pane, completes the pair."""
    dialog, _ = _dialog(qtbot, _state())

    dialog._on_pane_clicked("atlas", 100.0, 120.0)
    assert dialog._pairs == [], "one click is not a pair"
    assert dialog._pending is not None, "the first click must be held"
    assert "tissue" in dialog._pairs_label.text(), "it must say which pane is next"

    dialog._on_pane_clicked("tissue", 110.0, 130.0)
    assert dialog._pairs == [((100.0, 120.0), (110.0, 130.0))]
    assert dialog._pending is None


def test_the_pane_decides_the_side_so_either_order_works(qtbot) -> None:
    """Source is always the atlas click and target the tissue one, whichever came first."""
    dialog, _ = _dialog(qtbot, _state())

    dialog._on_pane_clicked("tissue", 50.0, 60.0)
    dialog._on_pane_clicked("atlas", 40.0, 55.0)

    source, target = dialog._pairs[0]
    assert source == (40.0, 55.0), "the atlas click is the source"
    assert target == (50.0, 60.0), "the tissue click is the target"


def test_clicking_the_same_pane_twice_moves_the_point(qtbot) -> None:
    """Never pair a feature with itself - that is a zero-displacement landmark."""
    dialog, _ = _dialog(qtbot, _state())

    dialog._on_pane_clicked("atlas", 10.0, 10.0)
    dialog._on_pane_clicked("atlas", 80.0, 90.0)

    assert dialog._pairs == [], "two clicks in one pane must not make a pair"
    assert dialog._pending == ("atlas", (80.0, 90.0)), "the later click wins"


def test_clicks_outside_the_image_are_ignored(qtbot) -> None:
    """Both panes sit on a black surround; a click there means nothing."""
    dialog, _ = _dialog(qtbot, _state())

    dialog._on_pane_clicked("atlas", -5.0, 10.0)
    dialog._on_pane_clicked("atlas", 10.0, 999.0)

    assert dialog._pending is None
    assert dialog._pairs == []


def test_landmarks_already_on_the_section_are_shown(qtbot) -> None:
    """Opening on a section that has pairs must show them, not start empty."""
    dialog, _ = _dialog(qtbot, _state(with_landmarks=True))

    assert dialog._pairs == [((10.0, 20.0), (12.0, 22.0)), ((30.0, 40.0), (33.0, 44.0))]
    assert "2 pair" in dialog._pairs_label.text()


def test_apply_writes_the_pairs_and_drops_the_box_transform(qtbot) -> None:
    """Landmarks and a box transform are mutually exclusive; landmarks win."""
    fired: list[int] = []
    state = _state()
    dialog, section = _dialog(qtbot, state, on_section_changed=lambda s: fired.append(s.index))
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


def test_apply_is_disabled_until_there_are_enough_pairs(qtbot) -> None:
    """A thin-plate spline through fewer than four points is not worth offering."""
    from atlastrack.gui.widgets.pair_points_dialog import _MIN_PAIRS

    dialog, _ = _dialog(qtbot, _state())
    assert dialog._apply_btn.isEnabled() is False

    for n in range(_MIN_PAIRS):
        dialog._on_pane_clicked("atlas", 10.0 + n, 20.0 + n)
        dialog._on_pane_clicked("tissue", 11.0 + n, 21.0 + n)

    assert dialog._apply_btn.isEnabled() is True


def test_apply_below_the_minimum_reports_without_a_modal_box(qtbot) -> None:
    """A message box with no one to click it is what hung the GUI suite before."""
    state = _state()
    dialog, section = _dialog(qtbot, state)

    dialog._on_pane_clicked("atlas", 10.0, 20.0)
    dialog._on_pane_clicked("tissue", 11.0, 21.0)
    dialog._apply_landmarks()  # must return, not block

    assert section.manual_landmarks is None
    assert "at least" in dialog._plane_status.text()


def test_clear_removes_saved_landmarks_too(qtbot) -> None:
    """Clearing has to reach the section, or they come back on reopen."""
    fired: list[int] = []
    state = _state(with_landmarks=True)
    dialog, section = _dialog(qtbot, state, on_section_changed=lambda s: fired.append(s.index))

    dialog._clear_landmarks()

    assert dialog._pairs == []
    assert section.manual_landmarks is None
    assert fired == [3]


def test_reset_transform_clears_both_kinds_of_correction(qtbot) -> None:
    """'Reset transform' means the registered overlay, so neither may remain."""
    state = _state(with_landmarks=True)
    dialog, section = _dialog(qtbot, state)
    section.manual_affine = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]

    dialog._reset_transform()

    assert section.manual_affine is None
    assert section.manual_landmarks is None
    assert dialog._pairs == []


def test_it_opens_without_an_atlas_and_says_so(qtbot) -> None:
    """The Register panel guards this, but the window must not crash on its own."""
    dialog, _ = _dialog(qtbot, _state())

    assert dialog._state.atlas is None
    assert "atlas" in dialog._plane_status.text().lower()
    # Pairing on the tissue side still works, so nothing is half-initialised.
    dialog._on_pane_clicked("tissue", 10.0, 10.0)
    assert dialog._pending is not None


def test_undo_takes_back_the_pending_click_first(qtbot) -> None:
    """Undo should cancel a half-finished pair before destroying a finished one."""
    dialog, _ = _dialog(qtbot, _state())
    dialog._on_pane_clicked("atlas", 10.0, 20.0)
    dialog._on_pane_clicked("tissue", 11.0, 21.0)
    dialog._on_pane_clicked("atlas", 50.0, 50.0)

    dialog._undo_pair()
    assert dialog._pending is None
    assert len(dialog._pairs) == 1, "the finished pair must survive"

    dialog._undo_pair()
    assert dialog._pairs == []
