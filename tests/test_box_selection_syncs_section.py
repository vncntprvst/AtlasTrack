"""Selecting a bounding box points the Adjustments "Section:" dropdown at it.

The box already *is* the section - it carries the section index as a shapes feature -
so making the user re-pick it in a dropdown adds a step whose only possible outcome is
getting it wrong. Adjusting the wrong section is silent: a flip or a levels change
lands on a neighbour and nothing says so.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.schema import Section, Slide

pytestmark = pytest.mark.qt


def _state(n_sections=3):
    state = WorkflowState()
    sections = [
        Section(index=i, slide_idx=0, ap_order=i,
                bbox_px=(10 + 100 * i, 10, 90 + 100 * i, 90))
        for i in range(n_sections)
    ]
    state.project.slides.append(Slide(image_path="s.png", sections=sections))
    state.active_slide_idx = 0
    state.slide_images[0] = np.zeros((200, 400), dtype=np.uint8)
    return state


def _widgets(qtbot, state):
    """Build the two widgets wired the way ``app._build_panel`` wires them."""
    import napari

    from atlastrack.gui.widgets.image_tools import ImageToolsWidget
    from atlastrack.gui.widgets.slide_loader import SlideLoaderWidget

    viewer = napari.Viewer(show=False)
    tools = ImageToolsWidget(state)
    loader = SlideLoaderWidget(
        state, viewer=viewer, on_section_selected=tools.select_section
    )
    qtbot.addWidget(tools)
    qtbot.addWidget(loader)
    loader._edit_boxes()
    return loader, tools, viewer


def _shape_of_section(loader, section_index):
    """Row in the shapes layer carrying ``section_index``."""
    idxs = [int(v) for v in loader._box_layer.features["idx"]]
    return idxs.index(section_index)


def test_selecting_a_box_selects_that_section_in_the_dropdown(qtbot):
    state = _state()
    loader, tools, viewer = _widgets(qtbot, state)
    try:
        loader._box_layer.selected_data = {_shape_of_section(loader, 2)}

        assert tools._section_combo.currentData() == 2
        assert state.active_section_idx == 2
    finally:
        viewer.close()


def test_it_follows_the_box_rather_than_the_row_order(qtbot):
    """Shapes are built in AP order, so row number and section index can differ."""
    state = _state()
    state.project.slides[0].sections[0].ap_order = 9  # section 0 is now last
    loader, tools, viewer = _widgets(qtbot, state)
    try:
        row = _shape_of_section(loader, 0)
        assert row != 0  # the ordering really did diverge

        loader._box_layer.selected_data = {row}

        assert tools._section_combo.currentData() == 0
    finally:
        viewer.close()


def test_selecting_several_boxes_leaves_the_dropdown_alone(qtbot):
    """No single section is named, so picking one of them would be a guess."""
    state = _state()
    loader, tools, viewer = _widgets(qtbot, state)
    try:
        loader._box_layer.selected_data = {_shape_of_section(loader, 1)}
        before = tools._section_combo.currentData()

        loader._box_layer.selected_data = {0, 1, 2}

        assert tools._section_combo.currentData() == before
    finally:
        viewer.close()


def test_clearing_the_selection_leaves_the_dropdown_alone(qtbot):
    """Deselecting is not a request to adjust some other section."""
    state = _state()
    loader, tools, viewer = _widgets(qtbot, state)
    try:
        loader._box_layer.selected_data = {_shape_of_section(loader, 1)}

        loader._box_layer.selected_data = set()

        assert tools._section_combo.currentData() == 1
        assert state.active_section_idx == 1
    finally:
        viewer.close()


def test_a_freshly_drawn_box_is_ignored_until_it_becomes_a_section(qtbot):
    """New shapes carry idx -1 until the data handler turns them into sections."""
    state = _state()
    loader, tools, viewer = _widgets(qtbot, state)
    try:
        loader._box_layer.selected_data = {_shape_of_section(loader, 1)}
        features = loader._box_layer.features
        features.loc[0, "idx"] = -1
        loader._box_layer.features = features

        loader._box_layer.selected_data = {0}

        assert tools._section_combo.currentData() == 1  # unchanged
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# select_section on its own
# ---------------------------------------------------------------------------


def test_select_section_fills_a_dropdown_that_was_never_populated(qtbot):
    """The list is only built when section scope is chosen; a click can come first."""
    from atlastrack.gui.widgets.image_tools import ImageToolsWidget

    state = _state()
    tools = ImageToolsWidget(state)
    qtbot.addWidget(tools)
    assert tools._section_combo.count() == 0

    assert tools.select_section(2) is True
    assert tools._section_combo.currentData() == 2


def test_select_section_reports_an_index_it_cannot_offer(qtbot):
    from atlastrack.gui.widgets.image_tools import ImageToolsWidget

    state = _state()
    tools = ImageToolsWidget(state)
    qtbot.addWidget(tools)

    assert tools.select_section(99) is False


def test_leaving_box_edit_restores_the_section_numbers(qtbot):
    """Entering edit mode hides the numbers; leaving it must bring them back.

    Before the button became a toggle there was no "leaving" at all: the boxes
    stayed yellow and the section numbers, hidden on entry, never returned - which
    read as "the numbers have been removed from the app".
    """
    from atlastrack.gui.widgets.slide_loader import (
        BOX_EDIT_ACTIVE_TEXT,
        BOX_EDIT_IDLE_TEXT,
    )

    state = _state()
    loader, _tools, viewer = _widgets(qtbot, state)  # already in edit mode
    try:
        assert loader._boxes_btn.isChecked(), "the button must show the mode is live"
        assert loader._boxes_btn.text() == BOX_EDIT_ACTIVE_TEXT
        assert "Edit boxes 0" in viewer.layers

        loader._on_edit_boxes_toggled(False)

        assert "Edit boxes 0" not in viewer.layers, "the yellow boxes must go"
        assert "Section numbers 0" in viewer.layers, "the numbers must come back"
        assert viewer.layers["Section numbers 0"].visible is True
        assert loader._boxes_btn.isChecked() is False
        assert loader._boxes_btn.text() == BOX_EDIT_IDLE_TEXT
    finally:
        viewer.close()


def test_escape_leaves_box_edit_mode(qtbot):
    """The canvas holds keyboard focus while dragging, so Esc is bound on the viewer."""
    state = _state()
    loader, _tools, viewer = _widgets(qtbot, state)
    try:
        assert loader._escape_bound, "Esc must be bound while the mode is live"
        # napari keys its keymap by KeyBinding objects, not strings.
        escape = next(k for k in viewer.keymap if str(k) == "Escape")
        viewer.keymap[escape](viewer)

        assert "Edit boxes 0" not in viewer.layers
        assert loader._boxes_btn.isChecked() is False
    finally:
        viewer.close()


def test_entering_box_edit_never_hides_the_section_numbers(qtbot):
    """Nothing may hide the numbers - there is no state where they should be gone.

    They used to be hidden on entry and restored only on an explicit exit, so
    anyone who clicked "Edit boxes" and then simply moved on - switched tab, loaded
    the atlas, showed the overlay - lost the numbers for the rest of the session
    with nothing on screen to suggest why.
    """
    from atlastrack.gui.app import _update_section_numbers

    state = _state()
    loader, _tools, viewer = _widgets(qtbot, state)  # already in edit mode
    try:
        _update_section_numbers(viewer, state, 0)
        assert viewer.layers["Section numbers 0"].visible is True

        # Walk away without leaving the mode.
        state.active_slide_idx = 0
        assert viewer.layers["Section numbers 0"].visible is True
        assert "Edit boxes 0" in viewer.layers, "still in edit mode"
    finally:
        viewer.close()


def test_the_editable_boxes_do_not_draw_their_own_numbers(qtbot):
    """One source of numbers. Two copies at the same point looked like smeared text."""
    state = _state()
    loader, _tools, viewer = _widgets(qtbot, state)
    try:
        layer = viewer.layers["Edit boxes 0"]
        values = list(getattr(layer.text, "values", []) or [])
        assert all(not str(v).strip() for v in values), f"unexpected text: {values}"
    finally:
        viewer.close()
