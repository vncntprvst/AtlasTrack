"""The per-section rotation control in Adjustments.

Rotation is baked into the working image, exactly like the flips, so it changes the
pixels a registration is computed against. That is the point - the exported series
and the fit then agree - but it means rotating a section that is already registered
leaves a stored fit describing pixels that have moved. The section still *has* a
registration; it is just quietly wrong, so the widget has to say so.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.project.schema import RegistrationResult, Section

pytestmark = pytest.mark.qt


def _anchoring(degrees):
    rad = math.radians(degrees)
    return [0.0, 0.0, 0.0, 0.0, math.sin(rad) * 100, math.cos(rad) * 100, 0.0, 100.0, 0.0]


def _tools(qtbot, *, predicted=None, registered=()):
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    state = WorkflowState()
    state.add_slide("s.png", np.zeros((40, 120), dtype=np.uint8))
    state.active_slide_idx = 0
    sections = [
        Section(index=i, slide_idx=0, bbox_px=(40 * i, 0, 40 * i + 40, 40), ap_order=i)
        for i in range(3)
    ]
    if predicted is not None:
        sections[0].deepslice_anchoring = _anchoring(predicted)
    for i in registered:
        sections[i].registration = RegistrationResult(
            anchoring=[0.0] * 9, output_size_px=(40, 40)
        )
    state.project.slides[0].sections = sections
    widget = ImageToolsWidget(state)
    qtbot.addWidget(widget)
    return widget, sections


# ---------------------------------------------------------------------------
# Storing the value
# ---------------------------------------------------------------------------


def test_the_spin_box_writes_to_the_selected_section_only(qtbot):
    tools, sections = _tools(qtbot)
    tools.select_section(1)

    tools._rotation_spin.setValue(12.25)

    assert sections[1].rotation_deg == pytest.approx(12.25)
    assert sections[0].rotation_deg == 0.0


def test_selecting_a_section_shows_its_own_value(qtbot):
    tools, sections = _tools(qtbot)
    sections[2].rotation_deg = -6.5

    tools.select_section(2)

    assert tools._rotation_spin.value() == pytest.approx(-6.5)


def test_showing_a_value_does_not_write_it_back(qtbot):
    """Merely looking at a section must not mark it as rotated."""
    tools, sections = _tools(qtbot)
    tools.select_section(1)

    assert sections[1].rotation_deg == 0.0


# ---------------------------------------------------------------------------
# The DeepSlice suggestion
# ---------------------------------------------------------------------------


def test_the_deepslice_button_fills_in_the_predicted_angle(qtbot):
    tools, sections = _tools(qtbot, predicted=7.5)
    tools.select_section(0)

    assert tools._rotation_ds_btn.isEnabled()
    tools._rotation_from_deepslice()

    assert sections[0].rotation_deg == pytest.approx(7.5)


def test_the_button_is_disabled_without_a_prediction(qtbot):
    """No silent no-op: a section DeepSlice never saw cannot offer an angle."""
    tools, _sections = _tools(qtbot, predicted=7.5)

    tools.select_section(1)

    assert not tools._rotation_ds_btn.isEnabled()


def test_a_prediction_is_never_applied_on_its_own(qtbot):
    """Applying it automatically would rotate every section the moment a pre-match
    ran, invalidating every registration in the project at once."""
    tools, sections = _tools(qtbot, predicted=7.5)

    tools.select_section(0)

    assert sections[0].rotation_deg == 0.0
    assert tools._rotation_spin.value() == 0.0


# ---------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------


def test_rotating_a_registered_section_warns(qtbot):
    tools, _sections = _tools(qtbot, registered=(1,))
    tools.select_section(1)
    assert tools._rotation_warning.text() == ""

    tools._rotation_spin.setValue(5.0)

    text = tools._rotation_warning.text()
    assert "Section 1" in text
    assert "re-register" in text.lower()


def test_rotating_an_unregistered_section_is_silent(qtbot):
    """Nothing to invalidate, so a warning here would just train people to ignore it."""
    tools, _sections = _tools(qtbot)
    tools.select_section(1)

    tools._rotation_spin.setValue(5.0)

    assert tools._rotation_warning.text() == ""


def test_returning_the_rotation_to_zero_clears_the_warning(qtbot):
    tools, _sections = _tools(qtbot, registered=(1,))
    tools.select_section(1)
    tools._rotation_spin.setValue(5.0)

    tools._rotation_spin.setValue(0.0)

    assert tools._rotation_warning.text() == ""


def test_the_warning_follows_the_selected_section(qtbot):
    """It describes one section, so it must not linger over another."""
    tools, sections = _tools(qtbot, registered=(1,))
    sections[1].rotation_deg = 5.0
    tools.select_section(1)
    assert tools._rotation_warning.text() != ""

    tools.select_section(2)

    assert tools._rotation_warning.text() == ""


# ---------------------------------------------------------------------------
# Selecting a section by clicking it in the canvas
# ---------------------------------------------------------------------------


def _clickable(qtbot):
    """The full app panel, with a slide of three sections laid out left to right."""
    import napari

    from histo_to_ccf.gui import app as gui_app
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    viewer = napari.Viewer(show=False)
    panel, viz_panel = gui_app._build_panel(viewer)
    qtbot.addWidget(panel)
    qtbot.addWidget(viz_panel)
    tools = panel.findChild(ImageToolsWidget)
    state = tools._state
    state.add_slide("s.png", np.zeros((60, 180), dtype=np.uint8))
    state.active_slide_idx = 0
    state.project.slides[0].sections = [
        Section(index=i, slide_idx=0, bbox_px=(60 * i, 0, 60 * i + 60, 60), ap_order=i)
        for i in range(3)
    ]
    # The panel is returned so the caller keeps it alive: dropping the last
    # Python reference deletes the C++ widget and every child with it, including
    # the combo these tests read.
    return viewer, panel, tools, state


def test_a_click_inside_a_box_finds_its_section(qtbot):
    """The outline layer paints only the border, so this hit-tests the bbox."""
    from histo_to_ccf.gui.app import _section_at

    viewer, _panel, _tools, state = _clickable(qtbot)
    try:
        assert _section_at(state, 30, 30) == 0
        assert _section_at(state, 30, 90) == 1
        assert _section_at(state, 30, 150) == 2
    finally:
        viewer.close()


def test_a_click_outside_every_box_selects_nothing(qtbot):
    from histo_to_ccf.gui.app import _section_at

    viewer, _panel, _tools, state = _clickable(qtbot)
    try:
        assert _section_at(state, 30, 500) is None
    finally:
        viewer.close()


def test_overlapping_boxes_resolve_to_the_smaller_one(qtbot):
    """The small box is the one a click was aimed at; the big one encloses it."""
    from histo_to_ccf.gui.app import _section_at

    viewer, _panel, _tools, state = _clickable(qtbot)
    try:
        state.project.slides[0].sections.append(
            Section(index=9, slide_idx=0, bbox_px=(0, 0, 180, 60), ap_order=9)
        )

        assert _section_at(state, 30, 90) == 1
    finally:
        viewer.close()


def test_clicking_selects_the_section_without_entering_edit_mode(qtbot):
    """No 'Edit boxes' first: the click works on a freshly detected slide."""
    viewer, _panel, tools, state = _clickable(qtbot)
    try:
        callback = viewer.mouse_drag_callbacks[-1]

        class _Event:
            button = 1
            modifiers = ()
            position = (30.0, 90.0)

        callback(viewer, _Event())

        assert state.active_section_idx == 1
        assert tools._section_combo.currentData() == 1
    finally:
        viewer.close()


def test_a_modified_click_is_left_to_the_other_tools(qtbot):
    """Probe picking and landmark placement use modifier clicks."""
    viewer, _panel, _tools, state = _clickable(qtbot)
    try:
        callback = viewer.mouse_drag_callbacks[-1]

        class _Event:
            button = 1
            modifiers = ("Shift",)
            position = (30.0, 90.0)

        callback(viewer, _Event())

        assert state.active_section_idx is None
    finally:
        viewer.close()
