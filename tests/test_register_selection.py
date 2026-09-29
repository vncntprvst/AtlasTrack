"""Registering a chosen set of sections, and what re-registering does to old fixes.

"Register all sections" was the only way to re-register, and it kept each
section's landmarks while replacing the morph they were placed on. Landmarks are
stored as positions on the registered overlay, so laid over a new morph they
pulled an outline that already fitted back out of place: on a real slide the
dorsal edge of two well-fitted sections went 25-70 px past the tissue.
"""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from atlastrack.gui.widgets.register_panel import REGISTER_ALL_TEXT
from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.schema import (
    ManualLandmarks,
    PlaneParams,
    RegistrationResult,
    Section,
    Slide,
)

pytestmark = pytest.mark.qt


def _reg():
    return RegistrationResult(anchoring=[400.0, 0, 0, 0, 0, 0, 0, 0, 0],
                              output_size_px=(10, 10), residual=0.1)


def _state(n=4, registered=(0, 1)):
    state = WorkflowState()
    state.project.slides.append(Slide(image_path="s.png", sections=[
        Section(index=i, slide_idx=0, ap_order=i, bbox_px=(60 * i, 0, 60 * i + 60, 60),
                registration=_reg() if i in registered else None)
        for i in range(n)
    ]))
    state.active_slide_idx = 0
    state.slide_images[0] = np.zeros((60, 60 * n), dtype=np.uint8)
    return state


@pytest.fixture
def panel(qtbot):
    import napari

    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    state = _state()
    widget = RegisterPanelWidget(state, viewer)
    qtbot.addWidget(widget)
    widget._refresh_residuals()
    yield widget
    viewer.close()


def test_every_section_is_listed_registered_or_not(panel) -> None:
    t = panel._residuals_table
    assert [t.item(r, 0).text() for r in range(t.rowCount())] == ["0", "1", "2", "3"]
    assert t.item(2, 2).text() == "-"


def test_the_button_names_what_it_will_register(panel) -> None:
    assert panel._reg_btn.text() == REGISTER_ALL_TEXT
    panel.pick_section_for_registration(2)
    assert panel._reg_btn.text() == "Register selected sections (1)"
    panel.toggle_section_selection(0)
    assert panel.selected_sections() == [0, 2]
    assert panel._reg_btn.text() == "Register selected sections (2)"
    panel.clear_section_selection()
    assert panel._reg_btn.text() == REGISTER_ALL_TEXT


def test_ctrl_toggles_and_shift_takes_a_range(panel) -> None:
    panel.pick_section_for_registration(1)
    panel.extend_section_selection(3)
    assert panel.selected_sections() == [1, 2, 3]
    panel.toggle_section_selection(2)
    assert panel.selected_sections() == [1, 3]


def test_the_selection_survives_a_table_refresh(panel) -> None:
    panel.pick_section_for_registration(2)
    panel._refresh_residuals()
    assert panel.selected_sections() == [2]
    assert panel._reg_btn.text() == "Register selected sections (1)"


def test_only_the_selected_sections_reach_the_worker(panel, monkeypatch, tmp_path) -> None:
    from atlastrack.gui import workers

    seen = {}

    class _Worker:
        yielded = returned = errored = type("S", (), {"connect": lambda *a: None})()

        def start(self):
            pass

    def fake(project, atlas, section_images, transforms_dir, **kw):
        seen["images"] = sorted(section_images)
        seen["preserve"] = kw["preserve_manual"]
        return _Worker()

    monkeypatch.setattr(workers, "register_worker_progressive", fake)
    panel._state.atlas = object()
    panel._preserve_manual.setChecked(True)
    panel._register_only = [1, 3]
    images = {i: np.zeros((5, 5), np.float32) for i in range(4)}
    panel._start_register(images, tmp_path, None)

    assert seen["images"] == [1, 3]
    assert seen["preserve"] is False, "a section picked by hand is picked to be redone"


def _click(callback, viewer, *, pos, mods=(), moved=False):
    class _Event:
        button = 1
        modifiers = mods
        position = pos
        type = "mouse_press"

    event = _Event()
    gen = callback(viewer, event)
    try:
        next(gen)
        if moved:
            event.type = "mouse_move"
            next(gen)
        event.type = "mouse_release"
        next(gen)
    except (StopIteration, TypeError):
        pass


def test_canvas_clicks_pick_sections_to_register(qtbot) -> None:
    import napari

    from atlastrack.gui import app as gui_app
    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        container, viz = gui_app._build_panel(viewer)
        qtbot.addWidget(container)
        qtbot.addWidget(viz)
        panel = container.findChild(RegisterPanelWidget)
        state = panel._state
        fresh = _state()
        state.project = fresh.project
        state.active_slide_idx = 0
        state.slide_images[0] = fresh.slide_images[0]
        panel._refresh_residuals()
        callback = viewer.mouse_drag_callbacks[-1]

        _click(callback, viewer, pos=(30.0, 90.0))                      # section 1
        assert panel.selected_sections() == [1]
        assert state.active_section_idx == 1
        _click(callback, viewer, pos=(30.0, 210.0), mods=("Control",))  # + section 3
        assert panel.selected_sections() == [1, 3]
        _click(callback, viewer, pos=(30.0, 30.0), mods=("Shift",))     # range 3 .. 0
        assert panel.selected_sections() == [0, 1, 2, 3]
        assert state.active_section_idx == 1, "a modifier click does not move the section"

        _click(callback, viewer, pos=(30.0, 150.0), moved=True)         # a pan
        assert panel.selected_sections() == [0, 1, 2, 3], "panning changed the selection"

        _click(callback, viewer, pos=(30.0, 900.0))                     # empty canvas
        assert panel.selected_sections() == []
        assert panel._reg_btn.text() == REGISTER_ALL_TEXT
    finally:
        viewer.close()


def test_modifier_clicks_are_left_to_a_layer_being_edited(qtbot) -> None:
    """Ctrl+drag re-anchors a landmark; Shift+click multi-selects points."""
    import napari

    from atlastrack.gui import app as gui_app
    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget

    viewer = napari.Viewer(show=False)
    try:
        container, viz = gui_app._build_panel(viewer)
        qtbot.addWidget(container)
        qtbot.addWidget(viz)
        panel = container.findChild(RegisterPanelWidget)
        fresh = _state()
        panel._state.project = fresh.project
        panel._refresh_residuals()
        points = viewer.add_points(np.zeros((1, 2)))
        points.mode = "select"
        viewer.layers.selection.active = points

        _click(viewer.mouse_drag_callbacks[-1], viewer, pos=(30.0, 90.0), mods=("Control",))
        assert panel.selected_sections() == []
    finally:
        viewer.close()


def _section_with_fixes(index=0):
    return Section(
        index=index, slide_idx=0, ap_order=index, bbox_px=(0, 0, 80, 40),
        plane=PlaneParams(ap_um=500.0, midline_px=40.0, dorsal_surface_px=0.0,
                          pixel_size_um=25.0),
        registration=_reg(),
        manual_landmarks=ManualLandmarks(source=[[1.0, 2.0]], target=[[3.0, 4.0]]),
        manual_affine=[[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )


def test_re_registering_clears_the_old_fixes(monkeypatch, tmp_path) -> None:
    from atlastrack.gui import workers
    from atlastrack.registration import pipeline

    monkeypatch.setattr(
        pipeline, "register_section_image",
        lambda img, atlas, **kw: (_reg(), sitk.Transform(2, sitk.sitkIdentity)),
    )
    from atlastrack.atlas.planes import Anchoring

    monkeypatch.setattr(
        pipeline, "anchoring_for_section",
        lambda section, anchorings, atlas: Anchoring(*_reg().anchoring),
    )
    state = WorkflowState()
    state.project.slides.append(Slide(image_path="s.png", sections=[
        _section_with_fixes(0), _section_with_fixes(1)]))

    class _Atlas:
        resolution = (25.0, 25.0, 25.0)
        reference = np.zeros((4, 4, 4), np.float32)
        shape = (4, 4, 4)

    gen = workers.register_worker_progressive.__wrapped__(
        state.project, _Atlas(), {0: np.zeros((40, 80), np.float32)}, tmp_path / "t",
        preserve_manual=False,
    )
    messages = []
    try:
        while True:
            messages.append(next(gen))
    except StopIteration:
        pass

    redone, untouched = state.project.slides[0].sections
    assert redone.manual_landmarks is None and redone.manual_affine is None
    assert untouched.manual_landmarks is not None, "a section not re-registered keeps its fixes"
    assert any(m.get("cleared_manual") == [0] for m in messages)


def test_the_non_gui_pipeline_clears_them_too() -> None:
    from atlastrack.registration.pipeline import clear_manual_correction

    section = _section_with_fixes()
    assert clear_manual_correction(section) is True
    assert section.manual_landmarks is None and section.manual_affine is None
    assert clear_manual_correction(section) is False
