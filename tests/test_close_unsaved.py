"""Close Project asks only when there is something unsaved to lose.

It asked on every close, saved or not - and a question whose answer is always
Yes gets clicked through, including the one time it mattered.
"""
from __future__ import annotations

import numpy as np

from atlastrack.gui.workflow import WorkflowState
from atlastrack.project.io import save_project
from atlastrack.project.schema import Section, Slide


def _with_a_section(state: WorkflowState) -> WorkflowState:
    state.project.slides.append(Slide(image_path="s.png", sections=[
        Section(index=0, slide_idx=0, ap_order=0, bbox_px=(0, 0, 10, 10))]))
    return state


def test_a_fresh_app_has_nothing_to_lose() -> None:
    assert not WorkflowState().has_unsaved_changes()


def test_work_never_saved_is_unsaved() -> None:
    assert _with_a_section(WorkflowState()).has_unsaved_changes()


def test_saved_then_edited(tmp_path) -> None:
    state = _with_a_section(WorkflowState())
    state.project_path = tmp_path / "p.json"
    save_project(state.project, state.project_path)
    assert not state.has_unsaved_changes()

    state.project.slides[0].sections[0].ap_order = 3
    assert state.has_unsaved_changes()

    save_project(state.project, state.project_path)   # e.g. an auto-save
    assert not state.has_unsaved_changes()


def test_a_missing_file_counts_as_unsaved(tmp_path) -> None:
    state = _with_a_section(WorkflowState())
    state.project_path = tmp_path / "gone.json"
    assert state.has_unsaved_changes()


def test_close_asks_only_when_something_would_be_lost(tmp_path, monkeypatch) -> None:
    from qtpy.QtWidgets import QMessageBox

    from atlastrack.gui.app import _confirm_close

    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: asked.append(a) or QMessageBox.No,
    )
    state = _with_a_section(WorkflowState())
    state.project_path = tmp_path / "p.json"
    save_project(state.project, state.project_path)
    state.slide_images[0] = np.zeros((4, 4), np.uint8)

    assert _confirm_close(None, state) is True
    assert asked == []

    state.project.slides[0].sections[0].ap_order = 5
    assert _confirm_close(None, state) is False, "No must keep the project open"
    assert len(asked) == 1
