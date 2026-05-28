"""pytest-qt smoke tests for the M4 GUI widgets.

These tests exercise widget construction and basic interactions without
requiring a live napari viewer or file I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.project.schema import (
    AtlasRef,
    ChannelLevels,
    PlaneParams,
    Project,
    Section,
    Slide,
)

# ---------------------------------------------------------------------------
# Schema: new flip/levels fields
# ---------------------------------------------------------------------------

def test_channel_levels_defaults() -> None:
    levels = ChannelLevels()
    assert levels.low == [0.0, 0.0, 0.0]
    assert levels.high == [1.0, 1.0, 1.0]


def test_slide_flip_fields() -> None:
    slide = Slide(image_path="x.png")
    assert slide.flip_h is False
    assert slide.flip_v is False
    assert slide.levels is None


def test_section_flip_fields() -> None:
    sec = Section(index=0, slide_idx=0, bbox_px=(0, 0, 100, 80))
    assert sec.flip_h is False
    assert sec.flip_v is False
    assert sec.levels is None


def test_project_roundtrip_with_flip_levels(tmp_path) -> None:
    from histo_to_ccf.project.io import load_project, save_project

    sec = Section(
        index=0, slide_idx=0, bbox_px=(0, 0, 100, 80),
        flip_h=True, levels=ChannelLevels(low=[0.1, 0.0, 0.0], high=[0.9, 1.0, 1.0]),
    )
    slide = Slide(image_path="s.png", sections=[sec], flip_v=True)
    project = Project(atlas=AtlasRef(), slides=[slide], probes=[])
    path = tmp_path / "test.json"
    save_project(project, path)
    reloaded = load_project(path)
    assert reloaded.slides[0].flip_v is True
    assert reloaded.slides[0].sections[0].flip_h is True
    assert reloaded.slides[0].sections[0].levels.high[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_add_slide() -> None:
    state = WorkflowState()
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    idx = state.add_slide("path/to/slide.png", img)
    assert idx == 0
    assert len(state.project.slides) == 1
    assert state.slide_images[0] is img
    idx2 = state.add_slide("path/to/slide2.png", img)
    assert idx2 == 1


def test_workflow_state_atlas_property() -> None:
    state = WorkflowState()
    assert state.atlas is None
    state.atlas = "mock_atlas"
    assert state.atlas == "mock_atlas"


# ---------------------------------------------------------------------------
# Widget smoke tests (require Qt via pytest-qt)
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_slide_loader_widget_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.slide_loader import SlideLoaderWidget

    state = WorkflowState()
    widget = SlideLoaderWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_image_tools_widget_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    state = WorkflowState()
    widget = ImageToolsWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_probe_picker_adds_probe(qtbot) -> None:
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget

    state = WorkflowState()
    widget = ProbePickerWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    widget._add_probe()
    assert len(state.project.probes) == 1
    assert state.project.probes[0].label == "probe1"


@pytest.mark.qt
def test_ordering_panel_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget

    state = WorkflowState()
    widget = OrderingPanelWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_save_panel_creates(qtbot) -> None:
    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget

    state = WorkflowState()
    widget = SavePanelWidget(state)
    qtbot.addWidget(widget)
    widget.show()
    assert widget.isVisible()


@pytest.mark.qt
def test_image_tools_flip_h_updates_state(qtbot) -> None:
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget

    state = WorkflowState()
    img = np.arange(6, dtype=np.uint8).reshape(2, 3)
    state.add_slide("fake.png", img.copy())
    state.active_slide_idx = 0

    widget = ImageToolsWidget(state)
    qtbot.addWidget(widget)
    # Flip H: image should be flipped and schema flag set
    widget._flip_slide("h")
    assert state.project.slides[0].flip_h is True
    np.testing.assert_array_equal(state.slide_images[0], np.fliplr(img))


@pytest.mark.qt
def test_save_panel_saves_json(qtbot, tmp_path) -> None:
    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget
    from histo_to_ccf.project.io import load_project

    state = WorkflowState()
    out_path = tmp_path / "out.json"
    widget = SavePanelWidget(state)
    qtbot.addWidget(widget)
    widget._path_edit.setText(str(out_path))
    widget._save()
    assert out_path.exists()
    p = load_project(out_path)
    assert p.version == 1
