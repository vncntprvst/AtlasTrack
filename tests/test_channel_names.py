"""Naming channels in the Histology tab."""
from __future__ import annotations

import pytest


@pytest.mark.qt
def test_channel_names_reach_the_project_and_align_on(qtbot) -> None:
    import napari

    from atlastrack.gui.app import _build_panel
    from atlastrack.gui.widgets.image_tools import ImageToolsWidget
    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget
    from atlastrack.project.schema import Slide

    viewer = napari.Viewer(show=False)
    try:
        panel, _ = _build_panel(viewer)
        qtbot.addWidget(panel)
        tools = panel.findChild(ImageToolsWidget)
        register = panel.findChild(RegisterPanelWidget)
        tools._state.project.slides.append(Slide(image_path="slide.tif"))

        tools._name_edits["blue"].setText("Nissl")
        tools._name_edits["red"].setText("DiI")
        tools._on_channel_names_edited()

        assert tools._state.project.slides[0].channel_names == {"red": "DiI", "blue": "Nissl"}
        items = [register._align_combo.itemText(i) for i in range(register._align_combo.count())]
        assert "Blue (Nissl)" in items and "Red (DiI)" in items

        tools._state.project.slides[0].channel_names = {"green": "GFP"}
        tools.refresh_after_load()
        assert tools._name_edits["green"].text() == "GFP"
        assert tools._name_edits["blue"].text() == ""
    finally:
        viewer.close()
