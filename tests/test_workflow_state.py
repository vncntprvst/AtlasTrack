"""WorkflowState helpers: reset clears project state but keeps the atlas."""
from __future__ import annotations

import numpy as np

from histo_to_ccf.gui.workflow import WorkflowState


def test_reset_clears_project_but_keeps_atlas() -> None:
    state = WorkflowState()
    state.add_slide("slide.png", np.zeros((4, 4), dtype=np.uint8))
    state.active_slide_idx = 0
    state.active_section_idx = 2
    state.slide_bands[0] = [(0, 4)]
    state.project_path = "p.histo2ccf.json"  # type: ignore[assignment]
    state.atlas = object()  # a stand-in for a loaded BrainGlobeAtlas
    sentinel = state.atlas

    state.reset()

    assert state.project.slides == []
    assert state.slide_images == {}
    assert state.slide_bands == {}
    assert state.active_slide_idx is None
    assert state.active_section_idx is None
    assert state.project_path is None
    # The (expensive) atlas object is intentionally kept loaded.
    assert state.atlas is sentinel
