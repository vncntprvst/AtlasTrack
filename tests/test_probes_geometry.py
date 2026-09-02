"""Tests for probes/geometry.py."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.probes.geometry import (
    SHANK_PITCH_UM,
    SHANK_THICKNESS_UM,
    SHANK_TIP_LENGTH_UM,
    SHANK_WIDTH_UM,
    probe_prism_mesh,
    shank_offsets,
)


def test_constants() -> None:
    assert SHANK_WIDTH_UM == 70.0
    assert SHANK_THICKNESS_UM == 24.0
    assert SHANK_TIP_LENGTH_UM == 175.0
    assert SHANK_PITCH_UM == 250.0


def test_probe_prism_mesh_keys() -> None:
    tip = (5400.0, 5700.0, 3000.0)
    entry = (5400.0, 5700.0, 500.0)
    mesh = probe_prism_mesh(tip, entry)
    assert set(mesh.keys()) == {"x", "y", "z", "i", "j", "k"}


def test_probe_prism_mesh_vertex_count() -> None:
    tip = (5400.0, 5700.0, 3000.0)
    entry = (5400.0, 5700.0, 500.0)
    mesh = probe_prism_mesh(tip, entry)
    n_verts = len(mesh["x"])
    assert n_verts == len(mesh["y"]) == len(mesh["z"])
    assert n_verts == 10  # 8 body corners + 2 tip taper


def test_probe_prism_mesh_face_count() -> None:
    tip = (5400.0, 5700.0, 3000.0)
    entry = (5400.0, 5700.0, 500.0)
    mesh = probe_prism_mesh(tip, entry)
    n_faces = len(mesh["i"])
    assert n_faces == len(mesh["j"]) == len(mesh["k"])
    assert n_faces > 0


def test_probe_prism_mesh_degenerate_does_not_crash() -> None:
    tip = (5400.0, 5700.0, 3000.0)
    entry = (5400.0, 5700.0, 3000.0)  # same point
    mesh = probe_prism_mesh(tip, entry)
    assert len(mesh["x"]) == 10


def test_shank_offsets_single() -> None:
    offsets = shank_offsets(1)
    assert offsets[0] == pytest.approx(0.0)


def test_shank_offsets_4shank() -> None:
    offsets = shank_offsets(4, pitch_um=250.0)
    assert len(offsets) == 4
    assert offsets.mean() == pytest.approx(0.0)
    assert offsets[1] - offsets[0] == pytest.approx(250.0)


@pytest.mark.qt
def test_register_panel_creates(qtbot) -> None:
    """RegisterPanelWidget can be constructed (no viewer interaction)."""
    import napari
    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget
    from atlastrack.gui.workflow import WorkflowState

    viewer = napari.Viewer(show=False)
    try:
        state = WorkflowState()
        widget = RegisterPanelWidget(state, viewer)
        qtbot.addWidget(widget)
        widget.show()
        assert widget.isVisible()
    finally:
        viewer.close()
