"""Tests for viz/plotly3d.py - no network, no real atlas needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from histo_to_ccf.project.schema import (
    AtlasRef,
    PlaneParams,
    Point2D,
    ProbeSpec,
    ProbeType,
    Project,
    Section,
    Shank,
    Slide,
)
from histo_to_ccf.viz.plotly3d import add_probe_traces, build_figure, save_html


def _project_with_coords() -> Project:
    """Minimal project with tip/entry CCF coords on one shank."""
    section = Section(index=0, slide_idx=0, bbox_px=(0, 0, 100, 80), ap_order=0)
    slide = Slide(image_path="fake.png", sections=[section])
    shank = Shank(
        index=0,
        tip_px=Point2D(x_px=50.0, y_px=70.0),
        tip_section_idx=0,
        tip_ccf_um=(5400.0, 5700.0, 3000.0),
        entry_px=Point2D(x_px=50.0, y_px=10.0),
        entry_section_idx=0,
        entry_ccf_um=(5400.0, 5700.0, 500.0),
    )
    probe = ProbeSpec(
        label="probe1",
        type=ProbeType(name="Neuropixels 1.0", n_shanks=1),
        shanks=[shank],
    )
    return Project(atlas=AtlasRef(), slides=[slide], probes=[probe])


def test_build_figure_no_atlas() -> None:
    """build_figure runs without atlas and returns a Figure with probe traces."""
    import plotly.graph_objects as go

    project = _project_with_coords()
    fig = build_figure(project, atlas=None)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # at least the probe line


def test_probe_traces_line_style() -> None:
    import plotly.graph_objects as go

    project = _project_with_coords()
    fig = go.Figure()
    add_probe_traces(fig, project, style="line")
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter3d)


def test_probe_traces_mesh_style() -> None:
    import plotly.graph_objects as go

    project = _project_with_coords()
    fig = go.Figure()
    add_probe_traces(fig, project, style="mesh")
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Mesh3d)


def test_probe_traces_both_style() -> None:
    import plotly.graph_objects as go

    project = _project_with_coords()
    fig = go.Figure()
    add_probe_traces(fig, project, style="both")
    assert len(fig.data) == 2  # line + mesh


def test_probe_traces_skips_missing_coords() -> None:
    import plotly.graph_objects as go

    project = _project_with_coords()
    project.probes[0].shanks[0].tip_ccf_um = None  # remove tip
    fig = go.Figure()
    add_probe_traces(fig, project, style="line")
    assert len(fig.data) == 0


def test_save_html(tmp_path: Path) -> None:
    project = _project_with_coords()
    fig = build_figure(project, atlas=None)
    out = save_html(fig, tmp_path / "test.html")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_multi_shank_probe() -> None:
    """4-shank probe produces 4 traces."""
    import plotly.graph_objects as go

    shanks = [
        Shank(
            index=i,
            tip_ccf_um=(5400.0, 5700.0 + i * 250.0, 3000.0),
            entry_ccf_um=(5400.0, 5700.0 + i * 250.0, 500.0),
        )
        for i in range(4)
    ]
    probe = ProbeSpec(
        label="np2-4shank",
        type=ProbeType(name="Neuropixels 2.0 (4-shank)", n_shanks=4),
        shanks=shanks,
    )
    project = Project(atlas=AtlasRef(), slides=[], probes=[probe])
    fig = go.Figure()
    add_probe_traces(fig, project, style="line")
    assert len(fig.data) == 4
