"""Plotly probe coords: use the placed tip/entry directly (no double-counted shank
offset), and optionally re-reference the axes to bregma."""
from __future__ import annotations

import pytest

from histo_to_ccf.project.schema import ProbeSpec, ProbeType, Project, Shank
from histo_to_ccf.viz.plotly3d import build_figure

pytest.importorskip("plotly.graph_objects")


def _project_with_shank3() -> Project:
    # A 4-shank probe; shank 3 placed just right of the midline (ML 5800 > 5700).
    shank = Shank(
        index=3,
        tip_ccf_um=(10700.0, 5800.0, 5000.0),    # (AP, ML, DV)
        entry_ccf_um=(10700.0, 5800.0, 1000.0),
    )
    return Project(probes=[ProbeSpec(
        label="P", type=ProbeType(name="NP4", n_shanks=4, shank_pitch_um=250.0),
        shanks=[shank])])


def _probe_trace(fig):
    return next(t for t in fig.data if getattr(t, "x", None) is not None)


def test_probe_line_uses_placed_ml_without_shank_offset() -> None:
    fig = build_figure(_project_with_shank3(), atlas=None, bregma_relative=False)
    probe = _probe_trace(fig)
    # ML is the placed 5800 for both endpoints - NOT shifted by a shank offset
    # (the old bug added ±375 and could shove shank 3 across the midline).
    assert list(probe.x) == [5800.0, 5800.0]


def test_bregma_referencing_shifts_ml_and_ap() -> None:
    fig = build_figure(_project_with_shank3(), atlas=None, bregma_relative=True)
    probe = _probe_trace(fig)
    # ML = midline - ML_ccf (5700 - 5800) -> -100 (Paxinos sign; flipped to keep the
    # dorsal-up z-reversal from mirroring L/R).
    assert list(probe.x) == [-100.0, -100.0]
    # AP from bregma (5400 - 10700): -5300 (posterior, negative).
    assert list(probe.y) == [-5300.0, -5300.0]
