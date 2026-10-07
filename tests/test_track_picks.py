"""Picking track points: many per shank, optionally unassigned, pixels -> CCF."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.project.schema import (
    Point2D,
    ProbeSpec,
    ProbeType,
    Project,
    Shank,
    TrackPick,
)
from atlastrack.registration.pipeline import _apply_track_picks


def _shank(**kwargs) -> Shank:
    return Shank(index=0, **kwargs)


def _pick(x: float, y: float, section: int = 0) -> TrackPick:
    return TrackPick(point=Point2D(x_px=x, y_px=y), section_idx=section)


# -- schema defaults: the minimum-information case --------------------------


def test_a_shank_starts_with_no_track_points():
    shank = _shank()

    assert shank.track_picks == []
    assert shank.track_points_ccf_um == []
    assert shank.entry_estimated is False


def test_a_probe_starts_with_no_unassigned_points():
    probe = ProbeSpec(label="p", type=ProbeType(name="NP", n_shanks=1),
                      shanks=[_shank()])

    assert probe.unassigned_track_picks == []
    assert probe.unassigned_track_points_ccf_um == []


def test_a_project_without_track_points_round_trips():
    """Old projects must load and save unchanged."""
    project = Project()
    project.probes.append(
        ProbeSpec(label="p", type=ProbeType(name="NP", n_shanks=1), shanks=[_shank()])
    )

    reloaded = Project.model_validate(project.model_dump())

    assert reloaded.probes[0].shanks[0].track_picks == []


# -- pixels -> CCF ----------------------------------------------------------


def test_picks_are_mapped_to_ccf_in_order():
    shank = _shank(track_picks=[_pick(10.0, 20.0), _pick(30.0, 40.0)])

    _apply_track_picks(shank, lambda pt, _s: (pt.x_px, pt.y_px, 0.0))

    assert shank.track_points_ccf_um == [(10.0, 20.0, 0.0), (30.0, 40.0, 0.0)]


def test_a_pick_on_an_unregistered_section_is_dropped_not_guessed():
    shank = _shank(track_picks=[_pick(10.0, 20.0, section=0), _pick(30.0, 40.0, section=9)])

    _apply_track_picks(shank, lambda pt, s: None if s == 9 else (pt.x_px, pt.y_px, 0.0))

    assert shank.track_points_ccf_um == [(10.0, 20.0, 0.0)]


def test_no_picks_leaves_existing_ccf_points_alone():
    """A project whose CCF points came from elsewhere must not be silently emptied."""
    shank = _shank(track_points_ccf_um=[(1.0, 2.0, 3.0)])

    _apply_track_picks(shank, lambda pt, _s: (0.0, 0.0, 0.0))

    assert shank.track_points_ccf_um == [(1.0, 2.0, 3.0)]


def test_picks_can_span_sections():
    """A track crossing sections is what makes it 3D rather than in-plane."""
    shank = _shank(track_picks=[_pick(10.0, 20.0, section=0), _pick(11.0, 21.0, section=1)])

    _apply_track_picks(shank, lambda pt, s: (pt.x_px, pt.y_px, float(s) * 100.0))

    assert [p[2] for p in shank.track_points_ccf_um] == [0.0, 100.0]


# -- the GUI layer ----------------------------------------------------------

pytest.importorskip("qtpy")


@pytest.mark.qt
def test_the_shank_combo_still_indexes_shanks_directly(qtbot) -> None:
    """Regression guard: an 'Unassigned' row here would shift every shank by one."""
    import napari

    from atlastrack.gui.widgets.click_overlay import ClickOverlayWidget
    from atlastrack.gui.workflow import WorkflowState

    state = WorkflowState()
    state.project.probes.append(
        ProbeSpec(label="p1", type=ProbeType(name="NP", n_shanks=3),
                  shanks=[Shank(index=i) for i in range(3)])
    )
    viewer = napari.Viewer(show=False)
    try:
        widget = ClickOverlayWidget(state, viewer)
        qtbot.addWidget(widget)
        widget._refresh_probe_combo()

        assert widget._shank_combo.count() == 3
        assert widget._shank_combo.itemText(0) == "Shank 0"
    finally:
        viewer.close()


@pytest.mark.qt
def test_old_track_points_are_not_drawn_and_survive_edits(qtbot) -> None:
    """Track points saved by earlier versions are no longer shown, and editing a
    track must not silently erase them from the project."""
    import napari

    from atlastrack.gui.widgets.click_overlay import ClickOverlayWidget
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.project.schema import Section, Slide

    state = WorkflowState()
    state.project.slides.append(
        Slide(image_path="s.tif",
              sections=[Section(index=0, slide_idx=0, bbox_px=(0, 0, 100, 100))])
    )
    state.active_slide_idx = 0
    state.project.probes.append(
        ProbeSpec(label="p1", type=ProbeType(name="NP", n_shanks=1),
                  shanks=[Shank(index=0, tip_px=Point2D(x_px=50.0, y_px=90.0),
                                tip_section_idx=0,
                                entry_px=Point2D(x_px=50.0, y_px=10.0), entry_section_idx=0,
                                track_picks=[_pick(50.0, 50.0)])])
    )
    viewer = napari.Viewer(show=False)
    try:
        widget = ClickOverlayWidget(state, viewer)
        qtbot.addWidget(widget)
        widget.refresh_after_load()
        assert "Track points" not in viewer.layers
        widget._move_marker(0, 0, "tip", 52.0, 88.0, commit=True)
        assert len(state.project.probes[0].shanks[0].track_picks) == 1
    finally:
        viewer.close()


