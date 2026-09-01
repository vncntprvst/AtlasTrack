"""Switching atlas on a project that already has section APs.

Section APs are stored as absolute distance from the atlas's anterior edge, so they
mean different anatomy in atlases whose volumes start in different places: Allen and
the BBP-augmented CCFv3 are 346 um apart, and the augmented volume is 38 slices
longer. Loading a different atlas recomputes nothing, so the change is invisible
unless the status line says so - and a wrong AP here is worth hundreds of microns in
every exported coordinate.
"""
from __future__ import annotations

import pytest

from histo_to_ccf.gui.workflow import WorkflowState
from histo_to_ccf.project.schema import PlaneParams, Section, Slide

pytestmark = pytest.mark.qt

ALLEN = "allen_mouse_25um"
AUGMENTED = "ccfv3augmented_mouse_25um"


class _FakeAtlas:
    """Enough of a BrainGlobeAtlas for the status line and the AP spin range."""

    def __init__(self, name, n_ap):
        self.atlas_name = name
        self.resolution = (25.0, 25.0, 25.0)
        self.root_dir = "/atlases"
        self.reference = type("_Ref", (), {"shape": (n_ap, 320, 456)})()


def _state(*, atlas_name=ALLEN, n_assigned=0, n_sections=3):
    state = WorkflowState()
    state.project.atlas.name = atlas_name
    sections = []
    for i in range(n_sections):
        sec = Section(index=i, slide_idx=0, bbox_px=(0, 0, 100, 80))
        if i < n_assigned:
            sec.plane = PlaneParams(ap_um=6000.0 + 200.0 * i)
            sec.ap_source = "manual"
        sections.append(sec)
    state.project.slides.append(Slide(image_path="s.png", sections=sections))
    return state


def _browser(qtbot, state):
    import napari

    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget

    viewer = napari.Viewer(show=False)
    widget = AtlasBrowserWidget(state, viewer)
    qtbot.addWidget(widget)
    return widget, viewer


def _load(widget, atlas_id, n_ap):
    """Drive the widget the way the load worker does when it returns."""
    from histo_to_ccf.gui.widgets.atlas_browser import _QUICK_PICKS

    widget._atlas_combo.setCurrentIndex(
        next(i for i, (_, aid) in enumerate(_QUICK_PICKS) if aid == atlas_id)
    )
    widget._on_atlas_loaded(_FakeAtlas(atlas_id, n_ap))
    return widget._atlas_status.text()


def test_switching_atlas_warns_that_assigned_aps_do_not_convert(qtbot):
    widget, viewer = _browser(qtbot, _state(atlas_name=ALLEN, n_assigned=2))
    try:
        status = _load(widget, AUGMENTED, 566)

        assert "2 section(s)" in status
        assert ALLEN in status
        assert "NOT converted" in status
    finally:
        viewer.close()


def test_the_warning_quantifies_the_shift_between_the_two_frames(qtbot):
    """A warning without the number leaves the user to guess how bad it is."""
    widget, viewer = _browser(qtbot, _state(atlas_name=ALLEN, n_assigned=1))
    try:
        status = _load(widget, AUGMENTED, 566)

        assert "+346" in status
    finally:
        viewer.close()


def test_reloading_the_same_atlas_is_not_a_switch(qtbot):
    """Re-loading the project's own atlas is routine and must stay quiet."""
    widget, viewer = _browser(qtbot, _state(atlas_name=ALLEN, n_assigned=3))
    try:
        status = _load(widget, ALLEN, 528)

        assert "NOT converted" not in status
    finally:
        viewer.close()


def test_a_project_with_no_assigned_aps_is_not_a_switch(qtbot):
    """Nothing to invalidate yet - picking an atlas first is the normal path."""
    widget, viewer = _browser(qtbot, _state(atlas_name=ALLEN, n_assigned=0))
    try:
        status = _load(widget, AUGMENTED, 566)

        assert "NOT converted" not in status
    finally:
        viewer.close()


def test_the_switch_warning_survives_an_atlas_with_no_bregma_anchor(qtbot):
    """Both caveats can apply at once; neither may swallow the other."""
    widget, viewer = _browser(qtbot, _state(atlas_name=ALLEN, n_assigned=2))
    try:
        widget._atlas_combo.setCurrentIndex(4)  # Custom ID
        widget._custom_id.setText("whs_sd_rat_39um")
        widget._on_atlas_loaded(_FakeAtlas("whs_sd_rat_39um", 512))
        status = widget._atlas_status.text()

        assert "No bregma anchor" in status
        assert "NOT converted" in status
        # No shift can be quoted when one of the two frames has no anchor.
        assert "the same anatomy sits" not in status.lower()
    finally:
        viewer.close()
