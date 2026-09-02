"""The menu bar, and the atlas reference sheet behind Settings ▸ Atlases.

Two problems this covers. Under napari's stylesheet Qt sized the Project menu at
165 px when "Save Project As" plus "Ctrl+Shift+S" alone need 145, so the shortcut was
drawn over the label. And the atlas choice quietly moves bregma - by 346 µm for the
augmented atlas, 102 µm for the isotropic Chon/Kim - with those figures measured for
this app and written down nowhere else, so the sheet has to read them from the live
table rather than repeat them.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt


def _menu(qtbot, entries):
    """A menu with ``entries`` of (label, shortcut-or-None)."""
    from qtpy.QtWidgets import QMenu, QWidget

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = QMenu("Test", parent)
    for label, shortcut in entries:
        action = menu.addAction(label)
        if shortcut:
            action.setShortcut(shortcut)
    return menu, parent


def _columns(menu):
    """(widest label, widest shortcut) in pixels, as Qt lays them out."""
    from qtpy.QtGui import QFontMetrics

    metrics = QFontMetrics(menu.font())
    labels, shortcuts = [0], [0]
    for action in menu.actions():
        if action.isSeparator():
            continue
        labels.append(metrics.horizontalAdvance(action.text()))
        if not action.shortcut().isEmpty():
            shortcuts.append(metrics.horizontalAdvance(action.shortcut().toString()))
    return max(labels), max(shortcuts)


# ---------------------------------------------------------------------------
# Menu width
# ---------------------------------------------------------------------------


def test_a_menu_is_widened_to_clear_its_shortcuts(qtbot):
    from histo_to_ccf.gui.app import _fit_menu_width

    menu, _parent = _menu(
        qtbot,
        [("Save Project", "Ctrl+S"), ("Save Project As", "Ctrl+Shift+S"),
         ("Close Project", None)],
    )
    before = menu.sizeHint().width()

    _fit_menu_width(menu)

    label_w, shortcut_w = _columns(menu)
    assert menu.minimumWidth() > before
    # Room for both columns and a real gap between them, not merely their sum.
    assert menu.minimumWidth() >= label_w + shortcut_w + 20


def test_a_menu_without_shortcuts_is_left_alone(qtbot):
    """Nothing can collide, so widening would only make it look wrong."""
    from histo_to_ccf.gui.app import _fit_menu_width

    menu, _parent = _menu(qtbot, [("Registration", None), ("Atlases", None)])

    _fit_menu_width(menu)

    assert menu.minimumWidth() == 0


def test_the_real_project_menu_clears_its_shortcuts(qtbot):
    """The reported bug, on the menu it was reported against."""
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        menu = next(
            a.menu()
            for a in viewer.window._qt_window.menuBar().actions()
            if a.menu() and a.menu().title().replace("&", "") == "Project"
        )

        label_w, shortcut_w = _columns(menu)
        assert menu.sizeHint().width() >= label_w + shortcut_w + 20
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# Settings menu
# ---------------------------------------------------------------------------


def _menus(viewer):
    return {
        (a.menu().title() if a.menu() else a.text()).replace("&", ""): a.menu()
        for a in viewer.window._qt_window.menuBar().actions()
        if a.isVisible()
    }


def test_the_menu_bar_is_project_and_settings(qtbot):
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)

        assert set(_menus(viewer)) == {"Project", "Settings"}
    finally:
        viewer.close()


def test_settings_holds_registration_and_atlases(qtbot):
    """"Parameters" said what they are, not what they configure."""
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        items = [a.text() for a in _menus(viewer)["Settings"].actions()]

        assert items == ["Registration", "Atlases"]
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# The reference sheet
# ---------------------------------------------------------------------------


def test_the_sheet_covers_every_atlas_the_app_offers():
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html

    html = atlas_reference_html()

    for atlas_id in (
        "allen_mouse_10um", "ccfv3augmented_mouse_10um", "kim_mouse_10um",
        "kim_mouse_isotropic_20um",
    ):
        assert atlas_id in html


def test_each_bregma_is_read_from_the_live_table_not_retyped(monkeypatch):
    """If the sheet repeated the numbers it would drift from what exports use."""
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html
    from histo_to_ccf.io import ccf_coords

    monkeypatch.setattr(
        ccf_coords, "BREGMA_AP_BY_ATLAS", {**ccf_coords.BREGMA_AP_BY_ATLAS,
                                           "allen_mouse": 1234.0}
    )

    assert "1234 µm from the anterior edge" in atlas_reference_html()


def test_an_atlas_with_no_anchor_says_so_rather_than_showing_a_number(monkeypatch):
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html
    from histo_to_ccf.io import ccf_coords

    monkeypatch.setattr(ccf_coords, "BREGMA_AP_BY_ATLAS", {})

    assert "no anchor recorded" in atlas_reference_html()


def test_the_measured_shifts_are_stated_with_their_provenance():
    """They were measured for this app; a reader needs to know that, and how."""
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html

    html = atlas_reference_html()

    assert "+346" in html and "25 compact nuclei" in html
    assert "+102" in html and "811" in html


@pytest.mark.parametrize(
    ("lightness", "expected"),
    [(255, "light"), (0, "dark")],
)
def test_the_link_colour_follows_the_background(qtbot, lightness, expected):
    """Qt's default link blue is unreadable on napari's dark ground."""
    from qtpy.QtGui import QColor, QPalette
    from qtpy.QtWidgets import QWidget

    from histo_to_ccf.gui.widgets.atlas_help_dialog import (
        LINK_ON_DARK,
        LINK_ON_LIGHT,
        link_colour_for,
    )

    widget = QWidget()
    qtbot.addWidget(widget)
    palette = widget.palette()
    palette.setColor(QPalette.Window, QColor(lightness, lightness, lightness))
    widget.setPalette(palette)

    got = link_colour_for(widget)

    assert got == (LINK_ON_LIGHT if expected == "light" else LINK_ON_DARK)


def test_the_dialog_is_reused_rather_than_stacked(qtbot):
    """Clicking the menu twice must not leave two copies open."""
    from qtpy.QtWidgets import QWidget

    from histo_to_ccf.gui.widgets.atlas_help_dialog import show_atlas_reference

    parent = QWidget()
    qtbot.addWidget(parent)

    first = show_atlas_reference(parent)
    second = show_atlas_reference(parent)

    assert first is second
    first.close()


def test_the_atlas_tab_offers_the_same_sheet(qtbot):
    """The "?" beside the atlas picker and Settings ▸ Atlases are one sheet."""
    import napari

    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.gui.workflow import WorkflowState

    viewer = napari.Viewer(show=False)
    try:
        widget = AtlasBrowserWidget(WorkflowState(), viewer)
        qtbot.addWidget(widget)

        assert callable(widget._show_atlas_reference)
        widget._show_atlas_reference()
        dialog = getattr(widget.window(), "_atlas_reference_dialog", None)
        assert dialog is not None
        dialog.close()
    finally:
        viewer.close()
