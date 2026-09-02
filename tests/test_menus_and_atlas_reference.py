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

        assert set(_menus(viewer)) == {"Project", "Settings", "Help"}
    finally:
        viewer.close()


def test_settings_holds_only_registration(qtbot):
    """"Parameters" said what they are, not what they configure."""
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        items = [a.text() for a in _menus(viewer)["Settings"].actions()]

        assert items == ["Registration"]
    finally:
        viewer.close()


def test_help_holds_the_three_documents(qtbot):
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        items = [a.text() for a in _menus(viewer)["Help"].actions()]

        assert items == ["Manual", "Tutorial", "Atlases"]
    finally:
        viewer.close()


def test_napari_own_help_menu_is_hidden(qtbot):
    """napari's is "&Help" and ours is "Help": matching by title keeps both."""
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        titles = [
            (a.menu().title() if a.menu() else a.text()).replace("&", "")
            for a in viewer.window._qt_window.menuBar().actions()
            if a.isVisible()
        ]

        assert titles.count("Help") == 1
    finally:
        viewer.close()


def test_menus_are_kept_by_identity_not_by_name(qtbot):
    """A same-named menu we did not build must still be hidden."""
    from qtpy.QtWidgets import QMainWindow, QMenu

    from histo_to_ccf.gui.app import _keep_only_menus

    window = QMainWindow()
    qtbot.addWidget(window)
    ours = QMenu("Help", window)
    theirs = QMenu("Help", window)
    window.menuBar().addMenu(ours)
    window.menuBar().addMenu(theirs)

    class _Viewer:
        class window:  # mimics viewer.window._qt_window
            pass

    viewer = _Viewer()
    viewer.window._qt_window = window
    _keep_only_menus(viewer, (ours,))

    visible = [a.menu() for a in window.menuBar().actions() if a.isVisible()]
    assert visible == [ours]


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


def test_the_measured_shifts_are_stated():
    """Both atlases move bregma by amounts big enough to matter, and neither figure
    is published anywhere - the sheet is the only place a user meets them. How they
    were measured is kept as a source comment beside each entry rather than shown."""
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html

    html = atlas_reference_html()

    assert "+346" in html
    assert "+102" in html


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


# ---------------------------------------------------------------------------
# The docked help panel
# ---------------------------------------------------------------------------


def test_the_panel_renders_the_repository_documents(qtbot):
    """Rendered, not duplicated: a copy in the GUI is a copy that goes stale."""
    from histo_to_ccf.gui.widgets.help_panel import HelpPanelWidget, find_doc

    assert find_doc("MANUAL.md") is not None
    assert find_doc("TUTORIAL.md") is not None

    panel = HelpPanelWidget()
    qtbot.addWidget(panel)

    assert panel.pages == ["Manual", "Tutorial", "Atlases"]
    for page in panel.pages:
        assert len(panel._views[page].toPlainText()) > 500


def test_a_missing_document_says_which_one(qtbot, monkeypatch):
    """A blank tab would read as "the manual is empty"."""
    from histo_to_ccf.gui.widgets import help_panel as module

    monkeypatch.setattr(module, "find_doc", lambda _name: None)
    panel = module.HelpPanelWidget()
    qtbot.addWidget(panel)

    assert "MANUAL.md was not found" in panel._views["Manual"].toPlainText()


def test_show_page_selects_the_tab_and_reports_an_unknown_one(qtbot):
    from histo_to_ccf.gui.widgets.help_panel import ATLASES, HelpPanelWidget

    panel = HelpPanelWidget()
    qtbot.addWidget(panel)

    assert panel.show_page(ATLASES) is True
    assert panel._tabs.tabText(panel._tabs.currentIndex()) == ATLASES
    assert panel.show_page("Nope") is False


def _app(qtbot):
    """The app with the help tab installed, as launch() assembles it."""
    import napari

    from histo_to_ccf.gui.app import _build_panel, _install_help_tab

    viewer = napari.Viewer(show=False)
    panel, viz_panel = _build_panel(viewer)
    qtbot.addWidget(panel)
    qtbot.addWidget(viz_panel)
    tabs = _install_help_tab(viewer, panel.help_panel)
    return viewer, panel, tabs


def test_help_lives_in_the_central_pane_beside_the_canvas(qtbot):
    """Not a dock. A dock sits in the sidebar and stays there over the project;
    the manual belongs where the project is, so switching is one gesture."""
    viewer, panel, tabs = _app(qtbot)
    try:
        assert tabs is not None
        assert [tabs.tabText(i) for i in range(tabs.count())] == ["Project", "Help"]
        assert tabs.indexOf(panel.help_panel) == 1
        assert viewer.window._qt_window.centralWidget() is tabs
    finally:
        viewer.close()


def test_the_canvas_survives_being_moved_into_the_tab(qtbot):
    """Reparenting a GL canvas is the risk in this; the viewer must still work."""
    import numpy as np

    viewer, _panel, tabs = _app(qtbot)
    try:
        viewer.add_image(np.zeros((16, 16)), name="slide")

        assert [layer.name for layer in viewer.layers] == ["slide"]
        canvas = viewer.window._qt_viewer.canvas.native
        # Still somewhere under the Project tab, just re-parented.
        assert tabs.isAncestorOf(canvas)
    finally:
        viewer.close()


def test_the_help_panel_is_not_added_as_a_dock(qtbot):
    """The sidebar is for the workflow; help there would cover it permanently."""
    from qtpy.QtWidgets import QDockWidget

    viewer, panel, _tabs = _app(qtbot)
    try:
        docked = [
            d.widget()
            for d in viewer.window._qt_window.findChildren(QDockWidget)
        ]

        assert panel.help_panel not in docked
    finally:
        viewer.close()


def test_the_menu_brings_the_help_tab_to_the_front(qtbot):
    from histo_to_ccf.gui.app import _show_help_page
    from histo_to_ccf.gui.widgets.help_panel import TUTORIAL

    viewer, panel, tabs = _app(qtbot)
    try:
        assert tabs.currentIndex() == 0  # the project is what you see first

        _show_help_page(viewer, panel.help_panel, TUTORIAL)

        assert tabs.currentIndex() == tabs.indexOf(panel.help_panel)
        assert panel.help_panel._tabs.tabText(
            panel.help_panel._tabs.currentIndex()
        ) == TUTORIAL
    finally:
        viewer.close()


# ---------------------------------------------------------------------------
# The "?" buttons and Custom ID
# ---------------------------------------------------------------------------


def test_the_atlas_help_button_is_an_icon_not_a_typed_question_mark(qtbot):
    import napari
    from qtpy.QtWidgets import QToolButton

    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.gui.workflow import WorkflowState

    viewer = napari.Viewer(show=False)
    try:
        widget = AtlasBrowserWidget(WorkflowState(), viewer)
        qtbot.addWidget(widget)
        buttons = widget.findChildren(QToolButton)

        assert buttons, "no help button found"
        assert any(not b.icon().isNull() for b in buttons)
    finally:
        viewer.close()


def test_the_atlas_help_button_uses_the_docked_page_when_there_is_one(qtbot):
    """In the app it selects Help ▸ Atlases; standalone it falls back to a window."""
    import napari

    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.gui.workflow import WorkflowState

    viewer = napari.Viewer(show=False)
    try:
        called = []
        widget = AtlasBrowserWidget(
            WorkflowState(), viewer, on_show_atlas_help=lambda: called.append(True)
        )
        qtbot.addWidget(widget)

        widget._show_atlas_reference()

        assert called == [True]
    finally:
        viewer.close()


def test_custom_id_is_explained_somewhere_the_user_can_reach():
    """The combo entry says "Custom ID" and nothing about what it accepts."""
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html

    html = atlas_reference_html()

    assert "Custom ID" in html
    assert "BrainGlobe atlas id" in html


def test_the_augmented_zenodo_link_points_at_the_current_record():
    from histo_to_ccf.gui.widgets.atlas_help_dialog import atlas_reference_html

    html = atlas_reference_html()

    assert "zenodo.org/records/18223882" in html
    assert "15176439" not in html


# ---------------------------------------------------------------------------
# Popping a page out, and coming back to the project
# ---------------------------------------------------------------------------


def test_a_page_can_be_popped_into_a_window_and_comes_back(qtbot):
    from histo_to_ccf.gui.widgets.help_panel import TUTORIAL, HelpPanelWidget

    panel = HelpPanelWidget()
    qtbot.addWidget(panel)
    panel.show_page(TUTORIAL)

    window = panel.detach_current()

    assert window is not None
    assert window.windowTitle() == TUTORIAL
    # The page has to be *visible* in the window, not merely re-parented into it:
    # removeTab hides it explicitly and Qt carries that across re-parenting, so a
    # bookkeeping-only assertion passes while the window shows nothing but a button.
    view = panel._views[TUTORIAL]
    assert view.isVisibleTo(window)
    assert len(view.toPlainText()) > 500
    assert panel.detached == [TUTORIAL]
    assert TUTORIAL not in [
        panel._tabs.tabText(i) for i in range(panel._tabs.count())
    ]
    # Still one of the panel's pages, just not docked.
    assert TUTORIAL in panel.pages

    window.close()

    assert panel.detached == []
    assert panel.pages == ["Manual", "Tutorial", "Atlases"]


def test_a_page_returns_to_the_slot_it_came_from(qtbot):
    """Popping the middle page out and back must not reorder the tabs."""
    from histo_to_ccf.gui.widgets.help_panel import TUTORIAL, HelpPanelWidget

    panel = HelpPanelWidget()
    qtbot.addWidget(panel)

    panel.detach(TUTORIAL)
    panel.attach(TUTORIAL)

    assert [panel._tabs.tabText(i) for i in range(panel._tabs.count())] == [
        "Manual", "Tutorial", "Atlases"
    ]
    # And it is showing again, not a blank tab where the text used to be.
    assert panel._views[TUTORIAL].isVisibleTo(panel._tabs)


def test_popping_every_page_explains_the_empty_area(qtbot):
    """A blank pane with no tabs would read as a broken panel."""
    from histo_to_ccf.gui.widgets.help_panel import HelpPanelWidget

    panel = HelpPanelWidget()
    qtbot.addWidget(panel)
    for title in list(panel.pages):
        panel.detach(title)

    assert panel._tabs.count() == 0
    assert panel._empty.isVisibleTo(panel)
    assert "own window" in panel._empty.text()


def test_asking_for_a_popped_out_page_raises_its_window(qtbot):
    """The user put it there; yanking it back into the tabs would undo that."""
    from histo_to_ccf.gui.widgets.help_panel import MANUAL, HelpPanelWidget

    panel = HelpPanelWidget()
    qtbot.addWidget(panel)
    panel.detach(MANUAL)

    assert panel.show_page(MANUAL) is True
    assert panel.detached == [MANUAL]  # still out


def test_loading_a_project_brings_the_project_tab_forward(qtbot):
    """Reading the manual should not hide the project you just opened."""
    from histo_to_ccf.gui.app import _show_help_page, _show_project_tab
    from histo_to_ccf.gui.widgets.help_panel import MANUAL

    viewer, panel, tabs = _app(qtbot)
    try:
        _show_help_page(viewer, panel.help_panel, MANUAL)
        assert tabs.tabText(tabs.currentIndex()) == "Help"

        _show_project_tab(panel.help_panel)

        assert tabs.tabText(tabs.currentIndex()) == "Project"
    finally:
        viewer.close()
