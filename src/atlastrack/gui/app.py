"""napari entrypoint - `launch()` builds the Viewer and docks workflow widgets."""
from __future__ import annotations

import sys
import traceback

import numpy as np


def launch() -> None:
    """Open the napari viewer with the atlastrack workflow docked."""
    import napari

    from atlastrack.gui import crashlog
    from atlastrack.gui.widgets.welcome_overlay import APP_TITLE

    # Armed before anything else: a fault in Qt/vispy/the GL driver kills the
    # process with no traceback and no dialog, and this is the only thing that
    # records what the app was doing when it happened.
    # Checked before install(), which starts a new session block in the log.
    crashed_before = crashlog.previous_session_crashed()
    log_path = crashlog.install()
    if crashed_before:
        # Only worth saying when there is something to look at. Printing the path
        # on every healthy launch is noise, and noise is what stops a warning
        # being read on the one occasion it matters.
        print(
            f"The previous session ended in a crash. Details: {log_path}",
            file=sys.stderr,
        )

    _install_exception_handler()
    try:
        viewer = napari.Viewer(title=APP_TITLE)
    except Exception as exc:
        # A dead GPU/OpenGL context (bad driver, RDP session, disabled GPU) makes
        # Viewer creation raise - print an actionable GL diagnosis instead of a
        # raw traceback, then exit.
        from atlastrack.gui.gl_diagnostics import report_launch_failure

        report_launch_failure(exc)
        raise SystemExit(1) from exc

    crashlog.log_gl_info()
    crashlog.install_input_tracer()
    panel, viz_panel = _build_panel(viewer)
    # Workflow (Registration) on the left; 3D visualization + export on the right.
    # Before the docks, so they size themselves against a central widget that is
    # already its final shape.
    _install_help_tab(viewer, panel.help_panel)
    viewer.window.add_dock_widget(panel, area="left", name="Registration", tabify=False)
    viewer.window.add_dock_widget(viz_panel, area="right", name="3D & Export", tabify=False)
    _hide_layer_panels(viewer)
    _defer_layer_controls(viewer)
    _install_welcome_overlay(viewer)
    _size_main_window(viewer)
    napari.run()


def _install_welcome_overlay(viewer: "napari.Viewer"):
    """Swap napari's welcome screen for the workflow schematic.

    napari's own screen shows its logo, a shortcut list and tips about menus this
    app hides, and it is the first thing anyone sees. Ours follows the same rule -
    visible exactly while the canvas holds no layers - so *Close Project* brings it
    back.

    Best-effort: if the private Qt handles are missing (headless, a future napari),
    the app runs with napari's own screen rather than refusing to start.
    """
    from atlastrack.gui.widgets.welcome_overlay import WelcomeOverlayWidget

    try:
        # Parent to the canvas, NOT to ``_qt_viewer``: that is a QSplitter, so a
        # child added to it becomes a splitter pane laid out below the canvas
        # instead of an overlay on top of it.
        canvas = viewer.window._qt_viewer.canvas.native
    except Exception:
        return None

    # Turn napari's off first: ours is opaque, but two welcome screens sharing one
    # canvas is a redraw bug waiting to happen.
    try:
        viewer.window._qt_viewer.show_welcome_screen = False
    except Exception:
        try:
            viewer.welcome_screen.visible = False
        except Exception:
            pass

    overlay = WelcomeOverlayWidget(canvas, theme=getattr(viewer, "theme", None))

    def _sync(_event=None) -> None:
        overlay.setGeometry(canvas.rect())
        empty = len(viewer.layers) == 0
        overlay.setVisible(empty)
        if empty:
            overlay.raise_()  # stay above the canvas after layers come and go

    viewer.layers.events.inserted.connect(_sync)
    viewer.layers.events.removed.connect(_sync)
    try:
        viewer.events.theme.connect(lambda _e: overlay.set_theme(viewer.theme))
    except Exception:
        pass
    _sync()
    return overlay


def _section_at(state: "WorkflowState", y: float, x: float):
    """Index of the section whose bbox covers ``(y, x)`` on the active slide.

    Overlapping boxes resolve to the smallest, which is the one a click was most
    likely aimed at. Returns None outside every box.
    """
    slide_idx = state.active_slide_idx
    if slide_idx is None or slide_idx >= len(state.project.slides):
        return None
    best, best_area = None, None
    for section in state.project.slides[slide_idx].sections:
        x0, y0, x1, y1 = section.bbox_px
        if not (x0 <= x < x1 and y0 <= y < y1):
            continue
        area = (x1 - x0) * (y1 - y0)
        if best_area is None or area < best_area:
            best, best_area = section.index, area
    return best


def _modifier_names(event) -> set[str]:
    """``event.modifiers`` as plain names ("Control", "Shift", ...)."""
    return {str(getattr(m, "name", m)) for m in (getattr(event, "modifiers", ()) or ())}


def _call_panels(panels, method: str, *args) -> None:
    for panel in panels:
        fn = getattr(panel, method, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 - one panel must not block the rest
                pass


def _install_section_click(viewer: "napari.Viewer", state: "WorkflowState", panels) -> None:
    """Pick sections with left clicks on the canvas, in any tab.

    A plain click selects the section under it everywhere that acts on "the
    section", and makes it the only one picked for registration; a plain click on
    empty canvas clears that pick, so the Register button goes back to all
    sections. Ctrl+click adds or removes a section, Shift+click picks the range
    from the last one - as in the section table.

    Acts on release, and only if the mouse did not move: a drag is a pan, and
    panning across the slide must not change what is selected.

    Hit-tests the stored bboxes rather than reading the outline Labels layer: that
    layer only paints the *border*, so a click in the middle of a section would
    return background and select nothing.
    """

    def _on_click(_viewer, event):
        if getattr(event, "button", 1) != 1:
            return
        mods = _modifier_names(event)
        if mods and mods not in ({"Control"}, {"Shift"}):
            return
        # Modifier clicks belong to a layer being edited (Ctrl+drag re-anchors a
        # landmark, Shift+click multi-selects points); only in plain navigation
        # are they free for picking sections.
        active = getattr(getattr(_viewer, "layers", None), "selection", None)
        active = getattr(active, "active", None)
        if mods and getattr(active, "mode", "pan_zoom") != "pan_zoom":
            return
        position = getattr(event, "position", None)
        if position is None or len(position) < 2:
            return
        y, x = float(position[-2]), float(position[-1])

        yield
        dragged = False
        while getattr(event, "type", "mouse_release") == "mouse_move":
            dragged = True
            yield
        if dragged:
            return

        index = _section_at(state, y, x)
        if "Control" in mods:
            if index is not None:
                _call_panels(panels, "toggle_section_selection", int(index))
            return
        if "Shift" in mods:
            if index is not None:
                _call_panels(panels, "extend_section_selection", int(index))
            return
        if index is None:
            _call_panels(panels, "clear_section_selection")
            return
        state.active_section_idx = int(index)
        _call_panels(panels, "select_section", int(index))
        _call_panels(panels, "pick_section_for_registration", int(index))

    try:
        viewer.mouse_drag_callbacks.append(_on_click)
    except Exception:  # noqa: BLE001 - headless viewer without a canvas
        pass


def _hide_layer_panels(viewer: "napari.Viewer") -> None:
    """Hide napari's built-in 'layer list' + 'layer controls' docks.

    The user drives everything through the Histo→CCF workflow panel; the raw
    layer list/controls only add confusion. Best-effort across napari versions -
    silently ignore if the private dock handles are not present.
    """
    try:
        qt_viewer = viewer.window._qt_viewer
        for attr in ("dockLayerList", "dockLayerControls"):
            dock = getattr(qt_viewer, attr, None)
            if dock is not None:
                dock.setVisible(False)
    except Exception:
        pass


def _defer_layer_controls(viewer: "napari.Viewer") -> bool:
    """Build napari's per-layer control widgets only while their dock is shown.

    napari builds a full Qt controls panel for every layer the moment it is added,
    whether or not anyone can see it. This app keeps that dock hidden, and "Show
    atlas overlay" adds a layer per section, so those panels were built for nothing:
    measured at roughly 0.3-0.45 s of an 18-section overlay.

    This does not change napari. It swaps which of napari's own handlers answer the
    layer events: building is skipped while the dock is hidden and done on demand
    when it is shown - napari's Window menu can show it - so the controls are still
    there for anyone who opens them. If napari's internals are not shaped the way
    this expects, nothing is touched and napari behaves exactly as before; returns
    whether the lazy version is in place.
    """
    from types import SimpleNamespace

    try:
        qt_viewer = viewer.window._qt_viewer
        container = qt_viewer.controls
        dock = qt_viewer.dockLayerControls
        widgets = container.widgets
        empty = container.empty_widget
        eager_add, eager_remove, eager_display = (
            container._add, container._remove, container._display
        )
        layers = viewer.layers
        inserted, removed = layers.events.inserted, layers.events.removed
        active = layers.selection.events.active
    except Exception:
        return False

    def _ensure(layer) -> None:
        if layer is not None and layer not in widgets:
            eager_add(SimpleNamespace(value=layer))

    def _on_active(event) -> None:
        layer = event.value
        if layer is not None and dock.isVisible():
            _ensure(layer)
        container.setCurrentWidget(widgets.get(layer, empty) if layer is not None else empty)

    def _on_removed(event) -> None:
        if event.value in widgets:
            eager_remove(event)

    def _on_dock_visibility(visible: bool) -> None:
        if not visible:
            return
        layer = layers.selection.active
        _ensure(layer)
        container.setCurrentWidget(widgets.get(layer, empty) if layer is not None else empty)

    try:
        inserted.disconnect(eager_add)
        removed.disconnect(eager_remove)
        active.disconnect(eager_display)
        active.connect(_on_active)
        removed.connect(_on_removed)
        dock.visibilityChanged.connect(_on_dock_visibility)
    except Exception:
        # Put napari's own wiring back exactly as it was, whatever got through.
        for emitter, ours, theirs in (
            (inserted, None, eager_add),
            (removed, _on_removed, eager_remove),
            (active, _on_active, eager_display),
        ):
            try:
                if ours is not None:
                    emitter.disconnect(ours)
                emitter.connect(theirs)
            except Exception:
                pass
        return False
    # Held here so nothing can collect the closures while the viewer lives.
    container._atlastrack_lazy_controls = (_on_active, _on_removed, _on_dock_visibility)
    return True


# Target aspect ratio (width : height) for the main window. 16:9 keeps the
# canvas wide and rectangular so the slide gets most of the horizontal room,
# with the docked workflow panel pinned to a compact column on the right.
_WINDOW_ASPECT = (16, 9)


def _size_main_window(viewer: "napari.Viewer") -> None:
    """Resize the napari window to a wide rectangle (``_WINDOW_ASPECT``).

    Height is taken as ~85 % of the available screen height; width follows from
    the aspect ratio, clamped to 95 % of the screen so it never overflows.
    Best-effort: any failure (headless, missing screen) is silently ignored.
    """
    try:
        from qtpy.QtWidgets import QApplication

        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        w_ratio, h_ratio = _WINDOW_ASPECT
        height = int(avail.height() * 0.85)
        width = min(int(height * w_ratio / h_ratio), int(avail.width() * 0.95))

        # napari.Window.resize delegates to the underlying QMainWindow; fall
        # back to the private handle if the public method is unavailable.
        try:
            viewer.window.resize(width, height)
        except Exception:
            viewer.window._qt_window.resize(width, height)
    except Exception:
        pass


def _install_exception_handler() -> None:
    """Replace sys.excepthook with one that shows a Qt error dialog."""
    from qtpy.QtWidgets import QApplication, QMessageBox

    _original = sys.excepthook

    def _handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            _original(exc_type, exc_value, exc_tb)
            return
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(None, f"Unhandled error: {exc_type.__name__}", tb[:2000])
        _original(exc_type, exc_value, exc_tb)

    sys.excepthook = _handler


def _scrollable(page: "QWidget") -> "QWidget":
    """``page`` in a scroll area that scrolls up and down only.

    It asks for the page's full width, so the panel can never be narrower than
    its controls (a plain scroll area lets it shrink and cuts them off on the
    right), and for almost no height, so the window can be as short as the screen.
    """
    from qtpy.QtCore import QSize, Qt
    from qtpy.QtWidgets import QFrame, QScrollArea

    class _VerticalScroll(QScrollArea):
        #: Called when the page's layout changes (its width needs may have grown).
        on_layout_changed = None

        def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt name
            from qtpy.QtCore import QEvent, QTimer

            if obj is self.widget() and event.type() == QEvent.LayoutRequest:
                self.updateGeometry()
                if self.on_layout_changed is not None:
                    QTimer.singleShot(0, self.on_layout_changed)
            return super().eventFilter(obj, event)

        def minimumSizeHint(self) -> QSize:
            inner = self.widget().minimumSizeHint() if self.widget() else QSize(0, 0)
            bar = self.verticalScrollBar().sizeHint().width()
            return QSize(inner.width() + bar, 60)

        def sizeHint(self) -> QSize:
            inner = self.widget().sizeHint() if self.widget() else QSize(0, 0)
            bar = self.verticalScrollBar().sizeHint().width()
            return QSize(inner.width() + bar, inner.height())

    scroll = _VerticalScroll()
    scroll.setWidget(page)
    page.installEventFilter(scroll)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    # The panel is kept wide enough (see _fit_width_to_tab); should it still end
    # up narrower, a scroll bar beats cutting the controls off.
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return scroll


def _narrow_combo_boxes(root: "QWidget", chars: int = 10) -> None:
    """Stop the combo boxes under ``root`` from widening the panel to their longest item.

    A combo box is as wide as its longest entry by default, so one long probe or
    stream name set the whole side panel's width. Here each is as wide as ``chars``
    characters (wider if its row allows), a longer current choice is shortened with
    "…" and given in full as a tooltip, and the drop-down list still shows every
    entry in full.
    """
    from qtpy.QtWidgets import QComboBox

    for combo in root.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(chars)

        def _fit_popup(_=None, combo=combo) -> None:
            view = combo.view()
            view.setMinimumWidth(view.sizeHintForColumn(0) + 24)
            text = combo.currentText()
            if text and not combo.toolTip():
                combo.setProperty("_auto_tip", True)
            if combo.property("_auto_tip"):
                combo.setToolTip(text)

        combo.currentTextChanged.connect(_fit_popup)
        combo.model().rowsInserted.connect(lambda *_, f=_fit_popup: f())
        _fit_popup()


def _build_panel(viewer: "napari.Viewer") -> "QWidget":
    """Construct the main dock panel and wire up all sub-widgets."""
    from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

    from atlastrack.config import load_app_settings, save_app_settings
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.gui.widgets.atlas_browser import AtlasBrowserWidget
    from atlastrack.gui.widgets.click_overlay import ClickOverlayWidget
    from atlastrack.gui.widgets.ephys_panel import EphysPanelWidget
    from atlastrack.gui.widgets.image_tools import ImageToolsWidget
    from atlastrack.gui.widgets.ordering_panel import OrderingPanelWidget
    from atlastrack.gui.widgets.probe_picker import ProbePickerWidget
    from atlastrack.gui.widgets.register_panel import RegisterPanelWidget
    from atlastrack.gui.widgets.slide_loader import SlideLoaderWidget
    from atlastrack.gui.widgets.viz_export_panel import VizExportPanelWidget

    from atlastrack.gui.widgets.help_panel import ATLASES, HelpPanelWidget

    settings = load_app_settings()
    state = WorkflowState()
    # Carried on the container rather than returned: every caller of _build_panel
    # unpacks two values, and the dock is only launch()'s business.
    help_panel = HelpPanelWidget()

    container = QWidget()
    container.setMinimumWidth(320)
    root = QVBoxLayout(container)
    root.setContentsMargins(0, 0, 0, 0)

    tabs = QTabWidget()
    root.addWidget(tabs)

    # -- Histology: load slide + detect sections + image tools --------------
    tab_load = QWidget()
    load_layout = QVBoxLayout(tab_load)
    load_layout.setContentsMargins(2, 2, 2, 2)
    image_tools = ImageToolsWidget(state, on_display_changed=lambda: _refresh_slide(viewer, state))
    slide_loader = SlideLoaderWidget(
        state,
        viewer=viewer,
        # A new slide also settles which face its sections are seen from.
        on_slide_loaded=lambda idx, img: (
            _on_slide_loaded(viewer, state, idx, img), image_tools._show_view()
        ),
        on_sections_detected=lambda secs: _on_sections_detected(viewer, state, secs),
        on_section_selected=image_tools.select_section,
    )
    load_layout.addWidget(slide_loader)
    load_layout.addWidget(image_tools)

    # -- Atlas: choose atlas + assign AP + section ordering ------------------
    tab_atlas = QWidget()
    atlas_layout = QVBoxLayout(tab_atlas)
    atlas_layout.setContentsMargins(2, 2, 2, 2)
    atlas_browser = AtlasBrowserWidget(
        state, viewer, settings=settings,
        on_show_atlas_help=lambda: _show_help_page(viewer, help_panel, ATLASES),
    )
    ordering = OrderingPanelWidget(state)
    # The matcher dialog (opened from the browser) syncs AP + spacing with these
    # widgets, so give the browser a handle to the ordering panel.
    atlas_browser.ordering_panel = ordering
    atlas_layout.addWidget(atlas_browser)
    atlas_layout.addWidget(ordering)

    # -- Probes: add probe + click tip/entry --------------------------------
    tab_annotate = QWidget()
    ann_layout = QVBoxLayout(tab_annotate)
    ann_layout.setContentsMargins(2, 2, 2, 2)
    probe_picker = ProbePickerWidget(state)
    click_overlay = ClickOverlayWidget(state, viewer)
    # After adding a probe, immediately arm tip-marker mode so the user can
    # click a tip point without first toggling the Tip/Entry selector.
    probe_picker.on_probe_added = click_overlay.arm_tip
    ann_layout.addWidget(probe_picker)
    ann_layout.addWidget(click_overlay)

    # -- Register + Results -------------------------------------------------
    tab_register = QWidget()
    reg_layout = QVBoxLayout(tab_register)
    reg_layout.setContentsMargins(2, 2, 2, 2)
    register_panel = RegisterPanelWidget(state, viewer)
    register_panel.apply_settings(settings)
    reg_layout.addWidget(register_panel)

    # -- Ephys alignment ----------------------------------------------------
    tab_ephys = QWidget()
    ephys_layout = QVBoxLayout(tab_ephys)
    ephys_layout.setContentsMargins(2, 2, 2, 2)
    ephys_panel = EphysPanelWidget(state, viewer)
    ephys_layout.addWidget(ephys_panel)

    # Tab order follows the workflow: load the histology, set each section's AP
    # against the atlas, register, then mark probes on the registered sections
    # and align ephys to them.
    # Each tab scrolls when the screen is shorter than its controls; without this
    # the tallest tab set the window's minimum height, which could exceed the
    # screen and stop the window being resized at all.
    tabs.addTab(_scrollable(tab_load), "Histology")
    tabs.addTab(_scrollable(tab_atlas), "Atlas")
    tabs.addTab(_scrollable(tab_register), "Register")
    tabs.addTab(_scrollable(tab_annotate), "Probes")
    tabs.addTab(_scrollable(tab_ephys), "Ephys")
    _narrow_combo_boxes(tabs)

    def _fit_width_to_tab(_index: int = 0) -> None:
        # As wide as the tab on show needs (its scroll bar included), so none is
        # ever cut off - and no wider, so narrow tabs don't pay for the widest.
        page = tabs.currentWidget()
        if page is not None:
            container.setMinimumWidth(max(320, page.minimumSizeHint().width() + 8))

    tabs.currentChanged.connect(_fit_width_to_tab)
    for k in range(tabs.count()):
        # A tab whose content grows later (a project loaded, a list filled) asks again.
        tabs.widget(k).on_layout_changed = (
            lambda k=k: _fit_width_to_tab() if tabs.currentIndex() == k else None
        )
    _fit_width_to_tab()

    # 3D visualization + export live in their own permanent panel (right dock),
    # not inside the Register tab.
    viz_panel = VizExportPanelWidget(state, viewer)
    viz_panel.apply_settings(settings)
    # Its atlas and alignment lists would otherwise make it about 500 px wide.
    _narrow_combo_boxes(viz_panel)

    # After a project load, redraw the canvas AND repopulate every tab's fields
    # from the loaded project (probes, tip/entry, atlas + AP, ordering, residuals,
    # ephys) - loading the data alone leaves the widgets showing stale defaults.
    panels = (slide_loader, image_tools, probe_picker, click_overlay,
              atlas_browser, ordering, register_panel, ephys_panel)

    def _refresh_panels() -> None:
        for panel in panels:
            refresh = getattr(panel, "refresh_after_load", None)
            if callable(refresh):
                try:
                    refresh()
                except Exception:  # noqa: BLE001 - one panel must not block the rest
                    pass

    # Renaming a probe must repopulate every panel's probe combo (Probes
    # tip/entry, Ephys) so they show the new label.
    probe_picker.on_probes_changed = _refresh_panels
    # Naming a channel in Histology updates the Register tab's "Align on" list.
    image_tools.on_channel_names_changed = register_panel._populate_align_combo

    def _after_match() -> None:
        # A match may flip sections and brings cells: redraw both.
        for slide_idx in list(state.slide_images):
            name = f"Slide {slide_idx}"
            if name in viewer.layers:
                viewer.layers[name].data = _display_image_for_slide(
                    state, slide_idx, state.slide_images[slide_idx]
                )
        _update_channel_layers(viewer, state)
        _update_cells_layer(viewer, state)

    register_panel.on_project_changed = _after_match
    # Channel images added or removed in Histology, and the one to show.
    image_tools.on_channel_images_changed = (
        lambda: _update_channel_layers(viewer, state, reload=True)
    )
    image_tools.on_show_channel = lambda name: _show_channel(viewer, state, name)

    def _after_planes_mirrored() -> None:
        # Overlays drawn for the old planes are out of date: redraw those shown.
        if any(layer.name.startswith("Atlas overlay ") for layer in viewer.layers):
            register_panel._show_overlay()
        register_panel._refresh_residuals()

    image_tools.on_planes_mirrored = _after_planes_mirrored

    def _on_project_loaded() -> None:
        # A project was just opened; whatever was being read, the thing to look at
        # now is the project. Reading is never lost - the Help tab keeps its page.
        _show_project_tab(help_panel)
        _reload_project_display(viewer, state)
        _refresh_panels()
        from atlastrack.project.checks import mirrored_sections_note

        note = mirrored_sections_note(state.project)
        if note:
            from napari.utils.notifications import show_warning

            show_warning(note)
        # Auto-load the project's atlas in the background so the overlay / 3D
        # brain are ready without a manual "Load atlas" click.
        atlas_browser.auto_load_atlas()

    def _on_project_cleared() -> None:
        # Remove every layer from the canvas and reset all tabs to the empty
        # project (state has already been reset by the menu action).
        try:
            viewer.layers.clear()
        except Exception:  # noqa: BLE001
            pass
        _refresh_panels()

    # Project save/load/close live in the menu bar (see _install_project_menu),
    # not a tab - they are file actions, not part of the left-to-right workflow.
    project_menu = _install_project_menu(
        viewer, state, settings=settings,
        on_loaded=_on_project_loaded, on_cleared=_on_project_cleared,
    )
    # Settings hosts the registration parameters (kept out of the panel); Help
    # raises the docked manual / tutorial / atlas sheet. napari's own menus are
    # hidden, including its Help, so ours is the only one in the bar.
    settings_menu = _install_settings_menu(viewer, register_panel)
    help_menu = _install_help_menu(viewer, help_panel)
    _keep_only_menus(viewer, (project_menu, settings_menu, help_menu))
    _install_wheel_pan(viewer)

    # Persist settings when the tab changes (cheap enough to do on every switch).
    def _on_tab_change(_idx: int) -> None:
        register_panel.collect_settings(settings)
        atlas_browser.collect_settings(settings)
        viz_panel.collect_settings(settings)
        save_app_settings(settings)

    tabs.currentChanged.connect(_on_tab_change)

    # Clicking a section in the canvas selects it everywhere that acts on "the
    # section": Adjustments in Histology and Manual atlas adjustment in Register.
    # Deliberately not tied to the Shapes layer that "Edit boxes" creates - having
    # to enter an edit mode before a box could be picked was the complaint.
    _install_section_click(viewer, state, (image_tools, register_panel))

    # One pass over the finished tree: Qt never wraps a plain-text tooltip, and
    # several here run past 300 characters.
    from atlastrack.gui.widgets.tooltips import wrap_tooltips

    wrap_tooltips(container)
    wrap_tooltips(viz_panel)
    container.help_panel = help_panel

    # Quitting closes the open project too; say so in the terminal like Close Project.
    from qtpy.QtWidgets import QApplication

    def _report_quit() -> None:
        if state.project_path is not None:
            from atlastrack.gui.widgets.save_panel import report_project

            report_project("Closed", state.project_path)

    qapp = QApplication.instance()
    if qapp is not None:
        qapp.aboutToQuit.connect(_report_quit)
    return container, viz_panel


def _recent_label(path: str) -> str:
    """Menu label for a recent project: parent folder + filename (no long path)."""
    from pathlib import Path

    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def _install_project_menu(
    viewer: "napari.Viewer", state: "WorkflowState", settings=None,
    on_loaded=None, on_cleared=None,
) -> None:
    """Add a "Project" menu (first in the menu bar) with Save / Save As / Load.

    These are file operations (not workflow steps), so they belong in the menu
    bar rather than a docked tab. Save/Load reuse :class:`SavePanelWidget`'s
    logic via a hidden instance so behaviour stays in one place. ``on_loaded``
    runs after a successful load (redraw canvas + repopulate tabs); it defaults to
    just redrawing the canvas. Best-effort: if the Qt main window or menu bar is
    unavailable (headless), do nothing.
    """
    from pathlib import Path

    from qtpy.QtWidgets import QMenu

    from atlastrack.gui.widgets.save_panel import SavePanelWidget

    try:
        menubar = viewer.window._qt_window.menuBar()
    except Exception:
        return

    if on_loaded is None:
        on_loaded = lambda: _reload_project_display(viewer, state)  # noqa: E731

    # A hidden helper widget owns the save/load implementation + file dialogs.
    helper = SavePanelWidget(state, on_project_loaded=on_loaded, settings=settings)
    helper.hide()
    # Keep it alive for the session (parent it to the main window).
    try:
        helper.setParent(viewer.window._qt_window)
    except Exception:
        pass

    # Insert "Project" as the first (left-most) menu, before napari's File menu.
    menu = QMenu("Project", menubar)
    existing = menubar.actions()
    if existing:
        menubar.insertMenu(existing[0], menu)
    else:
        menubar.addMenu(menu)

    def _save() -> None:
        # Save to the known project path if set, else prompt as Save As.
        if state.project_path is not None:
            helper._path_edit.setText(str(state.project_path))
            helper._save()
        else:
            _save_as()

    def _save_as() -> None:
        helper._path_edit.clear()
        helper._browse()
        if helper._path_edit.text().strip():
            helper._save()

    def _close() -> None:
        # Closing discards in-memory work, so confirm first - but only when there
        # is work to lose. A prompt on every close teaches clicking Yes blind.
        if _confirm_close(helper, state):
            from atlastrack.gui.widgets.save_panel import report_project

            closed = state.project_path
            state.reset()
            report_project("Closed", closed)
        if on_cleared is not None:
            on_cleared()

    from qtpy.QtCore import Qt

    save_action = menu.addAction("Save Project")
    save_action.triggered.connect(_save)
    save_as_action = menu.addAction("Save Project As")
    save_as_action.triggered.connect(_save_as)
    menu.addSeparator()
    load_action = menu.addAction("Load Project")
    load_action.triggered.connect(helper._load)

    # "Load recent ▸" - rebuilt each time it opens from settings.recent_projects.
    recent_menu = menu.addMenu("Load recent")

    def _rebuild_recent() -> None:
        recent_menu.clear()
        entries = list(getattr(settings, "recent_projects", []) or [])
        # Drop paths that no longer exist so the list stays trustworthy.
        entries = [p for p in entries if Path(p).exists()]
        if not entries:
            empty = recent_menu.addAction("(none yet)")
            empty.setEnabled(False)
            return
        for p in entries:
            act = recent_menu.addAction(_recent_label(p))
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, path=p: helper.load_path(path))
        recent_menu.addSeparator()
        clear = recent_menu.addAction("Clear recent")
        clear.triggered.connect(_clear_recent)

    def _clear_recent() -> None:
        if settings is None:
            return
        settings.recent_projects = []
        try:
            from atlastrack.config import save_app_settings

            save_app_settings(settings)
        except Exception:  # noqa: BLE001 - best-effort persistence
            pass

    recent_menu.aboutToShow.connect(_rebuild_recent)
    _rebuild_recent()  # populate once so it isn't empty before first open

    # "Import from another tool ▸": convert, then open the result like a project.
    import_menu = menu.addMenu("Import from another tool")

    def _import(kind: str) -> None:
        from qtpy.QtWidgets import QFileDialog, QMessageBox

        window = viewer.window._qt_window
        if not _confirm_close(window, state):
            return
        if kind == "slicereg":
            source = QFileDialog.getExistingDirectory(
                window, "slicereg project folder (holds project.json and slices/)"
            )
        else:
            source, _ = QFileDialog.getOpenFileName(
                window, "QuickNII / DeepSlice / VisuAlign file", "",
                "QUINT series (*.json *.waln *.wwrp)",
            )
        if not source:
            return
        from napari.qt.threading import thread_worker

        from atlastrack.io.importers import import_registration

        @thread_worker
        def _run():
            return import_registration(source)

        def _done(project_path) -> None:
            viewer.status = f"Imported {Path(source).name}"
            helper.load_path(str(project_path))

        def _failed(exc) -> None:
            viewer.status = "Import failed"
            QMessageBox.warning(window, "Import failed", str(exc)[:2000])

        viewer.status = f"Importing {Path(source).name}... (section images are written first)"
        worker = _run()
        worker.returned.connect(_done)
        worker.errored.connect(_failed)
        worker.start()

    slicereg_action = import_menu.addAction("slicereg project folder...")
    slicereg_action.setToolTip("Martin Dokholyan's cell-counting app (slicereg)")
    slicereg_action.triggered.connect(lambda: _import("slicereg"))
    quint_action = import_menu.addAction("QuickNII / DeepSlice / VisuAlign file...")
    quint_action.triggered.connect(lambda: _import("quint"))

    close_action = menu.addAction("Close Project")
    close_action.triggered.connect(_close)

    # Keyboard shortcuts (application-wide so they fire even with the napari
    # canvas focused): Ctrl+S save, Ctrl+Shift+S save-as, Ctrl+O load.
    for action, seq in (
        (save_action, "Ctrl+S"),
        (save_as_action, "Ctrl+Shift+S"),
        (load_action, "Ctrl+O"),
    ):
        action.setShortcut(seq)
        action.setShortcutContext(Qt.ApplicationShortcut)
    # After the shortcuts are set, so their width is known.
    _fit_menu_width(menu)
    return menu


def _confirm_close(parent, state: "WorkflowState") -> bool:
    """True to go ahead and close: nothing unsaved, or the user said Yes."""
    if not state.has_unsaved_changes():
        return True
    try:
        from qtpy.QtWidgets import QMessageBox

        resp = QMessageBox.question(
            parent, "Close project",
            "The project has unsaved changes. Close it anyway? This clears the "
            "loaded slides, sections, probes and registration from the app, and "
            "the unsaved changes are lost.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
    except Exception:  # noqa: BLE001 - headless: proceed without a prompt
        return True
    return resp == QMessageBox.Yes


def _fit_menu_width(menu) -> None:
    """Widen a menu so its shortcuts cannot be drawn over its labels.

    Under napari's stylesheet Qt's size hint accounts for the label text but leaves
    almost nothing for the icon column and the label-to-shortcut gap: the Project
    menu came out 165 px wide when "Save Project As" + "Ctrl+Shift+S" alone need
    145, so the two overlapped. Qt lays these out as columns, so the width needed
    is the widest label plus the widest shortcut plus the gap - not the widest
    label+shortcut pair.
    """
    from qtpy.QtGui import QFontMetrics

    metrics = QFontMetrics(menu.font())
    labels, shortcuts = [0], [0]
    for action in menu.actions():
        if action.isSeparator():
            continue
        labels.append(metrics.horizontalAdvance(action.text()))
        if not action.shortcut().isEmpty():
            shortcuts.append(metrics.horizontalAdvance(action.shortcut().toString()))
    if not max(shortcuts):
        return
    # Gap between the two columns, plus the icon/checkmark column and margins Qt
    # draws either side. Measured against the styled menu rather than guessed.
    gap, chrome = 28, 52
    menu.setMinimumWidth(max(labels) + gap + max(shortcuts) + chrome)


def _install_settings_menu(viewer: "napari.Viewer", register_panel) -> None:
    """Add a "Settings" menu: registration parameters, and the atlas reference.

    The registration parameters were moved out of the Register panel (the
    defaults are good); this is where to bring them back up when needed.
    Best-effort: no-op if the Qt menu bar is unavailable (headless).
    """
    from qtpy.QtWidgets import QMenu

    try:
        menubar = viewer.window._qt_window.menuBar()
    except Exception:
        return

    menu = QMenu("Settings", menubar)
    menubar.addMenu(menu)
    # "Registration", not "Parameters": the menu already says these are settings, so
    # the item should say what they are settings *for*.
    registration_action = menu.addAction("Registration")
    registration_action.triggered.connect(register_panel.open_parameters_dialog)

    _fit_menu_width(menu)
    return menu


def _install_help_menu(viewer: "napari.Viewer", help_panel) -> None:
    """Add a "Help" menu that raises the docked help panel on the right page.

    The items do not open windows: they select a tab in a dock that sits beside the
    workflow, so reading the manual never covers the project or loses its place.
    """
    from qtpy.QtWidgets import QMenu

    try:
        menubar = viewer.window._qt_window.menuBar()
    except Exception:
        return

    from atlastrack.gui.widgets.help_panel import ATLASES, MANUAL, TUTORIAL

    menu = QMenu("Help", menubar)
    menubar.addMenu(menu)
    for title, tip in (
        (MANUAL, "The full reference: concepts, GUI tour, recipes, troubleshooting."),
        (TUTORIAL, "A single linear walkthrough on example data."),
        (ATLASES, "What each atlas is, where it came from, and where its bregma sits."),
    ):
        action = menu.addAction(title)
        action.setToolTip(tip)
        action.triggered.connect(
            lambda _checked=False, page=title: _show_help_page(viewer, help_panel, page)
        )
    _fit_menu_width(menu)
    return menu


def _install_help_tab(viewer: "napari.Viewer", help_panel):
    """Put the help panel in the *central* pane, as a tab beside the canvas.

    Not a dock: a dock lives in the sidebar and stays there over the project. The
    manual belongs where the project is, so switching to it and back is the same
    gesture as switching between them.

    napari sets its central widget once during construction and never reads it back
    (``qt_main_window.py`` has the only reference), so wrapping that widget in a
    QTabWidget is safe - the QtViewer object itself is untouched, only reparented,
    which the GL canvas survives.
    """
    from qtpy.QtWidgets import QTabWidget

    try:
        window = viewer.window._qt_window
        central = window.centralWidget()
    except Exception:
        return None
    if central is None:
        return None

    tabs = QTabWidget()
    tabs.setDocumentMode(True)
    tabs.addTab(central, "Project")
    tabs.addTab(help_panel, "Help")
    window.setCentralWidget(tabs)
    help_panel._central_tabs = tabs
    return tabs


def _show_project_tab(help_panel) -> bool:
    """Bring the Project tab forward. False when there is no central tab bar."""
    tabs = getattr(help_panel, "_central_tabs", None)
    if tabs is None:
        return False
    index = tabs.indexOf(help_panel)
    # Two tabs, so "not the help one" is the project one.
    tabs.setCurrentIndex(1 if index == 0 else 0)
    return True


def _show_help_page(viewer: "napari.Viewer", help_panel, page: str) -> None:
    """Select ``page`` and bring the help tab to the front of the central pane."""
    help_panel.show_page(page)
    tabs = getattr(help_panel, "_central_tabs", None)
    if tabs is None:
        return
    index = tabs.indexOf(help_panel)
    if index >= 0:
        tabs.setCurrentIndex(index)


def _show_atlas_reference(viewer: "napari.Viewer"):
    """Open the atlas reference sheet, parented to the main window."""
    from atlastrack.gui.widgets.atlas_help_dialog import show_atlas_reference

    try:
        parent = viewer.window._qt_window
    except Exception:
        parent = None
    return show_atlas_reference(parent)


def _install_wheel_pan(viewer: "napari.Viewer") -> None:
    """Pan the canvas with **Ctrl+wheel** (horizontal) and **Shift+wheel** (vertical).

    The slides are tall composites, so plain wheel-zoom alone makes it awkward to
    move around. napari already *suppresses* wheel-zoom whenever a modifier is
    held (its canvas ignores modified wheel events), so these callbacks add
    panning without ever fighting the zoom. Wheel-up moves the view up / left.
    """

    def _pan(viewer, event) -> None:
        mods = set(getattr(event, "modifiers", ()))
        horizontal = "Control" in mods
        vertical = "Shift" in mods and not horizontal
        if not (horizontal or vertical):
            return
        delta = event.delta[1] if event.delta[1] else event.delta[0]
        if not delta:
            return
        if getattr(event.native, "inverted", lambda: False)():
            delta = -delta
        zoom = viewer.camera.zoom or 1.0
        # ~80 canvas px per wheel notch, scaled to world units by the zoom so the
        # pan feels the same at any magnification.
        step = float(delta) * 80.0 / zoom
        center = list(viewer.camera.center)  # (z, y, x)
        if horizontal:
            center[2] -= step
        else:
            center[1] -= step
        viewer.camera.center = tuple(center)

    viewer.mouse_wheel_callbacks.append(_pan)


def _keep_only_menus(viewer: "napari.Viewer", keep) -> None:
    """Hide every top-level menu except the ones we built.

    Matched by identity, not by title: napari's own menu is called "&Help" and ours
    is called "Help", so a title comparison keeps both and the bar ends up with two
    Help menus. Hiding (not removing) is reversible and survives napari re-adding
    its menus. Best-effort: no-op if the menu bar is unavailable.
    """
    try:
        menubar = viewer.window._qt_window.menuBar()
    except Exception:
        return

    ours = {id(menu) for menu in keep if menu is not None}
    for action in menubar.actions():
        if id(action.menu()) not in ours:
            action.setVisible(False)


# ---------------------------------------------------------------------------
# Viewer update helpers
# ---------------------------------------------------------------------------

def _on_slide_loaded(viewer: "napari.Viewer", state: "WorkflowState", slide_idx: int, img) -> None:
    name = f"Slide {slide_idx}"
    if name in viewer.layers:
        viewer.layers[name].data = img
    else:
        viewer.add_image(img, name=name, colormap="gray")
    viewer.reset_view()


def _on_sections_detected(viewer: "napari.Viewer", state: "WorkflowState", sections) -> None:
    """Render section outlines + index labels for the detected sections."""
    from atlastrack.gui.section_display import sections_to_outline_labels

    slide_idx = state.active_slide_idx
    if slide_idx is None:
        return
    img = state.slide_images.get(slide_idx)
    if img is None:
        return
    slide = state.project.slides[slide_idx]

    # --- Outline Labels layer ---
    labels = sections_to_outline_labels(img.shape[:2], slide.sections)
    outline_name = f"Sections {slide_idx}"
    if outline_name in viewer.layers:
        lyr = viewer.layers[outline_name]
        lyr.data = labels
    else:
        lyr = viewer.add_labels(
            labels, name=outline_name, opacity=0.85,
            blending=_section_outline_blending(),
        )

    # --- Section number text layer ---
    _update_section_numbers(viewer, state, slide_idx)


def _section_outline_blending() -> str:
    """The blend every slide overlay uses. See :mod:`atlastrack.gui.overlay_style`."""
    from atlastrack.gui.overlay_style import OVERLAY_BLENDING

    return OVERLAY_BLENDING


CHANNEL_LAYER_PREFIX = "Channel: "


def _channel_layer_name(state: "WorkflowState", slide_idx: int, name: str) -> str:
    many = len(state.project.slides) > 1
    return f"{CHANNEL_LAYER_PREFIX}{name}" + (f" (slide {slide_idx})" if many else "")


def _update_channel_layers(
    viewer: "napari.Viewer", state: "WorkflowState", *, reload: bool = False
) -> None:
    """Show each slide's channel images, laid out and flipped like the slide.

    ``reload`` reads them from disk again (after a project load or a new channel);
    otherwise the ones in memory are re-flipped to follow the slide's edits.
    Channel layers sit just above their slide and start hidden - the Histology
    tab's "Show" picks one.
    """
    from loguru import logger

    from atlastrack.project.images import apply_slide_edits, load_channel_image

    base_dir = state.project_path.parent if state.project_path else None
    wanted = set()
    for slide_idx, slide in enumerate(state.project.slides):
        if reload or len(state.channel_raw.get(slide_idx, [])) != len(slide.channel_images):
            raws = []
            for k, channel in enumerate(slide.channel_images):
                try:
                    raws.append(load_channel_image(slide, k, base_dir))
                except Exception as exc:  # noqa: BLE001 - a missing file must not stop the rest
                    logger.warning("channel image {!r} not shown: {}", channel.name, exc)
                    raws.append(None)
            state.channel_raw[slide_idx] = raws
        rotations = state.shown_rotations.get(slide_idx, {})
        slide_name = f"Slide {slide_idx}"
        for channel, raw in zip(slide.channel_images, state.channel_raw[slide_idx], strict=True):
            if raw is None:
                continue
            img = apply_slide_edits(raw.copy(), slide, rotations=rotations)
            name = _channel_layer_name(state, slide_idx, channel.name)
            wanted.add(name)
            if name in viewer.layers:
                viewer.layers[name].data = img
                continue
            kwargs = {"rgb": True} if img.ndim == 3 else {"colormap": channel.colour}
            layer = viewer.add_image(img, name=name, visible=False, **kwargs)
            if slide_name in viewer.layers:
                viewer.layers.move(viewer.layers.index(layer), viewer.layers.index(slide_name) + 1)
    for layer in list(viewer.layers):
        if layer.name.startswith(CHANNEL_LAYER_PREFIX) and layer.name not in wanted:
            viewer.layers.remove(layer)


def _show_channel(viewer: "napari.Viewer", state: "WorkflowState", name: str | None) -> None:
    """Show one channel image instead of the slide image, or the slide (``None``)."""
    for slide_idx, slide in enumerate(state.project.slides):
        slide_layer = f"Slide {slide_idx}"
        found = False
        for channel in slide.channel_images:
            layer_name = _channel_layer_name(state, slide_idx, channel.name)
            if layer_name in viewer.layers:
                on = channel.name == name
                viewer.layers[layer_name].visible = on
                found = found or on
        if slide_layer in viewer.layers:
            viewer.layers[slide_layer].visible = not found


def _update_cells_layer(viewer: "napari.Viewer", state: "WorkflowState") -> None:
    """Draw the project's counted cells on the sections they belong to."""
    from atlastrack.gui.overlay_style import OVERLAY_BLENDING
    from atlastrack.project.cells import cell_colour, cells_on_slide

    for slide_idx, slide in enumerate(state.project.slides):
        name = f"Cells {slide_idx}"
        pts, types = cells_on_slide(state.project, slide_idx)
        if name in viewer.layers:
            viewer.layers.remove(name)
        if not len(pts):
            continue
        # About 40 µm across when the pixel size is known: visible, not hiding much.
        size = 40.0 / slide.pixel_um if slide.pixel_um else 20.0
        colours = [cell_colour(state.project, t) for t in types]
        kwargs = {"name": name, "size": size, "face_color": colours, "opacity": 0.95,
                  "blending": OVERLAY_BLENDING}
        # A dark rim keeps each dot readable over bright labelling.
        try:
            viewer.add_points(pts, border_width=0.15, border_color="black", **kwargs)
        except TypeError:  # napari < 0.5
            viewer.add_points(pts, edge_width=0.15, edge_color="black", **kwargs)


def _update_section_numbers(
    viewer: "napari.Viewer", state: "WorkflowState", slide_idx: int
) -> None:
    """Refresh the Points layer that shows section indices at each centroid."""
    slide = state.project.slides[slide_idx]
    centroids, texts = [], []
    for sec in slide.sections:
        x0, y0, x1, y1 = sec.bbox_px
        centroids.append([(y0 + y1) / 2.0, (x0 + x1) / 2.0])
        texts.append(str(sec.index))

    name = f"Section numbers {slide_idx}"
    # Adding a layer makes it the active one; give that back afterwards, or a
    # redraw in the middle of editing (a box being dragged) takes the edit away.
    active = viewer.layers.selection.active
    if name in viewer.layers:
        viewer.layers.remove(name)
    if not centroids:
        return

    # Use opacity=1, transparent face so only the text is drawn.
    # napari ≥ 0.5 renamed edge_color → border_color; try both.
    from atlastrack.gui.overlay_style import OVERLAY_BLENDING

    _pt_kwargs: dict = {
        "size": 1, "face_color": "transparent", "opacity": 1.0,
        "blending": OVERLAY_BLENDING,
    }
    for _ec_key in ("border_color", "edge_color"):
        try:
            lyr = viewer.add_points(centroids, name=name, **{_ec_key: "transparent"}, **_pt_kwargs)
            break
        except TypeError:
            continue
    else:
        lyr = viewer.add_points(centroids, name=name, **_pt_kwargs)

    # Set text after creation.
    try:
        lyr.text = texts
        lyr.text.size = 18
        lyr.text.color = "yellow"
        lyr.text.anchor = "center"
    except Exception:
        pass

    # Keep the numbers as the topmost layer. They are added when sections are
    # detected or a project is loaded, and the atlas overlay arrives afterwards -
    # so without this the labels end up underneath everything drawn later, which
    # is how "the numbers are gone" starts.
    try:
        viewer.layers.move(viewer.layers.index(lyr), len(viewer.layers) - 1)
    except Exception:
        pass
    if active is not None and active in viewer.layers:
        viewer.layers.selection.active = active


def _reload_project_display(viewer: "napari.Viewer", state: "WorkflowState") -> None:
    """After loading a project, reload slide images and redraw section outlines.

    Registration results and CCF coordinates come back with the project JSON, so
    3D / HTML / HERBS / CSV exports work immediately. The atlas is not auto-loaded
    (click *Load atlas* if you want the overlay or the 3D brain); the atlas-overlay
    transform sidecars are resolved from the project folder when needed.
    """
    from atlastrack.gui.section_display import sections_to_outline_labels
    from atlastrack.project.images import rebuild_slide_image

    for slide_idx, slide in enumerate(state.project.slides):
        try:
            # Merged sources, whole-slide flips and per-section flips are all
            # re-applied here - shared with the headless CLI so the two can't drift.
            img, bands = rebuild_slide_image(
                slide,
                base_dir=state.project_path.parent if state.project_path else None,
            )
        except Exception:
            continue
        state.slide_bands[slide_idx] = bands
        state.slide_images[slide_idx] = img
        # The rotations this image was built with, for its channel images.
        state.shown_rotations[slide_idx] = {
            s.index: float(s.rotation_deg or 0.0) for s in slide.sections
        }
        state.active_slide_idx = slide_idx
        name = f"Slide {slide_idx}"
        disp = _display_image_for_slide(state, slide_idx, img)
        if name in viewer.layers:
            viewer.layers[name].data = disp
        else:
            viewer.add_image(disp, name=name, colormap="gray")
        if slide.sections:
            labels = sections_to_outline_labels(img.shape[:2], slide.sections)
            outline = f"Sections {slide_idx}"
            if outline in viewer.layers:
                viewer.layers[outline].data = labels
            else:
                viewer.add_labels(
                    labels, name=outline, opacity=0.85,
                    blending=_section_outline_blending(),
                )
            _update_section_numbers(viewer, state, slide_idx)

    _update_channel_layers(viewer, state, reload=True)
    _update_cells_layer(viewer, state)
    if state.project.slides:
        state.active_slide_idx = 0
    try:
        viewer.reset_view()
    except Exception:
        pass


def _refresh_slide(viewer: "napari.Viewer", state: "WorkflowState") -> None:
    slide_idx = state.active_slide_idx
    if slide_idx is None:
        return
    img = state.slide_images.get(slide_idx)
    if img is None:
        return
    name = f"Slide {slide_idx}"
    if name in viewer.layers:
        viewer.layers[name].data = _display_image_for_slide(state, slide_idx, img)
    # Flips change the slide image in place; its channel images follow.
    if state.project.slides[slide_idx].channel_images:
        _update_channel_layers(viewer, state)


def _window(channel, lo_frac: float, hi_frac: float):
    """Window a 2D channel to its own dtype using 0-1 fractions of full scale."""
    import numpy as np

    a = channel.astype(np.float32)
    full = 255.0 if a.max() <= 255.0 else float(a.max())
    lo, hi = lo_frac * full, hi_frac * full
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0) * full
    return out.astype(channel.dtype)


def _apply_levels(img, levels):
    """Return a copy of ``img`` with per-channel display levels applied."""
    import numpy as np

    if levels is None:
        return img
    low, high = levels.low, levels.high
    if img.ndim == 2:
        return _window(img, low[0], high[0])
    out = img.copy()
    for i in range(min(3, out.shape[2])):
        lo = low[i] if i < len(low) else 0.0
        hi = high[i] if i < len(high) else 1.0
        out[..., i] = _window(out[..., i], lo, hi)
    return out


def _display_image_for_slide(state: "WorkflowState", slide_idx: int, raw):
    """Build the display image for a slide: whole-slide levels + per-section levels.

    The raw array in ``state.slide_images`` is kept untouched (registration uses
    it); only this display copy is windowed. Flips are already baked into the raw
    array, so positions line up.
    """
    if slide_idx >= len(state.project.slides):
        return raw
    slide = state.project.slides[slide_idx]
    if slide.levels is None and not any(s.levels for s in slide.sections):
        return raw  # nothing to apply - show the raw array as-is
    disp = _apply_levels(raw, slide.levels)
    if disp is raw:
        disp = raw.copy()
    for sec in slide.sections:
        if sec.levels is None:
            continue
        x0, y0, x1, y1 = sec.bbox_px
        disp[y0:y1, x0:x1] = _apply_levels(raw[y0:y1, x0:x1], sec.levels)
    return disp
