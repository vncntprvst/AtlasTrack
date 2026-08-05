"""napari entrypoint - `launch()` builds the Viewer and docks workflow widgets."""
from __future__ import annotations

import sys
import traceback

import numpy as np


def launch() -> None:
    """Open the napari viewer with the histo-to-ccf workflow docked."""
    import napari

    _install_exception_handler()
    try:
        viewer = napari.Viewer(title="Histo-to-CCF")
    except Exception as exc:
        # A dead GPU/OpenGL context (bad driver, RDP session, disabled GPU) makes
        # Viewer creation raise - print an actionable GL diagnosis instead of a
        # raw traceback, then exit.
        from histo_to_ccf.gui.gl_diagnostics import report_launch_failure

        report_launch_failure(exc)
        raise SystemExit(1) from exc

    panel, viz_panel = _build_panel(viewer)
    # Workflow (Registration) on the left; 3D visualization + export on the right.
    viewer.window.add_dock_widget(panel, area="left", name="Registration", tabify=False)
    viewer.window.add_dock_widget(viz_panel, area="right", name="3D & Export", tabify=False)
    _hide_layer_panels(viewer)
    _size_main_window(viewer)
    napari.run()


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


def _build_panel(viewer: "napari.Viewer") -> "QWidget":
    """Construct the main dock panel and wire up all sub-widgets."""
    from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

    from histo_to_ccf.config import load_app_settings, save_app_settings
    from histo_to_ccf.gui.workflow import WorkflowState
    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget
    from histo_to_ccf.gui.widgets.ephys_panel import EphysPanelWidget
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget
    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget
    from histo_to_ccf.gui.widgets.slide_loader import SlideLoaderWidget
    from histo_to_ccf.gui.widgets.viz_export_panel import VizExportPanelWidget

    settings = load_app_settings()
    state = WorkflowState()

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
        on_slide_loaded=lambda idx, img: _on_slide_loaded(viewer, state, idx, img),
        on_sections_detected=lambda secs: _on_sections_detected(viewer, state, secs),
    )
    load_layout.addWidget(slide_loader)
    load_layout.addWidget(image_tools)

    # -- Atlas: choose atlas + assign AP + section ordering ------------------
    tab_atlas = QWidget()
    atlas_layout = QVBoxLayout(tab_atlas)
    atlas_layout.setContentsMargins(2, 2, 2, 2)
    atlas_browser = AtlasBrowserWidget(state, viewer, settings=settings)
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

    # Tab order: Histology → Atlas → Probes → Register → Ephys.
    tabs.addTab(tab_load, "Histology")
    tabs.addTab(tab_atlas, "Atlas")
    tabs.addTab(tab_annotate, "Probes")
    tabs.addTab(tab_register, "Register")
    tabs.addTab(tab_ephys, "Ephys")

    # 3D visualization + export live in their own permanent panel (right dock),
    # not inside the Register tab.
    viz_panel = VizExportPanelWidget(state, viewer)
    viz_panel.apply_settings(settings)

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

    def _on_project_loaded() -> None:
        _reload_project_display(viewer, state)
        _refresh_panels()
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
    _install_project_menu(viewer, state, settings=settings,
                          on_loaded=_on_project_loaded,
                          on_cleared=_on_project_cleared)
    # A "Registration" menu hosts the parameters dialog (kept out of the panel),
    # and napari's default menus are hidden - the user only wants Project +
    # Registration in the bar.
    _install_registration_menu(viewer, register_panel)
    _keep_only_menus(viewer, {"Project", "Registration"})
    _install_wheel_pan(viewer)

    # Persist settings when the tab changes (cheap enough to do on every switch).
    def _on_tab_change(_idx: int) -> None:
        register_panel.collect_settings(settings)
        atlas_browser.collect_settings(settings)
        viz_panel.collect_settings(settings)
        save_app_settings(settings)

    tabs.currentChanged.connect(_on_tab_change)

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
    """Add a "Project" menu (first in the menu bar) with Save / Save As… / Load.

    These are file operations (not workflow steps), so they belong in the menu
    bar rather than a docked tab. Save/Load reuse :class:`SavePanelWidget`'s
    logic via a hidden instance so behaviour stays in one place. ``on_loaded``
    runs after a successful load (redraw canvas + repopulate tabs); it defaults to
    just redrawing the canvas. Best-effort: if the Qt main window or menu bar is
    unavailable (headless), do nothing.
    """
    from pathlib import Path

    from qtpy.QtWidgets import QMenu

    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget

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
        # Closing discards in-memory work, so confirm first (best-effort dialog).
        try:
            from qtpy.QtWidgets import QMessageBox

            resp = QMessageBox.question(
                helper, "Close project",
                "Close the current project? This clears the loaded slides, "
                "sections, probes and registration from the app. Unsaved changes "
                "will be lost.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
        except Exception:  # noqa: BLE001 - headless: proceed without a prompt
            pass
        state.reset()
        if on_cleared is not None:
            on_cleared()

    from qtpy.QtCore import Qt

    save_action = menu.addAction("Save Project")
    save_action.triggered.connect(_save)
    save_as_action = menu.addAction("Save Project As…")
    save_as_action.triggered.connect(_save_as)
    menu.addSeparator()
    load_action = menu.addAction("Load Project…")
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
            from histo_to_ccf.config import save_app_settings

            save_app_settings(settings)
        except Exception:  # noqa: BLE001 - best-effort persistence
            pass

    recent_menu.aboutToShow.connect(_rebuild_recent)
    _rebuild_recent()  # populate once so it isn't empty before first open

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


def _install_registration_menu(viewer: "napari.Viewer", register_panel) -> None:
    """Add a "Registration" menu whose "Parameters" opens the params dialog.

    The registration parameters were moved out of the Register panel (the
    defaults are good); this is where to bring them back up when needed.
    Best-effort: no-op if the Qt menu bar is unavailable (headless).
    """
    from qtpy.QtWidgets import QMenu

    try:
        menubar = viewer.window._qt_window.menuBar()
    except Exception:
        return

    menu = QMenu("Registration", menubar)
    menubar.addMenu(menu)
    params_action = menu.addAction("Parameters")
    params_action.triggered.connect(register_panel.open_parameters_dialog)


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


def _keep_only_menus(viewer: "napari.Viewer", keep: set[str]) -> None:
    """Hide every top-level menu-bar menu except those whose title is in ``keep``.

    napari adds File / View / Plugins / Window / Help; the user only wants Project
    and Registration. Hiding (not removing) is reversible and robust to napari
    re-adding menus. Best-effort: no-op if the menu bar is unavailable.
    """
    try:
        menubar = viewer.window._qt_window.menuBar()
    except Exception:
        return

    for action in menubar.actions():
        sub = action.menu()
        title = (sub.title() if sub is not None else action.text()).replace("&", "")
        if title not in keep:
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
    from histo_to_ccf.gui.section_display import sections_to_outline_labels

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
        lyr = viewer.add_labels(labels, name=outline_name, opacity=0.85)

    # --- Section number text layer ---
    _update_section_numbers(viewer, state, slide_idx)


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
    if name in viewer.layers:
        viewer.layers.remove(name)
    if not centroids:
        return

    # Use opacity=1, transparent face so only the text is drawn.
    # napari ≥ 0.5 renamed edge_color → border_color; try both.
    _pt_kwargs: dict = {"size": 1, "face_color": "transparent", "opacity": 1.0}
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


def _reload_project_display(viewer: "napari.Viewer", state: "WorkflowState") -> None:
    """After loading a project, reload slide images and redraw section outlines.

    Registration results and CCF coordinates come back with the project JSON, so
    3D / HTML / HERBS / CSV exports work immediately. The atlas is not auto-loaded
    (click *Load atlas* if you want the overlay or the 3D brain); the atlas-overlay
    transform sidecars are resolved from the project folder when needed.
    """
    from histo_to_ccf.gui.section_display import sections_to_outline_labels
    from histo_to_ccf.project.images import rebuild_slide_image

    for slide_idx, slide in enumerate(state.project.slides):
        try:
            # Merged sources, whole-slide flips and per-section flips are all
            # re-applied here - shared with the headless CLI so the two can't drift.
            img, bands = rebuild_slide_image(slide)
        except Exception:
            continue
        state.slide_bands[slide_idx] = bands
        state.slide_images[slide_idx] = img
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
                viewer.add_labels(labels, name=outline, opacity=0.85)
            _update_section_numbers(viewer, state, slide_idx)

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
