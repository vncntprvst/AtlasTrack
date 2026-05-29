"""napari entrypoint — `launch()` builds the Viewer and docks workflow widgets."""
from __future__ import annotations

import sys
import traceback

import numpy as np


def launch() -> None:
    """Open the napari viewer with the histo-to-ccf workflow docked."""
    import napari

    _install_exception_handler()
    viewer = napari.Viewer(title="Histo-to-CCF")
    panel = _build_panel(viewer)
    viewer.window.add_dock_widget(panel, area="right", name="Histo→CCF", tabify=False)
    _size_main_window(viewer)
    napari.run()


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
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget
    from histo_to_ccf.gui.widgets.register_panel import RegisterPanelWidget
    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget
    from histo_to_ccf.gui.widgets.slide_loader import SlideLoaderWidget

    settings = load_app_settings()
    state = WorkflowState()

    container = QWidget()
    container.setMinimumWidth(320)
    root = QVBoxLayout(container)
    root.setContentsMargins(0, 0, 0, 0)

    tabs = QTabWidget()
    root.addWidget(tabs)

    # -- Tab 1: Load ---------------------------------------------------------
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
    tabs.addTab(tab_load, "Load")

    # -- Tab 2: Annotate -----------------------------------------------------
    tab_annotate = QWidget()
    ann_layout = QVBoxLayout(tab_annotate)
    ann_layout.setContentsMargins(2, 2, 2, 2)
    probe_picker = ProbePickerWidget(state)
    click_overlay = ClickOverlayWidget(state, viewer)
    ann_layout.addWidget(probe_picker)
    ann_layout.addWidget(click_overlay)
    tabs.addTab(tab_annotate, "Annotate")

    # -- Tab 3: Atlas --------------------------------------------------------
    tab_atlas = QWidget()
    atlas_layout = QVBoxLayout(tab_atlas)
    atlas_layout.setContentsMargins(2, 2, 2, 2)
    atlas_browser = AtlasBrowserWidget(state, viewer, settings=settings)
    ordering = OrderingPanelWidget(state)
    atlas_layout.addWidget(atlas_browser)
    atlas_layout.addWidget(ordering)
    tabs.addTab(tab_atlas, "Atlas")

    # -- Tab 4: Register + Results -------------------------------------------
    tab_register = QWidget()
    reg_layout = QVBoxLayout(tab_register)
    reg_layout.setContentsMargins(2, 2, 2, 2)
    register_panel = RegisterPanelWidget(state, viewer)
    register_panel.apply_settings(settings)
    reg_layout.addWidget(register_panel)
    tabs.addTab(tab_register, "Register")

    # -- Tab 5: Save ---------------------------------------------------------
    tab_save = QWidget()
    save_layout = QVBoxLayout(tab_save)
    save_layout.setContentsMargins(2, 2, 2, 2)
    save_panel = SavePanelWidget(state)
    save_layout.addWidget(save_panel)
    tabs.addTab(tab_save, "Save")

    # Persist settings when the tab changes (cheap enough to do on every switch).
    def _on_tab_change(_idx: int) -> None:
        register_panel.collect_settings(settings)
        atlas_browser.collect_settings(settings)
        save_app_settings(settings)

    tabs.currentChanged.connect(_on_tab_change)

    return container


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
    """Render section outlines + index labels; wire click-to-discard."""
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


def install_discard_handler(
    state: "WorkflowState",
    slide_idx: int,
    viewer: "napari.Viewer",
) -> None:
    """Arm one-click discard on the section-outline Labels layer.

    Activates the Sections layer, switches it to **pick** mode, and connects a
    one-shot handler to ``selected_label`` events.  The first non-zero label the
    user picks is discarded; pick mode is then cancelled automatically.

    This avoids ``viewer.mouse_press_callbacks`` which does not exist in
    napari 0.7.x (Viewer is a Pydantic model with only drag/wheel callbacks).
    """
    from histo_to_ccf.gui.section_display import sections_to_outline_labels

    outline_name = f"Sections {slide_idx}"
    if outline_name not in viewer.layers:
        return

    lyr = viewer.layers[outline_name]

    # Make the Labels layer active and enter pick (eyedropper) mode so that
    # clicking in the canvas updates selected_label.
    viewer.layers.selection.active = lyr
    lyr.mode = "pick"

    def _on_pick(event):
        label_val = int(lyr.selected_label)
        if label_val <= 0:
            return  # clicked background — stay armed

        # Disconnect and reset mode BEFORE mutating data to avoid re-entrancy.
        try:
            lyr.events.selected_label.disconnect(_on_pick)
        except Exception:
            pass
        lyr.mode = "pan_zoom"

        sec_index = label_val - 1  # labels stored as section.index + 1
        slide = state.project.slides[slide_idx]
        before = len(slide.sections)
        slide.sections = [s for s in slide.sections if s.index != sec_index]
        if len(slide.sections) == before:
            return

        new_labels = sections_to_outline_labels(lyr.data.shape[:2], slide.sections)
        lyr.data = new_labels
        _update_section_numbers(viewer, state, slide_idx)

    lyr.events.selected_label.connect(_on_pick)


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


def _refresh_slide(viewer: "napari.Viewer", state: "WorkflowState") -> None:
    slide_idx = state.active_slide_idx
    if slide_idx is None:
        return
    img = state.slide_images.get(slide_idx)
    if img is None:
        return
    name = f"Slide {slide_idx}"
    if name in viewer.layers:
        viewer.layers[name].data = img
