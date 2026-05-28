"""napari entrypoint — `launch()` builds the Viewer and docks workflow widgets."""
from __future__ import annotations


def launch() -> None:
    """Open the napari viewer with the histo-to-ccf workflow docked."""
    import napari

    viewer = napari.Viewer(title="Histo-to-CCF")
    panel = _build_panel(viewer)
    viewer.window.add_dock_widget(panel, area="right", name="Histo→CCF", tabify=False)
    napari.run()


def _build_panel(viewer: "napari.Viewer") -> "QWidget":
    """Construct the main dock panel and wire up all sub-widgets."""
    from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

    from histo_to_ccf.gui.workflow import WorkflowState
    from histo_to_ccf.gui.widgets.atlas_browser import AtlasBrowserWidget
    from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget
    from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget
    from histo_to_ccf.gui.widgets.ordering_panel import OrderingPanelWidget
    from histo_to_ccf.gui.widgets.probe_picker import ProbePickerWidget
    from histo_to_ccf.gui.widgets.save_panel import SavePanelWidget
    from histo_to_ccf.gui.widgets.slide_loader import SlideLoaderWidget

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
    atlas_browser = AtlasBrowserWidget(state, viewer)
    ordering = OrderingPanelWidget(state)
    atlas_layout.addWidget(atlas_browser)
    atlas_layout.addWidget(ordering)
    tabs.addTab(tab_atlas, "Atlas")

    # -- Tab 4: Save ---------------------------------------------------------
    tab_save = QWidget()
    save_layout = QVBoxLayout(tab_save)
    save_layout.setContentsMargins(2, 2, 2, 2)
    save_panel = SavePanelWidget(state)
    save_layout.addWidget(save_panel)
    tabs.addTab(tab_save, "Save")

    return container


# ---------------------------------------------------------------------------
# Viewer update helpers (called from widget callbacks)
# ---------------------------------------------------------------------------

def _on_slide_loaded(viewer: "napari.Viewer", state: "WorkflowState", slide_idx: int, img) -> None:
    """Show the newly loaded slide image in the viewer."""
    name = f"Slide {slide_idx}"
    if name in viewer.layers:
        viewer.layers[name].data = img
    else:
        viewer.add_image(img, name=name, colormap="gray")
    viewer.reset_view()


def _on_sections_detected(viewer: "napari.Viewer", state: "WorkflowState", sections) -> None:
    """Overlay section bounding boxes as a Shapes layer."""
    slide_idx = state.active_slide_idx
    if slide_idx is None:
        return
    slide = state.project.slides[slide_idx]
    rects = []
    for sec in slide.sections:
        x0, y0, x1, y1 = sec.bbox_px
        # napari Shapes: [[row, col], ...] i.e. [[y, x], ...]
        rects.append([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])

    name = f"Sections {slide_idx}"
    if name in viewer.layers:
        viewer.layers[name].data = rects
    else:
        viewer.add_shapes(
            rects,
            name=name,
            shape_type="polygon",
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
        )


def _refresh_slide(viewer: "napari.Viewer", state: "WorkflowState") -> None:
    """Refresh the active slide layer after flip or level change."""
    slide_idx = state.active_slide_idx
    if slide_idx is None:
        return
    img = state.slide_images.get(slide_idx)
    if img is None:
        return
    name = f"Slide {slide_idx}"
    if name in viewer.layers:
        viewer.layers[name].data = img
