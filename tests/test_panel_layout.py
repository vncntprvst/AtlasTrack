"""Panel grouping and ordering the user asked for, pinned so a refactor keeps it."""
from __future__ import annotations

import typing

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QGroupBox, QLabel, QPushButton, QWidget

from atlastrack.gui.widgets.click_overlay import ClickOverlayWidget
from atlastrack.gui.widgets.image_tools import ImageToolsWidget
from atlastrack.gui.widgets.separators import hline, section_header
from atlastrack.gui.workflow import WorkflowState

pytestmark = pytest.mark.qt


def _labels(widget: QWidget) -> list[str]:
    """Every visible label/button/group title, in layout order."""
    out: list[str] = []
    layout = widget.layout()
    for i in range(layout.count()):
        item = layout.itemAt(i)
        child = item.widget()
        if isinstance(child, QGroupBox):
            out.append(child.title())
        elif isinstance(child, QWidget):
            out.extend(t.text() for t in child.findChildren(QLabel) if t.text())
        elif item.layout() is not None:
            inner = item.layout()
            for j in range(inner.count()):
                w = inner.itemAt(j).widget()
                if isinstance(w, (QLabel, QPushButton)) and w.text():
                    out.append(w.text())
    return out


def test_section_header_shows_its_title(qtbot) -> None:
    header = section_header("Adjustments")
    qtbot.addWidget(header)

    assert [t.text() for t in header.findChildren(QLabel)] == ["Adjustments"]


def test_hline_is_a_horizontal_rule(qtbot) -> None:
    from qtpy.QtWidgets import QFrame

    line = hline()
    qtbot.addWidget(line)

    assert line.frameShape() == QFrame.HLine


def test_histology_groups_run_adjustments_scope_flip_levels(qtbot) -> None:
    """Scope first: it says what Flip and Levels below will act on."""
    widget = ImageToolsWidget(WorkflowState())
    qtbot.addWidget(widget)

    labels = _labels(widget)
    order = [labels.index(t) for t in ("Adjustments", "Scope", "Flip", "Levels (display)")]

    assert order == sorted(order), f"unexpected order: {labels}"


def test_probes_tab_heads_the_track_controls(qtbot) -> None:
    class _FakeViewer:
        layers: typing.ClassVar[list] = []
        mouse_drag_callbacks: typing.ClassVar[list] = []

    widget = ClickOverlayWidget(WorkflowState(), _FakeViewer())
    qtbot.addWidget(widget)

    labels = _labels(widget)
    assert "Probe tracks" in labels
    for gone in ("Marker type:", "Entry via:", "Select / move", "Clear selected"):
        assert gone not in labels
    assert labels.index("Probe tracks") < labels.index("Probe:")
    buttons = [b.text() for b in widget.findChildren(QPushButton)]
    assert buttons.index("Add track") < buttons.index("Clear all tracks")


def test_landmark_preview_matches_the_normal_overlay_width() -> None:
    """The drag preview must be a 1px contour like the ordinary overlay.

    Thickening the splat was the old fix for the holes the warp opens; it hid
    the anatomy. Gaps are closed instead.
    """
    from atlastrack.gui.widgets.register_panel import (
        _LANDMARK_CONTOUR_CLOSE_GAPS,
        _LANDMARK_CONTOUR_THICKNESS,
    )

    assert _LANDMARK_CONTOUR_THICKNESS == 0
    assert _LANDMARK_CONTOUR_CLOSE_GAPS >= 3


def test_closing_mends_the_warped_contour_without_widening_it() -> None:
    """Closing must bridge holes at roughly the ink of a plain 1px splat."""
    import numpy as np
    from scipy import ndimage as ndi

    from atlastrack.registration.landmarks_warp import warp_contour_image

    # A dense ring, warped by a mild stretch that pulls its pixels apart.
    theta = np.linspace(0, 2 * np.pi, 900, endpoint=False)
    edge_rc = np.column_stack([60 + 40 * np.sin(theta), 60 + 40 * np.cos(theta)])
    source = np.array([[20.0, 20.0], [100.0, 20.0], [20.0, 100.0], [100.0, 100.0]])
    target = source * 1.25

    plain = warp_contour_image(edge_rc, source, target, (160, 160))
    closed = warp_contour_image(edge_rc, source, target, (160, 160), close_gaps=3)
    thick = warp_contour_image(edge_rc, source, target, (160, 160), thickness=1)

    n_plain = int(plain.sum())
    struct = np.ones((3, 3))
    frag_plain = ndi.label(plain, structure=struct)[1]
    frag_closed = ndi.label(closed, structure=struct)[1]

    assert frag_closed <= frag_plain, "closing must not fragment the contour further"
    # Far less ink than thickening, which is the whole point.
    assert int(closed.sum()) < int(thick.sum())
    assert int(closed.sum()) < 2 * n_plain


@pytest.mark.qt
def test_side_panel_does_not_set_a_tall_window(qtbot) -> None:
    """Tabs scroll, so adding controls never makes the window taller than a screen."""
    import napari

    from atlastrack.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz)
        assert panel.minimumSizeHint().height() < 300
        # ...and never narrower than the tab on show, so nothing is cut off.
        from qtpy.QtWidgets import QTabWidget

        tabs = next(t for t in panel.findChildren(QTabWidget) if t.count() >= 5)
        for k in range(tabs.count()):
            tabs.setCurrentIndex(k)
            need = tabs.widget(k).widget().minimumSizeHint().width()
            assert panel.minimumWidth() >= need, tabs.tabText(k)
    finally:
        viewer.close()


@pytest.mark.qt
def test_side_panel_combo_boxes_do_not_size_to_their_longest_entry(qtbot) -> None:
    """A long combo entry (a probe or stream name) must not widen the whole panel.

    Combo boxes size to their longest entry by default; the Ephys "Probe map" list
    alone made the panel over 500 px wide. (Measured in the running app: with this,
    every tab fits in about 320 px.)
    """
    import napari
    from qtpy.QtWidgets import QComboBox

    from atlastrack.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz)
        # Both side panels: the registration tabs and 3D & Export.
        combos = panel.findChildren(QComboBox) + viz.findChildren(QComboBox)
        assert combos
        narrow = QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        wide = [c.currentText() or c.objectName() for c in combos if c.sizeAdjustPolicy() != narrow]
        assert not wide, wide
    finally:
        viewer.close()


def test_long_probe_names_do_not_widen_the_panel(qtbot) -> None:
    """The "N probe(s) in project: ..." line set the side panel's width (550 px with
    three long names), and the panel stayed that wide on every tab."""
    from atlastrack.gui.widgets.probe_picker import ProbePickerWidget
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.project.schema import ProbeSpec, ProbeType

    state = WorkflowState()
    for label in ("ProbeA_green_01302026", "ProbeA_red_01282026", "ProbeA_red_01292026"):
        state.project.probes.append(ProbeSpec(label=label, type=ProbeType(name="NP", n_shanks=1),
                                              shanks=[]))
    picker = ProbePickerWidget(state)
    qtbot.addWidget(picker)
    picker._refresh_rename_combo()
    picker._status.setText("3 probe(s) in project: " + ", ".join(p.label for p in state.project.probes))
    assert picker.minimumSizeHint().width() < 400
