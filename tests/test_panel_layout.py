"""Panel grouping and ordering the user asked for, pinned so a refactor keeps it."""
from __future__ import annotations

import typing

import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QGroupBox, QLabel, QPushButton, QWidget

from histo_to_ccf.gui.widgets.click_overlay import ClickOverlayWidget
from histo_to_ccf.gui.widgets.image_tools import ImageToolsWidget
from histo_to_ccf.gui.widgets.separators import hline, section_header
from histo_to_ccf.gui.workflow import WorkflowState

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


def test_histology_groups_run_adjustments_flip_levels_scope(qtbot) -> None:
    widget = ImageToolsWidget(WorkflowState())
    qtbot.addWidget(widget)

    labels = _labels(widget)
    order = [labels.index(t) for t in ("Adjustments", "Flip", "Levels (display)", "Scope")]

    assert order == sorted(order), f"unexpected order: {labels}"


def test_probes_tab_heads_the_marker_controls(qtbot) -> None:
    class _FakeViewer:
        layers: typing.ClassVar[list] = []
        mouse_drag_callbacks: typing.ClassVar[list] = []

    widget = ClickOverlayWidget(WorkflowState(), _FakeViewer())
    qtbot.addWidget(widget)

    labels = _labels(widget)
    assert "Probe markers" in labels
    # "Mode" said nothing about what it chose.
    assert "Marker type:" in labels
    assert "Mode:" not in labels
    assert labels.index("Probe markers") < labels.index("Marker type:")


def test_landmark_contour_is_thin_enough_to_see_through(qtbot) -> None:
    """The drag preview drew a 5px contour that covered the anatomy."""
    from histo_to_ccf.gui.widgets.register_panel import _LANDMARK_CONTOUR_THICKNESS

    assert _LANDMARK_CONTOUR_THICKNESS < 2
    # 0 would break the contour into dots once the points spread under the warp.
    assert _LANDMARK_CONTOUR_THICKNESS >= 1
