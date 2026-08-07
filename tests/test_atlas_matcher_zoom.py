"""The matcher's wheel zoom must stay inside a range the user can get back from."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent

from histo_to_ccf.gui.widgets.atlas_matcher import (
    _ZOOM_MAX,
    _ZOOM_MIN,
    _ImagePane,
    _to_pixmap,
)

pytestmark = pytest.mark.qt


def _pane(qtbot) -> _ImagePane:
    pane = _ImagePane()
    qtbot.addWidget(pane)
    pane.resize(400, 300)
    rng = np.random.default_rng(0)
    pane.set_base(_to_pixmap(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)), fit=True)
    return pane


def _wheel(pane: _ImagePane, direction: int) -> None:
    pane.wheelEvent(
        QWheelEvent(
            QPointF(200, 150), QPointF(200, 150), QPoint(0, 0),
            QPoint(0, 120 * direction), Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
        )
    )


def test_zooming_in_stops_at_the_limit(qtbot) -> None:
    pane = _pane(qtbot)

    for _ in range(200):
        _wheel(pane, +1)

    assert pane.transform().m11() <= _ZOOM_MAX


def test_zooming_out_stops_at_the_limit(qtbot) -> None:
    pane = _pane(qtbot)

    for _ in range(200):
        _wheel(pane, -1)

    assert pane.transform().m11() >= _ZOOM_MIN


def test_a_single_notch_still_zooms(qtbot) -> None:
    pane = _pane(qtbot)
    before = pane.transform().m11()

    _wheel(pane, +1)

    assert pane.transform().m11() == pytest.approx(before * 1.25)


def test_the_view_recovers_after_hitting_the_limit(qtbot) -> None:
    """A clamp that latched would be worse than no clamp - zooming back out
    after pinning at maximum must work."""
    pane = _pane(qtbot)
    for _ in range(200):
        _wheel(pane, +1)
    pinned = pane.transform().m11()

    _wheel(pane, -1)

    assert pane.transform().m11() < pinned
