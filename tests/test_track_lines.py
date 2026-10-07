"""Where a probe track's line is drawn on the sections."""
from __future__ import annotations

import numpy as np

from atlastrack.gui.widgets.click_overlay import track_segments


def test_one_section_line_runs_on_past_the_entry_to_the_box_edge() -> None:
    (a, b), = track_segments((50, 80), (0, 0, 100, 100), (50, 40), (0, 0, 100, 100))
    assert np.allclose(a, (50, 80))
    assert np.allclose(b, (50, 0))          # straight up through the entry, to the top


def test_tip_and_entry_on_different_sections() -> None:
    tip_box, entry_box = (0, 0, 100, 100), (200, 0, 300, 100)
    (t0, t1), (e0, e1) = track_segments((50, 80), tip_box, (260, 40), entry_box)
    # Tip's section: from the tip toward the entry placed at the same spot in this
    # box (60, 40), on to the box edge.
    assert np.allclose(t0, (50, 80)) and np.allclose(t1, (70, 0))
    # Entry's section: from the entry away from the tip placed likewise (250, 80).
    assert np.allclose(e0, (260, 40)) and np.allclose(e1, (270, 0))


def test_no_line_until_both_markers_are_placed() -> None:
    assert track_segments((50, 80), (0, 0, 100, 100), None, None) == []
