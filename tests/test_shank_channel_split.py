"""Which channels belong to which shank - and what happens when a shank has none."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.recordings import channels_for_shank, shank_index_from_x

PITCH = 250.0


def _four_shank_x(per_shank: int = 96) -> np.ndarray:
    """A full 4-shank recording: two columns per shank, shanks 250 µm apart."""
    return np.concatenate([
        np.tile([s * PITCH, s * PITCH + 32.0], per_shank // 2) for s in range(4)
    ])


def test_numeric_shank_ids_are_matched_directly():
    ids = np.array(["0"] * 96 + ["1"] * 96 + ["2"] * 96 + ["3"] * 96)

    for s in range(4):
        assert int(channels_for_shank(s, ids).sum()) == 96


def test_a_shank_that_was_not_recorded_gets_nothing():
    """The reported bug: every tab showed the same LFP map.

    LO_07_005 records a single column on one shank of a four-shank probe. Falling back
    to "all channels" when there are not four distinct ids gave shanks 1-3 a copy of
    shank 0's data - four identical panels that look like four measurements.
    """
    ids = np.array(["0"] * 384)

    assert int(channels_for_shank(0, ids).sum()) == 384
    for s in (1, 2, 3):
        assert int(channels_for_shank(s, ids).sum()) == 0


def test_x_positions_identify_the_shank_when_ids_are_missing():
    x = _four_shank_x()

    for s in range(4):
        assert int(channels_for_shank(s, None, x).sum()) == 96


def test_two_columns_of_one_shank_are_not_two_shanks():
    """32 µm apart is a column; 250 µm apart is a shank."""
    x = np.tile([0.0, 32.0], 192)

    assert int(channels_for_shank(0, None, x).sum()) == 384
    assert int(channels_for_shank(1, None, x).sum()) == 0


def test_a_single_column_on_a_far_shank_is_attributed_correctly():
    """LO_07_005 ProbeB: 'shank 4 single col', x around 750 µm -> shank 3."""
    x = np.tile([750.0, 782.0], 192)

    assert shank_index_from_x(x) == 3
    assert int(channels_for_shank(3, None, x).sum()) == 384
    assert int(channels_for_shank(0, None, x).sum()) == 0


def test_shank_index_from_x_is_none_when_several_shanks_are_present():
    assert shank_index_from_x(_four_shank_x()) is None


def test_no_information_at_all_returns_none():
    assert channels_for_shank(0, None, None) is None
    assert shank_index_from_x([]) is None


def test_non_numeric_ids_fall_through_to_x():
    ids = np.array(["probeA-shank-a"] * 384)
    x = np.tile([500.0, 532.0], 192)

    assert int(channels_for_shank(2, ids, x).sum()) == 384


def test_geometry_beats_a_group_id_that_restarts_at_zero():
    """LO_07_005 ProbeB: one shank at x=750 - physically shank 3 - reported group 0.

    SpikeInterface's ``group`` numbers the groups present in *this* recording, so a
    single-shank recording says 0 whichever shank it is. Believing it puts probe B's
    only data on the shank-0 tab and leaves shank 3 empty.
    """
    ids = np.array(["0"] * 384)
    x = np.tile([750.0, 782.0], 192)

    assert int(channels_for_shank(3, ids, x).sum()) == 384
    assert int(channels_for_shank(0, ids, x).sum()) == 0


def test_group_ids_still_used_when_there_is_no_geometry():
    ids = np.array(["0"] * 96 + ["1"] * 96 + ["2"] * 96 + ["3"] * 96)

    assert int(channels_for_shank(2, ids, None).sum()) == 96


@pytest.mark.qt
def test_the_dialog_leaves_unrecorded_shanks_empty(qtbot) -> None:
    """End to end: a single-shank recording must not populate the other tabs."""
    pytest.importorskip("pyqtgraph")
    import napari

    from atlastrack.gui.widgets.ephys_alignment_panel import EphysProbeAlignmentDialog
    from atlastrack.gui.workflow import WorkflowState
    from atlastrack.project.schema import ProbeSpec, ProbeType, Shank

    state = WorkflowState()
    state.project.probes.append(
        ProbeSpec(
            label="ProbeA", type=ProbeType(name="NP", n_shanks=4),
            shanks=[
                Shank(index=i, tip_ccf_um=(5000.0, 2000.0, 5000.0),
                      entry_ccf_um=(5000.0, 2000.0, 1000.0))
                for i in range(4)
            ],
        )
    )
    n = 64
    lfp = {
        "depths_um": np.linspace(0.0, 900.0, n),
        "psd": np.random.default_rng(0).random((n, 16)),
        "freqs": np.linspace(0.0, 300.0, 16),
        "shank_ids": np.array(["0"] * n),
        "x_um": np.zeros(n),
    }

    viewer = napari.Viewer(show=False)
    try:
        dlg = EphysProbeAlignmentDialog(state, 0, lfp_result=lfp)
        qtbot.addWidget(dlg)

        assert dlg.panels[0].view().lfp_data() is not None
        for i in (1, 2, 3):
            assert dlg.panels[i].view().lfp_data() is None, f"shank {i} got copied data"
            assert dlg.panels[i].view().available_modes() == []
    finally:
        viewer.close()
