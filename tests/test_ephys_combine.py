"""Stacking several recordings of one penetration onto one axis per shank."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys.combine import (
    RecordingFeatures,
    depths_from_tip,
    stack_penetration,
    stack_shank,
)
from atlastrack.ephys.epochs import common_median_reference
from atlastrack.ephys.features import covered_rows, power_image

FREQS = np.linspace(0.0, 300.0, 31)


def _rec(label, *, shanks=(0, 1, 2, 3), n_rows=48, insertion=4976.0, y0=0.0,
         columns=2, level=1.0, electrode_range=None) -> RecordingFeatures:
    """A synthetic recording with NP2.0 geometry: 15 µm rows, 250 µm shank pitch."""
    xs, ys = [], []
    for s in shanks:
        for r in range(n_rows):
            for c in range(columns):
                xs.append(250.0 * s + 32.0 * c)
                ys.append(y0 + 15.0 * r)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    # Power that varies with depth, so a misplacement shows up as a mismatch.
    psd = level * (1.0 + y[:, None] / 1000.0) * np.ones((y.size, FREQS.size))
    return RecordingFeatures(
        label=label, stream_name=f"{label}.ProbeA", insertion_depth_um=insertion,
        freqs_hz=FREQS, psd=psd, axial_um=y, x_um=x,
        shank_ids=np.zeros(y.size, dtype=int), electrode_range=electrode_range,
    )


# ------------------------------------------------------------------ referencing


def test_per_shank_reference_leaves_a_shank_specific_signal_alone():
    """The whole point: a shank's own signal must survive its own reference."""
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(500, 8))
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    traces[:, 4:] += 50.0  # a large offset on shank 1 only

    out = common_median_reference(traces, groups)

    assert abs(float(np.median(out[:, :4]))) < 1.0
    assert abs(float(np.median(out[:, 4:]))) < 1.0
    # Referencing across all channels would instead split the offset between shanks.
    naive = common_median_reference(traces)
    assert abs(float(np.median(naive[:, 4:]))) > 10.0


def test_reference_groups_must_match_the_channel_count():
    with pytest.raises(ValueError, match="3 entries for 8 channels"):
        common_median_reference(np.zeros((10, 8)), np.array([0, 1, 2]))


# --------------------------------------------------------------------- geometry


def test_depths_from_tip_adds_the_chisel_tip():
    rec = _rec("r", shanks=(0,), n_rows=2)
    mask = np.ones(rec.n_channels, dtype=bool)

    d = depths_from_tip(rec, mask)

    assert float(d.min()) == pytest.approx(175.0)  # tip length, not 0
    assert float(d.max()) == pytest.approx(175.0 + 15.0)


def test_depths_from_tip_does_not_double_count_an_absolute_bank():
    """A probe map covering the whole shank already includes the bank offset."""
    rec = _rec("bank2", shanks=(0,), n_rows=48, y0=720.0, electrode_range=(97, 192))
    mask = np.ones(rec.n_channels, dtype=bool)

    d = depths_from_tip(rec, mask)

    assert float(d.min()) == pytest.approx(720.0 + 175.0)


# ---------------------------------------------------------------------- stacking


def test_a_shank_no_recording_reached_gives_none():
    """LO_07_005 covers one shank; the others must come back empty, not copied."""
    single = _rec("005", shanks=(0,), n_rows=384, columns=1)

    assert stack_shank([single], 0) is not None
    assert stack_shank([single], 2) is None


def test_shanks_covered_by_different_recordings_all_get_data():
    """The LO_07 ProbeA case: 005 on shank 0, 004 on all four."""
    deep = _rec("005", shanks=(0,), n_rows=384, columns=1)
    bank = _rec("004", shanks=(0, 1, 2, 3), n_rows=48)

    stacks = stack_penetration([deep, bank], [0, 1, 2, 3])

    assert sorted(stacks) == [0, 1, 2, 3]
    assert len(stacks[0].contributions) == 2
    assert [c.label for c in stacks[1].contributions] == ["004"]
    # Shank 0 reaches the whole column; the others only the bottom bank.
    assert stacks[0].covered_spans_um()[-1][1] > 5000.0
    assert stacks[2].covered_spans_um()[-1][1] < 1000.0


def test_the_widest_recording_is_the_level_reference():
    deep = _rec("005", shanks=(0,), n_rows=384, columns=1)
    bank = _rec("004", shanks=(0,), n_rows=48)

    stack = stack_shank([bank, deep], 0)

    ref = [c for c in stack.contributions if c.is_level_reference]
    assert [c.label for c in ref] == ["005"]


def test_a_level_difference_is_measured_and_removed():
    """Two recordings of the same tissue at different absolute power must agree."""
    deep = _rec("005", shanks=(0,), n_rows=384, columns=1)
    bank = _rec("004", shanks=(0,), n_rows=48, level=10.0)  # a decade louder

    stack = stack_shank([deep, bank], 0)

    other = next(c for c in stack.contributions if c.label == "004")
    assert other.level_offset_dec == pytest.approx(1.0, abs=0.05)
    assert other.overlap_um > 700.0
    # After correction there is no *step* where the louder recording ends: the
    # synthetic power ramps with depth on purpose, so test bin-to-bin jumps, not the
    # overall range, which the ramp alone takes from 1.0 to 1.59 over 600 µm.
    column = stack.psd[stack.covered][:, 0]
    jumps = np.abs(np.diff(np.log10(column)))
    assert float(jumps.max()) < 0.05, "a decade step would show up here as ~1.0"


def test_no_overlap_means_no_level_claim():
    """Nothing can be said about the relative level of recordings that never meet."""
    low = _rec("bank1", shanks=(0,), n_rows=48)
    high = _rec("bank4", shanks=(0,), n_rows=48, y0=8640.0, electrode_range=(1153, 1248))

    stack = stack_shank([low, high], 0)

    offsets = {c.label: c.level_offset_dec for c in stack.contributions}
    assert offsets["bank4"] is None or offsets["bank1"] is None
    assert stack.gaps_um(), "a bank 8 mm further up must leave a reported gap"


def test_different_insertion_depths_are_placed_in_the_deepest_frame():
    """The same electrode at two insertion depths samples different tissue."""
    deeper = _rec("003", shanks=(0,), n_rows=48, insertion=4976.0)
    shallower = _rec("001", shanks=(0,), n_rows=48, insertion=4576.0)

    stack = stack_shank([deeper, shallower], 0)

    assert stack.reference_depth_um == pytest.approx(4976.0)
    by_label = {c.label: c for c in stack.contributions}
    # 400 µm shallower insertion -> its sites sit 400 µm further up the final track.
    assert by_label["001"].top_um - by_label["003"].top_um == pytest.approx(400.0)


def test_uncovered_depths_are_nan_not_zero():
    """"Nothing recorded here" must not be readable as "the LFP is quiet here"."""
    low = _rec("bank1", shanks=(0,), n_rows=48)
    high = _rec("bank3", shanks=(0,), n_rows=48, y0=2880.0, electrode_range=(385, 480))

    stack = stack_shank([low, high], 0)

    assert not stack.covered.all()
    assert np.isnan(stack.psd[~stack.covered]).all()
    gaps = stack.gaps_um()
    assert len(gaps) == 1
    assert gaps[0][1] - gaps[0][0] == pytest.approx(2880.0 - 705.0 - 15.0, abs=30.0)


def test_the_grid_is_uniform():
    """The display stretches the map between its first and last depth."""
    stack = stack_shank([_rec("a", shanks=(0,), n_rows=48)], 0)

    steps = np.diff(stack.depth_from_tip_um)
    assert np.allclose(steps, steps[0])


def test_abutting_banks_leave_no_gap():
    low = _rec("bank1", shanks=(0,), n_rows=48)
    nxt = _rec("bank2", shanks=(0,), n_rows=48, y0=720.0, electrode_range=(97, 192))

    stack = stack_shank([low, nxt], 0)

    assert stack.gaps_um() == []


def test_describe_names_the_recordings_and_their_reach():
    deep = _rec("005", shanks=(0,), n_rows=384, columns=1)
    bank = _rec("004", shanks=(0,), n_rows=48)

    text = stack_shank([deep, bank], 0).describe()

    assert "005" in text and "004" in text and "µm" in text


# ------------------------------------------------------------------- the image


def test_power_image_survives_uncovered_rows():
    psd = np.ones((5, 4))
    psd[2] = np.nan

    img = power_image(psd)

    assert img.dtype == np.uint8
    assert (img[2] == 0).all()
    assert covered_rows(psd).tolist() == [True, True, False, True, True]


def test_power_image_scaling_ignores_uncovered_rows():
    """An all-NaN row must not drag the normalisation of the real ones."""
    psd = np.array([[1.0], [10.0], [np.nan]])

    img = power_image(psd, per_freq=True)

    assert img[0, 0] == 0
    assert img[1, 0] == 255
    assert img[2, 0] == 0
