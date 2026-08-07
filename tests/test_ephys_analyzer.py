"""Reading spike features from a sorting analyzer.

The SpikeInterface-dependent read is exercised against real data elsewhere; these
cover the parts that are pure logic, including the off-site filter that a real
recording forced.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.analyzer import (
    SpikeFeatures,
    find_shank_analyzers,
    is_analyzer_dir,
    restrict_to_epochs,
)


def _features(n=100, seed=0) -> SpikeFeatures:
    rng = np.random.default_rng(seed)
    return SpikeFeatures(
        times_s=np.sort(rng.uniform(0, 100, n)),
        depth_um=rng.uniform(0, 700, n),
        amplitude=rng.normal(1.0, 0.1, n),
        unit_ids=rng.integers(0, 10, n),
        channel_depth_um=np.arange(48) * 15.0,
        channel_x_um=np.tile([0.0, 32.0], 24),
        duration_s=100.0,
        n_units=10,
    )


def test_mismatched_spike_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="but there are"):
        SpikeFeatures(
            times_s=np.zeros(5), depth_um=np.zeros(4), amplitude=np.zeros(5),
            unit_ids=np.zeros(5), channel_depth_um=np.zeros(2),
            channel_x_um=np.zeros(2), duration_s=1.0, n_units=1,
        )


def test_restrict_to_epochs_keeps_only_those_windows() -> None:
    f = _features(1000)

    out = restrict_to_epochs(f, [(10.0, 20.0), (50.0, 60.0)])

    assert out.times_s.size < f.times_s.size
    inside = ((out.times_s >= 10) & (out.times_s < 20)) | (
        (out.times_s >= 50) & (out.times_s < 60)
    )
    assert inside.all()
    assert out.duration_s == pytest.approx(20.0)


def test_restrict_keeps_every_array_in_step() -> None:
    f = _features(500)

    out = restrict_to_epochs(f, [(0.0, 50.0)])

    n = out.times_s.size
    assert out.depth_um.size == n and out.amplitude.size == n and out.unit_ids.size == n
    # Channel geometry is a property of the probe, not of the excerpt.
    assert out.channel_depth_um.size == f.channel_depth_um.size


def test_no_epochs_means_no_restriction_not_no_spikes() -> None:
    f = _features(200)

    assert restrict_to_epochs(f, []).times_s.size == 200


def test_is_analyzer_dir_rejects_a_plain_folder(tmp_path) -> None:
    assert is_analyzer_dir(tmp_path) is False
    assert is_analyzer_dir(tmp_path / "missing") is False


def test_is_analyzer_dir_accepts_the_zarr_layout(tmp_path) -> None:
    (tmp_path / "sorting").mkdir()
    (tmp_path / "extensions").mkdir()

    assert is_analyzer_dir(tmp_path) is True


def test_shank_analyzers_sort_numerically_not_lexically(tmp_path) -> None:
    """group10 must come after group2, or shanks get silently mislabelled."""
    root = tmp_path / "postprocessed"
    root.mkdir()
    for n in (0, 2, 10):
        d = root / f"block0_rec1_group{n}.zarr"
        (d / "sorting").mkdir(parents=True)
        (d / "extensions").mkdir()

    found = find_shank_analyzers(tmp_path)

    assert [p.name for p in found] == [
        "block0_rec1_group0.zarr",
        "block0_rec1_group2.zarr",
        "block0_rec1_group10.zarr",
    ]


def test_an_analyzer_path_itself_is_accepted(tmp_path) -> None:
    (tmp_path / "sorting").mkdir()
    (tmp_path / "extensions").mkdir()

    assert find_shank_analyzers(tmp_path) == [tmp_path]


def test_nothing_found_in_an_unrelated_folder(tmp_path) -> None:
    (tmp_path / "raw_ephys_data").mkdir()

    assert find_shank_analyzers(tmp_path) == []
