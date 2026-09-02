"""Curved shank tracks - and the guarantee that tip+entry alone keeps working."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.probes.catalog import get_layout
from atlastrack.probes.channels import (
    channel_ccf_coords,
    curved_channel_ccf_coords,
    shank_channel_coords,
)
from atlastrack.probes.track_path import (
    arc_lengths,
    max_deviation_um,
    path_length_um,
    points_at_distance,
    tangents_at_distance,
    track_polyline,
)
from atlastrack.project.schema import Shank

TIP = (5000.0, 4000.0, 5000.0)
ENTRY = (5000.0, 4000.0, 1000.0)


# -- the minimum-information case, which must never regress -----------------


def test_no_waypoints_is_the_straight_tip_to_entry_line():
    path = track_polyline(TIP, ENTRY, None)

    assert path.shape == (2, 3)
    assert np.allclose(path[0], TIP)
    assert np.allclose(path[-1], ENTRY)
    assert path_length_um(path) == pytest.approx(4000.0)


def test_empty_waypoint_list_behaves_as_none():
    assert np.allclose(track_polyline(TIP, ENTRY, []), track_polyline(TIP, ENTRY, None))


def test_curved_placement_matches_the_straight_one_when_there_is_no_curve():
    """A curvature feature that perturbs existing straight tracks would be a bug."""
    layout = get_layout("NP1")
    depths = layout.site_depths_from_tip_um()
    laterals = layout.site_lateral_offsets_um()

    straight = channel_ccf_coords(ENTRY, TIP, depths, site_lateral_offsets_um=laterals)
    curved = curved_channel_ccf_coords(TIP, ENTRY, [], depths,
                                       site_lateral_offsets_um=laterals)

    assert np.allclose(straight, curved, atol=1e-9)


def test_a_shank_with_no_waypoints_places_channels_exactly_as_before():
    layout = get_layout("NP1")
    plain = Shank(index=0, tip_ccf_um=TIP, entry_ccf_um=ENTRY)

    assert np.allclose(
        shank_channel_coords(plain, layout),
        channel_ccf_coords(ENTRY, TIP, layout.site_depths_from_tip_um(),
                           site_lateral_offsets_um=layout.site_lateral_offsets_um()),
        atol=1e-9,
    )


# -- ordering and hygiene ---------------------------------------------------


def test_waypoints_are_sorted_along_the_track_whatever_order_they_are_given():
    far = (5000.0, 4100.0, 2000.0)   # near the entry
    near = (5000.0, 4100.0, 4000.0)  # near the tip

    path = track_polyline(TIP, ENTRY, [far, near])

    # Tip first, then ascending distance from it, entry last.
    assert np.allclose(path[0], TIP)
    assert np.allclose(path[1], near)
    assert np.allclose(path[2], far)
    assert np.allclose(path[-1], ENTRY)


def test_duplicate_points_do_not_fold_the_path_back_on_itself():
    mid = (5000.0, 4100.0, 3000.0)

    path = track_polyline(TIP, ENTRY, [mid, mid])

    assert len(path) == 3
    assert np.all(np.diff(arc_lengths(path)) > 0)


def test_a_waypoint_on_the_line_leaves_the_length_unchanged():
    path = track_polyline(TIP, ENTRY, [(5000.0, 4000.0, 3000.0)])

    assert path_length_um(path) == pytest.approx(4000.0)
    assert max_deviation_um(path) == pytest.approx(0.0, abs=1e-9)


# -- curvature --------------------------------------------------------------


def test_a_curved_track_is_longer_than_the_chord():
    bowed = track_polyline(TIP, ENTRY, [(5000.0, 4200.0, 3000.0)])

    assert path_length_um(bowed) > 4000.0
    assert max_deviation_um(bowed) == pytest.approx(200.0, abs=1.0)


def test_sites_follow_the_bend_rather_than_the_chord():
    """The failure this feature exists to fix: a mid-shank site off by the sagitta."""
    bow = (5000.0, 4200.0, 3000.0)
    depths = np.array([2000.0])  # 2000 µm from the tip, near the bow

    straight = channel_ccf_coords(ENTRY, TIP, depths)
    curved = curved_channel_ccf_coords(TIP, ENTRY, [bow], depths)

    assert abs(curved[0, 1] - straight[0, 1]) > 100.0  # displaced in ML, as the bow is


def test_the_tip_anchor_holds_and_nearby_sites_move_only_slightly():
    """Arc length runs from the tip, so uncertainty at the far end stays at the far end.

    A waypoint does tilt the first segment - the path really does head towards it -
    so sites near the tip shift a little, in proportion to how far along they are.
    What must not move is the tip itself.
    """
    depths = np.array([0.0, 100.0, 2000.0])

    a = curved_channel_ccf_coords(TIP, ENTRY, [(5000.0, 4300.0, 1500.0)], depths)
    b = curved_channel_ccf_coords(TIP, ENTRY, [(5000.0, 3700.0, 1500.0)], depths)

    assert np.allclose(a[0], b[0], atol=1e-6)          # the tip is untouched
    near = float(np.linalg.norm(a[1] - b[1]))
    mid = float(np.linalg.norm(a[2] - b[2]))
    assert near < 20.0                                  # 100 µm along: barely moved
    assert mid > 10 * near                              # 2000 µm along: fully affected


def test_points_beyond_the_ends_extrapolate_rather_than_clip():
    path = track_polyline(TIP, ENTRY, [(5000.0, 4100.0, 3000.0)])

    out = points_at_distance(path, [-200.0, path_length_um(path) + 200.0])

    assert out[0, 2] > TIP[2]        # past the tip, i.e. deeper
    assert out[1, 2] < ENTRY[2]      # above the entry, i.e. outside the brain


def test_tangents_are_unit_length_and_follow_the_bend():
    path = track_polyline(TIP, ENTRY, [(5000.0, 4400.0, 3000.0)])
    lengths = arc_lengths(path)

    tangents = tangents_at_distance(path, [10.0, lengths[-1] - 10.0])

    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert not np.allclose(tangents[0], tangents[1])  # direction really changed


def test_a_shank_with_waypoints_moves_its_channels():
    layout = get_layout("NP1")
    plain = Shank(index=0, tip_ccf_um=TIP, entry_ccf_um=ENTRY)
    bowed = Shank(index=0, tip_ccf_um=TIP, entry_ccf_um=ENTRY,
                  track_points_ccf_um=[(5000.0, 4250.0, 3000.0)])

    a = shank_channel_coords(plain, layout)
    b = shank_channel_coords(bowed, layout)

    assert not np.allclose(a, b)
    # The path is pinned at *both* ends, so the bow displaces the middle: sites near
    # the tip and near the entry barely move, and the worst error is mid-shank. That
    # is exactly the error a straight-line placement makes, and why it is worth fixing.
    shift = np.linalg.norm(a - b, axis=1)
    deepest = int(np.argmax(a[:, 2]))
    shallowest = int(np.argmin(a[:, 2]))
    assert shift[deepest] < 40.0
    assert shift[shallowest] < 40.0
    assert shift.max() > 150.0
    worst_depth = a[int(np.argmax(shift)), 2]
    assert 2000.0 < worst_depth < 4000.0  # mid-track, out where the bow is


def test_degenerate_track_does_not_raise():
    path = track_polyline(TIP, TIP, [TIP])

    assert path_length_um(path) == pytest.approx(0.0)
    assert points_at_distance(path, [0.0, 100.0]).shape == (2, 3)
    assert max_deviation_um(path) == pytest.approx(0.0)
