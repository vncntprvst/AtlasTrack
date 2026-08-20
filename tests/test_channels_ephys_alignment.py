"""Phase 4: an ephys alignment must actually reach the exports.

Before this, `Shank.ephys` was stored and then ignored by every exporter, so placing
an alignment changed nothing you could ship.
"""
from __future__ import annotations

import csv
import json
from typing import ClassVar

import numpy as np
import pytest

from histo_to_ccf.probes.catalog import get_layout
from histo_to_ccf.probes.channels import (
    aligned_site_depths_from_tip,
    export_channel_csv,
    export_ibl_channel_locations,
    export_paxinos_csv,
    project_channel_coords_with_source,
    shank_channel_coords,
)
from histo_to_ccf.project.schema import (
    EphysAlignment,
    ProbeSpec,
    ProbeType,
    Project,
    Shank,
)

TRACK_UM = 4000.0


class _Atlas:
    structures: ClassVar[dict] = {
        "A": {"rgb_triplet": [1, 2, 3], "id": 111},
        "B": {"rgb_triplet": [4, 5, 6], "id": 222},
    }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        if dv < 0 or dv > 9000:
            return "Outside atlas"
        return "A" if dv < 2500 else "B"


def _project(ephys: EphysAlignment | None = None) -> Project:
    shank = Shank(
        index=0,
        # Straight down in DV: entry at 1000, tip at 5000 -> a 4000 µm track.
        entry_ccf_um=(5000.0, 2000.0, 1000.0),
        tip_ccf_um=(5000.0, 2000.0, 1000.0 + TRACK_UM),
        ephys=ephys,
    )
    project = Project()
    project.probes.append(
        ProbeSpec(label="p1", type=ProbeType(name="NP1", n_shanks=1), shanks=[shank])
    )
    return project


def _alignment(*pairs, **kwargs) -> EphysAlignment:
    """A landmark set with the two track end points, as the model stores it."""
    ordered = sorted(pairs)
    return EphysAlignment(
        feature_um=[0.0, *[p[0] for p in ordered], TRACK_UM],
        track_um=[0.0, *[p[1] for p in ordered], TRACK_UM],
        **kwargs,
    )


# -- the depth warp --------------------------------------------------------


def test_no_alignment_leaves_the_geometry_alone():
    shank = _project().probes[0].shanks[0]
    depths = np.array([0.0, 1000.0, 2000.0])

    out, used = aligned_site_depths_from_tip(shank, depths, TRACK_UM)

    assert used is False
    assert np.allclose(out, depths)


def test_an_alignment_with_no_user_landmarks_is_the_identity():
    """Two entries whose ends were never moved: nothing was pinned, nothing moves."""
    shank = _project(_alignment()).probes[0].shanks[0]
    depths = np.array([0.0, 1000.0, 2000.0])

    out, used = aligned_site_depths_from_tip(shank, depths, TRACK_UM)

    assert used is False
    assert np.allclose(out, depths)


def test_moving_only_the_end_points_is_still_a_real_alignment():
    """Identity is the test, not length.

    The end markers are draggable on purpose - "the brain starts here" is exactly the
    claim the LFP can contradict - so two entries whose ends were moved describe a
    perfectly good affine map. Bailing on ``len == 2`` exported geometry while
    reporting the shank as unaligned.
    """
    shifted = EphysAlignment(
        feature_um=[300.0, TRACK_UM + 300.0], track_um=[0.0, TRACK_UM]
    )
    shank = _project(shifted).probes[0].shanks[0]
    depths = np.array([0.0, 1000.0, 2000.0])

    out, used = aligned_site_depths_from_tip(shank, depths, TRACK_UM)

    assert used is True
    # Both ends moved by the same 300 µm, so it is a pure shift.
    assert np.allclose(out, depths + 300.0)


def test_moving_one_end_rescales_rather_than_shifts():
    """Surface moved, tip left alone: the map is anchored at the tip and stretches.

    ``depths`` are µm **from the tip**, so index 0 is the tip end and index 1 the
    surface end - which is the pair that moves here.
    """
    stretched = EphysAlignment(feature_um=[300.0, TRACK_UM], track_um=[0.0, TRACK_UM])
    shank = _project(stretched).probes[0].shanks[0]
    depths = np.array([0.0, TRACK_UM])

    out, used = aligned_site_depths_from_tip(shank, depths, TRACK_UM)

    assert used is True
    assert out[0] == pytest.approx(0.0, abs=1.0)   # the tip end is pinned
    assert out[1] > depths[1]                      # the surface end moves out by 300


def test_a_dragged_surface_marker_matches_the_equivalent_end_move():
    """The GUI turns a dragged end into a user landmark; both must agree."""
    from histo_to_ccf.ephys.landmarks import Landmarks

    dragged = Landmarks.identity(0.0, TRACK_UM).added(300.0, 0.0)
    as_landmark = _project(EphysAlignment(
        feature_um=list(dragged.feature_um), track_um=list(dragged.track_um)
    )).probes[0].shanks[0]
    as_ends = _project(EphysAlignment(
        feature_um=[300.0, TRACK_UM + 300.0], track_um=[0.0, TRACK_UM]
    )).probes[0].shanks[0]
    depths = np.array([175.0, 1000.0, 3000.0])

    a, a_used = aligned_site_depths_from_tip(as_landmark, depths, TRACK_UM)
    b, b_used = aligned_site_depths_from_tip(as_ends, depths, TRACK_UM)

    assert a_used and b_used
    assert np.allclose(a, b)


def test_a_length_mismatch_is_refused_rather_than_guessed():
    shank = _project(
        EphysAlignment(feature_um=[0.0, 100.0, TRACK_UM], track_um=[0.0, TRACK_UM])
    ).probes[0].shanks[0]
    depths = np.array([0.0, 1000.0])

    out, used = aligned_site_depths_from_tip(shank, depths, TRACK_UM)

    assert used is False
    assert np.allclose(out, depths)


def test_one_landmark_shifts_every_channel_by_the_same_amount():
    """Uniform extremes make a single landmark a pure offset - here 200 µm deeper."""
    shank = _project(_alignment((1500.0, 1700.0))).probes[0].shanks[0]
    depths = np.array([0.0, 1000.0, 2000.0, 3900.0])

    out, used = aligned_site_depths_from_tip(shank, depths, TRACK_UM)

    assert used is True
    # 200 µm deeper below the surface = 200 µm *less* far from the tip.
    assert np.allclose(out, depths - 200.0)


def test_the_warp_direction_is_not_flipped():
    """The failure that would look plausible: shank placed end for end."""
    shank = _project(_alignment((1000.0, 1400.0))).probes[0].shanks[0]

    out, _used = aligned_site_depths_from_tip(shank, np.array([0.0, 100.0]), TRACK_UM)

    # A site nearer the tip stays nearer the tip.
    assert out[0] < out[1]


def test_insertion_depth_offsets_the_feature_axis():
    """The manipulator depth and the histology track length are not the same number."""
    plain = _project(_alignment((1500.0, 1500.0))).probes[0].shanks[0]
    offset = _project(
        _alignment((1500.0, 1500.0), insertion_depth_um=TRACK_UM + 300.0)
    ).probes[0].shanks[0]
    depths = np.array([500.0, 2500.0])

    a, _ = aligned_site_depths_from_tip(plain, depths, TRACK_UM)
    b, _ = aligned_site_depths_from_tip(offset, depths, TRACK_UM)

    assert np.allclose(a, depths)  # identity landmark, matched axes
    assert np.allclose(b, depths - 300.0)  # electrodes sat 300 µm deeper than the track


def test_extremes_mode_is_honoured_when_stored():
    pairs = [(800.0, 880.0), (2000.0, 2200.0), (3200.0, 3520.0)]  # on a line, slope 1.1
    uniform = _project(_alignment(*pairs)).probes[0].shanks[0]
    linear = _project(_alignment(*pairs, extremes_mode="linear")).probes[0].shanks[0]
    depths = np.array([TRACK_UM])  # feature depth 0 = the surface, in the tail

    a, _ = aligned_site_depths_from_tip(uniform, depths, TRACK_UM)
    b, _ = aligned_site_depths_from_tip(linear, depths, TRACK_UM)

    assert not np.allclose(a, b)


# -- the exports -----------------------------------------------------------


def test_channel_coords_move_when_an_alignment_is_placed():
    layout = get_layout("NP1")
    plain = shank_channel_coords(_project().probes[0].shanks[0], layout)
    aligned = shank_channel_coords(
        _project(_alignment((1500.0, 1700.0))).probes[0].shanks[0], layout
    )

    assert plain is not None and aligned is not None
    # The landmark says feature depth 1500 belongs at track depth 1700 - 200 µm
    # *deeper* - so on a straight-down track every channel gains 200 µm of DV.
    assert np.allclose(aligned[:, 2], plain[:, 2] + 200.0)


def test_use_ephys_false_gives_the_geometric_placement():
    layout = get_layout("NP1")
    shank = _project(_alignment((1500.0, 1700.0))).probes[0].shanks[0]

    assert np.allclose(
        shank_channel_coords(shank, layout, use_ephys=False),
        shank_channel_coords(_project().probes[0].shanks[0], layout),
    )


def test_channel_csv_records_which_source_each_depth_came_from(tmp_path):
    path = tmp_path / "channels.csv"

    export_channel_csv(_project(_alignment((1500.0, 1700.0))), path)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows
    assert {r["depth_source"] for r in rows} == {"ephys_alignment"}

    plain = tmp_path / "plain.csv"
    export_channel_csv(_project(), plain)
    assert {r["depth_source"] for r in csv.DictReader(plain.open(encoding="utf-8"))} == {
        "geometry"
    }


def test_channel_csv_adds_regions_when_an_atlas_is_given(tmp_path):
    path = tmp_path / "channels.csv"

    export_channel_csv(_project(), path, atlas=_Atlas())

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert {r["region"] for r in rows} <= {"A", "B", ""}
    assert any(r["region"] == "A" for r in rows)
    assert any(r["region"] == "B" for r in rows)


def test_paxinos_export_follows_the_alignment_too(tmp_path):
    """The gap this closes: both exporters were ignoring Shank.ephys."""
    plain, aligned = tmp_path / "a.csv", tmp_path / "b.csv"

    export_paxinos_csv(_project(), plain)
    export_paxinos_csv(_project(_alignment((1500.0, 1700.0))), aligned)

    a = [r["dv_mm"] for r in csv.DictReader(plain.open(encoding="utf-8"))]
    b = [r["dv_mm"] for r in csv.DictReader(aligned.open(encoding="utf-8"))]
    assert a != b


def test_project_coords_report_the_source_per_shank():
    out = project_channel_coords_with_source(_project(_alignment((1500.0, 1700.0))))

    (_coords, used), = out.values()
    assert used is True


# -- IBL interchange -------------------------------------------------------


def test_ibl_export_writes_the_expected_keys(tmp_path):
    written = export_ibl_channel_locations(
        _project(_alignment((1500.0, 1700.0))), tmp_path, atlas=_Atlas()
    )

    locations = tmp_path / "p1_shank0" / "channel_locations.json"
    assert locations in written
    payload = json.loads(locations.read_text(encoding="utf-8"))
    assert payload["origin"]["depth_source"] == "ephys_alignment"
    # Not bregma - say so in the file rather than let a reader assume.
    assert "NOT bregma" in payload["origin"]["axes"]
    entry = payload["channel_0"]
    assert set(entry) == {"x", "y", "z", "axial", "lateral", "brain_region",
                          "brain_region_id"}
    assert entry["brain_region"] in {"A", "B", ""}
    assert entry["brain_region_id"] in {111, 222, 0}


def test_ibl_axis_naming_maps_ml_to_x_and_ap_to_y(tmp_path):
    """Our (AP, ML, DV) must land in IBL's (y, x, z), not be shipped in our order."""
    project = _project()
    export_ibl_channel_locations(project, tmp_path)

    payload = json.loads(
        (tmp_path / "p1_shank0" / "channel_locations.json").read_text(encoding="utf-8")
    )
    ap, ml, dv = shank_channel_coords(project.probes[0].shanks[0], get_layout("NP1"))[0]
    assert payload["channel_0"]["x"] == pytest.approx(ml)
    assert payload["channel_0"]["y"] == pytest.approx(ap)
    assert payload["channel_0"]["z"] == pytest.approx(dv)
    assert ap != ml  # the test would pass trivially if they matched


def test_prev_alignments_accumulate_rather_than_overwrite(tmp_path):
    first = _project(_alignment((1500.0, 1700.0), created_at="2026-01-01T00:00:00"))
    second = _project(_alignment((2500.0, 2400.0), created_at="2026-02-02T00:00:00"))

    export_ibl_channel_locations(first, tmp_path)
    export_ibl_channel_locations(second, tmp_path)

    prev = json.loads(
        (tmp_path / "p1_shank0" / "prev_alignments.json").read_text(encoding="utf-8")
    )
    assert sorted(prev) == ["2026-01-01T00:00:00", "2026-02-02T00:00:00"]
    feature, track = prev["2026-02-02T00:00:00"]
    assert feature[1] == pytest.approx(2500.0)
    assert track[1] == pytest.approx(2400.0)


def test_no_prev_alignments_file_without_an_alignment(tmp_path):
    export_ibl_channel_locations(_project(), tmp_path)

    assert not (tmp_path / "p1_shank0" / "prev_alignments.json").exists()


def test_unregistered_shank_is_skipped_not_written_blank(tmp_path):
    project = _project()
    project.probes[0].shanks[0].tip_ccf_um = None

    assert export_ibl_channel_locations(project, tmp_path) == []
