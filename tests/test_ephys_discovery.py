"""Recording discovery: what it derives, and how it behaves with no metadata table."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from atlastrack.ephys.discovery import (
    Penetration,
    RecordingCandidate,
    ShankCoverage,
    StreamInfo,
    attach_metadata,
    coverage_from_tip,
    find_record_nodes,
    group_penetrations,
    grouping_warnings,
    infer_path_ids,
    merge_spans,
    normalise_dye,
    parse_date,
    read_sidecar,
    session_folder,
)


def _stream(recording, probe, coverage, *, subject="LO_07",
            d=date(2026, 5, 8)) -> StreamInfo:
    return StreamInfo(
        recording_dir=f"F:/TJO/{subject}/2026_05_08/{recording}/raw_ephys_data/Record Node 104",
        stream_name=f"Record Node 104#Neuropix-PXI-100.{probe}",
        probe_label=probe, n_channels=384, sampling_rate_hz=30000.0,
        duration_s=900.0, coverage=tuple(coverage),
        subject=subject, session_date=d, recording_label=recording,
    )


def _bank(shank, top=0.0, bottom=705.0, n=96, cols=2) -> ShankCoverage:
    return ShankCoverage(shank=shank, n_sites=n, top_um=top, bottom_um=bottom,
                         step_um=15.0, n_columns=cols)


# ------------------------------------------------------------------ path inference


def test_infer_path_ids_anchors_on_the_date_directory():
    p = Path("F:/TJO/LO_07/2026_05_08/LO_07_005/raw_ephys_data/Record Node 104")
    assert infer_path_ids(p) == ("LO_07", date(2026, 5, 8), "LO_07_005")


def test_infer_path_ids_returns_none_rather_than_guessing():
    """A wrong subject silently splits one penetration in two, so refuse instead."""
    assert infer_path_ids(Path("D:/dump/Record Node 104")) == (None, None, None)


def test_find_record_nodes_skips_zarr_stores(tmp_path: Path):
    (tmp_path / "sess" / "raw" / "Record Node 104").mkdir(parents=True)
    (tmp_path / "sess" / "out.zarr" / "Record Node 999").mkdir(parents=True)

    found = find_record_nodes(tmp_path)
    assert [p.name for p in found] == ["Record Node 104"]


def test_find_record_nodes_on_a_missing_root_is_empty():
    assert find_record_nodes("Z:/nope/never") == []


def test_find_record_nodes_accepts_the_record_node_itself(tmp_path: Path):
    """Pointing the file dialog straight at the record node is the obvious thing."""
    node = tmp_path / "raw_ephys_data" / "Record Node 104"
    (node / "experiment1" / "recording1").mkdir(parents=True)

    assert find_record_nodes(node) == [node]
    assert find_record_nodes(node.parent) == [node]


def test_find_record_nodes_recognises_a_renamed_node_by_its_experiments(tmp_path: Path):
    """The folder name is the part people change; the experiments inside are not."""
    node = tmp_path / "sess" / "ephys_copy"
    (node / "experiment1" / "recording1").mkdir(parents=True)

    assert find_record_nodes(tmp_path) == [node]


def test_find_record_nodes_does_not_descend_past_a_node(tmp_path: Path):
    """A record node's children are experiments, not more nodes."""
    node = tmp_path / "Record Node 104"
    (node / "experiment1" / "Record Node 999").mkdir(parents=True)

    assert find_record_nodes(tmp_path) == [node]


def test_find_record_nodes_ignores_sorting_output_named_like_experiments(tmp_path: Path):
    """AIND writes ``experiment1_...zarr`` stores beside the raw data; not nodes."""
    node = tmp_path / "raw_ephys_data" / "Record Node 104"
    (node / "experiment1" / "recording1").mkdir(parents=True)
    post = tmp_path / "processed_data" / "spike_sorting_output_AIND_ephys" / "postprocessed"
    (post / "experiment1_Record Node 104#ProbeA-AP_recording1_group0.zarr").mkdir(
        parents=True
    )

    assert find_record_nodes(tmp_path) == [node]


# ---------------------------------------------------------------------- the sidecar


def test_parse_date_handles_both_spellings():
    assert parse_date("2026_05_08") == date(2026, 5, 8)
    assert parse_date("2026-05-08") == date(2026, 5, 8)
    assert parse_date("5/8/2026") == date(2026, 5, 8)   # US M/D/YYYY
    assert parse_date("8/25/2025") == date(2025, 8, 25)
    assert parse_date("") is None
    assert parse_date("last tuesday") is None


def test_normalise_dye_reduces_to_the_colour():
    """Colour is the link to the histology project, which is named for it."""
    assert normalise_dye("DiI (red)") == "red"
    assert normalise_dye("CM-DiI (red)") == "red"
    assert normalise_dye("DiD (far red)") == "far red"
    assert normalise_dye("deep red") == "far red"
    assert normalise_dye("green") == "green"
    assert normalise_dye("") is None


def _write_csv(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_read_sidecar_maps_the_labs_headers(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "s.csv", (
        "Subject,Date,Recording,Probe,Insertion depth,Orientation,"
        "Electrodes config,Dye,Notes\n"
        "LO_07,5/8/2026,LO_07_004,A,4976,20d pitch,all shanks 1-96,DiO (green),\n"
    ))
    rows = read_sidecar(csv_path)

    assert len(rows) == 1
    assert rows[0]["subject"] == "LO_07"
    assert rows[0]["date"] == date(2026, 5, 8)
    assert rows[0]["insertion_depth_um"] == pytest.approx(4976.0)
    assert rows[0]["dye"] == "green"
    assert rows[0]["electrodes_config"] == "all shanks 1-96"


def test_read_sidecar_leaves_a_blank_depth_unknown(tmp_path: Path):
    """A blank must not become 0.0 - that would place the tip at the brain surface."""
    csv_path = _write_csv(tmp_path / "s.csv",
                          "Subject,Date,Recording,Probe,Insertion depth\n"
                          "LO_07,5/8/2026,LO_07_004,A,\n"
                          "LO_07,5/8/2026,LO_07_005,A,not measured\n")
    rows = read_sidecar(csv_path)

    assert rows[0]["insertion_depth_um"] is None
    assert rows[1]["insertion_depth_um"] is None


def test_read_sidecar_rejects_an_unrecognisable_table(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "s.csv", "alpha,beta\n1,2\n")

    with pytest.raises(ValueError, match="no recognisable columns"):
        read_sidecar(csv_path)


def test_read_sidecar_accepts_a_custom_mapping(tmp_path: Path):
    csv_path = _write_csv(tmp_path / "s.csv", "animal_id,when,how_deep\nM1,2026-05-08,4200\n")
    rows = read_sidecar(csv_path, columns={
        "subject": ("animal_id",), "date": ("when",),
        "insertion_depth_um": ("how_deep",),
    })

    assert rows[0]["subject"] == "M1"
    assert rows[0]["insertion_depth_um"] == pytest.approx(4200.0)


def test_read_sidecar_refuses_a_spreadsheet(tmp_path: Path):
    xl = tmp_path / "s.xlsx"
    xl.write_bytes(b"PK\x03\x04")

    with pytest.raises(ValueError, match="export it to CSV"):
        read_sidecar(xl)


# --------------------------------------------------------------------- attaching


def test_attach_metadata_matches_on_subject_date_recording_probe():
    streams = [_stream("LO_07_004", "ProbeA", [_bank(0)])]
    rows = [{"subject": "LO_07", "date": date(2026, 5, 8), "recording": "LO_07_004",
             "probe": "A", "insertion_depth_um": 4976.0, "dye": "green",
             "electrodes_config": "all shanks 1-96", "orientation": None, "notes": None}]

    (c,) = attach_metadata(streams, rows)
    assert c.insertion_depth_um == pytest.approx(4976.0)
    assert c.depth_source == "sidecar"
    assert c.dye == "green"


def test_attach_metadata_leaves_unmatched_streams_usable():
    """No table at all is the normal case; discovery must still return structure."""
    streams = [_stream("LO_07_004", "ProbeA", [_bank(0)])]

    (c,) = attach_metadata(streams, None)
    assert c.insertion_depth_um is None
    assert c.depth_source == "unknown"
    assert c.stream.shanks == (0,)


def test_attach_metadata_will_not_borrow_a_depth_across_differing_rows():
    """Two depths in a session and no recording match: refuse rather than pick one."""
    streams = [_stream("LO_07_009", "ProbeA", [_bank(0)])]
    rows = [
        {"subject": "LO_07", "date": date(2026, 5, 8), "recording": "LO_07_001",
         "probe": "A", "insertion_depth_um": 4576.0},
        {"subject": "LO_07", "date": date(2026, 5, 8), "recording": "LO_07_004",
         "probe": "A", "insertion_depth_um": 4976.0},
    ]

    (c,) = attach_metadata(streams, rows)
    assert c.insertion_depth_um is None


def test_attach_metadata_falls_back_when_the_session_agrees():
    streams = [_stream("odd_label", "ProbeA", [_bank(0)])]
    rows = [
        {"subject": "LO_07", "date": date(2026, 5, 8), "recording": "LO_07_003",
         "probe": "A", "insertion_depth_um": 4976.0},
        {"subject": "LO_07", "date": date(2026, 5, 8), "recording": "LO_07_004",
         "probe": "A", "insertion_depth_um": 4976.0},
    ]

    (c,) = attach_metadata(streams, rows)
    assert c.insertion_depth_um == pytest.approx(4976.0)


def test_probe_labels_match_across_spellings():
    streams = [_stream("LO_07_004", "ProbeA", [_bank(0)])]
    for spelling in ("A", "probeA", "ProbeA", "probe A"):
        rows = [{"subject": "LO_07", "date": date(2026, 5, 8), "recording": "LO_07_004",
                 "probe": spelling, "insertion_depth_um": 4976.0}]
        (c,) = attach_metadata(streams, rows)
        assert c.insertion_depth_um == pytest.approx(4976.0), spelling


# ---------------------------------------------------------------------- grouping


def _cand(recording, probe, coverage, depth=None, dye=None, config=None):
    return RecordingCandidate(
        stream=_stream(recording, probe, coverage), insertion_depth_um=depth,
        dye=dye, depth_source="sidecar" if depth is not None else "unknown",
        stated_config=config,
    )


def test_group_penetrations_splits_by_probe_and_dye():
    cands = [
        _cand("LO_07_004", "ProbeA", [_bank(0)], 4976.0, "green"),
        _cand("LO_07_005", "ProbeA", [_bank(0)], 4976.0, "green"),
        _cand("LO_07_004", "ProbeB", [_bank(3)], 5400.0, "green"),
    ]
    pens = group_penetrations(cands)

    assert len(pens) == 2
    a = next(p for p in pens if p.probe_label == "ProbeA")
    assert len(a.recordings) == 2
    assert a.dye == "green"
    assert "LO_07" in a.label and "ProbeA" in a.label


def test_group_penetrations_works_with_no_dye_at_all():
    cands = [_cand("r1", "ProbeA", [_bank(0)]), _cand("r2", "ProbeA", [_bank(0)])]
    pens = group_penetrations(cands)

    assert len(pens) == 1
    assert pens[0].dye is None
    assert len(pens[0].missing_depths()) == 2


# ---------------------------------------------------------------------- coverage


def test_merge_spans_joins_abutting_banks_but_keeps_real_gaps():
    assert merge_spans([(0.0, 705.0), (720.0, 1425.0)]) == [(0.0, 1425.0)]
    assert merge_spans([(0.0, 705.0), (2880.0, 3585.0)]) == [
        (0.0, 705.0), (2880.0, 3585.0)]


def test_coverage_union_of_two_depths_reaches_further_than_either():
    """LO_07 2026-05-08 ProbeA: 4576 then 4976, so the union is 720 + 400 um."""
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("LO_07_002", "ProbeA", [_bank(1)], 4576.0, "green"),
        _cand("LO_07_004", "ProbeA", [_bank(1)], 4976.0, "green"),
    ])
    (span,) = coverage_from_tip(pen, 1)

    assert span[0] == pytest.approx(0.0)
    assert span[1] == pytest.approx(1105.0)   # 705 + 400


def test_coverage_places_the_deep_single_shank_alongside_the_bank():
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("LO_07_004", "ProbeA", [_bank(0)], 4976.0, "green"),
        _cand("LO_07_005", "ProbeA",
              [ShankCoverage(0, 384, 0.0, 5745.0, 15.0, 2)], 4976.0, "green"),
    ])
    (span,) = coverage_from_tip(pen, 0)

    assert span == pytest.approx((0.0, 5745.0))


def test_coverage_omits_a_recording_whose_depth_is_unknown():
    """Mixing a known and an unknown depth would be a guess, so drop the unknown."""
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", None, [
        _cand("a", "ProbeA", [_bank(0)], 4976.0),
        _cand("b", "ProbeA", [_bank(0, 2880.0, 3585.0)], None),
    ])

    assert coverage_from_tip(pen, 0) == [(0.0, 705.0)]
    assert len(pen.missing_depths()) == 1


def test_coverage_with_no_depths_at_all_assumes_one_insertion():
    """The multi-bank case: banks are placed by site position, no depth needed."""
    pen = Penetration(None, None, "ProbeA", None, [
        _cand("a", "ProbeA", [_bank(0, 0.0, 705.0)]),
        _cand("b", "ProbeA", [_bank(0, 720.0, 1425.0)]),
    ])

    assert coverage_from_tip(pen, 0) == [(0.0, 1425.0)]


def test_coverage_of_a_shank_no_recording_touched_is_empty():
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green",
                      [_cand("LO_07_005", "ProbeA", [_bank(0)], 4976.0, "green")])

    assert coverage_from_tip(pen, 2) == []


# ------------------------------------------------------------------- cross-checks


def test_config_mismatch_flags_notes_that_contradict_the_geometry():
    one_shank = _cand("LO_07_005", "ProbeA", [_bank(0)], 4976.0,
                      config="all shanks 1-96")
    assert "all shanks" in (one_shank.config_mismatch() or "")

    four = _cand("LO_07_004", "ProbeA", [_bank(i) for i in range(4)], 4976.0,
                 config="shank 1 single col")
    assert "single shank" in (four.config_mismatch() or "")


def test_config_mismatch_catches_a_bank_stated_wrongly():
    """Sites starting 720 um up the shank are bank 97-192, not 1-96."""
    upper = [_bank(i, 720.0, 1425.0) for i in range(4)]

    wrong = _cand("x", "ProbeA", upper, 4900.0, config="all shanks 1-96")
    assert "electrodes 1-96" in (wrong.config_mismatch() or "")

    right = _cand("x", "ProbeA", upper, 4900.0, config="all shanks 97-192")
    assert right.config_mismatch() is None


def test_no_stated_config_is_not_a_mismatch():
    assert _cand("x", "ProbeA", [_bank(0)], 4976.0).config_mismatch() is None


def test_describe_config_reads_like_the_notes_do():
    four = _stream("LO_07_004", "ProbeA", [_bank(i) for i in range(4)])
    assert four.describe_config() == "all 4 shanks, 0-705 µm from tip"

    deep = _stream("LO_07_005", "ProbeA",
                   [ShankCoverage(0, 384, 0.0, 5745.0, 15.0, 1)])
    assert deep.describe_config() == "shank 0, 0-5745 µm from tip, single column"


def test_absolute_geometry_detection_matches_the_lo06_case():
    """Bank 97-192 reporting y=720.. is absolute; adding the offset again is the bug."""
    absolute = _stream("LO_06_002", "ProbeA", [_bank(0, 720.0, 1425.0)])
    local = _stream("LO_06_002", "ProbeA", [_bank(0, 0.0, 705.0)])

    assert absolute.is_absolute_geometry is True
    assert local.is_absolute_geometry is False


def test_shank_coverage_extent():
    assert _bank(0, 0.0, 705.0).extent_um == pytest.approx(705.0)


def test_derive_electrode_range_inverts_the_bank_offset():
    """LO_06's bank starting 720 um up the shank is electrodes 97-192, as noted."""
    from atlastrack.ephys.discovery import derive_electrode_range

    assert derive_electrode_range(_bank(0, 0.0, 705.0)) == (1, 96)
    assert derive_electrode_range(_bank(0, 720.0, 1425.0)) == (97, 192)
    assert derive_electrode_range(_bank(0, 8640.0, 9345.0)) == (1153, 1248)


def test_derive_electrode_range_refuses_a_single_column_scan():
    """384 sites over 5745 um is one per row; '1-768' would claim twice the sites."""
    from atlastrack.ephys.discovery import derive_electrode_range

    assert derive_electrode_range(ShankCoverage(0, 384, 0.0, 5745.0, 15.0, 2)) is None
    assert derive_electrode_range(ShankCoverage(0, 96, 0.0, 705.0, 15.0, 1)) is None


def test_penetration_reports_its_shanks_and_depths():
    pen = Penetration("LO_07", date(2026, 5, 8), "ProbeA", "green", [
        _cand("LO_07_002", "ProbeA", [_bank(i) for i in range(4)], 4576.0, "green"),
        _cand("LO_07_005", "ProbeA", [_bank(0)], 4976.0, "green"),
    ])

    assert pen.shanks == (0, 1, 2, 3)
    assert pen.depths == [4576.0, 4976.0]
    assert pen.missing_depths() == []


def test_shank_coverage_from_real_geometry_numbers():
    """The two layouts in this dataset, built from their actual x/y values."""
    from atlastrack.ephys.discovery import _shank_coverage

    # LO_07_004: 4 shanks, 2 columns, 48 rows.
    x = np.concatenate([np.tile([s * 250.0, s * 250.0 + 32.0], 48) for s in range(4)])
    y = np.tile(np.repeat(np.arange(48) * 15.0, 2), 4)
    cov = _shank_coverage(x, y, None)
    assert [c.shank for c in cov] == [0, 1, 2, 3]
    assert all(c.n_sites == 96 and c.n_columns == 2 for c in cov)
    assert cov[0].bottom_um == pytest.approx(705.0)

    # LO_07_005 ProbeB: one shank, one site per row, 384 rows over 5745 um.
    x5 = np.tile([750.0, 782.0], 192)
    y5 = np.arange(384) * 15.0
    (c5,) = _shank_coverage(x5, y5, None)
    assert c5.shank == 3
    assert c5.n_sites == 384
    assert c5.bottom_um == pytest.approx(5745.0)
    assert c5.step_um == pytest.approx(15.0)


# ---------------------------------------------- when the path names no session


def _unnamed(recording_dir, probe, coverage, depth=None) -> RecordingCandidate:
    """A stream whose path carries no YYYY_MM_DD component."""
    return RecordingCandidate(
        stream=StreamInfo(
            recording_dir=recording_dir,
            stream_name=f"Record Node 104#Neuropix-PXI-100.{probe}",
            probe_label=probe, n_channels=384, sampling_rate_hz=30000.0,
            duration_s=900.0, coverage=tuple(coverage),
            subject=None, session_date=None, recording_label=None,
        ),
        insertion_depth_um=depth,
    )


def test_session_folder_skips_the_container_directory():
    assert session_folder("D:/x/mouse7/run_02/raw_ephys_data/Record Node 104") \
        == str(Path("D:/x/mouse7/run_02"))
    assert session_folder("D:/x/mouse7/run_02/Record Node 104") \
        == str(Path("D:/x/mouse7/run_02"))


def test_recordings_in_different_folders_do_not_merge_without_a_date():
    """A silent merge would stack different insertions onto one depth axis."""
    cands = [
        _unnamed("D:/x/run_01/raw_ephys_data/Record Node 104", "ProbeA", [_bank(0)]),
        _unnamed("D:/x/run_02/raw_ephys_data/Record Node 104", "ProbeA", [_bank(0)]),
    ]

    pens = group_penetrations(cands)

    assert len(pens) == 2, "these are two folders and possibly two insertions"
    assert {Path(p.folder_hint).name for p in pens} == {"run_01", "run_02"}


def test_two_record_nodes_in_one_folder_still_group_together():
    """Splitting is the fallback, not the goal: one folder is still one session."""
    cands = [
        _unnamed("D:/x/run_01/Record Node 101", "ProbeA", [_bank(0)]),
        _unnamed("D:/x/run_01/Record Node 102", "ProbeA", [_bank(1)]),
    ]

    assert len(group_penetrations(cands)) == 1


def test_a_dated_path_is_unaffected_by_the_fallback():
    a = _stream("LO_07_001", "ProbeA", [_bank(0)])
    b = _stream("LO_07_002", "ProbeA", [_bank(0)])
    pens = group_penetrations([RecordingCandidate(stream=a),
                               RecordingCandidate(stream=b)])

    assert len(pens) == 1
    assert pens[0].folder_hint is None
    assert pens[0].identified_from_path is True


def test_the_label_names_the_folder_when_the_session_is_unknown():
    pen = group_penetrations(
        [_unnamed("D:/x/session_final/Record Node 104", "ProbeA", [_bank(0)])]
    )[0]

    assert pen.label.startswith("session_final ProbeA")


def test_a_folder_grouped_scan_warns_that_it_guessed():
    pens = group_penetrations([
        _unnamed("D:/x/run_01/Record Node 104", "ProbeA", [_bank(0)]),
        _unnamed("D:/x/run_02/Record Node 104", "ProbeA", [_bank(0)]),
    ])

    warnings = grouping_warnings(pens)

    assert warnings
    assert "YYYY_MM_DD" in warnings[0]
    assert "one session folder at a time" in warnings[0]


def test_a_dated_scan_warns_about_nothing():
    pens = group_penetrations([RecordingCandidate(stream=_stream("r1", "ProbeA",
                                                                 [_bank(0)]))])

    assert grouping_warnings(pens) == []


def test_mixed_configs_with_no_depth_in_one_folder_are_called_out():
    """What a stack of different insertions looks like from the outside."""
    pens = group_penetrations([
        _unnamed("D:/x/run_01/Record Node 104", "ProbeA",
                 [_bank(i) for i in range(4)]),
        _unnamed("D:/x/run_01/Record Node 105", "ProbeA",
                 [ShankCoverage(0, 384, 0.0, 5745.0, 15.0, 1)]),
    ])

    warnings = grouping_warnings(pens)

    assert any("different electrode configurations" in w for w in warnings)
    assert any("cannot share a depth axis" in w for w in warnings)
