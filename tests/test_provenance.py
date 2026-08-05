"""Tests for project/provenance.py - detecting an export that has gone stale."""
from __future__ import annotations

import json

from histo_to_ccf.project.provenance import describe_staleness, write_export_provenance


def _project_file(tmp_path, text: str = '{"version": 1}'):
    path = tmp_path / "proj.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_records_source_and_options(tmp_path) -> None:
    project = _project_file(tmp_path)
    prov = write_export_provenance(
        tmp_path / "out.provenance.json",
        project_json=project,
        outputs=[tmp_path / "out.csv", tmp_path / "out - Paxinos.csv"],
        n_rows=3072,
        options={"rigid_array": True, "lock_spacing_um": 250},
    )

    record = json.loads(prov.read_text(encoding="utf-8"))
    assert record["tool"] == "histo2ccf"
    assert record["n_channel_rows"] == 3072
    assert record["source_project"] == str(project)
    assert record["outputs"] == ["out.csv", "out - Paxinos.csv"]
    assert record["options"]["lock_spacing_um"] == 250
    assert len(record["source_sha256"]) == 64


def test_unchanged_project_is_not_stale(tmp_path) -> None:
    project = _project_file(tmp_path)
    prov = write_export_provenance(
        tmp_path / "out.provenance.json",
        project_json=project,
        outputs=[tmp_path / "out.csv"],
        n_rows=10,
    )
    stale, message = describe_staleness(prov)
    assert stale is False
    assert "up to date" in message


def test_edited_project_makes_the_export_stale(tmp_path) -> None:
    project = _project_file(tmp_path)
    prov = write_export_provenance(
        tmp_path / "out.provenance.json",
        project_json=project,
        outputs=[tmp_path / "out.csv"],
        n_rows=10,
    )
    # This is the LO_06 situation: the project gets corrected after exporting.
    project.write_text('{"version": 1, "corrected": true}', encoding="utf-8")

    stale, message = describe_staleness(prov)
    assert stale is True
    assert "STALE" in message


def test_missing_project_is_reported_not_assumed_stale(tmp_path) -> None:
    project = _project_file(tmp_path)
    prov = write_export_provenance(
        tmp_path / "out.provenance.json",
        project_json=project,
        outputs=[tmp_path / "out.csv"],
        n_rows=10,
    )
    project.unlink()

    stale, message = describe_staleness(prov)
    assert stale is False
    assert "not found" in message


def test_unreadable_sidecar_is_reported(tmp_path) -> None:
    bad = tmp_path / "broken.provenance.json"
    bad.write_text("{not json", encoding="utf-8")
    stale, message = describe_staleness(bad)
    assert stale is False
    assert "could not read" in message
