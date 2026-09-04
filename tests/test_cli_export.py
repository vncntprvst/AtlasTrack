"""Tests for the ``atlastrack export`` command and the shared rigid-array helper.

The export path deliberately works from the coordinates already stored in a
project, so a corrected project can be re-exported without re-registering.
"""
from __future__ import annotations

import csv
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from atlastrack.cli import app
from atlastrack.probes.fitting import enforce_rigid_arrays
from atlastrack.project.io import save_project
from atlastrack.project.schema import (
    AtlasRef,
    Point2D,
    ProbeSpec,
    ProbeType,
    Project,
    Shank,
)

runner = CliRunner()


class _Atlas:
    """Two-region fake so ``export`` never loads (or downloads) a real atlas."""

    structures: ClassVar[dict] = {
        "A": {"rgb_triplet": [1, 2, 3], "id": 111},
        "B": {"rgb_triplet": [4, 5, 6], "id": 222},
    }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        return "A" if dv < 3500 else "B"


@pytest.fixture(autouse=True)
def _fake_atlas(monkeypatch):
    """Every test here gets the fake atlas; ``requested`` records what was asked."""
    requested: list[str] = []

    def _load(name):
        requested.append(name)
        return _Atlas()

    monkeypatch.setattr("atlastrack.cli._load_export_atlas", _load)
    return requested

# A 4-shank NP2.0 whose picks are deliberately uneven: the shanks should sit
# 250 µm apart in ML but shank 2 is pulled out to a 400 µm gap.
_TIP_ML = [5700.0, 5950.0, 6350.0, 6600.0]


def _make_project(*, n_shanks: int = 4) -> Project:
    shanks = [
        Shank(
            index=i,
            tip_px=Point2D(x_px=50.0 + i, y_px=70.0),
            tip_section_idx=0,
            tip_ccf_um=(7000.0, _TIP_ML[i], 6000.0),
            entry_px=Point2D(x_px=50.0 + i, y_px=10.0),
            entry_section_idx=0,
            entry_ccf_um=(7000.0, _TIP_ML[i], 1000.0),
        )
        for i in range(n_shanks)
    ]
    probe = ProbeSpec(
        label="ProbeA",
        type=ProbeType(name="Neuropixels 2.0 (4-shank)", n_shanks=n_shanks),
        shanks=shanks,
    )
    return Project(atlas=AtlasRef(), slides=[], probes=[probe])


def _read(path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def test_export_writes_ccf_and_paxinos(tmp_path) -> None:
    project_json = tmp_path / "LO_test_whole.json"
    save_project(_make_project(), project_json)

    result = runner.invoke(app, ["export", str(project_json)])
    assert result.exit_code == 0, result.output

    ccf = tmp_path / "LO_test_whole.csv"
    pax = tmp_path / "LO_test_whole - Paxinos.csv"
    assert ccf.exists() and pax.exists()

    header, rows = _read(ccf)
    # depth_source is appended, never inserted: the original six columns keep their
    # positions so a consumer reading by index is not silently shifted.
    assert header[:6] == ["probe", "shank", "channel", "ap_um", "ml_um", "dv_um"]
    assert header[6] == "depth_source"
    assert {r[6] for r in rows} == {"geometry"}  # no ephys alignment in this project
    assert len(rows) == 4 * 384  # 4 shanks x 384 channels
    assert {r[0] for r in rows} == {"ProbeA"}

    pax_header, pax_rows = _read(pax)
    assert pax_header == ["probe", "shank", "channel", "ap_mm", "ml_mm", "dv_mm"]
    assert len(pax_rows) == len(rows)


def test_export_adds_region_columns_from_the_project_atlas(tmp_path, _fake_atlas) -> None:
    """The gap this closes: the exporter could add regions but the CLI never asked."""
    project_json = tmp_path / "LO_test_whole.json"
    save_project(_make_project(), project_json)

    result = runner.invoke(app, ["export", str(project_json)])
    assert result.exit_code == 0, result.output

    header, rows = _read(tmp_path / "LO_test_whole.csv")
    assert header[-3:] == ["region", "region_id", "region_color"]
    regions = {r[-3] for r in rows}
    assert regions == {"A", "B"}
    assert {r[-2] for r in rows} == {"111", "222"}
    assert {r[-1] for r in rows} == {"#010203", "#040506"}
    assert _fake_atlas == ["allen_mouse_25um"]  # the project's atlas, not a hard-coded one

    import json

    prov = json.loads((tmp_path / "LO_test_whole.provenance.json").read_text())
    assert prov["options"]["atlas"] == "allen_mouse_25um"
    assert prov["options"]["region_columns"] == ["region", "region_id", "region_color"]


def test_export_no_regions_skips_the_atlas(tmp_path, _fake_atlas) -> None:
    project_json = tmp_path / "LO_test_whole.json"
    save_project(_make_project(), project_json)

    result = runner.invoke(app, ["export", str(project_json), "--no-regions"])
    assert result.exit_code == 0, result.output

    header, _rows = _read(tmp_path / "LO_test_whole.csv")
    assert header[-1] == "depth_source"
    assert _fake_atlas == []


def test_export_survives_an_atlas_that_fails_to_load(tmp_path, monkeypatch) -> None:
    """Coordinates must still ship; the sidecar records that regions are absent."""
    project_json = tmp_path / "LO_test_whole.json"
    save_project(_make_project(), project_json)
    monkeypatch.setattr("atlastrack.cli._load_export_atlas", lambda name: None)

    result = runner.invoke(app, ["export", str(project_json)])
    assert result.exit_code == 0, result.output
    assert "without region columns" in result.output

    header, rows = _read(tmp_path / "LO_test_whole.csv")
    assert header[-1] == "depth_source"
    assert len(rows) > 0

    import json

    prov = json.loads((tmp_path / "LO_test_whole.provenance.json").read_text())
    assert prov["options"]["atlas"] is None
    assert prov["options"]["region_columns"] == []


def test_export_respects_out_dir_and_name(tmp_path) -> None:
    project_json = tmp_path / "proj.json"
    save_project(_make_project(), project_json)
    out_dir = tmp_path / "Registration"

    result = runner.invoke(
        app,
        ["export", str(project_json), "--out-dir", str(out_dir), "--name", "LO_06 - red"],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "LO_06 - red.csv").exists()
    assert (out_dir / "LO_06 - red - Paxinos.csv").exists()


def test_export_does_not_modify_the_project_by_default(tmp_path) -> None:
    project_json = tmp_path / "proj.json"
    save_project(_make_project(), project_json)
    before = project_json.read_bytes()

    result = runner.invoke(app, ["export", str(project_json), "--rigid-array"])
    assert result.exit_code == 0, result.output
    assert project_json.read_bytes() == before, "export must not rewrite the project"


def test_export_save_project_to_persists_regularized_coords(tmp_path) -> None:
    project_json = tmp_path / "proj.json"
    save_project(_make_project(), project_json)
    out_json = tmp_path / "regularized.json"

    result = runner.invoke(
        app,
        [
            "export", str(project_json),
            "--rigid-array", "--rigid-tolerance", "0",
            "--lock-spacing-um", "250",
            "--save-project-to", str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output

    from atlastrack.project.io import load_project

    saved = load_project(out_json)
    tips = np.array([s.tip_ccf_um for s in saved.probes[0].shanks], dtype=float)
    gaps = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    assert gaps == pytest.approx([250.0, 250.0, 250.0], abs=1.0)


def test_export_warns_when_nothing_to_export(tmp_path) -> None:
    project = Project(
        atlas=AtlasRef(),
        slides=[],
        probes=[
            ProbeSpec(
                label="ProbeA",
                type=ProbeType(name="Neuropixels 1.0", n_shanks=1),
                shanks=[Shank(index=0)],  # no CCF coords
            )
        ],
    )
    project_json = tmp_path / "proj.json"
    save_project(project, project_json)

    result = runner.invoke(app, ["export", str(project_json)])
    assert result.exit_code == 0, result.output
    assert "nothing exported" in result.output


# ---------------------------------------------------------------------------
# enforce_rigid_arrays (shared by the CLI and the viz/export panel)
# ---------------------------------------------------------------------------

def test_enforce_rigid_arrays_evens_out_shank_spacing() -> None:
    project = _make_project()
    tips_before = np.array(
        [s.tip_ccf_um for s in project.probes[0].shanks], dtype=float
    )
    gaps_before = np.linalg.norm(np.diff(tips_before, axis=0), axis=1)
    assert gaps_before.std() > 50.0  # the picks really are uneven

    infos = enforce_rigid_arrays(project, tolerance=0.0)

    tips_after = np.array(
        [s.tip_ccf_um for s in project.probes[0].shanks], dtype=float
    )
    gaps_after = np.linalg.norm(np.diff(tips_after, axis=0), axis=1)
    assert gaps_after.std() == pytest.approx(0.0, abs=1e-6)
    assert "ProbeA" in infos
    assert infos["ProbeA"]["spacing_um"] > 0


def test_enforce_rigid_arrays_lock_spacing_overrides_the_estimate() -> None:
    project = _make_project()
    enforce_rigid_arrays(project, tolerance=0.0, lock_spacing_um=250.0)
    tips = np.array([s.tip_ccf_um for s in project.probes[0].shanks], dtype=float)
    gaps = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    assert gaps == pytest.approx([250.0, 250.0, 250.0], abs=1e-6)


def test_enforce_rigid_arrays_skips_probes_with_too_few_shanks() -> None:
    project = _make_project(n_shanks=2)
    before = [s.tip_ccf_um for s in project.probes[0].shanks]
    infos = enforce_rigid_arrays(project, tolerance=0.0)
    assert infos == {}
    assert [s.tip_ccf_um for s in project.probes[0].shanks] == before
