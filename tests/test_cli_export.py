"""Tests for the ``atlastrack export`` command and the shared rigid-array helper.

The export path deliberately works from the coordinates already stored in a
project, so a corrected project can be re-exported without re-registering.
"""
from __future__ import annotations

import csv

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
