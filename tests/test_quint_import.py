"""Import of a QUINT series (QuickNII / DeepSlice / VisuAlign JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from atlastrack.io.importers import detect, import_registration
from atlastrack.io.quint import atlas_for_target, deform_forward, deform_inverse
from atlastrack.project.io import load_project
from atlastrack.registration.transforms import build_registered_transform

W, H = 300, 200
# A coronal plane over the whole image, QuickNII axes (ML, AP, DV): image x runs
# along QuickNII +x (the animal's left -> right), image y from top to bottom.
ANCHORING = [0.0, 300.0, 320.0, 456.0, 0.0, 0.0, 0.0, 0.0, -320.0]
MARKERS = [[100, 80, 110, 85], [200, 120, 190, 128], [150, 60, 150, 50]]


class _Atlas:
    resolution = (25, 25, 25)

    class annotation:
        shape = (528, 320, 456)


def _write_series(root: Path, markers=MARKERS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    slices = []
    for nr in (1, 2):
        name = f"brain_s{nr:03d}.png"
        Image.fromarray(np.full((H, W, 3), 60 * nr, np.uint8)).save(root / name)
        slices.append({"filename": name, "nr": nr, "width": W, "height": H,
                       "anchoring": ANCHORING, "markers": markers if nr == 2 else []})
    path = root / "series.json"
    path.write_text(json.dumps({"target": "ABA_Mouse_CCFv3_2017_25um.cutlas",
                                "target-resolution": [456, 528, 320], "slices": slices}))
    return path


def _transform(project_path: Path, index: int):
    project = load_project(project_path)
    section = project.slides[0].sections[index]
    return section, build_registered_transform(
        section.registration, _Atlas(), project_dir=project_path.parent
    )


@pytest.fixture(autouse=True)
def _atlas_resolution(monkeypatch):
    import atlastrack.io.ccf_coords as cc

    monkeypatch.setattr(cc, "atlas_resolution_um", lambda atlas: (25.0, 25.0, 25.0))


def test_markers_move_their_points_and_the_inverse_undoes_it() -> None:
    m = np.asarray(MARKERS, dtype=float)
    mx, my = deform_forward(W, H, m, m[:, 0], m[:, 1])
    np.testing.assert_allclose(np.c_[mx, my], m[:, 2:], atol=1e-9)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, W, 200), rng.uniform(0, H, 200)
    fx, fy = deform_forward(W, H, m, x, y)
    bx, by = deform_inverse(W, H, m, fx, fy)
    np.testing.assert_allclose(np.c_[bx, by], np.c_[x, y], atol=1e-9)


def test_quint_import_maps_points_like_visualign(tmp_path: Path) -> None:
    series = _write_series(tmp_path / "quint")
    assert detect(series) == "quint" and detect(series.parent) == "quint"
    project_path = import_registration(series)
    assert project_path.name == "series.atlastrack.json"

    # Section without markers: the plane alone. QuickNII x runs from the animal's
    # left to its right, BrainGlobe ML from its right to its left: ML = 456 - x.
    _, plain = _transform(project_path, 0)
    left, right = plain.apply_many(np.array([[0.0, 100.0], [W, 100.0]]))
    assert left[1] == pytest.approx(456 * 25.0)  # QuickNII x 0 -> BrainGlobe ML 11400
    assert right[1] == pytest.approx(0.0)
    assert left[0] == pytest.approx((528 - 300) * 25.0)

    # Section with markers: a pixel maps to the plane point VisuAlign moved there.
    section, warped = _transform(project_path, 1)
    assert section.registration.bspline_transform_path == "series_transforms/section_001.h5"
    m = np.asarray(MARKERS, dtype=float)
    got = warped.apply_many(m[:, 2:])
    expected = plain.apply_many(m[:, :2])
    np.testing.assert_allclose(got, expected, atol=1e-6)

    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(5, W - 5, 50), rng.uniform(5, H - 5, 50)]
    ox, oy = deform_inverse(W, H, m, pts[:, 0], pts[:, 1])
    np.testing.assert_allclose(
        warped.apply_many(pts), plain.apply_many(np.c_[ox, oy]), atol=1.0  # µm
    )


def test_rat_atlases_are_refused() -> None:
    with pytest.raises(ValueError, match="not supported"):
        atlas_for_target("WHS_Rat_v4_39um.cutlas")
    assert atlas_for_target("ABA_Mouse_CCFv3_2017_25um.cutlas") == "allen_mouse_25um"
