"""Import of a slicereg project (github.com/mvdokh/cell-counting)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.interpolate import RBFInterpolator

from atlastrack.atlas.planes import Anchoring
from atlastrack.io.importers import detect, import_registration
from atlastrack.project.io import load_project
from atlastrack.registration.landmarks_warp import warp_points
from atlastrack.registration.transforms import RegisteredSectionTransform


class _SliceregModel:
    """slicereg's own mapping, written out from its transform.py and atlas.py."""

    def __init__(self, al: dict):
        self.al = al
        th = math.radians(al["rotation_deg"])
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        flip = np.diag([-1.0 if al["flip"] else 1.0, 1.0])
        self.m = rot @ np.diag([al["scale_x"], al["scale_y"]]) @ flip
        self.c = np.asarray(al["plane_size_um"], float) / 2
        self.t = np.array([al["tx"], al["ty"]])
        lm = np.asarray(al["landmarks"], float).reshape(-1, 4)
        if not len(lm):  # the global fit alone
            self.rbf = np.zeros_like
            return
        w, h = al["image_size"]
        xs, ys = [-0.1 * w, w / 2, 1.1 * w], [-0.1 * h, h / 2, 1.1 * h]
        anchors = np.array([[x, y] for x in xs for y in ys if not (x == w / 2 and y == h / 2)])
        src = self.affine(lm[:, :2])
        self.rbf = RBFInterpolator(
            np.vstack([src, anchors]),
            np.vstack([lm[:, 2:] - src, np.zeros_like(anchors)]),
            kernel="thin_plate_spline",
        )

    def affine(self, p):
        return (np.atleast_2d(p) - self.c) @ self.m.T + self.t

    def plane_to_image(self, p):
        y = self.affine(p)
        return y + self.rbf(y)

    def image_to_atlas(self, x):
        y = x - self.rbf(x)
        for _ in range(200):
            y = x - self.rbf(y)
        st = (y - self.t) @ np.linalg.inv(self.m).T + self.c
        p, yw = math.radians(self.al["pitch_deg"]), math.radians(self.al["yaw_deg"])
        r_p = np.array([[math.cos(p), math.sin(p), 0], [-math.sin(p), math.cos(p), 0], [0, 0, 1]])
        r_y = np.array([[math.cos(yw), 0, math.sin(yw)], [0, 1, 0], [-math.sin(yw), 0, math.cos(yw)]])
        r = r_y @ r_p
        u, v = r @ [0, 0, 1.0], r @ [0, 1.0, 0]
        w, h = self.al["plane_size_um"]
        o = np.array([self.al["ap_um"], h / 2, w / 2]) - w / 2 * u - h / 2 * v
        return o + st[:, :1] * u + st[:, 1:2] * v  # (AP, DV, ML)


def _write_project(root: Path) -> Path:
    proj = root / "Mouse1_project"
    sizes = {1: (80, 50, 2.0), 2: (120, 80, 1.0)}  # id: (w, h, pixel_um)
    meta = {
        "atlas": "allen_mouse_25um",
        "section_spacing_um": 80.0,
        "channels": [{"name": "rfp", "color": "red"}, {"name": "nissl", "color": "blue"}],
        "align_channel": "nissl",
        "slices": [],
    }
    rng = np.random.default_rng(3)
    for sid, (w, h, pix) in sizes.items():
        folder = proj / "slices" / f"slice_{sid:02d}"
        folder.mkdir(parents=True)
        image = np.zeros((h, w, 2), np.uint8)
        image[..., 1] = 40
        image[h // 2, w // 2, 0] = 200
        tifffile.imwrite(folder / "image.tif", image)
        scale = w / 11400 * 1.4
        alignment = {
            "atlas": "allen_mouse_25um", "plane_size_um": [11400.0, 8000.0],
            "image_size": [w, h], "ap_um": 9000.0 + 80 * sid,
            "pitch_deg": -4.5, "yaw_deg": 1.5, "rotation_deg": 3.0,
            "scale_x": scale, "scale_y": scale * 0.95, "tx": w / 2, "ty": h / 2,
            "flip": sid == 2, "landmarks": [],
        }
        # Landmarks: the global fit's atlas point for each pixel, nudged ~1 px.
        model = _SliceregModel(alignment)
        px = np.c_[rng.uniform(0, w, 10), rng.uniform(0, h, 10)]
        st = (px - model.t) @ np.linalg.inv(model.m).T + model.c
        st += rng.normal(0, 1.0 / scale, st.shape)
        alignment["landmarks"] = np.c_[st, px].tolist()
        (folder / "alignment.json").write_text(json.dumps(alignment))
        # Two counted cells, with the atlas position slicereg gives them.
        cell_px = np.array([[w * 0.4, h * 0.5], [w * 0.6, h * 0.4]])
        ap, dv, ml = _SliceregModel(alignment).image_to_atlas(cell_px).T
        rows = ["slice_id,cell_type,x_px,y_px,ap_um,dv_um,ml_um,region_acronym"]
        rows += [f"{sid},cell,{x},{y},{a},{d},{m},P5"
                 for (x, y), a, d, m in zip(cell_px, ap, dv, ml, strict=True)]
        (folder / "cells.csv").write_text("\n".join(rows) + "\n")
        meta["slices"].append({"id": sid, "pixel_um": pix, "col": sid, "row": 1})
    (proj / "project.json").write_text(json.dumps(meta))
    return proj


def test_slicereg_import_reproduces_its_mapping(tmp_path: Path) -> None:
    proj = _write_project(tmp_path)
    assert detect(proj) == "slicereg"
    project_path = import_registration(proj)
    assert project_path == tmp_path / "Mouse1_project_atlastrack" / "Mouse1.atlastrack.json"

    project = load_project(project_path)
    slide = project.slides[0]
    assert slide.image_path == ""
    assert slide.channel_names == {"red": "rfp", "blue": "nissl"}
    assert slide.align_channel == "blue"
    assert [s.slide_number for s in slide.sections] == [1, 2]

    rng = np.random.default_rng(0)
    for section in slide.sections:
        al = json.loads((proj / "slices" / f"slice_{section.slide_number:02d}"
                         / "alignment.json").read_text())
        model = _SliceregModel(al)
        factor = {1: 1.0, 2: 0.5}[section.slide_number]
        image = tifffile.imread(project_path.parent / section.image_path)
        x0, y0, x1, y1 = section.bbox_px
        assert image.shape[:2] == (y1 - y0, x1 - x0)

        src = np.asarray(section.manual_landmarks.source)
        dst = np.asarray(section.manual_landmarks.target)
        reg = section.registration
        transform = RegisteredSectionTransform(
            anchoring=Anchoring.from_iterable(reg.anchoring),
            output_size_px=tuple(reg.output_size_px), bspline=None,
            atlas_resolution_um=(25.0, 25.0, 25.0), manual_landmarks=(src, dst), landmarks_forward=True,
        )
        w, h = al["image_size"]
        px = np.c_[rng.uniform(0, w, 30), rng.uniform(0, h, 30)]
        expected = model.image_to_atlas(px)
        got = transform.apply_many((px + 0.5) * factor - 0.5)  # (AP, ML, DV)
        # The inverse stops at 1e-4 px; a pixel here is ~100 µm.
        np.testing.assert_allclose(got[:, [0, 2, 1]], expected, atol=0.02)

        # The overlay: a plane point drawn where slicereg draws it.
        st = np.c_[rng.uniform(3000, 8000, 30), rng.uniform(2000, 6000, 30)]
        overlay = model.affine(st)
        drawn = warp_points(src, dst, (overlay + 0.5) * factor - 0.5, forward=True)
        np.testing.assert_allclose((drawn + 0.5) / factor - 0.5, model.plane_to_image(st),
                                   atol=1e-6)


def test_slicereg_cells_come_in_on_their_sections(tmp_path: Path) -> None:
    project_path = import_registration(_write_project(tmp_path))
    project = load_project(project_path)
    cells = project.cell_sets[0].cells
    assert len(cells) == 4 and all(c.section_index is not None for c in cells)
    sections = {s.index: s for s in project.slides[0].sections}
    for c in cells:
        s = sections[c.section_index]
        reg = s.registration
        t = RegisteredSectionTransform(
            anchoring=Anchoring.from_iterable(reg.anchoring),
            output_size_px=tuple(reg.output_size_px), bspline=None,
            atlas_resolution_um=(25.0, 25.0, 25.0),
            manual_landmarks=(np.asarray(s.manual_landmarks.source),
                              np.asarray(s.manual_landmarks.target)),
            landmarks_forward=s.manual_landmarks.forward,
        )
        np.testing.assert_allclose(t.apply(c.x_px, c.y_px), (c.ap_um, c.ml_um, c.dv_um), atol=0.05)


def test_import_rejects_an_unknown_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither"):
        import_registration(tmp_path)
