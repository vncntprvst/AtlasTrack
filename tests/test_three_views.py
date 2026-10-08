"""The back / top / side PNG views (atlastrack.viz.views)."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.project.schema import EphysRecordingRef, Project, ProbeSpec, ProbeType, Shank

pytestmark = pytest.mark.qt


def _project(recordings=()):
    shanks = [Shank(index=i, tip_ccf_um=(11000.0, 6500.0 + 250 * i, 6000.0),
                    entry_ccf_um=(11000.0, 6500.0 + 250 * i, 1000.0)) for i in range(2)]
    return Project(probes=[ProbeSpec(label="P", type=ProbeType(
        name="Neuropixels 2.0 (4-shank)", n_shanks=2), shanks=shanks, recordings=list(recordings))])


def test_three_pngs_are_written_and_show_the_probe(qtbot, tmp_path) -> None:
    from PIL import Image

    from atlastrack.viz.views import render_three_views

    written = render_three_views(_project(), None, tmp_path, stem="S", size=(300, 200))

    assert [p.name for p in written] == ["S - back.png", "S - top.png", "S - side.png"]
    for p in written:
        img = np.asarray(Image.open(p))
        assert img.shape == (200, 300, 3)
        red = (img[..., 0] > 150) & (img[..., 1] < 90) & (img[..., 2] < 90)
        assert red.any(), f"no electrode drawn in {p.name}"


def test_only_the_recorded_electrodes_are_drawn() -> None:
    from atlastrack.viz.views import recorded_electrodes, scene_contents

    refs = [EphysRecordingRef(path="a", electrode_range=(1, 96)),
            EphysRecordingRef(path="b", electrode_range=(97, 192))]
    assert recorded_electrodes(_project(refs).probes[0]) == set(range(192))
    assert recorded_electrodes(_project().probes[0]) is None
    assert len(scene_contents(_project(refs), None)["electrodes"]) == 2 * 192
    assert len(scene_contents(_project(), None)["electrodes"]) == 2 * 384


def test_nothing_to_draw_says_so(tmp_path) -> None:
    from atlastrack.viz.views import render_three_views

    with pytest.raises(ValueError, match="nothing to draw"):
        render_three_views(Project(), None, tmp_path)


def test_labels_go_beside_their_region_and_never_on_an_electrode() -> None:
    from PIL import Image

    from atlastrack.viz.views import place_labels

    img = Image.new("RGB", (600, 400), "white")
    region = np.array([[250.0, 150.0], [350.0, 150.0], [350.0, 250.0], [250.0, 250.0]])
    # Electrodes fill the space to the region's left, so the name must go elsewhere.
    elec = np.stack(np.meshgrid(np.arange(60, 245, 4.0), np.arange(100, 300, 4.0)), -1).reshape(-1, 2)
    out = np.asarray(place_labels(img, [("Intermediate reticular nucleus", region)], elec,
                                  font_px=16))
    ink = np.argwhere(out.sum(-1) < 300)                     # dark text pixels (y, x)
    assert len(ink), "no label was written"
    assert ink[:, 1].min() > 245                              # not among the electrodes
