"""Exporting the section series: images, atlas-outline sidecars, and a manifest.

For a user who only registers histology, this is the whole product. Two things it
must get right: the series has to be *continuous* (in AP order, not scan order, with
the flips and rotation the registration itself saw), and the outline sidecar has to
be in its section's own frame. Both follow from taking the crops from
``rebuild_slide_image`` rather than re-deriving them here.
"""
from __future__ import annotations

import json
import math
from typing import ClassVar

import numpy as np
import pytest

from histo_to_ccf.io.series_export import export_section_series
from histo_to_ccf.project.images import (
    deepslice_rotation_deg,
    rebuild_slide_image,
    rotate_in_bbox,
)
from histo_to_ccf.project.schema import Project, Section, Slide


def _anchoring(degrees):
    """A stored anchoring whose u vector sits ``degrees`` off the ML axis."""
    rad = math.radians(degrees)
    return [0.0, 0.0, 0.0, 0.0, math.sin(rad) * 100, math.cos(rad) * 100, 0.0, 100.0, 0.0]


# ---------------------------------------------------------------------------
# Recovering the angle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degrees", [0.0, 3.5, -7.25, 15.0])
def test_the_angle_round_trips_through_the_anchoring(degrees):
    assert deepslice_rotation_deg(_anchoring(degrees)) == pytest.approx(degrees)


def test_a_square_section_needs_no_rotation():
    """u purely along ML is a section lying square on the slide."""
    assert deepslice_rotation_deg([0, 0, 0, 0, 0, 100, 0, 100, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Rotating
# ---------------------------------------------------------------------------


def test_rotation_keeps_the_bbox_shape_exactly():
    """bbox_px is the section frame everywhere else - overlay placement, landmarks,
    per-channel coordinates - so a rotation that resized it would desynchronise
    all of them."""
    image = np.zeros((40, 40), dtype=np.uint8)
    image[18:22, 5:35] = 255

    out = rotate_in_bbox(image, 30.0)

    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert out.sum() > 0


def test_a_zero_angle_returns_the_image_untouched():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)

    assert rotate_in_bbox(image, 0.0) is image


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


def _project(tmp_path, n=3, flip_h=False):
    """A slide of n bright squares on black, one section each."""
    import imageio.v3 as iio

    image = np.zeros((60, 60 * n), dtype=np.uint8)
    for i in range(n):
        image[20:40, 60 * i + 20 : 60 * i + 40] = 200 - 40 * i  # distinguishable
    path = tmp_path / "slide.png"
    iio.imwrite(path, image)

    sections = [
        Section(
            index=i,
            slide_idx=0,
            bbox_px=(60 * i, 0, 60 * i + 60, 60),
            ap_order=n - 1 - i,  # reversed, so AP order != index order
        )
        for i in range(n)
    ]
    project = Project()
    project.slides.append(
        Slide(image_path=str(path), sections=sections, flip_h=flip_h)
    )
    return project


def test_it_writes_one_image_per_section_named_in_series_order(tmp_path):
    project = _project(tmp_path)

    result = export_section_series(project, tmp_path / "out")

    assert result.sections == 3
    names = sorted(p.name for p in (tmp_path / "out").glob("*_section.png"))
    assert names == ["000_section.png", "001_section.png", "002_section.png"]


def test_the_files_follow_ap_order_not_detection_order(tmp_path):
    """A directory listing has to be the series; that is the point of the export."""
    project = _project(tmp_path)

    export_section_series(project, tmp_path / "out")

    manifest = json.loads((tmp_path / "out" / "series.json").read_text())
    assert [s["section_index"] for s in manifest["sections"]] == [2, 1, 0]


def test_the_manifest_records_each_section_rotation(tmp_path):
    project = _project(tmp_path)
    project.slides[0].sections[1].rotation_deg = -4.0

    export_section_series(project, tmp_path / "out")

    manifest = json.loads((tmp_path / "out" / "series.json").read_text())
    by_index = {s["section_index"]: s for s in manifest["sections"]}
    assert by_index[1]["rotation_baked_deg"] == pytest.approx(-4.0)
    assert by_index[0]["rotation_baked_deg"] == 0


def test_the_exported_crop_carries_the_rotation_the_registration_saw(tmp_path):
    """The export does not rotate anything: rebuild_slide_image already did, which
    is what keeps the image and its outline sidecar in one frame."""
    import imageio.v3 as iio

    plain = _project(tmp_path)
    turned = _project(tmp_path)
    turned.slides[0].sections[0].rotation_deg = 25.0

    export_section_series(plain, tmp_path / "a")
    export_section_series(turned, tmp_path / "b")

    # Section 0 is last in AP order (see _project), so it is position 002.
    before = iio.imread(tmp_path / "a" / "002_section.png")
    after = iio.imread(tmp_path / "b" / "002_section.png")
    assert before.shape == after.shape  # the bbox frame is preserved
    assert not np.array_equal(before, after)


def test_rebuild_bakes_rotation_into_the_working_image(tmp_path):
    """Registration reads this same function, so the fit and the export agree."""
    project = _project(tmp_path)
    flat, _ = rebuild_slide_image(project.slides[0])
    project.slides[0].sections[0].rotation_deg = 25.0
    turned, _ = rebuild_slide_image(project.slides[0])

    assert flat.shape == turned.shape
    x0, y0, x1, y1 = project.slides[0].sections[0].bbox_px
    assert not np.array_equal(flat[y0:y1, x0:x1], turned[y0:y1, x0:x1])
    # Only the rotated section changes; its neighbours are untouched.
    nx0, ny0, nx1, ny1 = project.slides[0].sections[1].bbox_px
    assert np.array_equal(flat[ny0:ny1, nx0:nx1], turned[ny0:ny1, nx0:nx1])


def test_the_flip_is_applied_exactly_as_registration_saw_it(tmp_path):
    """The crop must match the image the app worked on, or the outlines are wrong."""
    import imageio.v3 as iio

    plain = export_section_series(_project(tmp_path), tmp_path / "a")
    flipped = export_section_series(
        _project(tmp_path, flip_h=True), tmp_path / "b"
    )

    assert plain.sections == flipped.sections == 3
    first_plain = iio.imread(tmp_path / "a" / "000_section.png")
    first_flipped = iio.imread(tmp_path / "b" / "000_section.png")
    # flip_h reverses the slide, so the AP-first section is a different crop.
    assert not np.array_equal(first_plain, first_flipped)


def test_unregistered_sections_are_reported_not_silently_skipped(tmp_path):
    """A missing outline must be a stated gap, not an absent file nobody notices."""
    project = _project(tmp_path)

    result = export_section_series(project, tmp_path / "out", atlas=None)

    assert result.outlines == 0
    assert len(result.skipped_outlines) == 3
    assert all("atlas" in reason for _idx, reason in result.skipped_outlines)


def test_outlines_are_not_requested_when_switched_off(tmp_path):
    project = _project(tmp_path)

    result = export_section_series(project, tmp_path / "out", write_outlines=False)

    assert result.skipped_outlines == []
    assert list((tmp_path / "out").glob("*_outline.png")) == []


# ---------------------------------------------------------------------------
# Straightening at export (presentation only)
# ---------------------------------------------------------------------------


def test_with_nothing_set_by_hand_the_whole_deepslice_angle_is_removed():
    """The common case: the user never opens the rotation control at all."""
    from histo_to_ccf.io.series_export import straighten_angle_deg

    section = Section(index=0, slide_idx=0, bbox_px=(0, 0, 4, 4))
    section.deepslice_anchoring = _anchoring(9.0)

    assert straighten_angle_deg(section) == pytest.approx(9.0)


def test_a_baked_rotation_is_not_applied_a_second_time():
    """Pressing 'From DeepSlice' bakes the angle in; the export must then add none."""
    from histo_to_ccf.io.series_export import straighten_angle_deg

    section = Section(index=0, slide_idx=0, bbox_px=(0, 0, 4, 4))
    section.deepslice_anchoring = _anchoring(9.0)
    section.rotation_deg = 9.0

    assert straighten_angle_deg(section) == pytest.approx(0.0)


def test_only_the_remaining_tilt_is_straightened():
    from histo_to_ccf.io.series_export import straighten_angle_deg

    section = Section(index=0, slide_idx=0, bbox_px=(0, 0, 4, 4))
    section.deepslice_anchoring = _anchoring(9.0)
    section.rotation_deg = 4.0

    assert straighten_angle_deg(section) == pytest.approx(5.0)


def test_a_section_deepslice_never_saw_is_left_alone():
    from histo_to_ccf.io.series_export import straighten_angle_deg

    assert straighten_angle_deg(Section(index=0, slide_idx=0, bbox_px=(0, 0, 4, 4))) == 0.0


def test_the_export_straightens_by_default_and_records_both_rotations(tmp_path):
    project = _project(tmp_path)
    project.slides[0].sections[0].deepslice_anchoring = _anchoring(9.0)

    export_section_series(project, tmp_path / "out")

    manifest = json.loads((tmp_path / "out" / "series.json").read_text())
    entry = next(s for s in manifest["sections"] if s["section_index"] == 0)
    assert entry["rotation_straighten_deg"] == pytest.approx(9.0)
    assert entry["rotation_baked_deg"] == 0.0


def test_straightening_can_be_switched_off(tmp_path):
    project = _project(tmp_path)
    project.slides[0].sections[0].deepslice_anchoring = _anchoring(9.0)

    export_section_series(project, tmp_path / "out", straighten=False)

    manifest = json.loads((tmp_path / "out" / "series.json").read_text())
    entry = next(s for s in manifest["sections"] if s["section_index"] == 0)
    assert entry["rotation_straighten_deg"] == 0.0


def test_straightening_changes_the_pixels_it_writes(tmp_path):
    import imageio.v3 as iio

    tilted = _project(tmp_path)
    tilted.slides[0].sections[0].deepslice_anchoring = _anchoring(9.0)

    export_section_series(tilted, tmp_path / "a")
    export_section_series(tilted, tmp_path / "b", straighten=False)

    # Section 0 is last in AP order, so position 002.
    with_turn = iio.imread(tmp_path / "a" / "002_section.png")
    without = iio.imread(tmp_path / "b" / "002_section.png")
    assert with_turn.shape != without.shape  # the canvas grew to fit the rotation


# ---------------------------------------------------------------------------
# Outline sidecars, SVG and the region list
# ---------------------------------------------------------------------------


class _FakeAtlas:
    structures: ClassVar[dict] = {
        1: {"acronym": "VISp", "name": "Primary visual area"},
        2: {"acronym": "CA1", "name": "Field CA1"},
    }


def _with_registration(monkeypatch, tmp_path):
    """A one-section project whose warped labels are two known regions."""
    from histo_to_ccf.io import series_export
    from histo_to_ccf.project.schema import RegistrationResult

    def _labels(_section, _atlas, shape, _project_dir, _source_shape=None):
        lab = np.zeros(shape, dtype=np.int32)
        lab[8:26, 8:26] = 1
        lab[30:52, 12:48] = 2
        return lab

    monkeypatch.setattr(series_export, "_warped_labels", _labels)
    project = _project(tmp_path, n=1)
    project.slides[0].sections[0].registration = RegistrationResult(
        anchoring=[0.0] * 9, output_size_px=(60, 60)
    )
    return project


def test_the_outline_sidecar_is_black_on_white(tmp_path, monkeypatch):
    """It is a figure component, not a screen overlay: it must not need inverting."""
    import imageio.v3 as iio

    project = _with_registration(monkeypatch, tmp_path)

    export_section_series(project, tmp_path / "out", atlas=_FakeAtlas())

    sidecar = iio.imread(tmp_path / "out" / "000_outline.png")
    assert sorted(np.unique(sidecar).tolist()) == [0, 255]
    assert (sidecar == 255).mean() > 0.8  # white ground, thin black lines


def test_the_burnt_in_outline_is_white(tmp_path, monkeypatch):
    """Black would vanish into dark fluorescence, which is most of these images."""
    import imageio.v3 as iio

    project = _with_registration(monkeypatch, tmp_path)

    export_section_series(
        project, tmp_path / "out", atlas=_FakeAtlas(), write_overlays=True
    )

    overlay = iio.imread(tmp_path / "out" / "000_overlay.png")
    assert (overlay == 255).all(axis=-1).any()


def test_the_svg_carries_one_identified_path_per_region(tmp_path, monkeypatch):
    project = _with_registration(monkeypatch, tmp_path)

    result = export_section_series(
        project, tmp_path / "out", atlas=_FakeAtlas(), write_svg=True
    )

    assert result.svgs == 1
    svg = (tmp_path / "out" / "000_outline.svg").read_text(encoding="utf-8")
    assert 'id="VISp"' in svg and 'id="CA1"' in svg
    assert "Primary visual area" in svg  # the hover name, for labelling later
    assert 'stroke="#000"' in svg


def test_the_region_list_uses_the_names_the_canvas_shows_on_hover(tmp_path,
                                                                 monkeypatch):
    import csv

    project = _with_registration(monkeypatch, tmp_path)

    result = export_section_series(
        project, tmp_path / "out", atlas=_FakeAtlas(), write_regions=True
    )

    assert result.regions == 2
    rows = list(csv.DictReader((tmp_path / "out" / "regions.csv").open(encoding="utf-8")))
    assert [r["acronym"] for r in rows] == ["VISp", "CA1"]
    assert rows[0]["name"] == "Primary visual area"
    assert int(rows[1]["area_px"]) > 0


def test_svg_and_regions_are_off_unless_asked_for(tmp_path, monkeypatch):
    project = _with_registration(monkeypatch, tmp_path)

    result = export_section_series(project, tmp_path / "out", atlas=_FakeAtlas())

    assert result.svgs == 0 and result.regions == 0
    assert list((tmp_path / "out").glob("*.svg")) == []
    assert not (tmp_path / "out" / "regions.csv").exists()


def test_an_unknown_region_id_still_gets_a_row(tmp_path, monkeypatch):
    """An atlas that does not know an id is not a reason to drop the region."""
    from histo_to_ccf.io.series_export import _region_names

    assert _region_names(_FakeAtlas(), 999) == ("999", "")


def test_the_region_list_names_the_atlas_that_named_the_regions(tmp_path,
                                                                monkeypatch):
    """"M1" and "MOp" are the same region. A file that does not say which atlas
    named it cannot be read safely, so the nomenclature travels with the rows."""
    import csv

    project = _with_registration(monkeypatch, tmp_path)

    class _Named(_FakeAtlas):
        atlas_name = "kim_mouse_25um"

    export_section_series(
        project, tmp_path / "out", atlas=_Named(), write_regions=True
    )

    rows = list(csv.DictReader((tmp_path / "out" / "regions.csv").open(encoding="utf-8")))
    assert {r["atlas"] for r in rows} == {"kim_mouse_25um"}


def test_the_manifest_separates_the_registration_and_region_atlases(tmp_path,
                                                                   monkeypatch):
    """They are routinely different: registered on Allen, labelled with Chon/Kim."""
    project = _with_registration(monkeypatch, tmp_path)
    project.atlas.name = "allen_mouse_25um"

    class _Named(_FakeAtlas):
        atlas_name = "kim_mouse_25um"

    export_section_series(project, tmp_path / "out", atlas=_Named())

    manifest = json.loads((tmp_path / "out" / "series.json").read_text())
    assert manifest["registration_atlas"] == "allen_mouse_25um"
    assert manifest["region_atlas"] == "kim_mouse_25um"
