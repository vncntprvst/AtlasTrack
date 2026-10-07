"""Registering sections by matching them to a registered project of the same sections."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

from atlastrack.project.io import load_project, save_project
from atlastrack.project.schema import (
    CellPoint,
    CellSet,
    ManualLandmarks,
    Project,
    RegistrationResult,
    Section,
    Slide,
)
from atlastrack.registration.reference_match import apply_matches, match_projects
from atlastrack.registration.transforms import build_registered_transform


class _Atlas:
    pass


@pytest.fixture(autouse=True)
def _resolution(monkeypatch):
    import atlastrack.io.ccf_coords as cc

    monkeypatch.setattr(cc, "atlas_resolution_um", lambda atlas: (25.0, 25.0, 25.0))


def _tissue(h: int, w: int, seed: int) -> np.ndarray:
    """An asymmetric blob with texture inside, on black."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - w * 0.5) / (w * 0.4)) ** 2 + ((yy - h * 0.5) / (h * 0.35)) ** 2 < 1
    blob &= ~(((xx - w * 0.72) / (w * 0.1)) ** 2 + ((yy - h * 0.3) / (h * 0.12)) ** 2 < 1)
    blob |= ((xx - w * 0.25) / (w * 0.08)) ** 2 + ((yy - h * 0.8) / (h * 0.1)) ** 2 < 1
    texture = ndi.gaussian_filter(rng.random((h, w)), 6)
    texture = (texture - texture.min()) / np.ptp(texture)
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 2] = (blob * (80 + 170 * texture)).astype(np.uint8)
    return img


def _reference(tmp_path: Path) -> Path:
    image = _tissue(300, 400, 0)
    ref_dir = tmp_path / "confocal"
    ref_dir.mkdir()
    tifffile.imwrite(ref_dir / "slide.tif", image, photometric="rgb")
    src = [[50, 50], [350, 60], [60, 250], [340, 240], [200, 150]]
    dst = [[52, 48], [347, 63], [58, 252], [343, 238], [204, 153]]
    section = Section(
        index=0, slide_idx=0, bbox_px=(0, 0, 400, 300), slide_number=7,
        registration=RegistrationResult(
            anchoring=[400.0, 20.0, 30.0, 1.0, 0.0, 400.0, -2.0, 280.0, 0.0],
            output_size_px=(300, 400),
        ),
        manual_landmarks=ManualLandmarks(source=src, target=dst),
    )
    cells = CellSet(name="cells", cells=[
        CellPoint(ap_um=0, ml_um=0, dv_um=0, section_index=0, x_px=200.0, y_px=150.0),
    ])
    project = Project(slides=[Slide(image_path="slide.tif", sections=[section],
                                    align_channel="blue")], cell_sets=[cells])
    path = ref_dir / "confocal.atlastrack.json"
    save_project(project, path)
    return path


def _target(tmp_path: Path) -> Path:
    """The same tissue, smaller, mirrored and turned, beside another section."""
    from skimage.transform import rescale, rotate

    tissue = _tissue(300, 400, 0)
    small = rescale(tissue, 0.75, channel_axis=-1, preserve_range=True, anti_aliasing=True)
    small = rotate(small[:, ::-1], 3.0, preserve_range=True).astype(np.uint8)
    other = _tissue(220, 220, 5)
    slide = np.zeros((300, 600, 3), np.uint8)
    slide[40:40 + small.shape[0], 20:20 + small.shape[1]] = small
    slide[40:260, 360:580] = other
    tgt_dir = tmp_path / "slides"
    tgt_dir.mkdir()
    tifffile.imwrite(tgt_dir / "slide.tif", slide, photometric="rgb")
    sections = [
        Section(index=0, slide_idx=0, bbox_px=(10, 30, 330, 275)),
        Section(index=1, slide_idx=0, bbox_px=(350, 30, 590, 270)),
    ]
    path = tgt_dir / "slides.atlastrack.json"
    save_project(Project(slides=[Slide(image_path=str(tgt_dir / "slide.tif"), sections=sections,
                                       align_channel="blue")]), path)
    return path


def test_match_registers_the_counterpart_and_brings_its_cells(tmp_path: Path) -> None:
    ref_path, tgt_path = _reference(tmp_path), _target(tmp_path)
    target, reference = load_project(tgt_path), load_project(ref_path)

    matches = match_projects(target, tgt_path.parent, reference, ref_path.parent)
    assert [(m.target_index, m.reference_index, m.mirrored) for m in matches] == [(0, 0, True)]
    fit = matches[0].fit
    assert fit.overlap > 0.95
    assert np.sqrt(abs(np.linalg.det(fit.matrix[:2, :2]))) == pytest.approx(1 / 0.75, rel=0.03)

    done = apply_matches(target, tgt_path, reference, ref_path, matches)
    save_project(target, tgt_path)
    target = load_project(tgt_path)
    section = target.slides[0].sections[0]
    assert done == [0] and section.flip_h and section.ap_source == "reference"
    assert target.slides[0].sections[1].registration is None

    # A point and its counterpart land on the same atlas point.
    mine = build_registered_transform(section.registration, _Atlas(), project_dir=tgt_path.parent)
    ref_section = reference.slides[0].sections[0]
    theirs = build_registered_transform(
        ref_section.registration, _Atlas(), project_dir=ref_path.parent,
        manual_landmarks=ref_section.manual_landmarks,
    )
    m = np.asarray(section.reference.matrix)
    pts = np.array([[100.0, 80.0], [160.0, 120.0], [220.0, 150.0]])
    np.testing.assert_allclose(
        mine.apply_many(pts), theirs.apply_many(pts @ m[:2, :2].T + m[:2, 2]), atol=0.5
    )

    # The cell comes across, placed on the matched section.
    cell = target.cell_sets[0].cells[0]
    assert cell.section_index == 0
    back = np.array([cell.x_px, cell.y_px]) @ m[:2, :2].T + m[:2, 2]
    np.testing.assert_allclose(back, [200.0, 150.0], atol=1e-6)


def test_a_back_view_project_keeps_its_face_when_matched(tmp_path: Path) -> None:
    """The counterpart is seen from the front; these sections stay seen from the back."""
    ref_path, tgt_path = _reference(tmp_path), _target(tmp_path)
    target, reference = load_project(tgt_path), load_project(ref_path)
    target.seen_from = "back"

    matches = match_projects(target, tgt_path.parent, reference, ref_path.parent)
    assert [m.mirrored for m in matches] == [False]  # the target image is mirrored already
    apply_matches(target, tgt_path, reference, ref_path, matches)
    save_project(target, tgt_path)
    target = load_project(tgt_path)
    section = target.slides[0].sections[0]
    assert not section.flip_h
    assert section.registration.anchoring[5] < 0  # a back-view plane

    mine = build_registered_transform(section.registration, _Atlas(), project_dir=tgt_path.parent)
    ref_section = reference.slides[0].sections[0]
    theirs = build_registered_transform(
        ref_section.registration, _Atlas(), project_dir=ref_path.parent,
        manual_landmarks=ref_section.manual_landmarks,
    )
    m = np.asarray(section.reference.matrix)
    pts = np.array([[100.0, 80.0], [160.0, 120.0], [220.0, 150.0]])
    np.testing.assert_allclose(
        mine.apply_many(pts), theirs.apply_many(pts @ m[:2, :2].T + m[:2, 2]), atol=0.5
    )
