"""Per-section folders written on save, and a slide rebuilt from them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from atlastrack.project.images import rebuild_slide_image, section_images
from atlastrack.project.io import load_project, save_project
from atlastrack.project.schema import (
    ManualLandmarks,
    Project,
    RegistrationResult,
    Section,
    Slide,
)


def _slide_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (60, 100, 3), dtype=np.uint8)


def _project(tmp_path: Path) -> tuple[Project, Path]:
    slide_path = tmp_path / "slide.tif"
    tifffile.imwrite(slide_path, _slide_image(), photometric="rgb")
    anchoring = [400.0, 0.0, 0.0, 0.0, 0.0, 456.0, 0.0, 320.0, 0.0]
    sections = [
        Section(index=0, slide_idx=0, bbox_px=(5, 5, 45, 55),
                registration=RegistrationResult(anchoring=anchoring, output_size_px=(50, 40)),
                manual_landmarks=ManualLandmarks(
                    source=[[1, 1], [30, 2], [2, 40], [30, 40]],
                    target=[[1, 1], [31, 2], [2, 40], [30, 41]])),
        Section(index=1, slide_idx=0, bbox_px=(50, 10, 95, 50), flip_h=True),
    ]
    project = Project(slides=[Slide(image_path=str(slide_path), sections=sections)])
    return project, tmp_path / "proj" / "slide.atlastrack.json"


def test_save_cuts_section_images_and_writes_readable_summaries(tmp_path: Path) -> None:
    project, path = _project(tmp_path)
    save_project(project, path)
    loaded = load_project(path)
    slide_img, _ = rebuild_slide_image(loaded.slides[0], base_dir=path.parent)
    for section in loaded.slides[0].sections:
        assert section.image_path == f"slide_sections/section_{section.index:03d}/image.tif"
        x0, y0, x1, y1 = section.bbox_px
        np.testing.assert_array_equal(
            tifffile.imread(path.parent / section.image_path), slide_img[y0:y1, x0:x1]
        )
    summary = json.loads((path.parent / "slide_sections/section_000/section.json").read_text())
    assert summary["plane"]["ap_um"] == 10000.0  # coronal plane at voxel 400
    assert summary["plane"]["pitch_deg"] == 0 and summary["plane"]["mirrored"] is False
    assert len(summary["landmarks"]) == 4
    assert set(summary["landmarks"][0]) == {"image_px", "atlas_um"}


def test_a_section_image_is_cut_again_only_when_its_box_changes(tmp_path: Path) -> None:
    project, path = _project(tmp_path)
    save_project(project, path)
    image = path.parent / "slide_sections/section_001/image.tif"
    first = image.stat().st_mtime_ns
    save_project(project, path)
    assert image.stat().st_mtime_ns == first
    project.slides[0].sections[1].bbox_px = (50, 10, 90, 50)
    save_project(project, path)
    assert tifffile.imread(image).shape[:2] == (40, 40)


def test_slide_is_rebuilt_from_sections_when_its_image_is_gone(tmp_path: Path) -> None:
    project, path = _project(tmp_path)
    save_project(project, path)
    expected, _ = rebuild_slide_image(project.slides[0], base_dir=path.parent)
    Path(project.slides[0].image_path).unlink()

    loaded = load_project(path)
    rebuilt, bands = rebuild_slide_image(loaded.slides[0], base_dir=path.parent)
    assert bands == [(0, rebuilt.shape[0])]
    for section in loaded.slides[0].sections:
        x0, y0, x1, y1 = section.bbox_px
        np.testing.assert_array_equal(rebuilt[y0:y1, x0:x1], expected[y0:y1, x0:x1])


def test_section_images_use_the_align_channel(tmp_path: Path) -> None:
    project, path = _project(tmp_path)
    project.slides[0].align_channel = "green"
    save_project(project, path)
    crops = section_images(load_project(path), base_dir=path.parent)
    x0, y0, x1, y1 = project.slides[0].sections[0].bbox_px
    np.testing.assert_array_equal(crops[0], _slide_image()[y0:y1, x0:x1, 1].astype(np.float32))


def test_two_projects_in_one_folder_keep_their_own_files(tmp_path: Path) -> None:
    project, path = _project(tmp_path)
    save_project(project, path)
    other = path.with_name("other.atlastrack.json")
    project.slides[0].sections[0].bbox_px = (5, 5, 30, 30)
    save_project(project, other)  # "Save As" under another name, same folder
    first = load_project(path).slides[0].sections[0]
    second = load_project(other).slides[0].sections[0]
    assert first.image_path.startswith("slide_sections/")
    assert second.image_path.startswith("other_sections/")
    assert tifffile.imread(path.parent / first.image_path).shape[:2] == (50, 40)
    assert tifffile.imread(path.parent / second.image_path).shape[:2] == (25, 25)
