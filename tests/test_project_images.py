"""Tests for project/images.py - reproducing the pixels a section bbox refers to.

These cover the three ways the headless CLI used to disagree with the GUI: it
loaded only the first source of a merged slide, and it applied neither the
whole-slide nor the per-section flips.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from histo_to_ccf.io.image import merge_images
from histo_to_ccf.project.images import rebuild_slide_image, section_images
from histo_to_ccf.project.schema import AtlasRef, Project, Section, Slide


def _write(path, arr) -> str:
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def _gradient(h: int, w: int, base: int) -> np.ndarray:
    """A distinctive, non-symmetric pattern so flips are detectable."""
    row = np.arange(w, dtype=np.uint8)
    col = np.arange(h, dtype=np.uint8)[:, None]
    return (base + row + 3 * col).astype(np.uint8)


def test_single_source_slide_loads_directly(tmp_path) -> None:
    arr = _gradient(20, 30, 0)
    slide = Slide(image_path=_write(tmp_path / "a.png", arr), sections=[])
    img, bands = rebuild_slide_image(slide)
    assert np.array_equal(img, arr)
    assert bands == [(0, 20)]


def test_merged_slide_uses_every_source(tmp_path) -> None:
    a, b = _gradient(20, 30, 0), _gradient(14, 24, 100)
    paths = [_write(tmp_path / "a.png", a), _write(tmp_path / "b.png", b)]
    slide = Slide(image_path=paths[0], source_paths=paths, sections=[])

    img, bands = rebuild_slide_image(slide)

    expected = merge_images([a, b])
    assert img.shape == expected.shape
    assert np.array_equal(img, expected)
    # The second source must actually be present - the old CLI stopped at the first.
    assert img.shape[0] > a.shape[0]
    assert len(bands) == 2


def test_slide_flips_are_reapplied(tmp_path) -> None:
    arr = _gradient(20, 30, 0)
    path = _write(tmp_path / "a.png", arr)

    img_h, _ = rebuild_slide_image(Slide(image_path=path, flip_h=True, sections=[]))
    assert np.array_equal(img_h, np.fliplr(arr))

    img_v, _ = rebuild_slide_image(Slide(image_path=path, flip_v=True, sections=[]))
    assert np.array_equal(img_v, np.flipud(arr))

    img_hv, _ = rebuild_slide_image(
        Slide(image_path=path, flip_h=True, flip_v=True, sections=[])
    )
    assert np.array_equal(img_hv, np.flipud(np.fliplr(arr)))


def test_section_flip_is_applied_inside_its_bbox_only(tmp_path) -> None:
    arr = _gradient(20, 30, 0)
    path = _write(tmp_path / "a.png", arr)
    bbox = (4, 2, 16, 12)
    slide = Slide(
        image_path=path,
        sections=[Section(index=0, slide_idx=0, bbox_px=bbox, flip_h=True)],
    )

    img, _ = rebuild_slide_image(slide)

    x0, y0, x1, y1 = bbox
    assert np.array_equal(img[y0:y1, x0:x1], np.fliplr(arr[y0:y1, x0:x1]))
    # Everything outside the bbox is untouched.
    outside = np.ones(arr.shape, dtype=bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(img[outside], arr[outside])


def test_section_images_crops_and_grayscales(tmp_path) -> None:
    arr = np.stack([_gradient(20, 30, b) for b in (0, 50, 100)], axis=-1)
    path = _write(tmp_path / "rgb.png", arr)
    project = Project(
        atlas=AtlasRef(),
        slides=[
            Slide(
                image_path=path,
                sections=[
                    Section(index=0, slide_idx=0, bbox_px=(0, 0, 10, 8)),
                    Section(index=1, slide_idx=0, bbox_px=(10, 8, 30, 20)),
                ],
            )
        ],
    )

    images = section_images(project)

    assert set(images) == {0, 1}
    assert images[0].shape == (8, 10)   # (y1-y0, x1-x0), channels averaged away
    assert images[1].shape == (12, 20)
    assert images[0].dtype == np.float32


def test_section_images_skips_unreadable_slides(tmp_path) -> None:
    good = _write(tmp_path / "good.png", _gradient(20, 30, 0))
    project = Project(
        atlas=AtlasRef(),
        slides=[
            Slide(
                image_path=str(tmp_path / "missing.png"),
                sections=[Section(index=0, slide_idx=0, bbox_px=(0, 0, 5, 5))],
            ),
            Slide(
                image_path=good,
                sections=[Section(index=1, slide_idx=1, bbox_px=(0, 0, 5, 5))],
            ),
        ],
    )

    images = section_images(project)

    assert set(images) == {1}, "a missing source must not abort the whole project"


def test_relative_source_paths_resolve_against_base_dir(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    arr = _gradient(20, 30, 0)
    _write(data / "a.png", arr)
    slide = Slide(image_path="a.png", sections=[])

    img, _ = rebuild_slide_image(slide, base_dir=data)
    assert np.array_equal(img, arr)

    with pytest.raises(Exception):
        rebuild_slide_image(slide, base_dir=tmp_path / "nowhere")
