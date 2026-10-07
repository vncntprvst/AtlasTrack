"""Channel images: other images of a slide, laid out and flipped like it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from atlastrack.project.images import (
    apply_slide_edits,
    check_channel_layout,
    load_channel_image,
    rebuild_slide_image,
)
from atlastrack.project.schema import ChannelImage, Section, Slide


def _write(path: Path, arr: np.ndarray) -> str:
    tifffile.imwrite(path, arr)
    return str(path)


def _slide(tmp_path: Path) -> Slide:
    rng = np.random.default_rng(0)
    sources = [_write(tmp_path / f"s{i}.tif", rng.integers(0, 255, (40, 60, 3), np.uint8))
               for i in range(2)]
    # The "red" channel image of each source: its red plane alone.
    reds = [_write(tmp_path / f"s{i}_red.tif", tifffile.imread(s)[..., 0]) for i, s in enumerate(sources)]
    sections = [
        Section(index=0, slide_idx=0, bbox_px=(5, 5, 30, 35), flip_h=True),
        Section(index=1, slide_idx=0, bbox_px=(10, 50, 50, 75), flip_v=True, rotation_deg=10.0),
    ]
    return Slide(image_path=sources[0], source_paths=sources, sections=sections, flip_v=True,
                 channel_images=[ChannelImage(name="red", source_paths=reds, colour="red")])


def test_channel_image_follows_the_slide_edits(tmp_path: Path) -> None:
    slide = _slide(tmp_path)
    shown, _ = rebuild_slide_image(slide)
    channel = apply_slide_edits(load_channel_image(slide, 0), slide)
    assert channel.shape == shown.shape[:2]
    np.testing.assert_array_equal(channel, shown[..., 0])


def test_rotations_can_be_those_the_slide_was_shown_with(tmp_path: Path) -> None:
    slide = _slide(tmp_path)
    raw = load_channel_image(slide, 0)
    unrotated = apply_slide_edits(raw.copy(), slide, rotations={})
    slide.sections[1].rotation_deg = 0.0
    np.testing.assert_array_equal(unrotated, apply_slide_edits(raw.copy(), slide))


def test_channel_files_must_match_the_slide_images(tmp_path: Path) -> None:
    slide = _slide(tmp_path)
    with pytest.raises(ValueError, match="one per slide image"):
        check_channel_layout(slide, slide.channel_images[0].source_paths[:1])
    small = _write(tmp_path / "small.tif", np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="size"):
        check_channel_layout(slide, [small, small])
    check_channel_layout(slide, slide.channel_images[0].source_paths)
