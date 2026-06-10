"""Section splitting on synthetic + real composite slides."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skimage.draw import ellipse

from histo_to_ccf.sectioning.ordering import geometric_order, order_sections
from histo_to_ccf.sectioning.split import detect_sections


def _synth_composite(
    rows: int = 2,
    cols: int = 3,
    section_shape: tuple[int, int] = (80, 120),
    pad: int = 20,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Build a fake composite: ``rows × cols`` bright ellipses on a dark field."""
    rng = np.random.default_rng(seed)
    sh, sw = section_shape
    h = rows * (sh + pad) + pad
    w = cols * (sw + pad) + pad
    image = np.zeros((h, w), dtype=float)
    for r in range(rows):
        cy = pad + r * (sh + pad) + sh // 2
        for c in range(cols):
            cx = pad + c * (sw + pad) + sw // 2
            rr, cc = ellipse(cy, cx, sh // 2 - 4, sw // 2 - 4, shape=image.shape)
            image[rr, cc] = 0.9
    if noise > 0:
        image += rng.normal(0, noise, image.shape)
    return np.clip(image, 0, 1)


def test_detect_synthetic_grid() -> None:
    image = _synth_composite(rows=2, cols=3, noise=0.02)
    sections = detect_sections(image, min_area_px=500)
    assert len(sections) == 6, f"expected 6 sections, got {len(sections)}"


def test_ordering_groups_into_rows() -> None:
    image = _synth_composite(rows=3, cols=4)
    sections = detect_sections(image, min_area_px=500)
    ordered = order_sections(sections)
    assert len(ordered) == 12
    # Each row should have exactly 4 columns, indexed 0..3, in left-to-right order.
    for row in range(3):
        row_items = [o for o in ordered if o.row == row]
        assert len(row_items) == 4
        xs = [o.section.centroid_px[0] for o in row_items]
        assert xs == sorted(xs), "row not left-to-right"


def test_geometric_order_column_first() -> None:
    """Column-first numbers down each column before moving right."""
    # 2 rows x 3 cols grid of boxes (x0, y0, x1, y1).
    bboxes = []
    for col in range(3):
        for row in range(2):
            x0 = col * 100
            y0 = row * 100
            bboxes.append((x0, y0, x0 + 80, y0 + 80))
    # bboxes order: (c0r0, c0r1, c1r0, c1r1, c2r0, c2r1)
    ranks = geometric_order(bboxes, column_first=True)
    assert ranks == [0, 1, 2, 3, 4, 5]  # already in column-first order
    # Row-first: across row 0 (the 3 top boxes) then row 1.
    ranks_row = geometric_order(bboxes, column_first=False)
    # c0r0=0, c0r1=3, c1r0=1, c1r1=4, c2r0=2, c2r1=5
    assert ranks_row == [0, 3, 1, 4, 2, 5]


def test_order_sections_column_first_default() -> None:
    image = _synth_composite(rows=2, cols=3)
    sections = detect_sections(image, min_area_px=500)
    ordered = order_sections(sections)  # column_first=True by default
    # The section at (row 0, col 0) must be AP 0; (row 1, col 0) must be AP 1.
    by_rc = {(o.row, o.col): o.ap_order for o in ordered}
    assert by_rc[(0, 0)] == 0
    assert by_rc[(1, 0)] == 1
    assert by_rc[(0, 1)] == 2


def test_equalize_box_sizes_expands_undersized() -> None:
    from histo_to_ccf.sectioning.split import DetectedSection, _equalize_box_sizes

    def _sec(x0, y0, x1, y1):
        return DetectedSection(
            bbox_px=(x0, y0, x1, y1),
            mask=np.zeros((1000, 1000), dtype=bool),
            area_px=(x1 - x0) * (y1 - y0),
            centroid_px=((x0 + x1) / 2, (y0 + y1) / 2),
            aspect_ratio=1.0,
        )

    # Four ~100x100 boxes and one short 100x60 box (under-fit in height).
    sections = [
        _sec(0, 0, 100, 100),
        _sec(200, 0, 300, 100),
        _sec(400, 0, 500, 100),
        _sec(600, 20, 700, 80),  # height 60 → under-sized
    ]
    out = _equalize_box_sizes(sections, (1000, 1000), min_frac=0.85)
    # The short box grew to the median height (100), centred on y=50.
    x0, y0, x1, y1 = out[3].bbox_px
    assert (y1 - y0) == 100
    assert (y0 + y1) // 2 == 50
    # Correctly-sized boxes are untouched.
    assert out[0].bbox_px == (0, 0, 100, 100)


def test_equalize_box_sizes_noop_when_few_sections() -> None:
    from histo_to_ccf.sectioning.split import DetectedSection, _equalize_box_sizes

    secs = [
        DetectedSection((0, 0, 10, 10), np.zeros((20, 20), bool), 100, (5, 5), 1.0),
        DetectedSection((0, 0, 10, 4), np.zeros((20, 20), bool), 40, (5, 2), 2.5),
    ]
    assert _equalize_box_sizes(secs, (20, 20)) == secs  # < 3 sections → unchanged


def test_aspect_ratio_filter_removes_label_like_blob() -> None:
    """A tall narrow blob (like a slide-edge label) should be filtered."""
    image = _synth_composite(rows=1, cols=2, noise=0.0)
    # Paint a very tall thin rectangle to mimic a "MAS-CP" label.
    h, _w = image.shape
    image[10 : h - 10, image.shape[1] - 8 : image.shape[1] - 2] = 0.9
    sections = detect_sections(image, min_area_px=500, aspect_ratio_max=3.0)
    # The two ellipses survive; the label does not.
    assert len(sections) == 2


def test_expected_count_keeps_top_n() -> None:
    image = _synth_composite(rows=2, cols=3)
    # Add a tiny stray blob that would otherwise survive min_area_px=200.
    image[5:25, 5:25] = 0.9
    sections = detect_sections(image, min_area_px=200, expected_count=6)
    assert len(sections) == 6


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "example data").exists(),
    reason="real example slide not present (example data/ is gitignored)",
)
def test_real_slide_yields_expected_sections() -> None:
    from histo_to_ccf.io.image import load_image

    slide = (
        Path(__file__).parent.parent
        / "example data"
        / "L07_slide3_2x_whole_overlay.jpg"
    )
    image = load_image(slide)
    # Auto-split is a HINT - it correctly localizes most sections without manual
    # tuning, but anatomical gaps (cerebellum vs brainstem in caudal coronal
    # slices) can fragment a section into 2 blobs, and slide labels can leak in.
    # The GUI exposes the masks as an editable Labels layer for the user to
    # merge/split fragments before registration. The assertions below capture
    # the contract: we expect to find roughly the right number of sections in
    # roughly the right layout, not pixel-perfect detection.
    sections = detect_sections(image, min_area_px=20000, expected_count=15)
    assert 14 <= len(sections) <= 17, f"detected {len(sections)} (expected ~15)"

    ordered = order_sections(sections)
    rows = {o.row for o in ordered}
    assert 2 <= len(rows) <= 4, f"got {len(rows)} rows (expected 3)"

    # Sanity: section bounding boxes are roughly comparable in size - none
    # is wildly larger than the others (that would indicate a merged blob).
    areas = sorted(s.area_px for s in sections)
    assert areas[-1] / areas[0] < 6.0, "largest section is suspiciously bigger than smallest"
