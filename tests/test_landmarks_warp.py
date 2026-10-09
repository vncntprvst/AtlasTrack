"""Landmark thin-plate-spline warp: math, image warp, probe composition, schema."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.atlas.planes import Anchoring
from atlastrack.registration.landmarks_warp import (
    auto_landmarks,
    invert_points,
    salient_landmarks,
    warp_contour_image,
    warp_label_image,
    warp_points,
)
from atlastrack.registration.transforms import RegisteredSectionTransform


def _disk(h, w, cy, cx, r) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r * r


def test_auto_landmarks_count_and_inside_extent() -> None:
    ext = _disk(120, 160, 60, 80, 45)
    pts = auto_landmarks(ext, n_perimeter=6, n_inside=3)
    assert pts.shape == (9, 2)
    # All landmarks lie within the image and (roughly) inside the silhouette.
    for x, y in pts:
        assert 0 <= x < 160 and 0 <= y < 120
        assert ext[round(y), round(x)]


def _three_region_label(h=160, w=200) -> np.ndarray:
    """Three regions inside a disk that meet at a junction near the centre."""
    lbl = np.zeros((h, w), dtype=np.int32)
    disk = _disk(h, w, h // 2, w // 2, 70)
    lbl[disk] = 1
    yy, xx = np.ogrid[:h, :w]
    lbl[disk & (xx >= w // 2) & (yy < h // 2)] = 2  # top-right wedge
    lbl[disk & (xx >= w // 2) & (yy >= h // 2)] = 3  # bottom-right wedge
    return lbl


def test_salient_landmarks_targets_region_junction() -> None:
    lbl = _three_region_label()
    pts = salient_landmarks(lbl)
    h, w = lbl.shape
    assert pts.shape[0] >= 4 and pts.shape[1] == 2
    # All inside the silhouette.
    for x, y in pts:
        assert 0 <= x < w and 0 <= y < h
        assert (lbl > 0)[int(round(y)), int(round(x))]
    # The 1/2/3 wedges meet on the vertical mid-line near the centre; at least
    # one landmark should land close to that junction (x≈w/2, y≈h/2).
    near = [(x, y) for x, y in pts if abs(x - w / 2) < 18 and abs(y - h / 2) < 30]
    assert near, f"no landmark near the region junction: {pts}"


def test_salient_landmarks_spread_apart() -> None:
    pts = salient_landmarks(_three_region_label())
    # No two landmarks coincide (greedy spread enforces a minimum distance).
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.hypot(*(pts[i] - pts[j])) > 5.0


def test_salient_landmarks_falls_back_on_tiny_extent() -> None:
    tiny = np.zeros((20, 20), dtype=np.int32)
    tiny[9:11, 9:11] = 1
    pts = salient_landmarks(tiny)
    assert pts.shape[0] >= 1  # degenerate -> geometric fallback, never crashes


def test_warp_identity_when_undragged() -> None:
    src = auto_landmarks(_disk(100, 100, 50, 50, 40))
    q = np.array([[40.0, 55.0], [60.0, 45.0]])
    assert np.allclose(warp_points(src, src, q), q, atol=1e-6)


def test_warp_and_invert_are_inverses() -> None:
    src = np.array([[10, 10], [90, 10], [50, 50], [10, 90], [90, 90]], float)
    dst = src.copy()
    dst[2] += [15, -8]  # drag the centre point
    q = np.array([[50.0, 50.0], [30.0, 70.0]])
    fwd = warp_points(src, dst, q)
    back = invert_points(src, dst, fwd)
    # Two separately-fitted TPS are exact inverses AT the control points and
    # approximate between them (sub-pixel here) - fine for probe mapping.
    assert np.allclose(back, q, atol=1.5)


def test_warp_label_image_moves_feature() -> None:
    # A small labelled square; drag its location and check it moved.
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[45:55, 45:55] = 7
    src = np.array([[10, 10], [90, 10], [50, 50], [10, 90], [90, 90]], float)
    dst = src.copy()
    dst[2] += [20, 0]  # move centre handle +20 x
    warped = warp_label_image(labels, src, dst)
    # The label-7 blob's centroid should have shifted right by ~20.
    xs = np.nonzero(warped == 7)[1]
    assert xs.size > 0
    assert 60 <= xs.mean() <= 76


def test_warp_contour_image_identity_preserves_edges() -> None:
    # Undragged (source == target) -> the contour stays exactly where it was.
    edge_rc = np.array([[10, 10], [10, 40], [40, 40], [40, 10], [25, 25]])
    src = np.array([[10, 10], [90, 10], [50, 50], [10, 90], [90, 90]], float)
    img = warp_contour_image(edge_rc, src, src, (100, 100))
    for r, c in edge_rc:
        assert img[r, c] == 1
    assert img.sum() == len(edge_rc)


def test_warp_contour_image_follows_drag() -> None:
    # Drag the centre handle +20 in x; a boundary pixel near it should move right.
    edge_rc = np.array([[50, 50]])  # one contour pixel at (row=50, col=50)
    src = np.array([[10, 10], [90, 10], [50, 50], [10, 90], [90, 90]], float)
    dst = src.copy()
    dst[2] += [20, 0]  # +20 x on the centre control point
    img = warp_contour_image(edge_rc, src, dst, (100, 100))
    ys, xs = np.nonzero(img)
    assert xs.size == 1 and 65 <= xs[0] <= 75 and ys[0] == 50


def test_warp_contour_image_clips_out_of_bounds() -> None:
    edge_rc = np.array([[5, 5]])
    src = np.array([[10, 10], [90, 10], [5, 5], [10, 90], [90, 90]], float)
    dst = src.copy()
    dst[2] += [-100, -100]  # shove the contour pixel off-canvas
    img = warp_contour_image(edge_rc, src, dst, (100, 100))
    assert img.sum() == 0  # nothing rasterised outside the image


def test_warp_contour_image_thickness_grows_the_line() -> None:
    # A single point -> a (2t+1)^2 block, so spread-out warped points read as lines.
    edge_rc = np.array([[50, 50]])
    src = np.array([[10, 10], [90, 10], [50, 50], [10, 90], [90, 90]], float)
    thin = warp_contour_image(edge_rc, src, src, (100, 100), thickness=0)
    thick = warp_contour_image(edge_rc, src, src, (100, 100), thickness=2)
    assert thin.sum() == 1
    assert thick.sum() == 25  # 5x5 block centred on the point
    assert thick[50, 50] == 1


def test_landmarks_shift_probe_mapping() -> None:
    anchoring = Anchoring(ox=20.0, oy=0.0, oz=0.0, ux=0.0, uy=0.0, uz=80.0,
                          vx=0.0, vy=40.0, vz=0.0)
    src = np.array([[10, 10], [70, 10], [40, 20], [10, 30], [70, 30]], float)
    dst = src.copy()
    dst[:, 0] += 8.0  # translate all targets +8 in x  -> behaves like a shift
    base = RegisteredSectionTransform(
        anchoring=anchoring, output_size_px=(40, 80), bspline=None,
        atlas_resolution_um=(25.0, 25.0, 25.0),
    )
    warped = RegisteredSectionTransform(
        anchoring=anchoring, output_size_px=(40, 80), bspline=None,
        atlas_resolution_um=(25.0, 25.0, 25.0), manual_landmarks=(src, dst),
    )
    # Clicking +8 in x on the warped map ~= clicking at x on the base map.
    assert np.allclose(warped.apply(48.0, 20.0), base.apply(40.0, 20.0), atol=1.0)


def test_schema_round_trips_landmarks(tmp_path) -> None:
    from atlastrack.project.io import load_project, save_project
    from atlastrack.project.schema import (
        AtlasRef,
        ManualLandmarks,
        Project,
        Section,
        Slide,
    )

    lm = ManualLandmarks(source=[[1.0, 2.0], [3.0, 4.0]], target=[[1.5, 2.5], [3.0, 4.0]])
    sec = Section(index=0, slide_idx=0, bbox_px=(0, 0, 80, 40), manual_landmarks=lm)
    proj = Project(atlas=AtlasRef(), slides=[Slide(image_path="x.png", sections=[sec])])
    path = tmp_path / "p.atlastrack.json"
    save_project(proj, path)
    loaded = load_project(path)
    got = loaded.slides[0].sections[0].manual_landmarks
    assert got is not None and got.source == [[1.0, 2.0], [3.0, 4.0]]


def test_auto_landmarks_cover_the_section_not_just_its_busy_half() -> None:
    """Junctions outnumber every other candidate ~50:1 and sit in the dorsal half.

    Priority order alone therefore spent the whole budget up there: measured on two
    real sections, nine of twelve points landed in the top third and the geometric
    ring - the thing meant to pin the rest of the section - contributed nothing,
    because the budget ran out before it was reached.

    The synthetic section here reproduces that shape: a plain outline with a dense
    thicket of small regions confined to the top quarter, so a single greedy pass
    puts everything in the thicket.
    """
    import numpy as np

    from atlastrack.registration.landmarks_warp import salient_landmarks

    h, w = 300, 400
    labels = np.zeros((h, w), dtype=np.int32)
    labels[20:280, 30:370] = 1  # the section
    # A mosaic of TOUCHING regions across the top only. They must touch: junctions
    # are branch points of the shared-boundary skeleton, so isolated islands
    # produce none at all and would not exercise this in the first place.
    region = 2
    for yy in range(30, 100, 10):
        for xx in range(40, 360, 10):
            labels[yy:yy + 10, xx:xx + 10] = region
            region += 1
    from atlastrack.registration.landmarks_warp import _region_junctions

    assert len(_region_junctions(labels)) > 100, "the thicket must dominate"

    points = np.asarray(salient_landmarks(labels, max_points=12), dtype=float)
    assert len(points) >= 8, f"expected a usable set, got {len(points)}"

    top = sum(1 for _x, y in points if y < h / 3)
    assert top <= len(points) * 0.6, (
        f"{top}/{len(points)} points in the top third - the busy band has captured "
        f"the budget again"
    )

    cells = {(int(3 * y / h), int(3 * x / w)) for x, y in points}
    assert len(cells) >= 8, f"only {len(cells)} of 9 grid cells covered: {sorted(cells)}"


def test_spread_select_without_a_shape_is_the_plain_greedy_pass() -> None:
    """The coverage pass is opt-in, so existing callers keep their behaviour."""
    from atlastrack.registration.landmarks_warp import _spread_select

    cands = [(0.0, 0.0), (1.0, 1.0), (100.0, 100.0), (200.0, 200.0)]
    assert _spread_select(cands, min_dist=50.0, max_points=4) == [
        (0.0, 0.0), (100.0, 100.0), (200.0, 200.0)
    ]


def test_spread_select_still_honours_the_minimum_distance() -> None:
    """Covering a cell must never justify two points on top of each other."""
    from atlastrack.registration.landmarks_warp import _spread_select

    # Two candidates in different cells but only 2 px apart, astride a boundary.
    cands = [(199.0, 150.0), (201.0, 150.0), (20.0, 20.0)]
    chosen = _spread_select(cands, min_dist=50.0, max_points=3, shape=(300, 400))
    for i, (x1, y1) in enumerate(chosen):
        for x2, y2 in chosen[i + 1:]:
            assert (x1 - x2) ** 2 + (y1 - y2) ** 2 >= 50.0**2


def test_region_junctions_are_the_centres_of_each_branch_point() -> None:
    """Found in one pass (one full-image scan per junction took 3 s on LO_05)."""
    import numpy as np

    from atlastrack.registration.landmarks_warp import _region_junctions

    lab = np.zeros((60, 60), dtype=np.int32)
    lab[:30, :30], lab[:30, 30:], lab[30:, :] = 1, 2, 3     # a T where three meet
    pts = _region_junctions(lab)
    assert pts, "no junction found"
    x, y = min(pts, key=lambda p: abs(p[0] - 30) + abs(p[1] - 30))
    assert abs(x - 30) <= 2 and abs(y - 30) <= 2


def _symmetric_section() -> np.ndarray:
    """A mirror-symmetric 'brain' about x = 150, with paired regions either side."""
    yy, xx = np.mgrid[0:240, 0:300]
    lbl = np.zeros((240, 300), dtype=np.int32)
    r = np.hypot((xx - 150) / 120.0, (yy - 120) / 95.0)
    lbl[r <= 1] = 1
    lbl[(r > 0.8) & (r <= 0.9)] = 5  # cortex-like layers along the surface
    lbl[(r > 0.9) & (r <= 1)] = 6
    for dx in (-60, 60):  # a pair of nuclei
        lbl[((xx - 150 - dx) / 25.0) ** 2 + ((yy - 100) / 18.0) ** 2 <= 1] = 2
    lbl[(np.abs(xx - 150) <= 6) & (yy >= 70) & (yy <= 150)] = 3  # a midline ventricle
    lbl[(yy > 150) & (lbl == 1)] = 4  # a ventral region
    return lbl


def test_the_midline_is_found_upright_and_tilted() -> None:
    from scipy import ndimage

    from atlastrack.registration.landmarks_warp import find_midline

    a, b = find_midline(_symmetric_section())
    assert a == pytest.approx(150, abs=3) and b == pytest.approx(0, abs=0.03)

    turned = ndimage.rotate(_symmetric_section(), 10, order=0, reshape=False)
    a, b = find_midline(turned)
    # rotate() turns the content anticlockwise on screen, so the top leans left.
    assert b == pytest.approx(np.tan(np.deg2rad(10)), abs=0.05)
    assert a + b * 120 == pytest.approx(150, abs=4)  # through the centre


def test_one_hemisphere_has_no_midline() -> None:
    from atlastrack.registration.landmarks_warp import find_midline

    assert find_midline(_symmetric_section()[:, :150]) is None


def test_auto_placement_puts_points_at_the_top_and_bottom_of_the_midline() -> None:
    lbl = _symmetric_section()
    pts = salient_landmarks(lbl)
    (ys,) = np.nonzero(lbl[:, 150])
    for y_end in (ys.min(), ys.max()):
        d = np.hypot(pts[:, 0] - 150, pts[:, 1] - y_end)
        assert d.min() < 6, (y_end, pts)
