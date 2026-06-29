"""Landmark thin-plate-spline warp: math, image warp, probe composition, schema."""
from __future__ import annotations

import numpy as np

from histo_to_ccf.atlas.planes import Anchoring
from histo_to_ccf.registration.landmarks_warp import (
    auto_landmarks,
    invert_points,
    salient_landmarks,
    warp_contour_image,
    warp_label_image,
    warp_points,
)
from histo_to_ccf.registration.transforms import RegisteredSectionTransform


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
    from histo_to_ccf.project.io import load_project, save_project
    from histo_to_ccf.project.schema import (
        AtlasRef,
        ManualLandmarks,
        Project,
        Section,
        Slide,
    )

    lm = ManualLandmarks(source=[[1.0, 2.0], [3.0, 4.0]], target=[[1.5, 2.5], [3.0, 4.0]])
    sec = Section(index=0, slide_idx=0, bbox_px=(0, 0, 80, 40), manual_landmarks=lm)
    proj = Project(atlas=AtlasRef(), slides=[Slide(image_path="x.png", sections=[sec])])
    path = tmp_path / "p.histo2ccf.json"
    save_project(proj, path)
    loaded = load_project(path)
    got = loaded.slides[0].sections[0].manual_landmarks
    assert got is not None and got.source == [[1.0, 2.0], [3.0, 4.0]]
