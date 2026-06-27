"""DeepSlice pre-match: anchoring->AP reduction, crop fingerprint, Register reuse."""
from __future__ import annotations

import numpy as np

from histo_to_ccf.gui.workflow import WorkflowState, crop_fingerprint
from histo_to_ccf.registration.pipeline import anchoring_center_ap_um


def _anchoring9(ap_origin: float, ux: float = 20.0, vx: float = 0.0) -> list[float]:
    # ox + 0.5*ux + 0.5*vx is the plane centre's AP voxel coord.
    return [ap_origin, 0, 0, ux, 0, 0, vx, 0, 0]


def test_anchoring_center_ap_um_matches_coronal_anchoring_roundtrip() -> None:
    """ap_um -> coronal_anchoring -> anchoring_center_ap_um returns the same AP."""
    from histo_to_ccf.atlas.planes import coronal_anchoring

    class _Atlas:
        resolution = (25.0, 25.0, 25.0)
        annotation = np.zeros((100, 80, 90), dtype=np.int32)

    atlas = _Atlas()
    for ap_um in (250.0, 1000.0, 2375.0):
        anch = coronal_anchoring(atlas, ap_um).as_tuple()
        assert np.isclose(anchoring_center_ap_um(anch, 25.0), ap_um)


def test_anchoring_center_ap_um_uses_plane_centre_not_origin() -> None:
    # centre AP voxel = 90 + 0.5*20 + 0.5*10 = 105; at 10 µm/voxel -> 1050 µm.
    anch = _anchoring9(90.0, ux=20.0, vx=10.0)
    assert anchoring_center_ap_um(anch, 10.0) == 1050.0


def test_crop_fingerprint_distinguishes_content_and_shape() -> None:
    a = np.zeros((4, 4), dtype=np.float32)
    b = a.copy()
    b[0, 0] = 1.0  # same shape, different pixels (e.g. a swapped dye image)
    c = np.zeros((4, 5), dtype=np.float32)
    assert crop_fingerprint(a) == crop_fingerprint(a.copy())
    assert crop_fingerprint(a) != crop_fingerprint(b)
    assert crop_fingerprint(a) != crop_fingerprint(c)


def _seed_cache(state: WorkflowState, images: dict[int, np.ndarray]) -> None:
    for idx, img in images.items():
        state.deepslice_anchorings[idx] = _anchoring9(float(idx))
        state.deepslice_fingerprints[idx] = crop_fingerprint(img)


def _reuse(state: WorkflowState, section_images: dict) -> "dict | None":
    # Mirror RegisterPanel._reuse_prematch without constructing the Qt widget.
    anch = state.deepslice_anchorings
    fps = state.deepslice_fingerprints
    if not anch:
        return None
    out: dict = {}
    for idx, img in section_images.items():
        if idx not in anch or fps.get(idx) != crop_fingerprint(img):
            return None
        out[idx] = anch[idx]
    return out


def test_reuse_hits_when_cache_complete_and_fresh() -> None:
    state = WorkflowState()
    imgs = {0: np.full((4, 4), 3.0, np.float32), 1: np.full((4, 4), 7.0, np.float32)}
    _seed_cache(state, imgs)
    out = _reuse(state, imgs)
    assert out is not None and set(out) == {0, 1}


def test_reuse_misses_when_a_section_image_changed() -> None:
    """A swapped dye image (same index, new pixels) forces a fresh DeepSlice pass."""
    state = WorkflowState()
    imgs = {0: np.full((4, 4), 3.0, np.float32), 1: np.full((4, 4), 7.0, np.float32)}
    _seed_cache(state, imgs)
    imgs[1] = np.full((4, 4), 9.0, np.float32)  # different content
    assert _reuse(state, imgs) is None


def test_reuse_misses_when_a_section_is_not_cached() -> None:
    state = WorkflowState()
    imgs = {0: np.full((4, 4), 3.0, np.float32)}
    _seed_cache(state, imgs)
    imgs[2] = np.full((4, 4), 5.0, np.float32)  # never pre-matched
    assert _reuse(state, imgs) is None


def test_reset_clears_prematch_cache() -> None:
    state = WorkflowState()
    imgs = {0: np.full((4, 4), 3.0, np.float32)}
    _seed_cache(state, imgs)
    state.reset()
    assert not state.deepslice_anchorings and not state.deepslice_fingerprints


# -- AP-order safeguard (post-pre-match warning core) ------------------------

def test_prematch_order_issues_clean_series_has_none() -> None:
    from histo_to_ccf.registration.pipeline import prematch_ap_order_issues

    aps = [(0, 1000.0), (1, 1100.0), (2, 1200.0), (3, 1300.0)]
    assert prematch_ap_order_issues(aps) == ([], [])


def test_prematch_order_issues_flags_reversal() -> None:
    from histo_to_ccf.registration.pipeline import prematch_ap_order_issues

    # section 2 lands anterior to section 1 -> AP reverses on the 1->2 step.
    aps = [(0, 1000.0), (1, 1200.0), (2, 1150.0), (3, 1400.0)]
    rev, close = prematch_ap_order_issues(aps)
    assert rev == [(1, 2)] and close == []


def test_prematch_order_issues_flags_too_close() -> None:
    from histo_to_ccf.registration.pipeline import prematch_ap_order_issues

    # 2->3 step (5 µm) is a tiny fraction of the ~100 µm median -> "too close".
    aps = [(0, 1000.0), (1, 1100.0), (2, 1200.0), (3, 1205.0), (4, 1305.0)]
    rev, close = prematch_ap_order_issues(aps)
    assert rev == [] and close == [(2, 3)]


def test_prematch_order_issues_handles_descending_series() -> None:
    from histo_to_ccf.registration.pipeline import prematch_ap_order_issues

    # Overall posterior->anterior is fine; only a local reversal is flagged.
    aps = [(0, 1300.0), (1, 1200.0), (2, 1250.0), (3, 1000.0)]
    rev, close = prematch_ap_order_issues(aps)
    assert rev == [(1, 2)] and close == []


def test_prematch_order_issues_too_few_sections() -> None:
    from histo_to_ccf.registration.pipeline import prematch_ap_order_issues

    assert prematch_ap_order_issues([(0, 1000.0), (1, 1100.0)]) == ([], [])
