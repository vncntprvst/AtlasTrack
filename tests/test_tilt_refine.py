"""Tests for registration/tilt_refine.py (pure parts; the elastix search is
validated behaviourally in the LO_06 runs, too heavy for CI)."""
from __future__ import annotations

import numpy as np

from atlastrack.atlas.planes import Anchoring
from atlastrack.registration.tilt_refine import (
    _mutual_information,
    _perturbed,
    tilt_proxy_score,
)


def _anchoring() -> Anchoring:
    # ox,oy,oz, ux,uy,uz, vx,vy,vz
    return Anchoring.from_iterable([100.0, 50.0, 60.0, 5.0, 0.0, 40.0, 3.0, 30.0, 0.0])


def test_perturbed_touches_only_tilt_terms() -> None:
    a = _anchoring()
    b = _perturbed(a, 4.0, -2.0)
    at, bt = list(a.as_tuple()), list(b.as_tuple())
    assert bt[3] == at[3] + 4.0     # ux (left/right tilt)
    assert bt[6] == at[6] - 2.0     # vx (dorsal/ventral tilt)
    # every other term is unchanged
    for i in (0, 1, 2, 4, 5, 7, 8):
        assert bt[i] == at[i]


def test_mutual_information_self_is_high() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32)
    mask = np.ones((64, 64), dtype=bool)
    mi_self = _mutual_information(a, a, mask)
    mi_indep = _mutual_information(a, rng.random((64, 64)).astype(np.float32), mask)
    assert mi_self > mi_indep  # identical images share more information


def test_tilt_proxy_prefers_the_true_plane() -> None:
    """A synthetic 'atlas' volume + a section cut from it: the un-shifted plane
    should score >= a badly AP-tilted one."""
    # Volume with AP-varying texture so the plane's AP position matters.
    ap, dv, ml = 60, 40, 40
    z = np.arange(ap)[:, None, None]
    vol = (np.sin(z / 3.0) * 120 + 128).astype(np.float32) * np.ones((ap, dv, ml), np.float32)
    vol += np.linspace(0, 60, ml)[None, None, :]  # ML gradient for some 2D structure
    # A coronal-ish plane at AP=30 spanning DV and ML.
    anch = Anchoring.from_iterable([30.0, 0.0, 0.0, 0.0, 0.0, float(ml - 1),
                                    0.0, float(dv - 1), 0.0])
    from atlastrack.atlas.planes import sample_plane
    section = sample_plane(vol, anch, (dv, ml), order=1)
    section = np.stack([section] * 3, axis=-1)  # RGB-ish

    good = tilt_proxy_score(section, vol, anch)
    bad = tilt_proxy_score(section, vol, _perturbed(anch, 18.0, 0.0))
    assert good >= bad
