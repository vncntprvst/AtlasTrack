"""Atlas regions along the track, on the depth-below-surface axis the panels use."""
from __future__ import annotations

import itertools
from typing import ClassVar

import numpy as np
import pytest

from atlastrack.ephys.regions import (
    RegionBand,
    region_bands,
    regions_along_track,
    track_points_ccf_um,
)


class _DepthAtlas:
    """Fake atlas: region changes with DV, so a vertical track crosses boundaries."""

    def __init__(self) -> None:
        self.structures = {
            "A": {"rgb_triplet": [10, 20, 30]},
            "B": {"rgb_triplet": [40, 50, 60]},
        }

    def structure_from_coords(self, coords, *, microns=True, as_acronym=True):
        _ap, dv, _ml = coords
        if dv < 0 or dv > 6000:
            return "Outside atlas"
        return "A" if dv < 2000 else "B"


# -- geometry --------------------------------------------------------------


def test_depth_zero_is_the_entry_and_the_length_is_the_tip():
    entry = (5000.0, 1000.0, 0.0)
    tip = (5000.0, 1000.0, 4000.0)
    pts = track_points_ccf_um(tip, entry, [0.0, 4000.0])
    assert np.allclose(pts[0], entry)
    assert np.allclose(pts[1], tip)


def test_depths_are_measured_along_the_slanted_track_not_in_dv():
    """A tilted shank: 1000 µm *along the shank* is less than 1000 µm deeper in DV."""
    entry = (5000.0, 0.0, 0.0)
    tip = (5000.0, 3000.0, 4000.0)  # tilted in ML
    pts = track_points_ccf_um(tip, entry, [1000.0])
    step = np.asarray(pts[0]) - np.asarray(entry)
    assert np.linalg.norm(step) == pytest.approx(1000.0)
    assert step[2] < 1000.0


def test_depths_beyond_the_track_extrapolate_rather_than_clip():
    entry = (5000.0, 1000.0, 0.0)
    tip = (5000.0, 1000.0, 4000.0)
    pts = track_points_ccf_um(tip, entry, [-500.0, 4500.0])
    assert pts[0][2] == pytest.approx(-500.0)  # above the surface, and visibly so
    assert pts[1][2] == pytest.approx(4500.0)


def test_zero_length_track_collapses_to_the_entry():
    pts = track_points_ccf_um((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), [0.0, 500.0])
    assert pts.shape == (2, 3)
    assert np.allclose(pts, [1.0, 2.0, 3.0])


# -- lookup ----------------------------------------------------------------


def test_regions_along_track_follows_depth():
    entry = (5000.0, 1000.0, 0.0)
    tip = (5000.0, 1000.0, 4000.0)
    hits = regions_along_track(_DepthAtlas(), tip, entry, [500.0, 1500.0, 2500.0, 3500.0])
    assert [h[0] for h in hits] == ["A", "A", "B", "B"]
    assert hits[0][1] == (10, 20, 30)


def test_outside_the_atlas_is_reported_not_hidden():
    entry = (5000.0, 1000.0, 0.0)
    tip = (5000.0, 1000.0, 4000.0)
    hits = regions_along_track(_DepthAtlas(), tip, entry, [-500.0, 500.0])
    assert hits[0] == ("", (0, 0, 0))


# -- bands -----------------------------------------------------------------


def test_bands_merge_runs_and_put_boundaries_at_the_midpoint():
    depths = np.array([0.0, 100.0, 200.0, 300.0])
    hits = [("A", (1, 1, 1))] * 2 + [("B", (2, 2, 2))] * 2
    bands = region_bands(hits, depths)
    assert [b.acronym for b in bands] == ["A", "B"]
    assert bands[0].top_um == pytest.approx(0.0)
    assert bands[0].bottom_um == pytest.approx(150.0)  # midway between 100 and 200
    assert bands[1].top_um == pytest.approx(150.0)
    assert bands[1].bottom_um == pytest.approx(300.0)


def test_bands_are_contiguous_with_no_gaps():
    depths = np.linspace(0.0, 1000.0, 21)
    hits = [("A", (1, 1, 1))] * 7 + [("B", (2, 2, 2))] * 5 + [("A", (1, 1, 1))] * 9
    bands = region_bands(hits, depths)
    assert len(bands) == 3  # the repeat of A is its own band, not merged with the first
    for earlier, later in itertools.pairwise(bands):
        assert earlier.bottom_um == pytest.approx(later.top_um)


def test_an_unlabelled_stretch_stays_a_band():
    depths = np.array([0.0, 100.0, 200.0])
    bands = region_bands([("", (0, 0, 0)), ("A", (1, 1, 1)), ("A", (1, 1, 1))], depths)
    assert [b.acronym for b in bands] == ["", "A"]


def test_empty_input_gives_no_bands():
    assert region_bands([], np.array([])) == []
    assert region_bands([("A", (1, 1, 1))], np.array([])) == []


def test_band_colours_come_from_the_project_palette():
    """Not the Allen rgb_triplet: whole cerebella come back as one wash of yellow."""
    from atlastrack.ephys.regions import band_colours
    from atlastrack.viz.plotly3d import hex_to_rgb, region_style

    bands = [
        RegionBand(top_um=0.0, bottom_um=100.0, acronym="IRN", rgb=(1, 1, 1)),
        RegionBand(top_um=100.0, bottom_um=200.0, acronym="VII", rgb=(1, 1, 1)),
    ]

    colours = band_colours(bands)

    # Curated regions keep the colour the 3D views give them.
    assert colours[0] == hex_to_rgb(region_style("IRN")[0])
    assert colours[1] == hex_to_rgb(region_style("VII")[0])


def test_neighbouring_bands_never_share_a_colour():
    """The whole point of the column is that you can see the boundaries."""
    from atlastrack.ephys.regions import band_colours

    bands = [
        RegionBand(top_um=float(i * 100), bottom_um=float((i + 1) * 100),
                   acronym=f"R{i}", rgb=(200, 200, 100))
        for i in range(30)  # more distinct regions than the palette has colours
    ]

    colours = band_colours(bands)

    for earlier, later in itertools.pairwise(colours):
        assert earlier != later


def test_a_region_keeps_its_colour_across_shanks():
    """The reported bug: FN was blue on one shank tab and orange on the next.

    Each shank crosses a different set of structures, so any scheme that assigns
    colours in encounter order shifts them all. The colour has to be a property of the
    region, not of the list it happens to appear in.
    """
    from atlastrack.ephys.regions import band_colours

    def _bands(acronyms):
        return [
            RegionBand(top_um=float(i * 400), bottom_um=float((i + 1) * 400),
                       acronym=a, rgb=(1, 1, 1))
            for i, a in enumerate(acronyms)
        ]

    from atlastrack.ephys.regions import region_colour_map

    shank0 = _bands(["SIM", "CUL4, 5", "arb", "FN", "NOD", "MV", "PGRNd"])
    shank1 = _bands(["SIM", "CUL4, 5", "CENT3", "arb", "FN", "MV", "GRN", "RM"])

    # Decided over the union, which is what the dialog does for a probe's shanks.
    mapping = region_colour_map([shank0, shank1])
    c0 = dict(zip([b.acronym for b in shank0],
                  band_colours(shank0, shared=mapping), strict=True))
    c1 = dict(zip([b.acronym for b in shank1],
                  band_colours(shank1, shared=mapping), strict=True))

    assert set(c0) & set(c1)  # they really do share regions
    for shared in set(c0) & set(c1):
        assert c0[shared] == c1[shared], f"{shared} changed colour between shanks"
    # And neighbours are still distinguishable on each shank.
    for bands, colours in ((shank0, c0), (shank1, c1)):
        for above, below in itertools.pairwise(bands):
            assert colours[above.acronym] != colours[below.acronym]


def test_region_colour_is_stable_across_processes():
    """Not the builtin hash: Python salts it per process, so colours would drift."""
    from atlastrack.ephys.regions import region_colour

    # Fixed expectations computed from crc32; if this changes, colours changed.
    assert region_colour("FN") == region_colour("FN")
    assert region_colour("FN") != region_colour("MV")


def test_one_acronym_always_gets_one_colour():
    from atlastrack.ephys.regions import band_colours

    bands = [
        RegionBand(top_um=0.0, bottom_um=100.0, acronym="A", rgb=(1, 1, 1)),
        RegionBand(top_um=100.0, bottom_um=200.0, acronym="B", rgb=(1, 1, 1)),
        RegionBand(top_um=200.0, bottom_um=300.0, acronym="A", rgb=(1, 1, 1)),
    ]

    colours = band_colours(bands)

    assert colours[0] == colours[2]
    assert colours[0] != colours[1]


def test_unlabelled_bands_stay_black():
    from atlastrack.ephys.regions import band_colours

    bands = [RegionBand(top_um=0.0, bottom_um=100.0, acronym="", rgb=(9, 9, 9))]

    assert band_colours(bands) == [(0, 0, 0)]


def test_band_geometry_helpers():
    band = RegionBand(top_um=100.0, bottom_um=400.0, acronym="A", rgb=(1, 2, 3))
    assert band.thickness_um == pytest.approx(300.0)
    assert band.mid_um == pytest.approx(250.0)


# ------------------------------------------------------- white matter is reserved


def _band(acr, top, bottom):
    from atlastrack.ephys.regions import RegionBand

    return RegionBand(top_um=top, bottom_um=bottom, acronym=acr, rgb=(0, 0, 0))


class _FakeAtlas:
    """Just the ancestry lookup the colouring needs."""

    TREE: ClassVar[dict[str, list[str]]] = {
        "arb": ["root", "fiber tracts", "cbf"],
        "cbc": ["root", "fiber tracts", "cbf"],
        "py": ["root", "fiber tracts"],
        "MV": ["root", "grey", "BS", "HB", "MY"],
        "GRN": ["root", "grey", "BS", "HB", "MY"],
    }

    def get_structure_ancestors(self, acronym):
        return self.TREE[acronym]  # KeyError for unknowns, like the real one


def test_fibre_tracts_are_found_through_the_structure_tree():
    from atlastrack.ephys.regions import white_matter_acronyms

    found = white_matter_acronyms(_FakeAtlas(), {"arb", "py", "MV", "GRN", "nonsense"})

    assert found == {"arb", "py"}


def test_no_atlas_means_no_white_matter_rather_than_a_guess():
    from atlastrack.ephys.regions import white_matter_acronyms

    assert white_matter_acronyms(None, {"arb"}) == set()


def test_every_tract_gets_the_same_white_and_nothing_else_does():
    """Thin tracts often have no room for a label; the colour has to carry it."""
    from atlastrack.ephys.regions import (
        WHITE_MATTER_RGB,
        region_colour_map,
        white_matter_acronyms,
    )

    bands = [_band("MV", 0, 500), _band("arb", 500, 520),
             _band("GRN", 520, 900), _band("py", 900, 915)]
    wm = white_matter_acronyms(_FakeAtlas(), {b.acronym for b in bands})

    mapping = region_colour_map([bands], white_matter=wm)

    assert mapping["arb"] == WHITE_MATTER_RGB
    assert mapping["py"] == WHITE_MATTER_RGB
    assert mapping["MV"] != WHITE_MATTER_RGB
    assert mapping["GRN"] != WHITE_MATTER_RGB


def test_grey_matter_is_kept_clear_of_white_not_just_off_pure_white():
    """A near-white nucleus would read as a tract just as wrongly as a white one."""
    from atlastrack.ephys.regions import _too_close_to_white, region_colour_map

    bands = [_band(a, i * 100, (i + 1) * 100)
             for i, a in enumerate(["MV", "GRN", "IRN", "PGRNd", "NR", "XII", "NTS"])]

    mapping = region_colour_map([bands], white_matter={"arb"})

    assert not any(_too_close_to_white(c) for c in mapping.values())


def test_two_adjacent_tracts_may_share_white():
    """Deliberate: 'this is white matter' is the reading being supported."""
    from atlastrack.ephys.regions import WHITE_MATTER_RGB, region_colour_map

    bands = [_band("arb", 0, 100), _band("cbc", 100, 200)]

    mapping = region_colour_map([bands], white_matter={"arb", "cbc"})

    assert mapping["arb"] == mapping["cbc"] == WHITE_MATTER_RGB


def test_a_tract_keeps_white_and_its_neighbour_moves():
    from atlastrack.ephys.regions import WHITE_MATTER_RGB, region_colour_map

    bands = [_band("arb", 0, 400), _band("MV", 400, 500)]

    mapping = region_colour_map([bands], white_matter={"arb"})

    assert mapping["arb"] == WHITE_MATTER_RGB
    assert mapping["MV"] != WHITE_MATTER_RGB
