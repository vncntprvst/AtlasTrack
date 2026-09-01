"""Channel geometry for formats that store none, and honesty when there is none.

The failure this guards against is quiet: without a probe map, an Intan recording's
"depths" are channel indices, so a 32-channel Poly3 reports a span of 31 instead of
275 µm. That plots perfectly happily and is entirely wrong, so the no-geometry state
is a named value (``GeometrySource.CHANNEL_INDEX``) that callers must check, not a
fallback that looks like data.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.probemap import (
    NEURONEXUS_POLY3_A32_RHD2132,
    GeometrySource,
    ProbeMap,
    load_probe_map,
    map_from_catalog,
    read_csv_map,
    resolve_probe_map,
)
from histo_to_ccf.probes.catalog import NEURONEXUS_A1X32_POLY3


def _csv(tmp_path, text, name="map.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ------------------------------------------------------------------- the source


def test_only_the_index_case_is_non_physical():
    assert not GeometrySource.CHANNEL_INDEX.is_physical
    for source in (
        GeometrySource.RECORDING, GeometrySource.PROBE_MAP, GeometrySource.CATALOG
    ):
        assert source.is_physical


# ------------------------------------------------------------------- the map


def test_a_map_reports_the_span_that_makes_it_obviously_wrong():
    m = ProbeMap(depth_um=np.array([100.0, 200.0, 375.0]), x_um=np.zeros(3))

    assert m.n_channels == 3
    assert m.extent_um == pytest.approx(275.0)


def test_an_empty_map_has_no_extent_rather_than_erroring():
    m = ProbeMap(depth_um=np.array([]), x_um=np.array([]))

    assert m.n_channels == 0
    assert m.extent_um == 0.0


def test_mismatched_axes_are_refused_at_construction():
    with pytest.raises(ValueError, match="differ"):
        ProbeMap(depth_um=np.zeros(4), x_um=np.zeros(3))


def test_shank_ids_must_cover_every_channel():
    with pytest.raises(ValueError, match="one entry per channel"):
        ProbeMap(depth_um=np.zeros(4), x_um=np.zeros(4), shank_ids=np.zeros(2, int))


def test_a_map_that_does_not_fit_the_recording_is_refused_with_both_counts():
    """Recycling a 384-channel map onto 32 channels yields 32 confident wrong depths."""
    m = ProbeMap(depth_um=np.arange(384.0), x_um=np.zeros(384), name="NP1.0")

    with pytest.raises(ValueError) as exc:
        m.check_matches(32)
    assert "384" in str(exc.value) and "32" in str(exc.value)


def test_a_matching_map_passes_quietly():
    ProbeMap(depth_um=np.zeros(32), x_um=np.zeros(32)).check_matches(32)


# ------------------------------------------------------------------- CSV


def test_a_minimal_csv_needs_only_a_depth_column(tmp_path):
    p = _csv(tmp_path, "depth_um\n0\n25\n50\n")

    m = read_csv_map(p)

    np.testing.assert_allclose(m.depth_um, [0, 25, 50])
    np.testing.assert_allclose(m.x_um, [0, 0, 0])
    assert m.shank_ids is None
    assert m.source is GeometrySource.PROBE_MAP


def test_the_common_column_spellings_are_accepted(tmp_path):
    for header in ("depth_um", "depth", "y_um", "y"):
        m = read_csv_map(_csv(tmp_path, f"{header}\n10\n20\n", name=f"{header}.csv"))
        np.testing.assert_allclose(m.depth_um, [10, 20])


def test_a_channel_column_sets_the_order_rather_than_file_order(tmp_path):
    """File order is the assumption most likely to be silently wrong."""
    p = _csv(tmp_path, "channel,depth_um\n2,200\n0,0\n1,100\n")

    m = read_csv_map(p)

    np.testing.assert_allclose(m.depth_um, [0, 100, 200])


def test_x_and_shank_columns_are_read_when_present(tmp_path):
    p = _csv(tmp_path, "depth_um,x_um,shank\n0,-25,0\n25,0,0\n50,25,1\n")

    m = read_csv_map(p)

    np.testing.assert_allclose(m.x_um, [-25, 0, 25])
    np.testing.assert_array_equal(m.shank_ids, [0, 0, 1])


def test_a_csv_with_no_depth_column_says_which_names_it_wanted(tmp_path):
    p = _csv(tmp_path, "channel,foo\n0,1\n")

    with pytest.raises(ValueError, match="depth"):
        read_csv_map(p)


def test_an_empty_csv_is_refused(tmp_path):
    p = _csv(tmp_path, "depth_um\n")

    with pytest.raises(ValueError, match="no rows"):
        read_csv_map(p)


def test_a_partial_shank_column_is_dropped_rather_than_half_applied(tmp_path):
    """Half a shank map is not a shank map; guessing the rest would invent geometry."""
    p = _csv(tmp_path, "depth_um,shank\n0,0\n25,\n50,1\n")

    assert read_csv_map(p).shank_ids is None


# ------------------------------------------------------------------- dispatch


def test_load_dispatches_on_extension(tmp_path):
    p = _csv(tmp_path, "depth_um\n0\n25\n")

    assert load_probe_map(p).n_channels == 2


def test_an_unsupported_extension_lists_what_is_supported(tmp_path):
    p = tmp_path / "map.txt"
    p.write_text("0\n25\n")

    with pytest.raises(ValueError, match=r"\.json"):
        load_probe_map(p)


def test_a_missing_file_is_reported_as_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_probe_map(tmp_path / "nope.csv")


# ------------------------------------------------------------------- catalog


def test_the_poly3_comes_out_of_the_catalog_at_its_real_size():
    """The probe in the optotag rig: 32 sites over 275 µm, three columns."""
    m = map_from_catalog(NEURONEXUS_A1X32_POLY3)

    assert m.n_channels == 32
    assert m.extent_um == pytest.approx(275.0)
    assert sorted(set(np.round(m.x_um, 1))) == [-18.0, 0.0, 18.0]
    assert m.source is GeometrySource.CATALOG


def test_the_poly3_span_is_far_shorter_than_a_neuropixels_bank():
    """Why one Poly3 recording constrains almost nothing on a 5 mm track."""
    poly3 = map_from_catalog(NEURONEXUS_A1X32_POLY3)

    assert poly3.extent_um < 720.0  # one NP2.0 bank


def test_an_unknown_catalog_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="Neuropixels"):
        map_from_catalog("NotAProbe")


# ------------------------------------------------------------------- resolve


def test_resolving_none_stays_none_because_missing_geometry_is_the_callers_call():
    assert resolve_probe_map(None) is None


def test_a_probe_map_passes_through_unchanged():
    m = ProbeMap(depth_um=np.zeros(4), x_um=np.zeros(4))

    assert resolve_probe_map(m) is m


def test_a_catalog_name_resolves_without_touching_the_filesystem():
    m = resolve_probe_map(NEURONEXUS_A1X32_POLY3)

    assert m.n_channels == 32
    assert m.source is GeometrySource.CATALOG


def test_a_path_resolves_to_the_file(tmp_path):
    p = _csv(tmp_path, "depth_um\n0\n25\n")

    assert resolve_probe_map(str(p)).n_channels == 2


def test_the_channel_count_is_checked_when_given():
    with pytest.raises(ValueError, match="32"):
        resolve_probe_map(NEURONEXUS_A1X32_POLY3, n_channels=384)


def test_a_matching_channel_count_resolves():
    assert resolve_probe_map(NEURONEXUS_A1X32_POLY3, n_channels=32).n_channels == 32


# ------------------------------------------------------------------- probeinterface


def test_a_probeinterface_json_round_trips(tmp_path):
    pi = pytest.importorskip("probeinterface")

    probe = pi.Probe(ndim=2, si_units="um")
    positions = np.column_stack([np.zeros(8), np.arange(8) * 20.0])
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(8))
    path = tmp_path / "probe.json"
    pi.write_probeinterface(path, probe)

    m = load_probe_map(path)

    assert m.n_channels == 8
    np.testing.assert_allclose(m.depth_um, np.arange(8) * 20.0)


def test_the_device_channel_wiring_reorders_the_map(tmp_path):
    """The whole point of a map for Intan: the adapter permutes channel order."""
    pi = pytest.importorskip("probeinterface")

    probe = pi.Probe(ndim=2, si_units="um")
    positions = np.column_stack([np.zeros(4), [0.0, 10.0, 20.0, 30.0]])
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    # Contact at depth 0 is wired to recording channel 3, and so on - reversed.
    probe.set_device_channel_indices([3, 2, 1, 0])
    path = tmp_path / "wired.json"
    pi.write_probeinterface(path, probe)

    m = load_probe_map(path)

    np.testing.assert_allclose(m.depth_um, [30.0, 20.0, 10.0, 0.0])


def test_a_multi_probe_file_is_refused_rather_than_silently_taking_the_first(tmp_path):
    pi = pytest.importorskip("probeinterface")

    group = pi.ProbeGroup()
    for _ in range(2):
        probe = pi.Probe(ndim=2, si_units="um")
        probe.set_contacts(
            positions=np.column_stack([np.zeros(4), np.arange(4) * 10.0]),
            shapes="circle", shape_params={"radius": 5},
        )
        group.add_probe(probe)
    path = tmp_path / "two.json"
    pi.write_probeinterface(path, group)

    with pytest.raises(ValueError, match="2 probes"):
        load_probe_map(path)


# ---------------------------------------------------------------------------
# Intan RHX probe maps
#
# The RHX probe map is the one artifact that records the whole chain - probe,
# A32-OM32 adapter, RHD2132 headstage - so it is the map to prefer for an Intan
# recording. The fixtures below are the real thing, trimmed: channel numbers and
# positions are copied verbatim from the files the rig uses.
# ---------------------------------------------------------------------------

#: The four corner cases of the Poly3 map: the top centre site, the bottom centre
#: site, and one site from each side column. ``y`` is as written in the file, i.e.
#: from the lowest site, and the shank tip is a further 62 µm below that.
POLY3_SPOT_CHECKS = {0: (0.0, 275.0), 19: (0.0, 0.0), 23: (-18.0, 237.5), 3: (18.0, 12.5)}
POLY3_TIP_OFFSET_UM = 62.0


def _rhx(tmp_path, pages, name="probe.xml", outline=True):
    """Write an RHX probe map. ``pages`` is {page name: {channel: (x, y)}}."""
    lines = ['<?xml version="1.0"?>', '<IntanRHX version="3.0.0">', " <ProbeMapSettings>"]
    for page_name, sites in pages.items():
        lines.append(f'  <Page name="{page_name}">')
        if outline:
            lines.append(f'   <Line x1="0" x2="0" y1="{-POLY3_TIP_OFFSET_UM}" y2="275"/>')
        lines.append('   <Port name="A">')
        lines += [
            f'    <ElectrodeSite channelNumber="{c}" x="{x}" y="{y}"/>'
            for c, (x, y) in sites.items()
        ]
        lines += ["   </Port>", "  </Page>"]
    lines += [" </ProbeMapSettings>", "</IntanRHX>"]
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _poly3_sites():
    """The full 32-site Poly3 map, keyed by Intan channel, as written by RHX."""
    from histo_to_ccf.ephys.probemap import _POLY3_SITES_BY_PIN, A32_TO_RHD2132

    return {ch: _POLY3_SITES_BY_PIN[pin] for pin, ch in enumerate(A32_TO_RHD2132)}


def test_an_rhx_probe_map_puts_each_site_on_its_intan_channel(tmp_path):
    path = _rhx(tmp_path, {"Shanks": _poly3_sites()})

    got = load_probe_map(path)

    assert got.n_channels == 32
    assert got.extent_um == pytest.approx(275.0)
    for channel, (x, y) in POLY3_SPOT_CHECKS.items():
        assert got.x_um[channel] == pytest.approx(x)
        assert got.depth_um[channel] == pytest.approx(y + POLY3_TIP_OFFSET_UM)


def test_depths_are_referenced_to_the_tip_not_the_lowest_site(tmp_path):
    """The file quotes y from the deepest site; the outline says where silicon ends."""
    sites = _poly3_sites()

    with_outline = load_probe_map(_rhx(tmp_path, {"S": sites}, "a.xml"))
    without = load_probe_map(_rhx(tmp_path, {"S": sites}, "b.xml", outline=False))

    assert with_outline.depth_um.min() == pytest.approx(POLY3_TIP_OFFSET_UM)
    assert without.depth_um.min() == pytest.approx(0.0)


def test_each_page_is_a_shank(tmp_path):
    """Buzsaki32L shape: four shanks of eight, numbered straight through."""
    pages = {
        f"Shank{s + 1}": {s * 8 + i: (float(i), 20.0 * i) for i in range(8)}
        for s in range(4)
    }

    got = load_probe_map(_rhx(tmp_path, pages))

    assert got.shank_ids is not None
    assert list(np.bincount(got.shank_ids)) == [8, 8, 8, 8]
    assert got.shank_ids[0] == 0 and got.shank_ids[-1] == 3


def test_a_single_page_probe_reports_no_shank_split(tmp_path):
    got = load_probe_map(_rhx(tmp_path, {"Shanks": _poly3_sites()}))

    assert got.shank_ids is None


def test_an_rhx_settings_file_is_refused_and_names_the_file_wanted(tmp_path):
    """RHX writes two XMLs and only one is a probe map; picking the wrong one is easy."""
    p = tmp_path / "settings.xml"
    p.write_text('<?xml version="1.0"?>\n<IntanRHX><GeneralConfig/></IntanRHX>', "utf-8")

    with pytest.raises(ValueError, match=r"probe\.xml"):
        load_probe_map(p)


def test_a_channel_claimed_by_two_ports_is_refused(tmp_path):
    """Two ports each numbering 0..n cannot be laid on one recording channel axis."""
    path = tmp_path / "two_ports.xml"
    path.write_text(
        '<?xml version="1.0"?>\n<IntanRHX><ProbeMapSettings><Page name="S">'
        '<Port name="A"><ElectrodeSite channelNumber="0" x="0" y="0"/></Port>'
        '<Port name="B"><ElectrodeSite channelNumber="0" x="0" y="20"/></Port>'
        "</Page></ProbeMapSettings></IntanRHX>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one port per file"):
        load_probe_map(path)


def test_a_map_missing_a_channel_is_refused_rather_than_renumbered(tmp_path):
    sites = _poly3_sites()
    sites.pop(7)

    with pytest.raises(ValueError, match=r"up to 31 but is missing \[7\]"):
        load_probe_map(_rhx(tmp_path, {"Shanks": sites}))


# ---------------------------------------------------------------------------
# The wired built-in
# ---------------------------------------------------------------------------


def test_the_wired_builtin_matches_the_rhx_map_channel_for_channel(tmp_path):
    """The named map must stay equal to the file it stands in for."""
    from_file = load_probe_map(_rhx(tmp_path, {"Shanks": _poly3_sites()}))

    builtin = resolve_probe_map(NEURONEXUS_POLY3_A32_RHD2132, n_channels=32)

    assert builtin.depth_um == pytest.approx(from_file.depth_um)
    assert builtin.x_um == pytest.approx(from_file.x_um)


def test_the_wired_builtin_places_the_same_sites_as_the_catalog_layout():
    """Same probe, so the same site positions - only the channel order differs.

    This is what stops the catalog layout and the wired map drifting apart: they are
    written down separately because they are ordered differently (tip-to-base vs
    adapter pin order), and nothing else would notice if one were edited alone.
    """
    wired = resolve_probe_map(NEURONEXUS_POLY3_A32_RHD2132)
    catalog = map_from_catalog(NEURONEXUS_A1X32_POLY3)

    assert sorted(zip(wired.depth_um, wired.x_um, strict=True)) == pytest.approx(
        sorted(zip(catalog.depth_um, catalog.x_um, strict=True))
    )


def test_the_adapter_wiring_is_a_permutation_of_the_headstage_channels():
    """A dropped or repeated entry would silently mis-place every later site."""
    from histo_to_ccf.ephys.probemap import A32_TO_RHD2132

    assert sorted(A32_TO_RHD2132) == list(range(32))


def test_the_wiring_is_not_the_identity():
    """If it were, an unmapped Intan recording would have been fine all along."""
    from histo_to_ccf.ephys.probemap import A32_TO_RHD2132

    assert list(A32_TO_RHD2132) != list(range(32))


def test_the_wired_builtin_is_reported_as_a_real_map_not_a_bare_catalog_layout():
    got = resolve_probe_map(NEURONEXUS_POLY3_A32_RHD2132)

    assert got.source is GeometrySource.PROBE_MAP
    assert got.source.is_physical


def test_an_unknown_name_lists_both_the_catalog_and_the_wired_maps():
    with pytest.raises(KeyError, match="A32>RHD2132"):
        resolve_probe_map("not a probe")
