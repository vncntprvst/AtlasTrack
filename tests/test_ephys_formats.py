"""Telling Open Ephys, Intan and SpikeGLX apart from what they leave on disk.

Written when adding Intan and SpikeGLX support: the LFP code had grown three separate
Open Ephys entry points, and the detection rules are subtle enough to be worth pinning
down. In particular Intan and Open Ephys BOTH write a ``settings.xml``, so that file
cannot be a rule.
"""
from __future__ import annotations

import pytest

from atlastrack.ephys.formats import (
    INTAN,
    OPEN_EPHYS,
    SPIKEGLX,
    DetectedRecording,
    detect_format,
    select_lfp_stream,
    select_wideband_stream,
)


def _intan(tmp_path, name="info.rhd", *, with_settings=True):
    d = tmp_path / "raw_ephys_data"
    d.mkdir()
    (d / name).write_bytes(b"\x02\x27\x91\xc6" + b"\x00" * 60)
    (d / "amplifier.dat").write_bytes(b"\x00" * 32)
    if with_settings:
        (d / "settings.xml").write_text('<?xml version="1.0"?><IntanRHX Version="3.2.0"/>')
    return d


def _open_ephys(tmp_path, name="Record Node 104"):
    node = tmp_path / name
    (node / "experiment1" / "recording1").mkdir(parents=True)
    (node / "settings.xml").write_text("<SETTINGS/>")
    return node


def _spikeglx(tmp_path, *, run="sim_g0", with_bin=True):
    d = tmp_path / run / f"{run}_imec0"
    d.mkdir(parents=True)
    (d / f"{run}_t0.imec0.ap.meta").write_text("typeThis=imec\n")
    if with_bin:
        (d / f"{run}_t0.imec0.ap.bin").write_bytes(b"\x00" * 32)
    return tmp_path / run


# ------------------------------------------------------------------- detection


def test_intan_is_detected_from_its_info_file(tmp_path):
    d = _intan(tmp_path)
    got = detect_format(d)

    assert got.format is INTAN
    assert got.path == d / "info.rhd"   # the reader wants the file
    assert got.root == d                # what the user pointed at
    assert not got.carries_geometry


def test_pointing_straight_at_the_rhd_works_too(tmp_path):
    d = _intan(tmp_path)

    got = detect_format(d / "info.rhd")

    assert got.format is INTAN
    assert got.path == d / "info.rhd"


def test_an_rhs_file_is_intan_as_much_as_an_rhd(tmp_path):
    d = _intan(tmp_path, name="info.rhs")

    assert detect_format(d).format is INTAN


def test_a_lone_traditional_rhd_is_intan(tmp_path):
    """The single-file layout has no info.rhd - just one self-contained recording."""
    d = tmp_path / "session"
    d.mkdir()
    (d / "TJO_260511_174236.rhd").write_bytes(b"\x00" * 32)

    got = detect_format(d)

    assert got.format is INTAN
    assert got.path.name == "TJO_260511_174236.rhd"


def test_two_loose_rhd_files_are_not_guessed_between(tmp_path):
    """Picking one on the user's behalf would silently analyse the wrong recording."""
    d = tmp_path / "session"
    d.mkdir()
    (d / "a.rhd").write_bytes(b"\x00")
    (d / "b.rhd").write_bytes(b"\x00")

    assert detect_format(d) is None


def test_an_intan_settings_xml_is_not_mistaken_for_open_ephys(tmp_path):
    """Both systems write settings.xml, which is why it is not a detection rule."""
    d = _intan(tmp_path, with_settings=True)

    assert detect_format(d).format is INTAN


def test_a_record_node_is_open_ephys(tmp_path):
    node = _open_ephys(tmp_path)

    got = detect_format(node)

    assert got.format is OPEN_EPHYS
    assert got.path == node
    assert got.carries_geometry


def test_an_experiment_without_a_recording_is_not_a_record_node(tmp_path):
    """The AIND pipeline writes zarr stores named experiment1_... beside the raw data."""
    node = tmp_path / "postprocessed"
    (node / "experiment1_Record Node 104#ProbeA-AP_recording1_group0.zarr").mkdir(
        parents=True
    )

    assert detect_format(node) is None


def test_an_aind_analyzer_store_is_not_claimed_as_a_record_node(tmp_path):
    """Caught on the real LO_07 tree, where it claimed every sorting output.

    The AIND analyzer stores are named ``experiment1_...zarr`` AND each contains a
    child directory called plainly ``recording``, so matching ``experiment*`` with a
    ``recording*`` child hits both halves of the rule without being a recording.
    """
    node = tmp_path / "postprocessed"
    store = node / (
        "experiment1_Record Node 104#Neuropix-PXI-100.ProbeA_recording1_group0.zarr"
    )
    (store / "recording").mkdir(parents=True)
    (store / "sorting").mkdir()

    assert detect_format(node) is None


def test_a_recording_folder_is_recognised_by_its_oebin(tmp_path):
    rec = tmp_path / "recording1"
    rec.mkdir()
    (rec / "structure.oebin").write_text("{}")

    assert detect_format(rec).format is OPEN_EPHYS


def test_spikeglx_is_detected_through_its_imec_subfolder(tmp_path):
    run = _spikeglx(tmp_path)

    got = detect_format(run)

    assert got.format is SPIKEGLX
    assert got.path == run  # the reader takes the run folder
    assert got.carries_geometry


def test_a_meta_without_its_bin_is_not_a_readable_recording(tmp_path):
    """A stray .meta from a half-finished copy must not be claimed as a recording."""
    run = _spikeglx(tmp_path, with_bin=False)

    assert detect_format(run) is None


def test_pointing_at_a_spikeglx_meta_resolves_to_its_folder(tmp_path):
    run = _spikeglx(tmp_path)
    meta = next((run / "sim_g0_imec0").glob("*.meta"))

    got = detect_format(meta)

    assert got.format is SPIKEGLX
    assert got.path == meta.parent


def test_an_unrelated_folder_is_not_a_recording(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "readme.txt").write_text("hello")

    assert detect_format(tmp_path / "notes") is None


def test_a_missing_path_is_none_rather_than_an_error(tmp_path):
    assert detect_format(tmp_path / "nope") is None


def test_a_plain_file_is_not_a_recording(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hello")

    assert detect_format(f) is None


# ------------------------------------------------------------------- streams


def _det(fmt):
    from pathlib import Path

    return DetectedRecording(fmt, Path("."), Path("."))


def test_intan_never_reports_an_lfp_stream():
    """One wideband amplifier stream, so LFP is always derived from it."""
    streams = [
        "RHD2000 amplifier channel",
        "RHD2000 auxiliary input channel",
        "USB board digital input channel",
    ]

    assert select_lfp_stream(_det(INTAN), streams) is None
    assert select_wideband_stream(_det(INTAN), streams) == "RHD2000 amplifier channel"


def test_intan_without_an_amplifier_stream_selects_nothing():
    """Never fall through to "the first stream": on Intan that is a digital line."""
    streams = ["USB board digital input channel"]

    assert select_wideband_stream(_det(INTAN), streams) is None


def test_neuropixels_1_on_open_ephys_uses_its_real_lfp_stream():
    streams = [
        "Record Node 104#Neuropix-PXI-100.ProbeA-AP",
        "Record Node 104#Neuropix-PXI-100.ProbeA-LFP",
    ]

    assert select_lfp_stream(_det(OPEN_EPHYS), streams).endswith("-LFP")


def test_neuropixels_2_on_open_ephys_has_no_lfp_and_derives_from_ap():
    streams = [
        "Record Node 104#NI-DAQmx-103.PXIe-6341",
        "Record Node 104#Neuropix-PXI-100.ProbeA",
    ]

    assert select_lfp_stream(_det(OPEN_EPHYS), streams) is None
    assert select_wideband_stream(_det(OPEN_EPHYS), streams).endswith("ProbeA")


def test_the_spikeglx_sync_stream_is_not_mistaken_for_the_electrodes():
    """"imec0.ap-SYNC" contains ".ap" but is one square-wave line."""
    streams = ["imec0.ap", "imec0.ap-SYNC"]

    assert select_wideband_stream(_det(SPIKEGLX), streams) == "imec0.ap"
    assert select_lfp_stream(_det(SPIKEGLX), streams) is None


def test_the_nidq_stream_is_never_chosen_as_the_electrode_stream():
    streams = ["nidq", "imec0.ap"]

    assert select_wideband_stream(_det(SPIKEGLX), streams) == "imec0.ap"


def test_spikeglx_neuropixels_1_uses_its_lf_stream():
    streams = ["imec0.ap", "imec0.lf"]

    assert select_lfp_stream(_det(SPIKEGLX), streams) == "imec0.lf"


def test_no_streams_selects_nothing_rather_than_raising():
    for fmt in (OPEN_EPHYS, INTAN, SPIKEGLX):
        assert select_wideband_stream(_det(fmt), []) is None
        assert select_lfp_stream(_det(fmt), []) is None


# ------------------------------------------------------------------- real data


REAL_INTAN = (
    r"F:/TJO/TJO_optotag_07/2026_05_11"
    r"/TJO_optotag_07_2026_05_11_5028um_OT_260511_174236/raw_ephys_data"
)


@pytest.mark.slow
def test_a_real_intan_session_is_detected_and_its_streams_listed():
    """The 28 Intan sessions on F: are the reason this format is supported."""
    from pathlib import Path

    if not Path(REAL_INTAN).is_dir():
        pytest.skip("reference drive F: not mounted")
    pytest.importorskip("spikeinterface")
    from atlastrack.ephys.formats import list_streams

    got = detect_format(REAL_INTAN)
    assert got.format is INTAN
    assert got.path.name == "info.rhd"

    streams = list_streams(got)
    assert "RHD2000 amplifier channel" in streams
    assert select_wideband_stream(got, streams) == "RHD2000 amplifier channel"
    assert select_lfp_stream(got, streams) is None
