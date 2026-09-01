"""SpikeGLX read path, exercised against a synthetic run.

There is no SpikeGLX data on the reference drive - the 28 non-Neuropixels sessions
there are all Intan - so this builds a minimal but genuine SpikeGLX run and reads it
through the same code the GUI uses. Synthetic, but not a mock: SpikeInterface, neo and
probeinterface all parse it, so a change that breaks real SpikeGLX breaks this too.
"""
from __future__ import annotations

import numpy as np
import pytest

from histo_to_ccf.ephys.formats import (
    SPIKEGLX,
    detect_format,
    select_lfp_stream,
    select_wideband_stream,
)

N_ELECTRODES = 384
#: Neuropixels 1.0. probeinterface resolves geometry from the part number, so a meta
#: without ``imDatPrb_pn`` fails deep inside the probe builder with a bare KeyError.
PART_NUMBER = "NP1000"


def write_spikeglx_run(root, *, n_samples=6000, fs=30000.0, run="sim_g0"):
    """Write a single-probe SpikeGLX run and return its folder.

    The saved-channel count is electrodes + 1: SpikeGLX appends a sync word to every
    imec stream, which is why ``nSavedChans`` and the electrode count differ by one.
    """
    folder = root / run / f"{run}_imec0"
    folder.mkdir(parents=True)
    n_saved = N_ELECTRODES + 1

    rng = np.random.default_rng(0)
    traces = rng.normal(scale=50.0, size=(n_samples, n_saved)).astype(np.int16)
    bin_path = folder / f"{run}_t0.imec0.ap.bin"
    traces.tofile(bin_path)

    imro = "(0,384)" + "".join(f"({i} 0 0 500 250 1)" for i in range(N_ELECTRODES))
    # (nShank,nCol,nRow) then shank:col:row:used per electrode - two staggered columns.
    shank_map = "(1,2,480)" + "".join(
        f"(0:{i % 2}:{i // 2}:1)" for i in range(N_ELECTRODES)
    )
    chan_map = (
        "(384,384,1)"
        + "".join(f"(AP{i};{i}:{i})" for i in range(N_ELECTRODES))
        + "(SY0;768:768)"
    )
    fields = {
        "acqApLfSy": "384,0,1",
        "appVersion": "20230815",
        "fileName": str(bin_path),
        "fileSizeBytes": str(bin_path.stat().st_size),
        "fileTimeSecs": str(n_samples / fs),
        "firstSample": "0",
        "gateMode": "Immediate",
        "imAiRangeMax": "0.6",
        "imAiRangeMin": "-0.6",
        "imDatPrb_type": "0",
        "imDatPrb_pn": PART_NUMBER,
        "imDatPrb_sn": "1234",
        "imMaxInt": "512",
        "imSampRate": str(fs),
        "imroTbl": imro,
        "nSavedChans": str(n_saved),
        "snsApLfSy": "384,0,1",
        "snsSaveChanSubset": "all",
        "typeThis": "imec",
        "snsChanMap": chan_map,
        "snsShankMap": shank_map,
    }
    tilde = {"imroTbl", "snsChanMap", "snsShankMap"}
    (folder / f"{run}_t0.imec0.ap.meta").write_text(
        "".join(f"{'~' if k in tilde else ''}{k}={v}\n" for k, v in fields.items())
    )
    return root / run


@pytest.fixture
def spikeglx_run(tmp_path):
    return write_spikeglx_run(tmp_path)


def test_a_run_folder_is_detected_through_its_imec_subfolder(spikeglx_run):
    got = detect_format(spikeglx_run)

    assert got.format is SPIKEGLX
    assert got.path == spikeglx_run
    assert got.carries_geometry


def test_the_streams_are_listed_and_the_electrode_stream_chosen(spikeglx_run):
    pytest.importorskip("spikeinterface")
    from histo_to_ccf.ephys.formats import list_streams

    got = detect_format(spikeglx_run)
    streams = list_streams(got)

    assert "imec0.ap" in streams
    assert select_wideband_stream(got, streams) == "imec0.ap"
    # No .lf saved in this run, so LFP has to be derived from the wideband stream.
    assert select_lfp_stream(got, streams) is None


def test_opening_gives_the_electrodes_without_the_sync_word(spikeglx_run):
    pytest.importorskip("spikeinterface")
    from histo_to_ccf.ephys.formats import open_stream

    rec = open_stream(detect_format(spikeglx_run), "imec0.ap")

    assert rec.get_num_channels() == N_ELECTRODES
    assert rec.get_sampling_frequency() == pytest.approx(30000.0)


def test_geometry_comes_from_the_meta_rather_than_a_probe_map(spikeglx_run):
    """The contrast with Intan: SpikeGLX stores the imro table, so depths are free."""
    pytest.importorskip("spikeinterface")
    from histo_to_ccf.ephys.formats import open_stream

    rec = open_stream(detect_format(spikeglx_run), "imec0.ap")
    locations = np.asarray(rec.get_channel_locations(), dtype=float)

    assert locations.shape == (N_ELECTRODES, 2)
    # Neuropixels 1.0: two staggered columns, 20 µm between rows of the same column.
    assert len(set(np.round(locations[:, 0], 1))) >= 2
    assert np.ptp(locations[:, 1]) > 3000.0


def test_the_lfp_path_reports_geometry_from_the_recording(spikeglx_run):
    """End to end: the depth axis must be micrometres, not channel indices."""
    pytest.importorskip("spikeinterface")
    from histo_to_ccf.ephys.loader import load_lfp

    data = load_lfp(spikeglx_run, max_seconds=0.1)

    assert data.geometry_source == "recording"
    assert data.derived_from_ap  # no .lf stream, so LFP is derived
    assert data.channel_depths_um.size == N_ELECTRODES
    assert np.ptp(data.channel_depths_um) > 3000.0
