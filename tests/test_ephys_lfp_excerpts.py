"""Reading screened LFP excerpts instead of one central slab."""
from __future__ import annotations

import numpy as np
import pytest

from atlastrack.ephys import loader as loader_module
from atlastrack.ephys.loader import LfpExcerpts, excerpt_psd, load_lfp_excerpts

FS = 2500.0
N_CH = 8
DURATION_S = 120.0


class _FakeRecording:
    """A lazy recording stand-in: sine + noise, with an artifact burst at t=40-50 s.

    Only the handful of methods the reader actually calls, so the test exercises the
    real screening and assembly logic without SpikeInterface or a raw file.
    """

    def __init__(self, *, artifact_window=(40.0, 50.0)) -> None:
        self._n = int(DURATION_S * FS)
        self._artifact = artifact_window
        self.channel_ids = list(range(N_CH))
        self.reads: list[tuple[int, int]] = []

    def get_sampling_frequency(self) -> float:
        return FS

    def get_num_samples(self) -> int:
        return self._n

    def get_traces(self, *, start_frame, end_frame, return_in_uV=True):
        self.reads.append((start_frame, end_frame))
        n = end_frame - start_frame
        t = (start_frame + np.arange(n)) / FS
        rng = np.random.default_rng(int(start_frame))
        # Depth-varying amplitude, so activity_score is non-zero and channels differ.
        gain = np.linspace(1.0, 6.0, N_CH)[None, :]
        traces = gain * (np.sin(2 * np.pi * 8.0 * t)[:, None] + rng.normal(0, 0.3, (n, N_CH)))
        lo, hi = self._artifact
        burst = (t >= lo) & (t < hi)
        if burst.any():
            # A cross-channel transient on *every* channel at once - a lick artifact.
            traces[burst, :] += 60.0 * rng.normal(0, 1, (int(burst.sum()), 1))
        return traces

    def get_channel_locations(self):
        return np.column_stack([np.zeros(N_CH), np.arange(N_CH) * 15.0])

    def get_probe(self):
        raise RuntimeError("no probe attached")


@pytest.fixture
def fake_open(monkeypatch):
    """Patch the recording opener so no SpikeInterface or raw data is needed."""
    holder = {}

    def _open(recording_dir, stream_name, lfp_fs):
        rec = holder.get("rec") or _FakeRecording()
        holder["rec"] = rec
        return rec, "ProbeA-AP", True

    monkeypatch.setattr(loader_module, "_open_lfp_recording", _open)
    return holder


# -- reading ---------------------------------------------------------------


def test_excerpts_are_spread_across_the_recording_not_one_slab(fake_open) -> None:
    excerpts = load_lfp_excerpts("ignored", window_s=5.0, n_windows=4, reference=False)

    starts = sorted(s for s, _ in fake_open["rec"].reads)
    assert len(starts) == 4
    # Spread over most of the recording, rather than four adjacent windows.
    assert (starts[-1] - starts[0]) / FS > 0.5 * DURATION_S
    assert excerpts.derived_from_ap is True


def test_only_the_requested_windows_are_read(fake_open) -> None:
    """The whole point: 4 x 5 s read, not 120 s. Raw data lives on a slow disk."""
    load_lfp_excerpts("ignored", window_s=5.0, n_windows=4, reference=False)

    total = sum(stop - start for start, stop in fake_open["rec"].reads)
    assert total <= int(4 * 5.0 * FS)
    assert total < 0.25 * DURATION_S * FS


def test_windows_are_kept_separate_not_concatenated(fake_open) -> None:
    excerpts = load_lfp_excerpts("ignored", window_s=5.0, n_windows=4, reference=False)

    assert len(excerpts.windows) >= 2
    for window in excerpts.windows:
        assert window.shape == (int(5.0 * FS), N_CH)


def test_geometry_and_metadata_come_through(fake_open) -> None:
    excerpts = load_lfp_excerpts("ignored", window_s=5.0, n_windows=3, reference=False)

    assert excerpts.fs == FS
    assert excerpts.channel_depths_um.tolist() == (np.arange(N_CH) * 15.0).tolist()
    assert excerpts.channel_ids == list(range(N_CH))
    assert excerpts.channel_shank_ids is None  # probe raised; must degrade, not crash


# -- screening -------------------------------------------------------------


def test_the_artifact_window_is_rejected_and_the_reason_kept(fake_open) -> None:
    """A lick artifact is a cross-channel transient - reject it, and say why."""
    excerpts = load_lfp_excerpts("ignored", window_s=10.0, n_windows=6, reference=False)

    rejected = [v for v in excerpts.verdicts if not v.kept]
    assert rejected, "the artifact burst should have been caught"
    assert any("artifact" in (v.reject_reason or "") for v in rejected)
    # Rejected windows are reported, not dropped from the record.
    assert len(excerpts.verdicts) == 6
    assert len(excerpts.windows) == len(excerpts.kept_epochs)


def test_a_rejected_window_contributes_no_samples(fake_open) -> None:
    excerpts = load_lfp_excerpts("ignored", window_s=10.0, n_windows=6, reference=False)

    n_kept = sum(1 for v in excerpts.verdicts if v.kept)
    assert len(excerpts.windows) == n_kept
    assert excerpts.total_seconds == pytest.approx(n_kept * 10.0)


def test_keep_trims_to_the_most_informative_windows(fake_open) -> None:
    excerpts = load_lfp_excerpts("ignored", window_s=5.0, n_windows=6, keep=2,
                                 reference=False)

    assert len(excerpts.windows) == 2
    assert len(excerpts.verdicts) == 6  # the rest are recorded as not-kept


def test_a_permissive_tolerance_keeps_the_artifact_window(fake_open) -> None:
    """The threshold is a knob, not a hidden rule - prove it moves the outcome."""
    strict = load_lfp_excerpts("ignored", window_s=10.0, n_windows=6, reference=False)
    loose = load_lfp_excerpts("ignored", window_s=10.0, n_windows=6, reference=False,
                              artifact_tolerance=1.0)

    assert len(loose.windows) > len(strict.windows)


def test_common_median_reference_is_applied_when_asked(fake_open) -> None:
    plain = load_lfp_excerpts("ignored", window_s=5.0, n_windows=3, reference=False)
    referenced = load_lfp_excerpts("ignored", window_s=5.0, n_windows=3, reference=True)

    assert referenced.referenced is True
    assert plain.referenced is False
    # After CMR the across-channel median is zero at every sample.
    assert np.allclose(np.median(referenced.windows[0], axis=1), 0.0)
    assert not np.allclose(np.median(plain.windows[0], axis=1), 0.0)


def test_referencing_happens_before_screening(fake_open) -> None:
    """A window CMR can clean must not be thrown away for the artifact CMR removes.

    The burst here is identical on every channel, which is exactly what common median
    referencing subtracts. Screening the raw traces rejects the window; screening the
    referenced ones keeps it - and in a licking task that difference is most of the
    recording.
    """
    raw = load_lfp_excerpts("ignored", window_s=10.0, n_windows=6, reference=False)
    referenced = load_lfp_excerpts("ignored", window_s=10.0, n_windows=6, reference=True)

    assert len(referenced.windows) > len(raw.windows)


# -- spectra ---------------------------------------------------------------


def test_excerpt_psd_averages_windows_without_a_seam(fake_open) -> None:
    """Concatenating windows would put a broadband step at each join."""
    excerpts = load_lfp_excerpts("ignored", window_s=5.0, n_windows=3, reference=False)

    freqs, psd = excerpt_psd(excerpts)

    assert psd.shape == (N_CH, freqs.size)
    assert np.all(np.isfinite(psd))
    # The 8 Hz drive dominates, and power grows with the channel gain ramp.
    peak = freqs[np.argmax(psd[-1])]
    assert peak == pytest.approx(8.0, abs=1.5)
    assert psd[-1].max() > psd[0].max()


def test_normalise_band_power_removes_a_recording_level_offset():
    """The measured failure: two recordings of the same tissue, one 8x louder."""
    from atlastrack.ephys.features import normalise_band_power

    depth_structure = np.array([1.0, 2.0, 8.0, 3.0, 1.0])[:, None]
    quiet = depth_structure * np.array([[1.0, 10.0]])
    loud = quiet * 8.0  # same tissue, different gain

    a, b = normalise_band_power(quiet), normalise_band_power(loud)

    assert np.allclose(a, b)  # the structure survives, the level does not
    assert np.allclose(np.nanmedian(a, axis=0), 0.0)


def test_normalise_band_power_keeps_bands_independent():
    from atlastrack.ephys.features import normalise_band_power

    out = normalise_band_power(np.array([[1.0, 100.0], [10.0, 100.0], [100.0, 1.0]]))

    assert out.shape == (3, 2)
    assert np.allclose(np.nanmedian(out, axis=0), 0.0)
    assert out[0, 0] == pytest.approx(-1.0)  # a decade below its own band's median


def test_normalise_band_power_rejects_the_wrong_shape():
    from atlastrack.ephys.features import normalise_band_power

    with pytest.raises(ValueError, match="n_channels, n_bands"):
        normalise_band_power(np.array([1.0, 2.0, 3.0]))


def test_excerpt_psd_on_an_empty_set_is_empty_not_an_error():
    freqs, psd = excerpt_psd(
        LfpExcerpts(windows=[], fs=FS, channel_depths_um=np.empty(0),
                    channel_x_um=np.empty(0), channel_ids=[], stream_name="x")
    )
    assert psd.size == 0 and freqs.size == 0


def test_every_window_rejected_yields_no_spectrum_rather_than_a_wrong_one(fake_open) -> None:
    excerpts = load_lfp_excerpts("ignored", window_s=10.0, n_windows=2,
                                 artifact_tolerance=-1.0, reference=False)

    assert excerpts.windows == []
    assert all(not v.kept for v in excerpts.verdicts)
    assert excerpt_psd(excerpts)[1].size == 0
