"""Thread workers for expensive operations."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.qt.threading import thread_worker

from atlastrack.sectioning.ordering import OrderedSection, order_sections
from atlastrack.sectioning.split import detect_sections

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from atlastrack.project.schema import Project


@thread_worker
def detect_sections_worker(
    image: np.ndarray,
    *,
    min_area_px: int = 5000,
    closing_radius_px: int = 0,
    equalize_boxes: bool = True,
    band_bounds: list[tuple[int, int]] | None = None,
) -> list[OrderedSection]:
    """Detect and order sections in a slide image.

    ``band_bounds`` (per-source vertical bands of a merged multi-slide canvas)
    makes the ordering slide-aware so columns don't run across stacked slides.
    """
    sections = detect_sections(
        image,
        min_area_px=min_area_px,
        closing_radius_px=closing_radius_px,
        equalize_boxes=equalize_boxes,
    )
    return order_sections(sections, band_bounds=band_bounds)


@thread_worker
def load_atlas_worker(
    atlas_id: str, *, brainglobe_dir: str | None = None, max_retries: int = 3
) -> BrainGlobeAtlas:
    """Load a BrainGlobe atlas, retrying up to ``max_retries`` times on failure.

    ``brainglobe_dir`` overrides where atlases are downloaded to / loaded from;
    ``None`` uses the BrainGlobe default (``~/.brainglobe``).
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    # check_latest=False skips BrainGlobe's online "is there a newer version?"
    # check, which does an HTTP GET to the GIN server with NO timeout. When GIN
    # is slow/unreachable that call hangs the worker thread forever inside
    # BrainGlobeAtlas() (the GUI's "stuck on load_atlas_worker" freeze) - even
    # though the atlas is already on disk. A genuinely-missing atlas is still
    # downloaded; only the version courtesy-check is skipped.
    kwargs = {"check_latest": False}
    if brainglobe_dir:
        kwargs["brainglobe_dir"] = brainglobe_dir
    last_exc: Exception = RuntimeError("unknown error")
    for attempt in range(max_retries):
        try:
            return BrainGlobeAtlas(atlas_id, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


@thread_worker
def register_worker(
    project: Project,
    atlas: BrainGlobeAtlas,
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    *,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
    engine: str = "auto",
    bending_weight: float = 20.0,
    use_masks: bool = True,
    prealign: bool = True,
    boundary_snap: bool = True,
) -> Project:
    """Run the full registration pipeline in a background thread (no progress)."""
    from atlastrack.registration.pipeline import register_project_with_atlas

    return register_project_with_atlas(
        project,
        atlas,
        section_images=section_images,
        transforms_dir=transforms_dir,
        bspline_grid=bspline_grid,
        max_iterations=max_iterations,
        engine=engine,
        bending_weight=bending_weight,
        use_masks=use_masks,
        prealign=prealign,
        boundary_snap=boundary_snap,
    )


@thread_worker
def deepslice_worker(
    section_images: dict[int, np.ndarray],
    atlas: BrainGlobeAtlas,
    workdir: Path,
    *,
    species: str = "mouse",
    order: dict[int, int] | None = None,
) -> dict[int, list[float]]:
    """Predict per-section atlas anchorings with DeepSlice (background thread).

    ``order`` (section_idx -> AP-sequence rank) numbers the DeepSlice input by the
    user's intended anterior→posterior order, so its serial-section ordering isn't
    bound to the raw detection index.
    """
    from atlastrack.registration.deepslice_adapter import predict_anchorings

    return predict_anchorings(
        section_images, atlas, workdir=workdir, species=species, order=order
    )


@thread_worker
def trajectory_fit_worker(features: dict, tips, entries, atlas):
    """Fit a rigid probe adjustment to the detected LFP boundaries.

    Coarse grid first, then a fine one around its optimum. Measured on LO_07 ProbeB a
    single placement costs ~9 ms, so the full 33x13x9 grid is ~34 s of frozen UI while
    coarse-to-fine reaches the same answer in ~7 s. The refinement window is generous
    enough that the fine pass can leave the coarse cell it started in.

    Returns ``{"fit", "evidence", "notes"}``; ``notes`` carries the leave-one-out
    checks, which are what say whether any of the fitted numbers should be believed.
    """
    import numpy as np

    from atlastrack.probes.trajectory_fit import (
        evidence_from_features,
        fit_trajectory,
        leave_one_out,
    )

    yield {"current": 0, "total": 3, "msg": "Detecting boundaries in the features…"}
    evidence = evidence_from_features(features)
    if not evidence:
        return {"fit": None, "evidence": {}, "notes":
                "No shank produced a boundary above its own noise level, so there is "
                "nothing to fit to."}

    yield {"current": 1, "total": 3, "msg": "Searching placements (coarse)…"}
    coarse = fit_trajectory(
        tips, entries, evidence, atlas,
        offsets_um=np.arange(-400.0, 401.0, 50.0),
        rolls_deg=np.arange(-15.0, 15.1, 5.0),
        tilts_deg=np.arange(-10.0, 10.1, 5.0),
    )

    yield {"current": 2, "total": 3, "msg": "Refining and checking each shank…"}
    fit = fit_trajectory(
        tips, entries, evidence, atlas,
        offsets_um=np.arange(coarse.offset_um - 60.0, coarse.offset_um + 60.1, 15.0),
        rolls_deg=np.arange(coarse.roll_deg - 6.0, coarse.roll_deg + 6.1, 1.5),
        tilts_deg=np.arange(coarse.tilt_deg - 6.0, coarse.tilt_deg + 6.1, 1.5),
    )

    notes = []
    for name, values, tol, unit in (
        ("roll_deg", np.arange(-20.0, 20.1, 2.5), 5.0, "deg"),
        ("offset_um", np.arange(-400.0, 401.0, 25.0), 100.0, "um"),
    ):
        held = {"offset_um": fit.offset_um, "roll_deg": fit.roll_deg,
                "tilt_deg": fit.tilt_deg}
        held.pop(name)
        notes.append(
            leave_one_out(tips, entries, evidence, atlas, name=name, values=values,
                          **held).summary(tol, unit)
        )
    return {"fit": fit, "evidence": evidence, "notes": chr(10).join(notes)}


@thread_worker
def multi_lfp_power_worker(
    refs: list,
    shank_indices: list,
    *,
    window_s: float = 10.0,
    n_windows: int = 6,
    fmin: float = 0.0,
    fmax: float = 300.0,
    bin_um: float = 15.0,
):
    """Compute every attached recording's LFP and stack them per shank.

    Yields ``{"current", "total", "msg"}`` per recording - each one is a raw read off
    the reference disk, so a silent multi-minute wait is not acceptable - and returns
    ``{"stacks": {shank: ShankStack}, "recordings": [...], "failed": [...]}``.

    A recording that fails to read is reported and the rest are still stacked: losing
    one bank should not cost the user the shanks the others cover.
    """
    from atlastrack.ephys.combine import RecordingFeatures, stack_penetration
    from atlastrack.ephys.loader import excerpt_psd, load_lfp_excerpts

    computed: list = []
    failed: list[tuple[str, str]] = []
    total = len(refs)
    for i, ref in enumerate(refs, start=1):
        label = getattr(ref, "label", "") or Path(getattr(ref, "path", "")).name
        yield {"current": i - 1, "total": total, "msg": f"Reading {label} ({i}/{total})…"}
        try:
            data = load_lfp_excerpts(
                ref.path, getattr(ref, "stream_name", None),
                window_s=window_s, n_windows=n_windows,
                probe_map=getattr(ref, "probe_map", None),
            )
            # Depth-referenced features are meaningless on ordinal positions, and a
            # 32-channel probe spanning "31" would look plausible on a plot. Refuse
            # here so the recording is reported as failed with the reason, rather
            # than silently stacked against real micrometres from other recordings.
            if getattr(data, "geometry_source", "recording") == "channel_index":
                raise RuntimeError(
                    "this recording stores no channel geometry and no probe map was "
                    "set, so channel depths are indices rather than micrometres. Set "
                    "a probe map on the recording (a .json/.prb/.imro/.csv file, or a "
                    "catalog probe name)."
                )
            freqs, psd = excerpt_psd(data, fmin=fmin, fmax=fmax)
            if psd.size == 0:
                raise RuntimeError("every candidate window was artifact-dominated")
        except Exception as exc:  # one unreadable recording must not sink the rest
            failed.append((label, str(exc)[:300]))
            continue
        computed.append(RecordingFeatures(
            label=label,
            stream_name=data.stream_name,
            insertion_depth_um=float(getattr(ref, "insertion_depth_um", 0.0) or 0.0),
            freqs_hz=freqs,
            psd=psd,
            axial_um=np.asarray(data.channel_depths_um, dtype=float),
            x_um=np.asarray(data.channel_x_um, dtype=float),
            shank_ids=data.channel_shank_ids,
            electrode_range=getattr(ref, "electrode_range", None),
            channel_ids=list(data.channel_ids),
            reference_groups=int(getattr(data, "reference_groups", 0)),
        ))
    yield {"current": total, "total": total, "msg": "Stacking shanks…"}
    return {
        "stacks": stack_penetration(computed, shank_indices, bin_um=bin_um),
        "recordings": computed,
        "failed": failed,
    }


@thread_worker
def lfp_power_worker(
    recording_dir: Path,
    stream_name: str | None = None,
    *,
    window_s: float = 10.0,
    n_windows: int = 6,
    fmin: float = 0.0,
    fmax: float = 300.0,
    probe_map: object = None,
) -> dict:
    """Load an LFP segment and compute its depth x frequency power map.

    Returns a dict with: ``freqs`` (n_freq), ``psd`` (n_channels, n_freq, depth-
    sorted), ``image`` (uint8 power map), ``depths_um`` (sorted, µm from tip),
    ``x_um`` (shank column per channel), ``channel_ids``, ``stream_name`` and
    ``derived_from_ap``. Runs the SpikeInterface load in the background thread.
    """
    from atlastrack.ephys.features import power_image
    from atlastrack.ephys.loader import excerpt_psd, load_lfp_excerpts

    # Screened excerpts spread across the recording, not one central slab: windows
    # dominated by cross-channel artifact (licking) are rejected and reported instead
    # of being averaged in. This is why there is no "seconds to analyse" control.
    data = load_lfp_excerpts(
        recording_dir, stream_name, window_s=window_s, n_windows=n_windows,
        probe_map=probe_map,
    )
    if getattr(data, "geometry_source", "recording") == "channel_index":
        raise RuntimeError(
            "This recording stores no channel geometry (Intan writes none), and no "
            "probe map was supplied, so the depth axis would be channel indices "
            "rather than micrometres. Set a probe map for this recording."
        )
    freqs, psd = excerpt_psd(data, fmin=fmin, fmax=fmax)
    if psd.size == 0:
        raise RuntimeError(
            "Every candidate window was rejected as artifact-dominated, so there is "
            "no clean LFP to show. The recording may be unusable, or the artifact "
            "threshold too strict for it."
        )

    # Sort ascending by the probe y-location (distance along the shank from the
    # tip) so the display can put the tip at the bottom. Keep the ABSOLUTE y - do
    # NOT zero it at the lowest channel: the lowest recorded electrode usually sits
    # well above the physical tip (e.g. the NP2.0 chisel tip + bank offset), and
    # forcing it to depth 0 would wrongly pin it to the histology tip.
    order = np.argsort(data.channel_depths_um)
    depths = np.asarray(data.channel_depths_um)[order]
    psd_sorted = psd[order]
    sids = data.channel_shank_ids
    shank_ids = np.asarray(sids)[order] if sids is not None else None

    return {
        "freqs": freqs,
        "psd": psd_sorted,
        "image": power_image(psd_sorted),
        "depths_um": depths,
        "x_um": data.channel_x_um[order],
        "shank_ids": shank_ids,
        "channel_ids": [data.channel_ids[i] for i in order],
        "stream_name": data.stream_name,
        "derived_from_ap": data.derived_from_ap,
        # What the screening actually did, so a lossy read is never silent.
        "epochs_kept": len(data.windows),
        "epochs_total": len(data.verdicts),
        "seconds_used": data.total_seconds,
        "rejected": [
            (v.t_start_s, v.t_end_s, v.reject_reason)
            for v in data.verdicts if not v.kept
        ],
    }


@thread_worker
def register_worker_progressive(
    project: Project,
    atlas: BrainGlobeAtlas,
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    *,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
    anchorings: dict[int, list[float]] | None = None,
    engine: str = "auto",
    bending_weight: float = 20.0,
    use_masks: bool = True,
    prealign: bool = True,
    boundary_snap: bool = True,
    preserve_manual: bool = True,
    refine_tilt: bool = False,
):
    """Registration pipeline that yields per-section progress dicts.

    Each yielded value is a dict with keys:
        ``current``  int - sections completed so far
        ``total``    int - total sections to process
        ``msg``      str - human-readable status line

    The final ``return`` value is the updated :class:`Project`.
    """
    import SimpleITK as sitk

    from atlastrack.io.ccf_coords import atlas_resolution_um
    from atlastrack.registration.pipeline import (
        _apply_to_shank_registered,
        anchoring_for_section,
        register_section_image,
    )
    from atlastrack.registration.transforms import RegisteredSectionTransform

    anchorings = anchorings or {}

    # Collect work items: sections with a provided image AND either a DeepSlice
    # anchoring or a manually-assigned plane.
    tasks: list[tuple[int, object, object]] = []
    preserved: list[int] = []
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            if section.index not in section_images:
                continue
            # Don't re-register sections the user has hand-corrected: re-running
            # registration would recompute the B-spline and silently undo a
            # "Reset morph to plane" + box/landmark fix. Keep their existing result
            # (clear the correction via "Reset adjustment" to re-register one).
            if (
                preserve_manual
                and section.registration is not None
                and (section.manual_affine is not None
                     or section.manual_landmarks is not None)
            ):
                preserved.append(section.index)
                continue
            if section.index in anchorings or section.plane is not None:
                tasks.append((slide_idx, slide, section))

    if preserved:
        yield {
            "current": 0, "total": len(tasks),
            "msg": f"Keeping {len(preserved)} hand-corrected section(s) unchanged: "
                   f"{preserved}",
        }

    n_total = len(tasks)
    if n_total == 0:
        yield {"current": 0, "total": 0, "msg": "No sections to register."}
        return project

    from loguru import logger

    tfm_dir = Path(transforms_dir)
    tfm_dir.mkdir(parents=True, exist_ok=True)
    res_um = atlas_resolution_um(atlas)
    # Pass the raw (uint16) reference volume; each section's slice is
    # interpolated to float32 on the fly, so we never hold a full float32 copy
    # of the atlas (hundreds of MB) during the loop.
    ref_vol = atlas.reference
    registered: dict[tuple[int, int], RegisteredSectionTransform] = {}

    failed: list[int] = []
    for i, (slide_idx, slide, section) in enumerate(tasks):
        yield {"current": i, "total": n_total, "msg": f"Registering section {i + 1} of {n_total}"}
        logger.info("Registering section {} ({}/{})", section.index, i + 1, n_total)

        img = section_images[section.index]
        # A manually assigned AP takes precedence over a DeepSlice prediction
        # (see anchoring_for_section); DeepSlice, on by default, otherwise silently
        # overrode every hand-set AP.
        anchoring = anchoring_for_section(section, anchorings, atlas)

        # Optionally nudge the plane's tilt to better fit this section (fixes the
        # L/R-asymmetry where a paramedian nucleus sits on tissue one side but in a
        # gap the other). Conservative: only accepts a meaningful residual gain.
        if refine_tilt:
            from atlastrack.registration.tilt_refine import refine_tilt as _refine_tilt

            anchoring, tinfo = _refine_tilt(
                img, atlas, anchoring, reference_volume=ref_vol,
                register_kwargs=dict(
                    bspline_grid=bspline_grid, max_iterations=max_iterations,
                    engine=engine, bending_weight=bending_weight,
                    use_masks=use_masks, prealign=prealign,
                ),
            )
            if tinfo.get("refined"):
                logger.info(
                    "Section {} tilt refined (dux={}, dvx={}, resid {:.3f}->{:.3f})",
                    section.index, tinfo["d_ux"], tinfo["d_vx"],
                    tinfo["baseline_residual"], tinfo["residual"],
                )

        # One section failing (e.g. a plane with too little atlas overlap) must
        # not abort the whole batch - log it and carry on.
        try:
            reg, sitk_tf = register_section_image(
                img,
                atlas,
                anchoring=anchoring,
                bspline_grid=bspline_grid,
                max_iterations=max_iterations,
                reference_volume=ref_vol,
                engine=engine,
                bending_weight=bending_weight,
                use_masks=use_masks,
                prealign=prealign,
                boundary_snap=boundary_snap,
            )
        except Exception as exc:
            failed.append(section.index)
            logger.warning("Section {} failed: {}", section.index, exc)
            yield {
                "current": i + 1, "total": n_total,
                "msg": f"Section {i + 1} FAILED: {str(exc).splitlines()[-1][:120]}",
            }
            continue
        logger.info("Section {} done (residual={})", section.index, reg.residual)

        tfm_path = tfm_dir / f"section_{section.index:03d}.h5"
        sitk.WriteTransform(sitk_tf, str(tfm_path))
        reg.bspline_transform_path = str(tfm_path.relative_to(tfm_dir.parent))
        section.registration = reg

        registered[(slide_idx, section.index)] = RegisteredSectionTransform(
            anchoring=anchoring,
            output_size_px=reg.output_size_px,
            bspline=sitk_tf,
            atlas_resolution_um=res_um,
        )
        res_str = f"{reg.residual:.4f}" if reg.residual is not None else "n/a"
        yield {"current": i + 1, "total": n_total, "msg": f"Section {i + 1} done (residual={res_str})"}

    if failed:
        yield {
            "current": n_total, "total": n_total,
            "msg": f"Done with {len(failed)} failure(s): sections {failed}",
            "failed": list(failed),
        }

    # Project CCF positions onto shanks.
    for probe in project.probes:
        for shank in probe.shanks:
            _apply_to_shank_registered(shank, project, registered)

    return project


@thread_worker
def discover_recordings_worker(root: str, sidecar: str | None = None) -> list:
    """Scan ``root`` for Open Ephys recordings and group them into penetrations.

    Reads probe geometry only - no traces - but still walks the reference disk, which
    is slow enough to freeze the UI if run inline: ten streams under one LO_07 session
    took ~40 s off the spinning drive.
    """
    from atlastrack.ephys.discovery import discover

    return discover(root, sidecar=sidecar)
