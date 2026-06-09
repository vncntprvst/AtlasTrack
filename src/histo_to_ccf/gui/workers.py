"""Thread workers for expensive operations."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.qt.threading import thread_worker

from histo_to_ccf.sectioning.ordering import OrderedSection, order_sections
from histo_to_ccf.sectioning.split import detect_sections

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from histo_to_ccf.project.schema import Project


@thread_worker
def detect_sections_worker(
    image: np.ndarray,
    *,
    min_area_px: int = 5000,
    closing_radius_px: int = 0,
    equalize_boxes: bool = True,
) -> list[OrderedSection]:
    """Detect and order sections in a slide image."""
    sections = detect_sections(
        image,
        min_area_px=min_area_px,
        closing_radius_px=closing_radius_px,
        equalize_boxes=equalize_boxes,
    )
    return order_sections(sections)


@thread_worker
def load_atlas_worker(
    atlas_id: str, *, brainglobe_dir: str | None = None, max_retries: int = 3
) -> "BrainGlobeAtlas":
    """Load a BrainGlobe atlas, retrying up to ``max_retries`` times on failure.

    ``brainglobe_dir`` overrides where atlases are downloaded to / loaded from;
    ``None`` uses the BrainGlobe default (``~/.brainglobe``).
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    kwargs = {"brainglobe_dir": brainglobe_dir} if brainglobe_dir else {}
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
    project: "Project",
    atlas: "BrainGlobeAtlas",
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    *,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
    engine: str = "auto",
    bending_weight: float = 20.0,
    use_masks: bool = True,
) -> "Project":
    """Run the full registration pipeline in a background thread (no progress)."""
    from histo_to_ccf.registration.pipeline import register_project_with_atlas

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
    )


@thread_worker
def deepslice_worker(
    section_images: dict[int, np.ndarray],
    atlas: "BrainGlobeAtlas",
    workdir: Path,
    *,
    species: str = "mouse",
) -> dict[int, list[float]]:
    """Predict per-section atlas anchorings with DeepSlice (background thread)."""
    from histo_to_ccf.registration.deepslice_adapter import predict_anchorings

    return predict_anchorings(section_images, atlas, workdir=workdir, species=species)


@thread_worker
def lfp_power_worker(
    recording_dir: Path,
    stream_name: str | None = None,
    *,
    max_seconds: float = 60.0,
    fmin: float = 0.0,
    fmax: float = 300.0,
) -> dict:
    """Load an Open Ephys LFP segment and compute its depth x frequency power map.

    Returns a dict with: ``freqs`` (n_freq), ``psd`` (n_channels, n_freq, depth-
    sorted), ``image`` (uint8 power map), ``depths_um`` (sorted, µm from tip),
    ``x_um`` (shank column per channel), ``channel_ids``, ``stream_name`` and
    ``derived_from_ap``. Runs the SpikeInterface load in the background thread.
    """
    from histo_to_ccf.ephys.features import lfp_psd, power_image
    from histo_to_ccf.ephys.loader import load_lfp

    data = load_lfp(recording_dir, stream_name, max_seconds=max_seconds)
    freqs, psd = lfp_psd(data.traces, data.fs, fmin=fmin, fmax=fmax)

    # Depth = distance from tip: smallest y is the tip end. Sort ascending so the
    # display can put the tip at the bottom.
    order = np.argsort(data.channel_depths_um)
    depths = data.channel_depths_um[order]
    depths = depths - float(depths.min())  # zero at the tip
    psd_sorted = psd[order]

    return {
        "freqs": freqs,
        "psd": psd_sorted,
        "image": power_image(psd_sorted),
        "depths_um": depths,
        "x_um": data.channel_x_um[order],
        "channel_ids": [data.channel_ids[i] for i in order],
        "stream_name": data.stream_name,
        "derived_from_ap": data.derived_from_ap,
    }


@thread_worker
def register_worker_progressive(
    project: "Project",
    atlas: "BrainGlobeAtlas",
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    *,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
    anchorings: dict[int, list[float]] | None = None,
    engine: str = "auto",
    bending_weight: float = 20.0,
    use_masks: bool = True,
):
    """Registration pipeline that yields per-section progress dicts.

    Each yielded value is a dict with keys:
        ``current``  int — sections completed so far
        ``total``    int — total sections to process
        ``msg``      str — human-readable status line

    The final ``return`` value is the updated :class:`Project`.
    """
    import SimpleITK as sitk

    from histo_to_ccf.atlas.planes import Anchoring, anchoring_from_plane_params
    from histo_to_ccf.io.ccf_coords import atlas_resolution_um
    from histo_to_ccf.registration.pipeline import (
        _apply_to_shank_registered,
        register_section_image,
    )
    from histo_to_ccf.registration.transforms import RegisteredSectionTransform

    anchorings = anchorings or {}

    # Collect work items: sections with a provided image AND either a DeepSlice
    # anchoring or a manually-assigned plane.
    tasks: list[tuple[int, object, object]] = []
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            if section.index not in section_images:
                continue
            if section.index in anchorings or section.plane is not None:
                tasks.append((slide_idx, slide, section))

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
        yield {"current": i, "total": n_total, "msg": f"Registering section {section.index} ({i + 1}/{n_total})…"}
        logger.info("Registering section {} ({}/{})", section.index, i + 1, n_total)

        img = section_images[section.index]
        if section.index in anchorings:
            anchoring = Anchoring.from_iterable(anchorings[section.index])
        else:
            anchoring = anchoring_from_plane_params(atlas, section.plane)

        # One section failing (e.g. a plane with too little atlas overlap) must
        # not abort the whole batch — log it and carry on.
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
            )
        except Exception as exc:  # noqa: BLE001 — reported to the user, batch continues
            failed.append(section.index)
            logger.warning("Section {} failed: {}", section.index, exc)
            yield {
                "current": i + 1, "total": n_total,
                "msg": f"Section {section.index} FAILED: {str(exc).splitlines()[-1][:120]}",
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
        yield {"current": i + 1, "total": n_total, "msg": f"Section {section.index} done (residual={res_str})"}

    if failed:
        yield {
            "current": n_total, "total": n_total,
            "msg": f"Done with {len(failed)} failure(s): sections {failed}",
        }

    # Project CCF positions onto shanks.
    for probe in project.probes:
        for shank in probe.shanks:
            _apply_to_shank_registered(shank, project, registered)

    return project
