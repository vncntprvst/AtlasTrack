"""Orchestrate the per-section registration pipeline.

Two execution paths:

- **Manual (M1)**: every section has ``Section.plane`` populated by the user
  (midline + dorsal-surface anchors + AP). The pipeline fills shank
  coordinates by running a :class:`ManualSectionTransform`.

- **Registered (M3)**: every section gets an oblique atlas plane and an
  optional 2D B-spline refinement. The pipeline:

    1. Resolves an :class:`Anchoring` (either from PlaneParams or from a
       :class:`PlanePredictor`).
    2. Resamples the atlas reference at that plane.
    3. Runs :func:`refine_with_bspline` against the section image.
    4. Writes the SimpleITK transform to a sidecar file.
    5. Stores the resulting :class:`RegistrationResult` on the Section.
    6. Re-projects every shank's tip/entry pixel through the composed
       transform into CCF µm.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from histo_to_ccf.atlas.planes import (
    Anchoring,
    anchoring_from_plane_params,
    resample_atlas_at_plane,
)
from histo_to_ccf.io.ccf_coords import atlas_resolution_um
from histo_to_ccf.project.schema import Project, RegistrationResult, Section, Shank
from histo_to_ccf.registration.bspline import refine_with_bspline
from histo_to_ccf.registration.predictor import PlanePredictor
from histo_to_ccf.registration.transforms import (
    ManualSectionTransform,
    RegisteredSectionTransform,
    build_registered_transform,
)

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


# ── Manual (M1) pipeline ──────────────────────────────────────────────────────


def register_project(project: Project, predictor: PlanePredictor) -> Project:
    """Manual-mode (M1) pipeline. Mutates and returns ``project``."""
    transforms: dict[tuple[int, int], ManualSectionTransform] = {}
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            transforms[(slide_idx, section.index)] = _manual_transform_for(
                section, predictor
            )
    for probe in project.probes:
        for shank in probe.shanks:
            _apply_to_shank_manual(shank, project, transforms)
    return project


def _manual_transform_for(
    section: Section, predictor: PlanePredictor
) -> ManualSectionTransform:
    if section.plane is None:
        plane = predictor.predict(None, section_index=section.index)  # type: ignore[arg-type]
    else:
        plane = section.plane
    return ManualSectionTransform(plane=plane)


def _apply_to_shank_manual(
    shank: Shank,
    project: Project,
    transforms: dict[tuple[int, int], ManualSectionTransform],
) -> None:
    if shank.tip_px is not None and shank.tip_section_idx is not None:
        tx = _lookup_manual(project, shank.tip_section_idx, transforms)
        if tx is not None:
            shank.tip_ccf_um = tx.apply(shank.tip_px.x_px, shank.tip_px.y_px)
    if shank.entry_px is not None and shank.entry_section_idx is not None:
        tx = _lookup_manual(project, shank.entry_section_idx, transforms)
        if tx is not None:
            shank.entry_ccf_um = tx.apply(shank.entry_px.x_px, shank.entry_px.y_px)


def _lookup_manual(
    project: Project,
    section_idx: int,
    transforms: dict[tuple[int, int], ManualSectionTransform],
) -> ManualSectionTransform | None:
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            if section.index == section_idx:
                return transforms.get((slide_idx, section.index))
    return None


# ── Registered (M3) pipeline ──────────────────────────────────────────────────


def _resolve_engine(engine: str) -> str:
    """Map ``"auto"`` to the best available engine; validate explicit choices."""
    from histo_to_ccf.registration.elastix_bspline import ELASTIX_AVAILABLE

    if engine == "auto":
        return "elastix" if ELASTIX_AVAILABLE else "sitk"
    if engine == "elastix" and not ELASTIX_AVAILABLE:
        raise RuntimeError(
            "engine='elastix' requested but itk-elastix is not installed "
            "(install the 'elastix' extra)"
        )
    return engine


def _refine(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    engine: str,
    bspline_grid: tuple[int, int],
    max_iterations: int,
    bending_weight: float,
    use_masks: bool,
    moving_mask: np.ndarray | None = None,
):
    """Run the chosen refinement engine, returning a ``RegisterResult``."""
    resolved = _resolve_engine(engine)
    if resolved == "elastix":
        from histo_to_ccf.registration.elastix_bspline import refine_with_elastix

        return refine_with_elastix(
            reference,
            moving,
            grid_size=bspline_grid,
            bending_weight=bending_weight,
            max_iterations=max_iterations,
            use_masks=use_masks,
            moving_mask=moving_mask,
        )
    return refine_with_bspline(
        reference,
        moving,
        grid_size=bspline_grid,
        max_iterations=max_iterations,
    )


def register_section_image(
    section_image: np.ndarray,
    atlas: "BrainGlobeAtlas",
    *,
    anchoring: Anchoring,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
    reference_volume: np.ndarray | None = None,
    engine: str = "auto",
    bending_weight: float = 20.0,
    use_masks: bool = True,
) -> tuple[RegistrationResult, "object"]:
    """Run the M3 registration on one section.

    Returns the persistable :class:`RegistrationResult` plus the in-memory
    SimpleITK transform (caller writes it to a sidecar if desired).

    ``reference_volume`` lets the caller pass ``atlas.reference`` directly (the
    raw uint16 volume, no copy); the slice is interpolated into float32 on the
    fly. Otherwise the full volume is cast and the annotation needlessly
    resampled on every call, churning hundreds of MB per section.

    ``engine`` selects the refinement backend:

    - ``"elastix"`` — masked, bending-energy-regularized B-spline (ABBA-style;
      needs the ``itk-elastix`` extra).
    - ``"sitk"`` — the plain SimpleITK B-spline.
    - ``"auto"`` — elastix when installed, else SimpleITK.

    ``bending_weight`` and ``use_masks`` only apply to the elastix engine.
    """
    h, w = section_image.shape[:2]
    out_shape = (int(h), int(w))
    if reference_volume is not None:
        from histo_to_ccf.atlas.planes import sample_plane

        reference = sample_plane(
            reference_volume, anchoring, out_shape, order=1, out_dtype=np.float32
        )
    else:
        reference, _annot = resample_atlas_at_plane(atlas, anchoring, out_shape)

    # A plane that lands (almost) entirely outside the brain yields a constant
    # reference; the MI metric then can't establish overlap and ITK aborts with
    # a cryptic error. Fail early with an actionable message instead.
    if float(reference.max()) - float(reference.min()) < 1e-6:
        raise ValueError(
            "atlas plane is empty (no tissue overlap) — check the section's "
            "AP/anchoring"
        )

    moving = section_image
    moving_mask = None
    if moving.ndim == 3:
        # Build the metric mask from the RGB crop BEFORE collapsing to luminance,
        # so the bright fluorescent labels can be excluded (they have no atlas
        # counterpart and otherwise pull the fit).
        if use_masks:
            from histo_to_ccf.registration.masks import registration_moving_mask

            moving_mask = registration_moving_mask(section_image)
        # Use luminance for registration; preserves brain outline.
        moving = moving[..., :3].astype(np.float32).mean(axis=-1)

    result = _refine(
        reference,
        moving.astype(np.float32),
        engine=engine,
        bspline_grid=bspline_grid,
        max_iterations=max_iterations,
        bending_weight=bending_weight,
        use_masks=use_masks,
        moving_mask=moving_mask,
    )

    reg = RegistrationResult(
        anchoring=list(anchoring.as_tuple()),
        output_size_px=(out_shape[0], out_shape[1]),
        bspline_transform_path=None,
        residual=result.residual_rms,
    )
    return reg, result.transform


def register_project_with_atlas(
    project: Project,
    atlas: "BrainGlobeAtlas",
    *,
    section_images: dict[int, np.ndarray],
    transforms_dir: Path,
    bspline_grid: tuple[int, int] = (8, 8),
    max_iterations: int = 100,
    engine: str = "auto",
    bending_weight: float = 20.0,
    use_masks: bool = True,
) -> Project:
    """Drive the full M3 pipeline across every section in ``project``.

    ``section_images`` maps section index → preprocessed section image array
    (grayscale or RGB). ``transforms_dir`` is where .tfm sidecars get written
    (relative paths stored in the project).
    """
    transforms_dir = Path(transforms_dir)
    transforms_dir.mkdir(parents=True, exist_ok=True)
    res_um = atlas_resolution_um(atlas)
    # Pass the raw (uint16) reference volume — slices are interpolated to float
    # per section, so we never copy the whole volume to float32.
    ref_vol = atlas.reference

    registered: dict[tuple[int, int], RegisteredSectionTransform] = {}
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            img = section_images.get(section.index)
            if img is None or section.plane is None:
                logger.warning(
                    "skipping section {} (missing image or plane)", section.index
                )
                continue
            anchoring = anchoring_from_plane_params(atlas, section.plane)
            reg, sitk_transform = register_section_image(
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

            import SimpleITK as sitk

            tfm_path = transforms_dir / f"section_{section.index:03d}.h5"
            sitk.WriteTransform(sitk_transform, str(tfm_path))
            reg.bspline_transform_path = str(tfm_path.relative_to(transforms_dir.parent))
            section.registration = reg

            registered[(slide_idx, section.index)] = RegisteredSectionTransform(
                anchoring=anchoring,
                output_size_px=reg.output_size_px,
                bspline=sitk_transform,
                atlas_resolution_um=res_um,
            )

    for probe in project.probes:
        for shank in probe.shanks:
            _apply_to_shank_registered(shank, project, registered)
    return project


def _apply_to_shank_registered(
    shank: Shank,
    project: Project,
    transforms: dict[tuple[int, int], RegisteredSectionTransform],
) -> None:
    def to_ccf(point, section_idx: int) -> tuple[float, float, float] | None:
        for slide_idx, slide in enumerate(project.slides):
            for section in slide.sections:
                if section.index != section_idx:
                    continue
                tx = transforms.get((slide_idx, section.index))
                if tx is None:
                    return None
                # Clicked points are in slide-global pixels, but the transform is
                # defined on the section crop — convert to section-local coords
                # by subtracting the bbox origin.
                x0, y0 = section.bbox_px[0], section.bbox_px[1]
                return tx.apply(point.x_px - x0, point.y_px - y0)
        return None

    if shank.tip_px is not None and shank.tip_section_idx is not None:
        ccf = to_ccf(shank.tip_px, shank.tip_section_idx)
        if ccf is not None:
            shank.tip_ccf_um = ccf
    if shank.entry_px is not None and shank.entry_section_idx is not None:
        ccf = to_ccf(shank.entry_px, shank.entry_section_idx)
        if ccf is not None:
            shank.entry_ccf_um = ccf


def reload_registered_transforms(
    project: Project,
    atlas: "BrainGlobeAtlas",
    *,
    project_dir: Path | None = None,
) -> dict[tuple[int, int], RegisteredSectionTransform]:
    """Rebuild RegisteredSectionTransforms from persisted RegistrationResult."""
    out: dict[tuple[int, int], RegisteredSectionTransform] = {}
    for slide_idx, slide in enumerate(project.slides):
        for section in slide.sections:
            if section.registration is None:
                continue
            out[(slide_idx, section.index)] = build_registered_transform(
                section.registration, atlas, project_dir=project_dir,
                manual_affine=section.manual_affine,
            )
    return out
