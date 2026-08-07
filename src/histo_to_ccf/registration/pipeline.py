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


def anchoring_for_section(section: Section, anchorings: dict, atlas: "BrainGlobeAtlas"):
    """Pick a section's atlas plane: a (guided) DeepSlice anchoring, else the plane.

    When DeepSlice ran, ``anchorings`` holds its per-section prediction - already
    **guided** by any user-assigned AP (see :func:`guide_anchorings_with_planes`),
    so it is used directly. Sections DeepSlice didn't cover fall back to the
    hand-assigned plane.
    """
    if section.index in anchorings:
        return Anchoring.from_iterable(anchorings[section.index])
    return anchoring_from_plane_params(atlas, section.plane)


def _ap_center(anchoring9: "list[float] | tuple[float, ...]") -> float:
    """AP voxel coordinate of a plane's centre (su = sv = 0.5): ox + ½ux + ½vx."""
    return float(anchoring9[0]) + 0.5 * float(anchoring9[3]) + 0.5 * float(anchoring9[6])


def prematch_ap_order_issues(
    aps: "list[tuple[int, float]]", *, close_frac: float = 0.3
) -> "tuple[list[tuple[int, int]], list[tuple[int, int]]]":
    """Flag a DeepSlice pre-match's AP series as out-of-order or too-close.

    ``aps`` is ``[(section_index, ap_um), ...]`` already in the user's section
    order. Returns ``(reversed_pairs, close_pairs)`` of adjacent
    ``(index, index)`` whose AP step reverses the overall direction, or collapses
    below ``close_frac`` of the median |step|. DeepSlice only enforces *order*,
    not spacing, so two sections can land almost on top of each other - this is
    the headless core of the matcher's post-pre-match warning. Empty lists when
    there is nothing to flag (or fewer than 3 sections to judge).
    """
    from itertools import pairwise

    if len(aps) < 3:
        return [], []
    pairs = list(pairwise(aps))
    steps = [b_ap - a_ap for (_, a_ap), (_, b_ap) in pairs]
    med = float(np.median([abs(s) for s in steps]))
    if med < 1e-6:
        return [], []
    sign = 1.0 if (aps[-1][1] - aps[0][1]) >= 0 else -1.0
    reversed_pairs: list[tuple[int, int]] = []
    close_pairs: list[tuple[int, int]] = []
    for ((a_idx, _), (b_idx, _)), step in zip(pairs, steps, strict=True):
        if step * sign <= 0:
            reversed_pairs.append((a_idx, b_idx))
        elif abs(step) < close_frac * med:
            close_pairs.append((a_idx, b_idx))
    return reversed_pairs, close_pairs


def anchoring_center_ap_um(
    anchoring9: "list[float] | tuple[float, ...]", ap_res_um: float
) -> float:
    """Absolute AP (µm) of a plane's centre, given the atlas AP voxel size.

    The inverse of how :func:`coronal_anchoring` places a plane at ``ap_um``: it
    turns a DeepSlice anchoring (atlas-voxel ASR frame) into the scalar AP the
    Atlas matcher / ``PlaneParams.ap_um`` expect.
    """
    return _ap_center(anchoring9) * float(ap_res_um)


def guide_anchorings_with_planes(
    anchorings: dict, project: Project, atlas: "BrainGlobeAtlas"
) -> dict:
    """Anchor DeepSlice's predicted AP to the user's hand-assigned AP values.

    DeepSlice predicts each section's full plane (AP + tilt). For every section the
    user assigned an AP (Atlas tab), the plane is translated along AP so its centre
    sits **exactly** at that AP - keeping DeepSlice's tilt. Sections the user left
    unassigned are shifted by **interpolating** the assigned sections' shifts (as a
    function of DeepSlice's predicted AP), so the whole series follows the same
    correction. This guarantees a pinned section lands on its value while still
    letting DeepSlice place the rest. Returns the dict unchanged when there are no
    manual anchors.
    """
    sec_by_idx = {
        section.index: section
        for slide in project.slides
        for section in slide.sections
    }

    def user_center(section) -> float:
        return _ap_center(anchoring_from_plane_params(atlas, section.plane).as_tuple())

    # Per assigned section: (DeepSlice AP centre, exact shift onto the user's AP).
    anchors: list[tuple[float, float]] = []
    for idx, anch in anchorings.items():
        section = sec_by_idx.get(idx)
        if section is not None and section.plane is not None:
            anchors.append((_ap_center(anch), user_center(section) - _ap_center(anch)))
    if not anchors:
        return anchorings
    anchors.sort()
    ds_list = [a[0] for a in anchors]
    shift_list = [a[1] for a in anchors]

    corrected: dict = {}
    for idx, anch in anchorings.items():
        section = sec_by_idx.get(idx)
        if section is not None and section.plane is not None:
            shift = user_center(section) - _ap_center(anch)  # exact
        else:
            shift = float(np.interp(_ap_center(anch), ds_list, shift_list))
        new = list(anch)
        new[0] = float(anch[0]) + shift  # shift the AP origin (ox); tilt unchanged
        corrected[idx] = new
    return corrected


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
    prealign: bool = True,
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
            prealign=prealign,
        )
    return refine_with_bspline(
        reference,
        moving,
        grid_size=bspline_grid,
        max_iterations=max_iterations,
    )


# What elastix actually raises. The useful detail - "Too many samples map
# outside moving image buffer: 101 / 2048" - goes to elastix's own log, and the
# Python exception carries only the generic line, so that is what we match. The
# specific markers are kept in case a future itk-elastix propagates the detail.
_RETRYABLE_ELASTIX_MARKERS = (
    "internal elastix error",
    "map outside moving image buffer",
    "too many samples",
)


def _is_sample_coverage_failure(exc: BaseException) -> bool:
    """Is this an elastix abort that an unmasked retry can rescue?

    Deliberately narrow in effect rather than in matching: the caller only ever
    consults this when masks were **on**, and the retry simply turns them off.
    A failure that is not about the mask fails again and propagates.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_ELASTIX_MARKERS)


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
    prealign: bool = True,
    boundary_snap: bool = True,
) -> tuple[RegistrationResult, "object"]:
    """Run the M3 registration on one section.

    Returns the persistable :class:`RegistrationResult` plus the in-memory
    SimpleITK transform (caller writes it to a sidecar if desired).

    ``reference_volume`` lets the caller pass ``atlas.reference`` directly (the
    raw uint16 volume, no copy); the slice is interpolated into float32 on the
    fly. Otherwise the full volume is cast and the annotation needlessly
    resampled on every call, churning hundreds of MB per section.

    ``engine`` selects the refinement backend:

    - ``"elastix"`` - masked, bending-energy-regularized B-spline (ABBA-style;
      needs the ``itk-elastix`` extra).
    - ``"sitk"`` - the plain SimpleITK B-spline.
    - ``"auto"`` - elastix when installed, else SimpleITK.

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
            "atlas plane is empty (no tissue overlap) - check the section's "
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

    refine_kwargs = dict(
        engine=engine,
        bspline_grid=bspline_grid,
        max_iterations=max_iterations,
        bending_weight=bending_weight,
        prealign=prealign,
    )
    try:
        result = _refine(
            reference,
            moving.astype(np.float32),
            use_masks=use_masks,
            moving_mask=moving_mask,
            **refine_kwargs,
        )
    except Exception as exc:
        # elastix aborts the whole section when too few metric samples stay
        # inside the moving mask. On a low-contrast section (lightsheet renders,
        # faint periphery) the tissue mask under-covers the brain - measured at
        # ~42% of the crop against the atlas mask's ~73% on LO_03 - and the
        # valid-sample ratio collapses to ~5%, right on elastix's limit. Every
        # section below that line failed and every one above it passed.
        #
        # The mask is an accuracy aid (it excludes fluorescent labels), not a
        # requirement, so a section it sinks is better registered without it
        # than not at all. Retry once, unmasked, and let the caller say so.
        if not (use_masks and _is_sample_coverage_failure(exc)):
            raise
        logger.warning(
            "masked registration failed ({}); retrying without the tissue mask",
            str(exc).splitlines()[-1][:160],
        )
        result = _refine(
            reference,
            moving.astype(np.float32),
            use_masks=False,
            moving_mask=None,
            **refine_kwargs,
        )
        result.used_mask_fallback = True

    transform = result.transform
    if boundary_snap:
        transform = _apply_boundary_snap(
            transform, reference, section_image, out_shape
        )

    reg = RegistrationResult(
        anchoring=list(anchoring.as_tuple()),
        output_size_px=(out_shape[0], out_shape[1]),
        bspline_transform_path=None,
        residual=result.residual_rms,
        used_mask_fallback=getattr(result, "used_mask_fallback", False),
    )
    return reg, transform


def _apply_boundary_snap(
    transform: "object",
    reference: np.ndarray,
    section_image: np.ndarray,
    out_shape: tuple[int, int],
) -> "object":
    """Compose the outer-contour snap onto ``transform`` (no-op if it can't help).

    Snaps the warped atlas silhouette onto the section tissue silhouette - the
    automatic equivalent of the manual landmark drag. Returns the original
    transform unchanged when the snap is degenerate or would fold (see
    :mod:`registration.boundary_snap`). One failed snap must never break a
    section's registration, so any error is swallowed and the un-snapped
    transform is kept.
    """
    try:
        from histo_to_ccf.registration.boundary_snap import (
            boundary_snap_transform,
            compose_snap,
        )
        from histo_to_ccf.registration.masks import section_tissue_mask
        from histo_to_ccf.registration.transforms import _warped_atlas_extent

        ref = np.asarray(reference, dtype=np.float32)
        lo, hi = float(ref.min()), float(ref.max())
        if hi - lo < 1e-6:
            return transform
        atlas_fg = (ref - lo) / (hi - lo) > 0.02
        extent = _warped_atlas_extent(transform, atlas_fg, ref.shape, out_shape)
        tissue = section_tissue_mask(section_image)
        snap = boundary_snap_transform(extent, tissue)
        if snap is None:
            return transform
        return compose_snap(transform, snap)
    except Exception as exc:  # noqa: BLE001 - snap is best-effort; keep registration
        logger.warning("boundary snap skipped: {}", exc)
        return transform


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
    prealign: bool = True,
    boundary_snap: bool = True,
    reuse_stored_anchoring: bool = False,
) -> Project:
    """Drive the full M3 pipeline across every section in ``project``.

    ``section_images`` maps section index → preprocessed section image array
    (grayscale or RGB). ``transforms_dir`` is where .tfm sidecars get written
    (relative paths stored in the project).

    ``reuse_stored_anchoring`` keeps each section's already-registered atlas plane
    (``section.registration.anchoring``) instead of rebuilding one from its
    ``PlaneParams``. ``PlaneParams`` can only express a coronal plane plus two
    tilts, so rebuilding discards the oblique plane DeepSlice predicted - which is
    why a re-run could silently move sections. Leave it off when the plane's AP or
    tilt was edited by hand and should take effect.
    """
    transforms_dir = Path(transforms_dir)
    transforms_dir.mkdir(parents=True, exist_ok=True)
    res_um = atlas_resolution_um(atlas)
    # Pass the raw (uint16) reference volume - slices are interpolated to float
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
            stored = (
                section.registration.anchoring
                if reuse_stored_anchoring and section.registration is not None
                else None
            )
            if stored is not None:
                anchoring = Anchoring.from_iterable(stored)
                logger.debug("section {}: re-using stored anchoring", section.index)
            else:
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
                prealign=prealign,
                boundary_snap=boundary_snap,
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
                # defined on the section crop - convert to section-local coords
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
                manual_landmarks=section.manual_landmarks,
            )
    return out
