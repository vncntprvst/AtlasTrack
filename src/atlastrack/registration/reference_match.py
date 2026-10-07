"""Register sections by matching them to a registered project of the same sections.

The same tissue sections are often imaged twice: at high resolution for cells (a
confocal, say) and on a slide scanner for probe tracks. Once one set is registered
to the atlas, the other needs no atlas registration of its own - only a match to
its counterpart, which is a far easier problem: same tissue, same outline,
ventricles and cell clusters, differing by scale, a turn and perhaps a mirror.

1. :func:`pair_sections` pairs each section with its counterpart by the shape of
   its tissue outline (allowing for a mirror and a small turn).
2. :func:`fit_pair` finds the transform between a pair: a start from the outlines,
   then refined on the stain with mutual information (the stains need not look
   alike).
3. :func:`inherit_registration` gives a section its counterpart's registration
   through that transform: the plane, the warp and the landmark corrections, all
   in one, so atlas coordinates on the two sets agree.

Everything here works on section images (``(H, W)`` or ``(H, W, 3)``) in each
section's own pixels, after its flips and rotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atlastrack.project.schema import Project, RegistrationResult, Section

#: Side of the square the outlines are compared at when pairing.
_SHAPE_PX = 128
#: Longest side of the counterpart image when fitting.
_FIT_PX = 640
#: Pairs whose outlines overlap less than this are left unpaired.
MIN_SHAPE_OVERLAP = 0.75
#: The inherited warp is stored on a grid this many pixels apart.
WARP_STEP_PX = 8


def stain_image(image: np.ndarray, align_channel: str | None) -> np.ndarray:
    """The channel showing the tissue (``align_channel``), else the mean, as float."""
    from atlastrack.project.images import align_channel_image

    img = align_channel_image(np.asarray(image), align_channel)
    if img.ndim == 3:
        img = img[..., :3].astype(np.float32).mean(axis=-1)
    return img.astype(np.float32)


def _shrink(image: np.ndarray, longest: int) -> tuple[np.ndarray, tuple[float, float]]:
    """Resize so the longest side is ``longest``; returns it and (fx, fy) full/small."""
    from skimage.transform import resize

    h, w = image.shape[:2]
    k = max(h, w) / float(longest)
    if k <= 1:
        return image.astype(np.float32), (1.0, 1.0)
    shape = (max(1, round(h / k)), max(1, round(w / k)))
    small = resize(image, shape, order=1, anti_aliasing=True, preserve_range=True)
    return small.astype(np.float32), (w / shape[1], h / shape[0])


def tissue_mask(stain: np.ndarray) -> np.ndarray:
    from atlastrack.registration.masks import section_tissue_mask

    return section_tissue_mask(stain)


def _normalised_outline(mask: np.ndarray) -> np.ndarray:
    """The outline cropped to its box, centred in a square, at a fixed size."""
    from skimage.transform import resize

    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((_SHAPE_PX, _SHAPE_PX), bool)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = max(h, w)
    square = np.zeros((s, s), bool)
    square[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    return resize(square.astype(np.float32), (_SHAPE_PX, _SHAPE_PX), order=1) > 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass
class Outline:
    """A section's tissue outline, for pairing."""

    shape: np.ndarray  # normalised, _SHAPE_PX square

    @classmethod
    def of(cls, image: np.ndarray, align_channel: str | None) -> Outline:
        small, _ = _shrink(stain_image(image, align_channel), 4 * _SHAPE_PX)
        return cls(_normalised_outline(tissue_mask(small)))


def shape_overlap(target: Outline, reference: Outline) -> tuple[float, bool, float]:
    """Best outline overlap ``(overlap, mirrored, turn_deg)`` of target onto reference.

    ``mirrored`` means the target matches once flipped left-right; ``turn_deg``
    is the turn (anticlockwise, after the flip) that matched best.
    """
    from skimage.transform import rotate

    best = (0.0, False, 0.0)
    for mirrored in (False, True):
        shape = target.shape[:, ::-1] if mirrored else target.shape
        for turn in range(-12, 13, 3):
            turned = shape if turn == 0 else rotate(shape.astype(float), turn, order=0) > 0.5
            overlap = _iou(turned, reference.shape)
            if overlap > best[0]:
                best = (overlap, mirrored, float(turn))
    return best


@dataclass
class Pairing:
    target_index: int
    reference_index: int
    overlap: float
    mirrored: bool
    turn_deg: float
    runner_up: float  # the best overlap with any other reference section


def pair_sections(
    targets: dict[int, Outline],
    references: dict[int, Outline],
    *,
    min_overlap: float = MIN_SHAPE_OVERLAP,
) -> list[Pairing]:
    """Pair target sections with reference sections, one to one, by outline.

    The pairing maximises the total outline overlap; pairs below ``min_overlap``
    are dropped (a target section with no counterpart stays unpaired).
    """
    from scipy.optimize import linear_sum_assignment

    t_ids, r_ids = list(targets), list(references)
    if not t_ids or not r_ids:
        return []
    scores = np.zeros((len(t_ids), len(r_ids)))
    detail: dict[tuple[int, int], tuple[float, bool, float]] = {}
    for i, t in enumerate(t_ids):
        for j, r in enumerate(r_ids):
            detail[(i, j)] = shape_overlap(targets[t], references[r])
            scores[i, j] = detail[(i, j)][0]
    rows, cols = linear_sum_assignment(-scores)
    out = []
    for i, j in zip(rows, cols, strict=True):
        overlap, mirrored, turn = detail[(i, j)]
        if overlap < min_overlap:
            continue
        others = np.delete(scores[i], j)
        out.append(Pairing(
            target_index=t_ids[i], reference_index=r_ids[j], overlap=overlap,
            mirrored=mirrored, turn_deg=turn,
            runner_up=float(others.max()) if others.size else 0.0,
        ))
    out.sort(key=lambda p: p.target_index)
    return out


# --------------------------------------------------------------------------
# Fitting a pair
# --------------------------------------------------------------------------


def _scale_matrix(fx: float, fy: float) -> np.ndarray:
    """Small-image pixel -> full-image pixel, pixel centres kept."""
    return np.array([[fx, 0, 0.5 * fx - 0.5], [0, fy, 0.5 * fy - 0.5], [0, 0, 1.0]])


def _sitk_matrix(transform) -> np.ndarray:
    """3x3 (x, y) matrix of a SimpleITK 2-D similarity/affine transform."""
    a = np.asarray(transform.GetMatrix(), dtype=float).reshape(2, 2)
    c = np.asarray(transform.GetCenter(), dtype=float)
    t = np.asarray(transform.GetTranslation(), dtype=float)
    m = np.eye(3)
    m[:2, :2] = a
    m[:2, 2] = c + t - a @ c
    return m


def _mask_overlap_under(m_ref_to_tgt: np.ndarray, ref_mask, tgt_mask) -> float:
    """Overlap of the outlines when the target is pulled onto the reference."""
    from scipy.ndimage import map_coordinates

    h, w = ref_mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    pts = m_ref_to_tgt @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    pulled = map_coordinates(tgt_mask.astype(np.float32), [pts[1], pts[0]], order=0, cval=0)
    return _iou(pulled.reshape(h, w) > 0.5, ref_mask)


@dataclass
class Fit:
    matrix: np.ndarray  # 3x3: target pixel (after the mirror) -> reference pixel
    overlap: float  # outline overlap after the fit


def fit_pair(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    mirrored: bool,
    turn_deg: float = 0.0,
    target_channel: str | None = None,
    reference_channel: str | None = None,
) -> Fit:
    """Transform taking ``target`` pixels onto ``reference`` pixels.

    ``target`` is taken as already flipped when ``mirrored`` (the caller flips the
    section). The fit starts from the outlines - centres, sizes and the pairing's
    turn - then refines a similarity and an affine transform on the stain with
    mutual information, keeping whichever leaves the outlines overlapping more.
    """
    import SimpleITK as sitk

    t_stain = stain_image(target, target_channel)
    if mirrored:
        t_stain = t_stain[:, ::-1]
    r_small, (rfx, rfy) = _shrink(stain_image(reference, reference_channel), _FIT_PX)
    r_mask = tissue_mask(r_small)
    # Bring the target to the reference's working scale, judged by tissue area.
    t_probe, _ = _shrink(t_stain, _FIT_PX)
    t_probe_mask = tissue_mask(t_probe)
    scale = np.sqrt(r_mask.sum() / max(t_probe_mask.sum(), 1))
    t_small, (tfx, tfy) = _shrink(t_stain, max(t_probe.shape) * scale)
    t_mask = tissue_mask(t_small)

    def centroid(mask):
        ys, xs = np.nonzero(mask)
        return np.array([xs.mean(), ys.mean()])

    c_ref, c_tgt = centroid(r_mask), centroid(t_mask)
    fixed = sitk.GetImageFromArray(r_small / max(float(r_small.max()), 1e-6))
    moving = sitk.GetImageFromArray(t_small / max(float(t_small.max()), 1e-6))
    from scipy.ndimage import binary_dilation

    fixed_mask = sitk.GetImageFromArray(
        binary_dilation(r_mask, iterations=max(2, r_mask.shape[0] // 60)).astype(np.uint8)
    )

    def register(initial):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=48)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.3, seed=7)
        reg.SetMetricFixedMask(fixed_mask)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=300,
            relaxationFactor=0.6, gradientMagnitudeTolerance=1e-8,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        reg.SetInitialTransform(initial, inPlace=False)
        return reg.Execute(fixed, moving)

    # Fixed = reference, moving = target: the transform maps reference -> target.
    start = sitk.Similarity2DTransform()
    # Whatever size difference the resampling left (it never enlarges), from areas.
    start.SetScale(float(np.sqrt(max(t_mask.sum(), 1) / max(r_mask.sum(), 1))))
    start.SetCenter(tuple(float(v) for v in c_ref))
    start.SetAngle(np.deg2rad(-turn_deg))
    start.SetTranslation(tuple(float(v) for v in c_tgt - c_ref))
    candidates = [_sitk_matrix(start)]
    try:
        # Execute hands back a one-transform composite; unwrap it to its type.
        similar = sitk.Similarity2DTransform(
            sitk.CompositeTransform(register(start)).GetNthTransform(0)
        )
        candidates.append(_sitk_matrix(similar))
        affine = sitk.AffineTransform(2)
        affine.SetCenter(similar.GetCenter())
        affine.SetMatrix(similar.GetMatrix())
        affine.SetTranslation(similar.GetTranslation())
        fitted = sitk.AffineTransform(sitk.CompositeTransform(register(affine)).GetNthTransform(0))
        candidates.append(_sitk_matrix(fitted))
    except RuntimeError:  # the stain gave the optimiser nothing to work with
        pass

    # Keep whichever leaves the outlines overlapping most (ties: the simpler one).
    scored = [(_mask_overlap_under(m, r_mask, t_mask), -k, m) for k, m in enumerate(candidates)]
    overlap, _, m_small = max(scored, key=lambda s: (round(s[0], 4), s[1]))
    # m_small maps reference small -> target small.
    full = _scale_matrix(tfx, tfy) @ m_small @ np.linalg.inv(_scale_matrix(rfx, rfy))
    return Fit(matrix=np.linalg.inv(full), overlap=overlap)


# --------------------------------------------------------------------------
# Inheriting the counterpart's registration
# --------------------------------------------------------------------------


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts @ m[:2, :2].T + m[:2, 2]


class _ReferenceChain:
    """A registered section's map between its atlas-slice pixels and its pixels."""

    def __init__(self, section: Section, project_dir: Path):
        reg = section.registration
        assert reg is not None
        self.size = tuple(reg.output_size_px)  # (h, w)
        self.warp = self.unwarp = None
        if reg.bspline_transform_path:
            import SimpleITK as sitk

            from atlastrack.registration.transforms import (
                _invert_displacement,
                inverse_warp_path,
            )

            path = project_dir / reg.bspline_transform_path
            forward = sitk.ReadTransform(str(path))
            inverse_file = inverse_warp_path(path)
            if inverse_file.is_file():
                inverse = sitk.ReadTransform(str(inverse_file))
            else:
                try:
                    inverse = forward.GetInverse()
                except RuntimeError:
                    inverse = _invert_displacement(forward, self.size)
            self.warp, self.unwarp = self._sampled(forward), self._sampled(inverse)
        # The hand correction on top: landmarks win over the box affine, as in
        # RegisteredSectionTransform.
        lm = section.manual_landmarks
        self.source = self.target = self.affine = None
        self.forward = bool(lm.forward) if lm is not None else False
        if lm is not None:
            self.source = np.asarray(lm.source, float)
            self.target = np.asarray(lm.target, float)
        elif section.manual_affine is not None:
            swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])  # (row, col) <-> (x, y)
            self.affine = swap @ np.asarray(section.manual_affine, float) @ swap

    def _sampled(self, transform):
        """A transform as a fast lookup over the section grid (+ margin)."""
        import SimpleITK as sitk
        from scipy.ndimage import map_coordinates

        h, w = self.size
        pad = 0.25
        x0, y0 = -pad * w, -pad * h
        step = max(1.0, max(h, w) / 1024)
        nx, ny = int((1 + 2 * pad) * w / step) + 2, int((1 + 2 * pad) * h / step) + 2
        field = sitk.TransformToDisplacementField(
            transform, sitk.sitkVectorFloat64, (nx, ny), (x0, y0), (step, step)
        )
        arr = sitk.GetArrayFromImage(field)  # (ny, nx, 2)

        def apply(pts: np.ndarray) -> np.ndarray:
            cx = (pts[:, 0] - x0) / step
            cy = (pts[:, 1] - y0) / step
            dx = map_coordinates(arr[..., 0], [cy, cx], order=1, mode="nearest")
            dy = map_coordinates(arr[..., 1], [cy, cx], order=1, mode="nearest")
            return pts + np.c_[dx, dy]

        return apply

    def slice_to_section(self, pts: np.ndarray) -> np.ndarray:
        """Atlas-slice pixel -> section pixel, as the overlay draws it."""
        from atlastrack.registration.landmarks_warp import warp_points

        out = self.warp(pts) if self.warp is not None else pts
        if self.source is not None:
            out = warp_points(self.source, self.target, out, forward=self.forward)
        elif self.affine is not None:
            out = _apply(self.affine, out)
        return out

    def section_to_slice(self, pts: np.ndarray) -> np.ndarray:
        """Section pixel -> atlas-slice pixel, as points are mapped."""
        from atlastrack.registration.landmarks_warp import invert_points

        out = pts
        if self.source is not None:
            out = invert_points(self.source, self.target, out, forward=self.forward)
        elif self.affine is not None:
            out = _apply(np.linalg.inv(self.affine), out)
        return self.unwarp(out) if self.unwarp is not None else out

    @property
    def bends(self) -> bool:
        return self.warp is not None or self.source is not None or self.affine is not None


def inherit_registration(
    section: Section,
    reference: Section,
    reference_dir: Path,
    matrix: np.ndarray,
    project_path: Path,
    *,
    step_px: int = WARP_STEP_PX,
) -> RegistrationResult:
    """``section``'s registration: ``reference``'s, through ``matrix``.

    ``matrix`` maps ``section`` pixels onto ``reference`` pixels. The plane goes
    into the anchoring, exactly (both maps are linear); the counterpart's warp and
    landmark corrections become this section's warp file, stored with its exact
    inverse in the project's ``<name>_transforms`` folder.
    """
    from atlastrack.io.section_import import write_warp
    from atlastrack.project.schema import RegistrationResult

    ref_reg = reference.registration
    if ref_reg is None:
        raise ValueError(f"reference section {reference.index} is not registered")
    x0, y0, x1, y1 = section.bbox_px
    w, h = x1 - x0, y1 - y0
    rh, rw = ref_reg.output_size_px
    a = np.asarray(ref_reg.anchoring, float)
    o, u, v = a[:3], a[3:6], a[6:]

    def voxel(t_px: np.ndarray) -> np.ndarray:
        r = _apply(matrix, t_px)
        return o + np.outer(r[:, 0] / rw, u) + np.outer(r[:, 1] / rh, v)

    corners = voxel(np.array([[0.0, 0.0], [w, 0.0], [0.0, h]]))
    anchoring = [*corners[0], *(corners[1] - corners[0]), *(corners[2] - corners[0])]
    result = RegistrationResult(anchoring=[float(x) for x in anchoring], output_size_px=(h, w))

    chain = _ReferenceChain(reference, reference_dir)
    if not chain.bends:
        return result
    inv = np.linalg.inv(matrix)
    nx, ny = int(np.ceil(w / step_px)) + 1, int(np.ceil(h / step_px)) + 1
    gy, gx = np.mgrid[0:ny, 0:nx].astype(float) * step_px
    grid = np.c_[gx.ravel(), gy.ravel()]
    # Forward: this section's slice pixel -> reference slice -> reference pixel -> here.
    forward = _apply(inv, chain.slice_to_section(_apply(matrix, grid))) - grid
    # Inverse: this section's pixel -> reference pixel -> reference slice -> here.
    backward = _apply(inv, chain.section_to_slice(_apply(matrix, grid))) - grid
    from atlastrack.project.section_files import transforms_folder

    project_path = Path(project_path)
    rel = f"{transforms_folder(project_path)}/section_{section.index:03d}.h5"
    write_warp(
        project_path.parent / rel,
        forward.reshape(ny, nx, 2),
        backward.reshape(ny, nx, 2),
        spacing_px=step_px,
    )
    result.bspline_transform_path = rel
    return result


# --------------------------------------------------------------------------
# Whole projects
# --------------------------------------------------------------------------


@dataclass
class SectionMatch:
    target_index: int
    reference_index: int
    reference_label: int | None
    mirrored: bool
    shape_overlap: float
    runner_up: float
    fit: Fit | None = None


def _section_pixels(project: Project, section: Section, project_dir: Path,
                    slide_images: dict | None) -> np.ndarray:
    """A section's pixels as registration sees them."""
    from atlastrack.io.image import load_image
    from atlastrack.project.images import rebuild_slide_image

    image = None if slide_images is None else slide_images.get(section.slide_idx)
    if image is None and section.image_path and (project_dir / section.image_path).is_file():
        return load_image(project_dir / section.image_path)
    if image is None:
        image, _ = rebuild_slide_image(project.slides[section.slide_idx], base_dir=project_dir)
        if slide_images is not None:
            slide_images[section.slide_idx] = image
    x0, y0, x1, y1 = section.bbox_px
    return np.asarray(image)[y0:y1, x0:x1]


def match_projects(
    target: Project,
    target_dir: Path,
    reference: Project,
    reference_dir: Path,
    *,
    slide_images: dict | None = None,
    progress=None,
) -> list[SectionMatch]:
    """Pair ``target``'s sections with ``reference``'s registered ones and fit each pair.

    ``slide_images`` are the target's slides as the app holds them (else they are
    rebuilt from disk). ``progress(done, total, message)`` is called along the way.
    Nothing in either project is changed - see :func:`apply_matches`.
    """
    def report(done, total, msg):
        if progress is not None:
            progress(done, total, msg)

    t_secs = {s.index: s for sl in target.slides for s in sl.sections}
    r_secs = {s.index: s for sl in reference.slides for s in sl.sections
              if s.registration is not None}
    t_chan = {s.index: target.slides[s.slide_idx].align_channel for s in t_secs.values()}
    r_chan = {s.index: reference.slides[s.slide_idx].align_channel for s in r_secs.values()}
    total = len(t_secs) + len(r_secs)
    t_out, r_out, t_pix, r_pix = {}, {}, {}, {}
    for k, s in enumerate(t_secs.values()):
        report(k, total, f"Reading section {s.index}")
        t_pix[s.index] = _section_pixels(target, s, target_dir, slide_images)
        t_out[s.index] = Outline.of(t_pix[s.index], t_chan[s.index])
    for k, s in enumerate(r_secs.values()):
        report(len(t_secs) + k, total, f"Reading reference section {s.index}")
        r_pix[s.index] = _section_pixels(reference, s, reference_dir, None)
        r_out[s.index] = Outline.of(r_pix[s.index], r_chan[s.index])

    pairs = pair_sections(t_out, r_out)
    from atlastrack.project.orientation import project_view

    # Which face the target's sections must show once matched.
    view = project_view(target)
    matches = []
    for k, p in enumerate(pairs):
        report(k, len(pairs), f"Fitting section {p.target_index} to reference "
               f"{r_secs[p.reference_index].slide_number or p.reference_index}")
        fit = fit_pair(
            t_pix[p.target_index], r_pix[p.reference_index], mirrored=p.mirrored,
            turn_deg=p.turn_deg, target_channel=t_chan[p.target_index],
            reference_channel=r_chan[p.reference_index],
        )
        match = SectionMatch(
            target_index=p.target_index, reference_index=p.reference_index,
            reference_label=r_secs[p.reference_index].slide_number,
            mirrored=p.mirrored, shape_overlap=p.overlap, runner_up=p.runner_up, fit=fit,
        )
        _orient(match, t_secs[p.target_index], r_secs[p.reference_index], view)
        matches.append(match)
    report(len(pairs), len(pairs), "Done")
    return matches


def _orient(m: SectionMatch, section: Section, ref: Section, view: str) -> None:
    """Decide whether the section is flipped: so that it shows the face ``view``.

    The fit is for the section flipped when ``m.mirrored``; it is restated for the
    section as it is, then flipped again only if that shows the other face.
    Matching does not change which face a project's sections show - a project of
    back-face images stays one, whatever the counterpart's images show.
    """
    w = section.bbox_px[2] - section.bbox_px[0]
    flip_x = np.array([[-1.0, 0.0, w - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    as_is = m.fit.matrix @ flip_x if m.mirrored else m.fit.matrix
    ra = np.asarray(ref.registration.anchoring, float)
    rh, rw = ref.registration.output_size_px
    ml_along_x = ra[5] / rw * as_is[0, 0] + ra[8] / rh * as_is[1, 0]
    shows = "front" if ml_along_x >= 0 else "back"
    m.mirrored = shows != view
    m.fit = Fit(matrix=as_is @ flip_x if m.mirrored else as_is, overlap=m.fit.overlap)


def apply_matches(
    target: Project,
    target_path: Path,
    reference: Project,
    reference_path: Path,
    matches: list[SectionMatch],
    *,
    slide_images: dict | None = None,
) -> list[int]:
    """Give each matched section its counterpart's registration; returns their indices.

    ``target_path`` is where the target project is (or will be) saved; the warps
    are written beside it. A section is flipped left-right when that makes it show
    the face the target project's sections are seen from (front or back - see
    :mod:`atlastrack.project.orientation`), in the project and in ``slide_images``.
    Any hand correction of the section is cleared - it belonged to its old fit.
    Cells the reference carries are copied over, placed on the matched sections.
    """
    from atlastrack.project.orientation import project_view
    from atlastrack.project.schema import PlaneParams, ReferenceMatch
    from atlastrack.project.series import plane_values

    reference_dir = Path(reference_path).parent
    t_secs = {s.index: s for sl in target.slides for s in sl.sections}
    r_secs = {s.index: s for sl in reference.slides for s in sl.sections}
    view = project_view(target)
    done = []
    for m in matches:
        if m.fit is None:
            continue
        section, ref = t_secs[m.target_index], r_secs[m.reference_index]
        if m.mirrored:
            section.flip_h = not section.flip_h
            if slide_images is not None and section.slide_idx in slide_images:
                x0, y0, x1, y1 = section.bbox_px
                img = slide_images[section.slide_idx]
                img[y0:y1, x0:x1] = img[y0:y1, x0:x1][:, ::-1].copy()
        reg = inherit_registration(section, ref, reference_dir, m.fit.matrix, target_path)
        section.registration = reg
        section.manual_landmarks = None
        section.manual_affine = None
        section.deepslice_anchoring = None
        values = plane_values(reg.anchoring, reg.output_size_px, target.atlas.resolution_um)
        section.plane = PlaneParams(
            ap_um=values.ap_um, dv_tilt_deg=-values.pitch_deg, ml_tilt_deg=-values.yaw_deg,
        )
        section.ap_source = "reference"
        section.reference = ReferenceMatch(
            project_path=str(Path(reference_path).resolve()),
            section_index=ref.index,
            slide_number=ref.slide_number,
            matrix=m.fit.matrix.tolist(),
            mirrored=m.mirrored,
            overlap=round(m.fit.overlap, 4),
        )
        done.append(section.index)
    if done:
        target.atlas = reference.atlas.model_copy()
        target.seen_from = view
        copy_cells(target, reference, matches)
        _learn_pixel_size(target, reference, matches)
    return done


def _learn_pixel_size(target: Project, reference: Project, matches: list[SectionMatch]) -> None:
    """Give slides of unknown pixel size the one the matches imply.

    The fit's scale times the counterpart's pixel size is this slide's - one
    estimate per section; the median is kept.
    """
    t_slide = {s.index: s.slide_idx for sl in target.slides for s in sl.sections}
    r_pixel = {s.index: reference.slides[s.slide_idx].pixel_um
               for sl in reference.slides for s in sl.sections}
    estimates: dict[int, list[float]] = {}
    for m in matches:
        if m.fit is None or not r_pixel.get(m.reference_index):
            continue
        scale = float(np.sqrt(abs(np.linalg.det(m.fit.matrix[:2, :2]))))
        estimates.setdefault(t_slide[m.target_index], []).append(scale * r_pixel[m.reference_index])
    for slide_idx, values in estimates.items():
        slide = target.slides[slide_idx]
        if slide.pixel_um is None:
            slide.pixel_um = round(float(np.median(values)), 4)


def copy_cells(target: Project, reference: Project, matches: list[SectionMatch]) -> None:
    """Copy the reference's cell sets, placing each cell on its matched section."""
    from atlastrack.project.schema import CellSet

    to_target = {
        m.reference_index: (m.target_index, np.linalg.inv(m.fit.matrix))
        for m in matches if m.fit is not None
    }
    for cs in reference.cell_sets:
        # A re-match replaces the set it copied last time.
        target.cell_sets = [c for c in target.cell_sets if c.name != cs.name]
        cells = []
        for c in cs.cells:
            placed = to_target.get(c.section_index) if c.section_index is not None else None
            if placed is None or c.x_px is None or c.y_px is None:
                # Known in the atlas only: still drawn in 3-D, not on a section.
                cells.append(c.model_copy(update={"section_index": None, "x_px": None,
                                                  "y_px": None}))
                continue
            index, back = placed
            (x, y), = _apply(back, np.array([[c.x_px, c.y_px]]))
            cells.append(c.model_copy(update={"section_index": index,
                                              "x_px": float(x), "y_px": float(y)}))
        target.cell_sets.append(CellSet(name=cs.name, source=cs.source, cells=cells))
