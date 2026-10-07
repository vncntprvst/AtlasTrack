"""Section transforms: map a pixel inside a section to Allen CCF µm.

Two transform modes:

- **Manual** (M1): a :class:`PlaneParams` carries midline/dorsal-surface anchors
  and an AP/tilt-free pixel→CCF computation. Used when DeepSlice or the
  B-spline refinement has not been run.

- **Registered** (M3): a :class:`RegistrationResult` carries a QuickNII
  ``anchoring`` (3D atlas plane) and an optional SimpleITK B-spline transform
  for in-plane refinement. The histology pixel is first mapped through the
  B-spline into the atlas-slice coordinate system, then through the anchoring
  into atlas voxel coordinates, then to CCF µm.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from atlastrack.atlas.planes import Anchoring
from atlastrack.io.ccf_coords import MIDLINE_ML_UM
from atlastrack.project.schema import PlaneParams, RegistrationResult

#: Serialises reads of the ``.h5`` transform sidecars. The HDF5 library bundled
#: inside SimpleITK is built without thread safety: two concurrent ``ReadTransform``
#: calls fail with ``H5Gopen2 failed`` and can leave the library spinning in
#: "infinite loop closing library". Reading is ~17 ms of a ~270 ms warp, so holding
#: this for the read alone costs nothing and lets everything after it run in
#: parallel across sections.
_HDF5_LOCK = threading.Lock()

if TYPE_CHECKING:
    import SimpleITK as sitk
    from brainglobe_atlasapi import BrainGlobeAtlas


@dataclass(frozen=True)
class ManualSectionTransform:
    """The simple M1 pixel→CCF mapping (no atlas resampling)."""

    plane: PlaneParams

    def apply(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        p = self.plane
        dx_px = x_px - p.midline_px
        dy_px = y_px - p.dorsal_surface_px

        ml_offset_um = dx_px * p.pixel_size_um
        if not p.image_right_is_anatomical_right:
            ml_offset_um = -ml_offset_um
        dv_um = dy_px * p.pixel_size_um
        ml_ccf = MIDLINE_ML_UM + ml_offset_um
        ap_ccf = p.ap_um
        return ap_ccf, ml_ccf, dv_um

    def apply_many(self, pts_px: np.ndarray) -> np.ndarray:
        pts_px = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        out = np.empty((len(pts_px), 3), dtype=float)
        for i, (x, y) in enumerate(pts_px):
            out[i] = self.apply(float(x), float(y))
        return out


# Back-compat alias - older code imports SectionTransform.
SectionTransform = ManualSectionTransform


@dataclass(frozen=True)
class RegisteredSectionTransform:
    """The M3 pixel→CCF mapping: optional manual affine ∘ B-spline ∘ anchoring."""

    anchoring: Anchoring
    output_size_px: tuple[int, int]
    bspline: "sitk.Transform | None"
    atlas_resolution_um: tuple[float, float, float]  # (ap_res, dv_res, ml_res)
    # Section-local 3x3 manual-correction affine (row, col); None = identity.
    manual_affine: np.ndarray | None = None
    # Landmark TPS correction (section-local (x, y) source/target arrays); takes
    # precedence over manual_affine when present.
    manual_landmarks: tuple[np.ndarray, np.ndarray] | None = None
    # Which way the landmark spline is fitted (ManualLandmarks.forward).
    landmarks_forward: bool = False

    def _section_px_to_slice_px(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Map a histology pixel through the inverse B-spline to slice-px coords.

        The B-spline was trained mapping FIXED (atlas slice) → MOVING
        (histology). To go in the other direction we need the inverse. For a
        ``CompositeTransform(Affine, BSpline)`` SimpleITK's :func:`GetInverse`
        is exact for the affine; the B-spline inverse is iterative.
        """
        if self.bspline is None:
            return x_px, y_px
        slice_x, slice_y = self._inverse_warp.TransformPoint((float(x_px), float(y_px)))
        return float(slice_x), float(slice_y)

    @cached_property
    def _inverse_warp(self) -> "sitk.Transform":
        """The warp's inverse, built once: inverting a displacement field is costly."""
        # SITK TransformPoint maps fixed→moving; we want the inverse.
        try:
            return self.bspline.GetInverse()
        except RuntimeError:
            # Some B-splines need an iterative inverse displacement field.
            return _invert_displacement(self.bspline, self.output_size_px)

    def apply(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        ap, ml, dv = self.apply_many(np.array([[x_px, y_px]], dtype=float))[0]
        return float(ap), float(ml), float(dv)

    def apply_many(self, pts_px: np.ndarray) -> np.ndarray:
        """Section pixels (N, 2) in (x, y) -> CCF (AP, ML, DV) µm, (N, 3)."""
        pts = np.asarray(pts_px, dtype=float).reshape(-1, 2)
        # The clicked point is in the corrected (dragged) frame; pull it back into
        # the registered frame before the registration inverse. Landmarks (TPS)
        # take precedence over the box-handle affine.
        if self.manual_landmarks is not None:
            from atlastrack.registration.landmarks_warp import invert_points

            src, dst = self.manual_landmarks
            pts = invert_points(src, dst, pts, forward=self.landmarks_forward)
        elif self.manual_affine is not None:
            from atlastrack.registration.manual import invert_apply

            pts = np.array([invert_apply(self.manual_affine, x, y) for x, y in pts], dtype=float)
        sl = np.array([self._section_px_to_slice_px(x, y) for x, y in pts], dtype=float)
        sl = sl.reshape(-1, 2)
        h, w = self.output_size_px
        su = sl[:, 0] / max(w, 1)
        sv = sl[:, 1] / max(h, 1)
        a = self.anchoring
        ap_idx = a.ox + su * a.ux + sv * a.vx
        dv_idx = a.oy + su * a.uy + sv * a.vy
        ml_idx = a.oz + su * a.uz + sv * a.vz
        ap_res, dv_res, ml_res = self.atlas_resolution_um
        return np.stack([ap_idx * ap_res, ml_idx * ml_res, dv_idx * dv_res], axis=1)


def _invert_displacement(
    transform: "sitk.Transform", output_size_px: tuple[int, int]
) -> "sitk.Transform":
    """Fall-back B-spline inverse via a sampled displacement field."""
    import SimpleITK as sitk

    h, w = output_size_px
    ref = sitk.Image(int(w), int(h), sitk.sitkFloat32)
    disp = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        ref.GetSize(),
        ref.GetOrigin(),
        ref.GetSpacing(),
        ref.GetDirection(),
    )
    inverse_disp = sitk.InvertDisplacementField(
        disp,
        maximumNumberOfIterations=20,
        meanErrorToleranceThreshold=1e-3,
        maxErrorToleranceThreshold=1e-2,
        enforceBoundaryCondition=True,
    )
    return sitk.DisplacementFieldTransform(inverse_disp)


#: Recently warped label images, keyed by everything that determines them. See
#: :func:`_warp_key`. Each entry is one section's labels (~2-3 MB at 25 um), so
#: this comfortably holds a full slide without growing without bound.
_WARP_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_WARP_CACHE_MAX = 48
_WARP_CACHE_LOCK = threading.Lock()


def _warp_key(
    result: RegistrationResult,
    atlas: BrainGlobeAtlas,
    section_shape: tuple[int, int],
    project_dir: Path | None,
    source_shape: tuple[int, int, int] | None,
) -> tuple:
    """Everything the warped labels depend on - including the sidecar's contents.

    The transform file is identified by path **and** modification time and size,
    because re-registering a section rewrites the same path: keying on the path
    alone would keep serving the old overlay after a new fit.
    """
    stamp = None
    path_str = result.bspline_transform_path
    if path_str is not None:
        path = Path(path_str)
        if project_dir is not None and not path.is_absolute():
            path = project_dir / path
        try:
            st = path.stat()
            stamp = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = (str(path), None, None)
    return (
        tuple(float(v) for v in result.anchoring),
        tuple(int(v) for v in result.output_size_px),
        stamp,
        tuple(int(v) for v in section_shape),
        getattr(atlas, "atlas_name", None),
        id(atlas),
        tuple(int(v) for v in atlas.annotation.shape),
        None if source_shape is None else tuple(int(v) for v in source_shape),
    )


def clear_warp_cache() -> None:
    """Forget every overlay cached in memory. Files on disk are left alone."""
    with _WARP_CACHE_LOCK:
        _WARP_CACHE.clear()


# --------------------------------------------------------------------------
# On-disk overlay cache
# --------------------------------------------------------------------------
#
# One ``<sidecar>.overlay.npz`` beside each ``section_NNN.h5``, so every project
# load after the first gets its overlays back in milliseconds instead of re-running
# the inverse warp (~0.3 s a section). It is disposable: delete it and it is rebuilt.
#
# **The dangerous failure is a stale file, not a missing one.** A cache keyed only
# on its inputs would keep serving the old overlay after the warp code itself is
# fixed, with nothing on screen to say so. So the key also carries a fingerprint of
# the *source* of every function that produces the result, plus the versions of the
# libraries doing the maths; change any of them and every cached file misses.

#: Bump if the overlay changes for a reason the source fingerprint cannot see (a
#: data file, say). Code changes to the functions below invalidate on their own.
_OVERLAY_CACHE_VERSION = 1

#: Suffix of the cache file written next to each transform sidecar.
OVERLAY_CACHE_SUFFIX = ".overlay.npz"

_FINGERPRINT: str | None = None


def _overlay_code_fingerprint() -> str:
    """Hash of the code and libraries that produce a warped overlay. Computed once."""
    global _FINGERPRINT
    if _FINGERPRINT is not None:
        return _FINGERPRINT
    import hashlib
    import inspect

    import scipy
    import SimpleITK as sitk

    from atlastrack.atlas import planes

    parts = [
        f"v{_OVERLAY_CACHE_VERSION}",
        f"sitk={sitk.Version_VersionString()}",
        f"scipy={scipy.__version__}",
        f"numpy={np.__version__}",
    ]
    for fn in (
        _warp_annotation_to_section_uncached,
        _warped_atlas_extent,
        _invert_displacement,
        build_registered_transform,
        planes.annotation_at_plane,
        planes.sample_plane,
        planes.rescale_atlas_anchoring,
    ):
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError):
            # No source (a frozen build): fall back to the package version, which
            # still changes with every release that could have changed the code.
            from atlastrack import __version__

            parts.append(f"{fn.__qualname__}@{__version__}")
    _FINGERPRINT = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return _FINGERPRINT


def _sidecar_path(result: RegistrationResult, project_dir: Path | None) -> Path | None:
    if result.bspline_transform_path is None:
        return None
    path = Path(result.bspline_transform_path)
    if project_dir is not None and not path.is_absolute():
        path = project_dir / path
    return path


def _disk_digest(
    result: RegistrationResult,
    atlas: BrainGlobeAtlas,
    section_shape: tuple[int, int],
    sidecar: Path,
    source_shape: tuple[int, int, int] | None,
) -> str | None:
    """Everything the overlay depends on, portable across sessions and folders.

    Unlike the in-memory key this uses the sidecar's *name*, not its absolute path,
    so a project folder that is moved or copied keeps its cache.
    """
    import hashlib

    try:
        st = sidecar.stat()
    except OSError:
        return None
    metadata = getattr(atlas, "metadata", None) or {}
    key = (
        tuple(float(v) for v in result.anchoring),
        tuple(int(v) for v in result.output_size_px),
        sidecar.name, st.st_mtime_ns, st.st_size,
        tuple(int(v) for v in section_shape),
        getattr(atlas, "atlas_name", None),
        str(metadata.get("version")),
        tuple(int(v) for v in atlas.annotation.shape),
        None if source_shape is None else tuple(int(v) for v in source_shape),
        _overlay_code_fingerprint(),
    )
    return hashlib.sha256(repr(key).encode()).hexdigest()


def _cache_file(sidecar: Path) -> Path:
    return sidecar.with_name(sidecar.stem + OVERLAY_CACHE_SUFFIX)


def _load_cached_overlay(path: Path, digest: str) -> np.ndarray | None:
    """The cached labels if the file matches ``digest`` exactly, else None."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["digest"]) != digest:
                return None
            return np.asarray(data["labels"])
    except Exception:  # missing, truncated, or from an older layout: just rebuild
        return None


def _save_cached_overlay(path: Path, digest: str, labels: np.ndarray) -> None:
    """Write atomically, and never let a failed write break the overlay itself.

    Atomic because a crash mid-write must not leave a file that loads as garbage;
    silent because a read-only or full project folder is a reason to skip caching,
    not a reason to refuse to draw.
    """
    import contextlib
    import os
    import tempfile

    try:
        fd, tmp = tempfile.mkstemp(
            prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez_compressed(fh, digest=np.asarray(digest), labels=labels)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except Exception:
        return


def warp_annotation_to_section(
    result: RegistrationResult,
    atlas: BrainGlobeAtlas,
    section_shape: tuple[int, int],
    *,
    project_dir: Path | None = None,
    source_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Render the registered atlas annotation in a section's pixel grid.

    Cached: the same section is re-rendered constantly - every 'Place landmarks',
    every re-draw after a box edit, and every refresh of the split-panel window,
    which used to spend ~270 ms per drag release recomputing a result that could
    not have changed. A copy is returned so callers may modify it freely.
    """
    key = _warp_key(result, atlas, section_shape, project_dir, source_shape)
    with _WARP_CACHE_LOCK:
        hit = _WARP_CACHE.get(key)
        if hit is not None:
            _WARP_CACHE.move_to_end(key)
            return hit.copy()

    labels = None
    sidecar = _sidecar_path(result, project_dir)
    digest = cache_file = None
    if sidecar is not None:
        digest = _disk_digest(result, atlas, section_shape, sidecar, source_shape)
        if digest is not None:
            cache_file = _cache_file(sidecar)
            labels = _load_cached_overlay(cache_file, digest)
    if labels is None:
        labels = _warp_annotation_to_section_uncached(
            result, atlas, section_shape, project_dir=project_dir, source_shape=source_shape
        )
        if cache_file is not None and digest is not None:
            _save_cached_overlay(cache_file, digest, labels)

    with _WARP_CACHE_LOCK:
        _WARP_CACHE[key] = labels.copy()
        while len(_WARP_CACHE) > _WARP_CACHE_MAX:
            _WARP_CACHE.popitem(last=False)
    return labels


def warp_annotations_to_sections(
    jobs: list[tuple[RegistrationResult, tuple[int, int]]],
    atlas: BrainGlobeAtlas,
    *,
    project_dir: Path | None = None,
    source_shape: tuple[int, int, int] | None = None,
    max_workers: int | None = None,
) -> list[np.ndarray | Exception]:
    """Warp many sections at once, in parallel. One entry per job, in order.

    A failed section yields its exception in place of labels, so one bad sidecar
    does not cost the others their overlay.

    **Why the thread split.** ITK already runs each filter on every core, so
    sections launched side by side simply fight over the same 32 threads: that
    measured 1.8x. Giving each worker its share of ITK's threads instead measured
    2.9x on the same 18 sections, bit-identical to the serial result. ITK's
    thread count is process-wide, so it is restored before returning.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import SimpleITK as sitk

    if not jobs:
        return []

    def _one(job):
        result, shape = job
        try:
            return warp_annotation_to_section(
                result, atlas, shape, project_dir=project_dir, source_shape=source_shape
            )
        except Exception as exc:  # returned, not raised - see docstring
            return exc

    cores = os.cpu_count() or 1
    workers = max_workers or max(1, min(len(jobs), cores // 4))
    if workers <= 1:
        return [_one(job) for job in jobs]

    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, cores // workers))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, jobs))
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


def _warp_annotation_to_section_uncached(
    result: RegistrationResult,
    atlas: BrainGlobeAtlas,
    section_shape: tuple[int, int],
    *,
    project_dir: Path | None = None,
    source_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Render the registered atlas annotation in a section's pixel grid.

    Resamples the atlas annotation at the section's plane (``anchoring``), then
    warps that slice through the fitted B-spline into the histology section's
    pixel space, so the result can be overlaid directly on the section image.

    Returns an integer label image of shape ``section_shape`` (H, W). If no
    B-spline was stored the plane annotation is returned resized to the section
    (i.e. the un-refined plane), which is still a useful sanity overlay.
    """
    from atlastrack.atlas.planes import annotation_at_plane, rescale_atlas_anchoring

    anchoring = Anchoring.from_iterable(result.anchoring)
    # ``result.anchoring`` counts voxels of the atlas the fit was computed on. When
    # the labels are being drawn from a *different* atlas - the region-atlas picker -
    # that count has to be restated on the new grid, or every plane lands wrong.
    if source_shape is not None:
        target_shape = tuple(int(v) for v in atlas.reference.shape)
        if tuple(int(v) for v in source_shape) != target_shape:
            anchoring = rescale_atlas_anchoring(
                anchoring, source_shape=source_shape, target_shape=target_shape
            )
    h_slice, w_slice = result.output_size_px
    ann_slice = annotation_at_plane(atlas, anchoring, (int(h_slice), int(w_slice)))

    h_sec, w_sec = int(section_shape[0]), int(section_shape[1])
    if result.bspline_transform_path is None:
        # No refinement available: nearest-neighbour resize of the plane.
        ys = (np.linspace(0, h_slice - 1, h_sec)).round().astype(int)
        xs = (np.linspace(0, w_slice - 1, w_sec)).round().astype(int)
        return ann_slice[np.ix_(ys, xs)]

    import SimpleITK as sitk

    transform = build_registered_transform(result, atlas, project_dir=project_dir).bspline
    # The B-spline maps fixed (atlas slice) → moving (section). Resampling the
    # annotation into the section grid needs the section → slice map (inverse).
    try:
        inverse = transform.GetInverse()
    except RuntimeError:
        inverse = _invert_displacement(transform, (h_slice, w_slice))

    ann_img = sitk.GetImageFromArray(ann_slice.astype(np.int32))
    reference = sitk.Image(w_sec, h_sec, sitk.sitkInt32)
    warped = sitk.GetArrayFromImage(
        sitk.Resample(ann_img, reference, inverse, sitk.sitkNearestNeighbor, 0.0)
    )
    # The inverse displacement field extrapolates nonsense OUTSIDE the registered
    # atlas, painting boundary "stripes" far off the section. Clip the labels to
    # where the atlas actually lands (forward-warped extent) - this removes the
    # stripes while KEEPING every region outline inside the brain, including over
    # damaged/dim tissue (so the user still sees what region it was).
    extent = _warped_atlas_extent(
        transform, ann_slice > 0, (h_slice, w_slice), (h_sec, w_sec)
    )
    warped[~extent] = 0
    return warped


def _warped_atlas_extent(
    transform: "sitk.Transform",
    atlas_foreground: np.ndarray,
    atlas_shape: tuple[int, int],
    section_shape: tuple[int, int],
) -> np.ndarray:
    """Boolean mask of where the atlas brain lands in section space.

    Forward-splats the atlas-foreground pixels through ``transform``
    (fixed atlas -> moving section), then closes/fills the splat. Uses ONLY the
    forward field, so unlike an inverse resample it can't extrapolate coverage
    into regions the atlas never reached.
    """
    import SimpleITK as sitk
    from scipy import ndimage as ndi

    h_a, w_a = int(atlas_shape[0]), int(atlas_shape[1])
    h_s, w_s = int(section_shape[0]), int(section_shape[1])
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, (w_a, h_a),
        (0.0, 0.0), (1.0, 1.0), (1.0, 0.0, 0.0, 1.0),
    )
    disp = sitk.GetArrayFromImage(field)  # (h_a, w_a, 2): [...,0]=dx, [...,1]=dy
    ys, xs = np.nonzero(atlas_foreground)
    sx = np.round(xs + disp[ys, xs, 0]).astype(int)
    sy = np.round(ys + disp[ys, xs, 1]).astype(int)
    ok = (sx >= 0) & (sx < w_s) & (sy >= 0) & (sy < h_s)
    cov = np.zeros((h_s, w_s), dtype=bool)
    cov[sy[ok], sx[ok]] = True
    # Fill the small gaps left by local expansion of the warp.
    cov = ndi.binary_closing(cov, iterations=3)
    return ndi.binary_fill_holes(cov)


def annotation_boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean edge map: True where a label differs from a 4-neighbour."""
    labels = np.asarray(labels)
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edges[1:, :] |= labels[:-1, :] != labels[1:, :]
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    return edges


def inverse_warp_path(path: str | Path) -> Path:
    """Where an exact inverse of the warp file ``path`` is kept, if there is one."""
    path = Path(path)
    return path.with_name(f"{path.stem}_inverse{path.suffix}")


def build_registered_transform(
    result: RegistrationResult,
    atlas: "BrainGlobeAtlas",
    *,
    project_dir: Path | None = None,
    manual_affine: list[list[float]] | np.ndarray | None = None,
    manual_landmarks: "object | None" = None,
) -> RegisteredSectionTransform:
    """Construct a :class:`RegisteredSectionTransform` from persisted state.

    ``manual_landmarks`` is a ``Section.manual_landmarks``-like object (with
    ``source``/``target`` lists) or ``None``.
    """
    from atlastrack.io.ccf_coords import atlas_resolution_um

    anchoring = Anchoring.from_iterable(result.anchoring)
    bspline = None
    if result.bspline_transform_path is not None:
        import SimpleITK as sitk

        path = Path(result.bspline_transform_path)
        if project_dir is not None and not path.is_absolute():
            path = project_dir / path
        with _HDF5_LOCK:
            bspline = sitk.ReadTransform(str(path))
            inverse = inverse_warp_path(path)
            if inverse.is_file():
                # An exact inverse, e.g. VisuAlign's: points use it rather than
                # a numerical inversion, which a folded warp defeats.
                field = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(inverse)))
                # A copy the transform owns: the field image would otherwise still
                # belong to `field`, and points crash once `field` is freed.
                stored = field.GetDisplacementField()
                own = sitk.GetImageFromArray(sitk.GetArrayFromImage(stored), isVector=True)
                own.CopyInformation(stored)  # keep its spacing and origin
                bspline = sitk.DisplacementFieldTransform(bspline)
                bspline.SetInverseDisplacementField(own)
    ma = None if manual_affine is None else np.asarray(manual_affine, dtype=float).reshape(3, 3)
    lm = None
    if manual_landmarks is not None:
        lm = (
            np.asarray(manual_landmarks.source, dtype=float).reshape(-1, 2),
            np.asarray(manual_landmarks.target, dtype=float).reshape(-1, 2),
        )
    return RegisteredSectionTransform(
        anchoring=anchoring,
        output_size_px=tuple(result.output_size_px),  # type: ignore[arg-type]
        bspline=bspline,
        atlas_resolution_um=atlas_resolution_um(atlas),
        manual_affine=ma,
        manual_landmarks=lm,
        landmarks_forward=bool(getattr(manual_landmarks, "forward", False)),
    )
