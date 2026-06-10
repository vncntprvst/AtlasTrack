"""Optional DeepSlice plane predictor.

Wraps ``DeepSlice.DSModel`` so the rest of the pipeline can use it through the
:class:`PlanePredictor` protocol. The import happens lazily inside methods so
the base package installs without TensorFlow.

Install via the optional extra::

    pip install histo-to-ccf[deepslice]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from histo_to_ccf.io.quicknii import QuickNiiDocument, load_quicknii
from histo_to_ccf.project.schema import PlaneParams

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas


class DeepSlicePredictor:
    """Predict per-section atlas planes by running DeepSlice on an image folder.

    Unlike :class:`ManualPredictor`, this predictor is *batch* - it must see
    a whole folder of sections to apply :func:`propagate_angles` and
    :func:`enforce_index_order`. Use :meth:`predict_folder` once per slide,
    then :meth:`predict_single` looks up the cached result for each section.
    """

    def __init__(self, species: str = "mouse", *, ensemble: bool = True) -> None:
        self.species = species
        self.ensemble = ensemble
        self._cache: dict[str, PlaneParams] = {}

    def predict_folder(
        self,
        section_dir: Path | str,
        *,
        propagate_angles: bool = True,
        enforce_order: bool = True,
        output_xml: Path | str | None = None,
    ) -> QuickNiiDocument:
        """Run DeepSlice on every image in ``section_dir`` and cache results."""
        from DeepSlice import DSModel  # imported lazily

        model = DSModel(self.species)
        model.predict(str(section_dir), ensemble=self.ensemble)
        if propagate_angles:
            model.propagate_angles()
        if enforce_order:
            model.enforce_index_order()
        # DeepSlice.save_predictions(name) writes ``name.json`` and ``name.csv``
        # (it appends the extension), so pass a base path and read back ``.json``.
        if output_xml is None:
            output_base = Path(section_dir) / "deepslice_predictions"
        else:
            output_base = Path(output_xml).with_suffix("")
        model.save_predictions(str(output_base))
        doc = load_quicknii(str(output_base) + ".json")
        self._cache = self._build_cache(doc)
        return doc

    def _build_cache(self, doc: QuickNiiDocument) -> dict[str, PlaneParams]:
        """We don't bother converting QuickNII anchoring back to PlaneParams.

        Downstream code reads the full QuickNiiDocument; we keep this method
        as a hook in case a future caller wants per-filename PlaneParams.
        """
        return {}

    def predict(self, image: np.ndarray, *, section_index: int) -> PlaneParams:
        """Implements PlanePredictor - but only after predict_folder ran.

        The image argument is ignored; we keyed by section_index via the
        folder run. If you need a real per-image API, call DeepSlice directly.
        """
        del image, section_index
        raise NotImplementedError(
            "DeepSlicePredictor is batch-only. Call predict_folder(section_dir) "
            "and consume the returned QuickNiiDocument directly."
        )


# ---------------------------------------------------------------------------
# Folder-based anchoring prediction for the GUI register flow
# ---------------------------------------------------------------------------

# DeepSlice extracts a section number from an ``_s<number>`` token in the
# filename (regex ``_s\d+``), so we name crops ``section_s<idx>.png``.
_IDX_RE = re.compile(r"_s(\d+)")


def _section_filename(idx: int) -> str:
    return f"section_s{int(idx):03d}.png"


def _parse_section_index(filename: str) -> int | None:
    """Recover the section index from a ``section_s<idx>.png`` filename."""
    m = _IDX_RE.search(Path(filename).stem)
    return int(m.group(1)) if m else None


# DeepSlice/QuickNII "ABA_Mouse_CCFv3" 25 µm volume, in its native voxel order
# (ML, AP, DV) - i.e. dimensions 456 × 528 × 320. Expressed below in our
# (AP, DV, ML) order for scaling against a brainglobe atlas of the same family.
_QUICKNII_DIMS_APDVML = (528, 320, 456)

# Axis-direction differences between QuickNII ABA and the brainglobe ASR atlas.
# QuickNII's AP and DV run opposite to brainglobe (anterior/dorsal at the high
# end), so those axes are flipped. ML appears to share direction; flip it here
# if registered slices come out mirrored left↔right.
_FLIP_AP, _FLIP_DV, _FLIP_ML = True, True, False


def _quicknii_to_atlas_anchoring(
    anchoring: list[float],
    atlas_shape_apdvml: tuple[int, int, int],
) -> list[float]:
    """Convert a DeepSlice/QuickNII anchoring into our atlas's anchoring.

    Three transforms are applied:

    1. **Axis permutation.** QuickNII ABA voxels are ordered ``(ML, AP, DV)``;
       our :class:`~histo_to_ccf.atlas.planes.Anchoring` / ``sample_plane`` use
       ``(AP, DV, ML)`` (brainglobe ASR order). Each origin/u/v triplet is
       reordered ``(x, y, z) -> (y, z, x)``.
    2. **Resolution scaling.** QuickNII predicts in the 25 µm grid; components
       are scaled per axis to the loaded atlas's voxel grid.
    3. **Axis flips.** QuickNII AP and DV run opposite to brainglobe, so for a
       flipped axis the origin becomes ``size - o`` and the u/v components are
       negated (``P_k -> size - P_k``).
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = anchoring
    # (ML, AP, DV) -> (AP, DV, ML)
    o = [oy, oz, ox]
    u = [uy, uz, ux]
    v = [vy, vz, vx]
    # Per-axis scale to the loaded atlas grid.
    scale = [atlas_shape_apdvml[k] / _QUICKNII_DIMS_APDVML[k] for k in range(3)]
    o = [o[k] * scale[k] for k in range(3)]
    u = [u[k] * scale[k] for k in range(3)]
    v = [v[k] * scale[k] for k in range(3)]
    # Per-axis flips (origin -> size - origin; u, v negated).
    for k, flip in enumerate((_FLIP_AP, _FLIP_DV, _FLIP_ML)):
        if flip:
            o[k] = atlas_shape_apdvml[k] - o[k]
            u[k] = -u[k]
            v[k] = -v[k]
    return [*o, *u, *v]


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize a section image to 8-bit for DeepSlice (RGB or grayscale)."""
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def predict_anchorings(
    section_images: dict[int, np.ndarray],
    atlas: "BrainGlobeAtlas",
    *,
    workdir: Path | str,
    species: str = "mouse",
) -> dict[int, list[float]]:
    """Run DeepSlice on section crops and return per-section atlas anchorings.

    Writes each crop to ``workdir`` as ``section_s<idx>.png`` (DeepSlice reads
    the section number from the ``_s<idx>`` token), then runs DeepSlice **in a
    separate process** (see :mod:`deepslice_run`) so its TensorFlow memory is
    freed before the calling process moves on to registration. Returns
    ``{section_idx: anchoring9}`` converted to the loaded atlas's anchoring,
    ready to drop into the B-spline pipeline.
    """
    import subprocess

    import imageio.v3 as iio

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for idx, img in section_images.items():
        iio.imwrite(workdir / _section_filename(idx), _to_uint8(img))

    # Inherit stdout/stderr so DeepSlice's progress bar still shows in the
    # console; TensorFlow lives and dies inside this child process.
    result = subprocess.run(
        [sys.executable, "-m", "histo_to_ccf.registration.deepslice_run",
         str(workdir), species],
    )
    pred_json = workdir / "deepslice_predictions.json"
    if result.returncode != 0 or not pred_json.exists():
        raise RuntimeError(
            "DeepSlice prediction failed (see console output above). "
            f"Exit code {result.returncode}."
        )

    doc = load_quicknii(pred_json)
    shape = tuple(int(s) for s in atlas.annotation.shape)
    out: dict[int, list[float]] = {}
    for sl in doc.slices:
        idx = _parse_section_index(sl.filename)
        if idx is None:
            continue
        out[idx] = _quicknii_to_atlas_anchoring(list(sl.anchoring), shape)
    return out
