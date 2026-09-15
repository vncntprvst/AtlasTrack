"""Image loaders for histology slides.

Thin wrapper over :mod:`tifffile` / :mod:`imageio` so the rest of the code only
sees a numpy array. Pyramid handling will arrive with M2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Pillow's own ceiling on decoded pixels. Above it Pillow warns ("could be
#: decompression bomb DOS attack"); above **twice** it Pillow refuses to decode at
#: all (``DecompressionBombError``). Both guard against a small file crafted to
#: expand until it fills RAM - a real attack, and not what a slide scanner writes.
#: A 2x whole-slide scan sits right on this line (13927x7453 RGB = 103.8 MP), and a
#: larger one would hit the hard refusal, so the loader lifts the ceiling rather
#: than let a legitimate slide be warned about cryptically or rejected outright.
#: The size is reported to the user instead - see :func:`oversize_note`.
PILLOW_PIXEL_LIMIT = 89_478_485


def _lift_pillow_pixel_limit() -> None:
    """Stop Pillow warning about, or refusing, a large but legitimate slide.

    Done at load time rather than import time so merely importing this module does
    not mutate global Pillow state. Best-effort: Pillow may be absent (a TIFF-only
    install) and that must not stop a load.
    """
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass


def oversize_note(pixels: int) -> str | None:
    """A plain-language note if an image of ``pixels`` px is unusually large.

    Returns ``None`` for an ordinary image. Takes a pixel count rather than a path
    so it applies equally to a merged multi-slide canvas, which has no file of its
    own and can cross the line even when every source image is below it.

    This exists because the only signal the user previously got was Pillow's
    ``DecompressionBombWarning`` on stderr - which talks about a "DOS attack",
    says nothing about what it means for their slide, and scrolls past.
    """
    if pixels <= PILLOW_PIXEL_LIMIT:
        return None
    return (
        f"Large image: {pixels / 1e6:.1f} megapixels, above the {PILLOW_PIXEL_LIMIT / 1e6:.1f} MP "
        f"point where image libraries start warning. It loads and works normally - "
        f"but section detection scales with pixel count, so expect it to take longer."
    )


def load_image(path: str | Path) -> np.ndarray:
    """Load a 2D or 3D (H, W[, C]) image as a numpy array."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        return np.asarray(tifffile.imread(str(path)))

    # Pillow / imageio handles png/jpg/etc.
    _lift_pillow_pixel_limit()
    import imageio.v3 as iio

    return np.asarray(iio.imread(str(path)))


def crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``image`` to ``(x0, y0, x1, y1)`` in pixel coords (right/bottom exclusive)."""
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Coerce a 2D/RGB/RGBA array to a 3-channel (H, W, 3) array."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[2] >= 3:
        return arr[..., :3]
    # Single-channel-in-3D or 2-channel: replicate the first channel.
    return np.stack([arr[..., 0]] * 3, axis=-1)


def slide_bands(
    heights: list[int], *, gap_px: int = 40
) -> list[tuple[int, int]]:
    """Vertical ``(y_start, y_end)`` band of each source image in the merged canvas.

    :func:`merge_images` stacks sources top-to-bottom separated by ``gap_px``, so
    source ``i`` occupies rows ``[y_start, y_end)``. These bands let section
    detection stay **slide-aware**: order the sections within each source's band
    independently instead of letting a column run across two stacked slides. Must
    use the same ``gap_px`` as :func:`merge_images`.
    """
    bands: list[tuple[int, int]] = []
    y = 0
    for h in heights:
        bands.append((y, y + int(h)))
        y += int(h) + gap_px
    return bands


def merge_images(images: list[np.ndarray], *, gap_px: int = 40) -> np.ndarray:
    """Merge several slide images into one canvas, stacked vertically with a gap.

    Each image is placed top-left in a row; rows are padded to the widest image
    and separated by ``gap_px`` of background. Grayscale and colour inputs are
    mixed by promoting everything to RGB when any input is colour. The layout is
    a pure function of the input order, so a project that records its source
    paths (sorted) can reproduce the exact same combined image on reload - which
    keeps section bounding boxes valid.

    This is how the app supports "multiple slides": rather than tracking per-slide
    offsets through clicks / registration / 3D, all slides live in one image and
    therefore one coordinate space (a probe may enter on a section from one slide
    and have its tip on a section from another).
    """
    arrays = [np.asarray(im) for im in images]
    if not arrays:
        raise ValueError("merge_images requires at least one image")
    if len(arrays) == 1:
        return arrays[0]

    any_rgb = any(a.ndim == 3 for a in arrays)
    if any_rgb:
        arrays = [_to_rgb(a) for a in arrays]

    dtype = np.result_type(*[a.dtype for a in arrays])
    arrays = [a.astype(dtype, copy=False) for a in arrays]

    max_w = max(a.shape[1] for a in arrays)
    n = len(arrays)
    total_h = sum(a.shape[0] for a in arrays) + gap_px * (n - 1)
    shape = (total_h, max_w, 3) if any_rgb else (total_h, max_w)
    canvas = np.zeros(shape, dtype=dtype)

    y = 0
    for a in arrays:
        h, w = a.shape[:2]
        canvas[y:y + h, :w] = a
        y += h + gap_px
    return canvas
