"""Export a registered section series as images plus atlas-outline sidecars.

For the user who runs this app only to put histology into an atlas: the probe
exports are irrelevant to them, and what they want back is their own sections -
prepared the way the app prepared them, in series order - with the registered
atlas contours alongside.

The crops come straight from
:func:`histo_to_ccf.project.images.rebuild_slide_image`, so the flips *and* the
per-section rotation are already in them - the same pixels the registration was
computed against. That is what lets the outline sidecar be written in the section's
own frame with no correction: it cannot drift out of alignment with its image,
because both come from one source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from histo_to_ccf.project.schema import Project, Section

#: The sidecar is a figure component, not a screen overlay: black lines on white
#: drop straight into a paper or an illustration without inverting anything.
SIDECAR_INK = 0
SIDECAR_PAPER = 255

#: Burnt into the histology, white reads on dark fluorescence where black vanishes.
BURN_IN_RGB = (255, 255, 255)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale to 8-bit for PNG, preserving an already-8-bit image exactly."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (((arr - lo) / (hi - lo)) * 255.0).astype(np.uint8)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = _to_uint8(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    return arr[..., :3]


def _warped_labels(
    section: Section,
    atlas: BrainGlobeAtlas,
    shape,
    project_dir: Path | None,
    source_shape=None,
) -> np.ndarray | None:
    """Registered atlas region labels in this section's pixel grid."""
    if getattr(section, "registration", None) is None:
        return None
    from histo_to_ccf.registration.transforms import warp_annotation_to_section

    return warp_annotation_to_section(
        section.registration, atlas, shape,
        project_dir=project_dir, source_shape=source_shape,
    )


def outlines_to_svg(labels: np.ndarray, atlas: BrainGlobeAtlas) -> str:
    """One SVG path per atlas region, so the outlines stay editable.

    A PNG of contours can only be traced; a vector file can be restyled, split and
    labelled in Illustrator or Inkscape, which is what the outlines are usually for.
    Each path carries the region's acronym as its ``id`` and full name as a
    ``<title>``, so the shapes are identifiable once they are out of this app.
    """
    from skimage.measure import find_contours

    height, width = labels.shape[:2]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
    ]
    for region_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        acronym, name = _region_names(atlas, region_id)
        for contour in find_contours(labels == region_id, 0.5):
            # find_contours yields (row, col); SVG wants (x, y).
            points = " ".join(f"{c:.2f},{r:.2f}" for r, c in contour)
            parts.append(
                f'<g id="{_xml_escape(acronym)}"><title>{_xml_escape(name)}</title>'
                f'<polyline points="{points}" fill="none" stroke="#000" '
                'stroke-width="1"/></g>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _xml_escape(text: str) -> str:
    return (
        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _region_names(atlas: BrainGlobeAtlas, region_id: int) -> tuple[str, str]:
    """``(acronym, name)`` for an atlas id - the pair the hover readout shows."""
    try:
        structure = atlas.structures[int(region_id)]
        return str(structure["acronym"]), str(structure["name"])
    except Exception:
        return str(region_id), ""


def straighten_angle_deg(section: Section) -> float:
    """Residual tilt to remove at export, degrees.

    DeepSlice measures the angle of the section as it was *scanned*. Any rotation
    already baked into the working image has removed part of that, so what is left
    to straighten for presentation is the difference. When nothing was set by hand
    this is simply DeepSlice's angle; when the user took DeepSlice's angle with the
    "From DeepSlice" button it is zero, so the rotation is applied once, not twice.

    Returns 0 for a section DeepSlice never saw - there is nothing to measure from.
    """
    from histo_to_ccf.project.images import deepslice_rotation_deg

    anchoring = getattr(section, "deepslice_anchoring", None)
    if not anchoring or len(anchoring) < 6:
        return 0.0
    return deepslice_rotation_deg(anchoring) - float(
        getattr(section, "rotation_deg", 0.0) or 0.0
    )


def _rotate_for_display(image: np.ndarray, degrees: float, *, order: int) -> np.ndarray:
    """Rotate about the centre, growing the canvas so nothing is clipped.

    Unlike the baked-in rotation this is free to change shape: it is applied to the
    section and its outline mask with identical parameters, so the two stay the same
    size and stay aligned, and nothing downstream reads these pixels.
    """
    if abs(degrees) < 1e-6:
        return image
    from scipy.ndimage import rotate as _rotate

    return _rotate(
        image, degrees, axes=(1, 0), reshape=True, order=order, mode="constant", cval=0
    )


@dataclass
class SeriesExportResult:
    """What was written, and what could not be."""

    out_dir: Path
    sections: int = 0
    outlines: int = 0
    overlays: int = 0
    svgs: int = 0
    regions: int = 0
    #: (section index, reason) - a section with no registration cannot have outlines,
    #: and saying which ones is the difference between a gap and a silent omission.
    skipped_outlines: list[tuple[int, str]] = field(default_factory=list)


def export_section_series(
    project: Project,
    out_dir: str | Path,
    *,
    atlas: BrainGlobeAtlas | None = None,
    base_dir: Path | None = None,
    write_outlines: bool = True,
    write_overlays: bool = False,
    write_svg: bool = False,
    write_regions: bool = False,
    straighten: bool = True,
    source_shape=None,
) -> SeriesExportResult:
    """Write the section series to ``out_dir``, in AP order.

    Files per section ``NNN`` (the position in the series, so a directory listing is
    the series order - the section's own index is in the manifest):

    ``NNN_section.png``
        the crop, with the flips and rotation the registration also saw.
    ``NNN_outline.png``
        RGBA, the registered atlas contours on transparency, same size and rotation.
    ``NNN_overlay.png``
        the two composited, only when ``write_overlays``.

    ``source_shape`` is the voxel grid of the atlas the registration was computed
    on. Give it whenever ``atlas`` is a *different* atlas - the region-atlas picker -
    so the stored anchorings can be restated on that atlas's grid; the Chon/Kim
    isotropic atlas shares the CCF extent but samples it at 20 um, not 25.

    Plus ``series.json``: per section its index, slide, bbox, AP, rotation and where
    that rotation came from, so the export can be traced back to the project.
    """
    from histo_to_ccf.io.image import crop
    from histo_to_ccf.project.images import rebuild_slide_image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    atlas_name = str(getattr(atlas, "atlas_name", "") or "") if atlas else ""
    result = SeriesExportResult(out_dir=out)
    manifest: list[dict] = []
    regions: list[dict] = []

    ordered = sorted(
        ((sl_i, sec) for sl_i, sl in enumerate(project.slides) for sec in sl.sections),
        key=lambda pair: (pair[1].ap_order, pair[1].index),
    )
    slide_cache: dict[int, np.ndarray] = {}

    for position, (slide_idx, section) in enumerate(ordered):
        if slide_idx not in slide_cache:
            try:
                image, _ = rebuild_slide_image(project.slides[slide_idx], base_dir=base_dir)
            except Exception as exc:  # one bad source must not stop the rest
                result.skipped_outlines.append((section.index, f"slide unreadable: {exc}"))
                continue
            slide_cache[slide_idx] = image
        patch = crop(slide_cache[slide_idx], section.bbox_px)

        stem = f"{position:03d}"
        residual = straighten_angle_deg(section) if straighten else 0.0
        rgb = _rotate_for_display(_as_rgb(patch), residual, order=1)
        iio.imwrite(out / f"{stem}_section.png", rgb)
        result.sections += 1

        entry = {
            "position": position,
            "section_index": int(section.index),
            "slide_index": int(slide_idx),
            "bbox_px": list(section.bbox_px),
            "ap_um": None if section.plane is None else float(section.plane.ap_um),
            "ap_source": getattr(section, "ap_source", None),
            "rotation_baked_deg": round(
                float(getattr(section, "rotation_deg", 0.0) or 0.0), 4
            ),
            "rotation_straighten_deg": round(residual, 4),
            "section_image": f"{stem}_section.png",
        }

        if write_outlines:
            if atlas is None:
                result.skipped_outlines.append((section.index, "no atlas loaded"))
            else:
                labels = _warped_labels(
                    section, atlas, patch.shape[:2], base_dir, source_shape
                )
                if labels is None:
                    result.skipped_outlines.append(
                        (section.index, "section is not registered")
                    )
                else:
                    from histo_to_ccf.registration.transforms import (
                        annotation_boundaries,
                    )

                    # Rotate the labels, then find the boundaries: contours traced
                    # after the rotation are clean, where a rotated 1-pixel mask
                    # frays. order=0 keeps the labels as labels.
                    turned_labels = _rotate_for_display(labels, residual, order=0)
                    edges = annotation_boundaries(turned_labels)

                    sidecar = np.full(
                        (*edges.shape, 3), SIDECAR_PAPER, dtype=np.uint8
                    )
                    sidecar[edges] = SIDECAR_INK
                    iio.imwrite(out / f"{stem}_outline.png", sidecar)
                    entry["outline_image"] = f"{stem}_outline.png"
                    result.outlines += 1

                    if write_overlays:
                        burned = rgb.copy()
                        burned[edges] = BURN_IN_RGB
                        iio.imwrite(out / f"{stem}_overlay.png", burned)
                        entry["overlay_image"] = f"{stem}_overlay.png"
                        result.overlays += 1

                    if write_svg:
                        (out / f"{stem}_outline.svg").write_text(
                            outlines_to_svg(turned_labels, atlas), encoding="utf-8"
                        )
                        entry["outline_svg"] = f"{stem}_outline.svg"
                        result.svgs += 1

                    if write_regions:
                        for region_id in sorted(
                            int(v) for v in np.unique(turned_labels) if v > 0
                        ):
                            acronym, name = _region_names(atlas, region_id)
                            regions.append({
                                "position": position,
                                "section_index": int(section.index),
                                "region_id": region_id,
                                "acronym": acronym,
                                "name": name,
                                "area_px": int((turned_labels == region_id).sum()),
                                # Which nomenclature these names are in. "M1" and
                                # "MOp" are the same region; a file that does not
                                # say which atlas named it cannot be read safely.
                                "atlas": atlas_name,
                            })

        manifest.append(entry)

    if write_regions:
        import csv

        with (out / "regions.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "position", "section_index", "region_id", "acronym", "name",
                    "area_px", "atlas",
                ],
            )
            writer.writeheader()
            writer.writerows(regions)
        result.regions = len(regions)

    (out / "series.json").write_text(
        json.dumps(
            {
                "registration_atlas": getattr(project.atlas, "name", None),
                "region_atlas": atlas_name or None,
                "sections": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
