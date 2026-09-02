"""histo2ccf - guided histology->atlas registration with probe trajectory mapping.

Typical workflow
----------------
1. ``histo2ccf gui``           - launch the interactive napari GUI.
2. ``histo2ccf split image.tif`` - detect sections in a composite slide.
3. ``histo2ccf register-one `` - headless single-section registration.
4. ``histo2ccf register project.json`` - run the full M3 pipeline on a project.
5. ``histo2ccf export project.json`` - write per-channel CCF / Paxinos CSVs.

Run ``histo2ccf <command> --help`` for per-command options.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

# Windows consoles default to a legacy codepage (cp1252), which cannot encode the
# non-ASCII characters used in help text and messages ("µm", "->"). Without this,
# even `histo2ccf --help` dies with a UnicodeEncodeError. Reconfiguring the streams
# is a no-op where the encoding is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # not a reconfigurable text stream
        pass

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help=__doc__,
    rich_markup_mode=None,
)


def _parse_xy(s: str, name: str) -> tuple[float, float]:
    try:
        x_str, y_str = s.split(",")
        return float(x_str), float(y_str)
    except Exception as e:
        raise typer.BadParameter(f"--{name} expects 'x,y' (got {s!r})") from e


def _parse_bbox(s: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(v.strip()) for v in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)  # type: ignore[return-value]
    except Exception as e:
        raise typer.BadParameter(f"--bbox expects 'x0,y0,x1,y1' (got {s!r})") from e


@app.command()
def version() -> None:
    """Print the installed package version and exit."""
    from histo_to_ccf import __version__

    typer.echo(f"histo2ccf {__version__}")


@app.command()
def gui() -> None:
    """Launch the interactive napari GUI.

    If it fails with 'QOpenGLFramebufferObject: Unsupported framebuffer format',
    your GPU/OpenGL driver is the problem - run ``histo2ccf gl-info`` to see what
    renderer is active and how to fix it.
    """
    from histo_to_ccf.gui.app import launch

    launch()


@app.command("gl-info")
def gl_info() -> None:
    """Report the OpenGL renderer/driver and diagnose GUI launch failures."""
    from histo_to_ccf.gui.gl_diagnostics import format_gl_report

    typer.echo(format_gl_report())


@app.command()
def split(
    image: Annotated[Path, typer.Option(help="Path to a composite slide image.")],
    min_area_px: Annotated[
        int, typer.Option(help="Drop components below this area.")
    ] = 5000,
    closing_radius_px: Annotated[
        int,
        typer.Option(
            help=(
                "Morphological closing radius; bridges anatomical gaps within "
                "a section (cerebellum/brainstem). 0 disables."
            ),
        ),
    ] = 0,
    expected_count: Annotated[
        int,
        typer.Option(
            help="If > 0, keep only the N largest passing components.",
        ),
    ] = 0,
    output_json: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Sidecar JSON listing detected sections. Defaults to "
                "<image_dir>/<image_stem>.sections.json."
            ),
        ),
    ] = None,
) -> None:
    """Detect sections in a composite slide and write a sidecar JSON of bboxes."""
    import json

    from histo_to_ccf.io.image import load_image
    from histo_to_ccf.sectioning.ordering import order_sections
    from histo_to_ccf.sectioning.split import detect_sections

    img = load_image(image)
    sections = detect_sections(
        img,
        min_area_px=min_area_px,
        closing_radius_px=closing_radius_px,
        expected_count=expected_count if expected_count > 0 else None,
    )
    ordered = order_sections(sections)
    h, w = img.shape[:2]

    sidecar = {
        "image_path": str(image),
        "image_size_px": [int(w), int(h)],
        "sections": [
            {
                "index": o.ap_order,
                "row": o.row,
                "col": o.col,
                "ap_order": o.ap_order,
                "bbox_px": list(o.section.bbox_px),
                "centroid_px": list(o.section.centroid_px),
                "area_px": o.section.area_px,
                "aspect_ratio": round(o.section.aspect_ratio, 3),
            }
            for o in ordered
        ],
    }

    image_path = Path(image)
    if output_json is None:
        output_json = image_path.with_suffix(".sections.json")
    output_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    typer.echo(f"detected {len(ordered)} section(s) -> {output_json}")


@app.command("register-one")
def register_one_cmd(
    image: Annotated[Path, typer.Option(help="Path to the slide image (TIFF/PNG).")],
    ap_um: Annotated[float, typer.Option(help="AP position of this section in CCF µm.")],
    tip: Annotated[str, typer.Option(help="Shank tip pixel as 'x,y' in slide coords.")],
    entry: Annotated[str, typer.Option(help="Entry point pixel as 'x,y' in slide coords.")],
    midline_px: Annotated[float, typer.Option(help="Midline pixel (x) of this section.")],
    dorsal_surface_px: Annotated[
        float, typer.Option(help="Dorsal surface pixel (y) of this section.")
    ],
    pixel_size_um: Annotated[float, typer.Option(help="µm per pixel.")] = 1.0,
    bbox: Annotated[
        str, typer.Option(help="Section bbox in slide pixels: 'x0,y0,x1,y1'.")
    ] = "",
    probe_label: Annotated[str, typer.Option(help="Probe label.")] = "probe1",
    probe_type: Annotated[
        str, typer.Option(help="Probe type name (free-form).")
    ] = "neuropixels-1.0",
    n_shanks: Annotated[int, typer.Option(help="Number of shanks on the probe.")] = 1,
    flip_lr: Annotated[
        bool,
        typer.Option(
            "--flip-lr/--no-flip-lr",
            help="Set if image-right is anatomical LEFT (mirrored).",
        ),
    ] = False,
    output_json: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Where to write the project JSON. Defaults to "
                "<image_dir>/<image_stem>.histo2ccf.json."
            ),
        ),
    ] = None,
    output_pkl: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Optional HERBS-compatible pkl output. Pass a path, or 'auto' to "
                "write <image_dir>/<image_stem>.pkl."
            ),
        ),
    ] = None,
    n_pkl_samples: Annotated[
        int, typer.Option(help="Samples along the tip→entry line in the pkl.")
    ] = 128,
) -> None:
    """Headless single-section registration - the M1 vertical slice.

    Loads a section from an image, applies a manual plane prediction, maps a
    shank tip and entry from section pixels to CCF µm, and writes both a
    canonical project JSON and (optionally) a HERBS-compatible pkl.
    """
    import numpy as np

    from histo_to_ccf.io.herbs_writer import write_herbs_pkl
    from histo_to_ccf.io.image import load_image
    from histo_to_ccf.project.io import save_project
    from histo_to_ccf.project.schema import (
        AtlasRef,
        PlaneParams,
        Point2D,
        ProbeSpec,
        ProbeType,
        Project,
        Section,
        Shank,
        Slide,
    )
    from histo_to_ccf.registration.pipeline import register_project
    from histo_to_ccf.registration.predictor import ManualPredictor

    tip_xy = _parse_xy(tip, "tip")
    entry_xy = _parse_xy(entry, "entry")

    if bbox:
        bbox_tuple = _parse_bbox(bbox)
    else:
        img = load_image(image)
        h, w = img.shape[:2]
        bbox_tuple = (0, 0, int(w), int(h))
        logger.info("no bbox provided - using full image extent {}x{}", w, h)

    plane = PlaneParams(
        ap_um=ap_um,
        midline_px=midline_px,
        dorsal_surface_px=dorsal_surface_px,
        pixel_size_um=pixel_size_um,
        image_right_is_anatomical_right=not flip_lr,
    )
    section = Section(index=0, slide_idx=0, bbox_px=bbox_tuple, ap_order=0, plane=plane)
    slide = Slide(image_path=str(image), sections=[section])
    probe = ProbeSpec(
        label=probe_label,
        type=ProbeType(name=probe_type, n_shanks=n_shanks),
        shanks=[
            Shank(
                index=0,
                tip_px=Point2D(x_px=tip_xy[0], y_px=tip_xy[1]),
                tip_section_idx=0,
                entry_px=Point2D(x_px=entry_xy[0], y_px=entry_xy[1]),
                entry_section_idx=0,
            )
        ],
    )
    project = Project(atlas=AtlasRef(), slides=[slide], probes=[probe])

    predictor = ManualPredictor(plane)
    register_project(project, predictor)

    shank = project.probes[0].shanks[0]
    typer.echo(f"tip   CCF (AP, ML, DV) um: {shank.tip_ccf_um}")
    typer.echo(f"entry CCF (AP, ML, DV) um: {shank.entry_ccf_um}")

    image_path = Path(image)
    if output_json is None:
        output_json = image_path.with_suffix(".histo2ccf.json")
    save_project(project, output_json)
    typer.echo(f"wrote project -> {output_json}")

    if output_pkl is not None:
        if str(output_pkl).lower() == "auto":
            output_pkl = image_path.with_suffix(".pkl")
        if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
            raise typer.Exit(code=2)
        ccf = np.linspace(
            np.array(shank.entry_ccf_um, dtype=float),
            np.array(shank.tip_ccf_um, dtype=float),
            n_pkl_samples,
        )
        write_herbs_pkl(output_pkl, [ccf])
        typer.echo(f"wrote HERBS pkl -> {output_pkl}")


@app.command("register")
def register_cmd(
    project_json: Annotated[
        Path, typer.Argument(help="Path to the project JSON produced by register-one or the GUI.")
    ],
    atlas: Annotated[
        str, typer.Option(help="BrainGlobe atlas id (e.g. 'allen_mouse_25um').")
    ] = "allen_mouse_25um",
    bspline_grid: Annotated[
        str, typer.Option(help="B-spline control-point grid as 'NxM' (e.g. '8x8').")
    ] = "8x8",
    max_iterations: Annotated[
        int, typer.Option(help="Max LBFGSB iterations per section.")
    ] = 100,
    transforms_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory for .tfm sidecar files. "
                "Defaults to <project_dir>/transforms/."
            )
        ),
    ] = None,
    output_json: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Where to write the updated project JSON. "
                "Defaults to overwriting the input file."
            )
        ),
    ] = None,
    reuse_anchoring: Annotated[
        bool,
        typer.Option(
            help=(
                "Re-use each section's already-stored atlas plane instead of "
                "rebuilding a coronal one from its PlaneParams. Keeps a DeepSlice "
                "oblique plane intact. Turn off to re-derive the plane from a "
                "hand-edited AP/tilt."
            )
        ),
    ] = True,
) -> None:
    """Run the M3 atlas-registration pipeline on every section in a project.

    For each section that has a PlaneParams, resamples the atlas at that plane,
    refines the alignment with a 2D B-spline, and stores a RegistrationResult.
    The updated project JSON is written (overwriting by default).
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    from histo_to_ccf.project.images import section_images
    from histo_to_ccf.project.io import load_project, save_project
    from histo_to_ccf.registration.pipeline import register_project_with_atlas

    # Parse grid
    try:
        gn, gm = bspline_grid.lower().split("x")
        grid = (int(gn), int(gm))
    except Exception as e:
        raise typer.BadParameter(f"--bspline-grid expects 'NxM' (got {bspline_grid!r})") from e

    project = load_project(project_json)
    project_dir = project_json.parent

    logger.info("loading atlas {}", atlas)
    bg_atlas = BrainGlobeAtlas(atlas)

    if transforms_dir is None:
        transforms_dir = project_dir / "transforms"

    # Rebuild the exact pixels the stored bboxes were drawn against: merged
    # multi-source slides, whole-slide flips, per-section flips.
    images = section_images(project, base_dir=project_dir)

    logger.info(
        "registering {} section(s) with atlas={} grid={} reuse_anchoring={}",
        len(images), atlas, grid, reuse_anchoring,
    )
    register_project_with_atlas(
        project,
        bg_atlas,
        section_images=images,
        transforms_dir=transforms_dir,
        bspline_grid=grid,
        max_iterations=max_iterations,
        reuse_stored_anchoring=reuse_anchoring,
    )

    registered_count = sum(
        1
        for slide in project.slides
        for section in slide.sections
        if section.registration is not None
    )
    typer.echo(f"registered {registered_count} section(s)")

    out_path = output_json or project_json
    save_project(project, out_path)
    typer.echo(f"wrote project -> {out_path}")


@app.command("export")
def export_cmd(
    project_json: Annotated[
        Path, typer.Argument(help="Path to a registered project JSON.")
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory. Defaults to the project's own folder."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(help="Base name for the CSVs. Defaults to the project JSON stem."),
    ] = None,
    remap: Annotated[
        bool,
        typer.Option(
            help=(
                "Re-project every shank's tip/entry pixel through the stored "
                "transforms before exporting (needs the atlas). Off by default: the "
                "coordinates already saved in the project are used as-is."
            )
        ),
    ] = False,
    atlas: Annotated[
        str, typer.Option(help="BrainGlobe atlas id, used only with --remap.")
    ] = "allen_mouse_25um",
    rigid_array: Annotated[
        bool,
        typer.Option(
            help=(
                "Regularize each multi-shank probe to a parallel, evenly-spaced "
                "array before exporting."
            )
        ),
    ] = False,
    rigid_tolerance: Annotated[
        float,
        typer.Option(
            help="0 snaps exactly to the rigid array, 1 leaves picks unchanged."
        ),
    ] = 0.25,
    lock_spacing_um: Annotated[
        float | None,
        typer.Option(
            help="Force an exact inter-shank spacing in µm (e.g. 250 for NP2.0)."
        ),
    ] = None,
    paxinos_alignment: Annotated[
        str | None,
        typer.Option(help="CCF→stereotaxic preset (none/allen_forum/qiu2018/dorr2008)."),
    ] = None,
    save_project_to: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Also write the (re-mapped / regularized) project JSON here. "
                "Without this the project file is never modified."
            )
        ),
    ] = None,
) -> None:
    """Write per-channel CCF µm and Paxinos mm CSVs from a registered project.

    Regenerates coordinates from a project without re-running registration, so an
    export can be refreshed after the project has been corrected. Writes
    ``<name>.csv`` (CCF µm) and ``<name> - Paxinos.csv`` (stereotaxic mm).
    """
    from histo_to_ccf.probes.channels import export_channel_csv, export_paxinos_csv
    from histo_to_ccf.project.io import load_project, save_project
    from histo_to_ccf.project.provenance import write_export_provenance

    project = load_project(project_json)
    out_dir = out_dir or project_json.parent
    base = name or project_json.stem

    if remap:
        from brainglobe_atlasapi import BrainGlobeAtlas

        from histo_to_ccf.registration.pipeline import (
            _apply_to_shank_registered,
            reload_registered_transforms,
        )

        logger.info("loading atlas {}", atlas)
        bg_atlas = BrainGlobeAtlas(atlas)
        transforms = reload_registered_transforms(
            project, bg_atlas, project_dir=project_json.parent
        )
        n_remapped = 0
        for probe in project.probes:
            for shank in probe.shanks:
                _apply_to_shank_registered(shank, project, transforms)
                n_remapped += 1
        typer.echo(f"re-mapped {n_remapped} shank(s) from the stored transforms")

    if rigid_array:
        from histo_to_ccf.probes.fitting import enforce_rigid_arrays

        infos = enforce_rigid_arrays(
            project, tolerance=rigid_tolerance, lock_spacing_um=lock_spacing_um
        )
        for label, info in sorted(infos.items()):
            typer.echo(
                f"rigid array '{label}': spacing {info['spacing_um']:.0f} µm, "
                f"max shift {info['max_shift_um']:.0f} µm"
            )

    ccf_path = out_dir / f"{base}.csv"
    pax_path = out_dir / f"{base} - Paxinos.csv"
    n_ccf = export_channel_csv(project, ccf_path)
    n_pax = export_paxinos_csv(project, pax_path, alignment=paxinos_alignment)

    if n_ccf == 0:
        typer.echo(
            "warning: no shank had both a tip and an entry in CCF - nothing exported. "
            "Register the sections and map the probes first."
        )
    typer.echo(f"wrote {n_ccf} channel row(s) -> {ccf_path}")
    typer.echo(f"wrote {n_pax} channel row(s) -> {pax_path}")

    prov_path = out_dir / f"{base}.provenance.json"
    write_export_provenance(
        prov_path,
        project_json=project_json,
        outputs=[ccf_path, pax_path],
        n_rows=n_ccf,
        options={
            "remap": remap,
            "rigid_array": rigid_array,
            "rigid_tolerance": rigid_tolerance if rigid_array else None,
            "lock_spacing_um": lock_spacing_um if rigid_array else None,
            "paxinos_alignment": paxinos_alignment,
        },
    )
    typer.echo(f"wrote provenance -> {prov_path}")

    if save_project_to is not None:
        save_project(project, save_project_to)
        typer.echo(f"wrote project -> {save_project_to}")


if __name__ == "__main__":
    app()
