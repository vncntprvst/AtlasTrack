"""Command-line interface. `histo2ccf` is the entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


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
def gui() -> None:
    """Launch the napari-based GUI (milestone M4 — not yet implemented)."""
    from histo_to_ccf.gui.app import launch

    launch()


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
    """Headless single-section registration — the M1 vertical slice.

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
        logger.info("no bbox provided — using full image extent {}x{}", w, h)

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


if __name__ == "__main__":
    app()
