"""Project JSON round-trip."""
from __future__ import annotations

from pathlib import Path

from atlastrack.project.io import load_project, save_project
from atlastrack.project.schema import (
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


def _project() -> Project:
    plane = PlaneParams(
        ap_um=5400.0,
        midline_px=500.0,
        dorsal_surface_px=100.0,
        pixel_size_um=2.0,
    )
    return Project(
        atlas=AtlasRef(),
        slides=[
            Slide(
                image_path="slide.tif",
                sections=[Section(index=0, slide_idx=0, bbox_px=(0, 0, 1000, 800), plane=plane)],
            )
        ],
        probes=[
            ProbeSpec(
                label="probe1",
                type=ProbeType(name="neuropixels-1.0", n_shanks=1),
                shanks=[
                    Shank(
                        index=0,
                        tip_px=Point2D(x_px=600.0, y_px=400.0),
                        tip_section_idx=0,
                        entry_px=Point2D(x_px=510.0, y_px=120.0),
                        entry_section_idx=0,
                    )
                ],
            )
        ],
    )


def test_save_then_load(tmp_path: Path) -> None:
    original = _project()
    p = tmp_path / "project.histo2ccf.json"
    save_project(original, p)
    reloaded = load_project(p)
    assert reloaded.model_dump() == original.model_dump()
