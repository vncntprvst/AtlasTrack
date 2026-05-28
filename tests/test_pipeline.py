"""End-to-end pipeline test: project + ManualPredictor → filled CCF coords."""
from __future__ import annotations

from histo_to_ccf.io.ccf_coords import MIDLINE_ML_UM
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


def test_register_one_section() -> None:
    plane = PlaneParams(
        ap_um=5400.0,
        midline_px=500.0,
        dorsal_surface_px=100.0,
        pixel_size_um=2.0,
    )
    project = Project(
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
                type=ProbeType(name="np1", n_shanks=1),
                shanks=[
                    Shank(
                        index=0,
                        tip_px=Point2D(x_px=600.0, y_px=400.0),  # right of midline, ventral
                        tip_section_idx=0,
                        entry_px=Point2D(x_px=510.0, y_px=120.0),
                        entry_section_idx=0,
                    )
                ],
            )
        ],
    )

    register_project(project, ManualPredictor(plane))

    shank = project.probes[0].shanks[0]
    assert shank.tip_ccf_um is not None
    assert shank.entry_ccf_um is not None

    ap, ml, dv = shank.tip_ccf_um
    assert ap == 5400.0
    assert ml == MIDLINE_ML_UM + 200.0  # (600-500)*2
    assert dv == 600.0  # (400-100)*2

    ap_e, ml_e, dv_e = shank.entry_ccf_um
    assert ap_e == 5400.0
    assert ml_e == MIDLINE_ML_UM + 20.0  # (510-500)*2
    assert dv_e == 40.0  # (120-100)*2
