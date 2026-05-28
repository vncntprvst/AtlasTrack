"""Persisted project state (Pydantic v2).

The ``Project`` model is the single source of truth that the CLI and (later)
the GUI both read and write. JSON serialization is via :mod:`pydantic` directly.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Point2D(BaseModel):
    """A pixel coordinate inside a section."""

    model_config = ConfigDict(frozen=True)
    x_px: float
    y_px: float


class ProbeType(BaseModel):
    """Probe geometry parameters needed for trajectory + per-channel mapping."""

    name: str
    n_shanks: int = Field(ge=1)
    shank_pitch_um: float = 250.0
    shank_width_um: float = 70.0
    shank_thickness_um: float = 24.0
    shank_tip_length_um: float = 175.0


class Shank(BaseModel):
    """One shank annotation + its registered CCF coordinates."""

    index: int = Field(ge=0)
    tip_px: Point2D | None = None
    tip_section_idx: int | None = None
    entry_px: Point2D | None = None
    entry_section_idx: int | None = None

    # Filled by the registration pipeline.
    tip_ccf_um: tuple[float, float, float] | None = None  # (AP, ML, DV)
    entry_ccf_um: tuple[float, float, float] | None = None


class ProbeSpec(BaseModel):
    """One probe instance — references a type and carries per-shank annotations."""

    label: str
    type: ProbeType
    shanks: list[Shank]


class PlaneParams(BaseModel):
    """Per-section coarse atlas plane.

    ``ap_um`` is the AP position of the section center in CCF µm. ``ml_tilt_deg``
    and ``dv_tilt_deg`` describe how the cutting plane is tipped relative to a
    pure coronal slice. ``midline_px`` and ``dorsal_surface_px`` anchor the
    section in pixel space so the in-plane mapping pixel → CCF µm is well-defined
    without DeepSlice. ``image_right_is_anatomical_right`` controls ML sign.
    """

    ap_um: float
    ml_tilt_deg: float = 0.0
    dv_tilt_deg: float = 0.0
    midline_px: float
    dorsal_surface_px: float
    pixel_size_um: float = 1.0
    image_right_is_anatomical_right: bool = True


class Section(BaseModel):
    """One brain section extracted from a slide image."""

    index: int = Field(ge=0)
    slide_idx: int = Field(ge=0)
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1) in slide coords
    ap_order: int = 0
    plane: PlaneParams | None = None
    # Path (project-relative) to a serialized B-spline displacement field, or None.
    bspline_displacement_path: str | None = None


class Slide(BaseModel):
    """One slide image — typically holds several Sections."""

    image_path: str
    sections: list[Section] = []


class AtlasRef(BaseModel):
    """Which atlas the project is registered against."""

    name: str = "allen_mouse_25um"
    resolution_um: float = 25.0


class Project(BaseModel):
    """Top-level project model — serialized to ``project.json``."""

    model_config = ConfigDict(validate_assignment=True)

    version: int = 1
    atlas: AtlasRef = Field(default_factory=AtlasRef)
    slides: list[Slide] = []
    probes: list[ProbeSpec] = []
