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
    midline_px: float = 0.0
    dorsal_surface_px: float = 0.0
    pixel_size_um: float = 1.0
    image_right_is_anatomical_right: bool = True


class RegistrationResult(BaseModel):
    """Per-section registration outputs from the M3 pipeline.

    ``anchoring`` is the QuickNII 9-vector defining the atlas plane.
    ``bspline_transform_path`` points to a SimpleITK ``.tfm`` (project-relative).
    ``residual`` is the final optimizer metric (lower = better).
    ``output_size_px`` is ``(height, width)`` of the resampled atlas slice the
    B-spline transform is defined against.
    """

    anchoring: list[float] = Field(min_length=9, max_length=9)
    output_size_px: tuple[int, int]
    bspline_transform_path: str | None = None
    residual: float | None = None


class ChannelLevels(BaseModel):
    """Per-channel intensity levels for display normalisation (0–1 scale)."""

    low: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    high: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])


class Section(BaseModel):
    """One brain section extracted from a slide image."""

    index: int = Field(ge=0)
    slide_idx: int = Field(ge=0)
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1) in slide coords
    ap_order: int = 0
    plane: PlaneParams | None = None
    # Filled by M3 pipeline; absent for M1 manual-mode sections.
    registration: RegistrationResult | None = None
    flip_h: bool = False
    flip_v: bool = False
    levels: ChannelLevels | None = None


class Slide(BaseModel):
    """One slide image — typically holds several Sections.

    When several source images are opened they are merged into a single combined
    image (so every section shares one coordinate space). ``image_path`` is the
    primary/first source; ``source_paths`` lists all sources in merge order so
    the exact combined image can be reproduced on reload. For a single-image
    slide ``source_paths`` is empty and ``image_path`` is used directly.
    """

    image_path: str
    source_paths: list[str] = []
    sections: list[Section] = []
    flip_h: bool = False
    flip_v: bool = False
    levels: ChannelLevels | None = None


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
