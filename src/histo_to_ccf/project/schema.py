"""Persisted project state (Pydantic v2).

The ``Project`` model is the single source of truth that the CLI and (later)
the GUI both read and write. JSON serialization is via :mod:`pydantic` directly.
"""
from __future__ import annotations

from typing import Literal

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


class EphysEpoch(BaseModel):
    """One excerpt of a recording, and whether it is fit to use.

    Alignment features are built from a handful of short windows rather than the
    whole recording: it is far cheaper, and the useful signal is in *responsive*
    periods anyway. Rejected windows are kept (not dropped) with the reason, so a
    figure can be regenerated exactly and a questionable rejection can be reviewed.
    """

    t_start_s: float
    t_end_s: float
    kept: bool = True
    reject_reason: str | None = None


class EphysRecordingRef(BaseModel):
    """One recording contributing ephys features to a penetration.

    A single Neuropixels 2.0 bank spans only ~720 µm of shank (96 sites, 2 columns,
    15 µm row pitch) against insertion depths of 4.5-5.4 mm, so one recording
    constrains a small fraction of the track. Several recordings on the same
    insertion - different banks, or the probe advanced between them - together cover
    most of it, which is why these belong to the **probe**, not to one shank.

    The two fields that place a recording on the shared axis:

    * ``electrode_range`` gives ``bank_offset_um`` (how far up the shank its sites
      start),
    * ``insertion_depth_um`` gives how deep the tip was for *this* recording.

    See :func:`histo_to_ccf.ephys.recordings.depth_below_surface_um` - depth below
    the brain surface is the only axis on which recordings taken at different
    insertion depths can be compared.
    """

    path: str
    label: str = ""
    stream_name: str | None = None
    # SpikeInterface SortingAnalyzer / AIND postprocessed zarr. Spike depths and
    # amplitudes come from here, so most features need no raw access.
    analyzer_path: str | None = None
    insertion_depth_um: float = 0.0
    # 1-based inclusive electrode numbers, as the lab records them ("all shanks
    # 97-192"). None means the bank starts at the tip.
    electrode_range: tuple[int, int] | None = None
    # Derived from electrode_range by default; overridable when the numbering is
    # known to be wrong (see the LO_04 2025-08-26 bank discrepancy in dataset.md).
    bank_offset_um: float | None = None
    epochs: list[EphysEpoch] = []


class EphysAlignment(BaseModel):
    """Ephys-based refinement of a shank's depth->CCF mapping.

    The user pins ephys feature depths to histology track depths (``anchors``);
    that piecewise-linear warp places each channel (at ``channel_depths_um`` from
    the tip) along the tip->entry line, giving ``channel_ccf_um``. Stored so the
    alignment can be reviewed/edited and the per-channel CCF reproduced on reload.
    """

    recording_path: str | None = None
    stream_name: str | None = None
    shank_x_um: float | None = None  # which shank column was selected (multi-shank)
    channel_depths_um: list[float] = []  # µm from tip, one per channel
    anchors: list[tuple[float, float]] = []  # (feature_depth_um, track_depth_um)
    channel_ccf_um: list[tuple[float, float, float]] = []  # (AP, ML, DV) per channel

    # The IBL landmark model, stored as two parallel arrays rather than pairs:
    # feature_um[i] (depth along the electrode array) corresponds to track_um[i]
    # (depth along the histology track). Equivalent in content to ``anchors`` but
    # in the form the fit/undo machinery and the IBL interchange files use.
    # ``anchors`` stays for projects written before this field.
    feature_um: list[float] = []
    track_um: list[float] = []
    # Kept so the result is reviewable without recomputing: which channels these
    # were, and what region each landed in.
    channel_ids: list[str] = []
    channel_regions: list[str] = []
    created_at: str | None = None


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

    # Filled by the ephys alignment tab (optional).
    ephys: EphysAlignment | None = None


class ProbeSpec(BaseModel):
    """One probe instance - references a type and carries per-shank annotations."""

    label: str
    type: ProbeType
    shanks: list[Shank]

    # Recordings contributing ephys features to this penetration (see
    # EphysRecordingRef - a bank covers only ~720 µm, so several are needed).
    recordings: list[EphysRecordingRef] = []

    # Ephys-derived refinement of the histology trajectory. Kept separate from the
    # histology-derived tip/entry so the original placement is never overwritten and
    # any departure from it stays visible and reportable.
    #
    # ``array_roll_deg`` is the rotation of the shank row about the track axis. For
    # LO_03 and LO_06 this is currently *assumed* (~45°) rather than observed - the
    # largest unverified assumption in the probe placement - and per-shank depth
    # alignments disagreeing linearly with shank index are what constrain it.
    track_offset_ccf_um: tuple[float, float, float] | None = None
    array_roll_deg: float | None = None


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
    # True when the masked fit failed and the section was rescued by an
    # unmasked retry. It IS registered, but without the label-excluding mask,
    # so it is worth a visual check. Defaults False, so projects written before
    # this field load unchanged.
    used_mask_fallback: bool = False


class ChannelLevels(BaseModel):
    """Per-channel intensity levels for display normalisation (0–1 scale)."""

    low: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    high: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])


class ManualLandmarks(BaseModel):
    """User-dragged correspondence points for a thin-plate-spline atlas warp.

    ``source`` = auto-placed landmark positions on the registered atlas overlay;
    ``target`` = where the user dragged each. Both are section-local (x, y) pixel
    coordinates. The TPS maps source -> target (overlay) / target -> source (probe
    mapping). Mutually exclusive with ``Section.manual_affine`` (landmarks win).
    """

    source: list[list[float]]
    target: list[list[float]]


class Section(BaseModel):
    """One brain section extracted from a slide image."""

    index: int = Field(ge=0)
    slide_idx: int = Field(ge=0)
    bbox_px: tuple[int, int, int, int]  # (x0, y0, x1, y1) in slide coords
    ap_order: int = 0
    # Physical serial number of the slide/section this came from, when the series
    # was sampled unevenly (e.g. every 2nd section here, every 5th there). AP is
    # then spaced by slide-number gaps rather than by position in the list.
    # None = the series is evenly sampled and ap_order alone determines spacing.
    slide_number: int | None = None
    plane: PlaneParams | None = None
    # Where ``plane.ap_um`` came from, so the matcher can show it and the user can
    # tell a prediction from something they set. None = never assigned.
    ap_source: Literal["deepslice", "manual", "even_spacing"] | None = None
    # DeepSlice's raw predicted plane (QuickNII 9-vector). ``PlaneParams`` can only
    # express a coronal plane plus two tilts, so without this the predicted
    # obliquity is lost on reload and a re-register silently flattens the plane.
    deepslice_anchoring: list[float] | None = None
    # Content signature of the crop the prediction was made from, so a reloaded
    # anchoring is still rejected if the section's image has since been swapped.
    deepslice_fingerprint: list[float] | None = None
    # Filled by M3 pipeline; absent for M1 manual-mode sections.
    registration: RegistrationResult | None = None
    flip_h: bool = False
    flip_v: bool = False
    levels: ChannelLevels | None = None
    # Manual post-registration correction of the atlas overlay: a 3x3 affine in
    # section-local (row, col) pixel coordinates (napari convention), mapping a
    # registered atlas position to where the user dragged it. None = identity.
    # Composed into the overlay and the probe->CCF mapping.
    manual_affine: list[list[float]] | None = None
    # Thin-plate-spline correction from dragged landmarks (richer, non-rigid).
    # Takes precedence over manual_affine when present.
    manual_landmarks: ManualLandmarks | None = None


class Slide(BaseModel):
    """One slide image - typically holds several Sections.

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
    """Top-level project model - serialized to ``project.json``."""

    model_config = ConfigDict(validate_assignment=True)

    version: int = 1
    atlas: AtlasRef = Field(default_factory=AtlasRef)
    slides: list[Slide] = []
    probes: list[ProbeSpec] = []
    # Inter-section AP spacing (µm) chosen in the ordering panel; persisted so it
    # reloads with the project. ``None`` until the user sets/applies a spacing.
    section_spacing_um: float | None = None
