"""User-supplied fallback landmark anchors.

When auto-landmark detection fails (e.g., a damaged section with no visible
ventricles), the user can drop a few labeled points by hand. The registration
pipeline uses these as additional B-spline constraints.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AnchorPoint(BaseModel):
    """One user-placed landmark on a section."""

    model_config = ConfigDict(frozen=True)

    label: str  # e.g. "midline_top", "ventricle_left", "rostral_tip"
    x_px: float
    y_px: float


class AnchorSet(BaseModel):
    """All user anchors for one section."""

    section_index: int
    points: list[AnchorPoint] = []
