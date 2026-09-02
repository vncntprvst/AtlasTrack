"""Coarse atlas-plane prediction.

Defines a :class:`PlanePredictor` protocol so we can swap in DeepSlice (M3) or a
manual predictor (M1) behind the same interface. The pipeline itself never
imports DeepSlice directly - only the adapter does.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from atlastrack.project.schema import PlaneParams


class PlanePredictor(Protocol):
    """A strategy for producing a :class:`PlaneParams` from a section image."""

    def predict(
        self, image: np.ndarray, *, section_index: int
    ) -> PlaneParams:  # pragma: no cover - protocol
        ...


class ManualPredictor:
    """Always returns the same caller-supplied PlaneParams.

    Used when the user already knows the section's AP plane (e.g., entered on
    the CLI) or as a fallback when DeepSlice is not installed.
    """

    def __init__(self, params: PlaneParams) -> None:
        self._params = params

    def predict(self, image: np.ndarray, *, section_index: int) -> PlaneParams:
        del image, section_index  # unused - that's the point
        return self._params
