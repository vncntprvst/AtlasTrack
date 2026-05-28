"""Read and write QuickNII anchoring JSON.

QuickNII / VisuAlign / DeepSlice all share the same JSON schema for per-section
atlas-plane parameters:

    {
        "name": "MyExperiment",
        "target": "ABA_Mouse_CCFv3",
        "target-resolution": [528, 320, 456],
        "slices": [
            {
                "filename": "section_001.png",
                "nr": 1,
                "width": 1024,
                "height": 800,
                "anchoring": [ox, oy, oz, ux, uy, uz, vx, vy, vz]
            },
            ...
        ]
    }

Storing in this format lets users round-trip through QuickNII or VisuAlign
without losing data, and lets DeepSlice predictions drop straight into our
pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from histo_to_ccf.atlas.planes import Anchoring


class QuickNiiSlice(BaseModel):
    """One QuickNII slice entry."""

    model_config = ConfigDict(populate_by_name=True)

    filename: str
    nr: int = 1
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    anchoring: list[float] = Field(min_length=9, max_length=9)

    def get_anchoring(self) -> Anchoring:
        return Anchoring.from_iterable(self.anchoring)


class QuickNiiDocument(BaseModel):
    """A QuickNII JSON document — one experiment, many slices."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = "histo2ccf"
    target: str = "ABA_Mouse_CCFv3"
    target_resolution: list[int] = Field(
        default_factory=lambda: [528, 320, 456], alias="target-resolution"
    )
    slices: list[QuickNiiSlice] = []


def load_quicknii(path: str | Path) -> QuickNiiDocument:
    """Load a QuickNII JSON file."""
    return QuickNiiDocument.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_quicknii(doc: QuickNiiDocument, path: str | Path) -> Path:
    """Write ``doc`` to ``path`` as canonical QuickNII JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        doc.model_dump_json(indent=2, by_alias=True),
        encoding="utf-8",
    )
    return p
