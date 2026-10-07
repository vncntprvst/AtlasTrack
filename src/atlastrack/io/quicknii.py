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
pipeline. VisuAlign adds ``"markers"`` to a slice: ``[x, y, x', y']`` pairs in the
slice's ``width`` x ``height`` pixel frame, moving the point the plane puts at
``(x, y)`` to ``(x', y')``.

QuickNII's axes are ``(x, y, z)`` = (ML, AP, DV) running left -> right,
posterior -> anterior and inferior -> superior: every one the reverse of a
BrainGlobe ASR atlas. :func:`quicknii_to_atlas_anchoring` converts.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from atlastrack.atlas.planes import Anchoring

# QuickNII's Allen CCFv3 25 µm volume is 456 x 528 x 320 voxels (ML, AP, DV);
# here in (AP, DV, ML) order, to scale against a BrainGlobe atlas of the same family.
QUICKNII_DIMS_APDVML = (528, 320, 456)


def quicknii_to_atlas_anchoring(
    anchoring: Sequence[float],
    atlas_shape_apdvml: Sequence[int],
    *,
    quicknii_dims: Sequence[int] = QUICKNII_DIMS_APDVML,
) -> list[float]:
    """Convert a QuickNII/DeepSlice anchoring into a BrainGlobe-atlas anchoring.

    1. **Axis order.** QuickNII voxels are ``(ML, AP, DV)``; an
       :class:`~atlastrack.atlas.planes.Anchoring` is ``(AP, DV, ML)``.
    2. **Resolution.** Each axis is scaled from QuickNII's grid (``quicknii_dims``,
       (AP, DV, ML); 25 µm unless the file's ``target-resolution`` says otherwise)
       to the loaded atlas's grid.
    3. **Direction.** All three QuickNII axes run opposite to BrainGlobe's, so the
       origin becomes ``size - o`` and ``u``/``v`` are negated on every axis.

    The ML reversal follows PyNutil, the QUINT developers' own reader
    (``transpose([2, 0, 1])[::-1, ::-1, ::-1]`` between the two volumes). Without
    it a section comes out mirrored about the midline: the atlas is symmetric, so
    the overlay looks right, but ML and hemisphere are swapped.
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = anchoring
    o = [oy, oz, ox]
    u = [uy, uz, ux]
    v = [vy, vz, vx]
    scale = [atlas_shape_apdvml[k] / quicknii_dims[k] for k in range(3)]
    o = [atlas_shape_apdvml[k] - o[k] * scale[k] for k in range(3)]
    u = [-u[k] * scale[k] for k in range(3)]
    v = [-v[k] * scale[k] for k in range(3)]
    return [*o, *u, *v]


class QuickNiiSlice(BaseModel):
    """One QuickNII slice entry."""

    model_config = ConfigDict(populate_by_name=True)

    filename: str
    nr: int = 1
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    anchoring: list[float] = Field(min_length=9, max_length=9)
    # VisuAlign's non-linear adjustment, if any: [x, y, x', y'] per marker.
    markers: list[list[float]] | None = None

    def get_anchoring(self) -> Anchoring:
        return Anchoring.from_iterable(self.anchoring)


class QuickNiiDocument(BaseModel):
    """A QuickNII JSON document - one experiment, many slices."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = "atlastrack"
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
        doc.model_dump_json(indent=2, by_alias=True, exclude_none=True),
        encoding="utf-8",
    )
    return p
