"""AnchorPoint / AnchorSet serialization."""
from __future__ import annotations

from atlastrack.landmarks.anchors import AnchorPoint, AnchorSet


def test_anchor_set_roundtrip() -> None:
    aset = AnchorSet(
        section_index=2,
        points=[
            AnchorPoint(label="midline_top", x_px=512.0, y_px=80.0),
            AnchorPoint(label="ventricle_left", x_px=480.0, y_px=220.0),
        ],
    )
    payload = aset.model_dump_json()
    restored = AnchorSet.model_validate_json(payload)
    assert restored == aset


def test_anchor_point_is_frozen() -> None:
    p = AnchorPoint(label="x", x_px=1.0, y_px=2.0)
    try:
        p.x_px = 3.0  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("expected frozen model to reject mutation")
