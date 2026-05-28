"""QuickNII JSON read / write."""
from __future__ import annotations

import json
from pathlib import Path

from histo_to_ccf.io.quicknii import QuickNiiDocument, QuickNiiSlice, load_quicknii, save_quicknii


def test_round_trip(tmp_path: Path) -> None:
    doc = QuickNiiDocument(
        name="exp",
        target="ABA_Mouse_CCFv3",
        slices=[
            QuickNiiSlice(
                filename="sec_0.png",
                nr=1,
                width=512,
                height=384,
                anchoring=[10.0, 0.0, 0.0, 0.0, 0.0, 456.0, 0.0, 320.0, 0.0],
            ),
            QuickNiiSlice(
                filename="sec_1.png",
                nr=2,
                width=512,
                height=384,
                anchoring=[12.0, 0.0, 0.0, 0.0, 0.0, 456.0, 0.0, 320.0, 0.0],
            ),
        ],
    )
    out = tmp_path / "doc.json"
    save_quicknii(doc, out)

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["target-resolution"] == [528, 320, 456]
    assert len(on_disk["slices"]) == 2

    reloaded = load_quicknii(out)
    assert reloaded == doc


def test_anchoring_object_extraction() -> None:
    s = QuickNiiSlice(
        filename="x.png", width=1, height=1,
        anchoring=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    a = s.get_anchoring()
    assert a.ox == 1.0 and a.vz == 9.0
