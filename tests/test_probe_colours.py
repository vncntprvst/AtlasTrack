"""One colour per shank, the same in every view, and a probe's own colour."""
from __future__ import annotations

import pytest

from atlastrack.probes.colours import PALETTE, probe_colour, shank_colour
from atlastrack.project.schema import Project, ProbeSpec, ProbeType, Shank


def _project() -> Project:
    def probe(label, n):
        return ProbeSpec(label=label, type=ProbeType(name="x", n_shanks=n),
                         shanks=[Shank(index=i) for i in range(n)])

    return Project(probes=[probe("A", 2), probe("B", 3)])


def test_shanks_take_the_palette_in_order_across_probes() -> None:
    project = _project()
    assert [shank_colour(project, 0, s) for s in range(2)] == list(PALETTE[:2])
    assert [shank_colour(project, 1, s) for s in range(3)] == list(PALETTE[2:5])
    assert probe_colour(project, 1) == PALETTE[2]


def test_a_probe_colour_is_used_for_all_its_shanks_and_saved(tmp_path) -> None:
    from atlastrack.project.io import load_project, save_project

    project = _project()
    project.probes[1].color = "#123456"
    assert {shank_colour(project, 1, s) for s in range(3)} == {"#123456"}
    assert shank_colour(project, 0, 1) == PALETTE[1]  # the others keep theirs
    save_project(project, tmp_path / "p.json")
    assert load_project(tmp_path / "p.json").probes[1].color == "#123456"


@pytest.mark.qt
def test_the_colours_window_sets_and_resets_a_probe(qtbot) -> None:
    from atlastrack.gui.widgets.probe_colours_dialog import ProbeColoursDialog

    project = _project()
    changes = []
    dialog = ProbeColoursDialog(project, on_changed=lambda: changes.append(1))
    qtbot.addWidget(dialog)
    dialog._pick = lambda initial: "#00ff00"
    dialog._swatches[0].click()
    assert project.probes[0].color == "#00ff00" and changes == [1]
    dialog.set_colour(0, None)
    assert project.probes[0].color is None and changes == [1, 1]
    assert shank_colour(project, 0, 0) == PALETTE[0]
