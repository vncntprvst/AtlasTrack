"""Every layer drawn over the slide must composite, not depth-test.

This has been got wrong three times now - the atlas outline Labels layer, the
section-number Points layer, and the landmark Points layer - each time with the
same symptom: the layer exists, is visible, holds the right data, sits at the top
of the stack, and draws nothing. napari's default ``translucent`` depth-tests
against the slide image at the same z and culls the overlay.

It is untestable by the usual means, because every property reads correct and a
headless canvas renders nothing at all, so it survives review and unit tests and
is caught only by a person looking at the screen. So this test reads the source
instead: any ``add_points`` / ``add_shapes`` / ``add_labels`` in the GUI package
has to pass ``blending=`` explicitly. That is a cheap, total check, and it fails
the moment someone adds a fourth overlay without it.

``add_image`` is exempt: the slide itself is the thing being drawn *on*, not an
annotation over it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

GUI_ROOT = Path(__file__).resolve().parents[1] / "src" / "atlastrack" / "gui"

#: Layer constructors that produce an annotation drawn over the slide.
_OVERLAY_CTORS = {"add_points", "add_shapes", "add_labels"}


def _overlay_calls(path: Path):
    """Yield ``(lineno, func_name, has_blending)`` for each overlay constructor."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in _OVERLAY_CTORS:
            continue
        has = any(kw.arg == "blending" for kw in node.keywords)
        # ``**kwargs`` may carry it; those call sites build the dict nearby and are
        # checked by the dict literal containing "blending" in the same file.
        splat = any(kw.arg is None for kw in node.keywords)
        yield node.lineno, name, has or splat


def _gui_modules() -> list[Path]:
    return sorted(p for p in GUI_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_there_are_overlay_layers_to_check() -> None:
    """Guard the guard: an empty scan would pass while proving nothing."""
    total = sum(len(list(_overlay_calls(p))) for p in _gui_modules())
    assert total >= 8, f"expected the GUI to build several overlays, found {total}"


@pytest.mark.parametrize("module", _gui_modules(), ids=lambda p: p.name)
def test_every_overlay_layer_composites_over_the_slide(module: Path) -> None:
    missing = [
        f"{module.name}:{lineno} {name}(...) has no blending="
        for lineno, name, has_blending in _overlay_calls(module)
        if not has_blending
    ]
    assert not missing, (
        "These layers will be depth-tested against the slide and may draw nothing.\n"
        "Pass blending=OVERLAY_BLENDING (atlastrack.gui.overlay_style):\n  "
        + "\n  ".join(missing)
    )


def test_a_kwargs_dict_carrying_blending_counts() -> None:
    """The section numbers are built through a kwargs dict, not a literal call."""
    app = GUI_ROOT / "app.py"
    source = app.read_text(encoding="utf-8")
    assert '"blending": OVERLAY_BLENDING' in source, (
        "app.py builds its Points layer from a kwargs dict; that dict must carry "
        "the blend, or the ** splat exemption above hides a missing one"
    )
