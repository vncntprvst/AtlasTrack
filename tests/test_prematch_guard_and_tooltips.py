"""Pre-match must not silently discard hand-set APs, and must not cry wolf.

Two separate complaints, both about the same button:

* Pre-match writes an AP to every section, so fine-tuning one by hand and then
  clicking Pre-match again threw that work away with no warning.
* It always claimed "first run downloads the model and is slow", on every run. A
  warning that is always shown carries no information, and here it was usually
  false: DeepSlice caches its weights on disk after the first download.
"""
from __future__ import annotations

import pytest

from histo_to_ccf.registration.deepslice_adapter import (
    deepslice_run_note,
    deepslice_weights_missing,
)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# "first run is slow"
# ---------------------------------------------------------------------------


def test_the_note_is_empty_once_the_weights_are_cached_and_loaded(monkeypatch):
    """Nothing to warn about: no download pending and the model is already up."""
    import sys

    monkeypatch.setattr(
        "histo_to_ccf.registration.deepslice_adapter.deepslice_weights_missing",
        lambda species="mouse": [],
    )
    monkeypatch.setitem(sys.modules, "DeepSlice", object())

    assert deepslice_run_note() == ""


def test_a_pending_download_is_called_a_download(monkeypatch):
    monkeypatch.setattr(
        "histo_to_ccf.registration.deepslice_adapter.deepslice_weights_missing",
        lambda species="mouse": ["weights/Allen_Mixed_Best.h5"],
    )

    note = deepslice_run_note()

    assert "download" in note
    assert "1 model file" in note


def test_cached_weights_but_an_unloaded_model_says_so(monkeypatch):
    """The honest middle case: no download, but TensorFlow still has to come up."""
    import sys

    monkeypatch.setattr(
        "histo_to_ccf.registration.deepslice_adapter.deepslice_weights_missing",
        lambda species="mouse": [],
    )
    monkeypatch.delitem(sys.modules, "DeepSlice", raising=False)

    note = deepslice_run_note()

    assert "download" not in note
    assert "session" in note


def test_an_unknown_install_does_not_claim_either_way(monkeypatch):
    monkeypatch.setattr(
        "histo_to_ccf.registration.deepslice_adapter.deepslice_weights_missing",
        lambda species="mouse": None,
    )

    assert "download" not in deepslice_run_note()


def test_checking_the_weights_does_not_import_deepslice():
    """Importing it drags in TensorFlow - seconds of delay to draw a status line.

    It would also poison the "is the model loaded" test this note depends on.
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "from histo_to_ccf.registration.deepslice_adapter import"
        " deepslice_weights_missing as f;"
        "f('mouse');"
        "print('DeepSlice' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip().endswith("False")


def test_the_real_install_reports_a_concrete_answer():
    """Not None here: DeepSlice is a test dependency, so its config is readable."""
    missing = deepslice_weights_missing("mouse")

    assert missing is not None
    assert all(m.startswith("weights/") for m in missing)


# ---------------------------------------------------------------------------
# Tooltip wrapping
# ---------------------------------------------------------------------------


def test_a_long_tooltip_is_wrapped(qtbot):
    from qtpy.QtWidgets import QCheckBox

    from histo_to_ccf.gui.widgets.tooltips import wrap_tooltips

    widget = QCheckBox("Enforce rigid array")
    qtbot.addWidget(widget)
    widget.setToolTip("word " * 90)

    assert wrap_tooltips(widget) == 1
    assert "\n" in widget.toolTip()
    assert max(len(ln) for ln in widget.toolTip().splitlines()) <= 72


def test_a_hand_broken_tooltip_is_left_alone(qtbot):
    """Those newlines were chosen to separate a description from its caveat."""
    from qtpy.QtWidgets import QCheckBox

    from histo_to_ccf.gui.widgets.tooltips import wrap_tooltips

    widget = QCheckBox("x")
    qtbot.addWidget(widget)
    original = "A long first line that goes on for a while and then stops.\nA caveat."
    widget.setToolTip(original)

    assert wrap_tooltips(widget) == 0
    assert widget.toolTip() == original


def test_a_short_tooltip_is_left_alone(qtbot):
    from qtpy.QtWidgets import QCheckBox

    from histo_to_ccf.gui.widgets.tooltips import wrap_tooltips

    widget = QCheckBox("x")
    qtbot.addWidget(widget)
    widget.setToolTip("Short enough already.")

    assert wrap_tooltips(widget) == 0


def test_wrapping_reaches_children_and_is_idempotent(qtbot):
    from qtpy.QtWidgets import QCheckBox, QVBoxLayout, QWidget

    from histo_to_ccf.gui.widgets.tooltips import wrap_tooltips

    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    child = QCheckBox("child")
    child.setToolTip("word " * 90)
    layout.addWidget(child)

    assert wrap_tooltips(root) == 1
    wrapped = child.toolTip()
    assert wrap_tooltips(root) == 0  # second pass finds nothing to do
    assert child.toolTip() == wrapped


def test_the_rigid_array_tooltip_comes_out_wrapped_in_the_real_panel(qtbot):
    """The tooltip that prompted this: 343 characters on one line."""
    import napari

    from histo_to_ccf.gui.app import _build_panel

    viewer = napari.Viewer(show=False)
    try:
        panel, viz_panel = _build_panel(viewer)
        qtbot.addWidget(panel)
        qtbot.addWidget(viz_panel)
        tip = viz_panel._rigid_check.toolTip()

        assert "\n" in tip
        assert max(len(ln) for ln in tip.splitlines()) <= 72
    finally:
        viewer.close()
