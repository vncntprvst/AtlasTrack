"""Subprocess entry point: run DeepSlice on a folder of section crops.

Usage::

    python -m atlastrack.registration.deepslice_run <workdir> [species]

``<workdir>`` must already contain the section images named ``section_s<idx>.png``
(written by :func:`atlastrack.registration.deepslice_adapter.predict_anchorings`).
DeepSlice's QuickNII predictions are written to
``<workdir>/deepslice_predictions.json``.

Why a separate process: DeepSlice pulls in TensorFlow, whose resident memory
(~1–2 GB) would otherwise stay allocated in the GUI process for the rest of the
session and stack on top of the memory-heavy atlas registration - enough to OOM
the app on machines with other things open. Running it as its own process means
all that memory is released the moment this exits, before registration runs.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: deepslice_run <workdir> [species]", file=sys.stderr)
        return 2
    workdir = Path(argv[0])
    species = argv[1] if len(argv) > 1 else "mouse"

    # Imported here (not at module top) so importing this module is cheap and
    # TensorFlow only loads when the subprocess actually runs.
    from atlastrack.registration.deepslice_adapter import DeepSlicePredictor

    DeepSlicePredictor(species).predict_folder(workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
