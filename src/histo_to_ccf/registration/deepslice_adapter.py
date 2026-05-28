"""Optional DeepSlice plane predictor.

Wraps ``DeepSlice.DSModel`` so the rest of the pipeline can use it through the
:class:`PlanePredictor` protocol. The import happens lazily inside methods so
the base package installs without TensorFlow.

Install via the optional extra::

    pip install histo-to-ccf[deepslice]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from histo_to_ccf.io.quicknii import QuickNiiDocument, load_quicknii
from histo_to_ccf.project.schema import PlaneParams


class DeepSlicePredictor:
    """Predict per-section atlas planes by running DeepSlice on an image folder.

    Unlike :class:`ManualPredictor`, this predictor is *batch* — it must see
    a whole folder of sections to apply :func:`propagate_angles` and
    :func:`enforce_index_order`. Use :meth:`predict_folder` once per slide,
    then :meth:`predict_single` looks up the cached result for each section.
    """

    def __init__(self, species: str = "mouse", *, ensemble: bool = True) -> None:
        self.species = species
        self.ensemble = ensemble
        self._cache: dict[str, PlaneParams] = {}

    def predict_folder(
        self,
        section_dir: Path | str,
        *,
        propagate_angles: bool = True,
        enforce_order: bool = True,
        output_xml: Path | str | None = None,
    ) -> QuickNiiDocument:
        """Run DeepSlice on every image in ``section_dir`` and cache results."""
        from DeepSlice import DSModel  # imported lazily

        model = DSModel(self.species)
        model.predict(str(section_dir), ensemble=self.ensemble)
        if propagate_angles:
            model.propagate_angles()
        if enforce_order:
            model.enforce_index_order()
        if output_xml is None:
            output_xml = Path(section_dir) / "deepslice_predictions.json"
        model.save_predictions(str(output_xml))
        doc = load_quicknii(output_xml)
        self._cache = self._build_cache(doc)
        return doc

    def _build_cache(self, doc: QuickNiiDocument) -> dict[str, PlaneParams]:
        """We don't bother converting QuickNII anchoring back to PlaneParams.

        Downstream code reads the full QuickNiiDocument; we keep this method
        as a hook in case a future caller wants per-filename PlaneParams.
        """
        return {}

    def predict(self, image: np.ndarray, *, section_index: int) -> PlaneParams:
        """Implements PlanePredictor — but only after predict_folder ran.

        The image argument is ignored; we keyed by section_index via the
        folder run. If you need a real per-image API, call DeepSlice directly.
        """
        del image, section_index
        raise NotImplementedError(
            "DeepSlicePredictor is batch-only. Call predict_folder(section_dir) "
            "and consume the returned QuickNiiDocument directly."
        )
