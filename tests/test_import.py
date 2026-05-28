"""Smoke: the top-level package imports without pulling Qt or napari."""
from __future__ import annotations


def test_top_level_import() -> None:
    import histo_to_ccf

    assert histo_to_ccf.__version__


def test_core_subpackages_import() -> None:
    """Ensure each core subpackage is importable on its own."""
    import histo_to_ccf.atlas  # noqa: F401
    import histo_to_ccf.io  # noqa: F401
    import histo_to_ccf.landmarks  # noqa: F401
    import histo_to_ccf.probes  # noqa: F401
    import histo_to_ccf.project  # noqa: F401
    import histo_to_ccf.registration  # noqa: F401
    import histo_to_ccf.sectioning  # noqa: F401
    import histo_to_ccf.viz  # noqa: F401
