"""Smoke: the top-level package imports without pulling Qt or napari."""
from __future__ import annotations


def test_top_level_import() -> None:
    import atlastrack

    assert atlastrack.__version__


def test_core_subpackages_import() -> None:
    """Ensure each core subpackage is importable on its own."""
    import atlastrack.atlas  # noqa: F401
    import atlastrack.io  # noqa: F401
    import atlastrack.landmarks  # noqa: F401
    import atlastrack.probes  # noqa: F401
    import atlastrack.project  # noqa: F401
    import atlastrack.registration  # noqa: F401
    import atlastrack.sectioning  # noqa: F401
    import atlastrack.viz  # noqa: F401
