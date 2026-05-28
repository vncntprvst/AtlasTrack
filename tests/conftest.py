"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LEGACY_DIR = Path(__file__).parent.parent / "legacy"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to the bundled test-fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def legacy_dir() -> Path:
    """Path to the archived legacy code (HERBS scripts, old notebooks)."""
    return LEGACY_DIR
