"""App-wide paths, atlas cache config, and logging setup."""
from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, populated from env vars (prefix HISTO2CCF_) or defaults."""

    model_config = SettingsConfigDict(env_prefix="HISTO2CCF_", env_file=".env", extra="ignore")

    atlas_cache_dir: Path = Path.home() / ".brainglobe"
    project_dir: Path = Path.cwd()
    default_atlas: str = "allen_mouse_25um"
    log_level: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=_settings.log_level)
    return _settings
