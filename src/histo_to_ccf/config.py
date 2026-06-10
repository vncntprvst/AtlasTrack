"""App-wide paths, atlas cache config, logging setup, and persisted preferences."""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Runtime settings (env-vars / .env)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Persisted user preferences (~/.histo2ccf/settings.json)
# ---------------------------------------------------------------------------

_PREFS_DIR = Path.home() / ".histo2ccf"
_PREFS_FILE = _PREFS_DIR / "settings.json"


class AppSettings(BaseModel):
    """GUI preferences persisted between sessions."""

    last_atlas_id: str = "allen_mouse_25um"
    last_project_dir: str = ""
    atlas_dir: str = ""  # where BrainGlobe atlases are stored; "" = default
    bspline_grid: int = 8
    max_iterations: int = 100
    section_spacing_um: float = 80.0
    # Registration engine: "auto" (elastix if installed, else SimpleITK),
    # "elastix" (masked + bending-energy regularized), or "sitk".
    reg_engine: str = "auto"
    bending_energy_weight: float = 20.0  # elastix smoothness penalty
    use_tissue_mask: bool = True  # restrict the metric to tissue (elastix)
    prealign_similarity: bool = True  # per-section silhouette pre-align (elastix)
    boundary_snap: bool = True  # snap atlas outer contour onto tissue after fit

    @field_validator("bspline_grid")
    @classmethod
    def _clamp_grid(cls, v: int) -> int:
        return max(4, min(v, 24))

    @field_validator("max_iterations")
    @classmethod
    def _clamp_iter(cls, v: int) -> int:
        return max(10, min(v, 500))

    @field_validator("bending_energy_weight")
    @classmethod
    def _clamp_bending(cls, v: float) -> float:
        return max(0.0, min(v, 500.0))

    @field_validator("reg_engine")
    @classmethod
    def _valid_engine(cls, v: str) -> str:
        return v if v in {"auto", "elastix", "sitk"} else "auto"


def load_app_settings() -> AppSettings:
    """Load persisted preferences; return defaults on any error."""
    if _PREFS_FILE.exists():
        try:
            data = json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
            return AppSettings.model_validate(data)
        except Exception:
            pass
    return AppSettings()


def save_app_settings(settings: AppSettings) -> None:
    """Persist user preferences to disk."""
    _PREFS_DIR.mkdir(parents=True, exist_ok=True)
    _PREFS_FILE.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
