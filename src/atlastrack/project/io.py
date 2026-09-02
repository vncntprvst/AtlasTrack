"""Load / save the persisted project JSON."""
from __future__ import annotations

from pathlib import Path

from histo_to_ccf.project.schema import Project


def load_project(path: str | Path) -> Project:
    return Project.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_project(project: Project, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    return p
