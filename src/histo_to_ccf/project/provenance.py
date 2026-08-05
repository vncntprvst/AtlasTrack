"""Record which project version produced an export.

A registered project keeps getting corrected after coordinates have been exported
from it. Nothing in a bare CSV says which version of the project it came from, so
a stale export is invisible: on LO_06 the shipped per-channel CSV predated an AP
correction that moved every channel by ~350 µm on average, and only the file
mtimes gave it away.

Every export therefore writes a small sidecar recording the source project, its
content hash and modification time, the tool version, and the options used.
:func:`describe_staleness` compares a sidecar against the project as it stands
now, so "is this CSV current?" is answerable without re-deriving the coordinates.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def write_export_provenance(
    path: str | Path,
    *,
    project_json: str | Path,
    outputs: list[str | Path],
    n_rows: int,
    options: dict[str, Any] | None = None,
) -> Path:
    """Write the sidecar describing one export. Returns the path written."""
    from histo_to_ccf import __version__

    project_json = Path(project_json)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "tool": "histo2ccf",
        "tool_version": __version__,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_project": str(project_json),
        "source_sha256": _file_digest(project_json),
        "source_modified": _mtime_iso(project_json),
        "n_channel_rows": int(n_rows),
        "outputs": [Path(o).name for o in outputs],
        "options": options or {},
    }
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out_path


def describe_staleness(provenance_path: str | Path) -> tuple[bool, str]:
    """Is the export described by this sidecar still current?

    Returns ``(is_stale, message)``. An export is stale when the project it came
    from has since changed content. A missing project can't be checked, so it is
    reported as not stale with an explanatory message.
    """
    prov_path = Path(provenance_path)
    try:
        record = json.loads(prov_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"could not read provenance {prov_path.name}: {exc}"

    project_json = Path(record.get("source_project", ""))
    if not project_json.is_file():
        return False, f"source project not found: {project_json}"

    current = _file_digest(project_json)
    if current == record.get("source_sha256"):
        return False, f"up to date with {project_json.name}"
    return True, (
        f"STALE: {project_json.name} has changed since this export "
        f"(exported {record.get('exported_at', 'unknown')}); re-run `histo2ccf export`"
    )
