"""Save / export panel."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from atlastrack.gui.workflow import WorkflowState


class SavePanelWidget(QWidget):
    """Save / load the project JSON (the per-project configuration)."""

    def __init__(
        self,
        state: WorkflowState,
        on_project_loaded: Callable[[], None] | None = None,
        parent: QWidget | None = None,
        settings=None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        # Fired after a project is loaded so the viewer can redraw (wired in app).
        self._on_project_loaded = on_project_loaded
        # AppSettings (optional): used to reopen dialogs at the last-used folder and
        # to maintain the "Load recent" list. None in headless/tests.
        self._settings = settings
        self._build_ui()

    def _start_dir(self) -> str:
        """Folder the Load/Save dialogs should open in (last used, else cwd)."""
        if self._settings is not None:
            try:
                return self._settings.project_start_dir()
            except Exception:  # noqa: BLE001 - never let a bad pref block the dialog
                return ""
        return ""

    def _remember(self, path) -> None:
        """Record ``path`` as most-recent + persist, so the next dialog reopens here."""
        if self._settings is None:
            return
        try:
            from atlastrack.config import save_app_settings

            self._settings.remember_project(path)
            save_app_settings(self._settings)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Project JSON:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("project.atlastrack.json")
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save project")
        save_btn.clicked.connect(self._save)
        load_btn = QPushButton("Load project")
        load_btn.setToolTip(
            "Load a saved project (.atlastrack.json, or .histo2ccf.json from before "
            "the rename) - restores slides, sections, AP planes "
            "and the registration result (no need to re-run registration)."
        )
        load_btn.clicked.connect(self._load)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(load_btn)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project JSON",
            self._start_dir(),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def _save(self) -> None:
        raw = self._path_edit.text().strip()
        if not raw:
            # Default: next to the first slide's image
            slides = self._state.project.slides
            if slides:
                img_path = Path(slides[0].image_path)
                raw = str(img_path.with_suffix(".atlastrack.json"))
            else:
                self._status.setText("No slide loaded - provide a save path.")
                return

        out_path = Path(raw)
        self._state.project_path = out_path
        from atlastrack.project.io import save_project

        save_project(self._state.project, out_path)
        self._remember(out_path)
        self._status.setText(f"Saved → {out_path.name}")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project JSON", self._start_dir(),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self.load_path(path)

    def load_path(self, path: str | Path) -> bool:
        """Load a project from an explicit path (used by menu "Load recent").

        Returns True on success. Records the path in the recent list and fires the
        ``on_project_loaded`` callback so the canvas + tabs refresh, exactly like
        the file-dialog load.
        """
        from atlastrack.project.io import load_project

        try:
            project = load_project(path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Load failed: {exc}")
            return False
        self._state.project = project
        self._state.project_path = Path(path)
        self._path_edit.setText(str(path))
        self._remember(path)
        # Restore any stored DeepSlice planes, otherwise a re-register would
        # rebuild a flat coronal plane and lose the predicted tilt.
        self._state.deepslice_anchorings.clear()
        self._state.deepslice_fingerprints.clear()
        n_pre = self._state.seed_deepslice_cache_from_project()
        n_reg = sum(
            1 for slide in project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        pre = f", {n_pre} pre-matched plane(s) restored" if n_pre else ""
        self._status.setText(
            f"Loaded {Path(path).name} - {len(project.slides)} slide(s), "
            f"{n_reg} registered section(s){pre}."
        )
        if self._on_project_loaded is not None:
            self._on_project_loaded()
        return True
