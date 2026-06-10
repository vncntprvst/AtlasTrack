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

from histo_to_ccf.gui.workflow import WorkflowState


class SavePanelWidget(QWidget):
    """Save / load the project JSON (the per-project configuration)."""

    def __init__(
        self,
        state: WorkflowState,
        on_project_loaded: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        # Fired after a project is loaded so the viewer can redraw (wired in app).
        self._on_project_loaded = on_project_loaded
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Project JSON:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("project.histo2ccf.json")
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save project")
        save_btn.clicked.connect(self._save)
        load_btn = QPushButton("Load project…")
        load_btn.setToolTip(
            "Load a saved .histo2ccf.json - restores slides, sections, AP planes "
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
            "",
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
                raw = str(img_path.with_suffix(".histo2ccf.json"))
            else:
                self._status.setText("No slide loaded - provide a save path.")
                return

        out_path = Path(raw)
        self._state.project_path = out_path
        from histo_to_ccf.project.io import save_project

        save_project(self._state.project, out_path)
        self._status.setText(f"Saved → {out_path.name}")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project JSON", "", "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        from histo_to_ccf.project.io import load_project

        try:
            project = load_project(path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Load failed: {exc}")
            return
        self._state.project = project
        self._state.project_path = Path(path)
        self._path_edit.setText(path)
        n_reg = sum(
            1 for slide in project.slides
            for sec in slide.sections
            if sec.registration is not None
        )
        self._status.setText(
            f"Loaded {Path(path).name} - {len(project.slides)} slide(s), "
            f"{n_reg} registered section(s)."
        )
        if self._on_project_loaded is not None:
            self._on_project_loaded()
