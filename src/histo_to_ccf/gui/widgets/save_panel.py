"""Save / export panel."""
from __future__ import annotations

from pathlib import Path

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
    """Save project JSON and optionally export HERBS pkl."""

    def __init__(self, state: WorkflowState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
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

        save_btn = QPushButton("Save project")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

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
                self._status.setText("No slide loaded — provide a save path.")
                return

        out_path = Path(raw)
        self._state.project_path = out_path
        from histo_to_ccf.project.io import save_project

        save_project(self._state.project, out_path)
        self._status.setText(f"Saved → {out_path.name}")
