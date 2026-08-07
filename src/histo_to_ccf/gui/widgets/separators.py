"""Small shared layout helpers for grouping panel controls.

The workflow panels are tall single columns of controls, and without any visual
breaks unrelated groups run together - "which of these does the Scope radio
apply to?" is the sort of question that follows. A titled rule answers it
without the heavy border of a full ``QGroupBox``.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


def hline() -> QFrame:
    """A plain horizontal rule."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def section_header(title: str, *, top_margin: int = 10) -> QWidget:
    """A titled rule: the title, then a rule filling the rest of the row.

    Used to head a run of related controls. ``top_margin`` is the breathing
    space above it - the point is to separate, so it is not optional-looking.
    """
    holder = QWidget()
    outer = QVBoxLayout(holder)
    outer.setContentsMargins(0, top_margin, 0, 2)
    outer.setSpacing(2)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    label = QLabel(title)
    label.setStyleSheet("QLabel { font-weight: bold; }")
    row.addWidget(label)
    # The rule takes the leftover width, centred against the text.
    row.addWidget(hline(), 1, Qt.AlignVCenter)
    outer.addLayout(row)
    return holder
