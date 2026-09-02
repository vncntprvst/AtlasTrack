"""Manual, tutorial and atlas reference, docked beside the workflow.

Deliberately a dock rather than modal dialogs: reading the manual should not block
the project, and a user part-way through a registration needs to glance at a recipe
and go straight back. napari tabifies it against the workflow dock, so switching is
one click and the project stays exactly as it was.

The manual and tutorial are the repository's own ``MANUAL.md`` / ``TUTORIAL.md``,
rendered rather than duplicated - documentation that is copied into the GUI is
documentation that goes stale.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

#: Tab titles, and the markdown file each renders. ``None`` = built in this module.
MANUAL = "Manual"
TUTORIAL = "Tutorial"
ATLASES = "Atlases"

_DOCS = {MANUAL: "MANUAL.md", TUTORIAL: "TUTORIAL.md"}


def find_doc(filename: str) -> Path | None:
    """Locate a top-level markdown doc, or None.

    Tried in order: the repository root above the installed package (the usual
    editable install), then the package directory itself, which is where the files
    would land if they were ever shipped inside the wheel.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / filename,   # <repo>/MANUAL.md
        here.parents[3] / filename,   # <site-packages>/atlastrack/../
        here.parents[2] / filename,   # atlastrack/MANUAL.md
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _browser() -> QTextBrowser:
    view = QTextBrowser()
    view.setOpenExternalLinks(True)
    return view


class HelpPanelWidget(QWidget):
    """The Help dock: one tab per document."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._views: dict[str, QTextBrowser] = {}
        #: page title -> the window it was popped out into, while it is out.
        self._detached: dict[str, QDialog] = {}
        #: page title -> its slot in the tab bar, so it re-docks where it was.
        self._order: dict[str, int] = {}
        self._tabs = QTabWidget()

        for title, filename in _DOCS.items():
            view = _browser()
            path = find_doc(filename)
            if path is None:
                # Say which file is missing and where it was expected. A blank tab
                # would read as "the manual is empty".
                view.setMarkdown(
                    f"**{filename} was not found.**\n\n"
                    "It ships in the repository root; this looks like an install "
                    "without the documentation files alongside the package."
                )
            else:
                view.setMarkdown(path.read_text(encoding="utf-8"))
            self._views[title] = view
            self._order[title] = self._tabs.addTab(view, title)

        atlases = _browser()
        self._views[ATLASES] = atlases
        self._order[ATLASES] = self._tabs.addTab(atlases, ATLASES)
        self.refresh_atlases()

        # A visible button rather than a gesture: someone who wants the manual on a
        # second screen should not have to discover a right-click to get it there.
        self._pop_out_btn = QPushButton("Open in a window")
        self._pop_out_btn.setToolTip(
            "Move the page in front into its own window, so it can sit beside the "
            "app or on another screen. Closing that window puts the page back here."
        )
        self._pop_out_btn.clicked.connect(self.detach_current)
        self._tabs.setCornerWidget(self._pop_out_btn)

        # Shown only once every page has been popped out, so the area is never
        # simply blank with nothing to explain it.
        self._empty = QLabel(
            "Every page is open in its own window.\n"
            "Close a window to bring that page back here."
        )
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color: palette(mid);")
        self._empty.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._tabs)
        layout.addWidget(self._empty)

    def refresh_atlases(self) -> None:
        """Re-render the atlas sheet, picking up the current theme and anchors."""
        from atlastrack.gui.widgets.atlas_help_dialog import (
            atlas_reference_html,
            link_colour_for,
        )

        self._views[ATLASES].setHtml(atlas_reference_html(link_colour_for(self)))

    # -- popping a page out into its own window ---------------------------

    def detach_current(self) -> QDialog | None:
        """Move the page in front into its own window. None if there is none."""
        index = self._tabs.currentIndex()
        if index < 0:
            return None
        return self.detach(self._tabs.tabText(index))

    def detach(self, title: str) -> QDialog | None:
        """Move ``title`` into a window, or raise the window it is already in."""
        if title in self._detached:
            window = self._detached[title]
            window.raise_()
            window.activateWindow()
            return window
        index = next(
            (i for i in range(self._tabs.count()) if self._tabs.tabText(i) == title),
            -1,
        )
        if index < 0:
            return None

        view = self._tabs.widget(index)
        self._tabs.removeTab(index)
        window = _DetachedPage(
            title, view, on_close=lambda: self.attach(title), parent=self
        )
        self._detached[title] = window
        window.show()
        self._update_empty_notice()
        return window

    def attach(self, title: str) -> None:
        """Put a popped-out page back, in the slot it came from."""
        if self._detached.pop(title, None) is None:
            return
        view = self._views[title]
        view.setParent(self._tabs)
        view.show()  # same trap in reverse: the dialog's close hides it.
        index = min(self._order.get(title, self._tabs.count()), self._tabs.count())
        self._tabs.insertTab(index, view, title)
        self._tabs.setCurrentIndex(index)
        self._update_empty_notice()

    def _update_empty_notice(self) -> None:
        empty = self._tabs.count() == 0
        self._tabs.setVisible(not empty)
        self._empty.setVisible(empty)

    @property
    def detached(self) -> list[str]:
        return sorted(self._detached)

    def show_page(self, title: str) -> bool:
        """Bring one page to the front, wherever it currently lives.

        A page that has been popped out is raised in its own window rather than
        pulled back into the tabs: the user put it there deliberately.
        """
        if title in self._detached:
            window = self._detached[title]
            window.raise_()
            window.activateWindow()
            return True
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == title:
                self._tabs.setCurrentIndex(i)
                return True
        return False

    @property
    def pages(self) -> list[str]:
        """Every page this panel owns, whether docked or popped out."""
        docked = [self._tabs.tabText(i) for i in range(self._tabs.count())]
        return sorted(docked + list(self._detached), key=lambda t: self._order[t])


class _DetachedPage(QDialog):
    """One help page in its own window, which re-docks when it is closed."""

    def __init__(self, title, view, *, on_close, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Window, True)
        self.resize(720, 760)
        self._on_close = on_close

        dock_back = QPushButton("Dock back")
        dock_back.setToolTip("Return this page to the Help tab.")
        dock_back.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(dock_back)

        layout = QVBoxLayout(self)
        layout.addWidget(view)
        layout.addLayout(buttons)
        # QTabWidget.removeTab hides the page *explicitly*, and Qt carries an
        # explicit hide across re-parenting - so without this the window comes up
        # holding a correctly-parented but invisible page, i.e. empty.
        view.show()

    def closeEvent(self, event):  # Qt signature
        self._on_close()
        super().closeEvent(event)
