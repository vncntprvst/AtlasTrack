"""The figure shown on the empty canvas, in place of napari's welcome screen.

napari draws its own welcome overlay whenever the viewer holds no layers: the napari
logo, its version, a shortcut list and rotating tips pointing at the community chat
and ``File > Preferences``. None of that applies here - those menus are hidden (see
``app._keep_only_menus``) - and it is the first thing a new user sees, so it is
replaced with a schematic of this app's own workflow.

The figure is painted rather than shipped as an image so it stays crisp at any window
size and follows the napari theme. It is deliberately the *same* trigger as napari's:
visible exactly while the canvas is empty, so closing a project brings it back.
"""
from __future__ import annotations

from itertools import pairwise

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import QWidget

#: Window title and the name on the figure. One literal so the queued rename to
#: AtlasTrack has a single place to change.
APP_TITLE = "Histo-to-CCF"

_SUBTITLE = "histology  →  atlas  →  registered probe coordinates"

#: (tab name, what you do there). The order is the tab order in ``app._build_panel``;
#: if one moves, this must move with it or the figure teaches the wrong path.
STEPS: tuple[tuple[str, str], ...] = (
    ("Histology", "load slides,\ndetect sections"),
    ("Atlas", "pick an atlas,\nset each section's AP"),
    ("Register", "warp each section\nonto the atlas"),
    ("Probes", "click tip & entry\non the section"),
    ("Ephys", "align LFP/spike\ndepth to the track"),
)

_FOOTER = "Project ▸ Load Project  ·  Ctrl+O to reopen saved work"

#: Below this the diagram is unreadable, so only the title and tab names are drawn.
_COMPACT_BELOW_PX = 520

#: The figure is laid out for this size and scaled from it, so proportions hold.
_DESIGN_W, _DESIGN_H = 900.0, 560.0


class WelcomeOverlayWidget(QWidget):
    """Workflow schematic, sized to its parent and painted on demand.

    Created as a sibling of the canvas rather than a napari layer: a layer would
    itself make the canvas non-empty, would show up in exports and ``reset_view``,
    and would be cleared by *Close Project* - the moment the figure should return.
    """

    def __init__(self, parent: QWidget | None = None, *, theme: object = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._colors = _palette(theme)
        if parent is not None:
            parent.installEventFilter(self)
            self.setGeometry(parent.rect())

    # -- placement ---------------------------------------------------------

    #: Parent events that mean "the canvas is now a different size". ``Show`` is not
    #: redundant: Qt defers the resize event of a hidden widget until it is shown, so
    #: without it an overlay built before the window appears keeps its initial size.
    _FOLLOW_EVENTS = (QEvent.Resize, QEvent.Show)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Track the parent's size. The canvas has no resize signal to connect to."""
        if watched is self.parent() and event.type() in self._FOLLOW_EVENTS:
            self.setGeometry(watched.rect())
        return False

    def set_theme(self, theme: object) -> None:
        self._colors = _palette(theme)
        self.update()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        # Opaque: this stands in for the canvas, it does not tint it.
        painter.fillRect(self.rect(), self._colors["background"])

        w, h = self.width(), self.height()
        scale = _clamp(min(w / _DESIGN_W, h / _DESIGN_H), 0.55, 1.4)
        compact = w < _COMPACT_BELOW_PX

        block_h = (150.0 if compact else 300.0) * scale
        top = max(int(h * 0.5 - block_h * 0.5), int(20 * scale))

        y = self._draw_heading(painter, top, scale, compact)
        if compact:
            self._draw_tab_names(painter, y + int(18 * scale), scale)
        else:
            y = self._draw_steps(painter, y + int(46 * scale), scale)
            self._draw_footer(painter, y + int(34 * scale), scale)
        painter.end()

    def _draw_heading(self, painter: QPainter, top: int, scale: float, compact: bool) -> int:
        painter.setPen(QPen(self._colors["text"]))
        painter.setFont(_font(26 * scale, weight=QFont.DemiBold, spacing=2.5 * scale))
        line = int(38 * scale)
        painter.drawText(QRect(0, top, self.width(), line), Qt.AlignCenter, APP_TITLE)

        painter.setPen(QPen(self._colors["muted"]))
        painter.setFont(_font(11.5 * scale))
        sub = int(22 * scale)
        painter.drawText(
            QRect(0, top + line, self.width(), sub),
            Qt.AlignCenter,
            _SUBTITLE if not compact else "",
        )
        return top + line + (sub if not compact else 0)

    def _draw_steps(self, painter: QPainter, top: int, scale: float) -> int:
        """The numbered chain, then a caption under each step."""
        content = min(self.width() - int(48 * scale), int(860 * scale))
        x0 = (self.width() - content) // 2
        col = content / len(STEPS)
        radius = 15.0 * scale
        centre_y = top + radius

        centres = [QPoint(int(x0 + col * (i + 0.5)), int(centre_y)) for i in range(len(STEPS))]

        # Connector first, so the discs sit on top of it.
        painter.setPen(QPen(self._colors["line"], max(1.0, 1.6 * scale)))
        for left, right in pairwise(centres):
            painter.drawLine(
                QPoint(left.x() + int(radius) + 2, left.y()),
                QPoint(right.x() - int(radius) - 2, right.y()),
            )
        _draw_arrow_head(painter, centres[-1], radius, scale, self._colors["line"])

        name_font = _font(11.5 * scale, weight=QFont.DemiBold)
        body_font = _font(9.5 * scale)
        name_h = int(20 * scale)
        body_h = int(62 * scale)  # three wrapped lines at the smallest scale
        name_top = int(centre_y + radius + 12 * scale)

        for i, (centre, (name, blurb)) in enumerate(zip(centres, STEPS, strict=True)):
            painter.setBrush(self._colors["accent"])
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(centre, int(radius), int(radius))

            painter.setPen(QPen(self._colors["background"]))
            painter.setFont(_font(12 * scale, weight=QFont.Bold))
            disc = QRect(
                centre.x() - int(radius), centre.y() - int(radius),
                int(radius * 2), int(radius * 2),
            )
            painter.drawText(disc, Qt.AlignCenter, str(i + 1))

            cell = QRect(int(x0 + col * i), name_top, int(col), name_h)
            painter.setPen(QPen(self._colors["text"]))
            painter.setFont(name_font)
            painter.drawText(cell, Qt.AlignHCenter | Qt.AlignTop, name)

            painter.setPen(QPen(self._colors["muted"]))
            painter.setFont(body_font)
            painter.drawText(
                QRect(cell.x() + int(4 * scale), name_top + name_h,
                      int(col) - int(8 * scale), body_h),
                Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
                blurb,
            )
        return name_top + name_h + body_h

    def _draw_tab_names(self, painter: QPainter, top: int, scale: float) -> None:
        """Too narrow for the chain: name the tabs and say where to start."""
        painter.setPen(QPen(self._colors["text"]))
        painter.setFont(_font(11 * scale, weight=QFont.DemiBold))
        line = int(20 * scale)
        painter.drawText(
            QRect(0, top, self.width(), line), Qt.AlignCenter,
            "  ·  ".join(name for name, _ in STEPS),
        )
        painter.setPen(QPen(self._colors["muted"]))
        painter.setFont(_font(9.5 * scale))
        painter.drawText(
            QRect(0, top + line, self.width(), line * 2),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            "Start in Histology, or reopen saved work with Ctrl+O.",
        )

    def _draw_footer(self, painter: QPainter, top: int, scale: float) -> None:
        painter.setPen(QPen(self._colors["muted"]))
        painter.setFont(_font(10 * scale))
        painter.drawText(
            QRect(0, top, self.width(), int(20 * scale)), Qt.AlignCenter, _FOOTER
        )


# ---------------------------------------------------------------------------
# Painting helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _font(point_size: float, *, weight: int | None = None, spacing: float = 0.0) -> QFont:
    font = QFont()
    font.setPointSizeF(max(6.0, point_size))
    if weight is not None:
        font.setWeight(weight)
    if spacing:
        font.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return font


def _draw_arrow_head(
    painter: QPainter, centre: QPoint, radius: float, scale: float, color: QColor
) -> None:
    """A head past the last disc: the chain ends in a result, it is not a cycle."""
    tip_x = centre.x() + radius + 16 * scale
    base_x = centre.x() + radius + 4 * scale
    half = 4.5 * scale
    path = QPainterPath()
    path.moveTo(tip_x, centre.y())
    path.lineTo(base_x, centre.y() - half)
    path.lineTo(base_x, centre.y() + half)
    path.closeSubpath()
    painter.setPen(Qt.NoPen)
    painter.setBrush(color)
    painter.drawPath(path)


def _palette(theme: object) -> dict[str, QColor]:
    """Colours from the napari theme, so the figure follows light/dark.

    ``theme`` is a napari theme name or model; anything unusable falls back to the
    dark palette rather than leaving the figure unreadable.
    """
    colors = {
        "background": QColor("#262930"),
        "text": QColor("#f0f1f2"),
        "muted": QColor("#868e93"),
        "line": QColor("#5a626c"),
        "accent": QColor("#6a7380"),
    }
    try:
        from napari.utils.theme import get_theme

        resolved = get_theme(theme) if isinstance(theme, str) else theme
        if resolved is None:
            return colors
        for key, attr in (
            ("background", "background"), ("text", "text"),
            ("muted", "secondary"), ("line", "primary"), ("accent", "highlight"),
        ):
            value = getattr(resolved, attr, None)
            if value is not None:
                colors[key] = QColor(str(value))
    except Exception:
        pass
    return colors
