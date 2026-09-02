"""Keep tooltips readable.

Qt lays a plain-text tooltip out on a single line however long it is, so the longer
explanatory tooltips in this app - the longest is 391 characters - render as a strip
of text wider than the screen, which is worse than no tooltip at all.

Rather than hand-breaking three dozen strings (and re-breaking them whenever one is
edited), :func:`wrap_tooltips` walks a built widget tree once and re-wraps the long
ones. Tooltips that already contain newlines are left alone: whoever wrote those
chose the breaks deliberately, usually to separate a description from a caveat.
"""
from __future__ import annotations

import textwrap

from qtpy.QtWidgets import QWidget

#: Characters per line. Roughly the width at which a tooltip stays scannable without
#: becoming a tall narrow column.
DEFAULT_WIDTH = 72

#: Below this a tooltip already fits comfortably, so wrapping only adds a ragged
#: second line.
MIN_LENGTH = 90


def wrap_tooltips(
    root: QWidget, *, width: int = DEFAULT_WIDTH, min_length: int = MIN_LENGTH
) -> int:
    """Re-wrap long single-line tooltips on ``root`` and every child widget.

    Returns how many were rewrapped. Idempotent: a rewrapped tooltip contains
    newlines, so a second pass skips it.
    """
    count = 0
    for widget in [root, *root.findChildren(QWidget)]:
        if _wrap_one(widget, width=width, min_length=min_length):
            count += 1
    return count


def _wrap_one(widget: QWidget, *, width: int, min_length: int) -> bool:
    try:
        text = widget.toolTip()
    except Exception:  # a deleted C++ object must not stop the walk
        return False
    if not text or "\n" in text or len(text) < min_length:
        return False
    # Rich text is Qt's own wrapping path; re-flowing it would break the markup.
    if text.lstrip().startswith("<"):
        return False
    widget.setToolTip(textwrap.fill(text, width))
    return True
