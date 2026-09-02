"""Reference sheet for the atlases the app can register into or label from.

Which atlas you pick changes both the region names you get out and, less visibly,
where bregma sits - and two of the four are shifted relative to Allen by amounts big
enough to matter (346 µm and 102 µm). Those numbers were measured for this app rather
than published anywhere, so this is the one place a user can see them, next to the
paper that describes each atlas.

The bregma column is read from :data:`histo_to_ccf.io.ccf_coords.BREGMA_AP_BY_ATLAS`
rather than written out again here, so the sheet cannot drift from the value the
exports actually use.
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

#: (display name, brainglobe id prefix, one-line description, [(link text, url)], note)
_ATLASES = [
    (
        "Allen Mouse Common Coordinate Framework v3",
        "allen_mouse",
        "The reference this app registers into. Ids: "
        "<code>allen_mouse_10um / 25um / 50um / 100um</code>.",
        [("Wang et al. 2020, Cell", "https://doi.org/10.1016/j.cell.2020.04.007")],
        "840 structures. The bregma anchor is the usual community estimate, not a "
        "measurement from the volume - it is the baseline every other row is "
        "expressed against.",
    ),
    (
        "CCFv3-BBP Augmented",
        "ccfv3augmented_mouse",
        "Allen CCFv3 with olfactory-bulb layers, cerebellar granular / molecular / "
        "Purkinje layers across 16 lobules, barrel columns and spinal-cord coverage "
        "added. Ids: <code>ccfv3augmented_mouse_10um / 25um</code>.",
        [
            ("Bolaños-Puchet et al. 2024",
             "https://pmc.ncbi.nlm.nih.gov/articles/PMC12319842/"),
            ("Data on Zenodo (latest)", "https://zenodo.org/records/18223882"),
        ],
        "566 AP slices rather than 528, and the same anatomy sits <b>+346 µm</b> "
        "along the AP axis. "
        "Section APs assigned under Allen do <b>not</b> carry across.",
    ),
    # (measured here: volume centroids of 25 compact nuclei, sd 2.5 µm)
    (
        "Chon / Kim Unified Mouse Brain Atlas",
        "kim_mouse",
        "Franklin-Paxinos nomenclature mapped onto the CCF grid - M1, S1BF, 4V "
        "rather than MOp, SSp-bfd, V4. Ids: "
        "<code>kim_mouse_10um / 25um / 50um / 100um</code>.",
        [
            ("Chon et al. 2019, Nature Communications",
             "https://www.nature.com/articles/s41467-019-13057-w"),
        ],
        "Voxel-identical to Allen 25 µm (528 x 320 x 456) with 1356 structures "
        "against Allen's 840, including Paxinos's CA1 laminar subdivisions. Because "
        "the grid is shared, selecting it as <i>Region atlas</i> in 3D Visualization relabels without "
        "re-registering and without approximating anything.",
    ),
    (
        "Chon / Kim Unified v2, isotropic",
        "kim_mouse_isotropic",
        "The 2024 re-release: same volume sampled at 20 µm, with corrected labels "
        "and an updated ontology. Id: <code>kim_mouse_isotropic_20um</code>.",
        [
            ("BrainGlobe announcement",
             "https://brainglobe.info/blog/kim-isotropic-mouse-brain-atlas-added.html"),
        ],
        "660 x 400 x 570 at 20 µm - the same physical volume, a different grid, so "
        "anchorings are rescaled for it. Its annotation sits <b>+102 µm</b> "
        "posterior of the 25 µm release.",
    ),
    # (measured here: volume centroids over 811 structures, +101.8 ± 26.6 µm, a pure translation)
]

#: The atlas picker's last entry. Its label says "Custom ID" and nothing else, so
#: what it accepts has to be written down somewhere the user can reach from it.
_CUSTOM_ID = (
    "<h3 style='margin-bottom:2px;'>Custom ID</h3>"
    "<p style='margin-top:0;'>The last entry in the atlas list. Choosing it reveals "
    "a text box that takes <b>any BrainGlobe atlas id</b> - the identifier "
    "<code>brainglobe-atlasapi</code> uses, such as "
    "<code>kim_mouse_isotropic_20um</code> or <code>whs_sd_rat_39um</code>. The "
    "atlas is downloaded on first use and cached, so the first load of an unfamiliar "
    "id is slow.</p>"
    "<p>Use it to reach a resolution or species not in the list above. Two things to "
    "expect: an atlas with no bregma anchor here shows AP against Allen's anchor "
    "(the status line says so) and <b>refuses</b> to export Paxinos coordinates "
    "rather than borrow one; and an atlas on a different voxel grid or a different "
    "species will not share coordinates with the four above.</p><hr>"
)

_FOOTER = (
    "All four cover the same physical volume, so probe coordinates are identical "
    "across them and only the naming and the bregma anchor differ. Atlases are "
    'downloaded and cached by <a href="https://brainglobe.info/documentation/'
    'brainglobe-atlasapi/index.html">brainglobe-atlasapi</a>; any other BrainGlobe '
    "id can be typed into the atlas box, but one with no bregma anchor here will "
    "refuse to export Paxinos coordinates rather than borrow Allen's."
)


#: Link colours that stay legible either side of the theme. Qt's default link blue
#: is unreadable on napari's dark ground, and a colour picked for dark would be too
#: pale on white, so the ground decides.
LINK_ON_DARK = "#6ab7ff"
LINK_ON_LIGHT = "#0b5fa5"


def link_colour_for(widget: QWidget | None) -> str:
    """Pick a link colour from how dark the widget's own background is."""
    if widget is None:
        return LINK_ON_DARK
    try:
        from qtpy.QtGui import QPalette

        lightness = widget.palette().color(QPalette.Window).lightness()
    except Exception:  # a widget without a usable palette still needs a colour
        return LINK_ON_DARK
    return LINK_ON_LIGHT if lightness > 127 else LINK_ON_DARK


def atlas_reference_html(link_colour: str = LINK_ON_DARK) -> str:
    """The reference sheet, with each bregma read from the live anchor table."""
    from histo_to_ccf.io.ccf_coords import BREGMA_AP_BY_ATLAS

    parts = ["<html><body style='font-size: 10pt;'>"]
    for name, prefix, description, links, note in _ATLASES:
        bregma = BREGMA_AP_BY_ATLAS.get(prefix)
        anchor = (
            f"{bregma:.0f} µm from the anterior edge"
            if bregma is not None
            else "no anchor recorded"
        )
        linked = " &middot; ".join(
            f'<a href="{url}" style="color:{link_colour};">{text}</a>'
            for text, url in links
        )
        parts.append(
            f"<h3 style='margin-bottom:2px;'>{name}</h3>"
            f"<p style='margin-top:0;'>{description}</p>"
            f"<p>{linked}</p>"
            f"<p><b>Bregma:</b> {anchor}<br>{note}</p><hr>"
        )
    parts.append(_CUSTOM_ID)
    footer = _FOOTER.replace("<a href=", f'<a style="color:{link_colour};" href=')
    parts.append(f"<p>{footer}</p></body></html>")
    return "".join(parts)


class AtlasReferenceDialog(QDialog):
    """Non-modal reference sheet, opened from Settings and from the Atlas tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Atlases")
        self.resize(620, 620)

        body = QLabel(atlas_reference_html(link_colour_for(parent or self)))
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setOpenExternalLinks(True)
        body.setAlignment(Qt.AlignTop)
        body.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.TextSelectableByMouse
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(buttons)


def show_atlas_reference(parent: QWidget | None = None) -> AtlasReferenceDialog:
    """Open the sheet, reusing the one already on ``parent`` if it is open."""
    existing = getattr(parent, "_atlas_reference_dialog", None)
    if existing is None:
        existing = AtlasReferenceDialog(parent)
        if parent is not None:
            parent._atlas_reference_dialog = existing
    existing.show()
    existing.raise_()
    existing.activateWindow()
    return existing
