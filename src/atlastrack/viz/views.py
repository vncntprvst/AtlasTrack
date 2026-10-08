"""Three still views of the registered probes in the brain: back, top and side PNGs.

The scene is the 3D view's, drawn for a figure: a faint outline of the whole brain,
the regions the shank tips are in (plus any extra regions asked for), each probe's
tracks, and the electrodes as red dots - only those the attached recordings used,
when the recordings say which. Each view uses an orthographic camera, so
distances read the same everywhere in the picture.

Rendering is off-screen with vispy (napari's drawing library), so no window opens
and nothing beyond what AtlasTrack already needs is required.

Display frame, in µm: X = midline - ML (the animal's right is +X), Y = bregma - AP
(anterior is +Y), Z = -DV (dorsal is +Z). Seen from the back, the animal's left is
on the image's left, as for the sections.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

    from atlastrack.project.schema import Project

#: (file suffix, azimuth, elevation) for vispy's turntable camera with Z up.
#: Measured, not assumed: azimuth 0 looks from behind toward anterior with the
#: animal's right on the image's right; elevation 90 looks down with anterior up;
#: azimuth -90 looks from the animal's left side with anterior on the image's left.
VIEWS = (("back", 0.0, 0.0), ("top", 0.0, 90.0), ("side", -90.0, 0.0))

#: Which display axes run across and up the picture in each view.
_SCREEN_AXES = {"back": (0, 2), "top": (0, 1), "side": (1, 2)}

ELECTRODE_COLOUR = (0.86, 0.12, 0.12, 1.0)
BRAIN_COLOUR = (0.55, 0.55, 0.55, 0.07)


def _to_display(ccf_um, atlas_name: str | None) -> np.ndarray:
    """(N, 3) CCF (AP, ML, DV) µm -> the display frame described above."""
    from atlastrack.io.ccf_coords import MIDLINE_ML_UM, bregma_ap_for_display

    p = np.asarray(ccf_um, dtype=float).reshape(-1, 3)
    return np.stack([MIDLINE_ML_UM - p[:, 1],
                     bregma_ap_for_display(atlas_name) - p[:, 0],
                     -p[:, 2]], axis=1)


def _region_mesh(atlas, acronym: str):
    """(vertices (N, 3) as CCF (AP, ML, DV) µm, faces) of a region, or None."""
    from atlastrack.atlas.meshes import mesh_vertices_faces

    try:
        verts, faces = mesh_vertices_faces(atlas.mesh_from_structure(acronym))
    except Exception:  # noqa: BLE001 - no mesh for this structure
        return None
    verts = np.asarray(verts, dtype=float)                     # (AP, DV, ML)
    return np.stack([verts[:, 0], verts[:, 2], verts[:, 1]], axis=1), np.asarray(faces)


def recorded_electrodes(probe) -> set[int] | None:
    """Electrode indices (0 = lowest) the probe's recordings used, or None for all.

    From each attached recording's electrode range (1-based, the same on every
    shank). None when no recording gives one: then every electrode is drawn.
    """
    out: set[int] = set()
    for ref in getattr(probe, "recordings", None) or []:
        if ref.electrode_range:
            first, last = ref.electrode_range
            out.update(range(int(first) - 1, int(last)))
    return out or None


def _full_name(atlas, acronym: str) -> str:
    """A region's full atlas name, first letter capitalised; the acronym if none."""
    try:
        name = str(atlas.structures[acronym]["name"]).strip()
    except Exception:  # noqa: BLE001
        return acronym
    return name[:1].upper() + name[1:] if name else acronym


def _font(px: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def place_labels(image, regions, electrodes_px, *, font_px: int):
    """Write each region's name beside it, with a thin line to it. Returns the image.

    ``regions`` is ``[(name, points (N, 2) in pixels)]``; ``electrodes_px`` is (M, 2).
    A label goes just outside its region - on the side facing away from the
    electrodes first - and is moved to another side, then further out, until it
    covers no electrode, no other label and as little of the other regions as
    possible. A region that cannot be labelled that way is left unlabelled rather
    than labelled over the data.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    font = _font(font_px)
    w_img, h_img = image.size
    gap = font_px * 0.8
    pad = font_px * 0.4
    elec = np.asarray(electrodes_px, dtype=float).reshape(-1, 2)
    elec_centre = elec.mean(axis=0) if len(elec) else None
    boxes = [(n, pts.min(axis=0), pts.max(axis=0)) for n, pts in regions if len(pts)]
    taken: list[tuple[float, float, float, float]] = []

    def overlap(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    for name, pts in regions:
        if not len(pts):
            continue
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        centre = (lo + hi) / 2.0
        x0, y0, x1, y1 = draw.textbbox((0, 0), name, font=font, stroke_width=3)
        tw, th = x1 - x0, y1 - y0
        sides = {
            "right": lambda g: (hi[0] + g, centre[1] - th / 2),
            "left": lambda g: (lo[0] - g - tw, centre[1] - th / 2),
            "above": lambda g: (centre[0] - tw / 2, lo[1] - g - th),
            "below": lambda g: (centre[0] - tw / 2, hi[1] + g),
        }
        order = list(sides)
        if elec_centre is not None:
            away = centre - elec_centre
            horizontal = ["right", "left"] if away[0] >= 0 else ["left", "right"]
            vertical = ["below", "above"] if away[1] >= 0 else ["above", "below"]
            order = horizontal + vertical if abs(away[0]) >= abs(away[1]) else vertical + horizontal
        best = None
        for step, g in enumerate((gap, gap * 3, gap * 6)):
            for rank, side in enumerate(order):
                x, y = sides[side](g)
                rect = (x - pad, y - pad, x + tw + pad, y + th + pad)
                if rect[0] < 0 or rect[1] < 0 or rect[2] > w_img or rect[3] > h_img:
                    continue
                if len(elec) and np.any((elec[:, 0] > rect[0] - pad) & (elec[:, 0] < rect[2] + pad)
                                        & (elec[:, 1] > rect[1] - pad) & (elec[:, 1] < rect[3] + pad)):
                    continue
                if any(overlap(rect, t) > 0 for t in taken):
                    continue
                covered = sum(overlap(rect, (b_lo[0], b_lo[1], b_hi[0], b_hi[1]))
                              for n2, b_lo, b_hi in boxes if n2 != name)
                score = covered + (step * 4 + rank) * tw * th * 0.05
                if best is None or score < best[0]:
                    best = (score, x, y, rect)
        if best is None:
            continue
        _score, x, y, rect = best
        taken.append(rect)
        # Leader: from the label's nearest edge to the region's nearest point.
        label_centre = np.array([(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2])
        target = pts[np.argmin(np.linalg.norm(pts - label_centre, axis=1))]
        start = (min(max(target[0], rect[0]), rect[2]), min(max(target[1], rect[1]), rect[3]))
        draw.line([start, tuple(target)], fill=(90, 90, 90), width=max(1, font_px // 14))
        draw.text((x - x0, y - y0), name, font=font, fill=(45, 45, 45),
                  stroke_width=3, stroke_fill=(255, 255, 255))
    return image


def scene_contents(project: "Project", atlas: "BrainGlobeAtlas | None", *,
                   extra_regions=()) -> dict:
    """Everything the views draw, in CCF µm: regions, tracks and electrodes."""
    from atlastrack.probes.catalog import get_layout
    from atlastrack.probes.channels import shank_channel_coords
    from atlastrack.viz.plotly3d import hex_to_rgb, resolve_regions, styled_regions

    regions = []
    if atlas is not None:
        names = resolve_regions(project, atlas, extra_regions=list(extra_regions))
        for acronym, colour, opacity in styled_regions(names):
            mesh = _region_mesh(atlas, acronym)
            if mesh is not None:
                rgb = tuple(c / 255.0 for c in hex_to_rgb(colour))
                regions.append((acronym, mesh, (*rgb, max(0.25, float(opacity)))))
        full_names = {a: _full_name(atlas, a) for a, _m, _c in regions}
    else:
        full_names = {}
    tracks, electrodes = [], []
    for probe in project.probes:
        layout = get_layout(probe.type.name)
        recorded = recorded_electrodes(probe)
        for shank in probe.shanks:
            if shank.tip_ccf_um is None or shank.entry_ccf_um is None:
                continue
            tracks.append(np.array([shank.tip_ccf_um, shank.entry_ccf_um], dtype=float))
            coords = shank_channel_coords(shank, layout)
            if coords is None or not len(coords):
                continue
            coords = np.asarray(coords, dtype=float)
            if recorded is not None:
                coords = coords[[k for k in sorted(recorded) if k < len(coords)]]
            electrodes.append(coords)
    return {
        "brain": _region_mesh(atlas, "root") if atlas is not None else None,
        "regions": regions,
        "names": full_names,
        "tracks": tracks,
        "electrodes": np.concatenate(electrodes) if electrodes else np.empty((0, 3)),
    }


def _projected(visual, points) -> np.ndarray:
    """Display-frame points -> image pixels, through ``visual``'s current camera."""
    mapped = visual.get_transform("visual", "canvas").map(np.asarray(points, dtype=float))
    return mapped[:, :2] / mapped[:, 3:4]


def _projected_regions(region_visuals, contents, view_name: str, atlas_name):
    """(name, pixels) per region; one hemisphere's half where the two do not overlap."""
    electrodes = _to_display(contents["electrodes"], atlas_name)
    probe_side = np.sign(electrodes[:, 0].mean()) if len(electrodes) else 0.0
    out = []
    for acronym, mesh, visual in region_visuals:
        pts = _to_display(mesh[0], atlas_name)
        if view_name != "side" and probe_side != 0:
            other = pts[np.sign(pts[:, 0]) == -probe_side]
            if len(other) > 20:
                pts = other
        step = max(1, len(pts) // 4000)
        out.append((contents["names"].get(acronym, acronym), _projected(visual, pts[::step])))
    return out


def render_three_views(
    project: "Project",
    atlas: "BrainGlobeAtlas | None",
    out_dir: str | Path,
    *,
    stem: str = "views",
    extra_regions=(),
    size: tuple[int, int] = (2400, 1600),
    margin: float = 0.35,
    labels: bool = False,
) -> list[Path]:
    """Write ``<stem> - back.png``, ``- top.png`` and ``- side.png``; return their paths.

    The views are framed on the regions and electrodes (plus ``margin`` of that
    extent all round), at one zoom for all three; the whole-brain outline and the
    upper part of the tracks are context and may run off the edges.

    ``labels`` writes each region's full name beside it (see :func:`place_labels`).
    A region on both sides of the brain is labelled once, on the side without the
    probes, except in the side view where the two halves overlap.
    """
    from PIL import Image
    from vispy import scene

    atlas_name = getattr(getattr(project, "atlas", None), "name", None)
    contents = scene_contents(project, atlas, extra_regions=extra_regions)
    if not contents["tracks"]:
        raise ValueError("No probe has tip and entry coordinates yet: nothing to draw.")

    canvas = scene.SceneCanvas(show=False, size=size, bgcolor="white")
    view = canvas.central_widget.add_view()
    # Translucent surfaces must not hide each other, the tracks or the dots, so none
    # of them writes depth: everything stays visible through everything else.
    surface_state = dict(blend=True, depth_test=False, cull_face=False)

    def add_mesh(mesh, colour):
        verts, faces = mesh
        visual = scene.visuals.Mesh(vertices=_to_display(verts, atlas_name), faces=faces,
                                    color=colour, shading="smooth", parent=view.scene)
        visual.set_gl_state("translucent", **surface_state)
        return visual

    if contents["brain"] is not None:
        add_mesh(contents["brain"], BRAIN_COLOUR)
    region_visuals = [(name, mesh, add_mesh(mesh, colour))
                      for name, mesh, colour in contents["regions"]]
    for track in contents["tracks"]:
        line = scene.visuals.Line(pos=_to_display(track, atlas_name), color=(0.2, 0.2, 0.2, 0.8),
                                  width=max(2.0, size[0] / 800), parent=view.scene)
        line.set_gl_state("translucent", depth_test=False)
    dots = None
    if len(contents["electrodes"]):
        dots = scene.visuals.Markers(parent=view.scene)
        # Small: 384 electrodes per shank sit 7.5 µm apart, so larger dots merge
        # into a solid bar that hides the regions they are in.
        dots.set_data(_to_display(contents["electrodes"], atlas_name), face_color=ELECTRODE_COLOUR,
                      edge_width=0.0, size=max(3.0, size[0] / 600))
        dots.set_gl_state("translucent", depth_test=False)

    # Frame on what matters: the regions and the electrodes in them. Not the whole
    # tracks, which run up to the brain surface and would zoom out to most of it.
    focus = [_to_display(m[0], atlas_name) for _n, m, _c in contents["regions"]]
    if len(contents["electrodes"]):
        focus.append(_to_display(contents["electrodes"], atlas_name))
    if not focus:
        focus = [_to_display(t, atlas_name) for t in contents["tracks"]]
    pts = np.concatenate(focus)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = (lo + hi) / 2.0
    extent = np.maximum(hi - lo, 1.0) * (1.0 + 2.0 * margin)
    aspect = size[0] / size[1]
    # One zoom for all three, so the views compare at a glance: the one that fits
    # the widest of them.
    scale = max(float(max(extent[up], extent[across] / aspect))
                for across, up in _SCREEN_AXES.values())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, azimuth, elevation in VIEWS:
        view.camera = scene.TurntableCamera(fov=0.0, azimuth=azimuth, elevation=elevation,
                                            up="+z")
        # Fit what this view shows across and up; set_range would fit the largest
        # of all three axes and zoom out to the whole brain.
        view.camera.center = tuple(centre)
        view.camera.scale_factor = scale
        image = Image.fromarray(canvas.render()[..., :3])
        if labels and region_visuals:
            image = place_labels(image, _projected_regions(region_visuals, contents, suffix,
                                                           atlas_name),
                                 _projected(dots, _to_display(contents["electrodes"], atlas_name))
                                 if dots is not None else np.empty((0, 2)),
                                 font_px=max(12, size[1] // 45))
        path = out_dir / f"{stem} - {suffix}.png"
        image.save(path)
        written.append(path)
    canvas.close()
    return written
