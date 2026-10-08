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
        "tracks": tracks,
        "electrodes": np.concatenate(electrodes) if electrodes else np.empty((0, 3)),
    }


def render_three_views(
    project: "Project",
    atlas: "BrainGlobeAtlas | None",
    out_dir: str | Path,
    *,
    stem: str = "views",
    extra_regions=(),
    size: tuple[int, int] = (2400, 1600),
    margin: float = 0.35,
) -> list[Path]:
    """Write ``<stem> - back.png``, ``- top.png`` and ``- side.png``; return their paths.

    The views are framed on the regions and electrodes (plus ``margin`` of that
    extent all round), at one zoom for all three; the whole-brain outline and the
    upper part of the tracks are context and may run off the edges.
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
    for _name, mesh, colour in contents["regions"]:
        add_mesh(mesh, colour)
    for track in contents["tracks"]:
        line = scene.visuals.Line(pos=_to_display(track, atlas_name), color=(0.2, 0.2, 0.2, 0.8),
                                  width=max(2.0, size[0] / 800), parent=view.scene)
        line.set_gl_state("translucent", depth_test=False)
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
        image = canvas.render()[..., :3]
        path = out_dir / f"{stem} - {suffix}.png"
        Image.fromarray(image).save(path)
        written.append(path)
    canvas.close()
    return written
