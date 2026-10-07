# AtlasTrack

[![PyPI](https://img.shields.io/pypi/v/atlastrack)](https://pypi.org/project/atlastrack/)
[![Python](https://img.shields.io/pypi/pyversions/atlastrack)](https://pypi.org/project/atlastrack/)
[![License](https://img.shields.io/pypi/l/atlastrack)](LICENSE)

Register histological brain sections to a reference atlas, and map probe
trajectories into atlas coordinates.

This is a desktop app for wet-lab neuroscientists: load histology slide images,
place each section in the atlas, register the series automatically, click your
probe tracks, and export coordinates or figures.

<img src="https://raw.githubusercontent.com/vncntprvst/AtlasTrack/main/images/AT_GUI.png" alt="AtlasTrack GUI" width="600">

## What it does

Given one or more slide images and a little guidance:

1. **Finds the sections** in each slide and merges several slides into one
   image with shared coordinates.
2. **Places each section** at its front-to-back atlas level - by hand in a
   side-by-side matcher, or automatically with DeepSlice.
3. **Registers every section** to the atlas, bending the atlas to follow your
   tissue (see *How a section is fitted* below). Damaged sections can be
   corrected by hand, by moving the whole atlas outline or by dragging points.
4. **Maps probe tracks** you click into atlas coordinates, per shank and per
   channel; optionally refined from recorded LFP depth features.
5. **Exports** per-channel CSV (CCF µm or Paxinos stereotaxic mm), an interactive
   3-D HTML page, a HERBS `.pkl`, or your section series with atlas outlines.

## How a section is fitted

DeepSlice alone places each section with a *linear* fit: an atlas plane, tilted,
shifted and scaled, but not bent. On real tissue that is rarely enough. AtlasTrack
uses DeepSlice only to choose the plane, then fits the atlas to the tissue:

1. **Plane.** DeepSlice sees one tight crop per section, with your flips
   applied, numbered in your section order. It uses those numbers to keep the
   series in front-to-back order and to share one tilt across it. Any AP you set
   by hand (or by even spacing) is an **anchor**: that section's plane slides
   along AP to exactly your value, keeping DeepSlice's tilt. Sections without an
   anchor shift by interpolating their neighbouring anchors' corrections, so
   one anchor corrects the offset of the whole series and two or more also its
   spacing. APs that DeepSlice wrote itself are predictions, not anchors.
2. **Scale and shift.** The outline of the atlas slice is moved and scaled onto
   the outline of the tissue: same centre, same size, no rotation.
3. **Warp.** elastix, a registration program, bends the atlas onto the
   section: it moves a grid of control points until the light and dark
   patterns of the two images agree. It compares tissue pixels only -
   fluorescent labels are left out - and a penalty on bending keeps the warp
   smooth.
4. **Edge snap.** The atlas brain's outline is pulled onto the tissue border,
   with the interior held still. Pushes too large to be a fit (torn or missing
   tissue) are skipped, and a warp that would fold is rejected.
5. **Check.** If the result overlaps the tissue clearly worse than the plain
   plane, the plane is kept instead, for you to fit by hand.

Box and landmark corrections are applied on top. Re-registering a section
clears them, since they corrected the fit it replaces.

Atlases come from BrainGlobe: Allen CCFv3, CCFv3-BBP Augmented, Chon/Kim Unified
(Franklin-Paxinos names), and any other BrainGlobe id. All cover the same volume,
so regions can be re-named from a different atlas without re-registering.

[`TUTORIAL.md`](TUTORIAL.md) walks through one slide start to finish.  
[`MANUAL.md`](MANUAL.md) is the reference document.  
Both are also available in the app under **Help**.

## Install

From [PyPI](https://pypi.org/project/atlastrack/):

```bash
uv pip install "atlastrack[all]"    # recommended  (pip install also works)
atlastrack gui
```

The base install (`atlastrack`) is deliberately light. `[all]` adds three extras,
each installable on its own:

| Extra | Adds | Cost |
|---|---|---|
| `elastix` | The regularized registration engine - recommended | ~150 MB (ITK) |
| `deepslice` | Automatic front-to-back placement | ~1.65 GB (TensorFlow) |
| `ephys` | The Ephys tab (Open Ephys / SpikeGLX / Intan) | SpikeInterface |

From source, for development:

```bash
git clone https://github.com/vncntprvst/AtlasTrack
cd AtlasTrack
uv pip install -e ".[all,dev]"
```

> Put the package name in quotes and leave no spaces inside the brackets -
> `zsh` and PowerShell otherwise read `[...]` as a file pattern, and a space
> splits it in two.

## Commands

```bash
atlastrack gui        # the app
atlastrack version
atlastrack gl-info    # check the graphics setup if the window will not open
atlastrack split | register | export      # the same steps, without the window
atlastrack import SOURCE                  # convert a slicereg or QuickNII/VisuAlign registration
atlastrack match PROJECT.json REF.json    # register sections from the same sections in REF
```

## The window

The centre holds **Project** (your slide and the atlas overlay) and **Help**. On
the left, the workflow in order: **Histology → Atlas → Register → Probes →
Ephys**. On the right, **3D & Export**. Menus: **Project**, **Settings**, **Help**.

Wheel zooms; **Ctrl**+wheel and **Shift**+wheel pan.

## Layout

```
src/atlastrack/   io/ atlas/ sectioning/ landmarks/ registration/ probes/ viz/ gui/
tests/            pytest suite
```

The core packages run without a display and have their own tests; only `gui/`
and `viz/napari3d.py` import napari/Qt, which `lint-imports` checks. Coordinates are CCF
**(AP, ML, DV)** in µm throughout. A project is one Pydantic model serialized to
`<slide>.atlastrack.json`, with each section's registration saved in a file
alongside it.

## Testing

```bash
uv pip install -e ".[all,dev]"
pytest -q          # GUI tests need a display
lint-imports       # checks the core packages never import the GUI
```
