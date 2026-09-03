# AtlasTrack

[![PyPI](https://img.shields.io/pypi/v/atlastrack)](https://pypi.org/project/atlastrack/)
[![Python](https://img.shields.io/pypi/pyversions/atlastrack)](https://pypi.org/project/atlastrack/)
[![License](https://img.shields.io/pypi/l/atlastrack)](LICENSE)

Register histological brain sections to a reference atlas, and map probe
trajectories into atlas coordinates.

This is a desktop app for wet-lab neuroscientists: load histology slide images, 
place each section in the atlas, register the series automatically, click your 
probe tracks, and export coordinates or figures.

## What it does

Given one or more slide images and a little guidance:

1. **Finds the sections** in each slide and merges several slides into one
   coordinate space.
2. **Places each section** at its front-to-back atlas level - by hand in a
   side-by-side matcher, or automatically with DeepSlice.
3. **Registers every section** to the atlas: a regularized 2-D fit (elastix, with
   a bending-energy penalty and a tissue mask) plus a silhouette pre-align and an
   outer-contour snap. Damaged sections can be corrected by hand with a box
   transform or landmark points.
4. **Maps probe tracks** you click into atlas coordinates, per shank and per
   channel; optionally refined from recorded LFP depth features.
5. **Exports** per-channel CSV (CCF µm or Paxinos stereotaxic mm), an interactive
   3-D HTML page, a HERBS `.pkl`, or your section series with atlas outlines.

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

> Quote the target and use no spaces between extras - `zsh` and PowerShell treat
> `[...]` as a glob, and a space splits the argument.

## Commands

```bash
atlastrack gui        # the app
atlastrack version
atlastrack gl-info    # diagnose GPU/OpenGL if the window will not open
atlastrack split | register | export      # headless equivalents
```

`histo2ccf` still works as an alias for every command.

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

Core packages are headless and unit-tested; only `gui/` and `viz/napari3d.py`
import napari/Qt, enforced by an import-linter contract. Coordinates are CCF
**(AP, ML, DV)** in µm throughout. A project is one Pydantic model serialized to
`<slide>.atlastrack.json`, with transform sidecars alongside.

## Testing

```bash
uv pip install -e ".[all,dev]"
pytest -q          # GUI tests need a display
lint-imports       # the headless-core contract
```
