# histo-to-ccf

Guided registration of histological brain sections to a reference atlas
(Allen Mouse CCF), with probe (Neuropixels / NeuroNexus) trajectory mapping in 3D.

A napari desktop app for wet-lab neuroscientists: load slide images, click each
probe's tip/entry, register every section to the atlas **automatically**, and get
CCF coordinates plus an interactive 3D view - without dropping dozens of landmark
pairs per brain.

## What it does

Given one or more 4x/5x histology slide images of brain sections and a small
amount of user input, the app:

1. **Splits** each slide into individual sections (Otsu + connected components);
   multiple slides are merged into one coordinate space.
2. Lets the user click each probe shank's **tip** and **entry point** (a marker,
   or a trajectory line whose tissue-surface crossing is the entry).
3. Lets the user pick a rough atlas-plane match for a handful of sections (an
   AP-matcher dialog) and set the section spacing/ordering.
4. **Registers every section automatically** to the chosen atlas (Allen CCF 25 µm
   by default):
   - **DeepSlice** predicts a consistent set of atlas planes across the series;
   - a regularized 2D registration (**elastix**: bending-energy penalty + tissue
     mask, ABBA-style; falls back to a SimpleITK B-spline) refines each section
     in-plane;
   - a closed-form **silhouette pre-align** fixes per-section scale, and an
     automatic **outer-contour snap** pulls the atlas boundary onto the tissue
     border (the fix for "atlas lines just outside the section");
   - for the few genuinely damaged/asymmetric sections, **manual correction**
     tools (drag a box transform, or place landmark points) are available.
5. Returns **CCF coordinates** for every shank's tip and entry, plus per-channel
   coordinates when probe geometry is supplied.
6. Plots all probe trajectories with semi-transparent BrainGlobe meshes (Plotly
   HTML and an in-app napari 3D view, both bregma-referenced); regions can be drawn
   from any coordinate-compatible CCFv3 atlas (Allen, CCFv3-BBP Augmented, Chon/Kim
   Unified) without re-registering. Exports HERBS `.pkl`, per-channel CCF CSV, and
   per-channel **Paxinos** stereotaxic CSV.
7. (Optional, `ephys` extra) Refines each shank's depth→CCF mapping from LFP
   features: load an Open Ephys recording, align the depth×frequency LFP power
   map to atlas region boundaries with draggable anchors, and store per-channel
   CCF coordinates.

A full click-by-click walkthrough on the bundled example slide is in
[`TUTORIAL.md`](TUTORIAL.md); the comprehensive cookbook-style reference (concepts,
GUI reference, task recipes, troubleshooting) is in [`MANUAL.md`](MANUAL.md).

## Install

```bash
uv pip install histo-to-ccf                      # or:  pip install histo-to-ccf
histo2ccf gui                                    # launch the guided workflow
```

Optional features live behind **extras**:

- `deepslice` - automatic atlas-plane prediction (pulls TensorFlow).
- `elastix` - the regularized (bending-energy + masked) registration engine
  (pulls ITK ~150 MB). Without it the pipeline uses the plain SimpleITK B-spline.
- `ephys` - the Ephys-alignment tab: load Open Ephys LFP and refine shank depth
  (pulls SpikeInterface + neo/probeinterface).

```bash
uv pip install "histo-to-ccf[deepslice]"             # one extra
uv pip install "histo-to-ccf[deepslice,elastix]"     # several, comma-separated
```

To install for development (editable):

```bash
uv pip install -e .                                  # base only
uv pip install -e ".[deepslice]"                     # with DeepSlice
uv pip install -e ".[deepslice,elastix,ephys,dev]"   # everything + test/lint tools
```

> **Quote the target** (`".[...]"`) and use **no spaces** between extras -
> `zsh` and PowerShell otherwise treat `[...]` as a glob and the space splits the
> argument (e.g. `uv pip install -e .[deepslice, ephys]` fails to parse).

## Quick start

```bash
histo2ccf gui            # launch the GUI (the main entry point)
histo2ccf version        # print the installed version
histo2ccf gl-info        # diagnose GPU/OpenGL if the GUI won't start
```

The GUI shows a 5-tab **Registration** panel on the left (Histology → Atlas →
Probes → Register → Ephys), a permanent **3D & Export** panel on the right, and a
menu bar with **Project** (Save / Load / Close) and **Registration** (Parameters).
**Project → Close** clears the current slides/sections/probes so you can start a
fresh registration without restarting the app. In the canvas, the mouse wheel
zooms; **Ctrl+wheel** pans left/right and **Shift+wheel** pans up/down.

## Repository layout

```
src/histo_to_ccf/   # the package
  io/  atlas/  sectioning/  landmarks/  registration/  probes/  viz/  gui/
tests/              # pytest suite (run: pytest -q)
TUTORIAL.md         # step-by-step walkthrough on the example slide
MANUAL.md           # cookbook-style reference: concepts, GUI, recipes, troubleshooting
HANDOFF.md          # detailed design notes / engineering log
```

Core packages are headless and unit-tested; only `gui/` and `viz/napari3d.py`
import napari/Qt (enforced by an import-linter contract). Coordinates are CCF
**(AP, ML, DV)** in µm throughout. Project state is a single Pydantic model
serialized to `<slide>.histo2ccf.json` with B-spline transform sidecars alongside.

## Testing

```bash
uv pip install -e ".[dev]"
pytest -q                  # full suite (GUI tests need a display)
```
