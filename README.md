# histo-to-ccf

Guided registration of histological brain sections to a reference atlas
(Allen Mouse CCF), with probe trajectory mapping in 3D.

> **Status:** pre-alpha rewrite in progress. See `plans/` or open issues for the
> active milestone. The pre-rewrite code lives under [`legacy/`](legacy/) and is
> not imported by the new package.

## What it does (target)

Given one or more 4x/5x histology slide images of brain sections and a small
amount of user input, this app:

1. Splits each slide into individual sections.
2. Lets the user click each probe's shank **tip** and **entry point** on the
   sections where the probe is visible.
3. Lets the user pick a rough atlas-plane match for a handful of sections and
   provide the section spacing/ordering.
4. Automatically registers every section to the chosen atlas (Allen CCF 25 µm
   by default) — using DeepSlice for the coarse plane and a SimpleITK 2D
   B-spline for in-plane refinement, with auto-detected contour, midline, and
   ventricle landmarks.
5. Returns CCF coordinates for every shank's tip and entry, plus per-channel
   coordinates when probe geometry is supplied.
6. Plots all probe trajectories together with semi-transparent BrainGlobe
   meshes (Plotly HTML and an in-app napari 3D view).

## Install (target)

```bash
uv pip install histo-to-ccf            # or:  pip install histo-to-ccf
uv pip install histo-to-ccf[deepslice] # enable auto plane prediction
histo2ccf gui                          # launch the guided workflow
```

To install as dev: 
```bash
uv pip install -e . # To install without deepslice
uv pip install -e .[deepslice] # To install with deepslice
```

## Repository layout

```
src/histo_to_ccf/   # the new package (in progress)
tests/              # pytest suite + fixtures
legacy/             # archived pre-rewrite code (HERBS post-processing scripts,
                    # old plotting notebooks) — read-only reference
```

## Status of legacy code

The previous version of this repo (under [`legacy/`](legacy/)) consisted of:

- `HERBS_to_AllenCCF/herbs_probe_mapping.py` and `probe_visualization.py` —
  post-processing scripts for HERBS pickle output that produce a Plotly 3D
  view of probe trajectories in Allen CCF space.
- A handful of one-off Jupyter notebooks and ROI tables.

Reusable functions (HERBS pkl loader, probe-prism mesh, bregma↔CCF helpers,
dorsal-surface alignment) are being ported into the new package piece by
piece. Until then, the original scripts in `legacy/` still run as-is.
