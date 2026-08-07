# Histo-to-CCF - User Manual

_A cookbook-style reference for registering mouse histology to the Allen CCF and
mapping electrophysiology probe trajectories. Modeled on the HERBS cookbook:
concepts first, then a GUI reference, then task **recipes**, then troubleshooting._

For a single linear walkthrough on example data, see **TUTORIAL.md**. This manual
is the comprehensive reference.

---

## 1. About Histo-to-CCF

Histo-to-CCF is a [napari](https://napari.org)-based desktop app that:

- loads a composite slide image (many brain sections in one photo),
- detects and lets you edit each section,
- assigns each section an antero-posterior (**AP**) atlas plane (by hand, or
  automatically with **DeepSlice**),
- registers each section to the atlas with a masked, regularized 2-D B-spline,
- lets you hand-correct any section with a box transform or landmark warp,
- maps electrophysiology **probe trajectories** (tip/entry, per-channel) into
  atlas (CCF) coordinates, optionally refined from **LFP** depth features,
- and exports the result (3-D view, interactive HTML, HERBS `.pkl`, per-channel
  CSV in CCF or Paxinos stereotaxic space).

---

## 2. Concepts & coordinates

**CCF.** The Allen Common Coordinate Framework v3 is a 3-D reference mouse brain.
Positions are given in micrometres (µm) along three axes - **AP** (antero-posterior),
**ML** (medio-lateral), **DV** (dorso-ventral).

**AP from bregma.** The app shows AP relative to **bregma**: `0` = bregma,
**negative = posterior**, positive = anterior. Internally it stores an absolute
"distance from the anterior end"; the conversion anchor is 5400 µm.

**Supported atlases** (all fetched via BrainGlobe, cached under `~/.brainglobe`):

- Allen Mouse CCFv3 - 25 µm (default) and 100 µm
- CCFv3-BBP Augmented - 25 µm
- Chon / Kim Unified - 25 µm
- any custom BrainGlobe atlas ID (free-text).

**The pipeline at a glance:** section image → AP plane (atlas coronal slice) →
2-D B-spline warp of that slice onto the section → atlas region boundaries you can
overlay → probe pixels mapped through the same transform into CCF µm.

**Project file.** Everything is saved to a `*.histo2ccf.json` file. By
convention, **outputs go next to your input data**, not the repo folder.

---

## 3. Installation & launch

```bash
# base install (full histology→CCF workflow)
uv pip install -e .

# optional extras
uv pip install -e ".[deepslice]"   # DeepSlice AP prediction (pulls TensorFlow)
uv pip install -e ".[ephys]"       # Ephys tab (SpikeInterface)
uv pip install -e ".[deepslice,ephys]"

histo2ccf version      # confirm install
histo2ccf gui          # launch the app
histo2ccf gl-info      # diagnose a GUI/OpenGL failure to start
```

The app opens a napari viewer with a **left dock** of five workflow tabs -
**Histology · Atlas · Probes · Register · Ephys** - and a permanent **right dock**
**3D & Export** panel.

---

## 4. The interface (reference)

**Menu bar**

- **Project** - Save Project (Ctrl+S), Save Project As… (Ctrl+Shift+S),
  Load Project… (Ctrl+O), Close Project (clears to a fresh project).
- **Registration** - Parameters (the registration settings; see Recipe 5.5).

**Left-dock tabs**

| Tab | What it's for |
|-----|---------------|
| Histology | Load/merge slides, detect & edit sections, flip & adjust levels |
| Atlas | Load an atlas, assign AP planes (manually or via the matcher / DeepSlice) |
| Probes | Define probes, click tip/entry markers |
| Register | Run registration, inspect residuals, hand-correct sections |
| Ephys | Compute LFP power and align channel depths to anatomy |

**Right dock - 3D & Export** - 3-D view, HTML/HERBS/CSV export (always available).

---

## 5. Recipes

### 5.1 Load and prepare slides

1. **Histology** tab → **Open slide…** → pick one or more images
   (TIFF/PNG/JPEG). Selecting several **merges** them top-to-bottom into one
   canvas so all sections share a coordinate space.
2. Reopening a slide while one is loaded **swaps** the pixels (same-size keeps your
   sections + registration; different-size starts fresh) - useful to reuse a
   project on a different dye channel.
3. **Flip / levels** (Image tools): choose **Whole slide** or **Selected section**,
   click **Flip H** / **Flip V**, and set per-channel **R/G/B** low/high cutoffs
   (or **Auto**). Flips and levels are saved to the project.

### 5.2 Detect and edit sections

1. Set **Min area (px²)** (raise to drop debris), **Closing radius (px)** (bridge
   fragmented sections), and optionally **Equalize under-sized boxes**.
2. Click **Detect sections**.
3. **Edit boxes (resize / move / add / delete):** drag a handle to resize, drag
   inside to move, press **Delete** to remove the selected box, or use the
   rectangle tool to add one. (Deleting a hovered box is safe as of v0.2.33.)
4. **Draw new section…** then **Add drawn section** to add a missed section by hand.

### 5.3 Assign atlas AP planes (manual / matcher)

1. **Atlas** tab → choose an atlas (or type a custom BrainGlobe ID) → **Load atlas**
   (first download is slow; later loads are instant and offline-safe).
2. Quick manual path: set **AP from bregma (µm)** and a section index, then
   **Assign AP to section**.
3. Interactive path: **Open atlas matcher**
   - **Split** view shows histology left / atlas right; **Overlay** blends them
     (use the **opacity** slider and **Atlas edges**).
   - Step with **◀ Section / Section ▶**; tune **AP from bregma (µm)** until the
     atlas slice matches.
   - **Link** + **spacing (µm)** + **Anchor here**: pin one section, and every
     other section's AP follows by the spacing (sign sets direction). **Assign**
     the current section, or **Assign all (linked)** to fill the whole series.

> **Set the spacing before you pre-match** - it's what lets the app flag a bad
> DeepSlice result (Recipe 5.4).

### 5.4 Pre-match all sections with DeepSlice

DeepSlice predicts a consistent set of AP planes (with tilt) for the whole slide
in one pass - a fast way to seed AP before fine-tuning.

1. In the **atlas matcher**, make sure the section order is correct (no two
   sections share an order position; address missing sections in the order) and a
   **spacing** is set.
2. Click **Pre-match all (DeepSlice)** (first run downloads the model and is slow).
3. **Read the warnings.** DeepSlice enforces *order* but not *spacing*, so the app
   checks the result and warns if any sections come back **out of order** (AP
   reverses) or **too close together**. If warned, fix those sections before
   registering - set/confirm the spacing and use **Link + Assign all (linked)** to
   even them out, or correct individual APs.
4. Fine-tune any section's AP, then **Assign** / register.

### 5.5 Register the sections

Open **Registration → Parameters** to set:

- **Predict planes with DeepSlice** - predict planes during registration (no
  manual AP needed; reuses a prior pre-match when the section images are unchanged).
- **B-spline grid (N×N)** (default 8) and **Max iterations** (default 100).
- **Regularized registration (elastix)** - recommended: a bending-energy penalty
  + tissue mask keep atlas boundaries on the tissue.
- **Smoothness (bending energy)** (default 20) - higher = stiffer/smoother.
- **Restrict to tissue mask**, **Silhouette pre-align**, **Snap atlas contour to
  tissue** - masking/pre-alignment helpers (elastix).
- **Keep hand-corrected sections on re-run** - skip sections you've hand-corrected
  so a re-run doesn't recompute and lose them.

Then on the **Register** tab:

1. **Register all sections** (watch the progress bar + per-section status).
2. Check the **Per-section residuals** table (lower = better fit).
3. **Show atlas overlay on sections** to see the warped region boundaries on the
   histology.

### 5.6 Hand-correct a section

Pick the section in the **Section** dropdown of *Manual atlas adjustment*, then:

**Box transform** - toggle **Adjust atlas (drag in viewer)**: drag the body to
move, drag handles to scale/stretch/rotate; toggle again to apply (probes re-map,
project auto-saves).

**Landmarks** (for *local* distortions a box can't fix):

1. **Place landmarks** - drops correspondence points on **salient features**
   (silhouette tips, region-outline junctions, high-curvature corners) rather than
   on a fixed geometric ring, so they land where you'd want to grab.
2. Drag each point onto the matching tissue feature. The atlas contour **follows
   your cursor in real time**.
   - plain drag = warp; **Ctrl+drag** or **Move points** = relocate a point
     without warping; **Add points** then click = add; select + **Delete** = remove.
3. **Apply landmark warp** to commit (full warp + probe re-map + save).

Other buttons: **Reset morph to plane (keep AP/ML)** drops the auto B-spline but
keeps the AP plane (good for torn/missing tissue - redo the fit by hand);
**Reset adjustment** clears the manual correction for that section.

### 5.7 Probes and trajectories

1. **Probes** tab → choose a **Type** preset (Neuropixels 1.0, NP2.0 4-shank,
   NeuroNexus, Custom), a **Label**, and **Shanks**, then **Add probe**.
2. **Rename** an existing probe: pick it in the **Rename** dropdown, type a new
   label, **Rename** (labels must be unique - they're export keys).
3. Mark the track: choose **Tip** or **Entry** mode and the **Probe**/**Shank**,
   then **click** in the viewer to drop a marker. For entry you can instead pick
   **Trajectory line** and draw the trajectory - the entry is where it crosses the
   tissue surface.
4. **Select / move** to drag/select markers; **Clear selected** / **Clear all
   points** to remove.

### 5.8 Refine depth from LFP (Ephys, optional)

1. **Ephys** tab → select **Probe**/**Shank**, set the Open Ephys **Path** (or
   **Browse…**), optionally **List streams** and adjust **Seconds to analyse**.
2. **Load and compute LFP power** → then **Open alignment…**.
3. In the dialog, the LFP power map (depth × frequency) sits beside the atlas
   region strip. **Double-click** the map to drop a red **anchor**, drag it to line
   an LFP transition up with a region boundary. **Normalize per frequency** makes
   depth features stand out. **Save LFP power** exports the map (.npz/.csv).
4. **Apply** to store per-channel CCF on the shank.

### 5.9 Visualize and export (right dock)

- **View in napari 3D** - probes + brain regions in a 3-D viewer.
- **Export Plotly HTML** - standalone interactive 3-D scene.
- **Update coordinates** - re-map every tip/entry/channel through the current
  registration (incl. manual corrections) into CCF µm.
- **Export pkl file** - HERBS-compatible 128-point-per-shank trajectory.
- **Export per-channel CSV** - `probe, shank, channel, ap_um, ml_um, dv_um`.
- **Export per-channel Paxinos CSV** - same in Paxinos stereotaxic mm.

### 5.10 Headless / CLI

```bash
histo2ccf split --image IMAGE                  # detect sections → <stem>.sections.json
histo2ccf register-one --image IMAGE [...]     # single-section register → <stem>.histo2ccf.json
histo2ccf register PROJECT.json [--atlas ...]  # full-project registration → updates JSON
```

Transform sidecars are written to `<project_dir>/transforms/`.

---

## 6. Troubleshooting

**"Load atlas" hangs / the app won't exit.** Older versions hung on BrainGlobe's
online version check. Fixed in v0.2.35 (`check_latest=False`); loads are now
instant and offline-safe. If a process is ever stuck on a network read,
`Stop-Process -Name python -Force` (or close the terminal) clears it.

**A whole slide fails registration instantly** (every section "Internal elastix
error"). Usually the sections are stained in a colour the label mask mistook for a
fluorescent label (e.g. cyan = green+blue), so the tissue got excised from the
metric mask. Fixed in v0.2.34 - the mask now keeps tissue when label-exclusion
would remove most of it. Update and re-run.

**DeepSlice APs come out out-of-order or too close.** Expected with no spacing/
order set - see Recipe 5.4; set the spacing and section order first, and heed the
post-pre-match warning.

**Interior of the atlas overlay looks distorted but the outline is fine** (e.g. an
enlarged ventricle). Inherent to intensity-based warping where interior cues are
weak. Use **Place landmarks** on the structure (Recipe 5.6); raising smoothness or
the grid does not help.

**Elastix options are greyed out.** Install the elastix extra; otherwise the plain
SimpleITK B-spline is used.

**First DeepSlice run is slow.** It downloads the model once; later runs are fast.

**GUI won't start / OpenGL error.** Run `histo2ccf gl-info` and share the output.

---

## 7. Conventions

- **Outputs go next to your data**, never the repo working directory.
- Project files are `*.histo2ccf.json`; the atlas cache lives under `~/.brainglobe`.
- The app auto-saves the project after a manual correction (box/landmarks).
