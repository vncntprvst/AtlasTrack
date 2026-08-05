# Histo-to-CCF v0.2.14 - Test walkthrough

## Prerequisites

```
# Install with extras (DeepSlice for atlas-plane prediction, ephys for LFP-based depth refinement):
uv pip install -e ".[deepslice,ephys]"   
# or a subset:
#   uv pip install -e .                   # base only
#   uv pip install -e ".[deepslice]"      # just DeepSlice
#   uv pip install -e ".[ephys]"          # just the Ephys tab (SpikeInterface)

# Activate your environment
.venv\Scripts\activate # Windows
source .venv/bin/activate # macOS/Linux

histo2ccf version          # should print: histo2ccf 0.2.14
```

The **`ephys`** extra (§6b) pulls SpikeInterface + neo/probeinterface; the
**`deepslice`** extra pulls TensorFlow. Both are optional - the base install runs
the full histology→CCF workflow without them.

You need one composite slide image (multi-section TIFF or JPEG).  
The repo includes an example: `example data\L07_slide3_2x_whole_overlay.jpg`

---

## 1 - Launch

```
histo2ccf gui
```

A napari viewer opens with the **Registration** panel (5 tabs) docked on the
**left** - **Histology / Atlas / Probes / Register / Ephys** - and a permanent
**3D & Export** panel docked on the **right** (3D view + Plotly/HERBS/CSV export,
always available). The menu bar has two menus: **Project** (Save / Load / Close -
*Close* clears the current project to start fresh without restarting) and
**Registration** (Parameters - the registration settings, kept out of the panel).

---

## 2 - Histology tab: load the slide and detect sections

1. Click **Open slide…** → select your image. You can select **several images at
   once**; they are merged into one combined canvas, stacked top-to-bottom, so all
   sections share one coordinate space.  
   The slide appears as a gray layer in the viewer.  
   **Swapping the image:** once a slide is loaded (e.g. after reloading a project),
   **Open slide…** again *replaces* the current image instead of merging. If the new
   image is the **same size**, the section boxes + registration are kept - handy for
   reusing a registration on the same section imaged in a different channel/dye. A
   different-size image is treated as a new slide (sections cleared, re-detect).
2. Optionally adjust **Min area px** (default 5000) and **Closing r** (try 0 first).
3. Click **Detect sections**.  
   Yellow rectangles appear around the detected brain sections.  
   Status should read e.g. *"Detected 14 section(s)"*. Detection is **slide-aware**:
   sections are numbered column-by-column **within each source image**, so the
   ordering never runs a column across two stacked slides.

**Image tools** (same tab):

- Use **Flip H** or **Flip V** if the brain is mirrored relative to anatomical orientation.
- Adjust the **R / G / B** level spinboxes (or click **Auto**) if contrast is poor.

---

## 3 - Atlas tab: load atlas and assign AP planes

1. (Optional) Set the **Atlas folder** - where atlases are downloaded to and reused
   from. Leave the default (`~/.brainglobe`) unless you want them elsewhere.
2. Select **Allen CCFv3 25 µm** (default) and click **Load atlas**.  
   First run downloads ~400 MB; subsequent runs are instant because the atlas is
   reused from the folder above (the status line shows where it loaded from).
3. Set **AP from bregma (µm)** to match a section - **0 = bregma**, negative = posterior,
   positive = anterior. Click **Open atlas matcher…** to compare your section against
   atlas slices side-by-side / overlaid and assign the AP.
4. For each section: set **Assign to section idx**, then click **Assign AP to section**.
   (Midline / dorsal-surface anchoring is handled automatically by registration - no
   manual pixel entry needed.)
5. In **Section ordering**: the list is the **anterior→posterior sequence**, top =
   first. With the **Direction: Anterior → Posterior** default, the **top section is
   the most anterior** (and is marked *◄ anterior end*; the bottom is *◄ posterior
   end*) - check this matches your slides before registering. Drag sections to
   reorder, set **Section spacing (µm)** (persists with the project), pick the anchor
   section, and click **Apply spacing** to propagate AP values across all sections.

> **DeepSlice + your AP.** DeepSlice predicts every plane, then is **guided** by any
> AP you assigned: it shifts its predictions onto your values (one assigned section
> sets the overall offset, two or more set offset + scale). So you can let DeepSlice
> do the work and just pin a level or two to anchor it - your assigned AP is no
> longer overridden.

---

## 4 - Probes tab: add a probe and click tip + entry

1. **Add probe**: choose probe type (*Neuropixels 1.0*, *Neuropixels 2.0 (4-shank)*,
   or *NeuroNexus A1x32-Poly3-10mm-25s-177-OA32LP*), set a label, click **Add probe**.
   Select the probe (by label) and shank from the dropdowns - the same labels are
   used in the Ephys tab.
2. **Pick points** - just select a mode and click; no extra button:
   - Select **Tip** mode, then click the shank tip in the viewer on the section where
     the probe ends. The marker stays on top of the image automatically.
   - Select **Entry** mode for the brain-surface entry point. Two ways to set it:
     - **Marker**: click the surface directly.
     - **Trajectory line**: draw the probe track as a line; the point where it crosses
       the tissue surface is taken as the entry.
   Each shank gets its **own colour** (the tip and entry of one shank match; the
   colour cycles as you select another shank/probe). Tips are **discs**, entries are
   **triangles**. A new tip/entry for a shank replaces its previous one.
3. **Adjust / delete points**: click **Select / move** to drag a marker to a new
   spot or select markers (Shift-click for several); then press Delete or
   **Clear selected** to remove just those. **Clear all points** wipes everything.
4. The table at the bottom shows the stored coordinates for both tips **and** entries.

**Tip - moving around the canvas:** the mouse wheel zooms; hold **Ctrl** and scroll
to pan **left/right**, or **Shift** and scroll to pan **up/down**.

---

## 5 - Register tab: run the registration

The panel is intentionally lean - just **Register all sections**, the progress
bar, the residuals table, **Show atlas overlay**, and the manual-adjustment tools.

1. (Optional) Open **Registration menu → Parameters** to review the settings.
   The defaults are good and all toggles are **on**:
   - **Predict planes with DeepSlice** (top) - predicts a consistent set of atlas
     planes first, so you don't need to assign AP by hand. (Needs the `deepslice`
     extra; first run downloads the model and is slow.)
   - **Regularized registration (elastix)**, **Smoothness (bending energy)**,
     **Restrict to tissue mask** - the ABBA-style engine that keeps atlas
     boundaries on the tissue (needs the `elastix` extra; falls back to a plain
     SimpleITK B-spline otherwise).
   - **Silhouette pre-align** and **Snap atlas contour to tissue** - make the
     atlas outer contour follow the tissue border automatically.
   - **B-spline grid** (8) and **Max iterations** (100); for a quick first test
     try grid = 6, iterations = 60.
2. Click **Register all sections**.  
   The progress bar fills section-by-section; status shows the residual as each
   finishes. Typical runtime: ~30 s per section on CPU.
3. After completion the residuals table populates (residual = normalized-intensity
   RMS over the tissue; lower = better). Click **Show atlas overlay on sections**
   to see the registered region boundaries warped onto each section.
4. **If a section needs a touch-up** (usually only damaged/asymmetric ones), use
   **Manual atlas adjustment**: pick the section, then either **Box transform**
   (drag the overlay / box handles) or **Landmarks** (place + drag correspondence
   points). **Reset adjustment** clears it. Both re-map probes and auto-save.
   - **Badly distorted sections** (tissue torn apart, or a piece missing - e.g.
     brainstem but no cerebellum): click **Reset morph to plane (keep AP/ML)** in
     the Landmarks group. That drops the automatic warp but keeps the atlas plane,
     so the overlay returns to the clean, undistorted slice - then use **Place
     landmarks** to fit it by hand instead of fighting the distorted outline.

---

## 6 - View and export (3D & Export panel, right dock)

These live in the permanent **3D & Export** panel on the right - available at any
time, not only after registration.

**Update coordinates**  
Probe CCF coordinates are computed from your pixel clicks through the registration.
If you **move a tip/entry point** (Probes tab) or **correct a section's atlas**
after the 3D window is open, click **Update coordinates** to re-map every probe
into CCF, save, and refresh the 3D window. (Moving a point alone doesn't update the
CCF used by the 3D view + exports - click this first.) **View in napari 3D** updates
automatically; the pkl/CSV/Plotly exports use the last-updated coordinates, so click
**Update coordinates** before exporting if you've moved points.

**Region atlas** (3D Visualization)  
Optionally pick a **Region atlas** other than the one you registered with. The
Allen CCFv3, CCFv3-BBP Augmented and Chon/Kim Unified 25 µm atlases share the same
voxel space, so your probe coordinates stay identical - only the region
meshes/acronyms change. This lets you view (and Plotly-export) the same result in a
finer/alternative parcellation without re-registering.

**In-app 3D**  
Click **View in napari 3D** - opens a separate 3D window with the brain shell, tip
regions, probe tracks, and (after an ephys alignment) the per-channel positions.
Both 3D views are **bregma-referenced** and not mirrored.

**Plotly HTML**  
Click **Export Plotly HTML…** → save to e.g. `Desktop\probe_3d.html`.  
Opens automatically in your browser; rotate the atlas + probe interactively. The
axes are **referenced to bregma**: ML = 0 at the midline, AP = 0 at bregma
(anterior positive), so coordinates read like stereotaxic values.

**Save / Load (Project menu)**  
**Project → Save Project** writes `<slide_name>.histo2ccf.json` next to the image
by default. **Load Project** restores the slide, sections, registration, probes and
tip/entry, repopulates every tab's fields, and auto-loads the project's atlas in
the background.

To reload a session in Python:

```python
from histo_to_ccf.project.io import load_project
p = load_project(r"path\to\project.json")
```

**HERBS pkl** (legacy compatibility)  
Click **Export HERBS pkl…** - writes a `.pkl` readable by the old pipeline.

**Per-channel CSV**  
Click **Export per-channel CSV…** - writes `probe, shank, channel, ap_um, ml_um, dv_um` for all 384 channels.

**Per-channel Paxinos CSV**  
Pick a **Paxinos align** transform, then click **Export per-channel Paxinos CSV…** -
the same channels in **Paxinos stereotaxic mm** (bregma origin): `ap_mm`
anterior-positive, `ml_mm` 0 at the midline, `dv_mm` depth below bregma. CCFv3 is
pitched ~5° nose-down vs a flat-skull frame, so the transform **un-pitches 5°** and
applies published axis scaling (**Qiu 2018** is the default; *Dorr 2008*, *Allen
forum*, or *None* = plain mirror are also available). These are **estimates with
real variance - validate against histology** before trusting absolute values.

---

## 6b - Ephys tab: refine shank depth from LFP (optional)

Requires the `ephys` extra (`uv pip install -e ".[ephys]"`) and shanks that already
have CCF tip/entry from registration. This refines the depth→CCF mapping the way
the IBL ephys-alignment GUI does.

1. Pick the **Probe** and **Shank** to align.
2. Under **Recording (Open Ephys)**, **Browse…** to the Open Ephys record-node
   folder. Click **List streams** and leave **Stream** on *Auto* (uses the LFP
   stream; for Neuropixels 2.0, which has no LFP stream, LFP is derived from the
   AP stream).
3. Set **Seconds to analyse** (default 60) and click **Load & compute LFP power**.
4. Click **Open alignment…**. The dialog shows the depth×frequency **LFP power
   map** (left) beside the atlas **region colour strip** (right, now labelled with
   region **acronyms + names**), sharing a depth axis (tip at the bottom, surface at
   the top). The header states how many channels this **shank** has and how many
   **rows** (a Neuropixels 2.0 row holds 2 sites, so 96 channels = 48 rows), the
   electrode span, and the histology track length (always ≥ the electrode span).
   The depth axis is the **histology track**; the recorded electrodes occupy only
   the **green dashed bracket**, which sits above the tip (the shank tip extends
   below the lowest electrode).
5. Two **anchors are pre-set** at the recorded-block edges. Drag the red **anchor
   lines** so LFP power transitions line up with region boundaries (use **Add
   anchor (mid)** to create one, **Remove selected** / **Clear anchors** to manage
   them). **Save LFP power…** exports the per-channel power (.npz/.csv).
6. Click **Apply** - each channel is placed on the tip→entry line and the
   per-channel CCF coordinates (plus the anchors) are stored on the shank and saved
   with the project.

---

## 7 - Headless / CLI path (no GUI)

```
# Detect sections and write a sidecar JSON
histo2ccf split --image data\slide.tif

# Single-section registration (manual mode, no atlas download needed)
histo2ccf register-one ^
  --image data\slide.tif ^
  --ap-um 5400 ^
  --tip 512,900 ^
  --entry 512,120 ^
  --midline-px 512 ^
  --dorsal-surface-px 120 ^
  --pixel-size-um 2.0 ^
  --output-pkl auto

# Full M3 pipeline on a saved project
histo2ccf register project.histo2ccf.json --atlas allen_mouse_25um
```

---

## What "passing" looks like

| Check | Expected |
|---|---|
| Sections detected | ≥ 1 yellow rectangle per brain section |
| Atlas loaded | Status shows resolution; AP range updates |
| Registration completes | All residuals shown, no error dialog |
| Plotly HTML opens | Probe line visible inside semi-transparent brain |
| HERBS pkl | Written for the HERBS post-processing format |
| Per-channel CSV | 384 rows × 6 columns for NP 1.0 |
