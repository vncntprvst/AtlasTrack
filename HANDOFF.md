# Histo_to_CCF - Handoff

_Last updated: 2026-06-09 · version **0.2.13** · branch **newUI** (committed, not pushed)_

## TL;DR

A napari desktop app that registers mouse histology sections to the Allen CCF and
maps probe (Neuropixels / NeuroNexus) trajectories to CCF coordinates. The core
pipeline (sectioning → annotate → DeepSlice plane prediction → 2D B-spline →
per-channel CCF coords → 3D viz / exports) is implemented and **131 tests pass**.

**The GPU/OpenGL launch blocker is RESOLVED** - a reboot cleared the
`QOpenGLFramebufferObject: Unsupported framebuffer format` failure. The GUI
launches and **end-to-end registration completes without crashing** (TensorFlow
in the DeepSlice subprocess keeps the memory peak down). If the GL error ever
recurs, `histo2ccf gl-info` still diagnoses it and a reboot/driver fix is the cure.

> Work on `newUI` is **committed but not pushed** (ahead of `origin/newUI`).
> Recent commits: atlas-matcher dialog, Save/Load → menu, flip restore + lazy
> atlas on load, layer panels hidden, section-scope picker + working levels,
> Annotate→Probes rename.

## How to run

```powershell
# from repo root, with .venv active (editable install - no reinstall after edits)
histo2ccf gui            # launch the GUI
histo2ccf gl-info        # diagnose GPU/OpenGL if the GUI won't start
histo2ccf version
.venv\Scripts\python.exe -m pytest -q   # 131 tests
```

DeepSlice is installed (optional extra, pulls TensorFlow). It runs CPU-only on
Windows.

## Workflow (what the GUI does)

Layout: the **Registration** panel (5 tabs) docks on the **left**; a permanent
**3D & Export** panel (`VizExportPanelWidget`) docks on the **right**. Project
Save/Load live in the **Project menu** (inserted first in the menu bar). Tab order
left→right: **Histology → Atlas → Probes → Register → Ephys**.
1. **Histology** (was "Load") - open one or more composite slides; multiple opens
   are **merged into a single combined image** (see "Multiple slides" below).
   Auto-detect sections (Otsu + connected components). Auto-estimates min-area.
   "Equalize under-sized boxes" grows boxes smaller than the slide's median. **Edit
   boxes** turns detections into draggable napari rectangles, synced live. Image
   tools: flip H/V and per-channel levels, scoped to the whole (merged) slide or a
   **Selected section** chosen from a dropdown.
2. **Atlas** - choose atlas + storage folder; **AP shown relative to bregma**
   (0 = bregma). **Open atlas matcher…** (side-by-side / overlay AP matching;
   syncs AP + spacing with this tab on open/close). Assign AP per section, or
   reorder/space them in the ordering panel (spacing persists on the project;
   "Interpolate AP" fills gaps between hand-assigned sections).
3. **Probes** - pick a probe (Neuropixels 1.0, NP 2.0 4-shank, NeuroNexus …).
   "Add probe" auto-arms Tip-marker mode. **Probe + shank are selected by label**
   (combos, consistent with the Ephys tab). Click to drop tip; switch to Entry
   (Marker, or draw a Trajectory line whose tissue-surface crossing = entry).
4. **Register** - optionally **Predict planes with DeepSlice**, then per-section 2D
   refinement. **Engine: elastix (regularized) by default** - a bending-energy
   penalty + tissue mask (ABBA-style) keep atlas boundaries on the tissue; falls
   back to the plain SimpleITK B-spline when `itk-elastix` is absent. Controls:
   "Regularized registration (elastix)" toggle, "Smoothness (bending energy)"
   weight, "Restrict to tissue mask", **"Snap atlas contour to tissue"** (the
   automatic outer-contour fix, any engine - see below). Residuals table. "Show
   atlas overlay" warps
   registered region boundaries onto each section (lazily loads the atlas).
   Auto-saves the project. (See "Registration engine" below.)
5. **Ephys** - refine a registered shank's depth→CCF mapping from LFP features
   (see "Ephys alignment" below).

**3D & Export panel (right dock):** Extra-regions field, **View in napari 3D**,
**Export Plotly HTML**, **Export HERBS pkl**, **Export per-channel CSV** - always
available, not gated behind the Register tab.

Save/Load: **Histo→CCF menu** → Save / Save As… / Load Project (`.histo2ccf.json`).
Load restores the merged slide + sections + registration (CCF coords come back;
exports work without re-running), re-applies stored flips, and **auto-loads the
project's atlas in the background** (overlay / 3D ready without a manual click).
**Load also repopulates every tab's fields** from the project via per-widget
`refresh_after_load()` (probes + tip/entry markers/table, atlas selection + AP,
section ordering list + spacing, residuals, ephys combos) - wired through
`_on_project_loaded` in `app.py`. Section spacing now persists on the project
(`Project.section_spacing_um`).

3D / export: **Export Plotly HTML** and **View in napari 3D** (opens a *separate*
window; lazily loads the atlas so brain volumes show). HERBS pkl + per-channel CSV.

## Multiple slides → merged single image

Probes can have an entry on a section in one slide and a tip on a section in
another, so all sections must share **one coordinate space**. Rather than track
per-slide offsets through clicks/registration/3D (brittle), **opening multiple
slide images merges them into one combined image** (sections from all slides laid
out in a single canvas). The user is told this on load. After the merge the rest
of the pipeline is unchanged - there is effectively one slide. The napari built-in
**layer list / layer controls panels are hidden** (the workflow panel drives
everything), so slides are never managed as separate layers.

## Ephys alignment (v0.2)

IBL-style depth refinement of probe shanks from LFP. Pure-core package
`ephys/` (no Qt): `alignment.py` warps channel *feature depth* (µm from tip) to
*track depth* via anchor points (piecewise-linear + linear extrapolation) and
places each channel on the shank's `tip_ccf_um`→`entry_ccf_um` line;
`features.py` computes the depth×frequency LFP power map (Welch PSD); `regions.py`
samples the atlas region at each depth; `loader.py` reads Open Ephys LFP via
**SpikeInterface** (optional `ephys` extra - gated import, `si.read_openephys` /
`get_neo_streams`). NP 1.0 has a dedicated LFP stream; when none exists LFP is
derived from the AP stream (band-limit + resample). All the math is unit-tested
without SpikeInterface installed.

GUI: the **Ephys** tab (`gui/widgets/ephys_panel.py`) picks a probe+shank, loads
the recording (background `lfp_power_worker`), and opens
`EphysAlignmentDialog` - LFP power map (warped into track space) beside the atlas
region colour strip, shared depth axis (tip at the bottom), draggable red anchor
lines. Apply writes `Shank.ephys` (`EphysAlignment`: anchors + per-channel
`channel_ccf_um`), which round-trips through the project JSON.

The alignment dialog: **double-click the LFP map** to drop an anchor at that depth
(or "Add anchor (mid)"), then drag it; left margin shows top/bottom channel # +
depth µm (tip at bottom), header shows tip/entry CCF; **"Normalize per frequency"**
toggle (`power_image(per_freq=True)`) scales each frequency column independently so
depth structure pops. **Apply auto-saves** the project (if it has a path) and the
ephys per-channel CCF render as Points layers in the napari 3D view
(`add_ephys_channel_layers`). Compute does **not** cache - it re-reads/filters each
time.

Open follow-ups: per-channel region CSV export; richer anchor UX (snap to region
boundary); shank-column auto-detection for 4-shank probes is heuristic (sorted
unique x → shank index).

## Registration engine (v0.2.6 - elastix + regularization)

The per-section refinement has **two interchangeable engines**, both returning a
`sitk.Transform` (FIXED atlas-slice → MOVING histology) so the entire downstream
layer - point mapping, the iterative inverse in `registration/transforms.py`, and
`.h5` persistence - is **unchanged**:

- **`sitk`** (`registration/bspline.py`) - the original affine + 8×8 B-spline,
  Mattes MI, no mask, no deformation penalty. Kept as the fallback.
- **`elastix`** (`registration/elastix_bspline.py`, ABBA-style) - affine + B-spline
  via **itk-elastix** with a **`TransformBendingEnergyPenalty`** (smoothness) and a
  **tissue mask** on both images. This is the fix for "atlas boundaries flying off
  the bottom." Optional `elastix` extra (`pip install -e .[elastix]`, pulls ITK
  ~150 MB).

**Integration trick (don't undo this):** elastix has its own transform format, but
the rest of the package consumes a `sitk.Transform`. So we run elastix, pull the
**combined (affine∘bspline) deformation field** via `transformix`, and wrap it as a
`sitk.DisplacementFieldTransform` inside a `CompositeTransform`. That object
composes, inverts (via the existing `_invert_displacement` fallback - a DFT has no
cheap analytic inverse) and round-trips through `.h5` exactly like the native
B-spline. **Cost:** each sidecar is now a full displacement field (~4 MB for a
500² crop) instead of a few B-spline coefficients.

**Masks are DILATED, deliberately** (`_MASK_RIM_FRAC = 0.10`). A *tight* tissue
mask was empirically *worse* - it excludes the brain/background boundary, the
strongest alignment cue. Dilating keeps the outline + a background rim while still
excluding far-field junk (the green canvas border, debris, neighbouring-section
pixels caught in a crop's bbox). Verified on synthetic data: tight mask made MSE
worse (0.06→0.09); dilated mask recovered it (0.06→0.0034).

Engine selection: `engine="auto"` (default) uses elastix when installed, else
sitk; explicit `"elastix"` raises if itk-elastix is missing. Plumbed through
`AppSettings.reg_engine` / `bending_energy_weight` / `use_tissue_mask`,
`register_section_image`, `register_project_with_atlas`, both `register_worker*`,
and the Register panel UI.

**Overlay rendering - the dominant fix (v0.2.7).** The "atlas boundaries flying off
the bottom" was found, by rendering the *real* section-12 overlay, to be mostly a
**rendering artifact**, not bad registration. `warp_annotation_to_section` warps the
annotation through an **inverse displacement field** (`_invert_displacement`), which
**extrapolates nonsense outside the registered tissue** and draws boundary "stripes"
far off the section - and a displacement-field transform (the elastix engine)
inverts *worse* than the old `Affine∘BSpline`, which is why v0.2.6 looked worse than
the original. Fix (v0.2.8): `warp_annotation_to_section` clips the warped labels to
the **forward-warped atlas extent** (`_warped_atlas_extent` - forward-splat the atlas
foreground through the transform, then close+fill). This removes the stripes while
**keeping every region outline inside the brain, including over damaged/dim tissue**.
(An earlier v0.2.7 attempt clipped to the *tissue* silhouette, which wrongly deleted
outlines over damage and cut the outer contour at the tissue edge - don't
reintroduce that.) Internal *folds* in a damaged section are not removed by clipping;
that's a real registration limit - see manual-landmark follow-up.

**Masks (v0.2.7), `registration/masks.py` (headless, RGB-aware).**
`section_tissue_mask` (brightest-channel Otsu×0.5 + heavy close + largest component)
is far better than the old luminance-Otsu. `section_label_mask` flags the bright
green/magenta **fluorescent labels** (R or G high; DAPI tissue is blue).
`registration_moving_mask` = dilated tissue **minus** labels → the elastix metric
mask the pipeline now passes in (`register_section_image` builds it from the RGB crop
*before* the luminance collapse). This is the requested **saturated-label refinement**
- it stops the labels (which have no atlas counterpart) from pulling the fit. Also
added `RequiredRatioOfValidSamples=0.05` to the elastix params so a tight mask
doesn't abort with "too many samples map outside the moving image", plus `metric`
("mi"/"meansquares") and `deformable` knobs on `refine_with_elastix`.

**What was tried and rejected (don't redo):** registering the tissue **silhouette**
(binary mask / distance transform) with mean-squares - a full affine finds a
degenerate **shear**, and a flexible/stiff B-spline **folds** the asymmetric
forebrain (worse than intensity+clip). Plain elastix-MI on luminance, label-masked,
with the overlay clipped, was the best automatic result on section 12. The forebrain
of that section is genuinely damaged/asymmetric and needs manual correction.

**Caveats / open follow-ups for this engine:** the residual is a normalized-intensity
RMS over the fixed foreground (lower=better), *not* the old MI metric - old vs new
numbers aren't comparable. Tissue Dice as a QC metric and a **VisuAlign/BigWarp-style
manual landmark warp** (the real fix for damaged/asymmetric sections) are the two
biggest remaining levers.

## Silhouette pre-align (v0.2.10 - consistent outer-contour scale)

The atlas-plane-to-crop scale was inconsistent per section (§12 too big, §1 too
small) because intensity-MI never measures the silhouette. Fix: a **per-section,
closed-form similarity pre-align** before the B-spline. `masks.moment_similarity`
matches the atlas-mask and tissue-mask **centroids (translation)** and **area
(isotropic scale)** - 4-DOF, *cannot shear or fold*. It's computed independently
for each section from that section's own masks (nothing shared across sections).

**Isotropic on purpose:** area is rotation-invariant, so the scale can't be fooled
by a slightly-rotated section; an anisotropic per-axis scale *would* misread
rotation as a stretch (it broke the synthetic MSE test - don't switch it back
without handling rotation). Rotation DOF is dropped (the anchoring already orients
the plane; principal-axis angle has a sign/180-degree ambiguity).

Integration (`refine_with_elastix(prealign=True)`, default on): warp the atlas
reference + its mask into the section frame by `S`, run the B-spline on the
pre-aligned pair for the residual `R`, return **`CompositeTransform([R, S])`** -
last-in-list applies first, so `T(a) = R(S(a))`. Everything downstream (inverse,
overlay clip, persistence, probe map) is unchanged because it's still one
`sitk.Transform`. Verified on real §1 (was inset → now fills) and §12 (tightened),
+ synthetic round-trip. The external-initial-transform route was a dead end
(elastix "Not implemented" with the bending-energy metric). Toggle: Register tab
"Silhouette pre-align" / `AppSettings.prealign_similarity`. Composes *under* the
manual drag tool (pre-align gets close; drag finishes damaged sections).

## Automatic outer-contour snap (v0.2.13 - the real fix for "lines off the bottom")

The intensity B-spline (elastix MI) aligns *interior* structure but never
optimises the silhouette, so the atlas **outer contour** is left wherever the
affine pre-align put it - a few percent off the tissue, the classic "atlas lines
just outside the bottom of the section." Measured on the real example slide (15
sections, silhouette Dice = warped-atlas-extent vs section tissue mask) the match
**plateaus at ~0.89 no matter how the B-spline or pre-align is tuned** - because
MI doesn't see the boundary, and a PCA rotation/anisotropic pre-align only moved
it 0.896 -> 0.897 (the anchoring already orients sections). A free mean-squares
B-spline on the silhouettes folds (empirically reconfirmed: min Jacobian < 0 on
every section). So the engine now adds a **fold-proof boundary snap** as a final
step (`registration/boundary_snap.py`, headless):

- Sample points on the warped-atlas boundary; push each toward the nearest tissue
  edge (tissue **signed distance field** + its gradient).
- **Drop** any push larger than `_DROP_FRAC` (0.06) of the diagonal - *exclude* it,
  don't pin it (pinning beside a 50 px-snapped neighbour creases the field and
  folds it). This is what protects genuinely **damaged/asymmetric** tissue: where
  the atlas has no matching edge, the atlas is left alone there (§12's torn
  cerebellar midline stays put; only the intact bottom contour snaps).
- Pin a ring of **interior anchors** (zero displacement) so it's a boundary
  correction, not a global drift.
- Fit a **smoothed thin-plate spline** (`RBFInterpolator`, smoothing escalates
  2000 -> 4000 -> 8000) and **verify the forward field's min Jacobian > 0.02**;
  if every level still folds, return `None` (keep the un-snapped fit). Empirically
  fold-free at smoothing 2000 (worst Jacobian +0.11 across the slide).

Result on the real slide: **mean silhouette Dice 0.894 -> 0.942, every section
improved**, including damaged §12 (0.742 -> 0.79, gently). Rendered overlays
confirm it visually (the bottom-corner "fly-off" on §7/§12 is pulled onto the
tissue; the damaged region is untouched) - see `example data/reg_eval/` for the
before/after PNGs this was validated against.

**Integration (one `sitk.Transform`, nothing downstream changes):** the snap is a
`sitk.DisplacementFieldTransform` in the section (moving) frame mapping
*where the registered atlas landed* -> *tissue*. `register_section_image` composes
it as `CompositeTransform([snap, registration])` then **`FlattenTransform()`** -
the elastix engine already returns a composite, and HDF5 rejects a *nested*
composite ("Composite Transform can only be 1st transform in a file"), so it must
be flattened to persist. Verified: `.h5` write/read is exact (0.0000 px), the
iterative inverse used by probe->CCF round-trips (0.0014 px). The whole thing is
**best-effort**: any failure logs and keeps the un-snapped registration, so it can
never break a section. Toggle: Register tab **"Snap atlas contour to tissue"** /
`AppSettings.boundary_snap` (default on). Unlike the elastix-only knobs it works
with **any engine** (it's a post-fit step), so it's not gated on the elastix
checkbox. Threaded through `register_section_image`, `register_project_with_atlas`,
both `register_worker*`, and the panel.

This is the automatic equivalent of the manual landmark drag - for most sections
it removes the need for any manual correction; the manual tools remain for the
genuinely damaged ones the snap deliberately leaves alone.

## Manual atlas correction - two tools (mutually exclusive per section)

`Section.manual_affine` (box handles, v0.2.9) **or** `Section.manual_landmarks` (TPS,
v0.2.11) - landmarks take precedence in `RegisteredSectionTransform.apply` when both
somehow exist; the GUI clears the other when you apply one.

### Landmark / thin-plate-spline warp (v0.2.11 - the VisuAlign-style tool)

For *local* distortions a single affine can't fix (e.g. §12's damaged dorsal
cerebellar midline). **Register tab → pick section → "Place landmarks"** drops 6
border + 3 interior draggable points (`landmarks_warp.auto_landmarks`: ray-march the
warped-atlas extent for the ring, vertical midline for the interior). Drag each onto
the matching tissue (napari Points `select` mode), then **"Apply landmark warp"**.

A thin-plate spline (`scipy RBFInterpolator`, `kernel="thin_plate_spline"`) maps
**source→target** and is **baked into the overlay label image** (`warp_label_image`,
pull-back resample) - not the layer affine, since napari affine is linear-only. The
probe→CCF inverse uses a **second TPS fitted target→source** (`invert_points`); the
two are exact inverses *at* the control points and sub-pixel-approximate between
(fine for probes). Landmarks persist as section-local (x, y) `source`/`target` lists
and re-warp on overlay render / reload. `_place_landmarks` continues from stored
landmarks if present (drag, re-apply), else auto-places fresh. Validated on real §12
(smooth local deformation, rest pinned) + headless math/probe/schema tests + a qt
smoke test of apply/reset.

**Editing the points (v0.2.12).** Each landmark carries its **atlas anchor (source)
in the napari Points `features`** (`sy`/`sx`, world coords) so it travels with the
point through add/delete automatically; `layer.data` is the **target**.
`_on_landmark_data` (on `layer.events.data`) keeps them in sync:
- **plain drag** → moves the target only = warp (anchor in features unchanged);
- **Ctrl+drag** *or* **"Move points"** toggle → also shifts the anchor by the same
  delta = relocate, displacement preserved. Ctrl is read by a thin
  `mouse_drag_callbacks` generator (`_landmark_drag_modifier`) that just sets
  `self._lm_ctrl_drag`; the built-in drag still does the actual point move, so there's
  no conflict. (Modifiers aren't on `events.data`, hence the flag + a toggle fallback.)
- **"Add points"** toggle → Points `add` mode; new rows get their anchor set to the
  drop position (napari copies the last feature value on add, so we overwrite it).
- **Delete** key → features drop the row in lock-step (verified). Apply requires >=4.

napari gotcha: `layer.features[col]` is a **read-only** pandas Series - `np.array(...)`
(copy) before mutating, then reassign `layer.features = {...}`.

**Open polish:** live overlay preview is still on "Apply" (not during drag); no
source-anchor ghost/vector shown (anchors are invisible features); could seed targets
from a finer auto-fit.

### Box-handle affine (v0.2.9 - quick global nudge)

When automatic registration can't snap a damaged/asymmetric section, the user
corrects it by hand: **Register tab → "Manual atlas adjustment"** → pick a section
→ **"Adjust atlas (drag in viewer)"**. That puts the section's atlas-overlay Labels
layer into napari's built-in **`transform` mode** (drag the body to move; box handles
to scale / stretch X-Y / rotate). "Apply adjustment" commits; "Reset adjustment"
clears it.

Data model: napari writes a 3x3 **world** affine to `layer.affine`; we convert it to
a bbox-independent **section-local** affine (`registration/manual.py`,
`world_to_section` = conjugate by the bbox origin) and store it on
`Section.manual_affine` (round-trips through the project JSON). On overlay render /
project load the stored affine is pushed back onto the layer (`section_to_world`), so
corrections persist visually.

Composition (the correctness-critical bit): the manual affine `A` maps a *registered*
atlas position to where it was dragged. For probe -> CCF, a tissue point is fixed and
the atlas moved, so `RegisteredSectionTransform.apply` first pulls the clicked point
back with `A^-1` (`manual.invert_apply`) before the registration inverse. `Apply`
re-maps every probe (`reload_registered_transforms` now threads
`Section.manual_affine` into `build_registered_transform`) and auto-saves. Live drag
preview is free (napari renders `layer.affine`); only commit bakes it in.

Tested headless: world<->section round-trip, `A^-1` point map, a translation affine
shifting the probe CCF by the matching pixels, schema round-trip; plus a qt smoke test
(enter transform mode, set an affine, apply -> `manual_affine` stored + saved, reset).
**Open polish:** the adjust picker doesn't yet auto-select the section nearest the
viewport; per-section "stretch only" vs "move only" lock; an undo.

## Architecture notes / hard rules

- `src/histo_to_ccf/` layered: `io/`, `atlas/`, `sectioning/`, `landmarks/`,
  `registration/`, `probes/`, `viz/`, `gui/`. **Only `gui/` and `viz/napari3d.py`
  may import napari/Qt/magicgui** (import-linter contract). Core stays headless-
  testable.
- Coordinate convention: CCF **(AP, ML, DV)** in µm everywhere in our code.
- Project state is one Pydantic v2 `Project` (`project/schema.py`) serialized to
  `<slide>.histo2ccf.json`, with B-spline transform sidecars in a sibling
  `transforms/` dir.

## Hard-won gotchas (don't re-learn these)

- **napari 0.7**: `Viewer` is a Pydantic model with no `mouse_press_callbacks`;
  Points use `border_color` not `edge_color`; click-to-pick uses a Labels layer in
  `mode='pick'` + `events.selected_label`.
- **DeepSlice runs in a subprocess** (`registration/deepslice_run.py`). This is the
  fix for the original "crash after registration" - TensorFlow's ~2 GB is released
  before the memory-heavy atlas registration. `predict_anchorings` writes
  `section_s<idx>.png` (DeepSlice needs the `_s<n>` token), runs the subprocess,
  parses the QuickNII JSON. **Do not import DeepSlice in the GUI process.**
- **DeepSlice/QuickNII anchoring axis order is `(ML, AP, DV)`**, opposite of our
  `(AP, DV, ML)`. `_quicknii_to_atlas_anchoring` permutes **and flips AP + DV**
  (QuickNII runs those axes the other way). ML flip is the suspect if a registered
  slice ever looks mirrored (`_FLIP_ML` constant). Verified against real data:
  posterior sections land at AP≈420, dorsal at top.
- **Memory**: never cast the whole atlas to float32. `sample_plane(..., out_dtype=
  np.float32)` interpolates uint16→float per section slice. Registration is
  per-section with try/except so one bad section can't abort the batch.
- **DeepSlice `save_predictions(name)` appends `.json`** - pass a base path, read
  back `name.json`.
- Tip/entry are clicked in **slide-global** pixels but the section transform is on
  the **section crop** - `_apply_to_shank_registered` subtracts the bbox origin
  (this was the "probe 60 mm outside the brain" bug).
- 3D region colors use a **curated palette** (`viz/plotly3d.REGION_STYLE`), not the
  muddy native atlas colors; siblings get distinct fallback colors. Context shell
  (Isocortex/CB/BS) is faint + additive blending so probes show through; only
  shank-tip regions are shown by default; extra regions via a text field.
- **Windows console is cp1252** - keep CLI/diagnostic output ASCII (no `→`, `-`).

## The GPU/OpenGL launch blocker (RESOLVED)

Symptom (now gone): `histo2ccf gui` printed repeated `QOpenGLFramebufferObject:
Unsupported framebuffer format` and rendered nothing. It was a GPU driver /
Remote-Desktop / dual-GPU session issue, not a histo2ccf bug - and **a reboot
fixed it**. The GUI launches and registration runs end-to-end.

Diagnostics left in place in case it recurs:
- A previous `--software-gl` attempt was **wrong** (set Qt software GL but vispy
  uses PyOpenGL → hard `GLError` crash) and has been **removed**.
- `histo2ccf gl-info` probes OpenGL **in a subprocess** (survives a native GL
  segfault) and prints the renderer + diagnosis + ordered driver fixes.
- `launch()` catches a failed `napari.Viewer()` and prints that diagnostic
  instead of a traceback.

If the GL error returns: run `histo2ccf gl-info`. "Microsoft Basic Render
Driver"/"GDI Generic" or a crashing probe → RDP or dead driver (run on the
console / fix driver). Real GPU named → reboot (worked this time), then
update/roll back the GPU driver.

## State of testing

- `pytest -q` → **191 passed** (165 non-qt + 26 qt; +4 in `test_boundary_snap.py`
  for the outer-contour snap - improves silhouette Dice, never folds, drops
  damage-region mismatches, persists through `.h5`; the qt landmark test now also
  covers warp/relocate/add/delete editing; elastix engine adds 6 in
  `test_elastix_bspline.py` - skipped if itk-elastix absent - 4 in `test_masks.py`,
  3 in `test_overlay_extent.py`, 5 in `test_manual_affine.py`, +1 qt manual-adjust,
  +2 moment-similarity in `test_masks.py`, +1 pre-align in `test_elastix_bspline.py`,
  +6 landmark-TPS in `test_landmarks_warp.py`, +1 qt landmark; run qt per-process - the
  napari GL context corrupts across many viewers in one process on this machine,
  so a single `pytest -q` run can hit a Windows access violation mid-suite even
  though every test passes alone). Includes: core pipeline, sectioning/ordering,
  ephys alignment math + LFP features + region strip + schema round-trip,
  probe geometry/channels, transforms, DeepSlice anchoring conversion (permute +
  flip + scale, mocked subprocess), region styling, mesh extraction, and
  `@pytest.mark.qt` GUI smoke tests (full-panel build, edit-boxes, atlas bregma,
  ordering, click-overlay, atlas matcher, project-menu present, flip-restore on
  load, section-scope flip, GL diagnostic never-raises).
- Can't be tested in CI here: live GL rendering and a real DeepSlice model run -
  both **verified manually in the running GUI** (GUI launches, registration
  completes end-to-end without crashing).

## Open items / next steps

0. **Registration quality** - done: elastix engine (v0.2.6); RGB label-excluding
   metric mask + sampling robustness (v0.2.7); **forward-warped-atlas-extent overlay
   clip** (v0.2.8 - fixes "lines flying off" AND preserves outlines over damage).
   Validated on real sections 1 + 12. **Known remaining limit:** the automatic
   registration does **not reliably snap the atlas OUTER contour to the tissue
   border** - the intensity-MI fit doesn't optimise the silhouette, so global scale
   is inconsistent (section 12 atlas slightly too big, section 1 slightly too
   small). Silhouette-based fixes were tried and rejected (shear / fold on
   asymmetric sections). **Manual per-section correction shipped (v0.2.9)** - drag the
   atlas overlay (see "Manual atlas correction"). (Section detection bboxes are fine -
   ~0% tissue cut; the "atlas leaving the box" look is the atlas scale, not the box.)
   **Silhouette pre-align shipped (v0.2.10)** - fixes the per-section scale
   inconsistency (see "Silhouette pre-align"). **Automatic outer-contour snap
   shipped (v0.2.13)** - the actual fix for "atlas lines off the bottom" and for
   the inconsistent outer-contour fit: silhouette Dice 0.894 -> 0.942 across the
   real slide, every section improved, fold-proof, damaged tissue left alone (see
   "Automatic outer-contour snap"). This removes the need for manual correction on
   most sections. Remaining lever: surface the per-section silhouette **Dice as a QC
   metric** in the residuals table to flag the few sections still worth a manual
   look.
1. **Ephys per-channel CCF in 3D / export** - surface the ephys-refined channels
   (regions + CCF) in the napari 3D view and a per-channel region CSV (see "Ephys
   alignment" follow-ups). Done earlier: multiple-slide merge, Ephys tab.
2. Eyeball DeepSlice planes for a **left/right mirror** (flip `_FLIP_ML` if so).
3. **Push** `newUI` to origin when ready (committed locally, not pushed).
4. Possible follow-ups discussed but not built: auto-clean DeepSlice AP outliers
   (neighbor smoothing), a globally-coupled (vs per-section) registration, and a
   true Mesa software-GL option (the bundled `opengl32sw.dll` exists but vispy/
   PyOpenGL need `VISPY_GL_LIB` wiring - not done, and deprioritized in favor of
   fixing the driver).
```
