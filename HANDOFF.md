# Histo_to_CCF - Handoff

_Last updated: 2026-06-27 · version **0.2.37** · branch **dev**_

## Salient landmarks + probe rename + user manual (v0.2.37)

**Salient landmark placement.** `landmarks_warp.salient_landmarks(labels)` replaces
the by-angle `auto_landmarks` for the initial `Place landmarks` points: candidates
in priority order = silhouette tips (`_silhouette_extremes`) → region-outline
junctions (`_region_junctions`: skeleton branch points of the boundary) → high-
curvature corners (`_silhouette_corners`: Harris peaks) → the geometric ring as
fill; a greedy spread keeps them ≥10% of the image apart, and `_snap_into_extent`
pulls any off-mask corner back on. So points land on grabbable features where they
exist (busy dorsal cortex) while tips+ring guarantee coverage of cerebellum/
brainstem. Falls back to `auto_landmarks` on a degenerate extent. Wired in
`register_panel._place_landmarks`. Tested in `test_landmarks_warp.py`
(junction-targeting, spread, fallback); visually verified on sections 8 & 12.

**Rename probes.** `probe_picker.py` gains a Rename row (combo of probes + new-label
edit + button). `_rename_probe` updates `Probe.label`, rejects duplicates (labels
are export keys in `probes/channels.py`), and fires a new `on_probes_changed`
callback wired in `app.py` to `_refresh_panels` so the Probes tip/entry and Ephys
combos pick up the new name. Tested in `test_gui_smoke.py::test_probe_picker_rename`.

**User manual.** New `MANUAL.md` — cookbook-style reference (concepts, GUI
reference, task recipes, troubleshooting) modeled on the HERBS cookbook,
complementing the linear `TUTORIAL.md`. Linked from README. `pypdf` was installed
into the `.venv` to read the cookbook.


## Real-time landmark drag + DeepSlice AP-order safeguard (v0.2.36)

**Real-time landmark dragging** (Register tab). "Apply landmark warp" was slow
because one click did four heavy things: full-label TPS warp (RBF over every
pixel), reload all transforms, re-map all probes, save the whole project. Now a
drag shows a live preview by re-warping only the atlas **boundary contour**: on
`Place landmarks` the un-warped boundary is cached (`register_panel`:
`_lm_base_edge_rc`, subsampled ≤4000 pts); each `_on_landmark_data` calls
`_preview_landmark_warp`, which forward-TPS-warps that contour
(`landmarks_warp.warp_contour_image`) and rasterises it into the overlay layer -
follows the cursor in real time. The exact full warp + probe re-map + save stay
in `_apply_landmarks` (one click at the end). Pure core tested in
`test_landmarks_warp.py` (identity/drag/clip).

**DeepSlice pre-match AP-order safeguard** (Atlas matcher). DeepSlice only
`enforce_index_order()`s (monotonic), never spacing, so it could return sections
out of order or two almost on top of each other (the bug the user hit). Now:
- Before running: blocks if two sections share an `ap_order` rank (fix the order
  window first); warns if spacing is 0 µm (offers to proceed).
- After running: `_warn_if_prematch_disordered` flags AP steps that reverse the
  series direction or collapse below 30% of the median step, in a QMessageBox
  naming the offending section pairs, so the user evens them out before
  registering. Headless core = `pipeline.prematch_ap_order_issues`, tested in
  `test_prematch_deepslice.py`. Decision: warn, don't auto-force even spacing
  (wrong with missing sections) - see memory `project_deepslice_prematch_safeguard`.

Still open from this discussion (agreed, not yet built): place landmarks at
salient corners/junctions (not by-angle); rename probes (`Probe.label` exists,
set only at creation); a HERBS-cookbook-style user manual (no PDF lib in `.venv`
yet to read `C:\Code\Registration\HERBS\CookBook.pdf`).

## Fix: "Load atlas" hangs the GUI (and blocks exit) (v0.2.35)

Loading an already-downloaded atlas could freeze on `load_atlas_worker`
indefinitely (napari close dialog stuck "Executing load_atlas_worker"), and the
terminal wouldn't return after closing because the blocked worker thread(s) kept
the process alive. Re-clicking "Load atlas" stacked more hung workers.

Root cause: `BrainGlobeAtlas(...)` defaults to `check_latest=True`, which calls
`check_latest_version()` -> `remote_version` -> `utils.conf_from_url()` - an HTTP
GET to the GIN server with **no timeout**. When GIN is slow/unreachable that call
hangs forever inside the worker, even though the atlas is fully present on disk.

Fix:
- `workers.load_atlas_worker` now passes `check_latest=False` (skips only the
  online version courtesy-check; a genuinely-missing atlas is still downloaded).
  Load of `allen_mouse_25um` drops to ~0.2s and is offline-safe. Trade-off: no
  auto-notification when a newer atlas version exists - acceptable for reliability.
- `atlas_browser._load_atlas` guards re-entrancy: disables the Load button +
  ignores clicks while a load is in flight (re-enabled on the worker's `finished`
  signal), so a slow load can't stack multiple workers.

If a stuck process needs killing: it's blocked on a network read, so Ctrl+C may
not land - `Stop-Process -Name python -Force` (or close the terminal) clears it.

## Fix: cyan/green-stained sections failed registration (v0.2.34)

Whole slides where the sections are rendered **cyan** (high green + blue, e.g. a
DAPI channel mapped to green+blue) failed elastix with an instant, uniform
`Internal elastix error` for every section of that slide, while magenta sections
on the same image registered fine. Real ITK error (surfaced by running elastix
with `log_to_console=True`): `AdvancedMattesMutualInformationMetric: Too many
samples map outside moving image buffer: 20 / 3693` → `Error in metric`.

Root cause: `masks.section_label_mask` flags any `r>70 or g>70` pixel as a
fluorescent **label** to exclude from the registration metric mask (assuming
labels are green/red and DAPI tissue is pure blue). Cyan tissue has ~65% of its
own pixels above the green threshold, so `registration_moving_mask` subtracted
most of the tissue, leaving a ~15-19% scattered mask. elastix then drew nearly
all metric samples outside the valid (masked) moving region and aborted.

Fix (`masks.registration_moving_mask`): label-exclusion now backs off when it
would remove most of the tissue - if `kept < 0.5 * dilated_tissue`, the bright
R/G signal *is* the stain (not a sparse label) so the full tissue silhouette is
kept. Verified end-to-end on the user's project `LO_06_red_whole.json`: bottom
sections 12-23 now register (residuals 0.28-0.39, on par with the top slide's
0.21-0.36); magenta sections unchanged. Regression test
`test_masks.py::test_registration_mask_keeps_cyan_tissue_not_treated_as_label`.

NOTE (latent, not this bug): `slide_loader` saves `image_path = sources[0]` for a
merged multi-slide image; the merged composite is only reconstructed at load time
by re-merging `source_paths` (`app._reload_project_display`). Cropping straight
from `image_path` would miss every section past the first source's height - fine
today because load always re-merges, but brittle if that ever changes.

## Fix: deleting a hovered section box crashed napari (v0.2.33)

Pressing **Delete** on a selected box in "Edit boxes" raised `IndexError: list
index out of range` (an `EmitLoopError` from psygnal). Root cause is a napari
bug, not ours: `Shapes.remove()` removes the shape from the data view **before**
clearing the selection, but leaves the hovered-shape index (`layer._value`)
pointing at the now-deleted row. The selection-cleared highlight recompute then
calls `ShapeList.outline(stale_index)` on the shorter list → `IndexError`
(`shapes.py::_outline_shapes`).

Fix (`slide_loader._bind_safe_delete`, bound in `_edit_boxes`): rebind **Delete**
and **Backspace** on the editable Shapes layer to a wrapper that resets
`_value`/`_value_stored` to `(None, None)` before `remove_selected()`, so the
post-remove highlight guard short-circuits. Regression test
`test_gui_smoke.py::test_edit_boxes_delete_hovered_does_not_crash` hovers a shape
then fires the bound Delete and asserts no crash.

Also **removed the "Click to discard a box…" button** (and its now-dead
`app.install_discard_handler` one-shot pick handler): with the editor's Delete
fixed it was redundant — "Edit boxes → select → Delete" covers box removal.

## DeepSlice pre-match in the Atlas matcher (v0.2.32)

DeepSlice was only ever run **inside** the Register step. New **"Pre-match all
(DeepSlice)"** button in the Atlas matcher (`atlas_matcher._prematch_deepslice`)
runs it on the **active slide's** sections in one pass and fills each section's AP
so the user fine-tunes from a good start instead of from zero:

- Crops are built with the same `io.image.crop` Register uses, ordered by
  `ap_order` (so DeepSlice's series order follows the user's intended sequence).
- Each predicted plane is reduced to its **centre AP** via
  `pipeline.anchoring_center_ap_um` and written to `section.plane.ap_um` (existing
  tilt on the plane is preserved; the matcher displays/edits coronal AP). The link
  **spacing** is seeded from the median AP step.
- **Only AP is baked into the saved plane** - the tilt sign conventions of the
  DeepSlice→our-frame anchoring aren't independently validated, so a possibly-wrong
  tilt is *not* persisted. Instead the **full predicted plane (incl. tilt) is cached**
  on `WorkflowState.deepslice_anchorings` (+ a per-crop `deepslice_fingerprints`).
- **Register reuses the cache** (`register_panel._reuse_prematch`) when *every*
  section to register has a cached plane whose `crop_fingerprint` still matches -
  so a swapped dye image (same index, new pixels) or any missing section forces a
  fresh DeepSlice pass. Safe failure mode: a mismatch only re-runs DeepSlice, never
  feeds a stale prediction. Cache cleared on `WorkflowState.reset()`.
- Tests: `tests/test_prematch_deepslice.py` (pure: AP round-trip vs
  `coronal_anchoring`, fingerprint staleness, reuse guard) +
  `test_gui_smoke.py::test_atlas_matcher_prematch_deepslice` (worker stubbed, no TF).

## 3D orientation: anterior-positive both views, dorsal-up, no mirror (v0.2.27)

## 3D orientation: anterior-positive both views, dorsal-up, no mirror (v0.2.27)

Removing the Plotly z-reversal (v0.2.24) fixed the L/R mirror but left the brain
**upside-down** (the camera up-vector approach didn't hold). Root cause again is
handedness: **reversing one axis mirrors L/R; negating one axis mirrors L/R.** The
reliable combination:

- **Plotly**: keep the `zaxis` reversal (reliable dorsal-up) and **negate ML**
  (`x = midline - ML`) to cancel the mirror it causes; AP stays `bregma - AP`
  (anterior +). Camera looks from the posterior (`eye.y < 0`), `up = +z`.
- **napari**: the display affine now negates **both AP and ML**
  (`bregma - AP`, `midline - ML`) - two negations = a proper rotation (**det = +1**,
  no mirror) - giving anterior-positive AP to match Plotly. DV stays ventral-positive;
  dorsal-up from the camera (`up = -DV`), `view_direction = +AP`.
- Net: **both views are anterior-positive AP, ML 0 at midline (Paxinos sign),
  dorsal-up, un-mirrored.** `test_bregma_display_affine_does_not_mirror` still guards
  det == +1. **Verify L/R + up/down on a real Plotly export.**

## Paxinos 5° pitch correction (v0.2.26)

User correction: **CCFv3 is pitched ~5° nose-down vs flat-skull stereotaxic**, so
the v0.2.25 pure-mirror Paxinos transform was only right near bregma. Rebuilt
`ccf_um_to_paxinos_mm` as a proper affine: bregma-relative (now incl. **DV offset
440 µm**, `BREGMA_DV_FROM_ORIGIN_UM`) → **5° un-pitch in the sagittal AP-DV plane**
→ per-axis scaling → Paxinos signs/mm. Selectable presets in
`PAXINOS_ALIGNMENTS` (`none` = legacy mirror, `allen_forum`, `qiu2018` = default,
`dorr2008`); a "Paxinos align" dropdown in the export panel feeds
`export_paxinos_csv(..., alignment=...)`. Scale/pitch/bregma-DV are **estimates** -
validate with histology. The 3D views stay bregma-CCF (not pitch-corrected). See
memory `reference_paxinos_transform`.

## Bugfix: probe markers wiped from a partial layer (v0.2.31)

User report: marked ProbeA+ProbeB, saved, reloaded → only ProbeA. The saved JSON
had **both** probes, but ProbeB's shanks had `tip_ccf_um`/`entry_ccf_um` intact and
`tip_px`/`entry_px` = `None`. Cause: `click_overlay._sync_layer` rewrites the schema
from the marker layer's points, first nulling **every** shank's px of that kind; if
the layer is ever incomplete (a layer-list reset / stale layer left it partial), the
shanks missing from it get their px nulled while their CCF (computed earlier)
survives. Fix: `_sync_layer` now bails out and **repopulates from the schema** if a
non-add event leaves the layer 2+ markers short of the schema, instead of wiping.
Regression test `test_sync_layer_does_not_wipe_markers_from_partial_layer`. Related:
the [[stale layer]] marker fix (v0.2.29). **Recovered the user's ProbeB pixels** by
inverting the per-section registration (anchoring least-squares + forward B-spline
`TransformPoint`); validated on ProbeA (known px+CCF, <6 px error = B-spline
inverse residual), CCF round-trip 0-5 µm, all in-bbox.

## Box-adjustment persistence + optional preserve-on-reregister (v0.2.30)

User report: a **box** atlas correction (done after "Reset morph to plane") "keeps
resetting back"; trigger unknown (NOT re-registration, NOT reload - both ruled out
by the user / by code: `manual_affine` round-trips through save/load and the
world↔section affine math is exact; `_render_overlay` reapplies it). Changes made:

- **Hardened `_rerender_section_overlay`** to reapply a box `manual_affine` (via
  `section_to_world`) instead of forcing `layer.affine = identity`, mirroring
  `_render_overlay`. (Its 3 current callers all clear the correction first, so this
  is latent-bug insurance, not the observed cause.)
- **Optional** "Keep hand-corrected sections on re-run" checkbox (params dialog,
  **default off**, per user) → `register_worker_progressive(preserve_manual=...)`
  skips re-registering sections with `manual_affine`/`manual_landmarks` and yields a
  "Keeping N hand-corrected section(s)" note.
- **Still unexplained**: the observed reset. Leading remaining hypothesis is an
  *uncommitted* box (the drag only persists when the user clicks **Apply
  adjustment** / toggles the Adjust button off → `_commit_adjustment`; a drag left
  un-applied is lost on the next redraw). TODO if confirmed: live-commit on drag or
  auto-commit on section-switch / overlay-refresh.

## Bugfix: tip/entry markers vanish after close→reload (v0.2.29)

`_on_project_cleared` calls `viewer.layers.clear()`, but `ClickOverlayWidget` kept
its `_tip_layer`/`_entry_layer`/`_traj_layer` attributes pointing at the removed
layers. `_ensure_points_layers` only (re)creates a layer when its ref is `None`, so
after a close→reload it wrote marker data to **detached** layers: markers didn't
draw and `Select / move` set `selection.active` to a layer "not in the list".
Fix: `_drop_stale_layer_refs()` nulls any ref no longer in `viewer.layers`; called
from `_ensure_points_layers`, `_ensure_traj_layer`, and `refresh_after_load`.
Regression test: `test_markers_redraw_after_layers_cleared`.

## TODO: region-atlas overlay (Kim/BBP) not yet correct

The region-atlas picker (v0.2.25) renders Kim/BBP region meshes onto Allen-registered
probe coordinates assuming the atlases are pixel-perfectly co-registered to Allen
25 µm. **The user confirmed Kim/BBP overlay is NOT entirely correct** (a real
offset/mismatch vs the Allen coordinates) - revisit: check whether these BrainGlobe
atlases truly share the Allen voxel grid (origin + resolution + orientation), and
add a per-atlas alignment/offset if needed before trusting cross-atlas overlays.

## Region-atlas picker + Paxinos export (v0.2.25)

- **Region/display atlas picker** (3D & Export panel): render region meshes /
  acronyms from a *different but coordinate-compatible* CCFv3 25 µm atlas (Allen,
  CCFv3-BBP Augmented, Chon/Kim Unified) without re-registering. These share the
  Allen 25 µm voxel space, so **probe coordinates are identical** - only the region
  annotation differs. `viz_export_panel._ensure_display_atlas` resolves the choice
  into `self._display_atlas` (lazy-loads + caches a non-registration atlas; the
  registration atlas is still loaded for the probe remap), and the Plotly / napari
  builders are passed `_display_atlas` for regions. Probe CCF coords come from the
  registration atlas as before.
- **Paxinos export** (`Export per-channel Paxinos CSV`): linear CCF→Paxinos
  (Franklin-Paxinos) conversion in `io.ccf_coords.ccf_um_to_paxinos_mm` -
  `AP = (5400 - AP_ccf)/1000` (bregma 0, anterior +), `ML = (5700 - ML_ccf)/1000`
  (midline 0), `DV = DV_ccf/1000` (CCF depth). **No rotation** - a simple mirror
  affine. Uses the corrected bregma AP (5400 = `BREGMA_AP_FROM_ORIGIN_UM`); the
  legacy 6600 placed regions ~1.2 mm too anterior. `probes.channels.export_paxinos_csv`
  writes `probe,shank,channel,ap_mm,ml_mm,dv_mm`. Refs (forum + paper) in memory
  `reference_paxinos_transform`. **DV is not re-referenced to the bregma surface** -
  validate against real stereotaxic readings if absolute DV matters.

## Bregma-referenced 3D + L/R mirror fix (v0.2.24)

**Plotly L/R was genuinely mirrored** (not a camera angle): **reversing a single
axis flips the scene's handedness** and mirrors left/right. Plotly had
`zaxis: autorange='reversed'` (for dorsal-up), which - combined with the x=ML/y=AP
swap and the AP negation - left an **odd** number of reflections = a mirror. Fix:
**drop the z-reversal** and get dorsal-up from the **camera up vector** (`-DV`)
instead (the way napari already did it); the camera also looks from the posterior
(`eye.y < 0`) to match napari.

- **The same rule bit the napari change:** `_bregma_affine` must be a **pure
  translation** (det +1) - **never negate an axis** (det -1 = reflection = mirror).
  So the napari bregma display is offset-only: `AP - bregma` (stays
  posterior-positive), `ML - midline`, DV unchanged; original posterior-dorsal
  camera kept. `test_bregma_display_affine_does_not_mirror` guards det == +1.
- (Superseded by v0.2.27: both views are now anterior-positive AP - the napari
  affine negates AP **and** ML together to stay un-mirrored.)

## Atlas conversion + Paxinos export - PLANNED, not built (user request)

- **View/export in a different *compatible* atlas** (e.g. CCFv3-BBP Augmented, Kim
  unified): these share the CCFv3 25 µm voxel space, so probe **coordinates are
  identical** - only the region **annotation/meshes/acronyms** differ. Plan: a
  "Region atlas" picker in the 3D & Export panel that lazily loads the chosen atlas
  and uses it for region rendering/lookup while probe CCF stays as-is.
- **Paxinos coordinate export**: a real **affine** CCF->Paxinos (see
  https://community.brain-map.org/t/how-to-switch-between-the-3-coordinate-systems-of-mouse-connectivity/952
  and https://pmc.ncbi.nlm.nih.gov/articles/PMC10033636/). Needs the published
  transform encoded + validated before shipping (wrong coords are worse than none).

## Ephys alignment clarity + NP2.0 geometry (v0.2.23)

Driven by the user's Neuropixels feedback (captured in memory:
`reference_neuropixels_geometry`). In `ephys_align_dialog` / `ephys/loader` /
`lfp_power_worker`:

- **Split shanks by `shank_ids`, not x.** `loader` now reads
  `recording.get_probe().shank_ids` (per channel) into `LfpData.channel_shank_ids`;
  the dialog's `_shank_mask` selects by shank id, falling back to gap-clustering x
  into the probe's shank count (`_cluster_x_into_shanks`). Fixes "shank 3 shows 48
  channels not 96" - NP2.0 has **2 columns per shank**, so unique-x grabbed one
  column. The info line now states **"N recorded channels in M rows (~k sites/row)"**
  (derived, never hard-coded).
- **Absolute depth-from-tip.** `lfp_power_worker` no longer zeroes depths at the
  lowest channel; the recorded block keeps its real offset above the physical tip,
  and the dialog uses absolute `_depths`. Axis labels are now track endpoints
  ("surface"/"tip"), not channel ids.
- **Region acronyms** (+ short names) are drawn beside the colour strip
  (`_draw_region_labels`, `_region_caption`).
- **Recorded-block brackets**: green dashed lines mark where the recorded
  electrodes land on the track (`_draw_recorded_extent`, redrawn on anchor move),
  and two **anchors are pre-set** at the block edges for fresh shanks.
- **Save LFP power** (`_save_lfp_power`): export per-channel PSD + depths + freqs as
  `.npz`/`.csv`.
- **Not done (needs the full probe layout, not just recorded contacts):** lines for
  the *full physical* electrode-site extent (all 1,280 sites incl. the 175 µm tip
  taper). The histology track is always >= the electrode span.

## Probe ML double-count fix + bregma-referenced Plotly axes (v0.2.22)

**Bug:** the two 3D views (`napari3d.add_probe_layers`, `plotly3d.add_probe_traces`)
added a per-shank geometric `ml_offset` (from `shank_offsets(n, pitch)`) **on top
of** each shank's already-placed+registered `tip_ccf_um`/`entry_ccf_um`. Since the
user places/registers each shank's tip and entry individually, the ML is already
correct - the offset double-counted the shank separation and could push outer
shanks across the midline. The per-channel export (`channels.shank_channel_coords`)
never did this - it uses the placed coords directly. **Fix:** both 3D views now use
`tip_ccf_um`/`entry_ccf_um` directly (no offset), matching the export.

**Bregma-referenced Plotly axes:** `build_figure(bregma_relative=True)` (default)
post-processes **all** traces (regions + probes, so they stay aligned) via
`_rereference_traces_to_bregma`: `x = ML - MIDLINE_ML_UM` (0 at midline),
`y = BREGMA_AP_FROM_ORIGIN_UM - AP` (0 at bregma, anterior +, matching the Atlas/
ordering/residuals tabs); DV unchanged. Axis titles become "ML from midline (µm)" /
"AP from bregma (µm, ant +)". napari 3D still uses CCF µm.

## Default 3D camera (v0.2.21)

`show_3d_scene` left the camera at napari's default (a side/back view rotated 90°).
It now ends with `_set_default_camera`: `reset_view()` (fit) then
`camera.set_view_direction`. Data axes are **(AP, ML, DV)** (AP↑ posterior, DV↑
ventral), so the standard view is **from behind (posterior), dorsal up, tilted ~30°
down from the top**: `view_direction=(-1, 0, 0.5)` (toward anterior, slightly down),
`up_direction=(0, 0, -1)` (dorsal up). Both hemispheres come out symmetric with the
brainstem/probes facing the viewer. Tweak the two `_VIEW_DIRECTION` / `_UP_DIRECTION`
constants in `viz/napari3d.py` to change it.

## "Update coordinates" - when probe CCF is recomputed (v0.2.20)

Probe **CCF** coords (`shank.tip_ccf_um` / `entry_ccf_um`, used by the 3D view +
exports) are derived from the **pixel** clicks (`tip_px` / `entry_px`) through the
registration. They are recomputed when: registration runs, a **manual atlas
correction** is applied (`register_panel._remap_and_save`), or the new **"Update
coordinates"** button is clicked. **Moving a tip/entry point** in the Probes tab
only updates the *pixel* position (`click_overlay._sync_layer`) - it does **not**
re-map to CCF, and an already-open 3D window doesn't refresh. So a moved point (or
a correction made while the 3D window is open) wasn't reflected.

Fix (`viz_export_panel`): **"Update coordinates"** (top of the 3D & Export panel)
calls `_remap_probes` (`reload_registered_transforms` + `_apply_to_shank_registered`
over every shank, reading the live `tip_px`/`entry_px` and registration incl.
manual corrections), saves, and refreshes an open 3D window. **"View in napari
3D"** now also re-maps first (`_remap_then_render`), so the 3D view is always
current. Re-mapping is only done on these explicit actions (not per-drag), per the
user's "too expensive to do automatically" concern. Needs the atlas loaded.
**Note:** the pkl/CSV/Plotly exports still read the stored CCF, so click "Update
coordinates" after moving points before exporting.

## Reset morph to plane, for hard sections (v0.2.19)

For sections the deformable fit mangles (tissue torn apart, or a piece missing -
e.g. brainstem present but cerebellum gone), fixing the distorted outline by hand
is slower than starting over. **Register tab → Manual atlas adjustment → Landmarks
→ "Reset morph to plane (keep AP/ML)"** (`_reset_morph`) sets
`section.registration.bspline_transform_path = None` and clears any
`manual_affine`/`manual_landmarks`. With no B-spline path, `warp_annotation_to_section`
falls back to the **undistorted atlas slice resized to the bbox**, and the probe
mapping uses only the anchoring (`bspline=None`) - so the **AP/ML plane is kept**,
just the morph is dropped. The user then uses **Place landmarks** to fit it by hand
from the clean plane. Re-maps probes + auto-saves like the other corrections.

## Register residuals table consistency (v0.2.18)

The residuals table disagreed with the Atlas/ordering tab and confused users into
thinking the registration was wrong. Three fixes in `_refresh_residuals`:
- **AP from bregma** (`BREGMA_AP_FROM_ORIGIN_UM - ap_idx·res`), matching the Atlas
  tab, instead of raw CCF-origin µm (e.g. shows `-5300`, not `10700`).
- **Sorted by `ap_order`** (the AP sequence), so rows read 0,1,2,3 like the
  ordering list, not the project's storage/detection order (0,3,6,9…).
- Uses the **actual registered AP** = centre of `section.registration.anchoring`
  (incl. DeepSlice guidance), not `section.plane.ap_um` (the request) which can
  differ. Header now reads "AP from bregma µm".

## DeepSlice ordering + manual-AP guidance + Ctrl+S (v0.2.17)

- **Ctrl+S / Ctrl+Shift+S / Ctrl+O** save / save-as / load (application-wide
  QAction shortcuts on the Project menu).
- **DeepSlice is ordered by the user's AP sequence, not the detection index.**
  DeepSlice enforces a monotonic A→P order by its filename `_s<token>`. The token
  was `section.index` (detection order); reordering in the ordering panel only
  changes `ap_order`, so a reorder was **ignored by DeepSlice**. Now
  `register_panel._section_order` numbers the DeepSlice input by `ap_order` rank
  (`predict_anchorings(order=...)`), and the results map back to `section.index`.
  No reorder → rank == index → unchanged.
- **A manually assigned AP guides DeepSlice** (`pipeline.guide_anchorings_with_planes`,
  v0.2.18): after DeepSlice predicts, each **assigned** section's plane is
  translated along AP so its centre sits **exactly** at the user's AP (keeping
  DeepSlice's tilt); **unassigned** sections are shifted by interpolating the
  assigned sections' shifts (vs DeepSlice's predicted AP). This *guarantees* a
  pinned section lands on its value (an earlier least-squares-line version only got
  close - the cause of "section 0 didn't register to -5300 even though I assigned
  it"). Applied in `_start_register`; `anchoring_for_section` then uses the guided
  DeepSlice anchoring, falling back to the manual plane only where DeepSlice didn't
  cover. **Watch:** the residuals table shows the *actual* registered AP (the
  anchoring), so a mismatch there vs the assigned AP means guidance didn't apply.
- **Known gap (not yet fixed):** the Atlas-tab "Assign to section idx", the
  overlay, and `section_images` still key by `section.index`, which can differ
  from the intended A→P order after a reorder or an alphabetical multi-slide merge
  - so "section 0" isn't guaranteed to be the anterior-most section.

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

> The `newUI` line of work is **merged into `main`**. Recent themes: regularized
> elastix engine + automatic outer-contour snap, manual atlas correction (box +
> landmarks), Register-panel slimmed (parameters moved to the Registration menu),
> menu bar trimmed to Project + Registration.

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
**3D & Export** panel (`VizExportPanelWidget`) docks on the **right**. The menu bar
shows only two menus (v0.2.14): **Project** (Save / Save As… / Load) and
**Registration** (Parameters); napari's default File/View/Plugins/Window/Help menus
are **hidden** (`_keep_only_menus`, best-effort - they're set invisible, not removed).
Tab order left→right: **Histology → Atlas → Probes → Register → Ephys**.
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
   **Markers are colour-coded per shank** (v0.2.16): a shank's tip and entry share
   one colour, cycling as you pick another shank/probe, and tip vs entry differ by
   **symbol** (tip = disc, entry = triangle). **"Select / move"** enters napari
   select mode to drag markers or select them; **"Clear selected"** removes just
   the chosen ones (vs "Clear all"). A second tip/entry for a shank replaces its
   previous one. (See "Tip/entry markers" below.)
4. **Register** - the panel is deliberately lean (v0.2.14): just **"Register all
   sections"**, the progress/status, the residuals table, "Show atlas overlay", and
   the manual adjustment group. **All registration parameters** are **no longer shown
   inline** - the defaults are good, so they live in a dialog opened from
   **Registration menu → Parameters** (`register_panel.open_parameters_dialog`
   reparents the same widgets into the dialog, so what you set is what the run reads).
   In that dialog, **"Predict planes with DeepSlice"** sits at the **top** (it's the
   first thing registration does), followed by B-spline grid, max iterations,
   "Regularized registration (elastix)", "Smoothness (bending energy)", "Restrict to
   tissue mask", "Silhouette pre-align", "Snap atlas contour to tissue". All toggles
   default **on**. **Engine: elastix (regularized) by default** -
   a bending-energy penalty + tissue mask (ABBA-style) keep atlas boundaries on the
   tissue; falls back to plain SimpleITK B-spline when `itk-elastix` is absent. "Show
   atlas overlay" warps registered region boundaries onto each section (lazily loads
   the atlas). Auto-saves the project. (See "Registration engine" + "Automatic
   outer-contour snap" below.)
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
slide images merges them into one combined image** (`io.image.merge_images` stacks
sources **top-to-bottom** with a 40 px gap, sorted alphabetically so reload
reproduces the exact pixels). The user is told this on load. After the merge the
rest of the pipeline is unchanged - there is effectively one slide. The napari
built-in **layer list / layer controls panels are hidden** (the workflow panel
drives everything), so slides are never managed as separate layers.

**Swap-image-on-reopen (v0.2.28).** Multi-select still merges, but **Open slide…**
when a slide is *already loaded* now SWAPS rather than appends
(`slide_loader._replace_images`, routed from `_open_file`). Same-size new image =>
keep section boxes + registration (reuse a registration on the same section in a
different channel; flips re-applied, per-channel levels cleared, outlines redrawn).
Different-size => treated as a new slide (sections/flips cleared, re-detect). The
old incremental "open more later → merge" path is gone (use multi-select instead).

**Slide-aware section ordering (v0.2.15).** Because sources are *stacked
vertically*, a naive column-first ordering walks each column top-to-bottom across
the whole canvas - so column 0 runs through both stacked slides and interleaves
their sections. Fix: `io.image.slide_bands(heights)` returns each source's
`(y_start, y_end)` band (same `gap_px` as the merge), tracked in
`WorkflowState.slide_bands` (set on first load = one band, on merge, and on
reload). `order_sections(..., band_bounds=...)` then partitions sections by band
(centroid-y), orders each source independently with the existing column/row logic,
and concatenates `ap_order` top band first - so columns stay within a slide.
`detect_sections_worker` threads it; a single-source slide passes one band and
behaves exactly as before.

**Close/clear project (v0.2.15).** **Project menu → Close Project** (confirm
dialog) calls `WorkflowState.reset()` (wipes project, images, bands, active
selection, saved path; **keeps the loaded atlas object** in memory), clears all
napari layers, and re-runs every panel's `refresh_after_load()` - returning the
app to its just-launched state without a restart. Wired via an `on_cleared`
callback to `_install_project_menu`; `_refresh_panels()` now also covers the slide
loader + image tools (the slide loader gained a `refresh_after_load` that resets
its labels).

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

UI (v0.2.14): the "Manual atlas adjustment" group has the **Section** picker
(ordered by section index, not slide detection order), then two **outlined
sub-groups** - "Box transform" (the Adjust-atlas button) and "Landmarks" (Place /
Move / Add / Apply) - with **"Reset adjustment" set apart below a divider** since
it clears either tool.

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

## Tip/entry markers + canvas navigation (v0.2.16)

**Markers (`gui/widgets/click_overlay.py`).** Two Points layers, **Tips** (symbol
`disc`) and **Entries** (`triangle_up`); colour encodes the **shank**, not the
type, so a shank's tip and entry match and different shanks cycle a 16-colour
palette by their global ordinal (position in the flattened probe→shank list).
Each point carries its `(p, s)` (probe/shank positions) in the layer **features**,
so identity survives add/move/delete. The layers are **two-way synced** with the
schema in `_sync_layer`: on *add* the new (last) point is assigned to the selected
shank (napari may copy the previous point's features, so we overwrite the last
row); points are then deduped to **one tip/entry per shank** (newest wins); the
schema's `shank.tip_px/entry_px` is rewritten from the kept points; colours are
reapplied (`_recolor`). "Select / move" puts both layers in napari `select` mode
(drag to reposition, Delete / "Clear selected" → `remove_selected()` fires the
data event that clears those shanks); "Clear all" wipes everything.
**Gotcha:** do **not** set `current_face_color` to drive a preview colour - it
recurses through napari's colour-swatch control and blows the stack; the per-point
`face_color` array set in `_recolor` is what colours the markers. Also: the
shank-combo `currentIndexChanged` must **not** repopulate the shank combo (only
the *probe* combo does) or it recurses.

**Canvas navigation (`app._install_wheel_pan`).** Plain wheel still zooms;
**Ctrl+wheel pans horizontally, Shift+wheel pans vertically**. napari 0.7 already
*suppresses* wheel-zoom whenever a modifier is held (its `NapariSceneCanvas`
ignores modified wheel events) while still firing `viewer.mouse_wheel_callbacks`,
so the pan callback never fights the zoom. Pan step = `delta * 80 / camera.zoom`
world units (consistent feel at any magnification); honours natural-scroll
inversion. **Real canvas scrollbars were considered and dropped** - napari's
QtViewer exposes no insertable layout and overlaying widgets on the GL canvas is
fragile across versions; the wheel-pan covers the need.

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
3. **Push** `main` to origin when ready (newUI has been merged into main).
4. Possible follow-ups discussed but not built: auto-clean DeepSlice AP outliers
   (neighbor smoothing), a globally-coupled (vs per-section) registration, and a
   true Mesa software-GL option (the bundled `opengl32sw.dll` exists but vispy/
   PyOpenGL need `VISPY_GL_LIB` wiring - not done, and deprioritized in favor of
   fixing the driver).
```
