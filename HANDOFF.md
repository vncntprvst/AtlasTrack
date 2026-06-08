# Histo_to_CCF — Handoff

_Last updated: 2026-06-08 · version **0.2.4** · branch **newUI** (committed, not pushed)_

## TL;DR

A napari desktop app that registers mouse histology sections to the Allen CCF and
maps probe (Neuropixels / NeuroNexus) trajectories to CCF coordinates. The core
pipeline (sectioning → annotate → DeepSlice plane prediction → 2D B-spline →
per-channel CCF coords → 3D viz / exports) is implemented and **131 tests pass**.

**The GPU/OpenGL launch blocker is RESOLVED** — a reboot cleared the
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
# from repo root, with .venv active (editable install — no reinstall after edits)
histo2ccf gui            # launch the GUI
histo2ccf gl-info        # diagnose GPU/OpenGL if the GUI won't start
histo2ccf version
.venv\Scripts\python.exe -m pytest -q   # 131 tests
```

DeepSlice is installed (optional extra, pulls TensorFlow). It runs CPU-only on
Windows.

## Workflow (what the GUI does)

5 tabs, left→right (Save/Load moved to the **Histo→CCF menu**, not a tab):
1. **Load** — open one or more composite slides; multiple opens are **merged into
   a single combined image** (see "Multiple slides" below). Auto-detect sections
   (Otsu + connected components). Auto-estimates min-area. "Equalize under-sized
   boxes" grows boxes that come out smaller than the slide's median. **Edit boxes**
   turns detections into draggable napari rectangles (resize/move/Delete/draw-to-
   add), synced live. Image tools: flip H/V and per-channel levels, scoped to the
   whole (merged) slide or a **Selected section** chosen from a dropdown.
2. **Probes** — pick a probe (Neuropixels 1.0, NP 2.0 4-shank, NeuroNexus
   A1x32-Poly3-10mm-25s-177-OA32LP). "Add probe" auto-arms Tip-marker mode. Click
   to drop tip; switch to Entry (Marker, or draw a Trajectory line whose
   tissue-surface crossing = entry).
3. **Atlas** — choose atlas + storage folder; **AP shown relative to bregma**
   (0 = bregma). **Open atlas matcher…** (side-by-side / overlay AP matching;
   syncs AP + spacing with this tab on open/close). Assign AP per section, or
   reorder/space them in the ordering panel (column-first default; drag to reorder;
   "Interpolate AP" fills gaps between hand-assigned sections).
4. **Register** — optionally **Predict planes with DeepSlice** (cross-section-
   consistent), then per-section 2D B-spline. Residuals table. "Show atlas overlay"
   warps registered region boundaries onto each section (lazily loads the atlas if
   needed). Auto-saves the project.
5. **Ephys** — refine a registered shank's depth→CCF mapping from LFP features
   (see "Ephys alignment" below). Point at an Open Ephys recording folder, compute
   the LFP power map, then drag anchors in the alignment dialog to align power
   transitions with atlas region boundaries; Apply stores per-channel CCF.

Save/Load: **Histo→CCF menu** → Save / Save As… / Load Project (`.histo2ccf.json`).
Load restores the merged slide + sections + registration (CCF coords come back;
exports work without re-running), re-applies stored flips, and **auto-loads the
project's atlas in the background** (overlay / 3D ready without a manual click).
**Load also repopulates every tab's fields** from the project via per-widget
`refresh_after_load()` (probes + tip/entry markers/table, atlas selection + AP,
section ordering list + spacing, residuals, ephys combos) — wired through
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
of the pipeline is unchanged — there is effectively one slide. The napari built-in
**layer list / layer controls panels are hidden** (the workflow panel drives
everything), so slides are never managed as separate layers.

## Ephys alignment (v0.2)

IBL-style depth refinement of probe shanks from LFP. Pure-core package
`ephys/` (no Qt): `alignment.py` warps channel *feature depth* (µm from tip) to
*track depth* via anchor points (piecewise-linear + linear extrapolation) and
places each channel on the shank's `tip_ccf_um`→`entry_ccf_um` line;
`features.py` computes the depth×frequency LFP power map (Welch PSD); `regions.py`
samples the atlas region at each depth; `loader.py` reads Open Ephys LFP via
**SpikeInterface** (optional `ephys` extra — gated import, `si.read_openephys` /
`get_neo_streams`). NP 1.0 has a dedicated LFP stream; when none exists LFP is
derived from the AP stream (band-limit + resample). All the math is unit-tested
without SpikeInterface installed.

GUI: the **Ephys** tab (`gui/widgets/ephys_panel.py`) picks a probe+shank, loads
the recording (background `lfp_power_worker`), and opens
`EphysAlignmentDialog` — LFP power map (warped into track space) beside the atlas
region colour strip, shared depth axis (tip at the bottom), draggable red anchor
lines. Apply writes `Shank.ephys` (`EphysAlignment`: anchors + per-channel
`channel_ccf_um`), which round-trips through the project JSON.

The alignment dialog: **double-click the LFP map** to drop an anchor at that depth
(or "Add anchor (mid)"), then drag it; left margin shows top/bottom channel # +
depth µm (tip at bottom), and the header shows tip/entry CCF. **Apply auto-saves**
the project (if it has a path) and the ephys per-channel CCF now render as Points
layers in the napari 3D view (`add_ephys_channel_layers`). Compute does **not**
cache — it re-reads/filters the recording each time.

Open follow-ups: per-channel region CSV export; richer anchor UX (snap to region
boundary); per-frequency normalisation option for the power map; shank-column
auto-detection for 4-shank probes is heuristic (sorted unique x → shank index).

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
  fix for the original "crash after registration" — TensorFlow's ~2 GB is released
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
- **DeepSlice `save_predictions(name)` appends `.json`** — pass a base path, read
  back `name.json`.
- Tip/entry are clicked in **slide-global** pixels but the section transform is on
  the **section crop** — `_apply_to_shank_registered` subtracts the bbox origin
  (this was the "probe 60 mm outside the brain" bug).
- 3D region colors use a **curated palette** (`viz/plotly3d.REGION_STYLE`), not the
  muddy native atlas colors; siblings get distinct fallback colors. Context shell
  (Isocortex/CB/BS) is faint + additive blending so probes show through; only
  shank-tip regions are shown by default; extra regions via a text field.
- **Windows console is cp1252** — keep CLI/diagnostic output ASCII (no `→`, `—`).

## The GPU/OpenGL launch blocker (RESOLVED)

Symptom (now gone): `histo2ccf gui` printed repeated `QOpenGLFramebufferObject:
Unsupported framebuffer format` and rendered nothing. It was a GPU driver /
Remote-Desktop / dual-GPU session issue, not a histo2ccf bug — and **a reboot
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

- `pytest -q` → **156 passed** (131 non-qt + 25 qt; run qt tests per-process — the
  napari GL context corrupts across many viewers in one process on this machine,
  so a single `pytest -q` run can hit a Windows access violation mid-suite even
  though every test passes alone). Includes: core pipeline, sectioning/ordering,
  ephys alignment math + LFP features + region strip + schema round-trip,
  probe geometry/channels, transforms, DeepSlice anchoring conversion (permute +
  flip + scale, mocked subprocess), region styling, mesh extraction, and
  `@pytest.mark.qt` GUI smoke tests (full-panel build, edit-boxes, atlas bregma,
  ordering, click-overlay, atlas matcher, project-menu present, flip-restore on
  load, section-scope flip, GL diagnostic never-raises).
- Can't be tested in CI here: live GL rendering and a real DeepSlice model run —
  both **verified manually in the running GUI** (GUI launches, registration
  completes end-to-end without crashing).

## Open items / next steps

1. **Ephys per-channel CCF in 3D / export** — surface the ephys-refined channels
   (regions + CCF) in the napari 3D view and a per-channel region CSV (see "Ephys
   alignment" follow-ups). Done this session: multiple-slide merge, Ephys tab.
2. Eyeball DeepSlice planes for a **left/right mirror** (flip `_FLIP_ML` if so).
3. **Push** `newUI` to origin when ready (committed locally, not pushed).
4. Possible follow-ups discussed but not built: auto-clean DeepSlice AP outliers
   (neighbor smoothing), a globally-coupled (vs per-section) registration, and a
   true Mesa software-GL option (the bundled `opengl32sw.dll` exists but vispy/
   PyOpenGL need `VISPY_GL_LIB` wiring — not done, and deprioritized in favor of
   fixing the driver).
```
