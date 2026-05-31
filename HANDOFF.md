# Histo_to_CCF — Handoff

_Last updated: 2026-05-31 · version **0.1.22** · branch **newUI** (changes uncommitted)_

## TL;DR

A napari desktop app that registers mouse histology sections to the Allen CCF and
maps probe (Neuropixels / NeuroNexus) trajectories to CCF coordinates. The core
pipeline (sectioning → annotate → DeepSlice plane prediction → 2D B-spline →
per-channel CCF coords → 3D viz / exports) is implemented and **128 tests pass**.

**Previous GPU/OpenGL launch blocker is RESOLVED** — a reboot cleared the
`QOpenGLFramebufferObject: Unsupported framebuffer format` failure. The GUI now
launches and **end-to-end registration completes without crashing** (TensorFlow
in the DeepSlice subprocess keeps the memory peak down). If the GL error ever
recurs, `histo2ccf gl-info` still diagnoses it and a reboot/driver fix is the cure.

> All work this session is **uncommitted** working-tree changes on `newUI`
> (git HEAD is still `3a42ddd "Packaging app before tests"`). Nothing has been
> committed or pushed.

## How to run

```powershell
# from repo root, with .venv active (editable install — no reinstall after edits)
histo2ccf gui            # launch the GUI
histo2ccf gl-info        # diagnose GPU/OpenGL if the GUI won't start
histo2ccf version
.venv\Scripts\python.exe -m pytest -q   # 128 tests
```

DeepSlice is installed (optional extra, pulls TensorFlow). It runs CPU-only on
Windows.

## Workflow (what the GUI does)

5 tabs, left→right:
1. **Load** — open a composite slide; auto-detect sections (Otsu + connected
   components). Auto-estimates min-area. "Equalize under-sized boxes" grows boxes
   that come out smaller than the slide's median. **Edit boxes** turns detections
   into draggable napari rectangles (resize handles / move / Delete / draw-to-add),
   synced live to the project.
2. **Annotate** — pick a probe (Neuropixels 1.0, NP 2.0 4-shank, NeuroNexus
   A1x32-Poly3-10mm-25s-177-OA32LP). "Add probe" auto-arms Tip-marker mode. Click
   to drop tip; switch to Entry (Marker, or draw a Trajectory line whose
   tissue-surface crossing = entry).
3. **Atlas** — choose atlas + storage folder; **AP shown relative to bregma**
   (0 = bregma). Assign AP per section, or reorder/space them in the ordering panel
   (column-first default; drag to reorder; "Interpolate AP" fills gaps between
   hand-assigned sections).
4. **Register** — optionally **Predict planes with DeepSlice** (cross-section-
   consistent), then per-section 2D B-spline. Residuals table. "Show atlas overlay"
   warps registered region boundaries onto each section. Auto-saves the project.
5. **Save** — Save / **Load project** (`.histo2ccf.json`). Load restores slides +
   sections + registration (CCF coords come back; exports work without re-running).

3D / export: **Export Plotly HTML** and **View in napari 3D** (opens a *separate*
window so the 2D workspace is untouched). HERBS pkl + per-channel CSV export.

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

- `pytest -q` → **128 passed**. Includes: core pipeline, sectioning/ordering,
  probe geometry/channels, transforms, DeepSlice anchoring conversion (permute +
  flip + scale, mocked subprocess), region styling, mesh extraction, and
  `@pytest.mark.qt` GUI smoke tests (full-panel build, edit-boxes, atlas bregma,
  ordering, click-overlay, GL diagnostic never-raises).
- Can't be tested in CI here: live GL rendering and a real DeepSlice model run —
  both **verified manually in the running GUI this session** (GUI launches,
  registration completes end-to-end without crashing).

## Open items / next steps

1. ~~Resolve the GPU driver / confirm registration doesn't crash~~ — **done**
   (reboot fixed GL; registration verified end-to-end in the running GUI).
2. Eyeball DeepSlice planes for a **left/right mirror** (flip `_FLIP_ML` if so).
3. Decide whether to **commit** this session's work (currently all uncommitted).
4. Possible follow-ups discussed but not built: auto-clean DeepSlice AP outliers
   (neighbor smoothing), a globally-coupled (vs per-section) registration, and a
   true Mesa software-GL option (the bundled `opengl32sw.dll` exists but vispy/
   PyOpenGL need `VISPY_GL_LIB` wiring — not done, and deprioritized in favor of
   fixing the driver).
```
