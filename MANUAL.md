# AtlasTrack - User Manual

This is the reference manual. If you're a new user, start with **TUTORIAL.md** (or the **Tutorial** tab in the app), which
walks through one registration from start to finish.

---

## 1. What the app does

You give it one or multiple image(s) of brain sections. It:

- finds each section and lets you tidy the boxes,
- places each section at the right front-to-back level in the atlas,
- warps the atlas onto each section so region outlines follow your tissue,
- turns probe tracks you click into atlas coordinates,
- and exports the result - coordinates, figures, or your section series with
  region outlines.

The defaults settings should be good enough to do most of registration. 
Most of the work is looking at each section, checking the atlas matching, and correcting warping. 

---

## 2. Ideas and coordinates

**The atlas.** A 3-D reference brain. Positions are in micrometres (µm) along
three axes: **AP** front-to-back, **ML** left-right, **DV** top-to-bottom.

**AP from bregma.** The app shows front-to-back position relative to **bregma**,
the skull landmark: `0` = bregma, **negative = behind it**, positive = in front.

**Atlases you can use.** All are downloaded once and cached under `~/.brainglobe`.
**Help ▸ Atlases** describes each one, with links and the bregma it uses:

| Atlas | Why you would pick it |
|---|---|
| Allen CCFv3 | The default. |
| CCFv3-BBP Augmented | Adds cerebellar layers, olfactory bulb layers, barrel columns. |
| Chon / Kim Unified | Franklin-Paxinos region names (M1, S1BF, 4V) instead of Allen's. |
| Chon / Kim v2, isotropic | The 2024 re-release of the above, 20 µm. |
| Custom ID | Any other BrainGlobe atlas, typed in by name. |

**The pipeline in one line:** section image → atlas slice at the right level →
that slice warped onto your section → outlines and probe coordinates.

**Your project** is one `*.atlastrack.json` file. Outputs go **next to your data**,
not into the app's folder.

---

## 3. Install and launch

```bash
uv pip install "atlastrack[all]"   # everything - recommended
uv pip install atlastrack          # base only: the histology → atlas workflow

atlastrack gui       # launch
atlastrack version   # check the install
atlastrack gl-info   # if the window will not open, run this and send the output
```

`pip install` works just as well as `uv pip install`. Quote the target and leave
no spaces between extras - PowerShell and `zsh` both treat `[...]` as a pattern.

The base install is deliberately light. `[all]` adds three optional pieces, which
you can also install one at a time (`".[elastix]"` and so on):

| Extra | What it adds | Notes |
|---|---|---|
| `elastix` | The **regularized** registration engine | Recommended - it is the setting the Register step relies on. ~150 MB. Without it the app falls back to a plainer fit and that option is greyed out. |
| `deepslice` | Automatic front-to-back placement | ~1.65 GB (it pulls TensorFlow), so it is the one to skip if you are placing sections by hand. |
| `ephys` | The Ephys tab | Only needed if you are refining depth from recordings. |

---

## 4. The window

**Centre** - two tabs: **Project** (your slide and the atlas overlay) and
**Help** (this manual, the tutorial, and the atlas reference). Loading a project
switches you back to Project automatically. **Open in a window** moves a help page
onto a second screen; closing that window puts it back.

**Left** - the workflow, in the order you use it:

| Tab | What you do there |
|---|---|
| Histology | Load the slide, find sections, straighten and adjust them |
| Atlas | Choose an atlas, set each section's front-to-back level |
| Register | Run the fit, check it, hand-correct anything that missed |
| Probes | Add a probe, click its tip and entry |
| Ephys | Optional: refine depth from recorded activity |

**Right - 3D & Export**, always available: **Probe** (re-map coordinates),
**3D Visualization** (region atlas, 3-D view), **Export**.

**Menus**

- **Project** - Save (Ctrl+S), Save As (Ctrl+Shift+S), Load (Ctrl+O), Load
  recent, Close.
- **Settings** - Registration (the fitting options).
- **Help** - Manual, Tutorial, Atlases.

**Moving around the image:** wheel to zoom, **Ctrl**+wheel to pan sideways,
**Shift**+wheel to pan up and down.

---

## 5. Recipes

### 5.1 Load a slide

**Histology ▸ Open histology image(s)** - pick one image, or several to stack
them into one canvas so every section shares a coordinate space.

Opening an image when one is already loaded **replaces** it. Same size keeps your
sections and registration - handy for the same slide in a different dye. A
different size starts fresh.

### 5.2 Find the sections

1. **Detect sections**. Adjust **Min area** upward to ignore debris, or
   **Closing radius** upward to join a section that came out in pieces.
2. **Click any box to select that section** - you do not need edit mode for this.
   The Adjustments **Section** dropdown follows your click.
3. **Edit boxes** when a box needs changing: drag a handle to resize, drag inside
   to move, **Delete** to remove.
4. **Draw new bounding box** for a section that was missed. It becomes a section
   as soon as you finish the rectangle.

### 5.3 Straighten and adjust

In **Adjustments**, first choose **Scope**: the whole slide, or one selected
section.

- **Rotation ▸ Angle** straightens a section that was mounted crooked. **From
  DeepSlice** fills in the angle it measured. Rotating a section that is already
  registered invalidates that fit - the panel says so, and you re-register it.
  For a tidy exported series you usually need none of this: the section-series
  export straightens on its own (Recipe 5.9).
- **Flip H / Flip V** if the tissue is mirrored.
- **Levels** to brighten faint channels, or **Auto**.

### 5.4 Set the front-to-back level

**Atlas** tab. Choose an atlas and **Load atlas** (first download is slow, later
loads are instant). The **?** beside the picker explains each one.

Then either:

**Quick manual assignment** - type an **AP from bregma**, pick a section number,
**Assign AP to section**.

**Matching viewer ▸ Open atlas matcher** - your section beside the atlas, or
blended over it. Step through sections and turn the AP dial until they match.
Pin one section, set the **spacing** between sections, and **Assign all** fills
in the rest.

**Section order and spacing** lists the sections front to back. Drag to reorder,
set the spacing, **Apply spacing** to fill in the series.

> Set the section order and spacing **before** using DeepSlice - that is what lets
> the app tell you when a prediction came out wrong.

### 5.5 Let DeepSlice place them all

**Pre-match all (DeepSlice)** in the atlas matcher predicts every section at once.

- It **overwrites every AP on the slide**, so if you have set some by hand it asks
  first and names them.
- It fixes the *order* but not the *spacing*, so the app checks the result and
  warns if sections come back out of order or too close together. Fix those before
  registering.
- Any AP you set by hand also **guides** it: one pinned section sets the overall
  offset, two or more set offset and scale.

### 5.6 Register

**Settings ▸ Registration** holds the options. The defaults are good; the ones
worth knowing:

- **Predict planes with DeepSlice** - place the sections automatically as part of
  the run.
- **Regularized registration (elastix)** - keeps the atlas outline on the tissue.
  Recommended, and on when the elastix extra is installed.
- **Keep hand-corrected sections on re-run** - so a re-run does not throw away
  corrections you made by hand.

Then **Register ▸ Register all sections**. Watch the progress, then check the
**residuals** table (lower is a better fit) and switch on **Show atlas overlay on
sections** to see the outlines on your tissue.

### 5.7 Hand-correct a section

Click the section in the image, or pick it in **Manual atlas adjustment ▸
Section**.

**Box transform** - **Adjust atlas (drag in viewer)**: drag to move, drag the
handles to scale, stretch or rotate. Toggle off to apply.

**Landmarks**, for local distortion a box cannot fix:

1. **Place landmarks** drops points on recognisable features - outline tips,
   junctions, corners.
2. Drag each onto the matching spot on your tissue. The outline follows as you
   drag. **Ctrl+drag** moves a point without warping.
3. **Apply landmark warp**.

**Reset morph to plane** drops the automatic warp but keeps the level - the right
move for torn tissue or a missing piece, then place landmarks by hand.
**Reset adjustment** clears a correction.

### 5.8 Probes

1. **Probes ▸** choose a **Type**, give it a **Label**, set **Shanks**, **Add
   probe**. Labels must be unique - they name the columns you export.
2. Choose **Tip** or **Entry** and click in the image. For entry you can instead
   draw a **Trajectory line**; the entry is where it crosses the surface.
3. Each shank has its own colour; tips are discs, entries are triangles.
4. **Select / move** to drag a marker; **Clear selected** to remove some.

### 5.9 Export

Everything is in the right-hand **3D & Export** panel.

**Update probe coordinates** first if you have moved a marker or corrected a
section - exports use the last computed coordinates. **Enforce rigid array**
regularises a multi-shank probe to parallel, evenly spaced shanks.

**3D Visualization** - **Region atlas** names regions from a different atlas
without re-registering (this is how you get Franklin-Paxinos names). **3D view**
opens the brain and probes in a 3-D window.

**Export** - pick a **Format**, then **Export…**:

| Format | What you get |
|---|---|
| Per-channel coordinates (CSV) | One row per recording channel, with its atlas region |
| Probe tracks for Python / HERBS (pkl) | The tracks, for the older pipeline |
| 3D view as interactive HTML | A page you can send someone |
| Registered section series (folder) | Your sections, in order, with region outlines |

The per-channel CSV has columns `probe, shank, channel, ap_um, ml_um, dv_um,
depth_source, region, region_id, region_color`. `depth_source` says whether that
shank's depths came from the ephys alignment or from probe geometry alone. The
three region columns come from looking each channel up in the project's atlas:
the acronym, the Allen structure id, and the atlas colour as `#rrggbb`. A channel
outside the atlas has an empty acronym, id `0` and an empty colour. The atlas is
loaded on first use, so the first CSV export can take a moment.

**Convert to Paxinos stereotaxic coordinates** (CSV only) converts the finished
coordinates to millimetres from bregma. The **?** explains the choices and how far
apart they are - they are published estimates, so validate against your histology.

The **section series** writes your sections in front-to-back order, straightened,
with the atlas outlines as a separate black-on-white image per section, plus a
manifest. Options add outlines burnt onto the section, an editable **SVG** of the
outlines, and a **region list** naming every region in every section.

### 5.10 Refine depth from recordings (optional)

Needs the `ephys` extra and a probe with tip and entry already registered.

1. **Ephys ▸** pick the probe and shank, point at the recording folder.
2. Open Ephys and SpikeGLX store the probe layout; **Intan does not**, so for
   Intan pick a **Probe map** (your RHX `-probe.xml`, or the built-in wired map).
   Without one the app refuses rather than reporting depths in channel numbers.
3. **Compute features from recording**, then **Open alignment**.
4. Drag the anchor lines so features in the recording line up with region
   boundaries, then **Apply** to store per-channel coordinates.

### 5.11 Without the GUI

```bash
atlastrack split IMAGE                  # find sections
atlastrack register PROJECT.json        # register a whole project
atlastrack export PROJECT.json          # per-channel CCF and Paxinos CSVs
```

---

## 6. Troubleshooting

**Every section fails registration at once.** Usually the stain colour was
mistaken for a label and the tissue was masked out. Update and re-run.

**The atlas overlay's outline fits but the inside looks stretched** (an enlarged
ventricle, say). Intensity-based warping has little to go on inside. Place
landmarks on that structure - raising smoothness or the grid will not help.

**DeepSlice APs come out in the wrong order or bunched together.** Set the section
order and spacing first, then re-run, and read the warning.

**Elastix options are greyed out.** Install the elastix extra; without it the
plain B-spline is used.

**The first DeepSlice run in a session is slow.** It loads a large model once.

**The window will not open.** Run `atlastrack gl-info` and send the output.

**Region names look wrong.** Check **Region atlas** in 3D & Export - Allen and
Chon/Kim name the same tissue differently (MOp vs M1). And do not switch the
registration atlas mid-project: levels assigned under one do not carry to another.

---

## 7. Conventions

- Outputs go next to your data, never into the app's folder.
- Projects are `*.atlastrack.json`; atlases cache under `~/.brainglobe`.
- The project auto-saves after a hand correction.
