# AtlasTrack - User Manual

This is the reference manual. If you're a new user, start with **TUTORIAL.md** (or the **Tutorial** tab in the app), which
walks through one registration from start to finish.

---

## 1. What the app does

You give it one or more images of brain sections. It:

- finds each section and lets you tidy the boxes,
- places each section at the right front-to-back level in the atlas,
- warps the atlas onto each section so region outlines follow your tissue,
- turns probe tracks you click into atlas coordinates,
- and exports the result - coordinates, figures, or your section series with
  region outlines.

The default settings are good enough for most registrations. Most of the work
is looking at each section, checking how well the atlas matches, and correcting
the fit where it misses.

### How a section is fitted

If you have tried DeepSlice on its own, it only places each section with a
*linear* fit: an atlas plane, tilted, shifted and scaled, but never bent. Real
tissue is stretched and squashed by cutting and mounting, so that alone rarely
lines up. Here DeepSlice only chooses the plane; the fit is done after it.

1. **The plane.** DeepSlice gets one tight crop per section, with your flips
   applied, **numbered in the section order you set**. It uses those numbers to
   keep the series in front-to-back order and to share one tilt across it - which
   is why setting the order first matters (Recipe 5.4).
   - **Anchors.** Any AP you set by hand, or by even spacing, pins that section:
     its plane slides along AP to exactly your value, keeping DeepSlice's tilt.
   - **Interpolation.** Sections without an anchor shift by interpolating the
     corrections of their neighbouring anchors. One anchor fixes the offset of the
     whole series; two or more also fix its spacing.
   - APs that **Pre-match all** wrote are DeepSlice's own predictions, not
     anchors. So after a pre-match, correcting a few sections by hand also carries
     the sections between them.
   - Without DeepSlice, the plane is the AP (and tilt) you set.
2. **Scale and shift.** The outline of the atlas slice is moved and scaled onto
   the outline of your tissue: same centre, same size, no rotation.
3. **Warp.** elastix, a registration program, bends the atlas onto the section.
   It moves a grid of control points (a *B-spline*) until the light and dark
   patterns of the two images agree as well as they can (measured by *mutual
   information*). It compares tissue pixels only - bright fluorescent labels are
   left out, since the atlas has nothing like them - and a penalty on bending
   keeps the warp smooth.
4. **Edge snap.** The outline of the atlas brain is pulled onto the border of the
   tissue, with the inside held still. A push too large to be a fit - torn or
   missing tissue - is skipped, and a warp that would fold is rejected.
5. **Check.** If the result overlaps the tissue clearly worse than the plain
   plane did, the plane is kept instead and the Register panel names the section,
   for you to fit by hand (Recipe 5.7).

Your box and landmark corrections (Recipe 5.7) sit on top of this fit.
Re-registering a section clears them, because they corrected the fit it replaces.

---

## 2. Atlases and coordinates

**The atlas.** A 3-D reference brain. Positions are in micrometres (µm) along
three axes: **AP** front-to-back, **ML** left-right, **DV** top-to-bottom.

**AP from bregma.** The app shows front-to-back position relative to **bregma**,
the skull landmark: `0` = bregma, **negative = behind it**, positive = in front.

**Atlases you can use.** Each is downloaded once and then kept on your computer
(see Conventions, section 7).
**Help ▸ Atlases** describes each one, with links and the bregma it uses:

| Atlas | Why you would pick it |
|---|---|
| Allen CCFv3 | The default. |
| CCFv3-BBP Augmented | Adds cerebellar layers, olfactory bulb layers, barrel columns. |
| Chon / Kim Unified | Franklin-Paxinos region names (M1, S1BF, 4V) instead of Allen's. |
| Chon / Kim v2, isotropic | The 2024 re-release of the above, 20 µm. |
| Custom ID | Any other BrainGlobe atlas, typed in by name. |

**The whole process in one line:** section image → atlas slice at the right level →
that slice warped onto your section → outlines and probe coordinates.

**Your project** is one `*.atlastrack.json` file. Outputs go **next to your data**,
not into the app's folder.

**Each section also gets a folder** beside the project, `sections/section_NNN/`,
written when you save:

- `image.tif` - the section cut from the slide, with its flips and rotation, as
  registration sees it. It is cut again only if you change the section's box.
- `section.json` - the section in plain numbers: front-to-back level (AP), the
  cutting angle (pitch and yaw), rotation, µm per pixel, and each landmark as an
  image pixel paired with its atlas point. It is for reading; editing it changes
  nothing - the project file is what the app reads.

If the slide image is missing when you open a project, the app rebuilds the slide
from these section images, so you can share a project without its slide.

---

## 3. Install and launch

```bash
uv pip install "atlastrack[all]"   # everything - recommended
uv pip install atlastrack          # base only: the histology → atlas workflow

atlastrack gui       # launch
atlastrack version   # check the install
atlastrack gl-info   # if the window will not open, run this and send the output
```

`pip install` works as well as `uv pip install`. Put the package name in quotes
and leave no spaces inside the brackets - PowerShell and `zsh` both read
`[...]` as a file pattern otherwise.

The base install is deliberately light. `[all]` adds three optional pieces, which
you can also install one at a time (`".[elastix]"` and so on):

| Extra | What it adds | Notes |
|---|---|---|
| `elastix` | The registration engine that keeps the warp smooth (*regularized*) | Recommended - it is the setting the Register step relies on. ~150 MB. Without it the app falls back to a plainer fit and that option is greyed out. |
| `deepslice` | Automatic front-to-back placement | ~1.65 GB (it includes TensorFlow, a large machine-learning library), so it is the one to skip if you are placing sections by hand. |
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
them into one image so every section shares the same coordinates.

Opening an image when one is already loaded **replaces** it. Same size keeps your
sections and registration - handy for the same slide in a different dye. A
different size starts fresh.

**Channel images.** To keep the combined image and still see each dye on its own
- where tracks from several probes overlap, say - add the single-dye exports as
channel images: **Histology ▸ Channels ▸ Add channel image...**, one file per
slide image (e.g. `Slide 3_red.png` and `Slide 4_red.png`), the same size as the
slide images. **Show** then switches between the slide image and each channel.
They share the slide's sections, flips and registration, so one project can hold
the probes of every session; registration always uses the slide image. From the
command line: `atlastrack add-channel PROJECT.json NAME FILE...`.

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
  registered makes that fit out of date - the panel says so, and you register
  it again.
  For a tidy exported series you usually need none of this: the section-series
  export straightens on its own (Recipe 5.9).
- **Flip H / Flip V** if the tissue is mirrored.
- **Sections seen from** says which face of the sections your images show, once
  flipped as you want them: **Back** puts the animal's right on the image's right;
  **Front**, as atlas plates are drawn, puts it on the image's left. Set it once:
  it is remembered for new projects. The atlas is almost symmetric, so the tissue
  cannot tell the two apart - every plane, from DeepSlice or set by hand, is placed
  for this setting, and getting it wrong mirrors ML and swaps the hemispheres.
  Changing it on a project that already has planes offers to mirror them, keeping
  each fit, warp and landmark (`atlastrack seen-from PROJECT.json back
  --mirror-planes` does the same). Sections matched to another project are turned
  by matching again instead.
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
- APs you set or correct by hand **after** the pre-match guide it at
  registration: one sets the offset of the whole series, two or more also its
  spacing. So correct only the sections that came out wrong - the ones between
  them follow (see *How a section is fitted* in section 1).

### 5.6 Register

**Settings ▸ Registration** holds the options. The defaults are good; the ones
worth knowing:

- **Predict planes with DeepSlice** - place the sections automatically as part of
  the run. Leave it on after a pre-match: the run then reuses the pre-match
  instead of running DeepSlice again, and keeps the tilt it predicted.
- **Regularized registration (elastix)** - keeps the atlas outline on the tissue.
  Recommended, and on when the elastix extra is installed.
- **Keep hand-corrected sections on re-run** - so **Register all sections** skips
  sections you corrected by hand. Off (the default), running it again replaces their fit
  and clears the corrections, which belonged to the old fit.

**Align on** (above the Register button) says which colour channel holds the
tissue stain - Nissl, DAPI and the like. Registration then compares only that
channel with the atlas, and DeepSlice sees only that channel too. Left on **All
channels**, the app uses all of them and leaves out bright red and green, taking
them to be labels; that guess fails when the stain itself is red or green, so
name the channel when you can.

Then **Register ▸ Register all sections**. Watch the progress, then check the
**Residual** column - how much mismatch the fit left, lower is better - and
switch on **Show atlas overlay on
sections** to see the outlines on your tissue.

**Check series (AP and tilt)** plots every section's AP, pitch and yaw in series
order. Sections cut from one block should step evenly front to back at one angle,
so one out of order, or tilted more than 2° unlike the rest, is drawn in red and
named below the plot. Click a point to select that section. (The plot needs the
`ephys` extra, which includes pyqtgraph.)

**Only some sections.** Select them in the table - click one, **Ctrl**+click to
add or remove, **Shift**+click for a range - or with the same clicks on the
sections in the image. The button becomes **Register selected sections**, and
registers those even if they are hand-corrected. **Esc** in the table, or a click
on an empty part of the image, goes back to all.

### 5.7 Hand-correct a section

Click the section in the image, or pick it in **Manual atlas adjustment ▸
Section**.

**Box transform** - **Move / scale / rotate overlay** shifts the whole overlay
as one rigid piece. The view jumps to the section and draws a box round it: drag
inside the box to move, drag a corner or edge handle to scale or stretch, drag the
handle above the top edge to rotate. Click **Apply box transform** when done.

**Landmarks**, for local distortion a box cannot fix. Either place the points and
drag them, or click them in pairs.

Drag the auto-placed points:

1. **Place landmarks** drops points on recognisable features - outline tips,
   junctions, corners.
2. Drag each onto the matching spot on your tissue. The outline follows as you
   drag. **Ctrl+drag** moves a point without warping.
3. **Apply landmark warp**.

Or work side by side, the way HERBS does: **Open split panel window** shows the
section on the left and the atlas, as currently registered, on the right.

1. **Auto-place points** puts numbered dots on both. Dots you have not moved yet
   are amber and hold the atlas still where they are.
2. Drag each dot on the section onto the feature it marks on the atlas. Click
   empty space to add a pair; right-click a dot, or click it and press
   **Delete**, to remove it. **Undo** steps back one change.
3. **Preview warp** shows the result, **Apply landmark warp** keeps it.

The window can also lay the atlas over the tissue at any opacity, and nudge the
atlas plane (AP, ML tilt, DV tilt). **Apply plane** saves the new plane; register
the section again for the overlay to follow it.

Both routes fill the same landmark set, so you can start with the auto-placed
points and add hand-clicked pairs to them.

**Reset morph to plane** drops the automatic warp but keeps the level - the best
start for torn tissue or a missing piece, which you then fit with landmarks.
**Reset adjustment** clears a correction.

### 5.8 Probes

1. **Probes ▸** choose a **Type**, give it a **Label**, set **Shanks**, **Add
   probe**. Labels must be unique - they name the columns you export.
2. Under **Probe tracks**, pick the **Probe** and **Shank**, then **Add track**.
   The cursor becomes a probe. Click the end of the track (the shank tip), then
   the point where it went in (the entry). A dotted line follows the cursor
   between the two clicks. **Esc** cancels. Adding a track for a shank that has
   one replaces it.
3. Each shank has its own colour, used by its markers (tip = circle, entry =
   triangle), its line and its row in the list. The line runs from the tip
   through the entry to the edge of the section's box. If the tip and the entry
   are on different sections, each section shows its own part of the line. To
   draw it, the other marker is carried over to this section: through the atlas
   when both sections are registered, otherwise to the same place relative to
   the tissue (its midline, width, top and height). The atlas version is worked
   out in the background, so the line may shift slightly a few seconds after
   the tab opens.
4. To change a track, click its marker or line (or its row in the list) to
   select it. Drag a marker to move it; press **Delete** to remove the track.
   **Clear all tracks** removes every track.

### 5.9 Export

Everything is in the right-hand **3D & Export** panel.

**Update probe coordinates** first if you have moved a marker or corrected a
section - exports use the last computed coordinates. **Enforce rigid array**
straightens a multi-shank probe so its shanks are parallel and evenly spaced.

**3D Visualization** - **Region atlas** names regions from a different atlas
without re-registering (this is how you get Franklin-Paxinos names). **3D view**
opens the brain and probes in a 3-D window. **Update probe coordinates** redraws
the probes in that window and keeps your view. Closing the main window closes the
3-D window too.

**Export** - pick a **Format**, then **Export…**:

| Format | What you get |
|---|---|
| Per-channel coordinates (CSV) | One row per recording channel, with its atlas region |
| Probe tracks for Python / HERBS (pkl) | The tracks, for HERBS or your own Python scripts |
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
file listing them all. Options add outlines burnt onto the section, an editable **SVG** of the
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
atlastrack import SOURCE                # convert a registration made by another tool
atlastrack match PROJECT.json REF.json  # register sections from the same sections in REF
atlastrack add-cells PROJECT.json DIR   # add slicereg's counted cells to its import
atlastrack add-channel PROJECT.json NAME FILE...  # another image of the slide, e.g. one dye
atlastrack seen-from PROJECT.json back  # which face the images show (--mirror-planes)
```

### 5.12 Import a registration from another tool

**Project ▸ Import from another tool**, or `atlastrack import SOURCE`, reads:

- **slicereg** ([github.com/mvdokh/cell-counting](https://github.com/mvdokh/cell-counting)):
  pick the project folder, the one holding `project.json` and `slices/`.
- **QuickNII, DeepSlice or VisuAlign** (the QUINT tools): pick the JSON file that
  lists the sections (a VisuAlign `.waln` works too). The section images must be in
  the same folder. Only mouse atlases in the Allen space (Allen, Kim) are read.

The result goes into a new folder named after the source with `_atlastrack` added,
next to it; the source is not changed. The project opens straight away.

What comes across, and how:

- **Each section's image**, into its section folder. slicereg sections are brought
  to one pixel size (the coarsest; `--pixel-um` sets another) and their channels
  coloured as slicereg showed them; the alignment channel becomes **Align on**.
- **The plane**, exactly.
- **The non-linear correction**, exactly. slicereg's landmarks become landmark
  corrections you can edit (with 8 fixed points just outside the image, where
  slicereg holds its correction at zero). VisuAlign's markers bend the atlas in
  straight-edged triangles, which landmarks cannot copy, so they come in as the
  section's fitted warp instead; you can add landmarks on top.

- **slicereg's counted cells** (`cells.csv`), drawn on their sections and in 3-D.
  For a project imported before cells came across, `atlastrack add-cells
  PROJECT.json SLICEREG_FOLDER` adds them.

Atlas coordinates then match the source tool's to within 0.02 µm (slicereg) and
1 µm (VisuAlign). Not imported: slicereg's 3-D renders.

### 5.13 The same sections, imaged twice

When the sections on your slides were also imaged elsewhere and registered there
- on a confocal for cell counts, say - the slides need no atlas registration of
their own. Matching each section to its counterpart is quicker and more accurate,
and puts probe tracks and cells in exactly the same atlas coordinates.

1. Load the slides and find the sections as usual (5.1, 5.2). No need to flip
   them: the match finds out which are mirrored.
2. **Register ▸ Match to a registered project...** and pick the other project.
   Each section is paired with its counterpart by the shape of its tissue
   outline, then fitted to it on the stain (scale, turn, shift and a little
   stretch).
3. A table lists the pairs with how well the outlines overlap once fitted (1 =
   perfectly). Pairs below 0.93 are left unticked for you to look at. **Apply
   ticked**: those sections are flipped where needed and take their
   counterpart's registration - its plane, its warp and its hand corrections.
   Sections without a counterpart are left as they were; register them the
   usual way.
4. Place the probe tracks (5.8). The other project's cells come along, drawn on
   the sections and in the 3-D view and HTML export.

Each section remembers what it was matched to and how. The registration is a
copy: if the other project is corrected later, match again.

---

## 6. Troubleshooting

**"Planes placed as if seen from the front/back" when a project opens.** Some
sections' planes were placed for the other face of the sections than the project
says (Recipe 5.3, Sections seen from), so their ML is mirrored and the hemispheres
swapped, while the outlines still look right. Mirror them from **Sections seen
from**, or, for sections matched to another project, match again; then re-export.

**Every section fails registration at once.** Usually the stain colour was
mistaken for a fluorescent label and the tissue was left out of the fit. Update
AtlasTrack and register again.

**The atlas overlay's outline fits but the inside looks stretched** (an enlarged
ventricle, say). The warp follows patterns of light and dark, and there are few
inside the brain. Place landmarks on that structure - changing the warp settings
will not help.

**DeepSlice APs come out in the wrong order or bunched together.** Set the section
order and spacing first, then run the pre-match again and read the warning.

**Elastix options are greyed out.** Install the elastix extra; without it a
plainer fit is used.

**The first DeepSlice run in a session is slow.** It loads a large model once.

**The window will not open.** Run `atlastrack gl-info` and send the output.

**Region names look wrong.** Check **Region atlas** in 3D & Export - Allen and
Chon/Kim name the same tissue differently (MOp vs M1). And do not switch the
registration atlas mid-project: levels assigned under one do not carry to another.

---

## 7. Conventions

- Outputs go next to your data, never into the app's folder.
- Projects are saved as `.json` files (e.g., `*.atlastrack.json`). Atlases are
  kept where BrainGlobe stores them, by default in your home folder (e.g.,
  `~/.brainglobe` on Linux).
- The project auto-saves after a hand correction.
