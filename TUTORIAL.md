# AtlasTrack - Tutorial

One histology slide registration, start to finish. About 20 minutes, most of it 
waiting for the atlas to download and the fit to run. For the full reference see
 **MANUAL.md** (or the **Manual** tab in the app).

You need one image of a slide with brain sections on it - a whole-slide scan with
several sections on it is ideal. Any common format works (`.tif`, `.jpg`, `.png`).

---

## Before you start

```bash
uv pip install "atlastrack[all]"   # everything - recommended for this walkthrough

atlastrack version
atlastrack gui
```

`[all]` brings three optional pieces: **elastix** (the better registration
engine - worth having, it is what step 6 uses), **deepslice** (automatic
placement, step 5), and **ephys** (the optional last section). A plain
`uv pip install atlastrack` runs the walkthrough too, but the fit is plainer and
the DeepSlice step is unavailable.

---

## 1. Load the slide

**Histology ▸ Open histology image(s)** → pick the image.

It appears in the centre **Project** tab. If you pick several images they stack
into one canvas so all the sections share one coordinate space.

## 2. Find the sections

Click **Detect sections**.

Yellow boxes appear and the status says how many were found. If debris was picked
up, raise **Min area** and detect again. If one section came out as two pieces,
raise **Closing radius**.

Missing one? **Draw new bounding box** and drag a rectangle around it - it becomes
a section as soon as you let go. To change a box, click **Edit boxes** and drag its
handles.

Sections are numbered column by column, within each image you loaded.

## 3. Straighten, if needed

Click a section in the image - the **Adjustments ▸ Section** dropdown follows your
click. Set **Scope** to *Selected section*, then use **Flip H / Flip V** if the
tissue is mirrored, and **Levels ▸ Auto** if it is too dark to judge.

You can leave rotation alone. The exported series straightens itself later.

## 4. Load the atlas

**Atlas ▸ Load atlas** with *Allen CCFv3 25 µm*.

The first download is around 400 MB and takes a while; after that it loads
instantly from `~/.brainglobe`. The **?** beside the picker explains the other
atlases - notably **Chon / Kim**, which gives Franklin-Paxinos region names.

## 5. Put the sections at the right level

Tell the app the order and spacing first, under **Section order and spacing**:

- The list runs front to back, top first. Check it matches your slide - drag to
  reorder if not.
- Set **Section spacing (µm)** to how far apart your sections were cut.

Now place them. Either:

**By hand** - **Matching viewer ▸ Open atlas matcher**. Your section sits beside
the atlas slice. Turn **AP from bregma** until they match, **Assign**, then step
to the next. Pin one section and use **Assign all** to space the rest out from it.

**Automatically** (needs the `deepslice` extra) - **Pre-match all (DeepSlice)** in
the same window places every section in one pass. It warns you if any come back
out of order or bunched together, and asks before overwriting APs you set by hand.
Anything you did set also guides it, so pinning one or two good levels is worth
doing.

## 6. Register

**Register ▸ Register all sections**.

Roughly half a minute per section. When it finishes, the **residuals** table fills
in - lower is a better fit. Tick **Show atlas overlay on sections** to see the
region outlines warped onto your tissue. This is the moment to judge the result:
the outlines should follow the anatomy, not just the silhouette.

If one section missed, click it and use **Manual atlas adjustment**:

- **Box transform** for a section that is simply offset or the wrong size.
- **Landmarks** for local distortion: **Place landmarks**, drag each point onto
  the matching feature, **Apply landmark warp**.
- For torn tissue or a missing piece, **Reset morph to plane** first - that gives
  you back a clean atlas slice to fit by hand.

Corrections re-map probes and save automatically.

## 7. Add a probe

**Probes ▸** choose a **Type**, give it a **Label**, **Add probe**.

Select **Tip** and click where the shank ended; select **Entry** and click where it
went in, or draw a **Trajectory line** and let the app find the surface crossing.
Each shank gets its own colour - tips are discs, entries are triangles.

Wheel zooms; **Ctrl**+wheel and **Shift**+wheel pan.

## 8. Look at it in 3-D

Right panel, **3D & Export**:

1. **Update probe coordinates** - always do this after moving a marker or
   correcting a section. It re-maps everything through the current registration.
2. **3D view** opens the brain, the regions and your probe in a 3-D window.

Both 3-D views are referenced to bregma and are not mirrored.

## 9. Export

Pick a **Format** and click **Export…**:

- **Per-channel coordinates (CSV)** - one row per channel. Tick **Convert to
  Paxinos stereotaxic coordinates** for millimetres from bregma instead; the **?**
  explains the options and how much they disagree.
- **3D view as interactive HTML** - a page you can send to someone.
- **Registered section series** - your sections in order, straightened, with the
  atlas outlines beside them. Add the **SVG** for editable outlines, or the
  **region list** for a table of every region in every section.
- **Probe tracks for Python / HERBS (pkl)** - for the older pipeline.

Then **Project ▸ Save Project** - it writes `<slide>.atlastrack.json` next to your
image. **Load Project** brings everything back and reloads the atlas for you.

To open a project in Python:

```python
from atlastrack.project.io import load_project
p = load_project(r"path\to\project.atlastrack.json")
```

---

## Optional: refine depth from a recording

Needs the `ephys` extra and a probe that already has tip and entry.

1. **Ephys ▸** pick the probe and shank, then point at the recording folder.
2. If it is an **Intan** recording, choose a **Probe map** - Intan does not store
   where the sites are. Open Ephys and SpikeGLX do, so leave it on *From the
   recording* for those.
3. **Compute features from recording**, then **Open alignment**.
4. Drag the anchor lines so features in the recording line up with region
   boundaries, then **Apply**.

---

## Did it work?

| Check | What you should see |
|---|---|
| Sections | A box round each section, none round debris |
| Atlas | Status shows the atlas and resolution |
| Levels | Neighbouring sections a sensible distance apart |
| Registration | Every section has a residual, no error dialog |
| Overlay | Outlines follow the anatomy, not just the outer edge |
| 3-D view | The probe sits inside the brain, in the structure you expect |
| CSV | One row per channel, with its region, id and colour |

---

## If something goes wrong

The **Manual** has a troubleshooting section. The two most common:

- **The window will not open** - run `atlastrack gl-info` and send the output.
- **Every section failed at once** - usually the stain colour was mistaken for a
  label and the tissue was masked out. Update and re-run.
