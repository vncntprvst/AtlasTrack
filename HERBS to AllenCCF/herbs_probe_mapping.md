# How to: Visualize HERBS probe trajectories in Allen CCF space

**Script:** `docs/how-to/herbs_probe_mapping.py`

Loads HERBS pkl registration files, converts electrode positions to Allen CCF
coordinates, and produces an interactive 3D figure (Plotly) with semi-transparent
brain structure meshes — the same visualization used in the BAM book probe
trajectories page, but as a standalone Python script / notebook.

---

## Dependencies

```bash
pip install brainglobe-atlasapi plotly numpy
```

The first run downloads the `allen_mouse_100um` atlas (~100 MB) from BrainGlobe
and caches it in `~/.brainglobe/`.

---

## Quick start

1. Open `herbs_probe_mapping.py` and edit the configuration block at the top:

   ```python
   HERBS_PKL_FILES = [
       ("/path/to/probe_session1.pkl", "Session 1"),
       ("/path/to/probe_session2.pkl", "Session 2"),
   ]
   ```

   Each entry is `(path_to_pkl, label)`.  Any section pkl (`_1`, `_2`, `_3`, `_4`)
   encodes the same trajectory — pick any one.

2. Run:
   ```bash
   python docs/how-to/herbs_probe_mapping.py
   ```
   This writes `probe_trajectories.html` (~5 MB) which you can open in any browser.
   To open directly in the browser instead, set `OUTPUT_HTML = None`.

3. In a **Jupyter notebook**, call `make_figure()` directly:
   ```python
   from docs.how_to.herbs_probe_mapping import make_figure
   fig = make_figure(HERBS_PKL_FILES)
   fig.show()
   ```

---

## What the figure shows

- **Transparent meshes** — selected brain structures from the Allen CCF atlas
  (brainstem, cerebellum, medulla, VII, IRN, thalamus, striatum, motor cortex by default).
- **Colored scatter points** — each HERBS sample site, one color per shank per session.
  Hovering shows the session label, shank index, and HERBS region acronym.
- **Axes** — Allen CCF µm: x = ML (left→right), y = AP (anterior→posterior),
  z = DV (dorsal→ventral, inverted so dorsal faces up).

Use the mouse to rotate / zoom / pan.  Click a legend entry to toggle visibility.

---

## Customizing brain structures

Edit `BRAIN_STRUCTURES` in the configuration block.  Each entry is a 4-tuple:

```python
BRAIN_STRUCTURES = [
    ("BS",  "Brainstem",  "#b0a0c8", 0.07),   # acronym, label, hex color, opacity
    ("IRN", "IRt / IRN",  "#40a850", 0.30),
    # add any Allen CCF acronym here, e.g. "SCm", "PAG", "IO", "PB", ...
]
```

Allen CCF structure acronyms can be looked up at https://atlas.brain-map.org or via:
```python
from brainglobe_atlasapi import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_100um", check_latest=False)
# List all acronyms
acrs = [s["acronym"] for s in atlas.structures_list]
```

---

## HERBS coordinate system

HERBS stores electrode positions as voxel indices into the Allen CCF 10 µm atlas
(shape 1320 × 800 × 1140 in AP × DV × ML), but with **two axes reversed**:

| HERBS axis | Semantic direction | Allen CCF direction |
|---|---|---|
| axis0 | ML, left → right (reversed) | ML, right → left |
| axis1 | AP, caudal → rostral (reversed) | AP, anterior → posterior |
| axis2 | DV, dorsal → ventral (standard) | DV, dorsal → ventral |

The script applies the conversion automatically:

```
ML (µm) = (1139 − vox_axis0) × 10
DV (µm) =         vox_axis2  × 10
AP (µm) = (1319 − vox_axis1) × 10
```

The axis flip was confirmed by verifying that the probe-tip coordinates match the
stereotaxic insertion target within ~100 µm, and that atlas lookups return the
expected brain regions.

---

## Using probe geometry (recording channel coordinates)

The HERBS pkl provides ~128 sample points per shank, spaced ~40 µm apart along
the full probe track.  Recording channels are more densely spaced (e.g., 15 µm
pitch for NP2.0).  To assign CCF coordinates to individual recording channels,
interpolate along the HERBS trajectory:

```python
from scipy.interpolate import interp1d

shank = load_herbs_pkl(pkl_path)[0]      # first shank
ccf   = shank["ccf"]                     # (128, 3) in ML, DV, AP order
depth = ccf[:, 1].max() - ccf[:, 1]     # depth from tip (0 = tip), in µm

order   = np.argsort(depth)
depth_s = depth[order]
ccf_s   = ccf[order]

# Interpolator for each CCF component
f = {ax: interp1d(depth_s, ccf_s[:, i],
                  bounds_error=False,
                  fill_value=(ccf_s[0, i], ccf_s[-1, i]))
     for i, ax in enumerate(["ML", "DV", "AP"])}

# Example: NP2.0, 320 channels/shank, 15 µm pitch, depth = row * 15 µm
for ch in range(320):
    y_ch = (ch // 2) * 15.0        # depth from tip for this channel
    ml, dv, ap = f["ML"](y_ch), f["DV"](y_ch), f["AP"](y_ch)
```

See `scripts/ingestion/convert_shijia_to_nwb.py` (`_herbs_channel_coords`) for
a complete implementation used in the NWB ingestion pipeline.
