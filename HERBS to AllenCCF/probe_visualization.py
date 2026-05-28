#!/usr/bin/env python3
"""
General probe visualization for Allen CCF coordinates.

This script makes an interactive Plotly HTML scene with:
  1. Selected Allen / BrainGlobe brain-region meshes.
  2. Probe trajectories from either a coordinate CSV or a folder of HERBS .pkl files.
  3. Optional unit/effect points projected onto each probe face.

The code is intentionally dataset-agnostic. Animal IDs are read from the input
data, not from any hard-coded prefix, and feature/unit rows are matched to probes
by a general recording-name-to-animal-ID rule.

Dependencies
------------
    pip install brainglobe-atlasapi plotly numpy

Examples
--------
    # Original style coordinate CSV.
    python probe_visualization.py --csv-path ~/Desktop/Probe_coordinates_confirmed_allen_ccf.csv

    # Plot a different set of regions.
    python probe_visualization.py --csv-path probes.csv --brain-regions root,MOp,SSp-bfd,VIS

    # Read HERBS probe pickles from one folder.
    python probe_visualization.py --probe-dir ~/Desktop/Probes_Refined

    # Interpolate shanks 2 and 3 between measured shanks 1 and 4 for every animal
    # with those endpoint shanks.
    python probe_visualization.py --csv-path probes.csv --interpolate-shanks

    # Overlay an arbitrary effect column from a unit table.
    python probe_visualization.py --csv-path probes.csv --feature-csv unit_table.csv --feature-column mass_effect
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import plotly.graph_objects as go
from brainglobe_atlasapi import BrainGlobeAtlas


# Allen CCF extents in microns. These are used when converting HERBS voxel
# coordinates to physical coordinates and when converting bregma-relative CSVs.
AP_UM = 13200.0
DV_UM = 8000.0
ML_UM = 11400.0
BREGMA_AP_UM = 6600.0
MIDLINE_ML_UM = 5700.0

# Neuropixels 2.0 shank dimensions in microns. They are only used for drawing
# probe meshes and for projecting unit/effect points onto the probe face.
SHANK_WIDTH_UM = 70.0
SHANK_THICKNESS_UM = 24.0
SHANK_TIP_LENGTH_UM = 175.0
SHANK_PITCH_UM = 250.0
ELECTRODE_COLUMN_CENTER_UM = 16.0

# Default regions match the original visualization but can be replaced from the
# command line with --brain-regions or --region.
DEFAULT_REGIONS: tuple[tuple[str, str, str, float], ...] = (
    ("root", "Whole brain", "#9aa7ba", 0.26),
    ("SSp-bfd", "S1 Barrel field", "#e41a1c", 0.12),
    ("SSp-ul", "S1 Upper limb", "#4daf4a", 0.12),
    ("SSp-ll", "S1 Lower limb", "#377eb8", 0.12),
    ("MOp", "M1 primary motor", "#ffff33", 0.10),
    ("PTLp", "Posterior parietal", "#a65628", 0.10),
    ("VIS", "Visual areas", "#17becf", 0.10),
)

PLOT_COLORS: tuple[str, ...] = (
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#9a6324",
    "#469990",
    "#dcbeff",
    "#800000",
)

# Optional code-level default output path.
#
# Leave as None to save next to the input as:
#   probe_visualization.html
#   probe_visualization_with_features.html
#
# Or set to a string/Path to choose the HTML name and folder in the code:
DEFAULT_OUTPUT_HTML = "~/Desktop/my_probe_plot.html"
#
# The command-line --output argument still overrides this value.
DEFAULT_OUTPUT_HTML: Optional[str | Path] = None


@dataclass(frozen=True)
class RegionSpec:
    """One atlas mesh to draw."""

    acronym: str
    label: str
    color: str
    opacity: float


@dataclass
class ProbeRow:
    """One plotted shank trajectory in AP/ML/DV Allen CCF microns."""

    animal_id: str
    shank: int
    ccf: np.ndarray
    color: str
    source: str
    rotation_deg: float = 270.0
    regions: Optional[list[str]] = None
    input_kind: str = "csv"
    interpolation_note: str = ""
    raw_csv_row: Optional[dict[str, str]] = None


@dataclass
class FeaturePoint:
    """One unit/effect point before projection into CCF space."""

    animal_id: str
    recording_name: str
    shank: int
    depth_um_from_surface: float
    x_um_local: float
    effect_value: float
    effect_name: str
    source_row: dict[str, str]


def normalize_column_name(name: str) -> str:
    """Normalize CSV column names so 'Animal ID', 'animal_id', and 'Animal' match."""

    return re.sub(r"[^a-z0-9]+", "", name.lower())


def make_column_lookup(fieldnames: Iterable[str]) -> dict[str, str]:
    """Map normalized column names back to their exact CSV header spelling."""

    return {normalize_column_name(name): name for name in fieldnames}


def get_column(
    row: dict[str, str],
    lookup: dict[str, str],
    aliases: Iterable[str],
    *,
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    """Read a CSV value using several possible header names."""

    for alias in aliases:
        actual = lookup.get(normalize_column_name(alias))
        if actual is not None:
            return row.get(actual, default)
    if required:
        raise KeyError(f"CSV is missing one of these columns: {list(aliases)}")
    return default


def parse_float(value: object, *, default: Optional[float] = None) -> float:
    """Convert CSV text to float while giving clear errors for required values."""

    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError("missing numeric value")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        if default is not None:
            return default
        raise ValueError(f"could not parse numeric value {value!r}") from exc
    if not np.isfinite(out):
        if default is not None:
            return default
        raise ValueError(f"numeric value is not finite: {value!r}")
    return out


def parse_int(value: object) -> int:
    """Parse shank indices that may be stored as '1', '1.0', etc."""

    return int(float(str(value).strip()))


def default_csv_path() -> Path:
    """Find common coordinate CSV names near this script or on the Desktop."""

    candidates = [
        Path(__file__).resolve().parent / "Probe_coordinates_confirmed_allen_ccf.csv",
        Path(__file__).resolve().parent / "Probes_coordinates_confirmed.csv",
        Path.home() / "Desktop" / "Probe_coordinates_confirmed_allen_ccf.csv",
        Path.home() / "Desktop" / "Probes_coordinates_confirmed.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def default_feature_csv_path() -> Path:
    """Find a unit table near this script."""

    for name in ("unit_table_session.csv", "unit_table.csv"):
        path = Path(__file__).resolve().parent / name
        if path.is_file():
            return path
    return Path(__file__).resolve().parent / "unit_table.csv"


def parse_region_spec(text: str, index: int) -> RegionSpec:
    """
    Parse one region spec.

    Accepted forms:
      ACRONYM
      ACRONYM:Label
      ACRONYM:Label:#RRGGBB
      ACRONYM:Label:#RRGGBB:0.15
    """

    parts = [part.strip() for part in text.split(":")]
    acronym = parts[0]
    if not acronym:
        raise ValueError("Region acronym cannot be empty.")
    label = parts[1] if len(parts) > 1 and parts[1] else acronym
    color = parts[2] if len(parts) > 2 and parts[2] else PLOT_COLORS[index % len(PLOT_COLORS)]
    opacity = parse_float(parts[3], default=0.12) if len(parts) > 3 else 0.12
    return RegionSpec(acronym=acronym, label=label, color=color, opacity=opacity)


def build_region_specs(args: argparse.Namespace) -> list[RegionSpec]:
    """Resolve default or user-requested atlas regions."""

    if args.region:
        return [parse_region_spec(text, i) for i, text in enumerate(args.region)]

    if args.brain_regions:
        acronyms = [item.strip() for item in args.brain_regions.split(",") if item.strip()]
        return [parse_region_spec(acronym, i) for i, acronym in enumerate(acronyms)]

    return [RegionSpec(*item) for item in DEFAULT_REGIONS]


def assign_colors(animal_ids: Iterable[str]) -> dict[str, str]:
    """Assign a stable color to each animal ID in sorted order."""

    return {
        animal_id: PLOT_COLORS[i % len(PLOT_COLORS)]
        for i, animal_id in enumerate(sorted(set(animal_ids)))
    }


def relative_ap_ml_dv_to_ccf(ap_rel: float, ml_rel: float, dv_um: float) -> np.ndarray:
    """
    Convert bregma/midline-relative AP/ML/DV into Allen CCF AP/ML/DV microns.

    Positive AP is anterior to bregma. Positive ML is lateral from midline on the
    right hemisphere. Allen CCF AP and ML coordinates increase in the opposite
    directions used by the original spreadsheet, so both are subtracted from the
    bregma/midline offsets.
    """

    return np.array([BREGMA_AP_UM - ap_rel, MIDLINE_ML_UM - ml_rel, dv_um], dtype=float)


def row_has_ccf_columns(lookup: dict[str, str]) -> bool:
    """True when a coordinate CSV already includes Allen CCF columns."""

    required = (
        "Insertion AP CCF",
        "Insertion ML CCF",
        "Terminus AP CCF",
        "Terminus ML CCF",
        "Terminus DV CCF",
    )
    return all(normalize_column_name(col) in lookup for col in required)


def infer_coordinate_system_for_generic_columns(
    rows: list[dict[str, str]],
    lookup: dict[str, str],
) -> str:
    """
    Guess whether generic AP/ML columns are already CCF or bregma-relative.

    Original bregma-relative spreadsheets usually have AP/ML values near zero to
    a few thousand microns. Allen CCF AP/ML values usually sit near the middle of
    the atlas, for example AP around 5000 to 7000 and ML around 3000 to 6000.
    This heuristic keeps generic "Insertion AP" CSVs convenient while retaining
    --coordinate-system for explicit control.
    """

    sample_values: list[tuple[float, float]] = []
    for row in rows[:20]:
        try:
            insertion_ap = parse_float(get_column(row, lookup, ("Insertion AP", "Original Insertion AP")))
            insertion_ml = parse_float(get_column(row, lookup, ("Insertion ML", "Original Insertion ML")))
        except Exception:
            continue
        sample_values.append((insertion_ap, insertion_ml))

    if not sample_values:
        return "bregma"

    ap_values = np.asarray([value[0] for value in sample_values], dtype=float)
    ml_values = np.asarray([value[1] for value in sample_values], dtype=float)
    if float(np.nanmedian(ap_values)) > 3000.0 or float(np.nanmedian(ml_values)) > 3000.0:
        return "ccf"
    return "bregma"


def load_probe_rows_from_csv(
    csv_path: Path,
    *,
    coordinate_system: str = "auto",
    points_per_probe: int = 80,
) -> list[ProbeRow]:
    """
    Load one probe trajectory per CSV row.

    Supported input columns include:
      - Animal, Animal ID, animal_id, animalID
      - Shank, shank
      - Rotation (degrees), rotation
      - Insertion AP/ML or Insertion AP/ML CCF
      - Terminus AP/ML/DV or Terminus AP/ML/DV CCF

    When CCF columns are present, values are used directly. Otherwise insertion
    and terminus AP/ML are treated as bregma/midline-relative coordinates and are
    converted to Allen CCF microns.
    """

    with open(csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row.")
        rows = list(reader)
        lookup = make_column_lookup(reader.fieldnames)

    animal_values = [
        str(get_column(row, lookup, ("Animal ID", "Animal", "animal_id", "animalID"))).strip()
        for row in rows
    ]
    color_by_animal = assign_colors(animal_values)
    has_ccf = row_has_ccf_columns(lookup)
    inferred_system = (
        "ccf"
        if has_ccf
        else infer_coordinate_system_for_generic_columns(rows, lookup)
    )
    if coordinate_system == "auto":
        coordinate_system = inferred_system
        print(f"interpreting CSV coordinates as {coordinate_system.upper()}.")

    use_ccf = coordinate_system == "ccf"
    use_bregma = coordinate_system == "bregma" or not use_ccf

    out: list[ProbeRow] = []
    for raw in rows:
        animal_id = str(get_column(raw, lookup, ("Animal ID", "Animal", "animal_id", "animalID"))).strip()
        shank = parse_int(get_column(raw, lookup, ("Shank", "shank")))
        rotation = parse_float(
            get_column(raw, lookup, ("Rotation (degrees)", "Rotation", "rotation_degrees"), required=False),
            default=270.0,
        )

        if use_ccf:
            insertion_dv = parse_float(
                get_column(raw, lookup, ("Insertion DV CCF", "Insertion DV"), required=False),
                default=0.0,
            )
            insertion = np.array(
                [
                    parse_float(get_column(raw, lookup, ("Insertion AP CCF", "Insertion AP"))),
                    parse_float(get_column(raw, lookup, ("Insertion ML CCF", "Insertion ML"))),
                    insertion_dv,
                ],
                dtype=float,
            )
            terminus = np.array(
                [
                    parse_float(get_column(raw, lookup, ("Terminus AP CCF", "Terminus AP", "Teminus AP"))),
                    parse_float(get_column(raw, lookup, ("Terminus ML CCF", "Terminus ML"))),
                    parse_float(get_column(raw, lookup, ("Terminus DV CCF", "Terminus DV"))),
                ],
                dtype=float,
            )
        elif use_bregma:
            insertion = relative_ap_ml_dv_to_ccf(
                parse_float(get_column(raw, lookup, ("Insertion AP", "Original Insertion AP"))),
                parse_float(get_column(raw, lookup, ("Insertion ML", "Original Insertion ML"))),
                parse_float(get_column(raw, lookup, ("Insertion DV",), required=False), default=0.0),
            )
            terminus = relative_ap_ml_dv_to_ccf(
                parse_float(get_column(raw, lookup, ("Terminus AP", "Teminus AP", "Original Terminus AP"))),
                parse_float(get_column(raw, lookup, ("Terminus ML", "Original Terminus ML"))),
                parse_float(get_column(raw, lookup, ("Terminus DV",))),
            )
        else:
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

        ccf = np.linspace(insertion, terminus, int(points_per_probe))
        out.append(
            ProbeRow(
                animal_id=animal_id,
                shank=shank,
                rotation_deg=rotation,
                ccf=ccf,
                color=color_by_animal[animal_id],
                source=str(csv_path),
                regions=[f"{animal_id} shank {shank}"] * len(ccf),
                input_kind="csv",
                raw_csv_row=raw,
            )
        )

    return out


def is_sidecar_pkl(stem: str) -> bool:
    """Skip helper pickle files that are not actual probe exports."""

    lowered = stem.lower()
    return "depth" in lowered or "ml_angle" in lowered


def infer_voxel_size_um(sites_vox: list[np.ndarray]) -> int:
    """Infer whether HERBS voxel coordinates are in a 10 um or 25 um grid."""

    all_sites = np.concatenate([np.asarray(sites, dtype=float) for sites in sites_vox], axis=0)
    max_ap = float(all_sites[:, 1].max())
    max_ml = float(all_sites[:, 0].max())
    return 10 if max_ap > 800 or max_ml > 700 else 25


def load_probe_pkl(
    pkl_path: Path,
    *,
    flip_ml: bool,
    flip_ap: bool,
    flip_dv: bool,
    voxel_um: int | str,
) -> list[dict[str, object]]:
    """
    Load one HERBS pickle.

    Returns a list with one entry per shank. Each entry has:
      ccf: N x 3 AP/ML/DV coordinates in microns
      regions: one region acronym per site, when available in the pickle
    """

    with open(pkl_path, "rb") as stream:
        data = pickle.load(stream)["data"]

    sites_vox = data["sites_vox"]
    region_sites = [int(count) for count in data.get("region_sites", [])]
    label_acronym = data.get("label_acronym", [])

    labels: list[str] = []
    for acronym, count in zip(label_acronym, region_sites):
        labels.extend([str(acronym).strip()] * count)

    grid_um = infer_voxel_size_um(sites_vox) if voxel_um == "auto" else int(voxel_um)
    ap_max = int(AP_UM // grid_um) - 1
    dv_max = int(DV_UM // grid_um) - 1
    ml_max = int(ML_UM // grid_um) - 1

    shanks: list[dict[str, object]] = []
    label_offset = 0
    for raw_sites in sites_vox:
        sites = np.asarray(raw_sites, dtype=float)
        n_sites = len(sites)

        # HERBS stores columns as ML-like, AP-like, DV-like voxel indices.
        # The flips convert voxel indices into Allen CCF physical coordinates.
        ml_um = (ml_max - sites[:, 0]) * grid_um if flip_ml else sites[:, 0] * grid_um
        ap_um = (ap_max - sites[:, 1]) * grid_um if flip_ap else sites[:, 1] * grid_um
        dv_um = (dv_max - sites[:, 2]) * grid_um if flip_dv else sites[:, 2] * grid_um
        ccf = np.column_stack([ap_um, ml_um, dv_um])

        shank_labels = labels[label_offset : label_offset + n_sites]
        if len(shank_labels) != n_sites:
            shank_labels = [""] * n_sites
        shanks.append({"ccf": ccf, "regions": shank_labels})
        label_offset += n_sites

    return shanks


def split_probe_filename(stem: str) -> tuple[str, Optional[int]]:
    """
    Infer (animal_id, shank_number) from a probe pickle filename.

    Preferred naming is 'animalID_probeNumber.pkl' or 'animalID_probe_01.pkl'.
    For backwards compatibility, 'animalID_01.pkl' is also accepted.
    If no trailing shank number is found, the whole stem is treated as the animal
    ID and every shank inside the pickle is plotted.
    """

    match = re.match(r"(.+?)_(?:probe[_-]?)?0*([1-9]\d*)$", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    return stem, None


def discover_probe_pkls(probe_dir: Path) -> list[Path]:
    """Return candidate probe pickle files directly inside one folder."""

    return sorted(path for path in probe_dir.glob("*.pkl") if not is_sidecar_pkl(path.stem))


def load_probe_rows_from_pkls(
    probe_dir: Path,
    *,
    flip_ml: bool,
    flip_ap: bool,
    flip_dv: bool,
    voxel_um: int | str,
) -> list[ProbeRow]:
    """Load probe rows from every HERBS .pkl file in a folder."""

    pkl_paths = discover_probe_pkls(probe_dir)
    if not pkl_paths:
        raise ValueError(f"No probe .pkl files found in {probe_dir}")

    animal_ids = [split_probe_filename(path.stem)[0] for path in pkl_paths]
    color_by_animal = assign_colors(animal_ids)

    out: list[ProbeRow] = []
    for path in pkl_paths:
        animal_id, requested_shank = split_probe_filename(path.stem)
        shanks = load_probe_pkl(
            path,
            flip_ml=flip_ml,
            flip_ap=flip_ap,
            flip_dv=flip_dv,
            voxel_um=voxel_um,
        )

        if requested_shank is not None:
            shank_index = requested_shank - 1
            if len(shanks) == 1:
                selected = [(requested_shank, shanks[0])]
            elif 0 <= shank_index < len(shanks):
                selected = [(requested_shank, shanks[shank_index])]
            else:
                print(
                    f"warning: {path.name} names shank {requested_shank}, "
                    f"but contains {len(shanks)} shank trajectory/trajectories; using the first trajectory."
                )
                selected = [(requested_shank, shanks[0])]
        else:
            selected = [(index + 1, shank) for index, shank in enumerate(shanks)]

        for shank_number, shank_data in selected:
            ccf = np.asarray(shank_data["ccf"], dtype=float)
            out.append(
                ProbeRow(
                    animal_id=animal_id,
                    shank=int(shank_number),
                    ccf=ccf,
                    regions=list(shank_data.get("regions", [])),
                    color=color_by_animal[animal_id],
                    source=str(path),
                    input_kind="pkl",
                )
            )

    return deduplicate_probe_rows(out)


def resample_ccf_to_n(ccf: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample an AP/ML/DV trajectory to a requested number of points."""

    ccf = np.asarray(ccf, dtype=float)
    if len(ccf) == n_out:
        return ccf.copy()
    t_in = np.linspace(0.0, 1.0, len(ccf))
    t_out = np.linspace(0.0, 1.0, int(n_out))
    out = np.empty((int(n_out), 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.interp(t_out, t_in, ccf[:, axis])
    return out


def resample_labels_to_n(labels: Optional[list[str]], n_out: int) -> list[str]:
    """Nearest-neighbor resample region labels to match a resampled trajectory."""

    if not labels:
        return [""] * int(n_out)
    if len(labels) == n_out:
        return list(labels)
    indices = np.rint(np.linspace(0, len(labels) - 1, int(n_out))).astype(int)
    return [labels[index] for index in indices]


def deduplicate_probe_rows(rows: list[ProbeRow]) -> list[ProbeRow]:
    """Keep one row per animal/shank, preferring the first sorted source path."""

    best: dict[tuple[str, int], ProbeRow] = {}
    for row in sorted(rows, key=lambda item: (item.animal_id, item.shank, item.source)):
        best.setdefault((row.animal_id, row.shank), row)
    removed = len(rows) - len(best)
    if removed:
        print(f"removed {removed} duplicate probe row(s).")
    return list(best.values())


def interpolate_shanks_2_and_3_from_1_and_4(
    rows: list[ProbeRow],
    *,
    animal_ids: Optional[set[str]] = None,
    replace_existing: bool = True,
) -> list[ProbeRow]:
    """
    Optionally synthesize shanks 2 and 3 from measured shanks 1 and 4.

    For each selected animal:
      shank 2 = 1/3 of the way from shank 1 to shank 4 in AP, ML, and DV
      shank 3 = 2/3 of the way from shank 1 to shank 4 in AP, ML, and DV

    This is the only interpolation rule in the script. It works for any animal
    ID as long as shanks 1 and 4 are present.
    """

    by_key = {(row.animal_id, row.shank): row for row in rows}
    selected_animals = animal_ids if animal_ids is not None else {row.animal_id for row in rows}

    out: list[ProbeRow] = [
        row
        for row in rows
        if not (
            replace_existing
            and row.animal_id in selected_animals
            and row.shank in (2, 3)
            and (row.animal_id, 1) in by_key
            and (row.animal_id, 4) in by_key
        )
    ]

    for animal_id in sorted(selected_animals):
        shank_1 = by_key.get((animal_id, 1))
        shank_4 = by_key.get((animal_id, 4))
        if shank_1 is None or shank_4 is None:
            if any(row.animal_id == animal_id for row in rows):
                found = sorted(row.shank for row in rows if row.animal_id == animal_id)
                print(f"interpolation skipped for {animal_id}: need shanks 1 and 4, found {found}.")
            continue

        n_target = max(len(shank_1.ccf), len(shank_4.ccf))
        ccf_1 = resample_ccf_to_n(shank_1.ccf, n_target)
        ccf_4 = resample_ccf_to_n(shank_4.ccf, n_target)
        labels = resample_labels_to_n(shank_1.regions, n_target)

        for shank_number, fraction_from_1 in ((2, 1.0 / 3.0), (3, 2.0 / 3.0)):
            ccf = (1.0 - fraction_from_1) * ccf_1 + fraction_from_1 * ccf_4
            out.append(
                ProbeRow(
                    animal_id=animal_id,
                    shank=shank_number,
                    ccf=ccf,
                    regions=labels,
                    color=shank_1.color,
                    source=f"{Path(shank_1.source).name} + {Path(shank_4.source).name}",
                    rotation_deg=shank_1.rotation_deg,
                    input_kind=shank_1.input_kind,
                    interpolation_note=(
                        f"interpolated at {fraction_from_1:.3f} between shanks 1 and 4 "
                        f"({Path(shank_1.source).name}, {Path(shank_4.source).name})"
                    ),
                )
            )
        print(f"interpolated shanks 2 and 3 for {animal_id}.")

    return deduplicate_probe_rows(out)


def atlas_resolution_um(atlas: BrainGlobeAtlas) -> tuple[float, float, float]:
    """Return atlas AP/DV/ML resolution in microns."""

    resolution = atlas.resolution
    if isinstance(resolution, (tuple, list, np.ndarray)):
        return float(resolution[0]), float(resolution[1]), float(resolution[2])
    value = float(resolution)
    return value, value, value


def dorsal_surface_dv_um(atlas: BrainGlobeAtlas, ap_um: float, ml_um: float) -> Optional[float]:
    """
    Return the dorsal-most non-background atlas voxel at one AP/ML coordinate.

    Allen CCF DV increases ventrally, so the dorsal surface is the first
    non-background voxel along the DV axis.
    """

    annotation = atlas.annotation
    ap_res, dv_res, ml_res = atlas_resolution_um(atlas)
    ap_index = int(np.clip(round(ap_um / ap_res), 0, annotation.shape[0] - 1))
    ml_index = int(np.clip(round(ml_um / ml_res), 0, annotation.shape[2] - 1))

    def surface_index(a_index: int, m_index: int) -> Optional[int]:
        hits = np.flatnonzero(annotation[a_index, :, m_index] > 0)
        return int(hits[0]) if len(hits) else None

    hit = surface_index(ap_index, ml_index)
    if hit is not None:
        return hit * dv_res

    # Rounding can put the exact column just outside the brain. Search nearby
    # AP/ML columns and use the dorsal-most hit.
    candidates: list[int] = []
    for radius in range(1, 9):
        for da in range(-radius, radius + 1):
            for dm in range(-radius, radius + 1):
                if abs(da) != radius and abs(dm) != radius:
                    continue
                a_index = int(np.clip(ap_index + da, 0, annotation.shape[0] - 1))
                m_index = int(np.clip(ml_index + dm, 0, annotation.shape[2] - 1))
                nearby_hit = surface_index(a_index, m_index)
                if nearby_hit is not None:
                    candidates.append(nearby_hit)
        if candidates:
            return min(candidates) * dv_res
    return None


def align_csv_insertions_to_atlas_surface(probe_rows: list[ProbeRow], atlas: BrainGlobeAtlas) -> None:
    """Move CSV probe insertion points to the atlas surface while preserving direction."""

    for row in probe_rows:
        if row.input_kind != "csv":
            continue
        ccf = np.asarray(row.ccf, dtype=float)
        insertion = ccf[0].copy()
        terminus = ccf[-1].copy()
        surface_dv = dorsal_surface_dv_um(atlas, insertion[0], insertion[1])
        if surface_dv is None:
            print(
                f"warning: no atlas surface found for {row.animal_id} shank {row.shank}; "
                f"keeping insertion DV={insertion[2]:.1f}."
            )
            continue
        dv_offset = surface_dv - insertion[2]
        insertion[2] = surface_dv
        terminus[2] += dv_offset
        row.ccf = np.linspace(insertion, terminus, len(ccf))


def rotation_face_normal_ap_ml(rotation_deg: float) -> np.ndarray:
    """
    Convert spreadsheet rotation into an AP/ML face-normal vector.

    Convention from the original spreadsheet: 0=lateral, 90=anterior,
    180=medial, 270=posterior.
    """

    theta = np.deg2rad(rotation_deg + 90.0)
    return np.array([-np.sin(theta), -np.cos(theta), 0.0], dtype=float)


def probe_frame(ccf: np.ndarray, rotation_deg: float = 270.0) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Return probe long-axis, width-axis, thickness-axis, and length."""

    ccf = np.asarray(ccf, dtype=float)
    start = ccf[0]
    end = ccf[-1]
    axis = end - start
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        return None
    axis = axis / axis_norm

    thick_ref = rotation_face_normal_ap_ml(rotation_deg)
    thick_vec = thick_ref - np.dot(thick_ref, axis) * axis
    thick_norm = float(np.linalg.norm(thick_vec))
    if thick_norm < 1e-6:
        thick_ref = np.array([0.0, 1.0, 0.0])
        thick_vec = thick_ref - np.dot(thick_ref, axis) * axis
        thick_norm = float(np.linalg.norm(thick_vec))
    thick_vec = thick_vec / thick_norm

    width_vec = np.cross(thick_vec, axis)
    width_vec = width_vec / float(np.linalg.norm(width_vec))
    return axis, width_vec, thick_vec, axis_norm


def probe_prism_mesh(ccf: np.ndarray, *, rotation_deg: float = 270.0) -> Optional[dict[str, list[float]]]:
    """Build a simple 3D Neuropixels-style shank mesh around a centerline."""

    ccf = np.asarray(ccf, dtype=float)
    frame = probe_frame(ccf, rotation_deg=rotation_deg)
    if frame is None:
        return None
    axis, width_vec, thick_vec, axis_len = frame
    start = ccf[0]
    end = ccf[-1]
    half_width = 0.5 * SHANK_WIDTH_UM * width_vec
    half_thick = 0.5 * SHANK_THICKNESS_UM * thick_vec
    tip_length = min(SHANK_TIP_LENGTH_UM, axis_len * 0.8)
    tip_base = end - axis * tip_length

    vertices = np.array(
        [
            start - half_width - half_thick,
            start + half_width - half_thick,
            start + half_width + half_thick,
            start - half_width + half_thick,
            tip_base - half_width - half_thick,
            tip_base + half_width - half_thick,
            tip_base + half_width + half_thick,
            tip_base - half_width + half_thick,
            end - half_thick,
            end + half_thick,
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
            [4, 8, 5],
            [5, 8, 9],
            [5, 9, 6],
            [6, 9, 7],
            [7, 9, 8],
            [7, 8, 4],
        ],
        dtype=int,
    )
    return {
        "x": vertices[:, 1].tolist(),
        "y": vertices[:, 0].tolist(),
        "z": vertices[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
    }


def load_atlas_meshes(
    atlas: BrainGlobeAtlas,
    regions: Iterable[RegionSpec],
    *,
    rotate_atlas_180: bool,
) -> list[dict[str, object]]:
    """Load BrainGlobe meshes and convert vertices to Plotly x=ML, y=AP, z=DV."""

    meshes: list[dict[str, object]] = []
    for region in regions:
        try:
            mesh = atlas.mesh_from_structure(region.acronym)
        except Exception as exc:
            print(f"skipping mesh {region.acronym!r}: {exc}")
            continue

        triangles = None
        for block in mesh.cells:
            if block.type in ("triangle", "tri"):
                triangles = block.data
                break
        if triangles is None:
            for cell_name, cell_data in mesh.cells_dict.items():
                if "tri" in cell_name.lower():
                    triangles = cell_data
                    break
        if triangles is None:
            print(f"skipping mesh {region.acronym!r}: no triangle cells found.")
            continue

        vertices = mesh.points
        x = vertices[:, 2]
        y = vertices[:, 0]
        z = vertices[:, 1]
        if rotate_atlas_180:
            x = (ML_UM - 10.0) - x
            y = (AP_UM - 10.0) - y

        meshes.append(
            {
                "name": region.label,
                "color": region.color,
                "opacity": region.opacity,
                "x": x.tolist(),
                "y": y.tolist(),
                "z": z.tolist(),
                "i": triangles[:, 0].tolist(),
                "j": triangles[:, 1].tolist(),
                "k": triangles[:, 2].tolist(),
            }
        )
        print(f"loaded mesh {region.acronym} ({region.label}): {len(vertices):,} vertices.")
    return meshes


def infer_animal_id_from_recording(recording_name: str, known_animal_ids: Iterable[str]) -> str:
    """
    Match a recording name to a known animal ID.

    The safest rule is longest known animal-ID prefix. This handles names such as
    'AnimalA_20260401' without assuming any specific prefix like NVnpg. If no
    known prefix matches, fall back to the text before the first underscore.
    """

    clean_name = str(recording_name).strip()
    for animal_id in sorted(set(known_animal_ids), key=len, reverse=True):
        if clean_name == animal_id or clean_name.startswith(f"{animal_id}_") or clean_name.startswith(f"{animal_id}-"):
            return animal_id
    return clean_name.split("_", 1)[0]


def load_feature_points_csv(
    feature_csv_path: Path,
    *,
    feature_column: str,
    known_animal_ids: Iterable[str],
    include_unused: bool,
    shank_pitch_um: float,
) -> list[FeaturePoint]:
    """
    Load unit/effect rows from a table.

    Required columns are:
      recording name: rec_name, recording_name, recording
      shank: tt_idx, shank
      depth: depth
      x position: x_position, x_pos
      effect: selected with --feature-column

    If a 'use' column exists, rows with use != 1 are skipped unless
    --include-unused-units is set.
    """

    with open(feature_csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{feature_csv_path} has no header row.")
        rows = list(reader)
        lookup = make_column_lookup(reader.fieldnames)

    if normalize_column_name(feature_column) not in lookup:
        raise ValueError(f"Feature CSV is missing requested effect column {feature_column!r}.")

    out: list[FeaturePoint] = []
    skipped_unused = 0
    skipped_nan = 0
    for raw in rows:
        if not include_unused:
            use_value = get_column(raw, lookup, ("use", "include"), required=False)
            if use_value not in (None, "") and parse_int(use_value) != 1:
                skipped_unused += 1
                continue

        recording_name = str(get_column(raw, lookup, ("rec_name", "recording_name", "recording", "session"))).strip()
        shank = parse_int(get_column(raw, lookup, ("tt_idx", "shank")))
        effect = parse_float(get_column(raw, lookup, (feature_column,)), default=float("nan"))
        if not np.isfinite(effect):
            skipped_nan += 1
            continue

        x_position = parse_float(get_column(raw, lookup, ("x_position", "x_pos", "x")))
        x_local = x_position - (shank - 1) * shank_pitch_um
        out.append(
            FeaturePoint(
                animal_id=infer_animal_id_from_recording(recording_name, known_animal_ids),
                recording_name=recording_name,
                shank=shank,
                depth_um_from_surface=parse_float(get_column(raw, lookup, ("depth", "depth_um"))),
                x_um_local=x_local,
                effect_value=effect,
                effect_name=feature_column,
                source_row=raw,
            )
        )

    if skipped_unused:
        print(f"skipped {skipped_unused} unit row(s) with use != 1.")
    if skipped_nan:
        print(f"skipped {skipped_nan} unit row(s) with missing/non-finite {feature_column}.")
    return out


def project_feature_points_to_ccf(probe_rows: list[ProbeRow], feature_points: list[FeaturePoint]) -> list[dict[str, object]]:
    """Project feature points from probe-local coordinates to CCF AP/ML/DV."""

    probe_by_key = {(row.animal_id, int(row.shank)): row for row in probe_rows}
    projected: list[dict[str, object]] = []
    skipped = 0

    for point in feature_points:
        probe = probe_by_key.get((point.animal_id, int(point.shank)))
        if probe is None:
            skipped += 1
            continue

        frame = probe_frame(probe.ccf, rotation_deg=probe.rotation_deg)
        if frame is None:
            skipped += 1
            continue
        axis, width_vec, thick_vec, axis_len = frame

        depth = float(np.clip(point.depth_um_from_surface, 0.0, axis_len))
        x_offset = point.x_um_local - ELECTRODE_COLUMN_CENTER_UM
        surface = np.asarray(probe.ccf[0], dtype=float)
        ccf = (
            surface
            + axis * depth
            + width_vec * x_offset
            + thick_vec * (SHANK_THICKNESS_UM / 2.0 + 2.0)
        )

        projected.append(
            {
                "animal_id": point.animal_id,
                "recording_name": point.recording_name,
                "shank": point.shank,
                "depth_um_from_surface": point.depth_um_from_surface,
                "x_um_local": point.x_um_local,
                "effect_value": point.effect_value,
                "effect_name": point.effect_name,
                "ccf_ap": float(ccf[0]),
                "ccf_ml": float(ccf[1]),
                "ccf_dv": float(ccf[2]),
            }
        )

    if skipped:
        print(f"skipped {skipped} feature point(s) without matching probe geometry.")
    return projected


def transformed_effect_value(value: float, transform: str) -> float:
    """Transform effect values for color mapping."""

    if transform == "none":
        return float(value)
    if transform == "log2-ratio":
        return float(np.log2(value)) if value > 0.0 else float("nan")
    raise ValueError(f"Unknown feature transform: {transform}")


def add_feature_point_traces(
    traces: list[go.BaseTraceType],
    probe_rows: list[ProbeRow],
    feature_points: list[FeaturePoint],
    *,
    feature_transform: str,
    feature_min: Optional[float],
    feature_max: Optional[float],
) -> None:
    """Add projected unit/effect points as colored 3D markers."""

    projected = []
    for row in project_feature_points_to_ccf(probe_rows, feature_points):
        raw_value = float(row["effect_value"])
        if feature_min is not None and raw_value < feature_min:
            continue
        if feature_max is not None and raw_value > feature_max:
            continue
        color_value = transformed_effect_value(raw_value, feature_transform)
        if not np.isfinite(color_value):
            continue
        row["color_value"] = color_value
        projected.append(row)

    if not projected:
        print("no feature points remained after projection/filtering.")
        return

    values = np.asarray([float(row["color_value"]) for row in projected], dtype=float)
    if np.nanmin(values) < 0.0 < np.nanmax(values):
        cmin = -max(abs(float(np.nanpercentile(values, 2))), abs(float(np.nanpercentile(values, 98))))
        cmax = -cmin
        colorscale = [
            [0.0, "#08306b"],
            [0.25, "#4292c6"],
            [0.5, "#f7f7f7"],
            [0.75, "#fb6a4a"],
            [1.0, "#99000d"],
        ]
        cmid = 0.0
    else:
        cmin = float(np.nanpercentile(values, 2))
        cmax = float(np.nanpercentile(values, 98))
        if cmin == cmax:
            cmin -= 1.0
            cmax += 1.0
        colorscale = "Viridis"
        cmid = None

    points_by_probe: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in projected:
        key = (str(row["animal_id"]), int(row["shank"]))
        points_by_probe.setdefault(key, []).append(row)

    first_colorbar = True
    for (animal_id, shank), rows in sorted(points_by_probe.items()):
        marker: dict[str, object] = {
            "size": 5,
            "color": [float(row["color_value"]) for row in rows],
            "cmin": cmin,
            "cmax": cmax,
            "colorscale": colorscale,
            "opacity": 0.95,
            "line": {"color": "rgba(0,0,0,0.35)", "width": 0.5},
        }
        if cmid is not None:
            marker["cmid"] = cmid
        if first_colorbar:
            title = rows[0]["effect_name"]
            if feature_transform == "log2-ratio":
                title = f"log2({title})"
            marker["colorbar"] = {"title": title, "x": 1.18, "len": 0.72, "thickness": 18}

        hover = [
            f"{row['recording_name']}<br>"
            f"animal ID: {animal_id}<br>"
            f"shank: {shank}<br>"
            f"depth: {float(row['depth_um_from_surface']):.1f} um<br>"
            f"x position local: {float(row['x_um_local']):.1f} um<br>"
            f"CCF AP/ML/DV: {float(row['ccf_ap']):.1f}, {float(row['ccf_ml']):.1f}, {float(row['ccf_dv']):.1f}<br>"
            f"{row['effect_name']}: {float(row['effect_value']):.4g}"
            for row in rows
        ]
        traces.append(
            go.Scatter3d(
                x=[float(row["ccf_ml"]) for row in rows],
                y=[float(row["ccf_ap"]) for row in rows],
                z=[float(row["ccf_dv"]) for row in rows],
                mode="markers",
                name=f"{animal_id} shank {shank} feature points",
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker=marker,
                showlegend=False,
            )
        )
        first_colorbar = False


def add_probe_trace(traces: list[go.BaseTraceType], row: ProbeRow, *, probe_style: str, fade_for_features: bool) -> None:
    """Add one probe row as a mesh, line, or both."""

    ccf = np.asarray(row.ccf, dtype=float)
    name = f"{row.animal_id} (shank {row.shank})"
    note = f"<br>{row.interpolation_note}" if row.interpolation_note else ""

    if probe_style in ("mesh", "both"):
        mesh = probe_prism_mesh(ccf, rotation_deg=row.rotation_deg)
        if mesh is not None:
            traces.append(
                go.Mesh3d(
                    x=mesh["x"],
                    y=mesh["y"],
                    z=mesh["z"],
                    i=mesh["i"],
                    j=mesh["j"],
                    k=mesh["k"],
                    name=name,
                    text=(
                        f"{name}{note}<br>"
                        f"source: {Path(row.source).name}<br>"
                        f"rotation: {row.rotation_deg:.1f} deg<br>"
                        f"insertion AP/ML/DV: {ccf[0, 0]:.1f}, {ccf[0, 1]:.1f}, {ccf[0, 2]:.1f}<br>"
                        f"terminus AP/ML/DV: {ccf[-1, 0]:.1f}, {ccf[-1, 1]:.1f}, {ccf[-1, 2]:.1f}"
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    color="#d0d0d0" if fade_for_features else row.color,
                    opacity=0.18 if fade_for_features else 0.90,
                    showlegend=False if fade_for_features else True,
                    flatshading=False,
                    lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.15},
                )
            )

    if probe_style in ("line", "both"):
        regions = row.regions if row.regions and len(row.regions) == len(ccf) else [""] * len(ccf)
        hover = [
            f"{name}{note}<br>source: {Path(row.source).name}<br>region: {region}"
            for region in regions
        ]
        traces.append(
            go.Scatter3d(
                x=ccf[:, 1].tolist(),
                y=ccf[:, 0].tolist(),
                z=ccf[:, 2].tolist(),
                mode="lines+markers",
                name=name,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                line={"color": row.color, "width": 2},
                marker={"size": 1.8, "color": row.color, "opacity": 0.9},
                showlegend=True,
            )
        )


def build_figure(
    probe_rows: list[ProbeRow],
    *,
    atlas_name: str,
    regions: list[RegionSpec],
    rotate_atlas_180: bool,
    align_surface: bool,
    probe_style: str,
    feature_points: Optional[list[FeaturePoint]],
    feature_transform: str,
    feature_min: Optional[float],
    feature_max: Optional[float],
) -> go.Figure:
    """Build the full Plotly figure."""

    print(f"loading atlas {atlas_name!r}...")
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)

    if align_surface:
        align_csv_insertions_to_atlas_surface(probe_rows, atlas)

    print("loading brain meshes...")
    meshes = load_atlas_meshes(atlas, regions, rotate_atlas_180=rotate_atlas_180)

    traces: list[go.BaseTraceType] = []
    for mesh in meshes:
        traces.append(
            go.Mesh3d(
                x=mesh["x"],
                y=mesh["y"],
                z=mesh["z"],
                i=mesh["i"],
                j=mesh["j"],
                k=mesh["k"],
                color=mesh["color"],
                opacity=mesh["opacity"],
                name=mesh["name"],
                showlegend=True,
                hoverinfo="name",
                flatshading=False,
                lighting={"ambient": 0.6, "diffuse": 0.7, "specular": 0.1},
            )
        )

    fade_for_features = bool(feature_points)
    for row in sorted(probe_rows, key=lambda item: (item.animal_id, item.shank)):
        add_probe_trace(traces, row, probe_style=probe_style, fade_for_features=fade_for_features)

    if feature_points:
        add_feature_point_traces(
            traces,
            probe_rows,
            feature_points,
            feature_transform=feature_transform,
            feature_min=feature_min,
            feature_max=feature_max,
        )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            title="Probe Visualization - Allen CCF",
            template="plotly_white",
            height=760,
            margin={"r": 230},
            legend={"x": 1.02, "y": 0.98, "bgcolor": "rgba(255,255,255,0.85)"},
            scene={
                "xaxis": {"title": "ML (um)", "gridcolor": "#e0e0e0"},
                "yaxis": {"title": "AP (um)", "gridcolor": "#e0e0e0"},
                "zaxis": {"title": "DV (um)", "gridcolor": "#e0e0e0", "autorange": "reversed"},
                "aspectmode": "data",
                "bgcolor": "rgba(248,248,255,1)",
                "camera": {
                    "up": {"x": 0, "y": 0, "z": 1},
                    "center": {"x": 0, "y": 0, "z": 0},
                    "eye": {"x": -1.2, "y": -1.2, "z": 1.2},
                },
            },
        ),
    )


def filter_probe_rows(
    rows: list[ProbeRow],
    *,
    animal_ids: Optional[set[str]],
    shanks: Optional[set[int]],
) -> list[ProbeRow]:
    """Apply optional animal and shank filters."""

    out = []
    for row in rows:
        if animal_ids is not None and row.animal_id not in animal_ids:
            continue
        if shanks is not None and row.shank not in shanks:
            continue
        out.append(row)
    return out


def parse_csv_set(text: Optional[str]) -> Optional[set[str]]:
    """Parse comma-separated text into a set of strings."""

    if not text:
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_int_set(text: Optional[str]) -> Optional[set[int]]:
    """Parse comma-separated text into a set of integers."""

    if not text:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def resolve_output_path(input_path: Path, cli_output: Optional[Path], *, with_features: bool) -> Path:
    """
    Choose where to save the HTML output.

    Priority order:
      1. --output from the command line
      2. DEFAULT_OUTPUT_HTML edited near the top of this file
      3. Automatic name next to the input file/folder
    """

    if cli_output is not None:
        return cli_output.expanduser().resolve()

    if DEFAULT_OUTPUT_HTML is not None:
        return Path(DEFAULT_OUTPUT_HTML).expanduser().resolve()

    suffix = "_with_features" if with_features else ""
    return input_path.with_name(f"probe_visualization{suffix}.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--csv-path", type=Path, default=None, help="Probe coordinate CSV path.")
    input_group.add_argument("--probe-dir", type=Path, default=None, help="Folder containing probe .pkl files.")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Overrides DEFAULT_OUTPUT_HTML in the script.",
    )
    parser.add_argument("--atlas", default="allen_mouse_25um", help="BrainGlobe atlas name.")
    parser.add_argument(
        "--coordinate-system",
        choices=("auto", "ccf", "bregma"),
        default="auto",
        help="CSV coordinate interpretation. Auto uses CCF columns when present.",
    )
    parser.add_argument("--points-per-probe", type=int, default=80, help="Number of plotted points for CSV trajectories.")
    parser.add_argument(
        "--no-align-surface",
        action="store_true",
        help="Do not move CSV insertion DV to the atlas dorsal surface.",
    )

    parser.add_argument(
        "--brain-regions",
        default=None,
        help="Comma-separated atlas acronyms to plot, e.g. root,MOp,SSp-bfd.",
    )
    parser.add_argument(
        "--region",
        action="append",
        help="Repeatable region spec: ACRONYM[:Label[:#RRGGBB[:opacity]]]. Overrides --brain-regions.",
    )
    parser.add_argument(
        "--rotate-atlas-180",
        action="store_true",
        help="Rotate atlas meshes 180 degrees in the ML/AP plane.",
    )

    parser.add_argument("--animal-ids", default=None, help="Optional comma-separated animal IDs to plot.")
    parser.add_argument("--shanks", default=None, help="Optional comma-separated shank numbers to plot.")
    parser.add_argument(
        "--interpolate-shanks",
        action="store_true",
        help="Interpolate shanks 2 and 3 between shanks 1 and 4 for selected animals.",
    )
    parser.add_argument(
        "--interpolate-animal-ids",
        default=None,
        help="Comma-separated animal IDs for interpolation. Default: all loaded animals.",
    )
    parser.add_argument(
        "--keep-existing-interpolated-shanks",
        action="store_true",
        help="Keep measured shanks 2 and 3 instead of replacing them during interpolation.",
    )

    parser.add_argument(
        "--probe-style",
        choices=("auto", "mesh", "line", "both"),
        default="auto",
        help="How to draw probes. Auto uses mesh for CSV input and line for .pkl input.",
    )

    parser.add_argument("--feature-csv", type=Path, default=None, help="Optional unit/effect CSV to overlay.")
    parser.add_argument(
        "--feature-column",
        default="mass_effect",
        help="Effect column in the feature CSV. Other users can set this to any numeric effect.",
    )
    parser.add_argument(
        "--feature-transform",
        choices=("none", "log2-ratio"),
        default="log2-ratio",
        help="Color-transform for feature values. Use 'none' for arbitrary signed effects.",
    )
    parser.add_argument("--feature-min", type=float, default=None, help="Optional raw feature lower bound.")
    parser.add_argument("--feature-max", type=float, default=None, help="Optional raw feature upper bound.")
    parser.add_argument(
        "--include-unused-units",
        action="store_true",
        help="Include rows where use != 1 if the feature CSV has a use column.",
    )
    parser.add_argument(
        "--shank-pitch-um",
        type=float,
        default=SHANK_PITCH_UM,
        help="Spacing used to convert global x_position/x_pos into local shank x position.",
    )

    parser.add_argument("--flip-ml", action=argparse.BooleanOptionalAction, default=True, help="Flip HERBS ML voxel axis.")
    parser.add_argument("--flip-ap", action=argparse.BooleanOptionalAction, default=True, help="Flip HERBS AP voxel axis.")
    parser.add_argument("--flip-dv", action=argparse.BooleanOptionalAction, default=False, help="Flip HERBS DV voxel axis.")
    parser.add_argument("--voxel-um", default="auto", help="HERBS voxel size in um, or 'auto'.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv_path is None and args.probe_dir is None:
        csv_path = default_csv_path()
        if not csv_path.is_file():
            raise SystemExit("Provide --csv-path or --probe-dir. No default CSV was found.")
        args.csv_path = csv_path

    if args.csv_path is not None:
        input_path = args.csv_path.expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"CSV file not found: {input_path}")
        probe_rows = load_probe_rows_from_csv(
            input_path,
            coordinate_system=args.coordinate_system,
            points_per_probe=args.points_per_probe,
        )
        input_kind = "csv"
    else:
        input_path = args.probe_dir.expanduser().resolve()
        if not input_path.is_dir():
            raise SystemExit(f"Probe folder not found: {input_path}")
        probe_rows = load_probe_rows_from_pkls(
            input_path,
            flip_ml=args.flip_ml,
            flip_ap=args.flip_ap,
            flip_dv=args.flip_dv,
            voxel_um=args.voxel_um,
        )
        input_kind = "pkl"

    animal_filter = parse_csv_set(args.animal_ids)
    shank_filter = parse_int_set(args.shanks)
    probe_rows = filter_probe_rows(probe_rows, animal_ids=animal_filter, shanks=shank_filter)
    if not probe_rows:
        raise SystemExit("No probe rows remained after loading/filtering.")

    if args.interpolate_shanks:
        interpolation_animals = parse_csv_set(args.interpolate_animal_ids)
        probe_rows = interpolate_shanks_2_and_3_from_1_and_4(
            probe_rows,
            animal_ids=interpolation_animals,
            replace_existing=not args.keep_existing_interpolated_shanks,
        )

    feature_points = None
    if args.feature_csv is not None:
        feature_csv_path = args.feature_csv.expanduser().resolve()
        if not feature_csv_path.is_file():
            raise SystemExit(f"Feature CSV not found: {feature_csv_path}")
        feature_points = load_feature_points_csv(
            feature_csv_path,
            feature_column=args.feature_column,
            known_animal_ids=[row.animal_id for row in probe_rows],
            include_unused=args.include_unused_units,
            shank_pitch_um=args.shank_pitch_um,
        )
        print(f"loaded {len(feature_points)} feature point(s) from {feature_csv_path.name}.")

    if args.probe_style == "auto":
        probe_style = "mesh" if input_kind == "csv" else "line"
    else:
        probe_style = args.probe_style

    output_path = resolve_output_path(input_path, args.output, with_features=bool(feature_points))

    print(f"loaded {len(probe_rows)} probe row(s).")
    fig = build_figure(
        probe_rows,
        atlas_name=args.atlas,
        regions=build_region_specs(args),
        rotate_atlas_180=args.rotate_atlas_180,
        align_surface=not args.no_align_surface,
        probe_style=probe_style,
        feature_points=feature_points,
        feature_transform=args.feature_transform,
        feature_min=args.feature_min,
        feature_max=args.feature_max,
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
