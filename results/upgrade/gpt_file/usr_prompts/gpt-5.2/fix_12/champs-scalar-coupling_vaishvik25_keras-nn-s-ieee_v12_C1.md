# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the `scalar_coupling_constant` between atom pairs in molecules, given the two atom types (e.g., C and H), the coupling type (e.g., `2JHC`), and any features you are able to create from the molecule structure (`xyz`) files.

## Metric
Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types.

## Submission Format
```
id,scalar_coupling_constant
2324604,0.0
2324605,0.0
2324606,0.0
etc.
```

## Dataset
The training and test splits are by *molecule*, so that no molecule in the training data is found in the test data.

- **train.csv** - the training set, where the first column (`molecule_name`) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (`atom_index_0`) and third column (`atom_index_1`) is the atom indices of the atom-pair creating the coupling and the fourth column (`scalar_coupling_constant`) is the scalar coupling constant that we want to be able to predict
- **test.csv** - the test set; same info as train, without the target variable
- **sample_submission.csv** - a sample submission file in the correct format
- **structures.zip** - folder containing molecular structure (xyz) files, where the first line is the number of atoms in the molecule, followed by a blank line, and then a line for every atom, where the first column contains the atomic element (H for hydrogen, C for carbon etc.) and the remaining columns contain the X, Y and Z cartesian coordinates (a standard format for chemists and molecular visualization programs)
- **structures.csv** - this file contains the **same** information as the individual xyz structure files, but in a single file
- **dipole_moments.csv** - contains the molecular electric dipole moments. These are three dimensional vectors that indicate the charge distribution in the molecule. The first column (`molecule_name`) are the names of the molecule, the second to fourth column are the `X`, `Y` and `Z` components respectively of the dipole moment.
- **magnetic_shielding_tensors.csv** - contains the magnetic shielding tensors for all atoms in the molecules. The first column (`molecule_name`) contains the molecule name, the second column (`atom_index`) contains the index of the atom in the molecule, the third to eleventh columns contain the `XX`, `YX`, `ZX`, `XY`, `YY`, `ZY`, `XZ`, `YZ` and `ZZ` elements of the tensor/matrix respectively.
- **mulliken_charges.csv** - contains the mulliken charges for all atoms in the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`atom_index`) contains the index of the atom in the molecule, the third column (`mulliken_charge`) contains the mulliken charge of the atom.
- **potential_energy.csv** - contains the potential energy of the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`potential_energy`) contains the potential energy of the molecule.
- **scalar_coupling_contributions.csv** - The scalar coupling constants in `train.csv` (or corresponding files) are a sum of four terms. `scalar_coupling_contributions.csv` contain all these terms. The first column (`molecule_name`) are the name of the molecule, the second (`atom_index_0`) and third column (`atom_index_1`) are the atom indices of the atom-pair, the fourth column indicates the type of coupling, the fifth column (`fc`) is the Fermi Contact contribution, the sixth column (`sd`) is the Spin-dipolar contribution, the seventh column (`pso`) is the Paramagnetic spin-orbit contribution and the eighth column (`dso`) is the Diamagnetic spin-orbit contribution.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        input/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        working/
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
```

-> data/champs-scalar-coupling/dipole_moments.csv has 76510 rows and 4 columns.
The columns are: molecule_name, X, Y, Z

-> data/champs-scalar-coupling/magnetic_shielding_tensors.csv has 1379964 rows and 11 columns.
The columns are: molecule_name, atom_index, XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ

-> data/champs-scalar-coupling/mulliken_charges.csv has 1379964 rows and 3 columns.
The columns are: molecule_name, atom_index, mulliken_charge

-> data/champs-scalar-coupling/potential_energy.csv has 76510 rows and 2 columns.
The columns are: molecule_name, potential_energy

-> data/champs-scalar-coupling/sample_submission.csv has 467813 rows and 2 columns.
The columns are: id, scalar_coupling_constant

-> data/champs-scalar-coupling/scalar_coupling_contributions.csv has 4191263 rows and 8 columns.
The columns are: molecule_name, atom_index_0, atom_index_1, type, fc, sd, pso, dso

-> data/champs-scalar-coupling/structures.csv has 1379964 rows and 6 columns.
The columns are: molecule_name, atom_index, atom, x, y, z

-> data/champs-scalar-coupling/test.csv has 467813 rows and 5 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type

-> data/champs-scalar-coupling/train.csv has 4191263 rows and 6 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type, scalar_coupling_constant

-> (stopped after 10 files for performance)

# 5. Target score

-1.476322251698409

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'Your notebook is currently a “stacking/blending of other people’s submissions” script, but none of those external `../input/...` submission folders exist in this environment, so it fails before producing any `.csv`. I replace those missing-input parts with a minimal, fully self-contained baseline that trains from `train.csv` and predicts for `test.csv` using only the provided competition files. To preserve the existing “simple aggregation” core idea, the model be a per-`type` median lookup of `scalar_coupling_constant` (with a global fallback), which is fast, stable, and guaranteed to generate a valid `submission.csv`. I also remove notebook-only magic (`%matplotlib inline`) and deprecated `np.bool` usage to prevent runtime errors under the given package versions.'
- What this solution (achieved 1.18497) has done: 'I fix the root cause of the crash in feature creation: some test rows fail to merge with `structures.csv`, producing NaN coordinates/distances; casting those NaNs to `int32` for `dist_bin` triggers `IntCastingNaNError` and then downstream `dist_bin`/`submission` errors. The minimal safe fix is to (1) enforce consistent dtypes for merge keys in both `structures` and train/test, (2) compute `dist` robustly and allow missing distances, and (3) create `dist_bin` using pandas’ nullable integer (`Int32`) with a sentinel fill value. This preserves the existing hierarchical-median core logic and ensures the notebook completes and writes a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.4763), so we should improve but keep the same “hierarchical median lookup” core logic. The most impactful minimal change is to make the atom-pair features invariant to atom order by canonicalizing `(atom_0, atom_1)` (and also `(atom_index_0, atom_index_1)` for distance) so that lookups match regardless of how the pair is ordered in train vs test. We also reduce accidental sparsity by computing `dist_bin` per coupling `type` (bin width scaled by per-type distance std) while keeping the exact same aggregation/prediction hierarchy (g1→g2→g3→global median). These changes typically increase hit-rate of g1/g2 matches and reduce MAE without changing the modeling approach.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is still far from the target (-1.4763), so we should improve accuracy while keeping the same hierarchical-median lookup core logic. The smallest high-impact change is to enrich the existing `dist_bin` feature without changing the model: add per-`type` quantile bins of `dist` computed from train and applied consistently to test, then insert it into the same fallback hierarchy (g1→g2→g3→global). This reduces sparsity vs fixed-width bins and tends to lower MAE for each coupling type, moving the score toward the target. I also keep the existing atom-order canonicalization and robust merge/NaN handling unchanged.'
- What this solution (achieved 1.18497) has done: 'We need to improve your current baseline (score 1.18497, lower-is-better) toward the target (-1.4763), so we should reduce MAE while keeping the same “hierarchical median lookup” core logic. The smallest high-impact change is to reduce lookup sparsity by adding one extra, slightly coarser quantile bin level (`dist_qbin_coarse`) and inserting it into the same fallback chain between the current fine bin and the atom-pair median. We also speed up and stabilize the lookup (same predictions) by replacing Python list comprehensions with pandas `MultiIndex.reindex`, which avoids per-row Python overhead without changing the model. Everything else (features, medians, and submission format) stays the same.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is much worse than the target (-1.4763), so we should improve accuracy while keeping the exact same “hierarchical median lookup” modeling logic. The smallest high-impact fix is to correct an ordering bug in atom-pair canonicalization: you currently sort `atom_index_0/1` for distance, but you sort `atom_0/1` independently by element symbol, which can mismatch the distance bin with the atom-pair key and reduce g1/g1c hit-rate. I tie atom labels to the same canonicalized atom indices (so `atom_0` always corresponds to `atom_index_0`, etc.) and then use an order-invariant atom-pair key only for grouping/lookup. Everything else (features, quantile bins, fallback chain, and submission format) stays the same and still runs within time.'
- What this solution (achieved 1.18497) has done: 'I fix the crash in `apply_qbins_per_pair_with_fallback` by removing the incorrect `np.argsort` on a Python list of tuples (which returns an array that can’t index a list) and replacing it with a stable pandas-based group iteration that computes per-pair bins without changing the binning logic. This allow `dist_qbin`/`dist_qbin_coarse` to be created reliably so the downstream groupby medians work. I also add a small safety cast when building the output series to keep nullable integer handling consistent and avoid dtype-related surprises. With these minimal fixes, the script run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.4763), so we should improve accuracy while keeping the same hierarchical-median lookup approach. The biggest likely issue is that the model learns/predicts using an order-invariant *element* key (`atom_pair_0/1`) but uses a distance bin computed after sorting indices, which can misalign with how couplings are directional by `atom_index_0/1` and by type; we can reduce error by making the “pair key” respect the same canonical atom ordering used for distance (so atom symbols stay tied to the sorted indices). Then, to reduce sparsity without changing the approach, we add one extra fallback level that uses only `(type, dist_qbin)` (and coarse) medians before falling back to per-type median; this is still the same median-lookup model, just a slightly richer hierarchy. These two minimal changes usually reduce MAE across types (especially where atom-pair bins are sparse) and should move the score toward the target while keeping runtime within limits and preserving submission semantics.'
- What this solution (achieved 1.18497) has done: 'Your current score (1.18497, lower-is-better) is far worse than the target (-1.4763), so we should improve accuracy but keep the exact same “hierarchical median lookup” approach. The smallest likely high-impact issue is that we’re currently discarding directionality by sorting atom indices (and then creating an order-invariant atom-pair key), but many coupling types behave differently depending on which side is H/C/N, so collapsing direction can blur medians and increase MAE. I keep your existing invariant (unordered) lookup chain intact, but add a *directional* lookup level first: `(type, atom_0, atom_1, dist_qbin)` and its coarse variant, then fall back to your existing unordered levels. This adds information without changing the model class (still pure groupby-median + fallback) and should reduce MAE toward the target while preserving runtime and producing the same submission format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_PATH = "/kaggle/data/champs-scalar-coupling"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
STRUCTURES_PATH = os.path.join(BASE_PATH, "structures.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.exists(STRUCTURES_PATH), f"Missing: {STRUCTURES_PATH}"

print("Found files:")
print(" -", TRAIN_PATH)
print(" -", TEST_PATH)
print(" -", SAMPLE_SUB_PATH)
print(" -", STRUCTURES_PATH)



## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

required_train_cols = {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
required_sub_cols = {"id", "scalar_coupling_constant"}

missing_train = required_train_cols - set(train.columns)
missing_test = required_test_cols - set(test.columns)
missing_sub = required_sub_cols - set(sample_sub.columns)

assert not missing_train, f"train.csv missing columns: {missing_train}"
assert not missing_test, f"test.csv missing columns: {missing_test}"
assert not missing_sub, f"sample_submission.csv missing columns: {missing_sub}"

for df in (train, test):
    df["molecule_name"] = df["molecule_name"].astype("string")
    df["atom_index_0"] = pd.to_numeric(df["atom_index_0"], downcast="integer")
    df["atom_index_1"] = pd.to_numeric(df["atom_index_1"], downcast="integer")
    df["type"] = df["type"].astype("string")

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample_submission shape:", sample_sub.shape)
print("train types:", train["type"].nunique(), "unique")



## === cell 2
structures = pd.read_csv(
    STRUCTURES_PATH, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)

structures["molecule_name"] = structures["molecule_name"].astype("string")
structures["atom_index"] = pd.to_numeric(structures["atom_index"], downcast="integer")

structures["x"] = structures["x"].astype(np.float32, copy=False)
structures["y"] = structures["y"].astype(np.float32, copy=False)
structures["z"] = structures["z"].astype(np.float32, copy=False)
structures["atom"] = structures["atom"].astype("string")

structures = structures.sort_values(
    ["molecule_name", "atom_index"], kind="mergesort"
).reset_index(drop=True)


def add_atom_and_distance_features(df, structures_df):
    """
    Change (score improvement, same core logic):
    - Preserve the ORIGINAL (directional) atom_index_0/1 order in additional columns,
      so we can add a directional median lookup level (type, atom_0, atom_1, dist_qbin)
      before the existing order-invariant (atom_pair_0/1) hierarchy.
    - Keep the existing invariant hierarchy unchanged by still computing a canonical
      (sorted) pair for distance/binning and the atom_pair key.
    """
    df = df.copy()
    df["molecule_name"] = df["molecule_name"].astype("string")
    df["atom_index_0"] = pd.to_numeric(df["atom_index_0"], downcast="integer")
    df["atom_index_1"] = pd.to_numeric(df["atom_index_1"], downcast="integer")

    df["atom_index_0_orig"] = df["atom_index_0"]
    df["atom_index_1_orig"] = df["atom_index_1"]

    a0 = df["atom_index_0"].astype("int64")
    a1 = df["atom_index_1"].astype("int64")
    df["atom_index_0"] = np.minimum(a0, a1).astype(df["atom_index_0"].dtype, copy=False)
    df["atom_index_1"] = np.maximum(a0, a1).astype(df["atom_index_1"].dtype, copy=False)

    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0_canon",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left", sort=False)

    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1_canon",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )
    df = df.merge(s1, on=["molecule_name", "atom_index_1"], how="left", sort=False)

    dx = df["x0"].astype("float64") - df["x1"].astype("float64")
    dy = df["y0"].astype("float64") - df["y1"].astype("float64")
    dz = df["z0"].astype("float64") - df["z1"].astype("float64")
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)

    a0s = df["atom_0_canon"].astype("string")
    a1s = df["atom_1_canon"].astype("string")
    df["atom_pair_0"] = a0s.where(a0s <= a1s, a1s)
    df["atom_pair_1"] = a1s.where(a0s <= a1s, a0s)

    s0o = structures_df.rename(
        columns={
            "atom_index": "atom_index_0_orig",
            "atom": "atom_0",
        }
    )[["molecule_name", "atom_index_0_orig", "atom_0"]]
    df = df.merge(
        s0o, on=["molecule_name", "atom_index_0_orig"], how="left", sort=False
    )

    s1o = structures_df.rename(
        columns={
            "atom_index": "atom_index_1_orig",
            "atom": "atom_1",
        }
    )[["molecule_name", "atom_index_1_orig", "atom_1"]]
    df = df.merge(
        s1o, on=["molecule_name", "atom_index_1_orig"], how="left", sort=False
    )

    return df


train_f = add_atom_and_distance_features(train, structures)
test_f = add_atom_and_distance_features(test, structures)

global_bin_width = 0.05
type_dist_std = (
    train_f.groupby("type", observed=True)["dist"]
    .std()
    .replace([0.0, np.inf, -np.inf], np.nan)
)
global_std = float(train_f["dist"].std())
if not np.isfinite(global_std) or global_std <= 0:
    global_std = 1.0

type_bin_width = (global_bin_width * (type_dist_std / global_std)).clip(
    lower=0.02, upper=0.20
)

bw_train = (
    train_f["type"].map(type_bin_width).astype("float64").fillna(global_bin_width)
)
bw_test = test_f["type"].map(type_bin_width).astype("float64").fillna(global_bin_width)

train_f["dist_bin"] = np.floor(train_f["dist"] / bw_train).astype("Int32")
test_f["dist_bin"] = np.floor(test_f["dist"] / bw_test).astype("Int32")

SENTINEL_BIN = -1
train_f["dist_bin"] = train_f["dist_bin"].fillna(SENTINEL_BIN).astype("Int32")
test_f["dist_bin"] = test_f["dist_bin"].fillna(SENTINEL_BIN).astype("Int32")

for col in ["atom_pair_0", "atom_pair_1", "atom_0", "atom_1"]:
    train_f[col] = train_f[col].astype("string")
    test_f[col] = test_f[col].astype("string")

N_QBINS_PAIR = 32
N_QBINS_PAIR_COARSE = 8
MIN_PAIR_SAMPLES = 80
SENTINEL_QBIN = -1

N_QBINS = 64
N_QBINS_COARSE = 16

edges_by_type = {}
edges_by_type_coarse = {}
for t, grp in train_f.groupby("type", observed=True):
    d = grp["dist"].to_numpy(dtype="float64", copy=False)
    d = d[np.isfinite(d)]
    if d.size < 50:
        edges_by_type[t] = None
        edges_by_type_coarse[t] = None
        continue

    qs_f = np.linspace(0.0, 1.0, N_QBINS + 1)
    edges_f = np.quantile(d, qs_f)
    edges_f = np.unique(edges_f)
    edges_by_type[t] = edges_f if edges_f.size > 2 else None

    qs_c = np.linspace(0.0, 1.0, N_QBINS_COARSE + 1)
    edges_c = np.quantile(d, qs_c)
    edges_c = np.unique(edges_c)
    edges_by_type_coarse[t] = edges_c if edges_c.size > 2 else None


def _digitize_with_edges(dist_arr, edges):
    if edges is None:
        return None
    interior = edges[1:-1]
    if interior.size == 0:
        return None
    return np.digitize(dist_arr, interior, right=False).astype(np.int32)


def apply_qbins_per_type(df, edges_map, out_col):
    out = np.full(len(df), SENTINEL_QBIN, dtype=np.int32)
    types = df["type"].astype("string").to_numpy()
    dist = df["dist"].to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(dist)

    for t in pd.unique(types):
        edges = edges_map.get(t, None)
        m = (types == t) & finite
        if not m.any():
            continue
        b = _digitize_with_edges(dist[m], edges)
        if b is None:
            continue
        out[m] = b

    df[out_col] = pd.Series(out, index=df.index, dtype="int32").astype("Int32")
    return df


train_f = apply_qbins_per_type(train_f, edges_by_type, "dist_qbin_type")
test_f = apply_qbins_per_type(test_f, edges_by_type, "dist_qbin_type")
train_f = apply_qbins_per_type(train_f, edges_by_type_coarse, "dist_qbin_coarse_type")
test_f = apply_qbins_per_type(test_f, edges_by_type_coarse, "dist_qbin_coarse_type")


def build_pair_edges(train_df, n_qbins, min_samples):
    edges_map = {}
    grp = train_df.groupby(["type", "atom_pair_0", "atom_pair_1"], observed=True)
    for key, g in grp:
        d = g["dist"].to_numpy(dtype="float64", copy=False)
        d = d[np.isfinite(d)]
        if d.size < min_samples:
            continue
        qs = np.linspace(0.0, 1.0, n_qbins + 1)
        edges = np.quantile(d, qs)
        edges = np.unique(edges)
        if edges.size > 2:
            edges_map[key] = edges
    return edges_map


pair_edges_fine = build_pair_edges(train_f, N_QBINS_PAIR, MIN_PAIR_SAMPLES)
pair_edges_coarse = build_pair_edges(train_f, N_QBINS_PAIR_COARSE, MIN_PAIR_SAMPLES)


def apply_qbins_per_pair_with_fallback(df, pair_edges_map, fallback_col, out_col):
    out = (
        df[fallback_col]
        .astype("Int32")
        .fillna(SENTINEL_QBIN)
        .to_numpy(dtype=np.int32, copy=True)
    )
    dist = df["dist"].to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(dist)

    if len(pair_edges_map) == 0:
        df[out_col] = pd.Series(out, index=df.index, dtype="int32").astype("Int32")
        return df

    keys = list(pair_edges_map.keys())
    key_df = pd.DataFrame(keys, columns=["type", "atom_pair_0", "atom_pair_1"])
    key_df["type"] = key_df["type"].astype("string")
    key_df["atom_pair_0"] = key_df["atom_pair_0"].astype("string")
    key_df["atom_pair_1"] = key_df["atom_pair_1"].astype("string")

    tagged = (
        df[["type", "atom_pair_0", "atom_pair_1"]]
        .merge(
            key_df.assign(_has_pair_edges=1),
            on=["type", "atom_pair_0", "atom_pair_1"],
            how="left",
            sort=False,
        )["_has_pair_edges"]
        .to_numpy()
    )
    cand = (tagged == 1) & finite
    if not cand.any():
        df[out_col] = pd.Series(out, index=df.index, dtype="int32").astype("Int32")
        return df

    cand_df = df.loc[cand, ["type", "atom_pair_0", "atom_pair_1"]].copy()
    cand_df["_rowpos"] = np.flatnonzero(cand).astype(np.int64)

    for (t, ap0, ap1), g in cand_df.groupby(
        ["type", "atom_pair_0", "atom_pair_1"], observed=True, sort=False
    ):
        edges = pair_edges_map.get((t, ap0, ap1), None)
        if edges is None:
            continue
        rowpos = g["_rowpos"].to_numpy(dtype=np.int64, copy=False)
        b = _digitize_with_edges(dist[rowpos], edges)
        if b is None:
            continue
        out[rowpos] = b

    df[out_col] = pd.Series(out, index=df.index, dtype="int32").astype("Int32")
    return df


train_f = apply_qbins_per_pair_with_fallback(
    train_f, pair_edges_fine, "dist_qbin_type", "dist_qbin"
)
test_f = apply_qbins_per_pair_with_fallback(
    test_f, pair_edges_fine, "dist_qbin_type", "dist_qbin"
)
train_f = apply_qbins_per_pair_with_fallback(
    train_f, pair_edges_coarse, "dist_qbin_coarse_type", "dist_qbin_coarse"
)
test_f = apply_qbins_per_pair_with_fallback(
    test_f, pair_edges_coarse, "dist_qbin_coarse_type", "dist_qbin_coarse"
)

print(
    train_f[
        [
            "type",
            "atom_0",
            "atom_1",
            "atom_pair_0",
            "atom_pair_1",
            "dist",
            "dist_bin",
            "dist_qbin",
            "dist_qbin_coarse",
        ]
    ].head()
)

missing_train_atoms = (
    train_f["atom_pair_0"].isna().any()
    or train_f["atom_pair_1"].isna().any()
    or train_f["atom_0"].isna().any()
    or train_f["atom_1"].isna().any()
)
missing_test_atoms = (
    test_f["atom_pair_0"].isna().any()
    or test_f["atom_pair_1"].isna().any()
    or test_f["atom_0"].isna().any()
    or test_f["atom_1"].isna().any()
)
print("Any missing atoms in train?", bool(missing_train_atoms))
print("Any missing atoms in test?", bool(missing_test_atoms))
print("Missing dist in train:", int(train_f["dist"].isna().sum()))
print("Missing dist in test:", int(test_f["dist"].isna().sum()))



## === cell 3
global_median = float(train_f["scalar_coupling_constant"].median())

g0 = train_f.groupby(["type", "atom_0", "atom_1", "dist_qbin"], observed=True)[
    "scalar_coupling_constant"
].median()

g0c = train_f.groupby(["type", "atom_0", "atom_1", "dist_qbin_coarse"], observed=True)[
    "scalar_coupling_constant"
].median()

g1 = train_f.groupby(
    ["type", "atom_pair_0", "atom_pair_1", "dist_qbin"], observed=True
)["scalar_coupling_constant"].median()

g1c = train_f.groupby(
    ["type", "atom_pair_0", "atom_pair_1", "dist_qbin_coarse"], observed=True
)["scalar_coupling_constant"].median()

g2 = train_f.groupby(["type", "atom_pair_0", "atom_pair_1"], observed=True)[
    "scalar_coupling_constant"
].median()

g2d = train_f.groupby(["type", "dist_qbin"], observed=True)[
    "scalar_coupling_constant"
].median()
g2dc = train_f.groupby(["type", "dist_qbin_coarse"], observed=True)[
    "scalar_coupling_constant"
].median()

g3 = train_f.groupby(["type"], observed=True)["scalar_coupling_constant"].median()


def hierarchical_predict(df):
    idx0 = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["atom_0"].astype(str).to_numpy(),
            df["atom_1"].astype(str).to_numpy(),
            df["dist_qbin"].astype("int32").to_numpy(),
        ],
        names=["type", "atom_0", "atom_1", "dist_qbin"],
    )
    idx0c = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["atom_0"].astype(str).to_numpy(),
            df["atom_1"].astype(str).to_numpy(),
            df["dist_qbin_coarse"].astype("int32").to_numpy(),
        ],
        names=["type", "atom_0", "atom_1", "dist_qbin_coarse"],
    )

    idx1 = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["atom_pair_0"].astype(str).to_numpy(),
            df["atom_pair_1"].astype(str).to_numpy(),
            df["dist_qbin"].astype("int32").to_numpy(),
        ],
        names=["type", "atom_pair_0", "atom_pair_1", "dist_qbin"],
    )
    idx1c = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["atom_pair_0"].astype(str).to_numpy(),
            df["atom_pair_1"].astype(str).to_numpy(),
            df["dist_qbin_coarse"].astype("int32").to_numpy(),
        ],
        names=["type", "atom_pair_0", "atom_pair_1", "dist_qbin_coarse"],
    )
    idx2 = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["atom_pair_0"].astype(str).to_numpy(),
            df["atom_pair_1"].astype(str).to_numpy(),
        ],
        names=["type", "atom_pair_0", "atom_pair_1"],
    )

    idx2d = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["dist_qbin"].astype("int32").to_numpy(),
        ],
        names=["type", "dist_qbin"],
    )
    idx2dc = pd.MultiIndex.from_arrays(
        [
            df["type"].astype(str).to_numpy(),
            df["dist_qbin_coarse"].astype("int32").to_numpy(),
        ],
        names=["type", "dist_qbin_coarse"],
    )

    p = pd.Series(np.nan, index=df.index, dtype="float64")

    p0 = pd.Series(g0.reindex(idx0).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p0)

    p0c = pd.Series(g0c.reindex(idx0c).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p0c)

    p1 = pd.Series(g1.reindex(idx1).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p1)

    p1c = pd.Series(g1c.reindex(idx1c).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p1c)

    p2 = pd.Series(g2.reindex(idx2).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p2)

    p2d = pd.Series(g2d.reindex(idx2d).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p2d)

    p2dc = pd.Series(g2dc.reindex(idx2dc).to_numpy(), index=df.index, dtype="float64")
    p = p.fillna(p2dc)

    p3 = pd.Series(df["type"].map(g3), index=df.index, dtype="float64")
    p = p.fillna(p3)

    p = p.fillna(global_median)
    return p


pred = hierarchical_predict(test_f)
assert pred.notna().all()

submission = pd.DataFrame(
    {"id": test["id"].values, "scalar_coupling_constant": pred.values}
)
submission["id"] = submission["id"].astype(sample_sub["id"].dtype, copy=False)

print(submission.head())
print(submission["scalar_coupling_constant"].describe())



## === cell 4
OUT_PATH = "submission.csv"
submission.to_csv(OUT_PATH, index=False, float_format="%.6f")
print("Wrote:", OUT_PATH)
print("File size (bytes):", os.path.getsize(OUT_PATH))

check = pd.read_csv(OUT_PATH)
assert list(check.columns) == ["id", "scalar_coupling_constant"]
assert len(check) == len(test)
assert check["scalar_coupling_constant"].notna().all()
print("Submission verified:", check.shape)
