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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

-1.31695555864586

# 6. Current score

1.18497

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.18497) has done: 'The current notebook tries to ensemble four external submissions from Kaggle Datasets that are not available in your environment, causing the FileNotFoundError and preventing any submission from being written. I replace that dependency with an in-notebook baseline that uses only the provided `train.csv` and `test.csv`, while keeping the “blend/aggregation” spirit by predicting per-`type` medians from the training set. This run end-to-end quickly, avoid leakage (train/test are split by molecule already), and produce a valid `submission.csv` with the required columns. The prediction is score-reasonable for this competition and should move you toward the target vs. a constant guess, while primarily focusing on correctness and stability.'
- What this solution (achieved 1.18497) has done: 'Your current solution is a very simple per-`type` median baseline, which is stable but far from the target log-MAE. To move the score downward (lower is better) toward the target with minimal disruption, I keep the same “predict from aggregates” core idea but add a second, more informative aggregate: per-(`type`, `atom_0`, `atom_1`) medians using `structures.csv` to get the element symbols for each atom index. This preserves the lightweight, no-model training approach while typically reducing MAE substantially versus per-`type` only. I also keep safe fallbacks (to per-`type`, then global median) to ensure full coverage and a valid submission CSV.'
- What this solution (achieved 1.18497) has done: 'You’re already well above the target (lower is better): current_score 1.18497 vs target -1.31696, so we should improve (reduce) the error while keeping your “aggregate median lookup” core logic intact. The smallest meaningful lift is to add slightly richer but still purely-aggregated keys using element-pair symmetry and atom-index distance, which are strong signals in this competition and don’t change the approach into a real model. Concretely: compute medians for (`type`, unordered(atom0,atom1)), and for (`type`, unordered(atom0,atom1), bond_distance_bin) where distance is from `structures.csv`; then use a safe fallback chain to your existing (`type`,`atom_0`,`atom_1`) → `type` → global. This keeps runtime reasonable (single merge + vectorized distance) and preserves submission semantics while plausibly moving the score downward toward the target.'
- What this solution (achieved 1.18497) has done: 'I fix the runtime error caused by constructing a `Series` directly from a `MultiIndex` (unsupported in pandas 2.2+) by switching to an index-aligned `reindex` lookup against the median Series’ MultiIndex. This keeps your exact “aggregate median lookup with fallback chain” logic intact (no model changes), but makes it compatible with the current pandas version and ensures `pred` is always created. Then I ensure the submission is written end-to-end as `submission.csv` with the correct columns and no NaNs. These changes are score-neutral in intent (same medians and same fallback order), just making the mapping implementation work.'
- What this solution (achieved 1.18497) has done: 'Your current approach is a pure “median lookup with fallbacks”, which is stable but it’s leaving performance on the table because the keys are still too coarse and you’re not handling per-type scale differences. I keep the exact same core logic (groupby medians + fallback chain) and only add one more informative, still-aggregated key: a per-type standardized distance bin (z-score binning within each `type`) which usually groups interactions more consistently than global quantiles. I also add a small, safe shrinkage rule to back off overly-fragmented medians (computed on tiny groups) toward the parent median, which typically reduces MAE without changing the paradigm. These changes should reduce log-MAE (lower is better) and move you toward the target, while staying fast and producing the same submission format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "../input/champs-scalar-coupling",
    "../input",  # fallback if files are directly in ../input
]


def find_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(d):
            if os.path.exists(os.path.join(d, "train.csv")):
                return d
    for d in DATA_DIR_CANDIDATES:
        nested = os.path.join(d, "champs-scalar-coupling")
        if os.path.exists(os.path.join(nested, "train.csv")):
            return nested
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling data directory in known locations."
    )


DATA_DIR = find_data_dir()
print("Using DATA_DIR:", DATA_DIR)
print("Files (sample):", sorted(os.listdir(DATA_DIR))[:20])



## === cell 1
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("train shape:", train.shape)
print("test shape:", test.shape)
print("sample shape:", sample.shape)

required_train_cols = {
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
}
required_test_cols = {"id", "molecule_name", "atom_index_0", "atom_index_1", "type"}
if not required_train_cols.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {required_train_cols - set(train.columns)}"
    )
if not required_test_cols.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {required_test_cols - set(test.columns)}"
    )

if not os.path.exists(structures_path):
    raise FileNotFoundError(f"Missing structures.csv at: {structures_path}")

structures = pd.read_csv(
    structures_path, usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"]
)
print("structures shape:", structures.shape)
print("structures atoms:", structures["atom"].nunique())



## === cell 2
s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x_0",
        "y": "y_0",
        "z": "z_0",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x_1",
        "y": "y_1",
        "z": "z_1",
    }
)

train2 = train.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)
test2 = test.merge(s0, on=["molecule_name", "atom_index_0"], how="left").merge(
    s1, on=["molecule_name", "atom_index_1"], how="left"
)

missing_train_atoms = train2["atom_0"].isna().sum() + train2["atom_1"].isna().sum()
missing_test_atoms = test2["atom_0"].isna().sum() + test2["atom_1"].isna().sum()
print(
    "Missing atom labels - train:",
    int(missing_train_atoms),
    "test:",
    int(missing_test_atoms),
)


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dx = out["x_0"].to_numpy(dtype=np.float64) - out["x_1"].to_numpy(dtype=np.float64)
    dy = out["y_0"].to_numpy(dtype=np.float64) - out["y_1"].to_numpy(dtype=np.float64)
    dz = out["z_0"].to_numpy(dtype=np.float64) - out["z_1"].to_numpy(dtype=np.float64)
    out["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    return out


train2 = add_distance(train2)
test2 = add_distance(test2)

train2["atom_min"] = np.where(
    train2["atom_0"] <= train2["atom_1"], train2["atom_0"], train2["atom_1"]
)
train2["atom_max"] = np.where(
    train2["atom_0"] <= train2["atom_1"], train2["atom_1"], train2["atom_0"]
)
test2["atom_min"] = np.where(
    test2["atom_0"] <= test2["atom_1"], test2["atom_0"], test2["atom_1"]
)
test2["atom_max"] = np.where(
    test2["atom_0"] <= test2["atom_1"], test2["atom_1"], test2["atom_0"]
)

type_dist_stats = train2.groupby("type")["dist"].agg(["mean", "std"])
type_dist_stats["std"] = type_dist_stats["std"].replace(0.0, np.nan)

train2 = train2.merge(
    type_dist_stats, left_on="type", right_index=True, how="left", suffixes=("", "_t")
)
test2 = test2.merge(
    type_dist_stats, left_on="type", right_index=True, how="left", suffixes=("", "_t")
)

train2["dist_z"] = (train2["dist"] - train2["mean"]) / train2["std"]
test2["dist_z"] = (test2["dist"] - test2["mean"]) / test2["std"]

Z_MIN, Z_MAX = -4.0, 4.0
N_ZBINS = 32
z_edges = np.linspace(Z_MIN, Z_MAX, N_ZBINS + 1)


def zbin_from_z(z: pd.Series) -> pd.Series:
    zz = z.to_numpy(dtype=np.float64)
    bins = np.full(len(zz), -1, dtype=np.int16)
    ok = np.isfinite(zz)
    if ok.any():
        zc = np.clip(zz[ok], Z_MIN, Z_MAX)
        bins_ok = (np.searchsorted(z_edges, zc, side="right") - 1).astype(np.int16)
        bins_ok = np.clip(bins_ok, 0, len(z_edges) - 2)
        bins[ok] = bins_ok
    return pd.Series(bins, index=z.index, name="dist_zbin")


train2["dist_zbin"] = zbin_from_z(train2["dist_z"])
test2["dist_zbin"] = zbin_from_z(test2["dist_z"])

global_median = train2["scalar_coupling_constant"].median()
type_median = train2.groupby("type")["scalar_coupling_constant"].median()

type_atom_median = train2.groupby(["type", "atom_0", "atom_1"])[
    "scalar_coupling_constant"
].median()

type_symatom_median = train2.groupby(["type", "atom_min", "atom_max"])[
    "scalar_coupling_constant"
].median()

g_cols = ["type", "atom_min", "atom_max", "dist_zbin"]
train_g = train2[train2["dist_zbin"] >= 0].groupby(g_cols)["scalar_coupling_constant"]
type_symatom_distz_median = train_g.median()
type_symatom_distz_count = train_g.size()

parent_idx = type_symatom_distz_median.index.droplevel("dist_zbin")
parent_med_aligned = pd.Series(
    type_symatom_median.reindex(parent_idx).to_numpy(dtype=np.float64),
    index=type_symatom_distz_median.index,
)
cnt = type_symatom_distz_count.reindex(type_symatom_distz_median.index).to_numpy(
    dtype=np.float64
)
w = np.clip(cnt / 20.0, 0.0, 1.0)
type_symatom_distz_median_shrunk = pd.Series(
    w * type_symatom_distz_median.to_numpy(dtype=np.float64)
    + (1.0 - w) * parent_med_aligned.to_numpy(dtype=np.float64),
    index=type_symatom_distz_median.index,
)

idx1 = pd.MultiIndex.from_frame(test2[g_cols])
pred = type_symatom_distz_median_shrunk.reindex(idx1).to_numpy(dtype=np.float64)

idx2 = pd.MultiIndex.from_frame(test2[["type", "atom_min", "atom_max"]])
pred = pd.Series(pred, index=test2.index, dtype="float64").fillna(
    pd.Series(
        type_symatom_median.reindex(idx2).to_numpy(dtype=np.float64), index=test2.index
    )
)

idx3 = pd.MultiIndex.from_frame(test2[["type", "atom_0", "atom_1"]])
pred = pred.fillna(
    pd.Series(
        type_atom_median.reindex(idx3).to_numpy(dtype=np.float64), index=test2.index
    )
)

pred = pred.fillna(test2["type"].map(type_median))
pred = pred.fillna(global_median).astype(np.float64)

print("Per-type medians computed for types:", len(type_median))
print("Per-(type,atom0,atom1) medians computed:", len(type_atom_median))
print("Per-(type,atom_min,atom_max) medians computed:", len(type_symatom_median))
print(
    "Per-(type,atom_min,atom_max,dist_zbin) medians computed:",
    len(type_symatom_distz_median),
)
print("dist_zbin coverage in test (>=0):", float((test2["dist_zbin"] >= 0).mean()))
print("Prediction describe:\n", pred.describe())



## === cell 3
submission = sample.copy()
if (
    "id" not in submission.columns
    or "scalar_coupling_constant" not in submission.columns
):
    raise ValueError("sample_submission.csv does not have the expected columns.")

test_id_to_pred = pd.Series(pred.to_numpy(), index=test["id"].to_numpy())
submission["scalar_coupling_constant"] = submission["id"].map(test_id_to_pred)

submission["scalar_coupling_constant"] = (
    submission["scalar_coupling_constant"].fillna(global_median).astype(np.float64)
)

if submission.isna().any().any():
    raise ValueError("Submission contains NaNs after filling; aborting.")

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("submission shape:", submission.shape)
