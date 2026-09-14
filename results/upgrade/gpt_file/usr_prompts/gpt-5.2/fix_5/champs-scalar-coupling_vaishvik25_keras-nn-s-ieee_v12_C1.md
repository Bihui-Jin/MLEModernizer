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
    Minimal improvement toward target score:
    - Canonicalize atom-index pairs before merging so distance/atoms are invariant to pair order.
      This increases reuse of the same learned medians between train/test and reduces sparsity.
    - Keep robust merges and NaN handling exactly as before.
    """
    df = df.copy()
    df["molecule_name"] = df["molecule_name"].astype("string")
    df["atom_index_0"] = pd.to_numeric(df["atom_index_0"], downcast="integer")
    df["atom_index_1"] = pd.to_numeric(df["atom_index_1"], downcast="integer")

    a0 = df["atom_index_0"].astype("int64")
    a1 = df["atom_index_1"].astype("int64")
    df["atom_index_0"] = np.minimum(a0, a1).astype(df["atom_index_0"].dtype, copy=False)
    df["atom_index_1"] = np.maximum(a0, a1).astype(df["atom_index_1"].dtype, copy=False)

    s0 = structures_df.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    df = df.merge(s0, on=["molecule_name", "atom_index_0"], how="left", sort=False)

    s1 = structures_df.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
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

    a0s = df["atom_0"].astype("string")
    a1s = df["atom_1"].astype("string")
    df["atom_a"] = a0s.where(a0s <= a1s, a1s)
    df["atom_b"] = a1s.where(a0s <= a1s, a0s)
    df.drop(columns=["atom_0", "atom_1"], inplace=True)
    df.rename(columns={"atom_a": "atom_0", "atom_b": "atom_1"}, inplace=True)

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

bw_train = train_f["type"].map(type_bin_width).astype("float64")
bw_test = test_f["type"].map(type_bin_width).astype("float64")
bw_train = bw_train.fillna(global_bin_width)
bw_test = bw_test.fillna(global_bin_width)

train_f["dist_bin"] = np.floor(train_f["dist"] / bw_train).astype("Int32")
test_f["dist_bin"] = np.floor(test_f["dist"] / bw_test).astype("Int32")

SENTINEL_BIN = -1
train_f["dist_bin"] = train_f["dist_bin"].fillna(SENTINEL_BIN).astype("Int32")
test_f["dist_bin"] = test_f["dist_bin"].fillna(SENTINEL_BIN).astype("Int32")

train_f["atom_0"] = train_f["atom_0"].astype("string")
train_f["atom_1"] = train_f["atom_1"].astype("string")
test_f["atom_0"] = test_f["atom_0"].astype("string")
test_f["atom_1"] = test_f["atom_1"].astype("string")

print(train_f[["type", "atom_0", "atom_1", "dist", "dist_bin"]].head())

missing_train_atoms = train_f["atom_0"].isna().any() or train_f["atom_1"].isna().any()
missing_test_atoms = test_f["atom_0"].isna().any() or test_f["atom_1"].isna().any()
print("Any missing atoms in train?", bool(missing_train_atoms))
print("Any missing atoms in test?", bool(missing_test_atoms))
print("Missing dist in train:", int(train_f["dist"].isna().sum()))
print("Missing dist in test:", int(test_f["dist"].isna().sum()))



## === cell 3
global_median = float(train_f["scalar_coupling_constant"].median())

g1 = train_f.groupby(["type", "atom_0", "atom_1", "dist_bin"], observed=True)[
    "scalar_coupling_constant"
].median()
g2 = train_f.groupby(["type", "atom_0", "atom_1"], observed=True)[
    "scalar_coupling_constant"
].median()
g3 = train_f.groupby(["type"], observed=True)["scalar_coupling_constant"].median()


def hierarchical_predict(df):
    k1 = list(
        zip(
            df["type"].astype(str).values,
            df["atom_0"].astype(str).values,
            df["atom_1"].astype(str).values,
            df["dist_bin"].astype("int32").values,
        )
    )
    k2 = list(
        zip(
            df["type"].astype(str).values,
            df["atom_0"].astype(str).values,
            df["atom_1"].astype(str).values,
        )
    )

    p = pd.Series(np.nan, index=df.index, dtype="float64")

    p1 = pd.Series([g1.get(key, np.nan) for key in k1], index=df.index, dtype="float64")
    p = p.fillna(p1)

    p2 = pd.Series([g2.get(key, np.nan) for key in k2], index=df.index, dtype="float64")
    p = p.fillna(p2)

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
