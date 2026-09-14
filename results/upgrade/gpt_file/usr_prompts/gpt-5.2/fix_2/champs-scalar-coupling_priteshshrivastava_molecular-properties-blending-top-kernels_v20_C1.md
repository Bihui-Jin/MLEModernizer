# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-1.6838160788283738

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/data/champs-scalar-coupling"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/champs-scalar-coupling"

print("Using INPUT_DIR:", INPUT_DIR)
print("Files:", sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")])[:10])



## === cell 1

from sklearn.ensemble import RandomForestRegressor

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
structures_path = os.path.join(INPUT_DIR, "structures.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)

structures = structures[["molecule_name", "atom_index", "atom", "x", "y", "z"]]

atomic_num = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
structures["atom_num"] = structures["atom"].map(atomic_num).astype(np.int16)

s0 = (
    structures.rename(
        columns={
            "atom_index": "atom_index_0",
            "atom": "atom_0",
            "atom_num": "atom_num_0",
            "x": "x0",
            "y": "y0",
            "z": "z0",
        }
    )
    .drop(columns=["molecule_name"])
    .copy()
)
s1 = (
    structures.rename(
        columns={
            "atom_index": "atom_index_1",
            "atom": "atom_1",
            "atom_num": "atom_num_1",
            "x": "x1",
            "y": "y1",
            "z": "z1",
        }
    )
    .drop(columns=["molecule_name"])
    .copy()
)

train_feat = train.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
)
train_feat = train_feat.drop(columns=["atom_index", "atom", "x", "y", "z"])
train_feat = train_feat.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    suffixes=("_0tmp", "_1tmp"),
)
train_feat = train_feat.rename(
    columns={
        "atom_0tmp": "atom_0",
        "x_0tmp": "x0",
        "y_0tmp": "y0",
        "z_0tmp": "z0",
        "atom_num_0tmp": "atom_num_0",
        "atom_1tmp": "atom_1",
        "x_1tmp": "x1",
        "y_1tmp": "y1",
        "z_1tmp": "z1",
        "atom_num_1tmp": "atom_num_1",
    }
)
train_feat = train_feat.drop(columns=["atom_index"])

test_feat = test.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
)
test_feat = test_feat.drop(columns=["atom_index", "atom", "x", "y", "z"])
test_feat = test_feat.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    suffixes=("_0tmp", "_1tmp"),
)
test_feat = test_feat.rename(
    columns={
        "atom_0tmp": "atom_0",
        "x_0tmp": "x0",
        "y_0tmp": "y0",
        "z_0tmp": "z0",
        "atom_num_0tmp": "atom_num_0",
        "atom_1tmp": "atom_1",
        "x_1tmp": "x1",
        "y_1tmp": "y1",
        "z_1tmp": "z1",
        "atom_num_1tmp": "atom_num_1",
    }
)
test_feat = test_feat.drop(columns=["atom_index"])


def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    dx = df["x0"].astype(np.float32) - df["x1"].astype(np.float32)
    dy = df["y0"].astype(np.float32) - df["y1"].astype(np.float32)
    dz = df["z0"].astype(np.float32) - df["z1"].astype(np.float32)
    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["dist2"] = dx * dx + dy * dy + dz * dz
    df["dist"] = np.sqrt(df["dist2"].values, dtype=np.float32)
    df["inv_dist"] = (1.0 / (df["dist"].values + 1e-6)).astype(np.float32)
    df["atom_num_sum"] = (
        df["atom_num_0"].astype(np.int16) + df["atom_num_1"].astype(np.int16)
    ).astype(np.int16)
    df["atom_num_diff"] = (
        df["atom_num_0"].astype(np.int16) - df["atom_num_1"].astype(np.int16)
    ).astype(np.int16)
    return df


train_feat = add_pair_features(train_feat)
test_feat = add_pair_features(test_feat)

for col in ["x0", "y0", "z0", "x1", "y1", "z1", "atom_num_0", "atom_num_1"]:
    if train_feat[col].isna().any() or test_feat[col].isna().any():
        train_feat[col] = train_feat[col].fillna(0)
        test_feat[col] = test_feat[col].fillna(0)

feature_cols = [
    "atom_index_0",
    "atom_index_1",
    "atom_num_0",
    "atom_num_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
    "inv_dist",
    "atom_num_sum",
    "atom_num_diff",
]

for c in feature_cols:
    train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").fillna(0)
    test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce").fillna(0)

preds = np.zeros(len(test_feat), dtype=np.float32)

types = sorted(train_feat["type"].unique())
print("Training per type:", types)

for t in types:
    tr_idx = (train_feat["type"] == t).values
    te_idx = (test_feat["type"] == t).values

    X_tr = train_feat.loc[tr_idx, feature_cols]
    y_tr = train_feat.loc[tr_idx, "scalar_coupling_constant"].astype(np.float32)

    X_te = test_feat.loc[te_idx, feature_cols]

    model = RandomForestRegressor(
        n_estimators=60,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="auto",
    )
    model.fit(X_tr, y_tr)
    preds[te_idx] = model.predict(X_te).astype(np.float32)

submission = pd.read_csv(sample_sub_path)[["id"]].copy()
submission = submission.merge(
    test[["id"]], on="id", how="left"
)  # keep same ordering as sample_submission if possible
if submission["id"].isna().any() or len(submission) != len(test):
    submission = pd.DataFrame({"id": test["id"].values})

submission["scalar_coupling_constant"] = preds

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submission), "cols:", list(submission.columns))
print(submission.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'x0'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4228305929.py in <cell line: 0>()
    135 
    136 
--> 137 train_feat = add_pair_features(train_feat)
    138 test_feat = add_pair_features(test_feat)
    139 

/tmp/ipykernel_11/4228305929.py in add_pair_features(df)
    117 # Feature engineering (minimal, distance-based)
    118 def add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
--> 119     dx = df["x0"].astype(np.float32) - df["x1"].astype(np.float32)
    120     dy = df["y0"].astype(np.float32) - df["y1"].astype(np.float32)
    121     dz = df["z0"].astype(np.float32) - df["z1"].astype(np.float32)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'x0'
