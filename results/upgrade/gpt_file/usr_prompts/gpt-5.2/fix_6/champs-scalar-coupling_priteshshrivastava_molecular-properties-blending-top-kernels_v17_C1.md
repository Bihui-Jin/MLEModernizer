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

-1.6821991287997458

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

_CANDIDATE_DIRS = [
    "/kaggle/input/champs-scalar-coupling",
    "/kaggle/data/champs-scalar-coupling",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for d in _CANDIDATE_DIRS:
    if os.path.isdir(d):
        if os.path.basename(d) in ("input", "data"):
            comp = os.path.join(d, "champs-scalar-coupling")
            if os.path.isdir(comp):
                DATA_DIR = comp
                break
        else:
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "structures.csv")
            ):
                DATA_DIR = d
                break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate champs-scalar-coupling dataset directory. Tried: "
        + ", ".join(_CANDIDATE_DIRS)
    )

print("Using DATA_DIR:", DATA_DIR)
print("Listing DATA_DIR (head):", sorted(os.listdir(DATA_DIR))[:20])




## === cell 1
from sklearn.ensemble import RandomForestRegressor

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
structures_path = os.path.join(DATA_DIR, "structures.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
structures = pd.read_csv(structures_path)
sample_sub = pd.read_csv(sample_sub_path)

required_struct_cols = {"molecule_name", "atom_index", "x", "y", "z"}
missing = required_struct_cols - set(structures.columns)
if missing:
    raise ValueError(
        f"structures.csv is missing required columns: {missing}. "
        f"Available columns: {structures.columns.tolist()}"
    )

if "atom" not in structures.columns:
    structures["atom"] = "X"

for df in (train, test, structures):
    df["molecule_name"] = df["molecule_name"].astype(str)

for df in (train, test):
    df["atom_index_0"] = df["atom_index_0"].astype(np.int32)
    df["atom_index_1"] = df["atom_index_1"].astype(np.int32)

structures["atom_index"] = structures["atom_index"].astype(np.int32)

structures_small = structures[["molecule_name", "atom_index", "x", "y", "z"]].copy()

train_feat = train.merge(
    structures_small.rename(
        columns={
            "atom_index": "atom_index_0",
            "x": "x_0",
            "y": "y_0",
            "z": "z_0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
)

test_feat = test.merge(
    structures_small.rename(
        columns={
            "atom_index": "atom_index_0",
            "x": "x_0",
            "y": "y_0",
            "z": "z_0",
        }
    ),
    on=["molecule_name", "atom_index_0"],
    how="left",
)

train_feat = train_feat.merge(
    structures_small.rename(
        columns={
            "atom_index": "atom_index_1",
            "x": "x_1",
            "y": "y_1",
            "z": "z_1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

test_feat = test_feat.merge(
    structures_small.rename(
        columns={
            "atom_index": "atom_index_1",
            "x": "x_1",
            "y": "y_1",
            "z": "z_1",
        }
    ),
    on=["molecule_name", "atom_index_1"],
    how="left",
)

for name, df in (("train_feat", train_feat), ("test_feat", test_feat)):
    missing0 = int(df["x_0"].isna().sum())
    missing1 = int(df["x_1"].isna().sum())
    if missing0 or missing1:
        raise ValueError(
            f"{name}: missing merged structure rows. "
            f"atom0 coord NaNs={missing0}, atom1 coord NaNs={missing1}. "
            f"Likely molecule_name/atom_index dtype mismatch or wrong DATA_DIR."
        )

train_feat["atom_0"] = "X"
train_feat["atom_1"] = "X"
test_feat["atom_0"] = "X"
test_feat["atom_1"] = "X"

for df in (train_feat, test_feat):
    dx = df["x_0"] - df["x_1"]
    dy = df["y_0"] - df["y_1"]
    dz = df["z_0"] - df["z_1"]
    df["dx"] = dx
    df["dy"] = dy
    df["dz"] = dz
    df["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    df["dist2"] = dx * dx + dy * dy + dz * dz

all_atoms0 = pd.concat([train_feat["atom_0"], test_feat["atom_0"]], axis=0).astype(
    "category"
)
all_atoms1 = pd.concat([train_feat["atom_1"], test_feat["atom_1"]], axis=0).astype(
    "category"
)

atom0_cats = all_atoms0.cat.categories
atom1_cats = all_atoms1.cat.categories

train_feat["atom_0_i"] = pd.Categorical(
    train_feat["atom_0"], categories=atom0_cats
).codes.astype(np.int16)
train_feat["atom_1_i"] = pd.Categorical(
    train_feat["atom_1"], categories=atom1_cats
).codes.astype(np.int16)
test_feat["atom_0_i"] = pd.Categorical(
    test_feat["atom_0"], categories=atom0_cats
).codes.astype(np.int16)
test_feat["atom_1_i"] = pd.Categorical(
    test_feat["atom_1"], categories=atom1_cats
).codes.astype(np.int16)

for col in ("atom_0_i", "atom_1_i"):
    if (train_feat[col] < 0).any() or (test_feat[col] < 0).any():
        raise ValueError(
            f"Found negative categorical codes in {col}; unexpected missing/unseen atom values."
        )

feature_cols = [
    "atom_index_0",
    "atom_index_1",
    "atom_0_i",
    "atom_1_i",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist2",
]

preds = np.zeros(len(test_feat), dtype=np.float64)

GLOBAL_SEED = 42

types = sorted(train_feat["type"].unique().tolist())
test_type_groups = test_feat.groupby("type").indices
train_type_groups = train_feat.groupby("type").indices

type_median = train_feat.groupby("type")["scalar_coupling_constant"].median().to_dict()
global_median = float(train_feat["scalar_coupling_constant"].median())

for t in types:
    tr_idx = train_type_groups.get(t, [])
    te_idx = test_type_groups.get(t, [])
    if len(te_idx) == 0:
        continue

    if len(tr_idx) < 50:
        preds[te_idx] = type_median.get(t, global_median)
        continue

    X_tr = train_feat.loc[tr_idx, feature_cols]
    y_tr = train_feat.loc[tr_idx, "scalar_coupling_constant"].astype(np.float64)
    X_te = test_feat.loc[te_idx, feature_cols]

    model = RandomForestRegressor(
        n_estimators=80,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
    )
    model.fit(X_tr, y_tr)
    preds[te_idx] = model.predict(X_te)

submission = pd.DataFrame(
    {"id": test_feat["id"].values, "scalar_coupling_constant": preds}
)
submission = submission.sort_values("id").reset_index(drop=True)

submission = sample_sub[["id"]].merge(
    submission, on="id", how="left", validate="one_to_one"
)

if submission["scalar_coupling_constant"].isna().any():
    raise ValueError(
        "Submission contains NaN predictions after alignment; cannot write valid submission."
    )

out_path = "my_blend_1.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Columns:", submission.columns.tolist())
print("NaNs in preds:", int(submission["scalar_coupling_constant"].isna().sum()))

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2797255819.py in <cell line: 0>()
     93     if missing0 or missing1:
     94         # Keep the hard fail: without coords the model features are invalid.
---> 95         raise ValueError(
     96             f"{name}: missing merged structure rows. "
     97             f"atom0 coord NaNs={missing0}, atom1 coord NaNs={missing1}. "

ValueError: test_feat: missing merged structure rows. atom0 coord NaNs=467813, atom1 coord NaNs=467813. Likely molecule_name/atom_index dtype mismatch or wrong DATA_DIR.
