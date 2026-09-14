# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)



## === cell 1
DATA_DIR = "../input/champs-scalar-coupling"

train_cols = [
    "id",
    "molecule_name",
    "atom_index_0",
    "atom_index_1",
    "type",
    "scalar_coupling_constant",
]
test_cols = ["id", "molecule_name", "atom_index_0", "atom_index_1", "type"]
structures_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]

df_train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    usecols=train_cols,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
df_test = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    usecols=test_cols,
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    usecols=structures_cols,
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)



## === cell 2
sample_submission = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", usecols=["id", "scalar_coupling_constant"]
)



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
plt = None
sns = None



## === cell 12
pass



## === cell 13
pass



## === cell 14

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_x",
        "x": "x_x",
        "y": "y_x",
        "z": "z_x",
    }
)
s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_y",
        "x": "x_y",
        "y": "y_y",
        "z": "z_y",
    }
)

all_mols = pd.api.types.union_categoricals(
    [df_train["molecule_name"], df_test["molecule_name"], structures["molecule_name"]]
)
df_train["molecule_name"] = df_train["molecule_name"].astype(all_mols.dtype)
df_test["molecule_name"] = df_test["molecule_name"].astype(all_mols.dtype)
structures["molecule_name"] = structures["molecule_name"].astype(all_mols.dtype)
s0["molecule_name"] = s0["molecule_name"].astype(all_mols.dtype)
s1["molecule_name"] = s1["molecule_name"].astype(all_mols.dtype)

train = df_train.merge(
    s0[["molecule_name", "atom_index_0", "atom_x", "x_x", "y_x", "z_x"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    copy=False,
)
train = train.merge(
    s1[["molecule_name", "atom_index_1", "atom_y", "x_y", "y_y", "z_y"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    copy=False,
)

test = df_test.merge(
    s0[["molecule_name", "atom_index_0", "atom_x", "x_x", "y_x", "z_x"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
    copy=False,
)
test = test.merge(
    s1[["molecule_name", "atom_index_1", "atom_y", "x_y", "y_y", "z_y"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
    copy=False,
)



## === cell 15
pass



## === cell 16
pass



## === cell 17
pass



## === cell 18
train = train.drop(["molecule_name"], axis=1)
test = test.drop(["molecule_name"], axis=1)



## === cell 19
pass




## === cell 20
def atom_number(atom):
    if atom == "H":
        return 0
    elif atom == "C":
        return 1
    elif atom == "N":
        return 2
    elif atom == "O":
        return 3
    elif atom == "F":
        return 4
    return -1




## === cell 21
_atom_map = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

train["atom_y"] = train["atom_y"].map(_atom_map).fillna(-1).astype(np.int8)
train["atom_x"] = train["atom_x"].map(_atom_map).fillna(-1).astype(np.int8)
test["atom_y"] = test["atom_y"].map(_atom_map).fillna(-1).astype(np.int8)
test["atom_x"] = test["atom_x"].map(_atom_map).fillna(-1).astype(np.int8)



## === cell 22

type_cats = pd.api.types.union_categoricals(
    [df_train["type"], df_test["type"]]
).categories
train["type"] = train["type"].cat.set_categories(type_cats)
test["type"] = test["type"].cat.set_categories(type_cats)

X_train_full = pd.get_dummies(
    train.drop(columns=["scalar_coupling_constant"]),
    columns=["type"],
    drop_first=True,
)
y_train_full = train["scalar_coupling_constant"].copy()

X_test_full = pd.get_dummies(
    test,
    columns=["type"],
    drop_first=True,
)

X_train_full, X_test_full = X_train_full.align(
    X_test_full, join="outer", axis=1, fill_value=0
)

for c in X_train_full.columns:
    if X_train_full[c].dtype == np.uint8:
        X_train_full[c] = X_train_full[c].astype(np.int8, copy=False)
for c in X_test_full.columns:
    if X_test_full[c].dtype == np.uint8:
        X_test_full[c] = X_test_full[c].astype(np.int8, copy=False)



## === cell 23
pass



## === cell 24
dx = (train["x_y"].to_numpy() - train["x_x"].to_numpy()).astype(np.float32, copy=False)
dy = (train["y_y"].to_numpy() - train["y_x"].to_numpy()).astype(np.float32, copy=False)
dz = (train["z_y"].to_numpy() - train["z_x"].to_numpy()).astype(np.float32, copy=False)
train_dist = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32)
X_train_full["distance"] = train_dist

dx = (test["x_y"].to_numpy() - test["x_x"].to_numpy()).astype(np.float32, copy=False)
dy = (test["y_y"].to_numpy() - test["y_x"].to_numpy()).astype(np.float32, copy=False)
dz = (test["z_y"].to_numpy() - test["z_x"].to_numpy()).astype(np.float32, copy=False)
test_dist = np.sqrt(dx * dx + dy * dy + dz * dz, dtype=np.float32)
X_test_full["distance"] = test_dist



## === cell 25
pass



## === cell 26
X_train_full = X_train_full.drop(["id"], axis=1)
X_test_full = X_test_full.drop(["id"], axis=1)



## === cell 27
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)



## === cell 28
from lightgbm import LGBMRegressor
import lightgbm as lgb



## === cell 29
model = LGBMRegressor(
    random_state=42,
    n_estimators=10000,
    n_jobs=-1,
)
model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=10)],
)



## === cell 30
preds_val = model.predict(X_val, num_iteration=model.best_iteration_)



## === cell 31
test_predictions = model.predict(X_test_full, num_iteration=model.best_iteration_)



## === cell 32
pass



## === cell 33
submission = pd.DataFrame(
    {"id": df_test["id"].values, "scalar_coupling_constant": test_predictions}
)
submission = submission[["id", "scalar_coupling_constant"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
