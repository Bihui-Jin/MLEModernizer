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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics



## === cell 1
train_dtype = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "scalar_coupling_constant": np.float32,
}
train_original = pd.read_csv(
    "../input/train.csv",
    dtype=train_dtype,
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
)

structures_dtype = {
    "molecule_name": "category",
    "atom_index": np.int16,
    "atom": "category",
    "x": np.float32,
    "y": np.float32,
    "z": np.float32,
}
structures_original = pd.read_csv(
    "../input/structures.csv",
    dtype=structures_dtype,
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
)

test_dtype = {
    "id": np.int32,
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
}
test_original = pd.read_csv(
    "../input/test.csv",
    dtype=test_dtype,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
)



## === cell 2
potential_dtype = {"molecule_name": "category", "potential_energy": np.float32}
potential_energy = pd.read_csv("../input/potential_energy.csv", dtype=potential_dtype)

dipole_dtype = {
    "molecule_name": "category",
    "X": np.float32,
    "Y": np.float32,
    "Z": np.float32,
}
dipole = pd.read_csv("../input/dipole_moments.csv", dtype=dipole_dtype)
dipole["dipole_mag"] = np.sqrt(dipole[["X", "Y", "Z"]].pow(2).sum(axis=1))



## === cell 3
atom_counts = (
    structures_original.groupby(["molecule_name", "atom"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

structures = structures_original.merge(atom_counts, on="molecule_name", how="left")
del structures_original, atom_counts
gc.collect()

train = (
    train_original.merge(
        structures[
            [
                "molecule_name",
                "atom_index",
                "atom",
                "x",
                "y",
                "z",
                "C",
                "F",
                "H",
                "N",
                "O",
            ]
        ],
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom_nm_0", "x": "x_0", "y": "y_0", "z": "z_0"})
    .drop(columns=["atom_index"])
)

train = (
    train.merge(
        structures[
            [
                "molecule_name",
                "atom_index",
                "atom",
                "x",
                "y",
                "z",
                "C",
                "F",
                "H",
                "N",
                "O",
            ]
        ],
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
        suffixes=("", "_1"),
    )
    .rename(columns={"atom": "atom_nm_1", "x": "x_1", "y": "y_1", "z": "z_1"})
    .drop(columns=["atom_index"])
)

train = train[
    [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "atom_nm_0",
        "atom_nm_1",
        "type",
        "x_0",
        "y_0",
        "z_0",
        "x_1",
        "y_1",
        "z_1",
        "C",
        "F",
        "H",
        "N",
        "O",
        "scalar_coupling_constant",
    ]
]

train.sort_values(by=["id", "molecule_name"], inplace=True)
train.reset_index(inplace=True, drop=True)

test = (
    test_original.merge(
        structures[
            [
                "molecule_name",
                "atom_index",
                "atom",
                "x",
                "y",
                "z",
                "C",
                "F",
                "H",
                "N",
                "O",
            ]
        ],
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
        how="left",
    )
    .rename(columns={"atom": "atom_nm_0", "x": "x_0", "y": "y_0", "z": "z_0"})
    .drop(columns=["atom_index"])
)

test = (
    test.merge(
        structures[
            [
                "molecule_name",
                "atom_index",
                "atom",
                "x",
                "y",
                "z",
                "C",
                "F",
                "H",
                "N",
                "O",
            ]
        ],
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
        how="left",
        suffixes=("", "_1"),
    )
    .rename(columns={"atom": "atom_nm_1", "x": "x_1", "y": "y_1", "z": "z_1"})
    .drop(columns=["atom_index"])
)

test = test[
    [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "atom_nm_0",
        "atom_nm_1",
        "type",
        "x_0",
        "y_0",
        "z_0",
        "x_1",
        "y_1",
        "z_1",
        "C",
        "F",
        "H",
        "N",
        "O",
    ]
]

test.sort_values(by=["id", "molecule_name"], inplace=True)
test.reset_index(inplace=True, drop=True)

del train_original, test_original, structures
gc.collect()



## === cell 4
contrib_dtype = {
    "molecule_name": "category",
    "atom_index_0": np.int16,
    "atom_index_1": np.int16,
    "type": "category",
    "fc": np.float32,
    "sd": np.float32,
    "pso": np.float32,
    "dso": np.float32,
}
contrib = pd.read_csv("../input/scalar_coupling_contributions.csv", dtype=contrib_dtype)

train = train.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)
test = test.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
)

train = train.merge(potential_energy, on="molecule_name", how="left")
test = test.merge(potential_energy, on="molecule_name", how="left")

train = train.merge(
    dipole[["molecule_name", "dipole_mag"]],
    on="molecule_name",
    how="left",
)
test = test.merge(
    dipole[["molecule_name", "dipole_mag"]],
    on="molecule_name",
    how="left",
)

del contrib, potential_energy, dipole
gc.collect()



## === cell 5
train["dist"] = np.linalg.norm(
    train[["x_0", "y_0", "z_0"]].values - train[["x_1", "y_1", "z_1"]].values, axis=1
).astype(np.float32)
test["dist"] = np.linalg.norm(
    test[["x_0", "y_0", "z_0"]].values - test[["x_1", "y_1", "z_1"]].values, axis=1
).astype(np.float32)

train["dist_sq"] = (train["dist"] ** 2).astype(np.float32)
test["dist_sq"] = (test["dist"] ** 2).astype(np.float32)

train = train.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"])
test = test.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"])

gc.collect()



## === cell 6
cat_cols = ["type", "atom_nm_0", "atom_nm_1"]
for col in cat_cols:
    train[col] = pd.Categorical(train[col])
    test[col] = pd.Categorical(test[col])



## === cell 7
FEATURE_COLS = [
    "atom_nm_0",
    "atom_nm_1",
    "type",
    "C",
    "F",
    "H",
    "N",
    "O",
    "dist",
    "dist_sq",
    "fc",
    "sd",
    "pso",
    "dso",
    "potential_energy",
    "dipole_mag",
]

X = train[FEATURE_COLS]
y = train["scalar_coupling_constant"]

num_cols = [c for c in FEATURE_COLS if c not in cat_cols]
X[num_cols] = X[num_cols].astype(np.float32)
y = y.astype(np.float32)



## === cell 8
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.4, random_state=420
)



## === cell 9
model_params = {
    "objective": "regression_l1",
    "learning_rate": 0.03,
    "num_leaves": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "verbose": -1,
    "n_estimators": 5000,
    "n_jobs": 4,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=150, verbose=False),
    lgb.log_evaluation(period=100),
]

gbm = lgb.LGBMRegressor(**model_params)

gbm.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    categorical_feature=cat_cols,
    callbacks=callbacks,
)



## === cell 10
y_pred_valid = gbm.predict(X_valid)
mae = metrics.mean_absolute_error(y_pred_valid, y_valid)
print("Validation MAE :", mae)



## === cell 11
test_features = test[FEATURE_COLS]

if test_features.shape[0] == 0:
    raise ValueError(
        "Test feature matrix is empty. Check the merging steps preceding this cell."
    )

test_features[num_cols] = test_features[num_cols].astype(np.float32)

test_pred = gbm.predict(test_features)

submission_df = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": test_pred})

submission_path = "submissions.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
submission_df.head(10)
