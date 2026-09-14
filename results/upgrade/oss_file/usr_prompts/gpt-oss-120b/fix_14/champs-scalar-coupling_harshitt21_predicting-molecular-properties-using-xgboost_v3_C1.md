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
xgboost==2.0.3

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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

train = pd.read_csv(
    "../input/champs-scalar-coupling/train.csv",
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test = pd.read_csv(
    "../input/champs-scalar-coupling/test.csv",
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures = pd.read_csv(
    "../input/champs-scalar-coupling/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)
charges = pd.read_csv(
    "../input/champs-scalar-coupling/mulliken_charges.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "mulliken_charge": np.float32,
    },
)
dipole = pd.read_csv(
    "../input/champs-scalar-coupling/dipole_moments.csv",
    dtype={
        "molecule_name": "category",
        "X": np.float32,
        "Y": np.float32,
        "Z": np.float32,
    },
)
potential = pd.read_csv(
    "../input/champs-scalar-coupling/potential_energy.csv",
    dtype={"molecule_name": "category", "potential_energy": np.float32},
)



## === cell 1
train = pd.merge(
    train,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_0"),
)
train = train.rename(
    columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}
).drop(columns=["atom_index"])

train = pd.merge(
    train,
    charges,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_c0"),
)
train = train.rename(columns={"mulliken_charge": "charge_0"}).drop(
    columns=["atom_index"]
)

train = pd.merge(
    train,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_1"),
)
train = train.rename(
    columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}
).drop(columns=["atom_index"])

train = pd.merge(
    train,
    charges,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_c1"),
)
train = train.rename(columns={"mulliken_charge": "charge_1"}).drop(
    columns=["atom_index"]
)

test = pd.merge(
    test,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_0"),
)
test = test.rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"}).drop(
    columns=["atom_index"]
)

test = pd.merge(
    test,
    charges,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_c0"),
)
test = test.rename(columns={"mulliken_charge": "charge_0"}).drop(columns=["atom_index"])

test = pd.merge(
    test,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_1"),
)
test = test.rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"}).drop(
    columns=["atom_index"]
)

test = pd.merge(
    test,
    charges,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    suffixes=("", "_c1"),
)
test = test.rename(columns={"mulliken_charge": "charge_1"}).drop(columns=["atom_index"])

del structures, charges



## === cell 2
train["dist"] = np.sqrt(
    (train["x_1"] - train["x_0"]) ** 2
    + (train["y_1"] - train["y_0"]) ** 2
    + (train["z_1"] - train["z_0"]) ** 2
).astype(np.float32)
train["dx"] = (train["x_1"] - train["x_0"]).astype(np.float32)
train["dy"] = (train["y_1"] - train["y_0"]).astype(np.float32)
train["dz"] = (train["z_1"] - train["z_0"]).astype(np.float32)

test["dist"] = np.sqrt(
    (test["x_1"] - test["x_0"]) ** 2
    + (test["y_1"] - test["y_0"]) ** 2
    + (test["z_1"] - test["z_0"]) ** 2
).astype(np.float32)
test["dx"] = (test["x_1"] - test["x_0"]).astype(np.float32)
test["dy"] = (test["y_1"] - test["y_0"]).astype(np.float32)
test["dz"] = (test["z_1"] - test["z_0"]).astype(np.float32)

train = train.merge(dipole, how="left", on="molecule_name")
train = train.merge(potential, how="left", on="molecule_name")
test = test.merge(dipole, how="left", on="molecule_name")
test = test.merge(potential, how="left", on="molecule_name")

train["charge_diff"] = (train["charge_0"] - train["charge_1"]).astype(np.float32)
train["charge_sum"] = (train["charge_0"] + train["charge_1"]).astype(np.float32)
train["dipole_mag"] = np.sqrt(
    train["X"] ** 2 + train["Y"] ** 2 + train["Z"] ** 2
).astype(np.float32)

test["charge_diff"] = (test["charge_0"] - test["charge_1"]).astype(np.float32)
test["charge_sum"] = (test["charge_0"] + test["charge_1"]).astype(np.float32)
test["dipole_mag"] = np.sqrt(test["X"] ** 2 + test["Y"] ** 2 + test["Z"] ** 2).astype(
    np.float32
)



## === cell 3
for col in ["atom_0", "atom_1", "type"]:
    cat = pd.Categorical(pd.concat([train[col], test[col]], ignore_index=True))
    train[col + "_code"] = cat.codes[: len(train)].astype(np.int16)
    test[col + "_code"] = cat.codes[len(train) :].astype(np.int16)

train["dist_sq"] = (train["dist"] ** 2).astype(np.float32)
test["dist_sq"] = (test["dist"] ** 2).astype(np.float32)

train["inv_dist"] = (1.0 / (train["dist"] + 1e-6)).astype(np.float32)
test["inv_dist"] = (1.0 / (test["dist"] + 1e-6)).astype(np.float32)

train["charge_product"] = (train["charge_0"] * train["charge_1"]).astype(np.float32)
test["charge_product"] = (test["charge_0"] * test["charge_1"]).astype(np.float32)



## === cell 4
features = [
    "atom_index_0",
    "atom_index_1",
    "x_0",
    "y_0",
    "z_0",
    "x_1",
    "y_1",
    "z_1",
    "dx",
    "dy",
    "dz",
    "dist",
    "dist_sq",
    "inv_dist",
    "charge_product",
    "atom_0_code",
    "atom_1_code",
    "type_code",
    "charge_0",
    "charge_1",
    "charge_diff",
    "charge_sum",
    "X",
    "Y",
    "Z",
    "dipole_mag",
    "potential_energy",
]

train = train.dropna(subset=["scalar_coupling_constant"]).reset_index(drop=True)

feature_medians = train[features].median()
train[features] = train[features].fillna(feature_medians)
test[features] = test[features].fillna(feature_medians)



## === cell 5
X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    train[features].astype(np.float32),
    train["scalar_coupling_constant"].astype(np.float32),
    test_size=0.2,
    random_state=42,
)

y_train = np.log1p(y_train_raw).astype(np.float32)
y_val = np.log1p(y_val_raw).astype(np.float32)

valid_train = np.isfinite(y_train)
X_train = X_train[valid_train]
y_train = y_train[valid_train]

valid_val = np.isfinite(y_val)
X_val = X_val[valid_val]
y_val = y_val[valid_val]



## === cell 6
xgb = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    tree_method="hist",
)
xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="mae",
    early_stopping_rounds=150,
    verbose=False,
)
preds_log = xgb.predict(X_val)
preds = np.expm1(preds_log)



## === cell 7
print("Log‑MAE on validation:", np.log(metrics.mean_absolute_error(y_val_raw, preds)))



## === cell 8
test_pred_log = xgb.predict(test[features].astype(np.float32))
test_predictions = np.expm1(test_pred_log)



## === cell 9
submission = pd.DataFrame(
    {"id": test["id"], "scalar_coupling_constant": test_predictions}
)



## === cell 10
submission.to_csv("submission.csv", index=False)
