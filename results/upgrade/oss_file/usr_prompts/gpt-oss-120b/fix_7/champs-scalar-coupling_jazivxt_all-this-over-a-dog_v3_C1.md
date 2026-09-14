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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import lightgbm as lgb

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom"] = train["type"].map(lambda x: str(x)[3])
test["atom"] = test["type"].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train[f"type{i}"] = lbl.fit_transform(train["type"].map(lambda x: str(x)[i]))
    test[f"type{i}"] = lbl.transform(test["type"].map(lambda x: str(x)[i]))

type_lbl = preprocessing.LabelEncoder().fit(train["type"])
train["type_enc"] = type_lbl.transform(train["type"])
test["type_enc"] = type_lbl.transform(test["type"])

structures = pd.read_csv("../input/structures.csv").rename(
    columns={"atom_index": "atom_index_0", "x": "x0", "y": "y0", "z": "z0"}
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_0", "atom"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_0", "atom"]
)
del structures

structures = pd.read_csv("../input/structures.csv").rename(
    columns={"atom_index": "atom_index_1", "x": "x1", "y": "y1", "z": "z1"}
)
train = pd.merge(
    train, structures, how="left", on=["molecule_name", "atom_index_1", "atom"]
)
test = pd.merge(
    test, structures, how="left", on=["molecule_name", "atom_index_1", "atom"]
)
del structures

dipole = pd.read_csv("../input/dipole_moments.csv")
dipole["dipole_mag"] = np.sqrt((dipole[["X", "Y", "Z"]] ** 2).sum(axis=1))
train = train.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)
test = test.merge(
    dipole[["molecule_name", "dipole_mag"]], on="molecule_name", how="left"
)

pot = pd.read_csv("../input/potential_energy.csv")
train = train.merge(pot, on="molecule_name", how="left")
test = test.merge(pot, on="molecule_name", how="left")

charges = pd.read_csv("../input/mulliken_charges.csv")
charges0 = charges.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "charge0"}
)
charges1 = charges.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "charge1"}
)
train = train.merge(
    charges0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
test = test.merge(
    charges0[["molecule_name", "atom_index_0", "charge0"]],
    on=["molecule_name", "atom_index_0"],
    how="left",
)
train = train.merge(
    charges1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)
test = test.merge(
    charges1[["molecule_name", "atom_index_1", "charge1"]],
    on=["molecule_name", "atom_index_1"],
    how="left",
)

train["charge_sum"] = train["charge0"] + train["charge1"]
train["charge_diff"] = train["charge0"] - train["charge1"]
test["charge_sum"] = test["charge0"] + test["charge1"]
test["charge_diff"] = test["charge0"] - test["charge1"]

contrib = pd.read_csv("../input/scalar_coupling_contributions.csv")
train = train.merge(
    contrib, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)
test = test.merge(
    contrib, on=["molecule_name", "atom_index_0", "atom_index_1", "type"], how="left"
)

print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].fillna(0).values
train_p1 = train[["x1", "y1", "z1"]].fillna(0).values
test_p0 = test[["x0", "y0", "z0"]].fillna(0).values
test_p1 = test[["x1", "y1", "z1"]].fillna(0).values

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

train["dist_to_type_mean"] = train["dist"] / train.groupby("type")["dist"].transform(
    "mean"
)
test["dist_to_type_mean"] = test["dist"] / test.groupby("type")["dist"].transform(
    "mean"
)



## === cell 2
col = [
    c
    for c in train.columns
    if c not in ["id", "molecule_name", "scalar_coupling_constant"]
]

if "type" in train.columns:
    train["type"] = train["type"].astype("category")
    test["type"] = test["type"].astype("category")
if "atom" in train.columns:
    train["atom"] = train["atom"].astype("category")
    test["atom"] = test["atom"].astype("category")

type_counts = train["type"].value_counts()
train["sample_weight"] = train["type"].map(lambda t: 1.0 / type_counts[t])
train["sample_weight"] = train["sample_weight"].astype(float)

test["sample_weight"] = 1.0  # dummy; not used for prediction


def lgb_lmae_weighted(preds, dtrain):
    labels = dtrain.get_label()
    weight = dtrain.get_weight()
    mae = np.average(np.abs(labels - preds), weights=weight)
    return "lmae_weighted", np.log(mae + 1e-15), False  # lower is better


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "none",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "categorical_feature": ["type"],  # treat original type string as categorical
}

x_train, x_valid, y_train, y_valid, w_train, w_valid = model_selection.train_test_split(
    train[col],
    train["scalar_coupling_constant"],
    train["sample_weight"],
    test_size=0.2,
    random_state=99,
)

train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, reference=None)
valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, reference=train_set)

model = lgb.train(
    params,
    train_set,
    num_boost_round=5000,
    valid_sets=[valid_set],
    feval=lgb_lmae_weighted,
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

test["scalar_coupling_constant"] = model.predict(
    test[col], num_iteration=model.best_iteration
)

test[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
