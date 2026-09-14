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
import sklearn.metrics as metrics
import lightgbm as lgb

dtypes_train = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
dtypes_test = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}
train = pd.read_csv("../input/train.csv", dtype=dtypes_train)
test = pd.read_csv("../input/test.csv", dtype=dtypes_test)
sub = pd.read_csv("../input/sample_submission.csv")
print(train.shape, test.shape, sub.shape)

train["atom1"] = train["type"].str[2]
train["atom2"] = train["type"].str[3]
test["atom1"] = test["type"].str[2]
test["atom2"] = test["type"].str[3]

train["atom1"] = train["atom1"].astype("category")
train["atom2"] = train["atom2"].astype("category")
test["atom1"] = test["atom1"].astype("category")
test["atom2"] = test["atom2"].astype("category")

for i in range(4):
    le = preprocessing.LabelEncoder()
    train[f"type{i}"] = le.fit_transform(train["type"].str[i])
    test[f"type{i}"] = le.transform(test["type"].str[i])

type_lbl = preprocessing.LabelEncoder()
train["type_label"] = type_lbl.fit_transform(train["type"])
test["type_label"] = type_lbl.transform(test["type"])

struct_dtypes = {
    "molecule_name": "category",
    "atom_index": "int16",
    "atom": "category",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}
structures = pd.read_csv("../input/structures.csv", dtype=struct_dtypes)

struct0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
struct1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)

train = pd.merge(
    train, struct0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
test = pd.merge(
    test, struct0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
train = pd.merge(
    train, struct1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
test = pd.merge(
    test, struct1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
del structures, struct0, struct1

charges = pd.read_csv(
    "../input/mulliken_charges.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": "int16",
        "mulliken_charge": "float32",
    },
)
charges0 = charges.rename(
    columns={"atom_index": "atom_index_0", "mulliken_charge": "charge0"}
)
charges1 = charges.rename(
    columns={"atom_index": "atom_index_1", "mulliken_charge": "charge1"}
)
train = pd.merge(train, charges0, how="left", on=["molecule_name", "atom_index_0"])
test = pd.merge(test, charges0, how="left", on=["molecule_name", "atom_index_0"])
train = pd.merge(train, charges1, how="left", on=["molecule_name", "atom_index_1"])
test = pd.merge(test, charges1, how="left", on=["molecule_name", "atom_index_1"])
del charges, charges0, charges1

dipole = pd.read_csv(
    "../input/dipole_moments.csv",
    dtype={"molecule_name": "category", "X": "float32", "Y": "float32", "Z": "float32"},
)
train = pd.merge(train, dipole, how="left", on="molecule_name")
test = pd.merge(test, dipole, how="left", on="molecule_name")
train["dipole_mag"] = np.sqrt(train["X"] ** 2 + train["Y"] ** 2 + train["Z"] ** 2)
test["dipole_mag"] = np.sqrt(test["X"] ** 2 + test["Y"] ** 2 + test["Z"] ** 2)
del dipole

pot = pd.read_csv(
    "../input/potential_energy.csv",
    dtype={"molecule_name": "category", "potential_energy": "float32"},
)
train = pd.merge(train, pot, how="left", on="molecule_name")
test = pd.merge(test, pot, how="left", on="molecule_name")
del pot

contrib = pd.read_csv(
    "../input/scalar_coupling_contributions.csv",
    dtype={
        "molecule_name": "category",
        "atom_index_0": "int16",
        "atom_index_1": "int16",
        "type": "category",
        "fc": "float32",
        "sd": "float32",
        "pso": "float32",
        "dso": "float32",
    },
)
train = pd.merge(
    train,
    contrib,
    how="left",
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
)
test = pd.merge(
    test,
    contrib,
    how="left",
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
)
del contrib

type_stats = (
    train.groupby("type")["scalar_coupling_constant"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "type_mean", "std": "type_std", "count": "type_count"})
)
train = train.merge(type_stats, on="type", how="left")
test = test.merge(type_stats, on="type", how="left")
train["type_std"] = train["type_std"].fillna(0.0)
test["type_std"] = test["type_std"].fillna(0.0)

train_p0 = train[["x0", "y0", "z0"]].values.astype(np.float32)
train_p1 = train[["x1", "y1", "z1"]].values.astype(np.float32)
test_p0 = test[["x0", "y0", "z0"]].values.astype(np.float32)
test_p1 = test[["x1", "y1", "z1"]].values.astype(np.float32)

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1)

del train_p0, train_p1, test_p0, test_p1

train["dx"] = train["x0"] - train["x1"]
train["dy"] = train["y0"] - train["y1"]
train["dz"] = train["z0"] - train["z1"]
test["dx"] = test["x0"] - test["x1"]
test["dy"] = test["y0"] - test["y1"]
test["dz"] = test["z0"] - test["z1"]

train["dist_to_type_mean"] = train["dist"] / train.groupby("type")["dist"].transform(
    "mean"
)
test["dist_to_type_mean"] = test["dist"] / test.groupby("type")["dist"].transform(
    "mean"
)

train["charge_diff"] = train["charge0"] - train["charge1"]
train["charge_sum"] = train["charge0"] + train["charge1"]
test["charge_diff"] = test["charge0"] - test["charge1"]
test["charge_sum"] = test["charge0"] + test["charge1"]

print(train.shape, test.shape, sub.shape)

for df in (train, test):
    df["atom1"] = df["atom1"].astype("category")
    df["atom2"] = df["atom2"].astype("category")

cat_features = ["type", "atom1", "atom2", "molecule_name"]




## === cell 1
def lgb_lmae(preds, dtrain):
    labels = dtrain.get_label()
    mae = np.mean(np.abs(labels - preds))
    return "lmae", np.log(mae), False


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 512,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "force_col_wise": True,
    "n_jobs": -1,
    "max_bin": 255,
}

feature_exclude = ["id", "scalar_coupling_constant"]
col = [c for c in train.columns if c not in feature_exclude]

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
    train[col], train["scalar_coupling_constant"], test_size=0.2, random_state=99
)

train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_features)
valid_set = lgb.Dataset(
    x_valid, label=y_valid, reference=train_set, categorical_feature=cat_features
)

callbacks = [
    lgb.early_stopping(stopping_rounds=500, verbose=False),
    lgb.log_evaluation(period=100),
]

model = lgb.train(
    params,
    train_set,
    num_boost_round=3000,
    valid_sets=[valid_set],
    feval=lgb_lmae,
    callbacks=callbacks,
)

test["scalar_coupling_constant"] = model.predict(
    test[col], num_iteration=model.best_iteration
)
test[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
