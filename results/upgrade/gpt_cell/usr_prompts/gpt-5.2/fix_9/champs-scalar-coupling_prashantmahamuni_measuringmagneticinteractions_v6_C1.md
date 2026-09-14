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
import numpy as np
import pandas as pd

import gc
import os

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

SEED = 420
np.random.seed(SEED)

pd.options.mode.chained_assignment = None



## === cell 1
print(os.listdir("../input"))



## === cell 2
train_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
    "scalar_coupling_constant": "float32",
}
test_dtypes = {
    "id": "int32",
    "molecule_name": "category",
    "atom_index_0": "int16",
    "atom_index_1": "int16",
    "type": "category",
}
structures_dtypes = {
    "molecule_name": "category",
    "atom_index": "int16",
    "atom": "category",
    "x": "float32",
    "y": "float32",
    "z": "float32",
}

train_original = pd.read_csv(
    "../input/train.csv",
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype=train_dtypes,
)
structures_original = pd.read_csv(
    "../input/structures.csv",
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype=structures_dtypes,
)
test_original = pd.read_csv(
    "../input/test.csv",
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype=test_dtypes,
)
sample_sub = pd.read_csv(
    "../input/sample_submission.csv", usecols=["id"], dtype={"id": "int32"}
)



## === cell 3
train_original.head()



## === cell 4
structures_original.head()



## === cell 5
test_original.head()



## === cell 6
structures_original[structures_original["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 7
moleculeCount = (
    structures_original.groupby(["molecule_name", "atom"], sort=False)
    .size()
    .unstack(fill_value=0)
)

for el in ["C", "F", "H", "N", "O"]:
    if el not in moleculeCount.columns:
        moleculeCount[el] = 0

moleculeCount = moleculeCount[["C", "F", "H", "N", "O"]].astype(np.int16).reset_index()
moleculeCount.head()



## === cell 8
moleculeCount[moleculeCount["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 9
structures_feat = structures_original.merge(
    moleculeCount, how="left", on="molecule_name", copy=False
)[["molecule_name", "atom_index", "atom", "x", "y", "z", "C", "F", "H", "N", "O"]]

structures_idx = structures_feat.set_index(["molecule_name", "atom_index"], drop=True)


def _attach_atom_side(df_base, atom_col, prefix, need_counts):
    keys = pd.MultiIndex.from_arrays(
        [df_base["molecule_name"].values, df_base[atom_col].values],
        names=["molecule_name", "atom_index"],
    )
    cols = ["atom", "x", "y", "z"]
    if need_counts:
        cols += ["C", "F", "H", "N", "O"]
    got = structures_idx.reindex(keys)[cols]

    got = got.rename(
        columns={
            "atom": f"atom_nm_{prefix}",
            "x": f"x_{prefix}",
            "y": f"y_{prefix}",
            "z": f"z_{prefix}",
        }
    )
    if need_counts:
        got = got.rename(columns={"C": "C", "F": "F", "H": "H", "N": "N", "O": "O"})
    return got.reset_index(drop=True)


side0 = _attach_atom_side(train_original, "atom_index_0", "0", need_counts=True)
side1 = _attach_atom_side(train_original, "atom_index_1", "1", need_counts=False)

train = pd.concat([train_original.reset_index(drop=True), side0, side1], axis=1)

train = train[
    [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "atom_nm_0",
        "x_0",
        "y_0",
        "z_0",
        "atom_nm_1",
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
].rename(columns={"atom_index_0": "atom_0", "atom_index_1": "atom_1"})

train.reset_index(inplace=True, drop=True)

del side0, side1
gc.collect()

train.head()



## === cell 10
side0_t = _attach_atom_side(test_original, "atom_index_0", "0", need_counts=True)
side1_t = _attach_atom_side(test_original, "atom_index_1", "1", need_counts=False)

test = pd.concat([test_original.reset_index(drop=True), side0_t, side1_t], axis=1)

test = test[
    [
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "atom_nm_0",
        "x_0",
        "y_0",
        "z_0",
        "atom_nm_1",
        "x_1",
        "y_1",
        "z_1",
        "C",
        "F",
        "H",
        "N",
        "O",
    ]
].rename(columns={"atom_index_0": "atom_0", "atom_index_1": "atom_1"})

test.reset_index(inplace=True, drop=True)

del side0_t, side1_t
gc.collect()

test.head()



## === cell 11
train_original = None
del train_original
structures_original = None
del structures_original
test_original = None
del test_original
moleculeCount = None
del moleculeCount
structures_feat = None
del structures_feat
structures_idx = None
del structures_idx
gc.collect()



## === cell 12
a0 = train[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
a1 = train[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
train["dist"] = np.sqrt(((a0 - a1) ** 2).sum(axis=1), dtype=np.float32)

b0 = test[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
b1 = test[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test["dist"] = np.sqrt(((b0 - b1) ** 2).sum(axis=1), dtype=np.float32)

train.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)
test.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)

del a0, a1, b0, b1
gc.collect()



## === cell 13
cat_cols = ["atom_0", "atom_1", "type", "atom_nm_1"]

for c in cat_cols:
    combined = pd.concat([train[c], test[c]], axis=0, ignore_index=True)
    dtype = pd.api.types.CategoricalDtype(
        categories=pd.Index(combined.astype("category").cat.categories)
    )
    train[c] = train[c].astype(dtype).cat.codes.astype("int32", copy=False)
    test[c] = test[c].astype(dtype).cat.codes.astype("int32", copy=False)

train.head()



## === cell 14
test.head()



## === cell 15
X = train[["atom_0", "atom_1", "type", "atom_nm_1", "C", "F", "H", "N", "O", "dist"]]
y = train["scalar_coupling_constant"]



## === cell 16
mol = train["molecule_name"].to_numpy()
unique_mol = pd.unique(mol)

mol_train, mol_valid = train_test_split(unique_mol, test_size=0.4, random_state=SEED)

mol_train_set = set(mol_train.tolist())
is_train = np.fromiter((m in mol_train_set for m in mol), dtype=bool, count=len(mol))

X_train = X[is_train]
y_train = y[is_train]
X_test = X[~is_train]
y_test = y[~is_train]

del mol, unique_mol, mol_train, mol_valid, mol_train_set, is_train
gc.collect()



## === cell 17
categorical_feature = ["atom_0", "atom_1", "type", "atom_nm_1"]

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))
y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
y_test_np = y_test.to_numpy(dtype=np.float32, copy=False)

cat_idx = [X.columns.get_loc(c) for c in categorical_feature]

lgb_train = lgb.Dataset(
    X_train_np, y_train_np, free_raw_data=True, categorical_feature=cat_idx
)
lgb_eval = lgb.Dataset(
    X_test_np, y_test_np, free_raw_data=True, categorical_feature=cat_idx
)



## === cell 18
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "metric": "mae",
    "seed": SEED,
    "num_threads": max(1, (os.cpu_count() or 2)),
    "force_col_wise": True,
}
num_boost_round = 5000
early_stopping_rounds = 50



## === cell 19
gbm = lgb.train(
    params=params,
    train_set=lgb_train,
    num_boost_round=num_boost_round,
    valid_sets=[lgb_eval],
    valid_names=["valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
    ],
)



## === cell 20
y_predict = gbm.predict(X_test_np, num_iteration=gbm.best_iteration)
mae = metrics.mean_absolute_error(y_test_np, y_predict)
rmse = np.sqrt(metrics.mean_squared_error(y_test_np, y_predict))
print("Holdout MAE:", mae)
print("Holdout RMSE:", rmse)
print("Best iteration:", gbm.best_iteration)



## === cell 21
del X_train, X_test, y_train, y_test, lgb_train, lgb_eval, y_predict
del X_train_np, X_test_np, y_train_np, y_test_np
gc.collect()

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_np = y.to_numpy(dtype=np.float32, copy=False)

lgb_full = lgb.Dataset(X_np, y_np, free_raw_data=True, categorical_feature=cat_idx)
gbm_full = lgb.train(
    params=params,
    train_set=lgb_full,
    num_boost_round=gbm.best_iteration,
    valid_sets=None,
    valid_names=None,
)



## === cell 22
feature_cols = [
    "atom_0",
    "atom_1",
    "type",
    "atom_nm_1",
    "C",
    "F",
    "H",
    "N",
    "O",
    "dist",
]
X_sub = test[feature_cols]

if X_sub.shape[0] == 0:
    submission_df = sample_sub[["id"]].copy()
    submission_df["scalar_coupling_constant"] = 0.0
else:
    X_sub_np = np.ascontiguousarray(X_sub.to_numpy(dtype=np.float32, copy=False))
    test_pred = gbm_full.predict(X_sub_np, num_iteration=gbm_full.current_iteration())

    pred_df = pd.DataFrame(
        {"id": test["id"].values, "scalar_coupling_constant": test_pred}
    )
    submission_df = sample_sub[["id"]].merge(pred_df, on="id", how="left")

    if submission_df["scalar_coupling_constant"].isna().any():
        submission_df["scalar_coupling_constant"] = submission_df[
            "scalar_coupling_constant"
        ].fillna(0.0)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, header=True, index=False)
print("Wrote:", submission_path, "rows:", len(submission_df))
submission_df.head(10)
