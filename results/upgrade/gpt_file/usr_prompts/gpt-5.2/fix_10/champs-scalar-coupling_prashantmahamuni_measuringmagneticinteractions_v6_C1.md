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
from sklearn import metrics

np.random.seed(420)



## === cell 1
print(os.listdir("../input"))



## === cell 2
BASE = "../input/champs-scalar-coupling"
if not os.path.exists(BASE):
    BASE = "../input"

train_original = pd.read_csv(
    os.path.join(BASE, "train.csv"),
    usecols=[
        "id",
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "scalar_coupling_constant",
    ],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "scalar_coupling_constant": np.float32,
    },
)
test_original = pd.read_csv(
    os.path.join(BASE, "test.csv"),
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
structures_original = pd.read_csv(
    os.path.join(BASE, "structures.csv"),
    usecols=["molecule_name", "atom_index", "atom", "x", "y", "z"],
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
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
    structures_original.groupby(["molecule_name", "atom"], observed=True)["atom"]
    .count()
    .rename("count")
    .unstack(fill_value=0)
    .reset_index()
)

moleculeCount.head()



## === cell 8
moleculeCount[moleculeCount["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 9
structures = structures_original.merge(moleculeCount, on="molecule_name", how="inner")

structures.head()



## === cell 10
scols = ["molecule_name", "atom_index", "atom", "x", "y", "z", "C", "F", "H", "N", "O"]
s = structures[scols].copy()

s_idx = s.set_index(["molecule_name", "atom_index"], drop=True)

k0 = pd.MultiIndex.from_frame(
    train_original[["molecule_name", "atom_index_0"]].rename(
        columns={"atom_index_0": "atom_index"}
    )
)
k1 = pd.MultiIndex.from_frame(
    train_original[["molecule_name", "atom_index_1"]].rename(
        columns={"atom_index_1": "atom_index"}
    )
)

feat0 = s_idx.reindex(k0)
feat1 = s_idx.reindex(k1)

train = pd.DataFrame(
    {
        "id": train_original["id"].to_numpy(copy=False),
        "molecule_name": train_original["molecule_name"],
        "atom_0": train_original["atom_index_0"].to_numpy(copy=False),
        "atom_1": train_original["atom_index_1"].to_numpy(copy=False),
        "type": train_original["type"],
        "atom_nm_0": feat0["atom"].to_numpy(copy=False),
        "x_0": feat0["x"].to_numpy(copy=False),
        "y_0": feat0["y"].to_numpy(copy=False),
        "z_0": feat0["z"].to_numpy(copy=False),
        "atom_nm_1": feat1["atom"].to_numpy(copy=False),
        "x_1": feat1["x"].to_numpy(copy=False),
        "y_1": feat1["y"].to_numpy(copy=False),
        "z_1": feat1["z"].to_numpy(copy=False),
        "C": feat1["C"].to_numpy(copy=False),
        "F": feat1["F"].to_numpy(copy=False),
        "H": feat1["H"].to_numpy(copy=False),
        "N": feat1["N"].to_numpy(copy=False),
        "O": feat1["O"].to_numpy(copy=False),
        "scalar_coupling_constant": train_original["scalar_coupling_constant"].to_numpy(
            copy=False
        ),
    }
)

train.reset_index(inplace=True, drop=True)

del k0, k1, feat0, feat1
gc.collect()

train.head()



## === cell 11
scols = ["molecule_name", "atom_index", "atom", "x", "y", "z", "C", "F", "H", "N", "O"]
s = structures[scols].copy()
s_idx = s.set_index(["molecule_name", "atom_index"], drop=True)

k0 = pd.MultiIndex.from_frame(
    test_original[["molecule_name", "atom_index_0"]].rename(
        columns={"atom_index_0": "atom_index"}
    )
)
k1 = pd.MultiIndex.from_frame(
    test_original[["molecule_name", "atom_index_1"]].rename(
        columns={"atom_index_1": "atom_index"}
    )
)

feat0 = s_idx.reindex(k0)
feat1 = s_idx.reindex(k1)

test = pd.DataFrame(
    {
        "id": test_original["id"].to_numpy(copy=False),
        "molecule_name": test_original["molecule_name"],
        "atom_0": test_original["atom_index_0"].to_numpy(copy=False),
        "atom_1": test_original["atom_index_1"].to_numpy(copy=False),
        "type": test_original["type"],
        "atom_nm_0": feat0["atom"].to_numpy(copy=False),
        "x_0": feat0["x"].to_numpy(copy=False),
        "y_0": feat0["y"].to_numpy(copy=False),
        "z_0": feat0["z"].to_numpy(copy=False),
        "atom_nm_1": feat1["atom"].to_numpy(copy=False),
        "x_1": feat1["x"].to_numpy(copy=False),
        "y_1": feat1["y"].to_numpy(copy=False),
        "z_1": feat1["z"].to_numpy(copy=False),
        "C": feat1["C"].to_numpy(copy=False),
        "F": feat1["F"].to_numpy(copy=False),
        "H": feat1["H"].to_numpy(copy=False),
        "N": feat1["N"].to_numpy(copy=False),
        "O": feat1["O"].to_numpy(copy=False),
    }
)

test.reset_index(inplace=True, drop=True)

del k0, k1, feat0, feat1, s_idx, s
gc.collect()

test.head()



## === cell 12
train_original = None
del train_original
structures_original = None
del structures_original
test_original = None
del test_original
structures = None
del structures
gc.collect()



## === cell 13
for df in (train, test):
    for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(0.0)

a0 = train[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
a1 = train[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
train["dist"] = np.sqrt(((a0 - a1) ** 2).sum(axis=1)).astype(np.float32, copy=False)

b0 = test[["x_0", "y_0", "z_0"]].to_numpy(dtype=np.float32, copy=False)
b1 = test[["x_1", "y_1", "z_1"]].to_numpy(dtype=np.float32, copy=False)
test["dist"] = np.sqrt(((b0 - b1) ** 2).sum(axis=1)).astype(np.float32, copy=False)

train.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)
test.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)

del a0, a1, b0, b1
gc.collect()



## === cell 14
for df in (train, test):
    df["atom_pair"] = df["atom_nm_0"].astype(str) + "_" + df["atom_nm_1"].astype(str)

for col in ["type", "atom_nm_0", "atom_nm_1", "atom_pair"]:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    all_cats = pd.Index(train[col].cat.categories).union(
        pd.Index(test[col].cat.categories)
    )
    train[col] = train[col].cat.set_categories(all_cats)
    test[col] = test[col].cat.set_categories(all_cats)



## === cell 15
train.head()



## === cell 16
test.head()



## === cell 17
feature_cols = [
    "atom_0",
    "atom_1",
    "type",
    "atom_nm_0",
    "atom_nm_1",
    "atom_pair",
    "C",
    "F",
    "H",
    "N",
    "O",
    "dist",
]
X = train[feature_cols]
y = train["scalar_coupling_constant"]



## === cell 18
rng = np.random.RandomState(420)

params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "learning_rate": 0.05,
    "num_leaves": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.3,
    "metric": "l1",
    "num_threads": int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count(),
}

cat_features = ["type", "atom_nm_0", "atom_nm_1", "atom_pair"]
cat_idx = [feature_cols.index(c) for c in cat_features]

train_types = train["type"].astype(str)
test_types = test["type"].astype(str)
all_types = sorted(
    pd.Index(pd.concat([train_types, test_types], axis=0).unique()).tolist()
)

train_type_groups = train.groupby(train_types, sort=False).indices
test_type_groups = test.groupby(test_types, sort=False).indices

X_train_df = X.copy()
X_test_df = test[feature_cols].copy()

for c in cat_features:
    X_train_df[c] = X_train_df[c].cat.codes.astype(np.int32, copy=False)
    X_test_df[c] = X_test_df[c].cat.codes.astype(np.int32, copy=False)

X_train_np = np.ascontiguousarray(X_train_df.to_numpy(copy=False))
y_np = y.to_numpy(dtype=np.float32, copy=False)
X_test_np = np.ascontiguousarray(X_test_df.to_numpy(copy=False))

train_mol_np = train["molecule_name"].to_numpy(copy=False)
train_types_np = train_types.to_numpy(copy=False)

preds_test = np.empty(test.shape[0], dtype=np.float64)
preds_test[:] = np.nan

oof_mae = []

for t in all_types:
    train_rows = train_type_groups.get(t, np.array([], dtype=np.int64))
    test_rows = test_type_groups.get(t, np.array([], dtype=np.int64))

    if train_rows.size == 0:
        if test_rows.size:
            preds_test[test_rows] = 0.0
        continue

    mols_t = train_mol_np[train_rows]
    uniq_mols = pd.unique(mols_t)
    rng.shuffle(uniq_mols)

    split = int(len(uniq_mols) * 0.8)
    train_mols_arr = uniq_mols[:split]
    valid_mols_arr = uniq_mols[split:]  # kept for semantic equivalence (unused)

    train_mask_t = np.isin(mols_t, train_mols_arr)
    valid_mask_t = ~train_mask_t

    tr_idx = train_rows[train_mask_t]
    va_idx = train_rows[valid_mask_t]

    lgb_train = lgb.Dataset(
        X_train_np[tr_idx],
        y_np[tr_idx],
        categorical_feature=cat_idx,
        free_raw_data=True,
    )
    lgb_eval = lgb.Dataset(
        X_train_np[va_idx],
        y_np[va_idx],
        categorical_feature=cat_idx,
        free_raw_data=True,
    )

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_eval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    y_pred_valid = gbm.predict(X_train_np[va_idx], num_iteration=gbm.best_iteration)
    mae_t = metrics.mean_absolute_error(y_np[va_idx], y_pred_valid)
    oof_mae.append(mae_t)

    if test_rows.size:
        preds_test[test_rows] = gbm.predict(
            X_test_np[test_rows], num_iteration=gbm.best_iteration
        )

    del lgb_train, lgb_eval, gbm, y_pred_valid, train_rows, test_rows, tr_idx, va_idx
    if (len(oof_mae) % 2) == 0:
        gc.collect()

print("Trained per-type models:", len(all_types))
if len(oof_mae) > 0:
    print("Mean per-type valid MAE (not Kaggle metric):", float(np.mean(oof_mae)))



## === cell 19
if np.isnan(preds_test).any():
    missing = int(np.isnan(preds_test).sum())
    raise ValueError(
        f"Missing predictions for {missing} test rows; cannot write a valid submission."
    )

submission_df = pd.DataFrame(
    {
        "id": test["id"].to_numpy(copy=False),
        "scalar_coupling_constant": preds_test,
    }
)

submission_df.sort_values("id", inplace=True)
submission_df.to_csv("submission.csv", header=True, index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
submission_df.head(10)
