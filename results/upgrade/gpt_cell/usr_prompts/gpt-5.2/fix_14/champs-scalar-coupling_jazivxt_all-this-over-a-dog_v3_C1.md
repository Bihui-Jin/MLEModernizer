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
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing, model_selection, metrics

os.environ.setdefault("PYTHONHASHSEED", "0")

DATA_DIR = "../input/champs-scalar-coupling"

train = pd.read_csv(
    f"{DATA_DIR}/train.csv",
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
    f"{DATA_DIR}/test.csv",
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"id": np.int32})
print(train.shape, test.shape, sub.shape)

tr_type = train["type"].astype(str)
te_type = test["type"].astype(str)

train["atom"] = tr_type.str[3]
test["atom"] = te_type.str[3]

for i in range(4):
    all_chars = pd.concat(
        [tr_type.str[i], te_type.str[i]], axis=0, ignore_index=True
    ).astype(str)
    cat = pd.Categorical(all_chars)
    codes = cat.codes.astype(np.int16)
    ntr = len(train)
    train[f"type{i}"] = codes[:ntr]
    test[f"type{i}"] = codes[ntr:]
    del all_chars, cat, codes

structures = pd.read_csv(
    f"{DATA_DIR}/structures.csv",
    dtype={
        "molecule_name": "category",
        "atom_index": np.int16,
        "atom": "category",
        "x": np.float32,
        "y": np.float32,
        "z": np.float32,
    },
)

s0 = structures.rename(
    columns={
        "atom_index": "atom_index_0",
        "atom": "atom_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
    }
)[["molecule_name", "atom_index_0", "atom_0", "x0", "y0", "z0"]]
train = pd.merge(
    train, s0, how="left", on=["molecule_name", "atom_index_0"], sort=False, copy=False
)
test = pd.merge(
    test, s0, how="left", on=["molecule_name", "atom_index_0"], sort=False, copy=False
)

s1 = structures.rename(
    columns={
        "atom_index": "atom_index_1",
        "atom": "atom_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
    }
)[["molecule_name", "atom_index_1", "atom_1", "x1", "y1", "z1"]]
train = pd.merge(
    train, s1, how="left", on=["molecule_name", "atom_index_1"], sort=False, copy=False
)
test = pd.merge(
    test, s1, how="left", on=["molecule_name", "atom_index_1"], sort=False, copy=False
)

del structures, s0, s1
gc.collect()
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(dtype=np.float32, copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(dtype=np.float32, copy=False)

np.nan_to_num(train_p0, copy=False, nan=0.0)
np.nan_to_num(train_p1, copy=False, nan=0.0)
np.nan_to_num(test_p0, copy=False, nan=0.0)
np.nan_to_num(test_p1, copy=False, nan=0.0)

dtr = train_p0 - train_p1
dte = test_p0 - test_p1

train["dx"] = dtr[:, 0]
train["dy"] = dtr[:, 1]
train["dz"] = dtr[:, 2]
test["dx"] = dte[:, 0]
test["dy"] = dte[:, 1]
test["dz"] = dte[:, 2]

train["dist"] = np.sqrt((dtr * dtr).sum(axis=1)).astype(np.float32)
test["dist"] = np.sqrt((dte * dte).sum(axis=1)).astype(np.float32)

type_dist_mean = train.groupby("type", observed=True)["dist"].mean()
train["dist_to_type_mean"] = (
    (train["dist"] / train["type"].map(type_dist_mean))
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)
test["dist_to_type_mean"] = (
    (test["dist"] / test["type"].map(type_dist_mean))
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .astype(np.float32)
)

all_atoms = pd.concat(
    [train["atom_0"], train["atom_1"], test["atom_0"], test["atom_1"]],
    axis=0,
    ignore_index=True,
).astype(str)
atom_cat = pd.Categorical(all_atoms)
atom_codes = atom_cat.codes.astype(np.int16)
ntr = len(train)
nte = len(test)
train["atom_0_code"] = atom_codes[:ntr]
train["atom_1_code"] = atom_codes[ntr : 2 * ntr]
test["atom_0_code"] = atom_codes[2 * ntr : 2 * ntr + nte]
test["atom_1_code"] = atom_codes[2 * ntr + nte :]
del all_atoms, atom_cat, atom_codes
gc.collect()

del train_p0, train_p1, test_p0, test_p1, dtr, dte
gc.collect()



## === cell 2
contrib = pd.read_csv(
    f"{DATA_DIR}/scalar_coupling_contributions.csv",
    usecols=[
        "molecule_name",
        "atom_index_0",
        "atom_index_1",
        "type",
        "fc",
        "sd",
        "pso",
        "dso",
    ],
    dtype={
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
        "fc": np.float32,
        "sd": np.float32,
        "pso": np.float32,
        "dso": np.float32,
    },
)

train = train.merge(
    contrib,
    on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    how="left",
    sort=False,
    copy=False,
)
del contrib
gc.collect()

col = [
    c
    for c in train.columns
    if c
    not in [
        "id",
        "molecule_name",
        "scalar_coupling_constant",
        "type",
        "atom",
        "atom_0",
        "atom_1",
        "fc",
        "sd",
        "pso",
        "dso",
    ]
]

for c in col:
    if train[c].dtype == "float64":
        train[c] = train[c].astype(np.float32)
    if test[c].dtype == "float64":
        test[c] = test[c].astype(np.float32)


def lgb_champs_metric(preds, dtrain):
    labels = dtrain.get_label()
    mae = metrics.mean_absolute_error(labels, preds)
    score = float(np.log(mae + 1e-12))
    return "champs_lmae", score, False


params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "None",
    "learning_rate": 0.08,
    "num_leaves": 64,
    "min_data_in_leaf": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 99,
    "num_threads": max(1, (os.cpu_count() or 2) - 1),
}

test_pred = np.zeros(len(test), dtype=np.float64)

types = sorted(train["type"].unique())
targets = ["fc", "sd", "pso", "dso"]

train_type_arr = train["type"].to_numpy()
test_type_arr = test["type"].to_numpy()
train_idx_by_type = {t: np.flatnonzero(train_type_arr == t) for t in types}
test_idx_by_type = {t: np.flatnonzero(test_type_arr == t) for t in types}

gss = model_selection.GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=99)

X_train_all = train[col].to_numpy(dtype=np.float32, copy=False)
X_test_all = test[col].to_numpy(dtype=np.float32, copy=False)

y_train_all = {
    tgt: train[tgt].to_numpy(dtype=np.float32, copy=False) for tgt in targets
}

for t in types:
    tr_rows = train_idx_by_type[t]
    te_rows = test_idx_by_type[t]
    if te_rows.size == 0 or tr_rows.size == 0:
        continue

    groups_t = train["molecule_name"].to_numpy()[tr_rows]
    trn_rel, val_rel = next(gss.split(tr_rows, groups=groups_t))

    X_all_t = X_train_all[tr_rows]
    X_tr = np.ascontiguousarray(X_all_t[trn_rel])
    X_va = np.ascontiguousarray(X_all_t[val_rel])
    X_te = np.ascontiguousarray(X_test_all[te_rows])

    te_sum = np.zeros(te_rows.size, dtype=np.float64)

    dtrain = lgb.Dataset(X_tr, label=None, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=None, reference=dtrain, free_raw_data=False)

    dtrain._champs_type = t
    dvalid._champs_type = t

    for tgt in targets:
        y_all_t = y_train_all[tgt][tr_rows]
        y_tr = y_all_t[trn_rel]
        y_va = y_all_t[val_rel]

        dtrain.set_label(y_tr)
        dvalid.set_label(y_va)

        model = lgb.train(
            params,
            dtrain,
            2000,
            valid_sets=[dvalid],
            feval=lgb_champs_metric,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        te_sum += model.predict(X_te, num_iteration=model.best_iteration)
        del model
        gc.collect()

    test_pred[te_rows] = te_sum

    del X_all_t, X_tr, X_va, X_te, dtrain, dvalid, te_sum, groups_t, trn_rel, val_rel
    gc.collect()

test["scalar_coupling_constant"] = test_pred
test[["id", "scalar_coupling_constant"]].to_csv(
    "submission.csv", float_format="%.9f", index=False
)
print(
    "Wrote submission.csv with shape:", test[["id", "scalar_coupling_constant"]].shape
)
