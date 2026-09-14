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
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass


def _resolve_input_path(filename: str) -> str:
    candidates = [
        os.path.join("..", "input", filename),
        os.path.join("..", "input", "champs-scalar-coupling", filename),
        os.path.join("/kaggle", "input", "champs-scalar-coupling", filename),
        os.path.join("/kaggle", "data", "champs-scalar-coupling", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join("..", "input", "champs-scalar-coupling", filename)


train_path = _resolve_input_path("train.csv")
test_path = _resolve_input_path("test.csv")
sub_path = _resolve_input_path("sample_submission.csv")
structures_path = _resolve_input_path("structures.csv")

train = pd.read_csv(
    train_path,
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
test = pd.read_csv(
    test_path,
    usecols=["id", "molecule_name", "atom_index_0", "atom_index_1", "type"],
    dtype={
        "id": np.int32,
        "molecule_name": "category",
        "atom_index_0": np.int16,
        "atom_index_1": np.int16,
        "type": "category",
    },
)
sub = pd.read_csv(sub_path)
print(train.shape, test.shape, sub.shape)

train_type_str = train["type"].astype("string")
test_type_str = test["type"].astype("string")
train["atom1"] = train_type_str.str.slice(2, 3)
train["atom2"] = train_type_str.str.slice(3, 4)
test["atom1"] = test_type_str.str.slice(2, 3)
test["atom2"] = test_type_str.str.slice(3, 4)

for i in range(4):
    lbl = preprocessing.LabelEncoder()
    combined = pd.concat(
        [train_type_str.str.slice(i, i + 1), test_type_str.str.slice(i, i + 1)],
        axis=0,
        ignore_index=True,
    )
    lbl.fit(combined)
    train["type" + str(i)] = lbl.transform(train_type_str.str.slice(i, i + 1)).astype(
        np.int8
    )
    test["type" + str(i)] = lbl.transform(test_type_str.str.slice(i, i + 1)).astype(
        np.int8
    )

structures_base = pd.read_csv(
    structures_path,
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

structures0 = structures_base.rename(
    columns={
        "atom_index": "atom_index_0",
        "x": "x0",
        "y": "y0",
        "z": "z0",
        "atom": "atom1",
    }
)
train = pd.merge(
    train, structures0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)
test = pd.merge(
    test, structures0, how="left", on=["molecule_name", "atom_index_0", "atom1"]
)

structures1 = structures_base.rename(
    columns={
        "atom_index": "atom_index_1",
        "x": "x1",
        "y": "y1",
        "z": "z1",
        "atom": "atom2",
    }
)
train = pd.merge(
    train, structures1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)
test = pd.merge(
    test, structures1, how="left", on=["molecule_name", "atom_index_1", "atom2"]
)

del structures_base, structures0, structures1
print(train.shape, test.shape, sub.shape)



## === cell 1
train_p0 = train[["x0", "y0", "z0"]].to_numpy(copy=False)
train_p1 = train[["x1", "y1", "z1"]].to_numpy(copy=False)
test_p0 = test[["x0", "y0", "z0"]].to_numpy(copy=False)
test_p1 = test[["x1", "y1", "z1"]].to_numpy(copy=False)

train["dist"] = np.linalg.norm(train_p0 - train_p1, axis=1).astype(np.float32)
test["dist"] = np.linalg.norm(test_p0 - test_p1, axis=1).astype(np.float32)

type_mean_dist = train.groupby("type", sort=False)["dist"].mean()
train["dist_to_type_mean"] = (train["dist"] / train["type"].map(type_mean_dist)).astype(
    np.float32
)
test["dist_to_type_mean"] = (test["dist"] / test["type"].map(type_mean_dist)).astype(
    np.float32
)

test["dist_to_type_mean"] = (
    test["dist_to_type_mean"]
    .fillna(test["dist"] / train["dist"].mean())
    .astype(np.float32)
)



## === cell 2
col = [
    c
    for c in train.columns
    if c
    not in ["id", "molecule_name", "scalar_coupling_constant", "type", "atom1", "atom2"]
]


def lgb_lmae(preds, dtrain):
    labels = dtrain.get_label()
    score = np.log(metrics.mean_absolute_error(labels, preds))
    return "lmae", score, False


params = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": max(1, os.cpu_count() or 1),
}

test["scalar_coupling_constant"] = np.nan

callbacks = [
    lgb.early_stopping(stopping_rounds=200),
    lgb.log_evaluation(period=200),
]

all_types = sorted(train["type"].unique().tolist())
print("Training per-type models for", len(all_types), "types:", all_types)

train_type_arr = train["type"].to_numpy()
test_type_arr = test["type"].to_numpy()

train_type_idx = {t: np.flatnonzero(train_type_arr == t) for t in all_types}
test_type_idx = {t: np.flatnonzero(test_type_arr == t) for t in all_types}

X_train_all = train[col].to_numpy(copy=False)
y_train_all = train["scalar_coupling_constant"].to_numpy(copy=False)
X_test_all = test[col].to_numpy(copy=False)

mol_codes_all, mol_uniques = pd.factorize(train["molecule_name"], sort=False)
mol_codes_all = mol_codes_all.astype(np.int32, copy=False)

for t in all_types:
    tr_idx = train_type_idx[t]
    te_idx = test_type_idx[t]
    if te_idx.size == 0:
        continue

    tr_mol_codes = mol_codes_all[tr_idx]
    mols_codes = np.unique(tr_mol_codes)

    mol_train_codes, mol_valid_codes = model_selection.train_test_split(
        mols_codes, test_size=0.2, random_state=99
    )

    mol_train_codes_sorted = np.sort(mol_train_codes)
    in_tr = np.searchsorted(mol_train_codes_sorted, tr_mol_codes, side="left")
    in_tr = (in_tr < mol_train_codes_sorted.size) & (
        mol_train_codes_sorted[in_tr] == tr_mol_codes
    )
    in_va = ~in_tr

    idx_tr = tr_idx[in_tr]
    idx_va = tr_idx[in_va]

    x_train = X_train_all[idx_tr]
    y_train = y_train_all[idx_tr]
    x_valid = X_train_all[idx_va]
    y_valid = y_train_all[idx_va]

    dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=20000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=lgb_lmae,
        callbacks=callbacks,
    )

    preds = model.predict(X_test_all[te_idx], num_iteration=model.best_iteration)
    test.loc[te_idx, "scalar_coupling_constant"] = preds
    print(
        f"Finished type={t}: train_rows={tr_idx.size}, test_rows={te_idx.size}, best_iter={model.best_iteration}"
    )

if test["scalar_coupling_constant"].isna().any():
    fill_value = float(train["scalar_coupling_constant"].mean())
    test["scalar_coupling_constant"] = test["scalar_coupling_constant"].fillna(
        fill_value
    )

submission = test[["id", "scalar_coupling_constant"]].copy()
submission = submission.sort_values("id")
submission.to_csv("submission.csv", float_format="%.9f", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("NaNs in predictions:", submission["scalar_coupling_constant"].isna().sum())
