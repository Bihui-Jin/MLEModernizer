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

category_encoders==2.7.0
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
import gc
import lightgbm as lgbm
from sklearn.model_selection import GroupKFold
import os

DATA_DIR = "/kaggle/data/champs-scalar-coupling"

print("Listing data dir:", DATA_DIR)
print(sorted(os.listdir(DATA_DIR))[:50])

os.environ.setdefault("PYTHONHASHSEED", "0")




## === cell 1
gc.collect()




## === cell 2
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
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", dtype={"id": np.int32})
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

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"structures.shape: {structures.shape}")
print(f"sample_sub.shape: {sample_sub.shape}")




## === cell 3
X_train = train.drop(columns=["scalar_coupling_constant"])
y_train = train["scalar_coupling_constant"]
X_test = test

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")




## === cell 4
test_id = X_test["id"].copy()

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])




## === cell 5
def convert_object_to_categories(X_train_df, X_test_df):
    for col in X_train_df.columns:
        if X_train_df[col].dtype == "O":
            X_train_df[col] = X_train_df[col].astype("category")
            X_test_df[col] = X_test_df[col].astype("category")
    return X_train_df, X_test_df


X_train, X_test = convert_object_to_categories(X_train, X_test)




## === cell 6
def calc_score_from_type(type_arr, y_true, y_pred):
    type_codes, inv = np.unique(type_arr, return_inverse=True)
    err = np.abs(y_true - y_pred)

    sums = np.bincount(inv, weights=err)
    cnts = np.bincount(inv)
    means = sums / cnts
    return float(np.log(means).mean())


def calc_score(X_df, y_true, y_pred):
    t = X_df["type"].to_numpy()
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return calc_score_from_type(t, yt, yp)


print("Train types:", X_train["type"].unique())
print("Test types:", X_test["type"].unique())




## === cell 7
s = structures.set_index(["molecule_name", "atom_index"])[["atom", "x", "y", "z"]]

a0 = s.loc[
    pd.MultiIndex.from_arrays([X_train["molecule_name"], X_train["atom_index_0"]])
].reset_index(drop=True)
a0.columns = ["atom_0", "atom_index_0_x", "atom_index_0_y", "atom_index_0_z"]
X_train = pd.concat([X_train.reset_index(drop=True), a0], axis=1)

a0t = s.loc[
    pd.MultiIndex.from_arrays([X_test["molecule_name"], X_test["atom_index_0"]])
].reset_index(drop=True)
a0t.columns = ["atom_0", "atom_index_0_x", "atom_index_0_y", "atom_index_0_z"]
X_test = pd.concat([X_test.reset_index(drop=True), a0t], axis=1)

a1 = s.loc[
    pd.MultiIndex.from_arrays([X_train["molecule_name"], X_train["atom_index_1"]])
].reset_index(drop=True)
a1.columns = ["atom_1", "atom_index_1_x", "atom_index_1_y", "atom_index_1_z"]
X_train = pd.concat([X_train, a1], axis=1)

a1t = s.loc[
    pd.MultiIndex.from_arrays([X_test["molecule_name"], X_test["atom_index_1"]])
].reset_index(drop=True)
a1t.columns = ["atom_1", "atom_index_1_x", "atom_index_1_y", "atom_index_1_z"]
X_test = pd.concat([X_test, a1t], axis=1)

dx = (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]).to_numpy()
dy = (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]).to_numpy()
dz = (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]).to_numpy()
X_train["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

dx = (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]).to_numpy()
dy = (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]).to_numpy()
dz = (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]).to_numpy()
X_test["distance"] = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)

X_train, X_test = convert_object_to_categories(X_train, X_test)

print("After features:")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)

del structures, s, a0, a1, a0t, a1t
gc.collect()




## === cell 8
LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
)


def cross_val_by_type(X_df, y, group_col="molecule_name", type_col="type"):
    types = X_df[type_col]
    groups_all = X_df[group_col]
    y_arr = y.to_numpy()

    unique_types = (
        types.cat.categories
        if pd.api.types.is_categorical_dtype(types)
        else np.sort(types.unique())
    )
    scores = []

    splitter = GroupKFold(n_splits=5)

    for t in [str(x) for x in unique_types]:
        mask = types.astype(str).to_numpy() == t
        if not mask.any():
            continue

        X_t = X_df.loc[mask]
        y_t = y_arr[mask]
        groups_t = groups_all.loc[mask]

        fold_scores = []
        for tr_idx, va_idx in splitter.split(X_t, y_t, groups=groups_t):
            model = lgbm.LGBMRegressor(**LGB_PARAMS)
            model.fit(X_t.iloc[tr_idx, :], y_t[tr_idx])
            y_va = model.predict(X_t.iloc[va_idx, :])
            s = calc_score_from_type(
                X_t.iloc[va_idx, :]["type"].to_numpy(), y_t[va_idx], y_va
            )
            fold_scores.append(s)

        t_mean = float(np.mean(fold_scores))
        scores.append(t_mean)
        print(
            f"type={t}  CV(mean over folds)={t_mean:.6f}  folds_std={float(np.std(fold_scores)):.6f}"
        )

    print(
        f"By-type CV mean (avg over types): {float(np.mean(scores)):.6f}  std_over_types: {float(np.std(scores)):.6f}"
    )
    return scores


cv_scores = cross_val_by_type(
    X_train, y_train, group_col="molecule_name", type_col="type"
)




## === cell 9
type_models = {}
pred_test = np.empty(shape=(len(X_test),), dtype=np.float32)

train_type_str = X_train["type"].astype(str).to_numpy()
test_type_str = X_test["type"].astype(str).to_numpy()

unique_types = np.unique(train_type_str)
unique_types.sort()

for t in unique_types:
    tr_mask = train_type_str == t
    te_mask = test_type_str == t

    model = lgbm.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train.loc[tr_mask, :], y_train.loc[tr_mask])
    type_models[t] = model

    if te_mask.any():
        pred_test[te_mask] = model.predict(X_test.loc[te_mask, :]).astype(np.float32)

pred_train = np.empty(shape=(len(X_train),), dtype=np.float32)
for t, model in type_models.items():
    tr_mask = train_type_str == t
    pred_train[tr_mask] = model.predict(X_train.loc[tr_mask, :]).astype(np.float32)

print(
    f"training score (by-type models): {calc_score(X_train, y_train, pred_train):.6f}"
)




## === cell 10
y_pred_arr = np.asarray(pred_test).reshape(-1)

if len(y_pred_arr) == 0:
    y_pred_arr = np.full(shape=(len(sample_sub),), fill_value=np.nan, dtype=float)
elif len(y_pred_arr) != len(sample_sub):
    raise ValueError(
        f"y_predict length ({len(y_pred_arr)}) does not match sample_sub length ({len(sample_sub)})."
    )

sub = sample_sub.copy()
sub["scalar_coupling_constant"] = y_pred_arr
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
