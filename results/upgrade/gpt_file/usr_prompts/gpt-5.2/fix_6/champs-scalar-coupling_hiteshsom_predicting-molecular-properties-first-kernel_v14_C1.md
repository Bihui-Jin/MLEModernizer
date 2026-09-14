# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

3.20052

# 6. Current score

2.12217

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2.03465) has done: 'The row-count mismatch is caused by the `merge(..., sort=True)` on `structures`, which reorders rows and breaks alignment with `test_id`; it can also silently drop rows if any key is missing, leading to an empty/incorrect `X_test_new` and then LightGBM predict failures. I make the merges order-preserving (`sort=False`) and validate that no rows were lost, then rebuild `X_train_new`/`X_test_new` with stable indices so training and inference matrices align. I also ensure categorical columns are explicitly passed to LightGBM (same model/logic, but prevents category handling issues), and I write `submission.csv` with the exact sample_submission ordering.'
- What this solution (achieved 2.03465) has done: 'The merge with `structures.csv` is returning missing rows for some `test` pairs because the join keys’ dtypes don’t reliably match across files (notably `atom_index_*`), causing left-join misses and NaNs that trip your assertions and later break distance computation. I minimally enforce consistent dtypes for merge keys (`molecule_name` as string, atom indices as int) in both `train/test` and `structures`, then re-run the exact same merges with `sort=False` and keep your row-preservation/validation logic. I also add a small diagnostic (counts of missing structure matches) right before asserting, to make failures actionable if any remain. This is a correctness/stability fix and should not materially change the modeling logic; your training/prediction flow and submission formatting remain the same.'
- What this solution (achieved 2.12217) has done: 'I fix the merge misses that create NaNs in `X_test` by enforcing consistent join-key dtypes and removing hidden whitespace issues in `molecule_name` and `atom` across all files. Then I add a small diagnostic that reports how many `(molecule_name, atom_index)` keys from test are absent in `structures`, so we can confirm the root cause and fail fast only if truly necessary. Finally, to keep the pipeline running end-to-end and generate a valid submission, I impute any remaining missing structure coordinates/atoms with safe defaults (distance computed with 0s) instead of asserting, which is a stability fix and should only negligibly affect score while preserving the same model and features.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from copy import copy
import category_encoders as ce
import lightgbm as lgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as mae

import os

print(os.listdir("../input"))



## === cell 1
gc.collect()



## === cell 2
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")
structures = pd.read_csv("../input/structures.csv")
print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")



## === cell 3
for df in (train, test, structures):
    df["molecule_name"] = df["molecule_name"].astype(str).str.strip()

for col in ["atom_index_0", "atom_index_1"]:
    train[col] = pd.to_numeric(train[col], errors="raise").astype(np.int32)
    test[col] = pd.to_numeric(test[col], errors="raise").astype(np.int32)

structures["atom_index"] = pd.to_numeric(
    structures["atom_index"], errors="raise"
).astype(np.int32)
structures["atom"] = structures["atom"].astype(str).str.strip()

for c in ["x", "y", "z"]:
    structures[c] = pd.to_numeric(structures[c], errors="coerce").astype(np.float32)



## === cell 4
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()



## === cell 5
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 6
test_id = X_test["id"].copy()
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## === cell 7
X_train = X_train.reset_index(drop=True)
X_train["row_id"] = np.arange(len(X_train), dtype=np.int64)

X_test = X_test.reset_index(drop=True)
X_test["row_id"] = np.arange(len(X_test), dtype=np.int64)




## === cell 8
def convert_object_to_categories(X_train, X_test):
    for col in X_train.columns:
        if X_train[col].dtype == "O":
            X_train[col] = X_train[col].astype("category")
            if col in X_test.columns:
                X_test[col] = X_test[col].astype("category")
    return X_train, X_test


X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 9
import math

print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")


def calc_score(X_train, y_train, y_val):
    X_train_new = X_train.copy()
    y_train_new = y_train.copy()
    y_val_new = y_val.copy()
    y_val_new = pd.Series(y_val_new)
    X_train_new = X_train_new.reset_index(drop=True)
    y_train_new = y_train_new.reset_index(drop=True)
    X_train_new = X_train_new.merge(
        pd.DataFrame(y_train_new, columns=["scalar_coupling_constant"]),
        left_index=True,
        right_index=True,
    )
    X_train_new = X_train_new.merge(
        pd.DataFrame(y_val_new, columns=["y_val"]), left_index=True, right_index=True
    )
    X_train_new["error"] = (
        X_train_new["scalar_coupling_constant"] - X_train_new["y_val"]
    ).abs()
    X_train_new["count"] = 1
    score_df = X_train_new.groupby(by=["type"]).agg({"count": "count", "error": "sum"})
    score_df["error"] = (score_df["error"] / score_df["count"]).apply(
        np.log, dtype=float
    )
    score = (1 / score_df.shape[0]) * (score_df["error"].sum())
    return score




## === cell 10
def cross_val(X, y):
    print(X.shape)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    for train_index, val_index in kf.split(X):
        fold += 1
        lgbm_model = lgbm.LGBMRegressor()
        lgbm_model.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_val = lgbm_model.predict(X.iloc[val_index, :])
        print(
            f"fold{fold} score: {calc_score(X.iloc[val_index, :], y.iloc[val_index], y_val)}"
        )




## === cell 11
def report_missing_structure_keys(df_pairs, structures_df, atom_index_col):
    keys_pairs = (
        df_pairs[["molecule_name", atom_index_col]]
        .rename(columns={atom_index_col: "atom_index"})
        .drop_duplicates()
    )
    keys_struct = structures_df[["molecule_name", "atom_index"]].drop_duplicates()
    merged = keys_pairs.merge(
        keys_struct, on=["molecule_name", "atom_index"], how="left", indicator=True
    )
    missing = (merged["_merge"] == "left_only").sum()
    return int(missing), int(keys_pairs.shape[0])


missing0_keys, total0_keys = report_missing_structure_keys(
    X_test, structures, "atom_index_0"
)
missing1_keys, total1_keys = report_missing_structure_keys(
    X_test, structures, "atom_index_1"
)
print(
    f"Unique (molecule_name, atom_index_0) missing in structures: {missing0_keys}/{total0_keys}"
)
print(
    f"Unique (molecule_name, atom_index_1) missing in structures: {missing1_keys}/{total1_keys}"
)



## === cell 12
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
    validate="many_to_one",
)
X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_0_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)

X_test = X_test.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
    validate="many_to_one",
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_0_0",
        "x": "atom_index_0_x",
        "y": "atom_index_0_y",
        "z": "atom_index_0_z",
        "atom": "atom_0",
    }
)

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed after atom_index_0 merge (train)"
assert (
    X_test.shape[0] == test.shape[0]
), "Row count changed after atom_index_0 merge (test)"

missing0_train = (
    X_train[["atom_index_0_x", "atom_index_0_y", "atom_index_0_z", "atom_0"]]
    .isna()
    .any(axis=1)
    .sum()
)
missing0_test = (
    X_test[["atom_index_0_x", "atom_index_0_y", "atom_index_0_z", "atom_0"]]
    .isna()
    .any(axis=1)
    .sum()
)
print("Missing structure matches for atom_index_0 - train:", int(missing0_train))
print("Missing structure matches for atom_index_0 - test :", int(missing0_test))

if missing0_train > 0 or missing0_test > 0:
    for df_ in (X_train, X_test):
        df_["atom_0"] = df_["atom_0"].astype("object").fillna("X")
        for c in ["atom_index_0_x", "atom_index_0_y", "atom_index_0_z"]:
            df_[c] = (
                pd.to_numeric(df_[c], errors="coerce").fillna(0.0).astype(np.float32)
            )

X_train.head()



## === cell 13
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
    validate="many_to_one",
)
X_train = X_train.rename(
    columns={
        "atom_index": "atom_index_1_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)

X_test = X_test.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
    sort=False,
    validate="many_to_one",
)
X_test = X_test.rename(
    columns={
        "atom_index": "atom_index_1_1",
        "x": "atom_index_1_x",
        "y": "atom_index_1_y",
        "z": "atom_index_1_z",
        "atom": "atom_1",
    }
)

assert (
    X_train.shape[0] == train.shape[0]
), "Row count changed after atom_index_1 merge (train)"
assert (
    X_test.shape[0] == test.shape[0]
), "Row count changed after atom_index_1 merge (test)"

missing1_train = (
    X_train[["atom_index_1_x", "atom_index_1_y", "atom_index_1_z", "atom_1"]]
    .isna()
    .any(axis=1)
    .sum()
)
missing1_test = (
    X_test[["atom_index_1_x", "atom_index_1_y", "atom_index_1_z", "atom_1"]]
    .isna()
    .any(axis=1)
    .sum()
)
print("Missing structure matches for atom_index_1 - train:", int(missing1_train))
print("Missing structure matches for atom_index_1 - test :", int(missing1_test))

if missing1_train > 0 or missing1_test > 0:
    for df_ in (X_train, X_test):
        df_["atom_1"] = df_["atom_1"].astype("object").fillna("X")
        for c in ["atom_index_1_x", "atom_index_1_y", "atom_index_1_z"]:
            df_[c] = (
                pd.to_numeric(df_[c], errors="coerce").fillna(0.0).astype(np.float32)
            )

X_train.head()



## === cell 14
X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])



## === cell 15
X_train.head()



## === cell 16
X_train["distance"] = (
    (X_train["atom_index_0_x"] - X_train["atom_index_1_x"]) ** 2
    + (X_train["atom_index_0_y"] - X_train["atom_index_1_y"]) ** 2
    + (X_train["atom_index_0_z"] - X_train["atom_index_1_z"]) ** 2
) ** 0.5
X_test["distance"] = (
    (X_test["atom_index_0_x"] - X_test["atom_index_1_x"]) ** 2
    + (X_test["atom_index_0_y"] - X_test["atom_index_1_y"]) ** 2
    + (X_test["atom_index_0_z"] - X_test["atom_index_1_z"]) ** 2
) ** 0.5



## === cell 17
X_train["join_type"] = X_train["type"].str.slice(0, 2)
X_test["join_type"] = X_test["type"].str.slice(0, 2)



## === cell 18
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 19
X_train["num_atoms"] = X_train.groupby(["molecule_name", "atom_1"])["atom_1"].transform(
    "count"
)
X_test["num_atoms"] = X_test.groupby(["molecule_name", "atom_1"])["atom_1"].transform(
    "count"
)



## === cell 20
X_train["num_atoms"] = X_train["num_atoms"].astype("str")
X_test["num_atoms"] = X_test["num_atoms"].astype("str")



## === cell 21
X_train["num_atoms"] = X_train["num_atoms"] + X_train["atom_1"]
X_test["num_atoms"] = X_test["num_atoms"] + X_test["atom_1"]



## === cell 22
df = X_train.set_index(keys="row_id", drop=False).merge(
    pd.DataFrame(y_train, columns=["scalar_coupling_constant"]),
    left_index=True,
    right_index=True,
)
corr_numeric = df.select_dtypes(include=[np.number]).corr()
print(corr_numeric.head())



## === cell 23
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 24
X_train_new = X_train.set_index(keys="row_id", drop=True)
X_train_new.head()



## === cell 25
print("Prepared training matrix:", X_train_new.shape)



## === cell 26
X_train_new = X_train_new.sort_index(axis=0)
X_train_new.head()



## === cell 27
X_test_new = X_test.set_index(keys="row_id", drop=True)

missing_cols = [c for c in X_train_new.columns if c not in X_test_new.columns]
for c in missing_cols:
    X_test_new[c] = np.nan
X_test_new = X_test_new[X_train_new.columns]

print("Prepared test matrix:", X_test_new.shape)
assert X_test_new.shape[0] == test.shape[0], "Test feature matrix row count mismatch"
assert X_test_new.shape[1] == X_train_new.shape[1], "Feature column mismatch"



## === cell 28
cat_cols = [c for c in X_train_new.columns if str(X_train_new[c].dtype) == "category"]

lgbm_model = lgbm.LGBMRegressor()
lgbm_model.fit(X_train_new, y_train, categorical_feature=cat_cols)
y_predict = lgbm_model.predict(X_test_new)



## === cell 29
sub = pd.DataFrame({"id": test_id.values, "scalar_coupling_constant": y_predict})

sub = sample_sub[["id"]].merge(sub, on="id", how="left")

assert sub.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert (
    sub["scalar_coupling_constant"].isna().sum() == 0
), "Found missing predictions after id merge"

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
