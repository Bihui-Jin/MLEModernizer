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

# 5. Target score

0.7869

# 6. Current score

3.00563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'Diagnosis: `X_test_new` becomes empty because earlier merges against `structures` likely dropped all test rows due to key mismatches, and the explicit guard in this cell raises a `ValueError` and stops execution. Since we’re constrained to modify only the failing cell, we can’t change the upstream merges; instead we must make this cell robust to an empty test set without altering model training or prediction semantics when data is present. The minimal safe behavior is to keep the same training/prediction flow when `X_test_new` is non-empty, but if it is empty, return a deterministic fallback prediction vector of the correct length so the next cell can build a submission. This avoids the crash while preserving core logic for the normal (non-empty) case.

Patch summary: Remove the hard failure on empty `X_test_new` and replace it with a deterministic fallback: if `X_test_new` has zero rows, create `y_predict` as a vector filled with the mean of `y_train` (or 0.0 if mean is not finite), with length equal to `sample_sub.shape[0]`. Keep all existing model fitting and normal prediction logic unchanged when `X_test_new` is valid.

Updated cells / Compatibility notes for cell k+1 / Assumptions: Cell 36 expects `y_predict` to exist and have the same length as `sample_sub`; the patch guarantees this even if `X_test_new` is empty. Assumes `sample_sub` and `y_train` are available in scope (they are created in earlier cells) and that producing a fallback prediction is preferable to crashing when the upstream merge produced no test rows.'
- What this solution (achieved 3.00563) has done: 'Diagnosis: The crash happens because `y_predict` is an all-zero-length array at the moment it’s assigned into `sample_sub["scalar_coupling_constant"]`, while `sample_sub` has 467,813 rows. That means the inference loop in the previous cell produced no predictions (or `y_predict` got overwritten/emptied), so the submission assignment fails with a length mismatch. The minimal robust fix in this cell is to defensively rebuild a correctly-sized prediction vector aligned to `sample_sub` length (and/or `test` length if available), rather than assuming `y_predict` is valid.

Patch summary: In cell 29, before assigning into `sample_sub`, add a small guard that (1) converts `y_predict` to a 1D numpy array, (2) checks its length against `len(sample_sub)`, and (3) if mismatched/empty, replaces it with a fallback vector filled with the global mean of `y_train` (or 0.0 if unavailable). This preserves execution determinism and produces a valid `submission.csv` without changing the modeling logic upstream.

Updated cells: Only cell 29 is changed.

Compatibility notes for cell k+1: Cell 29 is the last provided cell; no downstream interface changes. The output `sample_sub` keeps the same columns and row order.

Assumptions: `y_train` is still in scope in cell 29 (it is defined earlier), and using its mean as a safe fallback is acceptable to unblock execution when predictions are missing.'
- What this solution (achieved 3.00563) has done: 'Your current score is far above the target (lower is better), and the biggest driver is that your predictions are not aligned to the required `id` order: after merges you reset/sort indices and then write predictions straight into `sample_sub` without mapping back to `test.id`. I make the smallest change that preserves your per-`type` LightGBM training/prediction logic but fixes prediction-to-row alignment by carrying `test.id` through preprocessing and merging predictions back onto `sample_submission` by `id`. This should substantially reduce the metric (toward the target band) without changing model architecture, loss, or training approach. I also keep a deterministic fallback if any ids are missing after merge so a valid submission is always produced.'
- What this solution (achieved 3.00563) has done: 'Your current score is much worse than the target (lower is better), so we should make a small correctness fix that meaningfully reduces error rather than tuning. The biggest remaining issue is that your CV split and sometimes your training array indexing are inconsistent with the intended molecule-wise split and with the `X_train_new` index, which can lead to training/validation leakage and misaligned labels; I fix this by switching CV to `GroupKFold` on `molecule_name` and by ensuring per-type training uses `y_train` aligned by `X_train_new.index`. I also keep your existing “predict by id merge” submission logic intact, but add a tiny guard to ensure `id` is never missing after merges (so predictions always align). These changes keep the same feature set, same per-type LightGBM approach, and same loss/metric semantics, but should move the score substantially toward your target.'
- What this solution (achieved 3.00563) has done: 'Your current gap to target is large (3.00563 vs 0.7869; lower is better), and the most likely remaining cause is that training labels are being misaligned with `X_train_new` after earlier merges/sorts because `y_train` is still in original row order. I make a minimal, core-logic-preserving fix: explicitly realign `y_train` to `X_train_new.index` right before the per-type LightGBM loop, so each feature row trains with its correct target. I also ensure we never accidentally use `test_id` as a fallback mapping when test rows were dropped/reordered by merges (that can silently attach wrong ids), and instead keep the existing “merge predictions onto sample_submission by id” path as the single source of alignment. These changes keep the same feature engineering and per-type LightGBM training approach, but should substantially reduce the metric toward your target by fixing correctness rather than tuning.'
- What this solution (achieved 3.00563) has done: 'Your current score is much worse than the target (3.00563 vs 0.7869; lower is better), so we should make a small correctness fix rather than tune for peak performance. The biggest likely remaining issue is a subtle but important bug: you convert object columns to categorical separately for train and test, which can assign different category codes across datasets and badly harm LightGBM predictions. I make a minimal change to ensure categorical columns in test use the *same category set/order* as train (and that any unseen test categories become NaN), preserving your feature set and per-type LightGBM loop. This should materially reduce error while keeping the core logic intact and still writing a valid `submission.csv`.'
- What this solution (achieved 3.00563) has done: 'Your score gap to the target is large (3.00563 vs 0.7869; lower is better), so we should focus on a single high-impact correctness fix rather than tuning. The most likely remaining issue is inconsistent categorical encodings created by repeatedly converting object columns to `category` after feature merges, which can silently change category sets/codes and hurt LightGBM generalization. I make the minimal change to enforce that *after all feature engineering*, test categorical columns are aligned to train categories (unseen test categories become NaN) right before modeling. This keeps your per-type LightGBM training/prediction logic intact and still writes a valid `submission.csv`.'
- What this solution (achieved 3.00563) has done: 'Your current score (3.00563, lower is better) is far from the target (0.7869), so we should apply one high-impact correctness fix rather than tuning. The most likely remaining source of large error is inconsistent categorical handling after merges: you repeatedly cast object columns to `category` separately on train/test, which can still leave mismatched category sets at the final modeling stage and hurt LightGBM predictions. I make a minimal change right before modeling to enforce that every categorical column in `X_test_new` uses the exact same categories as `X_train_new` (unseen test categories become NaN), while keeping your per-`type` LightGBM training loop and submission-by-`id` alignment intact. This should reduce MAE per type and move the score substantially toward the target without changing the core approach.'
- What this solution (achieved 3.00563) has done: 'Your score is far from the target (3.00563 vs 0.7869, lower is better), so we should apply one high-impact correctness fix rather than tuning: ensure that training labels are aligned to the exact row order of `X_train_new` after all the merges/sorts. Right now `y_train_aligned` is built using `X_train.index` (which is the post-merge row index), but the labels are keyed by the *pre-merge/reset* `"index"` column; this can silently mismatch features to wrong targets and severely hurt MAE. I rebuild `y_train_aligned` using the preserved `"index"` key (the original row id from before merges), then keep the existing per-`type` LightGBM loop and the existing submission-by-`id` merge logic unchanged. This should materially reduce error toward the target without changing the core modeling approach.'
- What this solution (achieved 3.00563) has done: 'Your score is far above the target (lower is better), so we should make one correctness fix that is very likely to reduce MAE substantially without changing your core per-`type` LightGBM approach. The most impactful remaining bug is label misalignment: after you `reset_index()` you keep the original row id in the `"index"` column, but `y_train_aligned` is currently reindexed against `X_train_new.index` using `0..len(y_train)-1`, which mismatches targets to feature rows after merges/sorting. I rebuild `y_train_aligned` keyed by the original `"index"` from `X_train` (the pre-merge train row id), then keep everything else (features, per-type training loop, submission-by-`id` merge) unchanged. This should move the score strongly toward your target without tuning or altering the modeling semantics.'
- What this solution (achieved 3.00563) has done: 'I keep your per-`type` LightGBM training loop and feature set unchanged, but fix one high-impact correctness issue that can keep your score very high: the molecule-level feature `num_atoms` is currently computed from `atom_index_0` only, which can undercount when `atom_index_1` has the maximum index in that molecule. I replace it with a symmetric computation using the max of both atom indices per molecule for train and test, which should reduce systematic bias in a core feature without altering the modeling approach. Everything else (categorical handling, label alignment by preserved `"index"`, and submission-by-`id` merge) remains the same to keep changes minimal and stable. The submission writing stays identical and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from copy import copy
import category_encoders as ce
import lightgbm as lgbm
from sklearn.model_selection import KFold, GroupKFold
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
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()



## === cell 4
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 5
test_id = X_test["id"].copy()

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## === cell 6
X_train = X_train.reset_index()
X_test = X_test.reset_index()




## === cell 7
def convert_object_to_categories(X_train, X_test):
    obj_cols = [c for c in X_train.columns if X_train[c].dtype == "O"]
    for col in obj_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
        X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)
    return X_train, X_test


X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 8
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




## === cell 9
def cross_val(X, y):
    print(X.shape)
    if "molecule_name" in X.columns:
        splitter = GroupKFold(n_splits=5)
        groups = X["molecule_name"].astype(str).values
        splits = splitter.split(X, y, groups=groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = splitter.split(X)

    fold = 0
    for train_index, val_index in splits:
        fold += 1
        lgbm_model = lgbm.LGBMRegressor()
        lgbm_model.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_val = lgbm_model.predict(X.iloc[val_index, :])
        print(
            f"fold{fold} score: {calc_score(X.iloc[val_index,:], y.iloc[val_index], y_val)}"
        )




## === cell 10
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    sort=True,
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
    sort=True,
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

X_train.head()



## === cell 11
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    sort=True,
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
    sort=True,
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

X_train.head()



## === cell 12
X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])



## === cell 13
X_train.head()



## === cell 14
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



## === cell 15
X_train["join_type"] = X_train["type"].str.slice(0, 2)
X_test["join_type"] = X_test["type"].str.slice(0, 2)



## === cell 16
print(X_train["atom_0"].unique())
print(X_test["atom_0"].unique())



## === cell 17
X_train["num_atoms"] = (
    X_train.groupby("molecule_name")[["atom_index_0", "atom_index_1"]]
    .transform("max")
    .max(axis=1)
    + 1
)
X_test["num_atoms"] = (
    X_test.groupby("molecule_name")[["atom_index_0", "atom_index_1"]]
    .transform("max")
    .max(axis=1)
    + 1
)



## === cell 18
df = X_train.set_index(keys="index", drop=False).merge(
    pd.DataFrame(y_train, columns=["scalar_coupling_constant"]),
    left_index=True,
    right_index=True,
)
df.corr(numeric_only=True)



## === cell 19
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 20
X_train_new = X_train.set_index(keys="index")
X_test_new = X_test.set_index(keys="index")



## === cell 21
X_train_new.head()



## === cell 22
X_train_new.head()



## === cell 23
X_train_new = X_train_new.sort_index(axis=0)
X_test_new = X_test_new.sort_index(axis=0)



## === cell 24
X_train_new.head(50)



## === cell 25
X_train_new["molecule_name"].nunique()



## === cell 26
import matplotlib.pyplot as plt
import seaborn as sn

df = X_train_new.merge(
    pd.DataFrame(data=y_train, columns=["scalar_coupling_constant"]),
    left_index=True,
    right_index=True,
)
df = df.groupby(by=["molecule_name", "type"]).agg(
    {"join_type": "count", "scalar_coupling_constant": "std"}
)



## === cell 27
df.head(50)



## === cell 28
X_train_new, X_test_new = convert_object_to_categories(
    X_train_new.reset_index(), X_test_new.reset_index()
)
X_train_new = X_train_new.set_index("index").sort_index(axis=0)
X_test_new = X_test_new.set_index("index").sort_index(axis=0)

cat_cols = [c for c in X_train_new.columns if str(X_train_new[c].dtype) == "category"]
for c in cat_cols:
    if c in X_test_new.columns:
        X_test_new[c] = (
            X_test_new[c]
            .astype("category")
            .cat.set_categories(X_train_new[c].cat.categories)
        )

y_train_aligned = pd.Series(y_train.values, index=X_train["index"].values).reindex(
    X_train_new.index
)

if y_train_aligned.isna().any():
    y_train_aligned = y_train_aligned.fillna(float(pd.Series(y_train).mean()))

if "test_id" not in globals():
    raise ValueError(
        "Expected 'test_id' from the earlier cell to align predictions by id."
    )

test_id_df = pd.DataFrame({"id": test_id.values})
test_id_df.index.name = "orig_row"
test_id_df = (
    test_id_df.reset_index().rename(columns={"orig_row": "index"}).set_index("index")
)

X_test_new = X_test_new.merge(test_id_df, left_index=True, right_index=True, how="left")

if not isinstance(X_test_new, pd.DataFrame):
    X_test_new = pd.DataFrame(X_test_new)

if "type" not in X_train_new.columns or "type" not in X_test_new.columns:
    raise ValueError(
        "Missing required 'type' column after preprocessing; cannot train per-type models."
    )

y_predict = np.zeros(X_test_new.shape[0], dtype=float)

base_params = dict(
    n_estimators=300,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

types_test = X_test_new["type"].astype(str).unique()

global_mean = float(pd.Series(y_train_aligned).mean())
if not np.isfinite(global_mean):
    global_mean = 0.0

for t in sorted(types_test):
    test_mask = (X_test_new["type"].astype(str) == t).values
    if not np.any(test_mask):
        continue

    train_mask = (X_train_new["type"].astype(str) == t).values
    if not np.any(train_mask):
        y_predict[test_mask] = global_mean
        continue

    X_tr_t = X_train_new.loc[train_mask].copy()
    y_tr_t = y_train_aligned.loc[X_tr_t.index]

    X_te_t = X_test_new.loc[test_mask].copy()

    drop_cols = []
    if "molecule_name" in X_tr_t.columns:
        drop_cols.append("molecule_name")
    if "id" in X_tr_t.columns:
        drop_cols.append("id")
    X_tr_t = X_tr_t.drop(columns=drop_cols, errors="ignore")
    X_te_t = X_te_t.drop(columns=drop_cols, errors="ignore")

    model = lgbm.LGBMRegressor(**base_params)
    model.fit(X_tr_t, y_tr_t)
    y_predict[test_mask] = model.predict(X_te_t)

nan_mask = ~np.isfinite(y_predict)
if np.any(nan_mask):
    y_predict[nan_mask] = global_mean



## === cell 29
pred_df = pd.DataFrame(
    {
        "id": X_test_new["id"].values,
        "scalar_coupling_constant": np.asarray(y_predict, dtype=float).reshape(-1),
    }
)

pred_df = pred_df.groupby("id", as_index=False)["scalar_coupling_constant"].mean()

sub = sample_sub[["id"]].merge(pred_df, on="id", how="left")

fallback = (
    float(pd.Series(y_train_aligned).mean()) if "y_train_aligned" in globals() else 0.0
)
if not np.isfinite(fallback):
    fallback = 0.0
sub["scalar_coupling_constant"] = (
    sub["scalar_coupling_constant"].astype(float).fillna(fallback)
)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Missing predictions filled with fallback:",
    int(sub["scalar_coupling_constant"].isna().sum()),
)
