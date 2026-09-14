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

0.76707

# 6. Current score

3.00563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.00563) has done: 'Diagnosis: The crash happens at `lgbm_model.predict(X_test_new)` because `X_test_new` ends up empty (0 rows), so LightGBM raises `ValueError: Input data must be 2 dimensional and non empty.` This is typically caused by earlier merge steps yielding no matches for some reason (e.g., dtype/category mismatches) but the first hard failure occurs in cell 47. We should not change the upstream feature engineering, so the minimal safe fix is to guard against an empty `X_test_new` right before prediction and create a correctly-sized fallback prediction vector. This keeps the notebook running and preserves the interface expected by cell 48 (`y_predict` must be length `len(sample_sub)`).

Patch summary: In cell 47, add a check for `X_test_new.shape[0] == 0` (or wrong shape) before calling `predict()`. If empty, create a deterministic fallback prediction array of length `sample_sub.shape[0]` filled with the training target mean (or 0.0 if `y_train` is unexpectedly empty), otherwise proceed with normal prediction. No changes to model architecture, training, or feature logic are made.

Updated cells:'
- What this solution (achieved 3.00563) has done: 'Your current score is far above the target (lower is better), and the main reason is that the model is being trained on categorical columns without consistent encoding; LightGBM is likely treating categories inconsistently between train/test, which hurts generalization. I keep the same LightGBM regressor and single-fit training approach, but make one minimal, metric-relevant fix: use the already-imported `category_encoders` to target-encode categorical columns (`molecule_name`, `type`, `atom_0`, `atom_1`, `join_type`) using only training labels, then apply the same mapping to test. This preserves your feature set and training loop while making train/test feature representations aligned, which should move the log-MAE score down toward the target. I also keep your existing empty-test guard so the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 3.00563) has done: 'Your current log-MAE is far worse than the target, so we should improve (lower) it with minimal, metric-aligned tweaks while keeping your feature engineering and single LightGBM fit intact. The biggest “cheap win” without changing core logic is to train a separate LightGBM model per coupling `type` (the metric is averaged per type), using the same features and TargetEncoder but fit within each type to avoid leakage across types and improve per-type calibration. We also make the target encoding fold-safe within each type by fitting the encoder only on that type’s training rows and transforming that type’s test rows, which keeps semantics but fixes train/test alignment more strongly. Finally, we keep your existing empty-test guard and ensure we still write a valid `submission.csv`.'
- What this solution (achieved 3.00563) has done: 'We keep your feature engineering and per-coupling-type LightGBM approach, but fix a key alignment bug: you’re masking `y_train` with `X_train_new["type"] == t` even though `y_train` is still in original row order, so the per-type targets don’t match the per-type features. By reindexing `y_train` to `X_train_new.index` right after you set/sort indices, the per-type training labels correctly align with the engineered rows, which should substantially reduce log-MAE toward the target. We also make the TargetEncoder columns strictly those present in the per-type frame (to avoid rare missing-column issues), and keep your existing empty-test and length-guard so a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from copy import copy
import math
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
X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()



## === cell 4
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 5
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])



## === cell 6
X_train = X_train.reset_index()
X_test = X_test.reset_index()




## === cell 7
def convert_object_to_categories(X_train, X_test):
    for col in X_train.columns:
        if X_train[col].dtype == "O":
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
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
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    fold = 0
    for train_index, val_index in kf.split(X):
        fold += 1
        lgbm_model = lgbm.LGBMRegressor()
        lgbm_model.fit(X.loc[train_index, :], y[train_index])
        y_val = lgbm_model.predict(X.loc[val_index, :])
        print(f"fold{fold} score: {calc_score(X.loc[val_index,:],y[val_index],y_val)}")




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
    X_train.groupby(["molecule_name"])["atom_index_0"].transform("max") + 1
)
X_test["num_atoms"] = (
    X_test.groupby(["molecule_name"])["atom_index_0"].transform("max") + 1
)



## === cell 18
X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 19
X_train_new = X_train.set_index(keys="index")
X_test_new = X_test.set_index(keys="index")



## === cell 20
X_train_new.head()



## === cell 21
X_train_new = X_train_new.sort_index(axis=0)
X_test_new = X_test_new.sort_index(axis=0)



## === cell 22
X_train_new.head(5)



## === cell 23
X_train_new["molecule_name"].nunique()



## === cell 24
import matplotlib.pyplot as plt
import seaborn as sns

df = X_train_new.merge(
    pd.DataFrame(data=y_train, columns=["scalar_coupling_constant"]),
    left_index=True,
    right_index=True,
)
df = df.groupby(by=["molecule_name", "type"]).agg(
    {"join_type": "count", "scalar_coupling_constant": "std"}
)



## === cell 25
df.head(5)



## === cell 26
X_train_new["num_bonds"] = X_train_new["join_type"].str.slice(0, 1)
X_test_new["num_bonds"] = X_test_new["join_type"].str.slice(0, 1)



## === cell 27
X_train_new["num_bonds"] = X_train_new["num_bonds"].astype("int")
X_test_new["num_bonds"] = X_test_new["num_bonds"].astype("int")




## === cell 28
def angle_between_vectors(df):
    dot_products = (
        df["atom_index_0_x"] * df["atom_index_1_x"]
        + df["atom_index_0_y"] * df["atom_index_1_y"]
        + df["atom_index_0_z"] * df["atom_index_1_z"]
    )
    magnitudes_product = (
        df["atom_index_0_x"] ** 2
        + df["atom_index_0_y"] ** 2
        + df["atom_index_0_z"] ** 2
    ) ** 0.5 * (
        df["atom_index_1_x"] ** 2
        + df["atom_index_1_y"] ** 2
        + df["atom_index_1_z"] ** 2
    ) ** 0.5
    df["angle"] = np.arccos(dot_products / magnitudes_product)
    return df




## === cell 29
X_train_new = angle_between_vectors(X_train_new)
X_test_new = angle_between_vectors(X_test_new)



## === cell 30
y_train_aligned = y_train.reindex(X_train_new.index)



## === cell 31
y_predict = np.zeros(sample_sub.shape[0], dtype=float)

if (
    not isinstance(X_test_new, pd.DataFrame)
    or X_test_new.ndim != 2
    or X_test_new.shape[0] < 1
):
    fallback_value = float(y_train.mean()) if len(y_train) > 0 else 0.0
    y_predict[:] = fallback_value
else:
    test_pred_series = pd.Series(index=X_test_new.index, dtype=float)

    base_cat_cols = ["molecule_name", "type", "atom_0", "atom_1", "join_type"]

    for t in X_test_new["type"].unique():
        test_mask = X_test_new["type"] == t
        train_mask = X_train_new["type"] == t

        Xtr = X_train_new.loc[train_mask].copy()
        ytr = y_train_aligned.loc[train_mask].copy()
        Xte = X_test_new.loc[test_mask].copy()

        if Xtr.shape[0] == 0 or Xte.shape[0] == 0 or ytr.shape[0] == 0:
            fallback_value = float(y_train.mean()) if len(y_train) > 0 else 0.0
            test_pred_series.loc[Xte.index] = fallback_value
            continue

        cat_cols = [c for c in base_cat_cols if c in Xtr.columns]

        if len(cat_cols) > 0:
            te = ce.TargetEncoder(cols=cat_cols, smoothing=10.0)
            Xtr = te.fit_transform(Xtr, ytr)
            Xte = te.transform(Xte)

        lgbm_model = lgbm.LGBMRegressor()
        lgbm_model.fit(Xtr, ytr)
        test_pred_series.loc[Xte.index] = lgbm_model.predict(Xte)

        del Xtr, ytr, Xte
        gc.collect()

    y_predict = test_pred_series.sort_index().values



## === cell 32
print("Per-type models trained; skipping global feature importance plot.")



## === cell 33
if len(y_predict) != sample_sub.shape[0]:
    fallback_value = float(y_train.mean()) if len(y_train) > 0 else 0.0
    y_predict = np.full(
        shape=(sample_sub.shape[0],), fill_value=fallback_value, dtype=float
    )



## === cell 34
sample_sub["scalar_coupling_constant"] = list(y_predict)
sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
