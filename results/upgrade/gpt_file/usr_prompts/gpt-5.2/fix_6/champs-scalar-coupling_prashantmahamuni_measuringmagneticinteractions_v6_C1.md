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

0.66661

# 6. Current score

1.9121

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.96173) has done: 'We fix the empty-test merge that makes `test` have 0 rows by using `how="left"` (like the train merge) and by retaining `atom_index_0/1` so we can safely join even when indices are missing. We also correct a small categorical-encoding bug: `atom_nm_0` is a categorical feature but the code encodes `atom_nm_1` instead; swapping to `atom_nm_0` is score-improving while keeping the same model and features. Finally, we keep the pipeline end-to-end stable and always write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 3.97889) has done: 'Your current split is row-wise random, which leaks molecule information across train/valid and makes training/early-stopping select a model that doesn’t generalize to the (molecule-disjoint) Kaggle test set, inflating the leaderboard error. I switch the validation split to be by `molecule_name` (still the same LightGBM regressor, same features, same loss/metric) so early stopping chooses iterations that better match the competition’s evaluation setting. I also avoid dropping `molecule_name` from the split inputs (but it still not be used as a feature), and keep the same submission writing logic/format so you get a valid `submission.csv`. These are minimal changes aimed specifically at reducing your large gap from 3.96 toward 0.666.'
- What this solution (achieved 1.81121) has done: 'Your current LightGBM is trained on all coupling types together while the Kaggle metric is an average over types, so a single global model tends to underfit type-specific offsets and hurts the per-type MAE. With minimal change to the core approach (still LightGBM regression on the same features and early stopping), I train one model per `type` and predict the test rows for that type, then concatenate predictions back in `id` order. I also align early stopping’s metric to MAE and use a molecule-disjoint split *within each type* so validation better matches the test setup. This should reduce the leaderboard error substantially (move down from ~3.98 toward your 0.6666 target) without changing feature engineering or the model family.'
- What this solution (achieved 1.9121) has done: 'You’re far from the target (1.81121 vs 0.66661; lower is better), so we need a small but meaningful generalization improvement without changing the core approach (LightGBM per coupling type on the same features). The biggest win with minimal logic change is to train per-type on a molecule-disjoint **train/valid split closer to typical practice (80/20 instead of 60/40)** and to make early stopping less twitchy by increasing patience slightly; both help select a more robust number of trees and usually reduce leaderboard error. I also use `metric='l1'`/`objective='regression_l1'` (still MAE) to align optimization more directly with the competition metric while keeping the same model family and features. The submission writing and feature pipeline stay identical, and it still produces `submission.csv`.'

# 9. Code solution

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

train_original = pd.read_csv(os.path.join(BASE, "train.csv"))
structures_original = pd.read_csv(os.path.join(BASE, "structures.csv"))
test_original = pd.read_csv(os.path.join(BASE, "test.csv"))



## === cell 3
train_original.head()



## === cell 4
structures_original.head()



## === cell 5
test_original.head()



## === cell 6
structures_original[structures_original["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 7
moleculeCount = structures_original.groupby(by=["molecule_name", "atom"])[
    ["atom"]
].count()
moleculeCount.rename(columns={"atom": "count"}, inplace=True)
moleculeCount = moleculeCount.unstack(fill_value=0)
moleculeCount = moleculeCount["count"].reset_index()

moleculeCount.head()



## === cell 8
moleculeCount[moleculeCount["molecule_name"] == "dsgdb9nsd_000015"]



## === cell 9
structures = pd.DataFrame.merge(
    structures_original,
    moleculeCount,
    how="inner",
    left_on=["molecule_name"],
    right_on=["molecule_name"],
)

structures.head()



## === cell 10
tmp_merge = pd.DataFrame.merge(
    train_original,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
)

tmp_merge = tmp_merge.merge(
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
)

tmp_merge.drop(
    columns=["atom_index_x", "atom_index_y", "C_x", "F_x", "H_x", "N_x", "O_x"],
    inplace=True,
)
tmp_merge.columns = [
    "id",
    "molecule_name",
    "atom_0",
    "atom_1",
    "type",
    "scalar_coupling_constant",
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

train = tmp_merge[
    [
        "id",
        "molecule_name",
        "atom_0",
        "atom_1",
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
]
train.sort_values(by=["id", "molecule_name"], inplace=True)
train.reset_index(inplace=True, drop=True)

tmp_merge = None

train.head()



## === cell 11
tmp_merge = pd.DataFrame.merge(
    test_original,
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
)

tmp_merge = tmp_merge.merge(
    structures,
    how="left",
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
)

tmp_merge.drop(
    columns=["atom_index_x", "atom_index_y", "C_x", "F_x", "H_x", "N_x", "O_x"],
    inplace=True,
)
tmp_merge.columns = [
    "id",
    "molecule_name",
    "atom_0",
    "atom_1",
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

test = tmp_merge[
    [
        "id",
        "molecule_name",
        "atom_0",
        "atom_1",
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
]

test.sort_values(by=["id", "molecule_name"], inplace=True)
test.reset_index(inplace=True, drop=True)

tmp_merge = None

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
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]] = df[
        ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]
    ].fillna(0.0)

train["dist"] = np.linalg.norm(
    train[["x_0", "y_0", "z_0"]].values - train[["x_1", "y_1", "z_1"]].values, axis=1
)
test["dist"] = np.linalg.norm(
    test[["x_0", "y_0", "z_0"]].values - test[["x_1", "y_1", "z_1"]].values, axis=1
)

train.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)
test.drop(columns=["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"], inplace=True)



## === cell 14
for col in ["type", "atom_nm_0"]:
    all_cats = pd.Index(
        pd.concat([train[col].astype(str), test[col].astype(str)], axis=0).unique()
    )
    train[col] = pd.Categorical(train[col].astype(str), categories=all_cats)
    test[col] = pd.Categorical(test[col].astype(str), categories=all_cats)



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
}

cat_features = ["type", "atom_nm_0"]

train_types = train["type"].astype(str)
test_types = test["type"].astype(str)
all_types = sorted(
    pd.Index(pd.concat([train_types, test_types], axis=0).unique()).tolist()
)

preds_test = np.empty(test.shape[0], dtype=np.float64)
preds_test[:] = np.nan

oof_mae = []

for t in all_types:
    train_idx = (train_types == t).values
    test_idx = (test_types == t).values

    if train_idx.sum() == 0:
        preds_test[test_idx] = 0.0
        continue

    X_t = X.loc[train_idx]
    y_t = y.loc[train_idx]
    mols_t = train.loc[train_idx, "molecule_name"].dropna().unique()
    rng.shuffle(mols_t)

    split = int(len(mols_t) * 0.8)
    train_mols = set(mols_t[:split])
    valid_mols = set(mols_t[split:])

    train_mask_t = train.loc[train_idx, "molecule_name"].isin(train_mols).values
    valid_mask_t = train.loc[train_idx, "molecule_name"].isin(valid_mols).values

    X_train, y_train = X_t.iloc[train_mask_t], y_t.iloc[train_mask_t]
    X_valid, y_valid = X_t.iloc[valid_mask_t], y_t.iloc[valid_mask_t]

    lgb_train = lgb.Dataset(
        X_train, y_train, categorical_feature=cat_features, free_raw_data=True
    )
    lgb_eval = lgb.Dataset(
        X_valid, y_valid, categorical_feature=cat_features, free_raw_data=True
    )

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_eval],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    y_pred_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    mae_t = metrics.mean_absolute_error(y_valid, y_pred_valid)
    oof_mae.append(mae_t)

    X_submit_t = test.loc[test_idx, feature_cols]
    preds_test[test_idx] = gbm.predict(X_submit_t, num_iteration=gbm.best_iteration)

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
        "id": test["id"].values,
        "scalar_coupling_constant": preds_test,
    }
)

submission_df.sort_values("id", inplace=True)
submission_df.to_csv("submission.csv", header=True, index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
submission_df.head(10)
