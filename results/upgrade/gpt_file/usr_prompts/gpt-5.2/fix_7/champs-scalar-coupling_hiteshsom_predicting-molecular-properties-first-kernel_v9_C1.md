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

2.91313

# 6. Current score

1.69271

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.92906) has done: 'I fix the categorical conversion bug by ensuring category lists never include nulls, which currently causes `ValueError: Categorical categories cannot be null` after merging structures. Then I make LightGBM accept the dataframe by converting remaining object columns (`molecule_name`, `atom_0`, `atom_1`, `type`) into proper pandas `category` dtype and passing their names via `categorical_feature` in `.fit()`. Finally, I ensure the pipeline trains once on the full engineered feature set, predicts on test, and always writes a valid `submission.csv` with the required columns and correct row alignment. These are minimal changes that unblock execution and should improve score versus the earlier (non-feature-engineered) baseline because the intended distance/atom features are now actually used.'
- What this solution (achieved 1.68951) has done: 'You’re currently training LightGBM on the raw categorical columns before adding the key structure-based features, and you also train a single global model across all coupling `type`s even though the metric averages log-MAE per type (types have very different target scales). To move the score down toward the 2.91313 target with minimal core-logic change, I (1) remove the early “baseline” training so the only training is on the engineered feature set, and (2) keep the same LightGBM regressor but train one model per `type` and predict per `type`, which is a small change that aligns directly with the evaluation. I also add a tiny set of stable, cheap geometric features (coordinate deltas + squared distance) derived from the same merged structure columns to help LightGBM without changing the overall approach. The output remains a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 1.7283) has done: 'Your current score (1.68951, lower is better) is much better than the target (2.91313), so we should *reduce* performance slightly to move closer to the target band without changing the overall approach. The smallest, safest way is to keep the same per-type LightGBM training but make it a bit more regularized/underfit by slightly increasing `min_child_samples`, adding small `lambda_l2`, and reducing tree complexity (`num_leaves`) while keeping the same features and workflow. I also set `random_state` for stability (so you can iterate toward the target more predictably) and keep categorical handling identical. This should worsen the score modestly (increase it) toward ~2.9 while still producing a valid `submission.csv`.'
- What this solution (achieved 1.69232) has done: 'Your current score (1.7283, lower is better) is much better than the target (2.91313), so to move *toward* the target we should intentionally reduce model performance a bit while keeping the same per-type LightGBM workflow and the same features. The most minimal, controllable way is to increase regularization/underfitting by lowering tree count/complexity and adding stronger L2 while keeping everything else identical (no early stopping, no sampling approximations). I also fix the type loop to iterate only over types actually present in the training data (avoids wasted iterations on empty categories), which preserves semantics but improves stability/runtime. The script still runs end-to-end and writes a valid `submission.csv` with the required columns and alignment.'
- What this solution (achieved 1.69271) has done: 'To move your score closer to the (worse) target 2.91313 while keeping the same per-type LightGBM workflow and the same engineered features, I intentionally increase underfitting a bit more in a controlled way. The smallest reliable lever is to reduce tree count and tree expressiveness while increasing regularization (without changing the training approach, features, or metric semantics). I also make the stochastic parts deterministic (set `subsample_freq`) so the resulting degradation is stable run-to-run, helping you iterate toward the target band predictably. The pipeline still train one model per coupling `type` and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import category_encoders as ce  # kept as in original, though unused
import lightgbm as lgbm
from sklearn.model_selection import KFold

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
print(f"structures.shape: {structures.shape}")



## === cell 3
test_ids = test["id"].copy()

X_train = train.drop(columns=["scalar_coupling_constant"]).copy()
y_train = train["scalar_coupling_constant"].copy()
X_test = test.copy()

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")



## === cell 4
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])




## === cell 5
def convert_object_to_categories(X_train_in, X_test_in):
    X_train = X_train_in.copy()
    X_test = X_test_in.copy()

    for col in X_train.columns:
        is_obj = (X_train[col].dtype == "O") or str(X_train[col].dtype).startswith(
            "string"
        )
        if is_obj:
            tr = X_train[col].astype("string")
            te = X_test[col].astype("string")

            all_vals = pd.concat([tr, te], axis=0)
            all_cats = pd.Index(all_vals.dropna().unique())

            X_train[col] = pd.Categorical(tr, categories=all_cats)
            X_test[col] = pd.Categorical(te, categories=all_cats)

    return X_train, X_test


X_train, X_test = convert_object_to_categories(X_train, X_test)



## === cell 6
import math

print(f"{X_train['type'].unique()}")
print(f"{X_test['type'].unique()}")


def calc_score(X_part, y_true, y_pred):
    df = X_part[["type"]].copy()
    df = df.reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    df["scalar_coupling_constant"] = y_true
    df["y_val"] = y_pred
    df["error"] = (df["scalar_coupling_constant"] - df["y_val"]).abs()
    df["count"] = 1
    score_df = df.groupby("type").agg({"count": "count", "error": "sum"})
    score_df["error"] = (score_df["error"] / score_df["count"]).apply(
        np.log, dtype=float
    )
    score = (1 / score_df.shape[0]) * (score_df["error"].sum())
    return score




## === cell 7
def _get_categorical_feature_names(X: pd.DataFrame):
    return [c for c in X.columns if str(X[c].dtype) == "category"]


def cross_val(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    cat_feats = _get_categorical_feature_names(X)

    for tr_idx, va_idx in kf.split(X):
        fold += 1
        model = lgbm.LGBMRegressor()
        model.fit(
            X.iloc[tr_idx, :],
            y.iloc[tr_idx],
            categorical_feature=cat_feats if len(cat_feats) > 0 else "auto",
        )
        y_val = model.predict(X.iloc[va_idx, :])
        print(
            f"fold{fold} score: {calc_score(X.iloc[va_idx, :], y.iloc[va_idx], y_val)}"
        )




## === cell 8
pass



## === cell 9
X_train.head()



## === cell 10
structures.head()



## === cell 11
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_0"],
    right_on=["molecule_name", "atom_index"],
    how="left",
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



## === cell 12
X_train = X_train.merge(
    structures,
    left_on=["molecule_name", "atom_index_1"],
    right_on=["molecule_name", "atom_index"],
    how="left",
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



## === cell 13
X_train = X_train.drop(columns=["atom_index_0_0", "atom_index_1_1"])
X_test = X_test.drop(columns=["atom_index_0_0", "atom_index_1_1"])

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)



## === cell 14
X_train.head()



## === cell 15
dx_tr = X_train["atom_index_0_x"] - X_train["atom_index_1_x"]
dy_tr = X_train["atom_index_0_y"] - X_train["atom_index_1_y"]
dz_tr = X_train["atom_index_0_z"] - X_train["atom_index_1_z"]

dx_te = X_test["atom_index_0_x"] - X_test["atom_index_1_x"]
dy_te = X_test["atom_index_0_y"] - X_test["atom_index_1_y"]
dz_te = X_test["atom_index_0_z"] - X_test["atom_index_1_z"]

X_train["dx"] = dx_tr
X_train["dy"] = dy_tr
X_train["dz"] = dz_tr
X_test["dx"] = dx_te
X_test["dy"] = dy_te
X_test["dz"] = dz_te

X_train["distance_sq"] = dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr
X_test["distance_sq"] = dx_te * dx_te + dy_te * dy_te + dz_te * dz_te

X_train["distance"] = np.sqrt(X_train["distance_sq"])
X_test["distance"] = np.sqrt(X_test["distance_sq"])



## === cell 16
X_train, X_test = convert_object_to_categories(X_train, X_test)




## === cell 17
def fit_predict_by_type(X_train, y_train, X_test, type_col="type"):
    y_pred_test = np.zeros(X_test.shape[0], dtype=np.float64)
    y_pred_train = np.zeros(X_train.shape[0], dtype=np.float64)

    types = pd.Index(pd.Series(X_train[type_col]).dropna().unique())

    lgb_params = dict(
        random_state=42,
        n_estimators=60,  # reduce capacity (worsen toward target)
        learning_rate=0.1,
        num_leaves=9,  # reduce capacity (worsen toward target)
        min_child_samples=400,  # more regularization (worsen toward target)
        subsample=0.8,
        subsample_freq=1,  # deterministic bagging with random_state
        colsample_bytree=0.8,
        reg_lambda=30.0,  # stronger L2 (worsen toward target)
        n_jobs=-1,
    )

    for t in types:
        tr_mask = X_train[type_col] == t
        te_mask = X_test[type_col] == t

        if tr_mask.sum() == 0:
            continue

        model = lgbm.LGBMRegressor(**lgb_params)

        Xtr = X_train.loc[tr_mask].copy()
        ytr = y_train.loc[tr_mask].copy()
        Xte = X_test.loc[te_mask].copy()

        cat_feats = _get_categorical_feature_names(Xtr)
        model.fit(
            Xtr,
            ytr,
            categorical_feature=cat_feats if len(cat_feats) > 0 else "auto",
        )

        y_pred_train[tr_mask.values] = model.predict(Xtr)
        if te_mask.sum() > 0:
            y_pred_test[te_mask.values] = model.predict(Xte)

        del model, Xtr, ytr, Xte
        gc.collect()

    return y_pred_train, y_pred_test




## === cell 18
y_fit, y_predict = fit_predict_by_type(X_train, y_train, X_test, type_col="type")
print(f"training score: {calc_score(X_train, y_train, y_fit)}")



## === cell 19
submission = pd.DataFrame(
    {"id": test_ids.values, "scalar_coupling_constant": y_predict}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
