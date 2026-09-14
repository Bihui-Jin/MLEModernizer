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
Predict the formation energy and bandgap energy of a material.

## Metric
Column-wise root mean squared logarithmic error.

## Submission Format
For each id in the test set, you must predict a value for both formation_energy_ev_natom and bandgap_energy_ev. The file should contain a header and have the following format:
```
id,formation_energy_ev_natom,bandgap_energy_ev
1,0.1779,1.8892
2,0.1779,1.8892
3,0.1779,1.8892
...
```

## Dataset
The following information has been included:

- Spacegroup (a label identifying the symmetry of the material)
- Total number of Al, Ga, In and O atoms in the unit cell ($\N_{total}$)
- Relative compositions of Al, Ga, and In (x, y, z)
- Lattice vectors and angles: lv1, lv2, lv3 (which are lengths given in units of angstroms ($10^{-10}$ meters) and $\alpha, \beta, \gamma$ (which are angles in degrees between 0° and 360°)

Note: For each line of the CSV file, the corresponding spatial positions of all of the atoms in the unit cell (expressed in Cartesian coordinates) are provided as a separate file.

train.csv - contains a set of materials for which the bandgap and formation energies are provided

test.csv - contains the set of materials for which you must predict the bandgap and formation energies

/{train|test}/{id}/geometry.xyz - files with spatial information about the material. The file name corresponds to the id in the respective csv files.

# 2. Python version

3.6

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        input/
            description.md (89 lines)
            sample_submission.csv (241 lines)
            sample_submission.csv.zip (765 Bytes)
            test.csv (241 lines)
            test.csv.zip (6.0 kB)
            test.zip (505.0 kB)
            train.csv (2161 lines)
            train.csv.zip (56.7 kB)
            train.zip (4.5 MB)
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
            test/
                1/
                    geometry.xyz (3.0 kB)
                10/
                    geometry.xyz (3.0 kB)
                ... and 239 other folders
            train/
                1/
                    geometry.xyz (5.5 kB)
                10/
                    geometry.xyz (2.3 kB)
                ... and 2159 other folders
        working/
            nomad2018-predict-transparent-conductors/
                description.md (89 lines)
                sample_submission.csv (241 lines)
                ... and 7 other files
                nomad2018-predict-transparent-conductors/
                test/
                    1/
                        geometry.xyz (3.0 kB)
                    10/
                        geometry.xyz (3.0 kB)
                    ... and 239 other folders
                train/
                    1/
                        geometry.xyz (5.5 kB)
                    10/
                        geometry.xyz (2.3 kB)
                    ... and 2159 other folders
```

-> data/nomad2018-predict-transparent-conductors/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/nomad2018-predict-transparent-conductors/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/nomad2018-predict-transparent-conductors/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> data/sample_submission.csv has 240 rows and 3 columns.
The columns are: id, formation_energy_ev_natom, bandgap_energy_ev

-> data/test.csv has 240 rows and 12 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree

-> data/train.csv has 2160 rows and 14 columns.
The columns are: id, spacegroup, number_of_total_atoms, percent_atom_al, percent_atom_ga, percent_atom_in, lattice_vector_1_ang, lattice_vector_2_ang, lattice_vector_3_ang, lattice_angle_alpha_degree, lattice_angle_beta_degree, lattice_angle_gamma_degree, formation_energy_ev_natom, bandgap_energy_ev

-> (stopped after 10 files for performance)

# 5. Target score

0.07034

# 6. Current score

0.08088

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09557) has done: 'I update deprecated/removed scikit-learn imports (e.g., `sklearn.cross_validation`, `sklearn.grid_search`) to their modern equivalents so the notebook runs in your current environment. I also fix the train/test slicing bug (it hard-coded 2400/3000 instead of using the actual train/test sizes), which currently breaks training/inference alignment and prevents a valid submission. Since the metric is RMSLE, I add a minimal, score-improving but semantics-preserving safety step by clipping predictions to be non-negative before writing the submission (RMSLE is undefined for negatives). Finally, I keep the original Ridge/Lasso/XGBoost core approach and ensure a valid `XGB_Nomad.csv` is always produced with the required columns.'
- What this solution (achieved 0.08085) has done: 'I fix the runtime errors by removing invalid `alpha=0` entries from `RidgeCV`/`LassoCV` alpha grids (newer scikit-learn requires strictly positive alphas for these CV estimators), which currently prevents the notebook from training and producing the Ridge submission. To move the score toward your (lower-is-better) target with minimal semantic change, I align the linear models with the RMSLE metric by training them on `log1p(y)` and converting predictions back with `expm1`, keeping the same Ridge/Lasso approach. I also ensure RMSLE-valid outputs by clipping predictions to be non-negative and by writing at least one valid submission CSV (keeping the existing `XGB_Nomad.csv` output as well). All changes are localized to the failing alpha lists and target transforms; the rest of the pipeline stays the same.'
- What this solution (achieved 0.08085) has done: 'You’re currently worse than the target (0.08085 vs 0.07034; lower is better), so the smallest score-moving change is to make the XGBoost formation-energy model consistent with the RMSLE metric the same way you already did for Ridge/Lasso and for XGB bandgap: train on `log1p(y)` and invert with `expm1`. This keeps the same core XGBoost CV + fit flow, but aligns the objective with the competition evaluation and typically reduces RMSLE without changing feature logic. I also make the XGB CV use `rmse` on the log-target explicitly (still `rmse`, but now in the same space you score), and keep the non-negativity clipping for RMSLE validity. The script still write all three submissions (`Ridge_Nomad.csv`, `Lasso_Nomad.csv`, `XGB_Nomad.csv`) end-to-end.'
- What this solution (achieved 0.08085) has done: 'Your current score (0.08085) is worse than the target (0.07034; lower is better), so we should make a small, metric-aligned improvement without changing the modeling approach. The biggest low-risk gain here is to fix a parameter inconsistency: the XGB formation-energy CV is done with one parameter set (depth=3, subsample=1, etc.) but the final fitted model uses a different set (depth=2 and missing several params), which typically hurts generalization and RMSLE. I make the final EF XGBRegressor match the CV-tuned parameters exactly (and keep the same log1p/expm1 target handling and clipping), which should improve the score while preserving core logic. I also make the random seed deterministic for numpy to reduce run-to-run variance (no semantic change).'
- What this solution (achieved 0.08085) has done: 'You’re currently worse than the target (0.08085 vs 0.07034; lower is better), so the smallest likely improvement is to better match the evaluation metric without changing the model family or feature logic. The competition uses *column-wise* RMSLE; your CV uses `neg_mean_squared_log_error` but on **log1p targets**, which is a mismatch and can mislead model selection. I keep the same Ridge/Lasso/XGB approach, but fix CV to compute RMSLE correctly in the original target space via a `TransformedTargetRegressor` wrapper for the linear models (log1p/expm1 stays, but CV becomes consistent). For XGBoost, I replace the internal `rmse`-on-log CV with an explicit 5-fold RMSLE CV in the original space while keeping the same params and still using the resulting best number of boosting rounds for final training/prediction.'
- What this solution (achieved 0.08088) has done: 'You’re currently worse than the target (0.08085 vs 0.07034; lower is better), so we make the smallest metric-aligned change likely to improve generalization without changing the model families or feature logic. The main issue is that `get_dummies` is applied on the concatenated train+test, which leaks test distribution into training and can slightly hurt leaderboard performance; we instead fit the one-hot categories on train only and then align test columns to train (no leakage, same semantics). To avoid instability from rare/unseen spacegroups in test, we keep an “other/unseen” bucket by aligning columns and filling missing with 0 so the models remain compatible. Everything else (Ridge/Lasso with log1p targets, XGB with log1p targets and RMSLE CV, clipping to non-negative, and CSV outputs) stays the same.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import numpy as np
import pandas as pd
import glob
import io
import math
import matplotlib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    KFold,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_log_error

import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    Ridge,
    RidgeCV,
    ElasticNet,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)

CANDIDATE_INPUTS = [
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/input",
    "../input",
    "../input/nomad2018-predict-transparent-conductors",
]
INPUT_DIR = None
for p in CANDIDATE_INPUTS:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            INPUT_DIR = p
            break
        if os.path.exists(
            os.path.join(p, "nomad2018-predict-transparent-conductors", "train.csv")
        ):
            INPUT_DIR = os.path.join(p, "nomad2018-predict-transparent-conductors")
            break

if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory with train.csv/test.csv."
    )

print("Using INPUT_DIR:", INPUT_DIR)
print("Top-level files:", sorted(os.listdir(INPUT_DIR))[:20])



## === cell 1
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))



## === cell 2
print("Training data shape")
print(train_df.shape)
print("Testing data shape")
print(test_df.shape)



## === cell 3
print("Training columns")
print(train_df.columns)
print("Testing columns")
print(test_df.columns)



## === cell 4
print(train_df.dtypes)
print(test_df.dtypes)



## === cell 5
Targets_df = pd.DataFrame()
Targets_df["bandgap_energy_ev"] = train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"] = train_df["formation_energy_ev_natom"].copy()
train_df = train_df.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)



## === cell 6
train_id_df = pd.DataFrame()
train_id_df["id"] = train_df["id"].copy()
train_df = train_df.drop(["id"], axis=1)

test_id_df = pd.DataFrame()
test_id_df["id"] = test_df["id"].copy()
test_df = test_df.drop(["id"], axis=1)



## === cell 7
num_cols = [
    "number_of_total_atoms",
    "percent_atom_al",
    "percent_atom_ga",
    "percent_atom_in",
    "lattice_vector_1_ang",
    "lattice_vector_2_ang",
    "lattice_vector_3_ang",
    "lattice_angle_alpha_degree",
    "lattice_angle_beta_degree",
    "lattice_angle_gamma_degree",
]

train_num = train_df[num_cols].copy()
test_num = test_df[num_cols].copy()

train_cat = pd.get_dummies(
    train_df[["spacegroup"]].copy(), prefix=["spacegroup"], columns=["spacegroup"]
)
test_cat = pd.get_dummies(
    test_df[["spacegroup"]].copy(), prefix=["spacegroup"], columns=["spacegroup"]
)

test_cat = test_cat.reindex(columns=train_cat.columns, fill_value=0)

training_examples = pd.concat([train_num, train_cat], axis=1)
test_examples = pd.concat([test_num, test_cat], axis=1)

print("training_examples shape:", training_examples.shape)
print("test_examples shape:", test_examples.shape)



## === cell 8
print("Total number of null values in train df")
print(training_examples.isna().sum().sum())
print("Total number of null values in test df")
print(test_examples.isna().sum().sum())




## === cell 9
def rmsle_cv_ttr(base_model, y_raw, cv=5, random_state=42):
    ttr = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    msle = -cross_val_score(
        ttr,
        training_examples,
        y_raw,
        scoring="neg_mean_squared_log_error",
        cv=cv,
        n_jobs=None,
    )
    return np.sqrt(msle)


def xgb_rmsle_cv_num_boost_round(
    params, X, y_raw, num_boost_round=500, early_stopping_rounds=100, nfold=5, seed=42
):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    best_iters = []
    fold_rmsle = []
    for tr_idx, va_idx in kf.split(X):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_tr_raw = y_raw.iloc[tr_idx].values
        y_va_raw = y_raw.iloc[va_idx].values

        y_tr = np.log1p(y_tr_raw)
        y_va = np.log1p(y_va_raw)

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)

        evals_result = {}
        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=num_boost_round,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
            evals_result=evals_result,
        )
        best_iter = int(booster.best_iteration + 1)
        best_iters.append(best_iter)

        preds_raw = np.expm1(booster.predict(dva, iteration_range=(0, best_iter)))
        preds_raw = np.clip(preds_raw, 0, None)
        y_va_clip = np.clip(y_va_raw, 0, None)
        rmsle = np.sqrt(mean_squared_log_error(y_va_clip, preds_raw))
        fold_rmsle.append(rmsle)

    return (
        int(np.median(best_iters)),
        float(np.mean(fold_rmsle)),
        best_iters,
        fold_rmsle,
    )




## === cell 10
y_bg_raw = Targets_df["bandgap_energy_ev"].copy()
base_ridge_bg = RidgeCV(
    alphas=[0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5
)
model_ridge_bg = TransformedTargetRegressor(
    regressor=base_ridge_bg,
    func=np.log1p,
    inverse_func=np.expm1,
    check_inverse=False,
).fit(training_examples, y_bg_raw)

print("Ridge best alpha (BG):", model_ridge_bg.regressor_.alpha_)
BG_rmsle = rmsle_cv_ttr(
    Ridge(alpha=model_ridge_bg.regressor_.alpha_, random_state=42), y_bg_raw, cv=5
).mean()
print("BG CV RMSLE (metric-aligned):", BG_rmsle)



## === cell 11
ridge_BG_preds = model_ridge_bg.predict(test_examples)



## === cell 12
y_ef_raw = Targets_df["formation_energy_ev_natom"].copy()
base_ridge_ef = RidgeCV(
    alphas=[0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], cv=5
)
model_ridge_ef = TransformedTargetRegressor(
    regressor=base_ridge_ef,
    func=np.log1p,
    inverse_func=np.expm1,
    check_inverse=False,
).fit(training_examples, y_ef_raw)

print("Ridge best alpha (EF):", model_ridge_ef.regressor_.alpha_)
EF_rmsle = rmsle_cv_ttr(
    Ridge(alpha=model_ridge_ef.regressor_.alpha_, random_state=42), y_ef_raw, cv=5
).mean()
print("EF CV RMSLE (metric-aligned):", EF_rmsle)



## === cell 13
print("Expected combined error (Ridge CV, metric-aligned)")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)



## === cell 14
ridge_EF_preds = model_ridge_ef.predict(test_examples)



## === cell 15
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = np.clip(ridge_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(ridge_BG_preds, 0, None)
Predictions_df.to_csv("Ridge_Nomad.csv", index=False)
print("Wrote Ridge_Nomad.csv with shape:", Predictions_df.shape)



## === cell 16
pass



## === cell 17
y_bg_raw = Targets_df["bandgap_energy_ev"].copy()
base_lasso_bg = LassoCV(
    alphas=[1, 0.1, 0.001, 0.0005, 0.0001, 0.00005, 1e-5, 1e-6],
    cv=5,
    max_iter=100000,
).fit(training_examples, np.log1p(y_bg_raw).values.ravel())

print("Lasso best alpha (BG):", base_lasso_bg.alpha_)
BG_rmsle = rmsle_cv_ttr(
    LassoCV(alphas=[base_lasso_bg.alpha_], cv=5, max_iter=100000),
    y_bg_raw,
    cv=5,
).mean()
print("BG CV RMSLE (metric-aligned):", BG_rmsle)

lasso_coef = pd.Series(base_lasso_bg.coef_, index=training_examples.columns)
print(
    "Lasso picked "
    + str(sum(lasso_coef != 0))
    + " variables and eliminated the other "
    + str(sum(lasso_coef == 0))
    + " variables"
)



## === cell 18
lasso_BG_preds = np.expm1(base_lasso_bg.predict(test_examples))



## === cell 19
y_ef_raw = Targets_df["formation_energy_ev_natom"].copy()
base_lasso_ef = LassoCV(
    alphas=[1, 0.1, 0.001, 0.0005, 1e-5],
    cv=5,
    max_iter=100000,
).fit(training_examples, np.log1p(y_ef_raw).values.ravel())

print("Lasso best alpha (EF):", base_lasso_ef.alpha_)
EF_rmsle = rmsle_cv_ttr(
    LassoCV(alphas=[base_lasso_ef.alpha_], cv=5, max_iter=100000),
    y_ef_raw,
    cv=5,
).mean()
print("EF CV RMSLE (metric-aligned):", EF_rmsle)

lasso_coef = pd.Series(base_lasso_ef.coef_, index=training_examples.columns)
print(
    "Lasso picked "
    + str(sum(lasso_coef != 0))
    + " variables and eliminated the other "
    + str(sum(lasso_coef == 0))
    + " variables"
)



## === cell 20
lasso_EF_preds = np.expm1(base_lasso_ef.predict(test_examples))



## === cell 21
print("Expected combined error (Lasso CV, metric-aligned)")
combined_rmsle = (EF_rmsle + BG_rmsle) / 2
print(combined_rmsle)



## === cell 22
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = np.clip(lasso_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(lasso_BG_preds, 0, None)
Predictions_df.to_csv("Lasso_Nomad.csv", index=False)
print("Wrote Lasso_Nomad.csv with shape:", Predictions_df.shape)



## === cell 23
pass



## === cell 24
pass



## === cell 25
if "model_xgb" in globals() and hasattr(model_xgb, "loc"):
    last = len(model_xgb.loc[:]) - 1
    print(model_xgb.loc[last])
else:
    print("model_xgb cv results not available yet; skipping.")



## === cell 26
y_bg_raw = Targets_df["bandgap_energy_ev"].copy()

params_bg = {
    "max_depth": 2,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 7,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}

best_n_estimators_bg, cv_rmsle_bg, best_iters_bg, fold_rmsle_bg = (
    xgb_rmsle_cv_num_boost_round(
        params=params_bg,
        X=training_examples,
        y_raw=y_bg_raw,
        num_boost_round=500,
        early_stopping_rounds=100,
        nfold=5,
        seed=42,
    )
)
print("XGB BG: chosen n_estimators (median best_iter):", best_n_estimators_bg)
print("XGB BG: CV RMSLE (metric-aligned):", cv_rmsle_bg)

training_targets = np.log1p(y_bg_raw.copy())
model_xgb_bg = xgb.XGBRegressor(
    n_estimators=best_n_estimators_bg,
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.6,
    colsample_bytree=0.6,
    min_child_weight=7,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)
model_xgb_bg.fit(training_examples, training_targets)

try:
    xgb.plot_importance(model_xgb_bg, max_num_features=20)
    plt.show()
except Exception:
    pass

xgb_BG_preds = np.expm1(model_xgb_bg.predict(test_examples))



## === cell 27
y_ef_raw = Targets_df["formation_energy_ev_natom"].copy()

params_ef = {
    "max_depth": 3,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 1,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}

best_n_estimators_ef, cv_rmsle_ef, best_iters_ef, fold_rmsle_ef = (
    xgb_rmsle_cv_num_boost_round(
        params=params_ef,
        X=training_examples,
        y_raw=y_ef_raw,
        num_boost_round=500,
        early_stopping_rounds=100,
        nfold=5,
        seed=42,
    )
)
print("XGB EF: chosen n_estimators (median best_iter):", best_n_estimators_ef)
print("XGB EF: CV RMSLE (metric-aligned):", cv_rmsle_ef)



## === cell 28
training_targets = np.log1p(y_ef_raw.copy())

model_xgb_ef = xgb.XGBRegressor(
    n_estimators=best_n_estimators_ef,
    max_depth=3,
    learning_rate=0.1,
    gamma=0,
    subsample=1,
    colsample_bytree=1,
    min_child_weight=1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)
model_xgb_ef.fit(training_examples, training_targets)

try:
    xgb.plot_importance(model_xgb_ef, max_num_features=20)
    plt.show()
except Exception:
    pass

xgb_EF_preds = np.expm1(model_xgb_ef.predict(test_examples))



## === cell 29
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()

Predictions_df["formation_energy_ev_natom"] = np.clip(xgb_EF_preds, 0, None)
Predictions_df["bandgap_energy_ev"] = np.clip(xgb_BG_preds, 0, None)

Predictions_df = Predictions_df[
    ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
]
Predictions_df.to_csv("XGB_Nomad.csv", index=False)
print("Wrote XGB_Nomad.csv with shape:", Predictions_df.shape)
print(Predictions_df.head())
