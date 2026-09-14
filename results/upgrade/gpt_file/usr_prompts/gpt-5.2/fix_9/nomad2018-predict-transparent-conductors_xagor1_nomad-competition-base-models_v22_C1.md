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

0.06924

# 6. Current score

0.05984

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08665) has done: 'I fix the import errors caused by deprecated scikit-learn modules and missing imports so the notebook can execute end-to-end. I also correct the train/test slicing bug (it hard-codes row counts that don’t match your dataset), ensuring the model trains on all 2160 training rows and predicts on all 240 test rows. Finally, I make predictions safe for the RMSLE metric by clipping to non-negative values before writing the submission, and I write a single valid `submission.csv` with the required columns.'
- What this solution (achieved 0.08665) has done: 'We make two minimal changes that directly align training/prediction with the competition RMSLE without changing the model types or training loops: (1) train the XGBoost bandgap model on `log1p(y)` (it was accidentally fit on raw `y` after CV), and (2) stop forcing formation energy predictions to be non-negative (RMSLE only requires `y_pred > -1`, and clamping to 0 can severely distort negative true values). We also apply a consistent “RMSLE-safe” clip at `-0.999` to both targets’ predictions right before writing the submission, preserving semantics while avoiding invalid values. These changes should reduce the current gap to the target score (lower is better) with minimal code edits.'
- What this solution (achieved 0.05817) has done: 'You’re currently worse than the target (0.08665 vs 0.06924; lower is better), so the smallest likely win is to remove two inconsistencies that hurt generalization without changing your model types or training loops. First, the bandgap XGB model is still being fit on the wrong target (raw `training_targets` gets reused), so we keep the log1p training target in a dedicated variable and train/predict consistently in log-space. Second, your XGB CV is choosing `num_boost_round` using RMSE on log-target, but the final `XGBRegressor` is trained with square error on raw labels; we align that by keeping log labels for training and expm1 for prediction. Everything else (feature set, model hyperparameters, CV approach, blending, RMSLE-safe clipping, and submission schema) stays the same.'
- What this solution (achieved 0.08665) has done: 'Your current score (0.05817) is better than the target (0.06924) and lower is better, so we should slightly *degrade* performance toward the target with minimal, safe changes. The smallest knob that predictably worsens RMSLE a bit without breaking core logic is to reduce the amount of the stronger XGB model in the bandgap blend (increase the weaker linear model weight), while keeping the same models, features, and training loops. To avoid any accidental invalid MSLE behavior, we keep the same final “>-1” clipping and ensure ids are sorted/aligned exactly as in the sample submission. This should move the score upward (worse) toward the target band while still producing a valid submission.csv.'
- What this solution (achieved 0.08665) has done: 'You’re currently worse than the target (0.08665 vs 0.06924; lower is better), so we should make a small, metric-aligned improvement rather than changing the modeling approach. The biggest low-risk issue is that your linear models are trained on the *transformed* features while XGBoost is trained on the *raw* features; this makes the bandgap blend less complementary and can hurt RMSLE. I minimally align the XGBoost training/prediction to use the same transformed feature matrix used by the linear models (same XGB params, CV, and log1p/expm1 handling), which typically improves consistency and should reduce the gap toward the target. I also keep the RMSLE-safety clipping at `>-1` and ensure submission ordering/columns remain exactly correct.'
- What this solution (achieved 0.0592) has done: 'Your current score (0.08665) is worse than the target (0.06924) and lower is better, so we make the smallest metric-aligned fixes that can realistically close the gap without changing the overall modeling approach. The main improvement is to make the XGBoost CV selection consistent with the final XGBRegressor training by using the same objective/metric and by choosing the best iteration by the *minimum* CV test RMSE (not just the last row returned), which often reduces over/under-training. We also apply the same “RMSLE-safe” log1p handling more consistently: keep bandgap strictly non-negative (physical and RMSLE-friendly) while keeping formation energy only clipped at -0.999 right before log1p (as you already do). Everything else (features, transforms, models, blend idea, and submission schema) stays the same.'
- What this solution (achieved 0.08665) has done: 'Your current score (0.0592) is better than the target (0.06924) and lower is better, so we should make a very small, predictable change that slightly worsens performance without changing the core modeling/training logic. The safest knob is the existing bandgap blend weight: increasing the linear model contribution (weaker than XGB here) typically raise RMSLE a bit while keeping everything else identical. I keep all features, transforms, CV, objectives, and training loops unchanged, and only adjust the final blend ratio. I also keep the same RMSLE-safety clipping and ensure submission ordering/columns remain exactly correct.'
- What this solution (achieved 0.05984) has done: 'You’re currently worse than the target (0.08665 vs 0.06924; lower is better), so we should make a small, metric-aligned improvement without changing the overall modeling approach. The biggest direct issue is that `rmsle_cv()` evaluates MSLE on raw (possibly negative) bandgap targets while the model is trained on transformed/log-safe targets later; we make CV consistent by evaluating on `log1p(max(y,0))` with an RMSE scorer in log-space (same semantics as your XGB CV). Then we slightly tune the fixed blend weight using a tiny out-of-fold CV on the training set (no model/feature changes) to pick a better weight than 0.70/0.30, which typically reduces RMSLE. Everything else (features, transforms, models, XGB hyperparams/CV, and RMSLE-safe clipping + submission schema) remains unchanged.'

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

candidate_inputs = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
    "/kaggle/input/nomad2018-predict-transparent-conductors",
]
print("Candidate input dirs exist:")
for p in candidate_inputs:
    if os.path.exists(p):
        print(" -", p, "->", "OK")
    else:
        print(" -", p, "->", "missing")



## === cell 1
base_path = "../input"
comp_path = os.path.join(base_path, "nomad2018-predict-transparent-conductors")

if os.path.exists(os.path.join(comp_path, "train.csv")):
    path = comp_path
else:
    path = base_path

train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))



## === cell 2
print("Training data shape", "\n")
print(train_df.shape, "\n")
print("Testing data shape", "\n")
print(test_df.shape, "\n")

print("Training columns", "\n")
print(train_df.columns, "\n")
print("Testing columns", "\n")
print(test_df.columns, "\n")

print("Train data types", "\n")
print(train_df.dtypes, "\n")
print("Test data types", "\n")
print(test_df.dtypes)



## === cell 3
Targets_df = pd.DataFrame()
Targets_df["bandgap_energy_ev"] = train_df["bandgap_energy_ev"].copy()
Targets_df["formation_energy_ev_natom"] = train_df["formation_energy_ev_natom"].copy()
train_df = train_df.drop(["formation_energy_ev_natom", "bandgap_energy_ev"], axis=1)

train_id_df = pd.DataFrame()
train_id_df["id"] = train_df["id"].copy()
train_df = train_df.drop(["id"], axis=1)

test_id_df = pd.DataFrame()
test_id_df["id"] = test_df["id"].copy()
test_df = test_df.drop(["id"], axis=1)

combined_df = pd.concat([train_df, test_df], ignore_index=True)
print("Total number of null values in the df", "\n")
print(combined_df.isna().sum().sum())



## === cell 4
numerical_df = pd.DataFrame.copy(
    combined_df[
        [
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
    ]
)

one_hot_df = pd.DataFrame.copy(combined_df[["spacegroup"]])

one_hot_df = pd.get_dummies(one_hot_df, prefix=["spacegroup"], columns=["spacegroup"])

features_df = pd.concat([numerical_df, one_hot_df], axis=1)

print("Original skew", "\n")
print(numerical_df.skew())
skewed_feats = numerical_df.skew()
skewed_feats = skewed_feats[skewed_feats > 0.1]
skewed_feats = skewed_feats.index

unskewed_feats = numerical_df.skew()
unskewed_feats = unskewed_feats[unskewed_feats < 0.1]
unskewed_feats = unskewed_feats.index

transform_df = pd.DataFrame()
transform_df[unskewed_feats] = (
    numerical_df[unskewed_feats] - numerical_df[unskewed_feats].mean()
) / (numerical_df[unskewed_feats].max() - numerical_df[unskewed_feats].min())
transform_df[skewed_feats] = np.log1p(numerical_df[skewed_feats])

print("Transformed skew", "\n")
print(transform_df.skew())

features_transform_df = pd.concat([transform_df, one_hot_df], axis=1)
features_df = pd.concat([numerical_df, one_hot_df], axis=1)



## === cell 5
n_train = Targets_df.shape[0]
n_test = test_id_df.shape[0]

training_examples = features_df.iloc[:n_train].copy()
test_examples = features_df.iloc[n_train : n_train + n_test].copy()

training_examples_transform = features_transform_df.iloc[:n_train].copy()
test_examples_transform = features_transform_df.iloc[n_train : n_train + n_test].copy()

print("Derived shapes:")
print("training_examples:", training_examples.shape)
print("test_examples:", test_examples.shape)
print("training_examples_transform:", training_examples_transform.shape)
print("test_examples_transform:", test_examples_transform.shape)




## === cell 6
def rmse_cv_log_target(model, y_raw, X, cv=5, seed=42):
    y_log = np.log1p(np.maximum(y_raw, 0.0))
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    rmses = []
    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_log.iloc[tr_idx], y_log.iloc[va_idx]
        m = model
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        rmse = float(np.sqrt(np.mean((pred - yva.values) ** 2)))
        rmses.append(rmse)
    return np.array(rmses)




## === cell 7
bg_raw = Targets_df["bandgap_energy_ev"].copy()
bg_log = np.log1p(np.maximum(bg_raw, 0.0))

model_linear_bg = LinearRegression().fit(training_examples_transform, bg_log)
linear_BG_pred = np.expm1(model_linear_bg.predict(test_examples_transform))

BG_rmse_log = rmse_cv_log_target(
    LinearRegression(), bg_raw, training_examples_transform, cv=5
).mean()
print("Band gap CV RMSE in log1p-space (aligned to RMSLE):")
print(BG_rmse_log, "\n")

training_targets_fe = Targets_df["formation_energy_ev_natom"].copy()
model_linear_fe = LinearRegression().fit(
    training_examples_transform, training_targets_fe
)
linear_EF_pred = model_linear_fe.predict(test_examples_transform)

print(
    "Formation Energy RMSLE: skipped (target can be negative; MSLE undefined for y_true < 0)\n"
)

Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy()
Predictions_df["formation_energy_ev_natom"] = linear_EF_pred
Predictions_df["bandgap_energy_ev"] = linear_BG_pred
Predictions_df.to_csv("Linear_Nomad.csv", index=False)
print("Wrote Linear_Nomad.csv")



## === cell 8
bg_log_target = np.log1p(np.maximum(Targets_df["bandgap_energy_ev"].copy(), 0.0))

dtrain = xgb.DMatrix(training_examples_transform, label=bg_log_target)
params = {
    "max_depth": 2,
    "eta": 0.1,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 1,
    "min_child_weight": 10,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}

model_xgb_cv = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    early_stopping_rounds=100,
    verbose_eval=False,
)

best_iter = int(model_xgb_cv["test-rmse-mean"].idxmin())
best_score = float(model_xgb_cv.loc[best_iter, "test-rmse-mean"])
print("BG best iteration (min test-rmse-mean):", best_iter)
print("BG best test-rmse-mean:", best_score)



## === cell 9
model_xgb_bg = xgb.XGBRegressor(
    n_estimators=best_iter,
    max_depth=2,
    learning_rate=0.1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=1,
    min_child_weight=10,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
)
model_xgb_bg.fit(training_examples_transform, bg_log_target)
xgb_BG_preds = np.expm1(model_xgb_bg.predict(test_examples_transform))



## === cell 10
fe_raw = Targets_df["formation_energy_ev_natom"].copy()
fe_safe = np.maximum(fe_raw, -0.999)
training_targets = np.log1p(fe_safe)

dtrain = xgb.DMatrix(training_examples_transform, label=training_targets)
params = {
    "max_depth": 4,
    "eta": 0.08,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 0.4,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "seed": 42,
}

model_xgb_cv = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    early_stopping_rounds=100,
    verbose_eval=False,
)

best_iter = int(model_xgb_cv["test-rmse-mean"].idxmin())
best_score = float(model_xgb_cv.loc[best_iter, "test-rmse-mean"])
print("EF best iteration (min test-rmse-mean):", best_iter)
print("EF best test-rmse-mean:", best_score)

model_xgb_ef = xgb.XGBRegressor(
    n_estimators=best_iter,
    max_depth=4,
    learning_rate=0.08,
    gamma=0,
    subsample=1,
    colsample_bytree=0.4,
    min_child_weight=3,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
)
model_xgb_ef.fit(training_examples_transform, training_targets)
xgb_EF_preds = np.expm1(model_xgb_ef.predict(test_examples_transform))


def oof_blend_weight_bg(X, y_raw, xgb_params, num_boost_round, weights, cv=5, seed=42):
    y_log = np.log1p(np.maximum(y_raw, 0.0)).values
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    oof_lin = np.zeros_like(y_log, dtype=float)
    oof_xgb = np.zeros_like(y_log, dtype=float)

    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr = y_log[tr_idx]

        lin = LinearRegression()
        lin.fit(Xtr, ytr)
        oof_lin[va_idx] = lin.predict(Xva)

        dtr = xgb.DMatrix(Xtr, label=ytr)
        dva = xgb.DMatrix(Xva)
        booster = xgb.train(xgb_params, dtr, num_boost_round=int(num_boost_round))
        oof_xgb[va_idx] = booster.predict(dva)

    best_w = None
    best_rmse = 1e18
    for w in weights:
        blend = w * oof_xgb + (1.0 - w) * oof_lin
        rmse = float(np.sqrt(np.mean((blend - y_log) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
    return best_w, best_rmse


weights_to_try = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
best_w, best_rmse = oof_blend_weight_bg(
    training_examples_transform,
    Targets_df["bandgap_energy_ev"].copy(),
    {
        "max_depth": 2,
        "eta": 0.1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 1,
        "min_child_weight": 10,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
    },
    best_iter,
    weights_to_try,
    cv=5,
    seed=42,
)
print("Chosen BG blend weight (XGB) from OOF log-RMSE:", best_w)
print("OOF log-RMSE at chosen weight:", best_rmse)

Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].astype(int).copy()

stacked_BG_preds = best_w * xgb_BG_preds + (1.0 - best_w) * linear_BG_pred

Predictions_df["formation_energy_ev_natom"] = np.maximum(xgb_EF_preds, -0.999)
Predictions_df["bandgap_energy_ev"] = np.maximum(stacked_BG_preds, 0.0)

Predictions_df = Predictions_df[
    ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
].sort_values("id")

Predictions_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Predictions_df.shape)
print(Predictions_df.head())
