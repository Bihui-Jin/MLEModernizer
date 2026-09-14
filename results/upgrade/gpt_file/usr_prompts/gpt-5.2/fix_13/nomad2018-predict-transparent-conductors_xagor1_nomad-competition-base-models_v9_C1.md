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

0.0702

# 6. Current score

0.08099

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09553) has done: 'I fix the immediate runtime error by removing the invalid `alpha=0` from both `RidgeCV` alpha grids (scikit-learn now requires strictly positive alphas). To keep the RMSLE evaluation well-defined and avoid training-time failures, I also clip both targets to be non-negative before fitting and cross-validation (and keep the existing non-negative clipping for predictions). These changes preserve the same modeling approach (RidgeCV on one-hot + numeric features) while ensuring the notebook runs end-to-end and writes a properly formatted `submission.csv`.'
- What this solution (achieved 0.08086) has done: 'To move the RMSLE down toward the 0.0702 target while keeping your RidgeCV + one-hot/numeric core logic intact, I make two minimal, metric-aligned changes: (1) train each RidgeCV on a `log1p(target)` transform and invert with `expm1` at prediction time (this matches RMSLE’s log nature and usually improves it without changing the model family), and (2) standardize only the numeric features (leave one-hot as-is) so Ridge regularization behaves more consistently across differently-scaled lattice/angle features. I keep the same cross-validation setup and output format, and still clip predictions to be non-negative for RMSLE validity. These changes should plausibly reduce the score gap (0.09553 → closer to 0.0702) without altering the overall approach.'
- What this solution (achieved 0.08085) has done: 'We need to reduce the current RMSLE score (0.08086) toward the lower-is-better target (0.0702), so we should make small, metric-aligned improvements without changing the overall RidgeCV + one-hot/numeric pipeline. The most leverage with minimal disruption is to tune the Ridge regularization more precisely by expanding the alpha grid around the currently competitive region using a log-spaced grid, while keeping the same CV=5 and log1p target approach. I also switch the CV scoring to neg_mean_squared_log_error on the original (clipped) targets for diagnostics only (the fitted model remains on log1p targets, so core logic stays the same), and keep the same non-negative clipping/inversion to ensure RMSLE validity. These changes are low-risk, keep runtime within limits, and typically close some of the remaining gap by improving regularization selection.'
- What this solution (achieved 0.08092) has done: 'You’re already close to the target (0.08085 vs 0.0702, lower-is-better), so the safest way to move closer is to keep the same RidgeCV-on-onehot+scaled-numerics core logic and only make metric-aligned tweaks that reduce RMSLE without changing the overall approach. The most direct fix is to make RidgeCV actually select alphas using an MSLE-like objective on the original (non-log) target via a `TransformedTargetRegressor` wrapper, instead of selecting alphas using MSE in log-space (which is only an approximation to RMSLE). This keeps the same model family (RidgeCV), same features, same CV=5, but aligns hyperparameter selection with the competition metric more tightly. I also keep the non-negativity clipping for RMSLE validity and still write a correctly formatted `submission.csv`.'
- What this solution (achieved 0.08101) has done: 'To reduce your RMSLE from 0.08092 toward the 0.0702 target (lower is better) while keeping the same RidgeCV + one-hot + scaled-numerics core logic, I make two minimal, metric-aligned changes. First, I tune RidgeCV using a custom CV loop that selects `alpha` by directly minimizing MSLE on the original target scale (not MSE in log-space), which better matches the competition metric without changing the underlying model family. Second, I keep the same log1p/expm1 target transform for the final fit/predict, but now with an `alpha` chosen under MSLE, and I ensure predictions are clipped to non-negative for RMSLE validity. This is a small change (hyperparameter selection only) and should plausibly move the score downward toward the target band.'
- What this solution (achieved 0.08102) has done: 'You’re worse than the target (0.08101 vs 0.0702, lower is better), so we make the smallest metric-aligned change that can plausibly reduce RMSLE without changing the model family or features. The current alpha-selection loop is effectively tuning Ridge on `log1p(y)` but still measuring MSLE on `y`; a closer match to the competition is to evaluate the exact RMSLE objective (sqrt MSLE) on each fold and select alpha by minimizing mean RMSLE (not mean MSLE), which can pick a different regularization level. We also ensure consistent KFold usage and remove an unnecessary diagnostic CV that re-cross-validates the already-fitted transformed model (keeping runtime down and reducing noise), while preserving the same final training/prediction semantics and submission format.'
- What this solution (achieved 0.08099) has done: 'To move your RMSLE down toward the 0.0702 target while keeping the same Ridge-on-(scaled numerics + one-hot spacegroup) core logic, I make two minimal, metric-aligned adjustments. First, I use an out-of-fold (OOF) prediction step and apply a single global non-negative “floor” shift per target (learned from OOF) so that predictions are less often near-zero where RMSLE is sensitive; this is a lightweight calibration layer and doesn’t change the model family or features. Second, I slightly reduce noise in alpha selection by using a repeated KFold (still 5 folds, but repeated) inside the same alpha-search routine; this keeps the same approach but makes the chosen alpha more stable, which typically improves public RMSLE a bit. The submission format and paths remain unchanged.'
- What this solution (achieved 0.08099) has done: 'Your current score (0.08099) is worse than the target (0.0702), so we should make a small, metric-aligned improvement without changing your Ridge-on-(scaled numerics + one-hot spacegroup) approach. The biggest low-risk issue is that your `StandardScaler` is fit on the concatenated train+test (data leakage), which can hurt generalization and public score; we fit the scaler on train only and apply it to test. While keeping your exact model family and CV selection logic, we also remove the “OOF floor shift” (a post-hoc bias that can miscalibrate RMSLE on unseen data) to reduce overfitting pressure and typically improve leaderboard RMSLE. Everything else (features, Ridge, log1p target, alpha selection via repeated CV RMSLE, submission format/path) remains the same.'
- What this solution (achieved 0.08099) has done: 'We’re still above the target (0.08099 vs 0.0702, lower is better), so the smallest likely win without changing your Ridge-on-(scaled numerics + one-hot spacegroup) core logic is to make the feature matrix a dense NumPy array and set `solver="auto"` explicitly, because repeated KFold fitting over a wide alpha grid is sensitive to overhead and numeric backend; this can slightly change chosen alphas and improve generalization. I also add a tiny “variance floor” to avoid taking `log1p` of values extremely close to 0 during training/prediction, which can reduce RMSLE instability around zero without changing semantics (still non-negative, same transform). Finally, I keep your repeated-CV RMSLE alpha selection exactly as-is, but I reduce redundant `.values` conversions inside loops to make the selection more consistent and faster within the 600s budget, which can help the same search complete reliably and pick better alphas.'
- What this solution (achieved 0.08099) has done: 'Your current score (0.08099, lower-is-better) is still worse than the target (0.0702), so we should make a small, low-risk improvement that preserves your Ridge + one-hot + scaled numerics approach. The most likely issue holding RMSLE back is that `spacegroup` is treated as a numeric magnitude via `get_dummies`, so Ridge can over/under-regularize it relative to the scaled numeric features; we can fix this by explicitly centering the one-hot block (without scaling its variance) so its contribution matches Ridge’s intercept/regularization assumptions better. This keeps the exact same features (same one-hot columns) and same model family, but typically improves generalization a bit for linear models with mixed feature types. I also make the CV selection use the already-built dense `X_train_values` consistently (instead of the DataFrame in helper functions) to avoid any subtle dtype/ordering differences and keep alpha selection stable.'
- What this solution (achieved 0.08099) has done: 'To move the RMSLE down toward the 0.0702 target (lower is better) while keeping your exact Ridge + (scaled numerics + centered one-hot) approach intact, I make one metric-aligned change: tune `alpha` using the competition’s actual RMSLE objective on each CV fold, but computed on the same final prediction semantics you use for submission (TransformedTargetRegressor log1p/expm1 + non-negativity clipping). This removes the current mismatch where alpha search fits a plain Ridge on log-target and then evaluates, while the final model is a wrapped regressor with slightly different behavior. I keep the same 5-fold CV with 2 repeats, the same alpha grid, and the same feature pipeline, and still write `submission.csv` with the required columns.'

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

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import (
    Ridge,
    RidgeCV,
    ElasticNet,
    LassoCV,
    LassoLarsCV,
    LinearRegression,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings("ignore")

PREFERRED_INPUT = "../input"
FALLBACK_INPUTS = [
    "/kaggle/input/nomad2018-predict-transparent-conductors",
    "/kaggle/input",
    "/kaggle/data/nomad2018-predict-transparent-conductors",
    "/kaggle/data",
]


def _resolve_input_dir():
    if os.path.exists(os.path.join(PREFERRED_INPUT, "train.csv")):
        return PREFERRED_INPUT
    for p in FALLBACK_INPUTS:
        if os.path.exists(os.path.join(p, "train.csv")):
            return p
    return "."


path = _resolve_input_dir()
print("Using data path:", path)
print("Files in path (sample):", sorted(os.listdir(path))[:20])



## === cell 1
train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))



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
numerical_cols = [
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

combined_for_ohe = pd.concat(
    [train_df[["spacegroup"]], test_df[["spacegroup"]]], ignore_index=True
)
one_hot_all = pd.get_dummies(
    combined_for_ohe, prefix=["spacegroup"], columns=["spacegroup"]
)

one_hot_train = one_hot_all.iloc[: train_df.shape[0]].copy()
one_hot_test = one_hot_all.iloc[train_df.shape[0] :].copy()

scaler = StandardScaler()
num_train = train_df[numerical_cols].astype(float).copy()
num_test = test_df[numerical_cols].astype(float).copy()

num_train_scaled = pd.DataFrame(
    scaler.fit_transform(num_train.values),
    columns=numerical_cols,
    index=num_train.index,
)
num_test_scaled = pd.DataFrame(
    scaler.transform(num_test.values),
    columns=numerical_cols,
    index=num_test.index,
)

one_hot_mean = one_hot_train.mean(axis=0)
one_hot_train_centered = one_hot_train - one_hot_mean
one_hot_test_centered = one_hot_test - one_hot_mean

training_examples = pd.concat(
    [num_train_scaled, one_hot_train_centered.reset_index(drop=True)], axis=1
)
test_examples = pd.concat(
    [num_test_scaled, one_hot_test_centered.reset_index(drop=True)], axis=1
)

X_train_values = training_examples.to_numpy(dtype=np.float64, copy=True)
X_test_values = test_examples.to_numpy(dtype=np.float64, copy=True)

print(
    "training_examples:", training_examples.shape, "test_examples:", test_examples.shape
)



## === cell 8
print(
    "Total number of null values in train features:",
    training_examples.isna().sum().sum(),
)
print("Total number of null values in test features:", test_examples.isna().sum().sum())




## === cell 9
def rmsle_cv_logtarget(model, y_log):
    rmsle = np.sqrt(
        -cross_val_score(
            model,
            X_train_values,
            y_log,
            scoring="neg_mean_squared_error",
            cv=5,
        )
    )
    return rmsle


def rmsle_cv_direct(model, y):
    msle = -cross_val_score(
        model,
        X_train_values,
        y,
        scoring="neg_mean_squared_log_error",
        cv=5,
    )
    return np.sqrt(msle)




## === cell 10
RIDGE_ALPHAS = np.unique(
    np.concatenate(
        [
            np.logspace(-6, 3, 120),
            np.array([0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]),
        ]
    )
)

_Y_EPS = 1e-12


def _select_alpha_by_cv_rmsle_repeated(
    X_values, y, alphas, cv=5, n_repeats=2, random_state=42
):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, None)

    best_alpha = None
    best_rmsle = np.inf

    for a in alphas:
        rep_scores = []
        for rep in range(n_repeats):
            kf = KFold(
                n_splits=cv,
                shuffle=True,
                random_state=int(random_state + 1000 * rep),
            )
            fold_rmsle = []
            for tr_idx, va_idx in kf.split(X_values):
                X_tr = X_values[tr_idx]
                X_va = X_values[va_idx]
                y_tr = y[tr_idx]
                y_va = y[va_idx]

                base = Ridge(alpha=float(a), random_state=random_state, solver="auto")
                reg = TransformedTargetRegressor(
                    regressor=base,
                    func=np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                )

                reg.fit(X_tr, np.maximum(y_tr, _Y_EPS))

                pred = reg.predict(X_va)
                pred = np.clip(pred, 0.0, None)

                fold_rmsle.append(
                    math.sqrt(
                        mean_squared_log_error(
                            np.maximum(y_va, _Y_EPS), np.maximum(pred, _Y_EPS)
                        )
                    )
                )

            rep_scores.append(float(np.mean(fold_rmsle)))

        rmsle = float(np.mean(rep_scores))
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_alpha = float(a)

    return best_alpha, best_rmsle


def _make_fixed_alpha_log_ridge(alpha):
    base = Ridge(alpha=float(alpha), random_state=42, solver="auto")
    ttr = TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return ttr


def _oof_floor_shift(X, y, alpha, cv=5, random_state=42, quantile=0.25):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, None)
    X_values = X.values

    oof_pred = np.zeros_like(y, dtype=float)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for tr_idx, va_idx in kf.split(X_values):
        reg = Ridge(alpha=float(alpha), random_state=random_state, solver="auto")
        reg.fit(X_values[tr_idx], np.log1p(np.maximum(y[tr_idx], _Y_EPS)))
        pred = np.expm1(reg.predict(X_values[va_idx]))
        oof_pred[va_idx] = np.clip(pred, 0.0, None)

    residual = y - oof_pred
    delta = float(np.quantile(residual, quantile))
    delta = float(np.clip(delta, 0.0, np.inf))
    return delta




## === cell 11
training_targets_bg = Targets_df["bandgap_energy_ev"].copy().values.astype(float)
training_targets_bg = np.clip(training_targets_bg, 0.0, None)

bg_alpha, bg_cv_rmsle = _select_alpha_by_cv_rmsle_repeated(
    X_train_values,
    training_targets_bg,
    RIDGE_ALPHAS,
    cv=5,
    n_repeats=2,
    random_state=42,
)
print(
    "BG selected alpha (repeated CV RMSLE):",
    bg_alpha,
    "CV RMSLE:",
    bg_cv_rmsle,
)

bg_floor = 0.0
print("BG floor shift set to:", bg_floor)

model_ridge_bg = _make_fixed_alpha_log_ridge(bg_alpha)
model_ridge_bg.fit(training_examples, np.maximum(training_targets_bg, _Y_EPS))



## === cell 12
ridge_BG_preds = model_ridge_bg.predict(test_examples)

ridge_BG_preds = ridge_BG_preds + bg_floor
ridge_BG_preds = np.clip(ridge_BG_preds, 0, None)

print("ridge_BG_preds summary:", pd.Series(ridge_BG_preds).describe())



## === cell 13
training_targets_ef = (
    Targets_df["formation_energy_ev_natom"].copy().values.astype(float)
)
training_targets_ef = np.clip(training_targets_ef, 0.0, None)

ef_alpha, ef_cv_rmsle = _select_alpha_by_cv_rmsle_repeated(
    X_train_values,
    training_targets_ef,
    RIDGE_ALPHAS,
    cv=5,
    n_repeats=2,
    random_state=42,
)
print(
    "EF selected alpha (repeated CV RMSLE):",
    ef_alpha,
    "CV RMSLE:",
    ef_cv_rmsle,
)

ef_floor = 0.0
print("EF floor shift set to:", ef_floor)

model_ridge_ef = _make_fixed_alpha_log_ridge(ef_alpha)
model_ridge_ef.fit(training_examples, np.maximum(training_targets_ef, _Y_EPS))



## === cell 14
print("Expected combined error (CV RMSLE diagnostic):")
combined_rmsle_direct = (ef_cv_rmsle + bg_cv_rmsle) / 2
print(combined_rmsle_direct)



## === cell 15
ridge_EF_preds = model_ridge_ef.predict(test_examples)

ridge_EF_preds = ridge_EF_preds + ef_floor
ridge_EF_preds = np.clip(ridge_EF_preds, 0, None)

print("ridge_EF_preds summary:", pd.Series(ridge_EF_preds).describe())



## === cell 16
Predictions_df = pd.DataFrame()
Predictions_df["id"] = test_id_df["id"].copy().astype(int)
Predictions_df["formation_energy_ev_natom"] = ridge_EF_preds
Predictions_df["bandgap_energy_ev"] = ridge_BG_preds

Predictions_df = Predictions_df.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
Predictions_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", Predictions_df.shape)
print(Predictions_df.head())
