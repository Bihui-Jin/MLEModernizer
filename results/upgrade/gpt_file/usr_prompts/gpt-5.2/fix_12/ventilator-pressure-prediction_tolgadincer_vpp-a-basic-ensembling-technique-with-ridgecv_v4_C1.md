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
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.1452315082076256

# 6. Current score

5.84876

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.52182) has done: 'I fix the fold-wise median fallback assignment that currently fails because `to_numpy()` can return a read-only view; making a writable copy (or constructing a writable array) resolves the runtime error and lets the CV loop finish. Once that runs, the downstream `NameError`s (`test_pred_ridge`/`test_pred` undefined) disappear because those variables are created after the CV loop completes. I also keep the blending and post-processing exactly as-is, only ensuring robust numpy array creation and consistent indexing so a valid `submission.csv` and `submission_pp.csv` are always written. No model/feature logic is changed—only the buggy array write path is corrected.'
- What this solution (achieved 4.13996) has done: 'Your current score (4.52 MAE) is far worse than the target (0.145 MAE), and the main reason is that the Ridge model is being trained across *all time steps mixed together* rather than learning a separate mapping per time position within a breath; this misses the strong “time-step specific” structure of the problem. Keeping the same feature set and the same Ridge training approach, the smallest meaningful fix is to add the standard `time_step` index-within-breath feature (0–79) and train Ridge *per time index* using GroupKFold, then stitch predictions back together. The median-lookup and the same RidgeCV blender remain in place; we only replace the Ridge submodel’s training granularity to match the data’s sequence nature, which should dramatically reduce MAE toward your target while still producing a valid submission CSV. I also ensure all arrays used for assignment are writable and that indexing alignment stays correct.'
- What this solution (achieved 4.13964) has done: 'Your current MAE (4.14) is far above the target (0.145), so we should improve performance while keeping the same overall approach (median-lookup baseline + per-timestep Ridge + RidgeCV blender + the same post-processing). The biggest remaining issue is an indexing bug in the per‑timestep Ridge loop: `X_train.iloc[tr_idx].loc[tr_m]` mixes label-based `.loc` with a boolean mask whose index doesn’t match after `.iloc`, which can silently mis-select rows and severely hurt learning. I fix that by applying boolean masks as numpy indices (or using `.iloc` twice) so train/valid/test rows for each `t_idx` are correctly aligned. I also make `ytrue/uout` plain numpy arrays for consistent, fast, unambiguous indexing (no semantic change), which stabilizes the fold computations and predictions.'
- What this solution (achieved 4.13431) has done: 'Your score (4.13964 MAE, lower-is-better) is far worse than the target (0.14523), so we should improve accuracy with minimal, metric-aligned fixes while keeping your median-lookup + per-timestep Ridge + RidgeCV blender + post-processing intact. The biggest remaining performance leak is that Ridge is currently trained/predicting on both inspiratory and expiratory rows even though expiratory is not scored; training only on inspiratory (`u_out==0`) per time index typically reduces noise and improves inspiratory MAE substantially without changing the model type or feature set. I also fix the median fallback to avoid `np.vectorize` (slow/fragile) and to handle missing `u_out` keys deterministically via `map` with a safe fill. Finally, I keep the submission writing logic identical but ensure alignment by using `test` length consistently.'
- What this solution (achieved 4.13472) has done: 'Your current MAE (4.134) is far above the target (0.145), so we should improve accuracy while keeping your exact pipeline (median lookup + per‑timestep Ridge + RidgeCV blending + the same post-processing). The biggest issue is that in the per‑timestep Ridge loop you’re fitting/predicting **with `u_out` included as a feature**, but you only train on inspiratory (`u_out==0`) rows—so `u_out` is constant and can make the fitted intercept/coefficients ill-conditioned and hurt generalization; removing this single constant feature is a minimal, metric-aligned fix that usually yields a large MAE drop. Additionally, your blender is trained on inspiratory rows but at inference it also predicts expiratory rows; setting expiratory predictions to the median-lookup baseline (or simply keep them as median) is a minimal, semantics-preserving way to avoid unstable extrapolation on unscored rows without changing the models. Everything else (folding, per-timestep training, medians, blender, rounding post-process, submission writing) stays the same.'
- What this solution (achieved 4.13455) has done: 'Your current MAE (4.1347, lower-is-better) is still far from the target (0.1452), and the biggest remaining issue is that the per‑timestep Ridge is effectively learning on a feature (`u_out` lags) that acts as a proxy for phase transitions while you only train on inspiratory rows; this adds noise/instability and harms generalization. I make the smallest metric-aligned change: exclude `u_out_lag1`/`u_out_lag2` from the Ridge features (keep them available for the median lookup key, unchanged). I also hard-enforce that Ridge-based predictions are only used for inspiratory rows (keeping expiratory rows as the median baseline, as you already do after blending) to avoid any accidental leakage of unstable values. Everything else—feature engineering, per‑timestep training, GroupKFold, median lookup, RidgeCV blender, and post-processing—remains the same and the script still write valid `submission.csv` and `submission_pp.csv`.'
- What this solution (achieved 4.13433) has done: 'Your MAE (4.13455, lower-is-better) is still far above the target (0.14523), so we should improve accuracy while keeping your exact pipeline (median lookup + per‑timestep Ridge + RidgeCV blending + same post-process). The dominant issue left is that your per‑timestep Ridge features omit the lung attributes `R` and `C`, even though pressure dynamics depend strongly on them; adding them as numeric features is a minimal change that preserves the same model/loop logic but typically yields a large MAE reduction. I also keep the median-lookup keys unchanged and preserve the same inspiratory-only training behavior and expiratory handling. Everything else (folding, models, blending, post-processing, and writing `submission.csv`) remains the same.'
- What this solution (achieved 4.1343) has done: 'Your current MAE (4.134, lower-is-better) is far above the target (0.145), and the most likely cause is that the Ridge per‑timestep model is still trying to learn the full pressure scale without using the strongest physics-inspired signals. Keeping your exact pipeline (median lookup + per‑timestep Ridge + RidgeCV blending + same post-process), the smallest high-impact change is to add the standard “integrated flow” proxy feature `u_in * dt` cumulative (“u_in_area”) and its lag, which improves inspiratory pressure tracking without changing the model family or training loops. I also make a minimal fix to ensure the blender is trained on predictions for inspiratory rows only (it already is) while guaranteeing Ridge OOF predictions for expiratory rows fall back to the median baseline to avoid injecting noise into the ridge component before blending. Everything else (folding, per‑timestep training, median lookup keys, blending, submission writing) remains intact and still produces valid `submission.csv` and `submission_pp.csv`.'
- What this solution (achieved 4.13389) has done: 'Your current MAE (4.1343) is far above the target (0.1452), so we need a real accuracy lift while keeping your same pipeline (median lookup + per‑timestep Ridge + RidgeCV blender + same post-process). The main remaining bug is that `oof_ridge_fold` is indexed by a boolean mask built on the full `va_idx` length, but the array is only `len(va_idx)` long; this silently mis-assigns (or fails) and effectively destroys OOF ridge quality. I fix this by writing fold predictions into a full-length temporary array (train-sized) and then slicing `va_idx`, preserving the exact model/loop logic. I also ensure any remaining unfilled inspiratory ridge OOF entries fall back to the median baseline (like you already do for expiratory) so the blender is trained on sensible inputs.'
- What this solution (achieved 5.84876) has done: 'Your MAE (4.13389, lower-is-better) is still far from the target (0.14523), so we need a small but high-impact correction without changing your model family or training loops. The main issue is that the Ridge per‑timestep model is still trying to learn across breaths without giving it the single most predictive signal: the previous pressure; adding a `pressure_lag1` feature for training (and using a fold-wise median-lookup estimate as its test-time proxy) keeps the same Ridge-per-timestep + GroupKFold structure but dramatically improves accuracy. This is metric-aligned and leakage-safe because within a breath, previous pressure is available during inference in a recursive sense, and we approximate it using the existing median lookup baseline (no label leakage from test). Everything else (median lookup, inspiratory-only Ridge, RidgeCV blender, expiratory handling, post-processing, and submission writing) stays intact.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold



## === cell 1
SEED = 42
np.random.seed(SEED)

pd.options.mode.copy_on_write = True



## === cell 2
DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"




## === cell 3
def mae(ytrue, ypred, uout=None):
    if isinstance(uout, (pd.Series, np.ndarray)):
        return np.mean(np.abs((ytrue - ypred)[uout == 0]))
    else:
        return np.mean(np.abs((ytrue - ypred)))




## === cell 4
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

ytrue = train["pressure"].astype(np.float32).to_numpy()
uout = train["u_out"].astype(np.int8).to_numpy()

print("train shape:", train.shape, "test shape:", test.shape)




## === cell 5
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["breath_id"] = df["breath_id"].astype(np.int32)
    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)
    df["u_out"] = df["u_out"].astype(np.int8)
    df["u_in"] = df["u_in"].astype(np.float32)
    df["time_step"] = df["time_step"].astype(np.float32)

    g = df.groupby("breath_id", sort=False)

    df["t_idx"] = g.cumcount().astype(np.int16)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0).astype(np.float32)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0).astype(np.float32)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)
    df["u_out_lag2"] = g["u_out"].shift(2).fillna(0).astype(np.int8)

    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)
    df["u_in_diff2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype(np.float32)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)
    df["u_in_mean"] = (
        df["u_in_cumsum"] / (g.cumcount() + 1).astype(np.float32)
    ).astype(np.float32)

    dt = (df["time_step"] - g["time_step"].shift(1)).fillna(0).astype(np.float32)
    df["area"] = (df["u_in"] * dt).astype(np.float32)
    df["area_cumsum"] = g["area"].cumsum().astype(np.float32)

    df["u_in_area"] = df["area_cumsum"].astype(np.float32)
    df["u_in_area_lag1"] = g["u_in_area"].shift(1).fillna(0).astype(np.float32)

    for r in [5, 20, 50]:
        df[f"R_{r}"] = (df["R"] == r).astype(np.int8)
    for c in [10, 20, 50]:
        df[f"C_{c}"] = (df["C"] == c).astype(np.int8)

    return df


train_fe = make_features(train)
test_fe = make_features(test)

gc.collect()



## === cell 6
train_fe["pressure_lag1"] = (
    train.groupby("breath_id", sort=False)["pressure"]
    .shift(1)
    .fillna(0)
    .astype(np.float32)
)
test_fe["pressure_lag1"] = np.float32(0.0)

FEATS = [
    "time_step",
    "t_idx",  # within-breath index
    "u_in",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_cumsum",
    "u_in_mean",
    "area",
    "area_cumsum",
    "u_in_area",
    "u_in_area_lag1",
    "R",
    "C",
    "R_5",
    "R_20",
    "R_50",
    "C_10",
    "C_20",
    "C_50",
    "pressure_lag1",
]

X_train = train_fe[FEATS]
X_test = test_fe[FEATS]

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 7
TIME_ROUND = 3  # time_step has limited decimals in this dataset
train_key = train[["R", "C", "u_out"]].copy()
train_key["time_step_r"] = train["time_step"].round(TIME_ROUND)

test_key = test[["R", "C", "u_out"]].copy()
test_key["time_step_r"] = test["time_step"].round(TIME_ROUND)

median_map = (
    pd.concat([train_key, train["pressure"]], axis=1)
    .groupby(["R", "C", "u_out", "time_step_r"], sort=False)["pressure"]
    .median()
    .astype(np.float32)
)

global_median_by_uout = (
    train.groupby("u_out", sort=False)["pressure"].median().astype(np.float32).to_dict()
)


def predict_median_lookup(df_key: pd.DataFrame) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(df_key[["R", "C", "u_out", "time_step_r"]])
    pred = median_map.reindex(idx).to_numpy(dtype=np.float32).copy()
    miss = np.isnan(pred)
    if miss.any():
        fallback = (
            df_key.loc[miss, "u_out"]
            .map(global_median_by_uout)
            .astype(np.float32)
            .to_numpy()
        )
        if np.isnan(fallback).any():
            fallback = np.nan_to_num(
                fallback, nan=float(np.median(list(global_median_by_uout.values())))
            ).astype(np.float32)
        pred[miss] = fallback
    return pred




## === cell 8
N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)
groups = train["breath_id"].to_numpy()

oof_ridge = np.zeros(len(train), dtype=np.float32)
oof_med = np.zeros(len(train), dtype=np.float32)

test_pred_ridge_folds = []
test_pred_med_folds = []

t_idx_train = train_fe["t_idx"].to_numpy()
t_idx_test = test_fe["t_idx"].to_numpy()
unique_t = np.unique(t_idx_train)

insp_train = uout == 0
insp_test = test["u_out"].to_numpy().astype(np.int8) == 0

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, ytrue, groups=groups), 1):
    y_tr = ytrue[tr_idx]
    y_va = ytrue[va_idx]

    oof_ridge_fold_full = np.zeros(len(train), dtype=np.float32)
    test_pred_ridge_fold = np.zeros(len(test), dtype=np.float32)

    tr_key_fold = train_key.iloc[tr_idx].copy()
    tr_key_fold["time_step_r"] = (
        train.loc[tr_idx, "time_step"].round(TIME_ROUND).to_numpy()
    )

    median_map_fold = (
        pd.concat(
            [tr_key_fold.reset_index(drop=True), pd.Series(y_tr, name="pressure")],
            axis=1,
        )
        .groupby(["R", "C", "u_out", "time_step_r"], sort=False)["pressure"]
        .median()
        .astype(np.float32)
    )
    global_median_by_uout_fold = (
        train.iloc[tr_idx]
        .groupby("u_out", sort=False)["pressure"]
        .median()
        .astype(np.float32)
        .to_dict()
    )

    idx_va = pd.MultiIndex.from_frame(
        pd.concat(
            [
                train_key.iloc[va_idx].reset_index(drop=True),
                train.loc[va_idx, "time_step"]
                .round(TIME_ROUND)
                .rename("time_step_r")
                .reset_index(drop=True),
            ],
            axis=1,
        )
    )
    pred_va = median_map_fold.reindex(idx_va).to_numpy(dtype=np.float32).copy()
    miss_va = np.isnan(pred_va)
    if miss_va.any():
        fallback = (
            train.loc[va_idx, "u_out"]
            .reset_index(drop=True)
            .loc[miss_va]
            .map(global_median_by_uout_fold)
            .astype(np.float32)
            .to_numpy()
        )
        if np.isnan(fallback).any():
            fallback = np.nan_to_num(
                fallback,
                nan=float(np.median(list(global_median_by_uout_fold.values()))),
            ).astype(np.float32)
        pred_va[miss_va] = fallback
    oof_med[va_idx] = pred_va.astype(np.float32)

    idx_test = pd.MultiIndex.from_frame(test_key[["R", "C", "u_out", "time_step_r"]])
    pred_test = median_map_fold.reindex(idx_test).to_numpy(dtype=np.float32).copy()
    miss_te = np.isnan(pred_test)
    if miss_te.any():
        fallback = (
            test["u_out"]
            .loc[miss_te]
            .map(global_median_by_uout_fold)
            .astype(np.float32)
            .to_numpy()
        )
        if np.isnan(fallback).any():
            fallback = np.nan_to_num(
                fallback,
                nan=float(np.median(list(global_median_by_uout_fold.values()))),
            ).astype(np.float32)
        pred_test[miss_te] = fallback
    test_pred_med_folds.append(pred_test.astype(np.float32))

    X_test_fold = X_test.copy()
    pred_test_lag1 = (
        pd.Series(pred_test, index=test.index)
        .groupby(test["breath_id"], sort=False)
        .shift(1)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    X_test_fold.loc[:, "pressure_lag1"] = pred_test_lag1

    for t in unique_t:
        tr_m = (t_idx_train[tr_idx] == t) & insp_train[tr_idx]
        va_m = (t_idx_train[va_idx] == t) & insp_train[va_idx]
        te_m = (t_idx_test == t) & insp_test

        if (not np.any(tr_m)) or (not np.any(va_m)):
            continue

        tr_pos = tr_idx[tr_m]
        va_pos = va_idx[va_m]

        X_tr_t = X_train.iloc[tr_pos]
        y_tr_t = ytrue[tr_pos]
        X_va_t = X_train.iloc[va_pos]

        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X_tr_t, y_tr_t)

        oof_ridge_fold_full[va_pos] = ridge.predict(X_va_t).astype(np.float32)

        if np.any(te_m):
            te_pos = np.flatnonzero(te_m)
            X_te_t = X_test_fold.iloc[te_pos]
            test_pred_ridge_fold[te_pos] = ridge.predict(X_te_t).astype(np.float32)

    miss_ridge_va = (oof_ridge_fold_full[va_idx] == 0.0) & (uout[va_idx] == 0)
    if np.any(miss_ridge_va):
        oof_ridge_fold_full[va_idx[miss_ridge_va]] = oof_med[va_idx[miss_ridge_va]]

    oof_ridge[va_idx] = oof_ridge_fold_full[va_idx]
    test_pred_ridge_folds.append(test_pred_ridge_fold.astype(np.float32))

    print(
        f"Fold {fold}/{N_SPLITS} | "
        f"MAE ridge insp: {mae(y_va, oof_ridge[va_idx], uout[va_idx]):.6f} | "
        f"MAE median insp: {mae(y_va, oof_med[va_idx], uout[va_idx]):.6f}"
    )

test_pred_ridge = np.mean(np.vstack(test_pred_ridge_folds), axis=0).astype(np.float32)
test_pred_med = np.mean(np.vstack(test_pred_med_folds), axis=0).astype(np.float32)

oof_ridge = oof_ridge.copy()
oof_ridge[uout == 1] = oof_med[uout == 1]

print("Overall OOF MAE ridge insp:", mae(ytrue, oof_ridge, uout))
print("Overall OOF MAE median insp:", mae(ytrue, oof_med, uout))



## === cell 9
insp_mask = uout == 0

X_ens = np.stack([oof_ridge, oof_med], axis=1)[insp_mask]
y_ens = ytrue[insp_mask]

print(f"Ensemble X shape: {X_ens.shape}")
print(f"Ensemble y shape: {y_ens.shape}")



## === cell 10
lin_reg = RidgeCV(alphas=np.logspace(-3, 10, 20), fit_intercept=True)
lin_reg.fit(X_ens, y_ens)

pred_ens_oof_insp = lin_reg.predict(X_ens).astype(np.float32)
print("Blender OOF MAE (insp):", mae(y_ens, pred_ens_oof_insp))
print(f"Ensemble Weights: {lin_reg.coef_}")
print(f"Ensemble Intercept: {lin_reg.intercept_}")
print(f"Sum of weights: {float(np.sum(lin_reg.coef_))}")



## === cell 11
X_test_ens = np.stack([test_pred_ridge, test_pred_med], axis=1)
test_pred = lin_reg.predict(X_test_ens).astype(np.float32)

test_uout = test["u_out"].to_numpy().astype(np.int8)
test_pred = test_pred.copy()
test_pred[test_uout == 1] = test_pred_med[test_uout == 1]



## === cell 12
pressure_sorted = np.sort(train["pressure"].unique())
PRESSURE_MIN = float(pressure_sorted[0])
PRESSURE_MAX = float(pressure_sorted[-1])
PRESSURE_STEP = float(pressure_sorted[1] - pressure_sorted[0])


def post_process(pressure_arr: np.ndarray) -> np.ndarray:
    pressure_arr = pressure_arr.astype(np.float32)
    pressure_arr = (
        np.round((pressure_arr - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
        + PRESSURE_MIN
    )
    pressure_arr = np.clip(pressure_arr, PRESSURE_MIN, PRESSURE_MAX)
    return pressure_arr.astype(np.float32)


test_pred_pp = post_process(test_pred)



## === cell 13
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission["pressure"] = test_pred.astype(np.float32)
submission.to_csv("submission.csv", index=False)

submission_pp = submission.copy()
submission_pp["pressure"] = test_pred_pp.astype(np.float32)
submission_pp.to_csv("submission_pp.csv", index=False)

print(submission.head())
print("Wrote: submission.csv and submission_pp.csv")
