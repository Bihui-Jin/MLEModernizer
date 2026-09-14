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

0.1389791601511788

# 6. Current score

9.90841

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.95924) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/.../submission.csv` files) which currently causes the pipeline to crash before any submission is written. Instead, I keep your existing feature engineering and pressure-grid calibration, but generate a lightweight in-notebook baseline prediction using a robust median pressure per `(R, C, time_step)` learned from the training set and applied to the test set. This preserves the existing “median + quantize to pressure step + clip” semantics while ensuring `pred` is always defined and aligned by `id`. Finally, I write a valid `submission.csv` (required suffix) plus your previous auxiliary outputs.'
- What this solution (achieved 9.90989) has done: 'You’re currently scoring far worse than the target (MAE 9.959 vs 0.139; lower is better), and the main reason is that the submission predictions are based only on `(R,C,time_step)` medians, ignoring the dominant control signals (`u_in`, `u_out`) and breath dynamics. To move sharply toward the target while keeping the same “median lookup baseline” core logic, I switch the median table to use richer keys that exist in both train and test: `(R, C, time_step, u_out, u_in_rounded)` plus a fallback cascade to avoid NaNs. I also compute `P_STEP` robustly from unique pressure values (instead of relying on the first two rows), then keep your existing quantize-to-grid + clip post-processing unchanged. This remains a lightweight, non-neural baseline and should reduce the MAE substantially without changing your overall approach.'
- What this solution (achieved 9.90905) has done: 'Your current score is far worse than the target (MAE 9.91 vs 0.139; lower is better), so we should improve predictions while keeping your existing “median-lookup baseline + quantize-to-pressure-grid + clip” semantics. The smallest high-impact issue is that your median table ignores breath dynamics; adding a lightweight, non-model “previous u_in / previous u_out” context to the group-by keys typically reduces error a lot without changing the overall approach. I implement a 2-level lag context (lag1) for both train/test, then use a safe fallback cascade (full key → no-lag key → RC/time → global median) so the pipeline always produces predictions. Finally, I keep your existing quantization/clipping and ensure the written `submission.csv` remains correctly aligned by `id`.'
- What this solution (achieved 9.90897) has done: 'Your current score (MAE ~9.91; lower is better) is far from the target (~0.139), so we need a meaningful improvement while keeping your existing “median lookup + pressure-grid quantize/clip” core approach. The main issue is that your lookup keys still don’t capture the dominant breath dynamics well enough; we can add a tiny amount of extra context without changing the modeling paradigm by including cumulative inspired volume proxy (`area`) and an additional lag (`lag2`) in the median table keys, plus a safe fallback cascade to avoid NaNs. This remains the same core logic (groupby-median retrieval) but usually cuts MAE substantially on this competition. I also keep your existing pressure-grid calibration and submission alignment checks unchanged to preserve evaluation semantics and guarantee a valid `submission.csv`.'
- What this solution (achieved 9.9087) has done: 'Your current MAE (~9.91) is far worse than the target (~0.139), so we should improve the median-lookup baseline without changing the overall “groupby-median → fallback cascade → quantize/clip” approach. The biggest missing signal is a better proxy for inspired volume: `area` should be the integral of `u_in` over time (≈ cumulative sum of `u_in * Δt`), not `time_step * u_in` accumulated, which distorts dynamics and hurts the lookup. I minimally fix `area` to use per-breath `dt` and cumulative integral in both train/test, then keep your same binning, groupby keys, fallback cascade, and final quantization/clipping. This is a small feature correction that should legitimately move the score down substantially while preserving core logic and producing a valid `submission.csv`.'
- What this solution (achieved 9.90841) has done: 'Your current submission is still based on the earlier median-lookup `pred` array and completely ignores the much richer engineered features you compute later, which is why MAE stays ~9.9. To move the score substantially toward the target while preserving your existing approach (feature engineering + RobustScaler + reshaping), I keep all your features and scaling exactly as-is and add a lightweight per-time-step linear regression fit on the scaled features to predict pressure. This is not a new training loop/architecture; it’s a closed-form baseline model that uses the already-built tensors and should dramatically reduce MAE versus the median table. Finally, I keep your existing pressure-grid quantization/clipping and write a valid `submission.csv` aligned by `id`.'
- What this solution (achieved 9.90841) has done: 'I fix the crash in the per-timestep closed-form linear regression by making the linear system solvable even when `Xt.T @ Xt` is singular. This is done with a minimal change: increase the ridge term and add a safe fallback to `np.linalg.lstsq` if `np.linalg.solve` still fails, preserving the same linear model semantics. Then I ensure `pred_lr` is always produced so the submission-writing cell runs, and I keep your existing pressure-grid quantization/clipping and `submission.csv` output unchanged. These changes are correctness/stability fixes and should also move the score substantially toward the target by enabling the intended model to run.'
- What this solution (achieved 9.90841) has done: 'You’re far worse than the target (MAE 9.91 vs 0.139; lower is better), and the main blocker is that your “linear regression on engineered features” is being trained on features that are misaligned/reshaped because you never explicitly sort `train_df`/`test_df` by `(breath_id, time_step)` before doing `reshape(-1, 80, ...)`. I make the minimal fix of sorting once (and keeping that order through feature engineering) so each 80-step sequence is coherent, which should materially reduce MAE while preserving your model/training approach. I also ensure train/test have identical one-hot columns (align after `get_dummies`) to prevent silent feature mismatches that can also degrade score. Everything else (features, RobustScaler, per-timestep ridge solve, pressure quantization/clipping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 8.26727) has done: 'I fix the feature mismatch that crashes the scaler by ensuring the `pressure` column is removed from the test feature matrix (and by aligning train/test columns in a way that can’t accidentally keep `pressure` in test). Then I fix the downstream shape error by guaranteeing the scaled arrays reshape into `(-1, 80, n_features)` before the per-timestep ridge regression runs, so `pred_lr` is always defined. These are execution/correctness fixes that also let your intended linear-regression-on-engineered-features model actually run, which should materially reduce MAE versus the current broken pipeline. I keep your existing feature engineering, scaler, ridge solve, and pressure-grid quantize/clip semantics unchanged, and still write a valid `submission.csv`.'
- What this solution (achieved 9.90841) has done: 'Your score is still far from the target (8.27 vs 0.139, lower is better), so we should improve prediction quality while preserving your existing core approach (engineered features → RobustScaler → per-time-step ridge/solve → quantize to pressure grid). The smallest high-impact issue is that your feature engineering in `add_features()` uses an incorrect “area” proxy (`time_step * u_in` cum-summed) even though you already implemented the correct integral form earlier; this hurts the linear regression substantially. I minimally fix `add_features()` to compute `dt`, `area_step = u_in * dt`, and cumulative `area` consistently for both train/test, without changing any model/loop logic. I also ensure `time_step_cumsum` remains available (now redundant but preserved) and keep all I/O paths and submission writing unchanged.'
- What this solution (achieved 8.38971) has done: 'Your current MAE (9.908) is far worse than the target (0.139; lower is better), so we need a real improvement while keeping your same overall pipeline (feature engineering → RobustScaler → per-timestep ridge regression → pressure-grid quantize/clip). The smallest high-impact fix is to stop training on expiratory timesteps (`u_out=1`) that are not scored, because including them injects noise and hurts MAE on the inspiratory-only metric. I keep the exact same model/solver, but apply a per-row sample-weight mask in the closed-form ridge solve so only `u_out==0` rows contribute to `Xt.T@Xt` and `Xt.T@y`. I also apply the same mask after prediction by forcing `u_out==1` test predictions to 0 (any value is unscored, but this preserves common competition semantics and avoids odd outputs).'
- What this solution (achieved 9.90841) has done: 'Your current MAE (8.38971) is still far worse than the target (0.13898; lower is better), so we need a meaningful improvement while preserving your existing pipeline (feature engineering → RobustScaler → per-timestep ridge regression → pressure-grid quantize/clip). The smallest high-impact bug is that you force all `u_out==1` test predictions to 0, which can be far from the true pressure and hurts MAE if the evaluator does not perfectly exclude those rows (or if any `u_out` leakage/misalignment occurs). I keep the same weighted ridge training on inspiratory rows, but for test-time `u_out==1` rows I use a safe, consistent fallback: the same median-lookup baseline you already computed earlier (instead of hard-coded 0). This preserves evaluation semantics, keeps changes minimal, and should move the score substantially downward toward the target.'
- What this solution (achieved 9.90841) has done: 'Your current MAE (9.908) is far above the target (0.139; lower is better), so we need a real accuracy gain while keeping your existing pipeline intact (feature engineering → RobustScaler → per-timestep ridge solve → quantize/clip). The smallest high-impact correction is that the per-timestep ridge is currently trained without an explicit feature centering/standardization consistency between train/test breaths at each time index; with RobustScaler applied globally, the added intercept column helps, but the solution is still sensitive to multicollinearity and scale shifts. I keep the same closed-form ridge-per-timestep model, but (1) increase numerical stability by computing the normal equations in float64 then casting back to float32 (same semantics, fewer solve pathologies), and (2) use the already-loaded `u_out` from `train_df/test_df` feature frames to build the inspiratory mask instead of re-reading CSVs (prevents any accidental ordering mismatch and keeps training aligned with your reshaped tensors). Finally, I keep your existing median fallback for `u_out==1`, and the same pressure-grid quantization/clipping and submission writing.'
- What this solution (achieved 9.90841) has done: 'Your current MAE (9.908; lower is better) is still extremely far from the target (0.139), so we need a meaningful accuracy gain while preserving your exact pipeline (feature engineering → RobustScaler → per-timestep ridge solve → pressure-grid quantize/clip). The biggest remaining issue is that the ridge model is being fit on *all* training rows, including expiratory (`u_out=1`) rows that are not scored, which dilutes the learned mapping for the inspiratory phase; we can fix this without changing the model by filtering the training rows to `u_out==0` *before* reshaping and fitting. Then, at prediction time, we keep your existing safe fallback for `u_out==1` test rows using the median-lookup baseline (instead of trusting the ridge there). This is a minimal, metric-aligned change that should move the score sharply downward toward the target while keeping all core logic intact and still producing a valid `submission.csv`.'
- What this solution (achieved 9.90841) has done: 'I fix the root cause of the reshape/scaler failures by ensuring the training data remains in full 80-step breath sequences while still aligning the regression with the inspiratory-only metric via per-row weights (instead of dropping `u_out==1` rows, which breaks the 80-step structure). I also fix the scaler feature-name mismatch by removing `pressure` from the feature matrix before fitting, while keeping `targets` separately. Finally, I compute `P_MIN/P_MAX/P_STEP` from the true training pressures (now always defined) and ensure the pipeline always produces `submission.csv` with `id,pressure` aligned to the sample submission.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "../input/ventilator-pressure-prediction"



## === cell 1
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(sub.head())
print(sub.shape)



## === cell 2
train_raw = pd.read_csv(
    f"{DATA_DIR}/train.csv",
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
test_raw = pd.read_csv(
    f"{DATA_DIR}/test.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
)

train_raw = train_raw.sort_values(["breath_id", "time_step"], kind="mergesort")
test_raw = test_raw.sort_values(["breath_id", "time_step"], kind="mergesort")

train_raw["u_in_lag1"] = train_raw.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
train_raw["u_out_lag1"] = (
    train_raw.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
)
train_raw["u_in_lag2"] = train_raw.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
train_raw["u_out_lag2"] = (
    train_raw.groupby("breath_id")["u_out"].shift(2).fillna(0).astype(np.int8)
)

test_raw["u_in_lag1"] = test_raw.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
test_raw["u_out_lag1"] = (
    test_raw.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
)
test_raw["u_in_lag2"] = test_raw.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
test_raw["u_out_lag2"] = (
    test_raw.groupby("breath_id")["u_out"].shift(2).fillna(0).astype(np.int8)
)

train_raw["dt"] = (
    train_raw.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
)
test_raw["dt"] = (
    test_raw.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
)
train_raw["area_step"] = (train_raw["dt"] * train_raw["u_in"]).astype(np.float32)
test_raw["area_step"] = (test_raw["dt"] * test_raw["u_in"]).astype(np.float32)
train_raw["area"] = (
    train_raw.groupby("breath_id")["area_step"].cumsum().astype(np.float32)
)
test_raw["area"] = (
    test_raw.groupby("breath_id")["area_step"].cumsum().astype(np.float32)
)

UIN_BIN = 1.0
AREA_BIN = 0.5

train_raw["u_in_bin"] = (np.round(train_raw["u_in"] / UIN_BIN) * UIN_BIN).astype(
    np.float32
)
test_raw["u_in_bin"] = (np.round(test_raw["u_in"] / UIN_BIN) * UIN_BIN).astype(
    np.float32
)

train_raw["u_in_lag1_bin"] = (
    np.round(train_raw["u_in_lag1"] / UIN_BIN) * UIN_BIN
).astype(np.float32)
test_raw["u_in_lag1_bin"] = (
    np.round(test_raw["u_in_lag1"] / UIN_BIN) * UIN_BIN
).astype(np.float32)

train_raw["u_in_lag2_bin"] = (
    np.round(train_raw["u_in_lag2"] / UIN_BIN) * UIN_BIN
).astype(np.float32)
test_raw["u_in_lag2_bin"] = (
    np.round(test_raw["u_in_lag2"] / UIN_BIN) * UIN_BIN
).astype(np.float32)

train_raw["area_bin"] = (np.round(train_raw["area"] / AREA_BIN) * AREA_BIN).astype(
    np.float32
)
test_raw["area_bin"] = (np.round(test_raw["area"] / AREA_BIN) * AREA_BIN).astype(
    np.float32
)

group_median_full_ctx = (
    train_raw.groupby(
        [
            "R",
            "C",
            "time_step",
            "u_out",
            "u_in_bin",
            "u_out_lag1",
            "u_in_lag1_bin",
            "u_out_lag2",
            "u_in_lag2_bin",
            "area_bin",
        ],
        sort=False,
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_med_full_ctx"})
)

group_median_full_lag = (
    train_raw.groupby(
        ["R", "C", "time_step", "u_out", "u_in_bin", "u_out_lag1", "u_in_lag1_bin"],
        sort=False,
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_med_full_lag"})
)

group_median_full = (
    train_raw.groupby(["R", "C", "time_step", "u_out", "u_in_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_med_full"})
)

group_median_uout = (
    train_raw.groupby(["R", "C", "time_step", "u_out"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_med_uout"})
)

group_median_rc_t = (
    train_raw.groupby(["R", "C", "time_step"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_med_rc_t"})
)

test_pred = (
    test_raw.merge(
        group_median_full_ctx,
        on=[
            "R",
            "C",
            "time_step",
            "u_out",
            "u_in_bin",
            "u_out_lag1",
            "u_in_lag1_bin",
            "u_out_lag2",
            "u_in_lag2_bin",
            "area_bin",
        ],
        how="left",
    )
    .merge(
        group_median_full_lag,
        on=["R", "C", "time_step", "u_out", "u_in_bin", "u_out_lag1", "u_in_lag1_bin"],
        how="left",
    )
    .merge(
        group_median_full, on=["R", "C", "time_step", "u_out", "u_in_bin"], how="left"
    )
    .merge(group_median_uout, on=["R", "C", "time_step", "u_out"], how="left")
    .merge(group_median_rc_t, on=["R", "C", "time_step"], how="left")
)

global_median = float(train_raw["pressure"].median())
test_pred["pressure_median"] = (
    test_pred["p_med_full_ctx"]
    .fillna(test_pred["p_med_full_lag"])
    .fillna(test_pred["p_med_full"])
    .fillna(test_pred["p_med_uout"])
    .fillna(test_pred["p_med_rc_t"])
    .fillna(global_median)
)

test_pred = test_pred.sort_values("id", kind="mergesort")
sub_sorted = sub.sort_values("id", kind="mergesort")
assert np.array_equal(
    test_pred["id"].values, sub_sorted["id"].values
), "ID alignment mismatch"

pred_median = test_pred["pressure_median"].to_numpy(dtype=np.float32)
uout_test_for_fallback = test_pred["u_out"].to_numpy(dtype=np.int8)

pred = np.array([pred_median])
print("pred shape:", pred.shape)

del (
    train_raw,
    test_raw,
    group_median_full_ctx,
    group_median_full_lag,
    group_median_full,
    group_median_uout,
    group_median_rc_t,
    test_pred,
)
gc.collect()



## === cell 3
mean = np.mean(pred, axis=0)
med = np.median(pred, axis=0)
std = np.std(pred, axis=0)

print(mean[:5], med[:5], std[:5])



## === cell 4
clipped_pres = np.clip(np.vstack(pred), mean - std, mean + std)
clipped_mean = np.mean(clipped_pres, axis=0)

print(clipped_mean[:5])



## === cell 5
sub_tmp = sub_sorted.copy()
sub_tmp["pressure"] = mean
sub_tmp.to_csv("submission_mean.csv", index=False)

sub_tmp = sub_sorted.copy()
sub_tmp["pressure"] = med
sub_tmp.to_csv("submission_median.csv", index=False)

sub_tmp = sub_sorted.copy()
sub_tmp["pressure"] = clipped_mean
sub_tmp.to_csv("submission_clipped_mean.csv", index=False)

print(pd.read_csv("submission_median.csv").head())



## === cell 6
train_df = (
    pd.read_csv(f"{DATA_DIR}/train.csv")
    .sort_values(["breath_id", "time_step"], kind="mergesort")
    .reset_index(drop=True)
)
test_df = (
    pd.read_csv(f"{DATA_DIR}/test.csv")
    .sort_values(["breath_id", "time_step"], kind="mergesort")
    .reset_index(drop=True)
)


def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0).astype(np.float32)
    df["area_step"] = (df["u_in"].astype(np.float32) * df["dt"]).astype(np.float32)
    df["area"] = df.groupby("breath_id")["area_step"].cumsum().astype(np.float32)

    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    print("Step-1...Completed")

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
    df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
    df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
    df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
    df = df.fillna(0)
    print("Step-2...Completed")

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )
    print("Step-3...Completed")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
    print("Step-4...Completed")

    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["breath_id__u_in_lag"] = df["breath_id__u_in_lag"] * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["breath_id__u_in_lag2"] = df["breath_id__u_in_lag2"] * df["breath_id_lag2same"]
    print("Step-5...Completed")

    df["time_step_diff"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["ewm_u_in_mean"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=9)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(
            {
                "15_in_sum": "sum",
                "15_in_min": "min",
                "15_in_max": "max",
                "15_in_mean": "mean",
            }
        )
        .reset_index(level=0, drop=True)
    )
    print("Step-6...Completed")

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    print("Step-7...Completed")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")

    return df


print("Train data...\n")
train = add_features(train_df)

print("\nTest data...\n")
test = add_features(test_df)

del train_df, test_df
gc.collect()



## === cell 7
if "pressure" not in train.columns:
    raise KeyError("Expected 'pressure' in engineered train dataframe.")

targets = train[["pressure"]].to_numpy(dtype=np.float32).reshape(-1, 80)

if "u_out" not in train.columns:
    raise KeyError("Expected 'u_out' in engineered train dataframe.")
mask_tr_rows = (train["u_out"].to_numpy(dtype=np.float32) == 0.0).astype(np.float32)

drop_cols_common = [
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]

X_train_df = train.drop(columns=["pressure"] + drop_cols_common, errors="ignore")
X_test_df = test.drop(columns=drop_cols_common, errors="ignore")

X_train_df, X_test_df = X_train_df.align(X_test_df, join="left", axis=1, fill_value=0)

print(
    f"X_train_df: {X_train_df.shape} X_test_df: {X_test_df.shape} targets: {targets.shape}"
)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)

if X_train.shape[0] % 80 != 0 or X_test.shape[0] % 80 != 0:
    raise ValueError(
        f"Row counts must be divisible by 80. Got train_rows={X_train.shape[0]}, test_rows={X_test.shape[0]}"
    )

X_train = X_train.reshape(-1, 80, X_train.shape[-1]).astype(np.float32)
X_test = X_test.reshape(-1, 80, X_train.shape[-1]).astype(np.float32)
mask_tr = mask_tr_rows.reshape(-1, 80).astype(np.float32)

print(f"X_train: {X_train.shape} X_test: {X_test.shape} mask_tr: {mask_tr.shape}")

del train, test, X_train_df, X_test_df, mask_tr_rows
gc.collect()



## === cell 8
pressure_flat = targets.reshape(-1).astype(np.float32)

p_unique = np.unique(pressure_flat)
P_MIN = float(p_unique.min())
P_MAX = float(p_unique.max())
diffs = np.diff(p_unique)
P_STEP = float(diffs[diffs > 0].min())

print("Min pressure: {}".format(P_MIN))
print("Max pressure: {}".format(P_MAX))
print("Pressure step: {}".format(P_STEP))
print("Unique values:  {}".format(p_unique.shape[0]))

del pressure_flat, p_unique, diffs
gc.collect()



## === cell 9
Xtr = X_train  # (n_breaths, 80, n_feat)
Ytr = targets.astype(np.float32)  # (n_breaths, 80)
Xte = X_test  # (n_breaths_test, 80, n_feat)

n_breaths, T, n_feat = Xtr.shape
n_breaths_te = Xte.shape[0]

Xtr_i = np.concatenate([Xtr, np.ones((n_breaths, T, 1), dtype=np.float32)], axis=2)
Xte_i = np.concatenate([Xte, np.ones((n_breaths_te, T, 1), dtype=np.float32)], axis=2)
p = n_feat + 1

ridge = 1e-1  # keep as-is (stability)
W = np.zeros((T, p), dtype=np.float32)
I64 = np.eye(p, dtype=np.float64)

for t in range(T):
    Xt = Xtr_i[:, t, :].astype(np.float64, copy=False)
    yt = Ytr[:, t].astype(np.float64, copy=False)

    w = mask_tr[:, t].astype(np.float64, copy=False)  # 1 for u_out==0 else 0
    Xw = Xt * w[:, None]
    A = Xt.T @ Xw + float(ridge) * I64
    b = Xt.T @ (yt * w)

    try:
        W[t] = np.linalg.solve(A, b).astype(np.float32)
    except np.linalg.LinAlgError:
        W[t] = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)

pred_test = np.zeros((n_breaths_te, T), dtype=np.float32)
for t in range(T):
    pred_test[:, t] = Xte_i[:, t, :] @ W[t]

pred_lr = pred_test.reshape(-1).astype(np.float32)
print("pred_lr shape:", pred_lr.shape)

if pred_median.shape[0] != pred_lr.shape[0]:
    raise ValueError(
        f"Fallback shape mismatch: pred_median has {pred_median.shape[0]} rows but pred_lr has {pred_lr.shape[0]}"
    )
pred_lr = np.where(uout_test_for_fallback > 0, pred_median, pred_lr).astype(np.float32)

del Xtr, Ytr, Xte, Xtr_i, Xte_i, W, I64, pred_test, X_train, X_test, targets, mask_tr
gc.collect()



## === cell 10
submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv").sort_values(
    "id", kind="mergesort"
)

if submission.shape[0] != pred_lr.shape[0]:
    raise ValueError(
        f"Row count mismatch: submission has {submission.shape[0]} rows but pred has {pred_lr.shape[0]}"
    )

submission["pressure"] = pred_lr

submission["pressure"] = (
    np.round((submission["pressure"].to_numpy() - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission["pressure"].to_numpy(), P_MIN, P_MAX)

submission = submission.sort_values("id", kind="mergesort")
submission.to_csv("submission.csv", index=False)
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: submission.csv and median_submission.csv")
