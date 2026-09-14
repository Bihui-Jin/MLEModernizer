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

9.9087

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.95924) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/.../submission.csv` files) which currently causes the pipeline to crash before any submission is written. Instead, I keep your existing feature engineering and pressure-grid calibration, but generate a lightweight in-notebook baseline prediction using a robust median pressure per `(R, C, time_step)` learned from the training set and applied to the test set. This preserves the existing “median + quantize to pressure step + clip” semantics while ensuring `pred` is always defined and aligned by `id`. Finally, I write a valid `submission.csv` (required suffix) plus your previous auxiliary outputs.'
- What this solution (achieved 9.90989) has done: 'You’re currently scoring far worse than the target (MAE 9.959 vs 0.139; lower is better), and the main reason is that the submission predictions are based only on `(R,C,time_step)` medians, ignoring the dominant control signals (`u_in`, `u_out`) and breath dynamics. To move sharply toward the target while keeping the same “median lookup baseline” core logic, I switch the median table to use richer keys that exist in both train and test: `(R, C, time_step, u_out, u_in_rounded)` plus a fallback cascade to avoid NaNs. I also compute `P_STEP` robustly from unique pressure values (instead of relying on the first two rows), then keep your existing quantize-to-grid + clip post-processing unchanged. This remains a lightweight, non-neural baseline and should reduce the MAE substantially without changing your overall approach.'
- What this solution (achieved 9.90905) has done: 'Your current score is far worse than the target (MAE 9.91 vs 0.139; lower is better), so we should improve predictions while keeping your existing “median-lookup baseline + quantize-to-pressure-grid + clip” semantics. The smallest high-impact issue is that your median table ignores breath dynamics; adding a lightweight, non-model “previous u_in / previous u_out” context to the group-by keys typically reduces error a lot without changing the overall approach. I implement a 2-level lag context (lag1) for both train/test, then use a safe fallback cascade (full key → no-lag key → RC/time → global median) so the pipeline always produces predictions. Finally, I keep your existing quantization/clipping and ensure the written `submission.csv` remains correctly aligned by `id`.'
- What this solution (achieved 9.90897) has done: 'Your current score (MAE ~9.91; lower is better) is far from the target (~0.139), so we need a meaningful improvement while keeping your existing “median lookup + pressure-grid quantize/clip” core approach. The main issue is that your lookup keys still don’t capture the dominant breath dynamics well enough; we can add a tiny amount of extra context without changing the modeling paradigm by including cumulative inspired volume proxy (`area`) and an additional lag (`lag2`) in the median table keys, plus a safe fallback cascade to avoid NaNs. This remains the same core logic (groupby-median retrieval) but usually cuts MAE substantially on this competition. I also keep your existing pressure-grid calibration and submission alignment checks unchanged to preserve evaluation semantics and guarantee a valid `submission.csv`.'
- What this solution (achieved 9.9087) has done: 'Your current MAE (~9.91) is far worse than the target (~0.139), so we should improve the median-lookup baseline without changing the overall “groupby-median → fallback cascade → quantize/clip” approach. The biggest missing signal is a better proxy for inspired volume: `area` should be the integral of `u_in` over time (≈ cumulative sum of `u_in * Δt`), not `time_step * u_in` accumulated, which distorts dynamics and hurts the lookup. I minimally fix `area` to use per-breath `dt` and cumulative integral in both train/test, then keep your same binning, groupby keys, fallback cascade, and final quantization/clipping. This is a small feature correction that should legitimately move the score down substantially while preserving core logic and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
print(sub.head())
print(sub.shape)



## === cell 2
train_raw = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
test_raw = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
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

pred = np.array([test_pred["pressure_median"].values.astype(np.float32)])
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
train_df = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
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
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(
    [
        "pressure",
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)

test = test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
)

print(f"train: {train.shape} \ntest: {test.shape}")



## === cell 8
scaler = RobustScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

print(f"train: {train.shape} \ntest: {test.shape} \ntargets: {targets.shape}")



## === cell 9
pressure = targets.squeeze().reshape(-1).astype("float32")

p_unique = np.unique(pressure)
P_MIN = float(p_unique.min())
P_MAX = float(p_unique.max())
diffs = np.diff(p_unique)
P_STEP = float(diffs[diffs > 0].min())

print("Min pressure: {}".format(P_MIN))
print("Max pressure: {}".format(P_MAX))
print("Pressure step: {}".format(P_STEP))
print("Unique values:  {}".format(p_unique.shape[0]))

del pressure, p_unique, diffs
gc.collect()



## === cell 10
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
).sort_values("id", kind="mergesort")

submission["pressure"] = np.median(np.vstack(pred), axis=0)

submission["pressure"] = (
    np.round((submission.pressure - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission.pressure, P_MIN, P_MAX)

submission = submission.sort_values("id", kind="mergesort")
submission.to_csv("submission.csv", index=False)
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: submission.csv and median_submission.csv")
