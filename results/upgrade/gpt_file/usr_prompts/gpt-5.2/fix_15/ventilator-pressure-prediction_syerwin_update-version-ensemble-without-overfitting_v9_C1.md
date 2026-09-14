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

0.1380936005324954

# 6. Current score

1.38009

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.95665) has done: 'I remove the dependency on missing external “../input/…” submission files (the cause of the FileNotFoundError/NameError chain) and instead build predictions only from data available in this environment. To keep the core logic intact, I preserve your feature engineering + scaling + reshaping, and produce a simple, stable per-(R,C,time_step,u_in,u_out) median lookup from train to predict test, with a fallback to (R,C,time_step,u_in) and then global median. Finally, I keep your pressure discretization (P_MIN/P_MAX/P_STEP rounding) and write a valid `median_submission.csv` with `id,pressure`. This run end-to-end and should achieve a reasonable MAE without changing your modeling semantics beyond replacing the unavailable ensemble inputs.'
- What this solution (achieved 4.14501) has done: 'Your current score (9.95665, lower-is-better) is far worse than the target (~0.138), so we need a real accuracy lift while keeping your overall approach intact (feature engineering + scaling + a non-ML predictor + pressure discretization). The main issue is that the current lookup keys include raw `time_step` and `u_in` floats, so almost none of the test rows match train exactly, causing heavy fallback to coarse medians and very poor MAE. I keep the same median-lookup core logic, but make the join robust by (1) quantizing `time_step` to the known 0.03 grid and (2) rounding `u_in` to a small resolution before grouping/merging, which dramatically increases exact matches without changing the modeling paradigm. I also apply the standard competition post-processing of forcing predictions to 0 during expiratory phase (`u_out==1`), which aligns with the metric (expiratory phase not scored) and typically reduces error.'
- What this solution (achieved 4.0069) has done: 'Your current MAE (4.145) is far above the target (0.138), so we need a meaningful accuracy lift while still keeping the same overall “train-median lookup + discretize to pressure grid” core logic. The biggest remaining issue is that rounding `u_in` to 0.1 is still too strict: train/test won’t match well because `u_in` is effectively continuous, so many rows fall back to coarse medians. I keep the exact same median-merge approach and post-processing, but make the lookup robust by binning `u_in` (rather than decimal rounding) and slightly relaxing the primary key to use `u_in_bin` first, with the exact `u_in` only as an optional refinement fallback. This increases exact-hit rates on the informative dimensions without changing the modeling paradigm, and should move MAE substantially toward the target.'
- What this solution (achieved 1.98712) has done: 'I keep your overall pipeline intact (same feature engineering + scaling + median-lookup predictor + pressure-grid rounding), but fix the main reason the lookup still falls back too often: `u_in` is continuous, so even with binning, the keys are too strict. I add a cumulative-volume feature (`u_in` integral/area within breath) and do the median mapping on (`R`,`C`,`time_step`,`area_bin`,`u_out`) first, which is much more repeatable between train/test and typically reduces MAE substantially without changing the “median table lookup” core logic. I keep your existing `u_in_bin` mapping as a secondary fallback, then your existing coarser fallbacks. This should move the score meaningfully toward the target while remaining simple and deterministic.'
- What this solution (achieved 2.16973) has done: 'We keep your exact “median table lookup + pressure-grid rounding” core logic, but make the lookup keys more consistent with how ventilator signals evolve across a breath so fewer rows fall back to coarse medians. Concretely, we (1) compute `area` using the same `time_step*u_in` cumsum definition you used in `add_features` (instead of `u_in*dt`, which can drift due to float/rounding), and (2) add a second repeatable state key `u_in_cumsum` (binned) to disambiguate similar `area` values and improve inspiratory-phase matching. We still keep your existing fallback chain (area-based → u_in_bin → coarser medians → global median) and keep the `u_out==1 => 0` post-process + pressure discretization unchanged. These are minimal, deterministic changes aimed specifically at reducing MAE from 1.987 toward the 0.138 target.'
- What this solution (achieved 2.11696) has done: 'Your current MAE (2.16973, lower-is-better) is still far above the target (0.13809), so we should improve accuracy while keeping your median-lookup core logic intact. The main low-risk win is to make the keying more “breath-phase aware” by adding a discretized within-breath `count` (time index) and the `u_in` lag-1 signal (binned), which better captures the control dynamics without changing the modeling approach. We keep your existing feature engineering/scaling cells untouched and only adjust the lookup-table construction + fallback chain to use these extra repeatable keys, plus we keep the `u_out==1 => 0` post-process and the pressure-grid rounding as-is. This should reduce fallbacks and tighten the median mapping, typically improving MAE toward the target.'
- What this solution (achieved 1.49136) has done: 'We keep your existing “median lookup table + pressure-grid rounding + u_out==1 → 0” core logic, but make the lookup keys more consistent between train and test by removing float-driven mismatches in the derived state features. Concretely, we compute `area` using a per-breath `dt` (time_step diff) like true integration instead of `time_step * u_in` cumsum, and we quantize `u_in` before computing `u_in_cumsum` so the cumulative signal aligns much better across breaths. These are minimal, deterministic changes localized to the lookup-table cell and should reduce fallback frequency, moving MAE down toward your target without altering the model family or training loop. The rest of your pipeline (feature engineering/scaling cells and submission writing) stays intact.'
- What this solution (achieved 1.49136) has done: 'Your current score (1.49136, lower-is-better) is still far above the target (0.13809), so we should improve accuracy with the smallest localized changes to your existing median-lookup approach. The main weakness remaining is that the lookup uses global (dataset-wide) `count` and `time_step` alignment but doesn’t explicitly incorporate the breath’s *inspiratory phase progression* beyond `u_out`, so many inspiratory states with similar keys still get mismatched and fall back. I keep the same overall “build median tables from train → merge onto test with fallbacks → set `u_out==1` to 0 → snap to pressure grid” logic, but add two repeatable within-breath state keys: a binned `time_from_insp_start` and a binned `u_in` integral since inspiratory start (`area_insp`). These two keys are deterministic, don’t change your training loop/architecture (none exists), and typically reduce fallbacks and tighten mapping during the scored inspiratory phase, moving MAE toward the target.'
- What this solution (achieved 1.49136) has done: 'We keep your exact median-lookup + fallback + pressure-grid rounding core logic, but make two minimal adjustments that usually reduce MAE for this competition: (1) don’t force `u_out==1` predictions to 0 (the expiratory phase is ignored in scoring, so this can only hurt if Kaggle still expects realistic pressures there), and (2) use a more conservative/global fallback specifically for `u_out==1` rows (so expiratory predictions don’t introduce unnecessary error if they are actually included). Everything else (feature engineering cells, scaling, discretization) is left intact to avoid changing semantics. This should move your score downward from 1.491 toward the target without rewriting the approach.'
- What this solution (achieved 1.49136) has done: 'We keep your median-lookup + fallback + pressure-grid rounding approach intact, but fix a key misalignment: the “inspiratory start” and “area since inspiratory start” features should reset at the first inspiratory timestep (after any initial u_out==1), not use the minimum time_step among inspiratory points. This makes the within-breath phase keys much more consistent between train and test and reduces unnecessary fallbacks during the scored inspiratory phase, which should move MAE down toward your target. We also compute inspiratory dt cleanly (dt only when both current and previous steps are inspiratory) so `area_insp` doesn’t accidentally include the step immediately after u_out flips. All other cells (feature engineering/scaling/discretization/submission writing) remain unchanged.'
- What this solution (achieved 1.49136) has done: 'I keep your current “median lookup tables + fallback chain + snap-to-pressure-grid” approach intact and only adjust the lookup keys to better match train/test during the inspiratory (scored) phase. Concretely, I fix the inspiratory-start indexing bug (currently mixing per-breath indices with global `count`), and I add a minimal “inspiratory step index” (`insp_count`) plus a correctly-reset `time_from_insp_start` based on per-breath inspiratory progression. This should reduce mismatches/fallbacks on scored timesteps without changing the overall modeling semantics or adding any new model/training. Everything else (feature engineering/scaling cells and submission writing) remains the same, and the script still write `median_submission.csv`.'
- What this solution (achieved 1.4953) has done: 'I fix the runtime KeyError in the inspiratory area feature by correcting the accidental tuple-based column indexing (`df["breath_id",]` → `df["breath_id"]`), which currently prevents any predictions from being produced. I also make the data path robust to this environment by falling back from `../input/...` to `/kaggle/input/...` and `/kaggle/data/...` if needed, without changing what files are read. Finally, I add a small safety check to ensure `test_pred_np` is always defined before writing the submission, so the pipeline always emits a valid `median_submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 1.38009) has done: 'Your current MAE (1.4953, lower-is-better) is still far above the target (0.1381), so we should improve accuracy without changing your overall “median lookup tables + fallback chain + snap-to-pressure-grid” approach. The smallest likely win is to reduce train↔test key mismatches caused by overly-fine binning of the derived cumulative signals (`area_insp_bin`, `area_bin`, `u_in_cumsum_bin`), which currently makes the primary lookup too sparse and forces noisy fallbacks. I only adjust those bin widths to be slightly coarser (more repeatable across breaths) while keeping the same features, groupby/merge logic, and submission post-processing. This should increase exact-hit rate on the higher-priority tables and move MAE downward toward the target without changing modeling semantics.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler



## === cell 1
CANDIDATE_DIRS = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",  # fallback if files are directly here
    "/kaggle/data",  # fallback if files are directly here
]

DATA_DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    for d in CANDIDATE_DIRS:
        if os.path.exists(
            os.path.join(d, "ventilator-pressure-prediction", "train.csv")
        ):
            DATA_DIR = os.path.join(d, "ventilator-pressure-prediction")
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv for ventilator-pressure-prediction in expected directories."
    )

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print(train_df.shape, test_df.shape, sub.shape)




## === cell 2
def add_features(df):
    df = df.copy()
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

gc.collect()



## === cell 3
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



## === cell 4
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

train_scaled = train_scaled.reshape(-1, 80, train.shape[-1])
test_scaled = test_scaled.reshape(-1, 80, train.shape[-1])

print(
    f"train: {train_scaled.shape} \ntest: {test_scaled.shape} \ntargets: {targets.shape}"
)



## === cell 5
pressure = targets.squeeze().reshape(-1, 1).astype("float32")

P_MIN = np.min(pressure)
P_MAX = np.max(pressure)
P_STEP = (pressure[1] - pressure[0])[0]
print("Min pressure: {}".format(P_MIN))
print("Max pressure: {}".format(P_MAX))
print("Pressure step: {}".format(P_STEP))
print("Unique values:  {}".format(np.unique(pressure).shape[0]))

del pressure
gc.collect()



## === cell 6
train_raw = train_df[
    ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
].copy()
test_raw = test_df[["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]].copy()

TIME_STEP_GRID = np.float32(0.03)

UIN_BIN_WIDTH = np.float32(0.5)

AREA_BIN_WIDTH = np.float32(0.5)
UIN_CUMSUM_BIN_WIDTH = np.float32(10.0)

UIN_LAG1_BIN_WIDTH = np.float32(0.5)

TIME_FROM_INSP_BIN = np.float32(0.03)
AREA_INSP_BIN_WIDTH = np.float32(0.5)

for df in (train_raw, test_raw):
    df["time_step"] = (
        np.round(df["time_step"].astype(np.float32) / TIME_STEP_GRID) * TIME_STEP_GRID
    ).astype(np.float32)

    u_in_f = df["u_in"].astype(np.float32)
    df["u_in_bin"] = (np.round(u_in_f / UIN_BIN_WIDTH) * UIN_BIN_WIDTH).astype(
        np.float32
    )
    df["u_in"] = u_in_f

for df in (train_raw, test_raw):
    df["count"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    df["u_in_lag1"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )
    df["u_in_lag1_bin"] = (
        np.round(df["u_in_lag1"] / UIN_LAG1_BIN_WIDTH) * UIN_LAG1_BIN_WIDTH
    ).astype(np.float32)

for df in (train_raw, test_raw):
    dt = (
        df.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )

    u_in_for_area = df["u_in_bin"].to_numpy(dtype=np.float32)
    df["area"] = (dt.to_numpy(dtype=np.float32) * u_in_for_area).astype(np.float32)
    df["area"] = df.groupby("breath_id", sort=False)["area"].cumsum().astype(np.float32)
    df["area_bin"] = (np.round(df["area"] / AREA_BIN_WIDTH) * AREA_BIN_WIDTH).astype(
        np.float32
    )

    df["u_in_cumsum"] = (
        df.groupby("breath_id", sort=False)["u_in_bin"].cumsum().astype(np.float32)
    )
    df["u_in_cumsum_bin"] = (
        np.round(df["u_in_cumsum"] / UIN_CUMSUM_BIN_WIDTH) * UIN_CUMSUM_BIN_WIDTH
    ).astype(np.float32)

    insp = df["u_out"].astype(np.int16).to_numpy() == 0
    df["_insp"] = insp.astype(np.int8)

    df["_insp_cum"] = (
        df.groupby("breath_id", sort=False)["_insp"].cumsum().astype(np.int16)
    )
    df["insp_count"] = np.where(insp, df["_insp_cum"] - 1, -1).astype(np.int16)

    df["time_from_insp_start"] = np.where(
        insp, df["insp_count"].astype(np.float32) * TIME_STEP_GRID, 0.0
    ).astype(np.float32)
    df["time_from_insp_start_bin"] = (
        np.round(df["time_from_insp_start"] / TIME_FROM_INSP_BIN) * TIME_FROM_INSP_BIN
    ).astype(np.float32)

    prev_insp = (
        pd.Series(insp, index=df.index)
        .groupby(df["breath_id"], sort=False)
        .shift(1)
        .fillna(False)
        .to_numpy()
    )
    dt_insp = np.where(insp & prev_insp, dt.to_numpy(dtype=np.float32), 0.0).astype(
        np.float32
    )

    area_insp = (dt_insp * u_in_for_area).astype(np.float32)

    df["area_insp"] = (
        pd.Series(area_insp, index=df.index)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.float32)
    )
    df["area_insp_bin"] = (
        np.round(df["area_insp"] / AREA_INSP_BIN_WIDTH) * AREA_INSP_BIN_WIDTH
    ).astype(np.float32)

    df.drop(columns=["_insp", "_insp_cum"], inplace=True)

for col in ["R", "C", "u_out"]:
    train_raw[col] = train_raw[col].astype(np.int16)
    test_raw[col] = test_raw[col].astype(np.int16)

g_area = (
    train_raw.groupby(
        [
            "R",
            "C",
            "count",
            "insp_count",
            "time_step",
            "time_from_insp_start_bin",
            "area_insp_bin",
            "area_bin",
            "u_in_cumsum_bin",
            "u_in_lag1_bin",
            "u_out",
        ],
        sort=False,
    )["pressure"]
    .median()
    .reset_index()
)
test_pred = test_raw.merge(
    g_area,
    on=[
        "R",
        "C",
        "count",
        "insp_count",
        "time_step",
        "time_from_insp_start_bin",
        "area_insp_bin",
        "area_bin",
        "u_in_cumsum_bin",
        "u_in_lag1_bin",
        "u_out",
    ],
    how="left",
)["pressure"]

if test_pred.isna().any():
    g_area2 = (
        train_raw.groupby(
            [
                "R",
                "C",
                "count",
                "insp_count",
                "time_step",
                "time_from_insp_start_bin",
                "area_insp_bin",
                "area_bin",
                "u_in_lag1_bin",
                "u_out",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
    )
    test_pred_area2 = test_raw.merge(
        g_area2,
        on=[
            "R",
            "C",
            "count",
            "insp_count",
            "time_step",
            "time_from_insp_start_bin",
            "area_insp_bin",
            "area_bin",
            "u_in_lag1_bin",
            "u_out",
        ],
        how="left",
    )["pressure"]
    test_pred = test_pred.fillna(test_pred_area2)

if test_pred.isna().any():
    g_area3 = (
        train_raw.groupby(
            [
                "R",
                "C",
                "count",
                "insp_count",
                "time_step",
                "time_from_insp_start_bin",
                "area_insp_bin",
                "area_bin",
                "u_out",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
    )
    test_pred_area3 = test_raw.merge(
        g_area3,
        on=[
            "R",
            "C",
            "count",
            "insp_count",
            "time_step",
            "time_from_insp_start_bin",
            "area_insp_bin",
            "area_bin",
            "u_out",
        ],
        how="left",
    )["pressure"]
    test_pred = test_pred.fillna(test_pred_area3)

if test_pred.isna().any():
    g_ctrl_insp = (
        train_raw.groupby(
            [
                "R",
                "C",
                "count",
                "insp_count",
                "time_step",
                "time_from_insp_start_bin",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_out",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
    )
    test_pred_ctrl_insp = test_raw.merge(
        g_ctrl_insp,
        on=[
            "R",
            "C",
            "count",
            "insp_count",
            "time_step",
            "time_from_insp_start_bin",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_out",
        ],
        how="left",
    )["pressure"]
    test_pred = test_pred.fillna(test_pred_ctrl_insp)

if test_pred.isna().any():
    g1 = (
        train_raw.groupby(
            ["R", "C", "count", "time_step", "u_in_bin", "u_in_lag1_bin", "u_out"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
    )
    test_pred1 = test_raw.merge(
        g1,
        on=["R", "C", "count", "time_step", "u_in_bin", "u_in_lag1_bin", "u_out"],
        how="left",
    )["pressure"]
    test_pred = test_pred.fillna(test_pred1)

if test_pred.isna().any():
    g2 = (
        train_raw.groupby(["R", "C", "count", "time_step", "u_out"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index()
    )
    test_pred2 = test_raw.merge(
        g2, on=["R", "C", "count", "time_step", "u_out"], how="left"
    )["pressure"]
    test_pred = test_pred.fillna(test_pred2)

if test_pred.isna().any():
    g3 = (
        train_raw.groupby(["R", "C", "time_step"], sort=False)["pressure"]
        .median()
        .reset_index()
    )
    test_pred3 = test_raw.merge(g3, on=["R", "C", "time_step"], how="left")["pressure"]
    test_pred = test_pred.fillna(test_pred3)

global_med = float(train_raw["pressure"].median())
test_pred = test_pred.fillna(global_med)

g_exp = (
    train_raw[train_raw["u_out"] == 1]
    .groupby(["R", "C", "count", "time_step"], sort=False)["pressure"]
    .median()
    .reset_index()
)
exp_pred = test_raw.merge(g_exp, on=["R", "C", "count", "time_step"], how="left")[
    "pressure"
].to_numpy(dtype=np.float32)

test_pred_np = test_pred.to_numpy(dtype=np.float32)
u_out_test = test_df["u_out"].to_numpy(dtype=np.int16)

use_exp = (u_out_test == 1) & np.isfinite(exp_pred)
test_pred_np = np.where(use_exp, exp_pred, test_pred_np).astype(np.float32)

print(
    "Pred stats:",
    np.nanmin(test_pred_np),
    np.nanmax(test_pred_np),
    np.mean(test_pred_np),
)

gc.collect()



## === cell 7
if "test_pred_np" not in globals():
    raise RuntimeError("test_pred_np was not created; check earlier cells for errors.")

submission = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
submission["pressure"] = test_pred_np

submission["pressure"] = (
    np.round((submission["pressure"].to_numpy() - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission["pressure"].to_numpy(), P_MIN, P_MAX)
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: median_submission.csv")
