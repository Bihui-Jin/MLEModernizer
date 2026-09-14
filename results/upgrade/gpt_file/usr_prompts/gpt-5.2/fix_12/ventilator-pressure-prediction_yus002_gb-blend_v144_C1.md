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

0.1367879229143963

# 6. Current score

1.35183

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13673) has done: 'The failure comes from trying to read two external “blend.csv” files from a Kaggle dataset (`gb-data-blending-recover`) that isn’t available in your environment. To make the notebook run end-to-end and still produce a valid `submission.csv`, I keep your existing pressure “snap-to-known-levels” logic, but replace the missing-blend step with a simple, deterministic baseline prediction computed only from `train.csv` (mean pressure per `(R,C,time_step)`), with a safe fallback to global mean for unseen keys. This is score-oriented (should be far better than all-zeros) while staying minimal and using only provided data. Finally, it writes a valid `submission.csv` with columns `id,pressure`.'
- What this solution (achieved 8.16937) has done: 'Your current score (8.13673 MAE) is far worse than the target (0.1368), so we should cautiously improve without changing the modeling approach. The biggest issue is that averaging by `(R, C, time_step)` ignores the dominant signal in `u_in` and the inspiratory/expiratory dynamics, so I keep your same “train-derived lookup baseline + snap-to-known-pressures” core logic but extend the grouping key to include `u_out` and a rounded `u_in` bin. This remains a deterministic, train-only mean-encoding baseline and should materially reduce MAE while preserving your evaluation semantics and nearest-pressure post-processing. The output still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.1221) has done: 'Your current MAE (8.169) is far above the target (0.1368), so we should improve it while keeping your same “train lookup baseline + snap-to-known-pressures” core logic. The main weakness is using raw `time_step` and exact `u_in` bins, which causes many unseen keys in test and forces fallback to the global mean; we reduce miss rate by using a discrete within-breath `step` (0–79) instead of float `time_step`, and by using slightly coarser `u_in` binning (0.5) to generalize better. We keep the same deterministic mean-encoding approach and the same `find_nearest` post-processing, but add a simple backoff: first try the full key, then fall back to a less specific key before using the global mean. This should materially reduce MAE without changing the overall approach, and it still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.64195) has done: 'Your current MAE (4.1221) is still far above the target (0.1368), so we should improve predictions while keeping your same “train lookup mean + backoff + snap-to-known-pressures” core approach. The biggest remaining misspecification is that pressure depends strongly on recent control history, not just the current step, so we minimally extend the grouping keys to include short lags of `u_in` and cumulative inspired volume proxy (`u_in` cumulative sum) in binned form. We keep the same deterministic merging/backoff structure (full key → simpler key → global mean) and the same `find_nearest` post-processing, so evaluation semantics stay the same. This should reduce MAE substantially by capturing within-breath dynamics without changing the overall modeling strategy.'
- What this solution (achieved 2.40495) has done: 'Your current MAE (2.64195) is still far above the target (0.13679), so we should improve predictions while keeping your same “train lookup mean + backoff + snap-to-known-pressures” core logic. The smallest high-impact fix is to replace the coarse cumulative-sum proxy with a physically closer proxy: integrate flow over time via `u_in * delta_time` within each breath, then bin it; this captures the fact that pressure tracks delivered volume. We keep your exact groupby-mean/merge/backoff structure and your `find_nearest` post-processing unchanged, only swapping one history feature and adding one extra backoff level that drops the new proxy when missing. This should reduce MAE meaningfully without changing the approach or requiring any new packages, and it still writes a valid `submission.csv`.'
- What this solution (achieved 2.47115) has done: 'We keep your same deterministic “train lookup mean + backoff + snap-to-known-pressures” approach, but add two minimal history signals that better encode the inspiratory dynamics without changing any modeling paradigm. First, we add lagged `u_out` (since the scoring ignores expiratory phase and valve state changes matter) and a second lag of binned `u_in` to reduce ambiguity across similar states. Second, we add one extra backoff level that drops only the new features before falling back to the coarse `(R,C,u_out,step)` mean, reducing NaNs while still using the richer keys when available. These changes should improve MAE (lower is better) from 2.40 toward the target, while preserving your core logic and still producing a valid `submission.csv`.'
- What this solution (achieved 2.19077) has done: 'Your current MAE (2.471) is far above the target (0.1368), so we should improve while keeping your exact “train lookup mean + backoff + snap-to-known-pressures” core logic. The most impactful minimal change is to reduce key sparsity (too many exact-key misses) by coarsening only the most cardinal history feature (`u_in_vol_bin`) and adding one extra backoff level that keeps the volume proxy but drops the most fragile lag2 term. This preserves the same deterministic groupby-mean/merge/backoff semantics and the same `find_nearest` post-processing, but should substantially increase hit-rate and lower MAE. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.65993) has done: 'Your current MAE (2.19077, lower is better) is still far above the target (0.13679), so we should improve with the smallest possible changes that keep your existing “train lookup mean + multi-level backoff + snap-to-known-pressures” approach intact. The main issue is key sparsity/mismatch between train and test due to rounding-based bins; we switch the binning to deterministic flooring (consistent discretization) and slightly coarsen the most fragile continuous-history bin (`u_in_bin` and `u_in_vol_bin`) to increase exact-key hit rate, which should reduce MAE. We also add one additional intermediate backoff that drops only `u_in_lag2_bin` (a particularly sparse key) while keeping the volume proxy, improving coverage without changing the modeling paradigm. The rest (groupby-means, merge-based backoff, and `find_nearest` pressure snapping) remains the same and it still write a valid `submission.csv`.'
- What this solution (achieved 1.35183) has done: 'Your current MAE (1.65993) is still far above the target (0.13679), so we should improve accuracy while keeping your same deterministic “train groupby-mean lookup + multi-level backoff + snap-to-known-pressures” approach intact. The smallest high-impact change is to reduce key sparsity by slightly coarsening the two most brittle bins (`u_in_bin` and `u_in_vol_bin`) so more test rows hit the richer lookups instead of falling through to weak backoffs/global mean. To avoid hurting cases where the finer binning was correct, we keep the exact same backoff structure but add one additional “fine-bin” lookup level before the coarser one, so the model uses fine bins when available and otherwise uses coarse bins. This should move the score down (better) toward the target without changing core logic, training paradigm, or post-processing.'
- What this solution (achieved 1.35183) has done: 'To move MAE down from 1.35 toward the 0.1368 target without changing your core “train groupby-mean lookup + multi-level backoff + snap-to-known-pressures” logic, the most impactful minimal change is to better align the lookup keys with the metric: the competition scores only inspiratory timesteps (`u_out==0`). I keep your full feature set and backoff structure, but compute the grouped means only on inspiratory rows (and add `u_out==1` backoffs that don’t rely on pressure labels that aren’t scored), which typically improves generalization for the scored region. I also add one very small “inspiratory-only” fallback level before the global mean, so fewer scored rows fall all the way back. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.35183) has done: 'We keep your exact deterministic “train groupby-mean lookup + multi-level backoff + snap-to-known-pressures” approach, but fix the biggest misalignment: you’re currently training all lookup tables only on inspiratory rows while still using `u_out` in the keys, which makes every `u_out==1` test row fall to weak fallbacks and also distorts boundaries around valve transitions. The minimal improvement is to build lookup tables from the full train set (both phases) while keeping the same feature set, backoff order, and nearest-pressure snapping; this typically lowers overall MAE substantially without changing the core logic. To preserve the metric focus, we only use inspiratory global mean as the last-resort fill (so the scored region isn’t pulled toward expiratory pressures). Finally, we ensure `id` alignment stays correct and a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd
import gc



## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 154
    splits = file_count // 2
    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

df_train = df_train.copy()
df_test = df_test.copy()

df_train["step"] = df_train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
df_test["step"] = df_test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

UIN_BIN_FINE = 1.0
UIN_BIN_COARSE = 2.0

df_train["u_in_bin_fine"] = np.floor(df_train["u_in"] / UIN_BIN_FINE).astype(np.int16)
df_test["u_in_bin_fine"] = np.floor(df_test["u_in"] / UIN_BIN_FINE).astype(np.int16)

df_train["u_in_bin"] = np.floor(df_train["u_in"] / UIN_BIN_COARSE).astype(np.int16)
df_test["u_in_bin"] = np.floor(df_test["u_in"] / UIN_BIN_COARSE).astype(np.int16)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)

    df["u_in_lag1_bin_fine"] = np.floor(df["u_in_lag1"] / UIN_BIN_FINE).astype(np.int16)
    df["u_in_lag2_bin_fine"] = np.floor(df["u_in_lag2"] / UIN_BIN_FINE).astype(np.int16)

    df["u_in_lag1_bin"] = np.floor(df["u_in_lag1"] / UIN_BIN_COARSE).astype(np.int16)
    df["u_in_lag2_bin"] = np.floor(df["u_in_lag2"] / UIN_BIN_COARSE).astype(np.int16)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    dt = (
        g["time_step"].diff().fillna(df["time_step"]).clip(lower=0.0).astype(np.float32)
    )
    df["u_in_dt"] = (df["u_in"].astype(np.float32) * dt).astype(np.float32)
    df["u_in_vol"] = g["u_in_dt"].cumsum().astype(np.float32)

    VOL_BIN_FINE = 1.0
    VOL_BIN_COARSE = 2.0
    df["u_in_vol_bin_fine"] = np.floor(df["u_in_vol"] / VOL_BIN_FINE).astype(np.int16)
    df["u_in_vol_bin"] = np.floor(df["u_in_vol"] / VOL_BIN_COARSE).astype(np.int16)

    return df


df_train = add_history_features(df_train)
df_test = add_history_features(df_test)

df_train_full = df_train
df_train_insp = df_train.loc[df_train["u_out"] == 0].copy()

grp_cols_full_fine = [
    "R",
    "C",
    "u_out",
    "u_out_lag1",
    "step",
    "u_in_bin_fine",
    "u_in_lag1_bin_fine",
    "u_in_lag2_bin_fine",
    "u_in_vol_bin_fine",
]
mean_pressure_full_fine = (
    df_train_full.groupby(grp_cols_full_fine, observed=True)["pressure"]
    .mean()
    .reset_index()
)
test_pred = df_test.merge(mean_pressure_full_fine, on=grp_cols_full_fine, how="left")

grp_cols_full = [
    "R",
    "C",
    "u_out",
    "u_out_lag1",
    "step",
    "u_in_bin",
    "u_in_lag1_bin",
    "u_in_lag2_bin",
    "u_in_vol_bin",
]
mean_pressure_full = (
    df_train_full.groupby(grp_cols_full, observed=True)["pressure"].mean().reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_full]
        .merge(mean_pressure_full, on=grp_cols_full, how="left")["pressure"]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

grp_cols_backoff00 = [
    "R",
    "C",
    "u_out",
    "u_out_lag1",
    "step",
    "u_in_bin",
    "u_in_lag1_bin",
    "u_in_vol_bin",
]
mean_pressure_backoff00 = (
    df_train_full.groupby(grp_cols_backoff00, observed=True)["pressure"]
    .mean()
    .reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_backoff00]
        .merge(mean_pressure_backoff00, on=grp_cols_backoff00, how="left")["pressure"]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

grp_cols_backoff00b = [
    "R",
    "C",
    "u_out",
    "step",
    "u_in_bin",
    "u_in_lag1_bin",
    "u_in_vol_bin",
]
mean_pressure_backoff00b = (
    df_train_full.groupby(grp_cols_backoff00b, observed=True)["pressure"]
    .mean()
    .reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_backoff00b]
        .merge(mean_pressure_backoff00b, on=grp_cols_backoff00b, how="left")["pressure"]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

grp_cols_backoff0 = [
    "R",
    "C",
    "u_out",
    "u_out_lag1",
    "step",
    "u_in_bin",
    "u_in_lag1_bin",
    "u_in_lag2_bin",
]
mean_pressure_backoff0 = (
    df_train_full.groupby(grp_cols_backoff0, observed=True)["pressure"]
    .mean()
    .reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_backoff0]
        .merge(mean_pressure_backoff0, on=grp_cols_backoff0, how="left")["pressure"]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

grp_cols_backoff1 = ["R", "C", "u_out", "step", "u_in_bin", "u_in_lag1_bin"]
mean_pressure_backoff1 = (
    df_train_full.groupby(grp_cols_backoff1, observed=True)["pressure"]
    .mean()
    .reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_backoff1]
        .merge(mean_pressure_backoff1, on=grp_cols_backoff1, how="left")["pressure"]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

grp_cols_backoff2 = ["R", "C", "u_out", "step"]
mean_pressure_backoff2 = (
    df_train_full.groupby(grp_cols_backoff2, observed=True)["pressure"]
    .mean()
    .reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_backoff2]
        .merge(mean_pressure_backoff2, on=grp_cols_backoff2, how="left")["pressure"]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

grp_cols_insp_fallback = ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_vol_bin"]
mean_pressure_insp_fallback = (
    df_train_insp.groupby(grp_cols_insp_fallback, observed=True)["pressure"]
    .mean()
    .reset_index()
)
missing_mask = test_pred["pressure"].isna()
if missing_mask.any():
    tmp = (
        df_test.loc[missing_mask, grp_cols_insp_fallback]
        .merge(mean_pressure_insp_fallback, on=grp_cols_insp_fallback, how="left")[
            "pressure"
        ]
        .to_numpy()
    )
    test_pred.loc[missing_mask, "pressure"] = tmp

global_mean = float(df_train_insp["pressure"].mean())
test_pred["pressure"] = test_pred["pressure"].fillna(global_mean)

test_pred["pressure"] = test_pred["pressure"].apply(find_nearest)

submission = test_pred.loc[:, ["id", "pressure"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("NaN pressures in submission:", submission["pressure"].isna().sum())
print(
    "pressure min/max:",
    float(submission["pressure"].min()),
    float(submission["pressure"].max()),
)
