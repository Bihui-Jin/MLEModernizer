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

3.9

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

0.1536168206104505

# 6. Current score

1.55383

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.46367) has done: 'I remove the dependency on missing external blend files and instead generate a valid submission directly from the provided competition data. To keep changes minimal and preserve your existing “snap predictions to nearest training pressure” core idea, I create a simple, deterministic baseline prediction (median inspiratory pressure per (R, C) from train) and then apply your existing `find_nearest` mapping. I also fix the notebook cell numbering to start at 1 and ensure the script always writes a properly named `submission.csv` with the required `id,pressure` columns. This run end-to-end in the Kaggle environment using only the provided `/kaggle/input/ventilator-pressure-prediction/*` files.'
- What this solution (achieved 6.02322) has done: 'Your current score is far above the target (lower is better), so we should make a small but meaningful improvement without changing the overall “simple deterministic baseline + snap-to-nearest-pressure” core idea. The biggest issue is that predicting a single constant per (R,C) ignores the strong time-series structure; we can still stay within your baseline approach by switching to a per-time_step median per (R,C,time_step) computed on inspiratory rows only, with a safe fallback hierarchy ((R,C,time_step) → (R,C) → global). This keeps the same semantics (median-based lookup + `find_nearest`) but should substantially reduce MAE because test breaths share the same discrete time grid. We also ensure correct alignment and stable output formatting to always produce a valid `submission.csv`.'
- What this solution (achieved 2.38657) has done: 'Your current score (6.02322, lower-is-better) is far from the target (0.1536), so we need a meaningful but still “lookup-based” improvement without changing your overall approach of deterministic aggregation + snapping to the nearest allowed pressure. The main missing piece is that the metric ignores expiratory phase (u_out==1), but your submission currently predicts non-zero pressures there, which gets fully penalized; we set predictions to 0 for u_out==1 in test to match the evaluation phase. To further move toward the target while preserving the same median-lookup core logic, we add a light feature extension using cumulative inspired volume proxy (`u_in` cumulative sum within each breath) and take median by (R,C,time_step,volume_bin) with safe fallbacks. All changes are localized to the prediction table building and keep `find_nearest` snapping intact while still writing a valid `submission.csv`.'
- What this solution (achieved 2.87634) has done: 'Your current score is still much worse than the target (lower is better), so we make a small, fully deterministic improvement while keeping your same “median lookup table + snap-to-nearest pressure” core idea. The biggest remaining gap comes from using a very coarse proxy (`u_in` cumulative sum bins) that can mismatch between breaths; we can improve by switching the lookup key to the actual control inputs at each time step via a light discretization of `u_in` (and optionally `u_in` lag), which stays within the same aggregation approach. We keep the inspiratory-only aggregation, keep setting `u_out==1` predictions to 0, keep `find_nearest`, and add a strict fallback chain so every row gets a valid prediction. This should reduce MAE materially without changing the overall method or requiring any new packages.'
- What this solution (achieved 1.53747) has done: 'We keep your deterministic “median lookup table + snap to nearest training pressure” approach, but make the lookup slightly more faithful to the control signal by using a fixed-width discretization of `u_in` (and `u_in` lag) instead of quantile bins, which are more sensitive to distribution differences between breaths. We also add one more very local key that often helps with ventilator dynamics while staying in the same aggregation family: the per-breath cumulative `u_in` (as a proxy for inspired volume) with coarse rounding, used only as the most specific lookup and with the same safe fallback chain. Finally, we ensure the merge keys are compact integer types for faster/stabler joins and keep the `u_out==1 -> 0` rule and `find_nearest` snapping unchanged to preserve evaluation semantics.'
- What this solution (achieved 1.56523) has done: 'Your current gap to the target is large (1.53747 vs 0.1536, lower-is-better), so we need a meaningful improvement while keeping your core “median lookup table + snap to nearest training pressure” logic intact. The biggest issue is setting all `u_out==1` test rows to 0, which is not aligned with the competition scoring (those rows are ignored, so we should avoid arbitrary values that can still be evaluated differently than expected); instead we keep a consistent prediction for all rows and only train the lookup on inspiratory rows as you already do. Next, we add one minimal but high-signal key that stays in the same lookup-family: discretized `u_in` *delta* (change from previous step), which captures valve dynamics without changing the modeling approach. Finally, we keep your fallback chain and `find_nearest` snapping, and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.735) has done: 'Your current MAE (1.565) is still far above the target (0.154, lower is better), so we should improve accuracy while keeping your core “median lookup + fallback chain + snap-to-nearest pressure” logic intact. The most impactful minimal fix is to align your engineered cumulative signal with the physics/time grid by making it time-weighted: use per-breath cumulative integral of `u_in` over `time_step` (a closer proxy for inspired volume) instead of a raw cumulative sum, and discretize it coarsely as before. Second, slightly refine `time_step` rounding from 0.01 to 0.001 to reduce key-collisions/mismatches while still being deterministic and cheap. Everything else (groupby medians, merge/fallback chain, and `find_nearest` snapping) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.735) has done: 'Your current MAE (1.735, lower-is-better) is still far from the target (0.1536), so we need a meaningful improvement while keeping your same core “deterministic median lookup + fallback chain + snap-to-nearest pressure” logic. The most impactful minimal change is to explicitly model inspiratory vs expiratory within the lookup by building separate median tables for `u_out==0` and `u_out==1` (instead of predicting everything from inspiratory-only medians). This keeps the exact same aggregation-and-merge semantics, just adds a parallel set of groupbys/merges and uses the correct table conditioned on `u_out`. We also keep your existing engineered keys and snapping unchanged, and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.58081) has done: 'Your current score is far above the target (1.735 vs 0.1536, lower-is-better), so we need a meaningful but still “lookup-median + fallback + snap-to-nearest-pressure” improvement. The biggest minimal win within your existing logic is to condition the lookup on the *breath progress index* (step number 0..79) rather than a rounded float `time_step`, because the grid is discrete and stable and float rounding/matching can silently reduce hit-rate. Second, we add one more low-cost, high-signal key that stays in the same aggregation family: discretized `u_in` *acceleration* (delta of delta), used only at the most-specific levels with the same safe fallback chain. Everything else (separate insp/exp tables, median aggregation, merge-based fallback, and `find_nearest` snapping, plus writing `submission.csv`) remains the same.'
- What this solution (achieved 1.55383) has done: 'Your current MAE is still far above the target (lower is better), so the smallest likely improvement within your exact “median lookup + fallback chain + snap-to-nearest-pressure” core logic is to increase lookup hit-rate without changing the approach. I do that by (1) using rounding (nearest) instead of flooring for the binned control features to reduce systematic bin-shift mismatches between train/test, and (2) adding a very light “u_in over time” feature (`u_in_dt`) into only the most-specific median tables to better distinguish rows that currently collide in the same bins. Everything else (separate insp/exp tables, median aggregation, merge-based fallback chain, and `find_nearest` snapping) stays the same, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
import copy
import glob
import random
from random import random as rd



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




## === cell 2
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
        weight1 = 0.6
        weight2 = 0.4
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**3
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = 0
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        for i in range(len(flist)):
            output.pressure += flist[i] * weight[i]
    output.pressure /= loop_time
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3

    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

train_insp = df_train[df_train["u_out"] == 0].copy()
train_exp = df_train[df_train["u_out"] == 1].copy()

for part in (train_insp, train_exp):
    part = part.sort_values(["breath_id", "time_step"], kind="mergesort")
    part["step"] = part.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    if part.shape[0] == train_insp.shape[0] and part["u_out"].iloc[0] == 0:
        train_insp = part
    else:
        train_exp = part

df_test = df_test.sort_values(["breath_id", "time_step"], kind="mergesort")
df_test["step"] = df_test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

for part in (train_insp, train_exp, df_test):
    part["u_in_lag1"] = (
        part.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
    )
    part["u_in_delta1"] = (part["u_in"] - part["u_in_lag1"]).fillna(0.0)

for part in (train_insp, train_exp, df_test):
    part["u_in_delta1_lag1"] = (
        part.groupby("breath_id", sort=False)["u_in_delta1"].shift(1).fillna(0.0)
    )
    part["u_in_acc1"] = (part["u_in_delta1"] - part["u_in_delta1_lag1"]).fillna(0.0)

bin_w = 2.0
delta_w = 2.0
acc_w = 2.0
for part in (train_insp, train_exp, df_test):
    part["u_in_bin"] = np.rint(part["u_in"] / bin_w).astype(np.int16)
    part["u_in_lag1_bin"] = np.rint(part["u_in_lag1"] / bin_w).astype(np.int16)
    part["u_in_delta1_bin"] = np.clip(
        np.rint(part["u_in_delta1"] / delta_w), -50, 50
    ).astype(np.int16)
    part["u_in_acc1_bin"] = np.clip(np.rint(part["u_in_acc1"] / acc_w), -50, 50).astype(
        np.int16
    )

for part in (train_insp, train_exp, df_test):
    dt = (
        part.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(part["time_step"])
    )
    dt = dt.clip(lower=0.0).astype(np.float32)
    part["u_in_cum"] = (
        (part["u_in"].astype(np.float32) * dt)
        .groupby(part["breath_id"], sort=False)
        .cumsum()
    )
    part["u_in_cum_r"] = (part["u_in_cum"] / 0.5).round(0).astype(np.int16)

    part["u_in_dt"] = (part["u_in"].astype(np.float32) * dt).astype(np.float32)
    part["u_in_dt_r"] = (part["u_in_dt"] / 0.5).round(0).astype(np.int16)

for part in (train_insp, train_exp, df_test):
    part["R"] = part["R"].astype(np.int16)
    part["C"] = part["C"].astype(np.int16)
    part["step"] = part["step"].astype(np.int16)


def build_median_tables(train_part: pd.DataFrame, prefix: str):
    rc_s_u_u1_du_da_cum_dt_median = (
        train_part.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_delta1_bin",
                "u_in_acc1_bin",
                "u_in_cum_r",
                "u_in_dt_r",
            ],
            sort=False,
        )["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_u1_du_da_cum_dt_pressure_median")
        .reset_index()
    )

    rc_s_u_u1_du_da_cum_median = (
        train_part.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_delta1_bin",
                "u_in_acc1_bin",
                "u_in_cum_r",
            ],
            sort=False,
        )["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_u1_du_da_cum_pressure_median")
        .reset_index()
    )

    rc_s_u_u1_du_da_median = (
        train_part.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_delta1_bin",
                "u_in_acc1_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_u1_du_da_pressure_median")
        .reset_index()
    )

    rc_s_u_u1_du_cum_median = (
        train_part.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_delta1_bin",
                "u_in_cum_r",
            ],
            sort=False,
        )["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_u1_du_cum_pressure_median")
        .reset_index()
    )

    rc_s_u_u1_du_median = (
        train_part.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_delta1_bin"],
            sort=False,
        )["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_u1_du_pressure_median")
        .reset_index()
    )

    rc_s_u_u1_cum_median = (
        train_part.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_r"],
            sort=False,
        )["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_u1_cum_pressure_median")
        .reset_index()
    )

    rc_s_u_u1_median = (
        train_part.groupby(["R", "C", "step", "u_in_bin", "u_in_lag1_bin"], sort=False)[
            "pressure"
        ]
        .median()
        .rename(f"{prefix}rc_s_u_u1_pressure_median")
        .reset_index()
    )

    rc_s_u_median = (
        train_part.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .median()
        .rename(f"{prefix}rc_s_u_pressure_median")
        .reset_index()
    )

    rc_s_median = (
        train_part.groupby(["R", "C", "step"], sort=False)["pressure"]
        .median()
        .rename(f"{prefix}rc_s_pressure_median")
        .reset_index()
    )

    rc_median = (
        train_part.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .rename(f"{prefix}rc_pressure_median")
        .reset_index()
    )

    global_median = float(train_part["pressure"].median())

    return {
        "rc_s_u_u1_du_da_cum_dt_median": rc_s_u_u1_du_da_cum_dt_median,
        "rc_s_u_u1_du_da_cum_median": rc_s_u_u1_du_da_cum_median,
        "rc_s_u_u1_du_da_median": rc_s_u_u1_du_da_median,
        "rc_s_u_u1_du_cum_median": rc_s_u_u1_du_cum_median,
        "rc_s_u_u1_du_median": rc_s_u_u1_du_median,
        "rc_s_u_u1_cum_median": rc_s_u_u1_cum_median,
        "rc_s_u_u1_median": rc_s_u_u1_median,
        "rc_s_u_median": rc_s_u_median,
        "rc_s_median": rc_s_median,
        "rc_median": rc_median,
        "global_median": global_median,
        "prefix": prefix,
    }


insp_tables = build_median_tables(train_insp, prefix="insp_")
exp_tables = build_median_tables(train_exp, prefix="exp_")


def apply_tables(test_df: pd.DataFrame, tables: dict):
    tp = test_df.merge(
        tables["rc_s_u_u1_du_da_cum_dt_median"],
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_delta1_bin",
            "u_in_acc1_bin",
            "u_in_cum_r",
            "u_in_dt_r",
        ],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_u1_du_da_cum_median"],
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_delta1_bin",
            "u_in_acc1_bin",
            "u_in_cum_r",
        ],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_u1_du_da_median"],
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_delta1_bin",
            "u_in_acc1_bin",
        ],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_u1_du_cum_median"],
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_delta1_bin",
            "u_in_cum_r",
        ],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_u1_du_median"],
        on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_delta1_bin"],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_u1_cum_median"],
        on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_r"],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_u1_median"],
        on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin"],
        how="left",
    )
    tp = tp.merge(
        tables["rc_s_u_median"], on=["R", "C", "step", "u_in_bin"], how="left"
    )
    tp = tp.merge(tables["rc_s_median"], on=["R", "C", "step"], how="left")
    tp = tp.merge(tables["rc_median"], on=["R", "C"], how="left")

    p = tables["prefix"]
    tp["pressure"] = tp[f"{p}rc_s_u_u1_du_da_cum_dt_pressure_median"]
    tp["pressure"] = tp["pressure"].fillna(
        tp[f"{p}rc_s_u_u1_du_da_cum_pressure_median"]
    )
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_u_u1_du_da_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_u_u1_du_cum_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_u_u1_du_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_u_u1_cum_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_u_u1_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_u_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_s_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tp[f"{p}rc_pressure_median"])
    tp["pressure"] = tp["pressure"].fillna(tables["global_median"]).astype(float)

    return tp[["id", "pressure"]]


test_insp = df_test[df_test["u_out"] == 0].copy()
test_exp = df_test[df_test["u_out"] == 1].copy()

pred_insp = apply_tables(test_insp, insp_tables)
pred_exp = apply_tables(test_exp, exp_tables)

submission = pd.concat([pred_insp, pred_exp], axis=0, ignore_index=True)
submission["pressure"] = submission["pressure"].map(find_nearest).astype(float)

submission = submission.sort_values("id", kind="mergesort")
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("pressure unique (first 10):", np.sort(submission["pressure"].unique())[:10])
