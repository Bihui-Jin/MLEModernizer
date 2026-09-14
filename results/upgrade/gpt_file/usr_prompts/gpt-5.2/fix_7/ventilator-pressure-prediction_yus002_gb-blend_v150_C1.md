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

0.1413495026664843

# 6. Current score

7.2386

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 3.89135) has done: 'Your notebook fails because it tries to blend two external submission files that are not present in this Kaggle environment. I remove that dependency and instead generate a baseline prediction directly from the provided `train.csv`/`test.csv` using a minimal group-mean mapping (by `R`, `C`, `time_step`, `u_out`, and binned `u_in`) and then snap predictions to the nearest valid pressure value as your code already intends. This keeps the core “nearest pressure grid” post-processing and produces a valid `submission.csv` end-to-end. It should also yield a non-trivial score (better than all-zeros) and move toward your target since you currently have no valid submission.'
- What this solution (achieved 8.19027) has done: 'Your current approach is a pure group-mean lookup, which leaves a lot of “unseen key” cases and noisy means, keeping MAE far from your target. I keep the same core logic (group mean mapping + fallback chain + snapping to nearest valid pressure), but strengthen it with two minimal changes: (1) add a more informative first-stage key by including lagged control features (`u_in`/`u_out` from the previous timestep within each breath), and (2) use the *median* instead of mean in the lookup tables to reduce sensitivity to outliers and better match MAE. The rest (fallback hierarchy, nearest-pressure post-processing, submission writing) stays the same, and it still run within the time limit and produce `submission.csv`.'
- What this solution (achieved 2.63402) has done: 'Your current gap to the target is huge (MAE 8.19 vs 0.141, lower is better), so we need a meaningful but still “same-core-idea” improvement: keep the lookup-table approach, but make it much more expressive so more test rows hit high-quality keys. I add two minimal, very standard time-series features within each breath—cumulative inspired volume estimate (`u_in` integral) and a short rolling mean of `u_in`—and include their lightly-binned versions in the first-stage mapping; this preserves the same group-median lookup + fallback chain + nearest-pressure snapping. I also fix a silent but major issue: you sort `train_key`/`test_key` for lag features, but then merge predictions back to `sample_submission` without restoring original row order, which can severely damage score; we keep and use the original test row order. These changes should improve the MAE substantially while staying within the same overall logic and runtime budget and still writing a valid `submission.csv`.'
- What this solution (achieved 2.63402) has done: 'Your current score (2.63402 MAE, lower is better) is still far from the target (0.1413), so we need a meaningful improvement while keeping the same “lookup-table with fallback + snap-to-valid-pressure-grid” core logic. The biggest low-risk gain here is to align the lookup keys with the evaluation: the metric ignores expiratory phase (u_out=1), so we should predict a stable, low-variance value there instead of trying to match noisy mappings. Concretely, we compute per-(R,C) inspiratory pressure medians and use them to fill all u_out=1 rows (and also as a stronger fallback than a global median), while keeping your existing hierarchical median maps and the final nearest-pressure snapping. This should reduce MAE substantially without changing the overall approach or adding any modeling/training.'
- What this solution (achieved 7.2386) has done: 'Your current MAE (2.634) is still far above the target (0.141, lower is better), so we should improve accuracy while keeping the same core “hierarchical group-median lookup + fallback + snap-to-valid-pressure-grid” approach. The biggest low-risk gain is to stop rounding `time_step` (it already matches exactly between train/test); rounding can create key collisions and blur distinct dynamics, hurting the lookup precision. I also add one more minimal, consistent time-series feature (lag2 of `u_in`/`u_out`) into the first-stage key (with a fallback that drops it), which increases exact-key hit rate without changing the overall method. Finally, I keep your expiratory-phase handling but compute the `(R,C)` fallback median from *all* phases (since test u_out=1 is not scored anyway), which stabilizes the fill values with more data.'
- What this solution (achieved 7.2386) has done: 'Your current score (7.2386 MAE, lower is better) is far above the target (0.1413), so we need a meaningful accuracy boost while keeping your same core approach: hierarchical group-median lookup tables with fallbacks plus snapping to the nearest valid pressure grid. The biggest low-risk gain is to add two more within-breath state features that strongly correlate with pressure but don’t change the “lookup” nature: a cumulative `u_out` time (how long we’ve been in expiratory) and a rolling mean of `u_in` over a slightly longer window. We then include lightly binned versions of these features only in the top-level key (and one fallback), keeping the rest of your fallback chain intact so coverage remains high. This should reduce error substantially versus the current keys while preserving identical evaluation semantics and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
import copy
import glob
import random
from random import random as rd
import gc

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
        weight1 = (l[1] / l_sum) + 0.15
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
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
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
    a.pressure = a.pressure * 0.69 + b.pressure * 0.31
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

train_key = df_train[
    ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
].copy()
test_key = df_test[["breath_id", "R", "C", "time_step", "u_in", "u_out"]].copy()

train_key["_row"] = np.arange(len(train_key), dtype=np.int64)
test_key["_row"] = np.arange(len(test_key), dtype=np.int64)

train_key.sort_values(["breath_id", "time_step"], inplace=True)
test_key.sort_values(["breath_id", "time_step"], inplace=True)

train_key["u_in_lag1"] = train_key.groupby("breath_id", sort=False)["u_in"].shift(1)
train_key["u_out_lag1"] = train_key.groupby("breath_id", sort=False)["u_out"].shift(1)
train_key["u_in_lag2"] = train_key.groupby("breath_id", sort=False)["u_in"].shift(2)
train_key["u_out_lag2"] = train_key.groupby("breath_id", sort=False)["u_out"].shift(2)

test_key["u_in_lag1"] = test_key.groupby("breath_id", sort=False)["u_in"].shift(1)
test_key["u_out_lag1"] = test_key.groupby("breath_id", sort=False)["u_out"].shift(1)
test_key["u_in_lag2"] = test_key.groupby("breath_id", sort=False)["u_in"].shift(2)
test_key["u_out_lag2"] = test_key.groupby("breath_id", sort=False)["u_out"].shift(2)

train_key["u_in_lag1"] = train_key["u_in_lag1"].fillna(train_key["u_in"])
train_key["u_out_lag1"] = train_key["u_out_lag1"].fillna(train_key["u_out"])
train_key["u_in_lag2"] = train_key["u_in_lag2"].fillna(train_key["u_in_lag1"])
train_key["u_out_lag2"] = train_key["u_out_lag2"].fillna(train_key["u_out_lag1"])

test_key["u_in_lag1"] = test_key["u_in_lag1"].fillna(test_key["u_in"])
test_key["u_out_lag1"] = test_key["u_out_lag1"].fillna(test_key["u_out"])
test_key["u_in_lag2"] = test_key["u_in_lag2"].fillna(test_key["u_in_lag1"])
test_key["u_out_lag2"] = test_key["u_out_lag2"].fillna(test_key["u_out_lag1"])

train_key["time_step_r"] = train_key["time_step"]
test_key["time_step_r"] = test_key["time_step"]

train_key["u_in_b"] = (train_key["u_in"] * 2).round().astype(np.int32) / 2.0
test_key["u_in_b"] = (test_key["u_in"] * 2).round().astype(np.int32) / 2.0

train_key["u_in_lag1_b"] = (train_key["u_in_lag1"] * 2).round().astype(np.int32) / 2.0
test_key["u_in_lag1_b"] = (test_key["u_in_lag1"] * 2).round().astype(np.int32) / 2.0
train_key["u_in_lag2_b"] = (train_key["u_in_lag2"] * 2).round().astype(np.int32) / 2.0
test_key["u_in_lag2_b"] = (test_key["u_in_lag2"] * 2).round().astype(np.int32) / 2.0

train_key["u_out_lag1"] = train_key["u_out_lag1"].astype(np.int8)
test_key["u_out_lag1"] = test_key["u_out_lag1"].astype(np.int8)
train_key["u_out_lag2"] = train_key["u_out_lag2"].astype(np.int8)
test_key["u_out_lag2"] = test_key["u_out_lag2"].astype(np.int8)

dt_train = train_key.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
dt_test = test_key.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)

train_key["u_in_cum"] = (
    (train_key["u_in"] * dt_train).groupby(train_key["breath_id"], sort=False).cumsum()
)
test_key["u_in_cum"] = (
    (test_key["u_in"] * dt_test).groupby(test_key["breath_id"], sort=False).cumsum()
)

train_key["u_in_roll3"] = (
    train_key.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
test_key["u_in_roll3"] = (
    test_key.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

train_key["u_in_cum_b"] = (train_key["u_in_cum"] * 10).round().astype(np.int32) / 10.0
test_key["u_in_cum_b"] = (test_key["u_in_cum"] * 10).round().astype(np.int32) / 10.0
train_key["u_in_roll3_b"] = (train_key["u_in_roll3"] * 2).round().astype(np.int32) / 2.0
test_key["u_in_roll3_b"] = (test_key["u_in_roll3"] * 2).round().astype(np.int32) / 2.0

train_key["u_in_roll5"] = (
    train_key.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
test_key["u_in_roll5"] = (
    test_key.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
train_key["u_in_roll5_b"] = (train_key["u_in_roll5"] * 2).round().astype(np.int32) / 2.0
test_key["u_in_roll5_b"] = (test_key["u_in_roll5"] * 2).round().astype(np.int32) / 2.0

train_key["u_out_cumtime"] = (
    (train_key["u_out"].astype(np.float32) * dt_train)
    .groupby(train_key["breath_id"], sort=False)
    .cumsum()
)
test_key["u_out_cumtime"] = (
    (test_key["u_out"].astype(np.float32) * dt_test)
    .groupby(test_key["breath_id"], sort=False)
    .cumsum()
)
train_key["u_out_cumtime_b"] = (train_key["u_out_cumtime"] * 20).round().astype(
    np.int32
) / 20.0
test_key["u_out_cumtime_b"] = (test_key["u_out_cumtime"] * 20).round().astype(
    np.int32
) / 20.0

group_cols_1 = [
    "R",
    "C",
    "time_step_r",
    "u_out",
    "u_in_b",
    "u_out_lag1",
    "u_in_lag1_b",
    "u_out_lag2",
    "u_in_lag2_b",
    "u_in_cum_b",
    "u_in_roll3_b",
    "u_in_roll5_b",
    "u_out_cumtime_b",
]
map1 = train_key.groupby(group_cols_1, sort=False)["pressure"].median().reset_index()
pred = test_key.merge(map1, on=group_cols_1, how="left")["pressure"].to_numpy()

if np.isnan(pred).any():
    group_cols_2 = [
        "R",
        "C",
        "time_step_r",
        "u_out",
        "u_in_b",
        "u_out_lag1",
        "u_in_lag1_b",
        "u_in_cum_b",
        "u_in_roll3_b",
        "u_in_roll5_b",
    ]
    map2 = (
        train_key.groupby(group_cols_2, sort=False)["pressure"].median().reset_index()
    )
    pred2 = test_key.merge(map2, on=group_cols_2, how="left")["pressure"].to_numpy()
    pred = np.where(np.isnan(pred), pred2, pred)

if np.isnan(pred).any():
    group_cols_3 = [
        "R",
        "C",
        "time_step_r",
        "u_out",
        "u_in_b",
        "u_in_cum_b",
        "u_in_roll3_b",
    ]
    map3 = (
        train_key.groupby(group_cols_3, sort=False)["pressure"].median().reset_index()
    )
    pred3 = test_key.merge(map3, on=group_cols_3, how="left")["pressure"].to_numpy()
    pred = np.where(np.isnan(pred), pred3, pred)

if np.isnan(pred).any():
    group_cols_4 = ["R", "C", "time_step_r", "u_out", "u_in_b"]
    map4 = (
        train_key.groupby(group_cols_4, sort=False)["pressure"].median().reset_index()
    )
    pred4 = test_key.merge(map4, on=group_cols_4, how="left")["pressure"].to_numpy()
    pred = np.where(np.isnan(pred), pred4, pred)

if np.isnan(pred).any():
    group_cols_5 = ["R", "C", "time_step_r", "u_out"]
    map5 = (
        train_key.groupby(group_cols_5, sort=False)["pressure"].median().reset_index()
    )
    pred5 = test_key.merge(map5, on=group_cols_5, how="left")["pressure"].to_numpy()
    pred = np.where(np.isnan(pred), pred5, pred)

if np.isnan(pred).any():
    group_cols_6 = ["R", "C", "u_out"]
    map6 = (
        train_key.groupby(group_cols_6, sort=False)["pressure"].median().reset_index()
    )
    pred6 = test_key.merge(map6, on=group_cols_6, how="left")["pressure"].to_numpy()
    pred = np.where(np.isnan(pred), pred6, pred)

rc_all_median = (
    train_key.groupby(["R", "C"], sort=False)["pressure"].median().reset_index()
)
rc_m = (
    test_key[["R", "C"]]
    .merge(rc_all_median, on=["R", "C"], how="left")["pressure"]
    .to_numpy()
)
pred = np.where(np.isnan(pred), rc_m, pred)

exp_mask = test_key["u_out"].to_numpy() == 1
pred = np.where(exp_mask, rc_m, pred)

if np.isnan(pred).any():
    pred = np.where(np.isnan(pred), df_train["pressure"].median(), pred)

pred = np.vectorize(find_nearest, otypes=[float])(pred)

test_rows = test_key[["_row"]].copy()
test_rows["pressure"] = pred.astype(float)
test_rows.sort_values("_row", inplace=True)
pred_ordered = test_rows["pressure"].to_numpy()

submission = sample_sub.copy()
submission["pressure"] = pred_ordered
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
