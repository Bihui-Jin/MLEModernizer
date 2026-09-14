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

0.1536854960861055

# 6. Current score

1.7529

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.73489) has done: 'The error is caused by referencing external Kaggle dataset files that are not available in your environment, so the blend step fails before any submission is written. I keep the existing rounding-to-nearest-pressure core logic, but make the pipeline robust by (1) auto-discovering any candidate submission CSVs under `../input/` (if present) and blending them, and (2) providing a safe fallback baseline submission (predicting a central pressure and rounding) when no external predictions exist. I also ensure the output is always written as `submission.csv` with the required `id,pressure` columns and correct row alignment with the provided sample submission.'
- What this solution (achieved 6.19023) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that the code is not actually modeling the time-series at all—it's usually falling back to a constant “median pressure” baseline (or blending arbitrary CSVs), which yields a huge MAE. To move the score strongly toward the target while keeping changes minimal and within your installed packages, I keep your existing “snap predictions to nearest valid pressure” core logic, but replace the fallback with a legitimate, fast baseline model trained from `train.csv`: a per-(R,C,time_step,u_out) mean-pressure lookup with sensible backoffs. This preserves evaluation semantics, runs in time, avoids external datasets, and produce a valid `submission.csv` aligned by `id`.'
- What this solution (achieved 6.19023) has done: 'Your current MAE (6.19, lower is better) is still far from the target (~0.154), and the biggest remaining issue is the hard rule `u_out==1 -> pressure=0`, which is incorrect because the metric ignores expiratory phase rather than forcing those pressures to zero. I keep your existing “lookup-table mean by (R,C,t_idx) with backoffs” approach and your “snap to nearest valid pressure” logic, but remove that zeroing so predictions remain consistent with learned training pressures. I also make the lookup slightly more specific (add a (R,C,t_idx,u_out) table first, then back off), which is a minimal extension of the same core aggregation logic and should reduce MAE without changing modeling families. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.94663) has done: 'Your current approach is a fast per-key mean lookup, but it’s missing the most important driver of pressure in this dataset: the evolution of `u_in` over time and its integral (cumulative inspired volume proxy). To move the MAE substantially toward the target while keeping the same “lookup-table mean + backoff + snap-to-nearest-pressure” core logic, I add two minimal derived features: `u_in` rounded (to reduce sparsity) and `cum_u_in` (cumulative sum per breath, rounded), then make the lookup more specific using these keys with safe backoffs to your existing tables. This preserves your evaluation semantics, stays within pandas/numpy, and still writes a valid `submission.csv` aligned by `id`. The changes are localized to feature creation + a slightly richer sequence of groupby tables and merges.'
- What this solution (achieved 1.74284) has done: 'Your current lookup-table approach is already in the right family, but it’s too sparse because `u_in_r` at 1.0 and `cum_u_in_r` at 2.0 create many unseen key combinations in test, forcing frequent backoff to weaker tables and hurting MAE. I keep the exact same “groupby-mean tables + backoff + snap-to-nearest-valid-pressure” core logic, but slightly coarsen the rounding (especially for `cum_u_in`) so the most specific table hits far more often. I also add one minimal intermediate backoff table that uses both rounded signals without `u_out`, which often differs only in the unscored region and can reduce backoff frequency without changing the modeling approach. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.74339) has done: 'Your current score (1.74284, lower is better) is still far from the target (~0.154), so we should improve performance, but with minimal changes and the same lookup-table + backoff + “snap to nearest valid pressure” core logic. The biggest remaining issue is sparsity/mismatch between train and test keys caused by using raw `cum_u_in` rounding; we can increase hit-rate by normalizing `cum_u_in` by `time_step` (a per-breath flow-like proxy) and using that rounded value in the most-specific lookup, while keeping your existing tables as backoffs. This is a small, local feature addition + one extra table/merge + a slight re-order so we prefer the more stable normalized key before falling back. The submission writing and ID alignment logic stays the same, and the code still runs end-to-end and produces `submission.csv`.'
- What this solution (achieved 2.15318) has done: 'Your current score (1.743, lower is better) is still far above the target (~0.154), and the biggest gap is likely from key mismatch/sparsity between train and test rather than from the “nearest valid pressure” snapping. I keep your exact lookup-table + backoff + snapping core logic, but make two minimal, metric-aligned fixes: (1) build `cum_u_in` from `u_in * delta_time` (a closer proxy to inspired volume than a plain sum) and (2) compute `cum_rate` using the same `delta_time` to avoid instability at very small `time_step`. These are small feature-definition changes that increase hit-rate/consistency for your existing groupby tables without changing the modeling family or training approach. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.74339) has done: 'Your current score is much worse than the target (lower is better), so we should improve, but with minimal changes and the same lookup-table + backoff + “snap to nearest valid pressure” core logic. The biggest issue is that the recent `dt`-weighted cumulative features (`cum_u_in`, `cum_rate`) add noise/mismatch because `dt` is nearly constant; this can reduce key hit-rate and force weaker backoffs, worsening MAE. I keep your tables/merges/backoff order intact, but redefine `cum_u_in` to the standard competition baseline (plain cumulative sum of `u_in` per breath) and define `cum_rate` as `cum_u_in / time_step` (stable, aligned with the dataset’s time grid). These are localized feature-definition changes only, and should move your score back toward your previous better range and closer to the target.'
- What this solution (achieved 1.7754) has done: 'Your score is still far above the target (lower is better), so we should improve accuracy while keeping your existing “groupby mean lookup tables + backoff + snap-to-nearest-valid-pressure” core logic unchanged. The biggest issue is that your current `cum_rate = cum_u_in / time_step` is unstable at very small `time_step` and creates noisy keys, which reduces hit-rate for the `cum_rate_r` table and pushes more rows to weaker backoffs. I redefine `cum_rate` to a more stable per-step flow proxy using `delta_u_in / delta_time` within each breath (with safe handling at the first timestep), and keep the same rounding/backoff structure. This is a localized feature-definition change plus keeping the same merges/tables, and it should move MAE downward toward your target without changing modeling family or training semantics.'
- What this solution (achieved 1.7529) has done: 'Your current score is still much worse than the target (lower is better), so we should legitimately increase accuracy while keeping your same “groupby mean lookup tables + backoff + snap-to-nearest-pressure” core logic. The smallest high-impact fix is to replace the unstable `cum_rate = du/dt` (which explodes at tiny `dt` and mismatches between breaths) with a stable, time-grid-aligned proxy `cum_rate = cum_u_in / (time_step + eps)`, then keep the same rounding and merge/backoff structure. This change keeps the same modeling family (pure aggregation lookups), but should increase table hit-rate and reduce noisy keys, moving MAE downward toward the target. I also ensure the `id` alignment stays correct and still write a valid `submission.csv`.'

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
    loop_time = 150
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
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

sample_sub = pd.read_csv(SAMPLE_PATH)
test = pd.read_csv(TEST_PATH)

train = df_train
train = train.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test = test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

train["t_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)
test["t_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

UIN_ROUND = 2.0  # keep your coarser u_in rounding
CUM_ROUND = 5.0  # keep your coarser cum_u_in rounding

train["u_in_r"] = (np.round(train["u_in"] / UIN_ROUND) * UIN_ROUND).astype(np.float32)
test["u_in_r"] = (np.round(test["u_in"] / UIN_ROUND) * UIN_ROUND).astype(np.float32)

train["cum_u_in"] = (
    train["u_in"]
    .astype(np.float32)
    .groupby(train["breath_id"])
    .cumsum()
    .astype(np.float32)
)
test["cum_u_in"] = (
    test["u_in"]
    .astype(np.float32)
    .groupby(test["breath_id"])
    .cumsum()
    .astype(np.float32)
)

train["cum_u_in_r"] = (np.round(train["cum_u_in"] / CUM_ROUND) * CUM_ROUND).astype(
    np.float32
)
test["cum_u_in_r"] = (np.round(test["cum_u_in"] / CUM_ROUND) * CUM_ROUND).astype(
    np.float32
)

EPS = np.float32(1e-6)
CUM_RATE_ROUND = 2.0  # keep same coarse rounding for match-rate


def add_flow_rate(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["time_step"].astype(np.float32)
    cum = df["cum_u_in"].astype(np.float32)
    rate = (cum / (ts + EPS)).astype(np.float32)

    rate = rate.clip(0.0, 2000.0)

    df["cum_rate"] = rate
    df["cum_rate_r"] = (
        np.round(df["cum_rate"] / CUM_RATE_ROUND) * CUM_RATE_ROUND
    ).astype(np.float32)
    return df


train = add_flow_rate(train)
test = add_flow_rate(test)

tbl_rc_t_u_cum_uin = (
    train.groupby(["R", "C", "t_idx", "u_out", "cum_u_in_r", "u_in_r"], observed=True)[
        "pressure"
    ]
    .mean()
    .rename("p_rc_t_u_cum_uin")
    .reset_index()
)

tbl_rc_t_u_cum = (
    train.groupby(["R", "C", "t_idx", "u_out", "cum_u_in_r"], observed=True)["pressure"]
    .mean()
    .rename("p_rc_t_u_cum")
    .reset_index()
)

tbl_rc_t_u_uin = (
    train.groupby(["R", "C", "t_idx", "u_out", "u_in_r"], observed=True)["pressure"]
    .mean()
    .rename("p_rc_t_u_uin")
    .reset_index()
)

tbl_rc_t_cum_uin = (
    train.groupby(["R", "C", "t_idx", "cum_u_in_r", "u_in_r"], observed=True)[
        "pressure"
    ]
    .mean()
    .rename("p_rc_t_cum_uin")
    .reset_index()
)

tbl_rc_t_u_rate_uin = (
    train.groupby(["R", "C", "t_idx", "u_out", "cum_rate_r", "u_in_r"], observed=True)[
        "pressure"
    ]
    .mean()
    .rename("p_rc_t_u_rate_uin")
    .reset_index()
)

tbl_rc_t_u = (
    train.groupby(["R", "C", "t_idx", "u_out"], observed=True)["pressure"]
    .mean()
    .rename("p_rc_t_u")
    .reset_index()
)

tbl_rc_t = (
    train.groupby(["R", "C", "t_idx"], observed=True)["pressure"]
    .mean()
    .rename("p_rc_t")
    .reset_index()
)

tbl_rc = (
    train.groupby(["R", "C"], observed=True)["pressure"]
    .mean()
    .rename("p_rc")
    .reset_index()
)

global_p = float(train["pressure"].mean())

pred_df = test[
    ["id", "R", "C", "t_idx", "u_out", "cum_u_in_r", "cum_rate_r", "u_in_r"]
].copy()

pred_df = pred_df.merge(
    tbl_rc_t_u_cum_uin,
    on=["R", "C", "t_idx", "u_out", "cum_u_in_r", "u_in_r"],
    how="left",
)
pred_df = pred_df.merge(
    tbl_rc_t_u_rate_uin,
    on=["R", "C", "t_idx", "u_out", "cum_rate_r", "u_in_r"],
    how="left",
)
pred_df = pred_df.merge(
    tbl_rc_t_u_cum,
    on=["R", "C", "t_idx", "u_out", "cum_u_in_r"],
    how="left",
)
pred_df = pred_df.merge(
    tbl_rc_t_u_uin,
    on=["R", "C", "t_idx", "u_out", "u_in_r"],
    how="left",
)
pred_df = pred_df.merge(
    tbl_rc_t_cum_uin,
    on=["R", "C", "t_idx", "cum_u_in_r", "u_in_r"],
    how="left",
)
pred_df = pred_df.merge(tbl_rc_t_u, on=["R", "C", "t_idx", "u_out"], how="left")
pred_df = pred_df.merge(tbl_rc_t, on=["R", "C", "t_idx"], how="left")
pred_df = pred_df.merge(tbl_rc, on=["R", "C"], how="left")

pred = pred_df["p_rc_t_u_cum_uin"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc_t_u_rate_uin"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc_t_u_cum"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc_t_u_uin"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc_t_cum_uin"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc_t_u"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc_t"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = pred_df.loc[mask_nan, "p_rc"].to_numpy()

mask_nan = np.isnan(pred)
if mask_nan.any():
    pred[mask_nan] = global_p

out = sample_sub.copy()
out = out.set_index("id").reindex(pred_df["id"].values).reset_index()
out["pressure"] = pred

out["pressure"] = out["pressure"].apply(find_nearest)

out = out.set_index("id").reindex(sample_sub["id"]).reset_index()
out[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(out.head())
print("Pred stats:", pd.Series(out["pressure"]).describe())
