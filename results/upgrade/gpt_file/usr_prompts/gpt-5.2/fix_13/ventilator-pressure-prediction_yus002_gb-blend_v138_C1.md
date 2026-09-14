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

0.1390536612619201

# 6. Current score

1.99578

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The current notebook errors because it tries to read external blended submission files (`gb-data-blending-recover/...`) that are not present in this environment. To make it run end-to-end and still keep the original “blend/nearest-pressure” core semantics, I replace that missing-input blend with a safe fallback that creates a valid submission from the provided `sample_submission.csv` (all zeros) and applies the same nearest-pressure snapping. I also fix the cell numbering to start at 1 as required, and ensure the output is written as a proper `.csv` submission file (`submission.csv`) with the required columns. This yield a valid (but not competitive) score since no real model predictions are available in the provided code/data.'
- What this solution (achieved 5.81838) has done: 'Your current score is very far from the target (lower is better), mainly because the pipeline falls back to predicting all-zeros (then snapping to nearest known pressure), which is essentially a non-model and yields huge MAE. To move the score substantially toward the target while keeping changes minimal, I keep your existing “snap to nearest valid pressure” post-processing and add a simple, fast, deterministic per-(R,C,time_step,u_in,u_out) median lookup built only from `train.csv`, then fall back to a per-(R,C,time_step,u_out) median when the exact `u_in` isn’t seen. This preserves your overall semantics (a prediction vector snapped to the known discrete pressure set) and avoids any new ML libraries or training loops. It run end-to-end within the time limit and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 3.76984) has done: 'Your current score (5.818) is far worse than the target (0.139, lower is better), so we need a real uplift while keeping your “train-lookup median → fallback → snap to nearest valid pressure” core semantics unchanged. The biggest weakness is that exact matching on continuous `u_in` almost never hits, so most rows fall back to a much weaker aggregate; we fix this minimally by quantizing `u_in` into small bins (same transform applied to both train/test) so the primary lookup fires much more often. We also add a third fallback keyed by `(R,C,time_step,u_in_bin)` (dropping `u_out`) to improve coverage when `u_out` differs, still preserving the same lookup-style logic (no new ML model/training loop). Finally, we ensure perfect `id` alignment by writing predictions directly in test row order (the competition expects one row per test `id`).'
- What this solution (achieved 2.2076) has done: 'Your current lookup-based approach is missing the most important structure of the problem: prediction is only scored on inspiratory timesteps (`u_out==0`), and pressure is highly dependent on the cumulative delivered volume (integral of `u_in` over time) rather than just instantaneous `u_in`. To move your score sharply toward the target while keeping the same “groupby-median lookup → fallbacks → snap to nearest valid pressure” core semantics, I add two lightweight engineered keys computed per breath: cumulative `u_in` (and a binned version) and `delta_time`. I then upgrade the primary lookup to use these cumulative features (still median lookup), keep your existing fallbacks, and finally add an inspiratory-only correction: when `u_out==1` in test, force pressure to the minimum training pressure (safe because those rows are not scored). This stays within your non-ML, deterministic lookup framework and should substantially reduce MAE toward the target band.'
- What this solution (achieved 1.75382) has done: 'Your current score (2.2076, lower is better) is still far from the target (0.139), so we need a meaningful uplift while keeping your lookup+snap core approach intact. The biggest minimal win is to make the “integrated volume proxy” physically closer by using a time-weighted cumulative `u_in * delta_time` (instead of plain cumsum of `u_in`), and use it consistently in both train/test group keys. This preserves your exact semantics (groupby-median with fallbacks + nearest-pressure snapping + u_out==1 forced low), but improves the primary matching signal substantially. I also keep your existing fallbacks unchanged so coverage remains stable and the pipeline still writes a valid `submission.csv`.'
- What this solution (achieved 1.75382) has done: 'Your current gap to the target is large (1.75382 vs 0.13905, lower is better), so we need a real uplift without changing the overall “groupby-median lookup → fallbacks → snap to nearest valid pressure (and force u_out==1 low)” core approach. The biggest minimal improvement is to make the cumulative volume proxy closer to the underlying physics by integrating **net inflow** (`u_in * (1-u_out) * delta_time`) rather than using `u_in * delta_time` even during exhalation. I keep all existing lookup tables and fallbacks, but compute additional “net” integral bins and try them early, which increases match quality on inspiratory steps while retaining the same semantics. This should move MAE meaningfully toward your target while staying deterministic and fast, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.73915) has done: 'Your current score (1.75382, lower-is-better) is still far above the target (0.13905), so we need a meaningful but minimal uplift while keeping your lookup→fallback→nearest-pressure snapping core logic intact. The biggest low-risk improvement is to make the “integrated volume proxy” closer to the true state by also incorporating a **leaky/relaxing memory** version of net inflow (an exponential moving accumulation), which often matches pressure dynamics better than a pure cumulative sum. Concretely, we add one additional engineered key (EMA-integral bin) computed per breath for both train/test, build one extra median lookup table on it, and insert it early in the fallback chain; everything else (merging, fallbacks, u_out==1 forcing, snapping) remains the same. This should reduce MAE notably without changing the overall approach or requiring any new packages, and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.73195) has done: 'Your current score (1.739, lower-is-better) is still far above the target (0.139), so we need a meaningful improvement but with minimal, lookup-style changes. The biggest low-risk uplift in this framework is to add a better “state proxy” than the EMA: a per-breath **leaky cumulative integral** (a decayed running sum) of net inflow, which preserves the same groupby-median lookup + fallback + nearest-pressure snapping semantics but matches pressure dynamics more directly. Concretely, we compute a leaky integral feature for train/test, build one additional median lookup table keyed on it (kept early in the fallback chain), and keep everything else (including u_out==1 forcing and snapping) unchanged. This should reduce MAE toward your target without changing the overall approach or requiring new packages.'
- What this solution (achieved 1.73195) has done: 'Your current score (1.73195, lower-is-better) is still far above the target (0.13905), so we should improve the lookup quality while keeping your exact “engineered-state → groupby-median lookups → fallback chain → force u_out==1 low → snap to nearest valid pressure” approach unchanged. The lowest-risk uplift is to add a physically motivated key: **lagged (previous-step) pressure median** during inspiration, which captures strong autoregressive structure without introducing any ML model or changing the post-processing. Concretely, we compute within-breath `u_in` lag and a binned version of the **previous true pressure** in train, build a median table for `pressure[t]` conditioned on `(R,C,time_step,u_out,u_in_bin,u_in_bin_prev,prev_pressure_bin)`, and merge it as an *early* fallback for `u_out==0` rows. Everything else (existing tables, fallbacks, u_out==1 forcing, nearest-pressure snapping, output format) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.72912) has done: 'Your current score (1.73195, lower-is-better) is still far above the target (0.13905), so we should improve the lookup accuracy while keeping your exact lookup→fallback→snap framework unchanged. The biggest minimal issue is that your AR-style table `g_ar` is trained on **true** previous pressure, but at inference you approximate it with `pred_first_prev_bin`, which can be noisy; we can tighten this by iterating once more (two-pass refinement) so the previous-step bin used for the AR lookup is based on a closer first refined prediction. Additionally, we can improve coverage without changing semantics by adding one extra early lookup keyed on the leaky-integral bin without `delta_time` (delta_time mismatches after rounding can block merges), and only use it as a fallback when the stronger `delta_time`-keyed tables miss. These are small, deterministic changes that keep the same overall approach and should move MAE down toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 1.99578) has done: 'Your current score is still far above the target (lower-is-better), so we need a measurable uplift while keeping your lookup→fallback→nearest-pressure snapping approach intact. The highest-impact minimal fix here is to align the engineered “integral state” features with the real per-breath sampling grid: in this competition, `time_step` is effectively an index (0..79) per breath, and rounding + using `delta_time` as a merge key can cause unnecessary mismatches; we add an explicit per-breath `step` index and use it for the strongest lookup tables instead of `time_step/delta_time`. We keep all of your existing tables and AR two-pass refinement, but insert step-index-based versions early in the chain to improve match coverage/precision without changing the overall semantics. Finally, we keep the same inspiratory-only AR usage, `u_out==1` forcing, and nearest-pressure snapping, and still write a valid `submission.csv`.'
- What this solution (achieved 1.99578) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest likely cause is a silent but catastrophic `id` misalignment: you’re building predictions in `df_test_small` order, then sorting only by `id`, but `id` is **not globally unique** in this dataset (it repeats 1..80 per breath), so the submission rows won’t line up with Kaggle’s expected test row order. I fix this minimally by preserving the original test row order via an explicit `row_id` and then writing predictions back in that exact order (or by merging back to the original `sub` order). This keeps your entire lookup/AR/snap logic identical, only changing how predictions are aligned to the required submission format, which should move your score sharply toward the target without changing modeling semantics. I also add a lightweight assert to guarantee the output row count and order match the provided `sample_submission.csv`.'

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
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
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
    a.pressure = a.pressure * 0.65 + b.pressure * 0.35
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)
sub = pd.read_csv(sample_path)

if not {"id", "pressure"}.issubset(sub.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns id and pressure, got {list(sub.columns)}"
    )

df_test = df_test.copy()
df_test["row_id"] = np.arange(len(df_test), dtype=np.int64)

UIN_BIN = 0.5  # keep your fine u_in quantization
CUMVOL_BIN = 0.25  # moderate bin for time-weighted integral proxy
TS_ROUND = 2  # keep your time_step rounding
DT_ROUND = 3  # delta_time is small; keep slightly higher precision

EMA_ALPHA = 0.98
EMA_BIN = 0.25

LEAK_GAMMA = 0.97  # decay factor per time step (close to 1.0 = long memory)
LEAK_BIN = 0.25

PREV_P_BIN = 0.5  # match pressure grid step; bins align with nearest-pressure snapping

df_train_small = df_train[
    ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
].copy()

df_train_small["time_step"] = (
    df_train_small["time_step"].round(TS_ROUND).astype(np.float32)
)
df_train_small["u_out"] = df_train_small["u_out"].astype(np.int8)
df_train_small["pressure"] = df_train_small["pressure"].astype(np.float32)

df_train_small.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

df_train_small["step"] = df_train_small.groupby("breath_id").cumcount().astype(np.int16)

df_train_small["delta_time"] = (
    df_train_small.groupby("breath_id")["time_step"]
    .diff()
    .fillna(0.0)
    .round(DT_ROUND)
    .astype(np.float32)
)

df_train_small["u_in_bin"] = (df_train_small["u_in"] / UIN_BIN).round().astype(np.int16)

df_train_small["u_in_int"] = (
    (df_train_small["u_in"].astype(np.float32) * df_train_small["delta_time"])
    .groupby(df_train_small["breath_id"])
    .cumsum()
    .astype(np.float32)
)
df_train_small["u_in_int_bin"] = (
    (df_train_small["u_in_int"] / CUMVOL_BIN).round().astype(np.int16)
)

df_train_small["u_in_int_net"] = (
    (
        df_train_small["u_in"].astype(np.float32)
        * (1.0 - df_train_small["u_out"].astype(np.float32))
        * df_train_small["delta_time"]
    )
    .groupby(df_train_small["breath_id"])
    .cumsum()
    .astype(np.float32)
)
df_train_small["u_in_int_net_bin"] = (
    (df_train_small["u_in_int_net"] / CUMVOL_BIN).round().astype(np.int16)
)


def _ema_net_inflow(group: pd.DataFrame) -> pd.Series:
    x = (
        group["u_in"].to_numpy(dtype=np.float32)
        * (1.0 - group["u_out"].to_numpy(dtype=np.float32))
        * group["delta_time"].to_numpy(dtype=np.float32)
    )
    y = np.empty_like(x, dtype=np.float32)
    prev = np.float32(0.0)
    a = np.float32(EMA_ALPHA)
    one_minus_a = np.float32(1.0) - a
    for i in range(x.shape[0]):
        prev = a * prev + one_minus_a * x[i]
        y[i] = prev
    return pd.Series(y, index=group.index, dtype=np.float32)


def _leaky_int_net_inflow(group: pd.DataFrame) -> pd.Series:
    x = (
        group["u_in"].to_numpy(dtype=np.float32)
        * (1.0 - group["u_out"].to_numpy(dtype=np.float32))
        * group["delta_time"].to_numpy(dtype=np.float32)
    )
    y = np.empty_like(x, dtype=np.float32)
    prev = np.float32(0.0)
    g = np.float32(LEAK_GAMMA)
    for i in range(x.shape[0]):
        prev = g * prev + x[i]
        y[i] = prev
    return pd.Series(y, index=group.index, dtype=np.float32)


df_train_small["u_in_ema_net"] = (
    df_train_small.groupby("breath_id", sort=False, group_keys=False)
    .apply(_ema_net_inflow)
    .astype(np.float32)
)
df_train_small["u_in_ema_net_bin"] = (
    (df_train_small["u_in_ema_net"] / EMA_BIN).round().astype(np.int16)
)

df_train_small["u_in_leak_net"] = (
    df_train_small.groupby("breath_id", sort=False, group_keys=False)
    .apply(_leaky_int_net_inflow)
    .astype(np.float32)
)
df_train_small["u_in_leak_net_bin"] = (
    (df_train_small["u_in_leak_net"] / LEAK_BIN).round().astype(np.int16)
)

df_train_small["u_in_bin_prev"] = (
    df_train_small.groupby("breath_id")["u_in_bin"].shift(1).fillna(-1).astype(np.int16)
)
df_train_small["pressure_prev"] = (
    df_train_small.groupby("breath_id")["pressure"]
    .shift(1)
    .fillna(np.nan)
    .astype(np.float32)
)
df_train_small["pressure_prev_bin"] = (
    (df_train_small["pressure_prev"] / PREV_P_BIN).round().astype("float32")
)
df_train_small.loc[df_train_small["pressure_prev"].isna(), "pressure_prev_bin"] = -999.0
df_train_small["pressure_prev_bin"] = df_train_small["pressure_prev_bin"].astype(
    np.int16
)

df_test_small = df_test[
    ["row_id", "id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
].copy()
df_test_small["time_step"] = (
    df_test_small["time_step"].round(TS_ROUND).astype(np.float32)
)
df_test_small["u_out"] = df_test_small["u_out"].astype(np.int8)

df_test_small.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

df_test_small["step"] = df_test_small.groupby("breath_id").cumcount().astype(np.int16)

df_test_small["delta_time"] = (
    df_test_small.groupby("breath_id")["time_step"]
    .diff()
    .fillna(0.0)
    .round(DT_ROUND)
    .astype(np.float32)
)
df_test_small["u_in_bin"] = (df_test_small["u_in"] / UIN_BIN).round().astype(np.int16)

df_test_small["u_in_int"] = (
    (df_test_small["u_in"].astype(np.float32) * df_test_small["delta_time"])
    .groupby(df_test_small["breath_id"])
    .cumsum()
    .astype(np.float32)
)
df_test_small["u_in_int_bin"] = (
    (df_test_small["u_in_int"] / CUMVOL_BIN).round().astype(np.int16)
)

df_test_small["u_in_int_net"] = (
    (
        df_test_small["u_in"].astype(np.float32)
        * (1.0 - df_test_small["u_out"].astype(np.float32))
        * df_test_small["delta_time"]
    )
    .groupby(df_test_small["breath_id"])
    .cumsum()
    .astype(np.float32)
)
df_test_small["u_in_int_net_bin"] = (
    (df_test_small["u_in_int_net"] / CUMVOL_BIN).round().astype(np.int16)
)

df_test_small["u_in_ema_net"] = (
    df_test_small.groupby("breath_id", sort=False, group_keys=False)
    .apply(_ema_net_inflow)
    .astype(np.float32)
)
df_test_small["u_in_ema_net_bin"] = (
    (df_test_small["u_in_ema_net"] / EMA_BIN).round().astype(np.int16)
)

df_test_small["u_in_leak_net"] = (
    df_test_small.groupby("breath_id", sort=False, group_keys=False)
    .apply(_leaky_int_net_inflow)
    .astype(np.float32)
)
df_test_small["u_in_leak_net_bin"] = (
    (df_test_small["u_in_leak_net"] / LEAK_BIN).round().astype(np.int16)
)

df_test_small["u_in_bin_prev"] = (
    df_test_small.groupby("breath_id")["u_in_bin"].shift(1).fillna(-1).astype(np.int16)
)

g0 = (
    df_train_small.groupby(
        ["R", "C", "time_step", "delta_time", "u_in_int_bin", "u_out"], sort=False
    )["pressure"]
    .median()
    .rename("p0")
    .reset_index()
)
g0n = (
    df_train_small.groupby(
        ["R", "C", "time_step", "delta_time", "u_in_int_net_bin", "u_out"], sort=False
    )["pressure"]
    .median()
    .rename("p0n")
    .reset_index()
)
g0e = (
    df_train_small.groupby(
        ["R", "C", "time_step", "delta_time", "u_in_ema_net_bin", "u_out"], sort=False
    )["pressure"]
    .median()
    .rename("p0e")
    .reset_index()
)
g0l = (
    df_train_small.groupby(
        ["R", "C", "time_step", "delta_time", "u_in_leak_net_bin", "u_out"], sort=False
    )["pressure"]
    .median()
    .rename("p0l")
    .reset_index()
)
g0l_ndt = (
    df_train_small.groupby(
        ["R", "C", "time_step", "u_in_leak_net_bin", "u_out"], sort=False
    )["pressure"]
    .median()
    .rename("p0l_ndt")
    .reset_index()
)

g0s = (
    df_train_small.groupby(["R", "C", "step", "u_in_int_bin", "u_out"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p0s")
    .reset_index()
)
g0ns = (
    df_train_small.groupby(["R", "C", "step", "u_in_int_net_bin", "u_out"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p0ns")
    .reset_index()
)
g0ls = (
    df_train_small.groupby(
        ["R", "C", "step", "u_in_leak_net_bin", "u_out"], sort=False
    )["pressure"]
    .median()
    .rename("p0ls")
    .reset_index()
)
g0es = (
    df_train_small.groupby(["R", "C", "step", "u_in_ema_net_bin", "u_out"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p0es")
    .reset_index()
)

g1 = (
    df_train_small.groupby(["R", "C", "time_step", "u_in_bin", "u_out"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p1")
    .reset_index()
)
g2 = (
    df_train_small.groupby(["R", "C", "time_step", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p2")
    .reset_index()
)
g3 = (
    df_train_small.groupby(["R", "C", "time_step", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("p3")
    .reset_index()
)
g4 = (
    df_train_small.groupby(["R", "C", "u_in_int_bin", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p4")
    .reset_index()
)
g4n = (
    df_train_small.groupby(["R", "C", "u_in_int_net_bin", "u_out"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p4n")
    .reset_index()
)

g_ar = (
    df_train_small.groupby(
        [
            "R",
            "C",
            "time_step",
            "u_out",
            "u_in_bin",
            "u_in_bin_prev",
            "pressure_prev_bin",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("par")
    .reset_index()
)

pred_df = (
    df_test_small.merge(g0s, on=["R", "C", "step", "u_in_int_bin", "u_out"], how="left")
    .merge(g0ns, on=["R", "C", "step", "u_in_int_net_bin", "u_out"], how="left")
    .merge(g0ls, on=["R", "C", "step", "u_in_leak_net_bin", "u_out"], how="left")
    .merge(g0es, on=["R", "C", "step", "u_in_ema_net_bin", "u_out"], how="left")
)

pred_df = pred_df.merge(
    g0, on=["R", "C", "time_step", "delta_time", "u_in_int_bin", "u_out"], how="left"
)
pred_df = pred_df.merge(
    g0n,
    on=["R", "C", "time_step", "delta_time", "u_in_int_net_bin", "u_out"],
    how="left",
)
pred_df = pred_df.merge(
    g0e,
    on=["R", "C", "time_step", "delta_time", "u_in_ema_net_bin", "u_out"],
    how="left",
)
pred_df = pred_df.merge(
    g0l,
    on=["R", "C", "time_step", "delta_time", "u_in_leak_net_bin", "u_out"],
    how="left",
)
pred_df = pred_df.merge(
    g0l_ndt, on=["R", "C", "time_step", "u_in_leak_net_bin", "u_out"], how="left"
)
pred_df = pred_df.merge(g1, on=["R", "C", "time_step", "u_in_bin", "u_out"], how="left")
pred_df = pred_df.merge(g4, on=["R", "C", "u_in_int_bin", "u_out"], how="left")
pred_df = pred_df.merge(g4n, on=["R", "C", "u_in_int_net_bin", "u_out"], how="left")
pred_df = pred_df.merge(g2, on=["R", "C", "time_step", "u_out"], how="left")
pred_df = pred_df.merge(g3, on=["R", "C", "time_step", "u_in_bin"], how="left")

global_med = float(df_train_small["pressure"].median())
min_pressure = float(sorted_pressures[0])

pred_first = pred_df["p0s"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0ns"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0ls"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0es"].to_numpy(dtype=np.float32)

mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0n"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0l"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0l_ndt"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p0e"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p1"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p4"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p4n"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p3"].to_numpy(dtype=np.float32)
mask = np.isnan(pred_first)
pred_first[mask] = pred_df.loc[mask, "p2"].to_numpy(dtype=np.float32)
pred_first = np.where(np.isnan(pred_first), global_med, pred_first).astype(np.float32)

pred_df["pred_first"] = pred_first


def _finalize_with_ar(prev_bin_col: str) -> np.ndarray:
    tmp = pred_df[
        [
            "row_id",
            "id",
            "breath_id",
            "R",
            "C",
            "step",
            "time_step",
            "u_out",
            "u_in_bin",
            "u_in_bin_prev",
            "p0s",
            "p0ns",
            "p0ls",
            "p0es",
            "p0",
            "p0n",
            "p0l",
            "p0l_ndt",
            "p0e",
            "p1",
            "p4",
            "p4n",
            "p3",
            "p2",
            prev_bin_col,
        ]
    ].copy()

    tmp = tmp.merge(
        g_ar,
        left_on=[
            "R",
            "C",
            "time_step",
            "u_out",
            "u_in_bin",
            "u_in_bin_prev",
            prev_bin_col,
        ],
        right_on=[
            "R",
            "C",
            "time_step",
            "u_out",
            "u_in_bin",
            "u_in_bin_prev",
            "pressure_prev_bin",
        ],
        how="left",
    )

    pred = tmp["p0s"].to_numpy(dtype=np.float32)

    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0ns"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0ls"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0es"].to_numpy(dtype=np.float32)

    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0n"].to_numpy(dtype=np.float32)

    mask = np.isnan(pred)
    mask_insp = mask & (tmp["u_out"].to_numpy(dtype=np.int8) == 0)
    pred[mask_insp] = tmp.loc[mask_insp, "par"].to_numpy(dtype=np.float32)

    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0l"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0l_ndt"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p0e"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p1"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p4"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p4n"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p3"].to_numpy(dtype=np.float32)
    mask = np.isnan(pred)
    pred[mask] = tmp.loc[mask, "p2"].to_numpy(dtype=np.float32)

    pred = np.where(np.isnan(pred), global_med, pred).astype(np.float32)
    return pred.astype(float)


pred_df["pred_first_prev"] = (
    pred_df.groupby("breath_id")["pred_first"]
    .shift(1)
    .fillna(np.nan)
    .astype(np.float32)
)
pred_df["pred_first_prev_bin"] = (
    (pred_df["pred_first_prev"] / PREV_P_BIN).round().astype("float32")
)
pred_df.loc[pred_df["pred_first_prev"].isna(), "pred_first_prev_bin"] = -999.0
pred_df["pred_first_prev_bin"] = pred_df["pred_first_prev_bin"].astype(np.int16)

pred_pass1 = _finalize_with_ar("pred_first_prev_bin").astype(np.float32)

pred_df["pred_pass1"] = pred_pass1
pred_df["pred_pass1_prev"] = (
    pred_df.groupby("breath_id")["pred_pass1"]
    .shift(1)
    .fillna(np.nan)
    .astype(np.float32)
)
pred_df["pred_pass1_prev_bin"] = (
    (pred_df["pred_pass1_prev"] / PREV_P_BIN).round().astype("float32")
)
pred_df.loc[pred_df["pred_pass1_prev"].isna(), "pred_pass1_prev_bin"] = -999.0
pred_df["pred_pass1_prev_bin"] = pred_df["pred_pass1_prev_bin"].astype(np.int16)

pred = _finalize_with_ar("pred_pass1_prev_bin").astype(float)

u_out_arr = pred_df["u_out"].to_numpy()
pred = np.where(u_out_arr == 1, min_pressure, pred)

pred = np.array([find_nearest(x) for x in pred], dtype=float)

out = pred_df[["row_id", "id"]].copy()
out["pressure"] = pred
out.sort_values("row_id", inplace=True, kind="mergesort")
out = out[["id", "pressure"]].reset_index(drop=True)

assert len(out) == len(sub), (len(out), len(sub))

out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("Pressure min/max:", float(out["pressure"].min()), float(out["pressure"].max()))
