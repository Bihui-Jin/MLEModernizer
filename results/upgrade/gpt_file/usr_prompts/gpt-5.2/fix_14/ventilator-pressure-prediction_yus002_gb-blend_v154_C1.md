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

0.136436628524095

# 6. Current score

2.00606

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.24999) has done: 'I remove the dependency on missing external Kaggle datasets (the `gb-data-blending-recover` paths) that causes the `FileNotFoundError`, and instead generate a valid submission directly from the provided `train.csv/test.csv`. To keep core intent minimal while still improving score vs. a zero baseline, I use a simple, leakage-free median-pressure lookup by `(R, C, time_step)` computed on inspiratory rows (`u_out == 0`), with sensible fallbacks when a key is missing. I keep your pressure “snapping” logic (`find_nearest`) so predictions match the known discrete pressure levels. The script always write `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 6.77765) has done: 'Your current 7.24999 MAE is far above the 0.1364 target (lower is better), so we should make a small but meaningful improvement without changing the overall “lookup + snap-to-known-pressures” core logic. The biggest miss is that the metric ignores expiratory steps (`u_out==1`), but your lookup doesn’t condition on `u_out` and also doesn’t use the strong within-breath autoregressive signal (pressure is smooth and highly dependent on previous steps). I keep the same median-lookup approach, but (1) build separate medians for `u_out==0/1` and use `u_in`-binned medians for inspiratory steps, and (2) add a minimal sequential smoothing step per `breath_id` (using previous predicted pressure) to better match the time-series nature while preserving evaluation semantics and still snapping to discrete pressure levels. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 3.94863) has done: 'We keep your current “median lookup + sequential smoothing + snap-to-known-pressures” core logic, but fix two high-impact issues that are likely causing the very large MAE: (1) `time_step` is a float and exact merges can miss due to float representation, so we merge on an integer `time_step_idx` instead; (2) your final `sub` merge is redundant and can silently reorder/duplicate in edge cases, so we directly assign predictions to `sample_submission` after sorting by `id` to guarantee perfect alignment. These are minimal, semantics-preserving changes that should substantially reduce error without changing the modeling approach. The output remains a valid `submission.csv` with `id,pressure` and the same discrete snapping.'
- What this solution (achieved 2.00717) has done: 'Your current MAE (3.94863, lower-is-better) is still far from the target (0.1364), so we should improve the lookup quality without changing the overall “median lookup + sequential smoothing + snap-to-known-pressures” approach. The biggest remaining gap is that the lookup ignores the very strong autoregressive relationship between pressure and cumulative inspired volume; we can add a minimal, leakage-free feature `u_in_cum` computed per breath and use it in the inspiratory median tables. To keep changes small and stable, we keep all existing fallbacks and smoothing, but replace the inspiratory primary lookup from `(R,C,time_step_idx,u_in_bin)` to `(R,C,time_step_idx,u_in_cum_bin)` with a coarse binning. This preserves the same semantics (still a median table lookup snapped to discrete pressures) while typically moving the score substantially toward the target band.'
- What this solution (achieved 2.21211) has done: 'We keep your exact “median lookup + fallbacks + per-breath smoothing + snap-to-discrete-pressures” approach, but make two small changes that usually move MAE substantially closer to the target without changing the modeling paradigm. First, we ensure `time_step_idx` is computed consistently with the dataset’s fixed 0.03s grid by using an integer index via division (more stable than `rint(time_step*100)`, which can mis-key merges). Second, we apply the sequential smoothing only during inspiratory steps *and* only until the first `u_out==1` within each breath (the scored region), avoiding unintended propagation of expiratory dynamics back into later steps. These are minimal, semantics-preserving adjustments aimed at improving lookup hit-rate and reducing error in the scored inspiratory phase, while still writing a valid `submission.csv`.'
- What this solution (achieved 2.19617) has done: 'Your current MAE (2.21211, lower-is-better) is still far above the target (0.13644), so we should improve the lookup hit-rate without changing the overall “median lookup + fallbacks + per-breath smoothing + snap-to-discrete-pressures” core logic. The most likely remaining high-impact issue is that `time_step_idx` creation via floating rounding can still mis-key merges; switching to a stable integer grid index computed by `round(time_step*100)/3` (exactly matching 0.03s steps) reduces missing joins and improves medians usage. Second, your inspiratory primary lookup uses cumulative `u_in` bins but the binning is quite coarse; tightening the bin width modestly (while keeping the same feature and lookup tables) usually improves MAE materially without changing semantics. Finally, we keep the same smoothing logic but make it numerically consistent by applying it only to the scored inspiratory region as you already do, leaving expiratory predictions untouched.'
- What this solution (achieved 2.5262) has done: 'Your current MAE (2.19617, lower-is-better) is still far above the target (0.13644), so we should make small, lookup-quality improvements without changing the overall “median lookup + fallbacks + per-breath smoothing + snap-to-discrete-pressures” approach. The biggest safe gain is to make the inspiratory lookup more specific by adding `u_in` itself (binned) alongside `u_in_cum_bin`, and to make the sequential smoothing respect the known monotonic increase of pressure during inspiration (a minimal constraint that reduces obvious lookup noise). I also make the `time_step_idx` mapping exactly match the 0.03s grid using a stable division/rounding formula to reduce missed-join NaNs. These are minimal changes that keep your core logic intact, but usually reduce MAE materially toward the target.'
- What this solution (achieved 2.91944) has done: 'Your current MAE (2.5262, lower-is-better) is still far above the target (0.1364), so we should make the smallest changes that increase median-lookup hit-rate and reduce obvious inspiratory noise without changing the overall “median lookup + fallbacks + per-breath smoothing + snap-to-known-pressures” logic. The biggest likely issue is key-mismatch due to binning/rounding: switching to more stable *floor* bins (instead of round) for `u_in` and `u_in_cum`, and using a more faithful cumulative feature (`u_in * Δt`) should make the lookup tables match test behavior better. I also keep your monotonic inspiratory smoothing, but make the first-step initialization per breath robust by starting `prev` from the already-snapped first prediction (reduces early drift). These changes preserve your approach and should move MAE meaningfully toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 2.4349) has done: 'We keep your same “median lookup + fallbacks + per-breath inspiratory smoothing + snap-to-known-pressures” core logic, but fix two likely causes of the MAE regression. First, we make `time_step_idx` match the exact 80-step structure by computing it per-breath from the row order (0..79) rather than float rounding/division, which can silently mis-key merges and reduce lookup hit-rate. Second, we remove the unintended monotonic “never decrease” constraint during inspiration (it can force systematic overestimation and worsen MAE) while keeping the same exponential smoothing and only applying it in the scored inspiratory region. These are minimal, semantics-preserving changes that should move the score back down toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 2.12675) has done: 'Your current MAE (2.4349, lower-is-better) is still far above the target (0.1364), so we should make small, safe changes that improve lookup hit-rate and reduce systematic error without changing the overall “median lookup + fallbacks + inspiratory smoothing + snap-to-discrete-pressures” logic. The biggest low-risk gain is to compute the cumulative inhaled volume using the *actual per-row* `Δt` within each breath (instead of a fixed 0.03), because the last step spacing can differ and this directly affects your `u_in_cum_bin` keys. Next, we slightly tighten `u_in_cum` bin width (from 6.0 to 4.0) to make medians more specific while keeping the same tables/fallbacks. Finally, we apply snapping (`find_nearest`) immediately after smoothing each inspiratory step so the “prev” state stays on valid discrete pressure levels, which stabilizes the sequential component and typically reduces MAE.'
- What this solution (achieved 1.90599) has done: 'Your current MAE (2.12675, lower-is-better) is still far above the target (0.13644), so we should make a small, low-risk improvement while keeping the same “median lookup + fallbacks + per-breath inspiratory smoothing + snap-to-discrete-pressures” core logic. The biggest safe gain is to stop treating `u_in` bins as globally comparable across breaths by normalizing `u_in` within each breath (z-score) and adding a coarse `u_in_z_bin` to the inspiratory median tables (with fallbacks preserved), which typically reduces systematic bias across different breath patterns. This is still just a median-lookup table keyed by binned features (no model/loop/loss change), and we keep the existing smoothing/snap logic and submission alignment checks intact. The rest of the pipeline and file paths are unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 2.00609) has done: 'Your current score (1.90599 MAE; lower is better) is still far above the target (0.13644), so we should make a small, safe improvement that keeps the same “median lookup + fallbacks + per-breath inspiratory smoothing + snap-to-discrete-pressures” core logic. The biggest low-risk issue is that the per-breath `u_in` z-score bins can be unstable (especially when within-breath std is small), which can hurt the primary lookup hit-rate; we stabilize it by clipping extreme z-scores and using a slightly coarser z-bin width while keeping the same keyed-median hierarchy and fallbacks. Second, we make the inspiratory smoothing slightly more conservative (increase reliance on the previous step) to reduce step-to-step lookup noise in the scored inspiratory phase, without changing the overall approach. All file paths stay the same and the script still writes an aligned `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.00606) has done: 'Your MAE (2.00609, lower-is-better) is still far above the target (0.13644), so we should make a small improvement that keeps your exact “median lookup + fallbacks + inspiratory-only smoothing + snap-to-discrete-pressures” pipeline intact. The biggest low-risk win is to add a breath-level context key by binning `u_in` *within each (R,C,time_step_idx)* group (a “relative u_in” bin), which helps separate different breath patterns without changing the modeling approach. We keep all existing tables and fallbacks, but insert this new key as the first-choice inspiratory lookup and fall back to your current hierarchy when it’s missing. Everything else (time_step_idx by cumcount, dt-based cumulative, smoothing, snapping, strict id alignment, and writing `submission.csv`) remains the same.'

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
    loop_time = 155
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
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


def add_time_step_idx(df):
    return df.groupby("breath_id", sort=False).cumcount().astype(np.int16)


df_train = df_train.copy()
df_test = df_test.copy()
df_train["time_step_idx"] = add_time_step_idx(df_train)
df_test["time_step_idx"] = add_time_step_idx(df_test)


def add_dt_and_u_in_cum(df):
    dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    dt = dt.clip(lower=0.0)  # safety against any tiny negative float noise
    u_in_cum = (df["u_in"] * dt).groupby(df["breath_id"], sort=False).cumsum()
    return dt, u_in_cum


df_train["dt"], df_train["u_in_cum"] = add_dt_and_u_in_cum(df_train)
df_test["dt"], df_test["u_in_cum"] = add_dt_and_u_in_cum(df_test)


def add_u_in_z(df):
    g = df.groupby("breath_id", sort=False)["u_in"]
    mu = g.transform("mean")
    sd = g.transform("std").fillna(0.0)
    z = (df["u_in"] - mu) / (sd.replace(0.0, 1.0))
    z = z.clip(-5.0, 5.0)
    return z.astype(np.float32)


df_train["u_in_z"] = add_u_in_z(df_train)
df_test["u_in_z"] = add_u_in_z(df_test)

cum_bin_width = 4.0
df_train["u_in_cum_bin"] = np.floor(df_train["u_in_cum"] / cum_bin_width).astype(
    np.int16
)
df_test["u_in_cum_bin"] = np.floor(df_test["u_in_cum"] / cum_bin_width).astype(np.int16)

bin_width = 2.0
df_train["u_in_bin"] = np.floor(df_train["u_in"] / bin_width).astype(np.int16)
df_test["u_in_bin"] = np.floor(df_test["u_in"] / bin_width).astype(np.int16)

u_in_z_bin_width = 0.35
df_train["u_in_z_bin"] = np.floor(df_train["u_in_z"] / u_in_z_bin_width).astype(
    np.int16
)
df_test["u_in_z_bin"] = np.floor(df_test["u_in_z"] / u_in_z_bin_width).astype(np.int16)


def add_u_in_rc_t_rel_bin(df, rel_bin_width=0.25):
    g = df.groupby(["R", "C", "time_step_idx"], sort=False)["u_in"]
    mn = g.transform("min")
    mx = g.transform("max")
    denom = (mx - mn).replace(0.0, 1.0)
    rel = ((df["u_in"] - mn) / denom).clip(0.0, 1.0)
    rel_bin = np.floor(rel / rel_bin_width).astype(np.int16)
    return rel_bin


df_train["u_in_rc_t_rel_bin"] = add_u_in_rc_t_rel_bin(df_train, rel_bin_width=0.25)
df_test["u_in_rc_t_rel_bin"] = add_u_in_rc_t_rel_bin(df_test, rel_bin_width=0.25)

rc_t_uo_median = (
    df_train.groupby(["u_out", "R", "C", "time_step_idx"], sort=False)["pressure"]
    .median()
    .rename("p_rc_t_uo")
    .reset_index()
)
rc_uo_median = (
    df_train.groupby(["u_out", "R", "C"], sort=False)["pressure"]
    .median()
    .rename("p_rc_uo")
    .reset_index()
)
t_uo_median = (
    df_train.groupby(["u_out", "time_step_idx"], sort=False)["pressure"]
    .median()
    .rename("p_t_uo")
    .reset_index()
)
global_uo_median = (
    df_train.groupby(["u_out"], sort=False)["pressure"]
    .median()
    .rename("p_global_uo")
    .reset_index()
)

train_insp = df_train[df_train["u_out"] == 0].copy()

rc_t_rel_uic_ui_uz_median = (
    train_insp.groupby(
        [
            "R",
            "C",
            "time_step_idx",
            "u_in_rc_t_rel_bin",
            "u_in_cum_bin",
            "u_in_bin",
            "u_in_z_bin",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_rc_t_rel_uic_ui_uz")
    .reset_index()
)

rc_t_uic_ui_uz_median = (
    train_insp.groupby(
        ["R", "C", "time_step_idx", "u_in_cum_bin", "u_in_bin", "u_in_z_bin"],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_rc_t_uic_ui_uz")
    .reset_index()
)

rc_t_uic_ui_median = (
    train_insp.groupby(
        ["R", "C", "time_step_idx", "u_in_cum_bin", "u_in_bin"], sort=False
    )["pressure"]
    .median()
    .rename("p_rc_t_uic_ui")
    .reset_index()
)
rc_uic_ui_median = (
    train_insp.groupby(["R", "C", "u_in_cum_bin", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("p_rc_uic_ui")
    .reset_index()
)

rc_t_uic_median = (
    train_insp.groupby(["R", "C", "time_step_idx", "u_in_cum_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_rc_t_uic")
    .reset_index()
)
rc_uic_median = (
    train_insp.groupby(["R", "C", "u_in_cum_bin"], sort=False)["pressure"]
    .median()
    .rename("p_rc_uic")
    .reset_index()
)

rc_t_ui_median = (
    train_insp.groupby(["R", "C", "time_step_idx", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("p_rc_t_ui")
    .reset_index()
)
rc_ui_median = (
    train_insp.groupby(["R", "C", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("p_rc_ui")
    .reset_index()
)

global_median = float(train_insp["pressure"].median())

test_pred = (
    df_test[
        [
            "id",
            "breath_id",
            "R",
            "C",
            "time_step",
            "time_step_idx",
            "u_out",
            "u_in_bin",
            "u_in_cum_bin",
            "u_in_z_bin",
            "u_in_rc_t_rel_bin",
        ]
    ]
    .merge(rc_t_uo_median, on=["u_out", "R", "C", "time_step_idx"], how="left")
    .merge(rc_uo_median, on=["u_out", "R", "C"], how="left")
    .merge(t_uo_median, on=["u_out", "time_step_idx"], how="left")
    .merge(global_uo_median, on=["u_out"], how="left")
)

test_pred = (
    test_pred.merge(
        rc_t_rel_uic_ui_uz_median,
        on=[
            "R",
            "C",
            "time_step_idx",
            "u_in_rc_t_rel_bin",
            "u_in_cum_bin",
            "u_in_bin",
            "u_in_z_bin",
        ],
        how="left",
    )
    .merge(
        rc_t_uic_ui_uz_median,
        on=["R", "C", "time_step_idx", "u_in_cum_bin", "u_in_bin", "u_in_z_bin"],
        how="left",
    )
    .merge(
        rc_t_uic_ui_median,
        on=["R", "C", "time_step_idx", "u_in_cum_bin", "u_in_bin"],
        how="left",
    )
    .merge(rc_uic_ui_median, on=["R", "C", "u_in_cum_bin", "u_in_bin"], how="left")
    .merge(rc_t_uic_median, on=["R", "C", "time_step_idx", "u_in_cum_bin"], how="left")
    .merge(rc_uic_median, on=["R", "C", "u_in_cum_bin"], how="left")
    .merge(rc_t_ui_median, on=["R", "C", "time_step_idx", "u_in_bin"], how="left")
    .merge(rc_ui_median, on=["R", "C", "u_in_bin"], how="left")
)

p = test_pred["p_rc_t_uo"].copy()
insp_mask = test_pred["u_out"].to_numpy() == 0

p_insp_new = test_pred.loc[insp_mask, "p_rc_t_rel_uic_ui_uz"]
p.loc[insp_mask] = p_insp_new

p_insp0 = test_pred.loc[insp_mask, "p_rc_t_uic_ui_uz"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp0)

p_insp = test_pred.loc[insp_mask, "p_rc_t_uic_ui"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp)

p = p.fillna(test_pred["p_rc_uo"])

p_insp_c2 = test_pred.loc[insp_mask, "p_rc_uic_ui"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp_c2)

p_insp_c3 = test_pred.loc[insp_mask, "p_rc_t_uic"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp_c3)

p_insp_c4 = test_pred.loc[insp_mask, "p_rc_uic"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp_c4)

p_insp_t2 = test_pred.loc[insp_mask, "p_rc_t_ui"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp_t2)

p_insp2 = test_pred.loc[insp_mask, "p_rc_ui"]
p.loc[insp_mask] = p.loc[insp_mask].fillna(p_insp2)

p = p.fillna(test_pred["p_t_uo"])
p = p.fillna(test_pred["p_global_uo"])
p = p.fillna(global_median)
p = p.astype(float).to_numpy()

alpha = 0.55  # weight on current lookup; (1-alpha) on previous prediction

order = np.lexsort(
    (test_pred["time_step_idx"].to_numpy(), test_pred["breath_id"].to_numpy())
)
inv_order = np.empty_like(order)
inv_order[order] = np.arange(order.size)

breath_sorted = test_pred["breath_id"].to_numpy()[order]
uout_sorted = test_pred["u_out"].to_numpy()[order]
p_sorted = p[order].copy()

prev = None
prev_breath = None
in_scored_insp = True
for i in range(p_sorted.shape[0]):
    b = breath_sorted[i]
    if prev_breath != b:
        prev_breath = b
        p_sorted[i] = float(find_nearest(p_sorted[i]))
        prev = p_sorted[i]
        in_scored_insp = True
    else:
        if in_scored_insp:
            if uout_sorted[i] == 0:
                sm = alpha * p_sorted[i] + (1.0 - alpha) * prev
                p_sorted[i] = float(find_nearest(sm))
            else:
                in_scored_insp = False
        prev = p_sorted[i]

p = p_sorted[inv_order]
p = pd.Series(p).apply(find_nearest).to_numpy()

sub = sub.sort_values("id").reset_index(drop=True)
pred_df = test_pred[["id"]].copy()
pred_df["pressure"] = p
pred_df = pred_df.sort_values("id").reset_index(drop=True)

if len(sub) != len(pred_df) or not np.array_equal(
    sub["id"].to_numpy(), pred_df["id"].to_numpy()
):
    raise ValueError(
        "ID alignment check failed; refusing to write a potentially invalid submission."
    )

sub["pressure"] = pred_df["pressure"].astype(float).to_numpy()
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "pressure min/max:", float(np.min(sub["pressure"])), float(np.max(sub["pressure"]))
)
