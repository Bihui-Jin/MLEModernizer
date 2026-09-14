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

0.1789658040286963

# 6. Current score

2.74878

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.10591) has done: 'I remove the dependency on missing external “gb-blending” input files (which is causing the FileNotFoundError) and instead generate a valid submission directly from the provided competition `train.csv`/`test.csv`. To keep changes minimal and score-improving vs “no submission”, I implement a simple, fast baseline that predicts mean inspiratory pressure per `(R, C, time_step)` learned from train (fallbacks to coarser group means if an exact key is missing). I keep your pressure “snapping” (`find_nearest`) so predictions stay on the known discrete pressure grid used in this competition. Finally, the script always write a valid `submission.csv` with exactly `id,pressure` columns.'
- What this solution (achieved 4.08578) has done: 'Your current baseline ignores the strongest available signal (`u_in`) during inspiration, which is why the MAE is very high; adding `u_in` to the same group-mean lookup keeps the approach identical (a fast mean-encoding baseline) but should move the score substantially toward the target. I keep your inspiratory-only training (`u_out==0`), your time_step rounding, and your pressure “snapping” (`find_nearest`) unchanged in semantics. To stay robust, I add a minimal hierarchical fallback chain: mean by `(R,C,time_step,u_in)` → `(R,C,time_step)` → `(R,C)` → global mean. The script still run end-to-end and always write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.21297) has done: 'Your current mean-encoding baseline is sound but it’s still underusing the strongest structure in this competition: the discrete pressure grid and the fact that the same `(R,C)` setting produces repeatable pressure trajectories over the inspiratory phase. To move the MAE down toward the target with minimal change, I (1) build the mean table using the *full* training set while scoring-aligned (still training only on inspiratory rows `u_out==0`), (2) increase the `u_in` key resolution slightly (round to 2 decimals instead of 1) to reduce averaging error without changing the approach, and (3) add one extra fallback level `(R,C,uin_key)` between `(R,C,ts,uin)` and `(R,C,ts)` to improve robustness when a `(ts, u_in)` pair is unseen. All core semantics remain identical: group-mean lookup + hierarchical fallbacks + snapping predictions to the known pressure grid, and it still writes a valid `submission.csv`.'
- What this solution (achieved 4.89374) has done: 'Your current score (5.21297, lower-is-better) is far from the target (0.17897), so we need a real accuracy lift while keeping your core “group-mean lookup + hierarchical fallbacks + snap-to-pressure-grid” logic intact. The biggest missing scoring alignment is that the metric ignores expiratory phase, but your test predictions currently include arbitrary values for `u_out==1`, which can hurt because those rows still exist in submission even if they’re not scored—however, model consistency improves if we set expiratory predictions to a stable baseline that matches train behavior. Also, `time_step` and `u_in` in this dataset are on a fixed 80-step grid per breath; using the within-breath step index as the primary time key is typically more stable than rounding floats, so we add `step` (0–79) and keep your existing keys as fallbacks. Finally, we add one additional minimal fallback that uses `(R,C,step)` before `(R,C,ts_key)` to improve hit-rate without changing the approach.'
- What this solution (achieved 4.95581) has done: 'Your current baseline is still far from the target (MAE 4.89 vs 0.179, lower-is-better), so we need a meaningful accuracy gain while keeping the exact same “group-mean lookup + hierarchical fallbacks + snap-to-pressure-grid” core logic. The biggest safe lift with minimal change is to stop using rounded float `time_step` as a primary key (it causes unnecessary key misses) and instead rely primarily on the within-breath integer `step` (0–79), which is the true stable time index in this dataset. We also improve the expiratory-phase fill value: rather than a global mean, use an `(R,C)`-conditioned expiratory mean (still a group-mean baseline) which better matches train behavior without changing semantics. Finally, we keep your existing tables as backstops, just reorder/extend the fallback chain to increase exact-match hit-rate and reduce averaging error.'
- What this solution (achieved 2.82325) has done: 'Your current approach is a hierarchical group-mean lookup with pressure-grid snapping, but it still misses a lot of signal because it does not use the strongest “state” feature engineered in this competition: cumulative inspired volume (`u_in` integrated over time). To move the MAE down toward the target while keeping the exact same core logic (mean tables + fallbacks + snapping), I add `area` and `area_key` (cumulative sum of `u_in` within each breath) and insert a new top-priority mean table keyed by `(R,C,step,area_key)` with a small fallback chain. I keep all your existing tables as backstops, and keep the expiratory fill behavior (u_out==1) as a stable `(R,C)` expiratory mean. This is a minimal, metric-aligned change that typically yields a large lift for this competition without changing the modeling paradigm.'
- What this solution (achieved 3.70046) has done: 'Your current score (2.82325, lower-is-better) is still far above the target (0.17897), so we need a modest but meaningful lift while keeping the exact same “hierarchical group-mean lookup + pressure-grid snapping” core logic. The biggest low-risk gain is to improve key hit-rate for your top tables by quantizing `area_key` more finely (it’s currently too coarse at 0.1 precision, causing harmful averaging), and to add one extra high-priority table using continuous `area` (rounded) together with `uin_key` at each `step`. I also ensure all merges keep `id` alignment stable and reduce accidental float-key mismatch by casting key dtypes consistently before grouping/merging. Everything else (training data usage, inspiratory-only means, expiratory fill, fallback chain, snapping) stays the same.'
- What this solution (achieved 3.69998) has done: 'Your current score (3.70046, lower-is-better) is still far above the target (0.17897), so we should improve accuracy while keeping the exact same “hierarchical group-mean lookup + fallbacks + snap-to-known-pressure-grid” core logic. The biggest minimal lift is to add one more high-signal, still-baseline key: within-breath lag features (previous `u_in`, previous `u_out`) and optionally `delta_u_in`, then build an additional top-priority mean table keyed by `(R,C,step,area_key,uin_key,u_in_lag1)` (with a small fallback) to reduce averaging error without changing the modeling paradigm. To avoid key-mismatch and merge blowups, we keep all keys in compact numeric dtypes and only add 1–2 extra merges. Everything else (inspiratory-only training, expiratory fill using `(R,C)` mean, fallback chain structure, and pressure snapping) remains the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 3.66764) has done: 'Your score is far worse than the target (MAE 3.69998 vs 0.17897, lower-is-better), so we should improve accuracy while keeping your exact “hierarchical group-mean lookup + fallbacks + snap-to-pressure-grid” approach. The most impactful minimal fix is to avoid float-key mismatches created by rounding `time_step/u_in/area` to float32 (which can silently change rounded values), by using integer-quantized keys (e.g., round to 2 decimals then store as `int`). This preserves the same semantics (same rounding), but increases merge hit-rate and reduces unintended fallbacks, typically lowering MAE substantially. I also add a tiny additional high-priority table keyed by `(R,C,step,area_key)` computed on inspiratory rows (you already have it) but ensure all merge keys share identical compact integer dtypes across train/test before grouping/merging.'
- What this solution (achieved 2.27642) has done: 'Your current score (3.66764 MAE) is much worse than the target (0.17897, lower-is-better), so we need a real lift while keeping the same “hierarchical group-mean lookup + fallbacks + snap-to-pressure-grid” core logic. The minimal high-impact change is to fix a key semantic issue: your `area` is currently a plain cumulative sum of `u_in`, but it should be time-integrated (`u_in * delta_time`) to better represent inspired volume; this remains the same feature idea, just computed correctly. I also add a single extra top-priority mean table keyed by `(R,C,step,area_key_dt)` (and corresponding merge), which improves exact-match hit rate without changing the modeling paradigm. Everything else (inspiratory-only means, expiratory fill by `(R,C)` mean, fallback chain style, and pressure snapping) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 2.81567) has done: 'Your current baseline is still suffering from key misses and noisy averaging because `area_key` and lag keys depend on cumulative sums that can drift slightly between breaths, causing lots of fallbacks. To move the MAE down toward the target with minimal change and the same “group-mean lookup + hierarchical fallbacks + snap-to-grid” core logic, I add one additional high-signal, stable key: the time-integrated cumulative volume `vol = cumsum(u_in * dt)` (you already effectively compute this as `area`) normalized within each breath by its final value, then quantized (`vol_frac_key`). This keeps semantics identical (still mean tables + merges), but improves match-rate across breaths by making the key scale-invariant, and I insert one new top-priority table `(R,C,step,vol_frac_key)` plus a small fallback `(R,C,vol_frac_key)` ahead of the existing chain. Everything else (inspiratory-only training, expiratory fill using `(R,C)` mean, pressure snapping, submission writing) is preserved.'
- What this solution (achieved 2.77171) has done: 'You’re still far above the target (2.81567 vs 0.17897 MAE, lower-is-better), so the goal is to reduce error with the smallest, score-relevant changes while keeping your exact “hierarchical group-mean lookup + fallbacks + snap-to-pressure-grid” approach. The biggest safe issue is that your highest-priority tables are competing with each other in a suboptimal order: the more specific `(R,C,step,uin_key,area_key,uin_lag1_key)` is good, but the next fallbacks should prefer stable, high-hit-rate “trajectory” keys before noisier ones, and your current chain effectively delays the most reliable `(R,C,step,vol_frac_key)` signal. I only (1) re-order the fallback chain to prioritize `vol_frac_key` trajectory tables earlier, and (2) add one minimal additional trajectory table `(R,C,step,vol_frac_key,uin_key)` that uses the same features you already compute (no new feature engineering), improving specificity without changing the modeling paradigm. Everything else (data, inspiratory-only training for mean tables, expiratory fill behavior, snapping via `find_nearest`, and writing `submission.csv`) remains the same.'
- What this solution (achieved 2.74878) has done: 'Your current gap to the target is large (2.77 vs 0.179 MAE; lower is better), so we need a real lift while keeping the exact same “group-mean tables + hierarchical fallbacks + snap-to-pressure-grid” core logic. The most score-relevant minimal change is to add the standard high-signal Ventilator feature `u_in_cum` (cumulative `u_in` within breath, not time-integrated), quantize it to an integer key, and add just one new top-priority mean table keyed by `(R,C,step,u_in_cum_key,uin_key)` with a small fallback `(R,C,step,u_in_cum_key)`. This preserves your existing feature set and tables, but improves specificity in a way that usually reduces MAE substantially for this competition. I also keep your expiratory handling and snapping unchanged, and only adjust the fallback order to consult the new cum-u_in trajectory tables early.'

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


def find_nearest(prediction: float) -> float:
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
    loop_time = 100
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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

df_train = df_train.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
df_test = df_test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

df_train["step"] = df_train.groupby("breath_id").cumcount().astype(np.int16)
df_test["step"] = df_test.groupby("breath_id").cumcount().astype(np.int16)

for df in (df_train, df_test):
    dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    df["dt"] = dt.astype(np.float32)
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()

for df in (df_train, df_test):
    df["u_in_cum"] = (
        df.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
    )

for df in (df_train, df_test):
    vol = df["area"].astype(np.float32)
    vol_end = vol.groupby(df["breath_id"], sort=False).transform("max")
    vol_frac = (vol / vol_end.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    df["vol_frac"] = vol_frac.astype(np.float32)

for df in (df_train, df_test):
    df["u_in_lag1"] = df.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
    df["u_out_lag1"] = (
        df.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    df["delta_u_in"] = df["u_in"] - df["u_in_lag1"]

train_insp = df_train[df_train["u_out"] == 0].copy()


def q2_int(x: pd.Series) -> pd.Series:
    return (x.round(2) * 100).astype(np.int32)


train_insp["ts_key"] = q2_int(train_insp["time_step"])
df_test["ts_key"] = q2_int(df_test["time_step"])

train_insp["uin_key"] = q2_int(train_insp["u_in"])
df_test["uin_key"] = q2_int(df_test["u_in"])

train_insp["area_key"] = q2_int(train_insp["area"])
df_test["area_key"] = q2_int(df_test["area"])

train_insp["uin_lag1_key"] = q2_int(train_insp["u_in_lag1"])
df_test["uin_lag1_key"] = q2_int(df_test["u_in_lag1"])

train_insp["uin_cum_key"] = q2_int(train_insp["u_in_cum"])
df_test["uin_cum_key"] = q2_int(df_test["u_in_cum"])


def q3_frac_to_int(x: pd.Series) -> pd.Series:
    return (x.clip(0.0, 1.0).round(3) * 1000).astype(np.int16)


train_insp["vol_frac_key"] = q3_frac_to_int(train_insp["vol_frac"])
df_test["vol_frac_key"] = q3_frac_to_int(df_test["vol_frac"])

for df in (train_insp, df_test):
    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)

mean_rcscuu = (
    train_insp.groupby(["R", "C", "step", "uin_cum_key", "uin_key"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcscuu"})
)
mean_rcscu = (
    train_insp.groupby(["R", "C", "step", "uin_cum_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcscu"})
)

mean_rcsvfu = (
    train_insp.groupby(["R", "C", "step", "vol_frac_key", "uin_key"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsvfu"})
)

mean_rcsvf = (
    train_insp.groupby(["R", "C", "step", "vol_frac_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsvf"})
)

mean_rcvf = (
    train_insp.groupby(["R", "C", "vol_frac_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcvf"})
)

mean_rcsa_dt = (
    train_insp.groupby(["R", "C", "step", "area_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsa_dt"})
)

mean_rcsua_ul1 = (
    train_insp.groupby(
        ["R", "C", "step", "uin_key", "area_key", "uin_lag1_key"], sort=False
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsua_ul1"})
)

mean_rcsa_ul1 = (
    train_insp.groupby(["R", "C", "step", "area_key", "uin_lag1_key"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsa_ul1"})
)

mean_rcsua = (
    train_insp.groupby(["R", "C", "step", "uin_key", "area_key"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsua"})
)

mean_rcsa = (
    train_insp.groupby(["R", "C", "step", "area_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsa"})
)

mean_rca = (
    train_insp.groupby(["R", "C", "area_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rca"})
)

mean_rcsu = (
    train_insp.groupby(["R", "C", "step", "uin_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcsu"})
)

mean_rctu = (
    train_insp.groupby(["R", "C", "ts_key", "uin_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rctu"})
)

mean_rcs = (
    train_insp.groupby(["R", "C", "step"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcs"})
)

mean_rcu = (
    train_insp.groupby(["R", "C", "uin_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcu"})
)

mean_rct = (
    train_insp.groupby(["R", "C", "ts_key"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rct"})
)

mean_rc = (
    train_insp.groupby(["R", "C"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc"})
)

global_mean = float(train_insp["pressure"].mean())

train_exp = df_train[df_train["u_out"] == 1].copy()
train_exp["R"] = train_exp["R"].astype(np.int16)
train_exp["C"] = train_exp["C"].astype(np.int16)

mean_rc_exp = (
    train_exp.groupby(["R", "C"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc_exp"})
)
global_exp_mean = float(train_exp["pressure"].mean()) if len(train_exp) else global_mean

pred = df_test[
    [
        "id",
        "R",
        "C",
        "step",
        "ts_key",
        "uin_key",
        "area_key",
        "uin_lag1_key",
        "vol_frac_key",
        "uin_cum_key",
        "u_out",
    ]
].copy()

pred = pred.merge(
    mean_rcscuu, on=["R", "C", "step", "uin_cum_key", "uin_key"], how="left"
)
pred = pred.merge(mean_rcscu, on=["R", "C", "step", "uin_cum_key"], how="left")

pred = pred.merge(
    mean_rcsvfu, on=["R", "C", "step", "vol_frac_key", "uin_key"], how="left"
)
pred = pred.merge(mean_rcsvf, on=["R", "C", "step", "vol_frac_key"], how="left")
pred = pred.merge(mean_rcvf, on=["R", "C", "vol_frac_key"], how="left")

pred = pred.merge(mean_rcsa_dt, on=["R", "C", "step", "area_key"], how="left")
pred = pred.merge(
    mean_rcsua_ul1,
    on=["R", "C", "step", "uin_key", "area_key", "uin_lag1_key"],
    how="left",
)
pred = pred.merge(
    mean_rcsa_ul1, on=["R", "C", "step", "area_key", "uin_lag1_key"], how="left"
)
pred = pred.merge(mean_rcsua, on=["R", "C", "step", "uin_key", "area_key"], how="left")
pred = pred.merge(mean_rcsa, on=["R", "C", "step", "area_key"], how="left")
pred = pred.merge(mean_rca, on=["R", "C", "area_key"], how="left")
pred = pred.merge(mean_rcsu, on=["R", "C", "step", "uin_key"], how="left")
pred = pred.merge(mean_rctu, on=["R", "C", "ts_key", "uin_key"], how="left")
pred = pred.merge(mean_rcs, on=["R", "C", "step"], how="left")
pred = pred.merge(mean_rcu, on=["R", "C", "uin_key"], how="left")
pred = pred.merge(mean_rct, on=["R", "C", "ts_key"], how="left")
pred = pred.merge(mean_rc, on=["R", "C"], how="left")
pred = pred.merge(mean_rc_exp, on=["R", "C"], how="left")

pred_pressure = (
    pred["p_rcsua_ul1"]
    .fillna(pred["p_rcsa_ul1"])
    .fillna(pred["p_rcsua"])
    .fillna(pred["p_rcscuu"])
    .fillna(pred["p_rcscu"])
    .fillna(pred["p_rcsvfu"])
    .fillna(pred["p_rcsvf"])
    .fillna(pred["p_rcvf"])
    .fillna(pred["p_rcsa_dt"])
    .fillna(pred["p_rcsa"])
    .fillna(pred["p_rca"])
    .fillna(pred["p_rcsu"])
    .fillna(pred["p_rctu"])
    .fillna(pred["p_rcs"])
    .fillna(pred["p_rcu"])
    .fillna(pred["p_rct"])
    .fillna(pred["p_rc"])
    .fillna(global_mean)
    .astype(float)
    .to_numpy()
)

exp_mask = pred["u_out"].to_numpy() == 1
exp_fill = (
    pred.loc[exp_mask, "p_rc_exp"].fillna(global_exp_mean).astype(float).to_numpy()
)
pred_pressure[exp_mask] = exp_fill

pred_pressure = np.array([find_nearest(x) for x in pred_pressure], dtype=float)

out = pd.DataFrame({"id": df_test["id"].to_numpy(), "pressure": pred_pressure})
out = out.sort_values("id").reset_index(drop=True)

out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("pressure summary:", out["pressure"].describe())
