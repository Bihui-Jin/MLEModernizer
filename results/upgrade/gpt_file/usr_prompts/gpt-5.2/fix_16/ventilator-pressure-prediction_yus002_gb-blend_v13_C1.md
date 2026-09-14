# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4374418897001822

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.445) has done: 'I remove the dependency on missing external “gb-blending” files (the cause of the FileNotFoundError) and instead generate a self-contained, valid submission directly from the provided competition data. Because we must keep the core logic intact and there is no trained model here, I implement a minimal, deterministic baseline that predicts a constant pressure equal to the training-set mean pressure, which is score-improving versus all-zeros while staying simple and stable. I also fix the cell numbering to start at 1 and ensure the output is written as a proper `submission.csv` with exactly the required `id,pressure` columns. The script automatically pick the correct `/kaggle/input/...` style path if available, otherwise fall back to the relative paths you listed.'
- What this solution (achieved 7.53006) has done: 'Your current 8.445 score comes from predicting a constant (train mean) pressure for every timestep; to move much closer to the 0.437 target (lower is better), the smallest legitimate improvement is to keep your overall pipeline but change the constant baseline into a simple per-(R,C) grouped baseline. This preserves the “no model / no training loop” core approach while leveraging the key lung attributes explicitly mentioned in the problem description. I also add an inspiratory-only mean (u_out==0) to better match the evaluation phase without changing submission semantics (we still must predict for all rows). The script still run end-to-end, remain deterministic, and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 6.19013) has done: 'Your current baseline is already using the key (R,C) grouping, but it still predicts a single constant pressure for an entire breath, which is far from the true within-breath time dynamics and keeps the MAE high. To move substantially closer to the 0.437 target while preserving the “no ML model / no training loop” core approach, the smallest effective improvement is to switch from per-(R,C) constants to a per-(R,C,time_step_index) lookup table computed from training inspiratory points only (u_out==0). This keeps the same semantics (deterministic, purely data-aggregation based), but adds the minimal time-series structure needed for a big MAE reduction. We also ensure the `id` alignment is exactly the same as `test.csv` and still write a valid `submission.csv`.'
- What this solution (achieved 6.10818) has done: 'We keep your pure aggregation-based “no ML” core logic, but make the time-step lookup more faithful to the scoring by using the per-time-step *median* (more MAE-robust than mean) on inspiratory points only. We also align test rows using the actual within-breath ordering by sorting on `time_step` before computing `t_idx`, which prevents mis-indexing if the CSV isn’t strictly ordered. Finally, we clip predictions to the observed pressure range in training to avoid rare outliers from sparse groups, which typically reduces MAE without changing the approach. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.01358) has done: 'Your current score (6.108, lower-is-better) is still far from the target (0.437), so we need a meaningful but still aggregation-based improvement without introducing a model/training loop. The biggest remaining miss is that pressure depends strongly on the current control input `u_in`, not just `(R,C,t_idx)`, so we keep your exact lookup-table approach but condition it additionally on a lightly-binned `u_in` value computed identically for train/test. To keep changes minimal and stable, we build a median table on inspiratory rows (`u_out==0`) for `(R,C,t_idx,u_in_bin)` with a safe fallback chain: full key → `(R,C,t_idx)` → global inspiratory median. We keep your sorting, `t_idx` construction, clipping, and submission alignment logic unchanged, and still write `submission.csv`.'
- What this solution (achieved 4.06948) has done: 'Your current approach is a deterministic lookup-table baseline; to move the MAE down toward the 0.437 target without changing the core logic, the smallest effective tweak is to make the `u_in` conditioning less sparse. Right now `UIN_BIN=1.0` creates many rare `(R,C,t_idx,u_in_bin)` keys, so test rows often fall back to coarser medians, hurting accuracy; increasing the bin width to a modest value should improve coverage while still capturing `u_in` dependence. I also add a one-step smoothing on the fallback `(R,C,t_idx)` median curve (within each `(R,C)` over `t_idx`) to reduce noise from per-timestep aggregation, which typically lowers MAE without introducing any new model/training. Everything else (data loading, inspiratory-only aggregation, merge keys, fallback chain, clipping, and submission writing) is kept the same.'
- What this solution (achieved 3.98296) has done: 'I fix the runtime error in `pd.cut` by ensuring the quantile-derived bin edges are strictly increasing (dropping duplicates) and by gracefully falling back to a simpler binning scheme when the data doesn’t support the requested number of bins. This keeps your exact aggregation/lookup-table core logic intact (median tables + fallback chain + smoothing + clipping) while making it robust and deterministic across pandas versions and edge cases. I also make sure `test` includes `u_out` (available in test.csv) so later extensions won’t break, and keep the submission writing unchanged (`submission.csv` with `id,pressure`). These changes are correctness/stability fixes and should preserve or improve score versus the currently failing run.'
- What this solution (achieved 4.1668) has done: 'Your current approach is already a deterministic lookup-table baseline; to move the MAE down toward the 0.437 target with minimal risk, we only make the `u_in` conditioning denser (fewer sparse keys) so fewer test rows fall back to coarser medians. Concretely, we replace the quantile-based `pd.cut` binning with a robust fixed-width binning on `u_in` (still deterministic, still only aggregation + fallback), which typically improves coverage and reduces MAE. We keep the same inspiratory-only median tables, the same fallback chain `(R,C,t_idx,u_in_bin) -> (R,C,t_idx) -> global`, the same smoothing on `(R,C,t_idx)`, and the same clipping and submission writing. The rest of the script is left intact; cells that are unused remain unchanged to preserve core logic.'
- What this solution (achieved 4.31066) has done: 'Your current MAE (4.1668, lower-is-better) is still far above the target (0.437), so we should make a small but meaningful improvement while keeping the same lookup-table core logic. The largest remaining avoidable error is that the fallback chain ignores the within-breath dependence on `u_in` whenever `(R,C,t_idx,u_in_bin)` is missing; we can reduce fallback frequency by using a slightly coarser `u_in` bin width so more test rows hit the most-specific median. I only change `UIN_BIN_WIDTH` (and keep the same inspiratory-only medians, smoothing, clipping, merges, and submission writing) to improve coverage and move the score down toward the target. Everything else remains identical to preserve semantics and stability.'
- What this solution (achieved 4.26098) has done: 'Your current score (4.31066, lower is better) is still far above the target (0.43744), so we should modestly improve accuracy while keeping the same deterministic lookup-table core logic. The main issue is remaining sparsity/mismatch in the `(R,C,t_idx,u_in_bin)` table; making `u_in` bins slightly finer should increase specificity without exploding missing keys. I change only `UIN_BIN_WIDTH` from 4.0 to 3.0 (everything else identical), which should reduce MAE by better capturing `u_in` dependence while preserving your aggregation + fallback + smoothing + clipping approach. I also fix the cell numbering to start at 1 (required by your format) without changing execution.'
- What this solution (achieved 4.2906) has done: 'We keep your exact aggregation/lookup-table pipeline (inspiratory-only medians, fallback chain, smoothing, clipping, deterministic submission) and only adjust the `u_in` binning granularity, since the current MAE is still far above the target and the main controllable error source is sparsity vs specificity in `(R,C,t_idx,u_in_bin)`. The last change to `UIN_BIN_WIDTH=3.0` likely made keys a bit too sparse again; we move to a slightly coarser bin width to increase hit-rate on the most-specific table and reduce fallback usage. Everything else stays identical, including sorting, `t_idx`, smoothing window, and submission alignment. This is the smallest change expected to reduce MAE (lower-is-better) toward your 0.437 target.'
- What this solution (achieved 4.33295) has done: 'Your current MAE (4.2906, lower-is-better) is still far above the target (0.4374), so we should improve accuracy while keeping your deterministic lookup-table core logic unchanged. The most impactful minimal tweak is to reduce avoidable fallback usage by making the `u_in` binning *slightly* coarser (more train/test key matches) while keeping the same `(R,C,t_idx,u_in_bin) -> (R,C,t_idx) -> global` fallback chain. I only change `UIN_BIN_WIDTH` from 3.5 to 5.0 (and keep sorting, inspiratory-only medians, smoothing, clipping, and submission writing identical). This should increase hit-rate on the most-specific table and move MAE down toward the target without altering the overall approach.'
- What this solution (achieved 4.35388) has done: 'We keep your exact lookup-table/median + fallback pipeline and only make the `u_in` binning slightly more expressive without increasing sparsity: switch from fixed-width bins to per-`(R,C,t_idx)` *quantile* bins computed on inspiratory rows. This preserves the same semantics (deterministic aggregation, no training loop/model), but should reduce MAE by matching `u_in`-pressure relationships locally at each time step and lung setting. To avoid missing keys, we keep your same fallback chain and additionally fall back from the local quantile-bin median to the existing fixed `(R,C,t_idx)` median and then global median. All I/O paths, sorting, `t_idx` creation, smoothing, clipping, and submission writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import copy
import random
from random import random as rd

import numpy as np
import pandas as pd




## === cell 1
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
        weight1 = 0.8
        weight2 = 0.2
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
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
def blend(a, b, out_path="blend.csv"):
    if (a is None) or (b is None) or (not os.path.exists(a)) or (not os.path.exists(b)):
        raise FileNotFoundError(
            f"Blend inputs not found. Got a={a} (exists={os.path.exists(a) if a else False}), "
            f"b={b} (exists={os.path.exists(b) if b else False})."
        )
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.73 + b["pressure"] * 0.27
    a.to_csv(out_path, index=False)
    return a




## === cell 3
def resolve_base_path():
    candidates = [
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "/kaggle/input",
        "/kaggle/data",
        "../input/ventilator-pressure-prediction",
        "../input",
        "/kaggle/data/ventilator-pressure-prediction/ventilator-pressure-prediction",
        "/kaggle/input/ventilator-pressure-prediction/ventilator-pressure-prediction",
        "/kaggle/data",
        "/kaggle/input",
    ]
    for p in candidates:
        if os.path.exists(p):
            train_path = os.path.join(p, "train.csv")
            test_path = os.path.join(p, "test.csv")
            sample_path = os.path.join(p, "sample_submission.csv")
            if (
                os.path.exists(train_path)
                and os.path.exists(test_path)
                and os.path.exists(sample_path)
            ):
                return p
    if os.path.exists("/kaggle/data/train.csv") and os.path.exists(
        "/kaggle/data/test.csv"
    ):
        return "/kaggle/data"
    raise FileNotFoundError(
        "Could not resolve dataset base path containing train.csv/test.csv/sample_submission.csv"
    )


BASE = resolve_base_path()
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

print("Using BASE:", BASE)

train = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)
sub = pd.read_csv(sample_path, usecols=["id", "pressure"])

if len(sub) != len(test) or not np.array_equal(sub["id"].values, test["id"].values):
    sub = test[["id"]].merge(sub, on="id", how="left", suffixes=("", "_y"))
    if "pressure_y" in sub.columns:
        sub = sub[["id", "pressure_y"]].rename(columns={"pressure_y": "pressure"})
    else:
        sub = sub[["id"]]
        sub["pressure"] = 0.0

train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)
test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

train["t_idx"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
test["t_idx"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

insp = train[train["u_out"] == 0]

global_insp_median = (
    float(insp["pressure"].median()) if len(insp) else float(train["pressure"].median())
)

insp = insp.copy()

LOCAL_Q_BINS = (
    12  # keep unchanged (core logic); only fix train/test bin assignment consistency
)

grp_keys = ["R", "C", "t_idx"]

uq = insp.groupby(grp_keys, sort=False)["u_in"].apply(
    lambda s: np.sort(s.to_numpy(np.float64))
)

insp["u_in_qbin"] = np.int16(-1)
for key, idx in insp.groupby(grp_keys, sort=False).indices.items():
    if key in uq.index:
        arr = uq.loc[key]
        n = arr.size
        if n > 0:
            uvals = insp.loc[idx, "u_in"].to_numpy(np.float64)
            pos = np.searchsorted(arr, uvals, side="right")  # 0..n
            q = pos / n  # [0,1]
            bins = np.floor(q * LOCAL_Q_BINS).astype(np.int16)
            bins = np.clip(bins, 0, LOCAL_Q_BINS - 1).astype(np.int16)
            insp.loc[idx, "u_in_qbin"] = bins

test = test.copy()
test["u_in_qbin"] = np.int16(-1)
for key, idx in test.groupby(grp_keys, sort=False).indices.items():
    if key in uq.index:
        arr = uq.loc[key]
        n = arr.size
        if n > 0:
            uvals = test.loc[idx, "u_in"].to_numpy(np.float64)
            pos = np.searchsorted(arr, uvals, side="right")
            q = pos / n
            bins = np.floor(q * LOCAL_Q_BINS).astype(np.int16)
            bins = np.clip(bins, 0, LOCAL_Q_BINS - 1).astype(np.int16)
            test.loc[idx, "u_in_qbin"] = bins

rctu_median = (
    insp.groupby(["R", "C", "t_idx", "u_in_qbin"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rctu"})
)

rct_median = (
    insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rct"})
)

rct_median = rct_median.sort_values(["R", "C", "t_idx"], kind="mergesort").reset_index(
    drop=True
)
rct_median["pressure_rct"] = (
    rct_median.groupby(["R", "C"], sort=False)["pressure_rct"]
    .transform(lambda s: s.rolling(window=3, center=True, min_periods=1).median())
    .astype(np.float64)
)

pred = test.copy()
pred = pred.merge(rctu_median, on=["R", "C", "t_idx", "u_in_qbin"], how="left")
pred = pred.merge(rct_median, on=["R", "C", "t_idx"], how="left")

pred_pressure = (
    pred["pressure_rctu"]
    .fillna(pred["pressure_rct"])
    .fillna(global_insp_median)
    .astype(np.float64)
)

pmin = float(train["pressure"].min())
pmax = float(train["pressure"].max())
pred_pressure = np.clip(pred_pressure, pmin, pmax)

sub = test[["id"]].copy()
sub["pressure"] = pred_pressure
sub = sub.sort_values("id").reset_index(drop=True)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

hit_rate = float(np.mean(~pd.isna(pred["pressure_rctu"]))) if len(pred) else 0.0
print(
    "Wrote",
    out_path,
    "with shape",
    sub.shape,
    "| global_insp_median =",
    global_insp_median,
    "| pressure clip =",
    (pmin, pmax),
    "| LOCAL_Q_BINS =",
    LOCAL_Q_BINS,
    "| rctu rows =",
    int(rctu_median.shape[0]),
    "| rct rows =",
    int(rct_median.shape[0]),
    "| most-specific hit-rate =",
    hit_rate,
)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2316942384.py in <cell line: 0>()
     93         n = arr.size
     94         if n > 0:
---> 95             uvals = insp.loc[idx, "u_in"].to_numpy(np.float64)
     96             pos = np.searchsorted(arr, uvals, side="right")  # 0..n
     97             q = pos / n  # [0,1]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1366         with suppress(IndexingError):
   1367             tup = self._expand_ellipsis(tup)
-> 1368             return self._getitem_lowerdim(tup)
   1369 
   1370         # no multi-index, so validate all of the indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
   1087                     return section
   1088                 # This is an elided recursive call to iloc/loc
-> 1089                 return getattr(section, self.name)[new_key]
   1090 
   1091         raise IndexingError("not applicable")

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: '[391, 698, 1986, 2554, 2950, 3256, 4357, 4749, 5027, 5235, 5809, 6623, 6840, 8143, 8442, 8532, 9744, 10287, 10316, 10798, 11668, 11819, 12953, 13108, 13256, 13315, 13799, 15083, 15353, 15502, 15961, 16685, 16777, 17199, 17503, 19150, 19663, 19814, 19996, 20142, 20694, 21909, 22031, 22154, 22516, 23187, 23403, 24616, 24674, 25167, 26129, 26769, 27257, 28295, 28385, 29633, 30301, 30605, 31350, 32398, 32456, 32696, 33177, 33636, 33972, 34605, 35027, 35813, 35905, 36122, 37097, 37485, 37515, 38067, 38369, 39097, 39188, 39276, 39491, 39640, 39732, 40364, 40634, 41029, 41480, 41509, 41718, 42210, 42360, 42696, 43749, 44056, 45082, 45327, 45743, 46230, 46290, 46555, 46618, 46833, 47017, 47166, 47674, 48277, 49161, 50385, 50600, 50780, 51175, 51661, 52060, 53028, 53722, 54291, 54778, 55267, 55355, 55568, 55630, 55873, 55902, 56295, 56992, 57081, 58300, 58360, 59475, 59504, 59987, 60110, 60140, 61590, 61957, 62536, 63084, 63113, 63757, 64947, 65191, 65584, 65893, 67159, 67252, 67916, 68431, 69036, 69494, 69739, 70140, 70201, 70289, 70716, 71265, 71327, 71506, 71901, 73641, 74368, 75241, 75754, 75964, 76148, 77243, 77421, 77510, 77722, 78513, 80457, 80852, 81241, 81328, 81749, 81873, 82639, 84058, 84208, 84297, 84689, 84844, 85653, 86072, 86955, 87104, 87316, 87501, 88041, 88474, 88999, 89335, 89975, 90129, 90433, 91096, 91646, 92368, 92516, 92604, 92753, 93662, 93962, 94298, 95481, 96219, 96918, 98607, 98794, 98855, 99004, 100315, 100555, 101092, 101241, 103243, 103515, 103969, 104766, 104795, 105007, 105341, 105554, 106037, 107115, 108297, 108691, 109416, 109659, 110150, 110211, 110545, 111278, 111402, 111739, 112355, 112836, 113956, 114077, 114284, 114314, 114993, 115880, 115971, 116063, 116614, 116793, 117039, 117311, 119164, 119255, 119493, 119585, 119830, 121343, 121401, 121494, 121892, 121953, 122132, 123260, 124056, 124542, 125391, 126283, 126953, 127652, 127712, 128600, 128690, 128719, 129265, 129748, 130077, 130624, 130713, 131589, 132130, 132533, 133747, 134443, 134841, 135564, 136134, 136316, 137106, 137164, 137343, 137650, 138231, 138444, 139024, 139759, 141185, 142030, 142397, 143671, 144372, 144771, 144920, 145338, 146948, 147101, 147161, 147499, 147587, 150075, 151031, 151731, 152538, 152839, 153716, 154046, 155658, 155966, 156617, 157097, 157882, 157912, 158361, 158454, 159153, 159182, 160371, 160430, 161279, 161554, 162613, 162854, 163741, 163954, 164592, 164711, 165676, 165793, 165823, 166217, 166680, 167312, 167492, 167673, 168155, 168216, 168369, 168398, 168915, 169554, 169733, 169913, 170271, 170388, 171421, 171755, 172304, 172366, 172396, 172521, 172704, 173181, 173333, 174225, 174470, 174530, 174711, 174925, 175875, 176029, 176361, 176935, 177335, 177724, 178145, 179186, 179550, 179816, 181489, 181727, 182238, 182635, 182841, 182989, 183719, 183960, 184113, 184957, 185502, 185801, 186065, 186368, 187438, 187498, 188072, 188282, 189162, 189799, 190471, 190531, 191412, 191805, 192133, 192527, 192707, 192859, 193038, 194466, 194527, 195405, 195588, 195647, 195835, 195959, 197259, 197588, 198132, 199105, 199350, 199958, 200438, 200469, 200558, 200772, 201013, 201403, 202690, 203504, 203593, 203900, 204291, 204383, 204478, 205907, 206273, 206940, 207030, 207881, 207910, 208545, 209037, 209488, 210945, 211006, 211406, 211735, 212040, 212192, 212314, 212776, 213353, 213655, 214051, 215728, 216029, 216120, 216214, 216367, 216396, 217035, 217395, 217426, 219111, 219654, 219833, 220534, 221260, 221815, 222364, 222456, 222514, 222784, 223241, 223489, 223518, 226286, 226530, 226680, 226832, 226862, 227167, 227196, 228073, 228675, 228855, 228915, 228944, 229741, 230290, 231320, 231905, 232548, 232699, 234766, 235160, 235886, 236956, 237348, 237832, 238350, 239083, 239959, 240287, 240861, 241160, 241339, 241635, 242297, 242879, 243029, 243571, 244207, 244237, 244541, 245188, 245246, 245874, 246300, 246362, 248276, 248638, 248914, 249273, 249576, 250029, 250789, 251098, 252536, 253024, 253234, 253322, 253743, 254203, 257110, 257805, 258040, 258954, 259253, 259314, 259741, 260713, 261411, 261589, 262295, 262537, 264299, 264631, 265263, 267759, 267878, 268271, 268610, 269102, 269253, 269311, 269490, 269799, 270434, 270861, 270953, 271594, 271656, 272051, 272113, 273753, 275895, 276201, 276473, 276781, 277824, 278314, 278435, 278714, 279265, 279415, 280931, 280992, 281420, 281482, 281661, 281749, 282542, 282633, 282692, 283188, 283738, 283887, 284126, 284394, 284670, 284911, 285031, 285179, 285239, 285580, 285731, 286157, 286677, 286918, 287197, 287409, 288715, 288958, 289506, 289750, 292150, 292271, 292756, 294513, 294789, 294850, 295184, 295816, 295908, 296360, 297667, 298120, 298449, 298603, 298632, 298690, 299580, 299671, 300157, 300217, 300307, 300398, 300919, 302295, 303513, 304125, 304157, 304275, 304363, 305033, 305489, 307405, 307954, 308753, 308933, 309410, 309744, 310232, 310475, 311895, 312202, 312875, 313177, 313575, 313634, 313877, 313966, 314116, 314520, 314762, 314911, 314940, 315423, 315484, 315910, 316152, 318912, 319917, 320376, 320617, 320923, 321321, 321631, 321907, 322299, 322511, 322601, 323025, 324859, 325316, 325437, 325742, 326475, 327951, 328687, 328716, 328924, 329952, 330770, 331322, 331414, 331472, 332073, 332279, 332308, 332529, 334290, 334441, 335806, 336358, 336760, 337501, 337592, 337833, 338443, 338867, 338926, 339636, 339816, 340059, 340118, 340211, 340758, 341430, 341676, 342192, 342773, 343107, 344237, 344362, 345421, 345910, 346914, 347278, 347403, 347646, 347738, 347887, 348553, 349010, 349102, 349741, 349801, 351811, 352447, 352476, 352955, 353470, 354874, 354934, 354992, 355173, 355416, 355475, 356686, 356773, 357010, 357491, 357551, 358155, 358679, 358829, 358859, 359557, 360532, 362530, 362833, 362923, 363163, 363193, 363412, 363658, 364689, 364837, 364925, 365325, 365720, 365749, 365902, 365961, 366053, 367240, 367390, 367510, 368397, 368794, 369439, 369563, 370467, 371343, 371823, 372516, 372853, 373002, 373031, 373274, 373903, 373994, 374692, 375116, 376696, 376851, 377515, 378122, 378213, 378304, 378363, 378935, 380064, 380515, 381336, 381640, 381822, 382064, 382610, 383004, 383184, 383277, 383734, 384686, 386363, 386451, 386513, 386699, 387001, 387515, 387977, 388223, 388952, 389838, 390353, 391020, 391718, 392114, 393554, 394037, 394310, 394398, 395187, 395828, 396071, 397011, 397162, 397253, 397313, 400158, 401156, 401338, 401880, 402152, 402211, 402299, 402759, 403397, 403517, 403725, 404550, 405076, 405105, 405253, 405409, 405895, 406291, 407508, 407753, 408858, 409072, 410042, 412055, 412941, 413482, 413793, 413823, 413881, 414630, 415086, 415509, 415570, 415759, 416213, 416397, 416455, 416910, 417091, 417332, 417483, 417729, 418458, 418796, 419830, 419890, 420532, 421974, 422277, 422460, 422520, 422609, 422788, 422993, 423883, 424066, 424220, 424518, 424911, 424940, 425000, 426670, 426999, 427641, 428040, 428194, 428316, 428440, 428469, 429319, 429742, 429804, 430192, 430709, 431325, 431571, 431820, 432309, 432367, 432610, 432670, 433095, 433153, 433182, 434391, 435001, 435188, 435277, 435427, 435727, 435814, 435967, 435996, 436239, 436997, 437236, 437732, 437792, 438365, 439157, 441015, 442236, 442851, 442912, 443157, 443187, 443671, 444621, 445017, 445504, 446146, 446478, 446595, 446717, 447317, 447676, 448131, 448463, 449099, 449344, 449677, 449738, 449886, 450217, 451072, 451742, 452043, 452192, 452833, 453075, 453165, 453195, 453650, 454199, 455272, 455813, 455904, 457819, 457877, 458678, 459675, 459984, 460044, 460318, 460779, 460839, 461474, 461872, 461966, 462363, 462784, 462873, 463818, 465113, 465919, 466383, 466834, 466953, 467105, 468525, 468554, 468705, 468948, 469196, 470592, 470621, 471596, 471897, 471988, 472050, 473116, 473391, 473817, 476213, 476454, 476793, 477251, 477979, 479100, 479488, 480031, 482214, 482364, 482849, 482912, 483003, 483514, 483759, 484220, 484280, 484916, 485556, 485645, 485918, 486035, 486274, 489150, 489426, 489759, 490542, 491002, 491248, 491640, 492609, 493424, 493482, 493971, 494362, 494391, 495027, 495329, 495483, 495720, 496152, 496272, 496362, 496392, 497031, 497914, 499096, 499155, 499481, 499754, 499905, 500059, 500120, 500149, 500942, 501492, 501579, 501885, 502279, 502430, 503591, 503838, 504291, 504381, 504717, 505320, 505871, 506207, 506478, 507176, 507566, 507657, 508117, 508599, 508874, 509178, 509661, 510772, 510833, 510862, 511076, 511166, 511198, 511317, 511411, 511502, 512458, 513012, 513104, 513342, 514073, 514612, 516862, 518224, 519321, 519352, 519500, 519589, 520527, 520920, 520952, 521012, 521102, 521588, 521646, 522159, 523185, 523490, 524440, 524680, 524709, 524768, 524857, 524917, 524946, 525243, 525664, 525913, 526463, 526799, 526859, 527192, 527593, 527654, 529099, 529494, 529676, 529796, 529825, 530068, 530998, 531328, 531476, 532120, 532149, 532851, 533185, 533642, 533730, 534031, 534212, 534456, 534763, 534911, 535032, 535094, 535401, 535676, 535825, 535974, 536036, 537321, 537722, 538112, 538783, 539425, 540678, 541012, 541071, 541499, 541677, 541827, 541916, 542371, 543005, 543877, 543966, 544689, 544718, 544840, 545087, 545816, 545992, 546297, 546604, 547485, 547817, 548672, 548702, 550372, 550945, 551187, 551431, 552547, 553652, 554076, 554472, 554590, 556046, 556351, 556381, 556779, 557260, 558353, 560115, 560144, 560233, 560384, 560476, 561349, 561999, 562511, 562753, 563266, 563480, 563968, 563997, 564205, 564843, 565817, 565904, 566627, 567235, 567811, 567898, 568529, 568928, 568959, 569267, 570141, 570233, 571107, 571434, 572319, 573336, 573487, 573518, 573974, 574033, 574304, 574547, 574697, 575001, 575433, 575886, 576553, 576706, 576858, 577401, 577829, 579888, 580196, 580704, 581272, 581880, 581967, 582638, 582698, 583030, 583572, 583725, 583874, 583963, 586760, 587159, 587404, 587744, 588043, 588711, 588863, 589345, 591721, 591904, 592293, 592386, 592535, 593238, 594029, 594399, 594913, 596956, 597748, 597837, 598476, 598692, 598992, 599883, 600375, 600676, 601231, 601590, 601979, 602220, 602399, 602518, 602698, 603496, 604136, 604445, 606435, 607316, 607654, 609192, 609586, 610138, 611419, 611716, 611901, 612052, 612355, 612475, 613757, 614031, 614782, 615084, 615995, 616387, 616624, 616684, 617658, 618510, 618752, 619346, 619435, 619582, 619642, 620307, 621033, 621092, 621153, 621393, 621580, 621671, 622150, 623759, 624462, 624521, 624672, 625250, 625893, 627258, 627555, 628440, 628684, 629169, 629439, 629650, 629710, 629888, 629917, 630214, 630515, 633592, 634352, 634839, 635197, 635470, 635960, 636598, 637337, 637426, 638152, 638458, 638786, 638846, 639239, 639637, 641411, 642319, 642379, 642558, 645098, 645275, 645489, 645736, 646035, 646461, 646762, 646851, 646911, 647427, 647677, 648069, 648464, 648552, 648859, 648922, 649256, 649885, 651326, 652205, 652876, 653638, 653818, 655010, 655314, 655675, 655888, 656457, 657001, 657150, 657272, 657334, 657581, 657798, 657829, 658284, 659109, 659964, 659993, 660111, 660684, 660865, 661111, 661262, 661353, 661504, 662030, 662368, 663648, 663800, 664191, 664311, 664680, 664797, 665193, 666768, 667172, 667325, 668154, 668308, 668948, 670034, 670303, 670516, 671004, 672068, 672433, 672703, 673096, 674430, 674853, 675829, 676225, 676286, 677555, 677672, 677825, 678279, 679882, 680548, 680944, 681278, 681639, 682279, 685192, 685310, 685400, 685429, 685550, 685731, 685792, 687317, 687346, 687404, 688308, 688518, 688759, 689891, 690039, 690158, 690608, 690701, 690949, 691010, 691806, 692292, 692909, 693576, 693725, 694519, 695098, 695733, 697119, 697181, 697631, 698778, 699172, 699230, 700858, 701560, 701745, 702143, 702236, 702604, 702751, 704307, 704521, 704613, 704920, 705560, 705837, 705958, 706204, 706293, 707117, 707421, 707574, 707884, 707913, 708215, 708514, 709092, 709335, 709429, 709578, 709819, 710213, 710302, 710392, 710603, 711237, 711327, 711906, 712536, 712843, 713358, 713633, 713754, 715358, 715635, 716763, 716945, 717005, 717155, 717889, 718755, 718876, 719420, 719599, 719995, 720057, 721995, 723500, 723590, 723894, 724044, 724465, 724838, 725112, 725725, 726610, 727487, 727728, 729271, 729969, 730362, 730781, 731359, 731633, 732951, 733313, 733342, 734638, 734759, 734876, 735246, 737016, 737434, 737951, 738072, 739073, 739410, 739647, 740312, 741015, 741808, 741896, 744479, 744600, 744786, 745089, 745966, 747177, 747358, 748202, 748631, 749083, 749509, 749569, 749967, 750029, 750544, 751093, 751797, 752124, 752213, 752757, 753251, 753495, 753643, 754798, 754948, 755501, 755714, 755986, 756078, 756232, 756381, 756594, 757020, 757716, 758231, 758476, 759088, 759792, 760306, 762297, 762701, 764146, 764788, 765723, 766116, 766475, 766538, 766629, 766689, 767721, 768115, 768297, 768785, 769813, 769964, 770238, 770879, 771266, 771476, 772552, 773576, 774632, 774692, 775150, 775328, 775358, 775634, 775790, 776129, 776373, 776677, 777101, 778113, 778233, 778359, 778756, 778876, 779112, 779170, 779437, 779556, 779738, 779801, 779832, 779890, 780127, 780275, 780304, 780633, 781352, 781507, 784219, 784280, 784915, 785399, 785733, 786130, 786220, 787713, 787744, 787832, 788044, 788075, 788465, 788678, 790137, 790195, 790466, 790711, 790770, 793099, 793494, 795410, 795590, 795827, 796353, 797269, 797951, 798226, 798376, 798439, 798832, 798924, 799560, 800235, 800442, 800953, 801073, 801103, 801834, 801985, 802624, 802776, 802866, 803558, 803795, 805243, 806682, 806776, 806839, 807512, 807815, 808691, 808752, 809759, 810151, 810209, 810869, 812283, 812589, 815329, 816118, 817599, 817749, 817899, 818381, 818715, 818925, 819107, 819316, 819409, 819472, 819648, 819954, 820715, 820775, 821016, 821074, 821650, 821740, 821828, 821980, 822308, 823276, 823886, 824130, 824281, 825072, 825678, 826432, 827255, 828474, 829321, 829677, 829893, 830631, 830780, 830840, 831484, 831882, 832610, 832697, 834468, 834923, 835163, 835193, 836239, 836298, 836689, 836780, 837197, 837256, 837592, 837716, 837956, 838291, 838378, 839899, 841094, 841913, 842280, 842310, 842852, 844433, 844769, 845745, 846450, 846870, 846994, 847480, 847509, 847570, 847720, 848236, 848384, 851156, 851277, 851337, 851487, 852124, 853155, 853549, 853886, 854280, 855712, 856286, 856715, 857085, 858995, 860238, 860390, 861564, 861837, 861989, 862285, 862465, 862710, 862925, 864788, 865183, 865271, 865726, 865757, 866123, 866152, 868463, 869406, 869735, 870137, 870200, 870349, 870377, 870618, 870954, 872136, 873168, 873258, 873415, 873596, 874053, 874145, 874629, 875325, 875508, 875727, 876397, 876915, 877096, 877157, 877837, 878145, 879004, 879819, 879998, 880029, 880208, 880752, 881023, 881358, 881570, 881807, 882359, 882451, 882605, 883274, 883392, 883971, 884152, 884306, 884461, 884613, 885006, 885889, 886040, 886557, 886767, 886798, 886859, 887193, 887793, 888132, 888528, 889549, 889878, 890149, 890361, 890847, 891177, 891268, 891664, 891817, 892207, 892695, 893092, 893156, 895117, 895178, 895635, 896061, 896304, 896547, 896638, 896912, 897732, 897793, 898275, 898456, 898792, 900139, 901439, 901471, 901796, 902073, 902316, 903327, 903876, 904521, 905419, 905510, 905721, 905964, 909094, 909519, 909792, 909968, 910117, 910932, 911116, 912277, 912703, 912859, 912922, 913962, 914357, 914386, 914690, 915319, 915652, 916134, 917326, 918550, 918610, 918853, 919192, 919253, 919313, 919344, 919433, 919677, 920378, 920534, 920719, 920958, 921751, 922998, 923392, 924111, 925570, 925720, 925813, 926147, 926272, 926782, 927234, 927354, 928147, 928206, 928448, 928478, 928992, 930119, 930453, 930785, 930875, 931086, 932691, 932749, 932837, 932927, 933439, 934538, 934929, 935357, 935480, 936238, 936298, 936877, 937329, 937599, 938757, 938850, 939249, 939640, 939671, 939823, 940066, 940851, 941274, 941396, 941725, 942235, 942296, 942777, 943083, 943750, 943839, 945152, 945969, 946209, 946779, 947176, 947236, 947670, 948306, 951654, 952285, 954353, 954931, 955410, 956543, 956848, 957427, 958128, 958192, 959108, 959198, 959318, 960472, 960769, 961009, 962077, 962534, 962869, 963575, 964674, 964861, 964921, 965341, 965671, 965974, 967310, 968557, 968959, 969321, 969505, 969838, 970049, 970109, 971427, 971832, 973235, 975635, 975664, 977317, 977711, 978139, 978716, 979800, 980073, 980376, 980591, 981969, 982032, 982365, 982394, 982787, 983584, 984767, 985157, 987496, 989759, 989999, 990541, 991552, 992837, 993078, 993257, 993316, 994075, 994229, 994290, 994475, 995084, 995511, 996599, 996786, 997434, 997826, 998190, 998921, 998950, 1000078, 1000226, 1001346, 1001648, 1001833, 1001985, 1002771, 1003168, 1003438, 1003497, 1003587, 1003675, 1004158, 1004706, 1004927, 1005197, 1005320, 1005713, 1006378, 1006676, 1007160, 1007980, 1009321, 1009350, 1009502, 1010198, 1010318, 1010379, 1012052, 1012140, 1013085, 1013176, 1013748, 1014451, 1014753, 1015233, 1015262, 1017325, 1017415, 1017965, 1020190, 1020372, 1020769, 1021015, 1021172, 1021594, 1021992, 1022384, 1023670, 1023731, 1025651, 1025712, 1026196, 1026468, 1027473, 1027592, 1028468, 1029319, 1029502, 1031000, 1031330, 1031513, 1031660, 1031720, 1032598, 1032718, 1033324, 1034757, 1034788, 1034938, 1035238, 1035599, 1036690, 1037653, 1037746, 1037807, 1037987, 1038204, 1038233, 1038754, 1038783, 1039025, 1039086, 1039237, 1039417, 1039723, 1040049, 1040538, 1040750, 1041889, 1042467, 1042915, 1043098, 1043250, 1043676, 1044397, 1045739, 1046876, 1047338, 1047428, 1047731, 1047975, 1048678, 1049348, 1050143, 1050837, 1050926, 1053404, 1053952, 1054040, 1054534, 1054594, 1054869, 1055959, 1056385, 1057089, 1057267, 1057484, 1057513, 1057661, 1058058, 1058453, 1059243, 1060271, 1060854, 1061004, 1061150, 1061424, 1061667, 1061819, 1062117, 1062512, 1063177, 1064932, 1065421, 1066385, 1066479, 1066718, 1067662, 1067752, 1067996, 1068452, 1070319, 1071022, 1071085, 1071174, 1071956, 1071985, 1072289, 1073592, 1073654, 1074688, 1074865, 1074956, 1075016, 1075105, 1075348, 1075436, 1076139, 1076533, 1077817, 1077970, 1078791, 1079487, 1080057, 1080359, 1080630, 1081026, 1081178, 1081981, 1082863, 1082952, 1083161, 1083888, 1083982, 1084225, 1084676, 1085014, 1085073, 1085491, 1086457, 1087005, 1087402, 1088110, 1088228, 1088684, 1088866, 1088924, 1089433, 1091502, 1092055, 1092200, 1092289, 1094078, 1094139, 1094923, 1095401, 1095497, 1095559, 1097022, 1097475, 1097658, 1098550, 1099317, 1099407, 1099471, 1099837, 1100139, 1100287, 1100316, 1100715, 1101082, 1101233, 1103880, 1103911, 1104154, 1104525, 1105010, 1105645, 1105973, 1106065, 1106453, 1106545, 1106940, 1107030, 1107155, 1107405, 1107589, 1107647, 1108133, 1108221, 1108920, 1109192, 1109796, 1110278, 1110370, 1110460, 1110523, 1112191, 1112310, 1112368, 1112549, 1112670, 1113795, 1114226, 1115088, 1115332, 1115391, 1115482, 1115824, 1117553, 1117887, 1118130, 1118463, 1119425, 1119518, 1120070, 1120946, 1121005, 1121092, 1121431, 1121731, 1122559, 1124473, 1125085, 1125713, 1125830, 1126042, 1126829, 1127824, 1128126, 1128213, 1128606, 1129036, 1129489, 1130068, 1130550, 1131347, 1131917, 1132127, 1132159, 1132463, 1132768, 1132798, 1133801, 1134042, 1134284, 1134313, 1134431, 1134759, 1135519, 1136072, 1136443, 1136751, 1139112, 1140553, 1140611, 1141192, 1141554, 1142549, 1143004, 1143246, 1143397, 1143910, 1144303, 1144391, 1144604, 1145091, 1145337, 1145399, 1145737, 1145887, 1146685, 1146840, 1147275, 1147397, 1147426, 1148279, 1148680, 1149076, 1149317, 1151635, 1151758, 1151967, 1153156, 1153246, 1153336, 1154759, 1154850, 1157310, 1157830, 1158375, 1158527, 1158709, 1159591, 1160115, 1160513, 1161661, 1161720, 1162055, 1162205, 1162389, 1162543, 1162851, 1162912, 1164678, 1166220, 1167038, 1167099, 1167912, 1168391, 1169032, 1169510, 1169901, 1170141, 1170352, 1171567, 1171724, 1171873, 1172300, 1172448, 1172537, 1173518, 1173971, 1174525, 1175113, 1175563, 1176140, 1177026, 1177325, 1177354, 1177502, 1177837, 1178231, 1178445, 1178537, 1178716, 1178839, 1179576, 1179634, 1182037, 1183579, 1184037, 1184157, 1184312, 1184672, 1184794, 1184853, 1184914, 1184943, 1185030, 1185400, 1185431, 1185492, 1186217, 1187252, 1187952, 1188283, 1188763, 1189095, 1189823, 1189884, 1190037, 1190855, 1191409, 1191591, 1192045, 1192955, 1193317, 1193406, 1194556, 1194708, 1195159, 1196517, 1196795, 1197099, 1197279, 1197554, 1197917, 1198674, 1199916, 1200315, 1200621, 1200771, 1200831, 1200919, 1201100, 1201159, 1201557, 1201803, 1201958, 1202470, 1203105, 1203258, 1203407, 1203651, 1204558, 1205232, 1205564, 1205806, 1205895, 1206074, 1206381, 1206774, 1207111, 1207259, 1207739, 1208381, 1209358, 1209571, 1210450, 1211091, 1211393, 1211796, 1212524, 1212767, 1212854, 1212914, 1213277, 1215728, 1216121, 1216433, 1217893, 1218319, 1219080, 1220078, 1221027, 1221747, 1223172, 1224765, 1225278, 1225578, 1225912, 1226064, 1227505, 1227711, 1227741, 1227833, 1227893, 1228379, 1228837, 1229171, 1229724, 1229814, 1230383, 1230629, 1231031, 1231276, 1231397, 1231487, 1232395, 1232609, 1233976, 1234034, 1234213, 1234276, 1234852, 1234914, 1235244, 1235915, 1235974, 1236221, 1236856, 1237654, 1238142, 1238542, 1239404, 1240306, 1240701, 1241006, 1241035, 1241430, 1242280, 1242704, 1242854, 1242914, 1243407, 1243829, 1244706, 1244947, 1246717, 1247720, 1247960, 1248237, 1248930, 1249023, 1249873, 1250835, 1252912, 1252941, 1253118, 1253270, 1253751, 1254450, 1254512, 1254843, 1254994, 1255267, 1257151, 1257757, 1257877, 1258271, 1258540, 1259177, 1259793, 1259913, 1261312, 1261830, 1262137, 1262621, 1262681, 1264109, 1265273, 1265392, 1265483, 1265573, 1265722, 1265752, 1267027, 1267724, 1267970, 1268707, 1269099, 1269435, 1269585, 1270527, 1271657, 1271717, 1271995, 1272698, 1273000, 1273580, 1274034, 1274371, 1274524, 1274798, 1275166, 1276077, 1276464, 1276523, 1277001, 1277090, 1277678, 1277799, 1278072, 1278375, 1278522, 1279552, 1280039, 1280458, 1280674, 1281343, 1281497, 1281558, 1281836, 1282686, 1282873, 1282932, 1283229, 1283503, 1283746, 1283807, 1284079, 1284231, 1284718, 1284779, 1284871, 1285022, 1285084, 1285115, 1286120, 1286238, 1286605, 1286940, 1287000, 1288111, 1288772, 1288864, 1288926, 1289013, 1289102, 1289340, 1290123, 1290523, 1291010, 1291193, 1291253, 1292234, 1293078, 1293197, 1293837, 1294079, 1294199, 1294867, 1295019, 1295110, 1295322, 1295503, 1296288, 1297015, 1297491, 1298713, 1298836, 1299316, 1300134, 1300315, 1301165, 1301894, 1302595, 1303593, 1304148, 1304357, 1304386, 1305115, 1305177, 1305574, 1306278, 1306460, 1307409, 1307592, 1307978, 1309007, 1309155, 1310074, 1311095, 1311393, 1314876, 1315276, 1315488, 1315517, 1315578, 1315907, 1316031, 1316274, 1316303, 1316392, 1316695, 1316753, 1317331, 1318226, 1318709, 1318858, 1319160, 1319951, 1320440, 1320681, 1321249, 1321670, 1321817, 1322154, 1322764, 1323402, 1323432, 1324612, 1324793, 1325321, 1325410, 1325654, 1326109, 1326230, 1326684, 1327018, 1327572, 1327819, 1327908, 1328788, 1328846, 1329029, 1329489, 1329671, 1330065, 1330309, 1330396, 1330879, 1330998, 1331724, 1331812, 1331901, 1331992, 1332052, 1333105, 1333586, 1334072, 1334223, 1334799, 1335168, 1337160, 1337252, 1337672, 1337735, 1338682, 1339421, 1339754, 1341275, 1341334, 1341516, 1342129, 1342471, 1343897, 1344142, 1344693, 1345176, 1345237, 1345414, 1345811, 1346360, 1346545, 1346879, 1346939, 1347094, 1347486, 1347875, 1348208, 1348599, 1349391, 1349910, 1351400, 1351642, 1351912, 1352282, 1352616, 1352798, 1353106, 1353198, 1353654, 1354145, 1354358, 1355332, 1355510, 1355660, 1355721, 1355962, 1357175, 1357634, 1357664, 1357813, 1357872, 1358542, 1359596, 1360233, 1360843, 1363113, 1363327, 1363356, 1364151, 1365177, 1366388, 1366780, 1367179, 1367753, 1368448, 1369107, 1369323, 1369352, 1369719, 1371230, 1371352, 1372077, 1372471, 1372862, 1373106, 1373507, 1373657, 1374297, 1374450, 1374936, 1374995, 1375480, 1376058, 1376393, 1377407, 1378076, 1378624, 1378716, 1379112, 1379173, 1379473, 1379752, 1379960, 1380454, 1380700, 1381250, 1381495, 1381888, 1381917, 1382454, 1382876, 1383085, 1383326, 1384273, 1384636, 1384941, 1385428, 1385486, 1385731, 1386036, 1386065, 1386709, 1387192, 1387495, 1387953, 1388684, 1389591, 1389957, 1391911, 1392950, 1393586, 1393736, 1393794, 1394548, 1395236, 1395478, 1396070, 1396469, 1396557, 1396711, 1397730, 1398396, 1399003, 1399033, 1399398, 1399639, 1399731, 1400057, 1400613, 1401156, 1401245, 1401427, 1401638, 1402763, 1403101, 1403643, 1404528, 1404557, 1405012, 1406444, 1406786, 1406995, 1407578, 1409398, 1409997, 1410932, 1411355, 1411477, 1411506, 1411749, 1411987, 1412384, 1413163, 1413979, 1414039, 1414129, 1414310, 1415741, 1416195, 1416682, 1416868, 1417653, 1417711, 1417740, 1418314, 1418836, 1419175, 1419235, 1419507, 1419567, 1420785, 1421663, 1423029, 1423726, 1424238, 1425025, 1425355, 1427422, 1428767, 1430459, 1431106, 1431166, 1431257, 1431596, 1431717, 1431747, 1432379, 1432438, 1432690, 1432837, 1433081, 1433265, 1433410, 1433710, 1433985, 1434044, 1434073, 1434283, 1434676, 1435346, 1435558, 1435837, 1436629, 1436689, 1436753, 1437177, 1437325, 1438291, 1438619, 1438863, 1438957, 1440296, 1440925, 1441503, 1441564, 1441746, 1442435, 1442711, 1443105, 1443193, 1443345, 1443592, 1443714, 1443898, 1443958, 1443987, 1444471, 1444530, 1445418, 1445477, 1446120, 1446274, 1446937, 1448123, 1448456, 1448517, 1448608, 1448790, 1449244, 1449575, 1450456, 1450636, 1450696, 1451028, 1452215, 1453339, 1453489, 1453579, 1453728, 1454055, 1455033, 1456278, 1457327, 1457875, 1457904, 1457962, 1458351, 1458472, 1458714, 1459503, 1459804, 1459896, 1459956, 1459986, 1460139, 1460531, 1460685, 1461108, 1461196, 1461559, 1461801, 1461891, 1462695, 1462932, 1462993, 1463274, 1463485, 1463575, 1463722, 1463751, 1464301, 1464762, 1465157, 1466034, 1466859, 1467589, 1468077, 1468199, 1468382, 1468752, 1468994, 1469474, 1469504, 1469969, 1470303, 1471814, 1471904, 1472523, 1473175, 1473754, 1473815, 1474210, 1474839, 1476374, 1476676, 1477509, 1478148, 1478302, 1478604, 1478995, 1479175, 1479719, 1481426, 1482059, 1482119, 1482298, 1482790, 1482911, 1483245, 1483274, 1483517, 1483666, 1484031, 1484392, 1484601, 1485171, 1485231, 1485413, 1485722, 1485754, 1485905, 1486145, 1486692, 1487156, 1488070, 1488524, 1488767, 1489475, 1489596, 1490299, 1490446, 1490693, 1491511, 1491661, 1492393, 1492451, 1492600, 1492998, 1493152, 1493981, 1495080, 1496705, 1496768, 1497978, 1498283, 1498313, 1498709, 1499745, 1500680, 1500768, 1501161, 1501736, 1501978, 1502068, 1502609, 1503005, 1503472, 1505157, 1505186, 1505831, 1506129, 1506925, 1507016, 1507197, 1507258, 1507346, 1507800, 1508541, 1508634, 1509177, 1509479, 1509754, 1510750, 1511322, 1511661, 1511965, 1512296, 1512355, 1512688, 1513153, 1513242, 1514206, 1514935, 1514997, 1515634, 1516615, 1516952, 1517718, 1518235, 1518352, 1518932, 1519023, 1519112, 1519597, 1520477, 1520840, 1520990, 1521110, 1521349, 1521739, 1522317, 1522377, 1522436, 1523312, 1524551, 1524702, 1525554, 1525644, 1527876, 1528792, 1528918, 1529953, 1530713, 1531173, 1531233, 1531569, 1532361, 1532849, 1533093, 1533248, 1533427, 1534613, 1534759, 1535397, 1537397, 1538269, 1538357, 1538717, 1538870, 1539423, 1540548, 1541029, 1541756, 1542273, 1542637, 1543153, 1543794, 1543825, 1544834, 1544955, 1545077, 1545833, 1546048, 1546445, 1546626, 1547496, 1547645, 1547734, 1547973, 1548129, 1548865, 1549016, 1549230, 1550310, 1550556, 1550763, 1551888, 1552194, 1552439, 1553177, 1553901, 1554850, 1555875, 1555904, 1555997, 1556151, 1557029, 1557328, 1557357, 1557508, 1557991, 1558204, 1558540, 1559401, 1560458, 1560852, 1561159, 1562136, 1562527, 1562679, 1563166, 1563897, 1563987, 1564197, 1564780, 1565241, 1565913, 1566153, 1566675, 1567190, 1567343, 1567403, 1568461, 1568829, 1569314, 1569439, 1569651, 1569745, 1570750, 1571182, 1571241, 1571599, 1572236, 1573177, 1573237, 1574539, 1574690, 1577181, 1577332, 1577393, 1577637, 1577874, 1578118, 1578452, 1578634, 1580792, 1581313, 1581641, 1582275, 1583097, 1583911, 1584517, 1584694, 1584845, 1585268, 1585358, 1585418, 1586510, 1586602, 1586999, 1587272, 1587734, 1588632, 1588782, 1589630, 1590209, 1590238, 1591034, 1591340, 1591400, 1591799, 1592439, 1593195, 1593315, 1593495, 1594125, 1594217, 1594461, 1595342, 1595498, 1595894, 1595953, 1597352, 1597414, 1597472, 1597723, 1597902, 1598874, 1599392, 1600939, 1602433, 1602675, 1603346, 1603898, 1604292, 1604683, 1605009, 1605434, 1605496, 1606063, 1606609, 1606757, 1606846, 1606996, 1607027, 1607266, 1607420, 1607974, 1608760, 1609313, 1609556, 1610215, 1610638, 1611880, 1612119, 1613001, 1613426, 1613634, 1613815, 1615031, 1615154, 1615734, 1615916, 1616193, 1616833, 1617015, 1617199, 1617508, 1618034, 1618941, 1620763, 1621982, 1622289, 1622531, 1622776, 1623588, 1623803, 1624043, 1624318, 1624557, 1626195, 1626440, 1626620, 1627648, 1627894, 1628043, 1628132, 1629658, 1630449, 1630873, 1630995, 1631026, 1631321, 1631505, 1632796, 1632854, 1632946, 1633493, 1633892, 1634714, 1635346, 1635676, 1635736, 1635955, 1636046, 1636138, 1636446, 1637237, 1638124, 1639251, 1639437, 1639975, 1640037, 1640701, 1641005, 1641585, 1641643, 1641791, 1642549, 1643158, 1643248, 1643676, 1643736, 1643828, 1643886, 1644226, 1645012, 1646604, 1646633, 1647153, 1647430, 1647646, 1647919, 1648194, 1648436, 1648950, 1650296, 1650593, 1651349, 1652541, 1653237, 1653324, 1653416, 1653568, 1653872, 1654538, 1655633, 1656119, 1657002, 1657183, 1658045, 1658442, 1658833, 1659169, 1659993, 1661270, 1661796, 1662531, 1664796, 1665672, 1665820, 1666517, 1666546, 1666943, 1667556, 1667797, 1668135, 1668290, 1668867, 1669169, 1669990, 1670296, 1671005, 1671094, 1671274, 1671574, 1672358, 1672541, 1672784, 1672844, 1673024, 1673791, 1674671, 1675163, 1675554, 1675583, 1676440, 1676471, 1676558, 1676617, 1676921, 1678034, 1678274, 1678917, 1679834, 1680351, 1680870, 1681024, 1681812, 1682229, 1682437, 1682957, 1683637, 1684223, 1685349, 1685500, 1686472, 1686627, 1687179, 1687237, 1687266, 1687508, 1687967, 1689678, 1689801, 1689986, 1690386, 1690479, 1690602, 1690631, 1690873, 1691024, 1691485, 1691514, 1691968, 1692116, 1692206, 1692600, 1692781, 1693805, 1693958, 1693987, 1694845, 1694993, 1695085, 1695239, 1695974, 1696270, 1696451, 1696633, 1696909, 1697177, 1697238, 1698958, 1699169, 1699895, 1699986, 1700045, 1701480, 1701663, 1701966, 1702366, 1703097, 1703798, 1703979, 1704556, 1704765, 1705009, 1705191, 1706221, 1706314, 1706525, 1707074, 1707163, 1707316, 1707590, 1707991, 1708530, 1708619, 1708708, 1708919, 1709253, 1709794, 1709883, 1709912, 1710398, 1711313, 1711496, 1711650, 1711897, 1712541, 1712600, 1714156, 1714455, 1715272, 1715425, 1715906, 1716056, 1717022, 1717508, 1718054, 1718115, 1718146, 1718871, 1719478, 1719720, 1719751, 1720057, 1720845, 1721507, 1722203, 1722297, 1722872, 1723565, 1723897, 1724200, 1724229, 1725393, 1725579, 1725726, 1725817, 1725876, 1726760, 1726850, 1727097, 1727339, 1727431, 1727735, 1727826, 1728313, 1728702, 1728853, 1729030, 1729119, 1729332, 1729577, 1729665, 1729873, 1730059, 1730368, 1730457, 1730518, 1730941, 1731276, 1731397, 1731645, 1732678, 1733081, 1733573, 1733663, 1734453, 1734753, 1735262, 1735324, 1736381, 1736776, 1737191, 1737336, 1737667, 1737821, 1739585, 1739643, 1740626, 1740930, 1740990, 1741325, 1741566, 1741993, 1742606, 1742846, 1743333, 1743666, 1744520, 1744671, 1744703, 1744792, 1744943, 1745747, 1745993, 1746295, 1746599, 1746933, 1746991, 1747722, 1747870, 1748360, 1748545, 1748603, 1748696, 1749028, 1749117, 1749750, 1749960, 1751007, 1751337, 1751516, 1751635, 1751726, 1751911, 1752518, 1752693, 1753666, 1754238, 1754602, 1755088, 1755484, 1755575, 1755789, 1755971, 1756121, 1756152, 1756452, 1756875, 1756933, 1758371, 1759165, 1759254, 1759348, 1760077, 1761656, 1761906, 1762059, 1762612, 1763071, 1763956, 1764139, 1764441, 1764532, 1764861, 1765077, 1765198, 1765409, 1765678, 1766551, 1767006, 1768622, 1768680, 1769105, 1769194, 1770364, 1770393, 1771312, 1771341, 1771497, 1771731, 1771974, 1772220, 1772430, 1772639, 1773157, 1773581, 1773640, 1773728, 1773757, 1773818, 1774061, 1774869, 1775020, 1775111, 1775957, 1776079, 1776139, 1776777, 1778357, 1778476, 1779270, 1779572, 1779754, 1780615, 1780918, 1781340, 1781585, 1782466, 1783916, 1784519, 1785577, 1786373, 1786460, 1786912, 1787398, 1787829, 1788550, 1788699, 1789117, 1789419, 1789832, 1789894, 1790447, 1790476, 1790870, 1791023, 1791513, 1792790, 1793918, 1794378, 1794622, 1795725, 1795754, 1795873, 1795993, 1796513, 1796634, 1796935, 1797355, 1797905, 1798541, 1798635, 1798696, 1799663, 1800599, 1800931, 1801110, 1801359, 1801571, 1802701, 1803400, 1804458, 1805186, 1806127, 1806947, 1808624, 1809873, 1809963, 1810051, 1810354, 1810445, 1810781, 1811089, 1811239, 1811998, 1812758, 1813678, 1813950, 1814286, 1814624, 1814713, 1815356, 1816360, 1817494, 1818070, 1818217, 1819259, 1819553, 1820834, 1821985, 1822847, 1822876, 1823508, 1823599, 1824544, 1824999, 1825510, 1827021, 1827231, 1827656, 1827897, 1827956, 1828591, 1829010, 1829161, 1829439, 1831470, 1831561, 1831710, 1831798, 1832196, 1832468, 1832620, 1832829, 1833196, 1833745, 1833986, 1834135, 1834559, 1834952, 1835716, 1836054, 1836115, 1836144, 1836865, 1837411, 1838225, 1839662, 1839908, 1839967, 1840453, 1840791, 1841815, 1842123, 1842154, 1843007, 1843190, 1843582, 1843917, 1844040, 1844193, 1844314, 1844557, 1845102, 1845977, 1846128, 1846465, 1846676, 1846707, 1846858, 1847165, 1847589, 1847710, 1848594, 1849718, 1850055, 1850144, 1850547, 1850607, 1851181, 1851333, 1851575, 1851724, 1851912, 1852768, 1853105, 1853257, 1853747, 1853985, 1854196, 1854225, 1854773, 1855014, 1855074, 1855162, 1855494, 1855744, 1855977, 1856522, 1856639, 1856698, 1856760, 1856942, 1857001, 1857274, 1857817, 1858280, 1858612, 1859982, 1860436, 1860558, 1861105, 1861258, 1861658, 1862046, 1862075, 1862194, 1862317, 1862526, 1863496, 1863554, 1863739, 1864279, 1864951, 1865073, 1865436, 1865978, 1866065, 1866124, 1866761, 1867638, 1868672, 1868943, 1869240, 1869327, 1869756, 1870556, 1870916, 1871679, 1871738, 1872193, 1872316, 1873824, 1873976, 1874855, 1874915, 1874946, 1875005, 1875249, 1875403, 1875980, 1876133, 1876523, 1876552, 1876771, 1877432, 1878070, 1878524, 1879194, 1879317, 1879711, 1880077, 1880231, 1881355, 1881565, 1881750, 1882302, 1882997, 1883151, 1884034, 1884124, 1884365, 1884395, 1884789, 1885176, 1885234, 1885417, 1885569, 1887271, 1887917, 1887978, 1889107, 1889165, 1889195, 1889348, 1889502, 1889833, 1889984, 1890865, 1891417, 1891480, 1891662, 1891722, 1891751, 1892360, 1892452, 1892997, 1893424, 1893983, 1894620, 1895352, 1896205, 1896754, 1897031, 1897636, 1897726, 1898389, 1898754, 1898935, 1899027, 1899086, 1899753, 1900156, 1900214, 1900853, 1901738, 1902075, 1902595, 1903107, 1904440, 1905739, 1905831, 1905919, 1905977, 1906368, 1906459, 1906851, 1907398, 1907552, 1908228, 1908778, 1909324, 1911829, 1912377, 1912626, 1912687, 1913997, 1914752, 1915116, 1915571, 1915661, 1915961, 1916601, 1916943, 1917033, 1917094, 1917341, 1918307, 1918548, 1918879, 1919733, 1919823, 1920126, 1920277, 1920306, 1920462, 1920858, 1921155, 1921832, 1922622, 1922930, 1923471, 1924866, 1925715, 1926476, 1926595, 1926932, 1927480, 1927662, 1927994, 1928543, 1929330, 1930466, 1931098, 1931674, 1932282, 1932770, 1932947, 1933038, 1933676, 1934128, 1934310, 1934698, 1935397, 1935490, 1935675, 1936074, 1936370, 1936462, 1936794, 1937000, 1937417, 1937659, 1937965, 1938235, 1938541, 1938843, 1939022, 1939114, 1939263, 1939351, 1939651, 1939714, 1939836, 1940071, 1940676, 1941013, 1941346, 1941672, 1942684, 1942775, 1943109, 1943652, 1943712, 1944203, 1944447, 1944542, 1944633, 1944936, 1945178, 1945484, 1945882, 1945911, 1946547, 1948471, 1948684, 1949479, 1949571, 1950125, 1950306, 1950365, 1950702, 1951188, 1951401, 1951492, 1951583, 1952033, 1952124, 1952461, 1952704, 1952763, 1953405, 1953987, 1954441, 1954470, 1954772, 1955475, 1955658, 1955751, 1955873, 1955905, 1956207, 1956847, 1957635, 1958149, 1958208, 1958300, 1959335, 1959967, 1960057, 1960302, 1960605, 1961711, 1962048, 1962110, 1962230, 1962289, 1962319, 1963561, 1963650, 1963805, 1964790, 1965101, 1966221, 1966469, 1966713, 1967256, 1967650, 1967679, 1968380, 1969198, 1969593, 1969963, 1970359, 1970479, 1970697, 1970849, 1970878, 1970941, 1971340, 1971430, 1971916, 1972067, 1972219, 1972399, 1973105, 1973408, 1974207, 1974691, 1976151, 1976271, 1976514, 1977034, 1977275, 1977396, 1978217, 1978614, 1978676, 1978705, 1978765, 1980158, 1981164, 1982865, 1983105, 1983409, 1983964, 1984146, 1984236, 1985424, 1986130, 1986556, 1987012, 1987653, 1987747, 1988709, 1988948, 1989334, 1989424, 1989515, 1989756, 1989816, 1990238, 1990877, 1991157, 1991433, 1991824, 1992379, 1993106, 1993501, 1993593, 1993713, 1993992, 1994297, 1994633, 1994782, 1994871, 1995083, 1995174, 1995422, 1996146, 1996599, 1996688, 1996994, 1997801, 1998309, 1998369, 1998548, 1998758, 1999897, 2000204, 2000538, 2000866, 2001078, 2001438, 2001496, 2001894, 2002287, 2002441, 2003113, 2003323, 2003717, 2003901, 2005909, 2006845, 2006995, 2007180, 2007334, 2007668, 2008066, 2008955, 2009407, 2010139, 2010233, 2010450, 2010538, 2010752, 2010934, 2011327, 2011727, 2012468, 2013159, 2013340, 2013400, 2013429, 2013643, 2013914, 2014309, 2014460, 2014552, 2014768, 2015013, 2015167, 2016136, 2016433, 2016707, 2017015, 2017653, 2017917, 2017980, 2018316, 2018374, 2018678, 2019251, 2019551, 2019638, 2020511, 2020930, 2020959, 2021079, 2021351, 2021994, 2022514, 2022605, 2023028, 2023150, 2023336, 2023489, 2024699, 2025735, 2025793, 2026070, 2026311, 2027958, 2028139, 2028619, 2028710, 2029489, 2029821, 2030772, 2030833, 2031078, 2031107, 2031631, 2031718, 2031869, 2031991, 2032477, 2032842, 2033328, 2034213, 2034697, 2034756, 2034999, 2035553, 2035820, 2035910, 2037279, 2039428, 2039973, 2040208, 2040785, 2041399, 2041429, 2041951, 2043012, 2043561, 2044633, 2045487, 2046062, 2047189, 2047517, 2048297, 2049114, 2049871, 2051191, 2051314, 2051585, 2051952, 2052076, 2052717, 2052872, 2053174, 2053473, 2053892, 2053953, 2053982, 2054676, 2054860, 2055251, 2055493, 2055884, 2056039, 2056130, 2056518, 2057036, 2057403, 2057795, 2057977, 2058068, 2059092, 2060768, 2061344] not in index'
