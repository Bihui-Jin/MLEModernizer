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

0.3755122186933007

# 6. Current score

3.39885

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.63464) has done: 'Your notebook fails because it tries to read out-of-environment blend files (`../input/gb-blending/...`) that don’t exist, so it never produces any submission. I remove that dependency and instead generate a simple, valid baseline submission directly from the provided competition data. To keep the core approach “prediction without training” minimal and robust, I predict a constant pressure equal to the mean inspiratory-phase pressure from the training set (matching the metric focus on inspiration via `u_out==0`). Finally, I write `submission.csv` with exactly `id,pressure` so Kaggle accepts it.'
- What this solution (achieved 6.19013) has done: 'Your current score (7.63464 MAE) is far above the target (0.3755), so we need a real predictive model rather than a constant baseline. To keep changes minimal while materially improving MAE, I preserve the “no deep learning / no custom training loop” spirit by adding a lightweight, deterministic, per-(R,C) lookup-table regressor: for each time step within a breath, predict the mean inspiratory pressure from training for that same (R,C,time_step_idx). This aligns directly with the competition metric (only `u_out==0` is scored) and is fast to compute with pandas groupby. Finally, I keep the submission writing logic intact and ensure `id,pressure` matches the test set ordering.'
- What this solution (achieved 4.12219) has done: 'Your current MAE (6.19013) is far worse than the target (0.3755, lower-is-better), so we need a meaningful but still lightweight improvement without changing the overall “pandas lookup-table regressor” core logic. The biggest issue with the current lookup is that it ignores the control input `u_in`, which strongly drives pressure during inspiration; adding `u_in` (binned) into the same group-mean lookup usually yields a large MAE drop while keeping the exact same non-ML approach. To stay robust and fast, we compute mean inspiratory pressure by `(R, C, time_step_idx, u_in_bin)` with fallbacks to `(R, C, time_step_idx)` then `(R, C)` then global inspiratory mean. This preserves evaluation semantics (inspiration-focused) and still writes a valid `submission.csv`.'
- What this solution (achieved 4.12219) has done: 'Your current lookup-table regressor is close in spirit to what works for this competition, but the big scoring gap suggests a subtle alignment bug rather than needing a new model. The `id` ranges you printed (1–2000) indicate you likely loaded the wrong/partial CSVs (or a truncated file) from a different folder, so the learned means and the submission alignment are invalid for the real test set. I make the data loading robust by selecting the correct existing dataset path (`/kaggle/data/...` fallback to `/kaggle/input/...`) and add strict sanity checks that `train`/`test` row counts match the known competition sizes and that `id` ranges are full, then rerun the exact same feature engineering and lookup logic. This is a minimal change that should move MAE sharply toward your target without changing the core approach.'
- What this solution (achieved 2.14279) has done: 'Your current MAE is still far above target, and the biggest remaining “minimal change” lever within your lookup-table approach is to better represent system state without changing the modeling paradigm. I keep the same group-mean lookup + fallbacks, but add two lightweight state features that are known to matter for this problem: cumulative inspired volume proxy (`u_in` integral) and lagged `u_in` (previous timestep), both binned like your existing `u_in_bin`. This stays as pure pandas aggregation/merge logic (no new model/training loop) while typically moving MAE substantially toward your target. I also make `u_out` available in test so we can optionally force expiratory predictions to a stable baseline (doesn’t affect metric but can avoid weird outputs) without changing inspiratory scoring behavior.'
- What this solution (achieved 3.15057) has done: 'Your current MAE (2.14279, lower-is-better) is still far above the target (0.3755), so we should improve the existing lookup-table regressor without changing its core “groupby mean + fallbacks” logic. The biggest remaining gap is that the mapping from controls to pressure is highly *continuous* and your current bin widths (especially 0.5) are too coarse, causing systematic bias; tightening the bins makes the same lookup substantially more specific while keeping identical semantics. I reduce `UIN_BIN_WIDTH` and `UIN_LAG_BIN_WIDTH` to 0.1 and `UIN_CUM_BIN_WIDTH` to 0.25 (still fast, still pure pandas), and I add a tiny, safe post-processing step that snaps predictions to the discrete set of pressures seen in training (a well-known property of this competition), which usually reduces MAE without changing the model type. All paths and submission-writing logic remain the same, and it still produces a valid `submission.csv`.'
- What this solution (achieved 5.7263) has done: 'Your current score (3.15057 MAE, lower-is-better) is still far from the target (0.3755), so we should make a small but meaningful improvement while keeping your exact “groupby mean lookup + fallbacks” core logic. The biggest minimally-invasive gain is to stop using a *binned* `u_in` (and binned lag/cum) and instead use the *exact* continuous values already present in the data; in this competition, `u_in` and `time_step` are generated on a fixed grid, so exact matching between train and test is common and typically reduces bias from binning. Concretely: keep all the same features and merges, but replace `u_in_bin/u_in_lag_bin/u_in_cum_bin` with quantized-to-original-resolution integers (e.g., `u_in*1000`, `u_in_lag*1000`, `u_in_cum*1000`) to act as stable join keys without changing the model type. We keep the same snapping-to-pressure-grid postprocess and the same submission-writing checks/paths.'
- What this solution (achieved 3.15057) has done: 'Your current score got worse after switching from binning to exact-key matching, which likely caused severe sparsity (exact `(u_in, u_in_lag, u_in_cum)` triples rarely repeat), pushing many rows into coarse fallbacks and increasing MAE. To move back toward the target with minimal change and the same “groupby mean lookup + fallbacks” core logic, I reintroduce *light* binning just for the join keys (not a new model), keeping your same features, merges, and pressure-grid snapping. I also make the cumulative key bin slightly coarser than `u_in` to stabilize state matching across breaths while preserving your evaluation semantics (inspiratory focus, expiratory set to baseline). This should reduce fallback usage and improve MAE toward the target without changing the overall approach.'
- What this solution (achieved 3.04143) has done: 'Your current MAE (3.15057, lower-is-better) is far above the target (0.3755), so we should improve within your existing “pandas groupby mean lookup + fallbacks + snap-to-grid” core logic. The main issue is that your most-specific table uses a 6D key that is still too sparse, causing many rows to fall back to much coarser averages; we can reduce sparsity with a minimal, metric-aligned tweak: use a *coarser* bin just for the cumulative-state key (keep `u_in`/lag bins unchanged). Additionally, instead of a hard snap-to-grid, we can do a tiny regularization that blends the raw prediction with its snapped value (keeps the same semantics, often lowers MAE when group means are noisy). These are minimal changes that keep the same data, features, merges, and fallback structure, while typically moving MAE materially downward toward your target.'
- What this solution (achieved 3.04146) has done: 'Your current MAE (3.041) is still far above the target (0.375, lower-is-better), so we should improve accuracy while keeping the exact same “pandas groupby mean lookup + fallbacks + pressure-grid snap” core logic. The biggest minimal win is to reduce key mismatch caused by float rounding: instead of float “bin keys”, we store all binned keys as small integers (scaled by bin width) in both train and test, so merges match exactly and fallbacks are used less often. We also compute `dt` from `time_step` with a per-breath stable shift (rather than groupby diff) to avoid tiny float artifacts, while keeping the same cumulative definition. Finally, we keep your SNAP_BLEND mechanism unchanged and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.8044) has done: 'Your current MAE (3.041) is still far above the target (0.375, lower-is-better), so we should improve within the same “pandas groupby-mean lookup + fallbacks + snap-to-grid” approach. The smallest change with a high chance of reducing MAE is to reduce sparsity in the most-specific table by dropping the cumulative-state key from that table only (keep cum key computed for compatibility, but don’t require it for the tightest join). In exchange, we slightly strengthen the conditioning where it matters by adding `u_out` into the fallback tables (it’s already in data; this is still just groupby means and keeps evaluation semantics). Everything else (paths, feature engineering, fallbacks structure, snapping/blend, and submission writing) stays the same.'
- What this solution (achieved 3.8044) has done: 'Your current MAE (3.8044, lower-is-better) is still far above the target (0.3755), so we should improve accuracy with the smallest possible change while preserving the same “groupby-mean lookup + fallbacks + snap-to-grid” core logic. The most likely cause of the regression is that we are (a) forcing *all* `u_out==1` predictions to a global mean (even though some inspiratory scoring rows can still be affected by earlier-state leakage) and (b) using mixed tables where the most-specific table is inspiratory-only but the fallbacks are built on all phases, which can bias means. I keep the exact same feature engineering and merge/fallback structure, but make the fallback tables *inspiration-only* (u_out==0) to match the evaluation, and remove the hard override for `u_out==1` (it’s not scored, and overriding can only hurt sequence realism indirectly without any upside). Finally, I keep the same snapping/blend postprocess and the same submission validation to ensure a valid `submission.csv` is written.'
- What this solution (achieved 3.72441) has done: 'Your current MAE (3.8044, lower-is-better) is far above the target (0.3755), so we need a small change that increases match-rate of the most-specific lookup without changing the core “groupby mean lookup + fallbacks + snap-to-grid” approach. The most likely issue is key-mismatch/sparsity from `u_in_lag_key_i` at 0.1 resolution; many rows miss the tight table and fall back to coarser means. I keep your exact feature set and fallback structure, but slightly coarsen only the lag bin (to 0.2) to reduce sparsity while leaving `u_in` binning unchanged, which should reduce fallbacks and lower MAE. Everything else (paths, inspiration-only tables, snapping/blend, and writing `submission.csv` with `id,pressure`) stays the same.'
- What this solution (achieved 3.39885) has done: 'Your current MAE (3.724) is still far above the target (0.376, lower-is-better), so we should improve accuracy while keeping the exact same “pandas groupby mean lookup + fallbacks + snap-to-grid” core logic. The smallest high-impact change here is to reduce sparsity in the most-specific table by using the *current* `u_in` and a *short history summary* (mean of the last 3 `u_in` values) instead of the single lag-1 key, which mismatches often and forces fallbacks. This keeps the same aggregation/merge paradigm and uses only existing signals, but typically increases hit-rate on the tight table and lowers MAE. Everything else (inspiration-only tables, fallback order, snapping/blend, and submission writing) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




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
def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.75 + b.pressure * 0.25
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
def resolve_data_dir():
    candidates = [
        "/kaggle/data/ventilator-pressure-prediction",
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data",
        "/kaggle/input",
    ]
    for base in candidates:
        tr = os.path.join(base, "train.csv")
        te = os.path.join(base, "test.csv")
        ss = os.path.join(base, "sample_submission.csv")
        if os.path.exists(tr) and os.path.exists(te) and os.path.exists(ss):
            return base
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/sample_submission.csv in expected Kaggle paths."
    )


DATA_DIR = resolve_data_dir()
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

UIN_BIN_WIDTH = 0.1
UIN_HIST_BIN_WIDTH = 0.2

UIN_CUM_BIN_WIDTH = (
    0.5  # keep computed for compatibility (not required in tightest join)
)
SNAP_BLEND = 0.85  # keep current snap blend

UIN_SCALE = int(round(1.0 / UIN_BIN_WIDTH))  # 10 for 0.1
UIN_HIST_SCALE = int(round(1.0 / UIN_HIST_BIN_WIDTH))  # 5 for 0.2
UIN_CUM_SCALE = int(round(1.0 / UIN_CUM_BIN_WIDTH))  # 2 for 0.5

usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
train = pd.read_csv(train_path, usecols=usecols_train)

if len(train) < 5_000_000:
    raise ValueError(
        f"train.csv seems too small ({len(train)} rows). Check DATA_DIR={DATA_DIR} and file integrity."
    )

train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
train["time_step_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)

u_in_f = train["u_in"].to_numpy(dtype=np.float32)
train["u_in_key_i"] = np.rint(u_in_f * UIN_SCALE).astype(np.int16)

u_in_hist = (
    train.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
train["u_in_hist_key_i"] = np.rint(
    u_in_hist.to_numpy(dtype=np.float32) * UIN_HIST_SCALE
).astype(np.int16)

t = train["time_step"].to_numpy(dtype=np.float32)
t_prev = (
    train.groupby("breath_id")["time_step"]
    .shift(1)
    .fillna(train["time_step"])
    .to_numpy(dtype=np.float32)
)
dt = (t - t_prev).astype(np.float32)
train["u_in_cum"] = (train["u_in"].to_numpy(dtype=np.float32) * dt).astype(np.float32)
train["u_in_cum"] = (
    pd.Series(train["u_in_cum"])
    .groupby(train["breath_id"])
    .cumsum()
    .to_numpy(dtype=np.float32)
)
train["u_in_cum_key_i"] = np.rint(
    train["u_in_cum"].to_numpy(dtype=np.float32) * UIN_CUM_SCALE
).astype(np.int32)

insp_train = train[train["u_out"] == 0].copy()

global_mean_insp = (
    float(insp_train["pressure"].mean())
    if len(insp_train)
    else float(train["pressure"].mean())
)

rc_t_u_hist_means = (
    insp_train.groupby(
        ["R", "C", "time_step_idx", "u_in_key_i", "u_in_hist_key_i"],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_t_u_hist"})
)

rc_t_u_means = (
    insp_train.groupby(["R", "C", "time_step_idx", "u_in_key_i"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_t_u"})
)

rc_t_means = (
    insp_train.groupby(["R", "C", "time_step_idx"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_t"})
)

rc_means = (
    insp_train.groupby(["R", "C"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc"})
)

pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)

usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
test = pd.read_csv(test_path, usecols=usecols_test)

if len(test) < 500_000:
    raise ValueError(
        f"test.csv seems too small ({len(test)} rows). Check DATA_DIR={DATA_DIR} and file integrity."
    )
if test["id"].min() != 0 and test["id"].min() != 1:
    raise ValueError(
        f"Unexpected id min={test['id'].min()} in test.csv; likely wrong file."
    )
if test["id"].nunique() != len(test):
    raise ValueError(
        "test.csv id is not unique per row; likely wrong file or corrupted read."
    )

test = test.sort_values(["breath_id", "time_step"], kind="mergesort")
test["time_step_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

u_in_te = test["u_in"].to_numpy(dtype=np.float32)
test["u_in_key_i"] = np.rint(u_in_te * UIN_SCALE).astype(np.int16)

u_in_hist_te = (
    test.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
test["u_in_hist_key_i"] = np.rint(
    u_in_hist_te.to_numpy(dtype=np.float32) * UIN_HIST_SCALE
).astype(np.int16)

t_te = test["time_step"].to_numpy(dtype=np.float32)
t_prev_te = (
    test.groupby("breath_id")["time_step"]
    .shift(1)
    .fillna(test["time_step"])
    .to_numpy(dtype=np.float32)
)
dt_te = (t_te - t_prev_te).astype(np.float32)
u_in_cum_te = (test["u_in"].to_numpy(dtype=np.float32) * dt_te).astype(np.float32)
u_in_cum_te = (
    pd.Series(u_in_cum_te)
    .groupby(test["breath_id"])
    .cumsum()
    .to_numpy(dtype=np.float32)
)
test["u_in_cum"] = u_in_cum_te
test["u_in_cum_key_i"] = np.rint(u_in_cum_te * UIN_CUM_SCALE).astype(np.int32)

pred = test.merge(
    rc_t_u_hist_means,
    on=["R", "C", "time_step_idx", "u_in_key_i", "u_in_hist_key_i"],
    how="left",
)
pred = pred.merge(
    rc_t_u_means,
    on=["R", "C", "time_step_idx", "u_in_key_i"],
    how="left",
)
pred = pred.merge(
    rc_t_means,
    on=["R", "C", "time_step_idx"],
    how="left",
)
pred = pred.merge(
    rc_means,
    on=["R", "C"],
    how="left",
)

pred["pressure"] = (
    pred["pressure_rc_t_u_hist"]
    .fillna(pred["pressure_rc_t_u"])
    .fillna(pred["pressure_rc_t"])
    .fillna(pred["pressure_rc"])
    .fillna(global_mean_insp)
)

p_raw = pred["pressure"].to_numpy(dtype=np.float32)
idx = np.searchsorted(pressure_grid, p_raw, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx0 = np.clip(idx - 1, 0, len(pressure_grid) - 1)
choose_left = np.abs(p_raw - pressure_grid[idx0]) <= np.abs(p_raw - pressure_grid[idx])
snapped = np.where(choose_left, pressure_grid[idx0], pressure_grid[idx]).astype(
    np.float32
)

pred["pressure"] = (SNAP_BLEND * snapped + (1.0 - SNAP_BLEND) * p_raw).astype(
    np.float32
)

sub = pred[["id", "pressure"]].sort_values("id")

sample = pd.read_csv(sample_path, usecols=["id"])
if len(sub) != len(sample):
    raise ValueError(
        f"Submission rowcount {len(sub)} != sample rowcount {len(sample)}; refusing to write invalid submission."
    )
sub = sample.merge(sub, on="id", how="left")
if sub["pressure"].isna().any():
    raise ValueError(
        "Some ids in sample_submission.csv did not get a prediction; check merge keys and feature engineering."
    )

sub.to_csv("submission.csv", index=False)

print("DATA_DIR:", DATA_DIR)
print("Wrote submission.csv")
print("UIN_BIN_WIDTH:", UIN_BIN_WIDTH)
print("UIN_HIST_BIN_WIDTH:", UIN_HIST_BIN_WIDTH)
print("UIN_CUM_BIN_WIDTH:", UIN_CUM_BIN_WIDTH)
print("SNAP_BLEND:", SNAP_BLEND)
print("global_mean_insp:", global_mean_insp)
print("pressure_grid_size:", len(pressure_grid))
print(sub.head())
print(sub.shape)
