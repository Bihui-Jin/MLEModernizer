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

0.2638215552274246

# 6. Current score

2.91477

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.14624) has done: 'I fix the runtime failure by removing the hardcoded dependency on missing `../input/gb-blending/*.csv` files and instead generate a valid submission directly from the provided competition files. Since your current code has no actual model and only blends external submissions, the safest minimal “score-improving” step (vs. all-zeros) is to create a deterministic baseline prediction from training data: per-(R,C,time_step) mean pressure, with a global fallback. This keeps the core intent (simple blending/aggregation) while ensuring the notebook runs end-to-end and writes a proper `submission.csv` with `id,pressure`. I also keep your helper functions but make the final cell robust to missing blend inputs.'
- What this solution (achieved 6.10605) has done: 'Your current baseline is good but it’s losing a lot of accuracy because `time_step` is a float and grouping/joining on exact float values can miss matches due to tiny representation differences, forcing many rows to fall back to the global mean (which inflates MAE). I keep the exact same “per-(R,C,time_step) mean pressure with global fallback” core logic, but make it robust by rounding `time_step` to a fixed precision before grouping and lookup so train/test keys align. I also switch the slow Python `zip/get` loop to a vectorized merge (same semantics) to reduce runtime risk within Kaggle limits and ensure the submission row order matches `test.csv` by `id`. These minimal changes should move the score substantially toward your target without changing the modeling approach.'
- What this solution (achieved 3.90146) has done: 'Your current baseline is still far from the target because it ignores two key control inputs (`u_in`, `u_out`) and also averages over all phases, while the metric only scores inspiratory phase (effectively when `u_out==0`). Keeping the same “lookup mean from train and fall back to a global mean” core logic, we (1) compute means using only inspiratory rows (`u_out==0`) and (2) condition the lookup on a lightly-binned `u_in` in addition to `(R,C,time_step)` so test rows match more specific situations without changing the approach. We also apply the standard competition post-processing of forcing predictions to 0 during expiration (`u_out==1`), which aligns with the evaluation semantics and typically reduces MAE. These are minimal, deterministic changes that should move the score substantially toward your target while still being a simple aggregation baseline.'
- What this solution (achieved 3.89833) has done: 'Your current gap to the target is large (3.90 vs 0.264; lower is better), so we should improve accuracy while keeping the same “lookup mean from train + fallback mean” core logic. The biggest remaining issue is that your lookup key is too sparse: exact `(R,C,time_step,u_in_bin)` matches still miss often, causing many fallbacks to a global mean; we can reduce that by adding a hierarchical backoff (progressively less specific group means) without changing the approach. We also align better with the metric by computing means on inspiratory rows only (keep) and by using an inspiratory fallback that is conditioned on `(R,C,time_step)` and `(R,C,time_step,u_in_bin)` before the global mean. Finally, we keep the standard post-processing of forcing predictions to 0 for `u_out==1`.'
- What this solution (achieved 3.95011) has done: 'Your current approach is a deterministic “group-mean lookup + hierarchical fallback”, but it’s still far from the target largely because the `(time_step, u_in_bin)` key is too strict for noisy/continuous `u_in`, causing many fallbacks and higher MAE. Keeping the exact same core logic, I (1) make `u_in` binning slightly coarser (2.0 instead of 1.0) to increase exact-match coverage, and (2) add a very small, still-aggregation-based backoff that uses a `(R,C,time_step_r, u_out)` mean (inspiratory matter; expiratory gets overwritten to 0 anyway) before falling back to broader keys. This should reduce reliance on the global mean while preserving your merge-based pipeline and the “force 0 on u_out==1” post-processing aligned with the metric. All paths and submission formatting stay unchanged, and it still run comfortably within the time limit.'
- What this solution (achieved 4.05787) has done: 'Your current score is much worse than the target (lower-is-better), so we should improve accuracy with minimal changes while keeping your same “group-mean lookup + hierarchical fallback + force 0 on u_out==1” core logic. The biggest remaining issue is still match coverage: even with coarse `u_in` bins, exact `(R,C,time_step_r,u_in_bin)` can miss, so we add one more aggregation-based backoff using a *rolling time-step* mean per `(R,C,u_in_bin)` to interpolate across nearby `time_step_r` values (still deterministic, still just means). We also make `u_in` binning slightly less strict by using `floor` instead of `round` so neighboring values don’t flip bins around boundaries, improving stability/match rate without changing the approach. These changes are aimed specifically at reducing fallback-to-global-mean frequency, which should move MAE toward your target.'
- What this solution (achieved 3.9305) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy while keeping the same deterministic “group-mean lookup + hierarchical fallback + force 0 when u_out==1” approach. The biggest remaining easy win is to align the aggregation with the competition’s *inspiratory-only* scoring by using inspiratory-only statistics **and** a more realistic expiration handling: instead of forcing 0 for `u_out==1`, predict a stable low-pressure baseline learned from the data (still just a mean lookup), which better matches typical expiratory pressures. Additionally, make `u_in` binning a bit finer (1.0) to reduce bias while keeping the same lookup mechanism, and add a very small extra backoff keyed on `(R,C,u_in_bin)` plus `time_step_r` (already present) but applied earlier to reduce global fallbacks. These are minimal, purely aggregation-based changes and should move MAE materially toward your target.'
- What this solution (achieved 1.72448) has done: 'Your current score (3.9305, lower-is-better) is far from the target (0.2638), so we should improve accuracy with minimal, still-aggregation-based changes. The largest likely error driver is that the lookup ignores breath dynamics: pressure depends heavily on accumulated volume/flow history, so adding simple cumulative features (cumulative u_in and a lag of u_in) to the grouping keys can materially reduce MAE without changing the “group-mean lookup + hierarchical fallback” core logic. We also keep inspiratory-only fitting and the expiratory prediction path, but make the expiratory baseline depend on time_step and u_out (already) and add a small extra backoff using the new cumulative features. All changes remain deterministic, fast (single-pass groupby/merge), and still produce a valid `submission.csv`.'
- What this solution (achieved 2.89822) has done: 'Your current aggregation baseline is losing accuracy because the “history” bins (`u_in_cum_bin`, `u_in_lag1_bin`) are computed on raw `u_in` (which can fluctuate) and the time alignment uses only `time_step`; we can improve match quality with a minimal, still-deterministic change: build the same features on `dt` and *integrated flow* (`u_in * dt`) to better approximate inspired volume. We keep the exact same group-mean lookup + hierarchical fallback structure, but swap `u_in_cum` for a physically closer `vol_cum` and compute lag on `u_in` as before; this tends to reduce reliance on broad fallbacks and lowers MAE without changing the “mean-table + merge” approach. We also add one extra, very cheap backoff mean keyed on `(R,C,time_step_r, vol_cum_bin)` between the most-specific and the simpler keys to further reduce global fallback frequency. All paths, submission schema, and deterministic behavior are preserved, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 3.12641) has done: 'Your current score (2.89822, lower-is-better) is still far from the target (0.26382), so we should improve accuracy while preserving your exact “deterministic group-mean lookup + hierarchical fallback” approach. The biggest remaining low-risk gain is to reduce fallback frequency by making `vol_cum_bin` and `u_in_lag1_bin` less sparse (more matches) and to add one extra intermediate backoff keyed on both `time_step_r` and `u_in_lag1_bin` (history) without changing the overall logic. These are purely aggregation/merge-table changes (same semantics), and should move MAE closer to the target by producing more specific, better-calibrated lookups. All file paths, evaluation semantics (inspiratory-only fitting and separate expiratory handling), and submission formatting remain unchanged.'
- What this solution (achieved 2.70698) has done: 'We keep your deterministic “group-mean lookup + hierarchical fallback” pipeline exactly as-is, but tighten two small knobs that mainly affect match coverage (and therefore MAE): (1) use a slightly finer `vol_cum_bin` so the history feature is less over-smoothed, and (2) use a slightly finer `u_in_lag1_bin` so the lag signal is less quantized. These are minimal, safe changes that don’t alter the modeling approach, training loop, or loss semantics—just the aggregation granularity—so they should improve accuracy from your current 3.126 MAE toward the target. Everything else (inspiratory-only fitting, expiratory handling, merge order, fallbacks, and submission format) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 2.91477) has done: 'Your current MAE (2.70698, lower-is-better) is still far above the target (0.2638), so we should make small, low-risk changes that improve lookup match quality without changing the core “group-mean lookup + hierarchical fallback” approach. The biggest win with minimal disruption is to (1) use slightly finer rounding for `time_step` so train/test keys align more precisely, and (2) make the `vol_cum_bin` / `u_in_lag1_bin` bins a bit coarser to reduce sparsity (fewer fallbacks to broad means). I keep the exact same feature set, same group-mean tables, same merge/fallback order, and the same expiratory-vs-inspiratory handling; we only tweak these discretization knobs to reduce mismatch/fallback frequency. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.91477) has done: 'We keep your exact deterministic “group-mean lookup + hierarchical fallback” approach, but fix a key mismatch that currently causes many unnecessary fallbacks: the binning of `u_in` and `u_in_lag1` can produce values above the train range in test (notably when `u_in==100`), creating unseen keys. By clipping those binned features to the training max bin (and min) we increase exact-match coverage without changing the modeling logic. This should reduce reliance on broad means and move MAE down toward your target while preserving runtime and submission semantics. No architecture/training changes are introduced; this is purely a safe discretization alignment fix.'

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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

TIME_STEP_ROUND = 3
UIN_BIN = 1.0

train = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)

train["time_step_r"] = train["time_step"].round(TIME_STEP_ROUND)
test["time_step_r"] = test["time_step"].round(TIME_STEP_ROUND)

train["u_in_bin"] = np.floor(train["u_in"] / UIN_BIN).astype(np.int16)
test["u_in_bin"] = np.floor(test["u_in"] / UIN_BIN).astype(np.int16)

train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)
test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

train["u_in_lag1"] = (
    train.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)
test["u_in_lag1"] = (
    test.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)

train["dt"] = (
    train.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(train["time_step"])
    .astype(np.float32)
    .clip(lower=0.0)
)
test["dt"] = (
    test.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(test["time_step"])
    .astype(np.float32)
    .clip(lower=0.0)
)

train["vol_cum"] = (
    (train["u_in"].astype(np.float32) * train["dt"])
    .groupby(train["breath_id"], sort=False)
    .cumsum()
)
test["vol_cum"] = (
    (test["u_in"].astype(np.float32) * test["dt"])
    .groupby(test["breath_id"], sort=False)
    .cumsum()
)

UIN_CUM_BIN = 20.0
UIN_LAG_BIN = 3.0

train["vol_cum_bin"] = np.floor(train["vol_cum"] / UIN_CUM_BIN).astype(np.int16)
test["vol_cum_bin"] = np.floor(test["vol_cum"] / UIN_CUM_BIN).astype(np.int16)

train["u_in_lag1_bin"] = np.floor(train["u_in_lag1"] / UIN_LAG_BIN).astype(np.int16)
test["u_in_lag1_bin"] = np.floor(test["u_in_lag1"] / UIN_LAG_BIN).astype(np.int16)

bin_cols = ["u_in_bin", "vol_cum_bin", "u_in_lag1_bin"]
for col in bin_cols:
    tr_min = int(train[col].min())
    tr_max = int(train[col].max())
    test[col] = test[col].clip(lower=tr_min, upper=tr_max).astype(np.int16)

train_insp = train.loc[
    train["u_out"].eq(0),
    [
        "R",
        "C",
        "time_step_r",
        "u_in_bin",
        "vol_cum_bin",
        "u_in_lag1_bin",
        "u_out",
        "pressure",
    ],
].copy()

train_exp = train.loc[
    train["u_out"].eq(1),
    [
        "R",
        "C",
        "time_step_r",
        "u_in_bin",
        "vol_cum_bin",
        "u_in_lag1_bin",
        "u_out",
        "pressure",
    ],
].copy()

grp_rc_t_u_h = (
    train_insp.groupby(
        ["R", "C", "time_step_r", "u_in_bin", "vol_cum_bin", "u_in_lag1_bin"],
        sort=False,
        as_index=False,
    )["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_t_u_h"})
)

grp_rc_t_u_lag = (
    train_insp.groupby(
        ["R", "C", "time_step_r", "u_in_bin", "u_in_lag1_bin"],
        sort=False,
        as_index=False,
    )["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_t_u_lag"})
)

grp_rc_t_vol = (
    train_insp.groupby(
        ["R", "C", "time_step_r", "vol_cum_bin"], sort=False, as_index=False
    )["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_t_vol"})
)

grp_rc_t_u = (
    train_insp.groupby(
        ["R", "C", "time_step_r", "u_in_bin"], sort=False, as_index=False
    )["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_t_u"})
)

grp_rc_t_uout = (
    train_insp.groupby(["R", "C", "time_step_r", "u_out"], sort=False, as_index=False)[
        "pressure"
    ]
    .mean()
    .rename(columns={"pressure": "m_rc_t_uout"})
)

grp_rc_t = (
    train_insp.groupby(["R", "C", "time_step_r"], sort=False, as_index=False)[
        "pressure"
    ]
    .mean()
    .rename(columns={"pressure": "m_rc_t"})
)

grp_rc_u = (
    train_insp.groupby(["R", "C", "u_in_bin"], sort=False, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_u"})
)

grp_rc_u_cum = (
    train_insp.groupby(
        ["R", "C", "u_in_bin", "vol_cum_bin"], sort=False, as_index=False
    )["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_u_cum"})
)

grp_rc = (
    train_insp.groupby(["R", "C"], sort=False, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc"})
)

global_mean_insp = float(train_insp["pressure"].mean())

ts_u = (
    train_insp.groupby(["R", "C", "u_in_bin", "time_step_r"], sort=False)["pressure"]
    .mean()
    .rename("m_rc_u_t_exact")
    .reset_index()
    .sort_values(["R", "C", "u_in_bin", "time_step_r"], kind="mergesort")
)
ts_u["m_rc_u_t_roll"] = (
    ts_u.groupby(["R", "C", "u_in_bin"], sort=False)["m_rc_u_t_exact"]
    .transform(lambda s: s.rolling(window=5, center=True, min_periods=1).mean())
    .astype(np.float64)
)
ts_u_roll = ts_u[["R", "C", "u_in_bin", "time_step_r", "m_rc_u_t_roll"]]

grp_rc_t_exp = (
    train_exp.groupby(["R", "C", "time_step_r"], sort=False, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "m_rc_t_exp"})
)
global_mean_exp = float(train_exp["pressure"].mean()) if len(train_exp) else 0.0

test2 = test.copy()

test2 = test2.merge(
    grp_rc_t_u_h,
    on=["R", "C", "time_step_r", "u_in_bin", "vol_cum_bin", "u_in_lag1_bin"],
    how="left",
    sort=False,
)
test2 = test2.merge(
    grp_rc_t_u_lag,
    on=["R", "C", "time_step_r", "u_in_bin", "u_in_lag1_bin"],
    how="left",
    sort=False,
)
test2 = test2.merge(
    grp_rc_t_vol, on=["R", "C", "time_step_r", "vol_cum_bin"], how="left", sort=False
)
test2 = test2.merge(
    grp_rc_t_u, on=["R", "C", "time_step_r", "u_in_bin"], how="left", sort=False
)
test2 = test2.merge(
    grp_rc_t_uout, on=["R", "C", "time_step_r", "u_out"], how="left", sort=False
)
test2 = test2.merge(grp_rc_t, on=["R", "C", "time_step_r"], how="left", sort=False)
test2 = test2.merge(
    ts_u_roll, on=["R", "C", "u_in_bin", "time_step_r"], how="left", sort=False
)
test2 = test2.merge(
    grp_rc_u_cum, on=["R", "C", "u_in_bin", "vol_cum_bin"], how="left", sort=False
)
test2 = test2.merge(grp_rc_u, on=["R", "C", "u_in_bin"], how="left", sort=False)
test2 = test2.merge(grp_rc, on=["R", "C"], how="left", sort=False)
test2 = test2.merge(grp_rc_t_exp, on=["R", "C", "time_step_r"], how="left", sort=False)

test2["pressure_pred_insp"] = (
    test2["m_rc_t_u_h"]
    .fillna(test2["m_rc_t_u_lag"])
    .fillna(test2["m_rc_t_vol"])
    .fillna(test2["m_rc_t_u"])
    .fillna(test2["m_rc_t_uout"])
    .fillna(test2["m_rc_t"])
    .fillna(test2["m_rc_u_t_roll"])
    .fillna(test2["m_rc_u_cum"])
    .fillna(test2["m_rc_u"])
    .fillna(test2["m_rc"])
    .fillna(global_mean_insp)
    .astype(np.float64)
)

test2["pressure_pred_exp"] = (
    test2["m_rc_t_exp"].fillna(global_mean_exp).astype(np.float64)
)

test2["pressure_pred"] = np.where(
    test2["u_out"].eq(1).to_numpy(),
    test2["pressure_pred_exp"].to_numpy(),
    test2["pressure_pred_insp"].to_numpy(),
).astype(np.float64)

sub = pd.read_csv(sample_path, usecols=["id"])
sub = sub.merge(test2[["id", "pressure_pred"]], on="id", how="left", sort=False)
sub = sub.sort_values("id", kind="mergesort").reset_index(drop=True)
sub = sub.rename(columns={"pressure_pred": "pressure"})[["id", "pressure"]]

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert list(sub.columns) == ["id", "pressure"]
assert len(sub) == len(test)
assert sub["pressure"].notna().all()
assert sub["id"].is_monotonic_increasing
