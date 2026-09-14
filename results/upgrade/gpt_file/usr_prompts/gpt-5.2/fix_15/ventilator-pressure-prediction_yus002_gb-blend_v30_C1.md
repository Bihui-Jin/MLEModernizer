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

0.1748184868742451

# 6. Current score

3.24699

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.86458) has done: 'I remove the dependency on missing external “gb-blending” files (the root cause of the FileNotFoundError) and instead generate a valid baseline submission directly from the provided competition data. To keep changes minimal and score-reasonable without altering any model architecture (there is none here), I compute a simple per-(R,C,time_step,u_in,u_out) mean pressure lookup from train and use it to predict test pressures, with a fallback to the global mean for unseen combinations. This runs end-to-end using only numpy/pandas and writes a properly formatted `submission.csv` with `id,pressure`. The unused/random blending functions are kept but no longer block execution.'
- What this solution (achieved 6.13071) has done: 'Your current MAE (7.86458) is far worse than the target (0.1748), so we should improve score with minimal, safe changes while keeping the same “train lookup table → merge into test → fallback” core logic. The biggest win with almost no logic change is to fix the keying: using raw `u_in` as a key creates massive sparsity (few exact matches), causing lots of global-mean fallbacks and very poor MAE. We keep the same groupby-mean approach but (1) quantize `u_in` slightly (rounding) to increase match rate, and (2) add a hierarchical fallback: first try a coarser lookup (drop `u_in`), then fall back to the global mean. This preserves evaluation semantics (still a mean lookup baseline) and should move the score much closer to the target band.'
- What this solution (achieved 4.08576) has done: 'Your current score (6.13071 MAE) is far worse than the target (0.1748), so we should improve it with the smallest possible changes while keeping the same “groupby-mean lookup → merge into test → hierarchical fallback” core logic. The biggest remaining mismatch is that `time_step` is still being used as an exact key (even with 5-decimal rounding), which is unnecessarily sparse and causes many fallbacks; we coarsen it slightly to increase match rate while staying faithful to the lookup-table approach. We also add one extra intermediate fallback (drop both `time_step` and `u_in`) before the global mean, which is still the same semantics (mean lookup) but reduces catastrophic misses. Finally, we ensure we don’t duplicate columns on merge and that the submission aligns exactly by `id`.'
- What this solution (achieved 4.33536) has done: 'Your current MAE (4.08576) is still far above the target (0.1748), so we should legitimately improve it with the smallest changes while keeping the same “groupby-mean lookup → merge into test → hierarchical fallback” core logic. The biggest remaining issue is that even with rounding, keying on `time_step` is still too sparse and drives many rows into coarse/global fallbacks; we coarsen `time_step` a bit more to increase match rate. We also add one additional intermediate fallback that uses `(R,C,u_in_r,u_out)` (dropping `time_step`) so inspiratory dynamics keyed by valve opening are captured without requiring exact timing alignment. These are minimal changes that preserve the existing approach and should move the score substantially toward the target without changing model type or training semantics.'
- What this solution (achieved 4.33536) has done: 'Your current MAE (4.33536, lower is better) is still far above the target (0.1748), so we should improve it with the smallest possible change while keeping the same “groupby mean lookup → merge into test → hierarchical fallback” core logic. The largest avoidable error source is that we’re averaging over both inspiratory and expiratory phases, but Kaggle scores only inspiratory rows (u_out==0), so we should build all lookup tables using only u_out==0 rows to better match the metric. To keep predictions defined for u_out==1 rows (not scored but required in submission), we additionally compute a simple per-(R,C) mean for u_out==1 and fall back to that for expiratory rows, otherwise using the inspiratory-trained hierarchy. This preserves the same approach (mean tables + fallbacks) but aligns training aggregation with the evaluation and should move the score substantially toward the target.'
- What this solution (achieved 4.26379) has done: 'We keep your exact “groupby mean lookup → merge into test → hierarchical fallback” approach, but make two metric-aligned tweaks that should materially reduce MAE (lower is better) toward the 0.1748 target. First, we align the lookup on `time_step` to the true sampling grid by converting it to an integer index (0–79) instead of rounding floats, which reduces key mismatch and unnecessary fallbacks without changing the model type. Second, because pressure is effectively quantized in this competition, we post-process predictions by snapping them to the nearest pressure value seen in the inspiratory training data, which typically improves MAE with minimal risk and no architecture/training changes. The rest (inspiratory-only tables, expiratory fallback, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 4.13357) has done: 'Your current MAE (4.26379; lower is better) is still far above the target (0.1748), so we should improve it with the smallest change that keeps your exact “groupby mean lookup → merge into test → hierarchical fallback” approach. The main issue is the `u_in` key: rounding to 0.1 is still too sparse, causing lots of fallbacks and large errors; we keep the same tables but switch to a slightly coarser `u_in` bin (0.5) to increase match rate. We also build a lookup for `u_out==1` rows directly from `train[u_out==1]` (including a fine `t_idx` table) so expiratory rows get reasonable values instead of relying on inspiratory/global means, without changing how inspiratory scoring works. Everything else (t_idx indexing, snapping to the pressure grid, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 4.2451) has done: 'We keep your exact “groupby mean lookup → merge into test → hierarchical fallback → snap to pressure grid” core logic, but improve the lookup keys so far fewer rows fall back to coarse/global means (which is what’s keeping MAE around 4). Specifically, we (1) use the true 80-step grid by deriving `t_idx` per-breath via `cumcount()` (robust to float noise), and (2) make `u_in` binning depend on phase: keep finer bins for inspiratory (scored) and coarser bins for expiratory (not scored) to increase match rate without changing semantics. We also add one minimal extra intermediate inspiratory fallback that drops `u_in` but keeps `(R,C,t_idx)` to better capture the typical pressure curve shape for a given lung setting. These are small changes, still pure pandas/numpy, and should move the score materially toward the 0.1748 target.'
- What this solution (achieved 6.04587) has done: 'Your current MAE (4.2451; lower is better) is far above the target (0.1748), so we should improve it with the smallest changes that preserve your exact “groupby mean lookup → hierarchical fallback → snap to pressure grid” approach. The biggest remaining avoidable error is that you’re using the same lookup keys for all inspiratory points, but the early inspiratory rise is better captured by cumulative volume/flow features that are derivable from the existing columns without changing the modeling paradigm. I add two per-breath cumulative features (`u_in_cum` and `u_in_lag1`) and include them only in the finest inspiratory lookup (with light rounding) to reduce ambiguity and improve match quality while keeping the same groupby-mean table logic. Everything else (phase-specific tables, fallbacks, snapping, submission writing) stays intact to keep changes minimal and stable.'
- What this solution (achieved 4.08052) has done: 'Your current MAE (6.04587, lower is better) is far above the target (0.1748), so we should improve with the smallest change that preserves your exact “groupby mean lookup → hierarchical fallback → snap to pressure grid” approach. The main issue is that the finest inspiratory key includes `u_in_cum_r`, which is essentially a per-breath running state and makes exact matches extremely rare in test, causing heavy fallback and large error. I keep your same tables and fallbacks, but (1) remove `u_in_cum_r` from the finest inspiratory lookup (still using `t_idx`, `u_in`, `u_in_lag1`, `R`, `C`, `u_out`) and (2) add one new intermediate table keyed by `(R,C,t_idx,u_in,u_out)` before coarser fallbacks to recover accuracy without adding any new modeling paradigm. Everything else (expiratory tables, snapping, submission writing) stays intact to minimize risk and keep runtime under the limit.'
- What this solution (achieved 3.90849) has done: 'We keep your exact “groupby-mean lookup → hierarchical fallback → snap to pressure grid” core logic, but adjust only the `u_in` binning for inspiratory rows to reduce sparsity and increase exact-match rate in the finest/near-finest tables (your current 0.1 bins still cause too many fallbacks). Specifically, we change inspiratory `u_in` rounding from 0.1 to 0.5 to better align test rows with train aggregates while leaving all tables/fallbacks/training semantics unchanged. This is a minimal, metric-aligned change (lower MAE is better) that should move the score down toward the 0.1748 target without introducing new models, features, or training loops. Everything else (t_idx via cumcount, lag feature, expiratory handling, snapping, submission writing) remains intact.'
- What this solution (achieved 3.19274) has done: 'Your current MAE (3.90849; lower is better) is still far above the target (0.1748), so we should improve it with the smallest change that preserves your exact “groupby-mean lookup → hierarchical fallback → snap to pressure grid” approach. The biggest remaining error source is that the finest inspiratory table still keys on `u_in_lag1_r`, which is highly continuous and makes exact matches rare, pushing many rows into much coarser fallbacks. We keep the same features and same overall fallback chain, but (1) coarsen `u_in_lag1_r` to 0.5 steps (matching `u_in_r_insp`) and (2) insert one extra intermediate table keyed by `(R,C,t_idx,u_in_lag1_r,u_out)` so we can still use lag information without requiring an exact `u_in` match. These are minimal, metric-aligned changes that should reduce fallback frequency and move MAE down toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 3.19448) has done: 'Your current MAE (3.19274, lower is better) is still far above the target (0.1748), so we should improve it with a minimal, semantics-preserving change to the existing “groupby-mean lookup → hierarchical fallback → snap to pressure grid” pipeline. The biggest remaining mismatch is that the finest inspiratory table still depends on `u_in_lag1_r`, which makes matches unnecessarily sparse; we keep the same features but make `u_in_lag1_r` slightly coarser (1.0 steps) to increase match rate and reduce fallback error. To avoid losing lag information entirely, we keep the intermediate `(R,C,t_idx,u_in_lag1_r,u_out)` table intact (now also denser) and leave the rest of the fallback chain and snapping unchanged. This should legitimately reduce MAE while staying within the same modeling approach and still writing a valid `submission.csv`.'
- What this solution (achieved 3.24699) has done: 'We keep your exact groupby-mean lookup + hierarchical fallback + pressure-grid snapping pipeline, but reduce unnecessary sparsity in the *scored* (u_out==0) phase. Concretely, we make the inspiratory `u_in` binning slightly finer (0.25 instead of 0.5) and also make the lag binning a bit finer (0.5 instead of 1.0) so more test rows can hit the higher-quality lookup tables before falling back to coarse/global means. This is a minimal, semantics-preserving adjustment (still just mean tables and the same fallback chain) that should decrease MAE from ~3.19 toward your 0.1748 target without changing any architecture/training loop (there is none). All paths and the submission writing remain unchanged and it still produce `submission.csv`.'

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
        weight1 = l[1] / l_sum
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**7
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
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

train = train.copy()
test = test.copy()

train["t_idx"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
test["t_idx"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

train["u_in_cum"] = (
    train.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)
test["u_in_cum"] = (
    test.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)
train["u_in_lag1"] = (
    train.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0).astype(np.float32)
)
test["u_in_lag1"] = (
    test.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0).astype(np.float32)
)

train["u_in_cum_r"] = (
    train["u_in_cum"] * 10
).round() / 10.0  # keep as-is (unused in keys)
test["u_in_cum_r"] = (test["u_in_cum"] * 10).round() / 10.0

train["u_in_lag1_r"] = (train["u_in_lag1"] * 2).round() / 2.0  # 0.5 steps (was 1.0)
test["u_in_lag1_r"] = (test["u_in_lag1"] * 2).round() / 2.0

train["u_in_r_insp"] = (
    train["u_in"] * 4
).round() / 4.0  # 0.25 increments for u_out==0 (was 0.5)
test["u_in_r_insp"] = (test["u_in"] * 4).round() / 4.0

train["u_in_r_exp"] = (train["u_in"] * 2).round() / 2.0  # 0.5 increments for u_out==1
test["u_in_r_exp"] = (test["u_in"] * 2).round() / 2.0

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()

keys_fine_insp = ["R", "C", "t_idx", "u_in_r_insp", "u_in_lag1_r", "u_out"]
mean_table_fine_insp = (
    train_insp.groupby(keys_fine_insp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_fine"})
)
test_pred = test.merge(mean_table_fine_insp, on=keys_fine_insp, how="left")

keys_rctlag_insp = ["R", "C", "t_idx", "u_in_lag1_r", "u_out"]
mean_table_rctlag_insp = (
    train_insp.groupby(keys_rctlag_insp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_rctlag"})
)
test_pred = test_pred.merge(mean_table_rctlag_insp, on=keys_rctlag_insp, how="left")

keys_rctu_insp = ["R", "C", "t_idx", "u_in_r_insp", "u_out"]
mean_table_rctu_insp = (
    train_insp.groupby(keys_rctu_insp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_rctu"})
)
test_pred = test_pred.merge(mean_table_rctu_insp, on=keys_rctu_insp, how="left")

keys_rcuu_insp = ["R", "C", "u_in_r_insp", "u_out"]
mean_table_rcuu_insp = (
    train_insp.groupby(keys_rcuu_insp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_rcuu"})
)
test_pred = test_pred.merge(mean_table_rcuu_insp, on=keys_rcuu_insp, how="left")

keys_rct_insp = ["R", "C", "t_idx", "u_out"]
mean_table_rct_insp = (
    train_insp.groupby(keys_rct_insp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_rct"})
)
test_pred = test_pred.merge(mean_table_rct_insp, on=keys_rct_insp, how="left")

keys_rcu_insp = ["R", "C", "u_out"]
mean_table_rcu_insp = (
    train_insp.groupby(keys_rcu_insp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_rcu"})
)
test_pred = test_pred.merge(mean_table_rcu_insp, on=keys_rcu_insp, how="left")

keys_fine_exp = ["R", "C", "t_idx", "u_in_r_exp", "u_out"]
mean_table_exp_fine = (
    train_exp.groupby(keys_fine_exp, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_exp_fine"})
)
test_pred = test_pred.merge(mean_table_exp_fine, on=keys_fine_exp, how="left")

keys_exp_rct = ["R", "C", "t_idx", "u_out"]
mean_table_exp_rct = (
    train_exp.groupby(keys_exp_rct, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_exp_rct"})
)
test_pred = test_pred.merge(mean_table_exp_rct, on=keys_exp_rct, how="left")

keys_exp_rc = ["R", "C", "u_out"]
mean_table_exp_rc = (
    train_exp.groupby(keys_exp_rc, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_exp_rc"})
)
test_pred = test_pred.merge(mean_table_exp_rc, on=keys_exp_rc, how="left")

keys_rcu_all = ["R", "C", "u_out"]
mean_table_rcu_all = (
    train.groupby(keys_rcu_all, observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_pred_rcu_all"})
)
test_pred = test_pred.merge(mean_table_rcu_all, on=keys_rcu_all, how="left")

global_mean_insp = float(train_insp["pressure"].mean())
global_mean_exp = (
    float(train_exp["pressure"].mean())
    if len(train_exp)
    else float(train["pressure"].mean())
)
global_mean_all = float(train["pressure"].mean())

insp_chain = (
    test_pred["pressure_pred_fine"]
    .fillna(test_pred["pressure_pred_rctlag"])
    .fillna(test_pred["pressure_pred_rctu"])
    .fillna(test_pred["pressure_pred_rcuu"])
    .fillna(test_pred["pressure_pred_rct"])
    .fillna(test_pred["pressure_pred_rcu"])
    .fillna(test_pred["pressure_pred_rcu_all"])
    .fillna(global_mean_insp)
    .fillna(global_mean_all)
)

exp_chain = (
    test_pred["pressure_pred_exp_fine"]
    .fillna(test_pred["pressure_pred_exp_rct"])
    .fillna(test_pred["pressure_pred_exp_rc"])
    .fillna(test_pred["pressure_pred_rcu_all"])
    .fillna(global_mean_exp)
    .fillna(global_mean_all)
)

test_pred["pressure_pred"] = np.where(
    test_pred["u_out"].to_numpy() == 1,
    exp_chain.to_numpy(),
    insp_chain.to_numpy(),
).astype(np.float32)

pressure_grid = np.sort(train_insp["pressure"].unique()).astype(np.float32)
pred = test_pred["pressure_pred"].to_numpy(dtype=np.float32)

idx = np.searchsorted(pressure_grid, pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx0 = np.clip(idx - 1, 0, len(pressure_grid) - 1)
choose_left = np.abs(pred - pressure_grid[idx0]) <= np.abs(pred - pressure_grid[idx])
pred_snapped = np.where(choose_left, pressure_grid[idx0], pressure_grid[idx]).astype(
    np.float32
)
test_pred["pressure_pred"] = pred_snapped

sub = sub[["id"]].merge(test_pred[["id", "pressure_pred"]], on="id", how="left")
sub["pressure"] = sub["pressure_pred"].fillna(global_mean_all).astype(np.float32)
sub = sub[["id", "pressure"]]

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Null pressures:", int(sub["pressure"].isna().sum()))
print("Pressure range:", float(sub["pressure"].min()), float(sub["pressure"].max()))
