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

0.1426497337903372

# 6. Current score

1.61322

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The current notebook fails because it tries to read two external blend files that don’t exist in this Kaggle environment; since no model is trained here, we must instead create a standalone prediction to produce a valid submission CSV. I keep your existing pressure-grid snapping logic (find_nearest) intact, and replace the failing blend step with a simple, deterministic baseline: predict a constant pressure equal to the median training pressure, then snap to the nearest allowed pressure value. This runs end-to-end on the provided dataset paths and writes a valid `submission.csv` with the required `id,pressure` columns. No score-targeting beyond making it runnable is possible without introducing a real model/training pipeline, which would violate the “minimal changes / preserve core logic” constraint given the current code only blends external submissions.'
- What this solution (achieved 5.8996) has done: 'Your current submission is a constant-pressure baseline (median train pressure snapped to the allowed pressure grid), which explains the very high MAE vs the target. To move the score toward the target with minimal changes and without altering the “pressure grid snapping” core logic, I replace the constant prediction with a simple, deterministic, per-time-step lookup baseline: predict the median training pressure conditioned on (R, C, time_step, u_out) using a rounded time_step key and fall back safely to global medians when a key is unseen. This preserves evaluation semantics, avoids leakage (no test labels), runs fast under 600s, and still outputs a valid `submission.csv` with `id,pressure` aligned to `test.csv`. Finally, predictions are snapped using your existing `find_nearest` to keep outputs on the valid pressure grid.'
- What this solution (achieved 8.24584) has done: 'Your current lookup baseline is leaving a lot of signal on the table because it ignores the strong autoregressive structure within each breath. To move the MAE down toward the target while keeping the “grouped-median lookup + pressure-grid snapping” core logic, I add minimal lagged-control features computed per breath (`u_in_lag1`, `u_in_cum`) and condition the median lookup on them (with safe backoffs to your existing keys). I also round `time_step` to 2 decimals (more robust matching across train/test) and add a final backoff keyed on `(R,C,time_step_r)` to reduce NaNs without changing evaluation semantics. The output remains snapped via your existing `find_nearest` and writes a valid `submission.csv`.'
- What this solution (achieved 8.39774) has done: 'Your current score is much worse than the target (lower is better), so we need a small but meaningful accuracy lift without changing the overall “grouped-median lookup + pressure-grid snapping” approach. The main issue is that your richer lookup discretization (lag and cumulative bins) is too lossy and often mismatches train/test, and you’re not using the strongest available signal (`u_in`) directly in the mapping. I keep the same pipeline structure but (1) add a `u_in`-binned median map at the same time resolution, and (2) make the lag/cumulative bins finer (still deterministic) to reduce collisions, then (3) blend the rich/base predictions with a simple fallback order to reduce NaNs. This preserves your core logic, keeps snapping via `find_nearest`, and still writes a valid `submission.csv`.'
- What this solution (achieved 8.45164) has done: 'Your current lookup baseline is underperforming largely because the join keys you use for medians don’t align well between train/test (especially time rounding and coarse binning), causing many fallbacks to weak global medians. I keep the exact same “grouped-median lookup → fallback chain → snap to nearest allowed pressure” core logic, but (1) change `time_step_r` rounding to 2 decimals to match the dataset’s typical 0.03 spacing and increase hit-rate, and (2) add a minimal extra fallback that conditions on `(R,C,u_out,time_step_r,u_in_r)` with a finer `u_in` bin (1.0) without changing the overall approach. This should reduce NaNs and make predictions more responsive to the strongest signal (`u_in`), moving MAE downward toward your target without introducing any new model/training loop. The output format and snapping remain identical and it still writes a valid `submission.csv`.'
- What this solution (achieved 8.52526) has done: 'Your current lookup tables are too “sparse” because the (R,C,u_out,time_step,lag,cum) exact-key merges miss frequently; the frequent fallbacks to global medians hurt MAE. I keep the same core logic (grouped-median lookup → fallback chain → snap to nearest allowed pressure) but make one minimal, high-impact adjustment: use a slightly coarser `time_step` rounding (1 decimal) for the richer keys to dramatically increase match rate, while keeping your existing 2-decimal maps as higher-resolution fallbacks. I also add one extra intermediate fallback keyed on `(R,C,u_out,time_step_r1,u_in_r_fine)` to reintroduce `u_in` signal when the rich map misses at 1-decimal time. This should reduce reliance on weak global fallbacks and move the MAE down toward the target while preserving your approach and producing a valid `submission.csv`.'
- What this solution (achieved 8.77132) has done: 'Your current score is far above the (lower-is-better) target, so we need a small but meaningful accuracy lift while keeping your same “grouped-median lookup → fallback chain → snap to nearest pressure grid” core logic. The biggest win available without changing the approach is to (1) add a strong, low-leakage physical feature (`u_in` integral = `u_in * delta_t` cumulative per breath) and use it in an additional median lookup, and (2) enforce the competition’s evaluation semantics by setting predictions to 0 during expiratory phase (`u_out==1`), which is not scored and commonly improves overall MAE stability. These are minimal extensions to your existing feature set and fallback chain, keep the same median-merge logic, and preserve your snapping. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 8.52537) has done: 'To move your MAE down toward the target while keeping the exact same “grouped-median lookup → fallback chain → snap to nearest pressure grid” core logic, I make three minimal, high-impact fixes: (1) stop forcing expiratory (`u_out==1`) predictions to 0 (those rows are not scored and zeroing them can badly distort dynamics around the phase boundary), (2) add a strictly-inspiratory (`u_out==0`) median lookup map (same features, same median approach) which better matches the evaluation semantics, and (3) slightly refine the `u_in_int` discretization (0.2 instead of 0.5) to reduce quantization loss while keeping the same integral feature and median mapping approach. Everything else (feature set, merge-based prediction, fallback order concept, snapping via `find_nearest`, and output format/path) remains intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 8.52538) has done: 'Your current MAE is far above the (lower-is-better) target, so we need a modest but real accuracy lift while keeping your same “grouped-median lookup → fallback chain → snap to nearest pressure grid” approach. The single biggest missing piece is that pressure strongly depends on the breath’s running state, and your lookup doesn’t use the most informative “state proxy” (`u_in` integral) at high-resolution time; we add one additional median map keyed on `(R,C,u_out,time_step_r,u_in_int_r)` and use it early in the fallback chain. To reduce sparsity/mismatch without changing semantics, we also add a very small rolling-mean smoothing of `u_in` (per breath) and create one extra lookup keyed on that smoothed control at `time_step_r`. Everything else (features, median-merge prediction, fallback style, and snapping) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.61322) has done: 'We keep your current “grouped-median lookup → fallback chain → snap to nearest pressure grid” core logic, but fix a key alignment issue that can severely inflate MAE: you sort `test` for feature generation/merging, then assign predictions to `sub` in the original `df_test` order, causing row misalignment. The minimal change is to carry `id` through the sorted `test_pred`, build predictions in that same sorted order, then merge/reindex back to the sample submission’s `id` order before writing. This doesn’t change your modeling semantics at all—only ensures each prediction is attached to the correct `id`. With this fix, your score should move sharply downward toward the target (lower is better) without altering the approach.'
- What this solution (achieved 1.61322) has done: 'Your current score is still far above the (lower-is-better) target, so we should try to reduce MAE with minimal, safe changes while keeping the same “grouped-median lookup → fallback chain → snap to nearest pressure grid” core logic. The most impactful low-risk improvement here is to align the lookup more closely with the scoring rule (only inspiratory phase is evaluated): build the median maps only from inspiratory rows (`u_out==0`) for the main lookups and, at inference time, explicitly use inspiratory-derived fallbacks when `u_out==0` (while leaving expiratory rows as a simple, stable fallback). This preserves your exact modeling style (median maps + fallbacks) but reduces noise introduced by mixing expiratory dynamics into the learned medians. I also keep the crucial `id` alignment logic intact so predictions attach to the correct rows.'
- What this solution (achieved 1.61322) has done: 'Your current score is still much worse than the (lower-is-better) target, so we should make a small, low-risk improvement without changing your core “grouped-median lookup → fallback chain → snap to nearest pressure grid” approach. The biggest remaining mismatch is that all your lookup maps are built only from inspiratory rows (`u_out==0`), yet many of the join keys include `u_out`, meaning test rows with `u_out==1` never match and fall back to weak global medians. I add a parallel set of median maps trained on expiratory rows (`u_out==1`) for the same key sets and use them only when `u_out==1`, while keeping your inspiratory maps and fallback order unchanged for `u_out==0`. This preserves the same semantics/logic, reduces unnecessary fallbacks, and should move MAE downward toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved 1.61322) has done: 'Your current score (1.61322, lower-is-better) is still far above the target, so we should improve accuracy with the smallest change that preserves your exact “grouped-median lookup → fallback chain → snap to nearest pressure grid” logic. The most likely remaining error is key mismatch due to float rounding/representation differences between train/test on `time_step` and derived discretized columns, which reduces hit-rate and forces weak fallbacks. I make the merge keys deterministic by converting rounded time keys to scaled integers (e.g., `time_step_r2_i=int(round(time_step*100))`, `time_step_r1_i=int(round(time_step*10))`) and likewise store the discretized bins as small integers, without changing which features you use or the fallback order. This typically increases exact-match rates and should move MAE downward toward your target while keeping runtime reasonable and still producing a valid `submission.csv`.'
- What this solution (achieved 1.61322) has done: 'Your current score is still far above the (lower-is-better) target, so we should improve MAE with a minimal change that keeps the same “grouped-median lookup → fallback chain → snap to nearest pressure grid” core logic. The strongest low-risk lift here is to add one more lookup that uses the most informative state proxy you already compute (`u_in_int_r2`) at the highest time resolution (`time_step_r2_i`), which reduces reliance on weaker fallbacks without changing the approach. Concretely, we add `u_in_int_r2 = round(u_in_int/0.5)` (a coarser, denser key than your 0.2 bin) and build median maps for both inspiratory and expiratory rows keyed on `(R,C,u_out,time_step_r2_i,u_in_int_r2)`. We then insert these predictions early in each phase’s fallback chain, leaving everything else (features, merges, snapping, and submission alignment) intact.'

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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

train = df_train[
    ["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"]
].copy()

test = df_test[["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"]].copy()

train.sort_values(["breath_id", "time_step"], inplace=True)
test.sort_values(["breath_id", "time_step"], inplace=True)

train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
test["u_in_lag1"] = test.groupby("breath_id")["u_in"].shift(1).fillna(0.0)

train["u_in_cum"] = train.groupby("breath_id")["u_in"].cumsum()
test["u_in_cum"] = test.groupby("breath_id")["u_in"].cumsum()

train["time_step_r2_i"] = np.rint(
    train["time_step"].to_numpy(dtype=np.float64) * 100.0
).astype(np.int16)
test["time_step_r2_i"] = np.rint(
    test["time_step"].to_numpy(dtype=np.float64) * 100.0
).astype(np.int16)
train["time_step_r1_i"] = np.rint(
    train["time_step"].to_numpy(dtype=np.float64) * 10.0
).astype(np.int16)
test["time_step_r1_i"] = np.rint(
    test["time_step"].to_numpy(dtype=np.float64) * 10.0
).astype(np.int16)

train["u_in_lag1_r"] = np.rint(
    train["u_in_lag1"].to_numpy(dtype=np.float64) / 2.0
).astype(np.int16)
test["u_in_lag1_r"] = np.rint(
    test["u_in_lag1"].to_numpy(dtype=np.float64) / 2.0
).astype(np.int16)

train["u_in_cum_r"] = np.rint(
    train["u_in_cum"].to_numpy(dtype=np.float64) / 5.0
).astype(np.int16)
test["u_in_cum_r"] = np.rint(test["u_in_cum"].to_numpy(dtype=np.float64) / 5.0).astype(
    np.int16
)

train["u_in_r"] = np.rint(train["u_in"].to_numpy(dtype=np.float64) / 2.0).astype(
    np.int16
)
test["u_in_r"] = np.rint(test["u_in"].to_numpy(dtype=np.float64) / 2.0).astype(np.int16)

train["u_in_r_fine"] = np.rint(train["u_in"].to_numpy(dtype=np.float64) / 1.0).astype(
    np.int16
)
test["u_in_r_fine"] = np.rint(test["u_in"].to_numpy(dtype=np.float64) / 1.0).astype(
    np.int16
)

train["delta_t"] = (
    train.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
)
test["delta_t"] = (
    test.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
)

train["u_in_int"] = (
    (train["u_in"] * train["delta_t"])
    .groupby(train["breath_id"])
    .cumsum()
    .astype(np.float32)
)
test["u_in_int"] = (
    (test["u_in"] * test["delta_t"])
    .groupby(test["breath_id"])
    .cumsum()
    .astype(np.float32)
)

train["u_in_int_r"] = np.rint(
    train["u_in_int"].to_numpy(dtype=np.float64) / 0.2
).astype(np.int32)
test["u_in_int_r"] = np.rint(test["u_in_int"].to_numpy(dtype=np.float64) / 0.2).astype(
    np.int32
)

train["u_in_int_r2"] = np.rint(
    train["u_in_int"].to_numpy(dtype=np.float64) / 0.5
).astype(np.int32)
test["u_in_int_r2"] = np.rint(test["u_in_int"].to_numpy(dtype=np.float64) / 0.5).astype(
    np.int32
)

train["u_in_smooth"] = (
    train.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
test["u_in_smooth"] = (
    test.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)

train["u_in_smooth_r"] = np.rint(
    train["u_in_smooth"].to_numpy(dtype=np.float64) / 1.0
).astype(np.int16)
test["u_in_smooth_r"] = np.rint(
    test["u_in_smooth"].to_numpy(dtype=np.float64) / 1.0
).astype(np.int16)

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()

grp_cols_rich = ["R", "C", "u_out", "time_step_r2_i", "u_in_lag1_r", "u_in_cum_r"]
median_map_rich = (
    train_insp.groupby(grp_cols_rich, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rich"})
)
median_map_rich_exp = (
    train_exp.groupby(grp_cols_rich, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rich_exp"})
)

grp_cols_rich_r1 = ["R", "C", "u_out", "time_step_r1_i", "u_in_lag1_r", "u_in_cum_r"]
median_map_rich_r1 = (
    train_insp.groupby(grp_cols_rich_r1, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rich_r1"})
)
median_map_rich_r1_exp = (
    train_exp.groupby(grp_cols_rich_r1, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rich_r1_exp"})
)

grp_cols_uin = ["R", "C", "u_out", "time_step_r2_i", "u_in_r"]
median_map_uin = (
    train_insp.groupby(grp_cols_uin, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin"})
)
median_map_uin_exp = (
    train_exp.groupby(grp_cols_uin, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_exp"})
)

grp_cols_uin_fine = ["R", "C", "u_out", "time_step_r2_i", "u_in_r_fine"]
median_map_uin_fine = (
    train_insp.groupby(grp_cols_uin_fine, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_fine"})
)
median_map_uin_fine_exp = (
    train_exp.groupby(grp_cols_uin_fine, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_fine_exp"})
)

grp_cols_uin_fine_r1 = ["R", "C", "u_out", "time_step_r1_i", "u_in_r_fine"]
median_map_uin_fine_r1 = (
    train_insp.groupby(grp_cols_uin_fine_r1, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_fine_r1"})
)
median_map_uin_fine_r1_exp = (
    train_exp.groupby(grp_cols_uin_fine_r1, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_fine_r1_exp"})
)

grp_cols_uint = ["R", "C", "u_out", "time_step_r1_i", "u_in_int_r"]
median_map_uint = (
    train_insp.groupby(grp_cols_uint, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uint"})
)
median_map_uint_exp = (
    train_exp.groupby(grp_cols_uint, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uint_exp"})
)

grp_cols_uint_r2 = ["R", "C", "u_out", "time_step_r2_i", "u_in_int_r"]
median_map_uint_r2 = (
    train_insp.groupby(grp_cols_uint_r2, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uint_r2"})
)
median_map_uint_r2_exp = (
    train_exp.groupby(grp_cols_uint_r2, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uint_r2_exp"})
)

grp_cols_uint_r2_coarse = ["R", "C", "u_out", "time_step_r2_i", "u_in_int_r2"]
median_map_uint_r2_coarse = (
    train_insp.groupby(grp_cols_uint_r2_coarse, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uint_r2c"})
)
median_map_uint_r2_coarse_exp = (
    train_exp.groupby(grp_cols_uint_r2_coarse, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uint_r2c_exp"})
)

grp_cols_uin_smooth = ["R", "C", "u_out", "time_step_r2_i", "u_in_smooth_r"]
median_map_uin_smooth = (
    train_insp.groupby(grp_cols_uin_smooth, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_smooth"})
)
median_map_uin_smooth_exp = (
    train_exp.groupby(grp_cols_uin_smooth, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_uin_smooth_exp"})
)

grp_cols_base = ["R", "C", "u_out", "time_step_r2_i"]
median_map_base = (
    train_insp.groupby(grp_cols_base, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_base"})
)
median_map_base_exp = (
    train_exp.groupby(grp_cols_base, observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_base_exp"})
)

median_map_rc_t = (
    train.groupby(["R", "C", "time_step_r2_i"], observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rc_t"})
)

median_map_rc_uout = (
    train.groupby(["R", "C", "u_out"], observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rc"})
)

median_map_base_insp = (
    train_insp.groupby(["R", "C", "time_step_r2_i"], observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_base_insp"})
)
median_map_rc_t_insp = (
    train_insp.groupby(["R", "C", "time_step_r2_i"], observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rc_t_insp"})
)
median_map_rc_insp = (
    train_insp.groupby(["R", "C"], observed=True)["pressure"]
    .median()
    .astype(np.float32)
    .reset_index()
    .rename(columns={"pressure": "pred_rc_insp"})
)

global_median = float(np.median(train["pressure"].values))
global_median_insp = float(np.median(train_insp["pressure"].values))
global_median_exp = (
    float(np.median(train_exp["pressure"].values)) if len(train_exp) else global_median
)

test_pred = test.merge(median_map_rich, on=grp_cols_rich, how="left")
test_pred = test_pred.merge(median_map_rich_r1, on=grp_cols_rich_r1, how="left")
test_pred = test_pred.merge(median_map_uin_fine, on=grp_cols_uin_fine, how="left")
test_pred = test_pred.merge(median_map_uin_fine_r1, on=grp_cols_uin_fine_r1, how="left")
test_pred = test_pred.merge(median_map_uin, on=grp_cols_uin, how="left")
test_pred = test_pred.merge(median_map_uint, on=grp_cols_uint, how="left")
test_pred = test_pred.merge(median_map_uint_r2, on=grp_cols_uint_r2, how="left")
test_pred = test_pred.merge(
    median_map_uint_r2_coarse, on=grp_cols_uint_r2_coarse, how="left"
)
test_pred = test_pred.merge(median_map_uin_smooth, on=grp_cols_uin_smooth, how="left")
test_pred = test_pred.merge(median_map_base, on=grp_cols_base, how="left")

test_pred = test_pred.merge(median_map_rich_exp, on=grp_cols_rich, how="left")
test_pred = test_pred.merge(median_map_rich_r1_exp, on=grp_cols_rich_r1, how="left")
test_pred = test_pred.merge(median_map_uin_fine_exp, on=grp_cols_uin_fine, how="left")
test_pred = test_pred.merge(
    median_map_uin_fine_r1_exp, on=grp_cols_uin_fine_r1, how="left"
)
test_pred = test_pred.merge(median_map_uin_exp, on=grp_cols_uin, how="left")
test_pred = test_pred.merge(median_map_uint_exp, on=grp_cols_uint, how="left")
test_pred = test_pred.merge(median_map_uint_r2_exp, on=grp_cols_uint_r2, how="left")
test_pred = test_pred.merge(
    median_map_uint_r2_coarse_exp, on=grp_cols_uint_r2_coarse, how="left"
)
test_pred = test_pred.merge(
    median_map_uin_smooth_exp, on=grp_cols_uin_smooth, how="left"
)
test_pred = test_pred.merge(median_map_base_exp, on=grp_cols_base, how="left")

test_pred = test_pred.merge(
    median_map_rc_t, on=["R", "C", "time_step_r2_i"], how="left"
)
test_pred = test_pred.merge(median_map_rc_uout, on=["R", "C", "u_out"], how="left")
test_pred = test_pred.merge(
    median_map_base_insp, on=["R", "C", "time_step_r2_i"], how="left"
)
test_pred = test_pred.merge(
    median_map_rc_t_insp, on=["R", "C", "time_step_r2_i"], how="left"
)
test_pred = test_pred.merge(median_map_rc_insp, on=["R", "C"], how="left")

u_out_arr = test_pred["u_out"].to_numpy(dtype=np.int8)

pred_insp = test_pred["pred_rich"].to_numpy(dtype=np.float32)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_rich_r1"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp),
    test_pred["pred_uin_fine"].to_numpy(dtype=np.float32),
    pred_insp,
)
pred_insp = np.where(
    np.isnan(pred_insp),
    test_pred["pred_uin_fine_r1"].to_numpy(dtype=np.float32),
    pred_insp,
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_uin"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_uint_r2"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp),
    test_pred["pred_uint_r2c"].to_numpy(dtype=np.float32),
    pred_insp,
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_uint"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp),
    test_pred["pred_uin_smooth"].to_numpy(dtype=np.float32),
    pred_insp,
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_base"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp),
    test_pred["pred_base_insp"].to_numpy(dtype=np.float32),
    pred_insp,
)
pred_insp = np.where(
    np.isnan(pred_insp),
    test_pred["pred_rc_t_insp"].to_numpy(dtype=np.float32),
    pred_insp,
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_rc_t"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_rc"].to_numpy(dtype=np.float32), pred_insp
)
pred_insp = np.where(
    np.isnan(pred_insp), test_pred["pred_rc_insp"].to_numpy(dtype=np.float32), pred_insp
)

pred_exp = test_pred["pred_rich_exp"].to_numpy(dtype=np.float32)
pred_exp = np.where(
    np.isnan(pred_exp),
    test_pred["pred_rich_r1_exp"].to_numpy(dtype=np.float32),
    pred_exp,
)
pred_exp = np.where(
    np.isnan(pred_exp),
    test_pred["pred_uin_fine_exp"].to_numpy(dtype=np.float32),
    pred_exp,
)
pred_exp = np.where(
    np.isnan(pred_exp),
    test_pred["pred_uin_fine_r1_exp"].to_numpy(dtype=np.float32),
    pred_exp,
)
pred_exp = np.where(
    np.isnan(pred_exp), test_pred["pred_uin_exp"].to_numpy(dtype=np.float32), pred_exp
)
pred_exp = np.where(
    np.isnan(pred_exp),
    test_pred["pred_uint_r2_exp"].to_numpy(dtype=np.float32),
    pred_exp,
)
pred_exp = np.where(
    np.isnan(pred_exp),
    test_pred["pred_uint_r2c_exp"].to_numpy(dtype=np.float32),
    pred_exp,
)
pred_exp = np.where(
    np.isnan(pred_exp), test_pred["pred_uint_exp"].to_numpy(dtype=np.float32), pred_exp
)
pred_exp = np.where(
    np.isnan(pred_exp),
    test_pred["pred_uin_smooth_exp"].to_numpy(dtype=np.float32),
    pred_exp,
)
pred_exp = np.where(
    np.isnan(pred_exp), test_pred["pred_base_exp"].to_numpy(dtype=np.float32), pred_exp
)
pred_exp = np.where(
    np.isnan(pred_exp), test_pred["pred_rc_t"].to_numpy(dtype=np.float32), pred_exp
)
pred_exp = np.where(
    np.isnan(pred_exp), test_pred["pred_rc"].to_numpy(dtype=np.float32), pred_exp
)

pred = np.where(u_out_arr == 0, pred_insp, pred_exp).astype(np.float32)

pred = np.where(np.isnan(pred) & (u_out_arr == 0), global_median_insp, pred)
pred = np.where(np.isnan(pred) & (u_out_arr == 1), global_median_exp, pred)
pred = np.where(np.isnan(pred), global_median, pred).astype(np.float32)

pred_df = pd.DataFrame({"id": test_pred["id"].to_numpy(), "pressure": pred})
pred_df["pressure"] = pred_df["pressure"].apply(find_nearest)

sub = sub[["id"]].merge(pred_df, on="id", how="left")
sub["pressure"] = sub["pressure"].fillna(global_median).astype(np.float32)
sub["pressure"] = sub["pressure"].apply(find_nearest)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Pred stats:",
    float(np.min(sub["pressure"].values)),
    float(np.mean(sub["pressure"].values)),
    float(np.max(sub["pressure"].values)),
)
