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

0.1546511874263363

# 6. Current score

1.54322

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.1959) has done: 'I fix the `FileNotFoundError` by removing the dependency on external “gb-submitted” inputs and instead generating a valid baseline submission directly from the provided train/test files. To keep core semantics (pressure quantization to known training pressures), predictions be mapped using your existing `find_nearest` logic. I also correct the notebook cell numbering (start at 1) and ensure the script always writes a `submission.csv` with the required `id,pressure` columns. This run end-to-end in the Kaggle environment and yield a valid submission suitable for scoring.'
- What this solution (achieved 4.1975) has done: 'I fix the boolean-index alignment bug causing the `IndexingError` by ensuring the “missing” mask is a NumPy array aligned positionally with `test_feat`, not an index-labeled Series that can become unalignable. I also make the groupby reindex outputs consistently use NumPy arrays for assignment, which avoids pandas alignment pitfalls while preserving the exact feature logic and nearest-pressure post-processing. Finally, I keep the same file paths and ensure a valid `submission.csv` with `id,pressure` is always written.'
- What this solution (achieved 4.1975) has done: 'Your current solution is a pure lookup baseline; the biggest reason it’s far from the target is that it predicts pressures for *all* timesteps (including expiratory phase) even though Kaggle scores only inspiratory timesteps (`u_out==0`). With minimal change and the same feature/lookup core, we keep your existing groupby-based prediction but add a post-processing step that forces `pressure=0` whenever `u_out==1`, which typically reduces MAE substantially for this competition. To avoid introducing new modeling logic, we won’t change your binning, group keys, or nearest-pressure quantization—only align the output with the metric’s scoring mask behavior. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.66741) has done: 'Your current baseline is still far from the target, so the most effective minimal change is to make the lookup features better match the time-series dynamics without changing the “groupby mean lookup + nearest-pressure quantization” core logic. I keep your same prediction pipeline, but add two lightweight, competition-standard lag features (`u_in_bin_prev1` and `u_in_bin_prev2`) into the primary group key and provide a staged fallback (drop lags first, then drop `u_out/u_in_bin`, then the existing `(R,C,step)` fallback). This remains a pure table-lookup approach (no model/training loop changes) and should reduce MAE materially while still writing a valid `submission.csv`. The `u_out==1 -> 0` post-processing and nearest-pressure mapping are preserved exactly.'
- What this solution (achieved 2.66741) has done: 'Your current score is much worse than the target (lower is better), so we should make the smallest changes that improve accuracy without changing the overall “groupby mean lookup + staged fallback + nearest-pressure quantization + u_out==1->0” approach. The biggest safe win here is to stop using a hardcoded `step = cumcount()` (which can drift if any row ordering quirks exist) and instead compute `step` from `time_step` rank within each breath, which aligns train/test timesteps more robustly while preserving the same lookup logic. Second, we keep your exact feature set but slightly strengthen the fallback chain by adding an intermediate fallback that drops only `u_out` before dropping lags/u_in_bin, which helps when `u_out` differs but the rest of the pattern matches. These changes should reduce MAE toward the target while keeping runtime and semantics stable and still writing a valid `submission.csv`.'
- What this solution (achieved 2.50329) has done: 'Your current lookup baseline is still far above the target MAE, so we make the smallest changes that legitimately improve accuracy while keeping the same “groupby mean lookup + staged fallback + nearest-pressure quantization + `u_out==1 -> 0`” core. The main issue is that `df_test["id"]` (and `sample_submission`) contain repeated ids (1..2000 per breath), so the produced submission is malformed for this competition and score extremely poorly; fixing the `id` alignment to use the real row-unique `id` from the file (or fallback to sample submission order) is essential. Next, we minimally strengthen the time-series lookup without changing the approach by adding the next-step lag (`u_in_bin_next1`) to the primary key (with staged fallbacks that drop it first), which typically reduces ambiguity at a given step. These changes keep runtime low, preserve your quantization and `u_out` handling, and ensure a valid `submission.csv`.'
- What this solution (achieved 1.93672) has done: 'Your current score is far above the target (lower is better), so we should improve accuracy with the smallest changes that keep the same “groupby mean lookup + staged fallback + nearest-pressure quantization + `u_out==1 -> 0`” core. The biggest remaining issue is that your submission `id` selection logic can silently fall back to `sample_submission` because `df_test["id"]` is not unique (it repeats 1..80 within each breath), which makes the submission invalidly aligned to test rows and can inflate MAE a lot. I change this to always use the row-wise `id` from `test.csv` (the competition’s required `id` is unique across the whole file) and add a strict alignment sanity-check so the output ordering matches the test file. I also preserve your existing lookup but add one minimal, standard “within-breath cumulative volume” feature (`u_in_cum`) into the primary key with staged fallbacks (drop it first) to reduce ambiguity at the same step without changing the approach.'
- What this solution (achieved 1.93672) has done: 'We keep your exact “groupby mean lookup + staged fallback + nearest-pressure quantization + `u_out==1 -> 0`” approach, but fix two small issues that commonly inflate MAE for this competition: (1) the `id` column must be used in the original test row order (don’t sort by `id`, which can silently misalign rows if `id` is not strictly monotonic in the loaded file), and (2) enforce inspiratory-only scoring semantics more faithfully by also zeroing predictions for timesteps after the first `u_out==1` within each breath (once exhalation starts, pressure is not scored and tends to follow a different regime). These are minimal post-processing/alignment changes that don’t alter your core feature engineering or lookup tables, but should move the MAE down toward your target. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.93672) has done: 'Your current score is still much worse than the target (MAE lower is better), so we should make the smallest changes that plausibly improve accuracy while keeping the same “groupby mean lookup + staged fallback + nearest-pressure quantization + expiratory masking” core. The main issue I address is that forcing pressure to 0 after exhalation starts is likely incorrect (expiratory rows are *ignored*, not required to be 0), and setting them to 0 can distort lookup fallbacks and create unnecessary error if Kaggle’s mask is only `u_out==0` (as described). I replace that with a safe, metric-aligned post-process: only fill expiratory (`u_out==1`) predictions with a within-breath forward-filled last inspiratory prediction (or 0 if none), which avoids injecting an arbitrary 0 while keeping inspiratory predictions unchanged. I also make the nearest-pressure mapping vectorized (same semantics, faster) to stay within time limits without changing the model logic.'
- What this solution (achieved 1.54322) has done: 'Your current MAE (1.9367, lower is better) is still far from the target, so we should improve accuracy with minimal, metric-aligned changes while keeping your same “groupby mean lookup + staged fallback + nearest-pressure quantization + expiratory handling” core. The biggest safe win is to stop using `u_in_cum` as an exact-match key (it explodes sparsity and increases missing lookups) and instead use a lightly binned cumulative feature (`u_in_cum_bin`) so the primary table still matches often. Second, we make `u_in_cum` for test consistent with train by computing it from `u_in` (as before) but binning after rounding in both sets, preserving semantics while improving generalization. Everything else (lags, fallback chain, nearest-pressure mapping, and submission writing) remains the same and still produces a valid `submission.csv`.'

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
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(TRAIN_PATH)

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


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx]
    lower = sorted_pressures[np.clip(idx - 1, 0, total_pressures_len - 1)]
    choose_lower = (idx > 0) & (np.abs(preds - lower) < np.abs(upper - preds))
    out = upper.copy()
    out[choose_lower] = lower[choose_lower]
    return out


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
    output = pd.read_csv(SAMPLE_SUB_PATH)
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
df_test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_feat = df_train.copy()
test_feat = df_test.copy()

train_feat["step"] = (
    train_feat.groupby("breath_id")["time_step"].rank(method="first").astype("int16")
    - 1
)
test_feat["step"] = (
    test_feat.groupby("breath_id")["time_step"].rank(method="first").astype("int16") - 1
)

UIN_BIN = 2.0  # keep same binning / core logic
train_feat["u_in_bin"] = (train_feat["u_in"] / UIN_BIN).round().astype("int16")
test_feat["u_in_bin"] = (test_feat["u_in"] / UIN_BIN).round().astype("int16")

train_feat["u_in_bin_next1"] = (
    train_feat.groupby("breath_id")["u_in_bin"].shift(-1).fillna(0).astype("int16")
)
test_feat["u_in_bin_next1"] = (
    test_feat.groupby("breath_id")["u_in_bin"].shift(-1).fillna(0).astype("int16")
)

for lag in (1, 2):
    col = f"u_in_bin_prev{lag}"
    train_feat[col] = (
        train_feat.groupby("breath_id")["u_in_bin"].shift(lag).fillna(0).astype("int16")
    )
    test_feat[col] = (
        test_feat.groupby("breath_id")["u_in_bin"].shift(lag).fillna(0).astype("int16")
    )

UIN_CUM_BIN = 50.0  # coarse bin to reduce key sparsity; still keeps "within-breath cumulative volume" signal
train_feat["u_in_cum_bin"] = (
    (train_feat.groupby("breath_id")["u_in"].cumsum().round() / UIN_CUM_BIN)
    .round()
    .astype("int16")
)
test_feat["u_in_cum_bin"] = (
    (test_feat.groupby("breath_id")["u_in"].cumsum().round() / UIN_CUM_BIN)
    .round()
    .astype("int16")
)

grp_primary = train_feat.groupby(
    [
        "R",
        "C",
        "step",
        "u_out",
        "u_in_bin",
        "u_in_bin_prev1",
        "u_in_bin_prev2",
        "u_in_bin_next1",
        "u_in_cum_bin",
    ],
    sort=False,
)["pressure"].mean()

grp_fb_drop_cum = train_feat.groupby(
    [
        "R",
        "C",
        "step",
        "u_out",
        "u_in_bin",
        "u_in_bin_prev1",
        "u_in_bin_prev2",
        "u_in_bin_next1",
    ],
    sort=False,
)["pressure"].mean()

grp_fb_next_drop = train_feat.groupby(
    ["R", "C", "step", "u_out", "u_in_bin", "u_in_bin_prev1", "u_in_bin_prev2"],
    sort=False,
)["pressure"].mean()

grp_fb1 = train_feat.groupby(
    ["R", "C", "step", "u_out", "u_in_bin", "u_in_bin_prev1"], sort=False
)["pressure"].mean()

grp_fb2 = train_feat.groupby(["R", "C", "step", "u_out", "u_in_bin"], sort=False)[
    "pressure"
].mean()

grp_fb_drop_uout = train_feat.groupby(
    ["R", "C", "step", "u_in_bin", "u_in_bin_prev1", "u_in_bin_prev2"], sort=False
)["pressure"].mean()

grp_fallback = train_feat.groupby(["R", "C", "step"], sort=False)["pressure"].mean()
global_mean = float(train_feat["pressure"].mean())

idx_primary = pd.MultiIndex.from_frame(
    test_feat[
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_bin",
            "u_in_bin_prev1",
            "u_in_bin_prev2",
            "u_in_bin_next1",
            "u_in_cum_bin",
        ]
    ]
)
pred_s = grp_primary.reindex(idx_primary).astype("float64")
missing = pred_s.isna().to_numpy()
pred = pred_s.to_numpy(dtype="float64")

if missing.any():
    idx_fb_drop_cum = pd.MultiIndex.from_frame(
        test_feat.loc[
            missing,
            [
                "R",
                "C",
                "step",
                "u_out",
                "u_in_bin",
                "u_in_bin_prev1",
                "u_in_bin_prev2",
                "u_in_bin_next1",
            ],
        ]
    )
    fb_drop_cum_vals = (
        grp_fb_drop_cum.reindex(idx_fb_drop_cum).astype("float64").to_numpy()
    )
    pred[missing] = fb_drop_cum_vals
    missing = np.isnan(pred)

if missing.any():
    idx_fb_next_drop = pd.MultiIndex.from_frame(
        test_feat.loc[
            missing,
            ["R", "C", "step", "u_out", "u_in_bin", "u_in_bin_prev1", "u_in_bin_prev2"],
        ]
    )
    fb_next_drop_vals = (
        grp_fb_next_drop.reindex(idx_fb_next_drop).astype("float64").to_numpy()
    )
    pred[missing] = fb_next_drop_vals
    missing = np.isnan(pred)

if missing.any():
    idx_fb1 = pd.MultiIndex.from_frame(
        test_feat.loc[
            missing, ["R", "C", "step", "u_out", "u_in_bin", "u_in_bin_prev1"]
        ]
    )
    fb1_vals = grp_fb1.reindex(idx_fb1).astype("float64").to_numpy()
    pred[missing] = fb1_vals
    missing = np.isnan(pred)

if missing.any():
    idx_fb2 = pd.MultiIndex.from_frame(
        test_feat.loc[missing, ["R", "C", "step", "u_out", "u_in_bin"]]
    )
    fb2_vals = grp_fb2.reindex(idx_fb2).astype("float64").to_numpy()
    pred[missing] = fb2_vals
    missing = np.isnan(pred)

if missing.any():
    idx_fb_drop_uout = pd.MultiIndex.from_frame(
        test_feat.loc[
            missing, ["R", "C", "step", "u_in_bin", "u_in_bin_prev1", "u_in_bin_prev2"]
        ]
    )
    fb_drop_uout_vals = (
        grp_fb_drop_uout.reindex(idx_fb_drop_uout).astype("float64").to_numpy()
    )
    pred[missing] = fb_drop_uout_vals
    missing = np.isnan(pred)

if missing.any():
    idx_fallback = pd.MultiIndex.from_frame(test_feat.loc[missing, ["R", "C", "step"]])
    fallback_vals = grp_fallback.reindex(idx_fallback).astype("float64").to_numpy()
    pred[missing] = fallback_vals
    missing = np.isnan(pred)

if np.isnan(pred).any():
    pred = pd.Series(pred).fillna(global_mean).to_numpy(dtype="float64")

pred = find_nearest_vec(pred).astype("float64")

u_out = df_test["u_out"].to_numpy(dtype=np.int8)
pred2 = pred.copy()
if (u_out == 1).any():
    tmp = pd.DataFrame(
        {
            "breath_id": df_test["breath_id"].to_numpy(),
            "u_out": u_out,
            "pred": pred2,
        }
    )
    tmp.loc[tmp["u_out"] == 1, "pred"] = np.nan
    tmp["pred"] = tmp.groupby("breath_id")["pred"].ffill()
    tmp["pred"] = tmp["pred"].fillna(0.0)
    pred2 = tmp["pred"].to_numpy(dtype="float64")

sub_ids = df_test["id"].to_numpy()
assert len(sub_ids) == len(pred2) == len(df_test), "Prediction length mismatch vs test."

sub = pd.DataFrame({"id": sub_ids, "pressure": pred2})

assert list(sub.columns) == ["id", "pressure"]
assert sub.shape[0] == df_test.shape[0]

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("pressure min/max:", float(sub["pressure"].min()), float(sub["pressure"].max()))
print("u_out==1 fraction:", float(df_test["u_out"].mean()))
print("NaN in pred:", int(np.isnan(pred2).sum()))
print("Test id unique:", bool(pd.Series(sub_ids).is_unique))
print("Submission id range:", int(sub["id"].min()), int(sub["id"].max()))
