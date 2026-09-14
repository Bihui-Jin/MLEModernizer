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

0.2304440568323064

# 6. Current score

3.89598

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13673) has done: 'The crash comes from trying to read two external blend files (`../input/gb-blending/...`) that are not present in your environment, so no submission is produced. I keep your nearest-pressure post-processing intact, but replace the missing-file blend step with a simple, deterministic baseline that uses only the provided train/test files. Specifically, we predict pressure as the mean training pressure for each `(R, C, time_step)` combination (a lightweight lookup), then fall back to the global mean if a key is missing, and finally snap to the nearest valid pressure value using your `find_nearest`. This runs end-to-end quickly and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 9.91828) has done: 'Your current score (8.13673, lower-is-better) is far from the target (0.23044), so we need a real accuracy improvement while still keeping the solution lightweight and preserving your “nearest valid pressure” post-processing. The biggest issue is that predicting by `(R, C, time_step)` ignores the dominant driver `u_in` (and to a lesser extent `u_out`), which causes very large MAE. I keep the same lookup-based core approach, but change the key to include `u_in` and `u_out` and use a robust median aggregation, with a deterministic fallback hierarchy if an exact key is missing. This remains purely train→test aggregation (no new model/loops) and still snaps predictions using your existing `find_nearest`, and it writes a valid `submission.csv`.'
- What this solution (achieved 9.911) has done: 'Your current MAE (9.918) is far above the target (0.230), so we need a legitimate accuracy lift while keeping your “train-lookup → merge → nearest-pressure snap” core logic intact. The biggest weakness is using an exact `u_in` float in the join key, which causes massive key-misses in test (float mismatch / sparsity) and forces fallbacks that ignore the dominant signal. I keep the same aggregation approach but add a minimal, deterministic `u_in` quantization for the lookup keys (while still using true `u_in` as a fallback), so more rows hit a meaningful median without changing the modeling paradigm. I also ensure `sorted_pressures` is strictly sorted unique and keep the same nearest snapping and submission writing.'
- What this solution (achieved 3.82253) has done: 'We need to move MAE down (lower is better) from ~9.911 toward 0.230, so we should improve accuracy while keeping your current “train lookup → merge → snap to nearest valid pressure” core logic. The biggest remaining issue is that exact matching on `time_step` and a single global `u_in` bin still causes many misses and overly-blunt fallbacks; we can fix this minimally by (1) quantizing `time_step` to a small, physically-meaningful grid and (2) using a slightly finer `u_in` bin, while keeping the same median-aggregation + fallback hierarchy. To avoid accidental misalignment, we also ensure predictions are written in the same row order as `test.csv` by carrying an index column through merges. These are deterministic, lightweight changes that should materially reduce the MAE without changing the overall approach.'
- What this solution (achieved 3.82315) has done: 'We need to reduce MAE (lower is better) from 3.82253 toward 0.23044, so we should improve the hit-rate and sharpness of your existing “train groupby lookup → merge → fallback → snap to nearest valid pressure” pipeline without changing its core approach. The biggest remaining weakness is using `round()` binning, which can push values across bin boundaries and create mismatches between train/test keys; switching to deterministic `floor()` binning typically increases exact-key matches and reduces reliance on blunt fallbacks. To further reduce fallback error while staying in the same lookup paradigm, we add one intermediate fallback that uses the binned `u_in` signal but drops `u_out` (often noisy for inspiration), before falling back to `(R,C,time_bin,u_out)` and `(R,C,time_bin)`. All changes are lightweight, deterministic, preserve your nearest-pressure snapping, and still write a valid `submission.csv`.'
- What this solution (achieved 3.90438) has done: 'Your current MAE (3.823, lower-is-better) is still far above the target (0.230), so we should improve accuracy while staying in your exact same “train groupby lookup → merge → fallback → snap to nearest valid pressure” paradigm. The biggest remaining error driver is that the lookup key still misses many test rows due to too-fine `u_in_bin` and `time_bin`, forcing weaker fallbacks; we slightly coarsen binning to increase exact-key hit-rate (less fallback), which usually lowers MAE in this competition for pure-lookup methods. To keep changes minimal and stable, we only adjust the two bin sizes and leave your aggregation (median), fallback hierarchy, and `find_nearest` snapping intact. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.89795) has done: 'Your current MAE (3.90438, lower-is-better) is still far above the target (0.23044), so we should improve accuracy while keeping your same “train groupby lookup → merge → fallback → snap to nearest valid pressure” core logic. The biggest issue is key sparsity: binning `u_in` and `time_step` at fixed steps can still miss many test rows, forcing weaker fallbacks; a small but effective improvement is to use *dual* binnings (a fine and a coarse grid) and prefer fine matches, then fall back to coarse matches before the existing weaker fallbacks. This preserves the exact same approach (deterministic median lookups + hierarchical fill + nearest snapping) and stays fast, but increases hit-rate and reduces fallback error. I keep I/O paths and submission schema the same and still write `submission.csv`.'
- What this solution (achieved 3.89795) has done: 'To move MAE down from 3.89795 toward the 0.23044 target while preserving your exact “train groupby median lookup → hierarchical fallback → snap to nearest valid pressure” logic, the smallest reliable gain is to reduce lookup misses without changing the paradigm. I add two extra intermediate fallbacks that use *mixed* bin granularities (fine time + coarse u_in, and coarse time + fine u_in) so more test rows find a reasonable median before dropping to weaker keys. I keep your existing fine/coarse binnings, merge-based implementation, and final `find_nearest` snapping unchanged in semantics, and still write `submission.csv` in the correct `id,pressure` format. This is deterministic, lightweight, and should improve hit-rate with minimal code changes.'
- What this solution (achieved 3.89598) has done: 'To move MAE down from 3.89795 toward the 0.23044 target while keeping your exact “train median lookup → hierarchical fallback → nearest-pressure snap” core logic, the smallest reliable lift is to reduce lookup misses caused by the `u_in` binning boundaries. I keep your existing fine/coarse bin features and merge pipeline, but add two additional *shifted* `u_in` bin versions (half-bin offsets) and corresponding lookup tables, then insert them early in the fallback chain so more test rows hit a meaningful median before dropping to weaker keys. This is deterministic, lightweight (a few extra groupbys/merges), preserves evaluation semantics, and still writes a valid `submission.csv` with `id,pressure`. No model/loops/loss changes are introduced.'

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
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(np.unique(unique_pressures))
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




## === cell 2
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
        weight1 = 0.6
        weight2 = 0.4
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**2
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
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4

    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

TIME_BIN_FINE = 0.02
UIN_BIN_FINE = 0.5
TIME_BIN_COARSE = 0.03
UIN_BIN_COARSE = 1.0

df_train = df_train.copy()
df_test = df_test.copy()

df_test["_row"] = np.arange(len(df_test), dtype=np.int64)

df_train["time_bin_f"] = np.floor(df_train["time_step"] / TIME_BIN_FINE).astype(
    np.int16
)
df_test["time_bin_f"] = np.floor(df_test["time_step"] / TIME_BIN_FINE).astype(np.int16)
df_train["u_in_bin_f"] = np.floor(df_train["u_in"] / UIN_BIN_FINE).astype(np.int16)
df_test["u_in_bin_f"] = np.floor(df_test["u_in"] / UIN_BIN_FINE).astype(np.int16)

df_train["time_bin_c"] = np.floor(df_train["time_step"] / TIME_BIN_COARSE).astype(
    np.int16
)
df_test["time_bin_c"] = np.floor(df_test["time_step"] / TIME_BIN_COARSE).astype(
    np.int16
)
df_train["u_in_bin_c"] = np.floor(df_train["u_in"] / UIN_BIN_COARSE).astype(np.int16)
df_test["u_in_bin_c"] = np.floor(df_test["u_in"] / UIN_BIN_COARSE).astype(np.int16)

df_train["u_in_bin_f_s"] = np.floor(
    (df_train["u_in"] + (UIN_BIN_FINE / 2.0)) / UIN_BIN_FINE
).astype(np.int16)
df_test["u_in_bin_f_s"] = np.floor(
    (df_test["u_in"] + (UIN_BIN_FINE / 2.0)) / UIN_BIN_FINE
).astype(np.int16)

df_train["u_in_bin_c_s"] = np.floor(
    (df_train["u_in"] + (UIN_BIN_COARSE / 2.0)) / UIN_BIN_COARSE
).astype(np.int16)
df_test["u_in_bin_c_s"] = np.floor(
    (df_test["u_in"] + (UIN_BIN_COARSE / 2.0)) / UIN_BIN_COARSE
).astype(np.int16)

key_full_f = ["R", "C", "time_bin_f", "u_out", "u_in_bin_f"]
key_rc_t_uin_f = ["R", "C", "time_bin_f", "u_in_bin_f"]
key_rc_t_uout_f = ["R", "C", "time_bin_f", "u_out"]
key_rc_t_f = ["R", "C", "time_bin_f"]

med_full_f = (
    df_train.groupby(key_full_f, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_full_f"})
)
test_with_pred = df_test.merge(med_full_f, on=key_full_f, how="left")

key_full_f_s = ["R", "C", "time_bin_f", "u_out", "u_in_bin_f_s"]
med_full_f_s = (
    df_train.groupby(key_full_f_s, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_full_f_s"})
)
test_with_pred = test_with_pred.merge(med_full_f_s, on=key_full_f_s, how="left")

med_rc_t_uin_f = (
    df_train.groupby(key_rc_t_uin_f, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_uin_f"})
)
test_with_pred = test_with_pred.merge(med_rc_t_uin_f, on=key_rc_t_uin_f, how="left")

key_rc_t_uin_f_s = ["R", "C", "time_bin_f", "u_in_bin_f_s"]
med_rc_t_uin_f_s = (
    df_train.groupby(key_rc_t_uin_f_s, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_uin_f_s"})
)
test_with_pred = test_with_pred.merge(med_rc_t_uin_f_s, on=key_rc_t_uin_f_s, how="left")

med_rc_t_uout_f = (
    df_train.groupby(key_rc_t_uout_f, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_uout_f"})
)
test_with_pred = test_with_pred.merge(med_rc_t_uout_f, on=key_rc_t_uout_f, how="left")

med_rc_t_f = (
    df_train.groupby(key_rc_t_f, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t_f"})
)
test_with_pred = test_with_pred.merge(med_rc_t_f, on=key_rc_t_f, how="left")

key_full_c = ["R", "C", "time_bin_c", "u_out", "u_in_bin_c"]
key_rc_t_uin_c = ["R", "C", "time_bin_c", "u_in_bin_c"]
key_rc_t_uout_c = ["R", "C", "time_bin_c", "u_out"]
key_rc_t_c = ["R", "C", "time_bin_c"]

med_full_c = (
    df_train.groupby(key_full_c, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_full_c"})
)
test_with_pred = test_with_pred.merge(med_full_c, on=key_full_c, how="left")

key_full_c_s = ["R", "C", "time_bin_c", "u_out", "u_in_bin_c_s"]
med_full_c_s = (
    df_train.groupby(key_full_c_s, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_full_c_s"})
)
test_with_pred = test_with_pred.merge(med_full_c_s, on=key_full_c_s, how="left")

med_rc_t_uin_c = (
    df_train.groupby(key_rc_t_uin_c, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_uin_c"})
)
test_with_pred = test_with_pred.merge(med_rc_t_uin_c, on=key_rc_t_uin_c, how="left")

key_rc_t_uin_c_s = ["R", "C", "time_bin_c", "u_in_bin_c_s"]
med_rc_t_uin_c_s = (
    df_train.groupby(key_rc_t_uin_c_s, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_uin_c_s"})
)
test_with_pred = test_with_pred.merge(med_rc_t_uin_c_s, on=key_rc_t_uin_c_s, how="left")

med_rc_t_uout_c = (
    df_train.groupby(key_rc_t_uout_c, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_uout_c"})
)
test_with_pred = test_with_pred.merge(med_rc_t_uout_c, on=key_rc_t_uout_c, how="left")

med_rc_t_c = (
    df_train.groupby(key_rc_t_c, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t_c"})
)
test_with_pred = test_with_pred.merge(med_rc_t_c, on=key_rc_t_c, how="left")

key_full_tf_uc = ["R", "C", "time_bin_f", "u_out", "u_in_bin_c"]
key_full_tc_uf = ["R", "C", "time_bin_c", "u_out", "u_in_bin_f"]

med_full_tf_uc = (
    df_train.groupby(key_full_tf_uc, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_full_tf_uc"})
)
test_with_pred = test_with_pred.merge(med_full_tf_uc, on=key_full_tf_uc, how="left")

med_full_tc_uf = (
    df_train.groupby(key_full_tc_uf, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_full_tc_uf"})
)
test_with_pred = test_with_pred.merge(med_full_tc_uf, on=key_full_tc_uf, how="left")

global_median = float(df_train["pressure"].median())

test_with_pred["pressure_pred"] = (
    test_with_pred["p_full_f"]
    .fillna(test_with_pred["p_full_f_s"])
    .fillna(test_with_pred["p_full_c"])
    .fillna(test_with_pred["p_full_c_s"])
    .fillna(test_with_pred["p_full_tf_uc"])
    .fillna(test_with_pred["p_full_tc_uf"])
    .fillna(test_with_pred["p_uin_f"])
    .fillna(test_with_pred["p_uin_f_s"])
    .fillna(test_with_pred["p_uin_c"])
    .fillna(test_with_pred["p_uin_c_s"])
    .fillna(test_with_pred["p_uout_f"])
    .fillna(test_with_pred["p_uout_c"])
    .fillna(test_with_pred["p_rc_t_f"])
    .fillna(test_with_pred["p_rc_t_c"])
    .fillna(global_median)
)

test_with_pred = test_with_pred.sort_values("_row", kind="mergesort")

submission = pd.DataFrame(
    {
        "id": test_with_pred["id"].astype(np.int64),
        "pressure": test_with_pred["pressure_pred"].astype(np.float64),
    }
)

submission["pressure"] = submission["pressure"].apply(find_nearest)
submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Pressure range:",
    float(submission.pressure.min()),
    float(submission.pressure.max()),
)
