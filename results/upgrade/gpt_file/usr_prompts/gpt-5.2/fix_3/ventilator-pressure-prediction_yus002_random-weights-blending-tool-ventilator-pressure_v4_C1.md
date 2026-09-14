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

0.1513939814206466

# 6. Current score

2.94989

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.31565) has done: 'The crash happens because `../input/gb-rwbt-files` does not exist in your environment, so the ensemble code finds zero files and divides by zero. I make `g()` robust by (1) validating the directory and file list, and (2) falling back to generating predictions from the provided train/test data using a simple groupwise median baseline (by `R`, `C`, and discretized `u_in`/`u_out`) so a valid submission is always produced. This keeps the overall pipeline intent (produce a pressure prediction vector and snap to nearest allowed pressure) while ensuring it runs end-to-end and writes a `.csv` submission. The submission filename be a standard `submission.csv` to satisfy Kaggle requirements.'
- What this solution (achieved 2.94989) has done: 'Your current score (6.31565 MAE) is far worse than the target (~0.151), so we should meaningfully improve predictions while keeping the same overall “non-ML baseline pipeline” core idea. The simplest high-impact fix is to make the fallback baseline respect the competition’s key structure: pressure depends heavily on within-breath dynamics, so we add lightweight time-series features (lagged `u_in`, cumulative integral/mean) and predict using a median lookup keyed by (`R`,`C`,`u_out`,`time_step`,`u_in_bin`,`u_in_lag_bin`). We keep your snapping-to-nearest-allowed-pressure exactly as-is, and we still fall back only when the external directory doesn’t exist, producing `submission.csv` end-to-end. These changes are minimal (still pure pandas groupby-median, no new model/training loop) but should move MAE substantially toward the target.'

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


def _fallback_baseline_submission():
    """
    Score-improvement change (still same core logic: groupwise median baseline + snap):
    Add minimal within-breath time-series features and a more specific median lookup key.
    This should reduce MAE vs a static (R,C,u_out,u_in) median while preserving semantics.
    """
    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

    def add_features(df, is_train=True):
        base_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
        if is_train:
            base_cols = base_cols + ["pressure"]
        out = df[base_cols].copy()

        out["t"] = np.round(out["time_step"], 2).astype(np.float32)

        out["u_in_bin"] = np.round(out["u_in"]).astype(np.int16)

        out["u_in_lag1"] = out.groupby("breath_id", sort=False)["u_in"].shift(1)
        out["u_in_lag1"] = out["u_in_lag1"].fillna(0.0)
        out["u_in_lag1_bin"] = np.round(out["u_in_lag1"]).astype(np.int16)

        dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        out["u_in_cum"] = (
            (out["u_in"] * dt).groupby(out["breath_id"], sort=False).cumsum()
        )
        out["u_in_cum_bin"] = np.round(out["u_in_cum"] / 10.0).astype(
            np.int16
        )  # coarse bin

        return out

    train_feat = add_features(df_train, is_train=True)
    test_feat = add_features(df_test, is_train=False)

    keys_full = ["R", "C", "u_out", "t", "u_in_bin", "u_in_lag1_bin"]
    med_full = train_feat.groupby(keys_full, sort=False)["pressure"].median()

    keys_mid = ["R", "C", "u_out", "t", "u_in_bin"]
    med_mid = train_feat.groupby(keys_mid, sort=False)["pressure"].median()

    keys_simple = ["R", "C", "u_out", "t"]
    med_simple = train_feat.groupby(keys_simple, sort=False)["pressure"].median()

    keys_rc = ["R", "C", "u_out"]
    med_rc = train_feat.groupby(keys_rc, sort=False)["pressure"].median()

    global_med = float(train_feat["pressure"].median())

    idx_full = pd.MultiIndex.from_frame(test_feat[keys_full])
    pred = med_full.reindex(idx_full).to_numpy()

    miss = np.isnan(pred)
    if miss.any():
        idx_mid = pd.MultiIndex.from_frame(test_feat.loc[miss, keys_mid])
        pred[miss] = med_mid.reindex(idx_mid).to_numpy()
        miss = np.isnan(pred)

    if miss.any():
        idx_s = pd.MultiIndex.from_frame(test_feat.loc[miss, keys_simple])
        pred[miss] = med_simple.reindex(idx_s).to_numpy()
        miss = np.isnan(pred)

    if miss.any():
        idx_rc = pd.MultiIndex.from_frame(test_feat.loc[miss, keys_rc])
        pred[miss] = med_rc.reindex(idx_rc).to_numpy()
        miss = np.isnan(pred)

    if miss.any():
        pred[miss] = global_med

    sub["pressure"] = pred
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub = sub[["id", "pressure"]]
    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    if (dp is None) or (not os.path.isdir(dp)):
        print(
            f"Input directory not found: {dp}. Using fallback baseline to create submission.csv"
        )
        return _fallback_baseline_submission()

    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    file_count = len(l)
    if file_count == 0:
        print(
            f"No files found in: {dp}. Using fallback baseline to create submission.csv"
        )
        return _fallback_baseline_submission()

    loop_time = 500 // file_count
    if loop_time <= 0:
        loop_time = 1  # safety; still preserves intent of looping/aggregation

    splits = file_count // 2
    if splits <= 0:
        splits = 1

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
    for j in range(loop_time):
        weight = []
        set_seed(j)
        for k in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            for k in range(len(weight)):
                weight[k] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for k in range(len(flist)):
            temp += flist[k] * weight[k]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output




## === cell 2
g("../input/gb-rwbt-files")
