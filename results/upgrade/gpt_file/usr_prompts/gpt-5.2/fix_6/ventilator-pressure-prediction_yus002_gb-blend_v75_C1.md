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

0.1606358961038999

# 6. Current score

6.64226

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.6492) has done: 'Your notebook fails because it tries to read external submissions from `../input/gb-blending/...`, which don’t exist in this environment, so no submission file is ever produced. I keep the existing blending/rounding logic intact, but make it robust by automatically finding any submission-like CSVs under `../input` (excluding the competition’s own sample submission) and blending them if present. If none are found, it fall back to writing a valid baseline submission (all zeros) in the required `id,pressure` format so you always get a `.csv` output. I also fix the cell numbering (start at 1) and ensure the output filename ends with `.csv`.'
- What this solution (achieved 6.54001) has done: 'Your current score is extremely poor because the notebook mostly falls back to the all-zero `sample_submission`, which is not a meaningful predictor for this task. To move toward the target (lower MAE), the smallest legitimate improvement without changing your blending core is to generate one strong in-notebook baseline submission from `train.csv` using a simple, fast, physically-relevant mapping (per-(R,C,time_step,u_in,u_out) median pressure), then let your existing blending/nearest-pressure rounding operate on it. This keeps your “blend/round-to-known-pressures” semantics intact while ensuring you always have at least one non-trivial candidate to blend, even when no external submissions exist. The code below adds that baseline as a candidate CSV and blends it with any other found submission-like CSVs when available, otherwise uses it alone.'
- What this solution (achieved 6.57085) has done: 'Your current score (6.54 MAE) is far worse than the target (0.1606), so we should make a small, legitimate improvement without changing your blending/nearest-pressure rounding semantics. The main issue is that your baseline predictor is too weak because it relies heavily on exact `(R,C,time_step,u_in,u_out)` matches, which are sparse; we can improve coverage by adding lightweight “history” features (`u_in`/`u_out` lags and cumulative integral) and still use the same groupby-median mapping approach. This keeps the core logic (median lookup baseline + optional blend + rounding to known pressures) intact, but should substantially reduce MAE by conditioning on breath dynamics. We also ensure strict alignment to `sample_submission` ids and keep output as `submission.csv`.'
- What this solution (achieved 6.64226) has done: 'Your current MAE (6.57) is far worse than the target (0.1606), so we need a real improvement while keeping your existing “median-lookup baseline → optional blend → round-to-known-pressures” core intact. The smallest high-impact fix is to (1) restrict training rows used for the median maps to the inspiratory phase only (`u_out==0`), matching the evaluation (expiratory is not scored), and (2) add a very lightweight, physically-relevant key: the **cumulative sum of `u_in`** (`u_in_cumsum`, discretized), which improves lookup coverage without changing the approach. We keep your existing lag + area features and the same fallback hierarchy, but compute medians on inspiratory rows and add one more groupby key to reduce ambiguity. Finally, we keep submission alignment to `sample_submission.csv` and still round predictions via `find_nearest`, producing `submission.csv` end-to-end.'
- What this solution (achieved 6.64226) has done: 'Your current MAE (6.64) is far worse than the target (0.1606), so we should improve predictions while keeping your core “median lookup baseline → optional blend → round-to-known-pressures” logic intact. The biggest leak in performance is that you compute medians only on inspiratory rows (`u_out==0`) but then apply them to *all* test rows including expiratory; because expiratory isn’t scored, the safest minimal improvement is to force expiratory predictions to a constant (global inspiratory median) to avoid injecting noise. Additionally, your blending step currently fills missing predictions with `0.0`, which is an unnecessarily bad fallback; changing that fallback to the same global inspiratory median improves robustness without changing semantics. These are small, metric-aligned changes that should reduce MAE substantially while staying within your existing approach.'

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
    loop_time = 150
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
    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
usecols_test = ["breath_id", "id", "R", "C", "time_step", "u_in", "u_out"]

df_test = pd.read_csv(test_path, usecols=usecols_test)

df_train_key = df_train[usecols_train].copy()
df_test_key = df_test.copy()

df_train_key = df_train_key.sort_values(["breath_id", "time_step"], kind="mergesort")
df_test_key = df_test_key.sort_values(["breath_id", "time_step"], kind="mergesort")

df_train_key["time_step"] = df_train_key["time_step"].round(3)
df_test_key["time_step"] = df_test_key["time_step"].round(3)

for lag in (1, 2):
    df_train_key[f"u_in_lag{lag}"] = (
        df_train_key.groupby("breath_id")["u_in"].shift(lag).fillna(0.0)
    )
    df_train_key[f"u_out_lag{lag}"] = (
        df_train_key.groupby("breath_id")["u_out"].shift(lag).fillna(0).astype(np.int8)
    )

    df_test_key[f"u_in_lag{lag}"] = (
        df_test_key.groupby("breath_id")["u_in"].shift(lag).fillna(0.0)
    )
    df_test_key[f"u_out_lag{lag}"] = (
        df_test_key.groupby("breath_id")["u_out"].shift(lag).fillna(0).astype(np.int8)
    )

dt_train = (
    df_train_key.groupby("breath_id")["time_step"]
    .diff()
    .fillna(df_train_key["time_step"])
    .clip(lower=0.0)
)
dt_test = (
    df_test_key.groupby("breath_id")["time_step"]
    .diff()
    .fillna(df_test_key["time_step"])
    .clip(lower=0.0)
)

df_train_key["u_in_area"] = (
    (df_train_key["u_in"] * dt_train).groupby(df_train_key["breath_id"]).cumsum()
)
df_test_key["u_in_area"] = (
    (df_test_key["u_in"] * dt_test).groupby(df_test_key["breath_id"]).cumsum()
)

df_train_key["u_in_cumsum"] = df_train_key.groupby("breath_id")["u_in"].cumsum()
df_test_key["u_in_cumsum"] = df_test_key.groupby("breath_id")["u_in"].cumsum()

df_train_key["u_in_area"] = df_train_key["u_in_area"].round(2)
df_test_key["u_in_area"] = df_test_key["u_in_area"].round(2)

df_train_key["u_in_cumsum"] = df_train_key["u_in_cumsum"].round(1)
df_test_key["u_in_cumsum"] = df_test_key["u_in_cumsum"].round(1)

for c in ["u_in", "u_in_lag1", "u_in_lag2"]:
    df_train_key[c] = df_train_key[c].round(1)
    df_test_key[c] = df_test_key[c].round(1)

train_key_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_lag2",
    "u_out_lag2",
    "u_in_area",
    "u_in_cumsum",
]

df_train_insp = df_train_key[df_train_key["u_out"] == 0].copy()

med_map = (
    df_train_insp.groupby(train_key_cols, sort=False)["pressure"].median().reset_index()
)

test_pred = df_test_key.merge(med_map, on=train_key_cols, how="left")

rc_t_map = (
    df_train_insp.groupby(["R", "C", "time_step", "u_out"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_t"})
)
rc_map = (
    df_train_insp.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc"})
)
global_med = float(df_train_insp["pressure"].median())

test_pred = test_pred.merge(
    rc_t_map, on=["R", "C", "time_step", "u_out"], how="left"
).merge(rc_map, on=["R", "C"], how="left")

test_pred["pressure"] = (
    test_pred["pressure"]
    .fillna(test_pred["pressure_rc_t"])
    .fillna(test_pred["pressure_rc"])
    .fillna(global_med)
).astype(float)

test_pred.loc[test_pred["u_out"] == 1, "pressure"] = global_med

test_pred["pressure"] = test_pred["pressure"].apply(find_nearest)

baseline_sub = pd.read_csv(sample_path)[["id"]].merge(
    test_pred[["id", "pressure"]], on="id", how="left"
)
baseline_sub["pressure"] = baseline_sub["pressure"].fillna(global_med).astype(float)
baseline_sub["pressure"] = baseline_sub["pressure"].apply(find_nearest)
baseline_sub.to_csv("baseline_median.csv", index=False)

print("Wrote baseline_median.csv with shape:", baseline_sub.shape)
print("global_med (inspiratory):", global_med)




## === cell 3
def _is_submission_like_csv(path):
    base = os.path.basename(path).lower()
    if base == "sample_submission.csv":
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = [c.lower() for c in df.columns]
    return ("id" in cols) and ("pressure" in cols)


candidate_paths = sorted(glob.glob("../input/**/*.csv", recursive=True))
candidate_paths = [p for p in candidate_paths if _is_submission_like_csv(p)]
candidate_paths = [
    "baseline_median.csv"
] + candidate_paths  # baseline first (always present)

sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

if len(candidate_paths) >= 2:
    _ = blend(candidate_paths[0], candidate_paths[1])
    blended = pd.read_csv("blend.csv")
    blended = blended[["id", "pressure"]]
    merged = sub[["id"]].merge(blended, on="id", how="left")
    merged["pressure"] = merged["pressure"].fillna(global_med).astype(float)
    merged["pressure"] = merged["pressure"].apply(find_nearest)
    merged.to_csv("submission.csv", index=False)
else:
    baseline = pd.read_csv(candidate_paths[0])[["id", "pressure"]]
    merged = sub[["id"]].merge(baseline, on="id", how="left")
    merged["pressure"] = merged["pressure"].fillna(global_med).astype(float)
    merged["pressure"] = merged["pressure"].apply(find_nearest)
    merged.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print("Used candidate submission CSVs:", candidate_paths[:2])
