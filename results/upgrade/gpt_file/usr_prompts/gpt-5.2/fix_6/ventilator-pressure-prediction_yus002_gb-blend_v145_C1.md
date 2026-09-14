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

0.1359299397598838

# 6. Current score

4.1478

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.70637) has done: 'You’re failing because the notebook tries to blend external prediction CSVs from a dataset that isn’t present (`../input/gb-data-blending-recover`), so no valid files are found and execution stops before writing `submission.csv`. I keep your blending logic intact, but add a safe fallback: if no external prediction files exist, generate a baseline prediction from the provided `test.csv` (inspiratory-phase-aware heuristic using `u_out` and `u_in`) and still snap to the nearest allowed pressure values using your `find_nearest`. This ensures the notebook runs end-to-end in the given environment and always produces a valid `submission.csv` with correct columns/length. The fallback is score-oriented (better than all-zeros) while remaining minimal and not introducing any new model/training loops.'
- What this solution (achieved 8.13563) has done: 'Your current score is far above the target (lower is better), and the biggest controllable issue in this no-training/blending-only script is that the fallback baseline is an arbitrary linear mapping of `u_in` to the full pressure range, which performs very poorly for this competition. I keep your core blending logic intact, but make the fallback much more competition-aligned by predicting from a per-(R,C,time_step,u_in,u_out) lookup built from `train.csv`, with a safe global fallback when a key isn’t found. This stays within your existing approach (no model, no training loop, same snapping via `find_nearest`) but should drastically reduce MAE versus the current heuristic, moving the score toward the target. I also ensure the fallback writes `submission.csv` exactly as required and remains fast enough by using an aggregate groupby rather than per-row searches.'
- What this solution (achieved 7.93873) has done: 'Your current score is far worse than the target (lower is better), and since your main blending directory isn’t available the run is effectively relying on the fallback. To move the MAE much closer to the target without changing your overall approach, I keep your blending logic and the “snap-to-allowed-pressures” postprocess, but make the fallback prediction much more competition-aligned by using a per-(R,C,time_step,u_out) mapping plus an (R,C,u_out,u_in_bin) residual correction learned from train. I also enforce the evaluation semantics by setting predictions to 0 during expiratory phase (`u_out==1`), because those rows are not scored and this reduces harmful variance. All changes are confined to `_fallback_baseline_submission` and keep I/O paths and output format identical.'
- What this solution (achieved 4.1478) has done: 'Your current score is still far above the target (lower is better), and in this environment you’re effectively using the fallback because the external blend directory is missing. To move closer to the target without changing your blending/core structure, I only improve `_fallback_baseline_submission` by (1) predicting inspiratory pressure using a much more specific lookup keyed by `(R,C,u_out,time_step_bin,u_in_bin)` built from `train.csv`, and (2) only forcing expiratory (`u_out==1`) predictions to 0 *after* nearest-pressure snapping so we don’t inject an out-of-manifold value. This keeps your “train-lookup + nearest-pressure snapping” semantics, but greatly reduces the fallback error, which is the dominant driver of your current MAE. All I/O paths remain unchanged and it still always writes a valid `submission.csv`.'

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


def wc(input_list, expected_len=603600):
    preds = []
    scores = []
    for path in input_list:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "pressure" not in df.columns:
            continue
        arr = df["pressure"].to_numpy().ravel()
        if arr.shape[0] != expected_len:
            continue

        sc = None
        try:
            sc = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            sc = None

        preds.append(arr)
        scores.append(sc)

    if len(preds) == 0:
        raise ValueError(
            "No valid prediction CSVs found (need 'pressure' column and correct row count)."
        )

    if len(preds) == 1:
        return preds[0]

    p0, p1 = preds[0], preds[1]
    s0, s1 = scores[0], scores[1]

    if (s0 is None) or (s1 is None) or ((s0 + s1) == 0):
        weight1, weight2 = 0.5, 0.5
    else:
        l_sum = s0 + s1
        weight1 = (s1 / l_sum) + 0.15
        weight2 = 1 - weight1

    return p0 * weight1 + p1 * weight2


def _fallback_baseline_submission(expected_len=603600):
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    if len(test) != expected_len:
        raise ValueError(
            f"Unexpected test length: {len(test)} (expected {expected_len})"
        )

    time_bin_size = 0.03
    u_in_bins = 200  # finer u_in discretization to reduce averaging error
    u_in_edges = np.linspace(0.0, 100.0, u_in_bins + 1)

    tr = df_train[["R", "C", "time_step", "u_out", "u_in", "pressure"]].copy()
    tr["time_bin"] = np.rint(
        tr["time_step"].to_numpy(dtype=np.float64) / time_bin_size
    ).astype(np.int16)
    tr["u_in_bin"] = np.clip(
        np.digitize(tr["u_in"].to_numpy(dtype=np.float64), u_in_edges) - 1,
        0,
        u_in_bins - 1,
    ).astype(np.int16)

    key_cols = ["R", "C", "u_out", "time_bin", "u_in_bin"]
    lut = (
        tr.groupby(key_cols, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_lut"})
    )

    coarse_cols = ["R", "C", "u_out", "time_bin"]
    coarse = (
        tr.groupby(coarse_cols, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_coarse"})
    )

    rc_cols = ["R", "C", "u_out"]
    rc = (
        tr.groupby(rc_cols, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_rc"})
    )

    te = test[["id", "R", "C", "time_step", "u_out", "u_in"]].copy()
    te["time_bin"] = np.rint(
        te["time_step"].to_numpy(dtype=np.float64) / time_bin_size
    ).astype(np.int16)
    te["u_in_bin"] = np.clip(
        np.digitize(te["u_in"].to_numpy(dtype=np.float64), u_in_edges) - 1,
        0,
        u_in_bins - 1,
    ).astype(np.int16)

    merged = te.merge(lut, on=key_cols, how="left")
    merged = merged.merge(coarse, on=coarse_cols, how="left")
    merged = merged.merge(rc, on=rc_cols, how="left")

    overall_mean = float(df_train["pressure"].mean())

    p = merged["p_lut"].to_numpy(dtype=np.float64)
    p_coarse = merged["p_coarse"].to_numpy(dtype=np.float64)
    p_rc = merged["p_rc"].to_numpy(dtype=np.float64)

    miss = np.isnan(p)
    if miss.any():
        p[miss] = p_coarse[miss]
    miss = np.isnan(p)
    if miss.any():
        p[miss] = p_rc[miss]
    miss = np.isnan(p)
    if miss.any():
        p[miss] = overall_mean

    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    if len(sub) != expected_len:
        raise ValueError(
            f"Unexpected sample_submission length: {len(sub)} (expected {expected_len})"
        )

    sub["pressure"] = p

    sub["pressure"] = sub["pressure"].apply(find_nearest)

    u_out_arr = te["u_out"].to_numpy()
    if (u_out_arr == 1).any():
        sub.loc[u_out_arr == 1, "pressure"] = find_nearest(0.0)

    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    expected_len = 603600
    loop_time = 154

    all_files = sorted(glob.glob(os.path.join(dp, "*.csv")))
    valid_files = []
    for f in all_files:
        try:
            tmp = pd.read_csv(f, usecols=["pressure"])
            if len(tmp) == expected_len:
                valid_files.append(f)
        except Exception:
            continue

    file_count = len(valid_files)

    if file_count == 0:
        return _fallback_baseline_submission(expected_len=expected_len)

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(file_count / splits)
        end = None if i == splits - 1 else (i + 1) * round(file_count / splits)
        group = valid_files[start:end]
        if len(group) == 0:
            continue
        flist.append(wc(group, expected_len=expected_len))

    if len(flist) == 0:
        return _fallback_baseline_submission(expected_len=expected_len)

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    blended = np.median(np.vstack(pred_list), axis=0)
    if blended.shape[0] != len(output):
        raise ValueError(
            f"Blended prediction length {blended.shape[0]} does not match submission length {len(output)}"
        )

    output["pressure"] = blended
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
print(pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").columns.tolist())
