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

0.1417941441510536

# 6. Current score

6.10591

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The crash happens because the blending directory is empty/unavailable, so `pred_list` ends up containing arrays of the wrong shape (or none), and `np.median(np.vstack(pred_list), ...)` produces a length-1 result that can’t be assigned to the 603,600-row submission. I make `g()` robust by (1) verifying the input directory and reading only valid submission-like CSVs, (2) enforcing that every loaded prediction is aligned to `sample_submission` `id` and has the correct length, and (3) providing a safe fallback prediction (all zeros snapped to nearest valid pressure) so a valid `.csv` is always produced. These fixes are score-neutral relative to the intended blending approach; they only prevent runtime errors and ensure a submission file is written. I also fix the notebook cell numbering to start from 1 so it runs cleanly in a script-like environment.'
- What this solution (achieved 6.10591) has done: 'Your current score (17.65 MAE) is far worse than the target (0.1418), so the biggest issue is that the script is blending (or falling back to zeros) without actually generating any meaningful model predictions in this environment. The most minimal legitimate improvement is to keep your existing blending core, but add an in-notebook baseline predictor that uses only the provided train.csv to compute a per-(R,C,u_out) mean pressure by time_step, then uses that as a “pseudo-submission” to blend when no external blend files exist. This preserves your blending logic and snapping-to-valid-pressures postprocess, but ensures `g()` always has at least one strong prediction source available, moving the MAE dramatically toward the target. The submission file naming and schema remain unchanged (`rwb 154 loops.csv` with `id,pressure`).'

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


def _load_pred_file(fp, sample_ids):
    """
    Bugfix: robustly load a prediction file and align to sample_submission ids.
    Returns a numpy array of shape (n_rows,) or None if invalid.
    """
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    if "pressure" not in df.columns:
        return None

    if "id" in df.columns:
        try:
            df2 = df[["id", "pressure"]].copy()
            df2 = df2.dropna(subset=["id", "pressure"])
            df2["id"] = df2["id"].astype(np.int64, errors="ignore")
            df2 = df2.set_index("id")
            aligned = df2.reindex(sample_ids)["pressure"].to_numpy()
        except Exception:
            aligned = df["pressure"].to_numpy()
    else:
        aligned = df["pressure"].to_numpy()

    if aligned.ndim != 1:
        aligned = aligned.reshape(-1)

    if len(aligned) != len(sample_ids):
        return None

    return aligned.astype(np.float64, copy=False)


def wc(input_list):
    l = []
    preds = []
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()

    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        pred = _load_pred_file(fp, sample_ids)
        if pred is None:
            continue
        l.append(public_lb_score)
        preds.append(pred)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _build_timebucket_baseline_predictions(sample_ids):
    """
    Score-improvement change (core blending logic preserved):
    If no external blend files are available, create a strong, fully legitimate
    baseline prediction using ONLY train.csv statistics.

    Approach: mean pressure by (R, C, u_out, time_step_rounded).
    This aligns with the metric (only inspiratory scored) and uses available covariates,
    producing far better than all-zeros while keeping runtime reasonable.
    """
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    ts_col = np.round(df_train["time_step"].to_numpy(), 2)
    tr = df_train[["R", "C", "u_out", "pressure"]].copy()
    tr["ts2"] = ts_col

    mean_map = (
        tr.groupby(["R", "C", "u_out", "ts2"], sort=False)["pressure"]
        .mean()
        .astype(np.float64)
    )

    mean_rcu = (
        tr.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .mean()
        .astype(np.float64)
    )
    global_mean = float(tr["pressure"].mean())

    te = test[["id", "R", "C", "u_out", "time_step"]].copy()
    te["ts2"] = np.round(te["time_step"].to_numpy(), 2)

    te = te.join(mean_map.rename("p_ts"), on=["R", "C", "u_out", "ts2"])
    te = te.join(mean_rcu.rename("p_rcu"), on=["R", "C", "u_out"])
    pred = te["p_ts"].to_numpy(dtype=np.float64)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te.loc[m, "p_rcu"].to_numpy(dtype=np.float64)
    m2 = np.isnan(pred)
    if m2.any():
        pred[m2] = global_mean

    pred_aligned = (
        te.set_index("id").reindex(sample_ids)["p_ts"].to_numpy(dtype=np.float64)
    )
    pred_filled_by_id = te.set_index("id").reindex(sample_ids).index.to_numpy()
    te["pred_filled"] = pred
    pred_aligned = (
        te.set_index("id").reindex(sample_ids)["pred_filled"].to_numpy(dtype=np.float64)
    )

    pred_aligned = np.asarray(pred_aligned, dtype=np.float64).reshape(-1)
    return pred_aligned


def g(dp):
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()
    n = len(sample)

    l = []
    if os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                l.append(i)

    file_count = len(l)
    loop_time = 154

    if file_count == 0:
        baseline_pred = _build_timebucket_baseline_predictions(sample_ids)
        merged = [baseline_pred]
    else:
        splits = max(1, file_count // 2)
        l.sort()
        flist = []
        for i in range(splits):
            if i == splits - 1:
                flist.append(l[i * round(len(l) / splits) :])
            else:
                flist.append(
                    l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
                )

        merged = []
        for i in range(len(flist)):
            pred = wc(flist[i])
            if pred is not None and len(pred) == n:
                merged.append(pred)

        if len(merged) == 0:
            baseline_pred = _build_timebucket_baseline_predictions(sample_ids)
            merged = [baseline_pred]

    pred_list = []
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for j in range(len(merged)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(merged)):
            temp += merged[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = sample.copy()
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()

    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)

    if "id" in a_df.columns:
        a_p = a_df.set_index("id").reindex(sample_ids)["pressure"].to_numpy()
    else:
        a_p = a_df["pressure"].to_numpy()
    if "id" in b_df.columns:
        b_p = b_df.set_index("id").reindex(sample_ids)["pressure"].to_numpy()
    else:
        b_p = b_df["pressure"].to_numpy()

    out = sample.copy()
    out["pressure"] = a_p * 0.6 + b_p * 0.4
    out["pressure"] = out["pressure"].apply(find_nearest)
    out.to_csv("blend.csv", index=False)
    return out




## === cell 2
g("../input/gb-data-blending-recover")
