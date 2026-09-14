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

0.1361364264782853

# 6. Current score

10.86378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because the blending directory you point to doesn’t contain valid per-row prediction CSVs (so `pred_list` ends up holding scalars/length-1 arrays), which makes the median result length mismatch the 603,600-row submission. I make `g()` robust by (1) locating files safely, (2) extracting predictions only from CSVs that have the correct `pressure` length, and (3) falling back to a safe baseline prediction (median train pressure snapped to the nearest allowed pressure) if no valid inputs are found. This keeps the same blending core logic (random-weighted blends + median + nearest-pressure snapping) while guaranteeing a valid `submission.csv` is written. I also fix the output filename to a Kaggle-friendly `submission.csv`.'
- What this solution (achieved 10.86378) has done: 'Your current score (10.86 MAE) indicates the script is effectively falling back to a constant baseline because it can’t find any valid prediction CSVs in `../input/gb-data-blending-recover`, so the “blend” never happens. To move toward the target, the smallest legitimate improvement is to make `g()` actually ingest predictions from any available submission-like CSVs in the input tree, not just that one folder, while still keeping the same blending logic (random-weighted blends → median → snap to nearest allowed pressure). I (1) add a safe discovery step that searches `../input/**.csv` for files with a `pressure` column and the correct length, and (2) use those files when the provided directory yields none. This should replace the constant baseline with real blended predictions and substantially reduce MAE without changing the model/blending approach.'
- What this solution (achieved 10.86378) has done: 'Your current score (10.86 MAE, lower is better) is consistent with blending not actually using any strong prediction files and effectively reverting to a constant baseline. The smallest change to move toward the target is to ensure `g()` reliably finds and loads valid full-length prediction CSVs anywhere under the available dataset trees (including `/kaggle/data`), while still keeping the same random-weight blending → median → nearest-pressure snapping logic. I also add a strict filter to avoid accidentally ingesting your own newly-written `submission.csv`/`blend.csv` and other non-prediction CSVs, which can silently break blending quality. These changes should improve score substantially without changing the core blending approach.'

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


def _read_pressure_file(path, expected_len):
    """
    Ensure we only load valid prediction files (must have a 'pressure' column
    and match expected length). Returns 1D np.array or None.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = np.asarray(df["pressure"]).ravel()
    if arr.shape[0] != expected_len:
        return None
    try:
        arr = arr.astype(np.float64, copy=False)
    except Exception:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _discover_candidate_prediction_csvs(expected_len, prefer_dir=None):
    """
    Score-improvement fix: expand discovery to all available Kaggle trees so we can
    actually blend real prediction files instead of falling back to a constant baseline.
    Core logic is unchanged: we only change *which valid prediction CSVs are found*.
    """
    candidates = []

    if prefer_dir is not None:
        try:
            for p in glob.iglob(f"{prefer_dir}/**/*.csv", recursive=True):
                candidates.append(p)
        except Exception:
            pass

    if len(candidates) == 0:
        search_roots = [
            "../input/**/*.csv",
            "/kaggle/input/**/*.csv",
            "/kaggle/data/**/*.csv",
        ]
        for pat in search_roots:
            try:
                for p in glob.iglob(pat, recursive=True):
                    candidates.append(p)
            except Exception:
                continue

    banned_names = {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "submission.csv",  # avoid ingesting our own output
        "blend.csv",  # avoid ingesting our own output
    }

    filtered = []
    for p in candidates:
        base = os.path.basename(p)
        if base in banned_names:
            continue
        arr = _read_pressure_file(p, expected_len)
        if arr is None:
            continue
        filtered.append(p)

    seen = set()
    uniq = []
    for p in filtered:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def wc(input_list, expected_len):
    """
    Keep core logic: for 1 file return it, for 2 files do a simple weighted blend based on
    parsed public score in filename. Skip files that don't parse/fit expected length.
    """
    preds = []
    l = []
    for p in input_list:
        arr = _read_pressure_file(p, expected_len)
        if arr is None:
            continue
        try:
            public_lb_score = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        preds.append(arr)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    if len(preds) > 2:
        idx = np.argsort(l)[::-1][:2]
        preds = [preds[i] for i in idx]
        l = [l[i] for i in idx]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def g(dp):
    """
    - Ensure we actually ingest valid prediction CSVs from available trees if dp is empty.
    - Keep the same blending: (optional wc preblend) -> random weights -> median -> snap.
    - Always write a valid Kaggle submission.csv
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    files = []
    if dp is not None:
        try:
            for i in glob.iglob(f"{dp}/**/*.csv", recursive=True):
                files.append(i)
        except Exception:
            files = []
    files.sort()

    if len(files) == 0:
        files = _discover_candidate_prediction_csvs(expected_len, prefer_dir=dp)

    valid_files = []
    for p in files:
        if _read_pressure_file(p, expected_len) is not None:
            valid_files.append(p)
    files = sorted(valid_files)

    file_count = len(files)
    loop_time = 154

    splits = max(1, file_count // 2)
    flist = []
    if file_count > 0:
        step = max(1, round(len(files) / splits))
        for i in range(splits):
            if i == splits - 1:
                flist.append(files[i * step :])
            else:
                flist.append(files[i * step : (i + 1) * step])

    new_flist = []
    for group in flist:
        pred = wc(group, expected_len)
        if pred is not None:
            new_flist.append(pred)
    flist = new_flist

    pred_list = []
    if len(flist) > 0:
        for loop_idx in range(loop_time):
            weight = []
            set_seed(loop_idx)
            for j in range(len(flist)):
                weight.append(rd())
            weight_sum = sum(weight) if sum(weight) != 0 else 1.0
            for j in range(len(weight)):
                weight[j] /= weight_sum
            weight.sort(reverse=True)

            temp = 0.0
            for j in range(len(flist)):
                temp = temp + flist[j] * weight[j]
            pred_list.append(temp)
            del temp
            gc.collect()

    if len(pred_list) == 0:
        baseline = float(np.median(df_train["pressure"].values))
        baseline = find_nearest(baseline)
        output["pressure"] = baseline
    else:
        stacked = np.vstack(pred_list)  # (loop_time, expected_len)
        output["pressure"] = np.median(stacked, axis=0)

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
