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

0.1443382667323867

# 6. Current score

6.43381

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because the blending directory is missing/empty in this Kaggle environment, so your `wc()` returns an empty/invalid prediction and `pred_list` ends up with arrays of length 1, causing the length-mismatch assignment. I make `g()` robust by (1) auto-locating a valid input directory among the provided dataset paths, (2) validating that each loaded submission has the expected length and `pressure` column, and (3) if no valid external submissions exist, falling back to a simple baseline submission (predicting the median train pressure) so a valid `.csv` is always produced. These changes are minimal and keep your blending/median-aggregation logic intact when the required files are actually present.'
- What this solution (achieved 6.43381) has done: 'Your current score is very far from the target (MAE ~10.86 vs ~0.144), and the reason is that your script is effectively producing a constant “median pressure” fallback because it cannot find any external blend files in this environment. To move the score sharply toward the target while preserving your overall “produce a submission by aggregating predictions + snap to valid pressure grid” core logic, I keep your `find_nearest()` post-processing and submission-writing intact, but change the fallback prediction to a lightweight, fully in-notebook, deterministic per-(R,C,time_step) median baseline learned from train and applied to test. This does not change model architecture/training loops (there are none), avoids leakage (uses only train to build statistics), and typically yields a large MAE improvement versus a constant baseline for this competition. The blending code remains as-is and still be used if valid external submissions are found; only the “no valid files” fallback is upgraded to be competitive and closer to the target.'

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


def _read_pred_file(path, expected_len):
    """Robustly read a submission-like file and validate length."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy()
    if len(arr) != expected_len:
        return None
    return arr


def wc(input_list, expected_len):
    """
    Read one or two submissions and do a simple weighted combine.
    Handles empty lists and invalid/missing files safely.
    """
    if input_list is None or len(input_list) == 0:
        return None

    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        arr = _read_pred_file(input_list[i], expected_len)
        if arr is None:
            continue
        preds.append(arr)

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    l = l[: len(preds)]
    l_sum = sum(l) if sum(l) != 0 else 1

    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _list_candidate_dirs(dp):
    """Return candidate dirs to search for blend files, including known Kaggle input locations."""
    cands = []
    if dp is not None:
        cands.append(dp)

    for base in [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        "../data",
    ]:
        if os.path.isdir(base):
            cands.append(base)

    out = []
    seen = set()
    for d in cands:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _find_blend_files(search_dir):
    """Find CSV files that look like submissions."""
    files = []
    for p in glob.iglob(os.path.join(search_dir, "**", "*.csv"), recursive=True):
        name = os.path.basename(p).lower()
        if "train" in name or "test" in name or "sample_submission" in name:
            continue
        files.append(p)
    files.sort()
    return files


def _fallback_pressure_submission():
    """
    Score-relevant improvement: instead of constant-median fallback (very high MAE),
    build a simple per-(R,C,time_step) median pressure lookup from train and apply to test.
    This preserves evaluation semantics and keeps changes minimal (still pure aggregation + snapping).
    """
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    test = pd.read_csv(test_path, usecols=["id", "R", "C", "time_step"])
    sub = pd.read_csv(sample_path)

    tr = df_train[["R", "C", "time_step", "pressure"]].copy()
    tr["time_step_r"] = tr["time_step"].round(5)
    test["time_step_r"] = test["time_step"].round(5)

    med_map = (
        tr.groupby(["R", "C", "time_step_r"], sort=False)["pressure"]
        .median()
        .reset_index()
    )

    test2 = test.merge(med_map, on=["R", "C", "time_step_r"], how="left")

    global_med = float(df_train["pressure"].median())
    test2["pressure"] = test2["pressure"].fillna(global_med)

    pred_by_id = test2.set_index("id")["pressure"]
    sub["pressure"] = sub["id"].map(pred_by_id).astype(float)

    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv("submission.csv", index=False)


def g(dp):
    """
    Blending driver: if valid external submissions are found, blend them.
    Otherwise, produce an improved in-notebook fallback submission.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    loop_time = 154

    all_files = []
    for cand in _list_candidate_dirs(dp):
        if os.path.isdir(cand):
            all_files.extend(_find_blend_files(cand))

    if dp is not None and os.path.isdir(dp):
        all_files.extend(
            [p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")]
        )

    all_files = sorted(list(dict.fromkeys(all_files)))

    valid_files = []
    for p in all_files:
        arr = _read_pred_file(p, expected_len)
        if arr is not None:
            valid_files.append(p)

    if len(valid_files) == 0:
        _fallback_pressure_submission()
        return

    file_count = len(valid_files)
    splits = max(1, file_count // 2)

    flist_paths = []
    step = round(len(valid_files) / splits) if splits > 0 else len(valid_files)
    for i in range(splits):
        if i == splits - 1:
            flist_paths.append(valid_files[i * step :])
        else:
            flist_paths.append(valid_files[i * step : (i + 1) * step])

    flist = []
    for i in range(len(flist_paths)):
        arr = wc(flist_paths[i], expected_len=expected_len)
        if arr is not None and len(arr) == expected_len:
            flist.append(arr)

    if len(flist) == 0:
        _fallback_pressure_submission()
        return

    pred_list = []
    for ii in range(loop_time):
        weight = []
        set_seed(ii)
        for jj in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for jj in range(len(weight)):
            weight[jj] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for jj in range(len(flist)):
            temp += flist[jj] * weight[jj]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
