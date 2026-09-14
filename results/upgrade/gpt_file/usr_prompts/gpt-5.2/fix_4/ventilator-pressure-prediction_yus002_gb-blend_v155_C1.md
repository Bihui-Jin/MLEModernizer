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

0.1360168427746761

# 6. Current score

10.21761

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because the blending directory is missing in this environment, so no prediction files are loaded and the code ends up producing a single scalar median (length 1) that cannot be assigned to 603600 submission rows. I (1) make the path/file discovery robust, (2) validate and align every loaded prediction to the sample_submission `id` order, and (3) add a safe fallback that produces a valid submission (all zeros) if no blend inputs exist, so you always get a `.csv` output. These changes are execution/stability fixes and keep the core “blend predictions then snap to nearest known pressure” logic intact; they just prevent empty/shape-mismatch cases and guarantee a correctly formatted submission.'
- What this solution (achieved 9.92091) has done: 'Your current score (17.65 MAE) indicates the script is falling back to an all-zero submission because it can’t find any blend input files in `../input/gb-data-blending-recover`, which is far from the target (0.136). To move toward the target with minimal logic changes, I keep your existing “blend predictions then snap to nearest known pressure” approach, but add a safe, competition-legal baseline generator that uses the training-set median pressure per (R,C,time_step) and merges it onto test (then snaps to nearest pressure). This only activates when no blend CSVs are found, so if blend files exist it preserves your original behavior. The baseline is fast (groupby + merge) and typically scores orders of magnitude better than zeros, pushing the score much closer to your target.'
- What this solution (achieved 10.21761) has done: 'Your current MAE (9.92, lower-is-better) is still far from the target (0.136), and the biggest remaining gap is that the fallback baseline (median pressure per (R,C,time_step)) ignores the most important driver: the inspiratory control signal history (`u_in`) and valve state (`u_out`). To move the score substantially toward the target with minimal changes and without altering your blend/snap core logic, I keep the same “if blend files exist, blend+median+snap” path, but strengthen the fallback baseline by adding simple, competition-legal lag and cumulative features computed per breath, and then using a median lookup on those features from train to test. This is still a pure train-statistics merge (no new model/training loop), runs fast (groupby + merge), and keeps your final “snap to nearest known pressure” post-processing identical. I also keep the original file-loading/alignment safeguards unchanged.'

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


def _load_pred_as_array(path, sample_ids):
    """
    Bugfix: robustly load a prediction file and align to sample_submission ids.
    Accepts either [id, pressure] or just [pressure] with correct length.
    Returns None if file is unusable.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "pressure" not in df.columns:
        return None

    if "id" in df.columns:
        df = df[["id", "pressure"]].dropna()
        df = df.set_index("id")
        aligned = df.reindex(sample_ids)
        if aligned["pressure"].isna().any():
            return None
        arr = aligned["pressure"].to_numpy(dtype=np.float32)
    else:
        arr = df["pressure"].to_numpy(dtype=np.float32)
        if arr.shape[0] != sample_ids.shape[0]:
            return None

    if not np.isfinite(arr).all():
        return None
    return arr


def wc(input_list):
    """
    Original intent: weighted combine of 1-2 submissions from a split.
    Bugfix: handle arbitrary filenames safely; if parsing score fails, use equal weights.
    """
    l = []
    preds = []
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()

    for p in input_list:
        try:
            public_lb_score = float(os.path.basename(p).split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1.0
        l.append(public_lb_score)

        arr = _load_pred_as_array(p, sample_ids)
        if arr is None:
            continue
        preds.append(arr)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    preds = preds[:2]
    l = l[:2]

    l_sum = sum(l) if sum(l) != 0 else 1.0
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _snap_to_known_pressures(pred):
    """
    Keep existing post-processing semantics ("snap to nearest known pressure"),
    but vectorized for speed and to avoid per-row apply overhead.
    """
    pred = pred.astype(np.float32, copy=False)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = idx

    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[upper_idx]

    choose_lower = np.abs(lower_val - pred) < np.abs(upper_val - pred)
    snapped = np.where(choose_lower, lower_val, upper_val).astype(np.float32)
    return snapped


def _add_basic_breath_features(df):
    """
    Score-improvement (fallback only): add simple lag/cumsum features per breath_id
    using only inputs available at prediction time (u_in, u_out, time_step, R, C).
    This keeps the approach as a pure train-statistics lookup (no new model).
    """
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)

    df["u_in_x_u_out"] = (df["u_in"] * df["u_out"]).astype(np.float32)

    return df


def _baseline_submission_from_train_stats():
    """
    Improvement fallback (used only when no blend inputs exist):
    Use train median pressure per (R,C,time_step,u_out,u_in_lag1,u_in_cumsum), merge onto test,
    then snap predictions to nearest known pressure value.

    This remains a pure train-statistics merge (no new model/training loop) and preserves
    the existing "snap to nearest pressure" post-processing semantics.
    """
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

    train_feat = _add_basic_breath_features(
        df_train[["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]]
    )
    test_feat = _add_basic_breath_features(
        test[["breath_id", "R", "C", "time_step", "u_in", "u_out"]]
    )

    for col in ["u_in_lag1", "u_in_lag2", "u_in_cumsum", "u_in"]:
        train_feat[col] = np.round(train_feat[col], 1)
        test_feat[col] = np.round(test_feat[col], 1)

    keys_primary = ["R", "C", "time_step", "u_out", "u_in_lag1", "u_in_cumsum"]
    stats_primary = (
        train_feat.groupby(keys_primary, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred"})
    )

    merged = test_feat.merge(stats_primary, on=keys_primary, how="left", sort=False)

    keys_secondary = ["R", "C", "time_step", "u_out", "u_in_lag1"]
    stats_secondary = (
        train_feat.groupby(keys_secondary, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred2"})
    )
    merged = merged.merge(stats_secondary, on=keys_secondary, how="left", sort=False)

    keys_tertiary = ["R", "C", "time_step"]
    stats_tertiary = (
        df_train.groupby(keys_tertiary, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred3"})
    )
    merged = merged.merge(stats_tertiary, on=keys_tertiary, how="left", sort=False)

    global_med = float(df_train["pressure"].median())

    pred = (
        merged["pred"]
        .fillna(merged["pred2"])
        .fillna(merged["pred3"])
        .fillna(global_med)
        .to_numpy(dtype=np.float32)
    )

    sub["pressure"] = _snap_to_known_pressures(pred)
    sub.to_csv("submission.csv", index=False)


def g(dp):
    """
    Bugfixes:
    - If dp doesn't exist / no usable files, write a valid submission.
      Improvement: instead of all-zeros (very poor MAE), emit a simple train-stats
      baseline to move score toward the target while keeping core blend+snap logic intact.
    - Ensure every loaded prediction is aligned to sample ids and has correct length.
    - Avoid empty pred_list causing shape (1,) medians.
    """
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = output["id"].to_numpy()
    n = len(output)

    files = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                files.append(i)
    files.sort()

    if len(files) == 0:
        _baseline_submission_from_train_stats()
        return

    file_count = len(files)
    loop_time = 156
    splits = file_count // 2  # keep original splitting idea

    if splits < 1:
        splits = 1

    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) == 0:
            continue
        blended = wc(chunk)
        if blended is None or blended.shape[0] != n:
            continue
        flist.append(blended.astype(np.float32, copy=False))

    if len(flist) == 0:
        _baseline_submission_from_train_stats()
        return

    pred_list = []
    for t in range(loop_time):
        weight = []
        set_seed(t)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float32)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, n)
    output["pressure"] = np.median(stacked, axis=0)

    output["pressure"] = _snap_to_known_pressures(
        output["pressure"].to_numpy(dtype=np.float32)
    )

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
