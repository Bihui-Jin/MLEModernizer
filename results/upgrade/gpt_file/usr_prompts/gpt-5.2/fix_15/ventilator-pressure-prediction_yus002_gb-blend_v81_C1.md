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

0.1481169828851033

# 6. Current score

3.46706

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The failure comes from `g()` reading no valid prediction files from `../input/gb-blending`, so `pred_list` ends up containing arrays of shape `(1,)` (or a single scalar-like object), making the median length 1 and not matching the 603600-row submission. I make `g()` robust by (1) locating the competition files from the existing `../input/ventilator-pressure-prediction/` path, (2) collecting only `.csv` files that contain a `pressure` column and have the correct length, and (3) raising a clear error if no valid files are found (instead of producing a broken submission). To ensure you always get a valid submission even when the blend directory is empty/unavailable, I add a safe fallback that writes the sample submission (all zeros) with the correct filename and format. These changes are execution/stability fixes and keep the core blending logic intact when blend files exist.'
- What this solution (achieved 9.92097) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that when `../input/gb-blending` is empty you fall back to the sample submission (all zeros), which scores very poorly. To move toward the target with minimal change and without changing the blending approach, I add a safe, deterministic baseline that predicts the per-(R,C,time_step) median pressure from the training data, and use it only as the fallback when blending files are unavailable. This keeps your core blending logic intact when blend files exist, but makes the “no blend files” path produce a reasonable submission (and thus a much lower MAE). I also ensure the submission file is always written as `submission.csv` for Kaggle convenience while still returning the path.'
- What this solution (achieved 7.32063) has done: 'Your current score (9.92097, lower is better) is still far from the target, so we should improve the fallback path (used when `../input/gb-blending` is empty) without changing the core blending logic. The smallest legitimate gain for this competition is to make the fallback respect the evaluation rule by predicting **only for inspiratory phase** and forcing **expiratory phase (`u_out==1`) predictions to 0**, since those rows are not scored and this prevents noisy guesses from hurting. Concretely, we keep your `(R,C,time_step)->median(pressure)` fallback, but compute it from **inspiratory-only training rows** and apply it to **inspiratory-only test rows**, setting expiratory predictions to 0. This should move MAE substantially toward the target while preserving your overall approach and keeping runtime within limits.'
- What this solution (achieved 7.24306) has done: 'Your current score is still far worse than the target (lower is better), and it’s coming from the fallback baseline being too weak when `../input/gb-blending` is empty/unavailable. With minimal changes and preserving your overall “blend if possible, otherwise fallback” logic, I strengthen only the fallback by switching from `(R,C,time_step)->median(pressure)` to a strictly richer but still simple lookup: per-(R,C,breath_time_index,u_in_rounded,u_out) median pressure learned from train, with a safe hierarchical backoff. This respects the competition’s inspiratory-only scoring by keeping `u_out==1` predictions at 0, and it should move MAE materially toward your target without changing model/training loops. I also keep your `find_nearest` discretization so submission values stay on valid pressure levels.'
- What this solution (achieved 7.24186) has done: 'Your score is far worse than the target (lower is better), and it’s coming from the fallback still being too coarse when the blend directory is empty. With minimal changes and keeping your “blend if possible, otherwise fallback” structure, I strengthen only the fallback by adding a within-breath cumulative volume feature (`u_in_cum`) and grouping on a discretized version of it (plus `R,C,time_step,u_in_rounded`) with a safe hierarchical backoff. This stays a pure lookup/median baseline (no training loop/model changes) but captures much more of the pressure dynamics and should move MAE substantially toward your target. I also ensure `u_out==1` predictions are forced to 0 (still valid since not scored) and keep the `find_nearest` snapping to valid pressure levels.'
- What this solution (achieved 1.53132) has done: 'I fix the fallback submission creation crash by avoiding a `merge()` that creates `pressure_x/pressure_y` columns and then referencing a non-existent `pressure` column. I instead build the submission by aligning predictions to the test `id` order and assigning directly into the sample submission’s `pressure` column, guaranteeing correct length and column names. This is a pure bug fix (score-neutral relative to your intended fallback logic) and allow the notebook to always finish and write a valid `submission.csv`. I also add a small safety check to ensure no missing predictions remain after alignment.'
- What this solution (achieved 1.50215) has done: 'Your current score (1.53132, lower is better) is still far above the target, so we should improve the *fallback* (used when `../input/gb-blending` is empty) while keeping your blending logic unchanged. The smallest meaningful gain here is to make the fallback compute the cumulative-volume feature (`u_in_cum`) in a way that matches the physical integration (use `dt` with `time_step` diff and shift so each row uses the previous interval), and to include `u_out` in the lookup key so inspiratory/expiratory contexts don’t get mixed. This keeps your same “median in a bucket / kNN within bucket + hierarchical backoff + snap to valid pressures” approach, but fixes a feature alignment issue and reduces collisions that inflate MAE. The submission writing remains identical and still forces `u_out==1` predictions to 0.'
- What this solution (achieved 1.50215) has done: 'Your current MAE (1.50215, lower is better) is still far above the target, so we should improve only the *fallback* path (used when `../input/gb-blending` is empty) while keeping your blending logic untouched. The smallest high-impact fix for this competition is to ensure the fallback predicts **only on inspiratory rows** and then **zeroes out expiratory rows**, but right now your lookup table is built from inspiratory-only train while still using `t_idx` computed over all test rows, which misaligns time indices and harms accuracy. I compute an `t_insp_idx` (cumcount within `u_out==0`) for both train and test and use that in the bucket key, keeping your same median/kNN-in-bucket + hierarchical backoff + pressure snapping logic. This should materially reduce MAE toward your target without changing the overall approach or adding new models/training loops.'
- What this solution (achieved 1.50215) has done: 'Your current score (1.50215 MAE, lower is better) is still far above the target, so we should improve only the fallback path (used when `../input/gb-blending` is empty) while keeping your blending logic untouched. The most likely remaining issue is that the fallback builds training buckets keyed by inspiratory index, but the test-side `t_insp_idx` currently becomes `-1` for `u_out==1` rows and can also misbehave around breath boundaries; we compute `t_insp_idx` cleanly as a per-breath cumcount on inspiratory-only rows and merge it back to all rows, ensuring consistent keys. In the same minimal spirit, we also build `dt_prev` in a consistent “previous interval” way for both train/test (diff then shift within breath) and compute `u_in_cum` on the full breath (then only use it for inspiratory predictions), which reduces feature mismatch. These changes keep the exact same lookup/kNN-median + hierarchical backoff + pressure snapping logic, but remove index/feature misalignment that inflates MAE.'
- What this solution (achieved 1.53132) has done: 'Your current score is much worse than the target (lower is better), and the blend directory is likely empty so you’re relying on the fallback; we improve only that fallback while keeping the overall “blend if possible, otherwise fallback” logic unchanged. The main minimal win is to stop snapping predictions to the nearest discrete pressure level: the competition metric is MAE on continuous pressures, so quantizing usually increases error. We also fix the cumulative-volume integration to be physically consistent by using the current interval `dt` (not a shifted `dt_prev`), which reduces feature mismatch between train/test and improves the kNN-in-bucket lookup without changing the approach. Finally, we keep `u_out==1` predictions at 0 (not scored) and maintain the same output format to always write a valid `submission.csv`.'
- What this solution (achieved 3.46706) has done: 'We keep your existing “blend if possible, otherwise fallback” structure and only improve the fallback path, since your current score (1.53132 MAE) is far above the target (0.1481) and the blend directory is likely empty. The biggest low-risk gain without changing the approach is to (1) build a stronger deterministic lookup baseline that keys on more informative discretized features (R, C, inspiratory time index, u_in, and cumulative u_in integral), (2) add a safe hierarchical backoff (full key → drop cum → drop u_in → (R,C,t) → (R,C) → global), and (3) avoid forcing expiratory predictions to 0 (expiratory rows are not scored, but Kaggle computes MAE only on inspiratory; setting 0 can still be safe, yet in practice some implementations evaluate all rows—so we keep expiratory reasonable via backoff but still prioritize inspiratory accuracy). We also remove the unused pressure snapping in the fallback (quantization usually worsens MAE) while leaving blending untouched. These are minimal changes focused on improving the fallback prediction quality and thus moving the score toward the target.'
- What this solution (achieved 3.46706) has done: 'Your current MAE (3.46706, lower is better) is far above the target, so we should improve only the fallback path (used when the blending directory is empty/unavailable) while leaving the blending logic untouched. The biggest low-risk fix is to respect the competition’s scoring by forcing expiratory rows (`u_out==1`) to a constant (0.0), since they are not scored and this avoids any accidental penalty from noisy expiratory guesses. We also correct the fallback’s non‑inspiratory default from an (R,C) median to 0.0, and keep inspiratory predictions exactly as your existing hierarchical median lookup (same core logic). Finally, we add a strict sanity check that the written submission has the right length/columns to prevent silent misalignment.'
- What this solution (achieved 3.46706) has done: 'We keep your “blend if possible, otherwise fallback” structure unchanged and only strengthen the fallback because your current MAE (3.46706, lower is better) is still far from the target (0.1481). The smallest high-impact fix is to stop leaving most expiratory rows at 0.0: while they aren’t scored, predicting 0 there can still hurt due to potential metric/implementation nuances and also destabilizes the learned mapping; we instead fill expiratory rows with a conservative per-(R,C,t_idx) inspiratory median so the full sequence stays plausible. We also add `u_out` into the fallback lookup keys (so we never mix contexts) while still training the table primarily from inspiratory rows, and we replace the slow Python row loop with vectorized joins/backoff (same hierarchical median logic, just implemented safely and faster). These changes are tightly scoped to the fallback path and keep your blending logic, file paths, and submission format intact.'

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
BASE_DIR = "../input/ventilator-pressure-prediction"
if not os.path.exists(BASE_DIR):
    if os.path.exists("../input"):
        candidates = glob.glob("../input/**/train.csv", recursive=True)
        if candidates:
            BASE_DIR = os.path.dirname(candidates[0])

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 0
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l) if sum(l) != 0 else 1
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    expected_len = len(sample)

    all_files = sorted(glob.glob(os.path.join(dp, "*")))
    valid_files = []
    for fp in all_files:
        if not fp.lower().endswith(".csv"):
            continue
        try:
            tmp = pd.read_csv(fp, usecols=["pressure"])
            if len(tmp) != expected_len:
                continue
            valid_files.append(fp)
        except Exception:
            continue

    if len(valid_files) == 0:
        raise FileNotFoundError(
            f"No valid prediction CSVs found in {dp}. "
            f"Expected .csv files with a 'pressure' column and {expected_len} rows."
        )

    file_count = len(valid_files)
    loop_time = 150
    splits = max(1, file_count // 2)

    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(valid_files[i * round(len(valid_files) / splits) :])
        else:
            flist.append(
                valid_files[
                    i
                    * round(len(valid_files) / splits) : (i + 1)
                    * round(len(valid_files) / splits)
                ]
            )

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = sample
    output.pressure = np.median(np.vstack(pred_list), axis=0)

    out_path = f"rwb {loop_time} loops.csv"
    output.to_csv(out_path, index=False)
    return out_path


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5

    a.to_csv("blend.csv", index=False)
    return a


def make_rc_time_median_fallback_submission(out_csv="submission.csv"):
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    expected_len = len(sample)

    test = pd.read_csv(
        TEST_PATH, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    train_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    train = df_train[train_cols].copy()

    train_insp = train.loc[train["u_out"] == 0, ["id", "breath_id"]].copy()
    train_insp["t_insp_idx"] = (
        train_insp.groupby("breath_id").cumcount().astype(np.int16)
    )

    test_insp = test.loc[test["u_out"] == 0, ["id", "breath_id"]].copy()
    test_insp["t_insp_idx"] = test_insp.groupby("breath_id").cumcount().astype(np.int16)

    train = train.merge(train_insp[["id", "t_insp_idx"]], on="id", how="left")
    test = test.merge(test_insp[["id", "t_insp_idx"]], on="id", how="left")

    train["dt"] = (
        train.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )
    test["dt"] = (
        test.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )

    train["u_in_cum"] = (
        (train["u_in"].astype(np.float32) * train["dt"])
        .groupby(train["breath_id"])
        .cumsum()
    )
    test["u_in_cum"] = (
        (test["u_in"].astype(np.float32) * test["dt"])
        .groupby(test["breath_id"])
        .cumsum()
    )

    train["u_in_r"] = np.round(train["u_in"].astype(np.float32), 1)
    test["u_in_r"] = np.round(test["u_in"].astype(np.float32), 1)
    train["u_in_cum_r"] = np.round(train["u_in_cum"].astype(np.float32), 2)
    test["u_in_cum_r"] = np.round(test["u_in_cum"].astype(np.float32), 2)

    train_insp2 = train.loc[train["u_out"] == 0].copy()

    key_full = ["R", "C", "u_out", "t_insp_idx", "u_in_r", "u_in_cum_r"]
    key_drop_cum = ["R", "C", "u_out", "t_insp_idx", "u_in_r"]
    key_drop_uin = ["R", "C", "u_out", "t_insp_idx"]
    key_rc = ["R", "C"]

    med_full = (
        train_insp2.groupby(key_full, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full"})
    )
    med_drop_cum = (
        train_insp2.groupby(key_drop_cum, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_drop_cum"})
    )
    med_drop_uin = (
        train_insp2.groupby(key_drop_uin, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_drop_uin"})
    )
    med_rc = (
        train_insp2.groupby(key_rc, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rc"})
    )
    global_median_insp = float(train_insp2["pressure"].median())

    feat = test[["id", "R", "C", "u_out", "t_insp_idx", "u_in_r", "u_in_cum_r"]].copy()

    feat = feat.merge(med_full, on=key_full, how="left")
    feat = feat.merge(med_drop_cum, on=key_drop_cum, how="left")
    feat = feat.merge(med_drop_uin, on=key_drop_uin, how="left")
    feat = feat.merge(med_rc, on=key_rc, how="left")

    p = feat["p_full"]
    p = p.fillna(feat["p_drop_cum"])
    p = p.fillna(feat["p_drop_uin"])
    p = p.fillna(feat["p_rc"])
    p = p.fillna(global_median_insp).astype(np.float32)

    sub = sample.copy()
    sub["pressure"] = p.to_numpy()

    if list(sub.columns) != ["id", "pressure"]:
        sub = sub[["id", "pressure"]]
    if len(sub) != expected_len:
        raise ValueError(f"Submission length mismatch: {len(sub)} vs {expected_len}")
    if sub["pressure"].isna().any():
        sub["pressure"] = sub["pressure"].fillna(global_median_insp)

    sub.to_csv(out_csv, index=False)
    return out_csv




## === cell 2
BLEND_DIR = "../input/gb-blending"

try:
    out_csv = g(BLEND_DIR)
    pd.read_csv(out_csv).to_csv("submission.csv", index=False)
    print(f"Wrote submission to: submission.csv (from {out_csv})")
except Exception as e:
    print(
        f"Blending failed ({type(e).__name__}: {e}). Writing RC-time median fallback submission instead."
    )
    out_csv = make_rc_time_median_fallback_submission(out_csv="submission.csv")
    print(f"Wrote fallback submission to: {out_csv}")

sub_check = pd.read_csv("submission.csv")
if list(sub_check.columns) != ["id", "pressure"]:
    raise ValueError(f"Bad submission columns: {sub_check.columns.tolist()}")
if len(sub_check) != len(pd.read_csv(SAMPLE_SUB_PATH)):
    raise ValueError(f"Bad submission length: {len(sub_check)}")
print("submission.csv looks valid.")
