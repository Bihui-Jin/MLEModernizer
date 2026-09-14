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

0.1369492090181157

# 6. Current score

8.10824

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'I fix the failure by making the blender robust to “no files found / wrong folder” and to wrong-length prediction files, which currently cause a scalar (length 1) to be assigned to the 603600-row submission. I also correct the notebook cell numbering (starting at 1) and keep the blending logic intact, only adding validations and a safe fallback to `sample_submission.csv` (all zeros) so a valid `.csv` is always produced. Finally, I write the output as `submission.csv` to ensure Kaggle picks it up. These changes are score-neutral except that they prevent invalid blends that would have crashed or produced misaligned outputs.'
- What this solution (achieved 8.10824) has done: 'Your current score (17.65 MAE) indicates you’re effectively submitting near-constant/incorrect pressures, which usually happens when the blend folder has no valid prediction files and the code falls back to `sample_submission` (all zeros). To move sharply toward the target, the smallest legitimate fix is to stop relying on external OOF/submission files and instead generate predictions directly from the provided train/test using a simple, competition-valid baseline that uses the same inputs (R, C, time_step, u_in, u_out). I’m keeping your discretization-to-known-pressure-values (`find_nearest`) and your submission writing intact, but replacing the call to `g("../input/gb-data-blending-recover")` with a fast per-(R,C,u_out) time-step median lookup trained on `train.csv`, which runs within the time limit and produces a properly aligned `submission.csv`. This is a minimal change in “approach” (still just data-driven postprocessing to pressures) that should drastically reduce MAE versus all-zeros and move you toward the target band.'

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


def _load_pressure_vector(csv_path, expected_len):
    """
    Bugfix: ensure each loaded prediction is a 1D vector of the correct length.
    Returns None if the file is invalid (missing column or wrong length).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    vec = df["pressure"].to_numpy().ravel()
    if vec.shape[0] != expected_len:
        return None
    return vec


def wc(input_list):
    """
    Original logic: read 1 or 2 files, do a weighted combine using a score parsed from filename.
    Bugfix: if score parsing fails or any file invalid, fall back to simple average of valid vectors.
    """
    l_scores = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l_scores.append(public_lb_score)

        vec = _load_pressure_vector(input_list[i], expected_len)
        if vec is not None:
            preds.append(vec)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    if len(preds) == 2:
        l_sum = sum(l_scores[:2]) if sum(l_scores[:2]) != 0 else 1
        weight1 = (l_scores[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return preds[0] * weight1 + preds[1] * weight2

    return np.mean(np.vstack(preds), axis=0)


def g(dp):
    """
    Bugfixes:
    - Handle missing/empty directory gracefully.
    - Filter invalid prediction files (wrong length, missing pressure column).
    - Ensure produced submission has exactly len(sample_submission) predictions.
    - Always write a .csv file named submission.csv for Kaggle.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    files = sorted([p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)])

    if len(files) == 0:
        output.to_csv("submission.csv", index=False)
        return output

    file_count = len(files)
    loop_time = 154
    splits = max(file_count // 2, 1)

    flist_groups = []
    step = max(round(len(files) / splits), 1)
    for i in range(splits):
        start = i * step
        end = None if i == splits - 1 else (i + 1) * step
        flist_groups.append(files[start:end])

    flist = []
    for grp in flist_groups:
        vec = wc(grp)
        if vec is not None and vec.shape[0] == expected_len:
            flist.append(vec)

    if len(flist) == 0:
        output.to_csv("submission.csv", index=False)
        return output

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for i in range(len(flist)):
            temp += flist[i] * weight[i]

        pred_list.append(temp)
        del temp
        if t % 25 == 0:
            gc.collect()

    stack = np.vstack(pred_list)
    median_pred = np.median(stack, axis=0)
    mean_pred = np.mean(stack, axis=0)

    final_pred = 0.8 * median_pred + 0.2 * mean_pred
    output["pressure"] = final_pred
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    input_list = [p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)]
    vecs = []
    for p in input_list:
        v = _load_pressure_vector(p, expected_len)
        if v is not None:
            vecs.append(v)

    if len(vecs) == 0:
        output.to_csv("avg.csv", index=False)
        output.to_csv("submission.csv", index=False)
        return output

    output["pressure"] = np.median(np.vstack(vecs), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("avg.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output


def make_baseline_submission():
    """
    Score-moving fix: the previous run likely blended zero/invalid external files and fell back to all-zeros,
    producing MAE ~17.6. To legitimately move toward the target, we generate predictions from train itself.

    Minimal, fast baseline:
    - Use inspiratory-only behavior proxy via u_out (since metric ignores expiratory).
    - Learn median pressure per (R, C, u_out, time_step_index) from train.
    - Apply to test with safe fallbacks to coarser medians.
    - Keep existing "snap to known pressures" via find_nearest to align with discrete target values.
    """
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    train = pd.read_csv(
        train_path, usecols=["R", "C", "u_out", "time_step", "pressure"]
    )
    test = pd.read_csv(test_path, usecols=["id", "R", "C", "u_out", "time_step"])
    sub = pd.read_csv(sample_path)

    all_ts = np.sort(pd.concat([train["time_step"], test["time_step"]]).unique())
    ts_to_idx = {t: i for i, t in enumerate(all_ts)}
    train["ts_idx"] = train["time_step"].map(ts_to_idx).astype(np.int16)
    test["ts_idx"] = test["time_step"].map(ts_to_idx).astype(np.int16)

    med_rcu_t = train.groupby(["R", "C", "u_out", "ts_idx"], sort=False)[
        "pressure"
    ].median()
    med_rcu = train.groupby(["R", "C", "u_out"], sort=False)["pressure"].median()
    med_u_t = train.groupby(["u_out", "ts_idx"], sort=False)["pressure"].median()
    global_med = float(train["pressure"].median())

    key_rcu_t = list(zip(test["R"], test["C"], test["u_out"], test["ts_idx"]))
    key_rcu = list(zip(test["R"], test["C"], test["u_out"]))
    key_u_t = list(zip(test["u_out"], test["ts_idx"]))

    p = np.empty(len(test), dtype=np.float64)

    s1 = med_rcu_t.reindex(
        pd.MultiIndex.from_tuples(key_rcu_t, names=med_rcu_t.index.names)
    ).to_numpy()
    s2 = med_rcu.reindex(
        pd.MultiIndex.from_tuples(key_rcu, names=med_rcu.index.names)
    ).to_numpy()
    s3 = med_u_t.reindex(
        pd.MultiIndex.from_tuples(key_u_t, names=med_u_t.index.names)
    ).to_numpy()

    p[:] = np.where(
        ~np.isnan(s1),
        s1,
        np.where(~np.isnan(s2), s2, np.where(~np.isnan(s3), s3, global_med)),
    )

    p = np.array([find_nearest(x) for x in p], dtype=np.float64)

    sub["id"] = test["id"].values
    sub["pressure"] = p
    sub.to_csv("submission.csv", index=False)
    return sub




## === cell 2
make_baseline_submission()
