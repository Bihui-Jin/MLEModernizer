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

0.1373797655861054

# 6. Current score

3.29709

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13623) has done: 'The crash happens because `g()` is trying to blend external submission files from a dataset path that doesn’t exist here, leaving `pred_list` with wrong-shaped arrays (or scalars), so the median becomes length 1 and can’t be assigned to 603600 rows. I make `g()` robust: it only use valid `.csv` prediction files that match the sample submission length; otherwise it fall back to a simple, legitimate baseline model (mean pressure per (R,C,time_step,u_out) with a safe global fallback) so a valid submission is always produced. I also ensure the output filename ends with `.csv` and is written to the working directory. This keeps the “blending then snap-to-nearest-pressure-grid” core idea, but guarantees end-to-end execution in this environment.'
- What this solution (achieved 3.29709) has done: 'Your current score (8.136) is far worse than the target (0.137), so we should improve the fallback model that runs when no external blend files exist (which is the case here). I keep the blending/snap-to-pressure-grid core logic intact, but replace the weak group-mean fallback with a stronger, still-simple “u_in integration” baseline commonly used for this competition (using per-(R,C) linear calibration from train and applying it to test), then snap predictions to the nearest valid pressure. I also ensure the fallback trains only on inspiratory-phase rows (u_out==0) since the metric ignores expiratory phase, and I keep output format/id alignment identical so a valid `submission.csv` is always produced.'

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
    """
    Weighted combine for 1-2 files. Original notebook expects a special filename format
    containing a score; keep behavior but make it robust if parsing fails.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i]
        try:
            public_lb_score = int(fn.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        arrs.append(pd.read_csv(fn)["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else 1
    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area"] = df["u_in"] * df["time_step"]
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["area_cumsum"] = df.groupby("breath_id")["area"].cumsum()
    return df


def _baseline_predictions_from_train(train_df, test_df):
    """
    Score-improving fallback baseline when no external blend files exist.

    Uses a simple per-(R,C) linear calibration:
      pressure ~= a*(u_in_cumsum) + b*(area_cumsum) + c
    fitted on inspiratory phase only (u_out==0), then applied to test.
    This is still lightweight, uses only provided data, and keeps evaluation semantics.
    """
    key_cols = ["R", "C"]

    tr = _add_engineered_features(train_df)
    te = _add_engineered_features(test_df)

    tr_fit = tr.loc[
        tr["u_out"] == 0, ["R", "C", "u_in_cumsum", "area_cumsum", "pressure"]
    ].copy()

    Xg = tr_fit[["u_in_cumsum", "area_cumsum"]].to_numpy(dtype=np.float64)
    yg = tr_fit["pressure"].to_numpy(dtype=np.float64)
    A = np.c_[Xg, np.ones(len(Xg), dtype=np.float64)]
    coef_g, _, _, _ = np.linalg.lstsq(A, yg, rcond=None)  # [a,b,c]

    coef_map = {}
    for (r, c), g in tr_fit.groupby(key_cols, sort=False):
        X = g[["u_in_cumsum", "area_cumsum"]].to_numpy(dtype=np.float64)
        y = g["pressure"].to_numpy(dtype=np.float64)
        A = np.c_[X, np.ones(len(X), dtype=np.float64)]
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        coef_map[(int(r), int(c))] = coef

    Xte = te[["u_in_cumsum", "area_cumsum"]].to_numpy(dtype=np.float64)
    rc = list(zip(te["R"].astype(int).to_numpy(), te["C"].astype(int).to_numpy()))
    pred = np.empty(len(te), dtype=np.float64)
    for i, k in enumerate(rc):
        a, b, c0 = coef_map.get(k, coef_g)
        pred[i] = a * Xte[i, 0] + b * Xte[i, 1] + c0

    pred = np.clip(pred, float(sorted_pressures[0]), float(sorted_pressures[-1]))
    return pred


def g(dp):
    """
    Original intent: read multiple external prediction csvs from dp, blend them with random weights,
    take median across loops, then snap to nearest valid pressure.
    Fix: make it robust to missing/invalid dp; always output a valid submission CSV.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    output = pd.read_csv(sample_path)
    n = len(output)

    files = []
    if isinstance(dp, str) and os.path.isdir(dp):
        for fp in glob.iglob(os.path.join(dp, "*")):
            if fp.lower().endswith(".csv"):
                files.append(fp)
    files.sort()

    valid_preds = []
    for fp in files:
        try:
            dfp = pd.read_csv(fp, usecols=["pressure"])
            arr = dfp["pressure"].to_numpy().ravel()
            if len(arr) == n and np.isfinite(arr).all():
                valid_preds.append(arr)
        except Exception:
            continue

    if len(valid_preds) == 0:
        df_test = pd.read_csv(test_path)
        pred = _baseline_predictions_from_train(df_train, df_test)
        output["pressure"] = pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return output

    loop_time = 154
    splits = max(1, len(valid_preds) // 2)

    flist = []
    if splits == 1:
        flist = [np.mean(np.vstack(valid_preds), axis=0)]
    else:
        half = len(valid_preds) // 2
        flist = [
            np.mean(np.vstack(valid_preds[:half]), axis=0),
            np.mean(np.vstack(valid_preds[half:]), axis=0),
        ]

    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    blended = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = blended
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(input_list):
    for i in range(len(input_list)):
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = sum(input_list) / len(input_list)
    return output




## === cell 2
g("../input/gb-data-blending-recover")
