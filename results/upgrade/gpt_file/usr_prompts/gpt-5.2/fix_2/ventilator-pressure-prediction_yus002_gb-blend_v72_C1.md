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

0.1492447888535536

# 6. Current score

4.12137

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.12137) has done: 'I fix the failure by making the blending code robust when the expected `../input/gb-blending` directory doesn’t exist (or has no valid submission files), which currently leads to a 1-element prediction array and a length mismatch. To ensure the notebook always runs end-to-end and produces a valid `.csv` submission, I add a safe fallback that generates a baseline submission from the provided `test.csv` (per-breath mean pressure by `(R,C,time_step,u_in,u_out)` with sensible fallbacks). I also make file discovery stricter (only CSVs with the right length/columns are used) and always write `submission.csv` with the required `id,pressure` columns. These changes keep the original blending core logic intact when blend inputs are present, and otherwise avoid crashing while producing a reasonable score-improving baseline versus all-zeros.'

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
    Original intent: read 1-2 submission files and compute a weighted combination.
    Bugfix: make parsing of score-in-filename optional/robust; validate length/columns.
    """
    arrs = []
    scores = []
    for path in input_list:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "pressure" not in df.columns:
            continue
        p = np.asarray(df["pressure"]).ravel()
        if p.shape[0] != 603600:
            continue

        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        scores.append(public_lb_score)
        arrs.append(p)

    if len(arrs) == 0:
        return None

    if len(arrs) == 1:
        return arrs[0]

    l_sum = sum(scores) if sum(scores) != 0 else 1
    weight1 = (scores[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return arrs[0] * weight1 + arrs[1] * weight2


def _build_fallback_predictions():
    """
    Fallback used ONLY when blend inputs are missing/invalid.
    Produces a valid submission using train-derived mean pressure by (R,C,time_step,u_in,u_out),
    with progressively coarser fallbacks to avoid NaNs.
    """
    test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )
    train = df_train  # already loaded

    key_cols = ["R", "C", "time_step", "u_in", "u_out"]
    mean_map_exact = train.groupby(key_cols, sort=False)["pressure"].mean()

    train2 = train.copy()
    test2 = test.copy()
    train2["time_step_r"] = train2["time_step"].round(2)
    test2["time_step_r"] = test2["time_step"].round(2)
    train2["u_in_r"] = train2["u_in"].round(1)
    test2["u_in_r"] = test2["u_in"].round(1)

    key_cols_r = ["R", "C", "time_step_r", "u_in_r", "u_out"]
    mean_map_round = train2.groupby(key_cols_r, sort=False)["pressure"].mean()

    key_cols_coarse = ["R", "C", "time_step_r", "u_out"]
    mean_map_coarse = train2.groupby(key_cols_coarse, sort=False)["pressure"].mean()

    global_mean = float(train["pressure"].mean())

    exact_key = list(
        zip(test["R"], test["C"], test["time_step"], test["u_in"], test["u_out"])
    )
    pred = pd.Series(exact_key).map(mean_map_exact).to_numpy(dtype=float)

    miss = np.isnan(pred)
    if miss.any():
        r_key = list(
            zip(
                test2.loc[miss, "R"],
                test2.loc[miss, "C"],
                test2.loc[miss, "time_step_r"],
                test2.loc[miss, "u_in_r"],
                test2.loc[miss, "u_out"],
            )
        )
        pred2 = pd.Series(r_key).map(mean_map_round).to_numpy(dtype=float)
        pred[miss] = pred2

    miss = np.isnan(pred)
    if miss.any():
        c_key = list(
            zip(
                test2.loc[miss, "R"],
                test2.loc[miss, "C"],
                test2.loc[miss, "time_step_r"],
                test2.loc[miss, "u_out"],
            )
        )
        pred3 = pd.Series(c_key).map(mean_map_coarse).to_numpy(dtype=float)
        pred[miss] = pred3

    miss = np.isnan(pred)
    if miss.any():
        pred[miss] = global_mean

    pred = np.array([find_nearest(x) for x in pred], dtype=float)
    return test["id"].to_numpy(), pred


def g(dp):
    """
    Original blending function.
    Bugfixes:
    - Handle missing/empty directory and invalid files; avoid creating length-1 outputs.
    - Ensure output file is a valid Kaggle submission with .csv suffix (submission.csv).
    """
    files = []
    if dp is not None and os.path.isdir(dp):
        for p in glob.iglob(f"{dp}/*"):
            if os.path.isfile(p) and p.lower().endswith(".csv"):
                files.append(p)
    files.sort()

    if len(files) == 0:
        sub = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        ids, pred = _build_fallback_predictions()
        pred_df = pd.DataFrame({"id": ids, "pressure": pred})
        pred_df = pred_df.sort_values("id")
        sub = sub.sort_values("id")
        sub["pressure"] = pred_df["pressure"].to_numpy()
        sub.to_csv("submission.csv", index=False)
        return sub

    file_count = len(files)
    loop_time = 150

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        flist.append(files[start:end])

    preds = []
    for group in flist:
        p = wc(group)
        if p is not None:
            preds.append(p)
    flist = preds

    if len(flist) == 0:
        sub = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        ids, pred = _build_fallback_predictions()
        pred_df = pd.DataFrame({"id": ids, "pressure": pred}).sort_values("id")
        sub = sub.sort_values("id")
        sub["pressure"] = pred_df["pressure"].to_numpy()
        sub.to_csv("submission.csv", index=False)
        return sub

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(flist)):
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

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    blended = np.median(np.vstack(pred_list), axis=0)

    if blended.shape[0] != len(output):
        ids, pred = _build_fallback_predictions()
        pred_df = pd.DataFrame({"id": ids, "pressure": pred}).sort_values("id")
        output = output.sort_values("id")
        output["pressure"] = pred_df["pressure"].to_numpy()
        output.to_csv("submission.csv", index=False)
        return output

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
g("../input/gb-blending")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
