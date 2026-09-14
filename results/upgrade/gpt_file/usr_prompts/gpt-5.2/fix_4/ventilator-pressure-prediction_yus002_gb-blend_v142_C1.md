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

0.137018805076577

# 6. Current score

9.90136

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92097) has done: 'The crash happens because `g()` assumes there are prediction files in `../input/gb-data-blending-recover`, but in your environment that directory doesn’t exist (so it silently builds empty arrays and ends up producing a single scalar median). I add strict checks so `g()` fails early with a clear message if no valid prediction CSVs are found, and I implement a safe fallback that still produces a valid submission by generating a baseline prediction from the provided `test.csv` using the median training pressure per `(R, C, time_step)` (and then snapping to the nearest allowed pressure, preserving your `find_nearest` core calibration step). This keeps the overall semantics (predict pressures and quantize to known pressure levels) and ensures a `.csv` submission is always written end-to-end. I also keep paths compatible with Kaggle’s `/kaggle/input/...` layout while not changing your existing relative paths.'
- What this solution (achieved 9.5126) has done: 'Your current score (9.92097 MAE) is far from the target (0.1370), so the issue is that you’re effectively submitting a very weak fallback instead of a meaningful model/blend. To move the score sharply toward the target while preserving your core “predict then snap to nearest allowed pressure” logic, I keep `g()` and the quantization exactly as-is but upgrade the fallback to a stronger, still-simple lookup: median pressure by `(R, C, u_out, time_step)` plus a cumulative-`u_in` correction per `(R, C, u_out)` computed from train. This remains a deterministic train-derived baseline (no new model/training loop) and should substantially reduce MAE versus the current coarse median-by-time_step only. The script still write `submission.csv` end-to-end even when the external blend directory is missing.'
- What this solution (achieved 9.90136) has done: 'Your current MAE (9.5126, lower-is-better) is still very far from the target (0.1370), which strongly suggests the code is nearly always falling back (no external blend files) and the fallback predictor is too weak. To move sharply toward the target while preserving your “lookup-based prediction + snap to nearest allowed pressure” core semantics, I keep `g()`/`wc()`/`find_nearest()` intact and only strengthen the fallback using a deterministic, train-derived KNN-by-key lookup that matches the metric better. Concretely, I replace the fallback with a group-median keyed on `(R,C,u_out,u_in_bin,time_step)` (binning `u_in` is a minimal, non-model change) and keep the final quantization step unchanged. This should materially reduce MAE without introducing any new model architecture/training loop and still always write a valid `submission.csv`.'

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
    Reads one or two prediction files and blends them based on their score encoded in filename.
    Original logic preserved; now includes minimal validation and robust parsing.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i].split("/")[-1]
        try:
            public_lb_score = int(fn.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        dfp = pd.read_csv(input_list[i])
        if "pressure" not in dfp.columns:
            raise ValueError(f"File {input_list[i]} missing 'pressure' column.")
        arrs.append(dfp["pressure"].to_numpy().ravel())

    base_len = len(arrs[0])
    for j, a in enumerate(arrs):
        if len(a) != base_len:
            raise ValueError(
                f"Prediction length mismatch in {input_list[j]}: {len(a)} vs {base_len}"
            )

    output = 0
    l_sum = sum(l)

    if len(arrs) == 1:
        output = arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = arrs[0] * weight1 + arrs[1] * weight2
    return output


def _fallback_baseline_submission(out_path="submission.csv"):
    """
    Change (score-improving, still deterministic & same overall semantics):
    The previous fallback was too coarse, so MAE stayed ~9.5.
    We keep the same "train-derived lookup -> predict -> snap to nearest allowed pressure" approach,
    but strengthen the lookup by incorporating u_in (binned) in addition to (R,C,u_out,time_step).
    This better matches how pressure responds to valve opening, without introducing any new model/training loop.
    """
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    df_test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_path)

    tr = df_train.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    te = df_test.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    BIN_W = 2.0  # 0..100 => ~50 bins; small enough to add signal, large enough to generalize
    tr["u_in_bin"] = (np.rint(tr["u_in"].to_numpy() / BIN_W) * BIN_W).astype(np.float32)
    te["u_in_bin"] = (np.rint(te["u_in"].to_numpy() / BIN_W) * BIN_W).astype(np.float32)

    key5 = ["R", "C", "u_out", "time_step", "u_in_bin"]
    lut5 = tr.groupby(key5, sort=False)["pressure"].median().rename("p5").reset_index()

    key4 = ["R", "C", "u_out", "time_step"]
    lut4 = tr.groupby(key4, sort=False)["pressure"].median().rename("p4").reset_index()
    key3 = ["R", "C", "time_step"]
    lut3 = tr.groupby(key3, sort=False)["pressure"].median().rename("p3").reset_index()

    global_med = float(tr["pressure"].median())

    tem = te.merge(lut5, on=key5, how="left")
    tem = tem.merge(lut4, on=key4, how="left")
    tem = tem.merge(lut3, on=key3, how="left")

    pred = tem["p5"].to_numpy()
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = tem.loc[mask, "p4"].to_numpy()
        mask = np.isnan(pred)
    if mask.any():
        pred[mask] = tem.loc[mask, "p3"].to_numpy()
        mask = np.isnan(pred)
    if mask.any():
        pred[mask] = global_med

    tem["_pred"] = pred.astype(np.float64)
    tem = tem.sort_values("id", kind="mergesort")
    pred = tem["_pred"].to_numpy()

    sub["pressure"] = pred
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return sub


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    if len(l) == 0:
        _fallback_baseline_submission("submission.csv")
        return

    file_count = len(l)
    loop_time = 154
    splits = file_count // 2

    if splits < 1:
        splits = 1

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
        if len(flist[i]) == 0:
            continue
        flist[i] = wc(flist[i])

    if len(flist) == 0:
        _fallback_baseline_submission("submission.csv")
        return

    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
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
    med = np.median(np.vstack(pred_list), axis=0)

    if len(med) != len(output):
        raise ValueError(
            f"Pred length {len(med)} does not match submission length {len(output)}"
        )

    output["pressure"] = med
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
