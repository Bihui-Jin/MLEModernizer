# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import random
from random import random as rd

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

unique_pressures = None
sorted_pressures = None
total_pressures_len = None


def init_pressure_lookup(df):
    """Initialize global pressure lookup arrays from a dataframe (expects a 'pressure' column)."""
    global unique_pressures, sorted_pressures, total_pressures_len
    unique_pressures = df["pressure"].unique()
    sorted_pressures = np.sort(unique_pressures)
    total_pressures_len = len(sorted_pressures)


def find_nearest_vec(predictions):
    """
    Vectorized nearest‑pressure lookup.
    Returns an array of the same shape as *predictions* where each value
    is the element of *sorted_pressures* closest to the corresponding prediction.
    """
    idx = np.searchsorted(sorted_pressures, predictions, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    left = np.maximum(idx - 1, 0)
    right = np.minimum(idx, total_pressures_len - 1)

    left_val = sorted_pressures[left]
    right_val = sorted_pressures[right]

    choose_left = np.abs(left_val - predictions) <= np.abs(right_val - predictions)
    return np.where(choose_left, left_val, right_val)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """Weighted combination of up to two prediction files."""
    scores = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        scores.append(public_lb_score)
        input_list[i] = pd.read_csv(
            input_list[i], usecols=["pressure"], dtype={"pressure": np.float32}
        ).pressure.values.ravel()
    output = 0
    l_sum = sum(scores)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (scores[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = input_list[0] * weight1 + input_list[1] * weight2
    return output


def baseline_submission():
    """Train a simple GBDT on the training rows and predict pressures for test."""
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    df_train = pd.read_csv(
        train_path,
        dtype={
            "R": np.int8,
            "C": np.int8,
            "u_in": np.float32,
            "u_out": np.int8,
            "time_step": np.float32,
            "pressure": np.float32,
        },
        usecols=["R", "C", "u_in", "u_out", "time_step", "pressure"],
    )
    init_pressure_lookup(df_train[["pressure"]])

    test_path = "../input/ventilator-pressure-prediction/test.csv"
    df_test = pd.read_csv(
        test_path,
        dtype={
            "R": np.int8,
            "C": np.int8,
            "u_in": np.float32,
            "u_out": np.int8,
            "time_step": np.float32,
            "id": np.int32,
        },
        usecols=["R", "C", "u_in", "u_out", "time_step", "id"],
    )

    feature_cols = ["R", "C", "u_in", "u_out", "time_step"]
    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train["pressure"].to_numpy(dtype=np.float32)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    X_test = df_test[feature_cols].to_numpy(dtype=np.float32)
    preds = model.predict(X_test)
    preds = find_nearest_vec(preds)

    pd.DataFrame({"id": df_test["id"], "pressure": preds}).to_csv(
        "submission_baseline.csv", index=False
    )


def g(dp):
    """Ensemble predictions from CSV files inside *dp*.
    If no files are found, fall back to the trained‑model baseline."""
    pred_files = [f for f in glob.iglob(f"{dp}/*") if f.lower().endswith(".csv")]
    if not pred_files:
        print("No prediction files found – using trained‑model baseline.")
        baseline_submission()
        return

    if sorted_pressures is None:
        dummy_train = pd.read_csv(
            "../input/ventilator-pressure-prediction/train.csv",
            usecols=["pressure"],
            nrows=5000,
            dtype={"pressure": np.float32},
        )
        init_pressure_lookup(dummy_train)

    file_count = len(pred_files)
    max_iters = 1000
    loop_time = min(file_count**2, max_iters)

    splits = max(1, file_count // 2)
    pred_files.sort()
    split_preds = []  # will hold NumPy arrays of shape (n_rows,)
    for i in range(splits):
        start = i * round(len(pred_files) / splits)
        end = (
            (i + 1) * round(len(pred_files) / splits)
            if i < splits - 1
            else len(pred_files)
        )
        split_preds.append(wc(pred_files[start:end]))

    n_rows = split_preds[0].shape[0]
    flist_arr = np.empty((len(split_preds), n_rows), dtype=np.float32)
    for idx, arr in enumerate(split_preds):
        flist_arr[idx] = arr

    pred_list = []
    n_files = flist_arr.shape[0]
    for i in range(loop_time):
        set_seed(i)
        weights = np.random.rand(n_files).astype(np.float32)
        weights /= weights.sum()
        order = np.argsort(-weights)
        temp = np.dot(weights[order], flist_arr[order])
        pred_list.append(temp)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        dtype={"id": np.int32, "pressure": np.float32},
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = find_nearest_vec(output["pressure"])
    output.to_csv("submission_gb_ensemble.csv", index=False)




## === cell 1
g("../input/gb-pred-files")
