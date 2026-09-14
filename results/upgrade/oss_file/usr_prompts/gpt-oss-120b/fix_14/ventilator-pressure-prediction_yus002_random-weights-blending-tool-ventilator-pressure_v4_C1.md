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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import glob
import random
from random import random as rd
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  # retained for compatibility
import concurrent.futures  # added for parallel file reading



## === cell 1
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
)
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
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


def find_nearest_vectorized(preds):
    """
    Fully vectorized nearest‑pressure mapping using np.searchsorted.
    Produces the same output as repeatedly calling find_nearest on each element.
    """
    preds = np.asarray(preds, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds)
    idx = np.clip(idx, 0, total_pressures_len - 1)
    lower_idx = np.maximum(idx - 1, 0)
    upper_idx = idx

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[upper_idx]

    use_lower = np.abs(lower - preds) < np.abs(upper - preds)
    return np.where(use_lower, lower, upper)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def read_pressure(path):
    """Read only the pressure column as float32 – used in parallel I/O."""
    return pd.read_csv(path, usecols=["pressure"], dtype=np.float32).pressure.values


def wc(sublist, scores):
    """
    Compute the weighted combination of the first two prediction files in *sublist*.
    Only the pressure column is read as float32 for speed.
    """
    if len(sublist) == 0:
        return np.array([], dtype=np.float32)

    arrs = [read_pressure(f) for f in sublist[:2]]

    if len(arrs) == 1:
        return arrs[0]

    weight1 = (scores[1] / sum(scores)) + 0.1
    weight2 = 1 - weight1
    return arrs[0] * weight1 + arrs[1] * weight2


def build_features(df):
    """
    Create a feature matrix with cheap interaction terms.
    Cast everything to float32 to reduce memory bandwidth and speed up training.
    """
    X = df[["R", "C", "time_step", "u_in", "u_out"]].astype(np.float32).copy()
    X["R_mul_C"] = X["R"] * X["C"]
    X["u_in_mul_u_out"] = X["u_in"] * X["u_out"]
    X["time_mul_R"] = X["time_step"] * X["R"]
    X["time_mul_C"] = X["time_step"] * X["C"]
    return X


def build_features_np(df):
    """
    Faster NumPy‑only feature construction that avoids the extra pandas copy.
    Returns a float32 NumPy array with the same column order as `build_features`.
    """
    R = df["R"].values.astype(np.float32)
    C = df["C"].values.astype(np.float32)
    ts = df["time_step"].values.astype(np.float32)
    u_in = df["u_in"].values.astype(np.float32)
    u_out = df["u_out"].values.astype(np.float32)

    R_mul_C = R * C
    u_in_mul_u_out = u_in * u_out
    time_mul_R = ts * R
    time_mul_C = ts * C

    return np.column_stack(
        (R, C, ts, u_in, u_out, R_mul_C, u_in_mul_u_out, time_mul_R, time_mul_C)
    )


def g(dp):
    pred_files = [f for f in glob.glob(f"{dp}/*") if f.lower().endswith(".csv")]

    if len(pred_files) == 0:
        print("No external prediction files found – training a fallback model.")
        train_path = "../input/ventilator-pressure-prediction/train.csv"
        test_path = "../input/ventilator-pressure-prediction/test.csv"
        sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

        usecols_tr = ["R", "C", "time_step", "u_in", "u_out", "pressure"]
        dtypes = {
            "R": np.float32,
            "C": np.float32,
            "time_step": np.float32,
            "u_in": np.float32,
            "u_out": np.float32,
            "pressure": np.float32,
        }
        df_tr = pd.read_csv(train_path, usecols=usecols_tr, dtype=dtypes)
        df_te = pd.read_csv(
            test_path,
            usecols=usecols_tr[:-1],
            dtype={k: np.float32 for k in usecols_tr[:-1]},
        )

        X_train = build_features_np(df_tr)
        y_train = df_tr["pressure"].values  # already float32 from dtype
        X_test = build_features_np(df_te)

        del df_tr, df_te
        gc.collect()

        seed = 2021
        set_seed(seed)

        model = RandomForestRegressor(
            n_estimators=100,  # kept as in original logic
            max_depth=15,
            n_jobs=-1,  # utilize all available cores
            random_state=seed,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        preds = find_nearest_vectorized(preds)

        submission = pd.read_csv(sample_path)
        submission["pressure"] = preds
        submission.to_csv("submission.csv", index=False)
        print("Fallback submission written to submission.csv")
        return

    file_count = len(pred_files)
    loop_time = max(1, 500 // file_count)  # avoid zero division

    splits = max(1, file_count // 2)
    pred_files.sort()
    split_lists = []
    split_scores = []
    for i in range(splits):
        start = i * round(len(pred_files) / splits)
        end = (
            (i + 1) * round(len(pred_files) / splits)
            if i != splits - 1
            else len(pred_files)
        )
        sub = pred_files[start:end]
        split_lists.append(sub)

        scores = []
        for f in sub:
            score = int(os.path.basename(f).split(".")[1].split(" ")[0])
            scores.append(score)
        split_scores.append(scores)

    n_rows = pd.read_csv(
        split_lists[0][0],
        usecols=["pressure"],
        dtype=np.float32,
    ).shape[0]
    pred_matrix = np.empty((splits, n_rows), dtype=np.float32)

    pressures = {}
    max_workers = max(1, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for idx, sub in enumerate(split_lists):
            for pos, f in enumerate(sub[:2]):  # only first two files matter
                future = executor.submit(read_pressure, f)
                future_to_key[future] = (idx, pos)
        for future in concurrent.futures.as_completed(future_to_key):
            idx, pos = future_to_key[future]
            pressures[(idx, pos)] = future.result()

    for idx, (sub, sc) in enumerate(zip(split_lists, split_scores)):
        arrs = [pressures.get((idx, 0)), pressures.get((idx, 1))]
        arrs = [a for a in arrs if a is not None]
        if len(arrs) == 0:
            pred_matrix[idx] = np.empty(n_rows, dtype=np.float32)
        elif len(arrs) == 1:
            pred_matrix[idx] = arrs[0]
        else:
            weight1 = (sc[1] / sum(sc)) + 0.1
            weight2 = 1 - weight1
            pred_matrix[idx] = arrs[0] * weight1 + arrs[1] * weight2

    if splits == 1:
        blended = pred_matrix[0]  # 1‑D array
    else:
        if loop_time == 1:
            rs = np.random.RandomState(0)
            weights = rs.rand(splits).astype(np.float32)
            weights /= weights.sum()
            blended = np.dot(weights, pred_matrix)  # shape (n_rows,)
        else:
            rs = np.random.RandomState(0)
            weights = rs.rand(loop_time, splits).astype(np.float32)
            weights /= weights.sum(axis=1, keepdims=True)  # normalize each row
            blended = weights @ pred_matrix  # shape (loop_time, n_rows)

    median_pred = np.median(blended, axis=0)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = median_pred
    output["pressure"] = find_nearest_vectorized(output["pressure"].values)
    output.to_csv("ensemble_submission.csv", index=False)
    print("Ensemble submission written to ensemble_submission.csv")




## === cell 2
g("../input/gb-rwbt-files")
