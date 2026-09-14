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
import gc
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # retained for reference
from sklearn.ensemble import RandomForestRegressor  # use as improved fallback model



## === cell 1
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
    dtype={"pressure": np.float32},
)
sorted_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = len(sorted_pressures)


def find_nearest_vec(preds):
    """Vectorized version of find_nearest for an array of predictions."""
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.maximum(idx - 1, 0)
    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[idx]

    use_lower = np.abs(lower_val - preds) < np.abs(upper_val - preds)
    return np.where(use_lower, lower_val, upper_val)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def add_features(df):
    """Add a few inexpensive engineered features to improve model capacity."""
    df["R_C"] = df["R"].astype(np.float32) * df["C"].astype(np.float32)
    df["u_in_sq"] = df["u_in"] ** 2
    df["time_step_sq"] = df["time_step"] ** 2
    return df


def wc(input_list):
    """Blend predictions from a list of CSV files (high‑score submissions)."""
    series_list = []
    scores = []
    allow = [1348, 1698]  # public LB scores we accept

    for path in input_list:
        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue
        if public_lb_score in allow:
            scores.append(public_lb_score)
            series = pd.read_csv(
                path, usecols=["pressure"], dtype={"pressure": np.float32}
            ).pressure.values.ravel()
            series_list.append(series)

    if not series_list:
        return np.array([])

    if len(series_list) == 1:
        return series_list[0]

    score_sum = sum(scores)
    if score_sum == 0:
        weight1 = weight2 = 0.5
    else:
        weight1 = (scores[0] / score_sum) + 0.1
        weight2 = 1 - weight1

    output = series_list[0] * weight1 + series_list[1] * weight2
    if len(series_list) > 2:
        extra = np.mean(series_list[2:], axis=0)
        output = (output + extra) / 2

    return output


def g(dp):
    allowed_files = []
    allow = [1348, 1698]
    for i in glob.iglob(os.path.join(dp, "*")):
        try:
            file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue
        if file_lb in allow:
            allowed_files.append(i)

    if not allowed_files:
        print(
            "No high‑score submission files found. Training lightweight fallback model."
        )
        train_path = "../input/ventilator-pressure-prediction/train.csv"
        test_path = "../input/ventilator-pressure-prediction/test.csv"

        cols = ["R", "C", "u_in", "u_out", "time_step"]
        dtypes = {
            "R": np.int8,
            "C": np.int8,
            "u_in": np.float32,
            "u_out": np.int8,
            "time_step": np.float32,
        }
        df_train_local = pd.read_csv(
            train_path,
            usecols=cols + ["pressure"],
            dtype={**dtypes, "pressure": np.float32},
        )
        df_test_local = pd.read_csv(test_path, usecols=cols, dtype=dtypes)

        df_train_feat = add_features(df_train_local)
        df_test_feat = add_features(df_test_local)

        X_train = df_train_feat.drop(columns=["pressure"]).astype(np.float32).values
        y_train = df_train_feat["pressure"].astype(np.float32).values
        X_test = df_test_feat.astype(np.float32).values

        set_seed(2021)
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,  # already uses all cores
            random_state=2021,
        )
        model.fit(X_train, y_train)
        del X_train, y_train, df_train_local, df_train_feat
        gc.collect()

        preds = model.predict(X_test).astype(np.float32)
        preds = find_nearest_vec(preds)

        submission = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        submission["pressure"] = preds
        submission.to_csv("submission.csv", index=False)
        print("Fallback submission saved as submission.csv")
        return

    file_count = len(allowed_files)
    loop_time = 125
    splits = 2
    allowed_files.sort()
    flist = []
    for i in range(splits):
        start = i * round(len(allowed_files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(allowed_files) / splits)
        flist.append(wc(allowed_files[start:end]))

    pred_stack = np.vstack(flist)  # (splits, n_rows)

    random_weights = np.random.rand(loop_time, len(flist))
    random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)
    random_weights = np.sort(random_weights, axis=1)[:, ::-1]

    pred_matrix = random_weights @ pred_stack
    median_pred = np.median(pred_matrix, axis=0)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = find_nearest_vec(median_pred)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
