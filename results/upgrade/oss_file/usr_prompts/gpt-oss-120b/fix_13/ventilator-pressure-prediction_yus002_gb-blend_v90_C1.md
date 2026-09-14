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

os.environ["OMP_NUM_THREADS"] = "8"  # use all available cores for faster tree fitting

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import random  # needed for set_seed




## === cell 1
usecols_train = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

dtypes = {
    "id": "uint16",
    "breath_id": "int32",
    "R": "uint8",
    "C": "uint8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "uint8",
    "pressure": "float32",
}

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=usecols_train,
    dtype=dtypes,
)

unique_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = unique_pressures.shape[0]


def find_nearest_vec(preds):
    """Vectorized nearest‑pressure lookup matching the original scalar logic."""
    idx = np.searchsorted(unique_pressures, preds, side="left")
    idx_clipped = np.clip(idx, 0, total_pressures_len - 1)

    lower = np.take(unique_pressures, np.maximum(idx_clipped - 1, 0))
    upper = np.take(unique_pressures, idx_clipped)

    beyond_max = idx >= total_pressures_len
    below_min = idx == 0

    choose_upper = np.abs(upper - preds) < np.abs(lower - preds)
    result = np.where(choose_upper, upper, lower)

    result = np.where(beyond_max, unique_pressures[-1], result)
    result = np.where(below_min, unique_pressures[0], result)
    return result


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state




## === cell 2
def train_and_predict():
    _ = set_seed(2021)

    global df_train, df_test

    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=usecols_test,
        dtype={k: dtypes[k] for k in usecols_test},
    )

    for df in (df_train, df_test):
        df["u_in_time"] = df["u_in"] * df["time_step"]
        df["u_out_time"] = df["u_out"] * df["time_step"]
        df["time_step_sq"] = df["time_step"] ** 2

        df["u_in_R"] = df["u_in"] * df["R"]
        df["u_in_C"] = df["u_in"] * df["C"]
        df["u_out_R"] = df["u_out"] * df["R"]
        df["u_out_C"] = df["u_out"] * df["C"]
        df["R_C"] = df["R"] * df["C"]

        df["u_in_R_C"] = df["u_in"] * df["R"] * df["C"]
        df["u_out_R_C"] = df["u_out"] * df["R"] * df["C"]
        df["time_step_R"] = df["time_step"] * df["R"]
        df["time_step_C"] = df["time_step"] * df["C"]

    for df in (df_train, df_test):
        df["cum_u_in"] = df.groupby("breath_id", sort=False)["u_in_time"].cumsum()
        df["cum_u_out"] = df.groupby("breath_id", sort=False)["u_out_time"].cumsum()

    feature_cols = [
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "u_in_time",
        "u_out_time",
        "time_step_sq",
        "u_in_R",
        "u_in_C",
        "u_out_R",
        "u_out_C",
        "R_C",
        "u_in_R_C",
        "u_out_R_C",
        "time_step_R",
        "time_step_C",
        "cum_u_in",
        "cum_u_out",
    ]

    X = df_train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df_train["pressure"].to_numpy(dtype=np.float32, copy=False)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=5000,  # more boosting rounds
        learning_rate=0.005,
        max_leaf_nodes=127,  # deeper trees
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train_split, y_train_split)

    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE (raw): {mae:.6f}")

    test_pred_raw = model.predict(
        df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)
    )
    test_pred = find_nearest_vec(test_pred_raw)

    submission = pd.DataFrame({"id": df_test["id"], "pressure": test_pred})
    submission.to_csv("submission.csv", index=False)
    print('Submission file "submission.csv" created.')

    del df_train, df_test, X, y, X_train_split, X_val, y_train_split, y_val
    gc.collect()




## === cell 3
train_and_predict()
