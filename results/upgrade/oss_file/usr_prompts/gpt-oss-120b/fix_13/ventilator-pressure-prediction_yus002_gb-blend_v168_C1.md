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
import random
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error




## === cell 1
dtype_train = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtype_test = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype=dtype_train,
    low_memory=False,
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest_vectorized(predictions):
    """Snap an array of predictions to the nearest training pressure values."""
    insert_idx = np.searchsorted(sorted_pressures, predictions, side="left")
    insert_idx = np.clip(insert_idx, 0, total_pressures_len - 1)
    lower_idx = np.maximum(insert_idx - 1, 0)
    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[insert_idx]
    choose_lower = np.abs(lower_val - predictions) <= np.abs(upper_val - predictions)
    return np.where(choose_lower, lower_val, upper_val)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state




## === cell 2
def train_and_predict():
    global df_train

    set_seed(2021)

    base_features = ["R", "C", "time_step", "u_in", "u_out"]

    df_train["R_C"] = df_train["R"] * df_train["C"]
    df_train["u_in_u_out"] = df_train["u_in"] * df_train["u_out"]
    df_train["time_R"] = df_train["time_step"] * df_train["R"]
    df_train["time_C"] = df_train["time_step"] * df_train["C"]
    df_train["R_div_C"] = df_train["R"] / (df_train["C"] + 1e-6)
    df_train["time_sq"] = df_train["time_step"] ** 2

    grp = df_train.groupby("breath_id", sort=False)
    df_train["u_in_cumsum"] = grp["u_in"].cumsum()
    df_train["u_out_cumsum"] = grp["u_out"].cumsum()
    df_train["time_step_norm"] = df_train["time_step"] / df_train["time_step"].max()
    df_train["u_in_prev"] = grp["u_in"].shift(1).fillna(0)
    df_train["u_out_prev"] = grp["u_out"].shift(1).fillna(0)
    df_train["u_in_diff"] = df_train["u_in"] - df_train["u_in_prev"]
    df_train["u_out_diff"] = df_train["u_out"] - df_train["u_out_prev"]

    feature_cols = base_features + [
        "R_C",
        "u_in_u_out",
        "time_R",
        "time_C",
        "R_div_C",
        "time_sq",
        "u_in_cumsum",
        "u_out_cumsum",
        "time_step_norm",
        "u_in_prev",
        "u_out_prev",
        "u_in_diff",
        "u_out_diff",
    ]

    target_col = "pressure"

    X = df_train[feature_cols].astype("float32").values
    y = df_train[target_col].astype("float32").values

    del df_train
    gc.collect()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=2021
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.01,  # smaller step size
        max_iter=4000,  # more boosting rounds
        max_leaf_nodes=31,
        max_bins=128,
        random_state=2021,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_valid)
    mae_raw = mean_absolute_error(y_valid, val_pred)
    mae_snapped = mean_absolute_error(y_valid, find_nearest_vectorized(val_pred))
    print(f"Validation MAE (raw): {mae_raw:.6f}")
    print(f"Validation MAE (snapped): {mae_snapped:.6f}")

    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        dtype=dtype_test,
        low_memory=False,
    )
    df_test["R_C"] = df_test["R"] * df_test["C"]
    df_test["u_in_u_out"] = df_test["u_in"] * df_test["u_out"]
    df_test["time_R"] = df_test["time_step"] * df_test["R"]
    df_test["time_C"] = df_test["time_step"] * df_test["C"]
    df_test["R_div_C"] = df_test["R"] / (df_test["C"] + 1e-6)
    df_test["time_sq"] = df_test["time_step"] ** 2

    grp_test = df_test.groupby("breath_id", sort=False)
    df_test["u_in_cumsum"] = grp_test["u_in"].cumsum()
    df_test["u_out_cumsum"] = grp_test["u_out"].cumsum()
    df_test["time_step_norm"] = df_test["time_step"] / df_test["time_step"].max()
    df_test["u_in_prev"] = grp_test["u_in"].shift(1).fillna(0)
    df_test["u_out_prev"] = grp_test["u_out"].shift(1).fillna(0)
    df_test["u_in_diff"] = df_test["u_in"] - df_test["u_in_prev"]
    df_test["u_out_diff"] = df_test["u_out"] - df_test["u_out_prev"]

    X_test = df_test[feature_cols].astype("float32").values
    test_pred = model.predict(X_test)

    test_pred = np.clip(test_pred, y.min(), y.max())

    submission = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    submission["pressure"] = test_pred
    submission = submission.sort_values("id")
    submission.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' written.")

    del X, y, X_train, X_valid, y_train, y_valid, X_test, test_pred
    gc.collect()


train_and_predict()
