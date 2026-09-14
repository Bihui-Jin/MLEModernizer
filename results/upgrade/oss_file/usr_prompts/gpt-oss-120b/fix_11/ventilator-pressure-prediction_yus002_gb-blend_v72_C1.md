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
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split  # added for validation split

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def add_features(df):
    """Create interaction / polynomial features plus cumulative and lag u_in."""
    df = df.copy()
    df["R_mul_C"] = df["R"] * df["C"]
    df["u_in_mul_R"] = df["u_in"] * df["R"]
    df["u_in_mul_C"] = df["u_in"] * df["C"]
    df["time_step_sq"] = df["time_step"] ** 2
    df["R_div_C"] = df["R"] / (df["C"] + 1e-6)
    df["u_in_div_R"] = df["u_in"] / (df["R"] + 1e-6)
    df["u_in_div_C"] = df["u_in"] / (df["C"] + 1e-6)
    df["time_step_mul_u_in"] = df["time_step"] * df["u_in"]
    df["cum_u_in"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()
    df["lag_u_in"] = df.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0)
    new_cols = [
        "R_mul_C",
        "u_in_mul_R",
        "u_in_mul_C",
        "time_step_sq",
        "R_div_C",
        "u_in_div_R",
        "u_in_div_C",
        "time_step_mul_u_in",
        "cum_u_in",
        "lag_u_in",
    ]
    df[new_cols] = df[new_cols].astype(np.float32)
    return df


dtype_map = {
    "R": np.int16,
    "C": np.int16,
    "breath_id": np.int32,
    "id": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}

train_path = "../input/ventilator-pressure-prediction/train.csv"
df_train = pd.read_csv(train_path, dtype=dtype_map)

df_train = add_features(df_train)

base_features = ["R", "C", "time_step", "u_in", "u_out"]
extra_features = [
    "R_mul_C",
    "u_in_mul_R",
    "u_in_mul_C",
    "time_step_sq",
    "R_div_C",
    "u_in_div_R",
    "u_in_div_C",
    "time_step_mul_u_in",
    "cum_u_in",
    "lag_u_in",
]
features = base_features + extra_features

X = df_train[features].astype(np.float32).to_numpy(copy=False)
y = df_train["pressure"].to_numpy(copy=False)

seed = 2021
set_seed(seed)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=seed, shuffle=True
)

gbr = HistGradientBoostingRegressor(
    max_iter=3000,
    learning_rate=0.005,
    max_depth=16,
    random_state=seed,
    verbose=0,
    l2_regularization=0.0,
    early_stopping=True,  # enable early stopping for faster convergence
    validation_fraction=0.1,  # use the provided validation split
    n_iter_no_change=20,  # stop after 20 rounds without improvement
    loss="absolute_error",
)

gbr.fit(
    X_train, y_train
)  # fit on the training split; early stopping uses validation data internally

train_pressure_vals = np.sort(df_train["pressure"].unique())


def nearest_pressure(arr, ref_vals):
    """Round each value in arr to the nearest value in ref_vals (sorted array)."""
    idx = np.searchsorted(ref_vals, arr)
    idx = np.clip(idx, 0, len(ref_vals) - 1)
    lower = np.maximum(idx - 1, 0)
    upper = idx
    lower_vals = ref_vals[lower]
    upper_vals = ref_vals[upper]
    use_upper = np.abs(arr - upper_vals) < np.abs(arr - lower_vals)
    return np.where(use_upper, upper_vals, lower_vals)




## === cell 1
test_path = "../input/ventilator-pressure-prediction/test.csv"
df_test = pd.read_csv(test_path, dtype=dtype_map)

df_test = add_features(df_test)

X_test = df_test[features].astype(np.float32).to_numpy(copy=False)

pred_raw = gbr.predict(X_test)

submission = pd.DataFrame({"id": df_test["id"], "pressure": pred_raw})
submission = submission[["id", "pressure"]]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path} with shape {submission.shape}")
