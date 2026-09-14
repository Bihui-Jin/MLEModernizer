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

No external packages required in the script and installed.

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
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"

BASE_FEATURES = ["R", "C", "time_step", "u_in", "u_out", "breath_id"]
TARGET_COL = "pressure"
USE_COLS = BASE_FEATURES + [TARGET_COL]

dtype_map = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}

train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=USE_COLS,
    dtype={k: dtype_map[k] for k in USE_COLS},
    engine="c",
)

p_min = train_df[TARGET_COL].min()
p_max = train_df[TARGET_COL].max()

test_df = pd.read_csv(
    TEST_PATH,
    usecols=["id"] + BASE_FEATURES,
    dtype={k: dtype_map[k] for k in ["id"] + BASE_FEATURES},
    engine="c",
)


def add_engineered_features(df):
    """Add cheap interaction, lag, and short‑window temporal features efficiently."""
    df["R_div_C"] = df["R"] / df["C"]
    df["R_mul_C"] = df["R"] * df["C"]
    df["u_in_R"] = df["u_in"] * df["R"]
    df["u_in_C"] = df["u_in"] * df["C"]
    df["time_R"] = df["time_step"] * df["R"]
    df["time_C"] = df["time_step"] * df["C"]
    df["u_in_time"] = df["u_in"] * df["time_step"]
    df["u_out_time"] = df["u_out"] * df["time_step"]
    df["u_out_R"] = df["u_out"] * df["R"]
    df["u_out_C"] = df["u_out"] * df["C"]
    df["time_step_sq"] = df["time_step"] ** 2
    df["is_inspiratory"] = (df["u_in"] > 0).astype(np.int8)

    g = df.groupby("breath_id", observed=True)

    df["cum_u_in"] = g["u_in"].cumsum()
    df["u_in_lag"] = g["u_in"].shift(fill_value=0)
    df["u_out_lag"] = g["u_out"].shift(fill_value=0)
    df["u_in_diff"] = df["u_in"] - df["u_in_lag"]
    df["u_out_diff"] = df["u_out"] - df["u_out_lag"]

    df["u_in_roll_mean_3"] = g["u_in"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["u_in_roll_std_3"] = g["u_in"].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    df["u_out_roll_mean_3"] = g["u_out"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["breath_len"] = g["breath_id"].transform("size")

    return df


train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)

FEATURE_COLS = BASE_FEATURES + [
    "R_div_C",
    "R_mul_C",
    "u_in_R",
    "u_in_C",
    "time_R",
    "time_C",
    "u_in_time",
    "u_out_time",
    "u_out_R",
    "u_out_C",
    "cum_u_in",
    "u_in_lag",
    "u_out_lag",
    "u_in_diff",
    "u_out_diff",
    "is_inspiratory",
    "breath_len",
    "time_step_sq",
    "u_in_roll_mean_3",
    "u_in_roll_std_3",
    "u_out_roll_mean_3",
]

X = train_df[FEATURE_COLS]
y = train_df[TARGET_COL]

X_tr_df, X_val_df, y_tr_df, y_val_df = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=False
)

X_tr = X_tr_df.to_numpy(dtype=np.float32, copy=False)
X_val = X_val_df.to_numpy(dtype=np.float32, copy=False)
y_tr = y_tr_df.to_numpy(dtype=np.float32, copy=False)
y_val = y_val_df.to_numpy(dtype=np.float32, copy=False)

del train_df, X, y, X_tr_df, X_val_df, y_tr_df, y_val_df
gc.collect()

model = HistGradientBoostingRegressor(
    max_iter=4000,  # keep original boosting rounds
    learning_rate=0.005,
    max_depth=12,  # keep original tree depth
    random_state=42,
)

model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)

coeff = np.polyfit(val_pred, y_val, 1)  # slope, intercept
val_pred_calibrated = coeff[0] * val_pred + coeff[1]

print("Validation MAE (raw):", mean_absolute_error(y_val, val_pred))
print("Validation MAE (calibrated):", mean_absolute_error(y_val, val_pred_calibrated))




## === cell 1
test_features = test_df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
test_pred = model.predict(test_features)
test_pred_calibrated = coeff[0] * test_pred + coeff[1]

submission = pd.DataFrame(
    {"id": test_df["id"], "pressure": test_pred_calibrated.astype("float32")}
)

submission["pressure"] = submission["pressure"].clip(p_min, p_max)

submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
