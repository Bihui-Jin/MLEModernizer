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
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

dtypes = {
    "breath_id": np.int32,
    "u_in": np.float32,
    "u_out": np.int8,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "pressure": np.float32,
}
usecols = ["breath_id", "u_in", "u_out", "R", "C", "time_step", "pressure"]
train_df = pd.read_csv(train_path, usecols=usecols, dtype=dtypes)
assert "pressure" in train_df.columns, "Training data must contain 'pressure' column."

train_df["lag_u_in"] = train_df.groupby("breath_id")["u_in"].shift(1).fillna(0)
train_df["cum_u_in"] = train_df.groupby("breath_id")["u_in"].cumsum()
train_df["lag_u_out"] = train_df.groupby("breath_id")["u_out"].shift(1).fillna(0)
train_df["cum_u_out"] = train_df.groupby("breath_id")["u_out"].cumsum()

train_df["u_in_R"] = train_df["u_in"] * train_df["R"]
train_df["u_in_C"] = train_df["u_in"] * train_df["C"]
train_df["R_C"] = train_df["R"] * train_df["C"]
train_df["time_step_sq"] = train_df["time_step"] ** 2

train_df["cum_u_in_per_ts"] = train_df["cum_u_in"] / (train_df["time_step"] + 1e-3)
train_df["cum_u_out_per_ts"] = train_df["cum_u_out"] / (train_df["time_step"] + 1e-3)
train_df["R_div_C"] = train_df["R"] / train_df["C"]
train_df["u_in_time"] = train_df["u_in"] * train_df["time_step"]

train_df["delta_u_in"] = train_df["u_in"] - train_df["lag_u_in"]
train_df["delta_u_out"] = train_df["u_out"] - train_df["lag_u_out"]

train_df["breath_len"] = train_df.groupby("breath_id")["u_in"].transform("size")
train_df["mean_u_in"] = train_df.groupby("breath_id")["u_in"].transform("mean")
train_df["sum_u_in"] = train_df.groupby("breath_id")["u_in"].transform("sum")
train_df["mean_u_out"] = train_df.groupby("breath_id")["u_out"].transform("mean")
train_df["sum_u_out"] = train_df.groupby("breath_id")["u_out"].transform("sum")

feature_cols = [
    "u_in",
    "u_out",
    "R",
    "C",
    "time_step",
    "lag_u_in",
    "cum_u_in",
    "lag_u_out",
    "cum_u_out",
    "u_in_R",
    "u_in_C",
    "R_C",
    "time_step_sq",
    "cum_u_in_per_ts",
    "cum_u_out_per_ts",
    "R_div_C",
    "u_in_time",
    "delta_u_in",
    "delta_u_out",
    "breath_len",
    "mean_u_in",
    "sum_u_in",
    "mean_u_out",
    "sum_u_out",
]

X = train_df[feature_cols].values
y = train_df["pressure"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)



## === cell 1
model = HistGradientBoostingRegressor(
    loss="absolute_error",  # MAE loss
    max_iter=2000,
    learning_rate=0.005,
    max_depth=12,
    l2_regularization=0.05,
    random_state=42,
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (trained on split): {val_mae:.5f}")

model.fit(X, y)



## === cell 2
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
submission = pd.read_csv(sample_sub_path)

test_usecols = ["breath_id", "u_in", "u_out", "R", "C", "time_step"]
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype={col: dtypes[col] for col in test_usecols},
)

test_df["lag_u_in"] = test_df.groupby("breath_id")["u_in"].shift(1).fillna(0)
test_df["cum_u_in"] = test_df.groupby("breath_id")["u_in"].cumsum()
test_df["lag_u_out"] = test_df.groupby("breath_id")["u_out"].shift(1).fillna(0)
test_df["cum_u_out"] = test_df.groupby("breath_id")["u_out"].cumsum()

test_df["u_in_R"] = test_df["u_in"] * test_df["R"]
test_df["u_in_C"] = test_df["u_in"] * test_df["C"]
test_df["R_C"] = test_df["R"] * test_df["C"]
test_df["time_step_sq"] = test_df["time_step"] ** 2

test_df["cum_u_in_per_ts"] = test_df["cum_u_in"] / (test_df["time_step"] + 1e-3)
test_df["cum_u_out_per_ts"] = test_df["cum_u_out"] / (test_df["time_step"] + 1e-3)
test_df["R_div_C"] = test_df["R"] / test_df["C"]
test_df["u_in_time"] = test_df["u_in"] * test_df["time_step"]

test_df["delta_u_in"] = test_df["u_in"] - test_df["lag_u_in"]
test_df["delta_u_out"] = test_df["u_out"] - test_df["lag_u_out"]

test_df["breath_len"] = test_df.groupby("breath_id")["u_in"].transform("size")
test_df["mean_u_in"] = test_df.groupby("breath_id")["u_in"].transform("mean")
test_df["sum_u_in"] = test_df.groupby("breath_id")["u_in"].transform("sum")
test_df["mean_u_out"] = test_df.groupby("breath_id")["u_out"].transform("mean")
test_df["sum_u_out"] = test_df.groupby("breath_id")["u_out"].transform("sum")

test_pred = model.predict(test_df[feature_cols].values)
submission["pressure"] = test_pred

P_MIN = train_df["pressure"].min()
P_MAX = train_df["pressure"].max()
submission["pressure"] = submission["pressure"].clip(P_MIN, P_MAX)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with {len(submission)} rows.")
