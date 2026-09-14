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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor  # faster on large data


def resolve_path(relative_path: str) -> Path:
    kaggle_root = Path("/kaggle/input")
    possible = [
        kaggle_root / relative_path,
        Path("./data") / relative_path,
        Path("./input") / relative_path,
        Path(relative_path),  # fallback to current directory
    ]
    for p in possible:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not locate {relative_path}")


train_path = resolve_path("ventilator-pressure-prediction/train.csv")
test_path = resolve_path("ventilator-pressure-prediction/test.csv")

dtype_spec = {
    "R": np.int8,
    "C": np.int8,
    "breath_id": np.int32,
    "id": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}

train = pd.read_csv(train_path, dtype=dtype_spec, usecols=list(dtype_spec.keys()))
test = pd.read_csv(
    test_path,
    dtype={k: v for k, v in dtype_spec.items() if k != "pressure"},
    usecols=[c for c in dtype_spec if c != "pressure"],
)

train["RC"] = train["R"].astype(np.int16) * train["C"].astype(np.int16)
test["RC"] = test["R"].astype(np.int16) * test["C"].astype(np.int16)

train["cum_u_in"] = train.groupby("breath_id")["u_in"].cumsum()
train["cum_u_out"] = train.groupby("breath_id")["u_out"].cumsum()
test["cum_u_in"] = test.groupby("breath_id")["u_in"].cumsum()
test["cum_u_out"] = test.groupby("breath_id")["u_out"].cumsum()

train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0)
train["u_out_lag1"] = train.groupby("breath_id")["u_out"].shift(1).fillna(0)
train["time_step_lag1"] = train.groupby("breath_id")["time_step"].shift(1).fillna(0)

test["u_in_lag1"] = test.groupby("breath_id")["u_in"].shift(1).fillna(0)
test["u_out_lag1"] = test.groupby("breath_id")["u_out"].shift(1).fillna(0)
test["time_step_lag1"] = test.groupby("breath_id")["time_step"].shift(1).fillna(0)

train["u_in_times_u_out"] = train["u_in"] * train["u_out"]
test["u_in_times_u_out"] = test["u_in"] * test["u_out"]

train["R_div_C"] = train["R"] / train["C"]
test["R_div_C"] = test["R"] / test["C"]

train["time_step_sq"] = train["time_step"] ** 2
test["time_step_sq"] = test["time_step"] ** 2

pressure_min = train["pressure"].min()
pressure_max = train["pressure"].max()




## === cell 1
train_split, val_split = train_test_split(train, test_size=0.2, random_state=42)

feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "RC",
    "cum_u_in",
    "cum_u_out",
    "u_in_lag1",
    "u_out_lag1",
    "time_step_lag1",
    "u_in_times_u_out",
    "R_div_C",
    "time_step_sq",  # include the new feature
]

X_train = train_split[feature_cols].to_numpy(dtype=np.float32)
y_train = train_split["pressure"].to_numpy(dtype=np.float32)
X_val = val_split[feature_cols].to_numpy(dtype=np.float32)
y_val = val_split["pressure"].to_numpy(dtype=np.float32)

gbr = HistGradientBoostingRegressor(
    max_iter=1500,  # more iterations for better fit
    learning_rate=0.02,  # slightly slower learning
    max_depth=6,
    loss="absolute_error",
    random_state=42,
)
gbr.fit(X_train, y_train)

val_pred_model = gbr.predict(X_val)

rc_means = train.groupby(["R", "C"])["pressure"].mean()
val_rc_key = list(zip(val_split["R"], val_split["C"]))
val_pred_baseline = np.array(
    [rc_means.get(key, rc_means.mean()) for key in val_rc_key], dtype=np.float32
)

candidate_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
best_weight = 0.7  # default
best_mae = float("inf")
for w in candidate_weights:
    blended = w * val_pred_model + (1 - w) * val_pred_baseline
    blended = np.clip(blended, pressure_min, pressure_max)
    mae = mean_absolute_error(y_val, blended)
    if mae < best_mae:
        best_mae = mae
        best_weight = w

blend_model_weight = best_weight
blend_baseline_weight = 1.0 - best_weight

val_pred = (
    blend_model_weight * val_pred_model + blend_baseline_weight * val_pred_baseline
)
val_pred = np.clip(val_pred, pressure_min, pressure_max)

mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (blended & clipped): {mae:.6f}")
print(
    f"Chosen blend weight – model: {blend_model_weight:.2f}, baseline: {blend_baseline_weight:.2f}"
)




## === cell 2
X_full = train[feature_cols].to_numpy(dtype=np.float32)
y_full = train["pressure"].to_numpy(dtype=np.float32)

gbr_full = HistGradientBoostingRegressor(
    max_iter=1500,
    learning_rate=0.02,
    max_depth=6,
    loss="absolute_error",
    random_state=42,
)
gbr_full.fit(X_full, y_full)

test_X = test[feature_cols].to_numpy(dtype=np.float32)
test_pred_model = gbr_full.predict(test_X)

test_rc_key = list(zip(test["R"], test["C"]))
test_pred_baseline = np.array(
    [rc_means.get(key, rc_means.mean()) for key in test_rc_key], dtype=np.float32
)

test_pred = (
    blend_model_weight * test_pred_model + blend_baseline_weight * test_pred_baseline
)
test_pred = np.clip(test_pred, pressure_min, pressure_max)

submission_file = pd.DataFrame({"id": test["id"], "pressure": test_pred})
submission_path = Path("submission.csv")
submission_file.to_csv(submission_path, index=False)
print(f"Submission file '{submission_path}' written with shape:", submission_file.shape)




## === cell 3
submission_file.head()
