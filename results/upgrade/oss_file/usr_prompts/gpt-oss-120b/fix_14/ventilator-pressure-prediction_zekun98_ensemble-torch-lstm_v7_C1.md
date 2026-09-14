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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import gc




## === cell 1
class config:
    paths = {
        "train": "../input/ventilator-pressure-prediction/train.csv",
        "test": "../input/ventilator-pressure-prediction/test.csv",
    }
    model_params = {
        "max_iter": 400,  # unchanged
        "learning_rate": 0.05,
        "max_depth": None,
        "l2_regularization": 0.0,
        "random_state": 42,
        "max_bins": 64,
    }
    post_processing = {
        "max_pressure": 64.82099173863948,
        "min_pressure": -1.8957442945646408,
    }




## === cell 2
train_usecols = ["R", "C", "time_step", "u_in", "u_out", "pressure", "breath_id"]
dtype_map = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
    "breath_id": "int32",
}
train_df = pd.read_csv(
    config.paths["train"],
    usecols=train_usecols,
    dtype=dtype_map,
)

train_df["prev_pressure"] = (
    train_df.groupby("breath_id")["pressure"].shift(1).fillna(0).astype(np.float32)
)

feature_cols = ["R", "C", "time_step", "u_in", "u_out", "prev_pressure"]
X = train_df[feature_cols].values.astype(np.float32)
y = train_df["pressure"].values.astype(np.float32)

del train_df
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=None
)

hgb = HistGradientBoostingRegressor(
    max_iter=config.model_params["max_iter"],
    learning_rate=config.model_params["learning_rate"],
    max_depth=config.model_params["max_depth"],
    l2_regularization=config.model_params["l2_regularization"],
    random_state=config.model_params["random_state"],
    max_bins=config.model_params["max_bins"],
)

hgb.fit(X_train, y_train)

val_pred = hgb.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (quick subset): {val_mae:.6f}")



## === cell 3
test_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id", "id"]
test_dtype_map = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "breath_id": "int32",
    "id": "int32",
}
test_df = pd.read_csv(
    config.paths["test"],
    usecols=test_usecols,
    dtype=test_dtype_map,
)

test_pred = np.empty(len(test_df), dtype=np.float32)

test_df["orig_idx"] = np.arange(len(test_df))

sorted_df = test_df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

R_arr = sorted_df["R"].astype(np.float32).values
C_arr = sorted_df["C"].astype(np.float32).values
t_arr = sorted_df["time_step"].astype(np.float32).values
u_in_arr = sorted_df["u_in"].astype(np.float32).values
u_out_arr = sorted_df["u_out"].astype(np.float32).values
breath_arr = sorted_df["breath_id"].values
orig_idx_arr = sorted_df["orig_idx"].values

feat = np.empty((1, 6), dtype=np.float32)

prev_pressure = 0.0
prev_breath = breath_arr[0] if len(breath_arr) > 0 else None

for i in range(len(sorted_df)):
    if breath_arr[i] != prev_breath:
        prev_pressure = 0.0
        prev_breath = breath_arr[i]

    feat[0, 0] = R_arr[i]
    feat[0, 1] = C_arr[i]
    feat[0, 2] = t_arr[i]
    feat[0, 3] = u_in_arr[i]
    feat[0, 4] = u_out_arr[i]
    feat[0, 5] = prev_pressure

    pred = hgb.predict(feat)[0]
    test_pred[orig_idx_arr[i]] = pred

    prev_pressure = pred

pp = config.post_processing
clipped = np.clip(test_pred, pp["min_pressure"], pp["max_pressure"])



## === cell 4
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "pressure": clipped,
    }
)
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written with', len(submission), "rows.")
