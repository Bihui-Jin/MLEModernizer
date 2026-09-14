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
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

data_dir = Path("../input/ventilator-pressure-prediction")
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"
sample_sub_path = data_dir / "sample_submission.csv"

train_dtypes = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "breath_id": "int32",
    "pressure": "float32",
    "id": "int16",
}
test_dtypes = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "breath_id": "int32",
    "id": "int16",
}

train_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id", "pressure", "id"]
test_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id", "id"]

train_df = pd.read_csv(
    train_path,
    dtype=train_dtypes,
    usecols=train_usecols,
    low_memory=False,
)
test_df = pd.read_csv(
    test_path,
    dtype=test_dtypes,
    usecols=test_usecols,
    low_memory=False,
)
sub = pd.read_csv(sample_sub_path)

R_tr = train_df["R"].to_numpy(dtype=np.float32)
C_tr = train_df["C"].to_numpy(dtype=np.float32)
time_tr = train_df["time_step"].to_numpy(dtype=np.float32)
u_in_tr = train_df["u_in"].to_numpy(dtype=np.float32)
u_out_tr = train_df["u_out"].to_numpy(dtype=np.float32)
breath_tr = train_df["breath_id"].to_numpy(dtype=np.int32)

R_te = test_df["R"].to_numpy(dtype=np.float32)
C_te = test_df["C"].to_numpy(dtype=np.float32)
time_te = test_df["time_step"].to_numpy(dtype=np.float32)
u_in_te = test_df["u_in"].to_numpy(dtype=np.float32)
u_out_te = test_df["u_out"].to_numpy(dtype=np.float32)
breath_te = test_df["breath_id"].to_numpy(dtype=np.int32)


def compute_lag(arr, breath):
    lag = np.empty_like(arr, dtype=np.float32)
    lag[0] = 0.0
    lag[1:] = arr[:-1]
    mask = breath[1:] != breath[:-1]
    lag[1:][mask] = 0.0
    return lag


def compute_cum(arr, breath):
    csum = np.cumsum(arr, dtype=np.float32)
    offset = np.empty_like(csum, dtype=np.float32)
    offset[0] = 0.0
    offset[1:] = csum[:-1]
    mask = breath[1:] != breath[:-1]
    offset[1:][mask] = 0.0
    return csum - offset


BASE_FEATURES = ["R", "C", "time_step", "u_in", "u_out"]


def build_features(R, C, time, u_in, u_out, breath):
    feats = [
        R * C,  # R_C
        R * u_in,  # R_u_in
        C * u_in,  # C_u_in
        R * u_out,  # R_u_out
        C * u_out,  # C_u_out
        time * u_in,  # time_step_u_in
        u_in**2,  # u_in_sq
        time**2,  # time_step_sq
        compute_lag(u_in, breath),  # u_in_lag1
        compute_lag(u_out.astype(np.float32), breath),  # u_out_lag1
        breath.astype(np.float32),  # breath_id_feat
        R / C,  # R_div_C
        C / R,  # C_div_R
        compute_cum(u_in, breath),  # cum_u_in
        compute_cum(time, breath),  # cum_time_step
    ]
    return np.column_stack(feats)


engineered_tr = build_features(R_tr, C_tr, time_tr, u_in_tr, u_out_tr, breath_tr)
engineered_te = build_features(R_te, C_te, time_te, u_in_te, u_out_te, breath_te)

X = np.column_stack([R_tr, C_tr, time_tr, u_in_tr, u_out_tr, engineered_tr]).astype(
    np.float32
)
y = train_df["pressure"].to_numpy(dtype=np.float32)

X_test = np.column_stack(
    [R_te, C_te, time_te, u_in_te, u_out_te, engineered_te]
).astype(np.float32)

del train_df, test_df, R_tr, C_tr, time_tr, u_in_tr, u_out_tr, breath_tr
del R_te, C_te, time_te, u_in_te, u_out_te, breath_te, engineered_tr, engineered_te
gc.collect()




## === cell 1
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_iter=3000,  # fewer trees than the original 6000
    learning_rate=0.06,  # larger step compensates for fewer iterations
    max_depth=15,
    max_bins=64,
    random_state=42,
    early_stopping=False,  # keep original early_stopping behaviour
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (approx.): {val_mae:.5f}")




## === cell 2
test_pred = model.predict(X_test)

sub["pressure"] = test_pred
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
