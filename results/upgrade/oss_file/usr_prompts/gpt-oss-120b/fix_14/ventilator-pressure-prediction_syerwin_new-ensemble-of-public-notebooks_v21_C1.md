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
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def resolve_data_path(relative_path: str) -> str:
    """
    Return an absolute path to a data file.
    Checks the standard Kaggle input directory first,
    then falls back to a relative path (../input/...).
    """
    kaggle_root = Path("/kaggle/input")
    possible_paths = [
        kaggle_root / relative_path,
        Path(relative_path).resolve(),
    ]
    for p in possible_paths:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Could not locate data file: {relative_path}")




## === cell 1
train_path = resolve_data_path("ventilator-pressure-prediction/train.csv")
test_path = resolve_data_path("ventilator-pressure-prediction/test.csv")
sample_sub_path = resolve_data_path(
    "ventilator-pressure-prediction/sample_submission.csv"
)

train_cols = ["R", "C", "time_step", "u_in", "u_out", "pressure", "id", "breath_id"]
test_cols = ["R", "C", "time_step", "u_in", "u_out", "id", "breath_id"]

dtype_map = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
    "id": "int16",
    "breath_id": "int32",
}

train_df = pd.read_csv(train_path, usecols=train_cols, dtype=dtype_map)
test_df = pd.read_csv(test_path, usecols=test_cols, dtype=dtype_map)
sample_sub = pd.read_csv(sample_sub_path)




## === cell 2
base_features = ["R", "C", "time_step", "u_in", "u_out", "breath_id"]
engineered_features = ["cum_u_in", "cum_u_out", "time_to_end"]
all_features = base_features + engineered_features

for df in (train_df, test_df):
    grp = df.groupby("breath_id", sort=False)
    df["cum_u_in"] = grp["u_in"].cumsum()
    df["cum_u_out"] = grp["u_out"].cumsum()
    max_time = grp["time_step"].transform("max")
    df["time_to_end"] = max_time - df["time_step"]




## === cell 3
_n_features = len(all_features)
_tri_i, _tri_j = np.triu_indices(_n_features, k=0)  # include diagonal for squared terms
_n_inter = len(_tri_i)  # number of interaction columns (45 for 9 original features)


def poly2_features(X: np.ndarray) -> np.ndarray:
    """
    Efficiently create second‑order polynomial features.
    Writes interactions directly into a pre‑allocated array to avoid
    extra temporary storage.
    """
    n_rows = X.shape[0]
    out = np.empty((n_rows, _n_features + _n_inter), dtype=np.float32)
    out[:, :_n_features] = X  # original features
    out[:, _n_features:] = X[:, _tri_i] * X[:, _tri_j]
    return out


X_train_base = train_df[all_features].astype("float32").to_numpy(copy=False)
y_train = train_df["pressure"].astype("float32").to_numpy(copy=False)
X_test_base = test_df[all_features].astype("float32").to_numpy(copy=False)

X_train_poly = poly2_features(X_train_base)  # (N, 54)
X_test_poly = poly2_features(X_test_base)  # (M, 54)

del X_train_base, X_test_base, train_df, test_df
import gc

gc.collect()

scaler = StandardScaler(copy=False)
X_train = scaler.fit_transform(X_train_poly)
X_test = scaler.transform(X_test_poly)

del X_train_poly, X_test_poly
gc.collect()

gbr = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_iter=1500,
    learning_rate=0.01,
    max_depth=6,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20,
)
gbr.fit(X_train, y_train)




## === cell 4
test_pred = gbr.predict(X_test)

pressure_min, pressure_max = y_train.min(), y_train.max()
test_pred = np.clip(test_pred, pressure_min, pressure_max)

submission = pd.DataFrame(
    {"id": pd.read_csv(test_path, usecols=["id"])["id"], "pressure": test_pred}
)




## === cell 5
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission.shape}")
