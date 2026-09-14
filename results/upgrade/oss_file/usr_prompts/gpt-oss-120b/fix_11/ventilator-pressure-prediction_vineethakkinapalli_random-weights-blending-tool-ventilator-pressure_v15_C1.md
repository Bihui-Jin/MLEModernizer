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
import gc
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor as GradientBoostingRegressor


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 1
dtypes = {
    "R": np.int8,
    "C": np.int8,
    "u_in": np.float32,
    "u_out": np.int8,
    "time_step": np.float32,
    "pressure": np.float32,
    "breath_id": np.int32,
    "id": np.int16,
}

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype=dtypes,
)

df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    dtype=dtypes,
)

for df in (df_train, df_test):
    grp = df.groupby("breath_id")
    df["u_in_cumsum"] = grp["u_in"].cumsum().astype(np.float32)
    df["u_in_diff"] = grp["u_in"].diff().fillna(0).astype(np.float32)
    df["RC"] = (df["R"].astype(np.float32) * df["C"]).astype(np.float32)
    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0).astype(np.float32)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0).astype(np.float32)
    df["time_step_sq"] = (df["time_step"] ** 2).astype(np.float32)

for df in (df_train, df_test):
    df["step_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)
    df["R_div_C"] = (df["R"].astype(np.float32) / df["C"]).astype(np.float32)
    df["u_in_R"] = (df["u_in"] * df["R"]).astype(np.float32)
    df["u_in_C"] = (df["u_in"] * df["C"]).astype(np.float32)
    df["u_out_cumsum"] = df.groupby("breath_id")["u_out"].cumsum().astype(np.float32)

unique_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = len(unique_pressures)


def round_to_nearest(arr: np.ndarray) -> np.ndarray:
    """Vectorized rounding of predictions to the nearest training‑set pressure."""
    idx = np.searchsorted(unique_pressures, arr, side="left")
    idx_clipped = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.maximum(idx_clipped - 1, 0)
    upper_idx = idx_clipped

    lower = unique_pressures[lower_idx]
    upper = unique_pressures[upper_idx]

    use_upper = np.abs(upper - arr) < np.abs(lower - arr)
    return np.where(use_upper, upper, lower)




## === cell 2
train_path = (
    "../input/ventilator-pressure-prediction/train.csv"  # retained for reference
)
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

feature_cols = [
    "R",
    "C",
    "u_in",
    "u_out",
    "time_step",
    "breath_id",
    "u_in_cumsum",
    "u_in_diff",
    "RC",
    "u_in_lag1",
    "u_in_lag2",
    "time_step_sq",
    "step_idx",
    "R_div_C",
    "u_in_R",
    "u_in_C",
    "u_out_cumsum",
]

X = df_train[feature_cols].values
y = df_train["pressure"].values
X_test = df_test[feature_cols].values

set_seed(42)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = GradientBoostingRegressor(
    loss="absolute_error",
    max_iter=1500,  # fewer boosting rounds → faster fit
    learning_rate=0.01,
    max_depth=10,
    random_state=42,
    early_stopping=True,
    max_bins=128,  # fewer bins → quicker histogram building
)

model.fit(X_tr, y_tr)

val_pred = model.predict(X_val)
val_pred_rounded = round_to_nearest(val_pred)
mae = mean_absolute_error(y_val, val_pred_rounded)
print(f"Validation MAE (rounded predictions): {mae:.5f}")

optimal_iters = model.n_iter_  # iterations actually performed before early stop

final_model = GradientBoostingRegressor(
    loss="absolute_error",
    max_iter=optimal_iters,
    learning_rate=0.01,
    max_depth=10,
    random_state=42,
    early_stopping=False,
    max_bins=128,
)

final_model.fit(X, y)

test_pred = final_model.predict(X_test)
test_pred_rounded = round_to_nearest(test_pred)

submission = pd.read_csv(sample_sub_path)
submission["pressure"] = test_pred_rounded.astype(float)
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## === cell 3
del df_train, df_test, X, y, X_test, model, final_model, test_pred, test_pred_rounded
gc.collect()
