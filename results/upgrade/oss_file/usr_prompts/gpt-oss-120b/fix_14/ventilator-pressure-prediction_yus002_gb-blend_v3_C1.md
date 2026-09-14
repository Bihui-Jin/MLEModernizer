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

3.9

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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

possible_roots = [
    os.path.join(".", "data"),
    os.path.join(".", "input"),
    os.path.join(".", "kaggle", "input"),
    os.path.join("..", "input"),
]

train_path = test_path = None
for root in possible_roots:
    cand_train = os.path.join(root, "train.csv")
    cand_test = os.path.join(root, "test.csv")
    if os.path.exists(cand_train) and os.path.exists(cand_test):
        train_path, test_path = cand_train, cand_test
        break
if train_path is None or test_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv and test.csv in expected directories."
    )

BASE_FEATURES = ["R", "C", "time_step", "u_in", "u_out", "breath_id"]
dtype_map = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
    "id": "int32",
    "breath_id": "int32",
}
train = pd.read_csv(train_path, usecols=BASE_FEATURES + ["pressure"], dtype=dtype_map)
test = pd.read_csv(test_path, usecols=["id"] + BASE_FEATURES, dtype=dtype_map)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features using fully vectorized pandas operations."""
    grp = df.groupby("breath_id", sort=False)

    df["cum_u_in"] = grp["u_in"].cumsum()
    df["cum_u_out"] = grp["u_out"].cumsum()

    df["R_C"] = df["R"] * df["C"]
    df["R_u_in"] = df["R"] * df["u_in"]
    df["C_u_in"] = df["C"] * df["u_in"]
    df["R_u_out"] = df["R"] * df["u_out"]

    df["time_step_sq"] = df["time_step"] ** 2
    df["u_in_time"] = df["u_in"] * df["time_step"]
    df["u_out_time"] = df["u_out"] * df["time_step"]

    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0)
    df["u_out_lag1"] = grp["u_out"].shift(1).fillna(0)

    df["u_in_roll3"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_roll5"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_in_roll5_std"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["u_in_diff"] = grp["u_in"].diff().fillna(0)
    df["u_out_diff"] = grp["u_out"].diff().fillna(0)
    df["u_in_cumdiff"] = grp["u_in"].cumsum().diff().fillna(0)

    float_cols = [
        "cum_u_in",
        "cum_u_out",
        "R_C",
        "R_u_in",
        "C_u_in",
        "R_u_out",
        "time_step_sq",
        "u_in_time",
        "u_out_time",
        "u_in_lag1",
        "u_out_lag1",
        "u_in_roll3",
        "u_in_roll5",
        "u_in_roll5_std",
        "u_in_diff",
        "u_out_diff",
        "u_in_cumdiff",
    ]
    df[float_cols] = df[float_cols].astype("float32")
    return df


train = add_engineered_features(train)
test = add_engineered_features(test)

FEATURES = BASE_FEATURES + [
    "cum_u_in",
    "cum_u_out",
    "R_C",
    "R_u_in",
    "C_u_in",
    "R_u_out",
    "time_step_sq",
    "u_in_time",
    "u_out_time",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_roll3",
    "u_in_roll5",
    "u_in_roll5_std",
    "u_in_diff",
    "u_out_diff",
    "u_in_cumdiff",
]

X = train[FEATURES]
y = train["pressure"]

unique_breaths = train["breath_id"].unique()
train_breaths, val_breaths = train_test_split(
    unique_breaths, test_size=0.2, random_state=42, shuffle=True
)
train_idx = train["breath_id"].isin(train_breaths)
val_idx = train["breath_id"].isin(val_breaths)

X_train, X_val = X.loc[train_idx], X.loc[val_idx]
y_train, y_val = y.loc[train_idx], y.loc[val_idx]

X_train_np = X_train.values
X_val_np = X_val.values
y_train_np = y_train.values
y_val_np = y_val.values

model = HistGradientBoostingRegressor(
    random_state=42,
    max_iter=3000,
    learning_rate=0.01,
    max_depth=10,
    loss="absolute_error",
    n_iter_no_change=50,
    tol=1e-4,
    early_stopping=True,
)

model.fit(X_train_np, y_train_np)

val_pred = model.predict(X_val_np)
mae = mean_absolute_error(y_val_np, val_pred)
print(f"Validation MAE (quick check): {mae:.5f}")




## === cell 1
test_pred = model.predict(test[FEATURES].values)

submission = pd.DataFrame({"id": test["id"], "pressure": test_pred})
submission = submission.sort_values("id").reset_index(drop=True)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
