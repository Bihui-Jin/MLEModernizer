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
from sklearn.ensemble import HistGradientBoostingRegressor  # faster GBM variant




## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

dtype_train = {
    "R": np.int8,
    "C": np.int8,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
    "breath_id": np.int32,
    "id": np.int32,
}
train_df = pd.read_csv(train_path, dtype=dtype_train)

dtype_test = {
    "R": np.int8,
    "C": np.int8,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "breath_id": np.int32,
    "id": np.int32,
}
test_df = pd.read_csv(test_path, dtype=dtype_test)

train_df["time_step_sq"] = train_df["time_step"] ** 2
train_df["u_in_sq"] = train_df["u_in"] ** 2
train_df["R_C"] = train_df["R"] * train_df["C"]
train_df["u_in_u_out"] = train_df["u_in"] * train_df["u_out"]
train_df["cum_u_in"] = train_df.groupby("breath_id")["u_in"].cumsum()
train_df["cum_time"] = train_df.groupby("breath_id")["time_step"].cumsum()

test_df["time_step_sq"] = test_df["time_step"] ** 2
test_df["u_in_sq"] = test_df["u_in"] ** 2
test_df["R_C"] = test_df["R"] * test_df["C"]
test_df["u_in_u_out"] = test_df["u_in"] * test_df["u_out"]
test_df["cum_u_in"] = test_df.groupby("breath_id")["u_in"].cumsum()
test_df["cum_time"] = test_df.groupby("breath_id")["time_step"].cumsum()

for df in (train_df, test_df):
    df["prev_u_in"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df["prev_u_out"] = df.groupby("breath_id")["u_out"].shift(1).fillna(0)
    df["delta_u_in"] = df["u_in"] - df["prev_u_in"]
    df["delta_u_out"] = df["u_out"] - df["prev_u_out"]
    df["cum_u_in_avg"] = df["cum_u_in"] / (df["cum_time"] + 1e-6)

breath_stats = (
    train_df.groupby("breath_id")
    .agg(
        u_in_mean=("u_in", "mean"),
        u_in_max=("u_in", "max"),
        u_in_min=("u_in", "min"),
        u_out_mean=("u_out", "mean"),
    )
    .reset_index()
)

train_df = train_df.merge(breath_stats, on="breath_id", how="left")
test_df = test_df.merge(breath_stats, on="breath_id", how="left")
global_u_in_mean = train_df["u_in"].mean()
global_u_in_max = train_df["u_in"].max()
global_u_in_min = train_df["u_in"].min()
global_u_out_mean = train_df["u_out"].mean()

for col, val in zip(
    ["u_in_mean", "u_in_max", "u_in_min", "u_out_mean"],
    [global_u_in_mean, global_u_in_max, global_u_in_min, global_u_out_mean],
):
    test_df[col].fillna(val, inplace=True)

feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "time_step_sq",
    "u_in_sq",
    "R_C",
    "u_in_u_out",
    "cum_u_in",
    "cum_time",
    "breath_id",
    "prev_u_in",
    "prev_u_out",
    "delta_u_in",
    "delta_u_out",
    "cum_u_in_avg",
    "u_in_mean",
    "u_in_max",
    "u_in_min",
    "u_out_mean",
]

X = train_df[feature_cols].astype(np.float32)
y = train_df["pressure"].astype(np.float32)

assert not X.isnull().any().any()
assert not y.isnull().any()




## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=train_df["breath_id"]
)

breath_mean = (
    pd.DataFrame({"breath_id": X_train["breath_id"], "pressure": y_train})
    .groupby("breath_id")["pressure"]
    .mean()
)

X_train["breath_mean_pressure"] = X_train["breath_id"].map(breath_mean)
X_val["breath_mean_pressure"] = X_val["breath_id"].map(breath_mean)

global_mean_pressure = y_train.mean()
X_train["breath_mean_pressure"].fillna(global_mean_pressure, inplace=True)
X_val["breath_mean_pressure"].fillna(global_mean_pressure, inplace=True)

feature_cols.append("breath_mean_pressure")

X_train_np = X_train[feature_cols].to_numpy()
X_val_np = X_val[feature_cols].to_numpy()
y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()




## === cell 3
gbr = HistGradientBoostingRegressor(
    max_iter=4000,  # more trees
    learning_rate=0.01,  # lower LR for finer fitting
    max_depth=10,  # deeper trees
    loss="absolute_error",
    random_state=42,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
)

gbr.fit(X_train_np, y_train_np)

val_pred = gbr.predict(X_val_np)
val_mae = mean_absolute_error(y_val_np, val_pred)
print(f"Validation MAE: {val_mae:.5f}")




## === cell 4
test_df["breath_mean_pressure"] = global_mean_pressure

test_features = test_df[feature_cols].astype(np.float32).to_numpy()
test_pred = gbr.predict(test_features)

test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

submission = pd.DataFrame({"id": test_df["id"], "pressure": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
