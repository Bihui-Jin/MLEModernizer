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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

dtype_train = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.float32,
    "C": np.float32,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
dtype_test = {
    "id": np.int32,
    "breath_id": np.int32,
    "R": np.float32,
    "C": np.float32,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
}
train = pd.read_csv(train_path, dtype=dtype_train)
test = pd.read_csv(test_path, dtype=dtype_test)

g_train = train.groupby("breath_id", observed=True)
train["dcount"] = g_train["id"].cumcount()
train["uo"] = 80 - g_train["u_out"].transform("sum")
train["cum_u_in"] = g_train["u_in"].cumsum()
train["cum_u_out"] = g_train["u_out"].cumsum()
train["u_in_diff"] = g_train["u_in"].diff().fillna(0)
train["prev_u_in"] = g_train["u_in"].shift(1).fillna(0)

g_test = test.groupby("breath_id", observed=True)
test["dcount"] = g_test["id"].cumcount()
test["uo"] = 80 - g_test["u_out"].transform("sum")
test["cum_u_in"] = g_test["u_in"].cumsum()
test["cum_u_out"] = g_test["u_out"].cumsum()
test["u_in_diff"] = g_test["u_in"].diff().fillna(0)
test["prev_u_in"] = g_test["u_in"].shift(1).fillna(0)

agg_train = (
    g_train["u_in"]
    .agg(["mean", "max", "std"])
    .rename(columns={"mean": "mean_u_in", "max": "max_u_in", "std": "std_u_in"})
)
agg_train["mean_u_out"] = g_train["u_out"].mean()
agg_train["max_u_out"] = g_train["u_out"].max()
agg_train["breath_len"] = g_train["id"].count()
agg_train = agg_train.reset_index()

agg_test = (
    g_test["u_in"]
    .agg(["mean", "max", "std"])
    .rename(columns={"mean": "mean_u_in", "max": "max_u_in", "std": "std_u_in"})
)
agg_test["mean_u_out"] = g_test["u_out"].mean()
agg_test["max_u_out"] = g_test["u_out"].max()
agg_test["breath_len"] = g_test["id"].count()
agg_test = agg_test.reset_index()

train = train.merge(agg_train, on="breath_id", how="left")
test = test.merge(agg_test, on="breath_id", how="left")
train["std_u_in"] = train["std_u_in"].fillna(0)
test["std_u_in"] = test["std_u_in"].fillna(0)

train["R_C"] = train["R"] * train["C"]
test["R_C"] = test["R"] * test["C"]
train["u_in_R"] = train["u_in"] * train["R"]
test["u_in_R"] = test["u_in"] * test["R"]
train["u_in_C"] = train["u_in"] * train["C"]
test["u_in_C"] = test["u_in"] * test["C"]
train["u_out_R"] = train["u_out"] * train["R"]
test["u_out_R"] = test["u_out"] * test["R"]
train["u_out_C"] = train["u_out"] * train["C"]
test["u_out_C"] = test["u_out"] * test["C"]
train["u_in_time"] = train["u_in"] * train["time_step"]
test["u_in_time"] = test["u_in"] * test["time_step"]
train["u_out_time"] = train["u_out"] * train["time_step"]
test["u_out_time"] = test["u_out"] * test["time_step"]

print("train shape:", train.shape, "test shape:", test.shape)




## === cell 1
features = [
    "u_in",
    "u_out",
    "R",
    "C",
    "time_step",
    "dcount",
    "uo",
    "cum_u_in",
    "cum_u_out",
    "u_in_diff",
    "prev_u_in",
    "mean_u_in",
    "max_u_in",
    "std_u_in",
    "mean_u_out",
    "max_u_out",
    "breath_len",
    "R_C",
    "u_in_R",
    "u_in_C",
    "u_out_R",
    "u_out_C",
    "u_in_time",
    "u_out_time",
]

X = train[features].astype(np.float32).to_numpy()
y = train["pressure"].astype(np.float32).to_numpy()

model = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.02,
    max_iter=2500,
    max_depth=15,
    random_state=42,
)

model.fit(X, y)

val_idx = np.random.RandomState(42).choice(
    len(y), size=int(0.1 * len(y)), replace=False
)
val_pred = model.predict(X[val_idx])
val_mae = mean_absolute_error(y[val_idx], val_pred)
print(f"Validation MAE: {val_mae:.5f}")




## === cell 2
X_test = test[features].astype(np.float32).to_numpy()
test_pred = model.predict(X_test)
test["pred"] = test_pred
print("Number of predictions:", test["pred"].notna().sum())




## === cell 3
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"
sub = pd.read_csv(sample_path)

sub = sub.merge(test[["id", "pred"]], on="id", how="left")
sub.loc[sub["pred"].notna(), "pressure"] = sub.loc[sub["pred"].notna(), "pred"]
sub = sub.drop(columns=["pred"])
sub = sub[["id", "pressure"]]

output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}, shape: {sub.shape}")
