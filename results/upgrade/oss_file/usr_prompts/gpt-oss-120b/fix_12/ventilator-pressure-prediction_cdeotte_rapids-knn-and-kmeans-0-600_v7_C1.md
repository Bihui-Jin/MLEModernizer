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

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
cuml-cu12==25.2.1
cupy-cuda12x==13.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
libcudf-cu12==25.2.2
libcuml-cu12==25.2.1
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
pylibcudf-cu12==25.2.2
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
import pandas as pd, numpy as np
import cudf, cupy
import matplotlib.pyplot as plt

print("RAPIDS version", cudf.__version__)



## === cell 1
train = cudf.read_csv("../input/ventilator-pressure-prediction/train.csv")

breath_grp = train.groupby("breath_id")
exhale_series = 80 - breath_grp["u_out"].sum()  # exhale per breath
time_len_series = breath_grp["time_step"].max()  # max time per breath

train["exhale"] = train["breath_id"].map(exhale_series)
train["time_length"] = train["breath_id"].map(time_len_series)

train["R_C"] = train["R"] * train["C"]
train["u_in_sq"] = train["u_in"] * train["u_in"]
train["u_in_R"] = train["u_in"] * train["R"]
train["u_in_C"] = train["u_in"] * train["C"]
train["u_out_R"] = train["u_out"] * train["R"]
train["u_out_C"] = train["u_out"] * train["C"]
train["time_u_in"] = train["time_step"] * train["u_in"]

train = train.sort_values(["breath_id", "time_step"])

train["cum_u_in"] = train.groupby("breath_id")["u_in"].transform("cumsum")
train["diff_u_in"] = train.groupby("breath_id")["u_in"].diff().fillna(0)

train["cum_u_out"] = train.groupby("breath_id")["u_out"].transform("cumsum")
train["diff_u_out"] = train.groupby("breath_id")["u_out"].diff().fillna(0)

train["time_step_sq"] = train["time_step"] * train["time_step"]
train["u_in_div_R"] = train["u_in"] / (train["R"] + 1e-5)
train["time_cum_u_in"] = train["time_step"] * train["cum_u_in"]

print("Train shape:", train.shape)
train.head()



## === cell 2
pass



## === cell 3
pass



## === cell 4
test = cudf.read_csv("../input/ventilator-pressure-prediction/test.csv")

breath_grp_test = test.groupby("breath_id")
exhale_test_series = 80 - breath_grp_test["u_out"].sum()
time_len_test_series = breath_grp_test["time_step"].max()

test["exhale"] = test["breath_id"].map(exhale_test_series)
test["time_length"] = test["breath_id"].map(time_len_test_series)

test["R_C"] = test["R"] * test["C"]
test["u_in_sq"] = test["u_in"] * test["u_in"]
test["u_in_R"] = test["u_in"] * test["R"]
test["u_in_C"] = test["u_in"] * test["C"]
test["u_out_R"] = test["u_out"] * test["R"]
test["u_out_C"] = test["u_out"] * test["C"]
test["time_u_in"] = test["time_step"] * test["u_in"]

test = test.sort_values(["breath_id", "time_step"])

test["cum_u_in"] = test.groupby("breath_id")["u_in"].transform("cumsum")
test["diff_u_in"] = test.groupby("breath_id")["u_in"].diff().fillna(0)

test["cum_u_out"] = test.groupby("breath_id")["u_out"].transform("cumsum")
test["diff_u_out"] = test.groupby("breath_id")["u_out"].diff().fillna(0)

test["time_step_sq"] = test["time_step"] * test["time_step"]
test["u_in_div_R"] = test["u_in"] / (test["R"] + 1e-5)
test["time_cum_u_in"] = test["time_step"] * test["cum_u_in"]

print("Test shape:", test.shape)
test.head()



## === cell 5
from cuml.ensemble import RandomForestRegressor

train["u_in_roll_mean_3"] = (
    train.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
train["u_in_roll_std_3"] = (
    train.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .std()
    .fillna(0)
    .reset_index(level=0, drop=True)
)
train["u_out_roll_mean_3"] = (
    train.groupby("breath_id")["u_out"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
train["u_out_roll_std_3"] = (
    train.groupby("breath_id")["u_out"]
    .rolling(window=3, min_periods=1)
    .std()
    .fillna(0)
    .reset_index(level=0, drop=True)
)

test["u_in_roll_mean_3"] = (
    test.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
test["u_in_roll_std_3"] = (
    test.groupby("breath_id")["u_in"]
    .rolling(window=3, min_periods=1)
    .std()
    .fillna(0)
    .reset_index(level=0, drop=True)
)
test["u_out_roll_mean_3"] = (
    test.groupby("breath_id")["u_out"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
test["u_out_roll_std_3"] = (
    test.groupby("breath_id")["u_out"]
    .rolling(window=3, min_periods=1)
    .std()
    .fillna(0)
    .reset_index(level=0, drop=True)
)

feature_cols = [
    "u_in",
    "u_out",
    "R",
    "C",
    "time_step",
    "exhale",
    "time_length",
    "R_C",
    "u_in_sq",
    "u_in_R",
    "u_in_C",
    "u_out_R",
    "u_out_C",
    "time_u_in",
    "cum_u_in",
    "diff_u_in",
    "cum_u_out",
    "diff_u_out",
    "time_step_sq",
    "u_in_div_R",
    "time_cum_u_in",
    "u_in_roll_mean_3",
    "u_in_roll_std_3",
    "u_out_roll_mean_3",
    "u_out_roll_std_3",
]

train[feature_cols] = train[feature_cols].fillna(0)
test[feature_cols] = test[feature_cols].fillna(0)

X_train = train[feature_cols].astype("float32")
y_train = train["pressure"].astype("float32")

del train

model = RandomForestRegressor(
    n_estimators=2500,
    max_depth=45,
    max_features="sqrt",
    random_state=42,
    n_bins=256,
)

model.fit(X_train, y_train)

X_test = test[feature_cols].astype("float32")
pred_test = model.predict(X_test)

pred_test_np = cupy.asnumpy(pred_test)
min_pressure = float(y_train.min())
max_pressure = float(y_train.max())
pred_test_np = np.clip(pred_test_np, min_pressure, max_pressure)



## === cell 6
print(
    "Prediction stats – mean:",
    pred_test_np.mean(),
    "min:",
    pred_test_np.min(),
    "max:",
    pred_test_np.max(),
)



## === cell 7
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
sub["pressure"] = pred_test_np
sub = sub.sort_values("id").reset_index(drop=True)
sub.to_csv("submission_rapids_knn.csv", index=False)
print("Submission shape:", sub.shape)
sub.head()
