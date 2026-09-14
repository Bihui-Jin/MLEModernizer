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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import gc
from sklearn.model_selection import train_test_split  # retained for compatibility
from sklearn.ensemble import GradientBoostingRegressor  # stronger residual correction


def find_nearest_vec(preds):
    """Return the training pressure value closest to each prediction."""
    preds = np.asarray(preds)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx_low = np.maximum(idx - 1, 0)
    idx_high = np.minimum(idx, len(sorted_pressures) - 1)
    low = sorted_pressures[idx_low]
    high = sorted_pressures[idx_high]
    choose_low = np.abs(preds - low) <= np.abs(high - preds)
    return np.where(choose_low, low, high)




## === cell 1
dtype_train = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtype_test = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=dtype_train
)
df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv", dtype=dtype_test
)

df_train["u_in_bin"] = df_train["u_in"].astype(int).astype("int16")
df_test["u_in_bin"] = df_test["u_in"].astype(int).astype("int16")

df_train["time_bin"] = (df_train["time_step"] * 100).astype(int).astype("int16")
df_test["time_bin"] = (df_test["time_step"] * 100).astype(int).astype("int16")

cat_cols = ["R", "C", "u_in_bin", "u_out", "time_bin"]
for col in cat_cols:
    df_train[col] = df_train[col].astype("category")
    df_test[col] = df_test[col].astype("category")

detailed_mean = (
    df_train.groupby(["R", "C", "u_in_bin", "u_out", "time_bin"], observed=True)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_detailed"})
    .set_index(["R", "C", "u_in_bin", "u_out", "time_bin"])
)

coarse_mean = (
    df_train.groupby(["R", "C"], observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_coarse"})
    .set_index(["R", "C"])
)

time_coarse_mean = (
    df_train.groupby(["R", "C", "time_bin"], observed=True)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_time"})
    .set_index(["R", "C", "time_bin"])
)


def merge_means(df):
    df = df.join(
        detailed_mean,
        on=["R", "C", "u_in_bin", "u_out", "time_bin"],
        how="left",
    )
    df = df.join(
        time_coarse_mean,
        on=["R", "C", "time_bin"],
        how="left",
    )
    df = df.join(
        coarse_mean,
        on=["R", "C"],
        how="left",
    )
    return df


rng = np.random.RandomState(2021)
perm = rng.permutation(len(df_train))
split_idx = int(0.8 * len(df_train))
train_idx, val_idx = perm[:split_idx], perm[split_idx:]
train_part = df_train.iloc[train_idx]
val_part = df_train.iloc[val_idx]

val_merged = merge_means(val_part)

fallback_val = (
    val_merged["pred_pressure_detailed"]
    .combine_first(val_merged["pred_pressure_time"])
    .combine_first(val_merged["pred_pressure_coarse"])
)
overall_mean = df_train["pressure"].mean()
for col in ["pred_pressure_detailed", "pred_pressure_time", "pred_pressure_coarse"]:
    val_merged[col].fillna(fallback_val, inplace=True)
    val_merged[col].fillna(overall_mean, inplace=True)

X_val = np.column_stack(
    [
        val_merged["pred_pressure_detailed"].values,
        val_merged["pred_pressure_time"].values,
        val_merged["pred_pressure_coarse"].values,
        np.ones_like(val_merged["pressure"].values),  # intercept term
    ]
).astype(np.float32)
y_val = val_merged["pressure"].values.astype(np.float32)

w_opt, *_ = np.linalg.lstsq(
    X_val, y_val, rcond=None
)  # returns [w_det, w_time, w_coarse, intercept]

w_means = np.clip(w_opt[:3], 0, None)
if w_means.sum() == 0:
    w_means = np.array([0.6, 0.3, 0.1])  # fallback weights
else:
    w_means = w_means / w_means.sum()  # renormalise to sum to 1

w_det, w_time, w_coarse = w_means.tolist()
intercept = w_opt[3]  # keep intercept as learned (can be positive or negative)

sorted_pressures = np.sort(df_train["pressure"].unique())

train_merged = merge_means(train_part)

fallback_train = (
    train_merged["pred_pressure_detailed"]
    .combine_first(train_merged["pred_pressure_time"])
    .combine_first(train_merged["pred_pressure_coarse"])
)
for col in ["pred_pressure_detailed", "pred_pressure_time", "pred_pressure_coarse"]:
    train_merged[col].fillna(fallback_train, inplace=True)
    train_merged[col].fillna(overall_mean, inplace=True)

baseline_train_pred = (
    w_det * train_merged["pred_pressure_detailed"]
    + w_time * train_merged["pred_pressure_time"]
    + w_coarse * train_merged["pred_pressure_coarse"]
    + intercept
)

residual_train = train_merged["pressure"] - baseline_train_pred

residual_features = train_merged[
    ["R", "C", "u_in", "u_out", "time_step"]
].values.astype(np.float32, copy=False)

residual_model = GradientBoostingRegressor(
    n_estimators=80,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
residual_model.fit(residual_features, residual_train.astype(np.float32, copy=False))

del df_train, train_part, val_part, train_merged, val_merged
gc.collect()




## === cell 2
df_test = merge_means(df_test)

fallback_test = (
    df_test["pred_pressure_detailed"]
    .combine_first(df_test["pred_pressure_time"])
    .combine_first(df_test["pred_pressure_coarse"])
)

for col in ["pred_pressure_detailed", "pred_pressure_time", "pred_pressure_coarse"]:
    df_test[col].fillna(fallback_test, inplace=True)
    df_test[col].fillna(overall_mean, inplace=True)

df_test["pred_pressure"] = (
    w_det * df_test["pred_pressure_detailed"]
    + w_time * df_test["pred_pressure_time"]
    + w_coarse * df_test["pred_pressure_coarse"]
    + intercept
)

test_features = df_test[["R", "C", "u_in", "u_out", "time_step"]].values.astype(
    np.float32, copy=False
)
correction = residual_model.predict(test_features)
df_test["pred_pressure"] = df_test["pred_pressure"] + correction

df_test["pred_pressure"].fillna(overall_mean, inplace=True)

df_test["pressure"] = df_test["pred_pressure"]

submission = df_test[["id", "pressure"]].copy()
submission.to_csv("submission.csv", index=False)
