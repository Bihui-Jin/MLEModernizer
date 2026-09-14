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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass  # fall back to regular scikit‑learn if sklearnex is unavailable

import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")




## === cell 1
def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]

    grouped = df.groupby("breath_id", sort=False, observed=True)

    df["area"] = grouped["area"].cumsum()
    df["time_step_cumsum"] = grouped["time_step"].cumsum()
    df["u_in_cumsum"] = grouped["u_in"].cumsum()

    for i in range(1, 5):
        df[f"u_in_lag{i}"] = grouped["u_in"].shift(i).fillna(0)
        df[f"u_out_lag{i}"] = grouped["u_out"].shift(i).fillna(0)
        df[f"u_in_lag_back{i}"] = grouped["u_in"].shift(-i).fillna(0)
        df[f"u_out_lag_back{i}"] = grouped["u_out"].shift(-i).fillna(0)

    u_in_max = grouped["u_in"].transform("max")
    u_in_mean = grouped["u_in"].transform("mean")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = u_in_max - df["u_in"]
    df["breath_id__u_in__diffmean"] = u_in_mean - df["u_in"]

    for i in range(1, 5):
        df[f"u_in_diff{i}"] = df["u_in"] - df[f"u_in_lag{i}"]
        df[f"u_out_diff{i}"] = df["u_out"] - df[f"u_out_lag{i}"]

    df["count"] = grouped.cumcount() + 1
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0).astype(np.int32)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0).astype(np.int32)
    df["breath_id_lagsame"] = (df["breath_id_lag"] == df["breath_id"]).astype(np.int8)
    df["breath_id_lag2same"] = (df["breath_id_lag2"] == df["breath_id"]).astype(np.int8)
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = (
        df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    )

    df["time_step_diff"] = grouped["time_step"].diff().fillna(0)

    roll = grouped["u_in"].rolling(15, min_periods=1)
    df["15_in_sum"] = roll.sum().reset_index(level=0, drop=True)
    df["15_in_min"] = roll.min().reset_index(level=0, drop=True)
    df["15_in_max"] = roll.max().reset_index(level=0, drop=True)
    df["15_in_mean"] = roll.mean().reset_index(level=0, drop=True)

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]

    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}
    df["R_code"] = df["R"].map(r_map).astype(np.int8)
    df["C_code"] = df["C"].map(c_map).astype(np.int8)
    df["R__C_code"] = (df["R_code"] * 10 + df["C_code"]).astype(np.int8)

    df.drop(columns=["R", "C"], inplace=True)

    num_cols = df.select_dtypes(
        include=["int64", "float64", "int8", "int16", "int32"]
    ).columns
    df[num_cols] = df[num_cols].astype(np.float32)

    return df


dtype_map = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_df = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=dtype_map
)
test_df = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv", dtype=dtype_map
)

train_df["__is_train"] = 1
test_df["__is_train"] = 0
full = pd.concat([train_df, test_df], ignore_index=True)

print("Adding features to combined dataset...")
full = add_features(full)

train = full[full["__is_train"] == 1].drop(columns=["__is_train"])
test = full[full["__is_train"] == 0].drop(columns=["__is_train"])

test = test.reindex(columns=train.columns, fill_value=0)
if "pressure" in test.columns:
    test = test.drop(columns=["pressure"])

del train_df, test_df, full
gc.collect()




## === cell 2
targets = train["pressure"].values.astype(np.float32)

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",  # retained for compatibility; column may be absent after optimization
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]
train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(
    columns=[c for c in drop_cols if c in test.columns]
)  # same list works for test

print(
    f"train shape: {train.shape}, test shape: {test.shape}, targets shape: {targets.shape}"
)




## === cell 3
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train).astype(np.float32)
test_scaled = scaler.transform(test).astype(np.float32)

print(f"Scaled shapes – train: {train_scaled.shape}, test: {test_scaled.shape}")




## === cell 4
P_MIN = targets.min()
P_MAX = targets.max()
print(f"Pressure range – min: {P_MIN:.4f}, max: {P_MAX:.4f}")




## === cell 5
gbr = GradientBoostingRegressor(random_state=42, n_estimators=50)
gbr.fit(train_scaled, targets)

test_pred = gbr.predict(test_scaled)
test_pred = np.clip(test_pred, P_MIN, P_MAX)

submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = test_pred
submission.to_csv("submission.csv", index=False)

print("Submission written to submission.csv (first rows):")
print(submission.head())
