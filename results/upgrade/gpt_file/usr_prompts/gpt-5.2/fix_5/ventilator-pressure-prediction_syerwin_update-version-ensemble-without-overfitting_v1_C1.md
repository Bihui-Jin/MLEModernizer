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
import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train_df = pd.read_csv(train_path, dtype=train_dtypes)
test_df = pd.read_csv(test_path, dtype=test_dtypes)
submission = pd.read_csv(sub_path, dtype={"id": "int32", "pressure": "float32"})

print(train_df.shape, test_df.shape, submission.shape)
print(train_df.columns.tolist())




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for Ventilator Pressure Prediction.

    Bugfixes (minimal, score-neutral except restoring intended features):
    - The original code created `gb = df.groupby(...)` before creating the 'one' column,
      so `gb["one"].cumsum()` raised KeyError. Fix by creating columns first and then grouping.
    - Use groupby.transform for EWM mean to ensure index alignment without relying on reset_index.
    """
    df = df.copy()

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["one"] = 1

    gb = df.groupby("breath_id", sort=False)

    df["_area_step"] = df["time_step"] * df["u_in"]
    df["area"] = gb["_area_step"].cumsum()
    df.drop(columns=["_area_step"], inplace=True)

    df["time_step_cumsum"] = gb["time_step"].cumsum()
    df["u_in_cumsum"] = gb["u_in"].cumsum()

    u_in_g = gb["u_in"]
    u_out_g = gb["u_out"]

    df["u_in_lag1"] = u_in_g.shift(1)
    df["u_out_lag1"] = u_out_g.shift(1)
    df["u_in_lag_back1"] = u_in_g.shift(-1)
    df["u_out_lag_back1"] = u_out_g.shift(-1)

    df["u_in_lag2"] = u_in_g.shift(2)
    df["u_out_lag2"] = u_out_g.shift(2)
    df["u_in_lag_back2"] = u_in_g.shift(-2)
    df["u_out_lag_back2"] = u_out_g.shift(-2)

    df["u_in_lag3"] = u_in_g.shift(3)
    df["u_out_lag3"] = u_out_g.shift(3)
    df["u_in_lag_back3"] = u_in_g.shift(-3)
    df["u_out_lag_back3"] = u_out_g.shift(-3)

    df["u_in_lag4"] = u_in_g.shift(4)
    df["u_out_lag4"] = u_out_g.shift(4)
    df["u_in_lag_back4"] = u_in_g.shift(-4)
    df["u_out_lag_back4"] = u_out_g.shift(-4)

    df = df.fillna(0)

    u_in_max = gb["u_in"].transform("max")
    u_in_mean = gb["u_in"].transform("mean")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = u_in_max - df["u_in"]
    df["breath_id__u_in__diffmean"] = u_in_mean - df["u_in"]

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]

    df["count"] = gb["one"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = (
        df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    )

    df["time_step_diff"] = gb["time_step"].diff().fillna(0)

    df["ewm_u_in_mean"] = gb["u_in"].transform(lambda x: x.ewm(halflife=9).mean())

    roll = gb["u_in"].rolling(window=15, min_periods=1)
    df["15_in_sum"] = roll.sum().reset_index(level=0, drop=True)
    df["15_in_min"] = roll.min().reset_index(level=0, drop=True)
    df["15_in_max"] = roll.max().reset_index(level=0, drop=True)
    df["15_in_mean"] = roll.mean().reset_index(level=0, drop=True)

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]
    df = pd.get_dummies(df)

    return df




## === cell 3
train_feat = add_features(train_df)
test_feat = add_features(test_df)

if "pressure" not in train_feat.columns:
    raise ValueError("Expected 'pressure' column in engineered train features.")

y = train_feat["pressure"].astype(np.float32).to_numpy(copy=False)
X_train = train_feat.drop(["pressure"], axis=1)
X_test = test_feat

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print("Aligned shapes:", X_train.shape, X_test.shape)

del train_df, test_df, train_feat, test_feat
gc.collect()



## === cell 4
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32, copy=False)
X_test_s = scaler.transform(X_test).astype(np.float32, copy=False)

del X_train, X_test
gc.collect()

print("Scaled:", X_train_s.shape, X_test_s.shape)



## === cell 5
unique_p = np.unique(y)
unique_p.sort()
P_MIN = float(unique_p[0])
P_MAX = float(unique_p[-1])
diffs = np.diff(unique_p)
pos_diffs = diffs[diffs > 0]
P_STEP = float(pos_diffs.min()) if pos_diffs.size else 0.0

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step: {P_STEP}")
print(f"Unique values: {unique_p.shape[0]}")



## === cell 6
models = [
    (
        "hgb_1",
        HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
    ),
    (
        "hgb_2",
        HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=8,
            learning_rate=0.03,
            max_iter=450,
            random_state=43,
        ),
    ),
    ("ridge", Ridge(alpha=1.0, random_state=42)),
    ("knn", KNeighborsRegressor(n_neighbors=25, weights="distance", p=2)),
]

pred_list = []
for name, model in models:
    print("Fitting:", name)
    model.fit(X_train_s, y)
    p = model.predict(X_test_s).astype(np.float32, copy=False)
    pred_list.append(p)

pred = np.vstack(pred_list)  # shape: (n_models, n_test_rows)
print("Ensemble pred shape:", pred.shape)



## === cell 7
if len(submission) != pred.shape[1]:
    test_ids = pd.read_csv(test_path, usecols=["id"], dtype={"id": "int32"})["id"]
    submission = pd.DataFrame({"id": test_ids})
else:
    submission = submission[["id"]].copy()

submission["pressure"] = np.median(pred, axis=0)

if P_STEP > 0:
    submission["pressure"] = (
        np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
    )

submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX).astype(
    np.float32
)

submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
