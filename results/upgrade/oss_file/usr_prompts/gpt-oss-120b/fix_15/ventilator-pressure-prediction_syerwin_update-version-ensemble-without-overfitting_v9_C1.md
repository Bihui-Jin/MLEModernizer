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
import pandas as pd
import numpy as np
import gc

from sklearnex import patch_sklearn

patch_sklearn()  # patches sklearn estimators in‑place

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)



## === cell 1
dtype_dict = {
    "id": np.uint16,
    "breath_id": np.int32,
    "R": np.int8,
    "C": np.int8,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.uint8,
    "pressure": np.float32,
}
train_df = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype=dtype_dict,
    low_memory=False,
)
test_df = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    dtype={k: v for k, v in dtype_dict.items() if k != "pressure"},
    low_memory=False,
)




## === cell 2
def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]

    grp = df.groupby("breath_id", sort=False)

    df["area"] = grp["area"].cumsum()
    df["time_step_cumsum"] = grp["time_step"].cumsum()
    df["u_in_cumsum"] = grp["u_in"].cumsum()

    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = grp["u_in"].shift(lag)
        df[f"u_out_lag{lag}"] = grp["u_out"].shift(lag)
        df[f"u_in_lag_back{lag}"] = grp["u_in"].shift(-lag)
        df[f"u_out_lag_back{lag}"] = grp["u_out"].shift(-lag)

    df["breath_id__u_in__max"] = grp["u_in"].transform("max")
    df["breath_id__u_in__mean"] = grp["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = df["breath_id__u_in__max"] - df["u_in"]
    df["breath_id__u_in__diffmean"] = df["breath_id__u_in__mean"] - df["u_in"]

    for lag in range(1, 5):
        df[f"u_in_diff{lag}"] = df["u_in"] - df[f"u_in_lag{lag}"]
        df[f"u_out_diff{lag}"] = df["u_out"] - df[f"u_out_lag{lag}"]

    df["one"] = 1
    df["count"] = grp["one"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = (df["breath_id_lag"] == df["breath_id"]).astype(np.int8)
    df["breath_id_lag2same"] = (df["breath_id_lag2"] == df["breath_id"]).astype(np.int8)

    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = (
        df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    )

    df["time_step_diff"] = grp["time_step"].diff().fillna(0)

    df["ewm_u_in_mean"] = (
        grp["u_in"].ewm(halflife=9).mean().reset_index(level=0, drop=True)
    )

    roll = (
        grp["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(["sum", "min", "max", "mean"])
        .reset_index(level=0, drop=True)
        .rename(
            columns={
                "sum": "15_in_sum",
                "min": "15_in_min",
                "max": "15_in_max",
                "mean": "15_in_mean",
            }
        )
    )
    df = pd.concat([df, roll], axis=1)

    df["R"] = df["R"].astype("category")
    df["C"] = df["C"].astype("category")
    df["R__C"] = df["R"].cat.codes.astype(str) + "__" + df["C"].cat.codes.astype(str)
    df = pd.get_dummies(df, dtype=np.uint8)

    df = df.fillna(0)

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].astype(np.float32)

    return df




## === cell 3
print("Engineering train features...")
train = add_features(train_df)
print("Engineering test features...")
test = add_features(test_df)

train_feature_cols = [c for c in train.columns if c != "pressure"]
test = test.reindex(columns=train_feature_cols, fill_value=0)

y = train["pressure"].values.astype(np.float32)
X = train.drop(
    [
        "pressure",
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
)

test_X = test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    errors="ignore",
)

print(f"Feature matrix shapes – train: {X.shape}, test: {test_X.shape}")

X_array = X.values.astype(np.float32)
test_X_array = test_X.values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X_array, y, test_size=0.2, random_state=42
)

model_params = dict(
    max_iter=3000,
    learning_rate=0.01,
    max_depth=20,
    random_state=42,
    loss="absolute_error",
    early_stopping=True,
)

val_model = HistGradientBoostingRegressor(**model_params)
val_model.fit(X_train, y_train)

val_pred = val_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Local validation MAE (approx.): {val_mae:.5f}")

full_model = HistGradientBoostingRegressor(**model_params)
full_model.fit(X_array, y)

del X, X_array, X_train, X_val, y, y_train, y_val, train, test, train_df, test_df
gc.collect()

test_pred = full_model.predict(test_X_array)

P_MIN, P_MAX = y.min(), y.max()
test_pred = np.clip(test_pred, P_MIN, P_MAX)

submission["pressure"] = test_pred
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv (rows:", submission.shape[0], ")")
