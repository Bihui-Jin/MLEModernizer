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

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

import numpy as np
import pandas as pd
import random
import gc
from sklearn.ensemble import HistGradientBoostingRegressor


def find_nearest(prediction, sorted_pressures):
    """
    Round a predicted pressure to the nearest value present in the training set.
    """
    total_pressures_len = len(sorted_pressures)
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def round_series_to_nearest(series, sorted_vals):
    """
    Vectorised version of find_nearest for an entire pandas Series / numpy array.
    """
    arr = series.values
    idx = np.searchsorted(sorted_vals, arr, side="left")
    idx = np.clip(idx, 0, len(sorted_vals) - 1)

    left = sorted_vals[np.maximum(idx - 1, 0)]
    right = sorted_vals[np.minimum(idx, len(sorted_vals) - 1)]

    use_left = np.abs(arr - left) < np.abs(right - arr)
    rounded = np.where(use_left, left, right)
    return pd.Series(rounded, index=series.index)


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 1
import joblib  # retained for possible future use; not used in the optimized loop

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

dtypes = {
    "R": "float32",
    "C": "float32",
    "breath_id": "int32",
    "id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(train_path, dtype=dtypes)
df_test = pd.read_csv(test_path, dtype=dtypes)

unique_pressures = np.sort(df_train["pressure"].unique())


def add_features(df):
    df["R"] = df["R"].astype("float32")
    df["C"] = df["C"].astype("float32")
    df["u_in"] = df["u_in"].astype("float32")
    df["u_out"] = df["u_out"].astype("float32")
    df["time_step"] = df["time_step"].astype("float32")

    df["RC"] = (df["R"] * df["C"]).astype("float32")
    df["u_in_R"] = (df["u_in"] * df["R"]).astype("float32")
    df["u_in_C"] = (df["u_in"] * df["C"]).astype("float32")
    df["time_u_in"] = (df["time_step"] * df["u_in"]).astype("float32")
    df["u_in_sq"] = (df["u_in"] ** 2).astype("float32")
    df["u_out_u_in"] = (df["u_out"] * df["u_in"]).astype("float32")
    df["time_u_out"] = (df["time_step"] * df["u_out"]).astype("float32")
    df["RC_u_in"] = (df["RC"] * df["u_in"]).astype("float32")

    df.sort_values(["breath_id", "time_step"], inplace=True)
    dt = df.groupby("breath_id")["time_step"].diff().fillna(0).astype("float32")

    df["cumulative_u_in"] = (
        (df["u_in"] * dt).groupby(df["breath_id"]).cumsum().astype("float32")
    )
    df["cumulative_u_out"] = (
        (df["u_out"] * dt).groupby(df["breath_id"]).cumsum().astype("float32")
    )

    df["max_time"] = (
        df.groupby("breath_id")["time_step"].transform("max").astype("float32")
    )
    df["time_ratio"] = (df["time_step"] / df["max_time"]).astype("float32")
    df["cumulative_time"] = dt.groupby(df["breath_id"]).cumsum().astype("float32")
    df["cumulative_u_in_sq"] = (
        ((df["u_in"] ** 2) * dt).groupby(df["breath_id"]).cumsum().astype("float32")
    )

    df.sort_index(inplace=True)
    return df


df_train = add_features(df_train)
df_test = add_features(df_test)

gc.collect()

feature_cols = [
    "R",
    "C",
    "u_in",
    "u_out",
    "time_step",
    "RC",
    "u_in_R",
    "u_in_C",
    "time_u_in",
    "u_in_sq",
    "u_out_u_in",
    "time_u_out",
    "RC_u_in",
    "cumulative_u_in",
    "cumulative_u_out",
    "max_time",
    "time_ratio",
    "cumulative_time",
    "cumulative_u_in_sq",
]

set_seed(2021)


def train_group(key, group_df):
    """Train a HistGradientBoostingRegressor for a specific (R, C) pair."""
    X = group_df[feature_cols].values
    y = group_df["pressure"].values
    model = HistGradientBoostingRegressor(
        max_iter=5000,
        learning_rate=0.02,
        max_depth=None,
        max_bins=255,
        random_state=2021,
        early_stopping=False,
    )
    model.fit(X, y)
    return (key, model)


group_items = list(df_train.groupby(["R", "C"]))

models = {}
for key, grp in group_items:
    _, mdl = train_group(key, grp)
    models[key] = mdl
    del grp
    gc.collect()

preds = np.empty(len(df_test), dtype="float32")
preds[:] = np.nan  # initialise with NaN for safety

for (r_val, c_val), model in models.items():
    mask = (df_test["R"] == r_val) & (df_test["C"] == c_val)
    if mask.any():
        preds[mask] = model.predict(df_test.loc[mask, feature_cols])

global_mean = df_train["pressure"].mean()
preds = np.where(np.isnan(preds), global_mean, preds)

df_test["pred_pressure_raw"] = preds

df_test["pred_pressure_raw"] = round_series_to_nearest(
    df_test["pred_pressure_raw"], unique_pressures
)




## === cell 2
df_test["pressure"] = df_test["pred_pressure_raw"]

submission = df_test[["id", "pressure"]].copy()
submission = submission.astype({"id": int, "pressure": float})

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with {len(submission)} rows.")
