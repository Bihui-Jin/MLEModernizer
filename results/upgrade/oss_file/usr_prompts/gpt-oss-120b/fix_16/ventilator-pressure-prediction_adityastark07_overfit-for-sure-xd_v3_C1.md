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
from sklearn.ensemble import HistGradientBoostingRegressor

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save RAM and speed up ops."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype(np.int32)
    return df


train_df = reduce_mem(pd.read_csv(train_path))
test_df = reduce_mem(pd.read_csv(test_path))
sample_sub = pd.read_csv(sample_sub_path)

base_features = ["R", "C", "time_step", "u_in", "u_out"]


def _process_group(g):
    """Create engineered columns for a single breath using NumPy (vectorised)."""
    R = g["R"].values
    C = g["C"].values
    ts = g["time_step"].values
    u_in = g["u_in"].values
    u_out = g["u_out"].values

    g["R_times_C"] = R * C
    g["u_in_sq"] = u_in**2
    g["time_u_in"] = ts * u_in
    g["R_div_C"] = R / (C + 1e-6)
    g["C_div_R"] = C / (R + 1e-6)
    g["u_in_R"] = u_in * R
    g["u_in_C"] = u_in * C
    g["u_out_R"] = u_out * R
    g["u_out_C"] = u_out * C
    g["is_insp"] = (u_in > 0).astype(np.int8)

    dt = np.empty_like(ts)
    dt[0] = 0.0
    dt[1:] = ts[1:] - ts[:-1]
    g["dt"] = dt
    g["u_in_cum"] = np.cumsum(u_in * dt)
    g["u_out_cum"] = np.cumsum(u_out * dt)

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[0] = 0.0
    u_in_lag1[1:] = u_in[:-1]
    u_out_lag1 = np.empty_like(u_out)
    u_out_lag1[0] = 0
    u_out_lag1[1:] = u_out[:-1]
    g["u_in_lag1"] = u_in_lag1
    g["u_out_lag1"] = u_out_lag1
    g["u_in_diff"] = u_in - u_in_lag1
    g["u_out_diff"] = u_out - u_out_lag1

    roll3 = pd.Series(u_in).rolling(window=3, min_periods=1).mean().values
    g["u_in_roll3"] = roll3
    roll3_out = pd.Series(u_out).rolling(window=3, min_periods=1).mean().values
    g["u_out_roll3"] = roll3_out

    roll5 = pd.Series(u_in).rolling(window=5, min_periods=1).mean().values
    g["u_in_roll5"] = roll5
    roll5_out = pd.Series(u_out).rolling(window=5, min_periods=1).mean().values
    g["u_out_roll5"] = roll5_out

    g["breath_len"] = len(g)

    return g


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply engineered features efficiently."""
    df = df.copy()
    df = df.groupby("breath_id", sort=False, group_keys=False).apply(_process_group)
    return df


train_df = add_interactions(train_df)
test_df = add_interactions(test_df)

feature_cols = base_features + [
    "R_times_C",
    "u_in_sq",
    "time_u_in",
    "dt",
    "u_in_cum",
    "u_out_cum",
    "R_div_C",
    "C_div_R",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_diff",
    "u_out_diff",
    "u_in_roll3",
    "u_out_roll3",
    "u_in_roll5",
    "u_out_roll5",
    "is_insp",
    "u_in_R",
    "u_in_C",
    "u_out_R",
    "u_out_C",
    "breath_len",
]

X_train_full = train_df[feature_cols].astype(np.float32)
y_train_full = train_df["pressure"].astype(np.float32)
X_test = test_df[feature_cols].astype(np.float32)

X_train = X_train_full
y_train = y_train_full




## === cell 1
model = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_depth=16,
    learning_rate=0.01,
    max_iter=1500,  # more iterations for higher capacity
    max_bins=128,
    l2_regularization=0.02,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=0,
)
model.fit(X_train, y_train)




## === cell 2
raw_pred = model.predict(X_test)

pred_df = pd.DataFrame({"breath_id": test_df["breath_id"].values, "pred": raw_pred})
pred_df["smooth_pred"] = pred_df.groupby("breath_id")["pred"].transform(
    lambda s: s.rolling(window=3, min_periods=1, center=True).mean()
)

final_pred = np.clip(pred_df["smooth_pred"].values, 0, 50)

submission = sample_sub.copy()
submission["pressure"] = final_pred
submission.to_csv("submission.csv", index=False)

print(submission.head())
