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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
for df in [train, test]:
    for lag in range(1, 6):  # lags 1‑5
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)
        df[f"u_in_diff{lag}"] = df["u_in"] - df[f"u_in_lag{lag}"]
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_cummean"] = df.groupby("breath_id")["u_in"].transform(
        lambda s: s.expanding().mean()
    )
    df["u_in_roll_mean3"] = df.groupby("breath_id")["u_in"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df["u_in_roll_std3"] = df.groupby("breath_id")["u_in"].transform(
        lambda s: s.rolling(window=3, min_periods=1).std().fillna(0)
    )

    for lag in range(1, 4):  # lags 1‑3
        df[f"u_out_lag{lag}"] = df.groupby("breath_id")["u_out"].shift(lag).fillna(0)
        df[f"u_out_diff{lag}"] = df["u_out"] - df[f"u_out_lag{lag}"]
    df["u_out_roll_mean3"] = df.groupby("breath_id")["u_out"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )

    for lag in range(1, 4):
        df[f"time_step_lag{lag}"] = (
            df.groupby("breath_id")["time_step"].shift(lag).fillna(0)
        )
        df[f"time_step_diff{lag}"] = df["time_step"] - df[f"time_step_lag{lag}"]
    df["time_step_roll_mean3"] = df.groupby("breath_id")["time_step"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )

    df["R_C_interaction"] = df["R"] * df["C"]
    df["u_in_R"] = df["u_in"] * df["R"]
    df["u_in_C"] = df["u_in"] * df["C"]
    df["u_in_R_C"] = df["u_in"] * df["R"] * df["C"]
    df["time_step_sq"] = df["time_step"] ** 2



## === cell 3
k = train["pressure"].mean() / train["u_in"].mean()
print(f"Scaling factor k = {k:.4f}")




## === cell 4
def predict_pressure(df: pd.DataFrame, k: float) -> np.ndarray:
    """
    Vectorised physics‑based pressure prediction per breath.
    Implements the same recurrence:
        p_i = p_{i-1} * exp(-dt_i / tau) + k * u_i * (1 - exp(-dt_i / tau))
    but operates on NumPy arrays for speed.
    """
    orig_idx = df.index.values
    sorted_df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    preds = np.empty(len(sorted_df), dtype=float)

    for _, group in sorted_df.groupby("breath_id", sort=False):
        R = group["R"].iloc[0]
        C = group["C"].iloc[0]
        tau = R * C
        times = group["time_step"].values
        u_ins = group["u_in"].values

        p = np.empty_like(u_ins, dtype=float)
        prev_p = 0.0
        prev_t = 0.0

        if tau == 0:
            p[:] = k * u_ins
        else:
            for i in range(len(u_ins)):
                dt = times[i] - prev_t
                decay = np.exp(-dt / tau)
                p[i] = prev_p * decay + k * u_ins[i] * (1.0 - decay)
                prev_p = p[i]
                prev_t = times[i]

        preds[group.index.values] = p

    result = np.empty(len(df), dtype=float)
    result[sorted_df.index.values] = preds
    final = np.empty(len(df), dtype=float)
    final[np.argsort(orig_idx)] = result
    return final


train["physics_pred"] = predict_pressure(train, k)
test["physics_pred"] = predict_pressure(test, k)



## === cell 5
exclude_cols = ["pressure", "breath_id", "id"]
feature_cols = [c for c in train.columns if c not in exclude_cols]

train_split, val_split = train_test_split(
    train, test_size=0.2, random_state=42, shuffle=True
)

X_train = train_split[feature_cols]
y_train = train_split["pressure"]
X_val = val_split[feature_cols]
y_val = val_split["pressure"]



## === cell 6
model = HistGradientBoostingRegressor(
    loss="absolute_error",  # optimises MAE directly
    max_iter=4000,  # more boosting rounds
    learning_rate=0.005,  # finer steps
    max_depth=None,  # allow deep trees
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20,  # patience for early stop
)

model.fit(X_train, y_train)



## === cell 7
val_pred = model.predict(X_val)
val_mae_model = mean_absolute_error(y_val, val_pred)
print("Validation MAE (HistGradientBoosting):", val_mae_model)

val_physics_pred = val_split["physics_pred"]
val_mae_physics = mean_absolute_error(y_val, val_physics_pred)
print("Validation MAE (Physics‑only):", val_mae_physics)



## === cell 8
test_features = test[feature_cols]
test_pred_ml = model.predict(test_features)

if val_mae_physics < val_mae_model:
    test_pred = test["physics_pred"]
    print("Using physics‑only predictions for submission.")
else:
    test_pred = 0.5 * test_pred_ml + 0.5 * test["physics_pred"]
    print("Blending ML and physics predictions for submission.")

submission_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
submission = pd.read_csv(submission_path)

assert len(submission) == len(test_pred), "Prediction length mismatch"

submission["pressure"] = test_pred
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created successfully.")
