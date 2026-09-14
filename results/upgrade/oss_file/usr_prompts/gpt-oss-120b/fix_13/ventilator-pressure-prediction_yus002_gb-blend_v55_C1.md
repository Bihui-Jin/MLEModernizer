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
import numpy as np
import pandas as pd
import random
import gc
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(2021)

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

usecols_train = ["R", "C", "time_step", "u_in", "u_out", "pressure", "id"]
usecols_test = ["R", "C", "time_step", "u_in", "u_out", "id"]

dtype_dict = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
    "id": "int16",
}

df_train = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_dict)
df_test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype={k: v for k, v in dtype_dict.items() if k != "pressure"},
)

sorted_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = len(sorted_pressures)


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    """Vectorized nearest‑pressure lookup for an array of predictions."""
    insert_idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx_low = np.clip(insert_idx - 1, 0, total_pressures_len - 1)
    idx_high = np.clip(insert_idx, 0, total_pressures_len - 1)

    low_vals = sorted_pressures[idx_low]
    high_vals = sorted_pressures[idx_high]

    choose_low = np.abs(low_vals - preds) <= np.abs(high_vals - preds)
    return np.where(choose_low, low_vals, high_vals)


df_train["time_bin"] = (df_train["time_step"] * 100).round() / 100
df_test["time_bin"] = (df_test["time_step"] * 100).round() / 100

for col in ["R", "C", "u_out", "time_bin"]:
    df_train[col] = df_train[col].astype("category")
    df_test[col] = df_test[col].astype("category")

group_means = (
    df_train.groupby(["R", "C", "u_out", "time_bin"], as_index=False, sort=False)[
        "pressure"
    ]
    .mean()
    .rename(columns={"pressure": "group_pressure"})
)

df_test = df_test.merge(group_means, on=["R", "C", "u_out", "time_bin"], how="left")
global_mean = df_train["pressure"].mean()
df_test["group_pressure"].fillna(global_mean, inplace=True)

df_train = df_train.merge(group_means, on=["R", "C", "u_out", "time_bin"], how="left")
df_train["group_pressure"].fillna(global_mean, inplace=True)

lr_features = ["R", "C", "u_in", "u_out", "time_step"]

X = df_train[lr_features].astype(np.float32).values
y = df_train["pressure"].astype(np.float32).values
group_pressure = df_train["group_pressure"].values
del df_train
gc.collect()

X_train, X_val, y_train, y_val, gp_train, gp_val = train_test_split(
    X, y, group_pressure, test_size=0.1, random_state=2021
)

lr = LinearRegression()
lr.fit(X, y)

gbr = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=2021,
)
gbr.fit(X, y)

lr_val = lr.predict(X_val)
gbr_val = gbr.predict(X_val)

mae_group = mean_absolute_error(y_val, gp_val)
mae_lr = mean_absolute_error(y_val, lr_val)
mae_gbr = mean_absolute_error(y_val, gbr_val)

inv_mae = np.array([1 / mae_group, 1 / mae_lr, 1 / mae_gbr])
weights = inv_mae / inv_mae.sum()
w_group, w_lr, w_gbr = weights

print(f"Blend weights – group: {w_group:.3f}, lr: {w_lr:.3f}, gbr: {w_gbr:.3f}")

batch_size = 100_000
lr_preds = []
gbr_preds = []
for start in range(0, len(df_test), batch_size):
    end = start + batch_size
    batch_X = df_test.loc[start : end - 1, lr_features].astype(np.float32).values
    lr_preds.append(lr.predict(batch_X))
    gbr_preds.append(gbr.predict(batch_X))

df_test["lr_pred_raw"] = np.concatenate(lr_preds)
df_test["gbr_pred_raw"] = np.concatenate(gbr_preds)

combined_raw = (
    w_group * df_test["group_pressure"]
    + w_lr * df_test["lr_pred_raw"]
    + w_gbr * df_test["gbr_pred_raw"]
)

df_test["pressure_pred"] = find_nearest_vec(combined_raw.values)

submission = df_test[["id"]].copy()
submission["pressure"] = df_test["pressure_pred"]
output_path = "submission.csv"
submission.to_csv(output_path, index=False)

print(f"Submission written to {output_path} – shape: {submission.shape}")




## === cell 1
def blend(file_a: str, file_b: str, out_path: str = "blend.csv"):
    """Blend two existing submission files (if both are present)."""
    if not (os.path.exists(file_a) and os.path.exists(file_b)):
        print("One or both blend files not found – skipping blending.")
        return None
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    df_a["pressure"] = df_a["pressure"] * 0.55 + df_b["pressure"] * 0.45
    df_a["pressure"] = find_nearest_vec(df_a["pressure"].values)
    df_a.to_csv(out_path, index=False)
    print(f"Blended submission saved to {out_path}")
    return df_a
