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

No external packages required in the script and installed.

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

os.environ["OMP_NUM_THREADS"] = "8"  # adjust to the number of cores available
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import gc

base_dir = "../input/ventilator-pressure-prediction"
if not os.path.exists(base_dir):
    base_dir = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

dtypes = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train = pd.read_csv(train_path, dtype=dtypes)
test = pd.read_csv(test_path, dtype=dtypes)

y = train["pressure"].values
breath_id_strat = train["breath_id"].values

train_feats = train.drop(columns=["pressure"]).copy()
train_feats["__is_train"] = 1
test_feats = test.copy()
test_feats["__is_train"] = 0

combined = pd.concat([train_feats, test_feats], ignore_index=True)


def add_features(df):
    cols = [
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
        "u_in_roll_mean3",
        "u_in_roll_std3",
        "u_in_roll_mean5",
        "u_out_lag1",
        "u_out_lag2",
        "u_out_lag3",
        "u_out_diff1",
        "u_out_diff2",
        "u_out_cumsum",
        "u_out_roll_mean3",
        "u_out_roll_std3",
        "u_in_times_u_out",
        "u_in_breath_mean",
        "u_in_breath_max",
        "u_in_breath_min",
        "u_out_breath_mean",
        "u_out_breath_max",
        "u_out_breath_min",
        "R_C",
        "R_div_C",
        "time_step_sq",
        "u_in_times_time",
    ]
    for c in cols:
        df[c] = np.empty(df.shape[0], dtype=np.float32)

    for _, group in df.groupby("breath_id", sort=False):
        idx = group.index.values

        u_in = group["u_in"].values.astype(np.float32)
        u_out = group["u_out"].values.astype(np.int8)
        ts = group["time_step"].values.astype(np.float32)

        lag1 = np.concatenate(([0.0], u_in[:-1]))
        lag2 = np.concatenate(([0.0, 0.0], u_in[:-2]))
        df.loc[idx, "u_in_lag1"] = lag1
        df.loc[idx, "u_in_lag2"] = lag2
        df.loc[idx, "u_in_diff1"] = u_in - lag1
        df.loc[idx, "u_in_diff2"] = u_in - lag2
        df.loc[idx, "u_in_cumsum"] = np.cumsum(u_in, dtype=np.float32)

        csum = np.cumsum(u_in, dtype=np.float64)
        csum = np.insert(csum, 0, 0.0)
        win3_sum = csum[3:] - csum[:-3]
        mean3 = win3_sum / 3.0
        mean3 = np.concatenate((np.cumsum(u_in[:2]) / np.arange(1, 3), mean3))
        df.loc[idx, "u_in_roll_mean3"] = mean3.astype(np.float32)

        csum2 = np.cumsum(u_in**2, dtype=np.float64)
        csum2 = np.insert(csum2, 0, 0.0)
        win3_sq_sum = csum2[3:] - csum2[:-3]
        var3 = (win3_sq_sum - (win3_sum**2) / 3.0) / 3.0
        std3 = np.sqrt(np.maximum(var3, 0.0))
        std3 = np.concatenate((np.zeros(2), std3))
        df.loc[idx, "u_in_roll_std3"] = std3.astype(np.float32)

        csum5 = np.cumsum(u_in, dtype=np.float64)
        csum5 = np.insert(csum5, 0, 0.0)
        win5_sum = csum5[5:] - csum5[:-5]
        mean5 = win5_sum / 5.0
        prefixes = np.cumsum(u_in[:4])
        mean5 = np.concatenate((prefixes / np.arange(1, 5), mean5))
        df.loc[idx, "u_in_roll_mean5"] = mean5.astype(np.float32)

        lag1_o = np.concatenate(([0], u_out[:-1]))
        lag2_o = np.concatenate(([0, 0], u_out[:-2]))
        lag3_o = np.concatenate(([0, 0, 0], u_out[:-3]))
        df.loc[idx, "u_out_lag1"] = lag1_o
        df.loc[idx, "u_out_lag2"] = lag2_o
        df.loc[idx, "u_out_lag3"] = lag3_o
        df.loc[idx, "u_out_diff1"] = u_out - lag1_o
        df.loc[idx, "u_out_diff2"] = u_out - lag2_o
        df.loc[idx, "u_out_cumsum"] = np.cumsum(u_out, dtype=np.int32)

        csum_o = np.cumsum(u_out, dtype=np.float64)
        csum_o = np.insert(csum_o, 0, 0.0)
        win3_sum_o = csum_o[3:] - csum_o[:-3]
        mean3_o = win3_sum_o / 3.0
        mean3_o = np.concatenate((np.cumsum(u_out[:2]) / np.arange(1, 3), mean3_o))
        df.loc[idx, "u_out_roll_mean3"] = mean3_o.astype(np.float32)

        csum2_o = np.cumsum(u_out**2, dtype=np.float64)
        csum2_o = np.insert(csum2_o, 0, 0.0)
        win3_sq_sum_o = csum2_o[3:] - csum2_o[:-3]
        var3_o = (win3_sq_sum_o - (win3_sum_o**2) / 3.0) / 3.0
        std3_o = np.sqrt(np.maximum(var3_o, 0.0))
        std3_o = np.concatenate((np.zeros(2), std3_o))
        df.loc[idx, "u_out_roll_std3"] = std3_o.astype(np.float32)

        df.loc[idx, "u_in_times_u_out"] = (u_in * u_out).astype(np.float32)
        df.loc[idx, "u_in_breath_mean"] = np.mean(u_in, dtype=np.float32)
        df.loc[idx, "u_in_breath_max"] = float(np.max(u_in))
        df.loc[idx, "u_in_breath_min"] = float(np.min(u_in))
        df.loc[idx, "u_out_breath_mean"] = np.mean(u_out, dtype=np.float32)
        df.loc[idx, "u_out_breath_max"] = float(np.max(u_out))
        df.loc[idx, "u_out_breath_min"] = float(np.min(u_out))

        R = group["R"].values.astype(np.int16)
        C = group["C"].values.astype(np.int16)
        df.loc[idx, "R_C"] = (R * C).astype(np.int16)
        df.loc[idx, "R_div_C"] = (R / C).astype(np.float32)
        df.loc[idx, "time_step_sq"] = (ts**2).astype(np.float32)
        df.loc[idx, "u_in_times_time"] = (u_in * ts).astype(np.float32)


add_features(combined)

train = (
    combined[combined["__is_train"] == 1]
    .drop(columns=["__is_train"])
    .reset_index(drop=True)
)
test = (
    combined[combined["__is_train"] == 0]
    .drop(columns=["__is_train"])
    .reset_index(drop=True)
)

del combined, train_feats, test_feats
gc.collect()




## === cell 1
X_train = train.drop(columns=["breath_id", "id"])
X_test = test.drop(columns=["breath_id", "id"])




## === cell 2
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.1, random_state=42, stratify=breath_id_strat
)

X_tr_f = X_tr.astype(np.float32)
X_val_f = X_val.astype(np.float32)
X_test_f = X_test.astype(np.float32)

scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr_f)
X_val_scaled = scaler.transform(X_val_f)
X_test_scaled = scaler.transform(X_test_f)

del X_train, X_test, X_tr, X_val, X_tr_f, X_val_f, X_test_f
gc.collect()




## === cell 3
model = HistGradientBoostingRegressor(
    max_depth=12,
    learning_rate=0.015,
    max_iter=1500,
    loss="absolute_error",
    random_state=42,
)
model.fit(X_tr_scaled, y_tr)




## === cell 4
val_pred = model.predict(X_val_scaled)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (approximate): {val_mae:.5f}")




## === cell 5
preds = model.predict(X_test_scaled)




## === cell 6
PRESSURE_MIN = y.min()
PRESSURE_MAX = y.max()
preds_clipped = np.clip(preds, PRESSURE_MIN, PRESSURE_MAX)




## === cell 7
submission = pd.read_csv(sample_sub_path)
submission["pressure"] = preds_clipped
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
