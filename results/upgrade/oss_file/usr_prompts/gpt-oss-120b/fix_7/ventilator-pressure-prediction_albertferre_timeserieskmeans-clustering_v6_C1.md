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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error



## === cell 1
path = "../input/ventilator-pressure-prediction"
if not os.path.isdir(path):
    path = "/kaggle/input/ventilator-pressure-prediction"
train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "test.csv"))



## === cell 2
train_ts = train["u_in"].values.reshape(-1, 80)
test_ts = test["u_in"].values.reshape(-1, 80)

breath_ids = train["breath_id"].unique()
train_breaths, val_breaths = train_test_split(
    breath_ids, test_size=0.1, random_state=42
)


def rows_of(breath_set):
    mask = train["breath_id"].isin(breath_set)
    return np.where(mask)[0]


train_idx = rows_of(train_breaths)
val_idx = rows_of(val_breaths)

train_ts_split = train_ts[train_idx // 80]  # one entry per breath
val_ts_split = train_ts[val_idx // 80]



## === cell 3
target_mae = 6.162
candidate_clusters = [60, 80, 100]
best_n = None
best_gap = np.inf
best_mae = None

for n in candidate_clusters:
    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    km.fit(train_ts_split)

    train_clusters = km.predict(train_ts_split)
    val_clusters = km.predict(val_ts_split)

    train["cluster"] = np.repeat(train_clusters, 80)
    val = train.loc[val_idx].copy()
    val["cluster"] = np.repeat(val_clusters, 80)

    train["step"] = train.groupby("breath_id").cumcount()
    val["step"] = val.groupby("breath_id").cumcount()

    avg_pressure_cluster = (
        train.groupby(["R", "C", "step", "cluster"])["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_cluster"})
    )
    avg_pressure_overall = (
        train.groupby(["R", "C", "step"])["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_overall"})
    )

    pred_val = val.merge(
        avg_pressure_cluster, how="left", on=["R", "C", "step", "cluster"]
    )
    pred_val = pred_val.merge(avg_pressure_overall, how="left", on=["R", "C", "step"])

    global_mean = train["pressure"].mean()
    pred_val["pressure"] = (
        pred_val["pressure_cluster"]
        .fillna(pred_val["pressure_overall"])
        .fillna(global_mean)
    )
    pred_val["pressure"] = pred_val["pressure"].clip(lower=0, upper=45)

    mae = mean_absolute_error(val["pressure"], pred_val["pressure"])
    gap = abs(mae - target_mae)
    if gap < best_gap:
        best_gap = gap
        best_n = n
        best_mae = mae

    print(f"N_CLUSTERS={n:3d} | validation MAE={mae:.4f} | gap={gap:.4f}")

print(f"\nChosen N_CLUSTERS={best_n} (MAE={best_mae:.4f}, gap={best_gap:.4f})")



## === cell 4
km = KMeans(n_clusters=best_n, random_state=42, n_init=10)
km.fit(train_ts)

train_clusters = km.predict(train_ts)
test_clusters = km.predict(test_ts)

train["cluster"] = np.repeat(train_clusters, 80)
test["cluster"] = np.repeat(test_clusters, 80)

train["step"] = train.groupby("breath_id").cumcount()
test["step"] = test.groupby("breath_id").cumcount()

avg_pressure_cluster = (
    train.groupby(["R", "C", "step", "cluster"])["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_cluster"})
)

avg_pressure_overall = (
    train.groupby(["R", "C", "step"])["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pressure_overall"})
)

global_mean_pressure = train["pressure"].mean()

pred = test.merge(avg_pressure_cluster, how="left", on=["R", "C", "step", "cluster"])
pred = pred.merge(avg_pressure_overall, how="left", on=["R", "C", "step"])

pred["pressure"] = (
    pred["pressure_cluster"]
    .fillna(pred["pressure_overall"])
    .fillna(global_mean_pressure)
)
pred["pressure"] = pred["pressure"].clip(lower=0, upper=45)

submission = pd.DataFrame({"id": test["id"], "pressure": pred["pressure"]})
submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' written. First rows:")
print(submission.head())
