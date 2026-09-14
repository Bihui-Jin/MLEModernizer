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
import multiprocessing
import numpy as np
import pandas as pd
import gc  # explicit memory cleanup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

os.environ["OMP_NUM_THREADS"] = "1"

CPU_COUNT = multiprocessing.cpu_count()
N_JOBS = max(1, CPU_COUNT)  # use all cores for RandomForest parallelism




## === cell 1
train_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id", "pressure", "id"]
dtypes_train = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "breath_id": "int32",
    "pressure": "float32",
    "id": "int16",
}
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=train_usecols,
    dtype=dtypes_train,
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype("float32")  # cache as float32
total_pressures_len = len(sorted_pressures)


def find_nearest_batch(preds):
    """Vectorized nearest‑training‑pressure lookup."""
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx_low = np.maximum(idx - 1, 0)
    idx_high = np.minimum(idx, total_pressures_len - 1)

    low_vals = sorted_pressures[idx_low]
    high_vals = sorted_pressures[idx_high]

    choose_low = np.abs(low_vals - preds) < np.abs(high_vals - preds)
    return np.where(choose_low, low_vals, high_vals)


def set_seed(seed=2021):
    np.random.seed(seed)
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 2
set_seed(2021)

a = "../input/gb-data-blending-recover/0.151 seed 1997.csv"
b = "../input/gb-data-blending-recover/0.152 blend.csv"


def blend(a_path, b_path):
    """Blend two existing submission files if they are present."""
    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    a_df.pressure = a_df.pressure * 0.55 + b_df.pressure * 0.45
    a_df["pressure"] = a_df["pressure"].apply(find_nearest_batch)  # vectorized
    a_df.to_csv("blend.csv", index=False)
    return a_df


if os.path.exists(a) and os.path.exists(b):
    blend(a, b)
else:
    R = df_train["R"].values.astype(np.float32)
    C = df_train["C"].values.astype(np.float32)
    time_step = df_train["time_step"].values
    u_in = df_train["u_in"].values
    u_out = df_train["u_out"].values.astype(np.float32)
    breath_id = df_train["breath_id"].values

    time_step_sq = time_step**2
    u_in_sq = u_in**2
    R_mul_C = R * C

    change_idx = np.where(np.diff(breath_id) != 0)[0] + 1
    start_idx = np.concatenate(([0], change_idx))
    cum = np.cumsum(u_in, dtype=np.float32)
    offsets = np.zeros_like(cum)
    offsets[start_idx[1:]] = cum[start_idx[1:] - 1]
    cum_u_in = cum - offsets

    X = np.column_stack(
        (R, C, time_step, u_in, u_out, time_step_sq, u_in_sq, R_mul_C, cum_u_in)
    ).astype(np.float32)

    y = df_train["pressure"].values.astype(np.float32)

    del df_train, R, C, time_step, u_in, u_out, breath_id
    gc.collect()

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=2021)

    rf = RandomForestRegressor(
        n_estimators=200,  # core model unchanged
        max_depth=15,
        min_samples_leaf=1,
        n_jobs=N_JOBS,  # parallelism across trees
        random_state=2021,
    )
    rf.fit(X_tr, y_tr)  # data already contiguous, no extra copy

    val_pred = rf.predict(X_val)
    val_pred = find_nearest_batch(val_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE (rounded to nearest training pressure): {val_mae:.5f}")

    test_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id", "id"]
    dtypes_test = {
        "R": "int8",
        "C": "int8",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "breath_id": "int32",
        "id": "int16",
    }
    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=test_usecols,
        dtype=dtypes_test,
    )
    test_ids = df_test["id"].values

    R_t = df_test["R"].values.astype(np.float32)
    C_t = df_test["C"].values.astype(np.float32)
    time_step_t = df_test["time_step"].values
    u_in_t = df_test["u_in"].values
    u_out_t = df_test["u_out"].values.astype(np.float32)
    breath_id_t = df_test["breath_id"].values

    time_step_sq_t = time_step_t**2
    u_in_sq_t = u_in_t**2
    R_mul_C_t = R_t * C_t

    change_idx_t = np.where(np.diff(breath_id_t) != 0)[0] + 1
    start_idx_t = np.concatenate(([0], change_idx_t))
    cum_t = np.cumsum(u_in_t, dtype=np.float32)
    offsets_t = np.zeros_like(cum_t)
    offsets_t[start_idx_t[1:]] = cum_t[start_idx_t[1:] - 1]
    cum_u_in_t = cum_t - offsets_t

    X_test = np.column_stack(
        (
            R_t,
            C_t,
            time_step_t,
            u_in_t,
            u_out_t,
            time_step_sq_t,
            u_in_sq_t,
            R_mul_C_t,
            cum_u_in_t,
        )
    ).astype(np.float32)

    del df_test, R_t, C_t, time_step_t, u_in_t, u_out_t, breath_id_t
    gc.collect()

    test_pred = rf.predict(X_test)  # already contiguous
    test_pred = find_nearest_batch(test_pred)

    submission = pd.DataFrame({"id": test_ids, "pressure": test_pred})
    submission.to_csv("submission.csv", index=False)
    print('Submission file "submission.csv" written successfully.')
