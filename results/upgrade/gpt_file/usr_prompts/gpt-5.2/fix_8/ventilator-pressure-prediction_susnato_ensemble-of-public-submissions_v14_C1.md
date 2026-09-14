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

3.9

# 3. Installed packages

geopandas==0.14.4
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

pd.options.mode.copy_on_write = False

BASE_PATHS = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "../kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/data",
    "/kaggle/input",
]


def first_existing_path(paths):
    for p in paths:
        if os.path.exists(p) and os.path.isdir(p):
            if os.path.exists(os.path.join(p, "train.csv")):
                return p
            cand = os.path.join(p, "ventilator-pressure-prediction")
            if os.path.exists(os.path.join(cand, "train.csv")):
                return cand
    return None


DATA_DIR = first_existing_path(BASE_PATHS)
if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find dataset directory in any of: {BASE_PATHS}")

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

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
sub_dtypes = {"id": "int32", "pressure": "float32"}

train = pd.read_csv(train_path, dtype=train_dtypes, engine="c")
test = pd.read_csv(test_path, dtype=test_dtypes, engine="c")
sub = pd.read_csv(sample_path, dtype=sub_dtypes, engine="c")

assert {"id", "pressure"}.issubset(sub.columns)
assert "pressure" in train.columns
assert "pressure" not in test.columns

train.shape, test.shape, sub.shape




## === cell 1
def add_features_inplace(df: pd.DataFrame) -> None:
    df.sort_values(
        ["breath_id", "time_step"], inplace=True, kind="mergesort"
    )  # stable sort
    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0).astype(np.float32)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    df["dt"] = g["time_step"].diff().fillna(0.0).astype(np.float32)

    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)
    df["u_in_diff2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype(np.float32)
    df["u_out_diff1"] = (df["u_out"] - df["u_out_lag1"]).astype(np.int8)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)
    df["u_out_cumsum"] = g["u_out"].cumsum().astype(np.int16)

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(copy=False)
    breath = df["breath_id"].to_numpy(copy=False)

    is_new_breath = np.empty(len(df), dtype=bool)
    is_new_breath[0] = True
    is_new_breath[1:] = breath[1:] != breath[:-1]

    prev_u_out = np.empty(len(df), dtype=u_out.dtype)
    prev_u_out[0] = 1
    prev_u_out[1:] = u_out[:-1]
    is_new_insp_segment = is_new_breath | (u_out == 1) | (prev_u_out == 1)

    seg_start = np.maximum.accumulate(
        np.where(is_new_insp_segment, np.arange(len(df), dtype=np.int32), 0)
    )
    idx_in_seg = (np.arange(len(df), dtype=np.int32) - seg_start).astype(np.int32)

    total_csum = np.cumsum(u_in, dtype=np.float64).astype(np.float32)
    start_total = np.zeros(len(df), dtype=np.float32)
    start_total[0] = 0.0
    seg_starts = np.flatnonzero(is_new_insp_segment)
    prev_total = np.zeros_like(seg_starts, dtype=np.float32)
    m = seg_starts > 0
    prev_total[m] = total_csum[seg_starts[m] - 1]
    seg_ends = np.r_[seg_starts[1:], len(df)]
    for s, e, pv in zip(seg_starts, seg_ends, prev_total):
        start_total[s:e] = pv
    seg_csum = total_csum - start_total
    df["u_in_cumsum_insp"] = seg_csum.astype(np.float32)

    prefix_excl = seg_csum - u_in  # sum of previous values in inspiration segment

    def rollmean_prev_k(
        prefix_excl_arr: np.ndarray, idx_arr: np.ndarray, k: int
    ) -> np.ndarray:
        i = idx_arr
        denom = np.minimum(k, i).astype(np.float32)
        out = np.zeros(len(prefix_excl_arr), dtype=np.float32)
        valid = denom > 0
        if not np.any(valid):
            return out
        take_k = i >= k
        prev_idx = np.arange(len(prefix_excl_arr), dtype=np.int32) - k
        prev_val = np.zeros(len(prefix_excl_arr), dtype=np.float32)
        prev_val[take_k] = prefix_excl_arr[prev_idx[take_k]]
        num = prefix_excl_arr - prev_val
        out[valid] = num[valid] / denom[valid]
        return out

    df["u_in_rollmean3"] = rollmean_prev_k(prefix_excl, idx_in_seg, 3)
    df["u_in_rollmean5"] = rollmean_prev_k(prefix_excl, idx_in_seg, 5)

    df["u_in_x_R"] = (df["u_in"] * df["R"].to_numpy(copy=False)).astype(np.float32)
    df["u_in_x_C"] = (df["u_in"] * df["C"].to_numpy(copy=False)).astype(np.float32)


add_features_inplace(train)
add_features_inplace(test)

FEATURES = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_out_lag1",
    "dt",
    "u_in_diff1",
    "u_in_diff2",
    "u_out_diff1",
    "u_in_cumsum",
    "u_out_cumsum",
    "u_in_rollmean3",
    "u_in_rollmean5",
    "u_in_x_R",
    "u_in_x_C",
    "u_in_cumsum_insp",
]

X = train[FEATURES]
y = train["pressure"].astype(np.float32)

w = (train["u_out"].to_numpy(copy=False) == 0).astype(np.float32)

test_X = test[FEATURES]

X.shape, test_X.shape, float(w.mean())



## === cell 2
from sklearn.ensemble import HistGradientBoostingRegressor

rng = 42

model = HistGradientBoostingRegressor(
    loss="absolute_error",  # aligns with MAE metric
    learning_rate=0.05,
    max_depth=6,
    max_iter=400,
    min_samples_leaf=40,
    l2_regularization=0.0,
    random_state=rng,
)

model.fit(X, y, sample_weight=w)



## === cell 3
pred = model.predict(test_X).astype(np.float32)

pred_final = pred.copy()
insp_mask = test["u_out"].to_numpy(copy=False) == 0

if np.any(insp_mask):
    exp_const = np.nanmedian(pred_final[insp_mask]).astype(np.float32)
else:
    exp_const = np.float32(np.nanmedian(pred_final))
pred_final[~insp_mask] = exp_const

pressure_grid = np.sort(train["pressure"].unique().astype(np.float32))
pred_snapped = pred_final.copy()

if insp_mask.any():
    insp_vals = pred_final[insp_mask]
    idx = np.searchsorted(pressure_grid, insp_vals, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)

    p_right = pressure_grid[idx]
    p_left = pressure_grid[idx_left]
    pred_snapped[insp_mask] = np.where(
        np.abs(insp_vals - p_left) <= np.abs(insp_vals - p_right),
        p_left,
        p_right,
    ).astype(np.float32)

submission = sub[["id"]].copy()
submission["pressure"] = pd.Series(
    pred_snapped, index=test.index, dtype=np.float32
).to_numpy()

if not (
    len(submission) == len(test)
    and np.array_equal(submission["id"].values, test["id"].values)
):
    temp = pd.DataFrame(
        {
            "id": test["id"].to_numpy(copy=False),
            "pressure": pred_snapped.astype(np.float32),
        }
    )
    submission = sub[["id"]].merge(temp, on="id", how="left")
    submission["pressure"] = (
        submission["pressure"]
        .fillna(np.float32(np.nanmedian(pred_snapped)))
        .astype(np.float32)
    )

submission.to_csv("submission.csv", index=False)
submission.head()
