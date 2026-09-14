# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.143497814200942

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.44705) has done: 'I remove the dependency on missing external Kaggle datasets (the blend inputs) by generating predictions from the provided train/test only, while keeping the existing feature engineering and pressure discretization logic intact. The core bug is that `pred` was never created because the referenced submission files do not exist, so I replace the blending section with a simple scikit-learn model that outputs a per-time-step pressure prediction aligned to the test `id` order. To avoid memory issues with the very large training set, I train on a deterministic subset of breaths (still legitimate, no label leakage) and predict on the full test set. Finally, I write a valid `median_submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 1.42642) has done: 'You’re far above the target MAE (1.447 vs 0.143), so we should improve score substantially while keeping your exact feature engineering and model family intact. The largest low-risk gain here is to stop throwing away most training breaths: training on all breaths (or a much larger fixed cap) typically improves this competition a lot without changing the core approach. Second, the current rounding-to-pressure-grid should use the true known grid step (0.07) to better match labels and metric, rather than estimating from data (which can be slightly off). Finally, we keep everything else the same (same features, same scaler, same HGBR, same loss), and still write `median_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(42)

BASE_PATH = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

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

train_df = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test_df = pd.read_csv(TEST_PATH, dtype=test_dtypes)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns.tolist())




## === cell 1
def _rolling15_stats_per_breath_fast(
    u: np.ndarray, breath_id: np.ndarray, window: int = 15
):
    n = u.shape[0]
    out_sum = np.empty(n, dtype=np.float32)
    out_min = np.empty(n, dtype=np.float32)
    out_max = np.empty(n, dtype=np.float32)
    out_mean = np.empty(n, dtype=np.float32)

    changes = np.empty(n, dtype=bool)
    changes[0] = True
    changes[1:] = breath_id[1:] != breath_id[:-1]
    starts = np.flatnonzero(changes)
    ends = np.r_[starts[1:], n]

    for s, e in zip(starts, ends):
        uu = u[s:e].astype(np.float32, copy=False)
        m = e - s
        if m == 0:
            continue

        cs = np.cumsum(uu, dtype=np.float64)
        idx = np.arange(m, dtype=np.int32)
        left = idx - window + 1
        left[left < 0] = 0
        sums = cs - np.where(left > 0, cs[left - 1], 0.0)
        cnt = np.minimum(idx + 1, window).astype(np.float32)
        out_sum[s:e] = sums.astype(np.float32, copy=False)
        out_mean[s:e] = (sums / cnt).astype(np.float32, copy=False)

        min_idx = np.empty(m, dtype=np.int32)
        min_head = 0
        min_tail = 0
        max_idx = np.empty(m, dtype=np.int32)
        max_head = 0
        max_tail = 0

        for i in range(m):
            wstart = i - window + 1
            if wstart < 0:
                wstart = 0

            while min_head < min_tail and min_idx[min_head] < wstart:
                min_head += 1
            while max_head < max_tail and max_idx[max_head] < wstart:
                max_head += 1

            ui = uu[i]
            while min_head < min_tail and uu[min_idx[min_tail - 1]] >= ui:
                min_tail -= 1
            min_idx[min_tail] = i
            min_tail += 1

            while max_head < max_tail and uu[max_idx[max_tail - 1]] <= ui:
                max_tail -= 1
            max_idx[max_tail] = i
            max_tail += 1

            out_min[s + i] = uu[min_idx[min_head]]
            out_max[s + i] = uu[max_idx[max_head]]

    return out_sum, out_min, out_max, out_mean


_R_MAP = {5: 0, 20: 1, 50: 2}
_C_MAP = {10: 0, 20: 1, 50: 2}
_RC_MAP = {
    (r, c): i
    for i, (r, c) in enumerate(
        [
            (5, 10),
            (5, 20),
            (5, 50),
            (20, 10),
            (20, 20),
            (20, 50),
            (50, 10),
            (50, 20),
            (50, 50),
        ]
    )
}
_RC_TABLE = np.empty((3, 3), dtype=np.int8)
for rr in (5, 20, 50):
    for cc in (10, 20, 50):
        _RC_TABLE[_R_MAP[rr], _C_MAP[cc]] = _RC_MAP[(rr, cc)]


def _per_breath_starts_ends(breath_id: np.ndarray):
    n = breath_id.shape[0]
    changes = np.empty(n, dtype=bool)
    changes[0] = True
    changes[1:] = breath_id[1:] != breath_id[:-1]
    starts = np.flatnonzero(changes).astype(np.int64, copy=False)
    ends = np.r_[starts[1:], n].astype(np.int64, copy=False)
    return starts, ends


def _shift_within_breath_vec(x: np.ndarray, breath_id: np.ndarray, k: int):
    n = x.shape[0]
    out = np.zeros(n, dtype=x.dtype)
    if k == 0:
        out[:] = x
        return out
    if k > 0:
        out[k:] = x[:-k]
        same = breath_id[k:] == breath_id[:-k]
        out[k:] *= same.astype(out.dtype, copy=False)
    else:
        kk = -k
        out[:-kk] = x[kk:]
        same = breath_id[:-kk] == breath_id[kk:]
        out[:-kk] *= same.astype(out.dtype, copy=False)
    return out


def _cumsum_within_breath_vec(x: np.ndarray, starts: np.ndarray):
    cs = np.cumsum(x, dtype=np.float32)
    out = cs.astype(np.float32, copy=False)
    if starts.size > 1:
        adj = np.zeros_like(out)
        st = starts[1:]
        adj[st] = out[st - 1]
        out = out - np.cumsum(adj, dtype=np.float32)
    return out


def _diff_within_breath_vec(x: np.ndarray, starts: np.ndarray):
    out = np.empty_like(x, dtype=np.float32)
    out[0] = 0.0
    out[1:] = x[1:] - x[:-1]
    out = out.astype(np.float32, copy=False)
    out[starts] = 0.0
    return out


def _count_within_breath_vec(n: int, starts: np.ndarray, ends: np.ndarray):
    idx = np.arange(n, dtype=np.int32)
    start_of_row = np.empty(n, dtype=np.int32)
    start_of_row[starts] = starts.astype(np.int32, copy=False)
    start_of_row = np.maximum.accumulate(start_of_row)
    count = (idx - start_of_row + 1).astype(np.int16, copy=False)
    return count


def _ewm_mean_per_breath_halflife(
    x: np.ndarray, starts: np.ndarray, ends: np.ndarray, halflife: float
):
    alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
    one_minus = 1.0 - alpha

    out = np.empty(x.shape[0], dtype=np.float32)
    for s, e in zip(starts, ends):
        m = e - s
        if m <= 0:
            continue
        num = 0.0
        den = 0.0
        xs = x  # alias
        oo = out
        for i in range(m):
            xi = float(xs[s + i])
            num = xi + one_minus * num
            den = 1.0 + one_minus * den
            oo[s + i] = num / den
    return out


def _max_mean_within_breath(u_in: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    n = u_in.shape[0]
    out_max = np.empty(n, dtype=np.float32)
    out_mean = np.empty(n, dtype=np.float32)
    for s, e in zip(starts, ends):
        seg = u_in[s:e]
        out_max[s:e] = np.max(seg).astype(np.float32, copy=False)
        out_mean[s:e] = np.mean(seg, dtype=np.float64).astype(np.float32, copy=False)
    return out_max, out_mean


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=False)

    breath_id = df["breath_id"].to_numpy(copy=False)
    starts, ends = _per_breath_starts_ends(breath_id)

    time_step = df["time_step"].to_numpy(copy=False).astype(np.float32, copy=False)
    u_in = df["u_in"].to_numpy(copy=False).astype(np.float32, copy=False)
    u_out = df["u_out"].to_numpy(copy=False).astype(np.int8, copy=False)

    df["cross"] = (u_in * u_out).astype(np.float32, copy=False)
    df["cross2"] = (time_step * u_out).astype(np.float32, copy=False)

    df["area"] = _cumsum_within_breath_vec(
        (time_step * u_in).astype(np.float32, copy=False), starts
    )
    df["time_step_cumsum"] = _cumsum_within_breath_vec(time_step, starts)
    df["u_in_cumsum"] = _cumsum_within_breath_vec(u_in, starts)

    df["u_in_lag1"] = _shift_within_breath_vec(u_in, breath_id, 1)
    df["u_out_lag1"] = _shift_within_breath_vec(u_out, breath_id, 1)
    df["u_in_lag_back1"] = _shift_within_breath_vec(u_in, breath_id, -1)
    df["u_out_lag_back1"] = _shift_within_breath_vec(u_out, breath_id, -1)

    df["u_in_lag2"] = _shift_within_breath_vec(u_in, breath_id, 2)
    df["u_out_lag2"] = _shift_within_breath_vec(u_out, breath_id, 2)
    df["u_in_lag_back2"] = _shift_within_breath_vec(u_in, breath_id, -2)
    df["u_out_lag_back2"] = _shift_within_breath_vec(u_out, breath_id, -2)

    df["u_in_lag3"] = _shift_within_breath_vec(u_in, breath_id, 3)
    df["u_out_lag3"] = _shift_within_breath_vec(u_out, breath_id, 3)
    df["u_in_lag_back3"] = _shift_within_breath_vec(u_in, breath_id, -3)
    df["u_out_lag_back3"] = _shift_within_breath_vec(u_out, breath_id, -3)

    df["u_in_lag4"] = _shift_within_breath_vec(u_in, breath_id, 4)
    df["u_out_lag4"] = _shift_within_breath_vec(u_out, breath_id, 4)
    df["u_in_lag_back4"] = _shift_within_breath_vec(u_in, breath_id, -4)
    df["u_out_lag_back4"] = _shift_within_breath_vec(u_out, breath_id, -4)

    u_in_max, u_in_mean = _max_mean_within_breath(u_in, starts, ends)
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = (u_in_max - u_in).astype(np.float32, copy=False)
    df["breath_id__u_in__diffmean"] = (u_in_mean - u_in).astype(np.float32, copy=False)

    u_in_lag1 = df["u_in_lag1"].to_numpy(copy=False)
    u_out_lag1 = df["u_out_lag1"].to_numpy(copy=False)
    u_in_lag2 = df["u_in_lag2"].to_numpy(copy=False)
    u_out_lag2 = df["u_out_lag2"].to_numpy(copy=False)
    u_in_lag3 = df["u_in_lag3"].to_numpy(copy=False)
    u_out_lag3 = df["u_out_lag3"].to_numpy(copy=False)
    u_in_lag4 = df["u_in_lag4"].to_numpy(copy=False)
    u_out_lag4 = df["u_out_lag4"].to_numpy(copy=False)

    df["u_in_diff1"] = (u_in - u_in_lag1).astype(np.float32, copy=False)
    df["u_out_diff1"] = (u_out - u_out_lag1).astype(np.int16, copy=False)
    df["u_in_diff2"] = (u_in - u_in_lag2).astype(np.float32, copy=False)
    df["u_out_diff2"] = (u_out - u_out_lag2).astype(np.int16, copy=False)
    df["u_in_diff3"] = (u_in - u_in_lag3).astype(np.float32, copy=False)
    df["u_out_diff3"] = (u_out - u_out_lag3).astype(np.int16, copy=False)
    df["u_in_diff4"] = (u_in - u_in_lag4).astype(np.float32, copy=False)
    df["u_out_diff4"] = (u_out - u_out_lag4).astype(np.int16, copy=False)

    n = len(df)
    count = _count_within_breath_vec(n, starts, ends)
    df["one"] = np.int8(1)
    df["count"] = count
    df["u_in_cummean"] = (
        df["u_in_cumsum"].to_numpy(copy=False) / count.astype(np.float32)
    ).astype(np.float32, copy=False)

    breath_id_lag = np.zeros(n, dtype=breath_id.dtype)
    breath_id_lag[1:] = breath_id[:-1]
    breath_id_lag2 = np.zeros(n, dtype=breath_id.dtype)
    breath_id_lag2[2:] = breath_id[:-2]
    df["breath_id_lag"] = breath_id_lag
    df["breath_id_lag2"] = breath_id_lag2
    lagsame = (breath_id_lag == breath_id).astype(np.int8, copy=False)
    lag2same = (breath_id_lag2 == breath_id).astype(np.int8, copy=False)
    df["breath_id_lagsame"] = lagsame
    df["breath_id_lag2same"] = lag2same

    u_in_shift1_global = np.zeros(n, dtype=np.float32)
    u_in_shift1_global[1:] = u_in[:-1]
    u_in_shift2_global = np.zeros(n, dtype=np.float32)
    u_in_shift2_global[2:] = u_in[:-2]
    df["breath_id__u_in_lag"] = (u_in_shift1_global * lagsame).astype(
        np.float32, copy=False
    )
    df["breath_id__u_in_lag2"] = (u_in_shift2_global * lag2same).astype(
        np.float32, copy=False
    )

    df["time_step_diff"] = _diff_within_breath_vec(time_step, starts)

    df["ewm_u_in_mean"] = _ewm_mean_per_breath_halflife(
        u_in.astype(np.float64, copy=False), starts, ends, halflife=9
    ).astype(np.float32, copy=False)

    s15, mn15, mx15, mean15 = _rolling15_stats_per_breath_fast(
        u_in, breath_id, window=15
    )
    df["15_in_sum"] = s15
    df["15_in_min"] = mn15
    df["15_in_max"] = mx15
    df["15_in_mean"] = mean15

    df["u_in_lagback_diff1"] = (
        u_in - df["u_in_lag_back1"].to_numpy(copy=False)
    ).astype(np.float32, copy=False)
    df["u_out_lagback_diff1"] = (
        u_out - df["u_out_lag_back1"].to_numpy(copy=False)
    ).astype(np.int16, copy=False)
    df["u_in_lagback_diff2"] = (
        u_in - df["u_in_lag_back2"].to_numpy(copy=False)
    ).astype(np.float32, copy=False)
    df["u_out_lagback_diff2"] = (
        u_out - df["u_out_lag_back2"].to_numpy(copy=False)
    ).astype(np.int16, copy=False)

    R_raw = df["R"].to_numpy(copy=False)
    C_raw = df["C"].to_numpy(copy=False)
    R_code = np.empty(R_raw.shape[0], dtype=np.int8)
    C_code = np.empty(C_raw.shape[0], dtype=np.int8)
    R_code[R_raw == 5] = 0
    R_code[R_raw == 20] = 1
    R_code[R_raw == 50] = 2
    C_code[C_raw == 10] = 0
    C_code[C_raw == 20] = 1
    C_code[C_raw == 50] = 2
    df["R"] = R_code
    df["C"] = C_code
    df["R__C"] = _RC_TABLE[R_code, C_code]

    return df


print("Feature engineering train ...")
train_fe = add_features(train_df)
print("Feature engineering test ...")
test_fe = add_features(test_df)

test_ids = test_fe["id"].to_numpy(copy=True)

del train_df, test_df
gc.collect()

print(train_fe.shape, test_fe.shape)



## === cell 2
pressure_all = train_fe["pressure"].astype("float32").to_numpy().reshape(-1, 1)
P_MIN = float(np.min(pressure_all))
P_MAX = float(np.max(pressure_all))

P_STEP = 0.07

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step (fixed): {P_STEP}")



## === cell 3
y = train_fe["pressure"].astype("float32").to_numpy()

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]
X = train_fe.drop(columns=[c for c in drop_cols if c in train_fe.columns])
X_test = test_fe.drop(columns=[c for c in drop_cols if c in test_fe.columns])

print("X:", X.shape, "X_test:", X_test.shape)

u_out_train = train_fe["u_out"].astype(np.int8).to_numpy(copy=False)
t_train = train_fe["time_step"].astype("float32").to_numpy(copy=False)
t_test = test_fe["time_step"].astype("float32").to_numpy(copy=False)

del train_fe, test_fe, pressure_all
gc.collect()



## === cell 4
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float32)
X_test_scaled = np.ascontiguousarray(X_test_scaled, dtype=np.float32)

del X, X_test
gc.collect()

print("Scaled:", X_scaled.shape, X_test_scaled.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/131490532.py in <cell line: 0>()
      2 
      3 X_scaled = scaler.fit_transform(X)
----> 4 X_test_scaled = scaler.transform(X_test)
      5 
      6 X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float32)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in transform(self, X)
   1575         """
   1576         check_is_fitted(self)
-> 1577         X = self._validate_data(
   1578             X,
   1579             accept_sparse=("csr", "csc"),

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/daal4py/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    132         elif dt == np.float32:
    133             if not d4p.daal_assert_all_finite(x_for_daal, allow_nan, 1):
--> 134                 raise ValueError(err)
    135     # First try an O(n) time, O(1) space solution for the common case that
    136     # everything is finite; fall back to O(n) space np.isfinite to prevent

ValueError: Input X contains infinity or a value too large for dtype('float32').

## === cell 5
t_unique = np.sort(np.unique(t_train))
test_t_unique = np.sort(np.unique(t_test))
print("Unique time_steps in train:", len(t_unique), "in test:", len(test_t_unique))

test_pred = np.zeros(X_test_scaled.shape[0], dtype=np.float32)
insp_mask_all = u_out_train == 0

t_unique_used = t_unique[np.isin(t_unique, test_t_unique, assume_unique=True)]

train_time_code = np.searchsorted(t_unique_used, t_train)
test_time_code = np.searchsorted(test_t_unique, t_test)


def build_code_slices(code: np.ndarray, n_codes: int):
    order = np.argsort(code, kind="mergesort")
    code_sorted = code[order]
    counts = np.bincount(code_sorted, minlength=n_codes)
    ends = np.cumsum(counts)
    starts = ends - counts
    return (
        order.astype(np.int32, copy=False),
        starts.astype(np.int64, copy=False),
        ends.astype(np.int64, copy=False),
    )


train_order, train_start, train_end = build_code_slices(
    train_time_code, len(t_unique_used)
)
test_order, test_start, test_end = build_code_slices(test_time_code, len(test_t_unique))

params = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_depth=6,
    max_iter=250,
    random_state=42,
)

pos_in_test_unique = np.searchsorted(test_t_unique, t_unique_used)

_Xs = X_scaled
_Xt = X_test_scaled
_y = y
_to = train_order
_ts = train_start
_te = train_end
_tto = test_order
_tts = test_start
_tte = test_end

insp_in_sorted = insp_mask_all[_to]

for i, (tpos, tr_s, tr_e) in enumerate(zip(pos_in_test_unique, _ts, _te)):
    te_s = _tts[tpos]
    te_e = _tte[tpos]
    if te_e <= te_s or tr_e <= tr_s:
        continue

    te_idx = _tto[te_s:te_e]
    tr_block = _to[tr_s:tr_e]
    if tr_block.size == 0:
        continue

    tr_insp_block = insp_in_sorted[tr_s:tr_e]
    if tr_insp_block.any():
        tr_idx = tr_block[tr_insp_block]
    else:
        tr_idx = tr_block

    model = HistGradientBoostingRegressor(**params)
    model.fit(_Xs[tr_idx], _y[tr_idx])
    test_pred[te_idx] = model.predict(_Xt[te_idx]).astype(np.float32, copy=False)

    if (i + 1) % 10 == 0:
        gc.collect()
        print(f"Trained {i+1}/{len(t_unique_used)} time_step models...")

del X_scaled, X_test_scaled, y, u_out_train, t_train, t_test
del train_order, train_start, train_end, test_order, test_start, test_end
del train_time_code, test_time_code, insp_in_sorted
gc.collect()

print("Pred shape:", test_pred.shape, "test_ids:", test_ids.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1333714576.py in <cell line: 0>()
      7 print("Unique time_steps in train:", len(t_unique), "in test:", len(test_t_unique))
      8 
----> 9 test_pred = np.zeros(X_test_scaled.shape[0], dtype=np.float32)
     10 insp_mask_all = u_out_train == 0
     11 

NameError: name 'X_test_scaled' is not defined

## === cell 6
submission = sample_sub.copy()
submission["id"] = test_ids

submission["pressure"] = np.round((test_pred - P_MIN) / P_STEP) * P_STEP + P_MIN
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission = submission[["id", "pressure"]]
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: median_submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/865671752.py in <cell line: 0>()
      2 submission["id"] = test_ids
      3 
----> 4 submission["pressure"] = np.round((test_pred - P_MIN) / P_STEP) * P_STEP + P_MIN
      5 submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)
      6 

NameError: name 'test_pred' is not defined
