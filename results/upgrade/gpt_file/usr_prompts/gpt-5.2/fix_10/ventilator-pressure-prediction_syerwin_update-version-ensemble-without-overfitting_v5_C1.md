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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    breath = df["breath_id"]

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    g_breath_time_uin = (df["time_step"] * df["u_in"]).groupby(breath, sort=False)
    df["area"] = g_breath_time_uin.cumsum()

    g_time = df["time_step"].groupby(breath, sort=False)
    g_uin = df["u_in"].groupby(breath, sort=False)
    g_uout = df["u_out"].groupby(breath, sort=False)

    df["time_step_cumsum"] = g_time.cumsum()
    df["u_in_cumsum"] = g_uin.cumsum()

    df["u_in_lag1"] = g_uin.shift(1)
    df["u_out_lag1"] = g_uout.shift(1)
    df["u_in_lag_back1"] = g_uin.shift(-1)
    df["u_out_lag_back1"] = g_uout.shift(-1)

    df["u_in_lag2"] = g_uin.shift(2)
    df["u_out_lag2"] = g_uout.shift(2)
    df["u_in_lag_back2"] = g_uin.shift(-2)
    df["u_out_lag_back2"] = g_uout.shift(-2)

    df["u_in_lag3"] = g_uin.shift(3)
    df["u_out_lag3"] = g_uout.shift(3)
    df["u_in_lag_back3"] = g_uin.shift(-3)
    df["u_out_lag_back3"] = g_uout.shift(-3)

    df["u_in_lag4"] = g_uin.shift(4)
    df["u_out_lag4"] = g_uout.shift(4)
    df["u_in_lag_back4"] = g_uin.shift(-4)
    df["u_out_lag_back4"] = g_uout.shift(-4)

    df = df.fillna(0)

    u_in_max = g_uin.transform("max")
    u_in_mean = g_uin.transform("mean")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = u_in_max - df["u_in"]
    df["breath_id__u_in__diffmean"] = u_in_mean - df["u_in"]

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]

    df["one"] = 1
    df["count"] = df["one"].groupby(breath, sort=False).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = (df["breath_id_lag"] == df["breath_id"]).astype(np.int8)
    df["breath_id_lag2same"] = (df["breath_id_lag2"] == df["breath_id"]).astype(np.int8)

    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = (
        df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    )

    df["time_step_diff"] = g_time.diff().fillna(0)

    df["ewm_u_in_mean"] = (
        df["u_in"]
        .groupby(breath, sort=False)
        .ewm(halflife=9)
        .mean()
        .reset_index(level=0, drop=True)
    )

    u = df["u_in"].to_numpy(copy=False)
    b = df["breath_id"].to_numpy(copy=False)
    s15, mn15, mx15, mean15 = _rolling15_stats_per_breath_fast(u, b, window=15)
    df["15_in_sum"] = s15
    df["15_in_min"] = mn15
    df["15_in_max"] = mx15
    df["15_in_mean"] = mean15

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]

    df["R"] = pd.Categorical(
        df["R"], categories=[5, 20, 50], ordered=True
    ).codes.astype(np.int8, copy=False)
    df["C"] = pd.Categorical(
        df["C"], categories=[10, 20, 50], ordered=True
    ).codes.astype(np.int8, copy=False)
    r_code = df["R"].to_numpy(copy=False)
    c_code = df["C"].to_numpy(copy=False)
    df["R__C"] = _RC_TABLE[r_code, c_code]

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

u_out_train = train_fe["u_out"].astype(np.int8).to_numpy()
t_train = train_fe["time_step"].astype("float32").to_numpy()
t_test = test_fe["time_step"].astype("float32").to_numpy()

del train_fe, test_fe, pressure_all
gc.collect()




## === cell 4
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X).astype(np.float32, copy=False)
X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)

X_scaled = np.ascontiguousarray(X_scaled)
X_test_scaled = np.ascontiguousarray(X_test_scaled)

del X, X_test
gc.collect()

print("Scaled:", X_scaled.shape, X_test_scaled.shape)




## === cell 5
t_unique = np.sort(np.unique(t_train))
test_t_unique = np.sort(np.unique(t_test))
print("Unique time_steps in train:", len(t_unique), "in test:", len(test_t_unique))

test_pred = np.zeros(X_test_scaled.shape[0], dtype=np.float32)
insp_mask_all = u_out_train == 0

t_unique_used = t_unique[np.isin(t_unique, test_t_unique)]

train_time_code = np.searchsorted(t_unique_used, t_train)
test_time_code = np.searchsorted(test_t_unique, t_test)


def build_code_slices(code: np.ndarray, n_codes: int):
    order = np.argsort(code, kind="mergesort")
    code_sorted = code[order]
    counts = np.bincount(code_sorted, minlength=n_codes)
    ends = np.cumsum(counts)
    starts = ends - counts
    return order, starts, ends


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

for i, (tpos, tr_s, tr_e) in enumerate(zip(pos_in_test_unique, train_start, train_end)):
    te_s = test_start[tpos]
    te_e = test_end[tpos]
    if te_e <= te_s or tr_e <= tr_s:
        continue

    te_idx = test_order[te_s:te_e]
    all_tr_idx = train_order[tr_s:tr_e]
    if all_tr_idx.size == 0:
        continue

    tr_mask = insp_mask_all[all_tr_idx]
    tr_idx = all_tr_idx[tr_mask]
    if tr_idx.size == 0:
        tr_idx = all_tr_idx

    model = HistGradientBoostingRegressor(**params)
    model.fit(X_scaled[tr_idx], y[tr_idx])
    test_pred[te_idx] = model.predict(X_test_scaled[te_idx]).astype(
        np.float32, copy=False
    )

    if (i + 1) % 10 == 0:
        gc.collect()
        print(f"Trained {i+1}/{len(t_unique_used)} time_step models...")

del X_scaled, X_test_scaled, y, u_out_train, t_train, t_test
del train_order, train_start, train_end, test_order, test_start, test_end
del train_time_code, test_time_code
gc.collect()

print("Pred shape:", test_pred.shape, "test_ids:", test_ids.shape)




## === cell 6
submission = sample_sub.copy()
submission["id"] = test_ids

submission["pressure"] = np.round((test_pred - P_MIN) / P_STEP) * P_STEP + P_MIN
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission = submission[["id", "pressure"]]
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: median_submission.csv")
