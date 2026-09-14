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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor

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


def _diff_within_breath_vec(x: np.ndarray, starts: np.ndarray):
    out = np.empty_like(x, dtype=np.float32)
    out[0] = 0.0
    out[1:] = x[1:] - x[:-1]
    out = out.astype(np.float32, copy=False)
    out[starts] = 0.0
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=False)

    breath_id = df["breath_id"].to_numpy(copy=False)
    n = breath_id.shape[0]

    changes = np.empty(n, dtype=bool)
    changes[0] = True
    changes[1:] = breath_id[1:] != breath_id[:-1]
    starts = np.flatnonzero(changes).astype(np.int64, copy=False)

    time_step = df["time_step"].to_numpy(copy=False).astype(np.float32, copy=False)
    u_in = df["u_in"].to_numpy(copy=False).astype(np.float32, copy=False)
    u_out = df["u_out"].to_numpy(copy=False).astype(np.int8, copy=False)

    df["cross"] = (u_in * u_out).astype(np.float32, copy=False)
    df["cross2"] = (time_step * u_out).astype(np.float32, copy=False)

    gb = df.groupby("breath_id", sort=False)
    df["area"] = gb.apply(
        lambda x: (
            x["time_step"].astype("float32") * x["u_in"].astype("float32")
        ).cumsum()
    ).to_numpy(dtype=np.float32, copy=False)
    df["time_step_cumsum"] = (
        gb["time_step"].cumsum().to_numpy(dtype=np.float32, copy=False)
    )
    df["u_in_cumsum"] = gb["u_in"].cumsum().to_numpy(dtype=np.float32, copy=False)

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

    u_in_max = gb["u_in"].transform("max").to_numpy(dtype=np.float32, copy=False)
    u_in_mean = gb["u_in"].transform("mean").to_numpy(dtype=np.float32, copy=False)
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

    count = (gb.cumcount() + 1).to_numpy(dtype=np.int16, copy=False)
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

    df["ewm_u_in_mean"] = (
        gb["u_in"]
        .ewm(halflife=9, adjust=True)
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )

    roll = gb["u_in"].rolling(window=15, min_periods=1)
    df["15_in_sum"] = (
        roll.sum()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )
    df["15_in_min"] = (
        roll.min()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )
    df["15_in_max"] = (
        roll.max()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )
    df["15_in_mean"] = (
        roll.mean()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float32, copy=False)
    )

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
    R_code = np.empty_like(R_raw, dtype=np.int8)
    C_code = np.empty_like(C_raw, dtype=np.int8)
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

drop_cols_train = [c for c in drop_cols if c in train_fe.columns]
drop_cols_test = [c for c in drop_cols if c in test_fe.columns]

X = train_fe.drop(columns=drop_cols_train)
X_test = test_fe.drop(columns=drop_cols_test)

X_test = X_test.reindex(columns=X.columns, fill_value=0.0)

print("X:", X.shape, "X_test:", X_test.shape)

u_out_train = train_fe["u_out"].astype(np.int8).to_numpy(copy=False)
t_train = train_fe["time_step"].astype("float32").to_numpy(copy=False)
t_test = test_fe["time_step"].astype("float32").to_numpy(copy=False)

del train_fe, test_fe, pressure_all
gc.collect()



## === cell 4
X_np = X.to_numpy(dtype=np.float32, copy=False)
X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)

np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X_np)
X_test_scaled = scaler.transform(X_test_np)

X_scaled = np.asarray(X_scaled, dtype=np.float32, order="C")
X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32, order="C")

np.nan_to_num(X_scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test_scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

del X, X_test, X_np, X_test_np
gc.collect()

print("Scaled:", X_scaled.shape, X_test_scaled.shape)



## === cell 5
t_unique = np.sort(np.unique(t_train))
test_t_unique = np.sort(np.unique(t_test))
print("Unique time_steps in train:", len(t_unique), "in test:", len(test_t_unique))

test_pred = np.zeros(X_test_scaled.shape[0], dtype=np.float32)
insp_mask_all = u_out_train == 0

t_unique_used = t_unique[np.isin(t_unique, test_t_unique, assume_unique=True)]

train_time_code = np.searchsorted(t_unique_used, t_train).astype(np.int32, copy=False)
test_time_code = np.searchsorted(test_t_unique, t_test).astype(np.int32, copy=False)


def build_code_slices(code: np.ndarray, n_codes: int):
    order = np.argsort(code, kind="mergesort")  # stable and deterministic
    code_sorted = code[order]
    counts = np.bincount(code_sorted, minlength=n_codes)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts.astype(np.int64, copy=False)
    return order.astype(np.int32, copy=False), starts, ends


train_order, train_start, train_end = build_code_slices(
    train_time_code, int(len(t_unique_used))
)
test_order, test_start, test_end = build_code_slices(
    test_time_code, int(len(test_t_unique))
)

params = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_depth=6,
    max_iter=250,
    random_state=42,
)

pos_in_test_unique = np.searchsorted(test_t_unique, t_unique_used).astype(
    np.int32, copy=False
)

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

    if (i + 1) % 25 == 0:
        gc.collect()
        print(f"Trained {i+1}/{len(t_unique_used)} time_step models...")

del X_scaled, X_test_scaled, y, u_out_train, t_train, t_test
del train_order, train_start, train_end, test_order, test_start, test_end
del train_time_code, test_time_code, insp_in_sorted
gc.collect()

print("Pred shape:", test_pred.shape, "test_ids:", test_ids.shape)



## === cell 6
if "test_pred" not in globals():
    test_pred = np.zeros(len(test_ids), dtype=np.float32)
np.nan_to_num(test_pred, copy=False, nan=P_MIN, posinf=P_MAX, neginf=P_MIN)

submission = sample_sub.copy()
submission["id"] = test_ids

submission["pressure"] = np.round((test_pred - P_MIN) / P_STEP) * P_STEP + P_MIN
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission = submission[["id", "pressure"]]
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: median_submission.csv")
print("Submission shape:", submission.shape)
