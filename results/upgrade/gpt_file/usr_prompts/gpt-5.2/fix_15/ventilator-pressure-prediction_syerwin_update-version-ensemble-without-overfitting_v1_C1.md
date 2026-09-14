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

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
np.random.seed(42)




## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

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

train_df = pd.read_csv(train_path, dtype=train_dtypes, engine="c")
test_df = pd.read_csv(test_path, dtype=test_dtypes, engine="c")
submission = pd.read_csv(
    sub_path, dtype={"id": "int32", "pressure": "float32"}, engine="c"
)

print(train_df.shape, test_df.shape, submission.shape)
print(train_df.columns.tolist())




## === cell 2
def _ensure_sorted_by_breath_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0:
        return df
    return df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )


train_df = _ensure_sorted_by_breath_time(train_df)
test_df = _ensure_sorted_by_breath_time(test_df)




## === cell 3
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    n = df.shape[0]
    if n == 0:
        return df.copy()

    bid = df["breath_id"].to_numpy(copy=False)
    if n % 80 != 0:
        raise ValueError(
            "Unexpected row count not divisible by 80; cannot use fast path safely."
        )
    n_breaths = n // 80

    bid2 = bid.reshape(n_breaths, 80)
    if not np.all(bid2[:, 0] == bid2[:, -1]):
        raise ValueError(
            "Breaths are not contiguous after sorting; cannot compute per-breath features."
        )

    u_in = (
        df["u_in"]
        .to_numpy(copy=False)
        .reshape(n_breaths, 80)
        .astype(np.float32, copy=False)
    )
    u_out = (
        df["u_out"]
        .to_numpy(copy=False)
        .reshape(n_breaths, 80)
        .astype(np.float32, copy=False)
    )
    t = (
        df["time_step"]
        .to_numpy(copy=False)
        .reshape(n_breaths, 80)
        .astype(np.float32, copy=False)
    )

    cross = (df["u_in"].to_numpy(copy=False) * df["u_out"].to_numpy(copy=False)).astype(
        np.float32, copy=False
    )
    cross2 = (
        df["time_step"].to_numpy(copy=False) * df["u_out"].to_numpy(copy=False)
    ).astype(np.float32, copy=False)

    area = np.cumsum(t * u_in, axis=1, dtype=np.float32)
    time_step_cumsum = np.cumsum(t, axis=1, dtype=np.float32)
    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)

    def lag(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        if k:
            out[:, k:] = arr[:, :-k]
        return out

    def lag_back(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        if k:
            out[:, :-k] = arr[:, k:]
        return out

    u_in_lag1, u_in_lag2, u_in_lag3, u_in_lag4 = (
        lag(u_in, 1),
        lag(u_in, 2),
        lag(u_in, 3),
        lag(u_in, 4),
    )
    u_out_lag1, u_out_lag2, u_out_lag3, u_out_lag4 = (
        lag(u_out, 1),
        lag(u_out, 2),
        lag(u_out, 3),
        lag(u_out, 4),
    )

    u_in_lag_back1, u_in_lag_back2, u_in_lag_back3, u_in_lag_back4 = (
        lag_back(u_in, 1),
        lag_back(u_in, 2),
        lag_back(u_in, 3),
        lag_back(u_in, 4),
    )
    u_out_lag_back1, u_out_lag_back2, u_out_lag_back3, u_out_lag_back4 = (
        lag_back(u_out, 1),
        lag_back(u_out, 2),
        lag_back(u_out, 3),
        lag_back(u_out, 4),
    )

    u_in_max = np.max(u_in, axis=1, keepdims=True).astype(np.float32, copy=False)
    u_in_mean = np.mean(u_in, axis=1, keepdims=True).astype(np.float32, copy=False)
    diffmax = u_in_max - u_in
    diffmean = u_in_mean - u_in

    u_in_diff1, u_in_diff2, u_in_diff3, u_in_diff4 = (
        u_in - u_in_lag1,
        u_in - u_in_lag2,
        u_in - u_in_lag3,
        u_in - u_in_lag4,
    )
    u_out_diff1, u_out_diff2, u_out_diff3, u_out_diff4 = (
        u_out - u_out_lag1,
        u_out - u_out_lag2,
        u_out - u_out_lag3,
        u_out - u_out_lag4,
    )

    count = (np.arange(1, 81, dtype=np.float32))[None, :]
    u_in_cummean = u_in_cumsum / count

    time_step_diff = np.zeros_like(t, dtype=np.float32)
    time_step_diff[:, 1:] = t[:, 1:] - t[:, :-1]

    alpha = np.float32(1.0 - np.exp(np.log(0.5) / 9.0))
    ewm = np.empty_like(u_in, dtype=np.float32)
    ewm[:, 0] = u_in[:, 0]
    one_minus_alpha = np.float32(1.0) - alpha
    for j in range(1, 80):
        ewm[:, j] = alpha * u_in[:, j] + one_minus_alpha * ewm[:, j - 1]

    pad = 14
    u_pad = np.pad(
        u_in, ((0, 0), (pad, 0)), mode="constant", constant_values=0.0
    ).astype(
        np.float32, copy=False
    )  # (n_breaths, 94)
    csum = np.cumsum(u_pad, axis=1, dtype=np.float32)

    j = np.arange(80, dtype=np.int32)
    L = np.minimum(15, j + 1).astype(np.int32)  # (80,)
    end = j + pad
    start = end - L + 1
    startm1 = start - 1
    roll_sum = (
        csum[:, end] - np.where(startm1[None, :] >= 0, csum[:, startm1], 0.0)
    ).astype(np.float32, copy=False)
    roll_mean = roll_sum / L.astype(np.float32)[None, :]

    win = 15
    sw = np.lib.stride_tricks.sliding_window_view(u_pad, window_shape=win, axis=1)
    roll_min = np.empty((n_breaths, 80), dtype=np.float32)
    roll_max = np.empty((n_breaths, 80), dtype=np.float32)

    full_idx = np.arange(14, 80, dtype=np.int32)
    roll_min[:, full_idx] = sw[:, full_idx, :].min(axis=2)
    roll_max[:, full_idx] = sw[:, full_idx, :].max(axis=2)

    for jj in range(14):
        ll = int(L[jj])
        ee = int(end[jj])
        ss = ee - ll + 1
        w = u_pad[:, ss : ee + 1]
        roll_min[:, jj] = w.min(axis=1)
        roll_max[:, jj] = w.max(axis=1)

    u_in_lagback_diff1 = u_in - u_in_lag_back1
    u_out_lagback_diff1 = u_out - u_out_lag_back1
    u_in_lagback_diff2 = u_in - u_in_lag_back2
    u_out_lagback_diff2 = u_out - u_out_lag_back2

    breath_id_lag = np.empty(n, dtype=np.int32)
    breath_id_lag[0] = 0
    breath_id_lag[1:] = bid[:-1].astype(np.int32, copy=False)

    breath_id_lag2 = np.empty(n, dtype=np.int32)
    breath_id_lag2[:2] = 0
    breath_id_lag2[2:] = bid[:-2].astype(np.int32, copy=False)

    breath_id_lagsame = np.tile(np.r_[0, np.ones(79, dtype=np.int8)], n_breaths)
    breath_id_lag2same = np.tile(np.r_[0, 0, np.ones(78, dtype=np.int8)], n_breaths)

    r_cat = df["R"].astype("category")
    c_cat = df["C"].astype("category")
    R_codes = r_cat.cat.codes.astype("int8").to_numpy(copy=False)
    C_codes = c_cat.cat.codes.astype("int8").to_numpy(copy=False)
    RC_codes = (R_codes.astype(np.int16) * 10 + C_codes.astype(np.int16)).astype(
        "int8", copy=False
    )

    out = df.copy(deep=False)

    out["cross"] = cross
    out["cross2"] = cross2

    out["one"] = np.ones(n, dtype=np.int8)
    out["area"] = area.reshape(-1)
    out["time_step_cumsum"] = time_step_cumsum.reshape(-1)
    out["u_in_cumsum"] = u_in_cumsum.reshape(-1)

    out["u_in_lag1"] = u_in_lag1.reshape(-1)
    out["u_out_lag1"] = u_out_lag1.reshape(-1)
    out["u_in_lag_back1"] = u_in_lag_back1.reshape(-1)
    out["u_out_lag_back1"] = u_out_lag_back1.reshape(-1)

    out["u_in_lag2"] = u_in_lag2.reshape(-1)
    out["u_out_lag2"] = u_out_lag2.reshape(-1)
    out["u_in_lag_back2"] = u_in_lag_back2.reshape(-1)
    out["u_out_lag_back2"] = u_out_lag_back2.reshape(-1)

    out["u_in_lag3"] = u_in_lag3.reshape(-1)
    out["u_out_lag3"] = u_out_lag3.reshape(-1)
    out["u_in_lag_back3"] = u_in_lag_back3.reshape(-1)
    out["u_out_lag_back3"] = u_out_lag_back3.reshape(-1)

    out["u_in_lag4"] = u_in_lag4.reshape(-1)
    out["u_out_lag4"] = u_out_lag4.reshape(-1)
    out["u_in_lag_back4"] = u_in_lag_back4.reshape(-1)
    out["u_out_lag_back4"] = u_out_lag_back4.reshape(-1)

    out["breath_id__u_in__max"] = np.repeat(u_in_max.reshape(-1), 80)
    out["breath_id__u_in__mean"] = np.repeat(u_in_mean.reshape(-1), 80)
    out["breath_id__u_in__diffmax"] = diffmax.reshape(-1)
    out["breath_id__u_in__diffmean"] = diffmean.reshape(-1)

    out["u_in_diff1"] = u_in_diff1.reshape(-1)
    out["u_out_diff1"] = u_out_diff1.reshape(-1)
    out["u_in_diff2"] = u_in_diff2.reshape(-1)
    out["u_out_diff2"] = u_out_diff2.reshape(-1)
    out["u_in_diff3"] = u_in_diff3.reshape(-1)
    out["u_out_diff3"] = u_out_diff3.reshape(-1)
    out["u_in_diff4"] = u_in_diff4.reshape(-1)
    out["u_out_diff4"] = u_out_diff4.reshape(-1)

    out["count"] = np.tile(np.arange(1, 81, dtype=np.float32), n_breaths)
    out["u_in_cummean"] = u_in_cummean.reshape(-1)

    out["breath_id_lag"] = breath_id_lag
    out["breath_id_lag2"] = breath_id_lag2
    out["breath_id_lagsame"] = breath_id_lagsame
    out["breath_id_lag2same"] = breath_id_lag2same
    out["breath_id__u_in_lag"] = (u_in_lag1.reshape(-1) * breath_id_lagsame).astype(
        np.float32, copy=False
    )
    out["breath_id__u_in_lag2"] = (u_in_lag2.reshape(-1) * breath_id_lag2same).astype(
        np.float32, copy=False
    )

    out["time_step_diff"] = time_step_diff.reshape(-1)
    out["ewm_u_in_mean"] = ewm.reshape(-1)

    out["15_in_sum"] = roll_sum.reshape(-1)
    out["15_in_min"] = roll_min.reshape(-1)
    out["15_in_max"] = roll_max.reshape(-1)
    out["15_in_mean"] = roll_mean.reshape(-1)

    out["u_in_lagback_diff1"] = u_in_lagback_diff1.reshape(-1)
    out["u_out_lagback_diff1"] = u_out_lagback_diff1.reshape(-1)
    out["u_in_lagback_diff2"] = u_in_lagback_diff2.reshape(-1)
    out["u_out_lagback_diff2"] = u_out_lagback_diff2.reshape(-1)

    out["R"] = R_codes
    out["C"] = C_codes
    out["R__C"] = RC_codes

    return out




## === cell 4
train_feat = add_features(train_df)
test_feat = add_features(test_df)

if "pressure" not in train_feat.columns:
    raise ValueError("Expected 'pressure' column in engineered train features.")

y = train_feat["pressure"].astype(np.float32).to_numpy(copy=False)

feature_cols = [c for c in train_feat.columns if c != "pressure"]
missing_in_test = [c for c in feature_cols if c not in test_feat.columns]
if missing_in_test:
    for c in missing_in_test:
        test_feat[c] = 0.0

X_train = train_feat[feature_cols]
X_test = test_feat[feature_cols]

print("Feature matrix shapes:", X_train.shape, X_test.shape)

test_ids = test_df["id"].to_numpy(copy=False)

del train_df, test_df
gc.collect()




## === cell 5
X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))

del X_train, X_test, train_feat, test_feat
gc.collect()

q25, q50, q75 = np.percentile(X_train_np, [25, 50, 75], axis=0, method="linear").astype(
    np.float32, copy=False
)

scale = (q75 - q25).astype(np.float32, copy=False)
scale_safe = np.where(scale == 0.0, 1.0, scale).astype(np.float32, copy=False)

X_train_s = np.ascontiguousarray(((X_train_np - q50) / scale_safe), dtype=np.float32)
X_test_s = np.ascontiguousarray(((X_test_np - q50) / scale_safe), dtype=np.float32)

del X_train_np, X_test_np, q25, q75, scale, scale_safe
gc.collect()

print("Scaled:", X_train_s.shape, X_test_s.shape)




## === cell 6
unique_p = np.unique(y)
unique_p.sort()
P_MIN = float(unique_p[0])
P_MAX = float(unique_p[-1])
diffs = np.diff(unique_p)
pos_diffs = diffs[diffs > 0]
P_STEP = float(pos_diffs.min()) if pos_diffs.size else 0.0

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step: {P_STEP}")
print(f"Unique values: {unique_p.shape[0]}")




## === cell 7
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PCA_N_COMPONENTS = 32
pca = PCA(n_components=PCA_N_COMPONENTS, svd_solver="randomized", random_state=42)

X_train_s = np.ascontiguousarray(X_train_s, dtype=np.float32)
X_test_s = np.ascontiguousarray(X_test_s, dtype=np.float32)

X_train_knn = np.ascontiguousarray(
    pca.fit_transform(X_train_s).astype(np.float32, copy=False)
)
X_test_knn = np.ascontiguousarray(
    pca.transform(X_test_s).astype(np.float32, copy=False)
)

models = [
    (
        "hgb_1",
        HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        ),
        X_train_s,
        X_test_s,
    ),
    (
        "hgb_2",
        HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=8,
            learning_rate=0.03,
            max_iter=450,
            random_state=43,
        ),
        X_train_s,
        X_test_s,
    ),
    ("ridge", Ridge(alpha=1.0, random_state=42), X_train_s, X_test_s),
    (
        "knn",
        KNeighborsRegressor(n_neighbors=25, weights="distance", p=2, n_jobs=-1),
        X_train_knn,
        X_test_knn,
    ),
]

pred_list = []
for name, model, Xtr, Xte in models:
    print("Fitting:", name)

    if name == "knn":
        n_neighbors = int(model.n_neighbors)
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm="brute",
            metric="minkowski",
            p=float(model.p),
            n_jobs=model.n_jobs,
        )
        nn.fit(Xtr)
        dists, idx = nn.kneighbors(Xte, return_distance=True)
        y_tr = y  # (n_train,)
        neigh_y = y_tr[idx]  # (n_test, k)
        zero_mask = dists == 0.0
        any_zero = zero_mask.any(axis=1)

        w = np.empty_like(dists, dtype=np.float32)
        np.divide(1.0, dists, out=w, where=~zero_mask)
        w[zero_mask] = 0.0
        num = (w * neigh_y).sum(axis=1, dtype=np.float32)
        den = w.sum(axis=1, dtype=np.float32)
        p = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        if np.any(any_zero):
            zsum = (neigh_y * zero_mask).sum(axis=1, dtype=np.float32)
            zcnt = zero_mask.sum(axis=1).astype(np.float32)
            p_zero = zsum / zcnt
            p[any_zero] = p_zero[any_zero]

        pred_list.append(p.astype(np.float32, copy=False))
        del (
            nn,
            dists,
            idx,
            neigh_y,
            zero_mask,
            any_zero,
            w,
            num,
            den,
            zsum,
            zcnt,
            p_zero,
            p,
        )
        gc.collect()
    else:
        model.fit(Xtr, y)
        p = model.predict(Xte).astype(np.float32, copy=False)
        pred_list.append(p)

pred = np.vstack(pred_list)
print("Ensemble pred shape:", pred.shape)

del X_train_s, X_test_s, X_train_knn, X_test_knn
gc.collect()




## === cell 8
if "pred" not in globals():
    raise RuntimeError("pred was not created; training/prediction failed earlier.")
if pred.ndim != 2 or pred.shape[1] != test_ids.shape[0]:
    raise RuntimeError(
        f"Unexpected pred shape {pred.shape} for test_ids {test_ids.shape}."
    )

submission = pd.DataFrame({"id": test_ids.astype(np.int32, copy=False)})
submission["pressure"] = np.median(pred, axis=0)

if P_STEP > 0:
    submission["pressure"] = (
        np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
    )

submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX).astype(
    np.float32
)

submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
