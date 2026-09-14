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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
test = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)
sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": np.int32, "pressure": np.float32})

assert "id" in test.columns and "id" in sub.columns
assert len(test) == len(sub)




## === cell 1
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    u_in = out["u_in"].to_numpy(np.float32, copy=False)
    u_out = out["u_out"].to_numpy(np.float32, copy=False)
    t = out["time_step"].to_numpy(np.float32, copy=False)

    n = len(out)
    if n % 80 != 0:
        raise ValueError(
            "Expected number of rows to be a multiple of 80 (one breath = 80 rows)."
        )

    u_in_2d = u_in.reshape(-1, 80)
    u_out_2d = u_out.reshape(-1, 80)
    t_2d = t.reshape(-1, 80)

    u_in_lag1_2d = np.zeros_like(u_in_2d, dtype=np.float32)
    u_in_lag2_2d = np.zeros_like(u_in_2d, dtype=np.float32)
    u_out_lag1_2d = np.zeros_like(u_out_2d, dtype=np.float32)

    u_in_lag1_2d[:, 1:] = u_in_2d[:, :-1]
    u_in_lag2_2d[:, 2:] = u_in_2d[:, :-2]
    u_out_lag1_2d[:, 1:] = u_out_2d[:, :-1]

    u_in_lag1 = u_in_lag1_2d.reshape(-1)
    u_in_lag2 = u_in_lag2_2d.reshape(-1)
    u_out_lag1 = u_out_lag1_2d.reshape(-1)

    out["u_in_lag1"] = u_in_lag1
    out["u_in_lag2"] = u_in_lag2
    out["u_out_lag1"] = u_out_lag1

    out["u_in_diff1"] = (u_in - u_in_lag1).astype(np.float32, copy=False)
    out["u_out_diff1"] = (u_out - u_out_lag1).astype(np.float32, copy=False)

    out["u_in_cumsum"] = np.cumsum(u_in_2d, axis=1, dtype=np.float32).reshape(-1)

    dt = np.zeros_like(t_2d, dtype=np.float32)
    dt[:, 1:] = t_2d[:, 1:] - t_2d[:, :-1]
    out["dt"] = dt.reshape(-1)
    out["t_cumsum"] = np.cumsum(t_2d, axis=1, dtype=np.float32).reshape(-1)

    return out


train_fe = add_breath_features(train)
test_fe = add_breath_features(test)



## === cell 2
base_features = ["R", "C", "time_step", "u_in", "u_out"]
extra_features = [
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_in_diff1",
    "u_out_diff1",
    "u_in_cumsum",
    "dt",
    "t_cumsum",
]
features = base_features + extra_features
target = "pressure"

train_fe["t_idx"] = np.tile(np.arange(80, dtype=np.int16), len(train_fe) // 80)
test_fe["t_idx"] = np.tile(np.arange(80, dtype=np.int16), len(test_fe) // 80)

u_in_bin_width = 1.0
train_fe["u_in_bin"] = np.floor(
    train_fe["u_in"].to_numpy(np.float32, copy=False) / u_in_bin_width
).astype(np.int16)
test_fe["u_in_bin"] = np.floor(
    test_fe["u_in"].to_numpy(np.float32, copy=False) / u_in_bin_width
).astype(np.int16)

rs = 42

max_breaths = 12000
unique_breaths = train_fe["breath_id"].unique()
if len(unique_breaths) > max_breaths:
    rng = np.random.RandomState(rs)
    keep_breaths = rng.choice(unique_breaths, size=max_breaths, replace=False)
    mask_keep = np.isin(train_fe["breath_id"].to_numpy(copy=False), keep_breaths)
    train_fit_df = train_fe.loc[mask_keep].copy()
else:
    train_fit_df = train_fe.copy()

breaths_fit = pd.unique(train_fit_df["breath_id"].to_numpy(copy=False))
b_tr, b_va = train_test_split(breaths_fit, test_size=0.1, random_state=rs)

breath_arr = train_fit_df["breath_id"].to_numpy(copy=False)
tr_mask = np.isin(breath_arr, b_tr)
va_mask = np.isin(breath_arr, b_va)

train_tr = train_fit_df.loc[tr_mask].copy()
train_va = train_fit_df.loc[va_mask].copy()

train_tr_fit = train_tr.loc[train_tr["u_out"].to_numpy(copy=False) == 0].copy()

global_mean_pressure_tr = float(train_tr_fit[target].mean())


def _make_key_rc_tidx(Rv, Cv, t_idxv):
    return (
        (Rv.astype(np.int32) * 10000)
        + (Cv.astype(np.int32) * 100)
        + t_idxv.astype(np.int32)
    )


def _make_key_rc_tidx_uin(Rv, Cv, t_idxv, uinbinv):
    return (
        (Rv.astype(np.int32) * 10000000)
        + (Cv.astype(np.int32) * 100000)
        + (t_idxv.astype(np.int32) * 1000)
        + uinbinv.astype(np.int32)
    )


def _add_prior_mean_rc_tidx(
    src_df: pd.DataFrame, dst_df: pd.DataFrame, dst_name: str, fill_value: float
):
    g = (
        src_df.groupby(["R", "C", "t_idx"], sort=False, observed=True)[target]
        .mean()
        .reset_index()
    )
    keys_g = _make_key_rc_tidx(
        g["R"].to_numpy(np.int16, copy=False),
        g["C"].to_numpy(np.int16, copy=False),
        g["t_idx"].to_numpy(np.int16, copy=False),
    )
    vals_g = g[target].to_numpy(np.float32, copy=False)
    order = np.argsort(keys_g, kind="mergesort")
    keys_sorted = keys_g[order]
    vals_sorted = vals_g[order]

    keys_dst = _make_key_rc_tidx(
        dst_df["R"].to_numpy(copy=False),
        dst_df["C"].to_numpy(copy=False),
        dst_df["t_idx"].to_numpy(copy=False),
    )
    pos = np.searchsorted(keys_sorted, keys_dst)

    out = np.empty(keys_dst.shape[0], dtype=np.float32)
    out[:] = np.float32(fill_value)

    valid = pos < keys_sorted.size
    if np.any(valid):
        posv = pos[valid]
        keysv = keys_dst[valid]
        hitv = keys_sorted[posv] == keysv
        if np.any(hitv):
            out_idx = np.flatnonzero(valid)[hitv]
            out[out_idx] = vals_sorted[posv[hitv]]

    dst_df[dst_name] = out


def _add_prior_mean_rc_tidx_uin(
    src_df: pd.DataFrame, dst_df: pd.DataFrame, dst_name: str
):
    g = (
        src_df.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False, observed=True)[
            target
        ]
        .mean()
        .reset_index()
    )
    keys_g = _make_key_rc_tidx_uin(
        g["R"].to_numpy(np.int16, copy=False),
        g["C"].to_numpy(np.int16, copy=False),
        g["t_idx"].to_numpy(np.int16, copy=False),
        g["u_in_bin"].to_numpy(np.int16, copy=False),
    )
    vals_g = g[target].to_numpy(np.float32, copy=False)
    order = np.argsort(keys_g, kind="mergesort")
    keys_sorted = keys_g[order]
    vals_sorted = vals_g[order]

    keys_dst = _make_key_rc_tidx_uin(
        dst_df["R"].to_numpy(copy=False),
        dst_df["C"].to_numpy(copy=False),
        dst_df["t_idx"].to_numpy(copy=False),
        dst_df["u_in_bin"].to_numpy(copy=False),
    )
    pos = np.searchsorted(keys_sorted, keys_dst)

    out = np.full(keys_dst.shape[0], np.nan, dtype=np.float32)
    valid = pos < keys_sorted.size
    if np.any(valid):
        posv = pos[valid]
        keysv = keys_dst[valid]
        hitv = keys_sorted[posv] == keysv
        if np.any(hitv):
            out_idx = np.flatnonzero(valid)[hitv]
            out[out_idx] = vals_sorted[posv[hitv]]

    dst_df[dst_name] = out


_add_prior_mean_rc_tidx(
    train_tr_fit, train_tr_fit, "pressure_rc_tidx_mean", global_mean_pressure_tr
)
_add_prior_mean_rc_tidx(
    train_tr_fit, train_va, "pressure_rc_tidx_mean", global_mean_pressure_tr
)

insp_full = train_fe.loc[train_fe["u_out"].to_numpy(copy=False) == 0]
insp_mean_full = float(insp_full[target].mean())
_add_prior_mean_rc_tidx(insp_full, test_fe, "pressure_rc_tidx_mean", insp_mean_full)

_add_prior_mean_rc_tidx_uin(train_tr_fit, train_tr_fit, "pressure_rc_tidx_uinbin_mean")
_add_prior_mean_rc_tidx_uin(train_tr_fit, train_va, "pressure_rc_tidx_uinbin_mean")
_add_prior_mean_rc_tidx_uin(insp_full, test_fe, "pressure_rc_tidx_uinbin_mean")

tr_p1 = train_tr_fit["pressure_rc_tidx_mean"].to_numpy(np.float32, copy=False)
va_p1 = train_va["pressure_rc_tidx_mean"].to_numpy(np.float32, copy=False)
te_p1 = test_fe["pressure_rc_tidx_mean"].to_numpy(np.float32, copy=False)

tr_p2 = train_tr_fit["pressure_rc_tidx_uinbin_mean"].to_numpy(np.float32, copy=False)
va_p2 = train_va["pressure_rc_tidx_uinbin_mean"].to_numpy(np.float32, copy=False)
te_p2 = test_fe["pressure_rc_tidx_uinbin_mean"].to_numpy(np.float32, copy=False)

train_tr_fit["pressure_rc_tidx_uinbin_mean"] = np.where(
    np.isfinite(tr_p2),
    tr_p2,
    np.where(np.isfinite(tr_p1), tr_p1, np.float32(global_mean_pressure_tr)),
).astype(np.float32, copy=False)
train_va["pressure_rc_tidx_uinbin_mean"] = np.where(
    np.isfinite(va_p2),
    va_p2,
    np.where(np.isfinite(va_p1), va_p1, np.float32(global_mean_pressure_tr)),
).astype(np.float32, copy=False)
test_fe["pressure_rc_tidx_uinbin_mean"] = np.where(
    np.isfinite(te_p2),
    te_p2,
    np.where(np.isfinite(te_p1), te_p1, np.float32(insp_mean_full)),
).astype(np.float32, copy=False)

features_plus = features + [
    "t_idx",
    "pressure_rc_tidx_mean",
    "pressure_rc_tidx_uinbin_mean",
    "u_in_bin",
]

R_levels = np.array([5, 20, 50], dtype=np.int16)
C_levels = np.array([10, 20, 50], dtype=np.int16)

_num_cols = [c for c in features_plus if c not in ("R", "C")]


def build_X(df: pd.DataFrame) -> np.ndarray:
    X_num = df[_num_cols].to_numpy(dtype=np.float32, copy=False)

    Rv = df["R"].to_numpy(np.int16, copy=False)
    Cv = df["C"].to_numpy(np.int16, copy=False)

    n = X_num.shape[0]
    X = np.empty((n, X_num.shape[1] + 6), dtype=np.float32)
    X[:, : X_num.shape[1]] = X_num

    X[:, X_num.shape[1] : X_num.shape[1] + 3] = (
        Rv[:, None] == R_levels[None, :]
    ).astype(np.float32)
    X[:, X_num.shape[1] + 3 :] = (Cv[:, None] == C_levels[None, :]).astype(np.float32)
    return X


X_tr = build_X(train_tr_fit)
X_va = build_X(train_va)
X_test = build_X(test_fe)

y_tr = train_tr_fit[target].to_numpy(dtype=np.float32, copy=False)
y_va = train_va[target].to_numpy(dtype=np.float32, copy=False)



## === cell 3
model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(n_neighbors=45, weights="distance", p=1, n_jobs=-1),
        ),
    ]
)

model.fit(X_tr, y_tr)

va_pred = model.predict(X_va).astype(np.float32, copy=False)
va_u_out = train_va["u_out"].to_numpy(copy=False)
insp_mask_va = va_u_out == 0

va_mae = np.mean(
    np.abs(va_pred[insp_mask_va] - y_va[insp_mask_va].astype(np.float32, copy=False))
)
print(
    "Validation MAE (breath_id split; trained on u_out==0 only; evaluated on u_out==0; R/C one-hot; +priors):",
    float(va_mae),
)



## === cell 4
test_pred = model.predict(X_test).astype(np.float32, copy=False)

ids = test_fe["id"].to_numpy(np.int32, copy=False)
u_out = test_fe["u_out"].to_numpy(np.int8, copy=False)

pred2d = test_pred.reshape(-1, 80).copy()  # keep copy because we overwrite
uout2d = u_out.reshape(-1, 80)

insp_mask2d = uout2d == 0
last_insp_idx = np.where(
    insp_mask2d.any(axis=1),
    insp_mask2d.shape[1] - 1 - insp_mask2d[:, ::-1].argmax(axis=1),
    0,
).astype(np.int64)

rows = np.arange(pred2d.shape[0], dtype=np.int64)
last_vals = pred2d[rows, last_insp_idx]
pred2d = np.where(insp_mask2d, pred2d, last_vals[:, None]).astype(
    np.float32, copy=False
)
pred_filled = pred2d.reshape(-1)

pressure_levels = np.sort(train["pressure"].unique().astype(np.float32))

is_insp = u_out == 0
pred_all = pred_filled.astype(np.float32, copy=False)

pred_snapped_all = pred_all.copy()
pred_insp = pred_all[is_insp]

idx = np.searchsorted(pressure_levels, pred_insp, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_levels) - 1)

right = pressure_levels[idx]
left = pressure_levels[idx_left]
choose_left = np.abs(pred_insp - left) <= np.abs(pred_insp - right)
pred_snapped_insp = np.where(choose_left, left, right).astype(np.float32)

pred_snapped_all[is_insp] = pred_snapped_insp

sub_sorted = sub.sort_values("id").reset_index(drop=True)
test_ids_sorted = np.sort(ids, kind="mergesort")
assert sub_sorted["id"].to_numpy().shape == test_ids_sorted.shape
assert np.array_equal(sub_sorted["id"].to_numpy(), test_ids_sorted)

order = np.argsort(ids, kind="mergesort")
pred_by_id = pred_snapped_all[order].astype(np.float32, copy=False)

sub_sorted["pressure"] = pred_by_id
sub_sorted.to_csv("submission.csv", index=False)

sub_sorted.head()
