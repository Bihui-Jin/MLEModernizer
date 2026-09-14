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

0.1429385765842324

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.12363) has done: 'I remove the dependency on missing external Kaggle datasets (the ensemble submissions) which is causing the FileNotFoundError, and instead generate predictions directly from the provided train/test data so the notebook runs end-to-end. To keep changes minimal while improving score versus an all-zero submission, I implement a simple, fast KNN regressor using only the original input features (R, C, time_step, u_in, u_out) and train it on a subsample for runtime safety. I also ensure the submission is aligned by `id`, has exactly the required columns, and is written to a `.csv` file. This should produce a valid submission and a non-trivial MAE without changing any unavailable model architecture (the original code was just ensembling submissions).'
- What this solution (achieved 8.55181) has done: 'Your current score is far worse than the target (lower is better), so we should improve accuracy with minimal, metric-relevant changes while preserving the same simple KNN-on-tabular approach. The biggest issue is that a random row split breaks the time-series/breath structure and teaches the model to interpolate across mixed breaths; we instead split by `breath_id` (group split) and train/predict on per-timestep features that include a small amount of causal history within each breath. Concretely, we add lightweight lag and cumulative-sum features for `u_in`/`u_out` (computed per `breath_id`) and keep the same KNN+StandardScaler pipeline. This keeps the core model/training loop intact but typically reduces MAE a lot for this competition, moving your score substantially toward the 0.1429 target.'
- What this solution (achieved 8.53942) has done: 'Your current MAE (8.55) is far worse than the target (0.143, lower is better), so we should improve accuracy with minimal, metric-aligned changes while keeping the same KNN+scaler pipeline. The biggest gap is that the model is learning pressure values that are never scored (expiratory phase), which dilutes training; we can filter training rows to only inspiratory phase (`u_out==0`) to better match the Kaggle evaluation without changing the model or loss. Additionally, ventilator pressures take on a discrete set of values; snapping predictions to the nearest training pressure level is a standard post-process that usually reduces MAE without altering core modeling. These two small changes are fast, preserve your approach, and should move the score substantially toward the target.'
- What this solution (achieved 8.79376) has done: 'Your current MAE (8.54, lower is better) is still far above the target (0.143), so we should improve accuracy with minimal, metric-aligned changes while keeping the same KNN+StandardScaler pipeline and the same engineered lag/cumsum features. The largest remaining issue is that we train only on inspiratory rows (`u_out==0`) but still predict freely on expiratory rows in test; since expiratory rows are not scored and have a very different regime, we can safely set predictions for `u_out==1` to 0 (or another constant) to avoid KNN extrapolation noise without affecting the scored part. Additionally, KNN is very sensitive to feature scaling when `R`/`C` are categorical-like and `u_out` is binary; encoding `R` and `C` as one-hot (rather than treating them as numeric distances) is a small change that usually reduces MAE substantially while preserving the same model family and training approach. Finally, we keep the “snap-to-pressure-levels” post-process, but apply it only on inspiratory predictions to keep expiratory handling deterministic.'
- What this solution (achieved 1.14691) has done: 'Your current MAE (8.79, lower is better) is still far from the 0.143 target, so we need a real accuracy gain while keeping your same KNN+scaler pipeline and the same breath-wise lag/cumsum feature idea. The biggest issue is that KNN on per-row tabular features can’t easily “memorize” the discrete pressure dynamics; a minimal, competition-proven improvement is to add a very lightweight neighbor-based target-encoding: compute the mean pressure per (R,C,time_step index) from training inspiratory rows and use that as an additional numeric feature for both train/test. This preserves your model, training loop, and loss, but gives KNN a strong prior aligned with the data’s structure. I also set expiratory predictions (u_out==1) to the last predicted inspiratory pressure within the same breath (instead of 0), which is still unscored but avoids extreme discontinuities that can hurt public LB in some edge cases due to any scoring quirks.'
- What this solution (achieved 1.21424) has done: 'Your current MAE (1.14691, lower-is-better) is still far above the target (0.14294), so we should improve accuracy with minimal, metric-aligned tweaks while keeping the same KNN+scaler pipeline, the same breath-wise lag/cumsum features, and the same RC+t_idx prior idea. The biggest low-risk gain here is to enforce the known discrete pressure grid more strongly by snapping **all** predictions (not just inspiratory) to the nearest valid pressure level, which often reduces MAE on this competition without changing the model. I also fix the expensive `groupby().apply(...)` fill to a vectorized `where(...).groupby(...).ffill()` (same semantics, faster/stabler within time limits). Finally, I add a second prior feature (mean pressure per `(R,C,t_idx,u_in_binned)`) using the same training data; this preserves your approach (a lightweight mean prior) but gives KNN a better anchor for the strong `u_in` dependence.'
- What this solution (achieved 1.23617) has done: 'Your current MAE (1.214) is still far above the target (0.143, lower is better), so we need a small but meaningful accuracy boost while keeping the same KNN+scaler pipeline and the same feature/prior strategy. The most likely low-risk gain is to (1) snap only inspiratory predictions to the discrete pressure grid (the metric ignores expiratory rows; snapping expiratory can inject unnecessary quantization noise into the filled values), and (2) make the `(R,C,t_idx,u_in_bin)` prior a bit more informative by using a finer `u_in` bin width (more specific mean lookup without changing modeling). These are minimal changes that preserve the core logic and should move the score downward toward the target. The rest of the code (breath-wise features, inspiratory-only training, breath_id split, KNN settings, and submission alignment) stays intact.'
- What this solution (achieved 1.35576) has done: 'We keep your exact KNN + scaler pipeline and the same breath-wise lag/cumsum feature set, but make two minimal, metric-aligned fixes that usually lower MAE in this competition. First, we train the KNN on *all* rows (both u_out phases) while still evaluating/validating MAE only on inspiratory rows; this preserves the model/training approach but gives KNN much denser coverage of feature space, improving inspiratory generalization. Second, we make the priors more robust by using out-of-fold (breath-wise) computation for the `(R,C,t_idx)` mean prior on the training fold only (to avoid leakage into validation and to keep priors consistent), while still using the full-train prior for test. Submission writing, alignment by `id`, and the inspiratory-only snapping-to-pressure-grid post-process remain unchanged.'
- What this solution (achieved 1.30386) has done: 'Your current MAE (1.35576, lower-is-better) is still far above the target (0.14294), so we should improve accuracy with the smallest changes that keep the same KNN+scaler pipeline and the same breath-wise feature/prior idea. The biggest low-risk gain is to stop fitting KNN on the full 5.4M rows (which tends to “average away” rare exact patterns) and instead fit only on the inspiratory phase (the only scored region), while still predicting for all rows in test and filling expiratory values as you already do. Additionally, the priors you already compute become more metric-aligned if they are also computed only from inspiratory rows (so their means aren’t diluted by expiratory dynamics). Everything else (features, one-hot R/C, KNN settings, snapping to pressure grid for inspiratory rows, submission alignment) is left intact.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
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
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
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
sub = pd.read_csv(
    SAMPLE_SUB_PATH,
    usecols=["id", "pressure"],
    dtype={"id": np.int32, "pressure": np.float32},
)

assert "id" in test.columns and "id" in sub.columns
assert len(test) == len(sub)




## === cell 1
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    u_in = out["u_in"].to_numpy(np.float32, copy=False)
    u_out = out["u_out"].to_numpy(np.float32, copy=False)
    t = out["time_step"].to_numpy(np.float32, copy=False)

    n = u_in.shape[0]
    if n % 80 != 0:
        raise ValueError(
            "Expected number of rows to be a multiple of 80 (one breath = 80 rows)."
        )

    nb = n // 80
    u_in_2d = u_in.reshape(nb, 80)
    u_out_2d = u_out.reshape(nb, 80)
    t_2d = t.reshape(nb, 80)

    u_in_lag1_2d = np.zeros_like(u_in_2d, dtype=np.float32)
    u_in_lag2_2d = np.zeros_like(u_in_2d, dtype=np.float32)
    u_out_lag1_2d = np.zeros_like(u_out_2d, dtype=np.float32)

    u_in_lag1_2d[:, 1:] = u_in_2d[:, :-1]
    u_in_lag2_2d[:, 2:] = u_in_2d[:, :-2]
    u_out_lag1_2d[:, 1:] = u_out_2d[:, :-1]

    out = out.copy(deep=False)
    out["u_in_lag1"] = u_in_lag1_2d.reshape(-1)
    out["u_in_lag2"] = u_in_lag2_2d.reshape(-1)
    out["u_out_lag1"] = u_out_lag1_2d.reshape(-1)

    u_in_lag1 = out["u_in_lag1"].to_numpy(np.float32, copy=False)
    u_out_lag1 = out["u_out_lag1"].to_numpy(np.float32, copy=False)

    out["u_in_diff1"] = u_in - u_in_lag1
    out["u_out_diff1"] = u_out - u_out_lag1

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

train_fe = train_fe.copy(deep=False)
test_fe = test_fe.copy(deep=False)
train_fe["t_idx"] = np.arange(len(train_fe), dtype=np.int32) % 80
test_fe["t_idx"] = np.arange(len(test_fe), dtype=np.int32) % 80

u_in_bin_width = 1.0
train_fe["u_in_bin"] = np.floor(
    train_fe["u_in"].to_numpy(np.float32, copy=False) / u_in_bin_width
).astype(np.int16, copy=False)
test_fe["u_in_bin"] = np.floor(
    test_fe["u_in"].to_numpy(np.float32, copy=False) / u_in_bin_width
).astype(np.int16, copy=False)

rs = 42

max_breaths = 12000
unique_breaths = train_fe["breath_id"].unique()
if len(unique_breaths) > max_breaths:
    rng = np.random.RandomState(rs)
    keep_breaths = rng.choice(unique_breaths, size=max_breaths, replace=False)
    keep_breaths_sorted = np.sort(keep_breaths)
    breath_arr_full = train_fe["breath_id"].to_numpy(np.int32, copy=False)
    pos = np.searchsorted(keep_breaths_sorted, breath_arr_full)
    mask_keep = (pos < keep_breaths_sorted.size) & (
        keep_breaths_sorted[pos] == breath_arr_full
    )
    train_fit_df = train_fe.loc[mask_keep]
else:
    train_fit_df = train_fe

breaths_fit = pd.unique(train_fit_df["breath_id"].to_numpy(copy=False))
b_tr, b_va = train_test_split(breaths_fit, test_size=0.1, random_state=rs)

breath_arr = train_fit_df["breath_id"].to_numpy(np.int32, copy=False)

b_tr_sorted = np.sort(b_tr.astype(np.int32, copy=False))
pos_tr = np.searchsorted(b_tr_sorted, breath_arr)
tr_mask = (pos_tr < b_tr_sorted.size) & (b_tr_sorted[pos_tr] == breath_arr)
va_mask = ~tr_mask  # since split is disjoint and covers breaths_fit

train_tr = train_fit_df.loc[tr_mask]
train_va = train_fit_df.loc[va_mask]

train_tr_fit = train_tr.loc[train_tr["u_out"].to_numpy(copy=False) == 0]

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
    g = src_df.groupby(["R", "C", "t_idx"], sort=False, observed=True)[target].mean()
    g = g.reset_index()

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
        dst_df["R"].to_numpy(np.int16, copy=False),
        dst_df["C"].to_numpy(np.int16, copy=False),
        dst_df["t_idx"].to_numpy(np.int32, copy=False),
    )
    pos = np.searchsorted(keys_sorted, keys_dst)

    out = np.full(keys_dst.shape[0], np.float32(fill_value), dtype=np.float32)

    valid = pos < keys_sorted.size
    if valid.any():
        posv = pos[valid]
        keysv = keys_dst[valid]
        hitv = keys_sorted[posv] == keysv
        if hitv.any():
            out_idx = np.flatnonzero(valid)[hitv]
            out[out_idx] = vals_sorted[posv[hitv]]

    dst_df[dst_name] = out


def _add_prior_mean_rc_tidx_uin(
    src_df: pd.DataFrame, dst_df: pd.DataFrame, dst_name: str
):
    g = src_df.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False, observed=True)[
        target
    ].mean()
    g = g.reset_index()

    keys_g = _make_key_rc_tidx_uin(
        g["R"].to_numpy(np.int16, copy=False),
        g["C"].to_numpy(np.int16, copy=False),
        g["t_idx"].to_numpy(np.int32, copy=False),
        g["u_in_bin"].to_numpy(np.int16, copy=False),
    )
    vals_g = g[target].to_numpy(np.float32, copy=False)

    order = np.argsort(keys_g, kind="mergesort")
    keys_sorted = keys_g[order]
    vals_sorted = vals_g[order]

    keys_dst = _make_key_rc_tidx_uin(
        dst_df["R"].to_numpy(np.int16, copy=False),
        dst_df["C"].to_numpy(np.int16, copy=False),
        dst_df["t_idx"].to_numpy(np.int32, copy=False),
        dst_df["u_in_bin"].to_numpy(np.int16, copy=False),
    )
    pos = np.searchsorted(keys_sorted, keys_dst)

    out = np.full(keys_dst.shape[0], np.nan, dtype=np.float32)
    valid = pos < keys_sorted.size
    if valid.any():
        posv = pos[valid]
        keysv = keys_dst[valid]
        hitv = keys_sorted[posv] == keysv
        if hitv.any():
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

train_tr_fit = train_tr_fit.copy(deep=False)
train_va = train_va.copy(deep=False)
test_fe = test_fe.copy(deep=False)

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
    p = X_num.shape[1]
    X = np.empty((n, p + 6), dtype=np.float32)
    X[:, :p] = X_num
    X[:, p : p + 3] = (Rv[:, None] == R_levels[None, :]).astype(np.float32, copy=False)
    X[:, p + 3 : p + 6] = (Cv[:, None] == C_levels[None, :]).astype(
        np.float32, copy=False
    )
    return X


X_tr = build_X(train_tr_fit)
X_va = build_X(train_va)
X_test = build_X(test_fe)

y_tr = train_tr_fit[target].to_numpy(dtype=np.float32, copy=False)
y_va = train_va[target].to_numpy(dtype=np.float32, copy=False)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1808759471.py in <cell line: 0>()
     40     pos = np.searchsorted(keep_breaths_sorted, breath_arr_full)
     41     mask_keep = (pos < keep_breaths_sorted.size) & (
---> 42         keep_breaths_sorted[pos] == breath_arr_full
     43     )
     44     train_fit_df = train_fe.loc[mask_keep]

IndexError: index 12000 is out of bounds for axis 0 with size 12000

## === cell 3
model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True, copy=False)),
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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2436871131.py in <cell line: 0>()
     10 )
     11 
---> 12 model.fit(X_tr, y_tr)
     13 
     14 va_pred = model.predict(X_va).astype(np.float32, copy=False)

NameError: name 'X_tr' is not defined

## === cell 4
test_pred = model.predict(X_test).astype(np.float32, copy=False)

ids = test_fe["id"].to_numpy(np.int32, copy=False)
u_out = test_fe["u_out"].to_numpy(np.int8, copy=False)

pred2d = test_pred.reshape(-1, 80).copy()
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

sub_sorted = sub.sort_values("id", kind="mergesort").reset_index(drop=True)
test_ids_sorted = np.sort(ids, kind="mergesort")
assert sub_sorted["id"].to_numpy().shape == test_ids_sorted.shape
assert np.array_equal(sub_sorted["id"].to_numpy(), test_ids_sorted)

order = np.argsort(ids, kind="mergesort")
pred_by_id = pred_snapped_all[order].astype(np.float32, copy=False)

sub_sorted["pressure"] = pred_by_id
sub_sorted.to_csv("submission.csv", index=False)

sub_sorted.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3731602539.py in <cell line: 0>()
----> 1 test_pred = model.predict(X_test).astype(np.float32, copy=False)
      2 
      3 ids = test_fe["id"].to_numpy(np.int32, copy=False)
      4 u_out = test_fe["u_out"].to_numpy(np.int8, copy=False)
      5 

NameError: name 'X_test' is not defined
