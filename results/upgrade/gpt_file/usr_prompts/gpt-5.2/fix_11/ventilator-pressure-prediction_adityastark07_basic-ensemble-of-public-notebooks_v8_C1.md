# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.1438098822599156

# 6. Current score

13.23182

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.49101) has done: 'Your current notebook fails because it tries to read several external ensemble submission files that are not present in this Kaggle environment, so execution stops before any valid `submission.csv` is created. To fix this end-to-end, I remove the unavailable external-input dependency and instead train a simple, fast baseline model directly from the provided `train.csv` and predict on `test.csv`, preserving the required `id,pressure` submission format. Since there is no current score (no submission yielded), the minimal legitimate way to move toward the target MAE is to produce a real model-based submission rather than all-zeros. I also ensure paths match your provided dataset layout and that the saved file has a `.csv` suffix and correct row alignment by `id`.'
- What this solution (achieved 1.29627) has done: 'Your current score (1.49101 MAE) is far worse than the target (0.1438), so we should improve performance while keeping the same overall approach (feature engineering + a tree model) intact. The biggest issue is that the model is trained on all timesteps including expiratory phases, while Kaggle scores only inspiratory timesteps (`u_out==0`), so training should be aligned to that metric. Also, pressure takes on a small discrete set of values in this competition; snapping predictions to the nearest valid pressure level (learned from train) is a small post-processing step that usually reduces MAE without changing the modeling core. Finally, keeping `R` and `C` as categorical-like integers (and relying on `RC_code`) is fine, but we preserve your features and only adjust training filtering and prediction post-processing.'
- What this solution (achieved 1.26646) has done: 'Your current MAE (1.296) is still far from the target (0.1438), so we should improve performance with minimal, metric-aligned tweaks rather than changing the modeling approach. The biggest remaining gap is that the model is trained as an i.i.d. tabular regressor per timestep, but pressure is strongly dependent on within-breath history; we can capture that by adding a few lightweight cumulative/lagged features (cumulative u_out, u_in differences, and simple moving averages) without changing the model class or training loop. Additionally, clipping predictions to the valid pressure range before snapping prevents rare out-of-range artifacts from worsening MAE. These changes keep the same feature-engineering + HistGradientBoostingRegressor core logic while typically moving scores substantially closer to strong baselines for this competition.'
- What this solution (achieved 1.26646) has done: 'Your current score is far worse than the target (lower is better), so we should improve MAE while keeping your exact core approach (tabular feature engineering + `HistGradientBoostingRegressor` + snapping to valid pressure levels). The biggest minimal win left is to stop teaching the model that expiratory timesteps matter: keep training on `u_out==0`, but also force test predictions at `u_out==1` to a stable baseline (last inspiratory prediction within the same breath, or 0 if a breath starts with `u_out==1`), since those rows are not scored and can otherwise introduce erratic outputs. Additionally, we can make the snapping step slightly stronger by snapping after this expiratory fill (not before), which typically reduces noise while preserving your existing post-processing semantics. These are small, metric-aligned tweaks that should move the score meaningfully toward the target without changing the model class, loss, or training loop.'
- What this solution (achieved 1.18635) has done: 'Your current approach is already metric-aligned (train on `u_out==0`, MAE loss, and pressure snapping), but it’s still missing the strongest “minimal” signal: within-breath cumulative dynamics that approximate delivered volume/flow history. I keep the same model class and training loop, and only add a few lightweight, groupby-based physical proxy features (`u_in` cumulative integral and cumulative sums, plus a couple extra short lags) that typically reduce MAE a lot for this competition without changing semantics. I also ensure the expiratory-phase fill happens before snapping (as you intended) and keep the same submission format and alignment checks. These changes should move the MAE down toward your target while staying within your constraints.'
- What this solution (achieved 12.75163) has done: 'The timeout is dominated by cell 4’s per-row `.predict()` call inside nested Python loops and repeated DataFrame indexing, which turns inference into hundreds of thousands of tiny sklearn calls. I preserve the exact autoregressive “previous predicted pressure” logic and snapping, but compute predictions per-breath using NumPy arrays and in-place feature updates, while batching model predictions in chunks to amortize sklearn overhead. I also speed up feature engineering by reusing groupby objects and avoiding repeated `groupby(...).shift(...)` constructions, without changing any computed feature values. Finally, I keep paths, model, hyperparameters, and evaluation semantics identical, only reducing constant-factor overhead.'
- What this solution (achieved 13.23182) has done: 'The timeout is dominated by two hotspots: Python loops over millions of rows in `add_features()` (multiple cumulative/rolling/ema computations) and the per-timestep `model.predict()` call inside the test breath loop (603,600 tiny predictions). I keep the exact same features, model, and autoregressive prediction semantics, but move the expensive per-row feature calculations into compiled NumPy group operations (`np.add.reduceat`, `np.maximum.reduceat`) and replace repeated `predict(1 row)` with `predict(one breath chunk)` while still updating `p_prev/p_prev2` sequentially (same logic, far fewer sklearn calls). I also avoid unnecessary pandas object creation during test-time forward-fill by doing an equivalent per-breath NumPy forward fill (same semantics) to remove a large groupby overhead. These changes are strictly performance-focused and preserve the algorithm and outputs up to negligible float differences.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sub_path = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(
    train_path,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float64",
        "u_in": "float64",
        "u_out": "int8",
        "pressure": "float64",
    },
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float64",
        "u_in": "float64",
        "u_out": "int8",
    },
)
sub = pd.read_csv(sub_path, dtype={"id": "int32", "pressure": "float64"})

train.shape, test.shape, sub.shape




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    breath = df["breath_id"].to_numpy(copy=False)
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)
    time_step = df["time_step"].to_numpy(dtype=np.float64, copy=False)
    R = df["R"].to_numpy(copy=False)
    C = df["C"].to_numpy(copy=False)
    n = len(df)

    new_breath = np.empty(n, dtype=bool)
    new_breath[0] = True
    new_breath[1:] = breath[1:] != breath[:-1]
    starts = np.flatnonzero(new_breath)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n

    def lag(arr, k, fill=0.0, dtype=None):
        out = np.empty(n, dtype=(arr.dtype if dtype is None else dtype))
        out[:k] = fill
        out[k:] = arr[:-k]
        out[new_breath] = fill
        return out

    df["u_in_lag1"] = lag(u_in, 1, fill=0.0, dtype=np.float64)
    df["u_in_lag2"] = lag(u_in, 2, fill=0.0, dtype=np.float64)
    df["u_in_lag3"] = lag(u_in, 3, fill=0.0, dtype=np.float64)
    df["u_in_lag4"] = lag(u_in, 4, fill=0.0, dtype=np.float64)

    df["u_out_lag1"] = lag(u_out, 1, fill=0, dtype=np.int8)
    df["u_out_lag2"] = lag(u_out, 2, fill=0, dtype=np.int8)

    dt = np.empty(n, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = time_step[1:] - time_step[:-1]
    dt[new_breath] = 0.0
    df["dt"] = dt

    area_inc = u_in * dt
    cs = np.cumsum(area_inc, dtype=np.float64)
    base = np.empty(n, dtype=np.float64)
    base[starts] = np.concatenate(([0.0], cs[starts[1:] - 1]))
    base = np.maximum.accumulate(base)  # propagate within breath
    df["area"] = cs - base

    df["RC"] = df["R"].astype(str) + "_" + df["C"].astype(str)

    diff1 = np.empty(n, dtype=np.float64)
    diff1[0] = 0.0
    diff1[1:] = u_in[1:] - u_in[:-1]
    diff1[new_breath] = 0.0
    df["u_in_diff1"] = diff1

    diff2 = np.empty(n, dtype=np.float64)
    diff2[:2] = 0.0
    diff2[2:] = u_in[2:] - u_in[:-2]
    diff2[new_breath] = 0.0
    idx_second = starts + 1
    idx_second = idx_second[idx_second < n]
    diff2[idx_second] = 0.0
    df["u_in_diff2"] = diff2

    roll3 = np.empty(n, dtype=np.float64)
    roll5 = np.empty(n, dtype=np.float64)
    k3 = np.ones(3, dtype=np.float64)
    k5 = np.ones(5, dtype=np.float64)
    for s, e in zip(starts, ends):
        seg = u_in[s:e]
        m = e - s

        c3 = np.convolve(seg, k3, mode="full")[:m]
        denom3 = np.minimum(np.arange(1, m + 1, dtype=np.float64), 3.0)
        roll3[s:e] = c3 / denom3

        c5 = np.convolve(seg, k5, mode="full")[:m]
        denom5 = np.minimum(np.arange(1, m + 1, dtype=np.float64), 5.0)
        roll5[s:e] = c5 / denom5

    df["u_in_roll_mean3"] = roll3
    df["u_in_roll_mean5"] = roll5

    u_out_i16 = u_out.astype(np.int16, copy=False)
    cs_o = np.cumsum(u_out_i16, dtype=np.int32).astype(np.int32, copy=False)
    base_o = np.empty(n, dtype=np.int32)
    base_o[starts] = np.concatenate(([0], cs_o[starts[1:] - 1]))
    base_o = np.maximum.accumulate(base_o)
    df["u_out_cum"] = (cs_o - base_o).astype(np.int16)

    df["u_in_x_R"] = u_in * R
    df["u_in_x_C"] = u_in * C

    cs_in = np.cumsum(u_in, dtype=np.float64)
    base_in = np.empty(n, dtype=np.float64)
    base_in[starts] = np.concatenate(([0.0], cs_in[starts[1:] - 1]))
    base_in = np.maximum.accumulate(base_in)
    df["u_in_cumsum"] = cs_in - base_in

    df["u_in_sq"] = u_in**2

    u_in_sq = df["u_in_sq"].to_numpy(dtype=np.float64, copy=False)
    cs_sq = np.cumsum(u_in_sq, dtype=np.float64)
    base_sq = np.empty(n, dtype=np.float64)
    base_sq[starts] = np.concatenate(([0.0], cs_sq[starts[1:] - 1]))
    base_sq = np.maximum.accumulate(base_sq)
    df["u_in_sq_cumsum"] = cs_sq - base_sq

    alpha = 0.3
    one_minus = 1.0 - alpha
    u_in_ema = np.empty(n, dtype=np.float64)
    for s, e in zip(starts, ends):
        seg = u_in[s:e]
        prev = seg[0]
        u_in_ema[s] = prev
        for i in range(s + 1, e):
            prev = alpha * u_in[i] + one_minus * prev
            u_in_ema[i] = prev
    df["u_in_ema"] = u_in_ema

    df["area_per_time"] = df["area"].to_numpy(copy=False) / (time_step + 1e-6)

    return df


train_fe = add_features(train)
test_fe = add_features(test)

rc_all = pd.Index(train_fe["RC"]).append(pd.Index(test_fe["RC"]))
rc_uniques = pd.Index(pd.unique(rc_all))
rc_map = pd.Series(np.arange(len(rc_uniques), dtype=np.int16), index=rc_uniques)

train_fe["RC_code"] = rc_map.loc[train_fe["RC"]].to_numpy()
test_fe["RC_code"] = rc_map.loc[test_fe["RC"]].to_numpy()

train_fe.head()



## === cell 3
from sklearn.ensemble import HistGradientBoostingRegressor

pressure_levels = np.sort(train["pressure"].unique().astype(np.float32))
pmin, pmax = float(pressure_levels.min()), float(pressure_levels.max())


def snap_to_levels(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = np.clip(x, levels.min(), levels.max())
    idx = np.searchsorted(levels, x, side="left")
    idx = np.clip(idx, 0, len(levels) - 1)
    idx_left = np.clip(idx - 1, 0, len(levels) - 1)
    right = levels[idx]
    left = levels[idx_left]
    choose_left = np.abs(x - left) <= np.abs(x - right)
    return np.where(choose_left, left, right).astype(np.float32)


train_pressure_snapped = snap_to_levels(
    train_fe["pressure"].to_numpy(np.float32, copy=False), pressure_levels
)

breath_tr = train_fe["breath_id"].to_numpy(copy=False)
new_breath_tr = np.empty(len(train_fe), dtype=bool)
new_breath_tr[0] = True
new_breath_tr[1:] = breath_tr[1:] != breath_tr[:-1]

p_prev = np.empty(len(train_fe), dtype=np.float32)
p_prev2 = np.empty(len(train_fe), dtype=np.float32)
p_prev[0] = 0.0
p_prev[1:] = train_pressure_snapped[:-1]
p_prev[new_breath_tr] = 0.0

p_prev2[:2] = 0.0
p_prev2[2:] = train_pressure_snapped[:-2]
p_prev2[new_breath_tr] = 0.0
idx_second_tr = np.flatnonzero(new_breath_tr) + 1
idx_second_tr = idx_second_tr[idx_second_tr < len(train_fe)]
p_prev2[idx_second_tr] = 0.0

train_fe["p_prev"] = p_prev
train_fe["p_prev2"] = p_prev2

test_fe["p_prev"] = 0.0
test_fe["p_prev2"] = 0.0

feature_cols = [
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_out_lag1",
    "u_out_lag2",
    "dt",
    "area",
    "R",
    "C",
    "RC_code",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_roll_mean3",
    "u_in_roll_mean5",
    "u_out_cum",
    "u_in_x_R",
    "u_in_x_C",
    "u_in_cumsum",
    "u_in_sq_cumsum",
    "u_in_lag4",
    "u_in_ema",
    "area_per_time",
    "p_prev",
    "p_prev2",
]

train_mask = train_fe["u_out"].to_numpy(copy=False) == 0
X_train = train_fe.loc[train_mask, feature_cols]
y_train = train_fe.loc[train_mask, "pressure"].astype(np.float32)

model = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.05,
    max_depth=6,
    max_iter=350,
    l2_regularization=0.0,
    random_state=42,
)
model.fit(X_train, y_train)



## === cell 4
u_out_test = test_fe["u_out"].to_numpy(copy=False)
breath_id_test = test_fe["breath_id"].to_numpy(copy=False)
n = len(test_fe)

X_test_full = test_fe[feature_cols].to_numpy(copy=True)  # mutated p_prev/p_prev2
col_idx = {c: i for i, c in enumerate(feature_cols)}
p_prev_col = col_idx["p_prev"]
p_prev2_col = col_idx["p_prev2"]

pred_seq = np.empty(n, dtype=np.float32)

breath_changes = np.flatnonzero(breath_id_test[1:] != breath_id_test[:-1]) + 1
starts = np.concatenate(([0], breath_changes))
ends = np.concatenate((breath_changes, [n]))

for s, e in zip(starts, ends):
    prev1 = np.float32(0.0)
    prev2 = np.float32(0.0)

    for j in range(s, e):
        X_test_full[j, p_prev_col] = prev1
        X_test_full[j, p_prev2_col] = prev2
        prev2 = prev1
        prev1 = np.float32(0.0)  # placeholder; will be overwritten after predicting

    raw = model.predict(X_test_full[s:e]).astype(np.float32, copy=False)
    snapped = snap_to_levels(raw, pressure_levels)

    prev1 = np.float32(0.0)
    prev2 = np.float32(0.0)
    for j in range(s, e):
        p = snapped[j - s]
        pred_seq[j] = p
        prev2 = prev1
        prev1 = p

pred_filled = pred_seq.copy()
mask_out = u_out_test == 1
pred_filled[mask_out] = np.nan

for s, e in zip(starts, ends):
    last = np.nan
    for j in range(s, e):
        v = pred_filled[j]
        if np.isnan(v):
            if not np.isnan(last):
                pred_filled[j] = last
        else:
            last = v
    if np.isnan(pred_filled[s]):
        for j in range(s, e):
            if np.isnan(pred_filled[j]):
                pred_filled[j] = 0.0
            else:
                break

pred_filled = np.clip(pred_filled, pmin, pmax).astype(np.float32)
pred_snapped = snap_to_levels(pred_filled, pressure_levels)

submission = sub.copy()
submission["pressure"] = pred_snapped

assert submission.shape[0] == test.shape[0], "Submission row count mismatch."
assert submission["id"].iloc[0] == test["id"].iloc[0], "ID alignment mismatch at start."
assert submission["id"].iloc[-1] == test["id"].iloc[-1], "ID alignment mismatch at end."

submission.to_csv("submission.csv", index=False)
submission.head()
