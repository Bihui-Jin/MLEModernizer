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

0.1437364723911435

# 6. Current score

3.14187

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.18407) has done: 'I remove the dependency on missing external Kaggle Dataset submissions (those `../input/.../submission*.csv` files are not available in your environment), and instead generate predictions directly from the provided `train.csv`/`test.csv`. To keep changes minimal while producing a strong baseline score, I implement a simple, competition-valid approach: build a lookup table of mean `pressure` by `(R, C, time_step, u_in, u_out)` from train and apply it to test, with safe backoff to progressively coarser group means when an exact combination is unseen. This fixes the runtime errors, runs end-to-end within the environment constraints, and writes a valid `submission.csv` with `id,pressure`. The logic is deterministic and avoids leakage (only uses train to estimate conditional means).'
- What this solution (achieved 6.11394) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that the solution ignores the breath-level time-series structure and the “only inspiratory phase is scored” detail, so the lookup is too coarse/noisy for a good MAE. I keep the same core idea (train-derived mean lookup + backoff) but make two minimal, metric-aligned upgrades: (1) build lookups separately for inspiratory-only rows (`u_out==0`) and expiratory rows (`u_out==1`) so predictions during the scored phase aren’t diluted, and (2) add a lightweight per-breath running integral feature (`u_in_cumsum`) to the keys so the mapping better reflects the system state without changing the overall approach. These are still pure aggregation-based predictions (no model/loops/loss changes) and should move the score substantially toward the target while keeping runtime under the limit and producing a valid `submission.csv`.'
- What this solution (achieved 6.10623) has done: 'Your current MAE is far above the target, so we should improve predictive fidelity with the smallest changes that keep your “train-derived lookup + backoff” core logic intact. The biggest low-risk gain here is to respect breath-level dynamics more: add a per-breath lagged control signal (`u_in_lag1`) and a simple per-breath cumulative `u_out` counter (`u_out_cumsum`) as extra rounded keys, so the lookup better matches similar states without changing the approach. I also fix a subtle but important issue: your `dt` is computed on a sorted copy but then grouped by an unsorted `breath_id` Series, which can misalign; computing `dt` via `groupby("breath_id")["time_step"].diff()` avoids that. These changes should move the score substantially toward the target while keeping runtime reasonable and still producing a valid `submission.csv`.'
- What this solution (achieved 5.10272) has done: 'Your current MAE (6.10623) is still far above the target (0.1437), so we should improve predictive fidelity with minimal, lookup-based changes that preserve your core “train-derived mean tables + backoff” approach. The biggest low-risk issue is key mismatch from rounding on continuous features (`u_in`, `u_in_int`, `u_in_lag1`), which causes many misses and forces coarse fallbacks; we replace rounding with binning via `pd.cut` on train-derived bin edges so train/test map to identical discrete keys. We also add a tiny breath-position key (`step` within breath) to stabilize matching across breaths (still a pure aggregation key, not a model change). Finally, we keep the inspiratory/expiratory split and the backoff cascade intact, just with binned keys to increase hit-rate and move MAE down toward the target.'
- What this solution (achieved 5.38077) has done: 'Your current MAE (5.10) is far above the target (0.1437), so we need a modest but meaningful fidelity boost while keeping your “train-derived lookup + backoff cascade” core logic unchanged. The main issue is that binning `u_in`/lags/integral into 200–300 uniform bins creates too many near-empty groups and forces coarse fallbacks; switching to quantile-based bins (computed on train and applied to test) increases key hit-rate without changing the approach. I also add a single extra breath-state key (`pressure_lag1_mean` = mean pressure in train for the previous step under the same keys) computed only from train and used as an additional lookup dimension in the first (most specific) table to better capture dynamics with minimal code. Finally, I keep your inspiratory/expiratory split, backoff list, clipping, and submission merge semantics intact.'
- What this solution (achieved 5.17151) has done: 'Your current MAE is far above the target (lower is better), and the most likely cause is key instability/over-fragmentation from quantile binning across many features, which forces frequent fallbacks to coarse means. To move the score down with minimal change while preserving the same “train-derived lookup tables + backoff cascade” core logic, I (1) replace quantile binning with fixed, physically meaningful bins for `u_in` (0–100) and a coarse but stable binning for `u_in_lag1`, and (2) normalize the integral feature by time (`u_in_int / (time_step+eps)`) before binning so it becomes scale-stable across breaths and reduces sparsity. I keep your inspiratory/expiratory split, the same backoff table structure, the same prediction fill procedure, and the same submission merge semantics. These changes mainly increase exact/near-exact table hit-rate during the inspiratory (scored) phase, which should reduce MAE toward the target without changing the overall approach.'
- What this solution (achieved 5.17151) has done: 'Your current MAE is far above the target (lower is better), so we should make the lookup hit-rate higher during the scored inspiratory phase without changing the overall “mean-table lookup + backoff cascade” approach. The smallest high-impact fix is to remove the over-fragmenting `pressure_lag1_b` dimension from the most specific tables (it’s derived from a noisy/incomplete backfilled lag-mean and causes many misses), while keeping it available only in a later, softer stage. We also add a metric-aligned post-process: when `u_out==1` (expiratory, not scored), set predictions to 0 to avoid wasting modeling capacity on that phase (this won’t hurt the leaderboard metric). These changes keep the same core logic (groupby means + cascading fallbacks), run fast, and still write a valid `submission.csv`.'
- What this solution (achieved 1.76906) has done: 'Your current MAE is far above the target (lower is better), so we should reduce avoidable lookup misses while keeping your same “mean-table lookup + backoff cascade” approach unchanged. The biggest low-risk issue is over-fragmentation from using a rounded float `time_step_r` as a key: even tiny representation differences can cause misses and force coarse fallbacks; switching to the already-computed integer `step` as the time-position key (and dropping `time_step_r` from lookup keys) makes matching stable without changing semantics. I also align the “prev step mean pressure” helper table to use the same stable keys so `pressure_lag1_mean` becomes less noisy (but still derived only from train). Everything else (feature engineering, inspiratory/expiratory split, cascade structure, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 1.76906) has done: 'Your current score is much worse than the target (lower is better), so we should make a small, metric-aligned change that improves predictions during the inspiratory (scored) phase without changing the overall “mean lookup + backoff cascade” approach. The most harmful current choice is forcing expiratory (`u_out==1`) predictions to 0; while expiratory isn’t scored, this hard override can still hurt because the metric masks by inspiratory *within breaths* and any misalignment/edge handling can bleed errors—so we remove that override and let the expiratory tables/global mean handle it. Separately, your most specific key includes `u_out_cumsum_b`, which fragments states heavily and reduces hit-rate; we keep the feature but drop it only from the first (most specific) table so more rows get high-specificity matches before falling back. These two minimal changes should reduce MAE toward the target while preserving your core logic, runtime, and submission format.'
- What this solution (achieved 8.51838) has done: 'We need to move your MAE down toward the target (0.1437) from 1.769, so the smallest likely win is to fix submission alignment and remove avoidable key-miss sources without changing the overall “train-derived mean lookup + backoff cascade” approach. Your current merge-by-`id` can silently misalign because `id` is not globally unique across the full file (it restarts each breath), which can drastically worsen MAE; switching to writing predictions in the exact `test` row order using `sample_submission.id` as the index fixes this while preserving semantics. Additionally, the most specific key list has a duplicated entry and an over-fragmenting early stage; we make it strictly monotonic (no duplicates) and add one very small metric-aligned improvement: compute mean tables with `observed=True` to reduce categorical cartesian blow-ups and ensure consistent groupby behavior (no logic change, just stability/perf). These changes keep the same features, same table cascade idea, and should materially improve score toward the target while remaining minimal and deterministic.'
- What this solution (achieved 8.51824) has done: 'Your current MAE is far worse than the target, so we should improve predictive accuracy without changing the core “train mean-lookup + backoff cascade” logic. The biggest low-risk win here is to post-process predictions to respect the competition’s *discrete* pressure levels: in this dataset pressure takes one of a fixed set of values, so snapping predictions to the nearest seen training pressure typically reduces MAE materially. This is purely a metric-aligned calibration step after your existing lookup predictions (no model/feature/loop changes), and it’s deterministic and fast. I also keep your submission-writing aligned to `sample_submission` as you already do.'
- What this solution (achieved 8.51542) has done: 'Your MAE is far worse than the target (lower is better), so we should make the smallest metric-aligned change that meaningfully reduces error without changing your core “mean lookup + backoff cascade” logic. The main remaining easy win is to respect the competition’s rule that only inspiratory phase is scored: keep your current predictions for `u_out==0`, but for `u_out==1` simply copy the last inspiratory prediction within the same breath (a common, valid post-process that improves consistency and avoids noisy expiratory estimates). This is a deterministic post-processing step applied after your existing tables, so it preserves your core approach and is very low risk. We keep the same submission alignment to `sample_submission` and still snap to the nearest discrete pressure value afterward.'
- What this solution (achieved 1.76887) has done: 'Your current MAE (8.515) is far worse than the target (0.1437), so we need to *improve* accuracy with the smallest possible change that preserves your mean-lookup + backoff cascade core logic. The dominant issue is almost certainly submission misalignment: your code assumes `sample_submission.id` aligns 1:1 with `test` row order, but `id` is the true unique identifier and should be taken directly from `test` to avoid row-order mistakes. I change submission writing to use `test["id"]` and add a strict safety check that `sub["id"]` matches `test["id"]` (if it does match, this change is score-neutral; if not, it fixes a catastrophic error). Everything else (features, binning, lookup tables, post-processing, snapping) remains unchanged.'
- What this solution (achieved 3.14187) has done: 'Your current MAE (1.76887, lower is better) is still far above the target (0.1437), so we should improve accuracy with the smallest changes that keep your “train-derived mean lookup + backoff cascade” core logic intact. The highest-impact minimal fix is to stop using `u_out_cumsum_b` as a key because it fragments groups heavily (and it’s always 0 in inspiratory where the metric is computed), which causes many unnecessary fallbacks; we keep the feature engineering but remove it from all lookup key lists and the prev-pressure helper table. Next, we add a tiny, breath-consistent post-process that enforces a non-decreasing inspiratory pressure within each breath (common for this dataset and aligned with the physical process), applied only where `u_out==0` so it directly targets the scored phase. Everything else (features, inspiratory/expiratory split, backoff cascade, discrete-pressure snapping, and submission writing) remains the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_PATH = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

required_train_cols = {"breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}
required_test_cols = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}
missing_train = required_train_cols - set(train.columns)
missing_test = required_test_cols - set(test.columns)
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")
if "id" not in sub.columns or "pressure" not in sub.columns:
    raise ValueError("sample_submission.csv must have columns: id, pressure")



## === cell 1
train = train.copy()
test = test.copy()

for df in (train, test):
    df.sort_values(["breath_id", "time_step"], inplace=True)

    dt = df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)

    df["u_in_int"] = (
        (df["u_in"].astype(np.float32) * dt).groupby(df["breath_id"]).cumsum()
    )
    df["u_in_lag1"] = (
        df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    df["u_out_cumsum"] = df.groupby("breath_id")["u_out"].cumsum().astype(np.int16)

    df["step"] = df.groupby("breath_id").cumcount().astype(np.int16)

    df["time_step_r"] = df["time_step"].round(5)

eps_t = np.float32(1e-6)
for df in (train, test):
    df["u_in_int_norm"] = (
        df["u_in_int"] / (df["time_step"].astype(np.float32) + eps_t)
    ).astype(np.float32)


def make_fixed_edges(min_v: float, max_v: float, step: float) -> np.ndarray:
    edges = np.arange(min_v, max_v + step, step, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([min_v, max_v], dtype=np.float64)
    eps = (edges[-1] - edges[0]) * 1e-9 + 1e-12
    edges[0] -= eps
    edges[-1] += eps
    return edges


u_in_edges = make_fixed_edges(0.0, 100.0, 0.5)
u_in_lag1_edges = make_fixed_edges(0.0, 100.0, 1.0)

uin_intn_lo = float(
    np.nanquantile(train["u_in_int_norm"].to_numpy(dtype=np.float64), 0.001)
)
uin_intn_hi = float(
    np.nanquantile(train["u_in_int_norm"].to_numpy(dtype=np.float64), 0.999)
)
if not np.isfinite(uin_intn_lo):
    uin_intn_lo = 0.0
if not np.isfinite(uin_intn_hi):
    uin_intn_hi = 1.0
if uin_intn_hi <= uin_intn_lo:
    uin_intn_hi = uin_intn_lo + 1.0

u_in_intn_edges = np.linspace(uin_intn_lo, uin_intn_hi, 161, dtype=np.float64)
eps = (u_in_intn_edges[-1] - u_in_intn_edges[0]) * 1e-9 + 1e-12
u_in_intn_edges[0] -= eps
u_in_intn_edges[-1] += eps


def apply_binning(df: pd.DataFrame):
    df["u_in_b"] = pd.cut(
        df["u_in"], bins=u_in_edges, labels=False, include_lowest=True
    ).astype("Int16")
    df["u_in_lag1_b"] = pd.cut(
        df["u_in_lag1"], bins=u_in_lag1_edges, labels=False, include_lowest=True
    ).astype("Int16")

    x = df["u_in_int_norm"].astype(np.float32)
    x = np.clip(x, uin_intn_lo, uin_intn_hi)
    df["u_in_int_b"] = pd.cut(
        x, bins=u_in_intn_edges, labels=False, include_lowest=True
    ).astype("Int16")

    df["u_out_cumsum_b"] = df["u_out_cumsum"].astype(np.int16)


apply_binning(train)
apply_binning(test)

train["pressure_lag1"] = (
    train.groupby("breath_id")["pressure"]
    .shift(1)
    .fillna(train["pressure"].mean())
    .astype(np.float32)
)

base_keys_prev = [
    "R",
    "C",
    "step",
    "u_in_b",
    "u_in_lag1_b",
    "u_in_int_b",
]

tbl_prev_pressure = train.groupby(base_keys_prev, sort=False, observed=True)[
    "pressure"
].mean()


def attach_prev_pressure_mean(df: pd.DataFrame):
    df_prev = df[base_keys_prev].copy()
    df_prev["step"] = (df_prev["step"].astype(np.int32) - 1).astype(np.int16)
    mi = pd.MultiIndex.from_frame(df_prev)
    prev_mean = tbl_prev_pressure.reindex(mi).to_numpy()
    prev_mean = np.where(
        pd.isna(prev_mean), float(train["pressure"].mean()), prev_mean
    ).astype(np.float32)
    df["pressure_lag1_mean"] = prev_mean


attach_prev_pressure_mean(train)
attach_prev_pressure_mean(test)


def make_quantile_edges(values: pd.Series, bins: int):
    v = values.to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([-1.0, 1.0], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    edges = np.quantile(v, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        x = float(edges[0])
        return np.array([x - 1.0, x + 1.0], dtype=np.float64)
    eps = (edges[-1] - edges[0]) * 1e-9 + 1e-12
    edges[0] -= eps
    edges[-1] += eps
    return edges.astype(np.float64)


p_lag1_edges = make_quantile_edges(train["pressure_lag1_mean"], bins=80)


def apply_pressure_lag1_binning(df: pd.DataFrame):
    df["pressure_lag1_b"] = pd.cut(
        df["pressure_lag1_mean"], bins=p_lag1_edges, labels=False, include_lowest=True
    ).astype("Int16")


apply_pressure_lag1_binning(train)
apply_pressure_lag1_binning(test)

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()

keys_list_common = [
    ["R", "C", "step", "u_in_b", "u_in_lag1_b", "u_in_int_b"],
    ["R", "C", "step", "u_in_b", "u_in_lag1_b", "u_in_int_b", "pressure_lag1_b"],
    ["R", "C", "step", "u_in_b", "u_in_int_b"],
    ["R", "C", "step", "u_in_b", "u_in_lag1_b"],
    ["R", "C", "step", "u_in_b"],
    ["R", "C", "step", "u_in_int_b"],
    ["R", "C", "step"],
    ["R", "C", "u_in_b", "u_in_lag1_b", "u_in_int_b"],
    ["R", "C", "u_in_b", "u_in_int_b"],
    ["R", "C", "u_in_b", "u_in_lag1_b"],
    ["R", "C", "u_in_b"],
    ["R", "C", "u_in_int_b"],
    ["R", "C"],
    ["step", "u_in_b", "u_in_lag1_b", "u_in_int_b"],
    ["step", "u_in_b", "u_in_int_b"],
    ["step", "u_in_b"],
    ["step", "u_in_int_b"],
    ["step"],
    ["u_in_b", "u_in_lag1_b", "u_in_int_b"],
    ["u_in_b", "u_in_int_b"],
    ["u_in_b"],
]


def build_mean_tables(df, keys_list):
    mean_tables = []
    for keys in keys_list:
        tbl = df.groupby(keys, sort=False, observed=True)["pressure"].mean()
        mean_tables.append((keys, tbl))
    return mean_tables


mean_tables_insp = build_mean_tables(train_insp, keys_list_common)
mean_tables_exp = build_mean_tables(train_exp, keys_list_common)

global_mean_insp = (
    float(train_insp["pressure"].mean())
    if len(train_insp)
    else float(train["pressure"].mean())
)
global_mean_exp = (
    float(train_exp["pressure"].mean())
    if len(train_exp)
    else float(train["pressure"].mean())
)

pred = np.full(len(test), np.nan, dtype=np.float32)


def fill_pred_for_mask(mask_rows, mean_tables, fallback_mean):
    if int(mask_rows.sum()) == 0:
        return
    pred_local = pred[mask_rows]
    test_part = test.loc[mask_rows]

    for keys, tbl in mean_tables:
        if np.isnan(pred_local).sum() == 0:
            break
        idx_missing = np.isnan(pred_local)
        mi = pd.MultiIndex.from_frame(test_part.loc[idx_missing, keys])
        vals = tbl.reindex(mi).to_numpy()
        found = ~pd.isna(vals)
        pred_local[idx_missing] = np.where(
            found, vals.astype(np.float32), pred_local[idx_missing]
        )

    pred_local = np.where(np.isnan(pred_local), fallback_mean, pred_local).astype(
        np.float32
    )
    pred[mask_rows] = pred_local


mask_insp = test["u_out"].values == 0
mask_exp = ~mask_insp

fill_pred_for_mask(mask_insp, mean_tables_insp, global_mean_insp)
fill_pred_for_mask(mask_exp, mean_tables_exp, global_mean_exp)

test_order = np.arange(len(test), dtype=np.int64)
tmp = pd.DataFrame(
    {
        "breath_id": test["breath_id"].values,
        "time_step": test["time_step"].values,
        "u_out": test["u_out"].values,
        "pred": pred,
        "_ord": test_order,
    }
).sort_values(["breath_id", "time_step"], kind="mergesort")

last_insp = (
    tmp["pred"].where(tmp["u_out"].values == 0).groupby(tmp["breath_id"]).ffill()
)
last_insp_final = last_insp.groupby(tmp["breath_id"]).transform("last")
tmp.loc[tmp["u_out"].values == 1, "pred"] = np.where(
    pd.isna(last_insp_final[tmp["u_out"].values == 1].to_numpy()),
    tmp.loc[tmp["u_out"].values == 1, "pred"].to_numpy(),
    last_insp_final[tmp["u_out"].values == 1].to_numpy(),
).astype(np.float32)

insp_mask = tmp["u_out"].values == 0
tmp.loc[insp_mask, "pred"] = (
    tmp.loc[insp_mask].groupby("breath_id")["pred"].cummax().astype(np.float32).values
)

pred = tmp.sort_values("_ord", kind="mergesort")["pred"].to_numpy(dtype=np.float32)

pmin, pmax = float(train["pressure"].min()), float(train["pressure"].max())
pred = np.clip(pred, pmin, pmax).astype(np.float32)

pressure_values = np.sort(train["pressure"].unique()).astype(np.float32)
idx = np.searchsorted(pressure_values, pred, side="left")
idx = np.clip(idx, 0, len(pressure_values) - 1)
idx0 = np.clip(idx - 1, 0, len(pressure_values) - 1)
p_hi = pressure_values[idx]
p_lo = pressure_values[idx0]
pred = np.where(np.abs(pred - p_lo) <= np.abs(p_hi - pred), p_lo, p_hi).astype(
    np.float32
)



## === cell 2
if len(sub) != len(test):
    raise ValueError(f"Sample submission length {len(sub)} != test length {len(test)}")

submission = pd.DataFrame({"id": test["id"].values, "pressure": pred}).copy()
submission["pressure"] = submission["pressure"].astype(np.float32)

if "id" in sub.columns:
    if not np.array_equal(sub["id"].to_numpy(), test["id"].to_numpy()):
        print(
            "Warning: sample_submission.id does not match test.id order; writing using test.id to avoid misalignment."
        )

submission.to_csv("submission.csv", index=False)
submission.head()
