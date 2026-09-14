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

0.1358446937940982

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the crash by removing hard-coded references to missing external blending files and instead blending only whatever prediction CSVs are actually present in the provided input directory. To keep the original core logic (random-weight blending + median/mean combination + snapping to nearest allowed pressure), I preserve those steps and just make them robust to an arbitrary number of files (including the case of only one). I also ensure the script always writes a valid Kaggle submission named `submission.csv` with the required `id,pressure` columns and correct row order/length. Finally, I update the paths to use the existing `/kaggle/input/...` structure so it runs end-to-end in the Kaggle environment.'
- What this solution (achieved 6.29564) has done: 'Your current score is extremely poor because the notebook is blending whatever CSVs it finds under `/kaggle/input` instead of actually training a model for this competition; that effectively produces near-random pressures. To move the score toward the target (lower is better) with minimal but necessary change, I keep your “snap predictions to the nearest allowed pressure” post-processing, but replace the “external prediction blending” with a lightweight, legitimate baseline model trained from `train.csv` and applied to `test.csv`. Specifically, we exploit a strong property of this dataset: pressures are highly quantized and correlate tightly with `u_in`, and during `u_out==1` the scored pressure can be safely set near the minimum (not scored during expiratory phase anyway). The result is a deterministic end-to-end script that always writes a valid `submission.csv` and should drastically reduce MAE toward your target band without changing evaluation semantics.'
- What this solution (achieved 6.29058) has done: 'Your current approach is a quantized lookup by `(R, C, u_in_bin)` with a global fallback; the big remaining error is that test has many unseen `(R, C, u_in_bin)` combos so the fallback is too coarse. To move the MAE down toward your target without changing the core “bin → group-median lookup → fallback → snap-to-allowed-pressures” logic, I’m adding a second, more specific fallback that uses `(R, u_in_bin)` and `(C, u_in_bin)` before falling back to `(u_in_bin)` and then global median. I’m also slightly tightening the bin size from `0.5` to `0.25` to reduce discretization error while still keeping the same basic lookup mechanism. Everything else (u_out handling, snapping, submission writing) stays the same.'
- What this solution (achieved 6.28994) has done: 'Your current score is far above the target (lower is better), so we need a legitimate accuracy improvement but with minimal change to your existing “bin → group-median lookup → fallbacks → snap-to-allowed-pressures” core logic. The biggest remaining issue is the coarse discretization of `u_in` into bins, which creates large quantization error and many unseen bins; we can reduce this error by shrinking the bin size while keeping the same lookup mechanism. We also add one more minimal, safe fallback using `(R, C, u_in_bin, u_out)` (still only learned from inspiratory `u_out==0` rows) so that the lookup is not forced to reuse inspiratory medians for expiratory rows before we overwrite them. Everything else (u_out handling, snapping, submission writing) stays the same to preserve semantics and stability.'
- What this solution (achieved 6.29058) has done: 'Your current MAE is far worse than the target (lower is better), so we need a legitimate accuracy gain while keeping your same “binned u_in → group-median lookup → fallbacks → snap-to-allowed pressures” core logic. The main issue is the very fine BIN=0.10, which creates many sparse/unseen bins and forces frequent fallback to coarser aggregations, hurting accuracy; increasing BIN slightly reduces sparsity and typically improves lookup stability. I’m also adding one minimal, higher-fidelity fallback keyed by `(R, C, u_in_bin)` (without `u_out`) before falling back to `(R,u_in_bin)` etc., which reduces unnecessary misses from including `u_out` in the main key while preserving your overall semantics. Everything else (inspiratory-only training for lookups, u_out==1 overwrite, and snapping) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 7.60671) has done: 'Your score is far worse than the target (lower is better), so we need a legitimate accuracy increase while keeping your same “binned u_in → group-median lookup → fallbacks → snap to allowed pressures” core logic. The biggest issue is that binning `u_in` and then doing median lookups throws away strong information contained in the exact `(R,C,u_in)` relationship; we can keep your lookup approach but replace binning with a minimal interpolation-based lookup per `(R,C)` group (still trained only on inspiratory `u_out==0`). This preserves your overall semantics (deterministic lookup + fallback + snapping + u_out handling) but greatly reduces discretization error and sparse-bin misses. We also fix the `id` alignment by merging predictions back to the sample submission on `id` directly (no reliance on `id` being a full 1..N range), which prevents silent misalignment from hurting MAE.'
- What this solution (achieved 6.81622) has done: 'I fix the `IndexError` by ensuring `baseline_interp()` (and later ridge fitting) uses positional indices (0..n-1) rather than the original DataFrame index values produced by filtering `train_insp`, which were causing out-of-bounds writes into a NumPy array. I do this with a minimal change: reset indices after sorting/feature creation and use `groupby(...).indices` (positional) instead of `.groups` (label-based). Then I make the ridge residual computation robust by converting the baseline predictions into a Series aligned to `train_insp`’s index before slicing per-group, preserving the exact core modeling logic (interp baseline + per-(R,C) ridge residual + snapping). Finally, I keep the submission merge-on-`id` behavior and ensure `submission.csv` is always written.'
- What this solution (achieved 11.26377) has done: 'Your score (6.81622, lower-is-better) is still far from the target (0.1358), so we need a real accuracy gain while keeping your same core structure: (1) per-(R,C) `u_in`→pressure interpolation baseline, (2) ridge-on-residual correction using the same FEATURES and closed-form solve, (3) overwrite expiratory `u_out==1`, and (4) snap to allowed pressures. The biggest accuracy bug is setting `u_out==1` to the minimum pressure; while those rows aren’t scored, this is a distribution shift that can hurt downstream consistency and (depending on scoring mask alignment) can still leak into scored rows, so we instead set expiratory predictions to the last inspiratory prediction within each breath (minimal semantic change: still deterministic and uses only test inputs). Additionally, we make the interpolation baseline use a stable monotone “envelope” of max pressure per `u_in` within each (R,C) group (instead of mean), which better matches the physical mapping and reduces underestimation without changing the interpolation approach. Everything else (modeling approach, ridge fit/apply, snapping, submission writing) stays the same.'
- What this solution (achieved 6.76401) has done: 'Your current score is far worse than the target (lower is better), so we need a real accuracy gain while keeping your same core pipeline (interp baseline per (R,C) → ridge on residuals → expiratory overwrite → snap). The biggest bug is that the interpolation baseline uses the max pressure per exact `u_in`, which biases predictions upward and hurts MAE; switching to a median per `u_in` preserves the same monotone lookup/interpolation logic but reduces this bias. Second, your ridge is trained and applied on expiratory (`u_out==1`) rows even though those aren’t scored and follow a different regime; training/applying ridge only on inspiratory rows and then carrying forward the last inspiratory prediction for expiratory rows keeps your semantics while avoiding regime-mismatch noise. These are minimal, targeted changes and should move MAE substantially toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 6.74694) has done: 'Your current MAE (6.764) is far above the target (0.136), so we need a legitimate accuracy increase while preserving your existing pipeline (interp baseline per (R,C) → ridge on residuals → expiratory carry-forward → snap). The smallest high-impact fix is to change the interpolation baseline from “median pressure per exact u_in” to “median pressure per (R,C,time_step)” which better matches the true generative process (pressure is primarily a function of time within a breath, with R and C) while keeping the same lookup/interp semantics. We keep the ridge residual correction exactly as-is, but now it learns only the residuals left after this stronger baseline, which should substantially reduce error. Submission writing, expiratory handling, snapping to allowed pressures, and file paths stay unchanged.'
- What this solution (achieved 3.3275) has done: 'Your current MAE is far above the target (lower is better), so we should make a small, high-impact accuracy fix without changing your overall “baseline interpolation → ridge residual correction → expiratory carry-forward → snap” pipeline. The main issue is that the baseline ignores the very strong dependence on `u_in`; switching the baseline to interpolate on `u_in` per `(R,C,time_step)` captures that dependence while keeping the same deterministic lookup/interp semantics. I keep the ridge residual model exactly the same, but because the baseline is stronger, we can safely reduce `L2` slightly so the residual correction isn’t overly damped (still the same closed-form ridge). Everything else (feature generation, inspiratory-only training, expiratory carry-forward, snapping, and submission writing) stays unchanged.'
- What this solution (achieved 3.29635) has done: 'Your current score (3.3275 MAE, lower-is-better) is still far above the target (0.1358), so we need a legitimate accuracy gain while keeping your same core pipeline (per-(R,C,time_step) u_in→pressure interpolation baseline → closed-form ridge residual correction → expiratory carry-forward → snap-to-allowed-pressures). The smallest high-impact fix is to build the per-time_step interpolation curve using the median pressure at each exact u_in (group by (time_step,u_in) and take median pressure), rather than median u_in at each pressure (which inverts the mapping and makes interpolation ill-posed). This preserves the exact modeling approach (still interpolation on u_in per time_step and (R,C), same ridge on residuals, same post-processing), but should materially reduce baseline error and thus the final MAE. I also make the curve construction robust by sorting by u_in and de-duplicating u_in via unique, while keeping all paths and submission formatting unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import gc

BASE_COMP_PATH = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = f"{BASE_COMP_PATH}/train.csv"
TEST_PATH = f"{BASE_COMP_PATH}/test.csv"
SAMPLE_SUB_PATH = f"{BASE_COMP_PATH}/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

sorted_pressures = np.sort(train["pressure"].unique())
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    if insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


min_pressure = float(sorted_pressures[0])




## === cell 1
def add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.sort_values(["breath_id", "time_step"], kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )
    g = df.groupby("breath_id", sort=False)

    df["u_in_cum"] = g["u_in"].cumsum().astype(np.float32)
    df["u_in_diff1"] = g["u_in"].diff().fillna(0.0).astype(np.float32)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)

    df["ts_idx"] = g.cumcount().astype(np.int16)
    return df


train_feat = add_ts_features(train)
test_feat = add_ts_features(test)

train_insp = train_feat[train_feat["u_out"] == 0].reset_index(drop=True).copy()
test_insp = test_feat[test_feat["u_out"] == 0].reset_index(drop=True).copy()

rc_groups = {}
r_groups = {}
c_groups = {}

for (r, c), g in train_insp.groupby(["R", "C"], observed=True):
    gg = g[["ts_idx", "u_in", "pressure"]].sort_values(["ts_idx", "u_in"])
    gg = gg.groupby(["ts_idx", "u_in"], as_index=False, observed=True)[
        "pressure"
    ].median()

    per_ts = {}
    for ts, h in gg.groupby("ts_idx", observed=True):
        h = h.sort_values("u_in")
        x = h["u_in"].to_numpy(np.float32)
        y = h["pressure"].to_numpy(np.float32)
        if len(x) < 2:
            continue
        ux, idx = np.unique(x, return_index=True)
        uy = y[idx]
        if len(ux) >= 2:
            per_ts[int(ts)] = (ux, uy)
    rc_groups[(int(r), int(c))] = per_ts

for r, g in train_insp.groupby(["R"], observed=True):
    gg = g[["ts_idx", "u_in", "pressure"]].sort_values(["ts_idx", "u_in"])
    gg = gg.groupby(["ts_idx", "u_in"], as_index=False, observed=True)[
        "pressure"
    ].median()
    per_ts = {}
    for ts, h in gg.groupby("ts_idx", observed=True):
        h = h.sort_values("u_in")
        x = h["u_in"].to_numpy(np.float32)
        y = h["pressure"].to_numpy(np.float32)
        if len(x) < 2:
            continue
        ux, idx = np.unique(x, return_index=True)
        uy = y[idx]
        if len(ux) >= 2:
            per_ts[int(ts)] = (ux, uy)
    r_groups[int(r)] = per_ts

for c, g in train_insp.groupby(["C"], observed=True):
    gg = g[["ts_idx", "u_in", "pressure"]].sort_values(["ts_idx", "u_in"])
    gg = gg.groupby(["ts_idx", "u_in"], as_index=False, observed=True)[
        "pressure"
    ].median()
    per_ts = {}
    for ts, h in gg.groupby("ts_idx", observed=True):
        h = h.sort_values("u_in")
        x = h["u_in"].to_numpy(np.float32)
        y = h["pressure"].to_numpy(np.float32)
        if len(x) < 2:
            continue
        ux, idx = np.unique(x, return_index=True)
        uy = y[idx]
        if len(ux) >= 2:
            per_ts[int(ts)] = (ux, uy)
    c_groups[int(c)] = per_ts

g_all = train_insp[["ts_idx", "u_in", "pressure"]].sort_values(["ts_idx", "u_in"])
g_all = g_all.groupby(["ts_idx", "u_in"], as_index=False, observed=True)[
    "pressure"
].median()

all_ts = {}
for ts, h in g_all.groupby("ts_idx", observed=True):
    h = h.sort_values("u_in")
    x = h["u_in"].to_numpy(np.float32)
    y = h["pressure"].to_numpy(np.float32)
    if len(x) < 2:
        continue
    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]
    if len(ux) >= 2:
        all_ts[int(ts)] = (ux, uy)

global_median = float(train_insp["pressure"].median())


def baseline_interp(df: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(df), dtype=np.float32)
    out[:] = np.nan

    ts_arr = df["ts_idx"].to_numpy(np.int16)
    uin_arr = df["u_in"].to_numpy(np.float32)
    r_arr = df["R"].to_numpy(np.int16)
    c_arr = df["C"].to_numpy(np.int16)

    for (r, c), pos_idx in df.groupby(["R", "C"], observed=True).indices.items():
        key_rc = (int(r), int(c))
        per_ts_rc = rc_groups.get(key_rc, None)
        per_ts_r = r_groups.get(int(r), None)
        per_ts_c = c_groups.get(int(c), None)

        for j in pos_idx:
            ts = int(ts_arr[j])
            uin = float(uin_arr[j])

            curve = None
            if per_ts_rc is not None:
                curve = per_ts_rc.get(ts, None)
            if curve is None and per_ts_r is not None:
                curve = per_ts_r.get(ts, None)
            if curve is None and per_ts_c is not None:
                curve = per_ts_c.get(ts, None)
            if curve is None:
                curve = all_ts.get(ts, None)

            if curve is None:
                out[j] = np.float32(global_median)
            else:
                x_uin, y_p = curve
                out[j] = np.interp(uin, x_uin, y_p).astype(np.float32)

    out = np.where(np.isfinite(out), out, np.float32(global_median))
    return out


base_train_insp = baseline_interp(train_insp)
base_test_insp = baseline_interp(test_insp)

FEATURES = ["u_in", "time_step", "u_in_cum", "u_in_diff1", "u_in_lag1"]
L2 = 0.1  # deterministic closed-form ridge


def fit_ridge(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)

    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xa = np.concatenate([X, ones], axis=1)  # intercept
    d = Xa.shape[1]
    A = Xa.T @ Xa
    A = A + (l2 * np.eye(d, dtype=np.float32))
    b = Xa.T @ y
    coef = np.linalg.solve(A, b).astype(np.float32)
    return coef


rc_coef = {}
Xg = train_insp[FEATURES].to_numpy(np.float32)
base_train_insp_s = pd.Series(base_train_insp, index=train_insp.index, dtype=np.float32)
yg = (train_insp["pressure"].astype(np.float32) - base_train_insp_s).to_numpy(
    np.float32
)
global_coef = fit_ridge(Xg, yg, L2)

MIN_SAMPLES = 5000
for (r, c), pos_idx in train_insp.groupby(["R", "C"], observed=True).indices.items():
    if len(pos_idx) < MIN_SAMPLES:
        continue
    Xrc = train_insp.loc[pos_idx, FEATURES].to_numpy(np.float32)
    yrc = (
        train_insp.loc[pos_idx, "pressure"].to_numpy(np.float32)
        - base_train_insp_s.loc[pos_idx].to_numpy(np.float32)
    ).astype(np.float32)
    rc_coef[(int(r), int(c))] = fit_ridge(Xrc, yrc, L2)


def apply_ridge(df: pd.DataFrame, base_pred: np.ndarray) -> np.ndarray:
    X = df[FEATURES].to_numpy(np.float32)
    pred = base_pred.astype(np.float32, copy=True)

    for (r, c), pos_idx in df.groupby(["R", "C"], observed=True).indices.items():
        key = (int(r), int(c))
        coef = rc_coef.get(key, global_coef)
        w = coef[:-1]
        b = coef[-1]
        pred[pos_idx] = pred[pos_idx] + (X[pos_idx] @ w + b).astype(np.float32)
    return pred


pred_insp = apply_ridge(test_insp, base_test_insp)

pred_full = np.empty(len(test_feat), dtype=np.float32)
pred_full[:] = np.float32(min_pressure)

u_out = test_feat["u_out"].to_numpy(dtype=np.int8)
breath_id = test_feat["breath_id"].to_numpy(np.int64)

insp_pos = 0
last_insp = np.float32(min_pressure)
prev_b = breath_id[0]
for i in range(len(test_feat)):
    b = breath_id[i]
    if b != prev_b:
        last_insp = np.float32(min_pressure)
        prev_b = b

    if u_out[i] == 0:
        last_insp = pred_insp[insp_pos]
        pred_full[i] = last_insp
        insp_pos += 1
    else:
        pred_full[i] = last_insp

pred_snapped = np.array([find_nearest(float(x)) for x in pred_full], dtype=np.float32)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2371776300.py in <cell line: 0>()
     66         if len(ux) >= 2:
     67             per_ts[int(ts)] = (ux, uy)
---> 68     r_groups[int(r)] = per_ts
     69 
     70 for c, g in train_insp.groupby(["C"], observed=True):

TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'

## === cell 2
pred_df = pd.DataFrame({"id": test_feat["id"].to_numpy(), "pressure": pred_snapped})

out = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
if "pressure_pred" in out.columns:
    out["pressure"] = out["pressure_pred"]
    out = out.drop(columns=["pressure_pred"])

if out["pressure"].isna().any():
    out["pressure"] = out["pressure"].fillna(np.float32(min_pressure))

out["pressure"] = out["pressure"].astype(np.float32)
out.to_csv("submission.csv", index=False)

del (
    train_feat,
    test_feat,
    train_insp,
    test_insp,
    pred_df,
    pred_insp,
    pred_full,
    pred_snapped,
)
gc.collect()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4249100975.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({"id": test_feat["id"].to_numpy(), "pressure": pred_snapped})
      2 
      3 out = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
      4 if "pressure_pred" in out.columns:
      5     out["pressure"] = out["pressure_pred"]

NameError: name 'pred_snapped' is not defined
