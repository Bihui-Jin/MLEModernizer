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

0.1533028098228836

# 6. Current score

3.66968

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.0062) has done: 'The current notebook fails because it expects external prediction CSVs under `../input/gb-blending`, but that dataset is not present, so the blending code ends up producing arrays of length 1 and crashes when assigning into the 603600-row submission. I keep the existing “blend/median + snap-to-nearest-pressure” logic intact, but add a safe fallback that generates a baseline prediction from the provided train/test files when no blend files are found. This baseline uses per-(R,C,time_step,u_out) median pressure from train and otherwise backs off to per-(R,C,time_step) and then global per-time_step medians—so it runs end-to-end and produces a valid `submission.csv`. I also fix the undefined `a, b` usage by making that cell optional and non-failing.'
- What this solution (achieved 9.00291) has done: 'Your current score (9.0062 MAE; lower is better) is far from the target (0.1533), so we should improve meaningfully while keeping your “train/test-only fallback + snap-to-nearest-pressure” core approach intact. The biggest issue in the fallback is that it ignores the sequential nature of a breath; adding simple lag features via previous `u_in/u_out` and `time_step` within each `breath_id` and using a slightly richer group-median lookup (still just medians, no model) typically yields a large MAE drop on this competition. I keep your blending logic unchanged, but upgrade only the fallback baseline to use these breath-wise lag keys with safe backoff, and I make sure the submission remains correctly aligned to `id` and written to `submission.csv`. This stays fast (groupby/merge only), deterministic, and preserves evaluation semantics (still snapping to valid pressure levels).'
- What this solution (achieved 3.7753) has done: 'Your current fallback baseline is still too coarse for this competition (MAE ~9), mainly because it ignores the “inspiratory only” scoring mask (`u_out==0`) and doesn’t exploit strong within-breath structure. To move the score sharply toward the 0.153 target while preserving your core “lookup medians + backoff + snap-to-nearest-pressure” approach (no model/training loop changes), I upgrade the fallback to (1) learn medians only on inspiratory rows and (2) use cumulative-within-breath signals (cumulative `u_in`, time index) plus a couple of lagged `u_in` values for more specific keys, with safe backoff. I also enforce the known physical constraint that expiratory phase isn’t scored by simply setting `pressure=0` when `u_out==1`, which typically reduces MAE substantially without changing evaluation semantics. Blending logic and snapping remain intact, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 3.77494) has done: 'Your current MAE (3.7753, lower-is-better) is still far from the target (0.1533), so we need a meaningful improvement while keeping your core “median lookup + backoff + snap-to-nearest-pressure” logic intact. The biggest easy win is to stop forcing `pressure=0` for `u_out==1` (that hurts because Kaggle evaluates only inspiratory rows rather than treating expiratory as zero-error), and instead simply predict normally for all rows while still *training the lookups on inspiratory rows only* to match the metric focus. I also add one extra within-breath feature (`u_in_lag2_r`) into the most specific lookup key to better capture dynamics with minimal additional complexity and runtime. Everything else (blending logic, snapping, I/O paths, submission writing) remains the same.'
- What this solution (achieved 3.77486) has done: 'Your current score (3.77494 MAE; lower is better) is still far above the target (0.1533), so we should improve meaningfully while keeping your core “group-median lookup + backoff + snap-to-nearest-pressure” approach unchanged. The largest remaining gain with minimal logic change is to add an additional, very lightweight signal that better captures breath progression: a rounded delta of `u_in` (first difference) within each breath, and include it only in the most-specific median key (with the same safe backoff chain as before). This preserves the exact same overall method (pure lookups/medians, no model), keeps runtime low, and tends to reduce MAE by making the top-level lookup less ambiguous. Everything else (file handling, blending fallback behavior, snapping, and writing `submission.csv`) stays intact.'
- What this solution (achieved 3.75822) has done: 'Your current MAE (3.77486) is still far above the target (0.1533), so we should improve the fallback baseline while preserving your existing “group-median lookup + backoff + snap-to-nearest-pressure” core logic. The biggest low-risk win is to stop learning/predicting “raw pressure” directly and instead learn the *residual above a simple physical baseline* `u_in / C` (per row) using the same median-lookup/backoff chain; this keeps the same semantics (still medians + backoff + snapping) but typically reduces error a lot. Concretely, we compute `res = pressure - u_in/C`, build the same group-median tables on `res` (inspiratory rows only), then predict `pressure = u_in/C + res_pred` and snap to valid pressure levels. Blending code/path handling stays unchanged, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 3.75822) has done: 'We need to move MAE down from 3.758 toward 0.153, so we should improve the existing fallback (median-lookup + backoff + snap) without changing the overall approach. The largest remaining low-risk gain is to ensure we predict only on the inspiratory phase correctly and leverage the known discrete pressure grid more strongly by predicting **pressure class** via residual snapping that is conditioned on (R,C,t_idx) rather than a single global snap. Concretely, we keep your residual-above-`u_in/C` idea, but (1) build per-(R,C,t_idx) residual medians and (2) snap using the closest **valid pressure values observed for that (R,C)** pair (backing off to global grid if unseen). This preserves your core logic (pure group medians + backoff + snapping) and typically improves MAE substantially with minimal extra runtime. The blending path behavior and submission writing remain unchanged and still produce `submission.csv` end-to-end.'
- What this solution (achieved 3.89743) has done: 'We need to reduce MAE from 3.758 toward 0.153 (lower is better), so we should strengthen the existing fallback while preserving your core “median lookup + backoff + snap-to-valid-grid” logic. The biggest minimal win is to make the residual baseline a bit more physically appropriate by adding an `R * flow` term using an approximate flow proxy (`u_in`), then learn/smooth only the remaining residual with the same group-median/backoff chain. We keep all I/O paths, keep inspiratory-only rows for fitting the lookup tables, and keep the same (R,C)-conditioned snapping grid. The change is small (one extra baseline term + medians computed on a smaller-magnitude residual) and should move the score down without introducing new modeling/training machinery.'
- What this solution (achieved 3.8815) has done: 'Your current MAE (3.897) is still far above the target (0.153, lower is better), so we should make a small, low-risk improvement inside the existing fallback (median lookup + backoff + snap) rather than changing the overall approach. The biggest issue is that the added “R*u_in” baseline term is not physically well-aligned and likely harms accuracy; I keep the residual-baseline idea but switch to a breath-wise **volume proxy** (`u_in * dt` cumulative) and use `R * flow_proxy` where `flow_proxy ≈ dV/dt`, then fit its coefficient exactly the same way as your current `k` fit (still just one scalar OLS). Everything else stays the same: inspiratory-only rows for fitting tables, same lookup/backoff chain, same (R,C)-conditioned pressure snapping, and the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 3.40608) has done: 'We need to move MAE down from 3.8815 toward 0.1533 (lower is better), and your current approach is a pure median-lookup fallback with snapping, so the smallest meaningful gain is to fix two low-level issues that degrade that fallback. First, your fallback’s snapping loop is Python-level over 603,600 rows; replacing it with a vectorized nearest-grid snap for the common case (global grid) and a small grouped snap for (R,C) grids keeps identical semantics but avoids timeouts and lets us safely add a slightly richer lookup without exceeding runtime. Second, your current “flow_proxy” ends up very close to `u_in` (because V is cumsum(u_in*dt) then diff/dt), so the fitted k-term contributes little; we keep the same exact single-scalar OLS baseline idea but compute `flow_proxy` directly as `u_in` smoothed with a 3-step within-breath rolling mean (still deterministic, still derived only from inputs), which typically reduces noise and improves the residual medians. Everything else (inspiratory-only fitting, backoff chain, snapping to valid pressure levels, and writing `submission.csv`) stays the same.'
- What this solution (achieved 3.77486) has done: 'We need to reduce MAE from 3.406 toward 0.153 (lower is better), so the fallback baseline must become more faithful to the pressure dynamics while keeping your core “median lookup + backoff + snap-to-valid-grid” approach intact. The smallest high-impact fix is to stop using the ad‑hoc baseline (`u_in/C + k*R*flow_proxy`) and instead learn pressure directly via grouped medians of **pressure** (not residuals), conditioned on strong, known drivers `(R,C,t_idx,u_in)` plus a couple of breath-wise lag/cum features, with the same backoff chain and the same snapping. This preserves the same modeling family (pure lookups/medians, no training loop/model), but typically moves this competition’s score down substantially. I also replace the remaining `.apply(... in dict)` row-wise mask with a vectorized key join for speed/stability without changing outputs.'
- What this solution (achieved 4.02513) has done: 'Your current score (3.77486 MAE; lower is better) is still far above the target (0.1533), so we need a meaningful accuracy gain while keeping your “median lookup + backoff + snap-to-valid-grid” core approach intact. The smallest high-impact fix is to learn and predict **delta-pressure increments** within each breath (pressure differences between consecutive timesteps) rather than absolute pressure, then reconstruct pressure via cumulative sum; this preserves the same lookup/backoff mechanics but better matches the breath dynamics. We keep inspiratory-only rows (`u_out==0`) for fitting lookup tables (as you already do), keep the same backoff chain idea, and keep the same (R,C)-conditioned snapping to valid pressure levels. Blending logic and I/O remain unchanged, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 3.93797) has done: 'Your current MAE (4.025) is far above the target (0.153, lower-is-better), so we need a meaningful accuracy improvement while keeping your existing “median lookup + backoff + snap-to-valid-grid” core approach intact. The smallest high-impact fix is to stop rounding continuous control features (`u_in`, lags, cumulative, delta) because that collapses many distinct states into the same lookup bucket and harms median specificity; we keep the exact same dp-lookup/cumsum reconstruction, but use unrounded floats for the most-specific keys and only use rounded `u_in` in the coarser backoff tables to avoid over-sparsity. We also make the dp lookup more stable by adding a lightweight extra backoff level that uses `(R,C,t_idx,u_in,u_in_lag1)` before dropping lags entirely—still the same median/backoff mechanics, just a slightly better chain. All I/O, blending behavior, snapping semantics, and `submission.csv` writing remain unchanged.'
- What this solution (achieved 3.66968) has done: 'We keep your existing “dp lookup + backoff + per-(R,C) snapping” approach, but make one minimal change that typically reduces MAE a lot on this competition: avoid using raw float keys (`u_in`, lags, cum, diff) in the most-specific groupby/merge, because train/test rarely match exactly at float precision so the top tables become almost entirely NaN and you fall back to coarse dp medians. Concretely, we quantize those continuous features only for the lookup keys (not changing the overall logic), using a small bin size to preserve resolution while ensuring join hit-rate. Everything else (inspiratory-only fitting, same backoff chain, same reconstruction, same snapping, same submission writing) remains intact and the script still runs end-to-end even if `../input/gb-blending` is missing.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd
import gc



## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )




## === cell 2
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Read 1-2 submission files and do a simple weighted combination based on the filename "score" token.
    Bugfix: make parsing robust; if score can't be parsed, fall back to equal weights.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        base = os.path.basename(input_list[i])
        score = None
        try:
            parts = base.split(".")
            if len(parts) >= 2:
                score_token = parts[1].split(" ")[0]
                score = float(score_token)
        except Exception:
            score = None
        l.append(score)
        preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())

    if len(preds) == 1:
        return preds[0]

    if (l[0] is None) or (l[1] is None) or (l[0] + l[1] == 0):
        weight1, weight2 = 0.5, 0.5
    else:
        l_sum = l[0] + l[1]
        weight1 = (l[1] / l_sum) + 0.1
        weight1 = min(max(weight1, 0.0), 1.0)
        weight2 = 1 - weight1

    return preds[0] * weight1 + preds[1] * weight2


def _vectorized_find_nearest_global(pred, grid_sorted):
    """
    Vectorized nearest-neighbor snapping onto a sorted pressure grid.
    Identical semantics to per-row find_nearest for the same grid, but avoids a python loop.
    """
    pred = np.asarray(pred, dtype=np.float64)
    idx = np.searchsorted(grid_sorted, pred, side="left")
    idx = idx.clip(0, grid_sorted.shape[0] - 1)

    prev_idx = (idx - 1).clip(0, grid_sorted.shape[0] - 1)
    next_idx = idx

    prev_val = grid_sorted[prev_idx]
    next_val = grid_sorted[next_idx]

    choose_prev = (idx > 0) & (np.abs(prev_val - pred) < np.abs(next_val - pred))
    out = np.where(choose_prev, prev_val, next_val)
    return out.astype(np.float64)


def _fallback_baseline_submission():
    """
    Core logic preserved: (median lookup + backoff + snap-to-valid-pressure-grid),
    still using dp prediction + within-breath cumulative reconstruction.

    Change (expected MAE improvement toward target, without changing overall approach):
      - Quantize continuous control features for the lookup keys to increase train/test key matches.
        Using raw floats as join keys makes the most-specific tables almost entirely miss and forces
        coarse backoff; small-bin quantization preserves resolution but greatly improves hit-rate.
      - Keep rounded u_in in coarser backoff tables (as before).
      - Keep the same backoff chain, reconstruction, and snapping.
    """
    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()
    te = df_test[["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]].copy()

    tr.sort_values(["breath_id", "time_step"], inplace=True)
    te.sort_values(["breath_id", "time_step"], inplace=True)

    tr["t_idx"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["t_idx"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_lag1"] = tr.groupby("breath_id")["u_in"].shift(1)
    tr["u_in_lag2"] = tr.groupby("breath_id")["u_in"].shift(2)
    te["u_in_lag1"] = te.groupby("breath_id")["u_in"].shift(1)
    te["u_in_lag2"] = te.groupby("breath_id")["u_in"].shift(2)

    tr["u_in_cum"] = tr.groupby("breath_id")["u_in"].cumsum()
    te["u_in_cum"] = te.groupby("breath_id")["u_in"].cumsum()

    tr["du_in"] = tr.groupby("breath_id")["u_in"].diff()
    te["du_in"] = te.groupby("breath_id")["u_in"].diff()

    tr["u_in_r"] = tr["u_in"].round(1)
    te["u_in_r"] = te["u_in"].round(1)

    def _q(series, step):
        s = series.to_numpy(dtype=np.float64, copy=False)
        out = np.round(s / step) * step
        return out.astype(np.float64)

    tr["u_in_q"] = _q(tr["u_in"], 0.1)
    te["u_in_q"] = _q(te["u_in"], 0.1)

    tr["u_in_lag1_q"] = _q(tr["u_in_lag1"], 0.1)
    tr["u_in_lag2_q"] = _q(tr["u_in_lag2"], 0.1)
    te["u_in_lag1_q"] = _q(te["u_in_lag1"], 0.1)
    te["u_in_lag2_q"] = _q(te["u_in_lag2"], 0.1)

    tr["u_in_cum_q"] = _q(tr["u_in_cum"], 1.0)
    te["u_in_cum_q"] = _q(te["u_in_cum"], 1.0)

    tr["du_in_q"] = _q(tr["du_in"], 0.5)
    te["du_in_q"] = _q(te["du_in"], 0.5)

    tri = tr[tr["u_out"] == 0].copy()

    tri["dp"] = tri.groupby("breath_id")["pressure"].diff()
    tri["p0"] = tri.groupby("breath_id")["pressure"].transform("first")

    p0_g2 = (
        tri[tri["t_idx"] == 0]
        .groupby(["R", "C", "u_in_q"], sort=False)["pressure"]
        .median()
        .rename("p0_2")
        .reset_index()
    )
    p0_g2b = (
        tri[tri["t_idx"] == 0]
        .groupby(["R", "C", "u_in_r"], sort=False)["pressure"]
        .median()
        .rename("p0_2b")
        .reset_index()
    )
    p0_g3 = (
        tri[tri["t_idx"] == 0]
        .groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .rename("p0_3")
        .reset_index()
    )
    p0_global = float(tri[tri["t_idx"] == 0]["pressure"].median())

    tri_dp = tri[tri["t_idx"] > 0].copy()

    g0 = (
        tri_dp.groupby(
            [
                "R",
                "C",
                "t_idx",
                "u_in_q",
                "u_in_lag1_q",
                "u_in_lag2_q",
                "u_in_cum_q",
                "du_in_q",
            ],
            sort=False,
        )["dp"]
        .median()
        .rename("dp0")
        .reset_index()
    )

    g0b = (
        tri_dp.groupby(["R", "C", "t_idx", "u_in_q", "u_in_lag1_q"], sort=False)["dp"]
        .median()
        .rename("dp0b")
        .reset_index()
    )

    g1 = (
        tri_dp.groupby(["R", "C", "t_idx", "u_in_r"], sort=False)["dp"]
        .median()
        .rename("dp1")
        .reset_index()
    )
    g2 = (
        tri_dp.groupby(["R", "C", "t_idx"], sort=False)["dp"]
        .median()
        .rename("dp2")
        .reset_index()
    )
    g3 = (
        tri_dp.groupby(["t_idx"], sort=False)["dp"].median().rename("dp3").reset_index()
    )

    m = (
        te[
            [
                "id",
                "breath_id",
                "R",
                "C",
                "t_idx",
                "u_in_q",
                "u_in_lag1_q",
                "u_in_lag2_q",
                "u_in_cum_q",
                "du_in_q",
                "u_in_r",
            ]
        ]
        .merge(
            g0,
            on=[
                "R",
                "C",
                "t_idx",
                "u_in_q",
                "u_in_lag1_q",
                "u_in_lag2_q",
                "u_in_cum_q",
                "du_in_q",
            ],
            how="left",
        )
        .merge(g0b, on=["R", "C", "t_idx", "u_in_q", "u_in_lag1_q"], how="left")
        .merge(g1, on=["R", "C", "t_idx", "u_in_r"], how="left")
        .merge(g2, on=["R", "C", "t_idx"], how="left")
        .merge(g3, on=["t_idx"], how="left")
    )

    dp = m["dp0"]
    dp = dp.fillna(m["dp0b"])
    dp = dp.fillna(m["dp1"])
    dp = dp.fillna(m["dp2"])
    dp = dp.fillna(m["dp3"])
    dp = dp.fillna(float(tri_dp["dp"].median()))
    dp = dp.to_numpy(dtype=np.float64)

    m0 = (
        m[m["t_idx"] == 0][["id", "R", "C", "u_in_q", "u_in_r"]]
        .merge(p0_g2, on=["R", "C", "u_in_q"], how="left")
        .merge(p0_g2b, on=["R", "C", "u_in_r"], how="left")
        .merge(p0_g3, on=["R", "C"], how="left")
    )
    p0_pred = (
        m0["p0_2"]
        .fillna(m0["p0_2b"])
        .fillna(m0["p0_3"])
        .fillna(p0_global)
        .to_numpy(dtype=np.float64)
    )

    p_abs = np.empty(m.shape[0], dtype=np.float64)
    p_abs[:] = 0.0
    idx0 = m.index[m["t_idx"] == 0].to_numpy()
    p_abs[idx0] = p0_pred

    dp_for_cumsum = dp.copy()
    dp_for_cumsum[m["t_idx"].to_numpy() == 0] = 0.0
    cum_dp = (
        pd.Series(dp_for_cumsum)
        .groupby(m["breath_id"])
        .cumsum()
        .to_numpy(dtype=np.float64)
    )

    p0_by_row = (
        pd.Series(p_abs)
        .groupby(m["breath_id"])
        .transform("first")
        .to_numpy(dtype=np.float64)
    )
    pred = p0_by_row + cum_dp

    rc_to_grid = (
        df_train.groupby(["R", "C"])["pressure"]
        .unique()
        .apply(lambda x: np.sort(np.asarray(x, dtype=np.float64)))
        .to_dict()
    )

    snapped = np.empty_like(pred, dtype=np.float64)

    rc_pairs = m[["R", "C"]].astype(np.int16).to_numpy()
    rc_key_arr = rc_pairs[:, 0].astype(np.int32) * 1000 + rc_pairs[:, 1].astype(
        np.int32
    )
    known_keys = np.array(
        [int(R) * 1000 + int(C) for (R, C) in rc_to_grid.keys()], dtype=np.int32
    )
    known_mask = np.isin(rc_key_arr, known_keys)

    if (~known_mask).any():
        snapped[~known_mask] = _vectorized_find_nearest_global(
            pred[~known_mask], sorted_pressures
        )

    if known_mask.any():
        known_idx = np.nonzero(known_mask)[0]
        known_df = m.iloc[known_idx][["R", "C"]].copy()
        known_df["idx"] = known_idx
        for (R, C), grp in known_df.groupby(["R", "C"], sort=False):
            idxs = grp["idx"].to_numpy()
            grid = rc_to_grid[(int(R), int(C))]
            snapped[idxs] = _vectorized_find_nearest_global(pred[idxs], grid)

    pred_by_id = pd.DataFrame({"id": m["id"].to_numpy(), "pressure": snapped})
    pred_by_id.sort_values("id", inplace=True)

    sub = sub.merge(pred_by_id, on="id", how="left", suffixes=("", "_pred"))
    sub["pressure"] = sub["pressure_pred"]
    sub.drop(columns=["pressure_pred"], inplace=True)

    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    """
    Original intent: read multiple submission files under dp, ensemble them with random weights,
    take median of ensembles, snap to nearest valid pressure, write CSV.

    Bugfix: handle missing/empty dp and ensure all predictions have correct length before stacking.
    If no valid files exist, fall back to a train/test-only baseline to still yield a submission.
    """
    files = sorted([p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)])
    if len(files) == 0:
        print(
            f"[g] No files found in {dp}. Creating fallback baseline submission.csv instead."
        )
        return _fallback_baseline_submission()

    file_count = len(files)
    loop_time = 100

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) > 0:
            flist.append(chunk)

    flist = [wc(chunk) for chunk in flist]

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    n = len(output)
    good = []
    for arr in flist:
        arr = np.asarray(arr).ravel()
        if arr.shape[0] == n:
            good.append(arr)

    if len(good) == 0:
        print(
            f"[g] No valid prediction files with length {n} in {dp}. Using fallback baseline."
        )
        return _fallback_baseline_submission()

    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(good))]
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(good)):
            temp += good[j] * weight[j]
        pred_list.append(temp)

        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
g("../input/gb-blending")



## === cell 4
a = None
b = None
if (
    isinstance(a, str)
    and isinstance(b, str)
    and os.path.exists(a)
    and os.path.exists(b)
):
    blend(a, b)
else:
    print("[blend] Skipped because a/b are not set to existing file paths.")
