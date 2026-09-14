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

0.1492447888535536

# 6. Current score

1.60794

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.12137) has done: 'I fix the failure by making the blending code robust when the expected `../input/gb-blending` directory doesn’t exist (or has no valid submission files), which currently leads to a 1-element prediction array and a length mismatch. To ensure the notebook always runs end-to-end and produces a valid `.csv` submission, I add a safe fallback that generates a baseline submission from the provided `test.csv` (per-breath mean pressure by `(R,C,time_step,u_in,u_out)` with sensible fallbacks). I also make file discovery stricter (only CSVs with the right length/columns are used) and always write `submission.csv` with the required `id,pressure` columns. These changes keep the original blending core logic intact when blend inputs are present, and otherwise avoid crashing while producing a reasonable score-improving baseline versus all-zeros.'
- What this solution (achieved 4.71809) has done: 'Your current score (4.12137, lower is better) is far from the target (0.1492), and the main issue is that your fallback is effectively an unconditional lookup/average that ignores the per-breath dynamics and the inspiratory-only scoring nuance, which leads to very poor MAE. I keep your blending logic intact, but upgrade only the fallback to a stronger yet still simple, fully “tabular” approach: build a per-(R,C) mean pressure curve over `time_step` during inspiration (`u_out==0`) and apply a small correction based on `u_in` via a lightweight linear coefficient learned from train residuals. I also ensure the fallback operates in a vectorized way (no Python loop over 603600 rows) and still snaps to the nearest allowed pressure values to match the label discretization. This should materially reduce MAE while keeping the rest of your pipeline unchanged and still producing `submission.csv` end-to-end.'
- What this solution (achieved 4.71809) has done: 'You’re far from the target MAE, so the safest way to move toward it without changing your blending core logic is to strengthen only the fallback predictor (used when no valid blend CSVs are found). I keep your aggregation-based approach but fix two score-hurting issues: (1) the fallback currently doesn’t enforce the known physical constraint that pressure is ~0 during expiration (`u_out==1`), and (2) the “small linear correction vs `u_in`” uses slow/unstable `groupby.apply` and imperfect key alignment, which can degrade the learned correction. I replace the slope computation with a fully vectorized per-group covariance/variance using `groupby.agg`, and set expiratory predictions to 0 (then snap to the nearest allowed pressure, which pick the minimum grid value). I also fix a typo bug (`pd.Data.DataFrame`) in the defensive length-mismatch fallback path to ensure the notebook always writes a valid `submission.csv`.'
- What this solution (achieved 5.0112) has done: 'Your current MAE (4.718) is far worse than the target (0.149), so we should improve score by strengthening only the fallback path (used when no valid blend CSVs exist) while keeping the blending logic intact. The main score issue is that the fallback predicts pressure during expiration even though those timesteps are not scored and the true pressure is near the minimum grid value; we set `u_out==1` predictions explicitly to the minimum pressure from the discrete grid (not hardcoded 0) to better match training distribution. We also reduce key-mismatch noise by building the base curve and correction on an integer time index (`time_step_idx` derived from median timestep) instead of rounding floats, which improves train/test alignment without changing the overall aggregation-based approach. Finally, we ensure prediction alignment by merging on `id` (not relying on positional alignment after sorts), preventing subtle misalignment that can severely hurt MAE.'
- What this solution (achieved 5.0112) has done: 'Your current MAE (5.0112, lower is better) is far above the target (0.1492), so we should improve score by strengthening only the fallback path that runs when no valid blend CSVs are found, while leaving the blending logic and snapping semantics intact. The main minimal win is to fix the per-row keying bug in `_build_fallback_predictions()` (it currently maps with a `Series(list(zip(...)))`, which is slow and can silently misalign); switching to a proper `MultiIndex`-based reindex gives correct, vectorized alignment and typically reduces MAE materially. I also add a tiny “hold-last-pressure during expiration” fallback (per-breath forward-fill) before snapping, which preserves the same aggregation/correction idea but better matches how pressure behaves when `u_out==1`. All outputs remain a valid `submission.csv` with `id,pressure`, and if blend files exist the original blending path is unchanged.'
- What this solution (achieved 5.0112) has done: 'Your current MAE (5.0112, lower is better) is far above the target (0.1492), so we should improve the fallback predictor (used when no valid blend CSVs are present) while keeping your blending logic unchanged. The biggest minimal gain is to stop training-time “expiration forward-fill” from leaking into scored inspiratory steps: we instead explicitly build the prediction in two phases—predict only inspiration (`u_out==0`) from the learned curve+u_in correction, and set expiration (`u_out==1`) to the minimum pressure grid value (since expiration is not scored and this reduces harmful drift). We also make the correction more stable by learning the `u_in` slope per `(R,C,time_step_idx)` using only inspiratory rows and applying it only where `u_out==0`, avoiding noisy corrections during expiration. All outputs remain snapped to the discrete pressure grid and the script still always writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.42011) has done: 'I keep your blending logic unchanged and only strengthen the fallback predictor path (used when `../input/gb-blending` has no valid submissions), because your current MAE is far above the target and most likely you’re effectively submitting the weak fallback. The minimal score-relevant change is to make the fallback respect the known “pressure grid” and the strong per-breath structure: I learn a per-(R,C,time_step_idx) mean inspiratory curve plus a small linear residual correction on `u_in` exactly as you already do, but I also add two tiny, competition-specific features (`u_in` cumulative sum and lag-1 `u_in`) into that linear correction in a purely aggregated way (no new model/loops), which is still the same tabular residual-correction core. I also ensure the time index alignment is stable by clipping `time_step_idx` to the observed range per breath length (80 steps), avoiding key misses that inflate MAE. The submission writing, id alignment, and snapping semantics remain the same, and it still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 5.4292) has done: 'Your current MAE (5.42011, lower is better) is far from the target (0.1492), and the most likely reason is that you’re effectively using the fallback predictor (no valid blend CSVs), so we should improve only that fallback while leaving the blending logic and snapping semantics unchanged. The smallest score-relevant fix is to make the fallback learn from the same feature space used at inference: compute the base curve and residual correction on the full train set but fit the residual slopes only on inspiratory rows and apply them only to inspiratory test rows, while keeping expiration fixed to `MIN_PRESSURE`. Additionally, we should ensure the time index alignment is exact for this dataset by using the known 80-step structure (rather than estimating `dt` from a single breath), which reduces key misses that inflate MAE. These changes preserve your “aggregated mean curve + linear residual correction + snap-to-grid” core logic and should move the score materially toward the target.'
- What this solution (achieved 3.19147) has done: 'Your current MAE (5.4292, lower is better) is far above the target (0.1492), so we should improve only the fallback path (likely the one being used if `../input/gb-blending` has no valid CSVs) while keeping your blending logic and snapping-to-grid semantics intact. The smallest high-impact fix is to stop using a global mean curve and instead condition the base curve on the control input by binning `u_in` (still an aggregated lookup, not a new model), which better matches the dynamics and typically drops MAE a lot. To avoid score-harming key misses, we keep the same 80-step `time_step_idx`, use vectorized MultiIndex reindexing, and keep expiration predictions at `MIN_PRESSURE`. Finally, we preserve your existing residual linear correction, but compute it against the improved binned base curve to make the correction smaller/more stable.'
- What this solution (achieved 2.6005) has done: 'Your current MAE (3.19, lower is better) is still far from the target (0.149), and the most likely limiter is the fallback predictor quality (since `../input/gb-blending` typically doesn’t exist). I keep your blending path and “aggregated lookup + linear residual correction + snap-to-grid” core logic unchanged, but make the fallback’s u_in conditioning slightly richer in a still-tabular way: use a *2D bin* on `(u_in, u_in_lag1)` for the base curve instead of only `u_in`. This is a minimal extension of the same groupby-mean lookup (no new model/training loop), and it should reduce MAE by better capturing valve dynamics while preserving expiration handling and snapping semantics. I also add a robust global fallback for any remaining key misses and keep the output aligned by `id` and written to `submission.csv`.'
- What this solution (achieved 1.99643) has done: 'Your current MAE (2.6005, lower is better) is still far above the target (0.1492), so we should improve only the fallback predictor quality (since `../input/gb-blending` typically doesn’t exist) while leaving the blending logic intact. The smallest high-impact change consistent with your “aggregated lookup + residual linear correction + snap-to-grid” core is to (1) predict inspiration (`u_out==0`) and expiration separately, but for expiration use a *data-driven* per-(R,C,time_step_idx) mean pressure from training expiration instead of forcing `MIN_PRESSURE`. Additionally, we can reduce key-miss noise by building the base curve on a slightly richer but still tabular key: `(R,C,time_step_idx,u_in_bin,u_in_lag1_bin,u_in_sum_bin)` where `u_in_sum_bin` is a coarse bin of cumulative `u_in` capturing within-breath filling without changing the approach. Everything remains vectorized, keeps the same snapping semantics, and still writes a valid `submission.csv`.'
- What this solution (achieved 2.39394) has done: 'Your current MAE (1.996) is still far from the target (0.149, lower is better), so we should cautiously improve the fallback predictor (likely what you’re using when no blend files exist) without changing the overall “aggregated lookup + residual linear correction + snap-to-grid” approach. The smallest high-impact change is to reduce key-misses and noise from coarse binning by replacing the uniform-width `u_in` bins with data-driven quantile bins (computed from train inspiration) for `u_in`, `u_in_lag1`, and `u_in_cumsum`, while keeping the exact same groupby-mean lookup semantics. This tends to improve coverage (fewer NaNs → fewer global-mean fallbacks) and makes the base curve closer to the true conditional mean, which should reduce MAE. Everything else (blending path, correction computation, snapping, file writing, and paths) remains unchanged and still produces a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.76321) has done: 'I keep your blending path intact and only adjust the fallback (which you’re likely using) to reduce MAE toward the target by cutting key-misses and better matching the pressure dynamics during inspiration. The minimal score-relevant change is to replace the very high-cardinality 3D quantile binning with a lower-cardinality, more stable quantile binning on `u_in` and `u_in_cumsum` only (dropping `u_in_lag1_bin` from the base lookup), while keeping the same aggregated-lookup + linear residual correction + snap-to-grid core. This typically improves generalization and reduces the “global mean fallback” frequency that inflates MAE. Everything still runs end-to-end, preserves evaluation semantics, and always writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.60794) has done: 'Your current MAE (1.763) is still much worse than the target (0.149, lower is better), so we should improve only the fallback predictor (since `../input/gb-blending` likely has no valid CSVs) while keeping your blending path and “aggregated lookup + linear residual correction + snap-to-grid” core logic intact. The smallest high-impact change is to reduce over-fragmentation in the base lookup by lowering bin counts (better coverage → fewer global-mean fallbacks) and to add back a light `u_in_lag1` conditioning into the base key with very low cardinality (captures valve dynamics without changing the approach). We also ensure expiration predictions remain data-driven (from `exp_curve`) as you already do, and keep the same residual correction and snapping semantics. This should move MAE down toward the target without changing architecture/training loops or introducing approximations.'

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
MIN_PRESSURE = float(sorted_pressures[0])


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: read 1-2 submission files and compute a weighted combination.
    Bugfix: make parsing of score-in-filename optional/robust; validate length/columns.
    """
    arrs = []
    scores = []
    for path in input_list:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "pressure" not in df.columns:
            continue
        p = np.asarray(df["pressure"]).ravel()
        if p.shape[0] != 603600:
            continue

        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        scores.append(public_lb_score)
        arrs.append(p)

    if len(arrs) == 0:
        return None

    if len(arrs) == 1:
        return arrs[0]

    l_sum = sum(scores) if sum(scores) != 0 else 1
    weight1 = (scores[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return arrs[0] * weight1 + arrs[1] * weight2


def _snap_to_nearest_pressure_vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorized snapping to the discrete pressure grid (same semantics as find_nearest,
    but much faster and avoids a Python loop over 603600 rows).
    """
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, x, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    prev_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    next_val = sorted_pressures[idx]
    prev_val = sorted_pressures[prev_idx]

    choose_prev = (idx > 0) & (np.abs(prev_val - x) < np.abs(next_val - x))
    out = next_val.copy()
    out[choose_prev] = prev_val[choose_prev]
    return out.astype(np.float64)


def _quantile_bin_from_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Assign quantile bins with precomputed edges. Returns integer bin indices in [0, n_bins-1].
    """
    x = np.asarray(x, dtype=np.float64)
    b = np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
    return b


def _build_fallback_predictions():
    """
    Fallback used ONLY when blend inputs are missing/invalid.

    Core logic preserved: aggregated lookup base curve + linear residual correction + snap-to-grid.

    Change (score, minimal and still same approach):
    - Reduce quantile-bin cardinality to improve coverage/generalization (fewer key misses -> lower MAE).
    - Add a very low-cardinality u_in_lag1 bin into the base key to capture short-term dynamics
      without changing the modeling approach (still a groupby-mean lookup).
    """
    test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )
    train = df_train

    dt = 0.03
    dt_inv = 1.0 / dt

    te = test.copy()

    train_idx = np.rint(train["time_step"].to_numpy(dtype=np.float64) * dt_inv).astype(
        np.int16
    )
    test_idx = np.rint(te["time_step"].to_numpy(dtype=np.float64) * dt_inv).astype(
        np.int16
    )
    train_idx = np.clip(train_idx, 0, 79)
    test_idx = np.clip(test_idx, 0, 79)

    tr = train[["breath_id", "R", "C", "u_in", "u_out", "pressure"]].copy()
    tr["time_step_idx"] = train_idx
    te["time_step_idx"] = test_idx

    tr = tr.sort_values(["breath_id", "time_step_idx"], kind="mergesort")
    te = te.sort_values(["breath_id", "time_step_idx"], kind="mergesort")

    tr["u_in_lag1"] = tr.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
    te["u_in_lag1"] = te.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)

    tr["u_in_cumsum"] = tr.groupby("breath_id", sort=False)["u_in"].cumsum()
    te["u_in_cumsum"] = te.groupby("breath_id", sort=False)["u_in"].cumsum()

    tr_insp = tr[tr["u_out"] == 0].copy()
    tr_exp = tr[tr["u_out"] == 1].copy()

    global_mean = float(tr["pressure"].mean())
    global_insp_mean = (
        float(tr_insp["pressure"].mean()) if len(tr_insp) else global_mean
    )
    global_exp_mean = float(tr_exp["pressure"].mean()) if len(tr_exp) else MIN_PRESSURE

    N_BINS_UIN = 25
    N_BINS_SUM = 31
    N_BINS_LAG1 = 9

    def _safe_qcut_edges(x: pd.Series, n_bins: int) -> np.ndarray:
        x = x.to_numpy(dtype=np.float64)
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x, qs)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges.astype(np.float64)

    uin_edges = _safe_qcut_edges(tr_insp["u_in"], N_BINS_UIN)
    sum_edges = _safe_qcut_edges(tr_insp["u_in_cumsum"], N_BINS_SUM)
    lag1_edges = _safe_qcut_edges(tr_insp["u_in_lag1"], N_BINS_LAG1)

    tr["u_in_bin"] = _quantile_bin_from_edges(tr["u_in"].to_numpy(), uin_edges)
    te["u_in_bin"] = _quantile_bin_from_edges(te["u_in"].to_numpy(), uin_edges)

    tr["u_in_sum_bin"] = _quantile_bin_from_edges(
        tr["u_in_cumsum"].to_numpy(), sum_edges
    )
    te["u_in_sum_bin"] = _quantile_bin_from_edges(
        te["u_in_cumsum"].to_numpy(), sum_edges
    )

    tr["u_in_lag1_bin"] = _quantile_bin_from_edges(
        tr["u_in_lag1"].to_numpy(), lag1_edges
    )
    te["u_in_lag1_bin"] = _quantile_bin_from_edges(
        te["u_in_lag1"].to_numpy(), lag1_edges
    )

    tr_insp = tr[tr["u_out"] == 0].copy()
    tr_exp = tr[tr["u_out"] == 1].copy()

    base_key = ["R", "C", "time_step_idx", "u_in_bin", "u_in_sum_bin", "u_in_lag1_bin"]
    base_curve = tr_insp.groupby(base_key, sort=False)["pressure"].mean()

    exp_key = ["R", "C", "time_step_idx"]
    exp_curve = tr_exp.groupby(exp_key, sort=False)["pressure"].mean()

    te_mi_base = pd.MultiIndex.from_frame(te[base_key])
    base_pred = base_curve.reindex(te_mi_base).to_numpy(dtype=np.float64)
    miss = np.isnan(base_pred)
    if miss.any():
        base_pred[miss] = global_insp_mean

    te_u_out = te["u_out"].to_numpy(dtype=np.int64)
    insp_mask = te_u_out == 0

    if np.any(~insp_mask):
        te_mi_exp = pd.MultiIndex.from_frame(te.loc[~insp_mask, exp_key])
        exp_pred = exp_curve.reindex(te_mi_exp).to_numpy(dtype=np.float64)
        exp_pred = np.where(np.isnan(exp_pred), global_exp_mean, exp_pred)
        base_pred[~insp_mask] = exp_pred

    grp_cols = ["R", "C", "time_step_idx"]
    feat_cols = ["u_in", "u_in_lag1", "u_in_cumsum"]

    tr_tmp = tr_insp[
        [
            "R",
            "C",
            "time_step_idx",
            "u_in_bin",
            "u_in_sum_bin",
            "u_in_lag1_bin",
            "u_in",
            "u_in_lag1",
            "u_in_cumsum",
            "pressure",
        ]
    ].copy()
    tr_mi_base = pd.MultiIndex.from_frame(tr_tmp[base_key])

    tr_tmp["base"] = base_curve.reindex(tr_mi_base).to_numpy(dtype=np.float64)
    tr_tmp["base"] = np.where(
        np.isnan(tr_tmp["base"]), global_insp_mean, tr_tmp["base"]
    )
    tr_tmp["resid"] = tr_tmp["pressure"].to_numpy(dtype=np.float64) - tr_tmp["base"]

    feat_means = tr_tmp.groupby(grp_cols, sort=False)[feat_cols].mean()
    tr_mi_grp = pd.MultiIndex.from_frame(tr_tmp[grp_cols])
    tr_feat_means = feat_means.reindex(tr_mi_grp).to_numpy(dtype=np.float64)
    global_feat_means = tr_tmp[feat_cols].mean().to_numpy(dtype=np.float64)
    tr_feat_means = np.where(np.isnan(tr_feat_means), global_feat_means, tr_feat_means)

    X = tr_tmp[feat_cols].to_numpy(dtype=np.float64) - tr_feat_means

    corr = np.zeros(len(te), dtype=np.float64)

    te_mi_grp = pd.MultiIndex.from_frame(te[grp_cols])
    te_feat_means = feat_means.reindex(te_mi_grp).to_numpy(dtype=np.float64)
    te_feat_means = np.where(np.isnan(te_feat_means), global_feat_means, te_feat_means)
    te_X = te[feat_cols].to_numpy(dtype=np.float64) - te_feat_means

    for j, f in enumerate(feat_cols):
        tr_tmp[f"{f}_c"] = X[:, j]
        tr_tmp[f"{f}_c2"] = tr_tmp[f"{f}_c"] ** 2
        tr_tmp[f"{f}_c_res"] = tr_tmp[f"{f}_c"] * tr_tmp["resid"]

        stats = tr_tmp.groupby(grp_cols, sort=False).agg(
            var=(f"{f}_c2", "mean"),
            cov=(f"{f}_c_res", "mean"),
        )
        slope = (stats["cov"] / stats["var"].replace(0.0, np.nan)).fillna(0.0)

        te_slope = slope.reindex(te_mi_grp).to_numpy(dtype=np.float64)
        te_slope = np.where(np.isnan(te_slope), 0.0, te_slope)

        corr[insp_mask] += te_slope[insp_mask] * te_X[insp_mask, j]

    pred = base_pred + corr
    pred = _snap_to_nearest_pressure_vec(pred)
    return te[["id"]].copy(), pred


def g(dp):
    """
    Original blending function.
    Bugfixes:
    - Handle missing/empty directory and invalid files; avoid creating length-1 outputs.
    - Ensure output file is a valid Kaggle submission with .csv suffix (submission.csv).
    - Ensure robust id alignment when filling sample_submission (merge on id).
    """
    files = []
    if dp is not None and os.path.isdir(dp):
        for p in glob.iglob(f"{dp}/*"):
            if os.path.isfile(p) and p.lower().endswith(".csv"):
                files.append(p)
    files.sort()

    if len(files) == 0:
        sub = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        ids_df, pred = _build_fallback_predictions()
        pred_df = ids_df.copy()
        pred_df["pressure"] = pred

        sub = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
        sub["pressure"] = sub["pressure_pred"].to_numpy(dtype=np.float64)
        sub = sub[["id", "pressure"]]
        sub.to_csv("submission.csv", index=False)
        return sub

    file_count = len(files)
    loop_time = 150

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        flist.append(files[start:end])

    preds = []
    for group in flist:
        p = wc(group)
        if p is not None:
            preds.append(p)
    flist = preds

    if len(flist) == 0:
        sub = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        ids_df, pred = _build_fallback_predictions()
        pred_df = ids_df.copy()
        pred_df["pressure"] = pred
        sub = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
        sub["pressure"] = sub["pressure_pred"].to_numpy(dtype=np.float64)
        sub = sub[["id", "pressure"]]
        sub.to_csv("submission.csv", index=False)
        return sub

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    blended = np.median(np.vstack(pred_list), axis=0)

    if blended.shape[0] != len(output):
        sub = output.copy()
        ids_df, pred = _build_fallback_predictions()
        pred_df = ids_df.copy()
        pred_df["pressure"] = pred
        sub = sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
        sub["pressure"] = sub["pressure_pred"].to_numpy(dtype=np.float64)
        sub = sub[["id", "pressure"]]
        sub.to_csv("submission.csv", index=False)
        return sub

    output["pressure"] = blended
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




## === cell 2
g("../input/gb-blending")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
