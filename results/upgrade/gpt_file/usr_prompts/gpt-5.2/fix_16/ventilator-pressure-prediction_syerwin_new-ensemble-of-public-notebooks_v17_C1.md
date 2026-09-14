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

0.156

# 6. Current score

2.54975

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.41508) has done: 'Your notebook fails because it tries to read several external dataset submissions that are not available in your environment, so `sub_1..sub_4` never load and the ensemble step crashes. To make it run end-to-end and produce a valid `submission.csv`, I’m replacing the missing-input ensemble with a self-contained baseline that trains a simple per-(R,C,u_out) mean-pressure lookup from `train.csv` (using only inspiratory samples as the metric scores) and predicts for `test.csv`. This keeps the “postprocess/aggregation” spirit of the original (a lightweight blend/lookup rather than a new model architecture) while yielding a non-trivial score instead of failing. The script also guarantees correct `id` alignment and writes `submission.csv` with the required columns.'
- What this solution (achieved 6.4545) has done: 'Your current lookup is too sparse because it uses `u_in` rounded to 0.1, which creates many unseen bins in test and forces lots of fallbacks to coarse means—this inflates MAE. To move the score down toward 0.156 with minimal change and the same “groupby-mean lookup” core logic, I (1) bin `u_in` more robustly by rounding to the nearest integer (0 decimals), and (2) add a slightly stronger but still lightweight fallback chain that uses `(R,C,u_in_bin)` when `u_out`-specific bins are missing. This keeps the same aggregation approach (no new model/training loop) while improving coverage and stability. The script still write a valid `submission.csv` with correct `id,pressure` columns.'
- What this solution (achieved 6.4545) has done: 'Your current groupby-mean lookup is being hurt most by ID misalignment: in this dataset `id` is only unique within each breath, so merging predictions onto `sample_submission` by `id` collapses many rows and forces repeated/fallback values, inflating MAE. The smallest change that keeps your exact “lookup table + fallback chain” core logic is to stop merging on `id` and instead write predictions in the same row order as `test.csv`/`sample_submission.csv` (they align 1:1). I also add a quick sanity assertion on row counts to prevent silent misalignment. This should move the score down substantially toward your target without changing the modeling approach.'
- What this solution (achieved 6.4545) has done: 'Your lookup-table approach is sound, but it’s currently throwing away almost all training signal by filtering to `u_out==0` while still trying to predict for both phases; since expiratory rows aren’t scored, we should explicitly use inspiratory-only logic at prediction time too. I keep the same groupby-mean core logic and fallback chain, but (1) build lookups from all training rows (so expiratory predictions have a sensible baseline), and (2) set `pressure=0` for `u_out==1` in test (a common, metric-aligned postprocess because those rows are unscored). This minimal change should drop MAE substantially toward your 0.156 target without changing architecture/training. The script still assert row alignment and write a valid `submission.csv`.'
- What this solution (achieved 6.45436) has done: 'Your current score is far above the 0.156 target (lower is better), and the main remaining issue is that a mean-lookup on raw `pressure` tends to predict values between the discrete pressure steps, which is heavily penalized by MAE in this competition. Keeping your exact “groupby-mean lookup + fallback chain + set u_out==1 to 0” core logic, the smallest high-impact change is to quantize (snap) all predicted inspiratory pressures to the nearest pressure value seen in the training set. This preserves evaluation semantics (still predicting pressure per row) while aligning outputs with the discrete label distribution and typically drops MAE a lot. I also add a tiny safety clamp to the train pressure range before snapping to avoid edge-case extrapolation.'
- What this solution (achieved 5.29503) has done: 'Your current MAE is far from the 0.156 target (lower is better), and the biggest remaining weakness in the same “lookup table + fallback + snap-to-grid” logic is that the lookup ignores the strong temporal structure within each breath (pressure depends on recent history, not just the current row). With minimal change and no new model/training loop, I add a tiny amount of lagged context by building the lookup on `(R, C, u_out, u_in_bin, u_in_bin_prev1, u_out_prev1)` computed per `breath_id` in both train and test. I keep your existing fallback chain and snapping, but insert this higher-resolution lookup as the first attempt so coverage improves without changing evaluation semantics. This should materially reduce MAE toward the target while staying within the same aggregation-based approach and still producing a valid `submission.csv`.'
- What this solution (achieved 3.20781) has done: 'Your current lookup is still too “instantaneous”: it only uses the immediately previous step, but pressure in this dataset is strongly driven by recent accumulation (integral of flow), which your model can capture with a tiny, core-logic-preserving addition. I keep the exact same groupby-mean lookup + fallback chain + u_out==1→0 + snap-to-discrete-grid postprocess, but add two lightweight history features (`u_in_bin_prev2` and cumulative `u_in` within breath binned to integer) and use them only as an extra highest-priority lookup level. This should reduce MAE materially (move down toward 0.156) without introducing any new model/training loop or changing evaluation semantics. I also keep the existing row-order alignment assertion and submission writing unchanged.'
- What this solution (achieved 2.44226) has done: 'Your current score (3.20781, lower-is-better) is far above the 0.156 target, so we should make a small change that legitimately improves accuracy without changing your core “multi-level lookup + fallback + u_out==1→0 + snap-to-grid” approach. The biggest low-risk gain is to make the cumulative history feature comparable between train and test by normalizing it to a per-breath time-step count (cumulative sum grows with step index and is sensitive to discretization); we bin it more coarsely so the lag2 lookup has much better coverage and fewer fallbacks. Concretely, we replace `u_in_cumsum` with a coarse, integer-binned `u_in_cumsum_bin` and use it in the highest-priority lookup only; all other lookups/fallbacks remain unchanged. This keeps the same architecture/semantics but should reduce MAE by increasing exact key matches for the best lookup level.'
- What this solution (achieved 2.84125) has done: 'I fix the runtime error by correctly building the per-(R,C) pressure grids: `groupby().unique()` already returns NumPy arrays, so calling `.to_numpy()` crashes and prevents `pressure_grids_rc` from being defined. With that fixed, cell 2 run unchanged and the snapping loop have the required lookup dictionary. I also add a small safety cast to ensure each grid is `float32` and sorted, keeping the exact same “snap-to-discrete-grid” postprocess logic. This is score-neutral-to-positive (it restores the intended RC-conditional snapping) and produce a valid `submission.csv`.'
- What this solution (achieved 2.84125) has done: 'Your current score (2.84125 MAE; lower is better) is still far from the 0.156 target, so we should make a small accuracy-improving change that keeps your exact “multi-level lookup + fallback + u_out==1→0 + snap-to-grid” core logic. The main remaining low-risk gain is to align the lookup target with the evaluation: MAE is computed only on inspiratory rows, so we should build all lookup tables (and RC grids used for snapping) using inspiratory-only training data. This avoids expiratory-phase pressure behavior polluting the conditional means and the snapping grids, which typically reduces MAE on inspiratory steps without changing your prediction pipeline structure. I keep your fallback chain and submission writing intact, only switching the training source for lookups/grids and keeping a sensible global fallback.'
- What this solution (achieved 2.84125) has done: 'Your current MAE is still far above the 0.156 target (lower is better), so the smallest legitimate push downward is to make the lookup targets better match the evaluation rule: only inspiratory (`u_out==0`) rows are scored. I keep your exact multi-level lookup + fallback + `u_out==1 -> 0` + snap-to-grid core logic, but (1) build the *lagged* lookups from inspiratory-only training as you already do, and additionally (2) compute the history features (`u_in_bin_prev*`, `u_out_prev1`, `u_in_cumavg_bin`) within inspiratory segments (reset after `u_out==1`) so the keys match between train/test during the scored phase. This is a minimal feature-definition change (no new model/training loop) that typically reduces fallbacks and improves the conditional means for inspiratory steps. Submission writing, row alignment, and snapping behavior remain unchanged.'
- What this solution (achieved 2.86381) has done: 'Your current pipeline is already self-contained and metric-aligned (inspiratory-only training, per-segment history features, fallback chain, and snapping), so the smallest likely gain is to reduce key-mismatch noise rather than change the approach. I keep the exact lookup/fallback structure but make `u_in` binning slightly more consistent with how the signal behaves by clipping it to the valid [0,100] range before rounding (avoids rare float artifacts) and by using integer rounding via `np.rint` for stability. I also make the highest-priority lookup a touch less sparse by coarsening only the *history* average bin (keep your main `u_in_bin` intact) to improve coverage and reduce fallbacks without changing the core logic. Submission writing, row-order alignment, `u_out==1 -> 0`, and RC-conditional snapping remain unchanged.'
- What this solution (achieved 2.91502) has done: 'Your current MAE (2.86381; lower is better) is still far above the 0.156 target, and the biggest issue left (without changing your lookup/fallback/snap core) is key sparsity: exact matching on multiple binned history features causes many fallbacks. I keep the exact same multi-level lookup + fallback chain + `u_out==1 -> 0` + RC-conditional snap-to-grid logic, but make the highest-priority history keys less sparse by (1) coarsening `u_in_bin_prev1/prev2` slightly and (2) using a coarser cumulative-average bin. This increases hit-rate on the best lookup level (reduces fallback noise) while preserving the same overall modeling approach and semantics. I also compute lookups with `sort=False` for consistency and keep the submission alignment assertions unchanged.'
- What this solution (achieved 2.54975) has done: 'To move your MAE down toward the 0.156 target without changing the core “multi-level lookup + fallback + u_out==1→0 + snap-to-grid” approach, the smallest high-impact fix is to add one more lightweight history feature that better matches the true dynamics: cumulative inspired volume proxy via the integral of `u_in` over time (`u_in_dt_cumsum_bin`) computed within inspiratory segments. We use it only in a new highest-priority lookup level (ahead of your existing lag2 key), so everything else (binning, fallback chain, snapping, and submission writing) stays intact. This typically reduces fallbacks and improves conditional means because it disambiguates states with similar instantaneous `u_in` but different accumulated flow. The rest of the pipeline, including inspiratory-only training for lookups and RC-conditional snapping, remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)


def add_features_insp_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    u_in_clipped = df["u_in"].clip(0.0, 100.0).to_numpy(np.float32)
    df["u_in_bin"] = np.rint(u_in_clipped).astype("int16")

    seg = df.groupby("breath_id", sort=False)["u_out"].cumsum().astype("int16")
    g = df.groupby(["breath_id", seg], sort=False)

    prev1 = g["u_in_bin"].shift(1).fillna(0).astype("int16")
    prev2 = g["u_in_bin"].shift(2).fillna(0).astype("int16")
    df["u_in_bin_prev1"] = ((prev1 + 1) // 2).astype("int16")  # ~2-unit buckets
    df["u_in_bin_prev2"] = ((prev2 + 1) // 2).astype("int16")  # ~2-unit buckets

    df["u_out_prev1"] = g["u_out"].shift(1).fillna(0).astype("int8")

    step_idx = g.cumcount().astype("int16")  # 0.. within segment
    u_in_cumsum = g["u_in_bin"].cumsum().astype("int32")
    u_in_cumavg = (u_in_cumsum / (step_idx + 1)).astype(np.float32)
    df["u_in_cumavg_bin"] = np.floor(u_in_cumavg / 5.0).clip(0, 100).astype("int16")

    dt = g["time_step"].diff().fillna(0.0).to_numpy(np.float32)
    dt = np.clip(dt, 0.0, 0.2)  # safety clamp; does not change core semantics
    u_in_dt = (u_in_clipped * dt).astype(np.float32)
    u_in_dt_cumsum = g.apply(lambda x: None)  # placeholder to keep structure identical

    keys = g.grouper.result_index
    codes0, codes1 = g.grouper.codes
    group_code = (codes0.astype(np.int64) << 32) + codes1.astype(np.int64)
    order = np.arange(len(df), dtype=np.int32)
    csum = np.empty(len(df), dtype=np.float32)
    last_code = None
    running = 0.0
    for i in order:
        gc = group_code[i]
        if last_code is None or gc != last_code:
            running = 0.0
            last_code = gc
        running += float(u_in_dt[i])
        csum[i] = running

    df["u_in_dt_cumsum_bin"] = np.floor(csum / 10.0).clip(0, 200).astype("int16")

    return df


train = add_features_insp_segment(train)
test = add_features_insp_segment(test)

train_insp = train[train["u_out"] == 0].copy()

grp_cols_int = [
    "R",
    "C",
    "u_out",
    "u_in_bin",
    "u_in_bin_prev1",
    "u_in_bin_prev2",
    "u_out_prev1",
    "u_in_cumavg_bin",
    "u_in_dt_cumsum_bin",
]
lookup_int = train_insp.groupby(grp_cols_int, observed=True, sort=False)[
    "pressure"
].mean()

grp_cols_lag2 = [
    "R",
    "C",
    "u_out",
    "u_in_bin",
    "u_in_bin_prev1",
    "u_in_bin_prev2",
    "u_out_prev1",
    "u_in_cumavg_bin",
]
lookup_lag2 = train_insp.groupby(grp_cols_lag2, observed=True, sort=False)[
    "pressure"
].mean()

grp_cols_lag = ["R", "C", "u_out", "u_in_bin", "u_in_bin_prev1", "u_out_prev1"]
lookup_lag = train_insp.groupby(grp_cols_lag, observed=True, sort=False)[
    "pressure"
].mean()

grp_cols = ["R", "C", "u_out", "u_in_bin"]
lookup = train_insp.groupby(grp_cols, observed=True, sort=False)["pressure"].mean()
lookup_rc_ubin = train_insp.groupby(["R", "C", "u_in_bin"], observed=True, sort=False)[
    "pressure"
].mean()
lookup_rcuout = train_insp.groupby(["R", "C", "u_out"], observed=True, sort=False)[
    "pressure"
].mean()

global_mean_insp = float(train_insp["pressure"].mean())
global_mean_all = float(train["pressure"].mean())

pressure_grid_global = np.sort(train_insp["pressure"].unique()).astype(np.float32)
pmin = float(pressure_grid_global[0])
pmax = float(pressure_grid_global[-1])

pressure_grids_rc = {
    k: np.sort(np.asarray(v, dtype=np.float32))
    for k, v in train_insp.groupby(["R", "C"], sort=False)["pressure"].unique().items()
}

idx_int = pd.MultiIndex.from_frame(test[grp_cols_int])
pred = lookup_int.reindex(idx_int).to_numpy()

mask = pd.isna(pred)
if mask.any():
    idx_lag2 = pd.MultiIndex.from_frame(test.loc[mask, grp_cols_lag2])
    pred_lag2 = lookup_lag2.reindex(idx_lag2).to_numpy()
    pred[mask] = pred_lag2

mask = pd.isna(pred)
if mask.any():
    idx_lag = pd.MultiIndex.from_frame(test.loc[mask, grp_cols_lag])
    pred_lag = lookup_lag.reindex(idx_lag).to_numpy()
    pred[mask] = pred_lag

mask = pd.isna(pred)
if mask.any():
    idx0 = pd.MultiIndex.from_frame(test.loc[mask, grp_cols])
    pred0 = lookup.reindex(idx0).to_numpy()
    pred[mask] = pred0

mask = pd.isna(pred)
if mask.any():
    idx_f1 = pd.MultiIndex.from_frame(test.loc[mask, ["R", "C", "u_in_bin"]])
    pred_f1 = lookup_rc_ubin.reindex(idx_f1).to_numpy()
    pred[mask] = pred_f1

mask = pd.isna(pred)
if mask.any():
    idx_f2 = pd.MultiIndex.from_frame(test.loc[mask, ["R", "C", "u_out"]])
    pred_f2 = lookup_rcuout.reindex(idx_f2).to_numpy()
    pred[mask] = pred_f2

pred = pd.Series(pred)
is_insp_test = test["u_out"].to_numpy() == 0
pred.loc[is_insp_test] = pred.loc[is_insp_test].fillna(global_mean_insp)
pred = pred.fillna(global_mean_all).to_numpy(dtype=np.float32)

pred = pred.copy()
pred[~is_insp_test] = 0.0

insp_idx = np.where(is_insp_test)[0]
insp_pred = pred[insp_idx]
insp_pred = np.clip(insp_pred, pmin, pmax)

R_arr = test["R"].to_numpy()
C_arr = test["C"].to_numpy()

snapped = np.empty_like(insp_pred, dtype=np.float32)
for i, row_i in enumerate(insp_idx):
    grid = pressure_grids_rc.get(
        (int(R_arr[row_i]), int(C_arr[row_i])), pressure_grid_global
    )
    x = insp_pred[i]
    pos = np.searchsorted(grid, x, side="left")
    if pos <= 0:
        snapped[i] = grid[0]
    elif pos >= len(grid):
        snapped[i] = grid[-1]
    else:
        left = grid[pos - 1]
        right = grid[pos]
        snapped[i] = left if (x - left) <= (right - x) else right

pred[insp_idx] = snapped.astype(np.float32)

assert (
    len(pred) == len(test) == len(sub)
), "Row misalignment: predictions/test/sample_submission must have same length."

sub_out = sub.copy()
sub_out["pressure"] = pred.astype(float)
sub_out.to_csv("submission.csv", index=False)

sub_out.head(5)
