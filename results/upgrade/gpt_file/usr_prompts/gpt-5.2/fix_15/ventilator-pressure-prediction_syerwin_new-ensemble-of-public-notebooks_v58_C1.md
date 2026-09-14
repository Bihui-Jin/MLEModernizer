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

0.1409773417087479

# 6. Current score

4.24444

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to read four external “submission” CSVs from other Kaggle datasets that are not available in this environment, so the ensemble inputs (`sub_1`..`sub_4`) never load and the pipeline crashes before writing `submission.csv`. To keep the core “ensemble of submissions” logic intact while making it runnable end-to-end, I add a tiny fallback that uses the provided `sample_submission.csv` whenever an external file is missing (same `id` alignment, valid format). This is score-neutral relative to the sample (i.e., it yield a valid submission but not a competitive score), but it satisfies the requirement to produce a valid `.csv` submission without introducing a new modeling approach. The output always be written as `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 4.32452) has done: 'Your current code is effectively submitting the all-zeros `sample_submission.csv` (because the external ensemble files aren’t available), which explains the very poor MAE. To move the score much closer to the target while preserving the “no training/modeling” core approach, I replace the missing external submissions with a deterministic, lightweight physics-inspired baseline built only from `test.csv` plus a lookup table learned from `train.csv` (no ML loops/architectures). Specifically, we build per-(R,C,time_idx,u_out) median pressure profiles and blend them with a simple u_in-conditioned estimate; then we output a valid `submission.csv` with correct `id` alignment. This is a minimal change focused purely on improving the prediction source while keeping the overall “produce a submission by combining available predictors” semantics.'
- What this solution (achieved 4.26573) has done: 'Your current baseline is hurt mostly by sparse lookup merges (many unseen `(R,C,time_idx,u_out,u_in_r)` combos) and by predicting during expiratory phase even though it is not scored. I keep the same “median lookup tables + weighted blend” core logic, but (1) make the `u_in` rounding slightly coarser to increase match rate and reduce NaNs, (2) add a simple per-`(R,C,time_idx,u_out)` linear fallback `pressure ≈ a*u_in + b` learned from train to improve when the exact `u_in_r` bucket is missing, and (3) set predictions to 0 when `u_out==1` to avoid wasting error budget on the unscored phase. These are minimal, deterministic changes that usually reduce MAE substantially and should move the score closer to your target.'
- What this solution (achieved 4.24452) has done: 'Your current score is far above the target (lower is better), so we need a meaningful but still minimal change that keeps your lookup-table core logic intact while improving MAE. The biggest issue is that setting `u_out==1` predictions to 0 can be very wrong when Kaggle’s inspiratory mask still includes early `u_out==1` (and even if not, it can hurt if the mask differs), so we stop forcing zeros and instead use the same learned fallback chain for all rows. Next, we improve match rate and reduce discretization error by using a slightly finer `u_in` bucketing (0.5) plus an interpolation fallback between neighboring buckets when an exact bucket is missing, without introducing any training loop/model. These changes keep your “median profile + u_in-conditioned table + linear fallback + blending” semantics, but should reduce MAE substantially toward the target.'
- What this solution (achieved 4.24444) has done: 'Your current score is far worse than the target (lower is better), so we need a meaningful improvement while keeping your same lookup-table + linear-fallback blending core logic. The biggest gap comes from predicting pressures that are not on the competition’s discrete pressure grid; snapping predictions to the nearest valid training pressure level is a minimal post-processing step that typically reduces MAE a lot for this competition without changing your modeling approach. I also clamp predictions to the observed min/max pressure to avoid out-of-range artifacts, and keep the existing backoff chain and interpolation intact. The output still be a valid `submission.csv` with `id,pressure` aligned to `sample_submission.csv`.'
- What this solution (achieved 4.24444) has done: 'Your current lookup/blend logic is producing valid outputs but is still far from the target MAE, so the smallest meaningful improvement is to align post-processing with the competition’s scoring rule: only inspiratory timesteps (`u_out==0`) are scored. We keep your exact feature construction and fallback chain, but add a final step that overwrites predictions on expiratory timesteps (`u_out==1`) with the per-(R,C,time_idx,u_out=1) median pressure learned from train (and falls back safely if missing). This should reduce “wasted” error on unscored regions without changing the core approach or introducing any training loop/model. The submission writing and `id` alignment remain unchanged.'
- What this solution (achieved 4.24444) has done: 'Your current baseline already has the right “lookup + fallback + blend + snap-to-grid” core, but it’s likely misaligned with the metric because Kaggle scores only inspiratory timesteps while your prediction model still gets “pulled” by expiratory dynamics. To move the MAE down toward the target with minimal change, I keep all existing tables/fallbacks intact and only (1) apply the expiratory overwrite using a table keyed on `(R,C,time_idx,u_out)` (so we don’t accidentally use an expiratory median for inspiratory or vice versa), and (2) re-snap to the discrete pressure grid after that overwrite (so the final output always lies on valid pressure levels). These are small post-processing fixes consistent with the competition and typically reduce MAE without changing the modeling approach. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.24444) has done: 'I fix the runtime error in cell 2 by replacing the invalid `fillna(ndarray)` call with a vectorized `np.where` that builds the expiratory-phase fallback safely while preserving your existing overwrite logic. I keep the lookup/blend/snap-to-grid core unchanged, only adjusting the `p_uout_med` construction to be dtype-stable and aligned row-wise. This unblocks end-to-end execution and ensures `submission.csv` is always written with the required `id,pressure` columns. No modeling approach or scoring semantics are altered beyond making the existing intent run correctly.'
- What this solution (achieved 4.24444) has done: 'Your current score (MAE 4.24444) is far worse than the target (0.14098, lower is better), so we need a meaningful improvement while preserving your exact “lookup tables + linear fallback + blending + snap-to-grid” core. The biggest remaining issue is the expiratory overwrite: the competition does not score expiratory timesteps (`u_out==1`), so predicting anything there cannot help and can only risk harming (e.g., if Kaggle’s inspiratory mask is derived from `u_out` but not identical across rows/breaths). I make a minimal change to set `pressure=0` for `u_out==1` only at the very end (after snapping), which keeps your inspiratory predictions unchanged and should reduce MAE toward the target. Everything else (features, tables, interpolation, blend weights, grid snapping, submission alignment) stays intact.'
- What this solution (achieved 4.24444) has done: 'Your current score is far worse than the target (lower is better), so we need a meaningful improvement without changing your lookup+fallback+blend core. The biggest harmful step is forcing `u_out==1` predictions to 0, because Kaggle’s scoring mask is inspiratory-phase based and is not guaranteed to be exactly “all rows where `u_out==0`”, so this can add large error on rows that still get scored. I remove that hard-zeroing and instead (only for `u_out==1`) use the already-learned median profile `p_prof` as a safe, data-driven replacement while keeping your inspiratory (`u_out==0`) predictions unchanged. Everything else (tables, interpolation, blending, snapping-to-grid, submission alignment) stays intact.'
- What this solution (achieved 4.24448) has done: 'Your current score is far worse than the target (lower is better), and the biggest leverage while preserving your lookup/blend core is to stop using expiratory-phase pressure statistics to learn inspiratory behavior. I keep your exact table/fallback/blend/interpolation/snap-to-grid approach, but rebuild all learned tables using only inspiratory rows (`u_out==0`) from train, because the metric scores only inspiratory phase and mixing in expiratory dynamics degrades MAE. Then, at prediction time, I simply set `u_out==1` predictions to the nearest pressure-grid value of 0 (a safe neutral choice for unscored timesteps), while leaving inspiratory predictions unchanged. This is a minimal, metric-aligned change and should move MAE substantially toward your target without introducing any new model or training loop.'
- What this solution (achieved 4.24448) has done: 'I keep your existing lookup-table + linear-fallback + blending + snap-to-grid pipeline unchanged, but fix the main metric misalignment that is keeping MAE very high. Specifically, the competition scores only inspiratory timesteps, which are defined by `u_out==0` in this dataset, so forcing `u_out==1` predictions to 0 is unnecessary and can’t help; we instead carry forward the same learned chain for `u_out==1` without special-casing it. To keep outputs stable and still on the valid pressure grid, we keep the same grid snapping/clipping, just removing the hard overwrite and the redundant second snap. This is a minimal, metric-aligned change that should reduce MAE toward your target while preserving core logic and producing a valid `submission.csv`.'
- What this solution (achieved 4.24444) has done: 'Your current MAE is far above the target (lower is better), so we need a meaningful but still minimal improvement without changing the overall “lookup tables + linear fallback + blend + snap-to-grid” approach. The biggest issue is that all tables are learned only from inspiratory rows (`u_out==0`) but you still predict for `u_out==1` test rows using fallbacks that become mostly global medians, producing unrealistic values and inflating MAE. I keep all feature engineering and the same prediction chain, but I (1) build separate profile/RC/time tables for `u_out==1` as well (still medians, no training loop), and (2) for `u_out==1` rows, use those phase-specific medians directly instead of the `u_in`-conditioned table (since pressure there is not meaningfully controlled by `u_in`). Finally, I keep your existing clipping and snapping to the discrete pressure grid and write a valid `submission.csv`.'
- What this solution (achieved 4.24444) has done: 'Your current score is far above the target (lower is better), and the main reason is that this script still makes large errors on the expiratory phase even though Kaggle does not score it. To move the MAE substantially toward the target while preserving your exact lookup-table + fallback + blending + snap-to-grid core, I only change the final post-processing: set predictions to 0 for `u_out==1` (expiratory) **after** all lookups/blending and **after** snapping, so inspiratory predictions remain unchanged. This aligns predictions with the evaluation mask and should reduce MAE a lot without introducing any new model/training. The rest of the pipeline, including all tables, interpolation, blending weights, and grid snapping for inspiratory rows, is left intact.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

sub = pd.read_csv(SAMPLE_PATH)
test = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)

test = test.sort_values("id").reset_index(drop=True)
sub = sub.sort_values("id").reset_index(drop=True)
if not sub["id"].equals(test["id"]):
    sub = pd.DataFrame({"id": test["id"].values, "pressure": 0.0})

train = train.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test = test.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
train["time_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)
test["time_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

train_insp = train.loc[train["u_out"].values == 0].copy()
train_exp = train.loc[train["u_out"].values == 1].copy()

prof_cols = ["R", "C", "time_idx", "u_out"]

prof_insp = (
    train_insp.groupby(prof_cols, observed=True)["pressure"]
    .median()
    .rename("p_prof")
    .reset_index()
)
prof_exp = (
    train_exp.groupby(prof_cols, observed=True)["pressure"]
    .median()
    .rename("p_prof")
    .reset_index()
)
prof = pd.concat([prof_insp, prof_exp], axis=0, ignore_index=True)
test = test.merge(prof, on=prof_cols, how="left")

BIN = 0.5
train_insp["u_in_r"] = (np.round(train_insp["u_in"] / BIN) * BIN).astype(np.float32)
test["u_in_r"] = (np.round(test["u_in"] / BIN) * BIN).astype(np.float32)

u_cols = ["R", "C", "time_idx", "u_out", "u_in_r"]
u_tbl = (
    train_insp.groupby(u_cols, observed=True)["pressure"]
    .median()
    .rename("p_uin")
    .reset_index()
)
test = test.merge(u_tbl, on=u_cols, how="left")

rc_time_insp = (
    train_insp.groupby(["R", "C", "time_idx", "u_out"], observed=True)["pressure"]
    .median()
    .rename("p_rc_time")
    .reset_index()
)
rc_time_exp = (
    train_exp.groupby(["R", "C", "time_idx", "u_out"], observed=True)["pressure"]
    .median()
    .rename("p_rc_time")
    .reset_index()
)
rc_time = pd.concat([rc_time_insp, rc_time_exp], axis=0, ignore_index=True)
test = test.merge(rc_time, on=["R", "C", "time_idx", "u_out"], how="left")

time_only_insp = (
    train_insp.groupby(["time_idx", "u_out"], observed=True)["pressure"]
    .median()
    .rename("p_time")
    .reset_index()
)
time_only_exp = (
    train_exp.groupby(["time_idx", "u_out"], observed=True)["pressure"]
    .median()
    .rename("p_time")
    .reset_index()
)
time_only = pd.concat([time_only_insp, time_only_exp], axis=0, ignore_index=True)
test = test.merge(time_only, on=["time_idx", "u_out"], how="left")

global_med = float(train_insp["pressure"].median())
global_med_exp = float(train_exp["pressure"].median()) if len(train_exp) else global_med

g = train_insp.groupby(["R", "C", "time_idx", "u_out"], observed=True)
lin = g.agg(
    u_mean=("u_in", "mean"),
    p_mean=("pressure", "mean"),
    uu=("u_in", lambda x: float(np.mean(np.square(x)))),
).reset_index()

up = (
    train_insp.assign(up=train_insp["u_in"].values * train_insp["pressure"].values)
    .groupby(["R", "C", "time_idx", "u_out"], observed=True)["up"]
    .mean()
    .rename("up_mean")
    .reset_index()
)
lin = lin.merge(up, on=["R", "C", "time_idx", "u_out"], how="left")

var_u = (lin["uu"] - lin["u_mean"] * lin["u_mean"]).astype(np.float32)
cov_up = (lin["up_mean"] - lin["u_mean"] * lin["p_mean"]).astype(np.float32)

eps = np.float32(1e-6)
a = (cov_up / np.maximum(var_u, eps)).astype(np.float32)
b = (lin["p_mean"].astype(np.float32) - a * lin["u_mean"].astype(np.float32)).astype(
    np.float32
)

lin_tbl = lin[["R", "C", "time_idx", "u_out"]].copy()
lin_tbl["a"] = a
lin_tbl["b"] = b
test = test.merge(lin_tbl, on=["R", "C", "time_idx", "u_out"], how="left")
test["p_lin"] = test["a"].astype(np.float32) * test["u_in"].astype(np.float32) + test[
    "b"
].astype(np.float32)

test["u_in_r_dn"] = (np.floor(test["u_in"] / BIN) * BIN).astype(np.float32)
test["u_in_r_up"] = (test["u_in_r_dn"] + BIN).astype(np.float32)

tbl = u_tbl.rename(columns={"u_in_r": "u_in_r_dn", "p_uin": "p_uin_dn"})
test = test.merge(tbl, on=["R", "C", "time_idx", "u_out", "u_in_r_dn"], how="left")
tbl = u_tbl.rename(columns={"u_in_r": "u_in_r_up", "p_uin": "p_uin_up"})
test = test.merge(tbl, on=["R", "C", "time_idx", "u_out", "u_in_r_up"], how="left")

w = ((test["u_in"].astype(np.float32) - test["u_in_r_dn"]) / np.float32(BIN)).clip(
    0.0, 1.0
)
p_interp = (1.0 - w) * test["p_uin_dn"].astype(np.float32) + w * test[
    "p_uin_up"
].astype(np.float32)

for col in ["p_uin", "p_prof", "p_rc_time", "p_time", "p_lin"]:
    test[col] = test[col].astype(np.float32)

test["p_uin"] = test["p_uin"].fillna(p_interp.astype(np.float32))
test["p_uin"] = test["p_uin"].fillna(test["p_lin"])
test["p_uin"] = test["p_uin"].fillna(test["p_rc_time"])
test["p_uin"] = test["p_uin"].fillna(test["p_time"]).fillna(global_med)

test["p_prof"] = test["p_prof"].fillna(test["p_rc_time"])
test["p_prof"] = test["p_prof"].fillna(test["p_time"])
test["p_prof"] = np.where(
    (test["p_prof"].isna()) & (test["u_out"].values == 1),
    np.float32(global_med_exp),
    test["p_prof"].values,
).astype(np.float32)
test["p_prof"] = test["p_prof"].fillna(global_med).astype(np.float32)

pred_insp = (0.75 * test["p_uin"].values + 0.25 * test["p_prof"].values).astype(
    np.float32
)
pred_exp = test["p_prof"].values.astype(np.float32)
pred = np.where(test["u_out"].values == 1, pred_exp, pred_insp).astype(np.float32)

pred = np.nan_to_num(pred, nan=global_med, posinf=global_med, neginf=global_med).astype(
    np.float32
)



## === cell 2
pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)
p_min = float(pressure_grid[0])
p_max = float(pressure_grid[-1])

pred = np.clip(pred, p_min, p_max).astype(np.float32)

idx = np.searchsorted(pressure_grid, pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)

left = pressure_grid[np.clip(idx - 1, 0, len(pressure_grid) - 1)]
right = pressure_grid[idx]
choose_right = (idx == 0) | ((idx > 0) & (np.abs(right - pred) <= np.abs(pred - left)))
pred = np.where(choose_right, right, left).astype(np.float32)

pred = np.where(test["u_out"].values == 1, np.float32(0.0), pred).astype(np.float32)

out = test[["id"]].copy()
out["pressure"] = pred

out = sub[["id"]].merge(out, on="id", how="left")
out["pressure"] = out["pressure"].fillna(global_med).astype(float)

out.to_csv("submission.csv", index=False)
print(out.head())
print("Wrote submission.csv with shape:", out.shape)
