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

0.1383325702636221

# 6. Current score

1.98216

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.81842) has done: 'The current notebook fails because it tries to read two external Kaggle datasets that are not present in your provided environment, so no submission gets created. I replace that dependency with a self-contained baseline that reads the provided `train.csv`/`test.csv`, builds a simple, deterministic per-(R,C,time_step,u_in,u_out) median pressure lookup from train, and uses safe fallbacks when an exact match isn’t found. This preserves the “median-ensemble/median-prediction” spirit while making the pipeline run end-to-end and reliably write `submission.csv` with the correct columns. The changes are minimal and focused on fixing I/O and producing a valid submission (and a non-trivial score versus all-zeros).'
- What this solution (achieved 4.36815) has done: 'Your current lookup uses an exact match on `u_in` (a continuous float), so it almost never hits and falls back to much coarser tables, driving the MAE way above target. I keep the same “median lookup with fallbacks” core logic, but make the primary key actually match by binning `u_in` to a small fixed resolution (0.1) in both train and test; this is a minimal, metric-aligned change that should substantially reduce error toward the 0.138 target. I also replace the final global fallback with a slightly more informative per-(R,C) median (still a median fallback, just less coarse), which should further lower MAE without changing the overall approach. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.76921) has done: 'Your current approach is already a “median lookup with fallbacks”, but it still misses a lot because `u_in` binning at 0.1 is too fine given noise/float representation and the dynamics; this keeps you in coarser fallbacks and MAE stays far above the 0.138 target. I keep the exact same lookup-and-fallback core logic, but (1) make the `u_in` bin slightly coarser (0.5) to increase primary-table hit rate, and (2) add one extra intermediate fallback that uses `(R,C,u_in,u_out)` (still a median table) before dropping `u_in` entirely. These are minimal, metric-aligned changes that should reduce MAE substantially without changing the overall method or introducing any new modeling/training. The pipeline remains deterministic, runs end-to-end, and writes a valid `submission.csv`.'
- What this solution (achieved 3.36051) has done: 'Your current score (3.77 MAE) is far above the target (0.138), so we need a meaningful but still “same-core-logic” improvement: keep the median-lookup-with-fallbacks approach, but make the keys better match what the metric scores. The metric ignores expiratory phase (u_out=1), so we explicitly force predictions to 0 during u_out=1 (as in sample submission) and focus the lookup tables on inspiratory rows only, which typically reduces MAE a lot for this competition. We also keep your u_in binning, but add a very small extra dynamic feature (`u_in_lag1` per breath) into the primary table to reduce ambiguity without changing the overall lookup/median method. All changes are deterministic, run end-to-end, and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.51119) has done: 'Your current gap to the target is very large (3.36 vs 0.138, lower is better), and most of that error comes from poor key match/hit-rate in the median lookup due to continuous-ish features (`time_step`, `u_in`, and especially `u_in_lag1`). I keep the exact same “median lookup with fallbacks” logic, but (1) quantize `time_step` to its natural grid (0.03s) instead of rounding to 0.02 (which creates many non-existent times), and (2) remove float-merge fragility by converting the binned `time_step/u_in/u_in_lag1` into small integer bins used for all groupby/merge keys. This is a minimal change that should substantially increase exact-match hits in the primary and intermediate tables (so fewer coarse fallbacks), moving MAE down toward your target, while preserving your overall approach and still writing a valid `submission.csv`.'
- What this solution (achieved 3.48967) has done: 'Your current MAE is still far above target, and the biggest remaining issue is that `ts_bin` is still merge-fragile because `time_step` is stored as float32 then divided/rounded; small representation differences can push a value to an adjacent bin and cause many primary-key misses. I keep the exact same “median lookup + sequential fallbacks” core logic, but compute `ts_bin` deterministically from the known 80-step grid (0..79) by ranking within each `breath_id` after sorting by `time_step`, which matches train/test exactly and dramatically increases hit rate. I also compute `uin_lag1_bin` for train on the full sequence (then filter inspiratory rows) so the lag corresponds to the real previous time step even when the previous row is expiratory, which improves key consistency without changing the approach. Everything else (tables, fallbacks, forcing `u_out==1` to 0, output format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 3.9946) has done: 'Your current MAE (3.48967) is far above the target (0.13833), so we should improve hit-rate of the “median lookup + fallbacks” without changing the overall approach. The main issue is that using `uin_lag1_bin` in the primary key makes exact matches much rarer in test (because the same `(R,C,ts,u_in,u_out)` can map to many `uin_lag1` histories), forcing coarse fallbacks and large MAE. I keep the same lookup-and-fallback structure, but remove `uin_lag1_bin` from the primary table key (and make it an intermediate fallback instead), which should substantially increase primary matches and reduce MAE. I also compute fallbacks from inspiratory-only data but keep merges on `u_out` where appropriate, preserving your evaluation semantics (including forcing `u_out==1` predictions to 0).'
- What this solution (achieved 3.9946) has done: 'Your current MAE is far above the target (lower is better), so we need a small change that improves the lookup hit-rate without changing the overall “median tables + sequential fallbacks” approach. The biggest issue is that you’re forcing `u_out==1` predictions to 0, but you build all median tables from `u_out==0` only; combined with merging on `u_out`, this guarantees lots of missing matches and pushes many inspiratory points into coarse fallbacks. I keep the exact same table/fallback structure, but build the tables from the full train (both `u_out` values) so keys that include `u_out` can actually match, while still setting `u_out==1` predictions to 0 at the end (so evaluation semantics remain the same). This is a minimal, deterministic fix that should move MAE substantially down toward your target.'
- What this solution (achieved 3.25347) has done: 'Your current MAE (3.9946, lower is better) is still far from the target (0.1383), and the biggest remaining issue is that the “best available” table you merge first is too coarse while a more informative table (`(R,C,ts_bin,uin_bin,uin_lag1_bin,u_out)`) exists but is only used as a fallback. I keep the exact same median-lookup + sequential fillna structure, but swap the order so we try the most specific lag-conditioned table first (higher hit quality), then fall back to the coarser tables as before. I also set the global fallback median to be computed on full train (not only inspiratory) since you now build/merge on both `u_out` values; we still force `u_out==1` predictions to 0 at the end, preserving your evaluation semantics. These are minimal changes focused on improving match quality and reducing MAE toward your target without changing the overall approach.'
- What this solution (achieved 3.27233) has done: 'Your MAE is still far above the target, so we need a meaningful-but-minimal improvement without changing the overall “median lookup tables + sequential fallbacks” approach. The biggest issue is that `ts_bin` as a plain `cumcount()` is not guaranteed to correspond to the same physical timestep across breaths if any breath has missing/duplicated rows (or slight ordering quirks), which harms exact-match hit rate and forces coarse fallbacks. I keep your same keys and fallback structure, but compute `ts_bin` by snapping `time_step` to the known 0.03s grid (0..79) in a deterministic integer way and only fall back to `cumcount()` when snapping is ambiguous; this typically increases primary-table matches and reduces MAE. I also keep all I/O and the final `u_out==1 -> 0` rule unchanged so evaluation semantics remain the same.'
- What this solution (achieved 1.8136) has done: 'Your current MAE (3.272) is far above the target (0.138, lower is better), and the main reason is that the lookup keys still don’t capture enough breath dynamics, so many points fall back to very coarse medians. Keeping the exact same “median tables + sequential fillna fallbacks” core logic, I add one more deterministic, lightweight dynamic feature (`u_in` cumulative integral per breath) and use it only as the *most specific* first-try table so matches are better when available, then fall back exactly as before. To avoid making hit-rate worse, I quantize this new feature into a small integer bin (like your other bins) and keep all existing keys/tables intact. This should reduce MAE materially toward your target while staying within the same method and still writing a valid `submission.csv`.'
- What this solution (achieved 1.81365) has done: 'Your current MAE (1.8136) is still far above the target (0.1383, lower is better), so we should improve lookup quality without changing the overall “median tables + sequential fallbacks” approach. The simplest high-impact fix is to use the known discrete pressure grid in this competition: after computing pressures via your existing tables, snap predictions to the nearest pressure value observed in train (this typically reduces MAE a lot while keeping semantics intact). To avoid harming already-good exact matches, we only snap after all fillna logic and after forcing `u_out==1` to 0. Finally, we compute the pressure grid once from train and apply a fast vectorized nearest-neighbor mapping.'
- What this solution (achieved 2.01258) has done: 'Your MAE (1.81365) is still far above the target (0.13833, lower is better), so we should improve match quality while keeping the exact same “median lookup tables + sequential fillna fallbacks + snap-to-pressure-grid” core approach. The most impactful minimal fix is to use the competition’s key domain fact: pressures depend heavily on the *cumulative inspired volume proxy* `u_in` integral, but binning it at 1.0 is too coarse and collapses distinct states into the same bin. I keep your same features/tables and ordering, but make `uin_cum_bin` finer (0.2) to improve the specificity of the first-try table without changing the logic. Everything else (keys, fallbacks, forcing `u_out==1` to 0, and snapping to the train pressure grid) stays unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.98216) has done: 'We make a single, minimal change that improves key hit-rate in your most-specific median table: compute `uin_cum_bin` on the *same integer timestep grid* (`ts_bin`) instead of relying on float `time_step` diffs, which can drift and mismatch between train/test. This preserves your exact “median lookup tables + sequential fallbacks + u_out==1 -> 0 + snap-to-pressure-grid” core logic, but makes the cumulative feature more consistent and thus more often matchable. Everything else (bins, tables, merge order, snapping, output format) stays the same to keep behavior stable while nudging MAE down toward the target. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.98216) has done: 'Your current score (1.98216 MAE, lower is better) is far from the target (0.13833), so we should make a minimal change that increases exact-match hit-rate in your most-informative table without changing the overall “median lookup + sequential fallbacks + u_out==1 -> 0 + snap-to-pressure-grid” logic. The most fragile key right now is `uin_cum_bin`: because you quantize it, small drift causes mismatches that push many rows into coarser fallbacks. I keep the cumulative feature but compute it on the known fixed 80-step grid (exact `0.03` per step) and explicitly reset to 0 at the start of each breath (ts_bin==0), making train/test consistent and improving merges. Everything else (tables, order, snapping, output schema) stays the same so behavior remains stable while nudging MAE down toward the target.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
BASE_PATHS = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for bp in BASE_PATHS:
        cand = os.path.join(bp, filename)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not find {filename} under any of: {BASE_PATHS}")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train_path, test_path, sample_path



## === cell 2
usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

train = pd.read_csv(train_path, usecols=usecols_train)
test = pd.read_csv(test_path, usecols=usecols_test)

for col in ["breath_id"]:
    train[col] = train[col].astype(np.int32)
    test[col] = test[col].astype(np.int32)

for col in ["R", "C", "u_out"]:
    train[col] = train[col].astype(np.int16)
    test[col] = test[col].astype(np.int16)

UIN_BIN = 0.5
train["u_in"] = train["u_in"].astype(np.float32)
test["u_in"] = test["u_in"].astype(np.float32)

train["uin_bin"] = np.rint(train["u_in"] / UIN_BIN).astype(np.int16)
test["uin_bin"] = np.rint(test["u_in"] / UIN_BIN).astype(np.int16)

train.sort_values(["breath_id", "time_step"], inplace=True)
test.sort_values(["breath_id", "time_step"], inplace=True)


def compute_ts_bin(df: pd.DataFrame) -> pd.Series:
    ts = df["time_step"].astype(np.float32).to_numpy()
    ts_snap = np.rint(ts / 0.03).astype(np.int16)

    g = df.groupby("breath_id", sort=False)
    size_ok = g["time_step"].transform("size").astype(np.int16).to_numpy() == 80
    min_ok = g["time_step"].transform("min").to_numpy()
    max_ok = g["time_step"].transform("max").to_numpy()

    in_range = (ts_snap >= 0) & (ts_snap <= 79)

    tmp = pd.DataFrame({"breath_id": df["breath_id"].values, "ts_snap": ts_snap})
    nunique = (
        tmp.groupby("breath_id", sort=False)["ts_snap"].transform("nunique").to_numpy()
        == 80
    )

    use_snap = size_ok & nunique & in_range
    use_snap &= (min_ok <= 1e-6) & (max_ok <= 3.0)

    ts_bin_cum = g.cumcount().astype(np.int16).to_numpy()
    out = np.where(use_snap, ts_snap, ts_bin_cum).astype(np.int16)
    return pd.Series(out, index=df.index, dtype=np.int16)


train["ts_bin"] = compute_ts_bin(train)
test["ts_bin"] = compute_ts_bin(test)

if train["ts_bin"].max() > 79 or test["ts_bin"].max() > 79:
    print(
        "Warning: ts_bin exceeds 79; unexpected number of timesteps per breath detected."
    )

train.shape, test.shape, train.head(2), test.head(2)



## === cell 3
train["uin_lag1_bin"] = (
    train.groupby("breath_id")["uin_bin"].shift(1).fillna(0).astype(np.int16)
)
test["uin_lag1_bin"] = (
    test.groupby("breath_id")["uin_bin"].shift(1).fillna(0).astype(np.int16)
)


def add_uin_cum_bins(df: pd.DataFrame, bin_size: float = 1.0) -> None:
    dt = (df["ts_bin"].astype(np.float32).to_numpy() > 0).astype(
        np.float32
    ) * np.float32(0.03)
    uin_cum = (df["u_in"].astype(np.float32).to_numpy() * dt).astype(np.float32)
    uin_cum = (
        pd.Series(uin_cum, index=df.index)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.float32)
    )
    df["uin_cum_bin"] = np.rint((uin_cum / np.float32(bin_size)).to_numpy()).astype(
        np.int16
    )


UIN_CUM_BIN_SIZE = 0.2
add_uin_cum_bins(train, bin_size=UIN_CUM_BIN_SIZE)
add_uin_cum_bins(test, bin_size=UIN_CUM_BIN_SIZE)

train_all = train.copy()

primary_keys = ["R", "C", "ts_bin", "uin_bin", "u_out"]
med_table = (
    train_all.groupby(primary_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred"})
)

fallback_lag_keys = ["R", "C", "ts_bin", "uin_bin", "uin_lag1_bin", "u_out"]
med_fallback_lag = (
    train_all.groupby(fallback_lag_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_f_lag"})
)

fallback0_keys = ["R", "C", "uin_bin", "uin_lag1_bin", "u_out"]
med_fallback0 = (
    train_all.groupby(fallback0_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_f0"})
)

fallback1_keys = ["R", "C", "ts_bin", "u_out"]
med_fallback1 = (
    train_all.groupby(fallback1_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_f1"})
)

fallback2_keys = ["R", "C", "ts_bin"]
med_fallback2 = (
    train_all.groupby(fallback2_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_f2"})
)

fallback3_keys = ["R", "C"]
med_fallback3 = (
    train_all.groupby(fallback3_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_f3"})
)

primary_cum_keys = ["R", "C", "ts_bin", "uin_bin", "uin_cum_bin", "u_out"]
med_table_cum = (
    train_all.groupby(primary_cum_keys, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_cum"})
)

global_median = float(train_all["pressure"].median())

pressure_grid = np.sort(train_all["pressure"].round(5).unique()).astype(np.float32)

(
    len(med_table_cum),
    len(med_table),
    len(med_fallback_lag),
    len(med_fallback0),
    len(med_fallback1),
    len(med_fallback2),
    len(med_fallback3),
    global_median,
    len(pressure_grid),
)



## === cell 4
pred = test[["id"] + primary_cum_keys + ["uin_lag1_bin"]].copy()

pred = pred.merge(med_table_cum, on=primary_cum_keys, how="left")
pred = pred.merge(med_table, on=primary_keys, how="left")
pred = pred.merge(med_fallback_lag, on=fallback_lag_keys, how="left")
pred = pred.merge(med_fallback0, on=fallback0_keys, how="left")
pred = pred.merge(med_fallback1, on=fallback1_keys, how="left")
pred = pred.merge(med_fallback2, on=fallback2_keys, how="left")
pred = pred.merge(med_fallback3, on=fallback3_keys, how="left")

pred["pressure"] = pred["pred_cum"]
pred["pressure"] = pred["pressure"].fillna(pred["pred_f_lag"])
pred["pressure"] = pred["pressure"].fillna(pred["pred"])
pred["pressure"] = pred["pressure"].fillna(pred["pred_f0"])
pred["pressure"] = pred["pressure"].fillna(pred["pred_f1"])
pred["pressure"] = pred["pressure"].fillna(pred["pred_f2"])
pred["pressure"] = pred["pressure"].fillna(pred["pred_f3"])
pred["pressure"] = pred["pressure"].fillna(global_median)

pred.loc[pred["u_out"] == 1, "pressure"] = 0.0

pred = pred[["id", "pressure"]]
pred["pressure"] = (
    pd.to_numeric(pred["pressure"], errors="coerce").fillna(0.0).astype(np.float32)
)

p = pred["pressure"].to_numpy(dtype=np.float32)
idx = np.searchsorted(pressure_grid, p, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)

right = pressure_grid[idx]
left = pressure_grid[idx_left]
choose_left = np.abs(p - left) <= np.abs(p - right)
p_snapped = np.where(choose_left, left, right).astype(np.float32)

pred["pressure"] = p_snapped

pred.head(), pred.isna().sum()



## === cell 5
submission = pred.sort_values("id").reset_index(drop=True)

assert list(submission.columns) == ["id", "pressure"]
assert submission["id"].is_monotonic_increasing
assert len(submission) == len(test)

submission.to_csv("submission.csv", index=False)

del (
    train,
    test,
    train_all,
    med_table_cum,
    med_table,
    med_fallback_lag,
    med_fallback0,
    med_fallback1,
    med_fallback2,
    med_fallback3,
    pred,
)
gc.collect()

submission.head(), submission.tail()
