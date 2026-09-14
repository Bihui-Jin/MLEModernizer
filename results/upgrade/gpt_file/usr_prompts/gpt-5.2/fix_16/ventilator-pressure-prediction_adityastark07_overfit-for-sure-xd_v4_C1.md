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

0.1437697818387486

# 6. Current score

8.73305

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.29945) has done: 'Your current notebook fails because it tries to read three external “dataset” submissions that are not present in this environment, so `sub_1/sub_2/sub_3` never get created and the ensemble step crashes. To keep the same overall approach (create a submission with the right format) while making it run end-to-end here, I replace those missing inputs with a simple, deterministic baseline trained from the provided `train.csv` only. Specifically, we predict the per-time_step mean pressure conditioned on `(R, C, time_step)` computed from train and merged onto test, with a global mean fallback for unseen combinations; this is fast, stable, and produces a valid `submission.csv`. The output file have exactly `id,pressure` and a `.csv` suffix.'
- What this solution (achieved 6.13071) has done: 'Your current score (6.29945 MAE) is far worse than the target (0.14377), so we should legitimately improve accuracy with minimal changes while keeping the same “groupby-mean lookup baseline” core logic. The biggest issue is that using only `(R, C, time_step)` ignores the dominant control signal `u_in` and the on/off state `u_out`, which largely determines pressure; adding them to the aggregation key is a small, direct improvement. To keep it robust, we also add a simple hierarchical fallback (exact key mean → partial key mean without `u_in` → global mean), which reduces errors when exact combinations are unseen. This preserves the same training approach (pure aggregation from train and merge onto test) and still writes a valid `submission.csv`.'
- What this solution (achieved 5.87704) has done: 'Your current baseline is a pure groupby-mean lookup; the main reason it’s far from the target is that rounding `u_in` to 0.1 creates many unseen `(R,C,time_step,u_out,u_in)` bins at test time, forcing frequent fallback to a coarse mean. To move the MAE down toward the target with minimal changes and the same core logic, I (1) avoid rounding `u_in` at all (use exact float values, which match between train/test), and (2) add one extra hierarchical fallback keyed by `(R,C,time_step,u_out,u_in_bin)` using a small bin width so near-by `u_in` values share statistics when exact matches are still missing. This keeps the same “aggregate from train → merge onto test → fillna fallbacks” approach, still runs fast, and still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.36569) has done: 'Your current lookup baseline is still far from the target because it tries to predict absolute pressure directly from raw control inputs, but the scoring ignores expiratory phase and the raw `u_in` scale is highly lung-dependent; a minimal, still-lookup-based improvement is to normalize `u_in` by `(R,C)` within each time step. I keep the same “groupby mean tables + hierarchical fallbacks” core logic, but (1) replace `u_in_exact` with a per-(R,C,time_step) z-scored `u_in` (computed from train only) to improve matching between train/test, and (2) add one extra fallback using the partial key plus the normalized bin, to reduce fallback-to-global. This should legitimately reduce MAE (lower is better) without changing the overall approach or adding any new model/training loop. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.32444) has done: 'Your current lookup baseline is still missing the biggest scoring detail: Kaggle evaluates MAE only on the inspiratory phase (`u_out == 0`), but your aggregation mixes inspiratory and expiratory pressures together, which corrupts the learned means and hurts predictions. With minimal change to the same “groupby-mean tables + hierarchical fallbacks” core logic, I compute all pressure mean tables from `train` filtered to `u_out==0` only, while keeping the same merge keys and fallback order for test. I also keep the `u_in` normalization stats computed on full train (safe and stable) but ensure the final pressure statistics are inspiratory-only, which should move MAE down substantially toward the target without changing the overall approach. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.32436) has done: 'Your current lookup is still very inaccurate mainly because it tries to predict absolute pressure directly, while pressure in this dataset is highly quantized (fixed set of discrete values). A minimal change that often yields a large MAE drop (without changing the “groupby-mean lookup + fallbacks” core logic) is to **snap predictions to the nearest allowed pressure level learned from train inspiratory data** (`u_out==0`), which better matches the evaluation target distribution. I keep all your existing mean tables and fallback order, and only add a final post-processing step that maps each predicted float to the nearest valid pressure value. This is fast, deterministic, and preserves the same training approach and submission semantics while pushing the score down toward the target.'
- What this solution (achieved 7.38218) has done: 'Your current lookup baseline is still far from the target because it tries to predict pressure directly from static per-timestep control values, while pressure is strongly driven by the *history* of `u_in` (integrated flow/volume) within each breath. To move the MAE down substantially (lower is better) without changing the overall “groupby mean lookup + hierarchical fallbacks” approach, I add two cumulative-history features computed per `breath_id` (`u_in_cum` and `area` = cumulative integral of `u_in` over time) and use them as additional coarse lookup keys (binned) in the mean tables. I also fix an ineffective fallback: `mean_table_partial_bin` duplicated the same key as `mean_table_bin`, so I replace it with a more useful fallback that ignores `u_out` (since train pressure means are inspiratory-only anyway) when matching binned history features. Finally, I keep your inspiratory-only training stats and “snap to nearest allowed pressure level” post-processing unchanged.'
- What this solution (achieved 7.66791) has done: 'Your current score is much worse than the target (lower-is-better), so we should make a small, legitimate accuracy improvement without changing the overall “groupby-mean lookup + hierarchical fallbacks + snap-to-valid-levels” approach. The main issue is that `mean_table_full` uses an *exact* floating key (`u_in_z`) which almost never matches between train/test, so it contributes little while adding overhead; replacing that first-stage lookup with a *binned* `u_in_z_bin` (already computed) should greatly increase hit-rate and reduce MAE. To keep changes minimal, I keep your history features, inspiratory-only pressure statistics, the remaining fallback tables, and the snapping post-process unchanged. The resulting pipeline stays deterministic, runs end-to-end fast, and still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 7.78036) has done: 'Your current pipeline is still a pure lookup baseline, but the merge priority is backwards for accuracy: it uses the coarser `(R,C,time_step,u_out,u_in_z_bin)` table first and only then tries the more informative history-conditioned table, so even when history matches exist they get ignored. To move the MAE down toward the target with a minimal change (no new model/training, same features/tables), we simply reorder the fallback so the most specific table `(… + u_in_cum_bin + area_bin)` is used first, then progressively back off to coarser tables. This keeps the same inspiratory-only training stats and the same “snap-to-valid-pressure-levels” post-processing. The result is still deterministic, runs end-to-end, and writes a valid `submission.csv`.'
- What this solution (achieved 7.785) has done: 'Your current lookup baseline is far worse than the target (lower MAE is better), so we should make a small change that increases “exact/near match” hit-rate without changing the overall groupby-mean + fallback + snapping approach. The highest-impact minimal fix is to **compute history bins with floor (not round)** so cumulative features are monotonic and don’t jitter across bin boundaries, which reduces mismatches between train/test for the history-conditioned table. In the same spirit, we also switch `u_in_z_bin` to floor-binning to stabilize bin assignment (still the same feature, just a more consistent discretization), and we keep all tables, merge keys, fallback order, and snapping unchanged. This should move MAE down (closer to the target) while preserving core logic and keeping runtime well under the limit.'
- What this solution (achieved 7.785) has done: 'Your current merge order computes both history-conditioned and non-history-conditioned mean tables, but the code merges the coarse table first and then fills from the more specific history table; this makes it easy for the pipeline to “settle” on less-informative matches. I make the smallest change that preserves your exact lookup-baseline core logic: merge the most-specific history table first, then back off to the coarser tables, and fill in that same priority order. This increases the chance that when a history match exists it is actually used, which should reduce MAE (lower is better) toward your target without changing features, loss, or adding any training loop. The submission writing remains identical and still produces a valid `submission.csv`.'
- What this solution (achieved 7.51387) has done: 'Your current MAE (7.785) is far worse than the target (0.1438), so we need a legitimate accuracy increase while keeping the same “groupby-mean lookup + hierarchical fallbacks + snap-to-valid-levels” core logic. The minimal high-impact fix is to stop using the per-(R,C,time_step) z-score for `u_in` as the primary matching signal, because it creates unstable bins and harms match-rate; instead we bin the raw `u_in` (which matches exactly between train/test) and build/merge the same mean tables using `u_in_bin` (keeping all history bins, inspiratory-only training, fallbacks, and snapping). This is still the same approach (aggregate means on discretized keys, then merge and fill), just with a more reliable discretization of the dominant control input. The output remains a valid `submission.csv` with `id,pressure` and should move MAE down substantially toward your target.'
- What this solution (achieved 8.43743) has done: 'Your current MAE (7.51) is far worse than the target (0.144), so we should improve accuracy with very small, legitimate changes while keeping the same “groupby-mean lookup + hierarchical fallbacks + snap-to-valid-levels” approach. The biggest bug hurting match-rate is the `time_step` handling: rounding and then using the float as a key causes avoidable train/test mismatches; instead, we use an integer `time_step_idx` per breath (0..79) as the join key, which is stable and preserves the same timestep semantics. To further improve hit-rate without changing the core logic, we slightly coarsen the history bins (u_in_bin/u_in_cum_bin/area_bin) so more test rows find a non-null mean in the higher-priority tables, reducing fallback-to-global. Everything else (inspiratory-only pressure tables, merge/fallback order, and snapping to allowed pressure levels) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 8.44641) has done: 'Your current score is far from the target (lower-is-better), so we should make a small, legitimate accuracy improvement while keeping the exact same “groupby mean lookup + hierarchical fallbacks + snap-to-known pressure levels” core logic. The biggest low-risk gain here is to (1) avoid unnecessary expiratory conditioning in the lookup keys since we already train pressure tables on inspiratory-only rows, and (2) add one extra backoff table that uses the strongest signals available at a timestep (`u_in_bin` plus `(R,C,time_step_idx)`) to reduce fallback-to-coarse means when history bins miss. These are minimal changes: we’re not adding a new model or training loop, just improving match-rate in the existing lookup hierarchy. The submission format and file writing remain unchanged.'
- What this solution (achieved 8.73305) has done: 'Your current MAE (8.446) is far above the target (0.144), so we should make the smallest legitimate change that improves accuracy while keeping the same “groupby mean lookup + hierarchical fallbacks + snap-to-known pressure levels” core logic. The main issue is that we train pressure tables on inspiratory rows only, but at test-time we also predict expiratory rows (u_out==1) using inspiratory statistics; while expiratory isn’t scored, it can still distort public LB if Kaggle’s internal alignment differs and it definitely hurts semantic correctness. I keep your exact tables and fallbacks, but add a minimal post-processing step: for `u_out==1` rows, set predictions to a stable low-pressure baseline (the minimum inspiratory pressure level) before snapping (and keep snapping for inspiratory rows). This preserves evaluation semantics (inspiratory remains driven by your lookup) and is a low-risk way to reduce overall MAE variability and typically improves LB slightly.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

required_train_cols = {"breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}
required_test_cols = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}
if not required_train_cols.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {required_train_cols - set(train.columns)}"
    )
if not required_test_cols.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {required_test_cols - set(test.columns)}"
    )
if not {"id", "pressure"}.issubset(sub.columns):
    raise ValueError("sample_submission.csv must have columns: id, pressure")




## === cell 2
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")
    df["time_step_idx"] = (
        df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )

    df["u_in_cum"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()

    dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0).to_numpy()
    u_prev = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(df["u_in"])
        .to_numpy()
    )
    u_cur = df["u_in"].to_numpy()
    inc = 0.5 * (u_prev + u_cur) * dt
    df["area"] = (
        pd.Series(inc, index=df.index).groupby(df["breath_id"], sort=False).cumsum()
    )

    return df


train_feat = add_history_features(train)
test_feat = add_history_features(test)

UIN_BIN_WIDTH = 1.0
train_feat["u_in_bin"] = np.floor(train_feat["u_in"] / UIN_BIN_WIDTH).astype(np.int16)
test_feat["u_in_bin"] = np.floor(test_feat["u_in"] / UIN_BIN_WIDTH).astype(np.int16)

UIN_CUM_BIN = 50.0
AREA_BIN = 25.0
train_feat["u_in_cum_bin"] = np.floor(train_feat["u_in_cum"] / UIN_CUM_BIN).astype(
    np.int16
)
test_feat["u_in_cum_bin"] = np.floor(test_feat["u_in_cum"] / UIN_CUM_BIN).astype(
    np.int16
)
train_feat["area_bin"] = np.floor(train_feat["area"] / AREA_BIN).astype(np.int16)
test_feat["area_bin"] = np.floor(test_feat["area"] / AREA_BIN).astype(np.int16)

insp_mask = train_feat["u_out"].to_numpy() == 0
train_tmp_insp = train_feat.loc[
    insp_mask,
    ["R", "C", "time_step_idx", "u_in_bin", "u_in_cum_bin", "area_bin", "pressure"],
].copy()

mean_table_bin_hist = (
    train_tmp_insp.groupby(
        ["R", "C", "time_step_idx", "u_in_bin", "u_in_cum_bin", "area_bin"],
        as_index=False,
    )["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_mean_bin_hist"})
)

mean_table_full_bin = (
    train_tmp_insp.groupby(["R", "C", "time_step_idx", "u_in_bin"], as_index=False)[
        "pressure"
    ]
    .mean()
    .rename(columns={"pressure": "pressure_mean_full_bin"})
)

mean_table_uin_only = (
    train_tmp_insp.groupby(["R", "C", "u_in_bin"], as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_mean_uin_only"})
)

mean_table_partial = (
    train_tmp_insp.groupby(["R", "C", "time_step_idx"], as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_mean_partial"})
)

global_mean_insp = float(train_feat.loc[insp_mask, "pressure"].mean())

pred_df = (
    test_feat.merge(
        mean_table_bin_hist,
        on=["R", "C", "time_step_idx", "u_in_bin", "u_in_cum_bin", "area_bin"],
        how="left",
    )
    .merge(
        mean_table_full_bin,
        on=["R", "C", "time_step_idx", "u_in_bin"],
        how="left",
    )
    .merge(
        mean_table_uin_only,
        on=["R", "C", "u_in_bin"],
        how="left",
    )
    .merge(
        mean_table_partial,
        on=["R", "C", "time_step_idx"],
        how="left",
    )
)

pred = pred_df["pressure_mean_bin_hist"]
pred = pred.fillna(pred_df["pressure_mean_full_bin"])
pred = pred.fillna(pred_df["pressure_mean_uin_only"])
pred = pred.fillna(pred_df["pressure_mean_partial"])
pred = pred.fillna(global_mean_insp).astype(np.float32).to_numpy()

pressure_levels = np.sort(
    train_feat.loc[insp_mask, "pressure"].unique().astype(np.float32)
)
pressure_levels = np.asarray(pressure_levels, dtype=np.float32)

test_u_out = test_feat["u_out"].to_numpy()
exp_mask_test = test_u_out == 1
if exp_mask_test.any():
    pred[exp_mask_test] = pressure_levels[0]

idx = np.searchsorted(pressure_levels, pred, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_lo = np.clip(idx - 1, 0, len(pressure_levels) - 1)

p_hi = pressure_levels[idx]
p_lo = pressure_levels[idx_lo]
choose_lo = (pred - p_lo) <= (p_hi - pred)
pred_snapped = np.where(choose_lo, p_lo, p_hi).astype(np.float32)

sub = sub.copy()
sub["pressure"] = pred_snapped

if len(sub) != len(test):
    raise ValueError(f"Submission length mismatch: sub={len(sub)} vs test={len(test)}")
if sub["pressure"].isna().any():
    raise ValueError("Found NaNs in predicted pressure")



## === cell 3
sub.to_csv("submission.csv", index=False)
sub.head()
