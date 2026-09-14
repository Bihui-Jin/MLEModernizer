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

0.1414530913604676

# 6. Current score

4.1905

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.77357) has done: 'Your notebook fails because it tries to read four external Kaggle Dataset submissions that don’t exist in your provided environment, so `sub_0` etc. are never created and the ensemble step crashes. To keep the “ensemble” core idea intact while making it runnable, I (1) load `train/test/sample_submission` from the available `/kaggle/input/ventilator-pressure-prediction/` path, (2) generate four surrogate submissions using simple, fast, deterministic baselines derived from training data (group medians/means by `(R,C,time_step)` and by `(R,C,u_in,u_out)`), then (3) apply your original fixed weights (0.656/0.344/0/0) to produce `submission.csv`. This run end-to-end within the Kaggle constraints and produce a valid `submission.csv` with the required columns.'
- What this solution (achieved 9.02501) has done: 'Your current MAE (8.77) is far from the target (0.141, lower is better), so we should make a real-but-minimal modeling improvement without changing the overall “fast deterministic baseline + weighted blend” structure. The biggest issue is that the current predictors ignore the sequential nature and the fact that pressure is only scored during inspiration; a simple way to move the score sharply toward the target is to (1) add a per-time-step estimate built from cumulative `u_in` (a common strong baseline for this competition) and (2) set predictions to 0 when `u_out==1` (expiratory phase, unscored) to avoid wasting error budget on those rows. We keep your existing two groupby-based predictors and the same fixed weights concept, but introduce the new baseline as `sub_2` and give it a modest nonzero weight, renormalizing weights to sum to 1. This remains purely pandas/numpy, runs fast, and still outputs a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 9.16832) has done: 'Your current score is much worse than the target (lower is better), so we should make a small but meaningful accuracy improvement without changing the “fast deterministic baseline + weighted blend” structure. The biggest easy win is to make the cumulative-`u_in` feature more faithful to the physics by integrating `u_in` over time (`u_in * delta_time`) rather than a plain cumulative sum, then bin that integrated signal for stable groupby lookup. We keep your existing sub_0/sub_1 logic intact, keep the same inference/blending flow, and still zero out `u_out==1` rows (unscored phase). This should move MAE substantially downward toward your target while staying purely pandas/numpy and within runtime limits.'
- What this solution (achieved 9.54222) has done: 'Your current MAE is far above the target (lower is better), so we need a small but meaningful accuracy gain while keeping the same “fast deterministic groupby baselines + weighted blend” structure. The easiest win is to avoid learning across fundamentally different inspiratory dynamics by conditioning the groupby tables on `u_out` (inspiration vs expiration) for both the `(R,C,time_step)` and integrated-`u_in` baselines; this keeps the core approach identical but reduces bias. I also use the more stable `median` for the `(R,C,u_in,u_out)` table (instead of `mean`) to reduce the impact of outliers/noise. Finally, I keep your `u_out==1 -> 0` post-processing and keep weights essentially the same (only a tiny adjustment to reflect the slightly stronger `sub_2`), ensuring it still runs fast and writes a valid `submission.csv`.'
- What this solution (achieved 4.61632) has done: 'Your current MAE is far above the target (lower is better), so we need a small-but-real accuracy gain without changing the core “fast deterministic groupby baselines + weighted blend” approach. The biggest low-risk improvement is to quantize/join on `time_step` (which is float and can cause subtle merge mismatches) by using an integer time index per breath, and to add a very light “previous-pressure” lookup baseline keyed by `(R,C,time_idx,u_in,u_out,u_in_prev,u_out_prev)` that captures local dynamics without introducing a new model. We keep your existing sub_0/sub_1/sub_2 logic and blending, just add sub_4 and give it a modest weight while renormalizing to sum to 1. We also keep your `u_out==1 -> 0` post-processing and still write a valid `submission.csv`.'
- What this solution (achieved 4.61634) has done: 'Your current MAE (4.616, lower is better) is still far from the target (0.141), so we should improve accuracy with the smallest possible changes while keeping your “fast deterministic groupby baselines + weighted blend” core intact. The biggest issue is forcing `u_out==1` predictions to 0: although expiratory rows are *not scored*, Kaggle still includes them in the MAE after masking targets by inspiration (so predicting 0 there can still hurt if they are included or if masking differs), and in practice the best public baselines do **not** hard-zero them. Instead, we (1) remove that hard zeroing and (2) add a tiny, safe post-processing step that snaps predictions to the discrete set of pressures seen in training (a well-known property of this dataset), which usually yields a large MAE drop without changing the modeling approach. Everything else (your feature tables, merges, medians, and blending) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 3.86507) has done: 'Your current MAE (4.616, lower is better) is still far from the target (0.141), so we should improve accuracy with a minimal, low-risk change that keeps your exact “groupby baselines + weighted blend + snap-to-known-pressures” core intact. The biggest remaining error source is using exact floating `u_in` in group keys (it’s effectively high-cardinality), which causes sparse tables and many fallbacks; we quantize `u_in` into small bins for the `g1` and `g4` tables so test rows match train more often. This is a small change to feature discretization (not a new model) and should reduce MAE materially while staying deterministic and fast. Everything else (tables, merges, weights, snapping, submission writing) remains the same.'
- What this solution (achieved 3.8652) has done: 'Your current MAE (3.865, lower-is-better) is still far above the target (0.141), so we should make a small, low-risk change that improves accuracy without changing your overall “groupby baselines + weighted blend + snap-to-known-pressures” structure. The biggest easy gain in this competition is enforcing the known pressure plateau after exhalation starts: once `u_out` becomes 1 within a breath, pressure stays essentially constant; we can compute that plateau from your existing blended+snapped predictions and overwrite all later timesteps in the breath with the first `u_out==1` predicted pressure. This is pure post-processing (no model/feature/training change), preserves evaluation semantics, and is fast/deterministic. Everything else (tables, merges, weights, snapping, file paths, and submission writing) stays the same.'
- What this solution (achieved 3.9353) has done: 'Your current MAE (3.8652, lower-is-better) is still far above the target (0.14145), so we should make a small, legitimate accuracy improvement without changing your core “groupby baselines + weighted blend + snap-to-known-pressures + plateau post-process” structure. The biggest low-risk fix is to make the plateau enforcement use the **observed pressure plateau from training** for each `(R,C,time_idx)` during `u_out==1`, instead of using your model’s first `u_out==1` prediction (which can be wrong and then gets propagated). This keeps the same post-processing idea (constant pressure after exhalation starts), but anchors it to a stable, data-driven value and only applies where `u_out==1` (unscored region anyway), reducing the chance of harming inspiratory predictions. Everything else (feature tables, weights, snapping, file paths, and submission writing) remains the same and the script still runs end-to-end to produce `submission.csv`.'
- What this solution (achieved 3.92322) has done: 'Your current MAE (3.9353, lower is better) is still far above the target (0.14145), so we should make a small, legitimate improvement that keeps your exact “groupby baselines + weighted blend + snap-to-known-pressures + plateau post-process” structure intact. The biggest remaining avoidable error is that your tables don’t explicitly use the known “last inspiratory value carries into expiration” property: pressure during `u_out==1` is essentially the last `u_out==0` pressure within the same breath. We add one extra baseline (`sub_5`) that predicts expiratory timesteps by carrying forward the last predicted inspiratory pressure (using your already-snapped blended predictions), and we give it a small weight; this primarily affects unscored rows but can still reduce leaderboard MAE if the masking differs or if your prior plateau anchoring is mismatched. Everything else (features, medians, bins, snapping, paths) remains the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 3.92322) has done: 'Your current MAE (3.923) is still far above the target (0.141, lower is better), so we need a small but real accuracy gain while keeping your exact “groupby baselines + weighted blend + snap-to-known-pressures + plateau/carry post-process” structure intact. The most impactful minimal fix here is to stop blending expiratory (`u_out==1`) behavior into the same lookup tables as inspiratory rows for the high-signal table: we build the `g4` “prev-step” table using only inspiratory rows (where the metric is computed) and use the existing fallbacks for the rest. This preserves your model family and inference flow, but reduces noise/aliasing from expiration dynamics contaminating the inspiratory predictor, which should move MAE down toward the target. Everything else (paths, weights, snapping, plateau/carry logic, and submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 3.92322) has done: 'Your current MAE (3.923, lower is better) is still far above the target (0.141), so we should improve accuracy with the smallest safe change that keeps your exact “groupby baselines + weighted blend + snap-to-known-pressures + plateau/carry post-process” structure. The main issue is that the carry-forward post-process is currently computed on the full time-series without explicitly resetting at the first inspiratory timestep, which can leak a wrong early value through a whole breath when the first rows are `u_out==1` (or when the first inspiratory estimate is missing). I adjust the carry-forward to explicitly use the last known inspiratory prediction per breath and only apply it to expiratory rows after inspiration has occurred, while leaving your lookup tables, binning, blending weights, snapping, and plateau anchoring unchanged. This should reduce avoidable error without changing the modeling family or adding any new training.'
- What this solution (achieved 3.86509) has done: 'We keep your exact “groupby lookup baselines + weighted blend + snap-to-known-pressures + plateau/carry post-process” structure, but fix one key alignment issue that can silently destroy accuracy: your `test` and `train` both show `id` ranges of 1–2000, so using `id` as an index to align predictions to `sample_submission` is wrong (it collapses many rows onto the same id). Instead, we preserve row-order alignment by never reindexing on `id` and build the submission directly from `test["id"]` after sorting once consistently. This is a minimal change, directly relevant to MAE, and should move the score sharply down toward your target while keeping everything else the same. We also add a couple of sanity checks to guarantee we output 603600 rows and no NaNs.'
- What this solution (achieved 4.15728) has done: 'Your current score is still far above the target (lower is better), so the smallest likely improvement without changing your “groupby lookups + weighted blend + snap-to-known-pressures + plateau/carry post-process” core is to make the blending weights slightly more conservative toward the strongest/safest lookup (the per-(R,C,time_idx,u_out) median) and reduce reliance on the higher-cardinality prev-step table that can be noisy/sparse. This keeps the exact same predictors, snapping, and post-processing, but shifts the mixture to reduce overfitting-like artifacts and should move MAE modestly downward toward your target. I also add one safety clamp to the known pressure range after post-processing (does not change semantics, just avoids any rare out-of-range float drift) and keep the submission alignment logic unchanged (row-order safe). The code still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 4.1905) has done: 'Your MAE is still far above the target (lower is better), so we should make a minimal change that’s likely to improve accuracy without changing your overall “groupby lookup baselines + weighted blend + snap-to-known-pressures + plateau/carry post-process” structure. The smallest high-impact fix is to ensure the plateau and carry post-processing are *consistent with the competition metric*: expiratory phase (`u_out==1`) is not scored, so any post-processing that changes expiratory predictions cannot improve the metric and can only hurt if the evaluator masking differs or if there’s any implementation mismatch. Therefore, we keep all your predictors/blending/snapping exactly the same, but remove the plateau/carry overwrite and instead apply a conservative smoothing only within inspiratory phase using a per-breath, 3-tap moving average on the already-snapped predictions (this is post-processing, not a new model). This should reduce small timestep noise during inspiration (scored region) and move MAE downward while remaining deterministic and fast and still producing a valid `submission.csv`.'

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

train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)
test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

train["time_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)
test["time_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

g0 = (
    train.groupby(["R", "C", "time_idx", "u_out"], as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_pred"})
)
test0 = test.merge(g0, on=["R", "C", "time_idx", "u_out"], how="left")
fallback0 = train["pressure"].median()
sub_0 = test0["pressure_pred"].fillna(fallback0).to_numpy(np.float32)

u_in_bin_size = 0.5
train["u_in_bin"] = ((train["u_in"] / u_in_bin_size).round()).astype(np.int16)
test["u_in_bin"] = ((test["u_in"] / u_in_bin_size).round()).astype(np.int16)

g1 = (
    train.groupby(["R", "C", "u_in_bin", "u_out"], as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_pred"})
)
test1 = test.merge(g1, on=["R", "C", "u_in_bin", "u_out"], how="left")
fallback1 = train["pressure"].median()
sub_1 = test1["pressure_pred"].fillna(fallback1).to_numpy(np.float32)

train_tmp = train[
    ["breath_id", "R", "C", "time_idx", "time_step", "u_in", "u_out", "pressure"]
].copy()
test_tmp = test[
    ["breath_id", "R", "C", "time_idx", "time_step", "u_in", "u_out"]
].copy()

train_tmp["dt"] = (
    train_tmp.groupby("breath_id")["time_step"]
    .diff()
    .fillna(train_tmp["time_step"])
    .astype(np.float32)
)
test_tmp["dt"] = (
    test_tmp.groupby("breath_id")["time_step"]
    .diff()
    .fillna(test_tmp["time_step"])
    .astype(np.float32)
)

train_tmp["u_in_int"] = (
    (train_tmp["u_in"].astype(np.float32) * train_tmp["dt"])
    .groupby(train_tmp["breath_id"])
    .cumsum()
)
test_tmp["u_in_int"] = (
    (test_tmp["u_in"].astype(np.float32) * test_tmp["dt"])
    .groupby(test_tmp["breath_id"])
    .cumsum()
)

bin_size = 0.25
train_tmp["u_in_int_bin"] = (train_tmp["u_in_int"] / bin_size).round().astype(np.int32)
test_tmp["u_in_int_bin"] = (test_tmp["u_in_int"] / bin_size).round().astype(np.int32)

g2 = (
    train_tmp.groupby(["R", "C", "time_idx", "u_out", "u_in_int_bin"], as_index=False)[
        "pressure"
    ]
    .median()
    .rename(columns={"pressure": "pressure_pred"})
)
test2 = test_tmp.merge(
    g2, on=["R", "C", "time_idx", "u_out", "u_in_int_bin"], how="left"
)

test2 = test2.merge(
    g0.rename(columns={"pressure_pred": "fallback_rc_t"}),
    on=["R", "C", "time_idx", "u_out"],
    how="left",
)
fallback2 = fallback0

sub_2 = (
    test2["pressure_pred"]
    .fillna(test2["fallback_rc_t"])
    .fillna(fallback2)
    .to_numpy(np.float32)
)

sub_3 = np.zeros(len(test), dtype=np.float32)

train_prev = train[
    ["breath_id", "R", "C", "time_idx", "u_in", "u_out", "pressure", "u_in_bin"]
].copy()
test_prev = test[
    ["breath_id", "R", "C", "time_idx", "u_in", "u_out", "u_in_bin"]
].copy()

train_prev["u_in_prev"] = (
    train_prev.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
)
train_prev["u_out_prev"] = (
    train_prev.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
)
test_prev["u_in_prev"] = (
    test_prev.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
)
test_prev["u_out_prev"] = (
    test_prev.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
)

u_in_prev_bin_size = 0.5
train_prev["u_in_prev_bin"] = (
    (train_prev["u_in_prev"] / u_in_prev_bin_size).round()
).astype(np.int16)
test_prev["u_in_prev_bin"] = (
    (test_prev["u_in_prev"] / u_in_prev_bin_size).round()
).astype(np.int16)

train_prev_insp = train_prev.loc[train_prev["u_out"] == 0].copy()

g4 = (
    train_prev_insp.groupby(
        ["R", "C", "time_idx", "u_out", "u_in_bin", "u_in_prev_bin", "u_out_prev"],
        as_index=False,
    )["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_pred"})
)
test4 = test_prev.merge(
    g4,
    on=["R", "C", "time_idx", "u_out", "u_in_bin", "u_in_prev_bin", "u_out_prev"],
    how="left",
)

test4 = test4.merge(
    g1.rename(columns={"pressure_pred": "fallback_rc_u"}),
    on=["R", "C", "u_in_bin", "u_out"],
    how="left",
)
test4 = test4.merge(
    g0.rename(columns={"pressure_pred": "fallback_rc_t"}),
    on=["R", "C", "time_idx", "u_out"],
    how="left",
)

sub_4 = (
    test4["pressure_pred"]
    .fillna(test4["fallback_rc_u"])
    .fillna(test4["fallback_rc_t"])
    .fillna(fallback0)
    .to_numpy(np.float32)
)

n_test = len(test)
assert (
    sub_0.shape[0] == n_test
    and sub_1.shape[0] == n_test
    and sub_2.shape[0] == n_test
    and sub_4.shape[0] == n_test
)
assert (
    np.isfinite(sub_0).all()
    and np.isfinite(sub_1).all()
    and np.isfinite(sub_2).all()
    and np.isfinite(sub_4).all()
)

pd.DataFrame({"id": test["id"].head().values, "p0": sub_0[:5], "p1": sub_1[:5]}).head()



## === cell 2
w0, w1, w2, w3, w4 = 0.48, 0.16, 0.30, 0.0, 0.06
ws = w0 + w1 + w2 + w3 + w4
w0, w1, w2, w3, w4 = [w / ws for w in (w0, w1, w2, w3, w4)]

pred = (
    (sub_0 * w0) + (sub_1 * w1) + (sub_2 * w2) + (sub_3 * w3) + (sub_4 * w4)
).astype(np.float32)

pressure_values = np.sort(train["pressure"].unique()).astype(np.float32)

idx = np.searchsorted(pressure_values, pred, side="left")
idx = np.clip(idx, 0, len(pressure_values) - 1)

idx_left = np.clip(idx - 1, 0, len(pressure_values) - 1)
p_right = pressure_values[idx]
p_left = pressure_values[idx_left]
choose_left = np.abs(pred - p_left) <= np.abs(pred - p_right)
pred_snapped = np.where(choose_left, p_left, p_right).astype(np.float32)

breath = test["breath_id"].to_numpy()
u_out = test["u_out"].to_numpy(np.int8)

s = pd.Series(pred_snapped)

prev_ = s.groupby(breath, sort=False).shift(1)
next_ = s.groupby(breath, sort=False).shift(-1)

prev_ = prev_.fillna(s)
next_ = next_.fillna(s)

smooth = (
    (prev_.to_numpy(np.float32) + pred_snapped + next_.to_numpy(np.float32)) / 3.0
).astype(np.float32)

pred_final = pred_snapped.copy()
insp_mask = u_out == 0
pred_final[insp_mask] = smooth[insp_mask]

pred_final = np.clip(pred_final, pressure_values.min(), pressure_values.max()).astype(
    np.float32
)

submission = pd.DataFrame({"id": test["id"].values, "pressure": pred_final})
assert len(submission) == len(test)
assert submission["pressure"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)
submission.head(5)
