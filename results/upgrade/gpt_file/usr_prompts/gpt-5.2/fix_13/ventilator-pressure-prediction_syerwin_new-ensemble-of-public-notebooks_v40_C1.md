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

0.1460308628842259

# 6. Current score

1.65558

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.1958) has done: 'The notebook currently fails because it tries to read four external “../input/…” submission files that are not present in your environment; that prevents any submission from being written. To make it run end-to-end and produce a valid `submission.csv`, I keep the same blending core idea but fall back to a simple, fully-local baseline whenever those external blend components are missing. The fallback uses only the provided competition train/test files and predicts the mean training pressure per (R, C, time_step index within breath) which is fast, stable, and typically non-trivial versus all-zeros. Finally, the code enforces correct `id,pressure` formatting and alignment to the sample submission.'
- What this solution (achieved 6.19013) has done: 'Your current MAE (6.1958) is far above the target (0.1460), so we need a real modeling lift while keeping the solution simple and end-to-end. The biggest issue is that the fallback baseline ignores key dynamics (`u_in`, `u_out`, `time_step`) and the fact that expiratory phase (`u_out==1`) is unscored; we can improve a lot by using an in-breath, time-aligned mean conditioned on `(R, C, u_out, step)` and a stronger hierarchical fallback, then explicitly forcing expiratory predictions to a neutral value derived from training. These are minimal changes that keep your “local mean-by-groups” core logic, but align it better to the competition’s scoring and data structure. The blending path logic is kept intact; we only upgrade the local baseline that is used when external submissions are missing.'
- What this solution (achieved 4.3088) has done: 'Your current score (6.19 MAE) is far worse than the target (0.146), so we need a meaningful lift while preserving your “local group-mean baseline” core logic. The biggest improvement available without changing the approach is to condition the group means on `u_in` as well (discretized into bins) because `u_in` is the primary control signal driving pressure; keeping `step` and `R,C,u_out` maintains the same time-aligned grouping idea. I’m also keeping your hierarchical fallback structure, just extending it to include the binned `u_in` where possible, and I keep the explicit stable expiratory (`u_out==1`) override. These changes should substantially reduce MAE while staying fast and fully local, producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.59285) has done: 'Your current MAE (4.3088) is still far above the target (0.1460), so we need a clear lift while keeping your same “group-mean with hierarchical backoff” core logic. The most score-relevant missing signal is temporal memory within a breath, so we add minimal lag features (`u_in_prev`, `u_out_prev`) and a cumulative integral proxy (`u_in_cum`) and condition the group means on discretized versions of these, keeping the same merge+fillna backoff pattern. We also align better with the evaluation by learning means only on inspiratory rows (`u_out==0`) and continuing to force expiratory predictions to a stable expiratory level. All I/O paths remain unchanged and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 2.02381) has done: 'To move your MAE down toward the 0.146 target (lower is better) without changing the overall “group-mean with hierarchical backoff” approach, I make two minimal, score-aligned upgrades. First, I add a small amount of additional within-breath state by including discretized `u_in_diff` (current minus previous) and `u_in_slope` (diff divided by `dt`) as extra grouping keys—this preserves the same merge/fillna logic but better captures dynamics that drive pressure. Second, I slightly refine the discretization for `u_in`/`u_in_prev` (1.0 instead of 2.0) while keeping cumulative binning coarse to avoid explosion in group cardinality and runtime. Everything else (inspiratory-only fitting, expiratory override, file paths, and submission writing) remains the same.'
- What this solution (achieved 1.97936) has done: 'Your current MAE (2.02381, lower is better) is still far above the target (0.14603), so we should improve the baseline while keeping the exact same “group-mean with hierarchical backoff” core logic. The biggest low-risk gain is to align grouping and fallback to the evaluation: train group means only on inspiratory rows (already done) and also make test-time matching more faithful by adding a discretized `time_step` key (in addition to `step`) to reduce cross-breath timing misalignment. To avoid over-fragmenting groups, we add `time_step_bin` only to the strongest table and keep the rest of the backoff chain unchanged. This is minimal, fast, and should reduce MAE by improving nearest-neighbor matching of dynamics without changing the overall approach.'
- What this solution (achieved 1.86995) has done: 'Your current MAE (1.97936; lower is better) is still far above the target (0.14603), so we should cautiously improve accuracy while keeping your exact “group-mean with hierarchical backoff” approach unchanged. The biggest low-risk issue is that `u_in_slope_bin` is computed from a raw signed slope, which creates many sparse/rare groups and hurts generalization; binning its magnitude instead (and clipping to a reasonable range) keeps the same feature but stabilizes group matching. Similarly, `u_in_diff_bin` can be clipped to avoid extreme bins that fragment groups without adding useful signal. These are minimal changes (same data, same grouping/merge/fillna chain, same inspiratory-only fitting and expiratory override) that should reduce MAE toward the target band.'
- What this solution (achieved 1.86995) has done: 'Your current MAE (1.86995, lower-is-better) is still far above the target (0.14603), so we need a small but meaningful accuracy lift while keeping the same “group-mean with hierarchical backoff” core logic intact. The most impactful missing signal that fits your existing approach is `u_out` itself: you currently learn all inspiratory means but don’t include `u_out` as a key, which can cause mismatches near phase transitions and forces reliance on weaker fallbacks. I add `u_out` to the strongest and first mid-level group tables (and their merges) while keeping the rest of the backoff chain unchanged, plus keep your existing expiratory override exactly as-is. This is a minimal, score-aligned change that should reduce MAE by improving exact-match hit rate without changing the modeling paradigm.'
- What this solution (achieved 1.86982) has done: 'Your current MAE (1.86995, lower-is-better) is far above the target (0.14603), so we should improve accuracy with minimal risk while preserving your exact “group-mean with hierarchical backoff” approach. The most impactful, still-on-paradigm fix is to account for the fact that many pressure values are effectively “quantized” to a fixed grid in this competition; snapping predictions to the nearest observed pressure level typically yields a large MAE reduction without changing the model. I add a tiny post-processing step that builds the sorted unique pressure grid from train and rounds all predicted pressures to the nearest grid value, then keeps your existing expiratory override. This keeps all I/O and the full grouping/merge/fillna logic intact and still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.86982) has done: 'Your current MAE (1.86982, lower-is-better) is still far above the target (0.14603), so we should improve accuracy while keeping the exact same “group-mean with hierarchical backoff” core logic. The smallest high-impact change within this paradigm is to stop forcing all expiratory (`u_out==1`) predictions to a single constant, because although expiratory rows are not scored, this can harm sequence-consistency features (prev-state bins) near the transition and can also be unnecessary. Instead, we keep the expiratory override but make it *time/setting-aware* by using a simple group-median table keyed by `(R,C,step)` for expiratory rows, with a safe fallback to the global expiratory median when missing. Everything else (features, grouping tables, merge+fillna chain, pressure-grid snapping, file paths, and submission writing) remains unchanged.'
- What this solution (achieved 1.87357) has done: 'Your MAE (1.86982; lower is better) is still far above the target (0.14603), so we should improve accuracy while keeping your exact “group-mean with hierarchical backoff + pressure-grid snapping” core logic. The most score-relevant issue is that we currently predict pressures for rows where `u_out==0` without conditioning on the *current* `u_out` in the strongest table at train-time/test-time consistently (because training is restricted to `u_out==0`, `u_out` is effectively constant and not informative), while the most important scoring mask is actually “inspiratory phase” which is better approximated by `u_out==0` but also benefits from conditioning on “first expiratory step” boundary. A minimal, on-paradigm improvement is to add a single additional key that captures phase transition: `is_exhale_transition` = (u_out==1 and u_out_prev==0) and also include an inspiratory-only version `is_insp_last` = (u_out==0 and next u_out==1) in the strongest table; this helps the model learn distinct pressures near the transition where your lag-based bins otherwise mismatch. Everything else (feature set style, merge/fillna backoff chain, inspiratory-only fitting, expiratory median override, and pressure-grid snapping) is kept intact, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 1.65558) has done: 'Your current MAE (1.87357, lower-is-better) is far above the target (0.14603), so we need a meaningful accuracy lift while keeping your exact “group-mean + hierarchical backoff + pressure-grid snapping” core logic intact. The biggest score-aligned fix that doesn’t change the approach is to condition the strongest/mid tables on the *breath trajectory* by adding discretized `breath_u_in_mean` and `breath_u_in_max` (computed per breath from control inputs), which helps distinguish different maneuver intensities under the same (R,C,step,local bins). This remains the same merge/fillna pattern (no model/optimization changes), but increases exact-match hit rate and reduces reliance on weak fallbacks. I also keep your inspiratory-only fitting and expiratory median override exactly as-is, and still write a valid `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
SAMPLE_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"

sub = pd.read_csv(SAMPLE_PATH)

blend_paths = {
    "sub_1": "/kaggle/input/vpp-lstm-baseline-median-pp/submission.csv",
    "sub_2": "/kaggle/input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    "sub_3": "/kaggle/input/gb-vpp-whoppity-dub-dub/median_submission.csv",
    "sub_4": "/kaggle/input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
}

loaded = {}
missing = []
for name, path in blend_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            missing.append(name)
        else:
            loaded[name] = df[["pressure"]].copy()
    else:
        missing.append(name)

print("Blend components found:", sorted(list(loaded.keys())))
print("Blend components missing:", sorted(missing))




## === cell 2
def _snap_to_pressure_grid(
    pred_pressure: np.ndarray, pressure_grid: np.ndarray
) -> np.ndarray:
    """
    Change (score-relevant, preserves evaluation semantics):
    Kaggle's ventilator pressures lie on a fixed discrete grid. Snapping continuous predictions
    to the nearest observed pressure level usually reduces MAE materially without changing
    the underlying modeling approach (still the same group-mean baseline).
    """
    pred_pressure = np.asarray(pred_pressure, dtype=np.float32)
    grid = np.asarray(pressure_grid, dtype=np.float32)
    if grid.size == 0:
        return pred_pressure

    idx = np.searchsorted(grid, pred_pressure, side="left")
    idx0 = np.clip(idx - 1, 0, grid.size - 1)
    idx1 = np.clip(idx, 0, grid.size - 1)
    g0 = grid[idx0]
    g1 = grid[idx1]
    choose1 = np.abs(pred_pressure - g1) < np.abs(pred_pressure - g0)
    snapped = np.where(choose1, g1, g0).astype(np.float32)
    return snapped


def build_local_baseline_submission(sample_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Local, fast baseline (no external submissions).

    Core logic unchanged: group-mean tables + hierarchical merge/fillna backoff, trained on
    inspiratory rows and with an expiratory override.
    """
    usecols_train = ["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"]
    usecols_test = ["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"]

    train = pd.read_csv(TRAIN_PATH, usecols=usecols_train)
    test = pd.read_csv(TEST_PATH, usecols=usecols_test)

    pressure_grid = np.sort(train["pressure"].unique().astype(np.float32))

    train["step"] = train.groupby("breath_id").cumcount().astype(np.int16)
    test["step"] = test.groupby("breath_id").cumcount().astype(np.int16)

    BIN_T = 0.02
    train["time_step_bin"] = np.floor(train["time_step"].values / BIN_T).astype(
        np.int16
    )
    test["time_step_bin"] = np.floor(test["time_step"].values / BIN_T).astype(np.int16)

    exp_mask = train["u_out"].values == 1
    if exp_mask.any():
        exp_pressure_global = float(train.loc[exp_mask, "pressure"].median())
    else:
        exp_pressure_global = float(train["pressure"].median())

    train["u_in_prev"] = (
        train.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    test["u_in_prev"] = (
        test.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )

    train["u_out_prev"] = (
        train.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    test["u_out_prev"] = (
        test.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    train["u_out_next"] = (
        train.groupby("breath_id")["u_out"].shift(-1).fillna(1).astype(np.int8)
    )
    test["u_out_next"] = (
        test.groupby("breath_id")["u_out"].shift(-1).fillna(1).astype(np.int8)
    )
    train["is_insp_last"] = (
        (train["u_out"].values == 0) & (train["u_out_next"].values == 1)
    ).astype(np.int8)
    test["is_insp_last"] = (
        (test["u_out"].values == 0) & (test["u_out_next"].values == 1)
    ).astype(np.int8)
    train["is_exhale_transition"] = (
        (train["u_out"].values == 1) & (train["u_out_prev"].values == 0)
    ).astype(np.int8)
    test["is_exhale_transition"] = (
        (test["u_out"].values == 1) & (test["u_out_prev"].values == 0)
    ).astype(np.int8)

    train["u_in_cum"] = train.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
    test["u_in_cum"] = test.groupby("breath_id")["u_in"].cumsum().astype(np.float32)

    train["u_in_diff"] = (train["u_in"].astype(np.float32) - train["u_in_prev"]).astype(
        np.float32
    )
    test["u_in_diff"] = (test["u_in"].astype(np.float32) - test["u_in_prev"]).astype(
        np.float32
    )

    train["dt"] = (
        train.groupby("breath_id")["time_step"]
        .diff()
        .fillna(train["time_step"])
        .astype(np.float32)
    )
    test["dt"] = (
        test.groupby("breath_id")["time_step"]
        .diff()
        .fillna(test["time_step"])
        .astype(np.float32)
    )

    train_dt = np.maximum(train["dt"].values, 1e-6).astype(np.float32)
    test_dt = np.maximum(test["dt"].values, 1e-6).astype(np.float32)

    train["u_in_slope"] = (train["u_in_diff"].values / train_dt).astype(np.float32)
    test["u_in_slope"] = (test["u_in_diff"].values / test_dt).astype(np.float32)

    BIN_UIN = 1.0
    BIN_UIN_PREV = 1.0
    BIN_UIN_CUM = 20.0
    BIN_UIN_DIFF = 1.0
    BIN_UIN_SLOPE = 5.0

    train["u_in_bin"] = np.floor(train["u_in"].values / BIN_UIN).astype(np.int16)
    test["u_in_bin"] = np.floor(test["u_in"].values / BIN_UIN).astype(np.int16)

    train["u_in_prev_bin"] = np.floor(train["u_in_prev"].values / BIN_UIN_PREV).astype(
        np.int16
    )
    test["u_in_prev_bin"] = np.floor(test["u_in_prev"].values / BIN_UIN_PREV).astype(
        np.int16
    )

    train["u_in_cum_bin"] = np.floor(train["u_in_cum"].values / BIN_UIN_CUM).astype(
        np.int16
    )
    test["u_in_cum_bin"] = np.floor(test["u_in_cum"].values / BIN_UIN_CUM).astype(
        np.int16
    )

    DIFF_CLIP = 30.0
    train_diff_clip = np.clip(train["u_in_diff"].values, -DIFF_CLIP, DIFF_CLIP)
    test_diff_clip = np.clip(test["u_in_diff"].values, -DIFF_CLIP, DIFF_CLIP)
    train["u_in_diff_bin"] = np.floor(train_diff_clip / BIN_UIN_DIFF).astype(np.int16)
    test["u_in_diff_bin"] = np.floor(test_diff_clip / BIN_UIN_DIFF).astype(np.int16)

    SLOPE_ABS_CLIP = 60.0
    train_slope_abs_clip = np.clip(
        np.abs(train["u_in_slope"].values), 0.0, SLOPE_ABS_CLIP
    )
    test_slope_abs_clip = np.clip(
        np.abs(test["u_in_slope"].values), 0.0, SLOPE_ABS_CLIP
    )
    train["u_in_slope_bin"] = np.floor(train_slope_abs_clip / BIN_UIN_SLOPE).astype(
        np.int16
    )
    test["u_in_slope_bin"] = np.floor(test_slope_abs_clip / BIN_UIN_SLOPE).astype(
        np.int16
    )

    BREATH_UIN_MEAN_BIN = 2.0
    BREATH_UIN_MAX_BIN = 2.0
    tr_breath_stats = (
        train.groupby("breath_id", sort=False)["u_in"]
        .agg(breath_u_in_mean="mean", breath_u_in_max="max")
        .reset_index()
    )
    te_breath_stats = (
        test.groupby("breath_id", sort=False)["u_in"]
        .agg(breath_u_in_mean="mean", breath_u_in_max="max")
        .reset_index()
    )
    tr_breath_stats["breath_u_in_mean_bin"] = np.floor(
        tr_breath_stats["breath_u_in_mean"].values / BREATH_UIN_MEAN_BIN
    ).astype(np.int16)
    tr_breath_stats["breath_u_in_max_bin"] = np.floor(
        tr_breath_stats["breath_u_in_max"].values / BREATH_UIN_MAX_BIN
    ).astype(np.int16)
    te_breath_stats["breath_u_in_mean_bin"] = np.floor(
        te_breath_stats["breath_u_in_mean"].values / BREATH_UIN_MEAN_BIN
    ).astype(np.int16)
    te_breath_stats["breath_u_in_max_bin"] = np.floor(
        te_breath_stats["breath_u_in_max"].values / BREATH_UIN_MAX_BIN
    ).astype(np.int16)

    train = train.merge(
        tr_breath_stats[["breath_id", "breath_u_in_mean_bin", "breath_u_in_max_bin"]],
        on="breath_id",
        how="left",
    )
    test = test.merge(
        te_breath_stats[["breath_id", "breath_u_in_mean_bin", "breath_u_in_max_bin"]],
        on="breath_id",
        how="left",
    )

    train_insp = train.loc[train["u_out"].values == 0].copy()

    global_mean = (
        float(train_insp["pressure"].mean())
        if len(train_insp)
        else float(train["pressure"].mean())
    )

    mean_strong = (
        train_insp.groupby(
            [
                "R",
                "C",
                "step",
                "time_step_bin",
                "u_in_bin",
                "u_in_prev_bin",
                "u_in_cum_bin",
                "u_in_diff_bin",
                "u_in_slope_bin",
                "u_out_prev",
                "is_insp_last",
                "breath_u_in_mean_bin",
                "breath_u_in_max_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
    )

    mean_mid1 = (
        train_insp.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_prev_bin",
                "u_in_diff_bin",
                "u_out_prev",
                "breath_u_in_mean_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_mid1"})
    )

    mean_mid2 = (
        train_insp.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_diff_bin", "u_out_prev"], sort=False
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_mid2"})
    )
    mean_mid3 = (
        train_insp.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_mid3"})
    )
    mean_rcs = (
        train_insp.groupby(["R", "C", "step"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_rcs"})
    )
    mean_rs = (
        train_insp.groupby(["R", "step"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_rs"})
    )
    mean_cs = (
        train_insp.groupby(["C", "step"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_cs"})
    )
    mean_s = (
        train_insp.groupby(["step"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p_s"})
    )

    train_exp = train.loc[train["u_out"].values == 1, ["R", "C", "step", "pressure"]]
    if len(train_exp):
        exp_rcs = (
            train_exp.groupby(["R", "C", "step"], sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "p_exp_rcs"})
        )
    else:
        exp_rcs = pd.DataFrame(columns=["R", "C", "step", "p_exp_rcs"])

    pred = test.merge(
        mean_strong,
        on=[
            "R",
            "C",
            "step",
            "time_step_bin",
            "u_in_bin",
            "u_in_prev_bin",
            "u_in_cum_bin",
            "u_in_diff_bin",
            "u_in_slope_bin",
            "u_out_prev",
            "is_insp_last",
            "breath_u_in_mean_bin",
            "breath_u_in_max_bin",
        ],
        how="left",
    )
    pred = pred.merge(
        mean_mid1,
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_prev_bin",
            "u_in_diff_bin",
            "u_out_prev",
            "breath_u_in_mean_bin",
        ],
        how="left",
    )
    pred = pred.merge(
        mean_mid2,
        on=["R", "C", "step", "u_in_bin", "u_in_diff_bin", "u_out_prev"],
        how="left",
    )
    pred = pred.merge(mean_mid3, on=["R", "C", "step", "u_in_bin"], how="left")
    pred = pred.merge(mean_rcs, on=["R", "C", "step"], how="left")
    pred = pred.merge(mean_rs, on=["R", "step"], how="left")
    pred = pred.merge(mean_cs, on=["C", "step"], how="left")
    pred = pred.merge(mean_s, on=["step"], how="left")

    if len(exp_rcs):
        pred = pred.merge(exp_rcs, on=["R", "C", "step"], how="left")
    else:
        pred["p_exp_rcs"] = np.nan

    pred["pressure"] = pred["pressure"].fillna(pred["p_mid1"])
    pred["pressure"] = pred["pressure"].fillna(pred["p_mid2"])
    pred["pressure"] = pred["pressure"].fillna(pred["p_mid3"])
    pred["pressure"] = pred["pressure"].fillna(pred["p_rcs"])
    pred["pressure"] = pred["pressure"].fillna(pred["p_rs"])
    pred["pressure"] = pred["pressure"].fillna(pred["p_cs"])
    pred["pressure"] = pred["pressure"].fillna(pred["p_s"])
    pred["pressure"] = pred["pressure"].fillna(global_mean)

    exp_test_mask = pred["u_out"].values == 1
    if exp_test_mask.any():
        exp_vals = pred.loc[exp_test_mask, "p_exp_rcs"].astype(np.float32).values
        fill_vals = np.where(
            np.isfinite(exp_vals), exp_vals, np.float32(exp_pressure_global)
        )
        pred.loc[exp_test_mask, "pressure"] = fill_vals

    out = sample_sub[["id"]].merge(pred[["id", "pressure"]], on="id", how="left")
    out["pressure"] = out["pressure"].fillna(global_mean).astype(np.float32)

    out["pressure"] = _snap_to_pressure_grid(out["pressure"].values, pressure_grid)

    test_uout_by_id = test[["id", "u_out", "R", "C", "step"]]
    out = out.merge(test_uout_by_id, on="id", how="left")
    if len(exp_rcs):
        out = out.merge(exp_rcs, on=["R", "C", "step"], how="left")
    else:
        out["p_exp_rcs"] = np.nan

    exp_out_mask = out["u_out"].values == 1
    if exp_out_mask.any():
        exp_vals2 = out.loc[exp_out_mask, "p_exp_rcs"].astype(np.float32).values
        fill_vals2 = np.where(
            np.isfinite(exp_vals2), exp_vals2, np.float32(exp_pressure_global)
        )
        out.loc[exp_out_mask, "pressure"] = fill_vals2

    out = out[["id", "pressure"]]
    return out




## === cell 3
if all(k in loaded for k in ["sub_1", "sub_2", "sub_3", "sub_4"]):
    sub["pressure"] = (
        (loaded["sub_1"]["pressure"].values * 0.0)
        + (loaded["sub_2"]["pressure"].values * 0.3)
        + (loaded["sub_3"]["pressure"].values * 0.5)
        + (loaded["sub_4"]["pressure"].values * 0.2)
    )
    submission = sub[["id", "pressure"]].copy()
else:
    submission = build_local_baseline_submission(sub)

assert list(submission.columns) == ["id", "pressure"]
assert len(submission) == len(sub)
assert submission["pressure"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
