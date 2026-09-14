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

0.1547552849455595

# 6. Current score

3.5027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.97534) has done: 'Your current notebook fails because it tries to read several external Kaggle Dataset/Notebook submission files that do not exist in this environment, so `sub_1`…`sub_5` are never created and blending crashes. To make the pipeline run end-to-end and produce a valid `submission.csv`, I replace that external-file blending with an in-notebook baseline that uses only the provided `train.csv` and `test.csv`. The minimal, stable approach is to predict pressure by the mean pressure learned from train for each `(R, C, time_step)` combination, with a global fallback mean for unseen combinations. This preserves evaluation semantics (no leakage) and should yield a reasonable MAE (likely worse than the target, but it run and generate a valid submission file).'
- What this solution (achieved 4.83808) has done: 'I fix the merge logic that drops the fallback key columns (causing the `KeyError: 'R'`) by doing fallback joins on `id` instead of re-merging on keys after those columns have been lost. I also ensure we preserve row order and uniqueness by building a single prediction frame aligned to `test` and then writing `submission.csv`. These changes are minimal and keep the same mean-encoding + hierarchical fallback + pressure-level snapping core logic, but make the pipeline run end-to-end and reliably produce a valid submission file.'
- What this solution (achieved 4.0349) has done: 'Your current approach is a pure lookup/mean-encoding with fallbacks; the biggest score gain without changing the core logic is to ensure train/test keys match as often as possible. I make the rounding consistent with the natural discretization of the data by snapping `time_step` to the nearest 0.02s grid (instead of arbitrary 0.001 rounding), which increases exact key hits and reduces reliance on coarse fallbacks. I also add one more minimal fallback level that drops `u_out` only when needed (still a mean-encoding fallback, not a new model), which should reduce error when `u_out` mismatches cause missing joins. Submission writing, schema, and pressure-level snapping remain unchanged.'
- What this solution (achieved 3.84077) has done: 'Your current mean-encoding lookup is missing one key piece of ventilator signal structure: pressure during inspiration depends strongly on the *current* `u_in` (not just its lag/cumsum), so many of your joins are effectively too coarse and fall back to weak averages, keeping MAE high. I make a minimal change that preserves the same “groupby means + hierarchical fallbacks + pressure snapping” core logic by adding a rounded `u_in` into the primary key and fallbacks, which increases exact/near-exact key hits without changing the overall approach. I also add one extra intermediate fallback that drops only `u_in_lag1_r` (but keeps `u_in_r`) before dropping `u_in_r`, to reduce fallback error while staying within the same semantics. This should move the score substantially down toward the 0.1548 target while keeping the pipeline stable and still producing a valid `submission.csv`.'
- What this solution (achieved 3.53517) has done: 'I keep your current “groupby means + hierarchical fallbacks + snap-to-known-pressure-levels” approach, but make the join keys match train/test more often to reduce fallback usage (which is currently driving the MAE). The smallest safe improvement is to (1) snap `u_in` to the natural 0.1 grid using an integer representation to avoid float join mismatches, and (2) add one more intermediate fallback that drops only `u_in_lag1` before dropping all lag/cumsum info. These changes preserve your core logic and semantics, but should materially decrease error (toward the 0.1548 target) by increasing exact key hits. The rest of the pipeline (file discovery, ordering, submission schema, and pressure snapping) stays the same.'
- What this solution (achieved 3.53514) has done: 'I fix the runtime error by removing the duplicate `u_out` column from `base_cols`, which currently makes `base.merge(..., on=KEY_COLS)` fail because `u_out` appears twice. Then I keep the exact same mean-lookup + hierarchical fallback + pressure-level snapping logic, but ensure the final `u_out==1 -> pressure=0` rule is applied using the already-present `u_out` column (no extra merge needed). Finally, I make sure the script always writes a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 4.3868) has done: 'Your current score (3.535) is far above the target (0.1548) for a lower-is-better MAE, so we need a meaningful improvement while keeping your same “groupby-mean lookup + hierarchical fallbacks + snap-to-known-levels” core logic. The biggest issue is that your keys are still too sparse (many test rows miss the main/fallback joins), and your `u_out==1 -> 0` post-rule is likely hurting because the metric ignores expiratory phase anyway (and pressure isn’t 0 there). I (1) add a very small set of extra “breath-local” features (`time_step_idx` within breath and `u_in_diff`) and include them only in the higher-priority keys to increase match quality without changing the approach, and (2) remove the `u_out==1 -> 0` override while keeping `u_out` in the grouping keys. This should reduce MAE substantially (toward the target) while remaining a pure lookup/mean-encoding solution that runs fast and writes a valid `submission.csv`.'
- What this solution (achieved 3.52177) has done: 'Your current MAE is far above the target, so we need a real improvement while keeping the same “groupby-mean lookup + hierarchical fallbacks + snap-to-known-pressure-levels” core logic. The biggest issue is that your primary keys are too specific and include noisy/continuous-derived features (`u_in_diff`, `u_in_lag1`, `u_in_cumsum`) that cause frequent join misses and push many rows into weak fallbacks/global mean. I keep all your existing features and fallbacks, but insert one additional, more “physically relevant” intermediate fallback keyed on `(R, C, time_step_r, time_step_idx, u_out, u_in_r10)` (and then a version dropping `u_out`) so more rows get a good mean estimate without changing the approach. This is a minimal change (just two extra groupby tables + two merges + fill order) and should move the score substantially downward toward the target while preserving semantics and producing the same valid `submission.csv`.'
- What this solution (achieved 3.5027) has done: 'Your score is still far above the target (lower-is-better MAE), so we need a meaningful improvement while keeping your same “groupby-mean lookup + hierarchical fallbacks + snap-to-known-pressure-levels” approach. The biggest, minimal-change win is to make your join keys less brittle by quantizing the high-variance derived features: instead of using raw per-step `u_in_diff_r10` and `u_in_cumsum_r10` everywhere (which causes many missed joins), we keep those features but add coarser-binned versions and insert them as higher-priority fallback tables before you drop the signal entirely. This should increase match rates to “good” group means and reduce reliance on weak fallbacks/global mean, moving MAE down toward the target without changing the core logic or post-processing. All I/O paths and the submission schema remain unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path



## === cell 1
BASE_DIR_CANDIDATES = [
    Path("../input/ventilator-pressure-prediction"),
    Path("/kaggle/input/ventilator-pressure-prediction"),
    Path("../kaggle/input/ventilator-pressure-prediction"),
    Path("../kaggle/data/ventilator-pressure-prediction"),
    Path("../input"),  # fallback if files are directly here
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]


def find_file(filename: str) -> Path:
    for d in BASE_DIR_CANDIDATES:
        p = d / filename
        if p.exists():
            return p
    p = Path(filename)
    if p.exists():
        return p
    raise FileNotFoundError(f"Could not find {filename} in any known input directory.")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

assert "pressure" in train.columns
assert "id" in test.columns and "id" in sub.columns
assert len(test) == len(
    sub
), "sample_submission and test must have same number of rows."

print(train.shape, test.shape, sub.shape)
print(train.head(2))




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_step_r"] = (
        np.round(df["time_step"].to_numpy(dtype=np.float64) / 0.02) * 0.02
    ).round(2)

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)

    df["time_step_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)
    df["u_in_diff"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

    df["u_in_r10"] = np.rint(df["u_in"].to_numpy(np.float64) * 10.0).astype(np.int32)
    df["u_in_cumsum_r10"] = np.rint(
        df["u_in_cumsum"].to_numpy(np.float64) * 10.0
    ).astype(np.int32)
    df["u_in_lag1_r10"] = np.rint(df["u_in_lag1"].to_numpy(np.float64) * 10.0).astype(
        np.int32
    )
    df["u_in_diff_r10"] = np.rint(df["u_in_diff"].to_numpy(np.float64) * 10.0).astype(
        np.int32
    )

    df["u_in_diff_r2"] = (np.rint(df["u_in_diff"].to_numpy(np.float64) * 2.0)).astype(
        np.int32
    )
    df["u_in_cumsum_r1"] = (
        np.rint(df["u_in_cumsum"].to_numpy(np.float64) * 1.0)
    ).astype(np.int32)

    return df


train_feat = add_features(train)
test_feat = add_features(test)

KEY_COLS = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_out",
    "u_in_r10",
    "u_in_diff_r10",
    "u_in_cumsum_r10",
    "u_in_lag1_r10",
]

mean_by_key = (
    train_feat.groupby(KEY_COLS, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred"})
)

fallback_0_cols = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_out",
    "u_in_r10",
    "u_in_diff_r10",
    "u_in_cumsum_r10",
]
mean_fallback_0 = (
    train_feat.groupby(fallback_0_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_f0"})
)

fallback_0b_cols = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_out",
    "u_in_r10",
    "u_in_diff_r10",
    "u_in_lag1_r10",
]
mean_fallback_0b = (
    train_feat.groupby(fallback_0b_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_f0b"})
)

fallback_1_cols = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_out",
    "u_in_r10",
    "u_in_diff_r10",
]
mean_fallback_1 = (
    train_feat.groupby(fallback_1_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_f1"})
)

fallback_1b_cols = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_in_r10",
    "u_in_diff_r10",
]
mean_fallback_1b = (
    train_feat.groupby(fallback_1b_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_f1b"})
)

fallback_coarse_0_cols = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_out",
    "u_in_r10",
    "u_in_diff_r2",
    "u_in_cumsum_r1",
]
mean_fallback_coarse_0 = (
    train_feat.groupby(fallback_coarse_0_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_fc0"})
)

fallback_coarse_1_cols = [
    "R",
    "C",
    "time_step_r",
    "time_step_idx",
    "u_in_r10",
    "u_in_diff_r2",
    "u_in_cumsum_r1",
]
mean_fallback_coarse_1 = (
    train_feat.groupby(fallback_coarse_1_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_fc1"})
)

fallback_uin_cols = ["R", "C", "time_step_r", "time_step_idx", "u_out", "u_in_r10"]
mean_fallback_uin = (
    train_feat.groupby(fallback_uin_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_fu"})
)

fallback_uin_nouout_cols = ["R", "C", "time_step_r", "time_step_idx", "u_in_r10"]
mean_fallback_uin_nouout = (
    train_feat.groupby(fallback_uin_nouout_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_fu2"})
)

fallback_2_cols = ["R", "C", "time_step_r", "time_step_idx", "u_out"]
mean_fallback_2 = (
    train_feat.groupby(fallback_2_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_f2"})
)

fallback_3_cols = ["R", "C", "time_step_r", "time_step_idx"]
mean_fallback_3 = (
    train_feat.groupby(fallback_3_cols, as_index=False)["pressure"]
    .mean()
    .rename(columns={"pressure": "pressure_pred_f3"})
)

global_mean = float(train_feat["pressure"].mean())

base_cols = ["id"] + KEY_COLS
base = test_feat[base_cols].copy()

pred_main = base.merge(mean_by_key, on=KEY_COLS, how="left")[["id", "pressure_pred"]]
pred_f0 = base[["id"] + fallback_0_cols].merge(
    mean_fallback_0, on=fallback_0_cols, how="left"
)[["id", "pressure_pred_f0"]]
pred_f0b = base[["id"] + fallback_0b_cols].merge(
    mean_fallback_0b, on=fallback_0b_cols, how="left"
)[["id", "pressure_pred_f0b"]]
pred_f1 = base[["id"] + fallback_1_cols].merge(
    mean_fallback_1, on=fallback_1_cols, how="left"
)[["id", "pressure_pred_f1"]]
pred_f1b = base[["id"] + fallback_1b_cols].merge(
    mean_fallback_1b, on=fallback_1b_cols, how="left"
)[["id", "pressure_pred_f1b"]]

pred_fc0 = test_feat[["id"] + fallback_coarse_0_cols].merge(
    mean_fallback_coarse_0, on=fallback_coarse_0_cols, how="left"
)[["id", "pressure_pred_fc0"]]
pred_fc1 = test_feat[["id"] + fallback_coarse_1_cols].merge(
    mean_fallback_coarse_1, on=fallback_coarse_1_cols, how="left"
)[["id", "pressure_pred_fc1"]]

pred_fu = base[["id"] + fallback_uin_cols].merge(
    mean_fallback_uin, on=fallback_uin_cols, how="left"
)[["id", "pressure_pred_fu"]]
pred_fu2 = base[["id"] + fallback_uin_nouout_cols].merge(
    mean_fallback_uin_nouout, on=fallback_uin_nouout_cols, how="left"
)[["id", "pressure_pred_fu2"]]

pred_f2 = base[["id"] + fallback_2_cols].merge(
    mean_fallback_2, on=fallback_2_cols, how="left"
)[["id", "pressure_pred_f2"]]
pred_f3 = base[["id"] + fallback_3_cols].merge(
    mean_fallback_3, on=fallback_3_cols, how="left"
)[["id", "pressure_pred_f3"]]

test_pred = (
    pred_main.merge(pred_f0, on="id", how="left")
    .merge(pred_f0b, on="id", how="left")
    .merge(pred_f1, on="id", how="left")
    .merge(pred_f1b, on="id", how="left")
    .merge(pred_fc0, on="id", how="left")
    .merge(pred_fc1, on="id", how="left")
    .merge(pred_fu, on="id", how="left")
    .merge(pred_fu2, on="id", how="left")
    .merge(pred_f2, on="id", how="left")
    .merge(pred_f3, on="id", how="left")
)

test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_f0"]
)
test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_f0b"]
)
test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_f1"]
)
test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_f1b"]
)

test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_fc0"]
)
test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_fc1"]
)

test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_fu"]
)
test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_fu2"]
)

test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_f2"]
)
test_pred["pressure_pred"] = test_pred["pressure_pred"].fillna(
    test_pred["pressure_pred_f3"]
)
test_pred["pressure_pred"] = (
    test_pred["pressure_pred"].fillna(global_mean).astype(np.float32)
)

pressure_levels = np.sort(train["pressure"].unique()).astype(np.float32)


def snap_to_levels(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(levels, x, side="left")
    idx = np.clip(idx, 0, len(levels) - 1)
    prev_idx = np.clip(idx - 1, 0, len(levels) - 1)
    next_level = levels[idx]
    prev_level = levels[prev_idx]
    choose_prev = np.abs(x - prev_level) <= np.abs(x - next_level)
    return np.where(choose_prev, prev_level, next_level)


test_pred["pressure_pred"] = snap_to_levels(
    test_pred["pressure_pred"].to_numpy(np.float32), pressure_levels
).astype(np.float32)

sub_out = sub[["id"]].merge(test_pred[["id", "pressure_pred"]], on="id", how="left")
sub_out = sub_out.rename(columns={"pressure_pred": "pressure"})
assert sub_out["pressure"].notna().all()
sub_out["pressure"] = sub_out["pressure"].astype(np.float32)

print(sub_out.head())
print(sub_out.shape)



## === cell 3
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.dtypes)
print(sub_out.head(3))
