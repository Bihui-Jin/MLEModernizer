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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.1634919059188457

# 6. Current score

1.34272

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.82641) has done: 'I remove notebook-only/IPython code and fix the TensorFlow/protobuf import crash by avoiding TensorFlow entirely (it isn’t needed to produce a valid submission and is currently preventing any run). I also remove the TPU-only strategy block and the missing external model-file loading, both of which currently hard-fail in Kaggle. To keep core feature engineering intact, I preserve your lag/diff/cumsum features and generate a simple, deterministic per-time-step baseline prediction from training data (median pressure by engineered feature bins), then write a valid `submission.csv` with `id,pressure`. This run end-to-end within the time limit and yield a nontrivial score (better than all-zeros), moving toward your target.'
- What this solution (achieved 2.14636) has done: 'Your current approach is a pure lookup/median baseline, so the biggest gap to the target likely comes from (1) not predicting anything meaningful for the expiratory phase (`u_out==1`) and (2) using bins that are a bit too coarse for `time_step` and `u_in` given the strong discretization of pressures in this competition. I keep the same core “median by grouped engineered features with hierarchical fallback” logic, but (a) build a separate mapping for `u_out==1` (since those rows exist in test even if not scored, wrong values can still hurt if Kaggle’s mask differs) and (b) tighten the binning to better match the 80-step sequence structure (bin by step index within breath rather than approximate seconds). Finally, I snap predictions to the nearest training pressure level (pressure is highly discretized), which tends to reduce MAE without changing the core method.'
- What this solution (achieved 1.76755) has done: 'Your current pipeline is a hierarchical median lookup, but it’s still too sparse due to the extra `u_in_cumsum_bin` and `u_in_diff_bin` keys, causing many fallbacks to coarse medians and a high MAE. I keep the same core “groupby keys → median → fallback → clip → snap-to-pressure-levels → write submission.csv” logic, but add one extra intermediate fallback level that drops only `u_in_diff_bin` (keeping `u_in_cumsum_bin`), which typically reduces missing matches with minimal behavioral change. I also build the grouped medians from inspiratory-only rows (`u_out==0`) and reuse them for both phases, because the competition metric scores only inspiratory steps and this usually moves MAE down without changing the modeling approach. Everything else (feature engineering, binning, snapping, submission writing) remains intact.'
- What this solution (achieved 1.63426) has done: 'Your current lookup is trained only on inspiratory rows (`u_out==0`) but still keeps `u_out` as a grouping key, which makes the `u_out==1` mappings empty and forces broad fallbacks that can also affect inspiratory predictions indirectly via merge sparsity. I keep the exact same median-lookup-with-fallback core logic, but (1) build the lookup tables without `u_out` (since we intentionally train on `u_out==0` only) and (2) add one more intermediate fallback that drops only `u_in_cumsum_bin` (keeping `u_in_diff_bin`) to reduce sparsity in a different direction. Everything else (feature engineering, binning, snapping to discrete pressure levels, and writing `submission.csv`) stays the same, and it still runs fast and produces a valid submission.'
- What this solution (achieved 1.55763) has done: 'Your current score is far worse than the target (lower is better), so we should improve the prediction accuracy but keep your core “hierarchical median lookup with fallbacks + snap to discrete pressure levels” logic unchanged. The biggest low-risk gain here is to incorporate the competition’s key hidden state (integrated flow/volume) by adding a per-breath cumulative integral feature (`u_in * dt` cumulative sum) and binning it, then using it as an additional lookup key with a minimal fallback chain. This preserves your approach (still just groupby→median→fallback) but makes the lookup better aligned with the underlying physics and typically reduces MAE a lot in this competition. I also remove the unused scaler/reshape block (it doesn’t affect predictions but costs time/memory) to keep runtime safely within limits while producing the same submission format.'
- What this solution (achieved 1.37061) has done: 'Your current lookup is still too sparse: the extra `u_in_dt_cumsum_bin` key (and its tight 0.5 bin width) dramatically increases unseen key combinations in test, forcing many fallbacks to coarse medians and keeping MAE high. To move your score down toward the target while preserving the same “groupby → median lookup → hierarchical fallbacks → snap to discrete pressure levels” core logic, I (1) widen the integral bin (less sparsity, better matches) and (2) add one extra intermediate fallback that keeps only `u_in_dt_cumsum_bin` (drops both `u_in_cumsum_bin` and `u_in_diff_bin`) before falling back to the very coarse maps. These are minimal, localized changes that typically improve coverage and reduce MAE without changing the overall approach or the submission format.'
- What this solution (achieved 1.4408) has done: 'Your current score (1.37061, lower is better) is still far above the target, so we should improve accuracy while keeping your same core approach: hierarchical median lookups on binned engineered features plus snapping to discrete pressure levels. The biggest low-risk issue is that `u_in_dt_cumsum_bin` is likely too coarse/inconsistent across breaths because it depends on floating-point `dt`, so we make it more stable by using the per-breath median `dt` (effectively constant sampling) and a slightly finer bin width to better match typical volume progression without exploding sparsity. To avoid sparsity from hard bin edges, we also add one minimal “neighbor-bin” fallback for `u_in_dt_cumsum_bin` (try ±1 bin) before dropping keys; this preserves the lookup logic but improves match coverage. All I/O and submission format stay unchanged.'
- What this solution (achieved 1.43419) has done: 'Your current MAE is far above the target (lower is better), so we should improve accuracy while keeping the same core “groupby-median lookup with hierarchical fallbacks + snapping to discrete pressure levels” approach. The biggest minimal win is to make the integral feature more physically consistent by integrating with the *actual per-row* `dt` (time_step diff) instead of a per-breath median `dt`, which reduces systematic drift across the breath and improves alignment between train/test keys. To keep sparsity under control (so we don’t hurt coverage), we slightly coarsen the `u_in_dt_cumsum_bin` width and keep the existing neighbor-bin (±1) fallback unchanged. Everything else (feature set, lookup chain structure, snapping, and submission writing) remains the same.'
- What this solution (achieved 1.4357) has done: 'Your current score is far above the target (lower is better), so we make a small accuracy-focused adjustment while keeping the same hierarchical median-lookup + fallback + snapping core logic. The main issue is sparsity from the integral-bin key, which forces frequent fallbacks; we add a very small extra “neighbor-bin” attempt for `u_in_dt_cumsum_bin` at ±2 (in addition to your existing ±1) before dropping keys, improving match coverage without changing the modeling approach. We also reuse a single helper to apply these neighbor merges in a controlled order to avoid mistakes and keep runtime within limits. Submission format and all I/O paths remain unchanged, and it still writes `submission.csv`.'
- What this solution (achieved 1.38102) has done: 'Your current score is far above the target (lower is better), so we should improve accuracy while keeping your same core approach: hierarchical median lookups on binned engineered features plus snapping to discrete pressure levels. The biggest minimal issue left is that your “neighbor-bin” recovery only perturbs `u_in_dt_cumsum_bin`, but many misses are also caused by bin-edge effects in `u_in_cumsum_bin` and `u_in_diff_bin`. I add two very small, controlled neighbor attempts: first try `u_in_cumsum_bin ± 1` (keeping all other keys), then try `u_in_diff_bin ± 1` (keeping all other keys), only filling remaining NaNs; this preserves the exact lookup/fallback semantics but improves match coverage without changing model type. All paths, feature engineering, snapping, and submission writing remain unchanged.'
- What this solution (achieved 1.35585) has done: 'Your current pipeline is still a sparse hierarchical median lookup, so the smallest safe way to move MAE down toward the target is to reduce “bin-edge miss” without changing the core approach. I keep your exact feature set and lookup/fallback structure, but (1) add a very small neighbor search for `u_in_bin` (±1) since it’s the most sensitive key and often causes misses, and (2) switch the main grouped statistic from median to mean (still a groupby aggregate lookup, but often closer under MAE when the target is discretized and noise is low). Everything else—feature engineering, dt-integral, fallback chain, snapping to known pressure levels, and submission writing—stays the same and still produces `submission.csv`.'
- What this solution (achieved 1.34272) has done: 'Your current score (1.35585, lower is better) is still far above the target, so we should reduce MAE while keeping the same core “groupby lookup with hierarchical fallbacks + snapping to discrete pressure levels” approach. The smallest high-impact fix is to align the lookup targets with the competition metric by training all lookup tables on inspiratory rows only **and** computing the aggregate only on inspiratory rows for each (the current mean is computed over `u_out==0` already, so we keep that), then, at inference, force predictions for expiratory rows (`u_out==1`) to a harmless constant (global inspiratory median) so they can’t introduce unnecessary noise if Kaggle’s masking differs or there are edge-case scoring quirks. Additionally, we add one very small, controlled neighbor search for `step` (±1) before dropping keys, which reduces “off-by-one step” mismatches without changing the model type. These are localized changes that improve coverage/robustness while preserving your exact feature set, lookup logic, snapping, and submission format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(2021)



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)



## === cell 2
assert {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}.issubset(
    train.columns
)
assert {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}.issubset(
    test.columns
)



## === cell 3
train["u_in_lag"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0)
train["u_in_diff"] = train["u_in"] - train["u_in_lag"]
train["u_in_cumsum"] = train["u_in"].groupby(train["breath_id"]).cumsum()

test["u_in_lag"] = test.groupby("breath_id")["u_in"].shift(1).fillna(0)
test["u_in_diff"] = test["u_in"] - test["u_in_lag"]
test["u_in_cumsum"] = test["u_in"].groupby(test["breath_id"]).cumsum()




## === cell 4
def add_dt_integral(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    raw_dt = out.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    raw_dt = raw_dt.clip(lower=0.0)

    dt_pos = raw_dt.where(raw_dt > 0.0)
    global_dt = (
        float(dt_pos.median(skipna=True))
        if float(dt_pos.median(skipna=True)) > 0
        else 0.03
    )
    out["dt"] = raw_dt.where(raw_dt > 0.0, global_dt).astype(np.float32)

    out["u_in_dt"] = (out["u_in"].astype(np.float32) * out["dt"]).astype(np.float32)
    out["u_in_dt_cumsum"] = (
        out["u_in_dt"].groupby(out["breath_id"]).cumsum().astype(np.float32)
    )
    return out


train = add_dt_integral(train)
test = add_dt_integral(test)




## === cell 5
def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["step"] = out.groupby("breath_id").cumcount().astype(np.int16)

    out["u_in_bin"] = np.clip(
        np.floor(out["u_in"] / 1.0).astype(np.int16), 0, 100
    )  # 101 bins

    out["u_in_cumsum_bin"] = np.clip(
        np.floor(out["u_in_cumsum"] / 10.0).astype(np.int16), 0, 400
    )

    out["u_in_diff_bin"] = np.clip(
        np.round(out["u_in_diff"]).astype(np.int16), -100, 100
    )

    out["u_in_dt_cumsum_bin"] = np.clip(
        np.floor(out["u_in_dt_cumsum"] / 0.75).astype(np.int16), 0, 1600
    )

    return out


train_b = add_bins(train)
test_b = add_bins(test)



## === cell 6
train_stat = train_b.loc[train_b["u_out"] == 0].copy()

group_keys = [
    "R",
    "C",
    "step",
    "u_in_bin",
    "u_in_cumsum_bin",
    "u_in_diff_bin",
    "u_in_dt_cumsum_bin",
]

median_map = train_stat.groupby(group_keys, observed=True)["pressure"].mean()

median_map_less0 = train_stat.groupby(
    ["R", "C", "step", "u_in_bin", "u_in_cumsum_bin", "u_in_dt_cumsum_bin"],
    observed=True,
)["pressure"].mean()

median_map_less0b = train_stat.groupby(
    ["R", "C", "step", "u_in_bin", "u_in_diff_bin", "u_in_dt_cumsum_bin"],
    observed=True,
)["pressure"].mean()

median_map_less0c = train_stat.groupby(
    ["R", "C", "step", "u_in_bin", "u_in_dt_cumsum_bin"],
    observed=True,
)["pressure"].mean()

median_map_less0d = train_stat.groupby(
    ["R", "C", "step", "u_in_dt_cumsum_bin"],
    observed=True,
)["pressure"].mean()

median_map_less1 = train_stat.groupby(["R", "C", "step", "u_in_bin"], observed=True)[
    "pressure"
].mean()

median_map_less2 = train_stat.groupby(["R", "C", "step"], observed=True)[
    "pressure"
].mean()

global_median_all = float(train_stat["pressure"].median())



## === cell 7
test_keys = test_b[group_keys].copy()

test_pred = test_keys.merge(
    median_map.rename("pred").reset_index(), on=group_keys, how="left"
)["pred"].to_numpy()


def apply_neighbor_dt_bin_merge(
    pred_arr: np.ndarray, df_b: pd.DataFrame, delta: int
) -> np.ndarray:
    mask_local = np.isnan(pred_arr)
    if not mask_local.any():
        return pred_arr
    base_cols = ["R", "C", "step", "u_in_bin", "u_in_cumsum_bin", "u_in_diff_bin"]
    tmp_df = df_b.loc[mask_local, base_cols + ["u_in_dt_cumsum_bin"]].copy()
    tmp_df["u_in_dt_cumsum_bin"] = np.clip(
        (tmp_df["u_in_dt_cumsum_bin"].astype(np.int32) + int(delta)).astype(np.int16),
        0,
        1600,
    )
    tmp = tmp_df.merge(
        median_map.rename("pred").reset_index(), on=group_keys, how="left"
    )["pred"].to_numpy()
    pred_arr[mask_local] = tmp
    return pred_arr


def apply_neighbor_bin_merge_for_col(
    pred_arr: np.ndarray, df_b: pd.DataFrame, col: str, delta: int, lo: int, hi: int
) -> np.ndarray:
    mask_local = np.isnan(pred_arr)
    if not mask_local.any():
        return pred_arr
    tmp_df = df_b.loc[mask_local, group_keys].copy()
    tmp_df[col] = np.clip(
        (tmp_df[col].astype(np.int32) + int(delta)).astype(np.int16), lo, hi
    )
    tmp = tmp_df.merge(
        median_map.rename("pred").reset_index(), on=group_keys, how="left"
    )["pred"].to_numpy()
    pred_arr[mask_local] = tmp
    return pred_arr


test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="step", delta=+1, lo=0, hi=200
)
test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="step", delta=-1, lo=0, hi=200
)

test_pred = apply_neighbor_dt_bin_merge(test_pred, test_b, delta=+1)
test_pred = apply_neighbor_dt_bin_merge(test_pred, test_b, delta=-1)
test_pred = apply_neighbor_dt_bin_merge(test_pred, test_b, delta=+2)
test_pred = apply_neighbor_dt_bin_merge(test_pred, test_b, delta=-2)

test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="u_in_cumsum_bin", delta=+1, lo=0, hi=400
)
test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="u_in_cumsum_bin", delta=-1, lo=0, hi=400
)

test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="u_in_diff_bin", delta=+1, lo=-100, hi=100
)
test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="u_in_diff_bin", delta=-1, lo=-100, hi=100
)

test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="u_in_bin", delta=+1, lo=0, hi=100
)
test_pred = apply_neighbor_bin_merge_for_col(
    test_pred, test_b, col="u_in_bin", delta=-1, lo=0, hi=100
)

mask = np.isnan(test_pred)
if mask.any():
    keys0 = ["R", "C", "step", "u_in_bin", "u_in_cumsum_bin", "u_in_dt_cumsum_bin"]
    tmp = (
        test_b.loc[mask, keys0]
        .merge(median_map_less0.rename("pred").reset_index(), on=keys0, how="left")[
            "pred"
        ]
        .to_numpy()
    )
    test_pred[mask] = tmp

mask = np.isnan(test_pred)
if mask.any():
    keys0b = ["R", "C", "step", "u_in_bin", "u_in_diff_bin", "u_in_dt_cumsum_bin"]
    tmp = (
        test_b.loc[mask, keys0b]
        .merge(median_map_less0b.rename("pred").reset_index(), on=keys0b, how="left")[
            "pred"
        ]
        .to_numpy()
    )
    test_pred[mask] = tmp

mask = np.isnan(test_pred)
if mask.any():
    keys0c = ["R", "C", "step", "u_in_bin", "u_in_dt_cumsum_bin"]
    tmp = (
        test_b.loc[mask, keys0c]
        .merge(median_map_less0c.rename("pred").reset_index(), on=keys0c, how="left")[
            "pred"
        ]
        .to_numpy()
    )
    test_pred[mask] = tmp

mask = np.isnan(test_pred)
if mask.any():
    keys0d = ["R", "C", "step", "u_in_dt_cumsum_bin"]
    tmp = (
        test_b.loc[mask, keys0d]
        .merge(median_map_less0d.rename("pred").reset_index(), on=keys0d, how="left")[
            "pred"
        ]
        .to_numpy()
    )
    test_pred[mask] = tmp

mask = np.isnan(test_pred)
if mask.any():
    keys1 = ["R", "C", "step", "u_in_bin"]
    tmp = (
        test_b.loc[mask, keys1]
        .merge(median_map_less1.rename("pred").reset_index(), on=keys1, how="left")[
            "pred"
        ]
        .to_numpy()
    )
    test_pred[mask] = tmp

mask = np.isnan(test_pred)
if mask.any():
    keys2 = ["R", "C", "step"]
    tmp = (
        test_b.loc[mask, keys2]
        .merge(median_map_less2.rename("pred").reset_index(), on=keys2, how="left")[
            "pred"
        ]
        .to_numpy()
    )
    test_pred[mask] = tmp

mask = np.isnan(test_pred)
if mask.any():
    test_pred[mask] = global_median_all

test_pred = test_pred.reshape(-1)

exp_mask = test_b["u_out"].to_numpy().reshape(-1) == 1
test_pred[exp_mask] = global_median_all



## === cell 8
p_min = float(train["pressure"].min())
p_max = float(train["pressure"].max())
test_pred = np.clip(test_pred, p_min, p_max)



## === cell 9
pressure_levels = np.sort(train["pressure"].unique()).astype(np.float64)

idx = np.searchsorted(pressure_levels, test_pred, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx0 = np.clip(idx - 1, 0, len(pressure_levels) - 1)
left = pressure_levels[idx0]
right = pressure_levels[idx]
choose_right = np.abs(right - test_pred) < np.abs(test_pred - left)
test_pred = np.where(choose_right, right, left)



## === cell 10
submission_file = pd.read_csv(SAMPLE_SUB_PATH)
assert len(submission_file) == len(test_pred), (len(submission_file), len(test_pred))
assert "id" in submission_file.columns and "pressure" in submission_file.columns

submission_file["pressure"] = test_pred.astype(np.float32)
submission_file.to_csv("submission.csv", index=False)

print(submission_file.head())
print("Wrote submission.csv with shape:", submission_file.shape)
print(
    "pressure stats:",
    float(submission_file["pressure"].min()),
    float(submission_file["pressure"].mean()),
    float(submission_file["pressure"].max()),
)
