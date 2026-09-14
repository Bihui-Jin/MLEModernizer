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

0.1552335480953755

# 6. Current score

4.24573

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I remove the dependency on the missing `../input/gb-submitted/0.154 blend.csv` by generating a valid baseline submission directly from the provided `sample_submission.csv`, then applying your existing “snap to nearest valid pressure” post-processing (which is score-improving vs arbitrary values and preserves your intended semantics). I also shift the notebook to Kaggle’s standard `../input/ventilator-pressure-prediction/` paths so it runs in your described environment. Finally, I keep your existing blending utilities but make them safe (avoid filename parsing crashes, division by zero, and extremely slow `file_count**3` loops) so the script always finishes and writes `submission.csv`.'
- What this solution (achieved 6.5212) has done: 'Your current score (17.65 MAE) indicates the submission is essentially predicting a constant (the nearest pressure to 0), which is far from the target 0.155. The smallest legitimate improvement without changing your “snap-to-nearest-valid-pressure” core logic is to generate a better continuous baseline prediction before snapping. Below, I keep your post-processing intact and add a simple, fast, leakage-safe baseline: for each (R, C, time_step index within breath, u_out), predict the mean pressure from train; fall back to less-specific group means when unseen, then snap to nearest valid pressure. This typically moves a constant baseline dramatically closer to a reasonable MAE while staying lightweight and within the constraints.'
- What this solution (achieved 7.46354) has done: 'Your current MAE (6.52, lower is better) is still far from the target (0.155), so we need a meaningful but still minimal improvement without changing the overall “group-mean baseline + snap-to-nearest-valid-pressure” core approach. The biggest issue is that your baseline ignores `u_in` (the primary control driving pressure), which makes the group means too coarse and yields large error. I keep the same pipeline but add a more specific first-level group that includes a binned `u_in` (fast, no new models), then fall back to your existing group levels and finally global mean; snapping stays identical. This should substantially reduce MAE while remaining deterministic, lightweight, and within Kaggle constraints.'
- What this solution (achieved 8.04514) has done: 'Your current score is much worse than the target (lower is better), so we need a real accuracy gain while keeping your same “group-mean baseline → snap to nearest valid pressure” core logic. The smallest high-impact fix is to incorporate `u_in` more directly without introducing a new model: replace the coarse `u_in` bin mean with a fast KNN-style lookup per `(R,C,t_idx,u_out)` group using the nearest `u_in` values from train (vectorized via `np.searchsorted`), then keep your exact fallback chain and snapping. This stays deterministic, leakage-safe, and should reduce MAE substantially versus binning, moving you closer to the target. The script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 8.63599) has done: 'Your current MAE (8.045, lower is better) is far above the target (0.155), so we need a meaningful accuracy gain while keeping the same “lookup/group-mean baseline → snap to nearest valid pressure” approach. The main issue is that the current KNN-style lookup uses `t_idx` and `u_out` but ignores breath history; pressure depends strongly on accumulated `u_in` and flow history, so same `(R,C,t_idx,u_out,u_in)` can still map to different pressures. I keep your exact snapping and fallback chain, but change the first-level lookup key to include two simple, deterministic history features computed per breath (`u_in_cum` and `u_in_lag1`) and then do the same nearest-`u_in` interpolation inside those more-specific groups. This is still the same core logic (grouped KNN lookup with fallbacks + snapping), but typically reduces MAE substantially for this competition while staying fast enough and producing a valid `submission.csv`.'
- What this solution (achieved 8.20556) has done: 'Your current MAE is far above the target (lower is better), so we need a real accuracy gain while keeping your same “grouped lookup (nearest u_in inside a key) + fallback means + snap-to-valid-pressure” approach. The biggest issue is that the current key uses coarse bins for history, which makes many test rows miss the exact group and fall back to very rough means, inflating error. I keep the identical core pipeline but (1) replace the coarse history bins with more discriminative, still-deterministic bins aligned to the data scale, and (2) add a single additional lightweight history feature (`u_in_cum_delta`) to stabilize mapping without changing the modeling approach. Everything else (fallback chain and snapping) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 8.25041) has done: 'Your score is far worse than the target (lower is better), so we need a real accuracy gain but with minimal, safe changes that preserve your existing “grouped nearest-`u_in` lookup → fallback means → snap-to-valid-pressure” pipeline. The main problem is that your history bins make the primary lookup key far too sparse, so most test rows miss and fall back to coarse means, inflating MAE. I keep the exact same lookup/interpolation and fallback chain, but (1) add a deterministic backoff KNN lookup with a less-specific key (drop the most fragile history bin(s)) before falling back to means, and (2) vectorize snapping to nearest valid pressures (same semantics, faster/less error-prone). This should move MAE substantially toward the target while staying within the same core logic and runtime constraints.'
- What this solution (achieved 6.08171) has done: 'Your current MAE (8.25041, lower is better) is far above the target, so the biggest low-risk gain is to stop “guessing” pressure from coarse history bins and instead use an exact lookup whenever the exact state/control tuple exists in train. I keep your overall pipeline (lookup → fallbacks → snap-to-valid-pressure) but replace the sparse binned-key KNN with a much denser, deterministic per-(R,C,t_idx,u_out,u_in) mean mapping that matches many rows exactly and only falls back when unseen. This preserves the same evaluation semantics and post-processing (snapping), avoids new models/training, and stays fast enough by building the mapping with a single groupby and using vectorized join. The rest of your fallback chain remains intact so it still produces a valid `submission.csv` end-to-end.'
- What this solution (achieved 4.22037) has done: 'Your current MAE (6.08, lower is better) is still far above the target (0.155), so we need a meaningful accuracy gain while keeping your same “exact lookup → fallback means → snap-to-valid-pressure” pipeline. The smallest high-impact improvement is to make the “exact lookup” key a bit less brittle by adding a second, still-deterministic lookup that uses a *rounded* `u_in` (to 0.1) so near-identical control values can match between train/test more often, while preserving the same groupby-mean semantics. We keep your existing exact `(R,C,t_idx,u_out,u_in)` map as the first priority, then back off to the rounded-u_in map, then your existing g1/g2/g3/global fallbacks, and finally the same snapping. This typically reduces error substantially because `u_in` values are continuous and exact float matches are often missed, but rounding keeps the same core logic and remains very fast.'
- What this solution (achieved 4.18981) has done: 'Your current score (4.22037 MAE; lower is better) is still far above the target (0.1552), so we should improve accuracy while keeping your existing “exact lookup → rounded-u_in lookup → mean fallbacks → snap to valid pressures” pipeline intact. The biggest minimal gain is to make the rounded-u_in lookup less brittle by trying a few additional deterministic roundings (coarser and finer) before falling back to group means, which increases train/test match coverage without changing the modeling approach. I also ensure the `id`→prediction alignment is correct by mapping directly from the already-sorted `df_test` `id` sequence (no semantic change, but removes any potential index mismatch risk). Everything else (feature creation, fallbacks, snapping, and writing `submission.csv`) stays the same.'
- What this solution (achieved 4.24763) has done: 'Your current MAE (4.18981, lower is better) is still far above the target (0.1552), so we need a real accuracy gain but with minimal changes that preserve your existing “lookup maps → mean fallbacks → snap-to-valid-pressure” pipeline. The biggest issue is that float `u_in` rarely matches exactly between train/test, and your current rounded maps (0.05/0.1/0.2) still miss too often; we add one *coarser* rounding (to integer `u_in`) and try it before falling back to group means, increasing match coverage without changing the modeling approach. We also ensure `id` alignment is strictly correct by building predictions in the same row order as `df_test` (already the case) and mapping to `df_sub` by `id` as you do now. No model/loop/training changes are introduced; this is just an extra deterministic backoff key that should move MAE meaningfully toward the target band.'
- What this solution (achieved 4.24573) has done: 'Your MAE is still far above the target (lower is better), so the smallest high-impact improvement without changing your overall “lookup maps → fallback means → snap-to-valid-pressure” pipeline is to make the lookup key less brittle to float mismatches in `u_in` while keeping the same groupby-mean semantics. I add two additional deterministic backoff maps using `u_in` rounded to 0.5 and 2.0 (in between your existing 1.0 and coarse mean fallbacks), and I also add a final “nearest-neighbor within (R,C,t_idx,u_out)” interpolation over sorted `u_in` when all rounding maps miss (still a lookup-based method, no training loop/model). This typically improves coverage a lot because test `u_in` values rarely match train exactly, moving MAE toward the target while preserving your snapping and fallback structure. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd

DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure", "id"]
usecols_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]

df_train = pd.read_csv(train_path, usecols=usecols_train)
df_test = pd.read_csv(test_path, usecols=usecols_test)
df_sub = pd.read_csv(sample_sub_path)

if not {"id", "pressure"}.issubset(df_sub.columns):
    raise ValueError(
        "sample_submission.csv does not have required columns: id, pressure"
    )

df_train = df_train.sort_values(["breath_id", "time_step"], kind="mergesort")
df_test = df_test.sort_values(["breath_id", "time_step"], kind="mergesort")
df_train["t_idx"] = df_train.groupby("breath_id").cumcount().astype(np.int16)
df_test["t_idx"] = df_test.groupby("breath_id").cumcount().astype(np.int16)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)


def snap_to_nearest_valid(pred_arr: np.ndarray) -> np.ndarray:
    pred_arr = pred_arr.astype(np.float32, copy=False)
    idx = np.searchsorted(sorted_pressures, pred_arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    p1 = sorted_pressures[idx]
    p0 = sorted_pressures[idx0]

    choose0 = np.abs(pred_arr - p0) < np.abs(p1 - pred_arr)
    return np.where(choose0, p0, p1).astype(np.float32, copy=False)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("breath_id", sort=False)
    df["u_in_cum"] = g["u_in"].cumsum().astype(np.float32)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_cum_lag1"] = g["u_in_cum"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_cum_delta"] = (df["u_in_cum"] - df["u_in_cum_lag1"]).astype(np.float32)
    return df


df_train = add_history_features(df_train)
df_test = add_history_features(df_test)

g1 = (
    df_train.groupby(["R", "C", "t_idx", "u_out"], sort=False)["pressure"]
    .mean()
    .rename("p1")
    .reset_index()
)
g2 = (
    df_train.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .mean()
    .rename("p2")
    .reset_index()
)
g3 = (
    df_train.groupby(["R", "C"], sort=False)["pressure"]
    .mean()
    .rename("p3")
    .reset_index()
)
global_mean = float(df_train["pressure"].mean())


def add_history_bins(df: pd.DataFrame) -> pd.DataFrame:
    df["u_in_cum_bin"] = np.floor(df["u_in_cum"] / 10.0).astype(np.int16)
    df["u_in_lag1_bin"] = np.floor(df["u_in_lag1"] / 2.0).astype(np.int16)
    df["u_in_cum_delta_bin"] = np.floor(df["u_in_cum_delta"] / 2.0).astype(np.int16)
    return df


df_train = add_history_bins(df_train)
df_test = add_history_bins(df_test)

KEY_COLS_EXACT = ["R", "C", "t_idx", "u_out", "u_in"]
exact_map = (
    df_train.groupby(KEY_COLS_EXACT, sort=False)["pressure"]
    .mean()
    .rename("p_exact")
    .reset_index()
)


def add_uin_round(df: pd.DataFrame, name: str, scale: float) -> None:
    df[name] = (np.round(df["u_in"].to_numpy(dtype=np.float32) * scale) / scale).astype(
        np.float32
    )


add_uin_round(df_train, "u_in_r10", 10.0)  # 0.1
add_uin_round(df_test, "u_in_r10", 10.0)

add_uin_round(df_train, "u_in_r20", 20.0)  # 0.05
add_uin_round(df_test, "u_in_r20", 20.0)

add_uin_round(df_train, "u_in_r5", 5.0)  # 0.2
add_uin_round(df_test, "u_in_r5", 5.0)

add_uin_round(df_train, "u_in_r1", 1.0)  # 1.0
add_uin_round(df_test, "u_in_r1", 1.0)

add_uin_round(df_train, "u_in_r2", 2.0)  # 0.5
add_uin_round(df_test, "u_in_r2", 2.0)

add_uin_round(df_train, "u_in_r05", 0.5)  # 2.0 steps (round to nearest 2)
add_uin_round(df_test, "u_in_r05", 0.5)

KEY_COLS_R20 = ["R", "C", "t_idx", "u_out", "u_in_r20"]
KEY_COLS_R10 = ["R", "C", "t_idx", "u_out", "u_in_r10"]
KEY_COLS_R5 = ["R", "C", "t_idx", "u_out", "u_in_r5"]
KEY_COLS_R2 = ["R", "C", "t_idx", "u_out", "u_in_r2"]
KEY_COLS_R1 = ["R", "C", "t_idx", "u_out", "u_in_r1"]
KEY_COLS_R05 = ["R", "C", "t_idx", "u_out", "u_in_r05"]

r20_map = (
    df_train.groupby(KEY_COLS_R20, sort=False)["pressure"]
    .mean()
    .rename("p_r20")
    .reset_index()
)
r10_map = (
    df_train.groupby(KEY_COLS_R10, sort=False)["pressure"]
    .mean()
    .rename("p_r10")
    .reset_index()
)
r5_map = (
    df_train.groupby(KEY_COLS_R5, sort=False)["pressure"]
    .mean()
    .rename("p_r5")
    .reset_index()
)
r2_map = (
    df_train.groupby(KEY_COLS_R2, sort=False)["pressure"]
    .mean()
    .rename("p_r2")
    .reset_index()
)
r1_map = (
    df_train.groupby(KEY_COLS_R1, sort=False)["pressure"]
    .mean()
    .rename("p_r1")
    .reset_index()
)
r05_map = (
    df_train.groupby(KEY_COLS_R05, sort=False)["pressure"]
    .mean()
    .rename("p_r05")
    .reset_index()
)

test_exact = df_test[KEY_COLS_EXACT].merge(exact_map, on=KEY_COLS_EXACT, how="left")
pred_cont = test_exact["p_exact"].to_numpy(dtype=np.float32, copy=True)

mask = np.isnan(pred_cont)
if mask.any():
    test_r20 = df_test.loc[mask, KEY_COLS_R20].merge(
        r20_map, on=KEY_COLS_R20, how="left"
    )
    pred_cont[mask] = test_r20["p_r20"].to_numpy(dtype=np.float32, copy=False)

mask = np.isnan(pred_cont)
if mask.any():
    test_r10 = df_test.loc[mask, KEY_COLS_R10].merge(
        r10_map, on=KEY_COLS_R10, how="left"
    )
    pred_cont[mask] = test_r10["p_r10"].to_numpy(dtype=np.float32, copy=False)

mask = np.isnan(pred_cont)
if mask.any():
    test_r5 = df_test.loc[mask, KEY_COLS_R5].merge(r5_map, on=KEY_COLS_R5, how="left")
    pred_cont[mask] = test_r5["p_r5"].to_numpy(dtype=np.float32, copy=False)

mask = np.isnan(pred_cont)
if mask.any():
    test_r2 = df_test.loc[mask, KEY_COLS_R2].merge(r2_map, on=KEY_COLS_R2, how="left")
    pred_cont[mask] = test_r2["p_r2"].to_numpy(dtype=np.float32, copy=False)

mask = np.isnan(pred_cont)
if mask.any():
    test_r1 = df_test.loc[mask, KEY_COLS_R1].merge(r1_map, on=KEY_COLS_R1, how="left")
    pred_cont[mask] = test_r1["p_r1"].to_numpy(dtype=np.float32, copy=False)

mask = np.isnan(pred_cont)
if mask.any():
    test_r05 = df_test.loc[mask, KEY_COLS_R05].merge(
        r05_map, on=KEY_COLS_R05, how="left"
    )
    pred_cont[mask] = test_r05["p_r05"].to_numpy(dtype=np.float32, copy=False)

KEY_G = ["R", "C", "t_idx", "u_out"]
mask = np.isnan(pred_cont)
if mask.any():
    train_g = df_train[KEY_G + ["u_in", "pressure"]].copy()
    train_g["u_in"] = train_g["u_in"].astype(np.float32)
    train_g["pressure"] = train_g["pressure"].astype(np.float32)
    train_g.sort_values(KEY_G + ["u_in"], inplace=True, kind="mergesort")

    ug = (
        train_g.groupby(KEY_G + ["u_in"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .sort_values(KEY_G + ["u_in"], kind="mergesort")
    )

    keys = ug[KEY_G].to_numpy()
    change = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    ends = np.concatenate((starts[1:], [len(ug)]))

    group_to_range = {}
    key_tuples = [tuple(x) for x in keys[starts]]
    for kt, s, e in zip(key_tuples, starts, ends):
        group_to_range[kt] = (s, e)

    u_in_vals = ug["u_in"].to_numpy(dtype=np.float32)
    p_vals = ug["pressure"].to_numpy(dtype=np.float32)

    miss_idx = np.flatnonzero(mask)
    for i in miss_idx:
        kt = (
            int(df_test.iloc[i]["R"]),
            int(df_test.iloc[i]["C"]),
            int(df_test.iloc[i]["t_idx"]),
            int(df_test.iloc[i]["u_out"]),
        )
        rng = group_to_range.get(kt)
        if rng is None:
            continue
        s, e = rng
        x = float(df_test.iloc[i]["u_in"])
        xs = u_in_vals[s:e]
        ys = p_vals[s:e]
        if xs.size == 1:
            pred_cont[i] = ys[0]
            continue
        j = int(np.searchsorted(xs, x, side="left"))
        if j <= 0:
            pred_cont[i] = ys[0]
        elif j >= xs.size:
            pred_cont[i] = ys[-1]
        else:
            x0, x1 = float(xs[j - 1]), float(xs[j])
            y0, y1 = float(ys[j - 1]), float(ys[j])
            if x1 == x0:
                pred_cont[i] = y0
            else:
                pred_cont[i] = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

pred = df_test.merge(g1, on=["R", "C", "t_idx", "u_out"], how="left")
pred = pred.merge(g2, on=["R", "C", "t_idx"], how="left")
pred = pred.merge(g3, on=["R", "C"], how="left")

mask = np.isnan(pred_cont)
if mask.any():
    pred_cont[mask] = pred.loc[mask, "p1"].to_numpy()
mask = np.isnan(pred_cont)
if mask.any():
    pred_cont[mask] = pred.loc[mask, "p2"].to_numpy()
mask = np.isnan(pred_cont)
if mask.any():
    pred_cont[mask] = pred.loc[mask, "p3"].to_numpy()
mask = np.isnan(pred_cont)
if mask.any():
    pred_cont[mask] = global_mean

pred_snapped = snap_to_nearest_valid(pred_cont)

df_sub = df_sub.sort_values("id", kind="mergesort").reset_index(drop=True)
df_test_ids = df_test["id"].to_numpy()
id2p = pd.Series(pred_snapped, index=df_test_ids)
df_sub["pressure"] = df_sub["id"].map(id2p).astype(float)

df_sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
print("pressure stats:", df_sub["pressure"].describe())




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Fix: make robust to arbitrary filenames (avoid int(...) parsing crash).
    Keep core behavior: read pressures and (if 2 files) blend with fixed 0.6/0.4.
    """
    arrs = []
    for p in input_list:
        arrs.append(pd.read_csv(p)["pressure"].to_numpy().ravel())

    if len(arrs) == 0:
        raise ValueError("wc() received an empty file list")
    if len(arrs) == 1:
        return arrs[0]

    weight1 = 0.6
    weight2 = 0.4
    return arrs[0] * weight1 + arrs[1] * weight2


def g(dp):
    """
    Fix: ensure it terminates quickly and doesn't divide by zero.
    Original used loop_time = file_count ** 3 which can be extremely slow.
    Since this is utility code (not required for submission generation), we cap the loop count.
    """
    l = [i for i in glob.iglob(f"{dp}/*") if i.lower().endswith(".csv")]
    file_count = len(l)
    if file_count == 0:
        raise FileNotFoundError(f"No CSV files found in: {dp}")

    loop_time = min(file_count**3, 2000)

    splits = max(file_count // 2, 1)
    l.sort()
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        chunk = l[start:end]
        if len(chunk) > 0:
            flist.append(chunk)

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    output = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    output["pressure"] = 0.0

    for it in range(loop_time):
        weight = []
        set_seed(it)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) or 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output["pressure"] += flist[j] * weight[j]

    output["pressure"] /= float(loop_time)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.6 + b["pressure"] * 0.4
    a.to_csv("blend.csv", index=False)
    return a
