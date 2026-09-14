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

0.1423689560412636

# 6. Current score

3.15055

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the failure by making the blending code robust to missing/empty input directories and to files that don’t contain the expected `pressure` column or correct length. The current error happens because `pred_list` ends up containing arrays of the wrong shape (likely due to no valid prediction files found), so the median becomes length 1 instead of 603600. I add validation and a safe fallback that produces a valid submission (using sample_submission’s zeros) while still preserving your core blending logic when valid prediction files are available. I also ensure the output filename ends with `.csv` and is written to `/kaggle/working/` so Kaggle picks it up.'
- What this solution (achieved 9.006) has done: 'Your current script is a “blender” that mostly falls back to all-zeros because `../input/gb-data-blending-recover` is not available here, which explains the very poor MAE (17.65). To move the score toward the target while preserving your core logic, I keep the same blending pipeline but add a minimal, deterministic fallback “base prediction” built from the training data: per-(R,C,time_step,u_out) median pressure, with a global time_step fallback if a key is unseen. This keeps the same submission semantics (predict pressure per row), still snaps predictions to the valid pressure grid via `find_nearest`, and substantially reduce MAE compared with zeros when no external prediction files exist. The code still preferentially blend external files if they exist and are valid, only using the fallback when needed, and it always write `/kaggle/working/submission.csv`.'
- What this solution (achieved 4.00094) has done: 'To move your score closer to the 0.142 target (lower is better) while preserving the same “blender + fallback” core logic, I make the fallback much more informative by incorporating `u_in` (the main control signal) and by using medians at several granularities with safe backoff (exact match → partial match → time_step-only → global). I also ensure the fallback map build is memory-safe (cast keys to smaller dtypes, select only needed columns) so it runs within the time limit. Finally, I keep your existing `find_nearest` snapping (important for this competition) and keep external blending behavior unchanged when those files exist and are valid.'
- What this solution (achieved 8.06642) has done: 'Your current score (4.00094 MAE) is far above the target (0.14237), so we need a real accuracy lift while keeping your “blender + fallback mapping + snap-to-grid” logic intact. The biggest issue is that the fallback mapping ignores the strongest sequence signal: within-breath history (previous `u_in` and previous pressure), so it collapses to a noisy static lookup that can’t match the dynamics well. I keep your same fallback concept (groupby medians + safe backoff) but minimally extend the keys with a within-breath lag feature (`u_in_lag1`, rounded) computed from `breath_id` order, and add a corresponding backoff level; this preserves your overall approach and evaluation semantics while typically improving MAE substantially. I also ensure the `id` alignment is correct by building predictions directly on the loaded `test.csv` order and then writing to `sample_submission.csv` (same length/order) as you already do.'
- What this solution (achieved 3.7039) has done: 'Your current score is far worse than the target (lower is better), so we should improve accuracy without changing the overall “blending + fallback lookup + snap-to-pressure-grid” approach. The biggest correctness/score issue in the fallback is misalignment: `fallback_predict()` sorts the test rows, merges, then tries to reindex back using mismatched indices, which can scramble predictions and inflate MAE. I fix this by carrying the original row index through the sort/merge and restoring predictions to the original order deterministically. This is a minimal change that preserves your mapping logic, blending behavior, and `find_nearest` snapping, but should materially reduce error.'
- What this solution (achieved 3.15558) has done: 'Your current MAE (3.7039, lower is better) is still far from the target (0.1424), so we should improve the fallback predictor (used when no external blend files exist) while keeping your overall “blender + fallback lookup + snap-to-grid” logic unchanged. The smallest high-impact fix is to align the fallback maps with the evaluation: only inspiratory phase (u_out == 0) is scored, so we should build fallback medians primarily from inspiratory rows, and use a dedicated expiratory fallback for u_out == 1. Additionally, I add a minimal extra sequential signal by including a rounded cumulative integral of u_in within-breath (a common physics proxy for delivered volume) as an additional key with backoff, without changing your architecture/training (still pure groupby-median lookup). This typically reduces MAE materially versus using only instantaneous u_in and u_in_lag1, while preserving your existing blending behavior and pressure-grid snapping.'
- What this solution (achieved 3.15258) has done: 'I keep your blender intact and focus on making the fallback lookup more informative in a minimal way, because your current MAE (3.15558, lower is better) is still far above the target (0.14237). The smallest high-impact improvement that preserves your “groupby-median maps + backoff + snap-to-grid” core logic is to add one more within-breath sequential key: a rounded lag-1 pressure proxy learned from training (`pressure_lag1_r`) and applied at inference via mapped medians (not a model change, still pure lookup). I build an additional inspiratory-phase map keyed by (R,C,time_step,u_out,u_in,u_in_lag1,u_in_cum,pressure_lag1) with safe backoff to your existing maps, and I compute the test-time lag proxy using the already-predicted previous step (teacher-forcing style, still deterministic). This should materially reduce error while keeping execution under time and maintaining identical submission semantics and `find_nearest` snapping.'
- What this solution (achieved 3.15099) has done: 'Your current score is far above the target (lower is better), so we need a small but meaningful accuracy lift without changing your overall “fallback lookup + backoff + snap-to-grid” approach. The highest-impact minimal fix is to make the `pressure_lag1` feature consistent at inference: right now it uses a “one-pass” lag from a coarse first prediction, but it never truly follows the within-breath recursion that `pressure_lag1` implies. I keep all your existing maps and backoff order, but change inference to compute predictions step-by-step within each breath so `pressure_lag1_r` is derived from the previously predicted step (same semantics you intended, just applied correctly). This typically reduces MAE materially while staying within time and still writing `/kaggle/working/submission.csv`.'
- What this solution (achieved 3.15099) has done: 'I fix the immediate runtime error by ensuring `pressure_lag1_r` exists in the test dataframe before any merge that uses it (the current KeyError). Then I keep your intended recursive within-breath update for `pressure_lag1_r`, but implement it without creating a one-row DataFrame+merge inside the inner loop (same lookup semantics, far faster and stable within Kaggle’s time limit). Finally, I keep your existing fallback/blending behavior and pressure-grid snapping unchanged, and ensure a valid `/kaggle/working/submission.csv` is always written.'
- What this solution (achieved 3.17188) has done: 'I keep your current blender + fallback-lookup + snap-to-pressure-grid logic intact and only make a minimal change to improve the fallback accuracy (since your environment likely has no external blend files). Specifically, I build and use the most-informative `p_full_lag_cum_plag` map directly in the base backoff chain (currently it’s only applied conditionally via the recursive dict, so many rows never benefit). Then, inside the per-breath recursion, I also add a safe backoff to the same-map without `pressure_lag1_r` when the exact recursive key is missing, which improves coverage without changing the overall approach. These changes should move MAE down from ~3.15 toward your target while preserving semantics and producing the same `/kaggle/working/submission.csv`.'
- What this solution (achieved 3.16871) has done: 'Your current MAE (3.17188, lower is better) is still far above the target (0.14237), and the biggest remaining lever without changing your core “lookup + backoff + snap-to-grid” logic is to make the fallback maps more consistent with how you use them at inference. Right now, the most-informative map (`p_full_lag_cum_plag`) is built only with a *true* `pressure_lag1` from train, but at inference you use a *predicted* `pressure_lag1`, so exact matches are very sparse and you mostly back off to weaker maps. I keep the same architecture/approach, but rebuild the `p_full_lag_cum_plag` map using a discretized “pressure_lag1 bucket” based on the nearest valid pressure grid (same snapping semantics you already use), and I apply the same bucketing to the recursive inference key—this greatly increases hit-rate while preserving the same lookup + recursion idea. Additionally, I add one minimal “lagged u_out” feature (previous `u_out`) into the most-informative map only, which helps distinguish phase transitions without changing your general pipeline.'
- What this solution (achieved 3.15054) has done: 'I keep your existing “fallback lookup + backoff + recursive lag + snap-to-grid” core logic intact and focus on two minimal, high-impact correctness/coverage issues that currently waste your strongest map. First, your `map_plag_nolag_dict` is accidentally built from the *full* key map (includes `pressure_lag1_g`) but then queried with a *shorter* key, so it almost never hits; I rebuild that dict from a properly grouped map without `pressure_lag1_g`. Second, your initial merge into `p_full_lag_cum_plag` is done with a constant `pressure_lag1_g`, which guarantees near-zero coverage for that column; I remove this merge dependency and rely on your existing per-breath recursion (which already computes `pressure_lag1_g` correctly) to apply the strongest map. These changes should reduce MAE from ~3.17 toward the target while staying deterministic and producing the same `/kaggle/working/submission.csv`.'
- What this solution (achieved 3.15055) has done: 'Your current MAE (3.15054, lower is better) is still far above the target (0.14237), so we need a small, safe accuracy lift while preserving your existing “fallback lookup + backoff + recursive lag + snap-to-grid” logic. The biggest low-risk improvement is to make your most-informative recursive maps less brittle by discretizing/quantizing the float keys (time_step/u_in/u_in_lag1/u_in_cum) into integer “bins” consistently in both train-map building and test inference, which greatly increases dictionary hit-rate without changing the underlying approach. I also add one extra minimal backoff level inside the recursion that uses the same key but drops `u_out_lag1` (keeps your semantics, improves coverage near phase boundaries). Everything else (blending behavior, pressure snapping, output path/format) is kept intact, and it still always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 3.15055) has done: 'Your current MAE (3.15055, lower is better) is still far above the target (0.14237), and since the external blend directory is likely missing, the score is dominated by the fallback lookup/recursion. To move toward the target without changing your overall “groupby-median maps + backoff + recursive lag + snap-to-grid” approach, I make one minimal but high-impact change: use `pressure_lag1_g` only when it is informative (during inspiration) and avoid carrying it through expiration, because expiration isn’t scored and it corrupts the next-step inspiratory lag signal. Concretely, I reset `prev_pred` to a safe inspiratory baseline at the start of each breath and whenever `u_out` is 1, so the recursive `pressure_lag1_g` stays consistent with how the maps were built (inspiratory-focused) and yields more hits in the strongest dicts. Everything else (maps, backoff chain, snapping, submission writing) remains unchanged.'

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


def bucket_pressure_to_grid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, x, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    prev_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx]
    lower = sorted_pressures[prev_idx]
    choose_lower = (idx > 0) & ((x - lower) <= (upper - x))
    out = upper.copy()
    out[choose_lower] = lower[choose_lower]
    return out.astype(np.float32)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    if input_list is None or len(input_list) == 0:
        return None

    l = []
    preds = []
    for path in input_list:
        try:
            try:
                public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
            except Exception:
                public_lb_score = 1
            df = pd.read_csv(path)
            if "pressure" not in df.columns:
                continue
            arr = df["pressure"].to_numpy().ravel()
            preds.append(arr)
            l.append(public_lb_score)
        except Exception:
            continue

    if len(preds) == 0:
        return None

    output = 0
    l_sum = sum(l) if sum(l) != 0 else len(l)
    if len(preds) == 1:
        output = preds[0]
    else:
        if len(l) >= 2:
            weight1 = (l[1] / l_sum) + 0.15
            weight2 = 1 - weight1
        else:
            weight1, weight2 = 0.5, 0.5
        output += preds[0] * weight1 + preds[1] * weight2

    return output


def _q_round(x: pd.Series, scale: float) -> pd.Series:
    return np.rint(x.astype(np.float32) * np.float32(scale)).astype(np.int32)


def build_fallback_maps(train_df: pd.DataFrame):
    cols = ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
    t = train_df[cols].copy()

    t["breath_id"] = t["breath_id"].astype(np.int32)
    t["R"] = t["R"].astype(np.int16)
    t["C"] = t["C"].astype(np.int16)
    t["u_out"] = t["u_out"].astype(np.int8)

    t["time_step_q"] = _q_round(t["time_step"], 100.0).astype(np.int16)  # 0.01s bins
    t["u_in_q"] = _q_round(t["u_in"], 10.0).astype(np.int16)  # 0.1 bins

    t.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    u_in_lag1 = (
        t.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    t["u_in_lag1_q"] = _q_round(u_in_lag1, 10.0).astype(np.int16)  # 0.1 bins

    dt = (
        t.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    u_in_cum = (
        (t["u_in"].astype(np.float32) * dt).groupby(t["breath_id"], sort=False).cumsum()
    )
    t["u_in_cum_q"] = _q_round(u_in_cum, 10.0).astype(np.int32)  # 0.1 bins

    t["u_out_lag1"] = (
        t.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    t["pressure_lag1"] = (
        t.groupby("breath_id", sort=False)["pressure"]
        .shift(1)
        .fillna(t["pressure"].median())
        .astype(np.float32)
    )
    t["pressure_lag1_g"] = bucket_pressure_to_grid(t["pressure_lag1"].to_numpy())

    t_in = t[t["u_out"] == 0].copy()

    median_map_full_lag_cum_plag_in = (
        t_in.groupby(
            [
                "R",
                "C",
                "time_step_q",
                "u_out",
                "u_out_lag1",
                "u_in_q",
                "u_in_lag1_q",
                "u_in_cum_q",
                "pressure_lag1_g",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag_cum_plag"})
    )

    median_map_full_lag_cum_plag_in_nouol1 = (
        t_in.groupby(
            [
                "R",
                "C",
                "time_step_q",
                "u_out",
                "u_in_q",
                "u_in_lag1_q",
                "u_in_cum_q",
                "pressure_lag1_g",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag_cum_plag_nouol1"})
    )

    median_map_full_lag_cum_plag_nolag_in = (
        t_in.groupby(
            [
                "R",
                "C",
                "time_step_q",
                "u_out",
                "u_out_lag1",
                "u_in_q",
                "u_in_lag1_q",
                "u_in_cum_q",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag_cum_plag_nolag"})
    )

    median_map_full_lag_cum_in = (
        t_in.groupby(
            ["R", "C", "time_step_q", "u_out", "u_in_q", "u_in_lag1_q", "u_in_cum_q"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag_cum"})
    )

    median_map_full_lag_in = (
        t_in.groupby(
            ["R", "C", "time_step_q", "u_out", "u_in_q", "u_in_lag1_q"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag"})
    )

    median_map_full_in = (
        t_in.groupby(["R", "C", "time_step_q", "u_out", "u_in_q"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full"})
    )

    median_map_rc_to_in = (
        t_in.groupby(["R", "C", "time_step_q", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rc_to"})
    )

    median_map_to_in = (
        t_in.groupby(["time_step_q", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_to"})
    )

    t_map_in = (
        t_in.groupby(["time_step_q"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_t"})
    )

    global_med_in = float(t_in["pressure"].median())

    t_out = t[t["u_out"] == 1].copy()
    if len(t_out) > 0:
        median_map_out = (
            t_out.groupby(["R", "C", "time_step_q", "u_out"], sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "p_out_rc_to"})
        )
        global_med_out = float(t_out["pressure"].median())
    else:
        median_map_out = pd.DataFrame(
            columns=["R", "C", "time_step_q", "u_out", "p_out_rc_to"]
        )
        global_med_out = global_med_in

    return (
        median_map_full_lag_cum_plag_in,
        median_map_full_lag_cum_plag_in_nouol1,
        median_map_full_lag_cum_plag_nolag_in,
        median_map_full_lag_cum_in,
        median_map_full_lag_in,
        median_map_full_in,
        median_map_rc_to_in,
        median_map_to_in,
        t_map_in,
        global_med_in,
        median_map_out,
        global_med_out,
    )


(
    FALLBACK_FULL_LAG_CUM_PLAG,
    FALLBACK_FULL_LAG_CUM_PLAG_NOUOL1,
    FALLBACK_FULL_LAG_CUM_PLAG_NOLAG,
    FALLBACK_FULL_LAG_CUM,
    FALLBACK_FULL_LAG,
    FALLBACK_FULL,
    FALLBACK_RC_TO,
    FALLBACK_TO,
    FALLBACK_T_MAP,
    FALLBACK_GLOBAL_IN,
    FALLBACK_OUT_RC_TO,
    FALLBACK_GLOBAL_OUT,
) = build_fallback_maps(df_train)


def fallback_predict(test_df: pd.DataFrame) -> np.ndarray:
    t = test_df[["breath_id", "R", "C", "time_step", "u_out", "u_in"]].copy()
    t["_orig_idx"] = np.arange(len(t), dtype=np.int32)

    t["breath_id"] = t["breath_id"].astype(np.int32)
    t["R"] = t["R"].astype(np.int16)
    t["C"] = t["C"].astype(np.int16)
    t["u_out"] = t["u_out"].astype(np.int8)

    t["time_step_q"] = _q_round(t["time_step"], 100.0).astype(np.int16)
    t["u_in_q"] = _q_round(t["u_in"], 10.0).astype(np.int16)

    t.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    u_in_lag1 = (
        t.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    t["u_in_lag1_q"] = _q_round(u_in_lag1, 10.0).astype(np.int16)

    dt = (
        t.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    u_in_cum = (
        (t["u_in"].astype(np.float32) * dt).groupby(t["breath_id"], sort=False).cumsum()
    )
    t["u_in_cum_q"] = _q_round(u_in_cum, 10.0).astype(np.int32)

    t["u_out_lag1"] = (
        t.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    m = t.merge(
        FALLBACK_FULL_LAG_CUM,
        how="left",
        on=["R", "C", "time_step_q", "u_out", "u_in_q", "u_in_lag1_q", "u_in_cum_q"],
    )
    m = m.merge(
        FALLBACK_FULL_LAG,
        how="left",
        on=["R", "C", "time_step_q", "u_out", "u_in_q", "u_in_lag1_q"],
    )
    m = m.merge(
        FALLBACK_FULL, how="left", on=["R", "C", "time_step_q", "u_out", "u_in_q"]
    )
    m = m.merge(FALLBACK_RC_TO, how="left", on=["R", "C", "time_step_q", "u_out"])
    m = m.merge(FALLBACK_TO, how="left", on=["time_step_q", "u_out"])
    m = m.merge(FALLBACK_T_MAP, how="left", on=["time_step_q"])
    m = m.merge(FALLBACK_OUT_RC_TO, how="left", on=["R", "C", "time_step_q", "u_out"])

    base_sorted = (
        m["p_full_lag_cum"]
        .fillna(m["p_full_lag"])
        .fillna(m["p_full"])
        .fillna(m["p_rc_to"])
        .fillna(m["p_to"])
        .fillna(m["p_t"])
        .fillna(m["p_out_rc_to"])
        .to_numpy(dtype=np.float64)
    )

    u_out_sorted = m["u_out"].to_numpy(dtype=np.int8)
    needs_global = ~np.isfinite(base_sorted)
    if needs_global.any():
        base_sorted = base_sorted.copy()
        base_sorted[needs_global & (u_out_sorted == 0)] = FALLBACK_GLOBAL_IN
        base_sorted[needs_global & (u_out_sorted == 1)] = FALLBACK_GLOBAL_OUT

    n = len(m)
    pred_sorted = base_sorted.copy()

    map_plag = FALLBACK_FULL_LAG_CUM_PLAG
    if len(map_plag) > 0:
        map_plag_dict = {
            (
                int(r),
                int(c),
                int(tsq),
                int(uo),
                int(uol1),
                int(uiq),
                int(uil1q),
                int(uicq),
                float(pl1g),
            ): float(p)
            for r, c, tsq, uo, uol1, uiq, uil1q, uicq, pl1g, p in map_plag[
                [
                    "R",
                    "C",
                    "time_step_q",
                    "u_out",
                    "u_out_lag1",
                    "u_in_q",
                    "u_in_lag1_q",
                    "u_in_cum_q",
                    "pressure_lag1_g",
                    "p_full_lag_cum_plag",
                ]
            ].itertuples(index=False, name=None)
        }
    else:
        map_plag_dict = {}

    map_plag_nouol1 = FALLBACK_FULL_LAG_CUM_PLAG_NOUOL1
    if len(map_plag_nouol1) > 0:
        map_plag_nouol1_dict = {
            (
                int(r),
                int(c),
                int(tsq),
                int(uo),
                int(uiq),
                int(uil1q),
                int(uicq),
                float(pl1g),
            ): float(p)
            for r, c, tsq, uo, uiq, uil1q, uicq, pl1g, p in map_plag_nouol1[
                [
                    "R",
                    "C",
                    "time_step_q",
                    "u_out",
                    "u_in_q",
                    "u_in_lag1_q",
                    "u_in_cum_q",
                    "pressure_lag1_g",
                    "p_full_lag_cum_plag_nouol1",
                ]
            ].itertuples(index=False, name=None)
        }
    else:
        map_plag_nouol1_dict = {}

    map_plag_nolag = FALLBACK_FULL_LAG_CUM_PLAG_NOLAG
    if len(map_plag_nolag) > 0:
        map_plag_nolag_dict = {
            (
                int(r),
                int(c),
                int(tsq),
                int(uo),
                int(uol1),
                int(uiq),
                int(uil1q),
                int(uicq),
            ): float(p)
            for r, c, tsq, uo, uol1, uiq, uil1q, uicq, p in map_plag_nolag[
                [
                    "R",
                    "C",
                    "time_step_q",
                    "u_out",
                    "u_out_lag1",
                    "u_in_q",
                    "u_in_lag1_q",
                    "u_in_cum_q",
                    "p_full_lag_cum_plag_nolag",
                ]
            ].itertuples(index=False, name=None)
        }
    else:
        map_plag_nolag_dict = {}

    breath_arr = m["breath_id"].to_numpy(dtype=np.int32)
    start_idx = np.r_[0, np.flatnonzero(breath_arr[1:] != breath_arr[:-1]) + 1]
    end_idx = np.r_[start_idx[1:], n]

    R_arr = m["R"].to_numpy(dtype=np.int16)
    C_arr = m["C"].to_numpy(dtype=np.int16)
    tsq_arr = m["time_step_q"].to_numpy(dtype=np.int16)
    uiq_arr = m["u_in_q"].to_numpy(dtype=np.int16)
    uil1q_arr = m["u_in_lag1_q"].to_numpy(dtype=np.int16)
    uicq_arr = m["u_in_cum_q"].to_numpy(dtype=np.int32)
    uol1_arr = m["u_out_lag1"].to_numpy(dtype=np.int8)

    for s, e in zip(start_idx, end_idx):
        prev_pred = FALLBACK_GLOBAL_IN
        for i in range(s, e):
            if u_out_sorted[i] == 1:
                prev_pred = FALLBACK_GLOBAL_IN
                continue

            plag_bucket = float(
                bucket_pressure_to_grid(np.array([prev_pred], dtype=np.float32))[0]
            )
            k = (
                int(R_arr[i]),
                int(C_arr[i]),
                int(tsq_arr[i]),
                int(u_out_sorted[i]),
                int(uol1_arr[i]),
                int(uiq_arr[i]),
                int(uil1q_arr[i]),
                int(uicq_arr[i]),
                plag_bucket,
            )
            hit = map_plag_dict.get(k, None)
            if hit is not None and np.isfinite(hit):
                pred_sorted[i] = float(hit)
            else:
                k_uol1_drop = (
                    int(R_arr[i]),
                    int(C_arr[i]),
                    int(tsq_arr[i]),
                    int(u_out_sorted[i]),
                    int(uiq_arr[i]),
                    int(uil1q_arr[i]),
                    int(uicq_arr[i]),
                    plag_bucket,
                )
                hit_uol1_drop = map_plag_nouol1_dict.get(k_uol1_drop, None)
                if hit_uol1_drop is not None and np.isfinite(hit_uol1_drop):
                    pred_sorted[i] = float(hit_uol1_drop)
                else:
                    k2 = (
                        int(R_arr[i]),
                        int(C_arr[i]),
                        int(tsq_arr[i]),
                        int(u_out_sorted[i]),
                        int(uol1_arr[i]),
                        int(uiq_arr[i]),
                        int(uil1q_arr[i]),
                        int(uicq_arr[i]),
                    )
                    hit2 = map_plag_nolag_dict.get(k2, None)
                    if hit2 is not None and np.isfinite(hit2):
                        pred_sorted[i] = float(hit2)

            prev_pred = pred_sorted[i]

    pred = np.empty(len(test_df), dtype=np.float64)
    pred[m["_orig_idx"].to_numpy(dtype=np.int32)] = pred_sorted
    return pred


def g(dp):
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    n = len(output)

    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    l.sort()

    file_count = len(l)
    loop_time = 154

    if file_count == 0:
        test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        output["pressure"] = fallback_predict(test_df)
        output["pressure"] = output["pressure"].apply(find_nearest)
        out_path = "/kaggle/working/submission.csv"
        output.to_csv(out_path, index=False)
        return out_path

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        flist.append(l[start:end])

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    flist = [arr for arr in flist if isinstance(arr, np.ndarray) and arr.shape[0] == n]

    if len(flist) == 0:
        test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        output["pressure"] = fallback_predict(test_df)
        output["pressure"] = output["pressure"].apply(find_nearest)
        out_path = "/kaggle/working/submission.csv"
        output.to_csv(out_path, index=False)
        return out_path

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            for j in range(len(weight)):
                weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)
    med = np.median(stacked, axis=0)
    if med.shape[0] != n:
        test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        med = fallback_predict(test_df)

    output["pressure"] = med
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_path = "/kaggle/working/submission.csv"
    output.to_csv(out_path, index=False)
    return out_path


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    out_path = "/kaggle/working/blend.csv"
    a.to_csv(out_path, index=False)
    return a




## === cell 2
submission_path = g("../input/gb-data-blending-recover")
print("Wrote:", submission_path)
print(pd.read_csv(submission_path).head())
print(
    "Rows:",
    len(pd.read_csv(submission_path)),
    "Cols:",
    pd.read_csv(submission_path).columns.tolist(),
)
