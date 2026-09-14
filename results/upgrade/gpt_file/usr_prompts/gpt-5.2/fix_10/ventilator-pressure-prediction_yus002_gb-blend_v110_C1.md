# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the failure by making the blending code robust to missing/empty input directories and to files that don’t contain the expected `pressure` column or correct length. The current error happens because `pred_list` ends up containing arrays of the wrong shape (likely due to no valid prediction files found), so the median becomes length 1 instead of 603600. I add validation and a safe fallback that produces a valid submission (using sample_submission’s zeros) while still preserving your core blending logic when valid prediction files are available. I also ensure the output filename ends with `.csv` and is written to `/kaggle/working/` so Kaggle picks it up.'
- What this solution (achieved 9.006) has done: 'Your current script is a “blender” that mostly falls back to all-zeros because `../input/gb-data-blending-recover` is not available here, which explains the very poor MAE (17.65). To move the score toward the target while preserving your core logic, I keep the same blending pipeline but add a minimal, deterministic fallback “base prediction” built from the training data: per-(R,C,time_step,u_out) median pressure, with a global time_step fallback if a key is unseen. This keeps the same submission semantics (predict pressure per row), still snaps predictions to the valid pressure grid via `find_nearest`, and substantially reduce MAE compared with zeros when no external prediction files exist. The code still preferentially blend external files if they exist and are valid, only using the fallback when needed, and it always write `/kaggle/working/submission.csv`.'
- What this solution (achieved 4.00094) has done: 'To move your score closer to the 0.142 target (lower is better) while preserving the same “blender + fallback” core logic, I make the fallback much more informative by incorporating `u_in` (the main control signal) and by using medians at several granularities with safe backoff (exact match → partial match → time_step-only → global). I also ensure the fallback map build is memory-safe (cast keys to smaller dtypes, select only needed columns) so it runs within the time limit. Finally, I keep your existing `find_nearest` snapping (important for this competition) and keep external blending behavior unchanged when those files exist and are valid.'
- What this solution (achieved 8.06642) has done: 'Your current score (4.00094 MAE) is far above the target (0.14237), so we need a real accuracy lift while keeping your “blender + fallback mapping + snap-to-grid” logic intact. The biggest issue is that the fallback mapping ignores the strongest sequence signal: within-breath history (previous `u_in` and previous pressure), so it collapses to a noisy static lookup that can’t match the dynamics well. I keep your same fallback concept (groupby medians + safe backoff) but minimally extend the keys with a within-breath lag feature (`u_in_lag1`, rounded) computed from `breath_id` order, and add a corresponding backoff level; this preserves your overall approach and evaluation semantics while typically improving MAE substantially. I also ensure the `id` alignment is correct by building predictions directly on the loaded `test.csv` order and then writing to `sample_submission.csv` (same length/order) as you already do.'
- What this solution (achieved 3.7039) has done: 'Your current score is far worse than the target (lower is better), so we should improve accuracy without changing the overall “blending + fallback lookup + snap-to-pressure-grid” approach. The biggest correctness/score issue in the fallback is misalignment: `fallback_predict()` sorts the test rows, merges, then tries to reindex back using mismatched indices, which can scramble predictions and inflate MAE. I fix this by carrying the original row index through the sort/merge and restoring predictions to the original order deterministically. This is a minimal change that preserves your mapping logic, blending behavior, and `find_nearest` snapping, but should materially reduce error.'
- What this solution (achieved 3.15558) has done: 'Your current MAE (3.7039, lower is better) is still far from the target (0.1424), so we should improve the fallback predictor (used when no external blend files exist) while keeping your overall “blender + fallback lookup + snap-to-grid” logic unchanged. The smallest high-impact fix is to align the fallback maps with the evaluation: only inspiratory phase (u_out == 0) is scored, so we should build fallback medians primarily from inspiratory rows, and use a dedicated expiratory fallback for u_out == 1. Additionally, I add a minimal extra sequential signal by including a rounded cumulative integral of u_in within-breath (a common physics proxy for delivered volume) as an additional key with backoff, without changing your architecture/training (still pure groupby-median lookup). This typically reduces MAE materially versus using only instantaneous u_in and u_in_lag1, while preserving your existing blending behavior and pressure-grid snapping.'
- What this solution (achieved 3.15258) has done: 'I keep your blender intact and focus on making the fallback lookup more informative in a minimal way, because your current MAE (3.15558, lower is better) is still far above the target (0.14237). The smallest high-impact improvement that preserves your “groupby-median maps + backoff + snap-to-grid” core logic is to add one more within-breath sequential key: a rounded lag-1 pressure proxy learned from training (`pressure_lag1_r`) and applied at inference via mapped medians (not a model change, still pure lookup). I build an additional inspiratory-phase map keyed by (R,C,time_step,u_out,u_in,u_in_lag1,u_in_cum,pressure_lag1) with safe backoff to your existing maps, and I compute the test-time lag proxy using the already-predicted previous step (teacher-forcing style, still deterministic). This should materially reduce error while keeping execution under time and maintaining identical submission semantics and `find_nearest` snapping.'
- What this solution (achieved 3.15099) has done: 'Your current score is far above the target (lower is better), so we need a small but meaningful accuracy lift without changing your overall “fallback lookup + backoff + snap-to-grid” approach. The highest-impact minimal fix is to make the `pressure_lag1` feature consistent at inference: right now it uses a “one-pass” lag from a coarse first prediction, but it never truly follows the within-breath recursion that `pressure_lag1` implies. I keep all your existing maps and backoff order, but change inference to compute predictions step-by-step within each breath so `pressure_lag1_r` is derived from the previously predicted step (same semantics you intended, just applied correctly). This typically reduces MAE materially while staying within time and still writing `/kaggle/working/submission.csv`.'

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


def build_fallback_maps(train_df: pd.DataFrame):
    cols = ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
    t = train_df[cols].copy()

    t["breath_id"] = t["breath_id"].astype(np.int32)
    t["R"] = t["R"].astype(np.int16)
    t["C"] = t["C"].astype(np.int16)
    t["u_out"] = t["u_out"].astype(np.int8)

    t["time_step_r"] = np.round(t["time_step"].astype(np.float32), 2)
    t["u_in_r"] = np.round(t["u_in"].astype(np.float32), 1)

    t.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    t["u_in_lag1_r"] = (
        t.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    t["u_in_lag1_r"] = np.round(t["u_in_lag1_r"], 1)

    dt = (
        t.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    t["u_in_cum"] = (
        (t["u_in"].astype(np.float32) * dt).groupby(t["breath_id"], sort=False).cumsum()
    )
    t["u_in_cum_r"] = np.round(t["u_in_cum"].astype(np.float32), 1)

    t["pressure_lag1_r"] = (
        t.groupby("breath_id", sort=False)["pressure"]
        .shift(1)
        .fillna(t["pressure"].median())
        .astype(np.float32)
    )
    t["pressure_lag1_r"] = np.round(t["pressure_lag1_r"], 2)

    t_in = t[t["u_out"] == 0].copy()

    median_map_full_lag_cum_plag_in = (
        t_in.groupby(
            [
                "R",
                "C",
                "time_step_r",
                "u_out",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_cum_r",
                "pressure_lag1_r",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag_cum_plag"})
    )

    median_map_full_lag_cum_in = (
        t_in.groupby(
            ["R", "C", "time_step_r", "u_out", "u_in_r", "u_in_lag1_r", "u_in_cum_r"],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag_cum"})
    )

    median_map_full_lag_in = (
        t_in.groupby(
            ["R", "C", "time_step_r", "u_out", "u_in_r", "u_in_lag1_r"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full_lag"})
    )

    median_map_full_in = (
        t_in.groupby(["R", "C", "time_step_r", "u_out", "u_in_r"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full"})
    )

    median_map_rc_to_in = (
        t_in.groupby(["R", "C", "time_step_r", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rc_to"})
    )

    median_map_to_in = (
        t_in.groupby(["time_step_r", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_to"})
    )

    t_map_in = (
        t_in.groupby(["time_step_r"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_t"})
    )

    global_med_in = float(t_in["pressure"].median())

    t_out = t[t["u_out"] == 1].copy()
    if len(t_out) > 0:
        median_map_out = (
            t_out.groupby(["R", "C", "time_step_r", "u_out"], sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "p_out_rc_to"})
        )
        global_med_out = float(t_out["pressure"].median())
    else:
        median_map_out = pd.DataFrame(
            columns=["R", "C", "time_step_r", "u_out", "p_out_rc_to"]
        )
        global_med_out = global_med_in

    return (
        median_map_full_lag_cum_plag_in,
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
    t["time_step_r"] = np.round(t["time_step"].astype(np.float32), 2)
    t["u_in_r"] = np.round(t["u_in"].astype(np.float32), 1)

    t.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    t["u_in_lag1_r"] = (
        t.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    t["u_in_lag1_r"] = np.round(t["u_in_lag1_r"], 1)

    dt = (
        t.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    t["u_in_cum"] = (
        (t["u_in"].astype(np.float32) * dt).groupby(t["breath_id"], sort=False).cumsum()
    )
    t["u_in_cum_r"] = np.round(t["u_in_cum"].astype(np.float32), 1)

    m = t.merge(
        FALLBACK_FULL_LAG_CUM_PLAG,
        how="left",
        on=[
            "R",
            "C",
            "time_step_r",
            "u_out",
            "u_in_r",
            "u_in_lag1_r",
            "u_in_cum_r",
            "pressure_lag1_r",  # will be added later during recursion; kept for schema consistency
        ],
    )

    if "p_full_lag_cum_plag" not in m.columns:
        m["p_full_lag_cum_plag"] = np.nan

    m = t.merge(
        FALLBACK_FULL_LAG_CUM,
        how="left",
        on=["R", "C", "time_step_r", "u_out", "u_in_r", "u_in_lag1_r", "u_in_cum_r"],
    )
    m = m.merge(
        FALLBACK_FULL_LAG,
        how="left",
        on=["R", "C", "time_step_r", "u_out", "u_in_r", "u_in_lag1_r"],
    )
    m = m.merge(
        FALLBACK_FULL, how="left", on=["R", "C", "time_step_r", "u_out", "u_in_r"]
    )
    m = m.merge(FALLBACK_RC_TO, how="left", on=["R", "C", "time_step_r", "u_out"])
    m = m.merge(FALLBACK_TO, how="left", on=["time_step_r", "u_out"])
    m = m.merge(FALLBACK_T_MAP, how="left", on=["time_step_r"])
    m = m.merge(FALLBACK_OUT_RC_TO, how="left", on=["R", "C", "time_step_r", "u_out"])

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

    map_plag_df = FALLBACK_FULL_LAG_CUM_PLAG

    breath_arr = m["breath_id"].to_numpy(dtype=np.int32)

    start_idx = np.r_[0, np.flatnonzero(breath_arr[1:] != breath_arr[:-1]) + 1]
    end_idx = np.r_[start_idx[1:], n]

    for s, e in zip(start_idx, end_idx):
        prev_pred = FALLBACK_GLOBAL_IN
        for i in range(s, e):
            if u_out_sorted[i] == 1:
                prev_pred = pred_sorted[i]
                continue

            pressure_lag1_r = float(np.round(np.float32(prev_pred), 2))

            key_df = pd.DataFrame(
                {
                    "R": [m.at[i, "R"]],
                    "C": [m.at[i, "C"]],
                    "time_step_r": [m.at[i, "time_step_r"]],
                    "u_out": [m.at[i, "u_out"]],
                    "u_in_r": [m.at[i, "u_in_r"]],
                    "u_in_lag1_r": [m.at[i, "u_in_lag1_r"]],
                    "u_in_cum_r": [m.at[i, "u_in_cum_r"]],
                    "pressure_lag1_r": [pressure_lag1_r],
                }
            )
            hit = key_df.merge(
                map_plag_df,
                how="left",
                on=[
                    "R",
                    "C",
                    "time_step_r",
                    "u_out",
                    "u_in_r",
                    "u_in_lag1_r",
                    "u_in_cum_r",
                    "pressure_lag1_r",
                ],
            )["p_full_lag_cum_plag"].iloc[0]

            if np.isfinite(hit):
                pred_sorted[i] = float(hit)

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

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3900883445.py in <cell line: 0>()
----> 1 submission_path = g("../input/gb-data-blending-recover")
      2 print("Wrote:", submission_path)
      3 print(pd.read_csv(submission_path).head())

/tmp/ipykernel_11/697538519.py in g(dp)
    395     if file_count == 0:
    396         test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
--> 397         output["pressure"] = fallback_predict(test_df)
    398         output["pressure"] = output["pressure"].apply(find_nearest)
    399         out_path = "/kaggle/working/submission.csv"

/tmp/ipykernel_11/697538519.py in fallback_predict(test_df)
    257     # Change (score-relevant, minimal): merge the strongest precomputed map first so it can contribute.
    258     # Previously this map was only used via a Python dict override, which is sparse and slow.
--> 259     m = t.merge(
    260         FALLBACK_FULL_LAG_CUM_PLAG,
    261         how="left",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1308                         #  the latter of which will raise
   1309                         lk = cast(Hashable, lk)
-> 1310                         left_keys.append(left._get_label_or_level_values(lk))
   1311                         join_names.append(lk)
   1312                     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'pressure_lag1_r'
