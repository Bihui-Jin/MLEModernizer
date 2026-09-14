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

0.1485792995591306

# 6. Current score

1.55588

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The error comes from `g()` building `pred_list` arrays of inconsistent shape: when the input directory is empty or has too few/invalid files, the current split/weighting logic can collapse to a scalar (length 1), causing the median vector to have length 1 and fail assignment to the 603600-row submission. I make `g()` robust by (1) validating the blend directory and CSV contents, (2) collecting only valid pressure vectors of the correct length, and (3) ensuring the final stacked predictions are always shape `(n_models, n_test_rows)`. I also keep the original blending/median + `find_nearest` core logic intact, but write the submission to a deterministic valid filename `submission.csv` so Kaggle always picks it up. These changes are correctness/stability focused; they should also move the score toward the target compared to “no submission” by producing a legitimate blended submission.'
- What this solution (achieved 10.86378) has done: 'Your current score (17.65 MAE) is far worse than the target (0.1486), and the most likely reason is that you’re effectively submitting all-zeros (or near-zeros) because `../input/gb-blending` doesn’t exist in this environment, triggering the fallback path. I keep your blending/median + `find_nearest` logic intact, but make `g()` automatically fall back to using `sample_submission.csv` located alongside the provided competition files and search a small set of likely blend directories under `../input` (including nested ones) so it actually finds model submissions if present. If no blend files exist anywhere, I switch the fallback from “all zeros” to a simple, legitimate baseline: predict the global median training pressure (then snap to nearest valid pressure), which should drastically reduce MAE without changing your core approach. The script still always write a valid `submission.csv`.'
- What this solution (achieved 7.46367) has done: 'Your score is far worse than the target (lower is better), and in this environment you likely have no real blend files available, so the code falls back to a constant global-median pressure which performs poorly for this competition. Keeping your core “blend → median ensemble → snap to nearest valid pressure” logic intact, the smallest legitimate improvement is to make the fallback baseline *use the provided test features*: predict the median **training inspiratory-phase** pressure for each (R, C) group, and (optionally) further condition on `u_out` (since expiratory phase isn’t scored, but this still helps match dynamics). This stays within your existing semantics (still producing a pressure vector then snapping to valid pressures), but should drastically reduce MAE toward the target when no external blend submissions exist. I also make the fallback robust to missing groups by backing off to (R,C) medians and then global median, and I keep output formatting identical (`submission.csv`, `id,pressure`).'
- What this solution (achieved 6.02322) has done: 'Your current score is much worse than the target (lower is better), and because there are no actual blend CSVs available here, your pipeline is effectively using the fallback baseline. Keeping your core logic (blend/median ensemble + snapping to nearest valid pressures) intact, the smallest meaningful improvement is to make the fallback baseline use more informative, still-leak-free features that exist in train/test: build per-(R,C,time_step,u_out)-median pressure from inspiratory-only rows (u_out==0), with a safe backoff chain to coarser group medians and finally global median. This better matches the per-timestep pressure trajectory without changing any modeling approach, and should reduce MAE substantially toward the target while still producing a valid `submission.csv`. I also ensure the fallback produces a prediction vector aligned 1:1 with test `id` ordering.'
- What this solution (achieved 1.47916) has done: 'Your current score (6.02 MAE, lower is better) is still far from the target (0.1486), which strongly suggests the code is running in “no external blend files found” mode and relying on the fallback baseline. Keeping your core semantics intact (build a prediction vector, then snap with `find_nearest`, and still prefer blend files if present), I minimally strengthen the fallback to use breath-level dynamics: add per-breath cumulative/lag features available in both train/test and predict pressure by median over fine-grained groups derived from those features (with a safe backoff chain). This is still a pure leak-free median-lookup baseline (no model change), but it typically reduces MAE a lot for this competition compared to using only (R,C,u_out,time_step). I also ensure alignment is strictly by `id` order and keep writing `submission.csv` with the required columns.'
- What this solution (achieved 1.24588) has done: 'Your current score (1.47916 MAE) is still far above the target (0.14858), and given the environment likely has no external blend files, the result is dominated by the fallback median-lookup baseline. To move the score toward the target without changing the core “lookup/ensemble → snap to nearest pressure” semantics, I minimally strengthen the fallback by adding one more within-breath dynamics feature (a coarse bin of the instantaneous `u_in` itself) and using a slightly finer time rounding, then extending the same safe backoff chain. This keeps the approach as a pure leak-free median mapping (no model/training changes), but makes the lookup closer to the true pressure trajectory. I also keep output alignment strictly by `id` and still write a valid `submission.csv`.'
- What this solution (achieved 1.21732) has done: 'Your current MAE (1.24588, lower is better) is still far above the target (0.14858), and given the environment likely has no blend CSVs available, your score is dominated by the fallback baseline. Keeping your core logic intact (median-lookup fallback → snap to nearest valid pressure; and still prefer blend files if present), I minimally strengthen the fallback lookup keys using one additional within-breath dynamic feature (`u_in_diff` binned) and make the “fine” time rounding actually finer (0.01) as the comment originally intended. This increases the chance of matching the correct pressure trajectory without changing any modeling/training approach, and preserves your existing safe backoff chain. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.44422) has done: 'Your fallback baseline is still being used (no blend files found), and its biggest accuracy limiter is that it ignores key breath dynamics that strongly determine pressure (instantaneous flow and volume evolution). To move MAE down toward the 0.1486 target without changing your core “median-lookup fallback → snap to nearest valid pressure” semantics, I keep the same structure but compute two standard leak-free ventilator features available in both train/test: `u_in`-integral over time (proxy for inspired volume) and `flow` proxy (`u_in` adjusted by `u_out`). Then I condition the median lookup on these binned dynamics with the same safe backoff chain, and I fix the “fine rounding” to actually be fine (0.01) as intended. Everything else (blending path discovery, median ensemble, snapping, and writing `submission.csv`) stays intact.'
- What this solution (achieved 1.50385) has done: 'Your current MAE (1.44422, lower is better) is still far above the target (0.14858), so we need a small but meaningful accuracy lift without changing the overall “median-lookup fallback → snap to nearest pressure” core logic. The biggest issue in your current fallback is that `time_step` rounding is accidentally stuck at 0.02 everywhere, which collapses the time resolution and harms the per-timestep trajectory match; I fix this to true fine rounding at 0.01 and use it as the primary time key. To further nudge predictions toward the right trajectory with minimal change, I add one more leak-free dynamic feature already implied by your setup: binned `u_in` integral over actual `dt` (not raw cumsum), and include it only in the most specific lookup level with the same backoff chain. Everything else (blend discovery, weighted combine, median ensemble, snapping, and writing `submission.csv`) stays intact and still runs end-to-end.'
- What this solution (achieved 1.54654) has done: 'Your score gap to the target is still large (1.50385 vs 0.14858 MAE; lower is better), and since this notebook likely has no external blend files, the fallback baseline dominates performance. Keeping your core “median-lookup fallback → snap to nearest valid pressure” logic intact, the smallest meaningful lift is to fix a key leakage-free alignment issue: your fallback currently computes `dt` from test/train time differences, but the first timestep becomes `dt=0`, hurting the volume/integral features and downstream grouping; we set `dt` to the known timestep spacing (~0.03) for the first row of each breath. I also add a tiny, competition-specific correction that preserves semantics: predictions for `u_out==1` (expiratory phase, unscored) are set to a constant (global inspiratory median) before snapping, which reduces noise without affecting the inspiratory-scored part. Everything else (blend discovery, weighted combine, median ensembling, snapping, and writing `submission.csv`) stays the same.'
- What this solution (achieved 1.54654) has done: 'Your current MAE (1.54654, lower is better) is still far above the target (0.14858), so we should cautiously improve accuracy without changing your overall “median-lookup fallback → snap to nearest valid pressure” and “blend if available” core logic. The biggest gain with minimal semantic change is to align the fallback baseline with the evaluation: the metric only scores inspiratory phase, so we should build lookup medians using inspiratory rows only (u_out==0) but *without* using `u_out` as a grouping key (since it’s constant there and harms matching when test has u_out==1). We then predict inspiratory-phase pressures for all rows (including u_out==1) from inspiratory-derived dynamics and only afterward set u_out==1 to a constant (still unscored), keeping your existing behavior while improving inspiratory predictions and reducing missing-key fallbacks. All paths, blending logic, snapping to nearest valid pressures, and submission writing stay the same.'
- What this solution (achieved 1.54458) has done: 'Your current MAE (1.54654, lower is better) is still far above the target (0.14858), and since this code is almost certainly running in “no blend files found” mode, the fallback baseline dominates your score. Keeping your core logic intact (median-lookup fallback → snap to nearest valid pressure; blend/median ensemble if files exist), the smallest meaningful lift is to make the fallback keys better match breath dynamics by adding an explicitly stable within-breath step index (0–79) and using it in the coarser lookup levels (instead of relying mainly on rounded time). This improves lookup hit-rate and trajectory alignment with minimal semantic change and no modeling/training changes. I also keep the existing backoff chain and ensure the output stays aligned by `id` and writes a valid `submission.csv`.'
- What this solution (achieved 1.55588) has done: 'Your current MAE (1.54458, lower-is-better) is still far above the target (0.14858), and since this environment likely has no real blend files, the score is dominated by the fallback median-lookup baseline. Keeping your exact core semantics (median-lookup fallback → snap to nearest valid pressure; blend/median ensemble if files exist), the smallest meaningful improvement is to compute the within-breath dynamics (`flow`, `vol`, `vol_cum`, lags/diffs) using the *known fixed 80-step structure* (constant dt from per-breath time span / 79) instead of per-row diffs with an arbitrary 0.03 for the first step, which reduces systematic feature error and improves lookup hit-rate. I also align `t_idx` strictly with the stable `id % 80` pattern (still leak-free) rather than relying on sorting/groupby cumcount, to avoid any accidental ordering issues. Everything else (blend discovery, weighted combining, median ensembling, snapping, and writing `submission.csv`) stays the same and still produces a valid submission.'

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
    """
    Original intent: read a list of submission files and do a simple weighted combine.
    Bugfix: make parsing of score-from-filename optional and robust; always return a 1D vector.
    """
    if len(input_list) == 0:
        return None

    pressures = []
    weights_hint = []

    for p in input_list:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "pressure" not in df.columns:
            continue
        arr = df["pressure"].to_numpy().ravel()
        pressures.append(arr)

        try:
            public_lb_score = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        weights_hint.append(public_lb_score)

    if len(pressures) == 0:
        return None
    if len(pressures) == 1:
        return pressures[0]

    if len(pressures) >= 2:
        l_sum = sum(weights_hint[:2]) if sum(weights_hint[:2]) != 0 else 2
        weight1 = (weights_hint[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return pressures[0] * weight1 + pressures[1] * weight2

    return np.mean(np.vstack(pressures), axis=0)


def _resolve_sample_submission_path():
    candidates = [
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        "../input/sample_submission.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "sample_submission.csv not found in expected ../input locations."
    )


def _resolve_test_path():
    candidates = [
        "../input/ventilator-pressure-prediction/test.csv",
        "../input/test.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("test.csv not found in expected ../input locations.")


def _discover_blend_dir(requested_dp):
    if requested_dp is not None and os.path.isdir(requested_dp):
        return requested_dp

    search_roots = ["../input"]
    name_hints = [
        "blend",
        "blending",
        "submission",
        "submissions",
        "gb-blending",
        "ensemble",
        "ensembling",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for depth_pattern in ["*", "*/*", "*/*/*"]:
            for d in glob.glob(os.path.join(root, depth_pattern)):
                if os.path.isdir(d):
                    dn = os.path.basename(d).lower()
                    if any(h in dn for h in name_hints):
                        return d

    return requested_dp  # may be None/nonexistent; g() will handle gracefully


def _add_breath_dynamics(df):
    """
    Score-relevant enrichment for the *fallback* (median lookup), preserving semantics.

    Change to move score toward target (minimal, no model/training change):
    - Use the known 80-step breath structure to compute a stable within-breath step index:
        t_idx = (id - 1) % 80
      This avoids any accidental ordering/groupby cumcount issues.
    - Compute dt per breath from the observed breath duration:
        dt = (max(time_step) - min(time_step)) / 79
      and assign this constant dt to all 80 steps. This is closer to the true sampling and avoids
      the first-step dt=0 (or arbitrary 0.03) distortion that propagates into vol/vol_cum features.
    - Keep existing derived features and binning logic unchanged otherwise.
    """
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    df["t_idx"] = ((df["id"].astype(np.int64) - 1) % 80).astype(np.int16)

    bmin = (
        df.groupby("breath_id", sort=False)["time_step"]
        .transform("min")
        .astype(np.float32)
    )
    bmax = (
        df.groupby("breath_id", sort=False)["time_step"]
        .transform("max")
        .astype(np.float32)
    )
    dt_breath = ((bmax - bmin) / 79.0).astype(np.float32)
    dt_breath = dt_breath.where(dt_breath > 0, other=np.float32(0.03))
    df["dt"] = dt_breath

    df["flow"] = (df["u_in"] * (1.0 - df["u_out"].astype(np.float32))).astype(
        np.float32
    )
    df["vol"] = (df["flow"] * df["dt"]).astype(np.float32)
    df["vol_cum"] = (
        df.groupby("breath_id", sort=False)["vol"].cumsum().astype(np.float32)
    )

    df["u_in_lag1"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    df["u_in_diff"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

    df["ts_001"] = (df["time_step"] / 0.01).round().astype(np.int16)
    df["time_step_round"] = df["time_step"].round(2)

    df["u_in_dt"] = (df["u_in"].astype(np.float32) * df["dt"]).astype(np.float32)
    df["u_in_dt_cum"] = (
        df.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
    )

    df["u_in_lag1_bin"] = (df["u_in_lag1"] / 5.0).astype(np.int16)
    df["u_in_bin"] = (df["u_in"] / 2.0).astype(np.int16)
    df["u_in_diff_bin"] = (df["u_in_diff"] / 2.0).astype(np.int16)

    df["flow_bin"] = (df["flow"] / 2.0).astype(np.int16)
    df["vol_cum_bin"] = (df["vol_cum"] / 0.5).astype(np.int16)

    df["u_in_dt_cum_bin"] = (df["u_in_dt_cum"] / 0.5).astype(np.int16)

    return df


def _fallback_pressure_baseline(output_df):
    """
    Median-lookup baseline -> snap (preserved core semantics).
    """
    test_path = _resolve_test_path()
    df_test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "u_out", "time_step", "u_in"]
    )
    df_test = _add_breath_dynamics(df_test)

    df_tr = df_train.loc[
        :, ["id", "breath_id", "R", "C", "u_out", "time_step", "u_in", "pressure"]
    ].copy()
    df_tr = _add_breath_dynamics(df_tr)

    df_tr_insp = df_tr.loc[df_tr["u_out"] == 0].copy()

    med_1 = (
        df_tr_insp.groupby(
            [
                "R",
                "C",
                "ts_001",
                "vol_cum_bin",
                "flow_bin",
                "u_in_dt_cum_bin",
                "u_in_lag1_bin",
                "u_in_bin",
                "u_in_diff_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .rename("m1")
        .reset_index()
    )
    med_2 = (
        df_tr_insp.groupby(
            ["R", "C", "ts_001", "vol_cum_bin", "flow_bin"],
            sort=False,
        )["pressure"]
        .median()
        .rename("m2")
        .reset_index()
    )
    med_3 = (
        df_tr_insp.groupby(["R", "C", "ts_001", "t_idx"], sort=False)["pressure"]
        .median()
        .rename("m3")
        .reset_index()
    )
    med_4 = (
        df_tr_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
        .median()
        .rename("m4")
        .reset_index()
    )
    med_5 = (
        df_tr_insp.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .rename("m5")
        .reset_index()
    )
    global_med = float(df_tr_insp["pressure"].median())

    df_pred = df_test.merge(
        med_1,
        on=[
            "R",
            "C",
            "ts_001",
            "vol_cum_bin",
            "flow_bin",
            "u_in_dt_cum_bin",
            "u_in_lag1_bin",
            "u_in_bin",
            "u_in_diff_bin",
        ],
        how="left",
    )
    df_pred = df_pred.merge(
        med_2, on=["R", "C", "ts_001", "vol_cum_bin", "flow_bin"], how="left"
    )
    df_pred = df_pred.merge(med_3, on=["R", "C", "ts_001", "t_idx"], how="left")
    df_pred = df_pred.merge(med_4, on=["R", "C", "t_idx"], how="left")
    df_pred = df_pred.merge(med_5, on=["R", "C"], how="left")

    pred = df_pred["m1"].to_numpy()
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = df_pred.loc[mask, "m2"].to_numpy()
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = df_pred.loc[mask, "m3"].to_numpy()
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = df_pred.loc[mask, "m4"].to_numpy()
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = df_pred.loc[mask, "m5"].to_numpy()
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = global_med

    exp_mask = df_pred["u_out"].to_numpy() == 1
    if exp_mask.any():
        pred[exp_mask] = global_med

    df_pred_out = df_pred.loc[:, ["id"]].copy()
    df_pred_out["pressure"] = pred
    df_pred_out.sort_values("id", inplace=True, kind="mergesort")

    pred_snapped = np.fromiter(
        (find_nearest(x) for x in df_pred_out["pressure"].to_numpy()),
        dtype=float,
        count=len(df_pred_out),
    )

    out = output_df.copy()
    out.sort_values("id", inplace=True, kind="mergesort")
    out["pressure"] = pred_snapped
    out.sort_values("id", inplace=True, kind="mergesort")
    return out


def g(dp):
    """
    Robust version of the original random-weight blending + median ensembling.
    """
    sample_path = _resolve_sample_submission_path()
    output = pd.read_csv(sample_path)
    if "id" not in output.columns or "pressure" not in output.columns:
        raise ValueError("sample_submission must have columns ['id','pressure']")
    n_test = len(output)

    dp = _discover_blend_dir(dp)

    file_list = []
    if dp is not None and os.path.isdir(dp):
        file_list = sorted([p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)])

    valid_files = []
    for p in file_list:
        if not p.lower().endswith(".csv"):
            continue
        try:
            tmp = pd.read_csv(p, usecols=["pressure"])
        except Exception:
            continue
        if len(tmp) != n_test:
            continue
        valid_files.append(p)

    if len(valid_files) == 0:
        output = _fallback_pressure_baseline(output)
        output.to_csv("submission.csv", index=False)
        return output

    splits = 2 if len(valid_files) >= 2 else 1
    chunk_size = int(np.ceil(len(valid_files) / splits))
    flist = []
    for i in range(splits):
        chunk = valid_files[i * chunk_size : (i + 1) * chunk_size]
        vec = wc(chunk)
        if vec is None:
            continue
        vec = np.asarray(vec).ravel()
        if vec.shape[0] != n_test:
            continue
        flist.append(vec)

    if len(flist) == 0:
        output = _fallback_pressure_baseline(output)
        output.to_csv("submission.csv", index=False)
        return output

    loop_time = 150
    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n_test, dtype=float)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    pred_stack = np.vstack([np.asarray(p).ravel() for p in pred_list])
    median_pred = np.median(pred_stack, axis=0)

    output["pressure"] = median_pred
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-blending")
