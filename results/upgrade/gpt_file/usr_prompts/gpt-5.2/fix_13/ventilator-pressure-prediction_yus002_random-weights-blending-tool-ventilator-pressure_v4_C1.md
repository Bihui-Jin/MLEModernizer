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

0.1513939814206466

# 6. Current score

1.80658

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.31565) has done: 'The crash happens because `../input/gb-rwbt-files` does not exist in your environment, so the ensemble code finds zero files and divides by zero. I make `g()` robust by (1) validating the directory and file list, and (2) falling back to generating predictions from the provided train/test data using a simple groupwise median baseline (by `R`, `C`, and discretized `u_in`/`u_out`) so a valid submission is always produced. This keeps the overall pipeline intent (produce a pressure prediction vector and snap to nearest allowed pressure) while ensuring it runs end-to-end and writes a `.csv` submission. The submission filename be a standard `submission.csv` to satisfy Kaggle requirements.'
- What this solution (achieved 2.94989) has done: 'Your current score (6.31565 MAE) is far worse than the target (~0.151), so we should meaningfully improve predictions while keeping the same overall “non-ML baseline pipeline” core idea. The simplest high-impact fix is to make the fallback baseline respect the competition’s key structure: pressure depends heavily on within-breath dynamics, so we add lightweight time-series features (lagged `u_in`, cumulative integral/mean) and predict using a median lookup keyed by (`R`,`C`,`u_out`,`time_step`,`u_in_bin`,`u_in_lag_bin`). We keep your snapping-to-nearest-allowed-pressure exactly as-is, and we still fall back only when the external directory doesn’t exist, producing `submission.csv` end-to-end. These changes are minimal (still pure pandas groupby-median, no new model/training loop) but should move MAE substantially toward the target.'
- What this solution (achieved 2.91187) has done: 'Your current MAE (2.94989, lower is better) is still far from the target (0.1514), so we should improve the fallback baseline while keeping your “groupwise median lookup + snap-to-nearest-pressure” core logic unchanged. The biggest remaining gap is that the fallback does not explicitly handle the “only inspiratory phase is scored” rule; we can legitimately improve by setting predictions to 0 during expiration (`u_out==1`) since those rows are not evaluated. Additionally, we can make the median lookup slightly more time-series-aware with one more minimal within-breath feature (`u_in_diff1_bin`) and keep the same hierarchical fallback keys. These are small, safe changes that preserve the non-ML approach and should move the score substantially toward the target without changing the overall pipeline structure.'
- What this solution (achieved 2.75842) has done: 'Your current MAE (2.91187, lower is better) is still far above the target (0.15139), so we should improve the fallback baseline while keeping the same core “groupwise median lookup + snap-to-nearest-pressure” logic. The most direct improvement with minimal conceptual change is to (1) use per-breath time index (`step` 0..79) instead of rounded `time_step` to avoid time rounding collisions, and (2) add one more lightweight within-breath feature (`u_in_lag2_bin`) to sharpen the median lookup without introducing any new model/training loop. We also keep the legitimate metric-aware handling of `u_out==1` (expiration not scored) by setting those predictions to 0. Finally, we keep the existing directory-ensemble logic intact and only enhance the fallback path that is actually being used in your environment.'
- What this solution (achieved 1.82233) has done: 'We keep your current “median lookup baseline + hierarchical fallback + snap-to-nearest-pressure” core logic intact, but make two minimal changes that materially reduce MAE toward the target. First, we enforce the competition’s evaluation rule by predicting only for inspiratory timesteps (`u_out==0`) and leaving expiratory timesteps at a safe constant, and we also restrict the median tables to inspiratory rows so expiratory behavior can’t pollute the lookup. Second, we add a tiny amount of extra per-breath state (cumulative volume surrogate and a 1-step pressure lag feature used only for lookup, not as a learned model) while keeping the same groupby-median approach and the same snapping post-process. These changes are localized to the fallback path (the path you’re actually using) and keep runtime within limits while improving alignment with the metric.'
- What this solution (achieved 1.81251) has done: 'Your current score (1.82233 MAE, lower is better) is still far above the target (0.1514), so the smallest safe way to move toward the target is to improve only the fallback baseline path you are actually using (since `../input/gb-rwbt-files` doesn’t exist). I keep the same core “groupby-median lookup + hierarchical fallback + snap to nearest allowed pressure” logic, but make the lookup slightly more physically/time-series consistent by adding a minimal within-breath lagged/rolling feature set computed from inputs only (no new model/training loop). In particular, I add binned `u_in` lag-3 and a short rolling mean of `u_in` (both for train/test) and use them only in the most specific median table keys, falling back exactly as before when missing. This typically reduces MAE noticeably for this competition while preserving your approach and producing the same `submission.csv` format.'
- What this solution (achieved 1.81147) has done: 'We keep your exact core approach (hierarchical groupby-median lookup with teacher-forced `p_lag1_bin` during inference, then snap to nearest allowed pressure) and only strengthen the fallback baseline that is actually used. The smallest high-impact improvement toward the target MAE is to add one more very lightweight, input-only within-breath state feature: a binned rolling std of `u_in` over the last 5 steps, used only in the most specific median-table keys and otherwise falling back exactly as before. This helps distinguish similar `u_in`/lag patterns that have different dynamics without changing your overall pipeline. We also keep the inspiratory-only table restriction and keep expiratory predictions at 0 (unscored), preserving evaluation semantics.'
- What this solution (achieved 1.81065) has done: 'Your current MAE (1.81147, lower is better) is still far above the target (0.15139), so we should make a small, legitimate improvement in the only path you actually use: the fallback baseline. Keeping your exact core logic (hierarchical groupby-median lookup with teacher-forced `p_lag1_bin`, then snap-to-nearest allowed pressure), I add one more tiny input-only within-breath state feature (`u_in_roll5_mean_bin`) and use it only in the most specific median table key to sharpen matching while preserving the same fallback hierarchy. I also keep the inspiratory-only table restriction and keep expiratory predictions at 0 (unscored), unchanged. This is a minimal feature/key extension (no new model, no new training loop) that typically nudges MAE downward toward your target.'
- What this solution (achieved 1.80772) has done: 'We keep your exact core fallback approach (hierarchical groupby-median lookup with teacher-forced `p_lag1_bin`, then snap to nearest allowed pressure) but make two small, metric-aligned upgrades that usually reduce MAE without changing the “model type.” First, we add a single additional within-breath input-only state feature (`u_in_ema_bin`) to slightly sharpen the most-specific median key while preserving the same fallback hierarchy. Second, we change the expiratory (`u_out==1`) predictions from a hard 0 to a constant median-by-(R,C) value computed from inspiratory data (still unscored, but can reduce any edge-case penalty and is harmless), while keeping inspiratory-only median tables intact. These are minimal changes localized to the fallback path you actually use (since `../input/gb-rwbt-files` is absent), and they preserve submission format and runtime constraints.'
- What this solution (achieved 1.80772) has done: 'Your current MAE (1.80772, lower is better) is still far above the target (0.15139), so we should make a small improvement in the fallback baseline path you are actually using (since `../input/gb-rwbt-files` is absent). Keeping the exact same core approach (hierarchical groupby-median lookup with teacher-forced `p_lag1_bin`, then snap-to-nearest allowed pressure), the most impactful minimal fix is to add a single extra physics-aligned, input-only state feature: the instantaneous flow surrogate `u_in * (1 - u_out)` (binned), used only in the most specific median-table keys. This helps separate inspiratory driving pressure patterns without changing the model type, training loop, or post-processing. Everything else (inspiratory-only tables, expiratory stable fill, and nearest-pressure snapping) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.80772) has done: 'Your current MAE (1.80772, lower is better) is still far above the target (0.15139), so we should make a small, low-risk improvement in the fallback baseline path that your environment actually uses (since `../input/gb-rwbt-files` is missing). Keeping the exact same core logic (hierarchical groupby-median lookup with teacher-forced `p_lag1_bin`, inspiratory-only tables, then snap-to-nearest allowed pressure), I only add one extra input-only state feature: a binned change-in-flow surrogate (`u_flow_diff1_bin`). This is computed from existing `u_flow` and its lag-1, and it is used only in the most specific lookup key (with the exact same fallback hierarchy), which typically nudges MAE downward without changing the approach. All I/O paths and the submission format remain unchanged, and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved 1.80658) has done: 'Your current MAE (1.80772, lower-is-better) is still far above the target (0.15139), so we should make a small, low-risk improvement in the fallback path you are actually using (since `../input/gb-rwbt-files` is missing). Keeping the exact same core logic (hierarchical groupby-median lookup with teacher-forced `p_lag1_bin`, inspiratory-only median tables, then snap-to-nearest allowed pressure), I add one additional lightweight, input-only within-breath state feature: a binned interaction `u_in * time_step` (a minimal proxy for “delivered effort so far” that often correlates with pressure). This new feature is used only in the most specific median-table key, preserving the fallback hierarchy and evaluation semantics while typically nudging MAE downward. All I/O paths remain unchanged and the script still writes a valid `submission.csv` end-to-end.'

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
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def _fallback_baseline_submission():
    """
    Keep core logic: inspiratory-only median lookup with hierarchical fallback + teacher-forced p_lag1_bin,
    then snap to nearest allowed pressure.

    Minimal score-improvement change toward target:
    - Add one tiny physics/time-series-aligned input-only feature: binned interaction `u_in * time_step`
      used ONLY in the most specific lookup key to sharpen matches while preserving the same hierarchy.
    """
    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

    def add_features_train(df):
        base_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
        out = df[base_cols].copy()

        out["step"] = out.groupby("breath_id", sort=False).cumcount().astype(np.int16)

        out["u_in_bin"] = np.round(out["u_in"]).astype(np.int16)

        out["u_in_lag1"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )
        out["u_in_lag1_bin"] = np.round(out["u_in_lag1"]).astype(np.int16)

        out["u_in_lag2"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(2).fillna(0.0)
        )
        out["u_in_lag2_bin"] = np.round(out["u_in_lag2"]).astype(np.int16)

        out["u_in_lag3"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(3).fillna(0.0)
        )
        out["u_in_lag3_bin"] = np.round(out["u_in_lag3"]).astype(np.int16)

        u_in_diff1 = out["u_in"] - out["u_in_lag1"]
        out["u_in_diff1_bin"] = np.round(u_in_diff1).astype(np.int16)

        dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        out["u_in_cum"] = (
            (out["u_in"] * dt).groupby(out["breath_id"], sort=False).cumsum()
        )
        out["u_in_cum_bin"] = np.round(out["u_in_cum"] / 10.0).astype(np.int16)

        roll3 = (
            out.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["u_in_roll3_bin"] = np.round(roll3).astype(np.int16)

        roll5_std = (
            out.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=5, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        out["u_in_roll5_std_bin"] = np.round(roll5_std * 2.0).astype(np.int16)

        roll5_mean = (
            out.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["u_in_roll5_mean_bin"] = np.round(roll5_mean).astype(np.int16)

        ema = (
            out.groupby("breath_id", sort=False)["u_in"]
            .ewm(alpha=0.3, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["u_in_ema_bin"] = np.round(ema).astype(np.int16)

        out["u_flow"] = out["u_in"] * (1.0 - out["u_out"].astype(np.float32))
        out["u_flow_bin"] = np.round(out["u_flow"]).astype(np.int16)

        out["u_flow_lag1"] = (
            out.groupby("breath_id", sort=False)["u_flow"].shift(1).fillna(0.0)
        )
        out["u_flow_diff1_bin"] = np.round(out["u_flow"] - out["u_flow_lag1"]).astype(
            np.int16
        )

        out["u_in_time"] = out["u_in"] * out["time_step"]
        out["u_in_time_bin"] = np.round(out["u_in_time"] / 2.0).astype(np.int16)

        out["p_lag1"] = (
            out.groupby("breath_id", sort=False)["pressure"].shift(1).fillna(0.0)
        )
        out["p_lag1_bin"] = np.round(out["p_lag1"] / 0.5).astype(np.int16)

        return out

    def add_features_test(df):
        base_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
        out = df[base_cols].copy()

        out["step"] = out.groupby("breath_id", sort=False).cumcount().astype(np.int16)

        out["u_in_bin"] = np.round(out["u_in"]).astype(np.int16)

        out["u_in_lag1"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )
        out["u_in_lag1_bin"] = np.round(out["u_in_lag1"]).astype(np.int16)

        out["u_in_lag2"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(2).fillna(0.0)
        )
        out["u_in_lag2_bin"] = np.round(out["u_in_lag2"]).astype(np.int16)

        out["u_in_lag3"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(3).fillna(0.0)
        )
        out["u_in_lag3_bin"] = np.round(out["u_in_lag3"]).astype(np.int16)

        u_in_diff1 = out["u_in"] - out["u_in_lag1"]
        out["u_in_diff1_bin"] = np.round(u_in_diff1).astype(np.int16)

        dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        out["u_in_cum"] = (
            (out["u_in"] * dt).groupby(out["breath_id"], sort=False).cumsum()
        )
        out["u_in_cum_bin"] = np.round(out["u_in_cum"] / 10.0).astype(np.int16)

        roll3 = (
            out.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["u_in_roll3_bin"] = np.round(roll3).astype(np.int16)

        roll5_std = (
            out.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=5, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        out["u_in_roll5_std_bin"] = np.round(roll5_std * 2.0).astype(np.int16)

        roll5_mean = (
            out.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["u_in_roll5_mean_bin"] = np.round(roll5_mean).astype(np.int16)

        ema = (
            out.groupby("breath_id", sort=False)["u_in"]
            .ewm(alpha=0.3, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out["u_in_ema_bin"] = np.round(ema).astype(np.int16)

        out["u_flow"] = out["u_in"] * (1.0 - out["u_out"].astype(np.float32))
        out["u_flow_bin"] = np.round(out["u_flow"]).astype(np.int16)

        out["u_flow_lag1"] = (
            out.groupby("breath_id", sort=False)["u_flow"].shift(1).fillna(0.0)
        )
        out["u_flow_diff1_bin"] = np.round(out["u_flow"] - out["u_flow_lag1"]).astype(
            np.int16
        )

        out["u_in_time"] = out["u_in"] * out["time_step"]
        out["u_in_time_bin"] = np.round(out["u_in_time"] / 2.0).astype(np.int16)

        return out

    train_feat = add_features_train(df_train)
    test_feat = add_features_test(df_test)

    train_insp = train_feat[train_feat["u_out"] == 0].copy()

    keys_rc = ["R", "C"]
    med_rc = train_insp.groupby(keys_rc, sort=False)["pressure"].median()
    global_med = float(train_insp["pressure"].median())

    keys_full = [
        "R",
        "C",
        "step",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_in_lag3_bin",
        "u_in_diff1_bin",
        "u_in_roll3_bin",
        "u_in_roll5_std_bin",
        "u_in_roll5_mean_bin",
        "u_in_ema_bin",
        "u_flow_bin",
        "u_flow_diff1_bin",
        "u_in_time_bin",  # Change: only added to most specific key
        "u_in_cum_bin",
        "p_lag1_bin",
    ]
    med_full = train_insp.groupby(keys_full, sort=False)["pressure"].median()

    keys_mid = [
        "R",
        "C",
        "step",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_in_cum_bin",
        "p_lag1_bin",
    ]
    med_mid = train_insp.groupby(keys_mid, sort=False)["pressure"].median()

    keys_simple = ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_bin"]
    med_simple = train_insp.groupby(keys_simple, sort=False)["pressure"].median()

    keys_step = ["R", "C", "step", "u_in_cum_bin"]
    med_step = train_insp.groupby(keys_step, sort=False)["pressure"].median()

    pred = np.empty(len(test_feat), dtype=np.float64)

    r_arr = test_feat["R"].to_numpy()
    c_arr = test_feat["C"].to_numpy()
    uout_arr = test_feat["u_out"].to_numpy()
    exp_mask = uout_arr == 1

    if exp_mask.any():
        exp_idx = np.flatnonzero(exp_mask)
        for i in exp_idx:
            v = med_rc.get((int(r_arr[i]), int(c_arr[i])), global_med)
            pred[i] = float(v)

    insp_mask = ~exp_mask
    insp_idx = np.flatnonzero(insp_mask)

    prev_p_bin = {}  # breath_id -> p_lag1_bin (int)

    for i in insp_idx:
        r = int(test_feat.at[i, "R"])
        c = int(test_feat.at[i, "C"])
        step = int(test_feat.at[i, "step"])
        u_in_bin = int(test_feat.at[i, "u_in_bin"])
        u1 = int(test_feat.at[i, "u_in_lag1_bin"])
        u2 = int(test_feat.at[i, "u_in_lag2_bin"])
        u3 = int(test_feat.at[i, "u_in_lag3_bin"])
        d1 = int(test_feat.at[i, "u_in_diff1_bin"])
        ur3 = int(test_feat.at[i, "u_in_roll3_bin"])
        urs5 = int(test_feat.at[i, "u_in_roll5_std_bin"])
        urm5 = int(test_feat.at[i, "u_in_roll5_mean_bin"])
        uema = int(test_feat.at[i, "u_in_ema_bin"])
        uflow = int(test_feat.at[i, "u_flow_bin"])
        uflowd1 = int(test_feat.at[i, "u_flow_diff1_bin"])
        uit = int(test_feat.at[i, "u_in_time_bin"])  # Change
        uc = int(test_feat.at[i, "u_in_cum_bin"])
        bid = int(test_feat.at[i, "breath_id"])

        pl1 = prev_p_bin.get(bid, 0)

        k_full = (
            r,
            c,
            step,
            u_in_bin,
            u1,
            u2,
            u3,
            d1,
            ur3,
            urs5,
            urm5,
            uema,
            uflow,
            uflowd1,
            uit,  # Change
            uc,
            pl1,
        )
        v = med_full.get(k_full, np.nan)

        if np.isnan(v):
            k_mid = (r, c, step, u_in_bin, u1, u2, uc, pl1)
            v = med_mid.get(k_mid, np.nan)

        if np.isnan(v):
            k_simple = (r, c, step, u_in_bin, u1, uc)
            v = med_simple.get(k_simple, np.nan)

        if np.isnan(v):
            k_step = (r, c, step, uc)
            v = med_step.get(k_step, np.nan)

        if np.isnan(v):
            v = med_rc.get((r, c), global_med)

        pred[i] = float(v)
        prev_p_bin[bid] = int(np.round(pred[i] / 0.5))

    sub["pressure"] = pred
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub = sub[["id", "pressure"]]
    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    if (dp is None) or (not os.path.isdir(dp)):
        print(
            f"Input directory not found: {dp}. Using fallback baseline to create submission.csv"
        )
        return _fallback_baseline_submission()

    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    file_count = len(l)
    if file_count == 0:
        print(
            f"No files found in: {dp}. Using fallback baseline to create submission.csv"
        )
        return _fallback_baseline_submission()

    loop_time = 500 // file_count
    if loop_time <= 0:
        loop_time = 1

    splits = file_count // 2
    if splits <= 0:
        splits = 1

    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    pred_list = []
    for j in range(loop_time):
        weight = []
        set_seed(j)
        for k in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            for k in range(len(weight)):
                weight[k] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for k in range(len(flist)):
            temp += flist[k] * weight[k]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output




## === cell 2
g("../input/gb-rwbt-files")
