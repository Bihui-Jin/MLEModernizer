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

0.1366625097300452

# 6. Current score

1.82488

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.31337) has done: 'I fix the runtime error by making the blending functions robust to missing/empty input directories (your `../input/gb-data-blending-recover` isn’t available here, so `input_list` is empty and `vstack` crashes). To still produce a valid submission end-to-end, I add a safe fallback that creates a simple baseline prediction from the provided data (group-mean by `(R,C,time_step,u_in,u_out)` from train, with a global-mean fallback), then snap predictions to the nearest valid pressure values as your original code intends. I also update the paths to the known-existing `/kaggle/input/ventilator-pressure-prediction/...` location to avoid relative path issues. This keeps the “core logic” of the notebook (blending + nearest-pressure postprocess) intact while ensuring a submission CSV is always written.'
- What this solution (achieved 4.21214) has done: 'Your current score (8.31 MAE) is far from the target (~0.137), and the main reason is the fallback baseline: averaging by `(R,C,time_step,u_in,u_out)` rarely matches in test due to continuous `u_in` and `time_step`, so it collapses to a global mean for most rows. I keep your blending/nearest-pressure core logic intact, but upgrade only the fallback to a stronger, still-simple group-mean that matches how the data is structured: compute per-breath cumulative `u_in` and use `(R,C,time_step,u_out,cum_u_in)` with rounding to increase hit-rate. I also ensure predictions are forced to `pressure=0` whenever `u_out==1` (expiratory phase), which aligns with the metric being inspiratory-only and is a common minimal post-process for this competition. These changes should move the MAE sharply downward toward the target without changing your overall approach.'
- What this solution (achieved 5.52868) has done: 'Your current MAE (4.21) is still far above the target (~0.137), so we should improve the fallback baseline (which is what’s actually being used because the blending directory is absent). I keep your “baseline via aggregation + nearest-pressure snapping + u_out==1 => 0” core logic intact, but make the aggregation much more aligned with the competition structure by grouping per-breath sequences using `breath_time` (time index within a breath) and discretized cumulative inhaled volume proxy (`cum_u_in`). This greatly increases key match-rate between train/test without changing the modeling approach. I also ensure the final submission preserves the exact `id` ordering from `sample_submission.csv`.'
- What this solution (achieved 3.20705) has done: 'Your current MAE (5.53, lower-is-better) is still far from the target (~0.137), and since the external blending directory isn’t present, your score is dominated by the fallback baseline. To move the score toward the target without changing the overall approach (aggregation baseline + u_out postprocess + nearest-pressure snapping), I make the fallback aggregation more “sequence-aligned” by using lagged `u_in/u_out` features and a stronger multi-stage backoff that matches train/test much more often. I also keep the inspiratory-only scoring behavior by forcing predictions to 0 when `u_out==1`, and preserve `sample_submission.csv` id ordering for a valid submission. These are minimal changes confined to `_baseline_submission()` and should improve MAE materially toward the target.'
- What this solution (achieved 4.03362) has done: 'Your MAE (3.207) is still far above the target (0.1367, lower-is-better), and since the external blending directory is missing, the score is dominated by `_baseline_submission()`. To move toward the target with minimal semantic change, I keep your aggregation+backoff+`u_out==1 => 0`+nearest-pressure snapping core logic, but improve the aggregation keys by adding a simple, competition-standard “volume proxy” (`u_in * delta_time` cumulative) and a couple of additional lags; this increases train/test key match-rate without introducing any new model or training loop. I also align discretization to be slightly less coarse (to reduce collisions) while still keeping exact same overall approach. Everything still runs end-to-end and writes `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.66045) has done: 'Your current score is far worse than the target (lower-is-better), and because the external blending directory is missing, the leaderboard score is dominated by `_baseline_submission()`. To move MAE substantially toward the target while preserving your core approach (aggregation/backoff + expiratory fix + nearest-pressure snapping), I make the aggregation more “breath-sequence aware” by adding a couple of minimal, competition-standard deterministic features: previous cumulative volume proxy (`cum_u_in_dt` lag), and a simple interaction (`u_in * (1-u_out)`) to better separate inspiratory dynamics. I also slightly adjust rounding to improve train/test key match-rate (too-fine rounding causes mismatches and backoff to global mean), while keeping the same multi-stage backoff logic and post-processing. The rest of the pipeline (including writing `submission.csv`) stays unchanged.'
- What this solution (achieved 1.75126) has done: 'Your current MAE (2.66045, lower-is-better) is still far above the target (~0.1367), and because the blending directory isn’t available the score is dominated by `_baseline_submission()`. I keep the exact same core approach (deterministic aggregation + multi-stage backoff + `u_out==1 => 0` + nearest-pressure snapping), but make the aggregation keys match train/test more often by adding a discretized “within-breath time-bin” and slightly coarsening the cumulative-volume rounding (too-fine rounding causes misses and global-mean fallback). I also compute and use a lightweight per-breath `u_in` rolling mean (no training loop/model) as an additional stable key that tends to repeat across breaths. These minimal changes are confined to `_baseline_submission()` and should move MAE downward toward the target without changing evaluation semantics or requiring extra files.'
- What this solution (achieved 1.89936) has done: 'Your current score (1.75126 MAE, lower-is-better) is still far above the target (0.13666), and since the external blending directory is absent the leaderboard score is dominated by `_baseline_submission()`. To move toward the target with minimal semantic change, I keep the same core approach (deterministic aggregation with multi-stage backoff + `u_out==1 => 0` + snap-to-valid-pressure), but improve key match-rate by adding a discretized per-step derivative feature (`du_in`) and using a slightly coarser, more stable time bin (based on the known 80 steps per breath). I also make the aggregation itself more robust by using the median (less sensitive to collisions/outliers) while staying within the same “group-aggregate then merge” logic. Everything still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.61821) has done: 'Your current MAE (1.89936, lower-is-better) is still far above the target (0.13666), so we should improve score by strengthening the deterministic aggregation fallback that is actually used when the blend directory is missing. I keep your exact core approach (multi-stage group aggregation/backoff + `u_out==1 => 0` + snap-to-nearest-valid-pressure) but make two minimal, high-impact adjustments: (1) add a more appropriate cumulative “inspired volume proxy” (`cum_u_in * dt`) alongside your existing `cum_u_in_dt` and (2) slightly coarsen the rounding of cumulative features to increase train/test key match-rate (reducing fallback-to-global-mean). This stays within your current semantics (no model/training loop changes) and should move MAE downward toward the target. The script still run end-to-end and write `submission.csv` with the correct `id,pressure` columns.'
- What this solution (achieved 1.89698) has done: 'Your current MAE (2.61821, lower-is-better) is still far above the target (0.13666), and since the external blending directory is absent, the leaderboard score is dominated by `_baseline_submission()`. To move the score downward toward the target without changing your overall “deterministic aggregation + multi-stage backoff + `u_out==1 => 0` + snap-to-nearest-valid-pressure” core logic, I make a minimal but impactful adjustment to discretization: align `breath_time` exactly to the known 80 steps per breath and use a slightly coarser, more stable rounding for cumulative volume proxies to improve train/test key match-rate (reducing fallbacks to global mean). I also add one extra very-light key (`u_in_lag2_r`) into the strongest key stage only (no new modeling), which tends to disambiguate sequences while staying within the same aggregation semantics. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.89698) has done: 'Your current MAE (1.89698, lower-is-better) is still far above the target (0.13666), so we should improve accuracy with the smallest possible change inside the existing deterministic aggregation/backoff baseline (since the external blend directory is missing). The biggest low-risk win is to align the “expiratory” post-process with the competition convention: during `u_out==1` the true pressure is typically carried forward (held) rather than forced to zero, so replacing `pressure=0` with “carry-forward last inspiratory prediction within each breath” usually reduces MAE substantially without changing the overall approach. I also keep your nearest-pressure snapping and multi-stage key backoff intact, only adjusting this one post-processing step. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.90609) has done: 'Your current score (1.89698 MAE, lower-is-better) is still far above the target (~0.1367), so we should improve the fallback baseline that’s being used (because the blend directory is missing). The smallest high-impact change that preserves your existing “deterministic aggregation + multi-stage backoff + carry-forward on u_out==1 + snap-to-nearest-valid-pressure” core logic is to fix a key mismatch issue: the feature `cum_u_in_x_dt_cum` currently depends on `dt` (which is ~0 for the first step), making it unstable and less transferable; replacing it with the standard cumulative volume proxy `cum_u_in_dt` (already computed) removes that instability without changing the approach. I also add one more lightweight, deterministic inspiratory-state indicator (`is_insp`) into only the strongest key stage to reduce collisions while keeping the same aggregation/backoff semantics. These changes are confined to `_baseline_submission()` feature engineering/keys and should move MAE downward toward the target without introducing any new model/training loop.'
- What this solution (achieved 1.82488) has done: 'We keep your core “deterministic aggregation baseline + multi-stage backoff + carry-forward on `u_out==1` + snap-to-nearest-valid-pressure” logic intact, but fix two small issues that are currently hurting the match-rate and therefore MAE. First, your `key_list` has an accidental duplicate stage early on; removing it reduces unnecessary overwriting/extra work and makes the backoff progression behave as intended. Second, we slightly coarsen the rounding of the cumulative volume proxies (`cum_u_in_dt*`) from 0.1 to 0.2 to improve train/test key matching (reducing the number of rows that fall all the way back to the global mean), which should move the MAE down toward the target without changing the modeling approach. The rest of the pipeline and submission writing remains unchanged.'
- What this solution (achieved 1.82488) has done: 'We keep your blending/aggregation/backoff approach unchanged and only adjust one deterministic feature discretization that’s currently causing excessive key mismatches (and thus too much fallback to the global mean). Specifically, we make `time_step_r` slightly coarser (3 decimals → 2 decimals) so the `"R,C,u_out,time_step_r"` backoff stage matches far more often between train and test without introducing any new modeling. This is a minimal change confined to `_baseline_submission()` feature engineering and should move MAE down from ~1.82 toward your 0.1367 target. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 1.82488) has done: 'We keep your existing deterministic aggregation + multi-stage backoff + carry-forward-on-`u_out==1` + nearest-pressure snapping core logic intact, but fix a key-alignment issue that is currently causing excessive fallback to the global mean (and thus high MAE). In this competition each breath has exactly 80 time steps, so using a stable within-breath index is more reliable than floating `time_step` rounding; we derive `breath_time` from `time_step` for both train/test and use it to replace the `"R,C,u_out,time_step_r"` backoff stage. This is a minimal change confined to feature creation and the weakest backoff stage, expected to improve train/test match-rate and move MAE down toward your target without changing any modeling/training approach. The script still run end-to-end and write `submission.csv` with `id,pressure`.'

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
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_csv(train_path)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
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


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 154
    splits = file_count // 2
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
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(sample_path)
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def _baseline_submission():
    """
    Keep core logic: deterministic aggregation baseline + multi-stage backoff + carry-forward for u_out==1 + snap.

    Change (toward target MAE, minimal semantic change):
    Use a stable within-breath step index derived from time_step (80 steps/breath) as a backoff key
    instead of float time_step rounding. This improves train/test key match-rate and reduces fallback-to-global-mean.
    """
    df_test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_path)

    for k in ["R", "C", "u_out", "breath_id"]:
        df_train[k] = df_train[k].astype(np.int64)
        df_test[k] = df_test[k].astype(np.int64)

    def add_features(df):
        df = df.copy()

        df["breath_time_ts"] = (
            (df["time_step"].astype(np.float64) / 0.033).round().astype(np.int16)
        )
        df["breath_time_ts"] = df["breath_time_ts"].clip(0, 79).astype(np.int16)

        df["breath_time"] = (
            df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
        )
        df["step_bin"] = (df["breath_time"] // 2).astype(np.int16)  # 40 bins

        df["u_in_lag1"] = (
            df.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )
        df["u_in_lag2"] = (
            df.groupby("breath_id", sort=False)["u_in"].shift(2).fillna(0.0)
        )
        df["u_out_lag1"] = (
            df.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0)
        )
        df["u_out_lag2"] = (
            df.groupby("breath_id", sort=False)["u_out"].shift(2).fillna(0)
        )

        df["du_in"] = (
            df["u_in"].astype(np.float64) - df["u_in_lag1"].astype(np.float64)
        ).astype(np.float64)

        dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        df["dt"] = dt.astype(np.float64)

        df["u_in_dt"] = (df["u_in"].astype(np.float64) * df["dt"]).astype(np.float64)
        df["cum_u_in_dt"] = df.groupby("breath_id", sort=False)["u_in_dt"].cumsum()
        df["cum_u_in_dt_lag1"] = (
            df.groupby("breath_id", sort=False)["cum_u_in_dt"].shift(1).fillna(0.0)
        )

        df["cum_u_in"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()

        df["u_in_insp"] = df["u_in"].astype(np.float64) * (
            1.0 - df["u_out"].astype(np.float64)
        )

        df["u_in_roll3"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float64)
        )

        df["is_insp"] = (1 - df["u_out"]).astype(np.int8)

        df["cum_u_in_dt_r"] = (df["cum_u_in_dt"] / 0.2).round(0) * 0.2
        df["cum_u_in_dt_lag1_r"] = (df["cum_u_in_dt_lag1"] / 0.2).round(0) * 0.2

        df["cum_u_in_dt2_r"] = ((df["cum_u_in_dt"] * df["cum_u_in_dt"]) / 0.2).round(
            0
        ) * 0.2
        df["cum_u_in_dt2_lag1_r"] = (
            (df["cum_u_in_dt_lag1"] * df["cum_u_in_dt_lag1"]) / 0.2
        ).round(0) * 0.2

        df["cum_u_in_r"] = df["cum_u_in"].round(1)
        df["u_in_r"] = df["u_in"].round(2)
        df["u_in_insp_r"] = df["u_in_insp"].round(2)
        df["u_in_lag1_r"] = df["u_in_lag1"].round(2)
        df["u_in_lag2_r"] = df["u_in_lag2"].round(2)
        df["u_in_roll3_r"] = df["u_in_roll3"].round(2)
        df["du_in_r"] = df["du_in"].round(1)

        df["time_step_r"] = df["time_step"].astype(np.float64).round(2)

        return df

    tr = add_features(df_train)
    te = add_features(df_test)

    global_mean = float(df_train["pressure"].mean())

    key_list = [
        [
            "R",
            "C",
            "u_out",
            "is_insp",
            "breath_time",
            "step_bin",
            "cum_u_in_dt2_r",
            "cum_u_in_dt2_lag1_r",
            "cum_u_in_dt_r",
            "cum_u_in_dt_lag1_r",
            "u_in_insp_r",
            "u_in_roll3_r",
            "u_in_lag1_r",
            "u_in_lag2_r",
            "du_in_r",
        ],
        [
            "R",
            "C",
            "u_out",
            "breath_time",
            "step_bin",
            "cum_u_in_dt_r",
            "u_in_insp_r",
            "u_in_roll3_r",
            "du_in_r",
        ],
        [
            "R",
            "C",
            "u_out",
            "breath_time",
            "step_bin",
            "cum_u_in_dt_r",
            "cum_u_in_dt_lag1_r",
            "u_in_insp_r",
            "u_in_roll3_r",
            "u_in_lag1_r",
            "du_in_r",
        ],
        ["R", "C", "u_out", "breath_time", "step_bin", "cum_u_in_dt_r", "du_in_r"],
        ["R", "C", "u_out", "breath_time", "cum_u_in_dt_r", "u_in_r", "u_in_lag1_r"],
        ["R", "C", "u_out", "breath_time", "cum_u_in_dt_r", "u_in_r"],
        ["R", "C", "u_out", "breath_time", "cum_u_in_dt_r"],
        ["R", "C", "u_out", "breath_time", "cum_u_in_r", "u_in_r", "u_in_lag1_r"],
        ["R", "C", "u_out", "breath_time", "cum_u_in_r", "u_in_r"],
        ["R", "C", "u_out", "breath_time", "cum_u_in_r"],
        ["R", "C", "u_out", "breath_time", "u_in_r"],
        ["R", "C", "u_out", "breath_time"],
        ["R", "C", "u_out", "breath_time_ts"],
        ["R", "C", "u_out"],
        ["R", "C"],
    ]

    pred = np.full(len(te), np.nan, dtype=np.float64)

    for keys in key_list:
        missing = np.isnan(pred)
        if not missing.any():
            break

        agg_by_key = tr.groupby(keys, sort=False)["pressure"].median().reset_index()

        filled = (
            te.loc[missing, keys]
            .merge(agg_by_key, on=keys, how="left")["pressure"]
            .to_numpy(dtype=np.float64)
        )
        pred[missing] = filled

    pred = np.where(np.isnan(pred), global_mean, pred)

    te_u_out = te["u_out"].to_numpy(dtype=np.int64)
    breath_ids = te["breath_id"].to_numpy(dtype=np.int64)

    pred_cf = pred.copy()
    last_val = global_mean
    last_breath = breath_ids[0]
    for i in range(len(pred_cf)):
        if breath_ids[i] != last_breath:
            last_breath = breath_ids[i]
            last_val = pred_cf[i]
        if te_u_out[i] == 0:
            last_val = pred_cf[i]
        else:
            pred_cf[i] = last_val
    pred = pred_cf

    pred = np.array([find_nearest(p) for p in pred], dtype=np.float64)

    out = sub[["id"]].copy()
    out["pressure"] = pred
    out.to_csv("submission.csv", index=False)
    return out


def avg(dp):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    if len(input_list) == 0:
        return _baseline_submission()

    arrays = []
    for p in input_list:
        dfp = pd.read_csv(p)
        if "pressure" not in dfp.columns:
            continue
        arr = dfp["pressure"].to_numpy().ravel()
        arrays.append(arr)

    if len(arrays) == 0:
        return _baseline_submission()

    output = pd.read_csv(sample_path)

    n = len(output)
    arrays = [
        a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)), mode="edge")
        for a in arrays
    ]

    output.pressure = np.median(np.vstack(arrays), axis=0)

    df_test_meta = pd.read_csv(test_path, usecols=["breath_id", "u_out"])
    b = df_test_meta["breath_id"].to_numpy(dtype=np.int64)
    uo = df_test_meta["u_out"].to_numpy(dtype=np.int64)
    p = output["pressure"].to_numpy(dtype=np.float64)

    last_val = float(p[0])
    last_breath = int(b[0])
    for i in range(len(p)):
        if b[i] != last_breath:
            last_breath = int(b[i])
            last_val = float(p[i])
        if uo[i] == 0:
            last_val = float(p[i])
        else:
            p[i] = last_val
    output["pressure"] = p

    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)  # Kaggle-compatible default filename
    return output




## === cell 2
avg("../input/gb-data-blending-recover")
