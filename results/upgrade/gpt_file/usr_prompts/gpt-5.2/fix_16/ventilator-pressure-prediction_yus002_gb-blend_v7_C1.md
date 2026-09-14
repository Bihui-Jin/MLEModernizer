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

0.3955626102132885

# 6. Current score

1.51411

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.445) has done: 'Your script doesn’t yield a Kaggle score because it never creates a valid submission in this environment: it tries to blend files from a folder (`../input/gb-blending`) that isn’t provided, and it also writes a filename with spaces that Kaggle won’t automatically pick up. To minimally fix this while keeping your “blending/weighting” core logic intact, I add a safe fallback: if no blend files are found, create a baseline prediction using the **mean training pressure** (legitimate, fast, and typically far better than all-zeros), aligned by `id` from `sample_submission.csv`. I also make sure the output is always written as `submission.csv` with the exact `id,pressure` schema so you can submit and get a score. These changes are directly aimed at producing a valid submission and improving MAE from “not yielded / effectively 0” toward your target.'
- What this solution (achieved 6.31568) has done: 'Your current score (8.445 MAE, lower is better) is far from the target (~0.396), and the main issue is that when the blending folder is missing you fall back to a constant mean-pressure prediction, which is a very weak baseline. To move the score much closer to the target while preserving your overall “generate prediction then write submission.csv” flow, I keep your blending logic intact but upgrade only the fallback to a fast, legitimate heuristic that uses the test inputs (R, C, u_in, u_out) and matches known discrete pressure levels from training (a common improvement for this competition). Specifically, I build a simple lookup: for each (R, C, u_out) and binned u_in value, predict the median training pressure, and then snap predictions to the nearest valid pressure value seen in train. This keeps runtime low, requires no new packages, and should reduce MAE substantially compared to a constant baseline.'
- What this solution (achieved 3.77562) has done: 'Your current MAE (6.31568, lower is better) is far above the target (~0.396), so we should improve the fallback path (used when `../input/gb-blending` is missing) while keeping your blending core logic intact. The largest issue in the fallback is that it ignores the per-breath time-series structure and `time_step`, which are crucial for this competition. I minimally upgrade the fallback to a per-(R,C,time_step,u_out,u_in_bin) grouped-median lookup (with safe hierarchical backoffs) and then snap predictions to the nearest valid discrete pressure from training, which typically yields a large MAE drop without changing any model/training approach. I also fix the output alignment bug in the fallback (it currently merges `sub` with `test2[["id"]]` and can misalign), ensuring predictions are exactly in `sample_submission.csv` row order.'
- What this solution (achieved 1.6622) has done: 'Your current MAE (3.77562, lower is better) is still far above the target (~0.3956), so we should further strengthen only the fallback path (since `../input/gb-blending` is missing) while keeping your overall “blend if available, otherwise fallback then write submission.csv” logic intact. The main upgrade is to use the true per-breath sequence index (0–79) instead of time_step binning, and to add simple within-breath cumulative features (`u_in_cum`, `u_out_cum`) that are legitimate and strongly predictive for this competition. We keep the same grouped-median lookup + hierarchical backoff approach, then snap to the nearest valid discrete pressure from training, and we preserve exact `sample_submission.csv` id order for a valid submission.'
- What this solution (achieved 1.62549) has done: 'We’re far above the target MAE (1.6622 vs 0.3956, lower is better), and since the blending folder is missing the score is entirely driven by the fallback heuristic. I keep your overall “blend if available, otherwise fallback then write submission.csv” logic identical, but minimally strengthen the fallback by adding one more legitimate per-breath state feature (`u_in_diff`) and using it only in the most specific lookup level (with safe hierarchical backoffs unchanged). This typically improves MAE in this competition because pressure dynamics depend on changes in flow/valve control, not just absolute values. I also keep the exact `id` order from `sample_submission.csv` and still snap predictions to valid discrete train pressures.'
- What this solution (achieved 1.67261) has done: 'We keep your blending logic exactly as-is and only strengthen the fallback heuristic (which is what’s driving your current 1.62549 MAE because `../input/gb-blending` is missing). The smallest high-impact fix for this competition is to (1) explicitly set predictions to **0 during expiratory phase** (`u_out==1`) since it’s not scored, and (2) add a very lightweight per-breath dynamic feature (`u_in_lag1_bin`) used only in the most specific grouped-median lookup, with the same hierarchical backoffs and the same snapping to valid discrete pressures. These changes are legitimate, preserve evaluation semantics, and typically reduce MAE materially toward your ~0.3956 target without changing any training/modeling approach. The script still always write a valid `submission.csv` with `id,pressure` in exact `sample_submission.csv` order.'
- What this solution (achieved 1.62549) has done: 'Your current MAE (1.67261, lower is better) is still far above the target (0.39556), and since `../input/gb-blending` is missing your score is entirely determined by the fallback lookup heuristic. I keep the same “group-median with hierarchical backoff + snap-to-discrete + write submission.csv” core logic, but make one minimal, high-signal improvement: replace the lag-1 `u_in` bin with a per-breath **time index since last u_out==1** (“inspiratory clock”), which better matches how pressure evolves during inspiration and reduces sparsity versus using raw step alone. I also stop forcing predictions to 0 for `u_out==1` (it’s not scored, but forcing 0 can slightly harm snapped calibration in transitions), and instead keep the learned median mapping there (still snapped to valid pressure levels), which typically improves overall MAE. All paths still produce a valid `submission.csv` with exact `id,pressure` in sample_submission order.'
- What this solution (achieved 1.61362) has done: 'Your current MAE (1.62549) is still far above the target (0.39556, lower is better), and since the `../input/gb-blending` folder is missing, only the fallback heuristic affects the score. I keep the same core “grouped median lookup with hierarchical backoff + snap to discrete train pressures” approach, but make a minimal upgrade to reduce lookup sparsity and better capture the inspiratory dynamics by adding a tiny, physical per-breath feature: the **previous predicted/observed pressure level** via a `pressure_lag1_bin` key (train uses true lagged pressure; test uses a first-pass prediction lag). This is implemented as a two-pass lookup (first pass same as now, second pass refines with lag), without changing the overall evaluation semantics or adding any new model/training. I also keep the output aligned to `sample_submission.csv` order and always write `submission.csv`.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.61362, lower is better) is still far above the target (0.39556), and since `../input/gb-blending` is missing the score is entirely determined by the fallback heuristic. To move closer without changing the overall “grouped median lookup with hierarchical backoff + snap-to-discrete + optional blending” core logic, I add one minimal, competition-relevant state feature: **time since inspiration started** (reset when `u_out` flips to 1, then counts within the next inspiration) alongside the existing `insp_step`, and use it only in the most specific (top) lookup key to reduce collisions and better match the pressure trajectory. I also make the lag-1 binning consistent by using the *nearest* discrete pressure index (not just `searchsorted`), which improves the lag-conditioned grouping without changing the discrete snapping behavior. All outputs remain aligned to `sample_submission.csv` order and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.51411, lower is better) is still far above the target (0.39556), so we should improve only the fallback path (since `../input/gb-blending` is missing and blending never runs) while keeping the same grouped-median + hierarchical backoff + snap-to-discrete core logic. The most leverage with minimal logic change in this competition is to ensure we never train/learn mappings from expiratory (`u_out==1`) rows at all, because the metric ignores them and they add noise to the lookup tables; we still output valid pressures for those rows, but they won’t pollute inspiratory dynamics. Concretely: build all group-median tables using only `u_out==0` rows, and when predicting, use those tables for `u_out==0` while falling back to a simple (R,C,u_out) median for `u_out==1`. This is a small, safe change that typically reduces MAE materially toward your target without changing architecture/training loops or the “lookup + backoff + snap” evaluation semantics, and it still writes a valid `submission.csv` in exact sample submission order.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.51411, lower-is-better) is still far above the target (~0.3956), and since `../input/gb-blending` is absent the score is entirely determined by `_fallback_predict_pressure`. To move the score closer with minimal disruption, I keep the same “grouped-median lookup + hierarchical backoff + lag-refinement + snap-to-discrete” core logic, but reduce noise by learning all inspiratory lookup tables strictly from inspiratory rows (`u_out==0`) and also using inspiratory-only medians for the coarsest `(R,C,u_out)` fallback when predicting inspiratory rows. Expiratory rows still get valid predictions via an expiratory-only median table (not scored, but keeps output sane). This is a small, targeted change that typically reduces MAE without altering your overall approach or I/O.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.51411, lower is better) is still far above the target (~0.3956), and since `../input/gb-blending` is missing the score is entirely determined by the fallback lookup heuristic. To move closer while preserving your exact core “grouped-median lookup + hierarchical backoff + lag-refinement + snap-to-discrete” approach, I make one minimal but high-impact correction: compute `insp_step` and `insp_time` using the *true inspiration start (first u_out==0 after u_out==1)* rather than “since last u_out==1” (which wrongly makes the first inspiratory row `insp_step==1` and uses a moving time origin). This better aligns train/test keys with the scored inspiratory dynamics and typically reduces MAE without changing the modeling approach. I also keep your inspiratory-only table training (no expiratory noise) and ensure output remains exactly aligned to `sample_submission.csv` and written as `submission.csv`.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.51411, lower-is-better) is still far above the target (~0.3956), and since `../input/gb-blending` is missing the score is driven entirely by the fallback lookup. To move closer while preserving the exact “grouped-median lookup + hierarchical backoff + lag-refinement + snap-to-discrete” core logic, I make one minimal, high-signal fix: build `u_in_cum` and `u_out_cum` in a way that resets at the true inspiration start (first `u_out==0` after `u_out==1`), so your lookup keys represent inspiratory progression rather than mixing in expiratory history. This reduces key noise/sparsity and better aligns train/test dynamics during the scored phase without changing model class, loss, or overall workflow. The script still writes a valid `submission.csv` with `id,pressure` in exact `sample_submission.csv` order.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.51411, lower-is-better) is still far above the target (~0.3956), and since `../input/gb-blending` is missing the score is entirely determined by the fallback lookup heuristic. To move closer while preserving your exact “grouped-median lookup + hierarchical backoff + lag-refinement + snap-to-discrete” core logic, I make one minimal but high-signal change: learn and apply the lag-refinement (`pressure_lag1_bin`) only during inspiration (`u_out==0`), so expiratory rows (not scored) don’t distort the per-breath lag dynamics and the lag table becomes denser/cleaner for the scored phase. Concretely, I build the lagged prediction sequentially within each breath only across inspiratory segments (resetting lag at inspiration starts), and I apply the lag-conditioned table only to inspiratory rows; expiratory rows keep the first-pass prediction. This is a targeted noise-reduction change that typically lowers inspiratory MAE without changing your overall approach, and the script still writes a valid `submission.csv` in exact `sample_submission.csv` order.'
- What this solution (achieved 1.51411) has done: 'Your current MAE (1.51411, lower-is-better) is still far above the target (~0.3956), and since `../input/gb-blending` is missing your score is entirely determined by the fallback heuristic. To move closer while preserving the exact same “grouped-median lookup + hierarchical backoff + lag-refinement + snap-to-discrete” core logic, I make one minimal correction that directly affects the scored phase: ensure the lag feature in training (`pressure_lag1`) is computed **within inspiratory segments** (reset at each inspiration start) instead of simply shifting within the whole breath, which currently lets expiratory/inspiration transitions pollute the lag table. I also make the test-side lag initialization reset at the true inspiration start (first `u_out==0` after `u_out==1`) so train/test lag keys match. This is a small, targeted noise reduction and should lower inspiratory MAE without changing your overall approach or I/O, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Weighted combine for a list of filepaths.
    Keeps original core behavior, but becomes robust if filenames do not contain the expected score pattern.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 0
        l.append(public_lb_score)
        preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())

    if len(preds) == 1:
        return preds[0]
    else:
        weight1 = 0.8
        weight2 = 0.2
        return preds[0] * weight1 + preds[1] * weight2


def _fallback_predict_pressure(train_path, test_path, sample_path):
    """
    Change rationale (score toward target, minimal & preserves core approach):
    - The metric scores only inspiration (u_out==0). Our lag-refinement is already learned/applied only on inspiratory rows,
      but the *training lag itself* was computed by shifting within the whole breath, which mixes the u_out==1 -> u_out==0
      boundary and pollutes the lag table.
    - Minimal change: compute training pressure_lag1 within inspiratory segments (reset at true inspiration starts), matching
      how we sequentially build lag on test. This aligns train/test keys and reduces noise in the lag-conditioned lookup.
    """
    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]
    )
    sub = pd.read_csv(sample_path, usecols=["id"])

    pressure_values = np.sort(train["pressure"].unique())

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort")

    train["step"] = train.groupby("breath_id").cumcount().astype(np.int16)
    test["step"] = test.groupby("breath_id").cumcount().astype(np.int16)

    train["u_in_bin"] = np.rint(train["u_in"].to_numpy()).astype(np.int16)
    test["u_in_bin"] = np.rint(test["u_in"].to_numpy()).astype(np.int16)

    train["u_in_diff"] = (
        train.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float64)
    )
    test["u_in_diff"] = (
        test.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float64)
    )
    train["u_in_diff_bin"] = np.rint(train["u_in_diff"].to_numpy() * 2.0).astype(
        np.int16
    )
    test["u_in_diff_bin"] = np.rint(test["u_in_diff"].to_numpy() * 2.0).astype(np.int16)

    def _insp_step_from_start(u_out_series: pd.Series) -> pd.Series:
        u = u_out_series.to_numpy(dtype=np.int16)
        out = np.zeros_like(u, dtype=np.int16)
        c = 0
        for i in range(u.shape[0]):
            if u[i] == 1:
                c = 0
                out[i] = 0
            else:
                out[i] = c
                c += 1
        return pd.Series(out, index=u_out_series.index)

    train["insp_step"] = (
        train.groupby("breath_id")["u_out"]
        .transform(_insp_step_from_start)
        .astype(np.int16)
    )
    test["insp_step"] = (
        test.groupby("breath_id")["u_out"]
        .transform(_insp_step_from_start)
        .astype(np.int16)
    )

    def _insp_time_from_start(df: pd.DataFrame) -> pd.Series:
        uo = df["u_out"].to_numpy(dtype=np.int16)
        ts = df["time_step"].to_numpy(dtype=np.float64)
        out = np.zeros_like(ts, dtype=np.float64)
        t0 = 0.0
        started = False
        for i in range(ts.shape[0]):
            if uo[i] == 1:
                started = False
                out[i] = 0.0
            else:
                if not started:
                    t0 = ts[i]
                    started = True
                out[i] = ts[i] - t0
        return pd.Series(out, index=df.index)

    train["insp_time"] = (
        train.groupby("breath_id", sort=False)
        .apply(_insp_time_from_start, include_groups=False)
        .reset_index(level=0, drop=True)
        .astype(np.float64)
    )
    test["insp_time"] = (
        test.groupby("breath_id", sort=False)
        .apply(_insp_time_from_start, include_groups=False)
        .reset_index(level=0, drop=True)
        .astype(np.float64)
    )
    train["insp_time_bin"] = np.rint(train["insp_time"].to_numpy() * 100.0).astype(
        np.int16
    )
    test["insp_time_bin"] = np.rint(test["insp_time"].to_numpy() * 100.0).astype(
        np.int16
    )

    train["_insp_mask"] = (train["u_out"].to_numpy(dtype=np.int16) == 0).astype(
        np.int16
    )
    test["_insp_mask"] = (test["u_out"].to_numpy(dtype=np.int16) == 0).astype(np.int16)

    train["u_in_cum"] = train["u_in"].to_numpy(dtype=np.float64) * train[
        "_insp_mask"
    ].to_numpy(dtype=np.int16)
    test["u_in_cum"] = test["u_in"].to_numpy(dtype=np.float64) * test[
        "_insp_mask"
    ].to_numpy(dtype=np.int16)
    train["u_out_cum"] = train["u_out"].to_numpy(dtype=np.int16) * train[
        "_insp_mask"
    ].to_numpy(dtype=np.int16)
    test["u_out_cum"] = test["u_out"].to_numpy(dtype=np.int16) * test[
        "_insp_mask"
    ].to_numpy(dtype=np.int16)

    train["_insp_group"] = (
        (train["insp_step"] == 0).groupby(train["breath_id"]).cumsum().astype(np.int16)
    )
    test["_insp_group"] = (
        (test["insp_step"] == 0).groupby(test["breath_id"]).cumsum().astype(np.int16)
    )

    train["u_in_cum"] = (
        pd.Series(train["u_in_cum"])
        .groupby([train["breath_id"], train["_insp_group"]])
        .cumsum()
        .astype(np.float64)
    )
    test["u_in_cum"] = (
        pd.Series(test["u_in_cum"])
        .groupby([test["breath_id"], test["_insp_group"]])
        .cumsum()
        .astype(np.float64)
    )

    train["u_out_cum"] = (
        pd.Series(train["u_out_cum"])
        .groupby([train["breath_id"], train["_insp_group"]])
        .cumsum()
        .astype(np.int16)
    )
    test["u_out_cum"] = (
        pd.Series(test["u_out_cum"])
        .groupby([test["breath_id"], test["_insp_group"]])
        .cumsum()
        .astype(np.int16)
    )

    train.drop(columns=["_insp_mask", "_insp_group"], inplace=True)
    test.drop(columns=["_insp_mask", "_insp_group"], inplace=True)

    train["u_in_cum_bin"] = np.rint(train["u_in_cum"].to_numpy() / 10.0).astype(
        np.int16
    )
    test["u_in_cum_bin"] = np.rint(test["u_in_cum"].to_numpy() / 10.0).astype(np.int16)

    global_med = float(train["pressure"].median())

    train_insp = train.loc[train["u_out"] == 0].copy()
    train_exp = train.loc[train["u_out"] == 1].copy()

    rc_uout_med_insp = (
        train_insp.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
    )
    rc_uout_med_exp = (
        train_exp.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .median()
        .reset_index()
    )

    def _snap_to_levels(preds_float: np.ndarray) -> np.ndarray:
        preds = preds_float.astype(np.float64, copy=False)
        idx = np.searchsorted(pressure_values, preds, side="left")
        idx = np.clip(idx, 0, len(pressure_values) - 1)
        left_idx = np.clip(idx - 1, 0, len(pressure_values) - 1)
        right = pressure_values[idx]
        left = pressure_values[left_idx]
        choose_left = np.abs(preds - left) <= np.abs(preds - right)
        return np.where(choose_left, left, right)

    def _nearest_level_index(preds_float: np.ndarray) -> np.ndarray:
        preds = preds_float.astype(np.float64, copy=False)
        idx = np.searchsorted(pressure_values, preds, side="left")
        idx = np.clip(idx, 0, len(pressure_values) - 1)
        left_idx = np.clip(idx - 1, 0, len(pressure_values) - 1)
        right = pressure_values[idx]
        left = pressure_values[left_idx]
        choose_left = np.abs(preds - left) <= np.abs(preds - right)
        return np.where(choose_left, left_idx, idx).astype(np.int16)

    def _predict_without_lag(
        train_df_insp: pd.DataFrame, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        grp = (
            train_df_insp.groupby(
                [
                    "R",
                    "C",
                    "insp_step",
                    "insp_time_bin",
                    "u_out",
                    "u_in_bin",
                    "u_in_cum_bin",
                    "u_out_cum",
                    "u_in_diff_bin",
                ],
                sort=False,
            )["pressure"]
            .median()
            .reset_index()
        )

        test2 = test_df.merge(
            grp,
            on=[
                "R",
                "C",
                "insp_step",
                "insp_time_bin",
                "u_out",
                "u_in_bin",
                "u_in_cum_bin",
                "u_out_cum",
                "u_in_diff_bin",
            ],
            how="left",
        )

        if test2["pressure"].isna().any():
            grp_b1 = (
                train_df_insp.groupby(
                    ["R", "C", "insp_step", "u_out", "u_in_cum_bin", "u_out_cum"],
                    sort=False,
                )["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": "pressure_b1"})
            )
            test2 = test2.merge(
                grp_b1,
                on=["R", "C", "insp_step", "u_out", "u_in_cum_bin", "u_out_cum"],
                how="left",
            )
            test2["pressure"] = test2["pressure"].fillna(test2["pressure_b1"])
            test2.drop(columns=["pressure_b1"], inplace=True)

        if test2["pressure"].isna().any():
            grp_b2 = (
                train_df_insp.groupby(
                    ["R", "C", "insp_step", "u_out", "u_in_bin"], sort=False
                )["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": "pressure_b2"})
            )
            test2 = test2.merge(
                grp_b2, on=["R", "C", "insp_step", "u_out", "u_in_bin"], how="left"
            )
            test2["pressure"] = test2["pressure"].fillna(test2["pressure_b2"])
            test2.drop(columns=["pressure_b2"], inplace=True)

        if test2["pressure"].isna().any():
            grp_b3 = (
                train_df_insp.groupby(["R", "C", "insp_step", "u_out"], sort=False)[
                    "pressure"
                ]
                .median()
                .reset_index()
                .rename(columns={"pressure": "pressure_b3"})
            )
            test2 = test2.merge(grp_b3, on=["R", "C", "insp_step", "u_out"], how="left")
            test2["pressure"] = test2["pressure"].fillna(test2["pressure_b3"])
            test2.drop(columns=["pressure_b3"], inplace=True)

        test2 = test2.merge(
            rc_uout_med_insp.rename(columns={"pressure": "pressure_rc_insp"}),
            on=["R", "C", "u_out"],
            how="left",
        )
        test2 = test2.merge(
            rc_uout_med_exp.rename(columns={"pressure": "pressure_rc_exp"}),
            on=["R", "C", "u_out"],
            how="left",
        )
        mask_insp = test2["u_out"].to_numpy(dtype=np.int16) == 0
        fill_vals = np.where(
            mask_insp,
            test2["pressure_rc_insp"].to_numpy(dtype=np.float64),
            test2["pressure_rc_exp"].to_numpy(dtype=np.float64),
        )
        test2["pressure"] = test2["pressure"].fillna(
            pd.Series(fill_vals, index=test2.index)
        )
        test2.drop(columns=["pressure_rc_insp", "pressure_rc_exp"], inplace=True)

        test2["pressure"] = test2["pressure"].fillna(global_med)
        test2["pressure"] = _snap_to_levels(
            test2["pressure"].to_numpy(dtype=np.float64)
        )
        return test2[["id", "breath_id", "time_step", "u_out", "pressure"]]

    pass1 = _predict_without_lag(train_insp, test)

    train_lag = train_insp.copy()
    train_lag["pressure_snap"] = _snap_to_levels(
        train_lag["pressure"].to_numpy(dtype=np.float64)
    )
    train_lag["_insp_seg"] = (
        (train_lag["insp_step"] == 0).groupby(train_lag["breath_id"]).cumsum()
    )
    train_lag["pressure_lag1"] = (
        train_lag.groupby(["breath_id", "_insp_seg"], sort=False)["pressure_snap"]
        .shift(1)
        .fillna(global_med)
        .astype(np.float64)
    )
    train_lag.drop(columns=["_insp_seg"], inplace=True)
    train_lag["pressure_lag1_bin"] = _nearest_level_index(
        train_lag["pressure_lag1"].to_numpy(dtype=np.float64)
    )

    grp_lag = (
        train_lag.groupby(
            [
                "R",
                "C",
                "insp_step",
                "u_out",
                "pressure_lag1_bin",
                "u_in_bin",
                "u_in_cum_bin",
                "u_out_cum",
                "u_in_diff_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .reset_index()
    )

    test_lag = test.merge(
        pass1[["id", "pressure"]].rename(columns={"pressure": "pressure_p1"}),
        on="id",
        how="left",
        validate="one_to_one",
    )
    test_lag = test_lag.sort_values(["breath_id", "time_step"], kind="mergesort")

    p1 = test_lag["pressure_p1"].to_numpy(dtype=np.float64)
    uo = test_lag["u_out"].to_numpy(dtype=np.int16)
    bid = test_lag["breath_id"].to_numpy(dtype=np.int64)

    pred_lag1 = np.full_like(p1, global_med, dtype=np.float64)
    last_b = -1
    last_insp_pred = global_med
    prev_uo = 1
    for i in range(p1.shape[0]):
        b = bid[i]
        if b != last_b:
            last_b = b
            last_insp_pred = global_med
            prev_uo = 1
        if uo[i] == 0:
            if prev_uo == 1:
                last_insp_pred = global_med
            pred_lag1[i] = last_insp_pred
            last_insp_pred = p1[i]
        else:
            pred_lag1[i] = global_med
        prev_uo = uo[i]

    test_lag["pressure_lag1_bin"] = _nearest_level_index(pred_lag1)
    test_lag.drop(columns=["pressure_p1"], inplace=True)

    test2 = test_lag.merge(
        grp_lag,
        on=[
            "R",
            "C",
            "insp_step",
            "u_out",
            "pressure_lag1_bin",
            "u_in_bin",
            "u_in_cum_bin",
            "u_out_cum",
            "u_in_diff_bin",
        ],
        how="left",
    )

    pass1_map = pass1[["id", "pressure"]].rename(columns={"pressure": "pressure_p1"})
    test2 = test2.merge(pass1_map, on="id", how="left", validate="one_to_one")

    insp_mask = test2["u_out"].to_numpy(dtype=np.int16) == 0
    refined = test2["pressure"].to_numpy(dtype=np.float64)
    p1_back = test2["pressure_p1"].to_numpy(dtype=np.float64)
    refined = np.where(insp_mask, refined, np.nan)
    test2["pressure"] = pd.Series(refined, index=test2.index).fillna(
        pd.Series(p1_back, index=test2.index)
    )
    test2["pressure"] = test2["pressure"].fillna(global_med)
    test2.drop(columns=["pressure_p1"], inplace=True)

    snapped = _snap_to_levels(test2["pressure"].to_numpy(dtype=np.float64))
    pred_by_id = pd.DataFrame({"id": test2["id"].to_numpy(), "pressure": snapped})

    out = sub.merge(pred_by_id, on="id", how="left")
    out["pressure"] = out["pressure"].fillna(global_med).astype(np.float64)
    return out


def g(dp):
    l = [i for i in glob.iglob(f"{dp}/*")]
    l.sort()
    file_count = len(l)

    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    output = pd.read_csv(sample_path)

    if file_count == 0:
        output = _fallback_predict_pressure(train_path, test_path, sample_path)
        output.to_csv("submission.csv", index=False)
        return output

    loop_time = file_count**3
    splits = max(1, file_count // 2)

    flist = []
    step = round(len(l) / splits) if splits > 0 else len(l)
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * step :])
        else:
            flist.append(l[i * step : (i + 1) * step])

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    output["pressure"] = 0.0

    for it in range(loop_time):
        weight = []
        set_seed(it)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output["pressure"] += flist[j] * weight[j]

    output["pressure"] /= float(loop_time)

    output.to_csv("submission.csv", index=False)
    return output




## === cell 2
g("../input/gb-blending")
