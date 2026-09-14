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

0.1443382667323867

# 6. Current score

6.90184

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because the blending directory is missing/empty in this Kaggle environment, so your `wc()` returns an empty/invalid prediction and `pred_list` ends up with arrays of length 1, causing the length-mismatch assignment. I make `g()` robust by (1) auto-locating a valid input directory among the provided dataset paths, (2) validating that each loaded submission has the expected length and `pressure` column, and (3) if no valid external submissions exist, falling back to a simple baseline submission (predicting the median train pressure) so a valid `.csv` is always produced. These changes are minimal and keep your blending/median-aggregation logic intact when the required files are actually present.'
- What this solution (achieved 6.43381) has done: 'Your current score is very far from the target (MAE ~10.86 vs ~0.144), and the reason is that your script is effectively producing a constant “median pressure” fallback because it cannot find any external blend files in this environment. To move the score sharply toward the target while preserving your overall “produce a submission by aggregating predictions + snap to valid pressure grid” core logic, I keep your `find_nearest()` post-processing and submission-writing intact, but change the fallback prediction to a lightweight, fully in-notebook, deterministic per-(R,C,time_step) median baseline learned from train and applied to test. This does not change model architecture/training loops (there are none), avoids leakage (uses only train to build statistics), and typically yields a large MAE improvement versus a constant baseline for this competition. The blending code remains as-is and still be used if valid external submissions are found; only the “no valid files” fallback is upgraded to be competitive and closer to the target.'
- What this solution (achieved 6.10817) has done: 'Your current score is far worse than the target and is dominated by the quality of the “no external blend files found” fallback, so the smallest safe improvement is to make that fallback much stronger while keeping your overall logic (aggregation + snap-to-valid-pressure grid + writing `submission.csv`) unchanged. I keep your blending path intact, but upgrade `_fallback_pressure_submission()` from a coarse (R,C,time_step) median to a per-(R,C,u_out,time_step) median, which better matches the metric because expiratory (u_out=1) behaves very differently and is not scored. To prevent time-step floating merge misses, I use an exact integer time index within each breath (`time_idx`) computed by ordering rows inside each `breath_id` (this is stable in this dataset) and then map predictions back to `id`. This stays deterministic, avoids leakage (train-only statistics), runs fast, and should move MAE substantially toward the target without changing any model/training core logic (there is none).'
- What this solution (achieved 4.33309) has done: 'Your current MAE (6.108) is far worse than the target (0.144), so we should improve the “no external blend files found” fallback, since that dominates your output in this environment. To keep your core logic intact (train-statistics aggregation → map to test → snap to valid pressure grid), I strengthen the fallback by conditioning on more informative, still-leak-free features that exist at inference time: (R, C, u_out, time_idx, u_in_bin). To make this robust and fast, I bin `u_in` into a small fixed number of bins and use a hierarchical backoff (full key → drop u_in_bin → drop u_out → global median) so every test row gets a prediction. Everything else (your blending flow, `find_nearest()`, and writing `submission.csv`) stays the same.'
- What this solution (achieved 2.21577) has done: 'Your current score is still far above the target, so we should improve the fallback predictions (used when no external blend files exist) while keeping your overall logic intact: train-statistics aggregation → map to test → snap to valid pressure grid → write `submission.csv`. The smallest high-impact change is to add an explicit “physics-like” feature (`u_in` cumulative integral per breath, i.e., `u_in_cum`) and condition the median lookup on it (with light binning) plus `u_out`, which better separates inspiratory dynamics and typically reduces MAE substantially. I keep your existing hierarchical backoff idea, but extend it with `u_in_cum_bin` and keep `time_idx` so merges stay exact and fast. Everything else (blending path, `find_nearest()`, output schema/filename) remains unchanged.'
- What this solution (achieved 1.87084) has done: 'Your current score is still far above the target (lower is better), and in this environment you’re almost certainly hitting the “no valid external blend files” path, so the only way to move toward the target is to make the fallback mapping closer to the true pressure dynamics without changing your overall approach. I keep your core logic intact (train-statistics aggregation → map to test → snap to valid pressure grid → write submission.csv), but strengthen the fallback by conditioning on more informative, inference-available features: lagged u_in/u_out and cumulative u_in (integral proxy), all discretized into bins. To avoid merge misses and keep it deterministic, I continue using per-breath `time_idx` and implement a strict hierarchical backoff so every row receives a prediction. This is a minimal, local change inside `_fallback_pressure_submission()` and should reduce MAE substantially toward your target while remaining fast enough for the timeout.'
- What this solution (achieved 1.88705) has done: 'Your current score is still far worse than the target (lower is better), and in this environment you’re almost certainly using the in-notebook fallback (no valid external blend submissions found), so the only realistic way to move toward the target is to strengthen that fallback while keeping your overall “train-statistics → map to test → snap-to-pressure-grid → write submission.csv” logic unchanged. I keep your blending driver and `find_nearest()` exactly as-is, but upgrade `_fallback_pressure_submission()` to use a slightly richer, still inference-available discretized key: add binned lags, binned cumulative `u_in` (already there), and also binned cumulative “exhaled” integral (`u_in * u_out`) to better separate expiratory behavior (not scored) from inspiratory behavior. I also add a final lightweight per-(breath_id,time_idx) smoothing step (median filter of width 3) applied only to the fallback predictions before snapping, which tends to reduce MAE without changing evaluation semantics. All changes are local to the fallback path and still produce a valid `submission.csv` end-to-end.'
- What this solution (achieved 2.01032) has done: 'Your current MAE (1.887) is still far worse than the target (0.144), so we should improve the in-notebook fallback (used when no valid external blend files are found) while keeping your overall “train-statistics aggregation → map to test → snap to valid pressure grid → write submission.csv” logic unchanged. The smallest high-impact tweak is to add more inference-available “state” context (lagged pressure surrogate) without introducing any new model/training: we compute a per-key median of the next-step pressure and then recursively generate a per-breath predicted pressure sequence for test, using the previous predicted pressure (binned) as an additional grouping key. This remains deterministic, leak-free (train-only stats), keeps `find_nearest()` and file writing identical, and should move MAE notably toward the target while staying within runtime. Blending logic is untouched; only the fallback path is strengthened.'
- What this solution (achieved 2.06224) has done: 'Your current score (MAE 2.01032, lower is better) is still far from the target, and in this environment you’re almost certainly using the in-notebook fallback (no external blend submissions found), so the only reliable way to move toward the target is to strengthen that fallback while keeping your overall “train-statistics → map to test → snap to pressure grid → write submission.csv” logic intact. I make one minimal, score-relevant correction: the fallback’s autoregressive “previous pressure” state should be reset to a sensible per-breath initial pressure instead of a global median, because pressure at the start of a breath depends strongly on (R,C) and early control context. Concretely, I learn a per-(R,C,time_idx=0,u_out) median pressure from train and use it as the initial `prev_p` for each test breath (with backoff to simpler keys/global median), leaving the rest of your grouping keys, sequential generation, smoothing, and snapping unchanged. This should reduce systematic bias at the beginning of each breath and propagate improvements through the autoregressive rollout, moving MAE downward toward your target.'
- What this solution (achieved 5.03146) has done: 'Your current MAE (2.062) is still much worse than the target (0.144, lower is better), and in this environment you’re almost certainly using the in-notebook fallback (no external blend CSVs found). The smallest score-relevant change that preserves your existing fallback’s “train-statistics + autoregressive prev_p + smoothing + snap-to-grid” core logic is to correct the key mismatch: your lookup uses `p_prev_bin` derived from the *previous predicted pressure*, but your grouped medians were built with `p_prev_bin` derived from the *previous true pressure*. I rebuild the training medians in a “teacher-forced” way by shifting the grouping: compute `p_next = pressure.shift(-1)` per breath, and group rows at time t by features at time t (including `p_prev_bin` from true previous) to predict `p_next` at t+1—this aligns training statistics with inference rollout semantics. Everything else (features, hierarchical backoff structure, smoothing, snapping, and submission writing) stays the same, and this alignment typically reduces systematic rollout error substantially toward your target.'
- What this solution (achieved 6.28891) has done: 'Your current MAE is far above the target, and in this environment you’re almost certainly using the in-notebook fallback path, so the only meaningful way to move toward the target (lower MAE) is to make that fallback closer to the competition’s known structure while keeping your overall “train-derived lookup → autoregressive rollout → smoothing → snap-to-pressure-grid → write submission.csv” logic intact. The smallest high-impact improvement that doesn’t change your approach is to replace the custom binning of previous pressure with a binning directly on the known discrete pressure grid index (the competition pressures live on a fixed 0.07 step grid), which reduces lookup noise and mismatch between train/inference. I also add a tiny, deterministic post-processing step used by many valid solutions: for rows where `u_out==1` (expiratory, not scored), copy the previous step’s pressure within each breath; this improves stability without affecting inspiratory scoring directly. Everything else (file discovery/blending, AR loop, smoothing window=3, `find_nearest()`, submission writing) stays the same.'
- What this solution (achieved 6.84461) has done: 'Your current MAE is far above the target, so we should keep your existing blend/aggregation and snapping logic intact and focus on improving the fallback predictions (which are almost certainly being used when no external blend CSVs exist). The smallest high-impact fix is to align the autoregressive lookup’s time index: you built training medians keyed by `time_idx_next = time_idx + 1`, but at inference you accidentally used the current `time_idx`, causing systematic key mismatch and poor rollouts. I correct `_lookup()` to use `time_idx_next = row.time_idx + 1` (clipped) so training and inference semantics match, without changing the overall approach. Everything else (features, hierarchical backoff, expiratory copy, smoothing, snapping, and submission writing) stays the same.'
- What this solution (achieved 6.84152) has done: 'Your current MAE (6.84461, lower is better) is still far above the target (0.1443), so we should improve only the fallback path (since this environment likely has no external blend CSVs). The smallest high-impact, metric-aligned change is to ensure we do not “force” expiratory (`u_out==1`) pressures to previous-step values (which can harm the autoregressive state and thus degrade subsequent inspiratory predictions), while still keeping expiratory predictions stable. Concretely, we (1) stop overwriting all `u_out==1` predictions globally, and instead apply that “copy previous pressure” rule only to the final submission values after smoothing/snapping (so it won’t contaminate the AR rollout), and (2) keep everything else (train-derived medians, AR lookup, hierarchical backoff, smoothing, snap-to-grid, and submission writing) unchanged. This is a local correction and should move MAE downward toward the target.'
- What this solution (achieved 6.90184) has done: 'I fix the crash in the fallback path by avoiding an invalid `astype(np.int8)` cast on a `shift(-1)` series that contains NaNs at the end of each breath. Specifically, I keep `u_out_next` nullable until after filtering out the last timestep (where `p_next` is NaN anyway), then cast safely. This is a correctness/stability fix only and preserves your current core logic (train-derived medians → AR rollout → smoothing → snap-to-grid → submission.csv). I also keep all paths and the submission-writing exactly the same so it runs end-to-end in the Kaggle environment.'

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


def _read_pred_file(path, expected_len):
    """Robustly read a submission-like file and validate length."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy()
    if len(arr) != expected_len:
        return None
    return arr


def wc(input_list, expected_len):
    """
    Read one or two submissions and do a simple weighted combine.
    Handles empty lists and invalid/missing files safely.
    """
    if input_list is None or len(input_list) == 0:
        return None

    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        arr = _read_pred_file(input_list[i], expected_len)
        if arr is None:
            continue
        preds.append(arr)

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    l = l[: len(preds)]
    l_sum = sum(l) if sum(l) != 0 else 1

    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _list_candidate_dirs(dp):
    """Return candidate dirs to search for blend files, including known Kaggle input locations."""
    cands = []
    if dp is not None:
        cands.append(dp)

    for base in [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        "../data",
    ]:
        if os.path.isdir(base):
            cands.append(base)

    out = []
    seen = set()
    for d in cands:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _find_blend_files(search_dir):
    """Find CSV files that look like submissions."""
    files = []
    for p in glob.iglob(os.path.join(search_dir, "**", "*.csv"), recursive=True):
        name = os.path.basename(p).lower()
        if "train" in name or "test" in name or "sample_submission" in name:
            continue
        files.append(p)
    files.sort()
    return files


def _fallback_pressure_submission():
    """
    Keep the same core logic: train statistics -> AR rollout -> smoothing -> snap-to-grid -> write submission.csv

    Bug fix:
    - pandas shift(-1) introduces NaNs at end-of-breath; casting directly to int raises IntCastingNaNError.
      We keep u_out_next as float/nullable, filter rows where p_next is notna, then cast safely.
    """
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]
    test = pd.read_csv(test_path, usecols=usecols_test)
    sub = pd.read_csv(sample_path)

    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
    ].copy()

    tr.sort_values(["breath_id", "time_step"], inplace=True)
    test.sort_values(["breath_id", "time_step"], inplace=True)

    tr["time_idx"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    test["time_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_prev"] = (
        tr.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    test["u_in_prev"] = (
        test.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )
    tr["u_out_prev"] = (
        tr.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    test["u_out_prev"] = (
        test.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    tr_dt = (
        tr.groupby("breath_id")["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
    te_dt = (
        test.groupby("breath_id")["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
    tr_uin = tr["u_in"].astype(np.float32).to_numpy()
    te_uin = test["u_in"].astype(np.float32).to_numpy()
    tr_uout = tr["u_out"].astype(np.float32).to_numpy()
    te_uout = test["u_out"].astype(np.float32).to_numpy()

    tr["u_in_cum"] = (
        pd.Series(tr_uin * tr_dt, index=tr.index)
        .groupby(tr["breath_id"])
        .cumsum()
        .astype(np.float32)
    )
    test["u_in_cum"] = (
        pd.Series(te_uin * te_dt, index=test.index)
        .groupby(test["breath_id"])
        .cumsum()
        .astype(np.float32)
    )

    tr["u_in_out_cum"] = (
        pd.Series(tr_uin * tr_uout * tr_dt, index=tr.index)
        .groupby(tr["breath_id"])
        .cumsum()
        .astype(np.float32)
    )
    test["u_in_out_cum"] = (
        pd.Series(te_uin * te_uout * te_dt, index=test.index)
        .groupby(test["breath_id"])
        .cumsum()
        .astype(np.float32)
    )

    n_uin_bins = 50
    n_cum_bins = 60
    n_outcum_bins = 50

    uin_max = float(np.quantile(tr["u_in"].to_numpy(), 0.999))
    if uin_max <= 0:
        uin_max = 1.0
    cum_max = float(np.quantile(tr["u_in_cum"].to_numpy(), 0.999))
    if cum_max <= 0:
        cum_max = float(tr["u_in_cum"].max())
        if cum_max <= 0:
            cum_max = 1.0
    outcum_max = float(np.quantile(tr["u_in_out_cum"].to_numpy(), 0.999))
    if outcum_max <= 0:
        outcum_max = float(tr["u_in_out_cum"].max())
        if outcum_max <= 0:
            outcum_max = 1.0

    def _bin_clip(arr, vmin, vmax, nbins):
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.clip(arr, vmin, vmax)
        step = (vmax - vmin) / nbins
        if step <= 0:
            step = 1.0
        b = np.floor((arr - vmin) / step).astype(np.int16)
        return np.clip(b, 0, nbins - 1).astype(np.int16)

    tr["u_in_bin"] = _bin_clip(tr["u_in"].to_numpy(), 0.0, uin_max, n_uin_bins)
    test["u_in_bin"] = _bin_clip(test["u_in"].to_numpy(), 0.0, uin_max, n_uin_bins)
    tr["u_in_prev_bin"] = _bin_clip(
        tr["u_in_prev"].to_numpy(), 0.0, uin_max, n_uin_bins
    )
    test["u_in_prev_bin"] = _bin_clip(
        test["u_in_prev"].to_numpy(), 0.0, uin_max, n_uin_bins
    )

    tr["u_in_cum_bin"] = _bin_clip(tr["u_in_cum"].to_numpy(), 0.0, cum_max, n_cum_bins)
    test["u_in_cum_bin"] = _bin_clip(
        test["u_in_cum"].to_numpy(), 0.0, cum_max, n_cum_bins
    )

    tr["u_in_out_cum_bin"] = _bin_clip(
        tr["u_in_out_cum"].to_numpy(), 0.0, outcum_max, n_outcum_bins
    )
    test["u_in_out_cum_bin"] = _bin_clip(
        test["u_in_out_cum"].to_numpy(), 0.0, outcum_max, n_outcum_bins
    )

    global_med = float(df_train["pressure"].median())

    def _pressure_to_grid_idx(p):
        p = float(p)
        idx = int(np.searchsorted(sorted_pressures, p))
        if idx <= 0:
            return 0
        if idx >= total_pressures_len:
            return total_pressures_len - 1
        lo = sorted_pressures[idx - 1]
        hi = sorted_pressures[idx]
        return idx - 1 if abs(p - lo) <= abs(hi - p) else idx

    tr["p_prev"] = (
        tr.groupby("breath_id")["pressure"]
        .shift(1)
        .fillna(global_med)
        .astype(np.float32)
    )
    tr["p_prev_idx"] = tr["p_prev"].map(_pressure_to_grid_idx).astype(np.int16)

    tr["p_next"] = tr.groupby("breath_id")["pressure"].shift(-1).astype(np.float32)

    tr["u_out_next"] = tr.groupby("breath_id")["u_out"].shift(-1)

    tr0 = tr[tr["time_idx"] == 0]
    init_med_rcu = tr0.groupby(["R", "C", "u_out"], sort=False)["pressure"].median()
    init_med_rc = tr0.groupby(["R", "C"], sort=False)["pressure"].median()
    init_med_u = tr0.groupby(["u_out"], sort=False)["pressure"].median()

    tr_map = tr[tr["p_next"].notna()].copy()
    tr_map["u_out_next"] = tr_map["u_out_next"].fillna(1).astype(np.int8)

    tr_map = tr_map[tr_map["u_out_next"] == 0].copy()
    tr_map["time_idx_next"] = (tr_map["time_idx"] + 1).astype(np.int16)

    full_keys = [
        "R",
        "C",
        "time_idx_next",
        "u_out",
        "u_out_prev",
        "u_in_bin",
        "u_in_prev_bin",
        "u_in_cum_bin",
        "u_in_out_cum_bin",
        "p_prev_idx",
    ]
    med_full = tr_map.groupby(full_keys, sort=False)["p_next"].median()

    back1_keys = [
        "R",
        "C",
        "time_idx_next",
        "u_out",
        "u_out_prev",
        "u_in_bin",
        "u_in_prev_bin",
        "u_in_cum_bin",
        "p_prev_idx",
    ]
    med_b1 = tr_map.groupby(back1_keys, sort=False)["p_next"].median()

    back2_keys = [
        "R",
        "C",
        "time_idx_next",
        "u_out",
        "u_out_prev",
        "u_in_bin",
        "u_in_prev_bin",
        "p_prev_idx",
    ]
    med_b2 = tr_map.groupby(back2_keys, sort=False)["p_next"].median()

    back3_keys = [
        "R",
        "C",
        "time_idx_next",
        "u_out",
        "u_out_prev",
        "u_in_bin",
        "p_prev_idx",
    ]
    med_b3 = tr_map.groupby(back3_keys, sort=False)["p_next"].median()

    back4_keys = ["R", "C", "time_idx_next", "u_out", "p_prev_idx"]
    med_b4 = tr_map.groupby(back4_keys, sort=False)["p_next"].median()

    back5_keys = ["R", "C", "time_idx_next", "u_out"]
    med_b5 = tr_map.groupby(back5_keys, sort=False)["p_next"].median()

    back6_keys = ["R", "C", "time_idx_next"]
    med_b6 = tr_map.groupby(back6_keys, sort=False)["p_next"].median()

    test2 = test.copy()
    test2.sort_values(["breath_id", "time_idx"], inplace=True)

    pred = np.empty(len(test2), dtype=np.float32)
    prev_p = global_med
    prev_bid = None

    def _lookup(row, pprev_idx):
        tnext = int(row.time_idx) + 1
        if tnext > 79:
            tnext = 79

        k = (
            int(row.R),
            int(row.C),
            tnext,
            int(row.u_out),
            int(row.u_out_prev),
            int(row.u_in_bin),
            int(row.u_in_prev_bin),
            int(row.u_in_cum_bin),
            int(row.u_in_out_cum_bin),
            int(pprev_idx),
        )
        v = med_full.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        k = (
            int(row.R),
            int(row.C),
            tnext,
            int(row.u_out),
            int(row.u_out_prev),
            int(row.u_in_bin),
            int(row.u_in_prev_bin),
            int(row.u_in_cum_bin),
            int(pprev_idx),
        )
        v = med_b1.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        k = (
            int(row.R),
            int(row.C),
            tnext,
            int(row.u_out),
            int(row.u_out_prev),
            int(row.u_in_bin),
            int(row.u_in_prev_bin),
            int(pprev_idx),
        )
        v = med_b2.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        k = (
            int(row.R),
            int(row.C),
            tnext,
            int(row.u_out),
            int(row.u_out_prev),
            int(row.u_in_bin),
            int(pprev_idx),
        )
        v = med_b3.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        k = (int(row.R), int(row.C), tnext, int(row.u_out), int(pprev_idx))
        v = med_b4.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        k = (int(row.R), int(row.C), tnext, int(row.u_out))
        v = med_b5.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        k = (int(row.R), int(row.C), tnext)
        v = med_b6.get(k)
        if v is not None and not pd.isna(v):
            return float(v)

        return global_med

    for i, row in enumerate(test2.itertuples(index=False)):
        bid = int(row.breath_id)
        if prev_bid is None or bid != prev_bid:
            key_rcu = (int(row.R), int(row.C), int(row.u_out))
            v0 = init_med_rcu.get(key_rcu)
            if v0 is None or pd.isna(v0):
                key_rc = (int(row.R), int(row.C))
                v0 = init_med_rc.get(key_rc)
            if v0 is None or pd.isna(v0):
                key_u = int(row.u_out)
                v0 = init_med_u.get(key_u)
            prev_p = float(v0) if (v0 is not None and not pd.isna(v0)) else global_med
            prev_bid = bid

        pprev_idx = _pressure_to_grid_idx(prev_p)
        p = _lookup(row, pprev_idx)
        pred[i] = p

        if int(row.u_out) == 0:
            prev_p = p

    test2["pressure"] = pred.astype(float)

    test2["pressure"] = (
        test2.groupby("breath_id", sort=False)["pressure"]
        .transform(lambda s: s.rolling(window=3, center=True, min_periods=1).median())
        .astype(float)
    )

    pred_by_id = test2.set_index("id")["pressure"]
    sub["pressure"] = sub["id"].map(pred_by_id).astype(float)

    sub = sub.merge(test[["id", "breath_id", "time_idx", "u_out"]], on="id", how="left")
    sub.sort_values(["breath_id", "time_idx"], inplace=True)
    sub["pressure"] = np.where(
        sub["u_out"].to_numpy() == 1,
        sub.groupby("breath_id", sort=False)["pressure"].shift(1).to_numpy(),
        sub["pressure"].to_numpy(),
    )
    sub["pressure"] = sub["pressure"].fillna(global_med).astype(float)

    sub["pressure"] = sub["pressure"].apply(find_nearest)

    sub = sub[["id", "pressure"]].sort_values("id")
    sub.to_csv("submission.csv", index=False)


def g(dp):
    """
    Blending driver: if valid external submissions are found, blend them.
    Otherwise, produce an improved in-notebook fallback submission.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    loop_time = 154

    all_files = []
    for cand in _list_candidate_dirs(dp):
        if os.path.isdir(cand):
            all_files.extend(_find_blend_files(cand))

    if dp is not None and os.path.isdir(dp):
        all_files.extend(
            [p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")]
        )

    all_files = sorted(list(dict.fromkeys(all_files)))

    valid_files = []
    for p in all_files:
        arr = _read_pred_file(p, expected_len)
        if arr is not None:
            valid_files.append(p)

    if len(valid_files) == 0:
        _fallback_pressure_submission()
        return

    file_count = len(valid_files)
    splits = max(1, file_count // 2)

    flist_paths = []
    step = round(len(valid_files) / splits) if splits > 0 else len(valid_files)
    for i in range(splits):
        if i == splits - 1:
            flist_paths.append(valid_files[i * step :])
        else:
            flist_paths.append(valid_files[i * step : (i + 1) * step])

    flist = []
    for i in range(len(flist_paths)):
        arr = wc(flist_paths[i], expected_len=expected_len)
        if arr is not None and len(arr) == expected_len:
            flist.append(arr)

    if len(flist) == 0:
        _fallback_pressure_submission()
        return

    pred_list = []
    for ii in range(loop_time):
        weight = []
        set_seed(ii)
        for jj in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for jj in range(len(weight)):
            weight[jj] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for jj in range(len(flist)):
            temp += flist[jj] * weight[jj]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
