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

0.1360168427746761

# 6. Current score

8.09179

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because the blending directory is missing in this environment, so no prediction files are loaded and the code ends up producing a single scalar median (length 1) that cannot be assigned to 603600 submission rows. I (1) make the path/file discovery robust, (2) validate and align every loaded prediction to the sample_submission `id` order, and (3) add a safe fallback that produces a valid submission (all zeros) if no blend inputs exist, so you always get a `.csv` output. These changes are execution/stability fixes and keep the core “blend predictions then snap to nearest known pressure” logic intact; they just prevent empty/shape-mismatch cases and guarantee a correctly formatted submission.'
- What this solution (achieved 9.92091) has done: 'Your current score (17.65 MAE) indicates the script is falling back to an all-zero submission because it can’t find any blend input files in `../input/gb-data-blending-recover`, which is far from the target (0.136). To move toward the target with minimal logic changes, I keep your existing “blend predictions then snap to nearest known pressure” approach, but add a safe, competition-legal baseline generator that uses the training-set median pressure per (R,C,time_step) and merges it onto test (then snaps to nearest pressure). This only activates when no blend CSVs are found, so if blend files exist it preserves your original behavior. The baseline is fast (groupby + merge) and typically scores orders of magnitude better than zeros, pushing the score much closer to your target.'
- What this solution (achieved 10.21761) has done: 'Your current MAE (9.92, lower-is-better) is still far from the target (0.136), and the biggest remaining gap is that the fallback baseline (median pressure per (R,C,time_step)) ignores the most important driver: the inspiratory control signal history (`u_in`) and valve state (`u_out`). To move the score substantially toward the target with minimal changes and without altering your blend/snap core logic, I keep the same “if blend files exist, blend+median+snap” path, but strengthen the fallback baseline by adding simple, competition-legal lag and cumulative features computed per breath, and then using a median lookup on those features from train to test. This is still a pure train-statistics merge (no new model/training loop), runs fast (groupby + merge), and keeps your final “snap to nearest known pressure” post-processing identical. I also keep the original file-loading/alignment safeguards unchanged.'
- What this solution (achieved 10.23542) has done: 'Your current score is far worse than the target (lower is better), and the remaining gap is mainly because the fallback baseline is still too coarse and often misses exact pressure levels. I keep your core “blend if files exist, otherwise fallback, then snap to known pressures” logic unchanged, but strengthen only the fallback by using a higher-coverage, more ventilator-relevant lookup: median pressure by (R,C,time_step,u_out,u_in,u_in_cumsum,u_in_lag1), with a structured backoff that also uses per-(R,C,time_step,u_out,u_in) and per-(R,C,time_step,u_out,u_in_cumsum). I also ensure joins are stable and memory-safe (categoricals, float32, pre-sorting) so it finishes within time. This should materially reduce MAE vs the current fallback while preserving identical evaluation semantics and output format.'
- What this solution (achieved 10.40091) has done: 'I keep your existing “blend predictions then take a median over random weights, then snap to nearest known pressure” logic untouched, and focus only on the fallback path that’s currently producing a very weak score. Specifically, I make the train-stats baseline more aligned with the competition metric by predicting pressure only for the inspiratory phase (u_out==0) and setting expiratory predictions to 0 (expiratory is not scored), which typically reduces MAE substantially without changing evaluation semantics. I also fix a subtle but important alignment issue: snapping must use the *sorted unique pressures* (not the unsorted unique array), otherwise the vectorized snap can choose incorrect nearest values. These are minimal, competition-legal changes that should move MAE down toward your target.'
- What this solution (achieved 8.15716) has done: 'Your current MAE (10.40091, lower-is-better) is far from the target (0.1360), so we should improve score with minimal, low-risk changes. The biggest issue is that the fallback baseline is still too inaccurate because it relies on exact floating `time_step` keys and weak backoff; I keep the same “train medians -> merge -> snap to known pressures” approach but (1) switch to an integer `step` index per breath for stable joins, and (2) add a stronger, ventilator-relevant lookup keyed by `(R,C,step,u_out,u_in,u_in_cumsum)` with structured backoff. I also preserve your existing “expiratory not scored” handling (set u_out==1 predictions to 0) and keep the blending path unchanged. These changes only affect the fallback path used when no blend files exist, and should materially reduce MAE toward the target while still running within the time limit.'
- What this solution (achieved 8.15707) has done: 'We keep your overall “blend if available, otherwise fallback baseline, then snap to known pressures” logic unchanged, but strengthen the fallback baseline to better match the competition target without introducing any new model/training loop. Concretely, we (1) add two very cheap per-breath features that substantially improve lookup hit-rate (`u_in_diff1` and a discretized `u_in_bin`), (2) use a more ventilator-relevant primary median key that includes these features, and (3) keep the structured backoff you already use so it remains stable and fast. We also ensure we don’t waste time rebuilding the `step` feature twice and keep expiratory predictions at 0 (not scored), preserving evaluation semantics. These are minimal, competition-legal changes focused on reducing MAE from your current 8.15716 toward the 0.136 target.'
- What this solution (achieved 7.92211) has done: 'Your current MAE (8.157) is still far above the target (0.136), so we should improve the fallback path (used when no blend files are found) with minimal, competition-legal changes that preserve your “train medians -> merge -> snap” core logic. The main fix is to stop using an inspiratory-only-trained table without correctly handling inspiratory predictions: we compute stats on inspiratory rows (u_out==0) but also add a very cheap and high-signal key using the per-breath “area under u_in” proxy (`u_in_cumsum`) plus `u_in_lag1`/`u_in_lag2`, and we add a deterministic backoff ladder that prioritizes history-based keys before coarse keys. We also ensure the expiratory rows are set to 0 (still not scored) and keep your blend path unchanged. These changes are purely in the fallback statistics merge (no new model/loops) and are designed to reduce MAE materially toward the target while staying stable and within the time limit.'
- What this solution (achieved 8.26039) has done: 'Your current MAE (7.922, lower-is-better) is still far above the target (0.136), and in this environment you’re almost certainly running the fallback path (no blend files found), so we should only strengthen that fallback while preserving your existing “train-stats merge → backoff → snap to known pressures” core logic. The main issue is that the current fallback depends on exact/rounded floating keys (especially `u_in_cumsum`), which causes massive train→test key mismatches and forces frequent backoff to coarse/global medians. I keep the same approach but make the keys more join-stable by using integer-quantized versions of `u_in`, lags, diffs, and cumsum (plus a cumulative “bucket” feature) computed identically for train and test. I also add one extra very-cheap history feature (`u_in_ema2`) into the *first* lookup key to improve hit-rate without changing any model/training loops, and keep your “u_out==1 → 0 pressure” handling and final snapping unchanged.'
- What this solution (achieved 8.11501) has done: 'Your current MAE (8.260) is still far above the target (0.136, lower-is-better), and given the environment you’re almost certainly using the fallback train-stats path (no blend files found). To move the score down with minimal change and without altering the overall “train medians → merge/backoff → snap” approach, I (1) build the fallback stats only on the inspiratory phase (u_out==0) so expiratory rows don’t dilute the medians, and (2) stop forcing expiratory predictions to 0 and instead use a safe constant (the global median pressure) there—since expiratory rows are not scored, this can reduce error leakage if any scoring mask mismatch happens and avoids extreme values. Everything else (feature engineering, quantization, backoff ladder, snapping to known pressures, blend path) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 7.52081) has done: 'Your current MAE (8.115, lower-is-better) is still far above the target (0.136), so we should improve the fallback path (used when no blend files are found) while keeping your overall “train medians → merge/backoff → snap” core logic intact. The biggest issue is that the fallback uses very strict, high-dimensional keys that rarely match between train and test, causing heavy backoff to coarse medians; to move score down, we (1) add a join-stable “ranked u_in within breath” feature (per-step quantile bin) and (2) add one intermediate backoff table keyed by `(R,C,step,u_out,u_in_rankbin)` that has much higher coverage than your current primary key but still preserves step-wise control-signal behavior. We keep your inspiratory-only stats computation, your expiratory handling, and your final snap-to-known-pressures unchanged. This is a minimal, competition-legal change (still pure train-statistics lookup, no new model/training loop) and should reduce MAE toward the target without risking runtime issues.'
- What this solution (achieved 8.15266) has done: 'Your current MAE (7.52081; lower is better) is far above the target (0.136), and in this environment you’re almost certainly running the fallback train-stats path (no blend files found), so the smallest productive change is to strengthen that fallback without touching your blend/median/snap core logic. The main issue is join sparsity: your highest-dimensional keys rarely match between train/test, causing frequent backoff to coarse medians; we reduce that by adding one more high-coverage intermediate table keyed by `(R,C,step,u_out,u_in_bin,u_in_rankbin)` and placing it early in the backoff chain. We keep inspiratory-only stats computation, keep expiratory handling unchanged (set to global median), and keep final snap-to-known-pressures identical. This is a minimal, competition-legal adjustment (still pure groupby-median merges, no new model/training loop) and should move MAE downward toward the target.'
- What this solution (achieved 8.45835) has done: 'Your score gap to the target is huge (8.15 vs 0.136, lower-is-better), and in this environment you’re effectively always running the fallback baseline because the blend directory is missing, so we should only strengthen that fallback while keeping your “train medians → merge/backoff → snap” core logic unchanged. The biggest issue is still join sparsity from over-specific, noisy keys; we add one high-coverage but still informative lookup table keyed by `(R,C,step,u_out,u_in_q)` plus a *quantized cumulative history bucket* (derived from `u_in_cumsum`) and place it early in the backoff chain to reduce fallback to coarse medians. We also compute `u_in_cumsum` from `u_in` in `float64` before quantization to make buckets deterministic between train/test (tiny float drift can otherwise break joins), but keep all model semantics the same. No training loops, architectures, or loss changes are introduced; we still end with snapping to the nearest known pressure and writing `submission.csv`.'
- What this solution (achieved 8.63676) has done: 'Your current MAE (8.458, lower-is-better) is still far from the target (0.136), and in this environment you’re almost certainly always taking the fallback path (because the blend directory doesn’t exist), so we should only strengthen the fallback while keeping your overall “train medians → merge/backoff → snap to known pressures” core logic intact. The smallest high-impact fix is to stop relying on very sparse high-dimensional exact-key joins and instead add a *high-coverage* lookup keyed by `(R,C,step,u_out,u_in_bin,u_in_cumsum_bucket)` early in the backoff chain, plus a slightly finer `(R,C,step,u_out,u_in_bin,u_in_cumsum_bucket2)` table. This preserves the same statistics-based approach (no new model/training loop), but materially increases match rate so you don’t fall back to coarse/global medians so often. Everything else (blend path behavior, snapping, submission writing) remains unchanged.'
- What this solution (achieved 8.09179) has done: 'We need to move MAE down (lower is better) from 8.64 toward 0.136, and your environment likely uses the fallback path because the blend directory doesn’t exist. Keeping your core “train medians → merge/backoff → snap to known pressures” logic, the smallest high-impact change is to make the *primary/early* lookup less sparse by adding a stable high-coverage table keyed by `(R,C,step,u_out,u_in_q)` (and `(R,C,step,u_out,u_in_bin)`), and to soften overly-specific quantization by using slightly coarser `u_in_q` (×50 instead of ×100) to increase train→test key matches. This preserves evaluation semantics and post-processing, doesn’t introduce a new model/training loop, and should reduce backoff-to-global-median frequency (which is what’s currently driving large MAE). All paths still end by snapping to the sorted known pressure grid and writing a valid `submission.csv`.'

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


def _load_pred_as_array(path, sample_ids):
    """
    Robustly load a prediction file and align to sample_submission ids.
    Accepts either [id, pressure] or just [pressure] with correct length.
    Returns None if file is unusable.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "pressure" not in df.columns:
        return None

    if "id" in df.columns:
        df = df[["id", "pressure"]].dropna()
        df = df.set_index("id")
        aligned = df.reindex(sample_ids)
        if aligned["pressure"].isna().any():
            return None
        arr = aligned["pressure"].to_numpy(dtype=np.float32)
    else:
        arr = df["pressure"].to_numpy(dtype=np.float32)
        if arr.shape[0] != sample_ids.shape[0]:
            return None

    if not np.isfinite(arr).all():
        return None
    return arr


def wc(input_list):
    """
    Weighted combine of 1-2 submissions from a split.
    If parsing score fails, use equal weights.
    """
    l = []
    preds = []
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()

    for p in input_list:
        try:
            public_lb_score = float(os.path.basename(p).split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1.0
        l.append(public_lb_score)

        arr = _load_pred_as_array(p, sample_ids)
        if arr is None:
            continue
        preds.append(arr)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    preds = preds[:2]
    l = l[:2]

    l_sum = sum(l) if sum(l) != 0 else 1.0
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _snap_to_known_pressures(pred):
    """
    Keep existing post-processing semantics ("snap to nearest known pressure"),
    vectorized for speed and correctness (uses sorted_pressures for searchsorted).
    """
    pred = pred.astype(np.float32, copy=False)

    sp = sorted_pressures
    idx = np.searchsorted(sp, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = idx

    lower_val = sp[lower_idx]
    upper_val = sp[upper_idx]

    choose_lower = np.abs(lower_val - pred) < np.abs(upper_val - pred)
    snapped = np.where(choose_lower, lower_val, upper_val).astype(np.float32)
    return snapped


def _add_basic_breath_features(df):
    """
    Fallback feature builder (train-stats baseline):
    - integer per-breath 'step' index (0..79) for stable joins
    - simple lags/cumsum and u_in bins
    """
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
    g = df.groupby("breath_id", sort=False)

    df["step"] = g.cumcount().astype(np.int16)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float64)

    df["u_in_x_u_out"] = (df["u_in"] * df["u_out"]).astype(np.float32)

    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

    df["u_in_ema2"] = (0.5 * df["u_in"] + 0.5 * df["u_in_lag1"]).astype(np.float32)

    df["u_in_bin"] = np.clip(np.floor(df["u_in"] / 2.0), 0, 50).astype(np.int16)

    r = g["u_in"].rank(method="average")
    denom = g["u_in"].transform("size").astype(np.float32)
    q = (r.to_numpy(np.float32) - 1.0) / np.maximum(
        denom.to_numpy(np.float32) - 1.0, 1.0
    )
    df["u_in_rankbin"] = np.minimum((q * 20.0).astype(np.int16), np.int16(19))

    return df


def _baseline_submission_from_train_stats():
    """
    Fallback (used only when no blend inputs exist):
    Keep the same core logic (train medians -> merge with backoff -> snap),
    but improve match-rate by adding early, high-coverage lookup tables and
    making u_in quantization slightly coarser (more join-stable).

    - Compute lookup medians on inspiratory rows only (u_out==0) so expiratory
      phase does not dilute the mapping.
    - Do not force expiratory predictions to 0; set them to a safe constant
      (global median) since expiratory rows are not scored.
    """
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

    for d in (df_train, test):
        d["R"] = d["R"].astype("int16", copy=False)
        d["C"] = d["C"].astype("int16", copy=False)
        d["u_out"] = d["u_out"].astype("int8", copy=False)
        d["time_step"] = d["time_step"].astype(np.float32, copy=False)
        d["u_in"] = d["u_in"].astype(np.float32, copy=False)

    train_feat = _add_basic_breath_features(
        df_train[["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]]
    )
    test_feat = _add_basic_breath_features(
        test[["breath_id", "R", "C", "time_step", "u_in", "u_out"]]
    )

    def _quantize_keys(df):
        df = df.copy()

        df["u_in_q"] = np.rint(df["u_in"].to_numpy(np.float32) * 50.0).astype(np.int32)
        df["u_in_lag1_q"] = np.rint(df["u_in_lag1"].to_numpy(np.float32) * 50.0).astype(
            np.int32
        )
        df["u_in_lag2_q"] = np.rint(df["u_in_lag2"].to_numpy(np.float32) * 50.0).astype(
            np.int32
        )
        df["u_in_diff1_q"] = np.rint(
            df["u_in_diff1"].to_numpy(np.float32) * 50.0
        ).astype(np.int32)
        df["u_in_ema2_q"] = np.rint(df["u_in_ema2"].to_numpy(np.float32) * 50.0).astype(
            np.int32
        )

        cs = df["u_in_cumsum"].to_numpy(np.float64)
        df["u_in_cumsum_q"] = np.rint(cs * 2.0).astype(np.int32)

        df["u_in_cumsum_bucket"] = (df["u_in_cumsum_q"] // 20).astype(np.int32)
        df["u_in_cumsum_bucket2"] = (df["u_in_cumsum_q"] // 50).astype(np.int32)
        return df

    train_feat = _quantize_keys(train_feat)
    test_feat = _quantize_keys(test_feat)

    train_insp = train_feat[train_feat["u_out"] == 0].copy()

    keys_p_uinq = ["R", "C", "step", "u_out", "u_in_q"]
    stats_p_uinq = (
        train_insp.groupby(keys_p_uinq, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_uinq"})
    )

    keys_p_ubin = ["R", "C", "step", "u_out", "u_in_bin"]
    stats_p_ubin = (
        train_insp.groupby(keys_p_ubin, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_ubin"})
    )

    keys_p_bin_cs1 = ["R", "C", "step", "u_out", "u_in_bin", "u_in_cumsum_bucket"]
    stats_p_bin_cs1 = (
        train_insp.groupby(keys_p_bin_cs1, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_bin_cs1"})
    )

    keys_p_bin_cs2 = ["R", "C", "step", "u_out", "u_in_bin", "u_in_cumsum_bucket2"]
    stats_p_bin_cs2 = (
        train_insp.groupby(keys_p_bin_cs2, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_bin_cs2"})
    )

    keys_p0 = [
        "R",
        "C",
        "step",
        "u_out",
        "u_in_bin",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_ema2_q",
        "u_in_cumsum_q",
    ]
    stats_p0 = (
        train_insp.groupby(keys_p0, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p0"})
    )
    merged = test_feat.merge(stats_p0, on=keys_p0, how="left", sort=False)

    merged = merged.merge(stats_p_uinq, on=keys_p_uinq, how="left", sort=False)
    merged = merged.merge(stats_p_ubin, on=keys_p_ubin, how="left", sort=False)

    merged = merged.merge(stats_p_bin_cs1, on=keys_p_bin_cs1, how="left", sort=False)
    merged = merged.merge(stats_p_bin_cs2, on=keys_p_bin_cs2, how="left", sort=False)

    keys_p_uinq_cs2 = ["R", "C", "step", "u_out", "u_in_q", "u_in_cumsum_bucket2"]
    stats_p_uinq_cs2 = (
        train_insp.groupby(keys_p_uinq_cs2, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_uinq_cs2"})
    )
    merged = merged.merge(stats_p_uinq_cs2, on=keys_p_uinq_cs2, how="left", sort=False)

    keys_p_binrank = ["R", "C", "step", "u_out", "u_in_bin", "u_in_rankbin"]
    stats_p_binrank = (
        train_insp.groupby(keys_p_binrank, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_binrank"})
    )
    merged = merged.merge(stats_p_binrank, on=keys_p_binrank, how="left", sort=False)

    keys_p_rank = ["R", "C", "step", "u_out", "u_in_rankbin"]
    stats_p_rank = (
        train_insp.groupby(keys_p_rank, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p_rank"})
    )
    merged = merged.merge(stats_p_rank, on=keys_p_rank, how="left", sort=False)

    keys_p1 = ["R", "C", "step", "u_out", "u_in_q", "u_in_lag1_q", "u_in_cumsum_q"]
    stats_p1 = (
        train_insp.groupby(keys_p1, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p1"})
    )
    merged = merged.merge(stats_p1, on=keys_p1, how="left", sort=False)

    keys_p2 = ["R", "C", "step", "u_out", "u_in_q", "u_in_cumsum_bucket"]
    stats_p2 = (
        train_insp.groupby(keys_p2, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p2"})
    )
    merged = merged.merge(stats_p2, on=keys_p2, how="left", sort=False)

    keys_p3 = ["R", "C", "step", "u_out", "u_in_q"]
    stats_p3 = (
        train_insp.groupby(keys_p3, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p3"})
    )
    merged = merged.merge(stats_p3, on=keys_p3, how="left", sort=False)

    keys_p4 = ["R", "C", "step"]
    stats_p4 = (
        train_insp.groupby(keys_p4, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pred_p4"})
    )
    merged = merged.merge(stats_p4, on=keys_p4, how="left", sort=False)

    global_med = float(df_train["pressure"].median())

    pred = (
        merged["pred_p0"]
        .fillna(merged["pred_p_uinq"])
        .fillna(merged["pred_p_ubin"])
        .fillna(merged["pred_p_bin_cs1"])
        .fillna(merged["pred_p_bin_cs2"])
        .fillna(merged["pred_p_uinq_cs2"])
        .fillna(merged["pred_p_binrank"])
        .fillna(merged["pred_p_rank"])
        .fillna(merged["pred_p1"])
        .fillna(merged["pred_p2"])
        .fillna(merged["pred_p3"])
        .fillna(merged["pred_p4"])
        .fillna(global_med)
        .to_numpy(dtype=np.float32)
    )

    u_out_arr = test_feat["u_out"].to_numpy(dtype=np.int8)
    pred = np.where(u_out_arr == 1, np.float32(global_med), pred).astype(np.float32)

    sub["pressure"] = _snap_to_known_pressures(pred)
    sub.to_csv("submission.csv", index=False)


def g(dp):
    """
    Blend if blend inputs exist; otherwise produce a train-stats baseline submission.
    """
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = output["id"].to_numpy()
    n = len(output)

    files = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                files.append(i)
    files.sort()

    if len(files) == 0:
        _baseline_submission_from_train_stats()
        return

    file_count = len(files)
    loop_time = 156
    splits = file_count // 2
    if splits < 1:
        splits = 1

    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) == 0:
            continue
        blended = wc(chunk)
        if blended is None or blended.shape[0] != n:
            continue
        flist.append(blended.astype(np.float32, copy=False))

    if len(flist) == 0:
        _baseline_submission_from_train_stats()
        return

    pred_list = []
    for t in range(loop_time):
        weight = []
        set_seed(t)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float32)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)
    output["pressure"] = np.median(stacked, axis=0)

    output["pressure"] = _snap_to_known_pressures(
        output["pressure"].to_numpy(dtype=np.float32)
    )
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
