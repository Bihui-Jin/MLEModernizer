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

0.1392501136279981

# 6. Current score

3.49943

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the crash by making the blending code robust to directories that don’t exist (your environment doesn’t have `ventilator-pressure-high-score-submissions`) and to cases where a split contains 0 or 1 files (which caused the `IndexError` in `wc`). Since no valid submission was produced, I also ensure we always write a properly named `submission.csv` with the required `id,pressure` columns. To keep the core logic intact, I won’t change the model/metric assumptions; if the external submissions folder is missing or empty, the code safely fall back to a simple baseline prediction (median train pressure snapped to nearest allowed pressure), which guarantees an end-to-end run and a valid file.'
- What this solution (achieved 4.1728) has done: 'Your current score (10.86378 MAE; lower is better) is far from the target (0.13925), and the main reason is that your code falls back to a constant-median baseline because the external “high-score submissions” directory doesn’t exist in this environment. To move the score sharply toward the target without changing the core “blend submissions if available” logic, I add a minimal, competition-legal fallback that generates a much stronger per-row prediction using only the provided train/test signals: map each (R, C, u_in, u_out, time_step) pattern to the median observed train pressure and use that for test, then snap to the nearest allowed pressure as you already do. This keeps your existing blending path intact (used if files exist), but replaces the weak constant baseline with a lightweight lookup-based predictor that runs fast and is aligned with MAE on inspiratory phase (u_out=0 tends to drive pressure). The output remains a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.12049) has done: 'Your current MAE (4.1728; lower is better) is still far above the target (0.13925), so we should improve the fallback path (used when the external submissions folder is missing) without changing the overall “blend if available, else fallback” core logic. The minimal, high-impact fix is to make the lookup fallback breath-aware by adding `breath_id` and the within-breath step index as keys (instead of relying on raw `time_step` rounding), which better matches how the data is generated and reduces collisions. We keep the same median-mapping idea, keep snapping to allowed pressures, and keep all file paths and output format unchanged. This should move MAE materially toward the target while staying lightweight and within the same evaluation semantics.'
- What this solution (achieved 7.08689) has done: 'Your fallback predictor is still too coarse for this competition, which is why MAE is stuck around ~4 while the target is ~0.139 (lower is better), so we need a stronger fallback without changing your overall “blend submissions if available, else fallback” core logic. The minimal high-impact improvement is to keep the same train→median lookup idea but make it truly time-series aware by keying on lagged `u_in/u_out` history within a breath (a tiny fixed window), which better captures the pressure dynamics while staying lightweight and legal. I also ensure the fallback aligns predictions back to the original test row order (your current fallback sorts by breath/time and then tries to unsort, but it relies on index behavior that can misalign). Everything else (snapping to allowed pressures, submission writing, blender behavior) stays the same.'
- What this solution (achieved 10.56725) has done: 'Your current score is far worse than the target (MAE 7.09 vs 0.139, lower is better) and the main reason is that the “high-score submissions” folder is absent, so your fallback predictor dominates. To move toward the target while preserving the same core “train→median lookup fallback + snap-to-allowed-pressures” logic, I fix a critical alignment bug: you currently assign predictions by position to `sample_submission`, which can silently misalign if row order ever differs; instead we merge predictions by `id`. I also make the fallback key slightly more informative but still in the same lookup-family by adding cumulative volume features (`u_in` integral) and a small lag window (keeping your existing lags) to better capture dynamics without changing the approach. The submission writing remains `submission.csv` with exactly `id,pressure` and correct row count.'
- What this solution (achieved 10.56725) has done: 'Your current MAE is far above the target, so we should strengthen the fallback path (used when the external submissions folder is missing) while preserving the same “train→median lookup, then snap to allowed pressures” core logic. The biggest gain with minimal conceptual change is to (1) make the lookup use a more physically meaningful within-breath state: cumulative inspired volume proxy via `u_in * dt` (already present) and a simple “time since last u_out==1” counter, and (2) add a final ultra-coarse backoff keyed only by `(R,C,step,u_out)` to avoid falling back to a global median too often. These changes keep the same evaluation semantics and prediction style (pure nonparametric median mapping) but should reduce lookup misses and thus MAE. The submission is still merged by `id` and written to `submission.csv` with the required columns.'
- What this solution (achieved 3.63088) has done: 'Your current MAE is far above the target, so we should strengthen the existing fallback (used when no external submissions are present) without changing its core “train→median lookup then snap to allowed pressures” logic. The biggest issue is that the current lookup keys are too specific and sparse, causing many misses and effectively reverting to coarse/global medians; we fix this by adding a backoff cascade that progressively relaxes the keys and by using an inspiratory-aware median (computed on `u_out==0`) for better alignment with the metric. We also make the within-breath state computation (`since_out1`) deterministic and fast using `groupby().cumcount()` on reset segments rather than `apply/explode`, which can misbehave and is slow at this scale. These are minimal changes that keep the same nonparametric approach and output semantics while materially reducing lookup miss rate.'
- What this solution (achieved 3.63119) has done: 'We keep your exact “blend submissions if available, else fallback train→median lookup + snap-to-allowed-pressures” approach, but make the fallback much less sparse so it misses far less often (your current 3.63 MAE indicates many rows still fall back to coarse medians). Concretely, we (1) reduce over-quantization of state features (`u_in_cum`, `u_in_area`) and add a tiny extra physically meaningful state (`u_in_diff`) while staying in the same nonparametric lookup family, and (2) add two additional intermediate backoff levels that retain dynamics (lags + cum/area) but relax the noisiest key (`since_out1`) and/or `u_in_area_r`. We also ensure the lookup medians are computed on inspiratory rows for inspiratory-like keys (u_out=0), which better matches the evaluation without changing semantics, and we keep the id-merge alignment and output format unchanged. These minimal changes should move MAE downward toward your target without changing the overall logic or requiring any new packages.'
- What this solution (achieved 3.63119) has done: 'Your current score (3.63119 MAE; lower is better) is still far above the target (0.13925), so we should improve the fallback path (used when the external submissions folder is missing) with minimal changes and the same nonparametric “train→median lookup + backoff + snap-to-allowed-pressures” core logic. The biggest low-risk gain is to align the lookup with the scoring rule by training the lookup primarily on inspiratory rows (`u_out==0`) and forcing expiratory predictions to a stable value (pressure at the first `u_out==1` step, else last inspiratory prediction), since expiratory phase is not scored and this reduces noise without changing evaluation semantics. We keep your existing features and backoff cascade but add a dedicated expiratory-handling post-process and a more consistent choice of `tr_sub` (use inspiratory-only for *all* backoff levels that include `u_out`, not just when `u_out` is in the key). These changes should reduce MAE materially while keeping the approach, paths, and output format unchanged.'
- What this solution (achieved 3.50519) has done: 'We keep your exact “blend external submissions if present, else train→median lookup with backoff + snap-to-allowed pressures” pipeline, but make the fallback lookup less sparse where it currently misses and falls back to coarse medians (driving MAE ~3.63). Concretely, we add two very small, score-aligned backoff levels that (a) keep the time-series dynamics but relax the noisiest keys (`since_out1` and the most precise integral rounding) and (b) add a very coarse inspiratory-only lookup keyed by `(R,C,step,u_in_r)` to better approximate the known discrete pressure trajectories without changing the nonparametric approach. We also make the “inspiratory-only training subset” choice consistent per backoff: only use `u_out==0` training when the key implies inspiratory behavior (either contains `u_out` or is explicitly inspiratory); this avoids contaminating inspiratory medians with expiratory data. Everything else (file paths, blending behavior, snapping, submission format) stays unchanged and still writes `submission.csv`.'
- What this solution (achieved 3.59105) has done: 'The timeout is dominated by the fallback path: it loads the full 5.4M-row training set, builds many intermediate columns, and then iterates row-by-row through the test set using `te.iloc[i]` inside nested loops, which is prohibitively slow in pure Python. I keep the exact same fallback logic (same engineered features, same backoff cascade, same sequential dependence on previous predicted pressure), but replace per-row pandas Series extraction with precomputed NumPy arrays and tuple keys, and replace the O(n) “find breath end” while-loop with precomputed breath boundaries. I also replace the slow `output["pressure"].apply(find_nearest)` with a vectorized nearest-pressure snap using `np.searchsorted`, which is exactly equivalent. These changes remove Python-level overhead and repeated pandas work while preserving predictions and evaluation semantics (only negligible FP diffs from dtype handling).'
- What this solution (achieved 3.49943) has done: 'Your current MAE (3.59105; lower is better) is still far above the target (0.13925), and the main limiter is the fallback lookup’s miss rate: too many test rows don’t find a key and revert to coarse medians. I keep the exact same nonparametric “train→median lookup + backoff cascade + sequential previous-pressure dependence + snap-to-allowed-pressures” core logic, but add two extremely low-risk backoff levels that retain dynamics while relaxing only the sparsest continuous keys (`u_in_area_r`/`u_in_cum_r`). I also make the lookup `.get()` faster and more reliable by using plain-Python dicts (same values) rather than pandas Series for key retrieval, which is correctness-preserving but reduces overhead so we can afford the extra backoffs within the time limit. Everything else (paths, blending behavior if external submissions exist, submission format) stays unchanged and still writes `submission.csv`.'

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
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)

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


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx]
    lower = sorted_pressures[lower_idx]
    choose_lower = np.abs(lower - preds) < np.abs(upper - preds)
    out = np.where(choose_lower, lower, upper)
    out = np.where(idx == 0, sorted_pressures[0], out)
    out = np.where(
        idx == total_pressures_len - 1,
        np.where(preds > sorted_pressures[-1], sorted_pressures[-1], out),
        out,
    )
    return out


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Blend a list of submission files into one prediction vector.

    Safe for 0/1/2/N files; weights derived from filename score when possible.
    """
    if input_list is None or len(input_list) == 0:
        return None

    scores = []
    preds = []
    for path in input_list:
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain a 'pressure' column.")
        preds.append(df["pressure"].to_numpy().ravel())

        base = os.path.basename(path)
        score_val = 1.0
        try:
            score_str = base.split(".")[1].split(" ")[0]
            score_val = float(score_str)
        except Exception:
            score_val = 1.0
        scores.append(score_val)

    if len(preds) == 1:
        return preds[0]

    l_sum = float(np.sum(scores)) if np.sum(scores) != 0 else float(len(scores))

    if len(preds) == 2:
        weight1 = (scores[1] / l_sum) + 0.1
        weight2 = 1.0 - weight1
        return preds[0] * weight1 + preds[1] * weight2

    weights = np.array(scores, dtype=float) / l_sum
    out = np.zeros_like(preds[0], dtype=float)
    for w, p in zip(weights, preds):
        out += w * p
    return out


def _fallback_predict_from_train_lookup(
    df_train_local: pd.DataFrame, df_test_local: pd.DataFrame
) -> pd.DataFrame:
    """
    Fallback used only when no external submission files are present.

    Core logic preserved: nonparametric train->median lookup with a backoff cascade,
    then later snapping to nearest allowed pressure.

    Score-improvement (minimal, same approach):
    - Add two additional backoff levels that keep the same features but relax the
      sparsest continuous keys (u_in_area/u_in_cum) by using coarser rounding.
      This reduces lookup misses (and thus MAE) without changing the modeling family.

    Performance/correctness-preserving:
    - Convert groupby-median Series to plain dicts for faster .get(key) in the
      per-row sequential loop (same values, just faster access).
    """
    train_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    test_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    tr = df_train_local[train_cols].copy()
    te = df_test_local[test_cols].copy()

    tr.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
    te.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    tr["step"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["step"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_r"] = tr["u_in"].round(1)
    te["u_in_r"] = te["u_in"].round(1)

    tr["dt"] = (
        tr.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )
    te["dt"] = (
        te.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )

    tr["u_in_cum"] = tr.groupby("breath_id")["u_in_r"].cumsum().astype(np.float32)
    te["u_in_cum"] = te.groupby("breath_id")["u_in_r"].cumsum().astype(np.float32)

    tr["u_in_area"] = (
        (tr["u_in_r"] * tr["dt"]).groupby(tr["breath_id"]).cumsum()
    ).astype(np.float32)
    te["u_in_area"] = (
        (te["u_in_r"] * te["dt"]).groupby(te["breath_id"]).cumsum()
    ).astype(np.float32)

    tr["u_in_cum_r"] = tr["u_in_cum"].round(2)
    te["u_in_cum_r"] = te["u_in_cum"].round(2)
    tr["u_in_area_r"] = tr["u_in_area"].round(4)
    te["u_in_area_r"] = te["u_in_area"].round(4)
    tr["u_in_area_r2"] = tr["u_in_area"].round(2)
    te["u_in_area_r2"] = te["u_in_area"].round(2)

    tr["u_in_cum_r1"] = tr["u_in_cum"].round(1)
    te["u_in_cum_r1"] = te["u_in_cum"].round(1)
    tr["u_in_area_r1"] = tr["u_in_area"].round(1)
    te["u_in_area_r1"] = te["u_in_area"].round(1)

    tr["u_in_diff"] = tr.groupby("breath_id")["u_in_r"].diff().fillna(0.0).round(1)
    te["u_in_diff"] = te.groupby("breath_id")["u_in_r"].diff().fillna(0.0).round(1)

    tr_reset = tr["u_out"].eq(1).groupby(tr["breath_id"], sort=False).cumsum()
    te_reset = te["u_out"].eq(1).groupby(te["breath_id"], sort=False).cumsum()
    tr["since_out1"] = (
        tr.groupby(["breath_id", tr_reset], sort=False).cumcount().astype(np.int16)
    )
    te["since_out1"] = (
        te.groupby(["breath_id", te_reset], sort=False).cumcount().astype(np.int16)
    )

    for lag in (1, 2, 3):
        tr[f"u_in_l{lag}"] = tr.groupby("breath_id")["u_in_r"].shift(lag).fillna(0.0)
        te[f"u_in_l{lag}"] = te.groupby("breath_id")["u_in_r"].shift(lag).fillna(0.0)
        tr[f"u_out_l{lag}"] = (
            tr.groupby("breath_id")["u_out"].shift(lag).fillna(0).astype(np.int8)
        )
        te[f"u_out_l{lag}"] = (
            te.groupby("breath_id")["u_out"].shift(lag).fillna(0).astype(np.int8)
        )

    tr["pressure_l1"] = (
        tr.groupby("breath_id")["pressure"].shift(1).fillna(tr["pressure"])
    )
    tr["pressure_l1"] = tr["pressure_l1"].astype(np.float32)
    tr["pressure_l1_r"] = tr["pressure_l1"].round(2)

    insp_tr = tr["u_out"].eq(0)
    global_median = (
        float(tr.loc[insp_tr, "pressure"].median())
        if insp_tr.any()
        else float(tr["pressure"].median())
    )

    pred_sorted = np.full(len(te), global_median, dtype=float)

    backoffs = [
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r",
            "pressure_l1_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r2",
            "pressure_l1_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_cum_r1",
            "u_in_area_r1",
            "pressure_l1_r",
        ],
        ["R", "C", "step", "u_out", "u_in_r", "u_in_l1", "u_out_l1", "pressure_l1_r"],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_cum_r",
            "u_in_area_r2",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_l1",
            "u_out_l1",
            "u_in_cum_r1",
            "u_in_area_r1",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_area_r2",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_cum_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
        ],
        ["R", "C", "step", "u_out", "since_out1", "u_in_r"],
        ["R", "C", "step", "u_out"],
        ["R", "C", "step", "u_in_r"],
    ]

    lookups = []
    for cols in backoffs:
        use_insp_only = ("u_out" in cols) or (cols == ["R", "C", "step", "u_in_r"])
        tr_sub = tr.loc[insp_tr] if use_insp_only and insp_tr.any() else tr
        ser = tr_sub.groupby(cols, sort=False)["pressure"].median()
        lookups.append(ser.to_dict())

    all_needed = sorted({c for cols in backoffs for c in cols if c != "pressure_l1_r"})
    te_arrays = {c: te[c].to_numpy() for c in all_needed}

    te_breath = te["breath_id"].to_numpy()
    change = np.empty(len(te_breath), dtype=bool)
    change[0] = True
    change[1:] = te_breath[1:] != te_breath[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(te_breath)]

    for s, e in zip(starts, ends):
        prev_p = global_median
        for i in range(s, e):
            pressure_l1_r = float(np.round(prev_p, 2))

            found = False
            for cols, lookup in zip(backoffs, lookups):
                if "pressure_l1_r" in cols:
                    key = tuple(
                        te_arrays[c][i] for c in cols if c != "pressure_l1_r"
                    ) + (pressure_l1_r,)
                else:
                    key = tuple(te_arrays[c][i] for c in cols)

                val = lookup.get(key, None)
                if val is not None and not pd.isna(val):
                    pred_sorted[i] = float(val)
                    found = True
                    break
            if not found:
                pred_sorted[i] = global_median

            prev_p = pred_sorted[i]

    out_df = te[["id", "breath_id", "step", "u_out"]].copy()
    out_df["pressure_pred"] = pred_sorted

    out_df["is_exp"] = out_df["u_out"].eq(1)

    exp_start = (
        out_df.loc[out_df["is_exp"], ["breath_id", "step", "pressure_pred"]]
        .sort_values(["breath_id", "step"], kind="mergesort")
        .groupby("breath_id", sort=False)
        .first()
        .rename(columns={"pressure_pred": "exp_start_pred"})
    )
    out_df = out_df.merge(exp_start[["exp_start_pred"]], on="breath_id", how="left")

    last_insp = (
        out_df.loc[~out_df["is_exp"], ["breath_id", "step", "pressure_pred"]]
        .sort_values(["breath_id", "step"], kind="mergesort")
        .groupby("breath_id", sort=False)
        .last()
        .rename(columns={"pressure_pred": "last_insp_pred"})
    )
    out_df = out_df.merge(last_insp[["last_insp_pred"]], on="breath_id", how="left")

    stable_exp = out_df["exp_start_pred"].where(
        ~out_df["exp_start_pred"].isna(), out_df["last_insp_pred"]
    )
    out_df.loc[out_df["is_exp"], "pressure_pred"] = stable_exp.loc[
        out_df["is_exp"]
    ].to_numpy()

    out_df = out_df[["id", "pressure_pred"]]
    return out_df


def g(dp):
    """
    Main blender:
    - If external submission csvs exist in dp: blend them as before.
    - Else: use train-derived median lookup fallback.
    Always writes submission.csv with id,pressure.

    Speed fix:
    - Replace slow scalar apply(find_nearest) with vectorized snapping.
    """
    files = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                files.append(i)
    files.sort()

    loop_time = 125
    splits = 2

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if len(files) == 0:
        df_test = pd.read_csv(
            "../input/ventilator-pressure-prediction/test.csv",
            usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
        )
        pred_df = _fallback_predict_from_train_lookup(df_train, df_test)

        output = output.merge(pred_df, on="id", how="left", validate="one_to_one")
        output["pressure"] = output["pressure_pred"].astype(float)
        output.drop(columns=["pressure_pred"], inplace=True)

        output["pressure"] = find_nearest_vec(output["pressure"].to_numpy())
        output.to_csv("submission.csv", index=False)
        return

    flist = []
    step = round(len(files) / splits) if splits > 0 else len(files)
    for i in range(splits):
        if i == splits - 1:
            flist.append(files[i * step :])
        else:
            flist.append(files[i * step : (i + 1) * step])

    blended = []
    for part in flist:
        pred = wc(part)
        if pred is not None:
            blended.append(pred)

    if len(blended) == 0:
        df_test = pd.read_csv(
            "../input/ventilator-pressure-prediction/test.csv",
            usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
        )
        pred_df = _fallback_predict_from_train_lookup(df_train, df_test)

        output = output.merge(pred_df, on="id", how="left", validate="one_to_one")
        output["pressure"] = output["pressure_pred"].astype(float)
        output.drop(columns=["pressure_pred"], inplace=True)

        output["pressure"] = find_nearest_vec(output["pressure"].to_numpy())
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(blended))]
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            weight = [w / weight_sum for w in weight]

        weight.sort(reverse=True)

        temp = 0.0
        for j in range(len(blended)):
            temp += blended[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = find_nearest_vec(output["pressure"].to_numpy())
    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
