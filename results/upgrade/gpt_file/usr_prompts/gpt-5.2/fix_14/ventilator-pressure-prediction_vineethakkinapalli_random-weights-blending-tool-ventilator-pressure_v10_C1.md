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

0.1422687045761128

# 6. Current score

4.10759

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02327) has done: 'I fix the runtime error by making the blending code robust when the expected external “high-score submissions” folder is missing or contains fewer files than assumed (which is what triggers the IndexError). Since that external dataset is not available in your provided paths, I keep the same “blend multiple submissions then snap to nearest known pressure” core idea, but fall back to generating predictions from train data statistics (a per-(R,C,time_step,u_out) median pressure lookup) so the notebook runs end-to-end and writes a valid `submission.csv`. I also correct the cell numbering (starting at cell 1) and ensure the output file has the required `id,pressure` columns and `.csv` suffix. This should yield a non-trivial MAE (better than an all-zero submission) while staying within the constraints and without changing the evaluation semantics.'
- What this solution (achieved 4.00096) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that the fallback predictor ignores `u_in` (the most informative control signal), so it collapses many distinct states to the same median pressure. I keep the same overall “fallback lookup from train statistics + snap to nearest known pressure” core logic, but minimally enrich the lookup keys with a rounded `u_in` bin so predictions vary appropriately with valve opening. I also make the merge hierarchical (fine lookup → coarser lookup → global median) to avoid NaNs while still using the best available match. This should substantially reduce MAE toward your target without changing the approach (still pure train-statistics lookup, no model training).'
- What this solution (achieved 4.00096) has done: 'Your current MAE (4.00096) is still far above the target (0.1423), so we should improve it, but with minimal changes and without introducing a new model. The biggest remaining mismatch is that the fallback lookup predicts pressure for all time steps, while the competition only scores inspiratory steps (u_out==0); during expiratory steps we should avoid adding noise by forcing a stable value (commonly 0) which improves overall MAE. I keep your existing “train-statistics lookup + hierarchical backoff + snap to nearest known pressure” core logic, but (1) make the lookup *only for u_out==0* and (2) set predictions to 0 for u_out==1. This is a small, metric-aligned post-processing change and should move the score substantially toward your target band.'
- What this solution (achieved 2.92162) has done: 'Your current MAE (4.00096, lower is better) is still far above the target (0.1423), so we should improve it with minimal, metric-aligned changes while keeping your core “train-statistics lookup + hierarchical backoff + snap-to-nearest pressure” approach. The biggest remaining gap is that the lookup does not use the most important dynamic information about the breath history; pressure depends strongly on cumulative flow/volume, not just instantaneous `u_in`. I minimally add two standard time-series features computed per `breath_id` (cumulative `u_in` integral and 1-step lag of `u_in`) and include them as additional rounded keys in the same hierarchical median lookup (fine → coarser → existing backoffs). This stays within your non-modeling lookup paradigm, keeps the inspiratory-only handling (`u_out==1` forced to 0), and should materially reduce MAE toward the target band.'
- What this solution (achieved 2.92162) has done: 'Your current MAE (2.92162, lower is better) is still far above the target (0.1423), so we should improve it while keeping the same “train-statistics lookup + hierarchical backoff + snap-to-nearest pressure + u_out==1 forced to 0” core logic. The biggest remaining issue is that your breath-history features (`cum_uin`, `lag_uin`) are computed on `tr` after filtering to `u_out==0`, which breaks continuity across the breath and makes those features inconsistent with test (computed on all rows). I compute `dt/cum_uin/lag_uin` on the full train breath sequence first (matching test), then filter to `u_out==0` only for building medians—this is a minimal, metric-aligned fix that typically improves lookup hit-rate and reduces MAE. I also fix the paths to match your available `/kaggle/input/...` layout and keep everything else unchanged (including snapping and submission schema).'
- What this solution (achieved 2.95727) has done: 'Your current MAE (2.92162, lower is better) is still far above the target (0.1423), so we should improve the fallback lookup while keeping the same overall “train-statistics lookup + hierarchical backoff + u_out==1 forced to 0 + snap to nearest known pressure” approach. The minimal high-impact fix here is to correct a major feature mismatch: `time_step` and `dt` are being rounded/used in a way that can desynchronize cumulative features vs. the keys you merge on, and the lookup keys are currently too sparse around `dt` variability. I (1) compute `dt` from a consistently rounded time grid (same in train/test) and add a rounded `dt` key into the same hierarchical median lookup (fine → coarser), and (2) ensure the merge is performed on identically-typed key columns to avoid silent merge misses. This keeps your core logic intact (still pure median lookups + snapping), but should significantly increase exact-match hit rate and reduce MAE toward your target band.'
- What this solution (achieved 3.93945) has done: 'Your current MAE (2.95727, lower is better) is far above the target, so we should improve accuracy with very small, metric-aligned changes while keeping your same “train-statistics hierarchical median lookup + u_out==1 forced to 0 + snap to nearest train pressure” approach. The biggest easy win without changing the paradigm is to (1) snap the *lookup medians* themselves to the discrete pressure grid (so merges return physically valid pressure levels directly), and (2) add one more minimal breath-history key (`prev_pressure` from train, and its test-time proxy via merged `lag_uin` medians) to increase specificity but still remain pure lookup/backoff. This improves match quality for inspiratory steps and should reduce MAE toward the target while preserving the overall logic and submission semantics. All paths and output schema remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved 3.82152) has done: 'Your current score (3.93945, lower is better) is still far above the target (0.1423), so we should improve the fallback lookup accuracy with minimal, metric-aligned fixes while keeping your same “hierarchical median lookup + u_out==1 forced to 0 + snap to nearest pressure grid” approach. The biggest issue is that `prevpr` for test is currently derived from a global `lag_uin`→`prev_p` median, which is a weak proxy and breaks the intended “previous pressure” conditioning. I instead estimate `prevpr` for test *sequentially within each breath* by using your already-built lookup tables: first predict pressure at each time step, then set next step’s `prevpr` to the previous predicted pressure; this keeps the exact same lookup paradigm but makes the key consistent and much more informative. I also ensure all lookup tables are built once from train and snapped to the discrete pressure grid, then applied in a single pass per breath to stay within time limits and keep output schema identical.'
- What this solution (achieved 3.79945) has done: 'Your current MAE (3.82152, lower is better) is still far above the target (0.1423), so we should improve it with minimal, metric-aligned fixes while keeping your same “hierarchical median lookup + u_out==1 forced to 0 + snap-to-pressure-grid + sequential prevpr per breath” core logic. The biggest issue is that your lookup keys are over-specified (especially `cumu` and `lag` at 0.1 resolution), causing many test rows to miss the fine tables and fall back to coarse/global medians. I keep the same hierarchy and features but slightly coarsen the rounding for `cumu` (and keep `lag/uin` as-is) and add one extra intermediate backoff table that drops `cumu` before collapsing all the way to `(R,C,ts,uin)`; this increases hit-rate without changing the paradigm. These are small changes that typically reduce MAE substantially for this competition while preserving the exact submission format and runtime constraints.'
- What this solution (achieved 3.40409) has done: 'I keep your hierarchical train-median lookup + sequential `prevpr` per breath + `u_out==1 -> 0` + snap-to-pressure-grid exactly as-is, and only make small, metric-aligned changes to reduce merge misses that are forcing you into coarse/global fallbacks (a key reason MAE is stuck around ~3.8). Specifically, I (1) stop rounding `u_in` for the lookup keys (use the raw float value) and only discretize it into a small integer bin for grouping, (2) add a tiny “local interpolation” backoff: when the exact `uin_bin` key is missing, try the nearest neighboring bins (±1) before collapsing to coarser tables, and (3) compute and use a more stable cumulative-volume proxy `cum_uin_dt` based on the unrounded `time_step` diff (but still keep your same feature set and backoff structure). These are minimal changes that typically increase hit-rate dramatically without introducing any new model/training, and should move the score downward toward your target. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.09773) has done: 'Your current MAE (3.40409; lower is better) is still far from the target (0.1423), so we should reduce error with minimal, metric-aligned changes while keeping your exact “hierarchical median lookup + sequential prevpr + u_out==1→0 + snap-to-pressure-grid” approach. The biggest accuracy loss now is (a) using an overly-coarse `uin_bin` (0.2 step) which smears pressure levels, and (b) merging on `ts`/`dtr` rounded too coarsely (0.01–0.02) which increases table sparsity and forces fallbacks. I only tighten discretization to increase exact-match hit-rate: use `uin_bin` at 0.05 resolution, round `ts`/`dtr` to 3 decimals (matching the dataset’s typical 0.03 cadence), and keep your neighbor-bin backoff (±1) which now corresponds to a tiny ±0.05 change. Everything else (tables, hierarchy, sequential inference, snapping, submission schema/path) stays the same.'
- What this solution (achieved 4.10759) has done: 'Your current MAE (4.09773, lower is better) is still far above the target (0.1423), so we should improve the lookup accuracy without changing your overall “hierarchical train-median lookup + sequential prevpr + u_out==1→0 + snap-to-pressure-grid” approach. The least invasive high-impact fix is to stop forcing `cumu` into an `int16` (it can overflow and silently corrupt keys), and to use a slightly finer but still compact discretization for cumulative volume so the keys remain consistent and informative. I also keep your existing hierarchy and neighbor-bin backoff unchanged, but make key dtypes consistent and stable to reduce merge misses. These changes keep the same core logic and should move the score downward (better) toward the target band while still running within time and producing a valid `submission.csv`.'
- What this solution (achieved 4.10759) has done: 'Your current MAE (4.10759; lower is better) is still far above the target (0.1423), so the goal is to reduce error with minimal, metric-aligned changes while keeping your exact “hierarchical train-median lookup + sequential prevpr per breath + u_out==1→0 + snap-to-pressure-grid” approach. The biggest low-risk win is to add a very small extra backoff step that uses `u_in` more faithfully when the fine tables miss: build additional medians keyed on a finer `u_in` bin (0.01 resolution) but only for the coarser tables (so memory stays reasonable), and try those tables before falling back to `(R,C,ts)`/global. This increases lookup hit-rate without changing the core logic, training, or metric semantics. I also tighten dtype consistency for the lookup keys (notably `uin_bin_fine`) to reduce silent merge/get misses that inflate MAE.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import random
from random import random as rd
import gc



## === cell 1
df_train = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
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
    """
    Weighted-combine of 1+ submission files; safe when folder missing/empty.
    """
    preds = []
    scores = []
    allow = [1348, 1698]

    for p in input_list:
        try:
            lb = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            lb = None

        if lb is None or lb in allow:
            df_sub = pd.read_csv(p)
            if "pressure" not in df_sub.columns:
                continue
            preds.append(df_sub["pressure"].to_numpy())
            if lb is not None:
                scores.append(lb)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    if len(scores) == len(preds) and len(scores) >= 2:
        w = np.array(scores, dtype=float)
        w = w / w.sum()
        if len(w) == 2:
            w0 = float(w[1] + 0.1)
            w1 = 1.0 - w0
            w = np.array([w0, w1], dtype=float)
            w = w / w.sum()
    else:
        w = np.ones(len(preds), dtype=float) / len(preds)

    out = np.zeros_like(preds[0], dtype=float)
    for wi, pi in zip(w, preds):
        out += wi * pi
    return out


def g(dp):
    """
    Core logic preserved:
      - Try blending external submissions if present.
      - Else: hierarchical median lookup from train + u_out==1 forced to 0 + snap to pressure grid.
      - prevpr for test generated sequentially per breath from previous predicted pressure.

    Minimal score-improving fixes (toward target, lower is better):
      1) Add an additional intermediate backoff keyed on a *finer* u_in bin (0.01) for coarser tables only.
         This keeps the same lookup paradigm but reduces fallback-to-global when u_in is informative.
      2) Keep existing fine-table keys/rounding; only add the extra tables and try them before med4/global.
    """
    files = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                files.append(i)
    files.sort()

    blended_base = None
    if len(files) > 0:
        splits = 2
        flist = []
        for s in range(splits):
            start = s * round(len(files) / splits)
            end = None if s == splits - 1 else (s + 1) * round(len(files) / splits)
            flist.append(files[start:end])

        preds_chunks = []
        for chunk in flist:
            arr = wc(chunk)
            if arr is not None:
                preds_chunks.append(arr)

        if len(preds_chunks) == 1:
            blended_base = preds_chunks[0]
        elif len(preds_chunks) >= 2:
            loop_time = 125
            pred_list = []
            for seed in range(loop_time):
                set_seed(seed)
                weight = [rd() for _ in range(len(preds_chunks))]
                weight_sum = sum(weight)
                weight = [w / weight_sum for w in weight]
                weight.sort(reverse=True)
                temp = 0.0
                for j in range(len(preds_chunks)):
                    temp += preds_chunks[j] * weight[j]
                pred_list.append(temp)
                del temp
                gc.collect()
            blended_base = np.median(np.vstack(pred_list), axis=0)

    if blended_base is None:
        df_test = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")

        tr_full = df_train[
            ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
        ].copy()
        te = df_test[["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]].copy()

        tr_full = tr_full.sort_values(["breath_id", "time_step"], kind="mergesort")
        te = te.sort_values(["breath_id", "time_step"], kind="mergesort")

        tr_full["ts"] = tr_full["time_step"].round(3).astype(np.float32)
        te["ts"] = te["time_step"].round(3).astype(np.float32)

        uin_bin_scale = 20.0  # 1/0.05
        tr_full["uin_bin"] = np.rint(tr_full["u_in"] * uin_bin_scale).astype(np.int16)
        te["uin_bin"] = np.rint(te["u_in"] * uin_bin_scale).astype(np.int16)

        uin_bin_fine_scale = 100.0  # 1/0.01
        tr_full["uin_bin_fine"] = np.rint(tr_full["u_in"] * uin_bin_fine_scale).astype(
            np.int16
        )
        te["uin_bin_fine"] = np.rint(te["u_in"] * uin_bin_fine_scale).astype(np.int16)

        tr_full["dt"] = (
            tr_full.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        )
        te["dt"] = te.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)

        tr_full["dtr"] = tr_full["dt"].round(3).astype(np.float32)
        te["dtr"] = te["dt"].round(3).astype(np.float32)

        tr_full["cum_uin"] = (
            (tr_full["u_in"] * tr_full["dt"])
            .groupby(tr_full["breath_id"], sort=False)
            .cumsum()
        )
        te["cum_uin"] = (
            (te["u_in"] * te["dt"]).groupby(te["breath_id"], sort=False).cumsum()
        )

        tr_full["lag_uin"] = (
            tr_full.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )
        te["lag_uin"] = te.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)

        tr_full["cumu"] = np.rint(tr_full["cum_uin"] * 10.0).astype(
            np.int32
        )  # 0.1 resolution
        te["cumu"] = np.rint(te["cum_uin"] * 10.0).astype(np.int32)

        tr_full["lag"] = tr_full["lag_uin"].round(1).astype(np.float32)
        te["lag"] = te["lag_uin"].round(1).astype(np.float32)

        tr_full["prev_p"] = (
            tr_full.groupby("breath_id", sort=False)["pressure"].shift(1).fillna(0.0)
        )
        tr_full["prevpr"] = tr_full["prev_p"].apply(find_nearest).astype(np.float32)

        tr = tr_full[tr_full["u_out"] == 0].copy()

        for k in ["R", "C"]:
            tr[k] = tr[k].astype(np.int16)
            te[k] = te[k].astype(np.int16)

        def _snap_series(s: pd.Series) -> pd.Series:
            return s.apply(find_nearest)

        med0 = tr.groupby(
            ["R", "C", "ts", "dtr", "uin_bin", "cumu", "lag", "prevpr"], sort=False
        )["pressure"].median()
        med0 = _snap_series(med0)

        med1 = tr.groupby(
            ["R", "C", "ts", "dtr", "uin_bin", "cumu", "prevpr"], sort=False
        )["pressure"].median()
        med1 = _snap_series(med1)

        med1b = tr.groupby(
            ["R", "C", "ts", "dtr", "uin_bin", "lag", "prevpr"], sort=False
        )["pressure"].median()
        med1b = _snap_series(med1b)

        med2 = tr.groupby(["R", "C", "ts", "dtr", "uin_bin", "prevpr"], sort=False)[
            "pressure"
        ].median()
        med2 = _snap_series(med2)

        med3 = tr.groupby(["R", "C", "ts", "uin_bin"], sort=False)["pressure"].median()
        med3 = _snap_series(med3)

        med3f = tr.groupby(["R", "C", "ts", "uin_bin_fine"], sort=False)[
            "pressure"
        ].median()
        med3f = _snap_series(med3f)

        med2f = tr.groupby(
            ["R", "C", "ts", "dtr", "uin_bin_fine", "prevpr"], sort=False
        )["pressure"].median()
        med2f = _snap_series(med2f)

        med4 = tr.groupby(["R", "C", "ts"], sort=False)["pressure"].median()
        med4 = _snap_series(med4)

        global_med = find_nearest(float(tr["pressure"].median()))

        te_pred = te.copy()
        te_pred["pressure"] = np.nan

        def _try_uin_neighbors(getter, uin_bin, max_delta=1):
            v = getter(uin_bin)
            if not pd.isna(v):
                return v
            for d in range(1, max_delta + 1):
                v = getter(uin_bin - d)
                if not pd.isna(v):
                    return v
                v = getter(uin_bin + d)
                if not pd.isna(v):
                    return v
            return np.nan

        for bid, idx in te_pred.groupby("breath_id", sort=False).indices.items():
            prevpr = 0.0
            for i in idx:
                if int(te_pred.at[i, "u_out"]) == 1:
                    te_pred.at[i, "pressure"] = 0.0
                    prevpr = 0.0
                    continue

                R = int(te_pred.at[i, "R"])
                C = int(te_pred.at[i, "C"])
                ts = float(te_pred.at[i, "ts"])
                dtr = float(te_pred.at[i, "dtr"])
                uinb = int(te_pred.at[i, "uin_bin"])
                uinbf = int(te_pred.at[i, "uin_bin_fine"])
                cumu = int(te_pred.at[i, "cumu"])
                lag = float(te_pred.at[i, "lag"])
                pprev = float(find_nearest(prevpr))

                def _get0(ub):
                    return med0.get((R, C, ts, dtr, ub, cumu, lag, pprev), np.nan)

                p = _try_uin_neighbors(_get0, uinb, max_delta=1)

                if pd.isna(p):

                    def _get1(ub):
                        return med1.get((R, C, ts, dtr, ub, cumu, pprev), np.nan)

                    p = _try_uin_neighbors(_get1, uinb, max_delta=1)

                if pd.isna(p):

                    def _get1b(ub):
                        return med1b.get((R, C, ts, dtr, ub, lag, pprev), np.nan)

                    p = _try_uin_neighbors(_get1b, uinb, max_delta=1)

                if pd.isna(p):

                    def _get2(ub):
                        return med2.get((R, C, ts, dtr, ub, pprev), np.nan)

                    p = _try_uin_neighbors(_get2, uinb, max_delta=1)

                if pd.isna(p):

                    def _get2f(ubf):
                        return med2f.get((R, C, ts, dtr, ubf, pprev), np.nan)

                    p = _try_uin_neighbors(_get2f, uinbf, max_delta=2)

                if pd.isna(p):

                    def _get3(ub):
                        return med3.get((R, C, ts, ub), np.nan)

                    p = _try_uin_neighbors(_get3, uinb, max_delta=1)

                if pd.isna(p):

                    def _get3f(ubf):
                        return med3f.get((R, C, ts, ubf), np.nan)

                    p = _try_uin_neighbors(_get3f, uinbf, max_delta=2)

                if pd.isna(p):
                    p = med4.get((R, C, ts), global_med)

                p = float(p)
                p = find_nearest(p)

                te_pred.at[i, "pressure"] = p
                prevpr = p

        blended_base = te_pred.sort_values("id")["pressure"].to_numpy()

    output = pd.read_csv(
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output["pressure"] = blended_base
    output["pressure"] = output["pressure"].apply(find_nearest)
    output = output[["id", "pressure"]]
    output.to_csv("submission.csv", index=False)




## === cell 2
g("/kaggle/input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
