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

0.1417941441510536

# 6. Current score

5.01493

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The crash happens because the blending directory is empty/unavailable, so `pred_list` ends up containing arrays of the wrong shape (or none), and `np.median(np.vstack(pred_list), ...)` produces a length-1 result that can’t be assigned to the 603,600-row submission. I make `g()` robust by (1) verifying the input directory and reading only valid submission-like CSVs, (2) enforcing that every loaded prediction is aligned to `sample_submission` `id` and has the correct length, and (3) providing a safe fallback prediction (all zeros snapped to nearest valid pressure) so a valid `.csv` is always produced. These fixes are score-neutral relative to the intended blending approach; they only prevent runtime errors and ensure a submission file is written. I also fix the notebook cell numbering to start from 1 so it runs cleanly in a script-like environment.'
- What this solution (achieved 6.10591) has done: 'Your current score (17.65 MAE) is far worse than the target (0.1418), so the biggest issue is that the script is blending (or falling back to zeros) without actually generating any meaningful model predictions in this environment. The most minimal legitimate improvement is to keep your existing blending core, but add an in-notebook baseline predictor that uses only the provided train.csv to compute a per-(R,C,u_out) mean pressure by time_step, then uses that as a “pseudo-submission” to blend when no external blend files exist. This preserves your blending logic and snapping-to-valid-pressures postprocess, but ensures `g()` always has at least one strong prediction source available, moving the MAE dramatically toward the target. The submission file naming and schema remain unchanged (`rwb 154 loops.csv` with `id,pressure`).'
- What this solution (achieved 5.57107) has done: 'Your current MAE (6.1059) is still far above the target (0.1418), so we should improve predictions while keeping your blending + snapping core intact. The biggest low-risk gain is to make the built-in baseline much stronger by conditioning on inspiratory history (`u_in` lag and cumulative integral) in addition to `(R,C,u_out,time_step)`, because pressure depends heavily on the prior control trajectory. I keep the same training-free “statistics from train.csv” approach (no new model/loops/loss), but extend the grouping keys minimally and add a safe fallback chain so it always produces a valid submission. I also remove a couple of redundant lines in the baseline alignment that currently do extra work but don’t change outputs, to keep runtime within limits.'
- What this solution (achieved 0.6643) has done: 'Your current MAE (5.57) is far above the target (0.1418), so we should improve the built-in baseline prediction (used when no external blend files exist) while keeping your blending and snapping-to-valid-pressures core intact. The most minimal high-impact fix is to replace the coarse “bucketed group-mean” baseline with a still-training-free nearest-neighbor lookup over the full control trajectory within each breath: for each test breath, find the closest train breath among matching (R,C) using u_in/u_out sequences, then reuse that train breath’s pressure curve. This stays within your original “statistics/lookup from train.csv” approach (no model, no training loops, no loss changes), but leverages the key fact that many breaths repeat very similar trajectories, dramatically reducing MAE. I also keep the original group-mean baseline as a fallback for any breath where the nearest-neighbor search fails, ensuring the script always produces a valid submission CSV.'
- What this solution (achieved 0.6643) has done: 'Your current MAE (0.6643, lower is better) is still far above the target (0.1418), so we should improve predictions while preserving your existing “training-free lookup + blending + snap-to-valid-pressures” core. The smallest high-impact fix is to keep your nearest-neighbor breath retrieval, but make it robust to repeated/control-identical breaths by using an exact-hash match on the full (u_in,u_out) trajectory within each (R,C) first, and only fall back to the current L2 nearest-neighbor when no exact match exists. This typically reduces error a lot on this competition because many breaths repeat the same control program, and it doesn’t change your overall approach (still a deterministic lookup from train.csv, no training loop/model). I also fix the cell numbering to start at 1 so the script runs cleanly in your required “cells” format, and keep the submission filename/format unchanged.'
- What this solution (achieved 0.6643) has done: 'Your current MAE (0.6643, lower is better) is still far above the target (0.1418), so we should improve the built-in baseline while keeping your “training-free lookup + blending + snap-to-valid-pressures” core intact. The minimal high-impact change is to strengthen the exact-match stage by making the trajectory key deterministic and robust: quantize `u_in` to the train/test rounding resolution (2 decimals) before hashing, and build the exact-match dictionary from train breaths using a stable bytes key, which typically increases the exact-match hit rate significantly. Then, when no exact match exists, keep your existing L2 nearest-neighbor fallback unchanged, and still fall back to the time-bucket baseline for any missing ids. This should reduce MAE toward your target without changing model/training semantics (still pure lookup from train.csv, no training loop).'
- What this solution (achieved 0.6643) has done: 'I keep your current “training-free lookup + blending + snap-to-valid-pressures” core intact, but strengthen the exact-match retrieval so more test breaths can reuse an identical train pressure curve (which is a big driver of MAE reduction on this competition). Specifically, I make the trajectory key include a robust quantization for both `u_in` and `u_out`, and also include `time_step` (rounded) in the key to prevent any subtle alignment mismatches across breaths. When multiple train breaths share the same key, I store and average their pressure curves (same as you already do) for stability, while keeping your L2 nearest-neighbor fallback unchanged. These are minimal, localized changes inside `_build_nn_breath_predictions` and should move the MAE down from 0.6643 toward your target.'
- What this solution (achieved 5.57107) has done: 'Your current MAE (0.6643) is still far above the target (0.1418), so we need a stronger prediction while keeping your existing “training-free lookup + blending + snap-to-valid-pressures” core intact. The most minimal high-impact fix is to replace the expensive L2 nearest-neighbor fallback (which tends to be noisy and slow) with a deterministic “exact match → per-(R,C) trajectory cluster mean” fallback: for each (R,C), quantize the full (time_step,u_in,u_out) trajectory and average the corresponding train pressure curves for that key, which is closer to the true mapping on this dataset. This stays within the same non-training lookup paradigm and keeps your exact-match path and time-bucket baseline fallback, but should reduce error substantially toward the target. I also keep submission writing unchanged and ensure predictions always align to `sample_submission` ids.'
- What this solution (achieved 6.10271) has done: 'Your current MAE (5.571) is far above the target (0.1418), so the main issue is that your “exact/relaxed trajectory match” is effectively not matching in this environment and you fall back to a weak global baseline. I keep your overall training-free lookup + blending + snap-to-nearest-valid-pressure logic identical, but make the trajectory key more matchable by removing `time_step` from the hash (it’s redundant given fixed 80 steps and can cause mismatches due to float rounding differences). Then I add an additional very-small-change fallback inside each (R,C): a per-step mean pressure curve by `(R,C,u_out,time_step_rounded)` so even when no full-trajectory match exists, predictions are much closer than the current coarse baseline. These changes are localized to `_build_nn_breath_predictions` and preserve the rest of your pipeline and submission writing unchanged.'
- What this solution (achieved 5.01493) has done: 'Your current MAE (6.10) is far above the target (0.1418), and the most likely cause is that the breath-trajectory matching is failing often and you’re effectively falling back to a weak baseline. I keep your overall “training-free lookup + blending + snap-to-valid-pressures” structure unchanged, but fix a key bug in the per-breath fallback (using a slow/fragile `.apply` on rows) and strengthen the fallback hierarchy inside each (R,C) block by adding a deterministic per-step mean curve that conditions on `u_in` (rounded) in addition to `(u_out, time_step_rounded)`. This is still the same core logic (no model/training), but it should greatly reduce fallback error and move the score down toward the target. I also ensure `pred_by_id` is built in a way that guarantees full coverage/alignment to `sample_submission` ids before writing the CSV.'

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


def _load_pred_file(fp, sample_ids):
    """
    Bugfix: robustly load a prediction file and align to sample_submission ids.
    Returns a numpy array of shape (n_rows,) or None if invalid.
    """
    try:
        df = pd.read_csv(fp)
    except Exception:
        return None

    if "pressure" not in df.columns:
        return None

    if "id" in df.columns:
        try:
            df2 = df[["id", "pressure"]].copy()
            df2 = df2.dropna(subset=["id", "pressure"])
            df2["id"] = df2["id"].astype(np.int64, errors="ignore")
            df2 = df2.set_index("id")
            aligned = df2.reindex(sample_ids)["pressure"].to_numpy()
        except Exception:
            aligned = df["pressure"].to_numpy()
    else:
        aligned = df["pressure"].to_numpy()

    if aligned.ndim != 1:
        aligned = aligned.reshape(-1)

    if len(aligned) != len(sample_ids):
        return None

    return aligned.astype(np.float64, copy=False)


def wc(input_list):
    l = []
    preds = []
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()

    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        pred = _load_pred_file(fp, sample_ids)
        if pred is None:
            continue
        l.append(public_lb_score)
        preds.append(pred)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _make_breath_features(df, is_train=False):
    """
    Existing baseline feature engineering (kept) for fallback group-mean lookup.
    """
    out = df.copy()
    out["ts2"] = np.round(out["time_step"].to_numpy(), 2)
    out["u_in_lag1"] = out.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
    out["u_in_cum"] = out.groupby("breath_id", sort=False)["u_in"].cumsum()
    out["u_in_lag1_b"] = np.round(out["u_in_lag1"].to_numpy(), 1)
    out["u_in_cum_b"] = np.round(out["u_in_cum"].to_numpy(), 1)
    if is_train:
        out["pressure"] = out["pressure"].astype(np.float64)
    return out


def _build_timebucket_baseline_predictions(sample_ids):
    """
    Existing (kept) training-free baseline for fallback.
    """
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    tr = _make_breath_features(
        df_train[["breath_id", "R", "C", "u_out", "time_step", "u_in", "pressure"]],
        is_train=True,
    )
    te = _make_breath_features(
        test[["id", "breath_id", "R", "C", "u_out", "time_step", "u_in"]],
        is_train=False,
    )

    mean_map_hist = (
        tr.groupby(["R", "C", "u_out", "ts2", "u_in_lag1_b", "u_in_cum_b"], sort=False)[
            "pressure"
        ]
        .mean()
        .astype(np.float64)
    )

    mean_map_ts = (
        tr.groupby(["R", "C", "u_out", "ts2"], sort=False)["pressure"]
        .mean()
        .astype(np.float64)
    )
    mean_rcu = (
        tr.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .mean()
        .astype(np.float64)
    )
    global_mean = float(tr["pressure"].mean())

    te = te.join(
        mean_map_hist.rename("p_hist"),
        on=["R", "C", "u_out", "ts2", "u_in_lag1_b", "u_in_cum_b"],
    )
    te = te.join(mean_map_ts.rename("p_ts"), on=["R", "C", "u_out", "ts2"])
    te = te.join(mean_rcu.rename("p_rcu"), on=["R", "C", "u_out"])

    pred = te["p_hist"].to_numpy(dtype=np.float64)
    m = np.isnan(pred)
    if m.any():
        pred[m] = te.loc[m, "p_ts"].to_numpy(dtype=np.float64)
    m2 = np.isnan(pred)
    if m2.any():
        pred[m2] = te.loc[m2, "p_rcu"].to_numpy(dtype=np.float64)
    m3 = np.isnan(pred)
    if m3.any():
        pred[m3] = global_mean

    te["pred_filled"] = pred
    pred_aligned = (
        te.set_index("id").reindex(sample_ids)["pred_filled"].to_numpy(dtype=np.float64)
    )
    pred_aligned = np.asarray(pred_aligned, dtype=np.float64).reshape(-1)
    return pred_aligned


def _build_nn_breath_predictions(sample_ids):
    """
    Keep the same lookup-only core, but improve match coverage and the per-(R,C) fallback:
    - Fix: replace slow/fragile row-wise apply with vectorized MultiIndex reindex.
    - Improve: when no full-trajectory match exists, use per-step mean curve conditioned on
      (u_out, time_step_rounded, u_in_rounded) before falling back to (u_out, time_step_rounded).
      This is still training-free aggregation from train.csv, but closer to true mapping.
    """
    test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )
    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()

    fallback_pred = _build_timebucket_baseline_predictions(sample_ids)

    pred_by_id = pd.Series(index=sample_ids, dtype=np.float64)

    def _traj_key(u_in_arr, u_out_arr, ui_dec=2):
        ui = np.ascontiguousarray(
            np.round(np.asarray(u_in_arr, dtype=np.float32), ui_dec)
        )
        uo = np.ascontiguousarray(np.asarray(u_out_arr, dtype=np.int8))
        return ui.tobytes() + b"|" + uo.tobytes()

    tr["_ts2"] = np.round(tr["time_step"].to_numpy(dtype=np.float64), 2)
    test["_ts2"] = np.round(test["time_step"].to_numpy(dtype=np.float64), 2)

    test["_uin2"] = np.round(test["u_in"].to_numpy(dtype=np.float64), 2)
    tr["_uin2"] = np.round(tr["u_in"].to_numpy(dtype=np.float64), 2)

    rc_keys = sorted(set(zip(test["R"].values.tolist(), test["C"].values.tolist())))
    for R, C in rc_keys:
        tr_block = tr[(tr["R"].values == R) & (tr["C"].values == C)]
        te_block = test[(test["R"].values == R) & (test["C"].values == C)]
        if te_block.empty or tr_block.empty:
            continue

        mean_step_uin = (
            tr_block.groupby(["u_out", "_ts2", "_uin2"], sort=False)["pressure"]
            .mean()
            .astype(np.float64)
        )
        mean_step = (
            tr_block.groupby(["u_out", "_ts2"], sort=False)["pressure"]
            .mean()
            .astype(np.float64)
        )

        tr_in = tr_block.groupby("breath_id", sort=False)["u_in"].apply(np.asarray)
        tr_out = tr_block.groupby("breath_id", sort=False)["u_out"].apply(np.asarray)
        tr_p = tr_block.groupby("breath_id", sort=False)["pressure"].apply(np.asarray)

        te_in = te_block.groupby("breath_id", sort=False)["u_in"].apply(np.asarray)
        te_out = te_block.groupby("breath_id", sort=False)["u_out"].apply(np.asarray)
        te_ids = te_block.groupby("breath_id", sort=False)["id"].apply(np.asarray)

        lengths_tr = tr_in.apply(len).to_numpy()
        if len(lengths_tr) == 0:
            continue
        modal_len = int(pd.Series(lengths_tr).mode().iloc[0])

        ok_tr = lengths_tr == modal_len
        tr_in = tr_in.loc[ok_tr]
        tr_out = tr_out.loc[ok_tr]
        tr_p = tr_p.loc[ok_tr]
        if tr_in.empty:
            continue

        lengths_te = te_in.apply(len).to_numpy()
        if len(lengths_te) == 0:
            continue
        ok_te = lengths_te == modal_len
        te_in_ok = te_in.loc[ok_te]
        te_out_ok = te_out.loc[ok_te]
        te_ids_ok = te_ids.loc[ok_te]
        if te_in_ok.empty:
            continue

        tr_keys = {}
        for bid in tr_in.index.to_numpy():
            k = _traj_key(tr_in.loc[bid], tr_out.loc[bid], ui_dec=2)
            p = np.asarray(tr_p.loc[bid], dtype=np.float32)
            if k in tr_keys:
                tr_keys[k][0] += p
                tr_keys[k][1] += 1
            else:
                tr_keys[k] = [p.copy(), 1]

        tr_keys_relaxed = {}
        for bid in tr_in.index.to_numpy():
            k2 = _traj_key(tr_in.loc[bid], tr_out.loc[bid], ui_dec=1)
            p = np.asarray(tr_p.loc[bid], dtype=np.float32)
            if k2 in tr_keys_relaxed:
                tr_keys_relaxed[k2][0] += p
                tr_keys_relaxed[k2][1] += 1
            else:
                tr_keys_relaxed[k2] = [p.copy(), 1]

        for bid in te_in_ok.index.to_numpy():
            k = _traj_key(te_in_ok.loc[bid], te_out_ok.loc[bid], ui_dec=2)
            if k in tr_keys:
                p_sum, cnt = tr_keys[k]
                p_avg = (p_sum / float(cnt)).astype(np.float64)
                pred_by_id.loc[te_ids_ok.loc[bid]] = p_avg
                continue

            k2 = _traj_key(te_in_ok.loc[bid], te_out_ok.loc[bid], ui_dec=1)
            if k2 in tr_keys_relaxed:
                p_sum, cnt = tr_keys_relaxed[k2]
                p_avg = (p_sum / float(cnt)).astype(np.float64)
                pred_by_id.loc[te_ids_ok.loc[bid]] = p_avg
                continue

            ids_arr = te_ids_ok.loc[bid]
            tmp = (
                te_block.loc[
                    te_block["id"].isin(ids_arr), ["id", "u_out", "_ts2", "_uin2"]
                ]
                .set_index("id")
                .reindex(ids_arr)
            )

            mi = pd.MultiIndex.from_arrays(
                [
                    tmp["u_out"].astype(np.int64).to_numpy(),
                    tmp["_ts2"].astype(np.float64).to_numpy(),
                    tmp["_uin2"].astype(np.float64).to_numpy(),
                ],
                names=["u_out", "_ts2", "_uin2"],
            )
            p = mean_step_uin.reindex(mi).to_numpy(dtype=np.float64)

            m = np.isnan(p)
            if m.any():
                mi2 = pd.MultiIndex.from_arrays(
                    [
                        tmp.loc[m, "u_out"].astype(np.int64).to_numpy(),
                        tmp.loc[m, "_ts2"].astype(np.float64).to_numpy(),
                    ],
                    names=["u_out", "_ts2"],
                )
                p[m] = mean_step.reindex(mi2).to_numpy(dtype=np.float64)

            pred_by_id.loc[ids_arr] = p

        del tr_block, te_block, tr_in, tr_out, tr_p, te_in, te_out, te_ids
        gc.collect()

    fb_map = pd.Series(fallback_pred, index=sample_ids, dtype=np.float64)
    missing = pred_by_id.isna()
    if missing.any():
        pred_by_id.loc[missing] = fb_map.loc[pred_by_id.index[missing]].to_numpy(
            dtype=np.float64
        )

    pred_aligned = pred_by_id.reindex(sample_ids).to_numpy(dtype=np.float64)
    return pred_aligned


def g(dp):
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()
    n = len(sample)

    l = []
    if os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                l.append(i)

    file_count = len(l)
    loop_time = 154

    if file_count == 0:
        baseline_pred = _build_nn_breath_predictions(sample_ids)
        merged = [baseline_pred]
    else:
        splits = max(1, file_count // 2)
        l.sort()
        flist = []
        for i in range(splits):
            if i == splits - 1:
                flist.append(l[i * round(len(l) / splits) :])
            else:
                flist.append(
                    l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
                )

        merged = []
        for i in range(len(flist)):
            pred = wc(flist[i])
            if pred is not None and len(pred) == n:
                merged.append(pred)

        if len(merged) == 0:
            baseline_pred = _build_nn_breath_predictions(sample_ids)
            merged = [baseline_pred]

    pred_list = []
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for j in range(len(merged)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(merged)):
            temp += merged[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = sample.copy()
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    sample_ids = sample["id"].to_numpy()

    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)

    if "id" in a_df.columns:
        a_p = a_df.set_index("id").reindex(sample_ids)["pressure"].to_numpy()
    else:
        a_p = a_df["pressure"].to_numpy()
    if "id" in b_df.columns:
        b_p = b_df.set_index("id").reindex(sample_ids)["pressure"].to_numpy()
    else:
        b_p = b_df["pressure"].to_numpy()

    out = sample.copy()
    out["pressure"] = a_p * 0.6 + b_p * 0.4
    out["pressure"] = out["pressure"].apply(find_nearest)
    out.to_csv("blend.csv", index=False)
    return out




## === cell 2
g("../input/gb-data-blending-recover")
