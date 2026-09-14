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

0.1369073225014495

# 6. Current score

4.46828

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the crash by making the blending function robust to missing/empty input folders and to submissions with the wrong row count, which currently causes `pred_list` to contain arrays of length 1 and breaks the assignment to the 603600-row submission. I also ensure the script always produces a valid `submission.csv` (with `id,pressure`) even when the external dataset `../input/gb-data-blending-recover` is unavailable in your environment. These changes keep the core logic (random-weight blending + median aggregation + snapping to nearest known pressure) intact, but add defensive loading, length checks, and a guaranteed fallback output. Finally, I keep the original output naming but also write `submission.csv` so Kaggle accept it.'
- What this solution (achieved 9.92376) has done: 'Your current score is far above the target (lower is better), and the main reason is that the pipeline effectively falls back to an all-zero (then “snap-to-nearest”) submission when the external blending folder isn’t available, which is catastrophically bad for MAE. To move toward the target with minimal change and without altering the overall “blend predictions then snap” semantics, I add a deterministic, lightweight fallback model that uses only `train.csv` to compute a per-(R,C,time_step,u_out) median pressure lookup and predicts test by merging on those keys (then snapping to the known pressure grid). This keeps the core idea of “aggregate then snap” and avoids heavy ML training while producing a dramatically better baseline than all zeros. If the blending folder exists and contains valid submissions, your original blending path is kept and used first; only if it fails/empty the fallback be used, ensuring stability and a valid `submission.csv`.'
- What this solution (achieved 9.72479) has done: 'I fix the `merge_asof` crash by ensuring both left and right frames are *globally* sorted by the `on` key (`time_step`) as required by pandas, while still keeping your fallback semantics (median by keys, then nearest time_step within `(R,C,u_out)`). I also make the fallback memory-safe by reading only needed columns and using stable sorting, and I keep the original blending path unchanged when the external folder exists. Finally, I ensure a valid `submission.csv` is always written with exactly `id,pressure` and the correct row count.'
- What this solution (achieved 7.7415) has done: 'Your score is far worse than the target (lower is better), and the main reason is that your fallback still can’t model the inspiratory dynamics well enough: it only uses a sparse lookup on `(R,C,u_out,time_step)` and then nearest-time matching, which misses the strong dependence on `u_in` and breath progression. Keeping your overall “no heavy ML, aggregate-from-train then snap-to-pressure-grid” approach, I minimally enrich the fallback by adding `u_in` to the median lookup and (only when needed) doing nearest matching in the joint space of `(time_step, u_in)` within each `(R,C,u_out)` group. This preserves your evaluation semantics (predict pressure per row, then snap to known pressure levels) while significantly improving predictions toward the target without changing the blending path when external files exist. I also keep the robust CSV writing and length checks unchanged.'
- What this solution (achieved 7.74269) has done: 'Your current MAE (7.7415, lower is better) is still far from the target, and the biggest issue is the fallback’s coarse nearest matching: it does nearest time_step, then (optionally) nearest u_in, but it does not use breath context (pressure is highly dependent on within-breath history). To move the score toward the target with minimal change and without introducing ML training, I keep your “aggregate-from-train then snap-to-known-pressure-grid” core semantics, but upgrade the fallback to a stronger lookup that uses within-breath engineered features (cumulative u_in area and lag u_in) while still predicting by nearest-neighbor matching inside each (R,C,u_out) group. I also fix a silent correctness bug in your environment paths: your data lives under `/kaggle/input/...`, so the current relative `../input/...` reads may be pulling the wrong files or failing depending on working directory; I make paths robust by trying both locations. These changes only affect the fallback path (used when the external blending folder is missing/invalid), keeping your original blending logic intact.'
- What this solution (achieved 7.74154) has done: 'Your current MAE is far worse than the target, so the fallback path (used when the external blending folder is missing/invalid) needs a small but meaningful accuracy lift without changing the overall “train-aggregate then snap-to-known-pressure-grid” approach. I keep your blend logic intact, but strengthen the fallback by (1) predicting only the inspiratory phase and explicitly setting expiratory (`u_out==1`) pressures to 0 (since those rows are not scored and this avoids polluting errors), and (2) adding a minimal, non-ML correction that leverages the strong physical relationship between pressure and integrated flow by using `u_in_cum/R` in the lookup keys (still median-aggregation + nearest matching). These are localized changes inside `_fallback_predict_from_train()` only and should move the score substantially toward your target while keeping runtime under the limit. The script still always write a valid `submission.csv` with exactly `id,pressure` and the correct row count.'
- What this solution (achieved 1.82387) has done: 'Your current MAE (7.74, lower is better) is still far from the target, so we need a modest accuracy lift in the fallback path (used when the external blend folder isn’t present/valid). Keeping your core “aggregate-from-train then snap-to-known-pressure-grid” logic intact, I make the fallback explicitly evaluate-aligned by training/merging only on inspiratory rows (`u_out==0`) and then filling expiratory predictions with a safe constant (still snapped) without letting expiratory rows distort the lookup tables. I also fix the biggest accuracy bug: setting `u_out==1` predictions to 0 hurts because Kaggle still includes your predictions at those indices in the file even if they’re not scored—using a neutral value closer to typical pressures (a per-(R,C) expiratory median) reduces collateral error if scoring mask differs and improves robustness. Finally, I make the nearest-neighbor step more faithful by using `merge_asof` on `u_in_cum_over_R` within `(R,C,time_step)` for missing values (instead of forcing exact `time_step`), which is still the same lookup+nearest semantics but with a more appropriate continuous key.'
- What this solution (achieved 3.8161) has done: 'Your current score (1.82387, lower is better) is still far above the target, so we need a small but meaningful accuracy lift in the fallback path (used when the external blending folder isn’t present/valid) without changing your overall “train-aggregate then snap-to-pressure-grid” semantics. I keep your blending logic untouched, but make the fallback lookup less sparse by adding a two-stage nearest fill that first matches on `(R,C,time_step,u_in_q)` and then (only if still missing) uses nearest on `u_in_q` within `(R,C,time_step)`; this preserves the same median-lookup + nearest-neighbor idea while better capturing the dominant `u_in` dependence. I also remove an expensive row-wise `.apply(lambda ...)` for expiratory mapping (no logic change) to keep runtime safely under 600s. The script still always write a valid `submission.csv` with exact `id,pressure` and the correct row count.'
- What this solution (achieved 3.8161) has done: 'Your current MAE (3.8161, lower is better) is still far above the target, and the largest win available without changing the overall “train-aggregate then snap-to-pressure-grid” fallback approach is to make the fallback explicitly mask/handle the inspiratory-only scoring: we should predict inspiratory (`u_out==0`) as before, but for expiratory (`u_out==1`) we should carry forward the last inspiratory prediction within each breath (a stable, history-consistent fill) instead of using a global/per-(R,C) median. This is a minimal semantic extension (still non-ML lookup + filling + snapping) that better matches the physical continuity of pressure and avoids injecting unrelated values into a large fraction of rows. I also keep your external blending path intact; these changes only apply when the blending folder is missing/invalid and the fallback is used. Finally, I ensure the final submission is in the original test row order by merging onto `sample_submission.csv` (as you already do) and then applying the within-breath carry-forward on that aligned frame.'
- What this solution (achieved 4.46828) has done: 'Your current MAE (3.8161, lower is better) is still far above the target, and the fastest way to move toward it without changing your overall “train-aggregate then snap-to-known-pressure-grid” fallback semantics is to make the fallback more evaluation-aligned and less sparse for inspiratory rows. I keep your external blending path untouched, but strengthen the fallback by (1) explicitly building the lookup tables from inspiratory-only rows and (2) adding a staged nearest-fill on `time_step` within `(R,C)` and then on `u_in` within `(R,C,time_step)` to reduce missingness without introducing ML training. I also keep your within-breath expiratory carry-forward fill (continuity-consistent) and ensure the final submission remains aligned to `sample_submission.csv` by `id`. All changes are localized to `_fallback_predict_from_train()` and preserve the existing “median aggregation + nearest matching + snap” core logic.'
- What this solution (achieved 4.46828) has done: 'I fix the `merge_asof` crash by ensuring the left/right frames are sorted exactly as pandas requires: globally by the `on` key while also grouping by the `by` keys, and I do this for both asof-merge stages (`time_step_q` and `u_in_q`). This is a minimal, localized bugfix inside the fallback path only, preserving your overall blend→median→snap core logic and keeping the external blending path unchanged when available. I also add a small guard to drop any NA keys before `merge_asof` (shouldn’t happen, but avoids hard crashes). The script then run end-to-end and always write a valid `submission.csv` with `id,pressure` and the correct row count.'

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
def _resolve_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        rel_path.replace("../input", "/kaggle/input"),
        "/kaggle/input/ventilator-pressure-prediction/" + os.path.basename(rel_path),
        "/kaggle/data/ventilator-pressure-prediction/" + os.path.basename(rel_path),
        "/kaggle/input/" + os.path.basename(rel_path),
        "/kaggle/data/" + os.path.basename(rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


df_train = pd.read_csv(
    _resolve_path("../input/ventilator-pressure-prediction/train.csv")
)
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


def _safe_read_pressure_csv(path, expected_len):
    """
    Bugfix: some files in the provided directory may be non-submission files or malformed
    (wrong length / missing column). Return None if invalid.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = np.asarray(df["pressure"]).ravel()
    if arr.shape[0] != expected_len:
        return None
    try:
        arr = arr.astype(np.float64, copy=False)
    except Exception:
        return None
    return arr


def wc(input_list, expected_len):
    """
    Keeps original intent: for a list of filepaths, read their pressure arrays and blend.
    Bugfix: if parsing of "public_lb_score" fails, use uniform weights instead of crashing.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        path = input_list[i]
        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        arr = _safe_read_pressure_csv(path, expected_len)
        if arr is None:
            continue
        l.append(public_lb_score)
        arrs.append(arr)

    if len(arrs) == 0:
        return None

    if len(arrs) == 1:
        return arrs[0]

    a0, a1 = arrs[0], arrs[1]
    l0, l1 = l[0], l[1]
    l_sum = l0 + l1 if (l0 + l1) != 0 else 1
    weight1 = (l1 / l_sum) + 0.05
    weight2 = 1.0 - weight1
    return a0 * weight1 + a1 * weight2


def _fallback_predict_from_train():
    """
    Bugfix: pandas.merge_asof requires both frames to be sorted primarily by the 'on' key.
    We sort by ['time_step_q','R','C'] (and similarly for u_in_q stage) and also drop any NA keys.
    This fixes: ValueError: left keys must be sorted

    Score intent preserved: aggregate-from-train then staged nearest-fill then snap.
    """
    train_path = _resolve_path("../input/ventilator-pressure-prediction/train.csv")
    test_path = _resolve_path("../input/ventilator-pressure-prediction/test.csv")
    sample_path = _resolve_path(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )

    for col in ["time_step", "u_in"]:
        train[col] = train[col].astype(np.float64)
        test[col] = test[col].astype(np.float64)

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    train_dt = (
        train.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .to_numpy()
    )
    test_dt = (
        test.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0).to_numpy()
    )

    train_u_in = train["u_in"].to_numpy()
    test_u_in = test["u_in"].to_numpy()

    train["u_in_lag1"] = (
        train.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float64)
    )
    test["u_in_lag1"] = (
        test.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float64)
    )

    train["u_in_cum"] = (train_u_in * train_dt).astype(np.float64)
    train["u_in_cum"] = (
        train.groupby("breath_id", sort=False)["u_in_cum"].cumsum().astype(np.float64)
    )

    test["u_in_cum"] = (test_u_in * test_dt).astype(np.float64)
    test["u_in_cum"] = (
        test.groupby("breath_id", sort=False)["u_in_cum"].cumsum().astype(np.float64)
    )

    train["u_in_cum_over_R"] = (
        train["u_in_cum"] / train["R"].astype(np.float64)
    ).astype(np.float64)
    test["u_in_cum_over_R"] = (test["u_in_cum"] / test["R"].astype(np.float64)).astype(
        np.float64
    )

    train["u_in_q"] = np.round(train["u_in"].to_numpy(), 2)
    test["u_in_q"] = np.round(test["u_in"].to_numpy(), 2)
    train["u_in_lag1_q"] = np.round(train["u_in_lag1"].to_numpy(), 2)
    test["u_in_lag1_q"] = np.round(test["u_in_lag1"].to_numpy(), 2)
    train["u_in_cum_over_R_q"] = np.round(train["u_in_cum_over_R"].to_numpy(), 2)
    test["u_in_cum_over_R_q"] = np.round(test["u_in_cum_over_R"].to_numpy(), 2)
    train["time_step_q"] = np.round(train["time_step"].to_numpy(), 2)
    test["time_step_q"] = np.round(test["time_step"].to_numpy(), 2)

    insp_train = train[train["u_out"] == 0].copy()

    pred_df = test[
        [
            "id",
            "breath_id",
            "R",
            "C",
            "u_out",
            "time_step",
            "time_step_q",
            "u_in_q",
            "u_in_lag1_q",
            "u_in_cum_over_R_q",
        ]
    ].copy()

    key_cols_exact = [
        "R",
        "C",
        "time_step_q",
        "u_in_q",
        "u_in_cum_over_R_q",
        "u_in_lag1_q",
    ]
    med_exact = (
        insp_train.groupby(key_cols_exact, as_index=False)["pressure"]
        .median()
        .rename(columns={"pressure": "pred"})
    )
    pred_df = pred_df.merge(med_exact, on=key_cols_exact, how="left")

    miss_mask2 = pred_df["pred"].isna() & (pred_df["u_out"] == 0)
    if miss_mask2.any():
        med_uin = (
            insp_train.groupby(["R", "C", "time_step_q", "u_in_q"], as_index=False)[
                "pressure"
            ]
            .median()
            .rename(columns={"pressure": "pred_uin"})
        )
        pred_df = pred_df.merge(
            med_uin, on=["R", "C", "time_step_q", "u_in_q"], how="left"
        )
        pred_df.loc[miss_mask2, "pred"] = pred_df.loc[miss_mask2, "pred_uin"].to_numpy()
        pred_df.drop(columns=["pred_uin"], inplace=True)

    miss_mask_ts = pred_df["pred"].isna() & (pred_df["u_out"] == 0)
    if miss_mask_ts.any():
        med_ts = (
            insp_train.groupby(["R", "C", "time_step_q"], as_index=False)["pressure"]
            .median()
            .rename(columns={"pressure": "pred_ts"})
        )

        med_ts_sorted = (
            med_ts.dropna(subset=["time_step_q", "R", "C"])
            .sort_values(["time_step_q", "R", "C"], kind="mergesort")
            .reset_index(drop=True)
        )

        miss = pred_df.loc[miss_mask_ts, ["id", "R", "C", "time_step_q"]].copy()
        miss = (
            miss.dropna(subset=["time_step_q", "R", "C"])
            .sort_values(["time_step_q", "R", "C"], kind="mergesort")
            .reset_index(drop=True)
        )

        if len(miss) > 0 and len(med_ts_sorted) > 0:
            miss2 = pd.merge_asof(
                miss,
                med_ts_sorted,
                on="time_step_q",
                by=["R", "C"],
                direction="nearest",
                allow_exact_matches=True,
            )
            id_to_pred = pd.Series(
                miss2["pred_ts"].to_numpy(), index=miss2["id"].to_numpy()
            )
            pred_df.loc[miss_mask_ts, "pred"] = (
                pred_df.loc[miss_mask_ts, "id"].map(id_to_pred).to_numpy()
            )

            del miss2, id_to_pred
        del med_ts, med_ts_sorted, miss
        gc.collect()

    miss_mask3 = pred_df["pred"].isna() & (pred_df["u_out"] == 0)
    if miss_mask3.any():
        uin_nn = (
            insp_train.groupby(["R", "C", "time_step_q", "u_in_q"], as_index=False)[
                "pressure"
            ]
            .median()
            .rename(columns={"pressure": "pred_uin_nn"})
        )

        uin_nn_sorted = (
            uin_nn.dropna(subset=["u_in_q", "R", "C", "time_step_q"])
            .sort_values(["u_in_q", "R", "C", "time_step_q"], kind="mergesort")
            .reset_index(drop=True)
        )

        miss = pred_df.loc[miss_mask3, ["id", "R", "C", "time_step_q", "u_in_q"]].copy()
        miss = (
            miss.dropna(subset=["u_in_q", "R", "C", "time_step_q"])
            .sort_values(["u_in_q", "R", "C", "time_step_q"], kind="mergesort")
            .reset_index(drop=True)
        )

        if len(miss) > 0 and len(uin_nn_sorted) > 0:
            miss2 = pd.merge_asof(
                miss,
                uin_nn_sorted,
                on="u_in_q",
                by=["R", "C", "time_step_q"],
                direction="nearest",
                allow_exact_matches=True,
            )

            id_to_pred = pd.Series(
                miss2["pred_uin_nn"].to_numpy(), index=miss2["id"].to_numpy()
            )
            pred_df.loc[miss_mask3, "pred"] = (
                pred_df.loc[miss_mask3, "id"].map(id_to_pred).to_numpy()
            )

            del miss2, id_to_pred
        del uin_nn, uin_nn_sorted, miss
        gc.collect()

    global_insp_med = float(insp_train["pressure"].median())
    pred_df.loc[pred_df["u_out"] == 0, "pred"] = (
        pred_df.loc[pred_df["u_out"] == 0, "pred"]
        .astype(np.float64)
        .fillna(global_insp_med)
    )

    pred_df["pred"] = pred_df["pred"].astype(np.float64)
    pred_df = pred_df.sort_values(
        ["breath_id", "time_step"], kind="mergesort"
    ).reset_index(drop=True)

    pred_df["pred_insp_only"] = pred_df["pred"].where(
        pred_df["u_out"].to_numpy() == 0, np.nan
    )
    pred_df["pred_ffill"] = pred_df.groupby("breath_id", sort=False)[
        "pred_insp_only"
    ].ffill()
    pred_df["pred_ffill"] = pred_df["pred_ffill"].fillna(global_insp_med)
    pred_df.loc[pred_df["u_out"] == 1, "pred"] = pred_df.loc[
        pred_df["u_out"] == 1, "pred_ffill"
    ].to_numpy()
    pred_df.drop(columns=["pred_insp_only", "pred_ffill"], inplace=True)

    sub = pd.read_csv(sample_path, usecols=["id", "pressure"])
    sub = sub.merge(pred_df[["id", "pred"]], on="id", how="left")
    sub["pressure"] = sub["pred"].astype(np.float64)
    sub.drop(columns=["pred"], inplace=True)

    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    """
    Bugfixes:
    - Handle missing/empty directories gracefully.
    - Filter out invalid files and wrong-length predictions so pred_list arrays match submission length.
    - Always write a valid 'submission.csv' for Kaggle.

    Core logic preserved: build blended base predictions -> random convex weights -> median -> snap.

    Score-improvement change (fallback only):
    - If dp is missing/empty or yields no valid prediction files, use a train-derived lookup fallback.
    """
    sample_path = _resolve_path(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    if not os.path.isdir(dp):
        return _fallback_predict_from_train()

    files = [p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)]
    files.sort()

    if len(files) == 0:
        return _fallback_predict_from_train()

    file_count = len(files)
    loop_time = 154
    splits = file_count // 2
    if splits < 1:
        splits = 1

    flist_paths = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        if i == splits - 1:
            flist_paths.append(files[start:])
        else:
            end = (i + 1) * round(len(files) / splits)
            flist_paths.append(files[start:end])

    flist = []
    for grp in flist_paths:
        arr = wc(grp, expected_len=expected_len)
        if arr is not None and len(arr) == expected_len:
            flist.append(arr)

    if len(flist) == 0:
        return _fallback_predict_from_train()

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weights = [rd() for _ in range(len(flist))]
        weight_sum = sum(weights) if sum(weights) != 0 else 1.0
        weights = [w / weight_sum for w in weights]
        weights.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for i in range(len(flist)):
            temp += flist[i] * weights[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    """
    Keep original blending helper; also add minimal safety to ensure lengths match.
    """
    sa = pd.read_csv(a)
    sb = pd.read_csv(b)
    if "pressure" not in sa.columns or "pressure" not in sb.columns:
        raise ValueError("Both files must contain a 'pressure' column.")
    if len(sa) != len(sb):
        raise ValueError("Files to blend must have the same number of rows.")
    sa["pressure"] = (
        sa["pressure"].astype(np.float64) * 0.58
        + sb["pressure"].astype(np.float64) * 0.42
    )
    sa["pressure"] = sa["pressure"].apply(find_nearest)
    sa.to_csv("blend.csv", index=False)
    return sa




## === cell 2
g("../input/gb-data-blending-recover")
