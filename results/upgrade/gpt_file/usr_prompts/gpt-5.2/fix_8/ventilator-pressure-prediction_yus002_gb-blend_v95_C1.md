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

0.1475959330294667

# 6. Current score

3.09631

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'I fix the failure by making `g()` robust to missing/empty blend directories and to mismatched submission lengths, because right now it silently builds an empty/invalid `pred_list` and then tries to assign a length-1 array to a 603600-row submission. I also make the file discovery only include `.csv` files and ensure every loaded prediction is flattened and validated to exactly the sample submission length (or skipped). Finally, if no valid external blend files exist (likely in your environment), the script fall back to a simple, valid baseline submission (`pressure=0`) so you always get a `.csv` submission end-to-end without changing the “core logic” of blending when files are present.'
- What this solution (achieved 6.5338) has done: 'Your current score (17.65486 MAE) is far from the target (0.1476), and the reason is that your code is effectively submitting a near-baseline (all zeros) because no valid blend files exist in `../input/gb-blending`. To move the score toward the target with minimal change and without altering the blending core, I keep your blending logic intact but change the fallback: when no blend CSVs are found/valid, train a simple per-time-step median baseline on the training set (using only inspiratory rows, matching the metric) and use it for test predictions. I also ensure predictions are aligned to test `id` order and still snapped to the nearest allowed pressure value via your existing `find_nearest`. This should dramatically reduce MAE compared to zeros while keeping runtime within the constraint.'
- What this solution (achieved 6.10817) has done: 'Your current MAE (6.5338, lower is better) is still far above the target (0.1476), and the biggest safe gain without changing your blending core is to make the fallback baseline stronger while keeping the same overall semantics (generate predictions then snap to nearest allowed pressure). I keep your blending logic intact, but replace the fallback “median-by-timestep only” with a slightly richer grouped median baseline using `(R, C, ts_idx)` on inspiratory rows, with a safe backoff to `(R, C)` then global-by-`ts_idx` medians. I also fix the test prediction alignment bug (the current reordering code can misalign predictions with `id`), ensuring predictions are produced in `id` order deterministically. These changes should move the score significantly toward the target while staying minimal and fast enough.'
- What this solution (achieved 6.10817) has done: 'Your current MAE (6.108) is still far above the target (0.1476), so we should improve the fallback model (used when `../input/gb-blending` is missing/empty) while keeping your blending logic intact. The biggest minimal gain is to make the grouped-median fallback match the evaluation semantics better by explicitly forcing **all expiratory (`u_out==1`) predictions to 0**, since those rows are not scored and this avoids injecting unnecessary error. Additionally, we make the median computation more robust by deriving `ts_idx` using `groupby().cumcount()` on already time-sorted data (as you do), and keep strict alignment to the sample submission `id` order. These changes do not alter your core approach (blending + nearest-pressure snapping) and should move the score materially closer to the target.'
- What this solution (achieved 4.99472) has done: 'We keep your blending logic intact and only strengthen the fallback (used because `../input/gb-blending` is typically missing/empty here) to better match the ventilator dynamics while staying fast. Specifically, we replace the grouped-median fallback with a minimal linear regression fit **per (R,C,ts_idx)** on inspiratory rows using only `u_in` (the dominant control) and an intercept; then we back off to grouped medians only when a group is too small or singular. We also keep your metric-aligned rule of forcing `u_out==1` predictions to 0, and we keep your existing snapping to the nearest allowed pressure values. This should reduce MAE from ~6 toward the target without changing the overall approach (generate predictions → snap → write submission).'
- What this solution (achieved 2.48636) has done: 'Your current MAE (4.99472, lower is better) is still far above the target (0.1476), so we should cautiously improve the fallback path (which is what runs when `../input/gb-blending` is missing/empty) without changing the overall “predict → snap-to-allowed-pressures → write submission” semantics. The smallest high-impact fix is to make the fallback model use the strongest available signals while staying within your existing lightweight per-group fitting approach: add `u_in_lag1` and cumulative integrated flow (`u_in_cum`) as additional linear features in the same per-(R,C,ts_idx) regression, and keep the same backoff chain to medians. This preserves your core logic (no new ML libraries, no training loops, no architecture changes) but typically reduces error a lot because ventilator pressure depends on recent/accumulated flow. We keep forcing `u_out==1` predictions to 0 and keep the existing nearest-pressure snapping and submission writing.'
- What this solution (achieved 3.09631) has done: 'Your current MAE (2.486) is still far above the target (0.1476), so we should improve the fallback path (used when `../input/gb-blending` is absent/empty) with the smallest change that better matches ventilator physics while keeping your overall “predict → snap-to-allowed-pressures → write submission” flow intact. The main issue in the current fallback is that it fits a separate linear model per `(R,C,ts_idx)`, which throws away the strong sequential dependence of pressure within a breath; we keep the same lightweight closed-form regression idea, but fit one regression per `(R,C)` across all inspiratory timesteps using a few already-computed time-series features, and we keep the existing median backoff chain. We also preserve your metric-aligned rule `u_out==1 -> pressure=0` and keep the nearest-pressure snapping unchanged. This should materially reduce error while staying within Kaggle constraints and runtime.'

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


def _safe_read_pressure_csv(path, expected_len):
    """
    Bugfix: ensure we only use valid prediction CSVs with a 'pressure' column
    and matching length. If not valid, return None so caller can skip.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy().ravel()
    if arr.shape[0] != expected_len:
        return None
    return arr


def wc(input_list, expected_len):
    """
    Original intent: weighted combination of 1 or 2 files.
    Bugfix: handle parsing failures and length mismatches safely.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1  # neutral fallback weight
        l.append(public_lb_score)

        arr = _safe_read_pressure_csv(input_list[i], expected_len)
        if arr is None:
            continue
        preds.append(arr)

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l[:2]) if sum(l[:2]) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.01
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _fallback_grouped_median(expected_len):
    """
    Score-improving fallback while preserving the "fallback predicts then snap" semantics.

    Change (directly for score, minimal to current approach):
    - Keep your same lightweight closed-form regression idea (no ML libs, no training loops),
      but fit it per (R,C) across all inspiratory rows instead of per (R,C,ts_idx).
      This leverages within-breath temporal dependence using already-computed features,
      and avoids the very noisy/sparse per-timestep fits.
    - Features: [1, u_in, u_in_lag1, u_in_cum, ts_idx] (ts_idx acts as a simple time effect).
    - Keep: backoff chain to medians (R,C,ts_idx) -> (R,C) -> (ts_idx) -> global.
    - Keep: force u_out==1 predictions to 0.0 because those rows are not scored.
    - Keep: snapping to nearest allowed pressure value outside this function.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    sub = pd.read_csv(sample_path)
    expected_len = len(sub) if expected_len is None else expected_len

    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "time_step", "u_in", "u_out", "R", "C"]
    )
    test_bt = test.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    test_bt["ts_idx"] = test_bt.groupby("breath_id").cumcount().astype(np.int16)

    test_bt["u_in_lag1"] = (
        test_bt.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float64)
    )
    test_bt["u_in_cum"] = (
        test_bt.groupby("breath_id")["u_in"].cumsum().astype(np.float64)
    )

    train = df_train[
        ["breath_id", "time_step", "u_in", "u_out", "pressure", "R", "C"]
    ].copy()
    train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
    train["ts_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)
    train["u_in_lag1"] = (
        train.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float64)
    )
    train["u_in_cum"] = train.groupby("breath_id")["u_in"].cumsum().astype(np.float64)

    insp = train[train["u_out"] == 0].copy()

    med_rc_t = (
        insp.groupby(["R", "C", "ts_idx"], sort=False)["pressure"]
        .median()
        .astype(float)
    )
    med_rc = insp.groupby(["R", "C"], sort=False)["pressure"].median().astype(float)
    med_t = insp.groupby(["ts_idx"], sort=False)["pressure"].median().astype(float)
    global_med = float(insp["pressure"].median())

    features = ["u_in", "u_in_lag1", "u_in_cum", "ts_idx"]
    d = len(features)  # 4
    grp_rc = insp.groupby(["R", "C"], sort=False)

    b0_s = pd.Series(dtype=np.float64)
    b1_s = pd.Series(dtype=np.float64)  # u_in
    b2_s = pd.Series(dtype=np.float64)  # u_in_lag1
    b3_s = pd.Series(dtype=np.float64)  # u_in_cum
    b4_s = pd.Series(dtype=np.float64)  # ts_idx
    ok_s = pd.Series(dtype=bool)

    b0 = {}
    b1 = {}
    b2 = {}
    b3 = {}
    b4 = {}
    ok = {}

    min_n = 2000  # ensure stable solves; groups are large in this dataset

    for (R, C), gdf in grp_rc:
        n = len(gdf)
        if n < min_n:
            ok[(R, C)] = False
            b0[(R, C)] = global_med
            b1[(R, C)] = 0.0
            b2[(R, C)] = 0.0
            b3[(R, C)] = 0.0
            b4[(R, C)] = 0.0
            continue

        X = gdf[features].to_numpy(dtype=np.float64)
        y = gdf["pressure"].to_numpy(dtype=np.float64)

        ones = np.ones((n, 1), dtype=np.float64)
        A = np.concatenate([ones, X], axis=1)  # (n, 1+d)
        XtX = (A.T @ A) / n
        Xty = (A.T @ y) / n

        try:
            coef = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            ok[(R, C)] = False
            b0[(R, C)] = (
                float(med_rc.loc[(R, C)]) if (R, C) in med_rc.index else global_med
            )
            b1[(R, C)] = 0.0
            b2[(R, C)] = 0.0
            b3[(R, C)] = 0.0
            b4[(R, C)] = 0.0
            continue

        if not np.all(np.isfinite(coef)):
            ok[(R, C)] = False
            b0[(R, C)] = (
                float(med_rc.loc[(R, C)]) if (R, C) in med_rc.index else global_med
            )
            b1[(R, C)] = 0.0
            b2[(R, C)] = 0.0
            b3[(R, C)] = 0.0
            b4[(R, C)] = 0.0
            continue

        ok[(R, C)] = True
        b0[(R, C)] = float(coef[0])
        b1[(R, C)] = float(coef[1])
        b2[(R, C)] = float(coef[2])
        b3[(R, C)] = float(coef[3])
        b4[(R, C)] = float(coef[4])

    idx_rc = pd.MultiIndex.from_tuples(list(ok.keys()), names=["R", "C"])
    ok_s = pd.Series([ok[k] for k in ok.keys()], index=idx_rc)
    b0_s = pd.Series([b0[k] for k in ok.keys()], index=idx_rc)
    b1_s = pd.Series([b1[k] for k in ok.keys()], index=idx_rc)
    b2_s = pd.Series([b2[k] for k in ok.keys()], index=idx_rc)
    b3_s = pd.Series([b3[k] for k in ok.keys()], index=idx_rc)
    b4_s = pd.Series([b4[k] for k in ok.keys()], index=idx_rc)

    pred_bt = np.zeros(len(test_bt), dtype=np.float64)
    insp_mask = test_bt["u_out"].to_numpy() == 0

    if insp_mask.any():
        keys_rc = test_bt.loc[insp_mask, ["R", "C"]]
        mi_rc = pd.MultiIndex.from_frame(keys_rc)

        ok_key = mi_rc.map(ok_s).fillna(False).to_numpy(dtype=bool)
        pred_vals = pd.Series(np.nan, index=keys_rc.index, dtype=np.float64)

        if ok_key.any():
            u1 = test_bt.loc[insp_mask, "u_in"].to_numpy(dtype=np.float64)
            u2 = test_bt.loc[insp_mask, "u_in_lag1"].to_numpy(dtype=np.float64)
            u3 = test_bt.loc[insp_mask, "u_in_cum"].to_numpy(dtype=np.float64)
            tt = test_bt.loc[insp_mask, "ts_idx"].to_numpy(dtype=np.float64)

            c0 = mi_rc.map(b0_s).to_numpy(dtype=np.float64)
            c1 = mi_rc.map(b1_s).to_numpy(dtype=np.float64)
            c2 = mi_rc.map(b2_s).to_numpy(dtype=np.float64)
            c3 = mi_rc.map(b3_s).to_numpy(dtype=np.float64)
            c4 = mi_rc.map(b4_s).to_numpy(dtype=np.float64)

            lin = c0 + c1 * u1 + c2 * u2 + c3 * u3 + c4 * tt
            pred_vals.loc[pred_vals.index[ok_key]] = lin[ok_key]

        miss = pred_vals.isna()
        if miss.any():
            keys_rct = test_bt.loc[miss.index, ["R", "C", "ts_idx"]]
            mi_rct = pd.MultiIndex.from_frame(keys_rct)
            med_full = mi_rct.map(med_rc_t)
            pred_vals.loc[miss] = med_full.loc[miss].to_numpy(dtype=np.float64)

        miss = pred_vals.isna()
        if miss.any():
            rc_vals = mi_rc[miss.to_numpy()].map(med_rc)
            pred_vals.loc[miss] = rc_vals.to_numpy(dtype=np.float64)

        miss = pred_vals.isna()
        if miss.any():
            t_vals = test_bt.loc[pred_vals.index[miss], "ts_idx"].map(med_t)
            pred_vals.loc[miss] = t_vals.to_numpy(dtype=np.float64)

        pred_bt[insp_mask] = pred_vals.fillna(global_med).to_numpy(dtype=np.float64)

    pred_bt[~insp_mask] = 0.0

    test_bt["pred"] = pred_bt
    pred = test_bt.sort_values("id", kind="mergesort")["pred"].to_numpy(
        dtype=np.float64
    )

    if pred.shape[0] != expected_len:
        pred = np.full(expected_len, global_med, dtype=np.float64)

    return pred


def g(dp):
    """
    Bugfixes:
    - Only consider CSV files.
    - If dp doesn't exist or no valid files, write a valid fallback submission.
    - Ensure median stacking dimensions match sample_submission length.
    - Ensure output filename ends with .csv.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    if not os.path.isdir(dp):
        output["pressure"] = _fallback_grouped_median(expected_len)
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    l = sorted([p for p in glob.glob(os.path.join(dp, "*.csv")) if os.path.isfile(p)])
    file_count = len(l)
    loop_time = 154

    if file_count == 0:
        output["pressure"] = _fallback_grouped_median(expected_len)
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    splits = max(1, file_count // 2)

    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        chunk = l[start:end]
        if len(chunk) == 0:
            continue
        blended = wc(chunk, expected_len)
        if blended is not None:
            flist.append(blended)

    if len(flist) == 0:
        output["pressure"] = _fallback_grouped_median(expected_len)
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for loop_idx in range(loop_time):
        set_seed(loop_idx)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)

        del temp
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, expected_len)
    output["pressure"] = np.median(stacked, axis=0)

    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-blending")
print("Wrote submission.csv")
