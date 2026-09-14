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

0.1358631054497741

# 6. Current score

1.70347

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02322) has done: 'I fix the pipeline so it no longer depends on the missing `../input/gb-data-blending-recover` dataset by falling back to generating a baseline prediction directly from `train.csv`/`test.csv`. The fallback be a lightweight, competition-valid approach: per-(R,C,time_step) median pressure computed from train inspiratory rows (`u_out==0`) and merged onto test, with a sensible per-(R,C) fallback when an exact time_step key is absent. I keep your existing pressure “snap to nearest valid pressure” post-processing (which is aligned with this competition) and still write `submission.csv` with the required `id,pressure` columns. This is minimal but ensures end-to-end execution and should achieve a reasonable MAE (better than a constant-zero submission) while staying within runtime constraints.'
- What this solution (achieved 4.00094) has done: 'Your current fallback uses only per-(R,C,time_step) medians, which ignores the strongest signal (`u_in`) during inspiration and typically yields a much worse MAE than necessary. I keep your overall pipeline and “snap to nearest valid pressure” post-processing, but improve the fallback by adding a second lookup keyed by (R,C,time_step,u_in) after rounding `u_in` to a small grid, then fall back to your existing (R,C,time_step) and (R,C) medians. This is a minimal change that preserves the same training-free baseline approach while making predictions much more conditionally accurate, which should move the score substantially toward the 0.1359 target. The submission writing and blending behavior remain unchanged; only the fallback prediction construction is strengthened.'
- What this solution (achieved 3.86198) has done: 'I keep your current training-free fallback structure and the “snap to nearest valid pressure” post-processing, but make the baseline conditioning closer to the true scoring setup by explicitly using only inspiratory rows in train *and* forcing expiratory predictions in test to a safe constant (since expiratory isn’t scored and can otherwise inject noise). I also strengthen the `u_in` conditioning with a tiny multi-resolution binning fallback (0.5 then 1.0) so test rows that miss the 0.1 grid still get a `u_in`-aware estimate before dropping to the coarser (R,C,time_step) median. These are minimal additions that don’t change your overall approach (still pure lookups/medians), but should reduce MAE substantially from ~4 toward your ~0.136 target. The output remains a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.70347) has done: 'Your current score (3.86198 MAE) is far worse than the target (0.1359), so we should improve the baseline substantially while keeping the same “training-free lookup/median” core logic and the same snapping-to-valid-pressures post-processing. The biggest missing piece is that pressure is much more determined by engineered state like integrated flow/volume, not raw `(time_step, u_in)` bins; we can still stay within your exact approach by adding a few deterministic, per-breath cumulative features in both train/test and then doing medians on those keys. Concretely, I add `dt`, `u_in_cum` and `u_in_cum_bin` (cumulative inspired volume proxy), plus a light `u_in_delta` term, and use them as higher-priority merge keys before your existing fallbacks. This remains pure groupby-median lookups (no model training, no loops changed) but should move the MAE strongly toward your target band.'
- What this solution (achieved 1.70347) has done: 'Your current MAE (1.70) is still far above the target (0.136), so we should improve accuracy while keeping your same “training-free median lookup + fallbacks + snap-to-valid-pressure” core logic. The minimal high-impact fix is to add one more deterministic state proxy: an approximate “flow” feature `u_in*(1-u_out)` and its cumulative integral per breath (a volume proxy that ignores expiratory steps), then use it as higher-priority lookup keys before your existing fallbacks. This preserves evaluation semantics (still medians from inspiratory train rows, still safe expiratory handling, still nearest-pressure snapping) but should materially reduce MAE. I also keep runtime safe by only adding a small number of additional groupby tables and reusing the same merge/fill pattern.'

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


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    pressures = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(input_list[i])
        if "pressure" not in df.columns:
            raise ValueError(
                f"File {input_list[i]} does not contain 'pressure' column."
            )
        pressures.append(df["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else len(l)

    if len(pressures) == 1:
        output = pressures[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = pressures[0] * weight1 + pressures[1] * weight2
    return output


def _add_cum_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Change (score-moving, still minimal, preserves core logic):
    Add deterministic per-breath cumulative features (no training) so median lookups
    can condition on better proxies of lung state than (time_step, u_in) alone.

    New in this patch:
    - u_in_eff = u_in*(1-u_out): approximate inspiratory flow, zeroed during expiration
    - u_in_eff_cum = cumulative integral of u_in_eff over dt: better "volume" proxy
    These remain deterministic engineering and keep the same groupby-median approach.
    """
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True)

    df["dt"] = (
        df.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )

    df["u_in_dt"] = (df["u_in"].astype(np.float32) * df["dt"]).astype(np.float32)
    df["u_in_cum"] = (
        df.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
    )

    df["u_in_delta"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )

    u_out_f = df["u_out"].astype(np.float32)
    df["u_in_eff"] = (df["u_in"].astype(np.float32) * (1.0 - u_out_f)).astype(
        np.float32
    )
    df["u_in_eff_dt"] = (df["u_in_eff"] * df["dt"]).astype(np.float32)
    df["u_in_eff_cum"] = (
        df.groupby("breath_id", sort=False)["u_in_eff_dt"].cumsum().astype(np.float32)
    )

    df["time_step_r2"] = df["time_step"].round(2)
    df["u_in_bin_01"] = df["u_in"].round(1)
    df["u_in_bin_05"] = (df["u_in"] / 0.5).round() * 0.5
    df["u_in_bin_10"] = df["u_in"].round(0)

    df["u_in_cum_bin_1"] = (df["u_in_cum"] / 1.0).round() * 1.0
    df["u_in_cum_bin_2"] = (df["u_in_cum"] / 2.0).round() * 2.0

    df["u_in_eff_cum_bin_1"] = (df["u_in_eff_cum"] / 1.0).round() * 1.0
    df["u_in_eff_cum_bin_2"] = (df["u_in_eff_cum"] / 2.0).round() * 2.0

    df["u_in_delta_bin_1"] = (df["u_in_delta"] / 1.0).round() * 1.0

    return df


def _make_baseline_submission(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline remains: median-lookup tables from train inspiratory rows + merge fallbacks,
    then snap predictions to nearest valid discrete pressure.

    Change (score-moving, minimal): add u_in_eff_cum-based keys (inspiratory-only volume proxy),
    then fall back to your existing tables unchanged.
    """
    test_df = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )

    tr_full = _add_cum_features(train_df, is_train=True)
    te_full = _add_cum_features(test_df, is_train=False)

    tr = tr_full.loc[
        tr_full["u_out"] == 0,
        [
            "R",
            "C",
            "time_step_r2",
            "u_in_bin_01",
            "u_in_bin_05",
            "u_in_bin_10",
            "u_in_cum_bin_1",
            "u_in_cum_bin_2",
            "u_in_eff_cum_bin_1",
            "u_in_eff_cum_bin_2",
            "u_in_delta_bin_1",
            "pressure",
        ],
    ].copy()

    med_rct_ecum2_u05 = (
        tr.groupby(
            ["R", "C", "time_step_r2", "u_in_eff_cum_bin_2", "u_in_bin_05"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rct_ecum2_u05"})
    )
    med_rct_ecum1_u05 = (
        tr.groupby(
            ["R", "C", "time_step_r2", "u_in_eff_cum_bin_1", "u_in_bin_05"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rct_ecum1_u05"})
    )

    med_rct_cum2_u05 = (
        tr.groupby(
            ["R", "C", "time_step_r2", "u_in_cum_bin_2", "u_in_bin_05"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rct_cum2_u05"})
    )
    med_rct_cum1_u05 = (
        tr.groupby(
            ["R", "C", "time_step_r2", "u_in_cum_bin_1", "u_in_bin_05"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rct_cum1_u05"})
    )

    med_rctu_01 = (
        tr.groupby(["R", "C", "time_step_r2", "u_in_bin_01"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rctu_01"})
    )
    med_rctu_05 = (
        tr.groupby(["R", "C", "time_step_r2", "u_in_bin_05"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rctu_05"})
    )
    med_rctu_10 = (
        tr.groupby(["R", "C", "time_step_r2", "u_in_bin_10"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rctu_10"})
    )

    med_rctu05_du1 = (
        tr.groupby(
            ["R", "C", "time_step_r2", "u_in_bin_05", "u_in_delta_bin_1"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rctu05_du1"})
    )

    med_rct = (
        tr.groupby(["R", "C", "time_step_r2"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rct"})
    )
    med_rc = (
        tr.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rc"})
    )

    merged = te_full.merge(
        med_rct_ecum2_u05,
        on=["R", "C", "time_step_r2", "u_in_eff_cum_bin_2", "u_in_bin_05"],
        how="left",
    )
    merged = merged.merge(
        med_rct_ecum1_u05,
        on=["R", "C", "time_step_r2", "u_in_eff_cum_bin_1", "u_in_bin_05"],
        how="left",
    )
    merged = merged.merge(
        med_rct_cum2_u05,
        on=["R", "C", "time_step_r2", "u_in_cum_bin_2", "u_in_bin_05"],
        how="left",
    )
    merged = merged.merge(
        med_rct_cum1_u05,
        on=["R", "C", "time_step_r2", "u_in_cum_bin_1", "u_in_bin_05"],
        how="left",
    )
    merged = merged.merge(
        med_rctu05_du1,
        on=["R", "C", "time_step_r2", "u_in_bin_05", "u_in_delta_bin_1"],
        how="left",
    )
    merged = merged.merge(
        med_rctu_01, on=["R", "C", "time_step_r2", "u_in_bin_01"], how="left"
    )
    merged = merged.merge(
        med_rctu_05, on=["R", "C", "time_step_r2", "u_in_bin_05"], how="left"
    )
    merged = merged.merge(
        med_rctu_10, on=["R", "C", "time_step_r2", "u_in_bin_10"], how="left"
    )
    merged = merged.merge(med_rct, on=["R", "C", "time_step_r2"], how="left")
    merged = merged.merge(med_rc, on=["R", "C"], how="left")

    global_med = float(tr["pressure"].median())
    global_low = float(tr["pressure"].quantile(0.10))

    pred_insp = (
        merged["p_med_rct_ecum2_u05"]
        .fillna(merged["p_med_rct_ecum1_u05"])
        .fillna(merged["p_med_rct_cum2_u05"])
        .fillna(merged["p_med_rct_cum1_u05"])
        .fillna(merged["p_med_rctu05_du1"])
        .fillna(merged["p_med_rctu_01"])
        .fillna(merged["p_med_rctu_05"])
        .fillna(merged["p_med_rctu_10"])
        .fillna(merged["p_med_rct"])
        .fillna(merged["p_med_rc"])
        .fillna(global_med)
        .astype(float)
    )

    pred = np.where(merged["u_out"].to_numpy() == 1, global_low, pred_insp.to_numpy())
    pred = pd.Series(pred, index=merged.index).map(find_nearest)

    submission = merged[["id"]].copy()
    submission["pressure"] = pred.values
    submission.to_csv("submission.csv", index=False)
    return submission


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            l.append(i)

    if len(l) == 0:
        return _make_baseline_submission(df_train)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    expected_len = len(output)

    l.sort()
    splits = max(1, len(l) // 2)  # keep original intent but avoid splits==0
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        flist.append(l[start:end])

    for i in range(len(flist)):
        vec = wc(flist[i])
        vec = np.asarray(vec).ravel()
        if vec.shape[0] != expected_len:
            raise ValueError(
                f"Blended vector length mismatch from files {flist[i][:2]}...: "
                f"got {vec.shape[0]}, expected {expected_len}. "
                f"Check that prediction files align with sample_submission."
            )
        flist[i] = vec

    pred_list = []
    loop_time = 154  # preserve original

    for t in range(loop_time):
        weight = []
        set_seed(t)
        for _ in range(len(flist)):
            weight.append(rd())

        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            for i in range(len(weight)):
                weight[i] /= weight_sum

        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for i in range(len(flist)):
            temp += flist[i] * weight[i]

        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, expected_len)
    median_pred = np.median(stacked, axis=0)
    mean_pred = np.mean(stacked, axis=0)

    final_pred = 0.8 * median_pred + 0.2 * mean_pred
    output["pressure"] = pd.Series(final_pred, index=output.index).apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)

    if "pressure" not in a.columns or "pressure" not in b.columns:
        raise ValueError("Both input files must contain a 'pressure' column.")
    if len(a) != len(b):
        raise ValueError(f"Blend length mismatch: len(a)={len(a)} vs len(b)={len(b)}")

    a["pressure"] = a["pressure"] * 0.7 + b["pressure"] * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    if len(input_list) == 0:
        raise FileNotFoundError(
            f"No .csv files found under {dp}. Cannot average predictions."
        )

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    expected_len = len(output)

    preds = []
    for path in input_list:
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain 'pressure' column.")
        vec = df["pressure"].to_numpy().ravel()
        if vec.shape[0] != expected_len:
            raise ValueError(
                f"File {path} has length {vec.shape[0]} but expected {expected_len}."
            )
        preds.append(vec)

    output["pressure"] = np.median(np.vstack(preds), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("avg.csv", index=False)
    return output




## === cell 2
g("../input/gb-data-blending-recover")
print("Wrote submission.csv")  # helps confirm successful end-to-end run
print(pd.read_csv("submission.csv").head())
