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

0.1485422921312373

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the immediate runtime errors by updating the `pd.concat` call to use `axis=1` (pandas 2.x no longer accepts positional `axis`) and by handling the case where the external fold submission files don’t exist in this environment. To keep the core “ensemble median + pressure-grid rounding/clipping” logic intact, I load any found `submission.csv` files, rename their `pressure` columns to `pressure_0..`, and then compute the median exactly as your later cells expect. If no fold files are found, I fall back to a safe baseline that still produces a valid `submission_base.csv` with the correct columns and length (using the provided sample submission’s 0s). This ensures the notebook runs end-to-end and always writes a valid `.csv` submission.'
- What this solution (achieved 6.32729) has done: 'Your current score is extremely far from the target (MAE 17.65 vs 0.148), which strongly suggests the “no external fold submissions found” fallback of all-zeros is being used and is the root cause of poor performance. The smallest legitimate change that preserves your core “ensemble median + pressure-grid rounding/clipping” logic is to generate a real set of predictions from `train.csv` when fold files are missing, instead of zeros. Below I add a minimal, fast, pure-pandas baseline: for each `(R, C, time_step, u_in, u_out)` pattern in train, use the median `pressure`, and predict test by merging on those same keys, then fall back to a coarser `(R, C, u_out, u_in_rounded)` median for any unseen combinations. This keeps your post-processing intact and should move the score dramatically toward the target while still finishing within the time limit.'
- What this solution (achieved 4.22257) has done: 'Your current MAE (6.33) is far above the target (0.148), so we should improve predictions while keeping your existing “median ensemble + pressure-grid rounding/clipping” submission logic unchanged. The smallest impactful change is to make the fallback mapping closer to the real dynamics by (1) mapping per-breath time index (`step` = 0..79) instead of floating `time_step` (avoids merge misses from float representation), and (2) using cumulative features (`u_in_cum`, `u_in_lag1`, `u_out_lag1`) in a second-stage fallback that still stays pure groupby/merge (no model/loop changes). This should substantially reduce the error versus the current coarse fallback, while remaining fast and deterministic. The external fold-submission loading path remains intact; these changes only affect the “no fold files found” path that is currently being used.'
- What this solution (achieved 4.61139) has done: 'Your current score (4.22257 MAE; lower is better) is far above the target (0.1485), so we should improve predictions while keeping your existing “median ensemble + pressure-grid rounding/clipping” post-processing intact. The smallest likely win is to (1) ensure the fallback prediction merge is done on a stable, quantized representation of `u_in` (float merge misses are common), and (2) add a very-lightweight “per-breath cumulative volume” correction key (`area` = cumulative `u_in` * dt) which is a known strong signal in this competition, without changing your overall approach (still pure groupby-median mapping with staged fallbacks). These changes only affect the “no external fold submissions found” path that you are currently using, and keep runtime within limits by reducing feature width and using observed groupbys. The rest of your pipeline (ensemble median, rounding to discrete pressure grid, clipping, and CSV writing) remains unchanged.'
- What this solution (achieved 4.23124) has done: 'Your current MAE (4.61, lower-is-better) is still far above the target (0.1485), so we should legitimately improve the fallback (no-fold-files) predictor while keeping your core “median ensemble + pressure-grid rounding/clipping” logic intact. The main issue is that the current staged groupby/merge mapping is still too sparse, causing many fallbacks to the global median and losing the strong “pressure as discrete classes” structure of this competition. I add one minimal-but-high-impact step: snap predictions to the discrete pressure values seen in train (nearest neighbor on the pressure grid), which usually reduces MAE a lot without changing modeling/training logic. Additionally, I make the primary mapping denser by switching from `u_in.round(2)` to `u_in.round(1)` in the primary key (fewer merge misses) while leaving your multi-stage fallback structure unchanged.'
- What this solution (achieved 4.23124) has done: 'Your current MAE (4.23124; lower is better) is still far above the target (0.1485), so we should improve the fallback predictor that’s used when no external fold submissions exist, while keeping your core “median ensemble + rounding/clipping to pressure grid” logic unchanged. The biggest low-risk gain is to prevent scoring on the expiratory phase: since Kaggle ignores `u_out==1` rows, we can copy the previous inspiratory prediction forward during expiration (per breath), which typically reduces MAE materially without changing the modeling approach. Additionally, the existing “snap to nearest train pressure level” step has a small bug (it never chooses the upper neighbor); fixing that keeps the same intent but improves correctness. These changes are confined to the no-fold-files path and keep runtime within the 600s limit.'
- What this solution (achieved 4.14542) has done: 'Your current MAE (4.23124; lower-is-better) is still far above the target (0.1485), so we should improve the “no external fold submissions found” fallback predictions while keeping your core pipeline (groupby-median mapping → snap to train pressure grid → expiratory fill → final rounding/clipping) intact. The most impactful minimal fix is to avoid blending expiratory (`u_out==1`) targets into the learned mapping tables, since the metric only scores inspiratory and expiratory pressures follow different dynamics. I therefore build all mapping tables using only inspiratory rows from train, and I also restrict the global fallback median to inspiratory-only pressure; this typically reduces MAE without changing your modeling approach. Everything else (file paths, ensemble median logic, discrete pressure snapping, and submission writing) stays the same and still produces a valid `submission_base.csv`.'
- What this solution (achieved 3.82711) has done: 'Your current pipeline already runs end-to-end and writes a valid submission, so the only way to move MAE toward the (much better) target is to improve the “no external fold submissions found” fallback predictions. I keep your core logic (groupby-median mapping → snap to train pressure grid → expiratory forward-fill → final rounding/clipping) intact, and make two minimal, directly-relevant adjustments that usually reduce MAE: (1) use `time_step` (rounded) in the primary key in addition to `step` to better separate identical step indices with slightly different timing, and (2) add one extra fallback mapping keyed by `(R,C,u_out,step,area_b)` to reduce misses while staying purely groupby/merge-based. These changes only affect the `test_pred is None` path and preserve evaluation semantics and post-processing.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os

SS_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

sample_sub = pd.read_csv(SS_PATH)
test_ids = pd.read_csv(TEST_PATH, usecols=["id"])  # fast and ensures correct ids
submission = test_ids.copy()
submission["pressure"] = 0.0



## === cell 1
tr = pd.read_csv(TRAIN_PATH)
max(tr[tr.u_out == 0]["pressure"]), min(tr[tr.u_out == 0]["pressure"])



## === cell 2
fold_paths = sorted(glob.glob("../input/pulp-fiction-fold5-fold*/submission.csv"))

if len(fold_paths) > 0:
    fold_dfs = []
    for i, p in enumerate(fold_paths):
        df = pd.read_csv(p)
        if "pressure" not in df.columns:
            raise ValueError(
                f"Expected column 'pressure' in {p}, got columns={list(df.columns)}"
            )
        if "id" in df.columns:
            df = df[["id", "pressure"]].copy()
            df = df.rename(columns={"pressure": f"pressure_{i}"})
        else:
            df = df[["pressure"]].copy()
            df = df.rename(columns={"pressure": f"pressure_{i}"})
        fold_dfs.append(df)

    if all("id" in d.columns for d in fold_dfs):
        test_pred = fold_dfs[0]
        for d in fold_dfs[1:]:
            test_pred = test_pred.merge(d, on="id", how="inner", validate="one_to_one")
        test_pred = submission[["id"]].merge(
            test_pred, on="id", how="left", validate="one_to_one"
        )
    else:
        test_pred = pd.concat(fold_dfs, axis=1)
else:
    test_pred = None



## === cell 3
if test_pred is None:
    te = pd.read_csv(TEST_PATH)

    tr2 = tr.copy()
    te2 = te.copy()

    tr2["step"] = tr2.groupby("breath_id").cumcount().astype(np.int16)
    te2["step"] = te2.groupby("breath_id").cumcount().astype(np.int16)

    tr2["time_step_ms"] = np.rint(
        tr2["time_step"].to_numpy(dtype=np.float64) * 1000.0
    ).astype(np.int16)
    te2["time_step_ms"] = np.rint(
        te2["time_step"].to_numpy(dtype=np.float64) * 1000.0
    ).astype(np.int16)

    tr2["u_in_q"] = tr2["u_in"].round(1)
    te2["u_in_q"] = te2["u_in"].round(1)

    tr2["u_in_cum"] = tr2.groupby("breath_id")["u_in"].cumsum()
    te2["u_in_cum"] = te2.groupby("breath_id")["u_in"].cumsum()

    tr2["u_in_lag1"] = tr2.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    te2["u_in_lag1"] = te2.groupby("breath_id")["u_in"].shift(1).fillna(0.0)

    tr2["u_out_lag1"] = (
        tr2.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    te2["u_out_lag1"] = (
        te2.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )

    tr2["dt"] = tr2.groupby("breath_id")["time_step"].diff()
    te2["dt"] = te2.groupby("breath_id")["time_step"].diff()

    tr2["dt_med"] = (
        tr2.groupby("breath_id")["dt"].transform("median").astype(np.float32)
    )
    te2["dt_med"] = (
        te2.groupby("breath_id")["dt"].transform("median").astype(np.float32)
    )

    tr2["dt"] = tr2["dt"].fillna(tr2["dt_med"]).fillna(0.0).astype(np.float32)
    te2["dt"] = te2["dt"].fillna(te2["dt_med"]).fillna(0.0).astype(np.float32)

    tr2["area"] = (tr2["u_in"] * tr2["dt"]).groupby(tr2["breath_id"]).cumsum()
    te2["area"] = (te2["u_in"] * te2["dt"]).groupby(te2["breath_id"]).cumsum()

    tr_insp = tr2[tr2["u_out"] == 0].copy()
    insp_median_pressure = float(tr_insp["pressure"].median())

    keys_primary = ["R", "C", "step", "time_step_ms", "u_in_q", "u_out"]
    map_primary = (
        tr_insp.groupby(keys_primary, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_pred"})
    )
    te_pred = te2.merge(map_primary, on=keys_primary, how="left")

    miss = te_pred["pressure_pred"].isna()
    if miss.any():
        tr_b = tr_insp[["R", "C", "step", "u_out", "u_in", "pressure"]].copy()
        te_b = te_pred.loc[miss, ["id", "R", "C", "step", "u_out", "u_in"]].copy()

        tr_b["u_in_bucket"] = tr_b["u_in"].round(1)
        te_b["u_in_bucket"] = te_b["u_in"].round(1)

        keys_fb1 = ["R", "C", "step", "u_out", "u_in_bucket"]
        map_fb1 = (
            tr_b.groupby(keys_fb1, observed=True)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pressure_fb1"})
        )
        te_b = te_b.merge(map_fb1, on=keys_fb1, how="left")
        te_pred.loc[miss, "pressure_pred"] = te_b["pressure_fb1"].to_numpy()

    miss = te_pred["pressure_pred"].isna()
    if miss.any():
        tr_d = tr_insp[["R", "C", "step", "u_out", "u_in_q", "area", "pressure"]].copy()
        te_d = te_pred.loc[
            miss, ["id", "R", "C", "step", "u_out", "u_in_q", "area"]
        ].copy()

        tr_d["area_b"] = tr_d["area"].round(1)
        te_d["area_b"] = te_d["area"].round(1)

        keys_fb_mid = ["R", "C", "step", "u_out", "u_in_q", "area_b"]
        map_fb_mid = (
            tr_d.groupby(keys_fb_mid, observed=True)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pressure_fbmid"})
        )
        te_d = te_d.merge(map_fb_mid, on=keys_fb_mid, how="left")
        te_pred.loc[miss, "pressure_pred"] = te_d["pressure_fbmid"].to_numpy()

    miss = te_pred["pressure_pred"].isna()
    if miss.any():
        tr_e = tr_insp[["R", "C", "step", "u_out", "area", "pressure"]].copy()
        te_e = te_pred.loc[miss, ["id", "R", "C", "step", "u_out", "area"]].copy()

        tr_e["area_b"] = tr_e["area"].round(1)
        te_e["area_b"] = te_e["area"].round(1)

        keys_fb_area = ["R", "C", "step", "u_out", "area_b"]
        map_fb_area = (
            tr_e.groupby(keys_fb_area, observed=True)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pressure_fbarea"})
        )
        te_e = te_e.merge(map_fb_area, on=keys_fb_area, how="left")
        te_pred.loc[miss, "pressure_pred"] = te_e["pressure_fbarea"].to_numpy()

    miss = te_pred["pressure_pred"].isna()
    if miss.any():
        tr_c = tr_insp[
            [
                "R",
                "C",
                "step",
                "u_out",
                "u_in_cum",
                "u_in_lag1",
                "u_out_lag1",
                "area",
                "pressure",
            ]
        ].copy()
        te_c = te_pred.loc[
            miss,
            [
                "id",
                "R",
                "C",
                "step",
                "u_out",
                "u_in_cum",
                "u_in_lag1",
                "u_out_lag1",
                "area",
            ],
        ].copy()

        tr_c["u_in_cum_b"] = tr_c["u_in_cum"].round(1)
        te_c["u_in_cum_b"] = te_c["u_in_cum"].round(1)
        tr_c["u_in_lag1_b"] = tr_c["u_in_lag1"].round(1)
        te_c["u_in_lag1_b"] = te_c["u_in_lag1"].round(1)

        tr_c["area_b"] = tr_c["area"].round(2)
        te_c["area_b"] = te_c["area"].round(2)

        keys_fb2 = [
            "R",
            "C",
            "step",
            "u_out",
            "u_out_lag1",
            "u_in_cum_b",
            "u_in_lag1_b",
            "area_b",
        ]
        map_fb2 = (
            tr_c.groupby(keys_fb2, observed=True)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pressure_fb2"})
        )
        te_c = te_c.merge(map_fb2, on=keys_fb2, how="left")
        te_pred.loc[miss, "pressure_pred"] = te_c["pressure_fb2"].to_numpy()

    miss = te_pred["pressure_pred"].isna()
    if miss.any():
        tr0 = tr_insp[tr_insp["step"] == 0][["R", "C", "u_in_q", "pressure"]].copy()
        te0 = te_pred.loc[miss, ["R", "C", "step", "u_in_q"]].copy()
        te0 = te0[te0["step"] == 0]
        if len(te0) > 0 and len(tr0) > 0:
            keys_start = ["R", "C", "u_in_q"]
            map_start = (
                tr0.groupby(keys_start, observed=True)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": "pressure_start"})
            )
            te0 = te0.merge(map_start, on=keys_start, how="left")
            te_pred.loc[miss & (te_pred["step"] == 0), "pressure_pred"] = te0[
                "pressure_start"
            ].to_numpy()

    miss = te_pred["pressure_pred"].isna()
    if miss.any():
        keys_rcs = ["R", "C", "step"]
        map_rcs = (
            tr_insp.groupby(keys_rcs, observed=True)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "pressure_rcs"})
        )
        te_rcs = te_pred.loc[miss, ["R", "C", "step"]].merge(
            map_rcs, on=keys_rcs, how="left"
        )
        te_pred.loc[miss, "pressure_pred"] = te_rcs["pressure_rcs"].to_numpy()

    te_pred["pressure_pred"] = te_pred["pressure_pred"].fillna(insp_median_pressure)

    pressure_levels = np.sort(tr_insp["pressure"].unique())
    p = te_pred["pressure_pred"].to_numpy(dtype=np.float64)
    idx = np.searchsorted(pressure_levels, p, side="left")
    idx = np.clip(idx, 0, len(pressure_levels) - 1)
    lo_idx = np.clip(idx - 1, 0, len(pressure_levels) - 1)
    hi_idx = idx
    lo_val = pressure_levels[lo_idx]
    hi_val = pressure_levels[hi_idx]
    choose_lo = np.abs(p - lo_val) <= np.abs(p - hi_val)
    snapped = np.where(choose_lo, lo_val, hi_val)
    te_pred["pressure_pred"] = snapped.astype(np.float32)

    te_pred = te_pred.sort_values(["breath_id", "step"], kind="mergesort")

    te_pred["pressure_pred_insp"] = te_pred["pressure_pred"].where(
        te_pred["u_out"].to_numpy() == 0, np.nan
    )
    te_pred["pressure_pred_ff"] = te_pred.groupby("breath_id")[
        "pressure_pred_insp"
    ].ffill()
    te_pred["pressure_pred_ff"] = (
        te_pred["pressure_pred_ff"].fillna(insp_median_pressure).astype(np.float32)
    )
    te_pred["pressure_pred"] = np.where(
        te_pred["u_out"].to_numpy() == 1,
        te_pred["pressure_pred_ff"].to_numpy(),
        te_pred["pressure_pred"].to_numpy(),
    ).astype(np.float32)

    test_pred = te_pred[["id", "pressure_pred"]].rename(
        columns={"pressure_pred": "pressure_0"}
    )



## === cell 4
if test_pred is not None:
    test_pred.to_csv("submission_all.csv", index=False)
else:
    pd.DataFrame({"note": ["no external fold submissions found in ../input/"]}).to_csv(
        "submission_all.csv", index=False
    )



## === cell 5
if test_pred is not None:
    pressure_cols = [c for c in test_pred.columns if c.startswith("pressure_")]
    if len(pressure_cols) == 0:
        raise ValueError(
            f"No 'pressure_*' columns found in test_pred. Columns: {list(test_pred.columns)}"
        )
    if "id" in test_pred.columns:
        test_pred = submission[["id"]].merge(
            test_pred, on="id", how="left", validate="one_to_one"
        )
    test_preds = [test_pred[c].to_numpy() for c in pressure_cols]
else:
    test_preds = [submission["pressure"].to_numpy()]




## === cell 6
class config:
    paths = {
        "train": TRAIN_PATH,
        "test": TEST_PATH,
        "ss": SS_PATH,
    }

    model_params = {
        "is_train": True,
        "debug": False,
        "EPOCH": 300,
        "BATCH_SIZE": 1024,
        "NUM_FOLDS": 10,
    }

    post_processing = {
        "max_pressure": 64.82099173863948,
        "min_pressure": -1.8957442945646408,
        "diff_pressure": 0.07030214545121617,
    }




## === cell 7
submission["pressure"] = np.median(np.vstack(test_preds), axis=0)

submission["pressure"] = (
    np.round(
        (submission.pressure - config.post_processing["min_pressure"])
        / config.post_processing["diff_pressure"]
    )
    * config.post_processing["diff_pressure"]
    + config.post_processing["min_pressure"]
)
submission["pressure"] = np.clip(
    submission["pressure"],
    config.post_processing["min_pressure"],
    config.post_processing["max_pressure"],
)

submission = submission[["id", "pressure"]]

te_ids_full = pd.read_csv(TEST_PATH, usecols=["id"])
if len(submission) != len(te_ids_full):
    raise ValueError(
        f"Submission rowcount {len(submission)} != test rowcount {len(te_ids_full)}"
    )
if not submission["id"].equals(te_ids_full["id"]):
    submission = te_ids_full.merge(
        submission, on="id", how="left", validate="one_to_one"
    )
    if submission["pressure"].isna().any():
        submission["pressure"] = submission["pressure"].fillna(0.0)

print(submission.head())
print("rows:", len(submission), "cols:", submission.columns.tolist())

submission.to_csv("submission_base.csv", index=False)
print("Wrote submission_base.csv ->", os.path.abspath("submission_base.csv"))
