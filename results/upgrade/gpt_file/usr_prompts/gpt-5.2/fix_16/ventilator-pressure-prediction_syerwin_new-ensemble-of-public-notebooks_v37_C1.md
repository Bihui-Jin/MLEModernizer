# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

2.8234809420754745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 6.91232) has done: 'Your notebook fails because it depends on four external “../input/…” submission files that are not present in this environment, so `sub_1`…`sub_4` never load and blending can’t run. To keep the core “blend predictions into sample_submission” logic intact while making it runnable end-to-end, I (1) add a small, safe loader that uses those files if they exist, otherwise falls back to generating reasonable baseline predictions from the provided train/test data, and (2) ensure all prediction frames are aligned by `id` before blending to avoid silent row-order mismatches. This produce a valid `submission.csv` with the required columns and should yield a non-trivial score (likely much better than all-zeros), moving toward your target. No model architecture/training loop is introduced; the only added logic is a deterministic, lightweight fallback predictor so the pipeline can run without missing inputs.'
- What this solution (achieved 5.2966) has done: 'You’re currently far above the target MAE (6.91 vs 2.82, lower is better), so we need a modest accuracy gain without changing the overall “blend multiple submissions into sample_submission” core logic. The biggest safe win here is to make the fallback predictors closer to the metric by (1) using only inspiratory-phase statistics (u_out==0) when building the lookup table, since expiratory rows are not scored, and (2) adding a lightweight per-(R,C,u_in_bin,time_step_bin) table to better respect the time-series nature without introducing any model/training loop. Finally, we keep the same 4-way blend structure but slightly reweight toward the more granular time-aware fallback (still just blending), which should move MAE down toward your target. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 5.2962) has done: 'Your current MAE (5.2966, lower-is-better) is still far above the target (2.823), so we need a small but meaningful accuracy lift without changing the core “blend multiple submissions” logic. The biggest safe gain here is to make the fallback predictors respect the competition’s *discrete pressure levels* by snapping predictions to the nearest valid pressure observed in the training inspiratory phase (u_out==0), which typically reduces MAE materially for this dataset. I keep your existing fallback tables and 4-way blending, but add this pressure-quantization as a final post-process for both fallback predictions and the final blended output (harmless even when external submissions exist). This keeps semantics intact (still predicting pressure per id) and should move the score down toward your target band.'
- What this solution (achieved 4.92853) has done: 'We need to reduce your MAE (5.2962) toward the target (2.8235), so we keep the exact same fallback-table + 4-way blend structure but make the fallback predictions better match the metric’s inspiratory-only scoring. The minimal, high-impact fix is to ensure we never force inspiratory-phase predictions to expiratory-like values: for rows where `u_out==1` in test, we switch to a conservative per-(R,C) inspiratory mean (still snapped to valid pressure levels), rather than using time/u_in bins built on inspiratory data that can misbehave on expiratory patterns. We also make blending slightly favor the more time-aware fallback (sub_1/sub_2) only when we’re using fallbacks (i.e., when external submissions are missing), keeping your original equal-weights blend when all external files exist. This preserves the core logic (load/blend submissions; fallback via grouped statistics; snap to levels) while moving predictions closer to how the leaderboard is computed.'
- What this solution (achieved 5.01808) has done: 'We keep your exact “load 4 submissions or fall back, then blend, then snap to valid pressure levels” pipeline, but make the fallback tables more faithful to the time-series structure with minimal risk. Specifically, we add a lightweight per-(R,C,breath_time_index) inspiratory lookup (using the implicit 80 time steps per breath via `cumcount`) and blend it into the existing `pred_time` path; this tends to reduce MAE without changing any model/training logic. We also make the expiratory (`u_out==1`) fallback slightly safer by using a per-(R,C,step) mean when available (still inspiratory-derived, still snapped), otherwise falling back to your existing per-(R,C) mean. All changes are confined to the fallback generator and keep submission format/paths unchanged.'
- What this solution (achieved 5.01579) has done: 'We need to move MAE down from 5.018 toward 2.823 (lower is better) while keeping your core “fallback-table → 4-way blend → snap-to-levels” logic intact. The smallest likely win is to make the fallback lookups more faithful to the true discretization by using exact per-breath time index (`step`) directly (instead of a coarse `t_bin`) as the primary time key, and to prefer medians at that granularity for robustness. This keeps the same training-free grouped-statistics approach, but reduces bias from time-binning and should improve inspiratory predictions without changing evaluation semantics. I also keep your expiratory handling and final snapping unchanged, and still produce `submission.csv` end-to-end.'
- What this solution (achieved 5.18545) has done: 'Your current MAE (5.01579, lower-is-better) is still far from the target (2.82348), so we make a small, metric-aligned improvement while preserving your exact “fallback grouped-statistics → 4-way blend → snap-to-levels” core logic. The highest-impact minimal change is to compute fallback statistics using the same inspiratory-only mask the metric uses, but weighted toward the `u_in`-driven inspiratory portion by filtering `u_out==0` *and* excluding the near-zero-flow tail (`u_in` very small) that behaves more like expiration and can bias the lookups. We keep all existing tables and blending structure, but rebuild the aggregations on this slightly cleaner inspiratory subset and use it consistently for the step-based fallback. This should reduce systematic bias in the fallback predictions and move MAE down toward your target without changing any modeling/training approach.'
- What this solution (achieved 4.93586) has done: 'We need to move MAE down from 5.185 toward the target 2.823 (lower is better), and your current pipeline’s accuracy is dominated by the fallback (since external blend files usually don’t exist here). The most impactful minimal change that preserves your “grouped-statistics fallback → 4-way blend → snap-to-levels” core logic is to make the fallback features closer to the true ventilator dynamics by adding a cumulative integral feature (`u_in_cum`) per breath and using it in an additional lookup table blended into the existing time-aware prediction. This keeps everything training-free and deterministic, but typically reduces bias vs using only instantaneous `u_in` and step. Finally, we keep expiratory handling and the final snapping unchanged, and still write a valid `submission.csv`.'
- What this solution (achieved 4.91892) has done: 'We keep your exact “fallback grouped-statistics → 4-way blend → snap-to-levels” pipeline, but make one metric-aligned improvement in the fallback: use the inspiratory-only `time_step` grid (0..79 steps) more precisely by building an additional per-(R,C,step,prev_u_in_bin) lookup and blending it lightly into `pred_time`. This is still just deterministic grouped statistics (no new model/training loop), and it targets the main weakness of the current fallback: it can’t distinguish early/late dynamics at the same `u_in` value. We also keep your expiratory handling and final snapping unchanged, and still always write a valid `submission.csv`. The change is intentionally small (a modest extra table + a small blend weight) to move MAE down from 4.94 toward 2.82 without rewriting the approach.'
- What this solution (achieved 4.69913) has done: 'We need to move MAE down from 4.9189 toward 2.8235 (lower is better) while keeping your exact “grouped-statistics fallback → 4-way blend → snap-to-levels” core pipeline intact. The smallest likely win is to build one extra fallback table that better matches the ventilator physics: pressure correlates more with *delivered volume* (integral of flow) than instantaneous `u_in`, so we add a per-(R,C,step,u_in_cum_bin) *residual* correction on top of the existing per-(R,C,step) mean. We then blend this correction lightly into `pred_time` (only affects fallback mode) and keep all existing expiratory handling and final snapping unchanged. This is deterministic, training-free, preserves evaluation semantics, and should reduce systematic bias enough to move the score closer to the target.'
- What this solution (achieved 4.69913) has done: 'We keep your exact “fallback grouped-statistics → 4-way blend → snap-to-levels” pipeline, but make one small, metric-aligned improvement to reduce MAE from 4.699 toward the 2.823 target. The main change is to build an extra high-signal fallback table keyed by `(R, C, step, u_in_bin)` but using *cumulative delivered volume* (`u_in_cum`) statistics inside each key (median `u_in_cum` and median pressure), then at inference linearly interpolate pressure based on the test row’s `u_in_cum`—this preserves the same non-ML lookup logic, but better matches ventilator dynamics than a single median per key. We only apply this refinement in fallback mode (when external submissions are missing), and we blend it lightly into your existing `pred_time` so risk is low. Submission writing, alignment, expiratory handling, and final snapping remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sub = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/sample_submission.csv")

blend_paths = {
    "sub_1": "/kaggle/input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
    "sub_2": "/kaggle/input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    "sub_3": "/kaggle/input/gb-vpp-whoppity-dub-dub/median_submission.csv",
    "sub_4": "/kaggle/input/gb-vpp-whoppity-dub-dub/mean_submission.csv",
}


def _load_submission_if_exists(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" not in df.columns or "pressure" not in df.columns:
            raise ValueError(
                f"Found file at {path} but missing required columns ['id','pressure']"
            )
        df = df[["id", "pressure"]].copy()
        df["id"] = df["id"].astype(np.int64)
        df["pressure"] = df["pressure"].astype(np.float64)
        return df
    return None


loaded = {k: _load_submission_if_exists(p) for k, p in blend_paths.items()}




## === cell 2
def _get_pressure_levels() -> np.ndarray:
    train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
    tr = pd.read_csv(train_path, usecols=["u_out", "pressure"])
    tr = tr[tr["u_out"] == 0]
    levels = np.sort(tr["pressure"].unique().astype(np.float64))
    return levels


_PRESSURE_LEVELS = _get_pressure_levels()


def _snap_to_pressure_levels(pred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64, copy=False)
    if pred.size == 0:
        return pred
    idx = np.searchsorted(levels, pred, side="left")
    idx0 = np.clip(idx - 1, 0, levels.size - 1)
    idx1 = np.clip(idx, 0, levels.size - 1)
    v0 = levels[idx0]
    v1 = levels[idx1]
    choose1 = np.abs(pred - v1) < np.abs(pred - v0)
    return np.where(choose1, v1, v0)


def _make_fallback_predictions() -> dict[str, pd.DataFrame]:
    train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
    test_path = "/kaggle/input/ventilator-pressure-prediction/test.csv"

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "u_in", "u_out", "time_step", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "u_in", "u_out", "time_step"]
    )

    for df in (train, test):
        df["R"] = df["R"].astype(np.int16)
        df["C"] = df["C"].astype(np.int16)
        df["breath_id"] = df["breath_id"].astype(np.int64)

    train["step"] = (
        train.groupby("breath_id", observed=True).cumcount().astype(np.int16)
    )
    test["step"] = test.groupby("breath_id", observed=True).cumcount().astype(np.int16)

    train["u_in_cum"] = train.groupby("breath_id", observed=True)["u_in"].cumsum()
    test["u_in_cum"] = test.groupby("breath_id", observed=True)["u_in"].cumsum()

    train_insp = train[(train["u_out"] == 0) & (train["u_in"] > 0.1)].copy()

    ubin = 200  # 0.5 increments (0..200)
    train_insp["u_in_bin"] = np.clip(
        np.rint(train_insp["u_in"].to_numpy() * 2.0), 0, ubin
    ).astype(np.int16)
    test["u_in_bin"] = np.clip(np.rint(test["u_in"].to_numpy() * 2.0), 0, ubin).astype(
        np.int16
    )

    ucb = 400  # 0.5 increments for cumulative signal
    train_insp["u_in_cum_bin"] = np.clip(
        np.rint(train_insp["u_in_cum"].to_numpy() * 2.0), 0, ucb
    ).astype(np.int16)
    test["u_in_cum_bin"] = np.clip(
        np.rint(test["u_in_cum"].to_numpy() * 2.0), 0, ucb
    ).astype(np.int16)

    train_insp["prev_u_in_bin"] = (
        train_insp.groupby("breath_id", observed=True)["u_in_bin"]
        .shift(1)
        .fillna(train_insp["u_in_bin"])
        .astype(np.int16)
    )
    test["prev_u_in_bin"] = (
        test.groupby("breath_id", observed=True)["u_in_bin"]
        .shift(1)
        .fillna(test["u_in_bin"])
        .astype(np.int16)
    )

    grp_cols = ["R", "C", "u_in_bin"]
    agg = (
        train_insp.groupby(grp_cols, observed=True)["pressure"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    grp_cols_step_u = ["R", "C", "u_in_bin", "step"]
    agg_step_u = (
        train_insp.groupby(grp_cols_step_u, observed=True)["pressure"]
        .agg(["median", "mean", "count"])
        .reset_index()
        .rename(
            columns={
                "median": "su_median",
                "mean": "su_mean",
                "count": "su_count",
            }
        )
    )

    grp_cols_step_uc = ["R", "C", "u_in_cum_bin", "step"]
    agg_step_uc = (
        train_insp.groupby(grp_cols_step_uc, observed=True)["pressure"]
        .agg(["median", "mean", "count"])
        .reset_index()
        .rename(
            columns={
                "median": "suc_median",
                "mean": "suc_mean",
                "count": "suc_count",
            }
        )
    )

    grp_cols_step_prev = ["R", "C", "prev_u_in_bin", "step"]
    agg_step_prev = (
        train_insp.groupby(grp_cols_step_prev, observed=True)["pressure"]
        .agg(["median", "mean", "count"])
        .reset_index()
        .rename(
            columns={
                "median": "sp_median",
                "mean": "sp_mean",
                "count": "sp_count",
            }
        )
    )

    agg_step = (
        train_insp.groupby(["R", "C", "step"], observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "step_mean"})
    )

    train_step_mean = train_insp.merge(
        agg_step, on=["R", "C", "step"], how="left", validate="many_to_one"
    )
    train_step_mean["resid"] = (
        train_step_mean["pressure"] - train_step_mean["step_mean"]
    )
    agg_step_uc_resid = (
        train_step_mean.groupby(grp_cols_step_uc, observed=True)["resid"]
        .median()
        .reset_index()
        .rename(columns={"resid": "suc_resid_median"})
    )

    rc_mean = (
        train_insp.groupby(["R", "C"], observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "rc_mean"})
    )
    global_mean = float(train_insp["pressure"].mean())

    _grp_v = ["R", "C", "step", "u_in_bin"]
    v_stats = (
        train_insp.groupby(_grp_v, observed=True)
        .agg(
            v_u_cum_med=("u_in_cum", "median"),
            v_p_med=("pressure", "median"),
            v_cnt=("pressure", "size"),
        )
        .reset_index()
    )

    test2 = (
        test.merge(agg_step_u, on=grp_cols_step_u, how="left")
        .merge(agg_step_uc, on=grp_cols_step_uc, how="left")
        .merge(agg_step_prev, on=grp_cols_step_prev, how="left")
        .merge(agg, on=grp_cols, how="left")
        .merge(agg_step, on=["R", "C", "step"], how="left")
        .merge(agg_step_uc_resid, on=grp_cols_step_uc, how="left")
        .merge(rc_mean, on=["R", "C"], how="left")
    )

    v_bins = (
        v_stats[["R", "C", "step", "u_in_bin"]]
        .drop_duplicates(subset=["R", "C", "step", "u_in_bin"])
        .sort_values(["R", "C", "step", "u_in_bin"], kind="mergesort")
        .reset_index(drop=True)
    )

    left_sorted = (
        test2[["id", "R", "C", "step", "u_in_bin"]]
        .sort_values(["R", "C", "step", "u_in_bin", "id"], kind="mergesort")
        .reset_index(drop=True)
    )

    lo = pd.merge_asof(
        left_sorted,
        v_bins,
        on="u_in_bin",
        by=["R", "C", "step"],
        direction="backward",
        allow_exact_matches=True,
    ).rename(columns={"u_in_bin_y": "u_in_bin_lo", "u_in_bin_x": "u_in_bin"})

    hi = pd.merge_asof(
        left_sorted,
        v_bins,
        on="u_in_bin",
        by=["R", "C", "step"],
        direction="forward",
        allow_exact_matches=True,
    ).rename(columns={"u_in_bin_y": "u_in_bin_hi", "u_in_bin_x": "u_in_bin"})

    test2 = test2.merge(
        lo[["id", "u_in_bin_lo"]], on="id", how="left", validate="one_to_one"
    ).merge(hi[["id", "u_in_bin_hi"]], on="id", how="left", validate="one_to_one")

    v1 = v_stats.rename(
        columns={
            "u_in_bin": "u_in_bin_lo",
            "v_u_cum_med": "v_u_cum_med_1",
            "v_p_med": "v_p_med_1",
            "v_cnt": "v_cnt_1",
        }
    )
    v2 = v_stats.rename(
        columns={
            "u_in_bin": "u_in_bin_hi",
            "v_u_cum_med": "v_u_cum_med_2",
            "v_p_med": "v_p_med_2",
            "v_cnt": "v_cnt_2",
        }
    )

    test2 = test2.merge(
        v1, on=["R", "C", "step", "u_in_bin_lo"], how="left", validate="many_to_one"
    )
    test2 = test2.merge(
        v2, on=["R", "C", "step", "u_in_bin_hi"], how="left", validate="many_to_one"
    )

    su_mean = test2["su_mean"].to_numpy()
    su_med = test2["su_median"].to_numpy()

    suc_mean = test2["suc_mean"].to_numpy()
    suc_med = test2["suc_median"].to_numpy()

    sp_mean = test2["sp_mean"].to_numpy()
    sp_med = test2["sp_median"].to_numpy()

    base_mean = test2["mean"].to_numpy()
    base_med = test2["median"].to_numpy()
    rc_fallback = test2["rc_mean"].to_numpy()
    step_mean = test2["step_mean"].to_numpy()

    pred_time = np.where(np.isnan(su_med), su_mean, su_med)
    pred_time = np.where(np.isnan(pred_time), base_med, pred_time)
    pred_time = np.where(np.isnan(pred_time), base_mean, pred_time)
    pred_time = np.where(np.isnan(pred_time), rc_fallback, pred_time)
    pred_time = np.where(np.isnan(pred_time), global_mean, pred_time)

    pred_cum = np.where(np.isnan(suc_med), suc_mean, suc_med)
    pred_cum = np.where(np.isnan(pred_cum), pred_time, pred_cum)

    pred_prev = np.where(np.isnan(sp_med), sp_mean, sp_med)
    pred_prev = np.where(np.isnan(pred_prev), pred_time, pred_prev)

    pred_base_mean = np.where(np.isnan(base_mean), rc_fallback, base_mean)
    pred_base_med = np.where(np.isnan(base_med), rc_fallback, base_med)
    pred_base_mean = np.where(np.isnan(pred_base_mean), global_mean, pred_base_mean)
    pred_base_med = np.where(np.isnan(pred_base_med), global_mean, pred_base_med)

    step_fallback = np.where(np.isnan(step_mean), rc_fallback, step_mean)
    step_fallback = np.where(np.isnan(step_fallback), global_mean, step_fallback)

    pred_time = 0.85 * pred_time + 0.15 * step_fallback
    pred_time = 0.80 * pred_time + 0.20 * pred_cum
    pred_time = 0.90 * pred_time + 0.10 * pred_prev

    suc_resid = test2["suc_resid_median"].to_numpy()
    suc_resid = np.where(np.isnan(suc_resid), 0.0, suc_resid)
    pred_time = pred_time + (0.15 * suc_resid)

    u_cum_test = test2["u_in_cum"].to_numpy(dtype=np.float64)
    uc1 = test2["v_u_cum_med_1"].to_numpy(dtype=np.float64)
    uc2 = test2["v_u_cum_med_2"].to_numpy(dtype=np.float64)
    p1 = test2["v_p_med_1"].to_numpy(dtype=np.float64)
    p2 = test2["v_p_med_2"].to_numpy(dtype=np.float64)

    ok = (
        (~np.isnan(uc1))
        & (~np.isnan(uc2))
        & (~np.isnan(p1))
        & (~np.isnan(p2))
        & (uc2 != uc1)
    )
    t = np.zeros_like(u_cum_test, dtype=np.float64)
    t[ok] = (u_cum_test[ok] - uc1[ok]) / (uc2[ok] - uc1[ok])
    t = np.clip(t, 0.0, 1.0)
    pred_v = (1.0 - t) * p1 + t * p2
    pred_time = np.where(ok, (0.90 * pred_time + 0.10 * pred_v), pred_time)

    mix_1 = 0.70 * pred_time + 0.30 * pred_base_mean
    mix_2 = 0.70 * pred_time + 0.30 * pred_base_med

    u_out_test = test2["u_out"].to_numpy(dtype=np.int8)
    rc_only = np.where(np.isnan(rc_fallback), global_mean, rc_fallback)

    exp_only = np.where(np.isnan(step_mean), rc_only, step_mean)
    exp_only = np.where(np.isnan(exp_only), rc_only, exp_only)

    pred_time = np.where(u_out_test == 1, exp_only, pred_time)
    pred_base_mean = np.where(u_out_test == 1, exp_only, pred_base_mean)
    pred_base_med = np.where(u_out_test == 1, exp_only, pred_base_med)
    mix_1 = np.where(u_out_test == 1, exp_only, mix_1)
    mix_2 = np.where(u_out_test == 1, exp_only, mix_2)

    pred_time_s = _snap_to_pressure_levels(pred_time, _PRESSURE_LEVELS)
    pred_base_mean_s = _snap_to_pressure_levels(pred_base_mean, _PRESSURE_LEVELS)
    pred_base_med_s = _snap_to_pressure_levels(pred_base_med, _PRESSURE_LEVELS)
    mix_1_s = _snap_to_pressure_levels(mix_1, _PRESSURE_LEVELS)
    mix_2_s = _snap_to_pressure_levels(mix_2, _PRESSURE_LEVELS)

    ids = test2["id"].to_numpy(dtype=np.int64)
    df_base_mean = pd.DataFrame({"id": ids, "pressure": pred_base_mean_s})
    df_base_med = pd.DataFrame({"id": ids, "pressure": pred_base_med_s})
    df_mix_1 = pd.DataFrame({"id": ids, "pressure": mix_1_s})
    df_mix_2 = pd.DataFrame({"id": ids, "pressure": mix_2_s})

    return {
        "sub_1": df_mix_1,
        "sub_2": df_mix_2,
        "sub_3": df_base_med,
        "sub_4": df_base_mean,
    }


missing_any = any(v is None for v in loaded.values())
if missing_any:
    fallback = _make_fallback_predictions()
    for k in loaded:
        if loaded[k] is None:
            loaded[k] = fallback[k]

sub_1, sub_2, sub_3, sub_4 = (
    loaded["sub_1"],
    loaded["sub_2"],
    loaded["sub_3"],
    loaded["sub_4"],
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1643282379.py in <cell line: 0>()
    336 missing_any = any(v is None for v in loaded.values())
    337 if missing_any:
--> 338     fallback = _make_fallback_predictions()
    339     for k in loaded:
    340         if loaded[k] is None:

/tmp/ipykernel_11/1643282379.py in _make_fallback_predictions()
    192     )
    193 
--> 194     lo = pd.merge_asof(
    195         left_sorted,
    196         v_bins,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge_asof(left, right, on, left_on, right_on, left_index, right_index, by, left_by, right_by, suffixes, tolerance, allow_exact_matches, direction)
    706         direction=direction,
    707     )
--> 708     return op.get_result()
    709 
    710 

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in get_result(self, copy)
   1924 
   1925     def get_result(self, copy: bool | None = True) -> DataFrame:
-> 1926         join_index, left_indexer, right_indexer = self._get_join_info()
   1927 
   1928         left_join_indexer: npt.NDArray[np.intp] | None

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_join_info(self)
   1149             )
   1150         else:
-> 1151             (left_indexer, right_indexer) = self._get_join_indexers()
   1152 
   1153             if self.right_index:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_join_indexers(self)
   2236 
   2237         # initial type conversion as needed
-> 2238         left_values = self._convert_values_for_libjoin(left_values, "left")
   2239         right_values = self._convert_values_for_libjoin(right_values, "right")
   2240 

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _convert_values_for_libjoin(self, values, side)
   2180             if isna(values).any():
   2181                 raise ValueError(f"Merge keys contain null values on {side} side")
-> 2182             raise ValueError(f"{side} keys must be sorted")
   2183 
   2184         if isinstance(values, ArrowExtensionArray):

ValueError: left keys must be sorted

## === cell 3
def _align_to_sub(base_sub: pd.DataFrame, pred_sub: pd.DataFrame) -> np.ndarray:
    merged = base_sub[["id"]].merge(
        pred_sub[["id", "pressure"]], on="id", how="left", validate="one_to_one"
    )
    if merged["pressure"].isna().any():
        merged["pressure"] = merged["pressure"].fillna(0.0)
    return merged["pressure"].to_numpy(dtype=np.float64)


p1 = _align_to_sub(sub, sub_1)
p2 = _align_to_sub(sub, sub_2)
p3 = _align_to_sub(sub, sub_3)
p4 = _align_to_sub(sub, sub_4)

if missing_any:
    w1, w2, w3, w4 = 0.35, 0.35, 0.15, 0.15
else:
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25

sub["pressure"] = (p1 * w1) + (p2 * w2) + (p3 * w3) + (p4 * w4)

sub["pressure"] = _snap_to_pressure_levels(
    sub["pressure"].to_numpy(dtype=np.float64), _PRESSURE_LEVELS
)

sub.to_csv("submission.csv", index=False)
sub.head(5)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2537295345.py in <cell line: 0>()
      8 
      9 
---> 10 p1 = _align_to_sub(sub, sub_1)
     11 p2 = _align_to_sub(sub, sub_2)
     12 p3 = _align_to_sub(sub, sub_3)

NameError: name 'sub_1' is not defined
