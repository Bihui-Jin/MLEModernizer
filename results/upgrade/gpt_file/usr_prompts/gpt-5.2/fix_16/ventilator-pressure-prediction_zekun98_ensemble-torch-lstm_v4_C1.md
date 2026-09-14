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

0.1671

# 6. Current score

4.59

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because the notebook expects `../input/torch*` folders with `oof.csv` and `submission.csv`, but none exist in your environment, so `pd.concat` receives an empty list. I keep the ensemble logic intact when those files are present, and add a safe fallback that generates a valid `submission.csv` directly from `sample_submission.csv` (all-zero pressures) when they are not. This makes the pipeline run end-to-end and always write a correctly formatted `.csv` submission file. The fallback is score-poor but is the minimal correctness fix needed since no model artifacts are available to ensemble.'
- What this solution (achieved 8.13469) has done: 'I fix the `KeyError: 'pressure'` by avoiding the merge that creates `pressure_x/pressure_y` columns and instead assign predictions directly onto the existing `sub['pressure']` after aligning by `id`. I also make the input path resolution robust by trying both `../input/...` and `/kaggle/input/...` so it runs in your provided filesystem layout. The fallback model logic (group means by `R,C,time_step` then `R,C` then global mean) is preserved exactly; the change is only to ensure correct column handling and a valid `submission.csv` is always written.'
- What this solution (achieved 8.13042) has done: 'Your current fallback predictor is leaving a lot of score on the table because it ignores the strongest signal in this competition: the fact that pressure values are *discrete* and come from a fixed set seen in training. I keep your exact fallback logic (group means by `R,C,time_step` then `R,C` then global mean) and only add a minimal post-processing step that “snaps” predictions to the nearest known training pressure value, which typically reduces MAE substantially for this task. I also slightly extend the fallback features without changing the modeling approach by including `u_in` and `u_out` in the most-granular group key (still just a group-mean lookup), which should move your score closer to the 0.1671 target. The ensemble-from-existing-submissions path remains unchanged and still takes priority when present, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 8.13194) has done: 'Your fallback currently predicts pressure as a pure lookup on \[R,C,time_step,u_in,u_out\] and then snaps to discrete pressure levels, but it still wastes signal by not using the strongest time-series features that remain “lookup-friendly” (lagged u_in/u_out and cumulative u_in). I keep the exact same core approach (group-mean tables + left merges + hierarchical fill + snapping), and only (1) add a few deterministic, cheap engineered keys (u_in lag1/lag2 and cumulative u_in per breath, all quantized), and (2) slightly tune quantization so more test rows hit the most-granular table, which should move MAE down toward your 0.1671 target. The ensemble path (using existing `../input/torch*/submission.csv`) remains unchanged and still takes priority when present. The script still run end-to-end in your filesystem layout and always write a valid `submission.csv`.'
- What this solution (achieved 4.93923) has done: 'Your current fallback is still far from the target, so the fastest way to move MAE down (without changing the overall “group-mean lookup + hierarchical fill + snapping” logic) is to (1) ensure your `time_step` join key matches train/test exactly by quantizing it consistently, and (2) add one more very cheap, deterministic lookup signal that’s highly predictive in this dataset: the within-breath time index (0..79) and a quantized cumulative `u_in` built on the same grid. These changes increase hit-rate of the most-granular lookup tables and reduce reliance on coarse/global means, which should reduce MAE toward your 0.1671 target. The ensemble-from-existing-submissions path is kept intact and still takes priority when those files exist, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 4.93941) has done: 'I keep your exact “hierarchical group-mean lookup + fill + snap-to-discrete-levels” fallback logic, but make two minimal changes that typically reduce MAE a lot for this competition: (1) enforce the competition’s scoring semantics by forcing pressure to 0 on expiratory steps (`u_out==1`) in the test predictions (those rows are not scored, and this avoids injecting noisy values), and (2) build the mean tables using only inspiratory rows from train (`u_out==0`) so the lookups match the evaluated regime better. Everything else (keys, quantization, merges, hierarchical fallback order, snapping) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 4.90455) has done: 'Your current fallback already does hierarchical group-mean lookups and snapping, but it still overfits to a too-granular key and then backs off to very coarse means when unseen combinations appear in test, which keeps MAE high. I keep the exact same overall approach (mean-table lookups → hierarchical fill → snap-to-known pressure levels → force expiratory to 0), but add one extra *intermediate* lookup table that drops only the most brittle lag/cum features while keeping the key drivers (`R,C,time_step,t_idx,u_in,u_out`). This should increase the hit rate of a “near-best” table and reduce reliance on the coarse `R,C,time_step`/`R,C` means, moving MAE down toward your target. I also ensure the train/test sort order for feature generation is identical by sorting on `breath_id,t_idx` after creating `t_idx` (still the same features, just more consistent alignment).'
- What this solution (achieved 4.90455) has done: 'Your current fallback is still missing one of the strongest “legal” tricks for this competition: the fact that pressure is not only discrete, but also tightly coupled to the inspiratory regime, so mixing expiratory behavior into the snapping set and/or leaving inspiratory predictions unbounded can keep MAE high. I keep your exact hierarchical mean-lookup → fill → snap approach, but (1) build the snapping levels from the full train pressure set (not just inspiratory) to better match the competition’s discrete grid, and (2) clip predictions to the observed pressure range before snapping to avoid snapping artifacts at the extremes. These are minimal post-processing adjustments that don’t change your modeling approach, and they should move the MAE down from ~4.90 toward the 0.1671 target without altering file paths or output format. The ensemble-from-existing-submissions path remains unchanged.'
- What this solution (achieved 4.90455) has done: 'Your current fallback is still far above the target (MAE 4.90 vs 0.1671), so we need a small change that meaningfully reduces error while keeping your “group-mean lookup → hierarchical fill → snap-to-discrete-levels → force expiratory to 0” core logic intact. The biggest remaining mismatch is that your inspiratory predictions can be systematically shifted; we can correct that with a single global calibration offset computed on train inspiratory rows (predicting train from the same lookup tables, then taking the median residual) and then applying that offset to test before snapping. This does not change the model family or features—just a deterministic post-calibration consistent with MAE minimization—and is usually a sizable improvement for this competition. Everything else (paths, merges, fallback order, snapping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 4.92511) has done: 'Your current score (4.90455 MAE) is far worse than the target (0.1671), so we should make a minimal change that legitimately reduces error without changing the core “hierarchical group-mean lookup → fill → snap-to-discrete-levels → force expiratory to 0” approach. The biggest remaining systematic issue is that all your mean tables are built from *all inspiratory rows*, including rows where the pressure is changing rapidly; smoothing that training signal slightly within each breath (using a small centered rolling mean on pressure only for building lookup tables) typically improves MAE for this competition while keeping the same lookup logic. We keep calibration, snapping, and expiratory forcing identical, and we only change the pressure values used to compute the group means (not the labels used for calibration residuals). This should move MAE down toward the target without introducing new models or altering file paths/output format.'
- What this solution (achieved 4.92516) has done: 'We’re far from the target (MAE 4.93 vs 0.1671, lower is better), so we need a meaningful improvement while keeping your core “hierarchical group-mean lookup → fill → calibration offset → snap-to-discrete → force expiratory to 0” logic intact. The biggest remaining mismatch is that the scoring ignores expiratory steps (`u_out==1`), but your calibration offset is computed on *all* inspiratory rows equally; weighting calibration toward later inspiratory steps (where pressure is more stable and your lookup is more reliable) typically reduces systematic bias and MAE without changing the model family. I implement a minimal calibration tweak: compute the offset on a filtered subset of inspiratory rows (e.g., excluding the first few timesteps in each breath) and use the **median** residual there, then apply it exactly as before. Everything else (paths, mean tables, smoothing, merge keys, snapping, expiratory forcing, and writing `submission.csv`) stays the same.'
- What this solution (achieved 4.925) has done: 'Your current fallback is still far above the target (4.93 vs 0.1671 MAE), so we need a small but impactful improvement without changing the overall “hierarchical group-mean lookup → fill → calibration offset → snap-to-discrete → force expiratory to 0” approach. The biggest remaining weakness is that the calibration offset is computed once globally, but prediction bias varies strongly by lung settings; we can reduce MAE by computing the same median-residual offset **per (R,C)** on inspiratory rows and applying it in test based on each row’s (R,C), falling back to the global offset when needed. This keeps the exact model family and lookup tables intact—only the deterministic calibration step becomes slightly more specific. Everything else (paths, features/keys, smoothing, merge order, snapping, expiratory forcing, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 4.58974) has done: 'Your current score (MAE ~4.925, lower is better) is still far from the 0.1671 target, so we need a small change that can materially reduce error while keeping your exact “hierarchical group-mean lookup → fill → (R,C) calibration offset → snap-to-discrete → force expiratory to 0” core logic intact. The biggest remaining mismatch is that your lookup tables don’t use the strongest non-leaky breath-level state proxy: `area` (cumulative integral of `u_in` over time), which is highly predictive of pressure and is safe because it uses only inputs. I add `area` (quantized) as an additional key in a new top-priority mean table (and a mid table) while leaving all existing tables and fallback order in place, so it only improves rows where the richer key matches and otherwise behaves exactly as before. I also compute `area` identically for train/test using the existing per-breath ordering to avoid any alignment changes, and keep your calibration/snapping/expiratory forcing unchanged.'
- What this solution (achieved 4.59) has done: 'Your current fallback is still far above the target (4.58974 vs 0.1671 MAE, lower is better), so we need a minimal change that reduces error without changing the core “hierarchical group-mean lookup → fill → (R,C) calibration offset → snap-to-discrete → force expiratory to 0” approach. The biggest remaining gain within the same logic is to add one more high-signal, non-leaky input-derived state: **cumulative volume** `u_in * dt` (separate from your existing `area` which is the cumulative integral), then use it only as an additional key in new top/mid mean tables so it improves matches where available and otherwise falls back exactly as before. I keep all your existing tables and fallback order intact, simply inserting these new `*_vol_*` tables ahead of the existing ones, and compute `dt/area` consistently per breath to avoid train/test mismatch. This should increase “exact key hit rate” on inspiratory rows and move MAE downward toward the target band while preserving semantics and producing the same `submission.csv` output format.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import glob
import os

paths = glob.glob("../input/torch*")

valid_oof_paths = [p for p in paths if os.path.isfile(os.path.join(p, "oof.csv"))]
valid_sub_paths = [
    p for p in paths if os.path.isfile(os.path.join(p, "submission.csv"))
]

print(
    f"Found {len(paths)} torch* paths, {len(valid_oof_paths)} with oof.csv, {len(valid_sub_paths)} with submission.csv."
)



## === cell 1
df = None
if len(valid_oof_paths) > 0:
    df = pd.concat(
        [pd.read_csv(os.path.join(p, "oof.csv")) for p in valid_oof_paths],
        ignore_index=True,
    )
    if "pred" in df.columns:
        df = df[df.pred != 0]
else:
    print("No oof.csv files found under ../input/torch*. Skipping OOF evaluation.")



## === cell 2
if df is not None and {"pred", "pressure"}.issubset(df.columns):
    print("OOF MAE:", np.mean(np.abs(df["pred"] - df["pressure"])))
else:
    print("OOF MAE not computed (missing df or required columns).")




## === cell 3
def first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


sample_path = first_existing_path(
    [
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

train_path = first_existing_path(
    [
        "../input/ventilator-pressure-prediction/train.csv",
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
    ]
)

test_path = first_existing_path(
    [
        "../input/ventilator-pressure-prediction/test.csv",
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/input/test.csv",
    ]
)

sub = pd.read_csv(sample_path)

if len(valid_sub_paths) > 0:
    preds = []
    for p in valid_sub_paths:
        s = pd.read_csv(os.path.join(p, "submission.csv"))
        if "pressure" not in s.columns:
            raise ValueError(f"Missing 'pressure' column in {p}/submission.csv")
        if len(s) != len(sub):
            raise ValueError(
                f"Row count mismatch for {p}/submission.csv: got {len(s)} expected {len(sub)}"
            )
        preds.append(s["pressure"].to_numpy(dtype=np.float64))
    sub["pressure"] = np.mean(np.vstack(preds), axis=0)

else:
    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )

    train_ins = train[train["u_out"] == 0].copy()

    train_ins["time_step_q"] = train_ins["time_step"].round(2).astype(np.float32)
    test["time_step_q"] = test["time_step"].round(2).astype(np.float32)

    train_ins["u_in_q"] = train_ins["u_in"].round(2).astype(np.float32)
    test["u_in_q"] = test["u_in"].round(2).astype(np.float32)

    for df_ in (train_ins, test):
        df_.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

        df_["t_idx"] = df_.groupby("breath_id", sort=False).cumcount().astype(np.int16)

        df_["u_in_lag1"] = (
            df_.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )
        df_["u_in_lag2"] = (
            df_.groupby("breath_id", sort=False)["u_in"].shift(2).fillna(0.0)
        )
        df_["u_out_lag1"] = (
            df_.groupby("breath_id", sort=False)["u_out"]
            .shift(1)
            .fillna(0)
            .astype(np.int8)
        )

        df_["u_in_lag1_q"] = df_["u_in_lag1"].round(2).astype(np.float32)
        df_["u_in_lag2_q"] = df_["u_in_lag2"].round(2).astype(np.float32)

        df_["u_in_cum"] = df_.groupby("breath_id", sort=False)["u_in"].cumsum()
        df_["u_in_cum_q"] = df_["u_in_cum"].round(2).astype(np.float32)

        df_["dt"] = (
            df_.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )

        df_["vol"] = (
            df_["u_in"].astype(np.float64) * df_["dt"].astype(np.float64)
        ).astype(np.float64)
        df_["vol_q"] = df_["vol"].round(2).astype(np.float32)

        df_["area"] = df_["vol"].groupby(df_["breath_id"], sort=False).cumsum()
        df_["area_q"] = df_["area"].round(2).astype(np.float32)

        df_.sort_values(["breath_id", "t_idx"], inplace=True, kind="mergesort")

    key_cols_rc_t = ["R", "C", "time_step_q"]
    key_cols_rc_t_u = ["R", "C", "time_step_q", "u_in_q", "u_out"]

    key_cols_rc_t_u_ext = [
        "R",
        "C",
        "time_step_q",
        "t_idx",
        "u_in_q",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_out_lag1",
        "u_in_cum_q",
    ]

    key_cols_rc_t_u_mid = ["R", "C", "time_step_q", "t_idx", "u_in_q", "u_out"]

    key_cols_rc_t_u_area_ext = key_cols_rc_t_u_ext + ["area_q"]
    key_cols_rc_t_u_area_mid = key_cols_rc_t_u_mid + ["area_q"]

    key_cols_rc_t_u_vol_area_ext = key_cols_rc_t_u_ext + ["vol_q", "area_q"]
    key_cols_rc_t_u_vol_area_mid = key_cols_rc_t_u_mid + ["vol_q", "area_q"]

    train_ins_for_means = train_ins.copy()
    train_ins_for_means["pressure_smooth"] = (
        train_ins_for_means.groupby("breath_id", sort=False)["pressure"]
        .transform(lambda s: s.rolling(window=3, center=True, min_periods=1).mean())
        .astype(np.float64)
    )

    mean_rc_t_u_vol_area_ext = (
        train_ins_for_means.groupby(key_cols_rc_t_u_vol_area_ext, sort=False)[
            "pressure_smooth"
        ]
        .mean()
        .rename("pressure_mean_rc_t_u_vol_area_ext")
        .reset_index()
    )

    mean_rc_t_u_vol_area_mid = (
        train_ins_for_means.groupby(key_cols_rc_t_u_vol_area_mid, sort=False)[
            "pressure_smooth"
        ]
        .mean()
        .rename("pressure_mean_rc_t_u_vol_area_mid")
        .reset_index()
    )

    mean_rc_t_u_area_ext = (
        train_ins_for_means.groupby(key_cols_rc_t_u_area_ext, sort=False)[
            "pressure_smooth"
        ]
        .mean()
        .rename("pressure_mean_rc_t_u_area_ext")
        .reset_index()
    )

    mean_rc_t_u_area_mid = (
        train_ins_for_means.groupby(key_cols_rc_t_u_area_mid, sort=False)[
            "pressure_smooth"
        ]
        .mean()
        .rename("pressure_mean_rc_t_u_area_mid")
        .reset_index()
    )

    mean_rc_t_u_ext = (
        train_ins_for_means.groupby(key_cols_rc_t_u_ext, sort=False)["pressure_smooth"]
        .mean()
        .rename("pressure_mean_rc_t_u_ext")
        .reset_index()
    )

    mean_rc_t_u_mid = (
        train_ins_for_means.groupby(key_cols_rc_t_u_mid, sort=False)["pressure_smooth"]
        .mean()
        .rename("pressure_mean_rc_t_u_mid")
        .reset_index()
    )

    mean_rc_t_u = (
        train_ins_for_means.groupby(key_cols_rc_t_u, sort=False)["pressure_smooth"]
        .mean()
        .rename("pressure_mean_rc_t_u")
        .reset_index()
    )

    mean_rc_t = (
        train_ins_for_means.groupby(key_cols_rc_t, sort=False)["pressure_smooth"]
        .mean()
        .rename("pressure_mean_rc_t")
        .reset_index()
    )

    mean_rc = (
        train_ins_for_means.groupby(["R", "C"], sort=False)["pressure_smooth"]
        .mean()
        .rename("pressure_mean_rc")
        .reset_index()
    )

    global_mean = float(train_ins_for_means["pressure_smooth"].mean())

    train_ins_pred_frame = train_ins.merge(
        mean_rc_t_u_vol_area_ext, on=key_cols_rc_t_u_vol_area_ext, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t_u_vol_area_mid, on=key_cols_rc_t_u_vol_area_mid, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t_u_area_ext, on=key_cols_rc_t_u_area_ext, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t_u_area_mid, on=key_cols_rc_t_u_area_mid, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t_u_ext, on=key_cols_rc_t_u_ext, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t_u_mid, on=key_cols_rc_t_u_mid, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t_u, on=key_cols_rc_t_u, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc_t, on=key_cols_rc_t, how="left"
    )
    train_ins_pred_frame = train_ins_pred_frame.merge(
        mean_rc, on=["R", "C"], how="left"
    )

    train_pred = train_ins_pred_frame["pressure_mean_rc_t_u_vol_area_ext"]
    train_pred = train_pred.fillna(
        train_ins_pred_frame["pressure_mean_rc_t_u_vol_area_mid"]
    )
    train_pred = train_pred.fillna(
        train_ins_pred_frame["pressure_mean_rc_t_u_area_ext"]
    )
    train_pred = train_pred.fillna(
        train_ins_pred_frame["pressure_mean_rc_t_u_area_mid"]
    )
    train_pred = train_pred.fillna(train_ins_pred_frame["pressure_mean_rc_t_u_ext"])
    train_pred = train_pred.fillna(train_ins_pred_frame["pressure_mean_rc_t_u_mid"])
    train_pred = train_pred.fillna(train_ins_pred_frame["pressure_mean_rc_t_u"])
    train_pred = train_pred.fillna(train_ins_pred_frame["pressure_mean_rc_t"])
    train_pred = train_pred.fillna(train_ins_pred_frame["pressure_mean_rc"])
    train_pred = train_pred.fillna(global_mean).astype(np.float64)

    calib_mask = train_ins_pred_frame["t_idx"] >= 5
    resid_series = train_ins_pred_frame.loc[calib_mask, "pressure"].astype(
        np.float64
    ) - train_pred.loc[calib_mask].astype(np.float64)
    rc_offsets = (
        train_ins_pred_frame.loc[calib_mask, ["R", "C"]]
        .assign(resid=resid_series.to_numpy(dtype=np.float64))
        .groupby(["R", "C"], sort=False)["resid"]
        .median()
        .rename("calib_offset_rc")
        .reset_index()
    )
    calib_offset_global = float(np.median(resid_series.to_numpy(dtype=np.float64)))
    print(
        "Calibration offset global (median residual, train inspiratory t_idx>=5):",
        calib_offset_global,
    )
    print("Calibration offsets per (R,C):", rc_offsets.shape[0])

    test = test.merge(
        mean_rc_t_u_vol_area_ext, on=key_cols_rc_t_u_vol_area_ext, how="left"
    )
    test = test.merge(
        mean_rc_t_u_vol_area_mid, on=key_cols_rc_t_u_vol_area_mid, how="left"
    )
    test = test.merge(mean_rc_t_u_area_ext, on=key_cols_rc_t_u_area_ext, how="left")
    test = test.merge(mean_rc_t_u_area_mid, on=key_cols_rc_t_u_area_mid, how="left")
    test = test.merge(mean_rc_t_u_ext, on=key_cols_rc_t_u_ext, how="left")
    test = test.merge(mean_rc_t_u_mid, on=key_cols_rc_t_u_mid, how="left")
    test = test.merge(mean_rc_t_u, on=key_cols_rc_t_u, how="left")
    test = test.merge(mean_rc_t, on=key_cols_rc_t, how="left")
    test = test.merge(mean_rc, on=["R", "C"], how="left")

    pred = test["pressure_mean_rc_t_u_vol_area_ext"]
    pred = pred.fillna(test["pressure_mean_rc_t_u_vol_area_mid"])
    pred = pred.fillna(test["pressure_mean_rc_t_u_area_ext"])
    pred = pred.fillna(test["pressure_mean_rc_t_u_area_mid"])
    pred = pred.fillna(test["pressure_mean_rc_t_u_ext"])
    pred = pred.fillna(test["pressure_mean_rc_t_u_mid"])
    pred = pred.fillna(test["pressure_mean_rc_t_u"])
    pred = pred.fillna(test["pressure_mean_rc_t"])
    pred = pred.fillna(test["pressure_mean_rc"])
    pred = pred.fillna(global_mean).astype(np.float64)

    test = test.merge(rc_offsets, on=["R", "C"], how="left")
    offset_vec = (
        test["calib_offset_rc"].fillna(calib_offset_global).to_numpy(dtype=np.float64)
    )
    pred = (pred.to_numpy(dtype=np.float64) + offset_vec).astype(np.float64)

    train_p_min = float(train["pressure"].min())
    train_p_max = float(train["pressure"].max())
    pred_np = np.clip(pred, train_p_min, train_p_max)

    pressure_levels = np.sort(train["pressure"].unique()).astype(np.float64)

    idx = np.searchsorted(pressure_levels, pred_np, side="left")
    idx0 = np.clip(idx - 1, 0, len(pressure_levels) - 1)
    idx1 = np.clip(idx, 0, len(pressure_levels) - 1)
    p0 = pressure_levels[idx0]
    p1 = pressure_levels[idx1]
    snapped = np.where(np.abs(pred_np - p0) <= np.abs(pred_np - p1), p0, p1).astype(
        np.float64
    )

    pred_by_id = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": snapped})
    pred_by_id = pred_by_id.drop_duplicates(subset=["id"], keep="first").set_index(
        "id"
    )["pressure"]

    sub["pressure"] = sub["id"].map(pred_by_id).astype(np.float64)
    sub["pressure"] = sub["pressure"].fillna(global_mean).astype(np.float64)

    test_uout_by_id = test.drop_duplicates(subset=["id"], keep="first").set_index("id")[
        "u_out"
    ]
    uout_aligned = sub["id"].map(test_uout_by_id).fillna(0).astype(np.int8).to_numpy()
    sub.loc[uout_aligned == 1, "pressure"] = 0.0

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("pressure dtype:", sub["pressure"].dtype)
print("pressure stats:", float(sub["pressure"].min()), float(sub["pressure"].max()))
