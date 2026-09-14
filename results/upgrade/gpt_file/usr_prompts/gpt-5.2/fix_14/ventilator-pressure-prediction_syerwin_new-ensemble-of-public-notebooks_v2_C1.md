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

3.9

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

0.1623496609212457

# 6. Current score

2.78922

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.54738) has done: 'Your notebook fails because it tries to read four external Kaggle Dataset paths that don’t exist in this environment, so `sub_1..sub_4` are never created and the blend crashes. To keep the “blend submissions” core logic while making it runnable end-to-end, I add a tiny fallback that searches `../input/**/submission.csv` for any available submission files and blends them; if none are found, it falls back to a simple baseline prediction built only from `train.csv` (median pressure during inspiratory phase) so a valid `submission.csv` is always produced. I also harden alignment by merging on `id` to avoid ordering mismatches and ensure the output columns exactly match `id,pressure`. These changes are execution-unblocking and should yield a non-trivial MAE (better than all-zeros) without changing the fundamental “weighted ensemble of submissions” intent.'
- What this solution (achieved 3.76812) has done: 'Your current score is far worse than the target (lower-is-better), and the main reason is that the fallback predictor ignores the key scoring rule (only inspiratory phase is scored) and ignores breath dynamics, so it produces near-constant pressures. To move the MAE sharply toward the target while preserving the “submission blending” core logic, I keep your blending as-is when external submissions exist, but replace the weak constant fallback with a minimal, purely training-data-based kNN-style regressor over the instantaneous control/state features (`R,C,time_step,u_in,u_out`) restricted to inspiratory samples. This stays within the same pipeline (read train/test → produce `submission.csv`) and doesn’t introduce any new ML framework or training loop, but it yields a much more realistic pressure mapping and should substantially reduce MAE versus a constant. I also ensure correct `id` alignment by merging predictions back onto the sample submission’s `id` order.'
- What this solution (achieved 2.83506) has done: 'I fix the runtime KeyError by ensuring the `tq2` feature is created consistently for both train and test (it was computed in `test_k2` but then mistakenly accessed from `test_k`). I keep the existing fallback modeling logic intact (the hierarchical median lookup with breath-derived features and pressure snapping), only correcting the feature pipeline so it runs end-to-end. I also make the CSV writing a bit more robust by always writing `submission.csv` with the required `id,pressure` columns and ensuring predictions are numeric. These changes are execution-unblocking and should improve the score versus not yielding any submission, without altering the intended approach.'
- What this solution (achieved 2.83506) has done: 'Your current MAE (2.83506) is still far above the target (0.1623; lower is better), so we should improve the fallback predictor while keeping the same overall “blend external submissions else build a train-based lookup and snap to pressure grid” core logic. The simplest high-impact fix is to ensure the fallback also predicts *zero pressure during expiratory phase* (`u_out==1`), because those rows are not scored and setting them to 0 generally reduces overall error without affecting the inspiratory-scored part. I keep your hierarchical median lookup exactly as-is for inspiratory rows, but compute it only where `u_out==0` and then explicitly set expiratory predictions to 0 before writing the submission. This is a minimal behavioral change aligned with the metric and should move the score substantially toward the target.'
- What this solution (achieved 2.78854) has done: 'We need to reduce MAE (lower is better) from 2.835 toward 0.162, so we should improve the fallback predictor while keeping your current “hierarchical median lookup + pressure snapping + optional blending” core logic intact. The biggest metric-aligned issue left is that the fallback currently mixes inspiratory-derived lookup tables with expiratory rows in test (even though you later set expiratory predictions to 0), which can still pollute merges and increase missingness/poor fills for inspiratory rows; we generate predictions only for inspiratory test rows and then combine with explicit 0 for expiratory. Next, we add one more minimal breath-dynamics key (a lagged `u_out` and `breath_time_idx`) to tighten grouping without changing the modeling approach, improving lookup hit-rate/precision on inspiratory. Finally, we keep the same snapping-to-grid and submission alignment-by-id behavior to preserve evaluation semantics and stability.'
- What this solution (achieved 2.78854) has done: 'I keep your overall “blend external submissions else use train-derived lookup + pressure snapping” pipeline unchanged, but tighten the fallback so it matches the scoring rule more closely. Specifically, I compute the lookup tables only on inspiratory samples (`u_out==0`) and also generate predictions only for inspiratory test rows, preventing expiratory rows from influencing any merges/fills. Then I add one minimal, high-signal key (`R*C` interaction as a categorical-like feature) into the existing hierarchical median tables to increase lookup hit-rate without changing the modeling approach. Finally, I preserve your explicit `u_out==1 -> 0.0` post-processing and `id`-aligned merge to ensure a valid submission.csv.'
- What this solution (achieved 2.78854) has done: 'I keep your existing “blend external submissions else use train-derived hierarchical median lookup + pressure grid snapping” core logic, but make one metric-aligned correction and one minimal robustness improvement to reduce MAE toward the target. First, instead of forcing `u_out==1` predictions to `0.0` (which is arbitrary and can be very wrong, even if those rows aren’t scored), we set expiratory predictions to a neutral value: the inspiratory median pressure from training; this avoids injecting extreme errors on those rows while preserving inspiratory predictions unchanged. Second, we ensure any `id` missing from the lookup (rare) is filled with the same inspiratory median (not 0), preventing occasional large penalties from NaNs/zeros. These are small, safe changes that keep your modeling approach identical but should move the score down from ~2.79 toward the target.'
- What this solution (achieved 2.78854) has done: 'I keep your end-to-end pipeline and hierarchical median-lookup + snapping core logic intact, but make two minimal metric-aligned adjustments that typically reduce MAE substantially on this competition. First, I stop forcing expiratory (`u_out==1`) predictions to an arbitrary constant (median), and instead carry forward the last predicted inspiratory pressure within each breath (a simple “hold” that better matches typical expiratory behavior without changing the model). Second, I add one additional, very lightweight lookup level using a slightly different time bin (`tq2`) and `cumu_q` to improve hit-rate when the strictest table misses, while still using the same median-table approach and the same pressure-grid snapping. These changes should move your score down (lower is better) toward the 0.162 target without changing the overall method or requiring new packages.'
- What this solution (achieved 2.78854) has done: 'I keep your existing fallback “hierarchical median lookup + pressure-grid snapping + within-breath carry-forward fill” logic intact, but make two metric-aligned adjustments that typically reduce MAE without changing the approach. First, I include `u_out` itself (not just its lag) as a grouping key in the lookup tables so inspiratory/expiratory regimes don’t get mixed in coarse backoff levels, improving median-table precision on inspiratory rows. Second, I ensure expiratory predictions are filled by a per-breath hold of the last inspiratory prediction (as you already do via ffill/bfill), but avoid any accidental overwriting/missingness by building the final `sub_out` in a single breath-sorted frame and filling once. These are minimal changes aimed at lowering your current 2.78854 score toward the 0.16235 target (lower is better), while preserving end-to-end execution and submission format.'
- What this solution (achieved 2.78854) has done: 'Your fallback (the path you’re actually using when no external submissions exist) is still far from the target MAE, so we keep the exact same hierarchical median-lookup + snapping + within-breath fill core logic, but make two metric-aligned, minimal adjustments to improve lookup precision. First, we build the median tables from *all* phases (not just inspiratory) while keeping the scoring semantics unchanged (we still predict for all rows, and the metric only evaluates inspiratory), because expiratory dynamics help the “hold/ffill” behavior and reduce mismatches around the phase boundary. Second, we add one extra backoff table keyed on a more stable combination (`R,C,RC,bidx_q,tq,uin_q,u_out`) to increase hit-rate without changing the approach, then keep the same snapping-to-pressure-grid and per-breath forward/backward fill.'
- What this solution (achieved 2.78854) has done: 'We keep your exact fallback approach (hierarchical median lookup → backoff fills → snap to pressure grid → per-breath ffill/bfill) and only make minimal, metric-aligned fixes that should reduce MAE toward the 0.162 target. The biggest remaining issue is that your lookup is diluted by mixing expiratory (`u_out==1`) rows into the median tables even though the metric scores only inspiratory; we build the median tables using inspiratory rows only while still generating predictions for all rows (and keeping your within-breath fill for expiratory). Next, we add one tiny, high-signal key already available in your features (`u_in`-integral bin `cumu_q` at a coarser backoff level) to improve hit-rate/precision without changing the modeling paradigm. Finally, we ensure the external-submission blending path also snaps predictions to the known pressure grid for a small, safe improvement when that path is used.'
- What this solution (achieved 2.78922) has done: 'We need to move your MAE down from 2.78854 toward 0.16235 (lower is better), so we should improve the fallback path you’re likely using (no external submissions found) with the smallest changes that keep your hierarchical median-lookup + snapping + per-breath fill core logic intact. The biggest metric-aligned win is to explicitly predict only inspiratory rows using lookup tables built from inspiratory training rows, then for expiratory rows hold the last inspiratory prediction within each breath (your ffill/bfill already does this, but we should ensure expiratory rows don’t interfere with lookup merges and that phase boundary is handled cleanly). Next, we add one minimal, high-signal grouping key that doesn’t change the approach: quantized delta-u_in (`duin_q`) to better capture valve movement dynamics, improving lookup precision/hit-rate. Finally, we also apply the same within-breath fill to the blended-submissions path (if it ever triggers) to keep phase continuity and avoid unnecessary errors from noisy external preds.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
sub = pd.read_csv(sub_path)

expected_paths = [
    "../input/improvement-base-on-tensor-bidirect-lstm-0-173/submission.csv",
    "../input/finetune-of-tensorflow-bidirectional-lstm/submission.csv",
    "../input/lightautoml-bidirectional-lstm/submission.csv",
    "../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv",
]

available_paths = [p for p in expected_paths if os.path.exists(p)]

if len(available_paths) == 0:
    discovered = sorted(glob.glob("../input/**/submission.csv", recursive=True))
    available_paths = [p for p in discovered if "sample_submission" not in p.lower()]

submission_dfs = []
for p in available_paths:
    try:
        df = pd.read_csv(p, usecols=["id", "pressure"])
        submission_dfs.append(df)
    except Exception:
        pass

if len(submission_dfs) == 0:
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    use_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    train = pd.read_csv(train_path, usecols=use_cols)
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    train_all = train.copy()
    test_all = test.copy()

    pressure_grid = np.sort(train_all["pressure"].unique())

    def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.sort_values(["breath_id", "time_step"], kind="mergesort")

        out["u_in_lag1"] = (
            out.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
        )

        out["u_out_lag1"] = (
            out.groupby("breath_id", sort=False)["u_out"]
            .shift(1)
            .fillna(0)
            .astype(np.int8)
        )
        out["breath_time_idx"] = (
            out.groupby("breath_id", sort=False).cumcount().astype(np.int16)
        )

        out["du_in"] = (out["u_in"] - out["u_in_lag1"]).astype(np.float64)

        dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
        out["cum_u_in"] = (
            (out["u_in"] * dt).groupby(out["breath_id"], sort=False).cumsum()
        )
        return out

    train_f = add_breath_features(train_all)
    test_f = add_breath_features(test_all)

    def make_keys(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["tq"] = (out["time_step"] * 100).round().astype(np.int32)  # ~0.01s bins
        out["uin_q"] = (out["u_in"] * 2).round().astype(np.int32)  # 0.5 bins
        out["uin_lag_q"] = (out["u_in_lag1"] * 2).round().astype(np.int32)  # 0.5 bins
        out["cumu_q"] = (out["cum_u_in"] * 200).round().astype(np.int32)  # coarse bins
        out["tq2"] = (out["time_step"] * 50).round().astype(np.int32)  # ~0.02s bins
        out["bidx_q"] = (out["breath_time_idx"] // 2).astype(np.int16)
        out["RC"] = (out["R"].astype(np.int16) * out["C"].astype(np.int16)).astype(
            np.int16
        )
        out["u_out"] = out["u_out"].astype(np.int8)

        out["duin_q"] = (out["du_in"] * 2).round().astype(np.int32)  # 0.5 bins
        return out

    train_k = make_keys(train_f)
    test_k = make_keys(test_f)

    def median_table(df: pd.DataFrame, grp_cols, pred_name: str):
        return (
            df.groupby(grp_cols, observed=True)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": pred_name})
        )

    train_k_insp = train_k.loc[train_k["u_out"] == 0].copy()

    grp1 = [
        "R",
        "C",
        "RC",
        "u_out",
        "u_out_lag1",
        "bidx_q",
        "tq",
        "uin_q",
        "uin_lag_q",
        "duin_q",
        "cumu_q",
    ]
    tbl1 = median_table(train_k_insp, grp1, "pred1")

    grp2 = ["R", "C", "RC", "u_out", "u_out_lag1", "bidx_q", "tq", "uin_q", "uin_lag_q"]
    tbl2 = median_table(train_k_insp, grp2, "pred2")

    grp3 = ["R", "C", "RC", "u_out", "u_out_lag1", "bidx_q", "tq", "uin_q"]
    tbl3 = median_table(train_k_insp, grp3, "pred3")

    grp4 = ["R", "C", "RC", "u_out", "u_out_lag1", "bidx_q", "tq2", "uin_q"]
    tbl4 = median_table(train_k_insp, grp4, "pred4")

    grp5 = ["R", "C", "RC", "u_out", "tq", "uin_q"]
    tbl5 = median_table(train_k_insp, grp5, "pred5")

    grp6 = ["R", "C", "RC", "u_out", "tq2", "uin_q", "cumu_q"]
    tbl6 = median_table(train_k_insp, grp6, "pred6")

    grp7 = ["R", "C", "RC", "bidx_q", "tq", "uin_q", "u_out"]
    tbl7 = median_table(train_k_insp, grp7, "pred7")

    grp8 = ["R", "C", "RC", "u_out", "bidx_q", "tq2", "uin_q", "cumu_q"]
    tbl8 = median_table(train_k_insp, grp8, "pred8")

    insp_mask = test_k["u_out"].to_numpy() == 0
    test_k_insp = test_k.loc[insp_mask].copy()

    merged = test_k_insp[
        [
            "id",
            "breath_id",
            "time_step",
            "R",
            "C",
            "RC",
            "u_out",
            "u_out_lag1",
            "bidx_q",
            "tq",
            "uin_q",
            "uin_lag_q",
            "duin_q",
            "cumu_q",
            "tq2",
        ]
    ].copy()
    merged = merged.merge(tbl1, on=grp1, how="left")
    merged = merged.merge(tbl2, on=grp2, how="left")
    merged = merged.merge(tbl3, on=grp3, how="left")
    merged = merged.merge(tbl4, on=grp4, how="left")
    merged = merged.merge(tbl5, on=grp5, how="left")
    merged = merged.merge(tbl6, on=grp6, how="left")
    merged = merged.merge(tbl7, on=grp7, how="left")
    merged = merged.merge(tbl8, on=grp8, how="left")

    merged["pred"] = (
        merged["pred1"]
        .fillna(merged["pred2"])
        .fillna(merged["pred3"])
        .fillna(merged["pred4"])
        .fillna(merged["pred5"])
        .fillna(merged["pred6"])
        .fillna(merged["pred7"])
        .fillna(merged["pred8"])
    )

    insp_median = float(train.loc[train["u_out"] == 0, "pressure"].median())
    merged["pred"] = merged["pred"].fillna(insp_median).astype(np.float64)

    idx = np.searchsorted(pressure_grid, merged["pred"].to_numpy(), side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    left_idx = np.clip(idx - 1, 0, len(pressure_grid) - 1)
    right = pressure_grid[idx]
    left = pressure_grid[left_idx]
    x = merged["pred"].to_numpy()
    snapped = np.where(np.abs(x - left) <= np.abs(x - right), left, right)
    merged["pred"] = snapped.astype(np.float64)

    sub_out = sub[["id"]].merge(
        test[["id", "breath_id", "time_step", "u_out"]], on="id", how="left"
    )
    sub_out = sub_out.merge(merged[["id", "pred"]], on="id", how="left")
    sub_out = sub_out.sort_values(["breath_id", "time_step"], kind="mergesort")

    sub_out["pressure"] = sub_out["pred"].astype(np.float64)
    sub_out.drop(columns=["pred"], inplace=True)

    sub_out["pressure"] = sub_out.groupby("breath_id", sort=False)["pressure"].ffill()
    sub_out["pressure"] = sub_out.groupby("breath_id", sort=False)["pressure"].bfill()
    sub_out["pressure"] = sub_out["pressure"].fillna(insp_median).astype(np.float64)

    sub_out = sub_out.sort_values("id", kind="mergesort")
    sub = sub_out[["id", "pressure"]]
    sub.to_csv("submission.csv", index=False)

else:
    base = sub[["id"]].copy()
    for i, df in enumerate(submission_dfs):
        df = df.rename(columns={"pressure": f"pressure_{i}"})
        base = base.merge(df, on="id", how="left")

    pred_cols = [c for c in base.columns if c.startswith("pressure_")]

    if len(pred_cols) > 0:
        for c in pred_cols:
            med = float(base[c].median()) if base[c].notna().any() else 0.0
            base[c] = base[c].fillna(med)
        overall_med = float(pd.concat([base[c] for c in pred_cols], axis=0).median())
    else:
        overall_med = 0.0

    if len(pred_cols) == 4:
        weights = np.array([0.20, 0.30, 0.27, 0.23], dtype=np.float64)
    else:
        weights = np.ones(len(pred_cols), dtype=np.float64) / max(len(pred_cols), 1)

    if len(pred_cols) > 0:
        preds = base[pred_cols].to_numpy(dtype=np.float64)
        blended = (preds * weights.reshape(1, -1)).sum(axis=1)
    else:
        blended = np.full(len(base), overall_med, dtype=np.float64)

    train_path = "../input/ventilator-pressure-prediction/train.csv"
    pressure_grid = np.sort(
        pd.read_csv(train_path, usecols=["pressure"])["pressure"].unique()
    )
    idx = np.searchsorted(pressure_grid, blended, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    left_idx = np.clip(idx - 1, 0, len(pressure_grid) - 1)
    right = pressure_grid[idx]
    left = pressure_grid[left_idx]
    snapped = np.where(np.abs(blended - left) <= np.abs(blended - right), left, right)

    test_path = "../input/ventilator-pressure-prediction/test.csv"
    test_meta = pd.read_csv(test_path, usecols=["id", "breath_id", "time_step"])

    sub2 = pd.DataFrame(
        {"id": base["id"].to_numpy(), "pressure": snapped.astype(np.float64)}
    )
    sub2 = sub2.merge(test_meta, on="id", how="left")
    sub2 = sub2.sort_values(["breath_id", "time_step"], kind="mergesort")
    sub2["pressure"] = sub2.groupby("breath_id", sort=False)["pressure"].ffill()
    sub2["pressure"] = sub2.groupby("breath_id", sort=False)["pressure"].bfill()
    sub2 = sub2.sort_values("id", kind="mergesort")

    sub = sub2[["id", "pressure"]]
    sub.to_csv("submission.csv", index=False)



## === cell 2
print(sub.head())
print(sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
print("pressure stats:", pd.Series(sub["pressure"]).describe())
