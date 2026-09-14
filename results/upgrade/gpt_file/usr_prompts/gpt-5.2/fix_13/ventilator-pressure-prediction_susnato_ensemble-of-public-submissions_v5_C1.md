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

0.2220053254330167

# 6. Current score

1.76657

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The current notebook fails because it tries to read other notebooks’ submission files from `../input/...` paths that do not exist in your environment, so `sub_1/sub_2/sub_3` are never created and the blend step crashes. To make this run end-to-end and produce a valid `submission.csv`, I keep the “blend multiple submissions” core logic but add a safe fallback: if those external files are missing, we default to the provided `sample_submission.csv` (all-zero pressure) so a valid CSV is always written. I also add basic alignment checks on `id` to avoid accidental row-order mismatches when blending. This is score-neutral relative to the current (non-running) state, but at least yield a valid submission file you can score.'
- What this solution (achieved 6.02313) has done: 'Your current score is extremely far from the target (17.65 vs 0.222, lower is better) because the notebook is only blending external submissions that aren’t available in your environment, so it falls back to all-zero pressures (which scores very poorly). To move sharply toward the target while preserving your “blend submissions” core logic, I add a minimal in-notebook model-based fallback that trains on `train.csv` and predicts `test.csv` when none of the external blend files exist. This fallback use a fast, simple per-(R,C,time_step) median pressure lookup (and a global median backup) which is legitimate, runs within the time limit, and typically lands much closer to the 0.2–0.3 MAE range than zeros. I also keep your existing `id` alignment checks and still blend any external submissions if they happen to exist.'
- What this solution (achieved 4.00111) has done: 'Your current score is far worse than the target because the fallback model ignores the inspiratory-phase nature of the metric and uses only `(R,C,time_step)` medians, which can’t track breath dynamics. To move the MAE much closer to ~0.22 while keeping the same “blend external submissions else fallback” core logic, I upgrade only the fallback to a stronger but still simple and fast lookup: per-(R,C,u_out,time_step_round) median pressure, plus a second-stage per-(R,C,u_out,u_in_round,time_step_round) correction when available. This preserves evaluation semantics, stays within time limits (uses only groupby/merge), and should substantially reduce error versus the current fallback. I also keep strict `id` alignment checks and still use external submissions if present.'
- What this solution (achieved 17.65486) has done: 'Your current score is far above (worse than) the target, so we should improve the fallback (used when external submissions are missing) while keeping the same “blend if available else fallback” structure. The smallest high-impact fix is to make the fallback respect the metric: only inspiratory phase (u_out==0) matters, so we predict pressures only there and set expiratory (u_out==1) to 0 to avoid injecting noisy guesses. To better capture breath dynamics without changing the overall approach (still pure groupby/merge lookups), we add breath-level cumulative features (cumulative u_in and cumulative area) and use medians keyed on (R,C,time_step_round, cumu_in_bin, cumu_area_bin), with safe fallbacks to your existing tables and then the global median. This stays fast (groupby + merges), preserves evaluation semantics, and should move MAE substantially toward ~0.22.'
- What this solution (achieved 1.70848) has done: 'I fix the runtime `KeyError: 'time_step_round'` by creating `time_step_round`/`u_in_round` on the full `train` dataframe before the groupby tables that use them (they were only created on `train_insp`). I also keep your existing “blend external submissions else fallback predictor” logic unchanged, but make the `id` merge more robust by forcing consistent dtypes/sorting so the submission aligns exactly with `sample_submission.csv`. These changes are correctness/stability fixes and should substantially improve score versus the current broken run (and avoid falling back to invalid/zero outputs). The script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.64242) has done: 'We need to improve MAE (lower is better) from 1.70848 toward 0.222, so we should strengthen only the fallback predictor (used when external submissions are absent) while keeping your “blend if available else fallback” structure unchanged. The current fallback already uses median lookups but bins cumulative features too coarsely and uses fixed rounding that can miss matches; I make those bins data-driven (quantile bins) to increase hit-rate and reduce error without changing the core groupby/merge logic. I also add one minimal additional safe fallback table keyed by (R,C,u_out,time_step_round,u_in_round_prev) to better capture dynamics while still being the same lookup approach. Finally, I keep strict `id` alignment and ensure `submission.csv` is always written.'
- What this solution (achieved 1.64242) has done: 'Your current MAE (1.642) is far worse than the target (0.222; lower is better), so we should improve only the fallback predictor (used when external submissions are missing) while keeping your “blend-if-available else fallback” structure unchanged. The biggest gain with minimal change is to exploit the fact that true pressures take only a small set of discrete values: after your median lookups, snap predictions to the nearest known pressure level learned from train, which typically reduces MAE substantially. To keep this stable and metric-aligned, we snap only inspiratory-phase predictions (u_out==0) and continue forcing expiratory predictions to 0. Finally, we keep your strict `id` alignment/merge validation so the submission remains correct.'
- What this solution (achieved 17.65486) has done: 'We need to move MAE (lower is better) from 1.642 toward 0.222, so we should improve only the fallback predictor while keeping your overall “blend external submissions else fallback” structure unchanged. The current fallback likely underperforms because it (a) uses expiratory rows in training medians even though they’re not scored and have different dynamics, and (b) uses a very slow/buggy snapping implementation that can distort predictions; I compute all median lookup tables from inspiratory-only rows and replace snapping with a correct, vectorized nearest-level mapping. I also add one minimal, high-signal lookup keyed by `(R,C,time_step_round,u_in_round)` within inspiratory rows (no `u_out` needed since it’s always 0 there) to increase match rate, while preserving your existing staged fallbacks. All I/O paths and submission schema remain the same, and `submission.csv` always be written.'
- What this solution (achieved 1.64242) has done: 'I fix the immediate runtime error by creating `u_in_round_prev` inside the inspiratory-only dataframe (`train_insp`/`test_insp`) before it’s used in the `groupby`, instead of only creating it on the full `train`/`test` and then grouping `train_insp` which doesn’t have that column. I also make the `pd.cut(...).astype("int16")` binning robust by filling NaNs (values outside bin edges) so it won’t crash on unseen ranges in test. These changes keep your existing “blend if available else fallback lookup predictor + snapping” core logic intact, but allow the fallback to run end-to-end and should improve score substantially versus the current failing run. The script still always write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 7.30044) has done: 'Your MAE (1.642) is still far above the target (0.222; lower is better), so we should improve only the fallback predictor while keeping your existing “blend external submissions else fallback” structure unchanged. The biggest low-risk gain is to make the lookup keys match the underlying discrete structure of the data better: use the original exact `time_step` (it’s already on a fixed grid) instead of rounding, and rely less on quantile-binned cumulative features that can miss matches. I also switch the main lookup aggregation from `median` to `mean` for the finest-grained table, which usually reduces MAE for this competition without changing the overall lookup-based core logic. Finally, I keep your inspiratory-only training, expiratory forcing to 0, strict `id` alignment, and the same submission writing behavior.'
- What this solution (achieved 1.60678) has done: 'Your current MAE (7.30; lower is better) is far worse than the target (0.222), so we should improve only the fallback predictor while keeping your existing “blend external submissions else fallback” structure unchanged. The biggest issue is that expiratory rows are currently forced to 0; since Kaggle evaluates only inspiratory rows, we can safely set expiratory predictions to any constant, and setting them to a realistic baseline (global inspiratory median snapped to valid pressure levels) avoids pathological behavior if any evaluation quirks/row filtering differ. Next, the main lookup tables still use float `time_step_key`; switching to a stable integer time index (time_step * 100 rounded) makes joins exact and increases hit-rate without changing the core groupby/merge logic. Finally, we compute the cumulative-feature bins using `pd.qcut(..., duplicates="drop")` to avoid -1 bins and reduce unmatched merges, keeping the rest of your staged fallbacks and snapping intact.'
- What this solution (achieved 1.76657) has done: 'We need to reduce MAE (lower is better) from 1.60678 toward the target 0.222, so we keep your existing “blend external submissions else fallback lookup” structure but strengthen only the fallback predictions. The biggest minimal win in this competition is to respect the known discrete pressure levels more aggressively: instead of snapping only at the very end, we also snap each lookup table’s aggregated pressure to the nearest valid pressure level (learned from train inspiratory), which reduces systematic bias and typically lowers MAE without changing your approach. We also slightly refine the time-step key to match the dataset’s fixed grid exactly (time_step * 1000 as int) to improve merge hit-rate while still being the same groupby/merge logic. All paths, schema, and submission writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
sub = pd.read_csv(sub_path)

blend_candidates = [
    ("../input/tensorflow-bidirectional-lstm-0-234/submission.csv", 0.30),
    ("../input/i-am-groot/submission.csv", 0.40),
    ("../input/tensorflow/submission.csv", 0.30),
]

loaded = []
for p, w in blend_candidates:
    if os.path.exists(p):
        df = pd.read_csv(p)
        loaded.append((df, w, p))

sub["id"] = sub["id"].astype("int64")
sub = sub.sort_values("id", kind="mergesort").reset_index(drop=True)

if len(loaded) == 0:
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )

    train["time_step_key"] = (
        (train["time_step"].astype("float64") * 1000.0).round().astype("int16")
    )
    test["time_step_key"] = (
        (test["time_step"].astype("float64") * 1000.0).round().astype("int16")
    )

    train["u_in_round"] = train["u_in"].round(1).astype("float32")
    test["u_in_round"] = test["u_in"].round(1).astype("float32")

    train_insp = train[train["u_out"] == 0].copy()
    test_insp = test[test["u_out"] == 0].copy()

    train_insp = train_insp.sort_values(["breath_id", "time_step"], kind="mergesort")
    test_insp = test_insp.sort_values(["breath_id", "time_step"], kind="mergesort")

    train_insp["u_in_round_prev"] = (
        train_insp.groupby("breath_id", sort=False)["u_in_round"]
        .shift(1)
        .fillna(0.0)
        .astype("float32")
    )
    test_insp["u_in_round_prev"] = (
        test_insp.groupby("breath_id", sort=False)["u_in_round"]
        .shift(1)
        .fillna(0.0)
        .astype("float32")
    )

    train_insp["dt"] = (
        train_insp.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    )
    test_insp["dt"] = (
        test_insp.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    )

    train_insp["cumu_in"] = train_insp.groupby("breath_id", sort=False)["u_in"].cumsum()
    test_insp["cumu_in"] = test_insp.groupby("breath_id", sort=False)["u_in"].cumsum()

    train_insp["cumu_area"] = (
        (train_insp["u_in"] * train_insp["dt"])
        .groupby(train_insp["breath_id"], sort=False)
        .cumsum()
    )
    test_insp["cumu_area"] = (
        (test_insp["u_in"] * test_insp["dt"])
        .groupby(test_insp["breath_id"], sort=False)
        .cumsum()
    )

    def _qcut_bins(train_s: pd.Series, test_s: pd.Series, q: int):
        tr = pd.Series(train_s).astype("float64")
        te = pd.Series(test_s).astype("float64")
        tr_bin, edges = pd.qcut(tr, q=q, labels=False, retbins=True, duplicates="drop")
        te_clip = te.clip(lower=float(edges[0]), upper=float(edges[-1]))
        te_bin = pd.cut(te_clip, bins=edges, labels=False, include_lowest=True)
        return (
            tr_bin.astype("int16"),
            te_bin.fillna(0).astype("int16"),
        )

    train_insp["cumu_in_bin"], test_insp["cumu_in_bin"] = _qcut_bins(
        train_insp["cumu_in"], test_insp["cumu_in"], q=80
    )
    train_insp["cumu_area_bin"], test_insp["cumu_area_bin"] = _qcut_bins(
        train_insp["cumu_area"], test_insp["cumu_area"], q=80
    )

    global_median_insp = float(train_insp["pressure"].median())

    pressure_levels = (
        train_insp["pressure"].dropna().astype("float64").round(5).unique()
    )
    pressure_levels = np.sort(pressure_levels)

    def _snap_to_levels_arr(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(levels, x, side="left")
        idx = np.clip(idx, 0, len(levels) - 1)
        left_idx = np.clip(idx - 1, 0, len(levels) - 1)
        right_idx = idx
        left = levels[left_idx]
        right = levels[right_idx]
        choose_left = np.abs(x - left) <= np.abs(x - right)
        return np.where(choose_left, left, right)

    def _snap_to_levels(pred: pd.Series, levels: np.ndarray) -> pd.Series:
        x = pred.astype("float64").to_numpy()
        snapped = _snap_to_levels_arr(x, levels)
        return pd.Series(snapped, index=pred.index, dtype="float64")

    def _snap_table_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
        df = df.copy()
        arr = df[col].astype("float64").to_numpy()
        df[col] = _snap_to_levels_arr(arr, pressure_levels)
        return df

    agg_fn_fine = "mean"

    med_rc_ts_u = (
        train_insp.groupby(["R", "C", "time_step_key", "u_in_round"], sort=False)[
            "pressure"
        ]
        .agg(agg_fn_fine)
        .reset_index()
        .rename(columns={"pressure": "p_med_rc_ts_u"})
    )
    med_rc_ts_u = _snap_table_col(med_rc_ts_u, "p_med_rc_ts_u")

    med_rcuot = (
        train_insp.groupby(["R", "C", "u_out", "time_step_key"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rcuot"})
    )
    med_rcuot = _snap_table_col(med_rcuot, "p_med_rcuot")

    med_rcuotu = (
        train_insp.groupby(
            ["R", "C", "u_out", "u_in_round", "time_step_key"], sort=False
        )["pressure"]
        .agg(agg_fn_fine)
        .reset_index()
        .rename(columns={"pressure": "p_med_rcuotu"})
    )
    med_rcuotu = _snap_table_col(med_rcuotu, "p_med_rcuotu")

    med_rcuotu_prev = (
        train_insp.groupby(
            ["R", "C", "u_out", "u_in_round_prev", "time_step_key"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rcuotu_prev"})
    )
    med_rcuotu_prev = _snap_table_col(med_rcuotu_prev, "p_med_rcuotu_prev")

    med_rc_ts_cum = (
        train_insp.groupby(
            ["R", "C", "time_step_key", "cumu_in_bin", "cumu_area_bin"], sort=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_med_rc_ts_cum"})
    )
    med_rc_ts_cum = _snap_table_col(med_rc_ts_cum, "p_med_rc_ts_cum")

    test_insp_pred = (
        test_insp.merge(
            med_rc_ts_cum,
            on=["R", "C", "time_step_key", "cumu_in_bin", "cumu_area_bin"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            med_rc_ts_u,
            on=["R", "C", "time_step_key", "u_in_round"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            med_rcuotu,
            on=["R", "C", "u_out", "u_in_round", "time_step_key"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            med_rcuotu_prev,
            on=["R", "C", "u_out", "u_in_round_prev", "time_step_key"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            med_rcuot,
            on=["R", "C", "u_out", "time_step_key"],
            how="left",
            validate="many_to_one",
        )
    )

    test_insp_pred["pressure"] = test_insp_pred["p_med_rc_ts_cum"]
    test_insp_pred["pressure"] = test_insp_pred["pressure"].fillna(
        test_insp_pred["p_med_rc_ts_u"]
    )
    test_insp_pred["pressure"] = test_insp_pred["pressure"].fillna(
        test_insp_pred["p_med_rcuotu"]
    )
    test_insp_pred["pressure"] = test_insp_pred["pressure"].fillna(
        test_insp_pred["p_med_rcuotu_prev"]
    )
    test_insp_pred["pressure"] = test_insp_pred["pressure"].fillna(
        test_insp_pred["p_med_rcuot"]
    )
    test_insp_pred["pressure"] = test_insp_pred["pressure"].fillna(global_median_insp)
    test_insp_pred["pressure"] = test_insp_pred["pressure"].astype("float64")

    test_insp_pred["pressure"] = _snap_to_levels(
        test_insp_pred["pressure"], pressure_levels
    )

    test = test.copy()
    test["id"] = test["id"].astype("int64")
    test = test.sort_values("id", kind="mergesort").reset_index(drop=True)

    test_pred_full = test[["id", "u_out"]].copy()

    exp_baseline = float(
        _snap_to_levels(pd.Series([global_median_insp]), pressure_levels).iloc[0]
    )
    test_pred_full["pressure"] = exp_baseline

    test_pred_full = test_pred_full.merge(
        test_insp_pred[["id", "pressure"]],
        on="id",
        how="left",
        suffixes=("", "_insp"),
        validate="one_to_one",
    )
    insp_mask = test_pred_full["u_out"].values == 0
    test_pred_full.loc[insp_mask, "pressure"] = test_pred_full.loc[
        insp_mask, "pressure_insp"
    ].values
    test_pred_full.drop(columns=["pressure_insp"], inplace=True)

    base = sub[["id"]].copy()
    merged = base.merge(
        test_pred_full[["id", "pressure"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if merged["pressure"].isna().any():
        missing = int(merged["pressure"].isna().sum())
        raise ValueError(
            f"Fallback predictor is missing {missing} ids present in sample_submission.csv"
        )
    sub["pressure"] = merged["pressure"].values

else:
    base = sub[["id"]].copy()
    blended = pd.Series(0.0, index=base.index, dtype="float64")

    for df, w, p in loaded:
        if "id" not in df.columns or "pressure" not in df.columns:
            raise ValueError(
                f"File {p} must have columns ['id','pressure'], got {df.columns.tolist()}"
            )

        df2 = df[["id", "pressure"]].copy()
        df2["id"] = df2["id"].astype("int64")
        df2 = df2.sort_values("id", kind="mergesort")

        merged = base.merge(df2, on="id", how="left", validate="one_to_one")
        if merged["pressure"].isna().any():
            missing = int(merged["pressure"].isna().sum())
            raise ValueError(
                f"File {p} is missing {missing} ids present in sample_submission.csv"
            )

        blended += merged["pressure"].astype("float64").values * float(w)

    sub["pressure"] = blended.values



## === cell 2
sub.to_csv("submission.csv", index=False)
sub.head(5)
