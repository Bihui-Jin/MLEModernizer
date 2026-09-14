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

0.1564

# 6. Current score

2.27832

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Diagnosis: Cell 1 crashes because it tries to load four external Kaggle “../input/...” notebook output submissions that do not exist in this environment, so `pd.read_csv()` raises `FileNotFoundError`. The `_resolve_input_path()` helper only remaps paths under `/kaggle/input` and `/kaggle/data`, but those specific dataset folders are absent here. Since cell 2 expects `sub_1..sub_4` DataFrames with a `pressure` column aligned to `sub`, the safest minimal fix is to gracefully fall back to the base `sub` (zero pressures) whenever an external submission file is missing, preserving shapes and column names.

Patch summary: In cell 1 only, wrap the external submission loads in a small helper that attempts to read the CSV and, on `FileNotFoundError`, returns `sub[['id','pressure']]` as a compatible fallback. This prevents the crash while keeping the downstream ensemble computation in cell 2 working without interface changes.

Updated cells: cell 1 only (no other cells modified).

Compatibility notes for cell k+1: `sub`, `sub_1`, `sub_2`, `sub_3`, and `sub_4` always exist as DataFrames with `pressure` columns and the same row count as `sub`, so the weighted sum in cell 2 remains valid and deterministic.

Assumptions: If an external submission file is unavailable, using the provided sample submission (all zeros) is an acceptable fallback to unblock execution without changing downstream code structure.'
- What this solution (achieved 6.10613) has done: 'Your current score is far worse than the target (lower is better), because all four ensemble inputs are missing and you’re effectively submitting all zeros (or near-zeros), which performs very poorly. The smallest legitimate improvement without changing the “ensemble of 4 submissions” core logic is to replace each missing external submission with a simple, deterministic baseline model trained from `train.csv` and applied to `test.csv`. Specifically, we build a per-(R,C,time_idx,u_out) mean-pressure lookup from the training data and use it to fill each missing `sub_k`, falling back to a global mean when unseen—this preserves the weighted ensembling code path while dramatically improving MAE. We also ensure `pressure` is float and row-aligned by `id`, and still write `submission.csv` exactly as required.'
- What this solution (achieved 3.8916) has done: 'Your current score (6.10613, lower is better) is far from the target (0.1564), and the main reason is the fallback baseline is too weak (a coarse mean lookup by `(R,C,time_idx,u_out)` ignores the dominant signal from `u_in`). To move the score much closer to the target while keeping the same “4 submissions + weighted ensemble” core logic, I strengthen only the fallback builder by adding `u_in` into the lookup key (rounded to a small bin), and keep a safe hierarchical fallback (full-key mean → no-`u_in` mean → global mean) so it always produces predictions for every test row. I also ensure the fallback predictions are aligned to the sample submission `id` order to avoid any accidental misalignment. No changes are made to the ensembling cell beyond benefiting from better fallback `sub_1..sub_4` inputs.'
- What this solution (achieved 2.94434) has done: 'Your score is still far from the target (lower is better), and the main issue is that all four “missing external submissions” are being replaced by the same simple lookup baseline, so the ensemble doesn’t add signal. Keeping the exact same ensemble core logic, I strengthen only the fallback baseline by (1) using a more faithful within-breath `time_idx` (0–79 per breath) instead of a rounded absolute time_step, and (2) adding physically-relevant cumulative features (`u_in_cum`, `u_in_lag1`, `u_in_diff1`) binned and used in a hierarchical mean-encoding lookup with safe fallbacks. This remains deterministic, fast, and produces a valid `submission.csv`, while typically moving MAE sharply downward for this competition without changing your ensembling cell.'
- What this solution (achieved 2.36272) has done: 'Your current score (2.94434, lower is better) is still far from the target (0.1564), so we should legitimately improve the fallback predictions while keeping the same “4 submissions → weighted ensemble” core logic unchanged. The smallest high-impact change for this competition is to add the standard “cumulative volume” feature (`u_in` integrated over time, i.e., `area`) and use it in the same hierarchical mean-lookup you already have; pressure is strongly correlated with this integral. To avoid over-fragmentation, we bin `area` coarsely and use it in a new top-level key, with safe fallbacks to your existing keys and then the global mean so a submission is always produced. Everything else (file paths, ensemble weights, output CSV schema) remains identical.'
- What this solution (achieved 4.20197) has done: 'Your fallback baseline is still too weak because it relies on sparse, exact-match mean lookups that miss many test rows and collapse to coarse/global means. To move the MAE closer to the target without changing your ensemble structure, I keep the same hierarchical lookup approach but replace it with “smoothed” hierarchical means: compute group counts and means, then blend each level with its parent using simple empirical-Bayes shrinkage so predictions don’t jump to the global mean when keys are rare. I also make the per-breath `dt` computation safer (use known `time_step` grid per `time_idx`) so `area` is consistent between train/test. Everything else (four submissions, weights, output schema/filename) remains unchanged.'
- What this solution (achieved 4.20197) has done: 'The crash happens in `_build_baseline_submission()` when constructing `pred`: the code concatenates `["id","breath_id","time_step","u_out"]` with `list(dict.fromkeys(keys_area + keys_full))`, but `keys_area/keys_full` also include `"u_out"`, creating duplicate `"u_out"` columns. In pandas 2.x, merging with a non-unique column label raises `ValueError: The column label 'u_out' is not unique.` The minimal fix is to build the feature column list while explicitly excluding `"u_out"` so it appears only once. This preserves the same merge keys and prediction logic, and keeps `sub_1..sub_4` as dataframes with `["id","pressure"]` for cell 2.'
- What this solution (achieved 4.202) has done: 'Your current score (4.20197, lower is better) is far from the target (0.1564), so we should make a legitimate but still minimal improvement inside the existing fallback baseline builder (since your ensemble inputs are missing and all 4 subs come from the fallback). The highest-impact low-risk change for this competition is to add a deterministic “pressure discretization” post-process: map predictions to the nearest pressure value seen in the training set, which aligns well with the dataset’s quantized target and typically reduces MAE without changing the core modeling approach. I also ensure this snapping happens after the inspiratory/expiratory handling and before writing the final `sub_k` frames so the downstream ensemble remains unchanged. No changes are made to the ensemble weights or file paths; the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 4.09777) has done: 'Your score is much worse than the target (lower is better), and the biggest issue is that the fallback baseline is over-fragmented and then “snapped” using `train["pressure"]` levels that include expiratory-phase points (which are not scored and can distort the discretization). I keep your exact ensemble structure and hierarchical lookup/shrinkage logic, but (1) compute the pressure quantization levels from `train_insp` only (u_out==0) so snapping matches the scored distribution, and (2) add one minimal, deterministic physical signal: a binned “lung compliance elastic term” `u_in_cum / C`, which strongly tracks inspiratory pressure and is cheap to add to the existing mean-encoding hierarchy. These are localized changes inside `_build_baseline_submission()` only, preserve the same training approach (group means + shrinkage), and still write a valid `submission.csv`.'
- What this solution (achieved 3.32239) has done: 'Your current score is far worse than the target (lower is better), and the main bottleneck is that the fallback baseline is a coarse mean-encoding that still can’t capture the strong, smooth relationship between pressure and within-breath cumulative flow (`area`) plus lung attributes. To move the score substantially toward the target without changing the ensemble structure, training loop style, or using new external files, I make one localized improvement inside `_build_baseline_submission()`: add a deterministic “template matching” correction using training breaths. Specifically, for each `(R,C)` and binned peak `u_in` pattern, we retrieve the closest training inspiratory pressure curve (by L1 distance on the `u_in` sequence) and blend it slightly into the existing prediction, keeping expiratory handling and final snapping intact. This preserves the overall approach (non-ML lookup/statistics baseline + postprocess) and should materially reduce MAE while still producing a valid `submission.csv`. All file paths, output schema, and the downstream weighted ensemble cell remain unchanged.'
- What this solution (achieved 2.27832) has done: 'Your current score (3.32239, lower is better) is still far from the target (0.1564), so we should improve the fallback predictions (since all 4 ensemble inputs are missing and come from the fallback) while keeping your ensemble structure and “lookup/statistics + postprocess” core logic intact. The smallest high-impact change here is to strengthen the template-matching correction so it’s less sparse and more useful: (1) build templates only from the first inspiratory segment (before first `u_out==1`) to better match the scored phase, (2) add a second, coarser signature fallback per `(R,C)` so more test breaths get a template, and (3) blend adaptively based on match distance rather than a fixed 0.25. Everything else (hierarchical mean+shrinkage, expiratory fill, snapping, output CSV) remains the same.'
- What this solution (achieved 2.27832) has done: 'Your current MAE (2.27832, lower is better) is still far above the target (0.1564), so we need a legitimate improvement while preserving your existing “fallback baseline + weighted ensemble” structure. The smallest high-impact change is to make the hierarchical lookup operate at the breath level (predicting the full inspiratory pressure curve per breath) rather than per-row merges; this keeps the same core logic (group means + shrinkage + template correction + expiratory fill + snapping) but avoids many sparse-key misses that currently push rows toward coarse/global means. We keep the same features/keys and template-matching idea, just apply them via precomputed dictionaries and per-breath vectors for better coverage and consistency. The ensemble cell remains unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 2.27832) has done: 'We keep your ensemble structure and all existing feature/lookup/template logic intact, but fix one key issue that’s inflating error: expiratory-phase (`u_out==1`) pressures should be filled from the *last inspiratory value* only, whereas the current `ffill()+bfill()` can leak future inspiratory values backward and distort the whole curve. We replace that section with a strict per-breath “last inspiratory pressure” computation (cummax of time index where `u_out==0`), then fill `u_out==1` rows from that value only. This is a localized change inside `_build_baseline_submission()` and should materially reduce MAE toward the target without changing your overall approach or file outputs. The script still runs end-to-end and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd



## === cell 1
import os


def _resolve_input_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        candidate = os.path.join("/kaggle/input", p[len("../input/") :])
        if os.path.exists(candidate):
            return candidate
        candidate2 = os.path.join("/kaggle/data", p[len("../input/") :])
        if os.path.exists(candidate2):
            return candidate2
    if p.startswith("/kaggle/") and os.path.exists(p):
        return p
    return p  # let pandas raise a clear error if still missing


def _build_baseline_submission(
    sample_sub_path: str, train_path: str, test_path: str
) -> pd.DataFrame:
    """
    Change rationale (score-improvement toward target, minimal core-logic impact):
    - Preserve the same approach: hierarchical mean-lookup + shrinkage + template correction +
      expiratory fill + snapping.
    - Fix expiratory fill: Kaggle metric ignores expiratory, but bad expiratory handling can
      contaminate inspiratory curve if we accidentally backfill future inspiratory values.
      We now fill u_out==1 strictly from the last available inspiratory value in time order
      (no bfill), which is consistent with standard VPP postprocessing and improves MAE.
    """
    import numpy as np

    sub_local = pd.read_csv(_resolve_input_path(sample_sub_path), usecols=["id"])

    train = pd.read_csv(
        _resolve_input_path(train_path),
        usecols=["breath_id", "time_step", "u_in", "u_out", "R", "C", "pressure"],
    )
    test = pd.read_csv(
        _resolve_input_path(test_path),
        usecols=["id", "breath_id", "time_step", "u_in", "u_out", "R", "C"],
    )

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    train["time_idx"] = train.groupby("breath_id").cumcount().astype("int16")
    test["time_idx"] = test.groupby("breath_id").cumcount().astype("int16")

    dt_by_idx = (
        train.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .groupby(train["time_idx"], sort=False)
        .median()
        .astype("float32")
        .to_dict()
    )

    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["u_in_cum"] = (
            df.groupby("breath_id", sort=False)["u_in"].cumsum().astype("float32")
        )
        df["u_in_lag1"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .astype("float32")
        )
        df["u_in_diff1"] = (df["u_in"].astype("float32") - df["u_in_lag1"]).astype(
            "float32"
        )

        df["dt"] = df["time_idx"].map(dt_by_idx).fillna(0.0).astype("float32")
        df["area"] = (
            (df["u_in"].astype("float32") * df["dt"])
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .astype("float32")
        )

        df["vol_over_C"] = (df["u_in_cum"] / df["C"].astype("float32")).astype(
            "float32"
        )
        return df

    train = _add_features(train)
    test = _add_features(test)

    train_insp = train.loc[train["u_out"] == 0].copy()

    train_insp["u_in_bin"] = (
        (train_insp["u_in"].astype("float32") * 4.0).round().astype("int16")
    )
    test["u_in_bin"] = (test["u_in"].astype("float32") * 4.0).round().astype("int16")

    train_insp["u_in_cum_bin"] = (
        train_insp["u_in_cum"].round().clip(lower=0).astype("int16")
    )
    test["u_in_cum_bin"] = test["u_in_cum"].round().clip(lower=0).astype("int16")

    train_insp["u_in_diff_bin"] = (
        (train_insp["u_in_diff1"] * 2.0).round() + 400
    ).astype("int16")
    test["u_in_diff_bin"] = ((test["u_in_diff1"] * 2.0).round() + 400).astype("int16")

    train_insp["area_bin"] = (
        (train_insp["area"] / 0.2).round().clip(lower=0).astype("int16")
    )
    test["area_bin"] = (test["area"] / 0.2).round().clip(lower=0).astype("int16")

    train_insp["volC_bin"] = (
        (train_insp["vol_over_C"] / 0.05).round().clip(lower=0).astype("int16")
    )
    test["volC_bin"] = (test["vol_over_C"] / 0.05).round().clip(lower=0).astype("int16")

    for col, dt in [("R", "int16"), ("C", "int16"), ("u_out", "int8")]:
        train_insp[col] = train_insp[col].astype(dt)
        test[col] = test[col].astype(dt)

    train_insp["pressure"] = train_insp["pressure"].astype("float32")
    global_mean = float(train_insp["pressure"].mean())

    keys_area = ["R", "C", "time_idx", "u_out", "u_in_bin", "area_bin"]
    keys_full = [
        "R",
        "C",
        "time_idx",
        "u_out",
        "u_in_bin",
        "u_in_cum_bin",
        "u_in_diff_bin",
    ]
    keys_mid = ["R", "C", "time_idx", "u_out", "u_in_bin", "u_in_cum_bin"]
    keys_uin = ["R", "C", "time_idx", "u_out", "u_in_bin"]
    keys_coarse = ["R", "C", "time_idx", "u_out"]
    keys_volc = ["R", "C", "time_idx", "u_out", "u_in_bin", "volC_bin"]

    def _group_stats(df: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
        g = (
            df.groupby(keys, sort=False)["pressure"]
            .agg(["mean", "count"])
            .reset_index()
        )
        g = g.rename(columns={"mean": f"m_{name}", "count": f"c_{name}"})
        g[f"c_{name}"] = g[f"c_{name}"].astype("int32")
        g[f"m_{name}"] = g[f"m_{name}"].astype("float32")
        return g

    g_area = _group_stats(train_insp, keys_area, "area")
    g_full = _group_stats(train_insp, keys_full, "full")
    g_mid = _group_stats(train_insp, keys_mid, "mid")
    g_uin = _group_stats(train_insp, keys_uin, "uin")
    g_coarse = _group_stats(train_insp, keys_coarse, "coarse")
    g_volc = _group_stats(train_insp, keys_volc, "volc")

    def _to_dict(g: pd.DataFrame, keys: list[str], mcol: str, ccol: str):
        k_tuples = list(map(tuple, g[keys].to_numpy()))
        return dict(zip(k_tuples, zip(g[mcol].to_numpy(), g[ccol].to_numpy())))

    d_coarse = _to_dict(g_coarse, keys_coarse, "m_coarse", "c_coarse")
    d_uin = _to_dict(g_uin, keys_uin, "m_uin", "c_uin")
    d_mid = _to_dict(g_mid, keys_mid, "m_mid", "c_mid")
    d_full = _to_dict(g_full, keys_full, "m_full", "c_full")
    d_area = _to_dict(g_area, keys_area, "m_area", "c_area")
    d_volc = _to_dict(g_volc, keys_volc, "m_volc", "c_volc")

    def _shrink_scalar(child_m, child_c, parent_m, alpha: float) -> float:
        if child_m is None or child_c is None or child_c <= 0:
            return float(parent_m)
        w = float(child_c) / (float(child_c) + float(alpha))
        return w * float(child_m) + (1.0 - w) * float(parent_m)

    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    pressure_pred = np.full(len(test), global_mean, dtype=np.float32)

    for bid, g in test.groupby("breath_id", sort=False):
        idxs = g.index.to_numpy()
        Rv = int(g["R"].iloc[0])
        Cv = int(g["C"].iloc[0])

        for j, row in enumerate(g.itertuples(index=False)):
            if int(row.u_out) != 0:
                continue

            key_coarse = (Rv, Cv, int(row.time_idx), int(row.u_out))
            m_coarse = d_coarse.get(key_coarse, (global_mean, 0))[0]

            key_uin = (Rv, Cv, int(row.time_idx), int(row.u_out), int(row.u_in_bin))
            mu, cu = d_uin.get(key_uin, (None, 0))
            m_uin_v = _shrink_scalar(mu, cu, m_coarse, alpha=80.0)

            key_mid = (
                Rv,
                Cv,
                int(row.time_idx),
                int(row.u_out),
                int(row.u_in_bin),
                int(row.u_in_cum_bin),
            )
            mm, cm = d_mid.get(key_mid, (None, 0))
            m_mid_v = _shrink_scalar(mm, cm, m_uin_v, alpha=60.0)

            key_full = (
                Rv,
                Cv,
                int(row.time_idx),
                int(row.u_out),
                int(row.u_in_bin),
                int(row.u_in_cum_bin),
                int(row.u_in_diff_bin),
            )
            mf, cf = d_full.get(key_full, (None, 0))
            m_full_v = _shrink_scalar(mf, cf, m_mid_v, alpha=40.0)

            key_area = (
                Rv,
                Cv,
                int(row.time_idx),
                int(row.u_out),
                int(row.u_in_bin),
                int(row.area_bin),
            )
            ma, ca = d_area.get(key_area, (None, 0))
            m_area_v = _shrink_scalar(ma, ca, m_full_v, alpha=30.0)

            key_volc = (
                Rv,
                Cv,
                int(row.time_idx),
                int(row.u_out),
                int(row.u_in_bin),
                int(row.volC_bin),
            )
            mv, cv = d_volc.get(key_volc, (None, 0))
            m_volc_v = _shrink_scalar(mv, cv, m_area_v, alpha=25.0)

            pressure_pred[idxs[j]] = np.float32(m_volc_v)

    pred = test[["id", "breath_id", "time_step", "u_out", "time_idx"]].copy()
    pred["pressure"] = pressure_pred.astype("float32")

    def _build_templates(train_df: pd.DataFrame, max_templates_per_rc: int = 600):
        first_uout1_idx = (
            train_df["u_out"]
            .astype("int8")
            .eq(1)
            .groupby(train_df["breath_id"], sort=False)
            .idxmax()
        )
        has_uout1 = (
            train_df.groupby("breath_id", sort=False)["u_out"].max().astype("int8")
        )

        mask_inspseg = train_df["u_out"].astype("int8").eq(0).copy()
        first_map = first_uout1_idx.reindex(train_df["breath_id"]).to_numpy()
        has1_map = has_uout1.reindex(train_df["breath_id"]).to_numpy()
        idx_arr = train_df.index.to_numpy()
        mask_inspseg &= ~(has1_map == 1) | (idx_arr < first_map)

        seg = train_df.loc[mask_inspseg].copy()
        seg = seg.sort_values(["breath_id", "time_step"], kind="mergesort")

        grp = seg.groupby("breath_id", sort=False)
        meta = seg.groupby("breath_id", sort=False).agg(
            R=("R", "first"), C=("C", "first")
        )

        sig_idx = [0, 3, 6, 10, 15, 20, 30, 40]
        sig_idx2 = [0, 6, 15, 30, 40]  # coarser fallback signature

        def _sig(u, idxs, denom):
            u = u.to_numpy(dtype=np.float32)
            if len(u) <= max(idxs):
                last = u[-1] if len(u) else 0.0
                uu = np.pad(u, (0, max(idxs) + 1 - len(u)), constant_values=last)
            else:
                uu = u
            s = np.round(uu[idxs] / denom).astype(np.int16)
            return tuple(s.tolist())

        uin_seq = grp["u_in"].apply(lambda s: s.astype("float32").to_numpy())
        p_seq = grp["pressure"].apply(lambda s: s.astype("float32").to_numpy())

        sig1 = grp["u_in"].apply(lambda s: _sig(s, sig_idx, denom=4.0))
        sig2 = grp["u_in"].apply(lambda s: _sig(s, sig_idx2, denom=8.0))

        tmp = pd.DataFrame(
            {
                "breath_id": uin_seq.index.values,
                "R": meta["R"].astype("int16").values,
                "C": meta["C"].astype("int16").values,
                "sig1": sig1.values,
                "sig2": sig2.values,
                "u_in_seq": uin_seq.values,
                "p_seq": p_seq.values,
            }
        )

        templates_sig1 = {}
        templates_sig2 = {}
        templates_rc = {}

        for (R, C, k), g in tmp.groupby(["R", "C", "sig1"], sort=False):
            g = g.head(max_templates_per_rc)
            templates_sig1[(int(R), int(C), k)] = list(
                zip(g["u_in_seq"].tolist(), g["p_seq"].tolist())
            )

        for (R, C, k), g in tmp.groupby(["R", "C", "sig2"], sort=False):
            g = g.head(max_templates_per_rc)
            templates_sig2[(int(R), int(C), k)] = list(
                zip(g["u_in_seq"].tolist(), g["p_seq"].tolist())
            )

        for (R, C), g in tmp.groupby(["R", "C"], sort=False):
            g = g.head(max_templates_per_rc)
            templates_rc[(int(R), int(C))] = list(
                zip(g["u_in_seq"].tolist(), g["p_seq"].tolist())
            )

        return templates_sig1, templates_sig2, templates_rc, sig_idx, sig_idx2

    templates1, templates2, templates_rc, sig_idx, sig_idx2 = _build_templates(
        train, max_templates_per_rc=450
    )

    test_insp = test.loc[test["u_out"] == 0].copy()
    test_insp = test_insp.sort_values(["breath_id", "time_step"], kind="mergesort")
    test_uin_seq = test_insp.groupby("breath_id", sort=False)["u_in"].apply(
        lambda s: s.astype("float32").to_numpy()
    )
    test_meta = test_insp.groupby("breath_id", sort=False).agg(
        R=("R", "first"), C=("C", "first")
    )

    def _sig_arr(u, idxs, denom):
        u = np.asarray(u, dtype=np.float32)
        if len(u) <= max(idxs):
            last = u[-1] if len(u) else 0.0
            uu = np.pad(u, (0, max(idxs) + 1 - len(u)), constant_values=last)
        else:
            uu = u
        s = np.round(uu[idxs] / denom).astype(np.int16)
        return tuple(s.tolist())

    corr_by_breath = {}
    dist_by_breath = {}

    for bid, u_seq in test_uin_seq.items():
        R = int(test_meta.loc[bid, "R"])
        C = int(test_meta.loc[bid, "C"])

        k1 = _sig_arr(u_seq, sig_idx, denom=4.0)
        bank = templates1.get((R, C, k1))

        if not bank:
            k2 = _sig_arr(u_seq, sig_idx2, denom=8.0)
            bank = templates2.get((R, C, k2))

        if not bank:
            bank = templates_rc.get((R, C))

        if not bank:
            continue

        best_p = None
        best_d = None
        for u_t, p_t in bank:
            L = min(len(u_seq), len(u_t), len(p_t))
            if L == 0:
                continue
            Lm = min(L, 40)
            d = float(np.abs(u_seq[:Lm] - u_t[:Lm]).mean())
            if (best_d is None) or (d < best_d):
                best_d = d
                best_p = p_t[:L]
        if best_p is None:
            continue
        corr_by_breath[int(bid)] = best_p.astype(np.float32)
        dist_by_breath[int(bid)] = float(best_d if best_d is not None else 1e9)

    if corr_by_breath:
        pred = pred.sort_values(
            ["breath_id", "time_step"], kind="mergesort"
        ).reset_index(drop=True)
        b = pred["breath_id"].to_numpy()
        t = pred["time_idx"].to_numpy()
        uo = pred["u_out"].to_numpy()
        p = pred["pressure"].to_numpy(dtype=np.float32)

        for i in range(len(p)):
            if int(uo[i]) != 0:
                continue
            bid = int(b[i])
            seq = corr_by_breath.get(bid)
            if seq is None:
                continue
            ti = int(t[i])
            if ti < 0 or ti >= len(seq):
                continue

            d = dist_by_breath.get(bid, 1e9)
            blend = np.float32(0.55 - min(0.45, 0.45 * (d / 8.0)))
            if blend < np.float32(0.10):
                blend = np.float32(0.10)
            p[i] = (1.0 - blend) * p[i] + blend * np.float32(seq[ti])

        pred["pressure"] = p.astype("float32")

    pred = pred.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    is_insp = pred["u_out"].astype("int8").eq(0)
    tidx = pred["time_idx"].to_numpy(dtype=np.int16)

    last_insp_tidx = (
        pd.Series(np.where(is_insp.to_numpy(), tidx, -1), index=pred.index)
        .groupby(pred["breath_id"], sort=False)
        .cummax()
        .astype("int16")
    )

    filled = pred["pressure"].to_numpy(dtype=np.float32).copy()
    insp_lookup = pred.loc[is_insp, ["breath_id", "time_idx", "pressure"]].copy()
    insp_lookup["k"] = (
        insp_lookup["breath_id"].astype("int32").astype(str)
        + "_"
        + insp_lookup["time_idx"].astype("int16").astype(str)
    )
    insp_map = dict(zip(insp_lookup["k"].values, insp_lookup["pressure"].values))

    exp_mask = ~is_insp
    exp_idx = pred.index[exp_mask]
    if len(exp_idx):
        bid_str = pred.loc[exp_idx, "breath_id"].astype("int32").astype(str)
        li = last_insp_tidx.loc[exp_idx].astype("int16").astype(str)
        ks = (bid_str + "_" + li).values
        exp_vals = np.array([insp_map.get(k, np.nan) for k in ks], dtype=np.float32)
        filled[exp_idx.to_numpy()] = exp_vals

    pred["pressure"] = pd.Series(filled).fillna(global_mean).astype("float32")

    pressure_levels = (
        train_insp["pressure"].astype("float32").dropna().sort_values().unique()
    )
    levels = pressure_levels.astype("float32")
    x = pred["pressure"].to_numpy(dtype=np.float32)
    idx = np.searchsorted(levels, x, side="left")
    idx0 = np.clip(idx - 1, 0, len(levels) - 1)
    idx1 = np.clip(idx, 0, len(levels) - 1)
    choose1 = np.abs(levels[idx1] - x) < np.abs(levels[idx0] - x)
    snapped = levels[idx0]
    snapped[choose1] = levels[idx1][choose1]
    pred["pressure"] = snapped.astype("float32")

    pred = pred[["id", "pressure"]]
    sub_local = sub_local.merge(pred, on="id", how="left")
    sub_local["pressure"] = sub_local["pressure"].fillna(global_mean).astype("float32")
    sub_local = sub_local[["id", "pressure"]]
    return sub_local


sub = pd.read_csv(
    _resolve_input_path("../input/ventilator-pressure-prediction/sample_submission.csv")
)


def _read_submission_or_fallback(path: str, fallback_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.read_csv(_resolve_input_path(path))
        if "id" in df.columns and "pressure" in df.columns:
            df = df[["id", "pressure"]].copy()
        else:
            raise ValueError(f"Submission at {path} missing required columns.")
        return df
    except FileNotFoundError:
        return _build_baseline_submission(
            sample_sub_path="../input/ventilator-pressure-prediction/sample_submission.csv",
            train_path="../input/ventilator-pressure-prediction/train.csv",
            test_path="../input/ventilator-pressure-prediction/test.csv",
        )
    except Exception:
        return _build_baseline_submission(
            sample_sub_path="../input/ventilator-pressure-prediction/sample_submission.csv",
            train_path="../input/ventilator-pressure-prediction/train.csv",
            test_path="../input/ventilator-pressure-prediction/test.csv",
        )


sub_1 = _read_submission_or_fallback(
    "../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv", sub
)
sub_2 = _read_submission_or_fallback(
    "../input/finetune-of-tensorflow-bidirectional-lstm/submission.csv", sub
)
sub_3 = _read_submission_or_fallback(
    "../input/a-dummy-approach-to-improve-your-score-postprocess/submission.csv", sub
)
sub_4 = _read_submission_or_fallback(
    "../input/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv", sub
)



## === cell 2
sub["pressure"] = (
    (sub_1["pressure"].astype("float32").values * 0.1)
    + (sub_2["pressure"].astype("float32").values * 0.12)
    + (sub_3["pressure"].astype("float32").values * 0.18)
    + (sub_4["pressure"].astype("float32").values * 0.6)
).astype("float32")

sub.to_csv("submission.csv", index=False)
sub.head(5)
