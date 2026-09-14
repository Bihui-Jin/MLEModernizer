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

0.1598031777632238

# 6. Current score

1.7296

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to read ensemble submission files from other Kaggle datasets that are not available in this environment, so `sub_1..sub_4` never load and the blend crashes. To keep the core “ensemble by weighted averaging” logic while making it run end-to-end, I (1) discover which `/kaggle/input/**/submission.csv` files actually exist, (2) load any valid ones, and (3) blend them using your original weights when possible, otherwise fall back safely to an average (or ultimately the sample submission zeros). This produces a valid `submission.csv` with the required `id,pressure` columns and correct row alignment with the sample submission.'
- What this solution (achieved 8.14624) has done: 'Your current score is extremely poor because the code is (legitimately) blending whatever `submission.csv` files it finds under `/kaggle/input`, which in this environment are not strong model predictions and can even be unrelated—so the ensemble outputs are effectively garbage. To move the MAE down toward the target with minimal core-logic change, I keep the “ensemble by weighted averaging” approach but constrain candidates to only those that match this competition’s test size and id set exactly, and I also automatically exclude any candidate that is identical to the all-zero sample submission. If no valid candidates remain, we fall back to a simple, fast baseline prediction computed from `train.csv` (mean pressure per (R,C,time_step) with a global-mean fallback), which is still consistent with “producing a submission” and should drastically reduce the MAE from ~17.65. This should move the score much closer to the 0.1598 target without changing model architecture/training loops (there are none here).'
- What this solution (achieved 3.05825) has done: 'Your MAE is far above the target (lower is better), so we need a real improvement rather than tuning. Keeping your core “ensemble if available, otherwise baseline” logic intact, the smallest meaningful upgrade is to make the fallback baseline match the competition’s scoring (only inspiratory phase) and use a stronger but still simple breath-dynamics feature: cumulative inspired volume (`u_in` integrated over time) plus `u_out`, conditioned on (R,C,time_step). This stays within your current approach (no model/training loops), but typically moves scores dramatically closer to the ~0.16 range than a plain (R,C,time_step) mean. I also make the `time_step` join robust by rounding (float merge keys are fragile), and ensure candidate submissions are aligned to `sub`’s `id` order before stacking (prevents silent misalignment).'
- What this solution (achieved 4.78016) has done: 'Your current MAE (3.058) is far worse than the target (0.1598, lower is better), so we should improve the fallback baseline while keeping the same “ensemble if available, otherwise baseline” core logic. The main weakness is that the baseline uses rounded float merge keys and a coarse 2-decimal binning for cumulative volume/time, which loses information and creates many unseen bins at test time. I keep the same groupby-mean approach but switch to exact per-step indexing within each breath (`step` 0–79) and add two simple, deterministic breath-dynamics features (`u_in_cum` and `u_in_lag1`) with finer rounding to reduce collisions and missing joins. This is still the same modeling approach (lookup table from train inspiratory rows), but it should move the score substantially downward toward the target band without changing any training loops or model architectures.'
- What this solution (achieved 2.28375) has done: 'Your current score is much worse than the target (lower is better), and the remaining issue is that the fallback baseline is still too sparse: it keys on exact rounded `u_in_cum`/`u_in_lag1`, which creates many unseen combinations at test time and forces coarse fallbacks. I keep the same core “lookup-table from train inspiratory rows + hierarchical fallbacks” logic, but add a slightly stronger, still-deterministic feature (`u_in_cum` plus a simple `u_in_diff1`) and make the grouping less brittle by binning these dynamics features at a consistent resolution and adding one more intermediate fallback level. I also ensure the “step” index is robustly 0–79 per breath (the competition’s fixed length) so train/test align perfectly. These minimal changes should reduce missing joins and move MAE down toward the target without changing the overall approach or producing an invalid submission.'
- What this solution (achieved 1.4732) has done: 'Your current MAE is far above the target (lower is better), so we should strengthen the fallback baseline while keeping the same core “lookup-table from train inspiratory rows + hierarchical fallbacks / ensemble if available” approach. The biggest issue is sparsity/mismatch from binning the breath-dynamics features; I keep the exact same features but switch to quantile-based discretization learned on train (then applied to test) so bins are well-populated and reduce unseen-combination fallbacks. I also add one more intermediate fallback level that drops only the most brittle discretized feature first, which should reduce NA rates without changing the overall logic. The ensemble path remains unchanged; only the fallback baseline changes, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.49467) has done: 'Your current MAE (1.4732, lower is better) is still far above the target (0.1598), so we need a modest but meaningful improvement in the fallback baseline while keeping the same “lookup-table from train inspiratory rows + hierarchical fallbacks / ensemble if available” approach. The main weakness is that the current discretized dynamics features are still too sparse/misaligned; instead of changing the approach, we keep the same groupby-mean lookup but use a *more stable* breath-dynamics signal: integrated flow as `u_in_cum` computed with the known fixed sampling (`dt≈0.033`) and add a simple `u_in_ewm` (within-breath exponentially weighted mean) to smooth control input without introducing any training loops. We also add a final, very safe fallback keyed only by `(R,C,step,u_out,u_in_binned)` to reduce missing joins; this preserves evaluation semantics and should move MAE downward toward the target. The ensemble-loading/blending path remains unchanged, and the script still writes a valid `submission.csv` with correct `id,pressure` alignment.'
- What this solution (achieved 1.49007) has done: 'Your current MAE (1.49467, lower is better) is far above the target (0.1598), so we should improve the *fallback baseline* while keeping the exact same overall approach: “blend valid candidate submissions if present, otherwise use a deterministic train-lookup baseline.” The biggest gap is that the baseline predicts pressures for expiratory timesteps too, even though those rows are not scored; setting test predictions to 0 when `u_out==1` is a minimal, metric-aligned change that usually yields a large MAE drop without changing the model class. Additionally, the baseline’s dynamics use a fixed `dt=0.033`; switching to the actual per-row `delta_time` from `time_step` makes the integrated signal consistent across breaths and reduces train/test feature mismatch while preserving the same feature/lookup-table logic. No changes are made to the ensemble-loading/blending path, and the script still writes a valid `submission.csv` with correct `id,pressure` alignment.'
- What this solution (achieved 1.48985) has done: 'Your current score (1.49007, lower is better) is far worse than the target (0.1598), so we should make the smallest metric-aligned improvement to the fallback baseline without changing the overall “ensemble if available else deterministic train-lookup” approach. The key fix is to make the lookup table less sparse by using the known discrete target structure: pressures take on a fixed set of discrete values, so rounding baseline predictions to the nearest training pressure level typically reduces MAE substantially. This keeps the same features/groupby/merging logic and only changes the final post-processing of predictions, plus it continues to force `u_out==1` rows to 0 (not scored). The ensemble path is unchanged; if no valid candidate submissions exist, the improved baseline be used and written to `submission.csv`.'
- What this solution (achieved 1.7296) has done: 'Your current MAE (1.48985, lower is better) is still far above the target (0.1598), so we should improve the fallback baseline while keeping the same “train lookup-table with hierarchical fallbacks (and ensemble if available)” core logic. The smallest high-impact fix for this competition is to use the known discrete pressure levels and predict the most likely level per state instead of a mean: we switch the lookup aggregations from mean to **mode (most frequent pressure)**, then keep your existing hierarchical fallback chain intact. This aligns better with MAE on a discretized target and typically reduces error without changing the overall approach or adding training loops/models. We keep the existing snapping-to-grid (still helpful) and the `u_out==1 -> 0` rule (metric-aligned, not scored).'
- What this solution (achieved 1.70912) has done: 'The timeout is dominated by repeatedly materializing huge pandas objects and doing expensive `groupby().agg(value_counts)` modes plus multiple full merges on a 5.4M-row train. To keep identical prediction semantics while making it fit in 600s, I replace the mode aggregation with an equivalent “most frequent then smallest pressure” computation using `groupby().size()` + `sort_values()` + `drop_duplicates()`, which avoids per-group Python work. I also eliminate unnecessary DataFrame copies, use categorical encodings for keys to speed up grouping/merging, and compute per-breath features (`dt`, cumulative integral, lag, EWM) with vectorized numpy/`groupby` only once. Candidate submission loading is also made cheaper by first reading just headers/required columns and avoiding repeated merges when ids already align.'
- What this solution (achieved 1.7296) has done: 'Your current MAE (1.70912, lower is better) is still far above the target, so we should improve the fallback baseline while keeping the same “ensemble if available, otherwise deterministic train lookup-table with hierarchical fallbacks” approach. The biggest metric-aligned gain with minimal logic change is to explicitly force predictions to 0 for `u_out==1` (expiratory, not scored) and to build the lookup tables using only inspiratory rows (`u_out==0`) so the mode tables aren’t polluted by expiratory dynamics. To reduce sparsity without changing the model class, we also add one very safe extra fallback level that drops `u_out` in the keys (since inspiratory-only tables already imply `u_out=0`), which tends to reduce missing joins. All other core mechanics (feature computation, quantile binning, mode-by-count lookup, snapping to pressure grid, and CSV writing) remain the same.'
- What this solution (achieved 1.7296) has done: 'I fix the immediate `KeyError: 'pred'` by ensuring the first merge creates a `pred` column even when `test_insp` already contains a placeholder `pred` (pandas suffix it otherwise). This keeps the exact same lookup-table and hierarchical fallback logic, but makes the column naming unambiguous and robust. I also make the baseline path explicitly set expiratory (`u_out==1`) test predictions to 0.0 (not scored) instead of using an expiratory default value, which is metric-aligned and should move the MAE substantially toward the target without changing the modeling approach. Finally, I keep the ensemble-loading/blending path unchanged and still write a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input"
COMP_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "ventilator-pressure-prediction"),
    os.path.join(BASE_INPUT, "data", "ventilator-pressure-prediction"),
    os.path.join(BASE_INPUT, "data"),
]
comp_dir = next((p for p in COMP_DIR_CANDIDATES if os.path.exists(p)), None)
if comp_dir is None:
    raise FileNotFoundError(
        "Could not locate ventilator-pressure-prediction data directory under /kaggle/input."
    )

sample_path = os.path.join(comp_dir, "sample_submission.csv")
test_path = os.path.join(comp_dir, "test.csv")
train_path = os.path.join(comp_dir, "train.csv")

sub = pd.read_csv(sample_path, usecols=["id", "pressure"])
sub = sub[["id", "pressure"]].copy()

sub_ids = sub["id"].to_numpy()
test_ids = pd.read_csv(test_path, usecols=["id"])["id"].to_numpy()
if len(test_ids) != len(sub_ids) or not np.array_equal(
    np.sort(test_ids), np.sort(sub_ids)
):
    raise ValueError(
        "test.csv ids do not match sample_submission ids; cannot safely build submission."
    )

required_ids_sorted = np.sort(sub_ids)




## === cell 2
def load_candidate_submissions(base_input="/kaggle/input"):
    paths = glob.glob(os.path.join(base_input, "**", "submission.csv"), recursive=True)
    candidates = []
    for p in paths:
        if os.path.abspath(p) == os.path.abspath("submission.csv"):
            continue
        try:
            df = pd.read_csv(p, usecols=["id", "pressure"])
        except Exception:
            continue
        if not {"id", "pressure"}.issubset(df.columns):
            continue

        pr = pd.to_numeric(df["pressure"], errors="coerce")
        if pr.isna().any():
            continue
        df = df[["id"]].copy()
        df["pressure"] = pr.astype(np.float64, copy=False)
        candidates.append((p, df))
    return candidates


def is_valid_comp_submission(df, required_ids_sorted):
    if len(df) != len(required_ids_sorted):
        return False
    df_ids = df["id"].to_numpy()
    if df_ids.dtype != required_ids_sorted.dtype:
        df_ids = df_ids.astype(required_ids_sorted.dtype, copy=False)
    if not np.array_equal(np.sort(df_ids), required_ids_sorted):
        return False
    return True


candidates = load_candidate_submissions(BASE_INPUT)

valid_loaded = []
sub_id_index = None  # cached mapping for alignment when needed
for p, df in sorted(candidates, key=lambda x: x[0]):
    if not is_valid_comp_submission(df, required_ids_sorted):
        continue

    df_ids = df["id"].to_numpy()
    if np.array_equal(df_ids, sub_ids):
        arr = df["pressure"].to_numpy(dtype=np.float64, copy=False)
    else:
        if sub_id_index is None:
            sub_id_index = pd.Index(sub_ids)
        s = pd.Series(
            df["pressure"].to_numpy(dtype=np.float64, copy=False), index=df_ids
        )
        aligned = s.reindex(sub_id_index)
        if aligned.isna().any():
            continue
        arr = aligned.to_numpy(np.float64, copy=False)

    if np.allclose(arr, 0.0):
        continue
    valid_loaded.append((p, arr))

print(f"Found {len(candidates)} raw candidate submission.csv files under {BASE_INPUT}.")
print(f"Using {len(valid_loaded)} valid, aligned, non-trivial candidates for blending.")
for p, _ in valid_loaded[:10]:
    print(" -", p)




## === cell 3
def baseline_from_train_groupby(train_csv, test_csv):
    use_train_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    use_test_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(train_csv, usecols=use_train_cols)
    test = pd.read_csv(test_csv, usecols=use_test_cols)

    train = train.sort_values(
        ["breath_id", "time_step"], kind="mergesort", ignore_index=True
    )
    test = test.sort_values(
        ["breath_id", "time_step"], kind="mergesort", ignore_index=True
    )

    pressure_grid = np.sort(train["pressure"].unique().astype(np.float64, copy=False))

    def add_features(df):
        df["R"] = df["R"].astype(np.int16, copy=False)
        df["C"] = df["C"].astype(np.int16, copy=False)
        df["u_out"] = df["u_out"].astype(np.int8, copy=False)

        df["step"] = (
            df.groupby("breath_id", sort=False).cumcount().astype(np.int16, copy=False)
        )

        dt = (
            df.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(np.float64, copy=False)
        )
        u_in = df["u_in"].to_numpy(np.float64, copy=False)

        u_in_cum = (
            pd.Series(u_in * dt)
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .to_numpy(np.float64, copy=False)
        )

        u_in_lag1 = (
            df.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .to_numpy(np.float64, copy=False)
        )
        u_in_diff1 = (u_in - u_in_lag1).astype(np.float64, copy=False)

        u_in_ewm = (
            df.groupby("breath_id", sort=False)["u_in"]
            .ewm(alpha=0.3, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
            .to_numpy(np.float64, copy=False)
        )

        df["u_in_cum"] = u_in_cum
        df["u_in_lag1"] = u_in_lag1
        df["u_in_diff1"] = u_in_diff1
        df["u_in_ewm"] = u_in_ewm
        return df

    train = add_features(train)
    test = add_features(test)

    train_insp = train[train["u_out"] == 0].copy()

    global_mean_insp = (
        float(train_insp["pressure"].mean())
        if len(train_insp)
        else float(train["pressure"].mean())
    )

    def make_qbins(series, q):
        qs = np.linspace(0, 1, q + 1)
        edges = np.quantile(series.to_numpy(np.float64, copy=False), qs)
        edges = np.unique(edges)
        if len(edges) < 3:
            mn = float(np.min(series))
            mx = float(np.max(series))
            if mn == mx:
                edges = np.array([mn - 1.0, mn, mn + 1.0], dtype=np.float64)
            else:
                edges = np.array([mn, (mn + mx) / 2.0, mx], dtype=np.float64)
        return edges

    edges_cum = make_qbins(train_insp["u_in_cum"], q=90)
    edges_lag = make_qbins(train_insp["u_in_lag1"], q=60)
    edges_diff = make_qbins(train_insp["u_in_diff1"], q=60)
    edges_ewm = make_qbins(train_insp["u_in_ewm"], q=70)
    edges_uin = make_qbins(train_insp["u_in"], q=80)

    def apply_bins(df):
        df["u_in_cum_b"] = pd.cut(
            df["u_in_cum"], bins=edges_cum, labels=False, include_lowest=True
        )
        df["u_in_lag1_b"] = pd.cut(
            df["u_in_lag1"], bins=edges_lag, labels=False, include_lowest=True
        )
        df["u_in_diff1_b"] = pd.cut(
            df["u_in_diff1"], bins=edges_diff, labels=False, include_lowest=True
        )
        df["u_in_ewm_b"] = pd.cut(
            df["u_in_ewm"], bins=edges_ewm, labels=False, include_lowest=True
        )
        df["u_in_b"] = pd.cut(
            df["u_in"], bins=edges_uin, labels=False, include_lowest=True
        )

        for c in ["u_in_cum_b", "u_in_lag1_b", "u_in_diff1_b", "u_in_ewm_b", "u_in_b"]:
            df[c] = pd.Series(df[c]).fillna(-1).astype(np.int16)
        return df

    train_insp = apply_bins(train_insp)
    test = apply_bins(test)

    for c in [
        "R",
        "C",
        "u_out",
        "step",
        "u_in_cum_b",
        "u_in_ewm_b",
        "u_in_lag1_b",
        "u_in_diff1_b",
        "u_in_b",
    ]:
        train_insp[c] = train_insp[c].astype("category")
        test[c] = test[c].astype("category")

    def fast_mode_table(train_df, keys, pred_name):
        tmp = train_df[keys + ["pressure"]].copy()
        cnt = (
            tmp.groupby(keys + ["pressure"], sort=False, observed=True)
            .size()
            .reset_index(name="cnt")
        )
        cnt = cnt.sort_values(
            keys + ["cnt", "pressure"],
            ascending=[True] * len(keys) + [False, True],
            kind="mergesort",
        )
        out = cnt.drop_duplicates(subset=keys, keep="first")[
            keys + ["pressure"]
        ].rename(columns={"pressure": pred_name})
        return out

    insp_mask = (test["u_out"].astype(np.int8) == 0).to_numpy()
    test["pred"] = np.nan

    test_insp = test.loc[insp_mask].copy()

    if "pred" in test_insp.columns:
        test_insp = test_insp.drop(columns=["pred"])

    keys1 = [
        "R",
        "C",
        "step",
        "u_out",
        "u_in_cum_b",
        "u_in_ewm_b",
        "u_in_lag1_b",
        "u_in_diff1_b",
    ]
    grp = fast_mode_table(train_insp, keys1, "pred")
    test_insp = test_insp.merge(grp, on=keys1, how="left", sort=False)

    missing = test_insp["pred"].isna()
    if missing.any():
        keys_mid0 = [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_cum_b",
            "u_in_ewm_b",
            "u_in_diff1_b",
        ]
        grp_mid0 = fast_mode_table(train_insp, keys_mid0, "pred_mid0")
        test_insp = test_insp.merge(grp_mid0, on=keys_mid0, how="left", sort=False)
        test_insp.loc[missing, "pred"] = test_insp.loc[missing, "pred_mid0"]
        test_insp = test_insp.drop(columns=["pred_mid0"])
    missing = test_insp["pred"].isna()

    if missing.any():
        keys_mid = [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_cum_b",
            "u_in_ewm_b",
            "u_in_lag1_b",
        ]
        grp_mid = fast_mode_table(train_insp, keys_mid, "pred_mid")
        test_insp = test_insp.merge(grp_mid, on=keys_mid, how="left", sort=False)
        test_insp.loc[missing, "pred"] = test_insp.loc[missing, "pred_mid"]
        test_insp = test_insp.drop(columns=["pred_mid"])
    missing = test_insp["pred"].isna()

    if missing.any():
        keys2 = ["R", "C", "step", "u_out", "u_in_cum_b", "u_in_ewm_b"]
        grp2 = fast_mode_table(train_insp, keys2, "pred2")
        test_insp = test_insp.merge(grp2, on=keys2, how="left", sort=False)
        test_insp.loc[missing, "pred"] = test_insp.loc[missing, "pred2"]
        test_insp = test_insp.drop(columns=["pred2"])
    missing = test_insp["pred"].isna()

    if missing.any():
        keys_uin = ["R", "C", "step", "u_out", "u_in_b"]
        grp_uin = fast_mode_table(train_insp, keys_uin, "pred_uin")
        test_insp = test_insp.merge(grp_uin, on=keys_uin, how="left", sort=False)
        test_insp.loc[missing, "pred"] = test_insp.loc[missing, "pred_uin"]
        test_insp = test_insp.drop(columns=["pred_uin"])
    missing = test_insp["pred"].isna()

    if missing.any():
        keys3 = ["R", "C", "step", "u_out"]
        grp3 = fast_mode_table(train_insp, keys3, "pred3")
        test_insp = test_insp.merge(grp3, on=keys3, how="left", sort=False)
        test_insp.loc[missing, "pred"] = test_insp.loc[missing, "pred3"]
        test_insp = test_insp.drop(columns=["pred3"])
    missing = test_insp["pred"].isna()

    if missing.any():
        keys4 = ["R", "C", "step"]
        grp4 = fast_mode_table(train_insp, keys4, "pred4")
        test_insp = test_insp.merge(grp4, on=keys4, how="left", sort=False)
        test_insp.loc[missing, "pred"] = test_insp.loc[missing, "pred4"]
        test_insp = test_insp.drop(columns=["pred4"])

    test_insp["pred"] = (
        test_insp["pred"].fillna(global_mean_insp).astype(np.float64, copy=False)
    )

    if pressure_grid.size > 1:
        x = test_insp["pred"].to_numpy(np.float64, copy=False)
        idx = np.searchsorted(pressure_grid, x, side="left")
        idx = np.clip(idx, 0, pressure_grid.size - 1)
        left = pressure_grid[np.clip(idx - 1, 0, pressure_grid.size - 1)]
        right = pressure_grid[idx]
        choose_right = (idx == 0) | (
            (idx > 0) & (np.abs(x - right) <= np.abs(x - left))
        )
        test_insp["pred"] = np.where(choose_right, right, left).astype(
            np.float64, copy=False
        )

    test.loc[insp_mask, "pred"] = test_insp["pred"].to_numpy(np.float64, copy=False)

    test.loc[~insp_mask, "pred"] = 0.0

    out = (
        sub[["id"]]
        .merge(test[["id", "pred"]], on="id", how="left", sort=False)["pred"]
        .to_numpy(np.float64, copy=False)
    )

    if np.isnan(out).any():
        out = np.nan_to_num(out, nan=global_mean_insp)

    return out


if len(valid_loaded) >= 4:
    weights = np.array([0.32, 0.29, 0.21, 0.18], dtype=np.float64)
    preds_stack = np.vstack([valid_loaded[i][1] for i in range(4)])
    blended = (weights[:, None] * preds_stack).sum(axis=0)
elif len(valid_loaded) > 0:
    weights = np.ones(len(valid_loaded), dtype=np.float64) / len(valid_loaded)
    preds_stack = np.vstack([arr for _, arr in valid_loaded])
    blended = (weights[:, None] * preds_stack).sum(axis=0)
else:
    blended = baseline_from_train_groupby(train_path, test_path)

sub["pressure"] = blended.astype(np.float64, copy=False)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("pressure summary:", pd.Series(sub["pressure"]).describe())
