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

0.1551276069962956

# 6. Current score

2.56246

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to ensemble four external submission files that are not present in this Kaggle environment, so `sub_1`..`sub_4` never load and the rest crashes. To keep the core “blend submissions into sample_submission” logic while making it run end-to-end, I add a small fallback that checks which candidate submission files exist and blends only those; if none exist, it create a valid baseline submission (all zeros) so you can at least submit. I also fix the cell numbering (start at 1) and ensure the output is written as `submission.csv` with the required `id,pressure` columns. This is primarily a correctness/stability fix; without the missing external inputs, we can’t reproduce the intended high-scoring ensemble, but the code run and produce a valid `.csv`.'
- What this solution (achieved 7.2068) has done: 'I fix the KeyError by preventing `sub.merge(...)` from creating `pressure_x/pressure_y` columns and instead directly assigning the computed predictions into the existing `sub["pressure"]` aligned by `id`. I also make the `id` alignment robust by reindexing predictions to the exact `sub["id"]` order, ensuring the submission has the correct row order and no missing values. These are minimal correctness fixes that keep your fallback “groupby mean” core logic unchanged while guaranteeing a valid `submission.csv` is written end-to-end. Finally, I update the cell numbering to start at 1 to match the required format.'
- What this solution (achieved 5.9754) has done: 'Your current fallback model is limited because it groups by raw `time_step` floats, which don’t match exactly between train and test and cause heavy fallback to coarse means (hurting MAE). To move the score closer to the 0.155 target without changing the core “groupby mean then merge” logic, I make the grouping robust by rounding `time_step` to a fixed precision (matching the dataset’s 0.03-ish grid) and using that rounded value consistently in both train and test keys. I also add a slightly more informative intermediate fallback (grouping on `R,C,u_out,time_step_rounded`) → then `R,C,u_out` → then global mean, keeping semantics identical but reducing missing merges. The ensemble path remains unchanged if external submissions exist; these changes only improve the built-in fallback that’s currently producing ~7.2.'
- What this solution (achieved 3.88866) has done: 'Your current fallback is still far from the target because it predicts pressure from very coarse averages that ignore the core dynamics and the fact that only inspiratory timesteps (u_out=0) matter for scoring. To move the MAE substantially closer to the target without changing the overall “simple non-ML fallback based on training aggregates then merge by keys” approach, I (1) build a per-breath cumulative integral feature of `u_in` (a standard proxy for delivered volume) and (2) use a slightly richer grouped mean key that includes this cumulative feature, while still keeping the same groupby→merge→fallback ladder. This keeps the core logic (groupby means + deterministic merging) intact but makes the lookup much more physically aligned and reduces error a lot. The ensemble path remains unchanged and still be used if those external submissions exist.'
- What this solution (achieved 3.89235) has done: 'Your current fallback aggregates are still too coarse because they ignore the strongest available proxy for pressure dynamics: the instantaneous flow `u_in` itself (and its lag). Without changing the core “groupby means → merge by keys → fallback ladder” approach, I minimally enrich the grouping keys to include a lightly-rounded `u_in` and a per-breath lagged `u_in` so the lookup better matches the true pressure curve. I also keep your existing integral feature and ladder, but add these new keys at the top so predictions improve while preserving semantics and runtime. This should move MAE substantially down from 3.89 toward your 0.155 target without introducing any ML training or changing submission formatting.'
- What this solution (achieved 2.55994) has done: 'Your current fallback is still missing two low-risk, high-impact “physics proxy” keys that can be added without changing the core groupby→merge→fallback ladder: (1) a per-breath cumulative sum of raw `u_in` (not time-weighted) and (2) a simple “phase” index within the breath (`time_step` rank). Adding these as additional top-priority grouping keys make train/test lookups much more specific and should reduce MAE substantially from ~3.89 toward your 0.155 target, while keeping the same deterministic aggregation logic. I also keep your existing keys and fallbacks intact, just inserting the new aggregates above them. The ensemble path remains unchanged if any external submission files exist.'
- What this solution (achieved 2.55894) has done: 'I keep your current “groupby means → merge → fallback ladder” approach unchanged, but make the top-level lookup less sparse by (1) using the known fixed 80-step structure directly (instead of deriving `step` from sorting) and (2) adding one more very-low-risk dynamic proxy (`u_in_diff`) with light rounding as an additional *highest-priority* key. This increases exact train/test key matches without changing the overall semantics, and should reduce MAE from 2.56 toward your 0.155 target. I also make the `step` computation robust to any ordering by using `id` within each `breath_id` and keep the final `id` alignment logic identical so the submission remains valid. All paths and the ensemble branch remain untouched.'
- What this solution (achieved 2.55896) has done: 'Your current score (2.55894 MAE) is far worse than the target (0.15513), so we should safely improve accuracy without changing the core “groupby means → merge → fallback ladder” logic. The highest-impact minimal change for this competition is to apply the known discrete-pressure postprocessing: in the training data, pressure takes on a fixed set of ~950 discrete values, so snapping predictions to the nearest allowed value usually reduces MAE a lot while preserving your existing prediction approach. I compute the sorted unique pressure levels from `train.csv` and, after your current `pred` is formed (either from ensembling or the fallback ladder), map each prediction to the nearest allowed pressure level using a fast `np.searchsorted`-based nearest-neighbor. This is deterministic, fast, and keeps your pipeline end-to-end producing a valid `submission.csv`.'
- What this solution (achieved 2.55896) has done: 'Your current MAE (2.55896) is far above the target (0.15513), so we should improve accuracy with the smallest changes that keep your existing “groupby means → merge → fallback ladder (+snap)” core logic intact. The main issue is that your lookup keys are still too sparse for capturing breath dynamics; a minimal, high-signal fix is to add a single additional top-level aggregate keyed by (`R`,`C`,`u_out`,`step`,`u_in_int_r`) and use it early in the ladder, because pressure is strongly determined by delivered-volume proxy at each step for a given lung. This reuses features you already compute (`step`, `u_in_int_r`) and only adds one more groupby+merge and one `fillna` line. Everything else (including snapping to discrete pressures and submission formatting) stays the same.'
- What this solution (achieved 2.56246) has done: 'Your current MAE (2.55896) is far above the target (0.15513), so we should improve accuracy with a minimal change that keeps your existing “groupby means → merge → fallback ladder (+snap)” logic intact. The biggest remaining gap is that the lookup still often misses because the keys don’t capture the strongest time-dynamics proxy: within-breath lagged pressure itself. Since train has pressure, we can add a deterministic, leak-free feature `p_lag1` (previous timestep pressure) and use its rounded value as an additional high-priority grouping key; at inference we can generate `p_lag1` sequentially from our own prior predictions. This preserves your aggregation approach (still only grouped means + merges/fallbacks + snapping), but greatly increases specificity and typically drops MAE substantially. I also keep the existing ensembling path unchanged; the new sequential fallback only runs when external submissions are missing.'
- What this solution (achieved 2.56246) has done: 'We keep your existing groupby→merge fallback ladder and the discrete-pressure snapping intact, but fix the one thing that’s likely hurting you now: the sequential `p_lag1` lookup is being done with `base.at[i, ...]` on a DataFrame whose index is not guaranteed to be `0..n-1` after merges, so you can end up reading the wrong rows and producing noisy predictions (worsening MAE). The minimal safe change is to reset the index after all merges and then use `iloc` (positional) inside the sequential loop so each timestep uses the correct features and breath boundaries. This doesn’t change your model logic, only makes it correctly apply what you already intended, and should move the MAE down toward the 0.155 target. All paths, fallbacks, and submission writing remain unchanged.'
- What this solution (achieved 2.56246) has done: 'Your current MAE (2.56246, lower is better) is still far from the target (0.15513), so we should improve the fallback predictions while keeping the same “groupby means → merge → sequential p_lag1 lookup → snapping → write submission.csv” core logic. The smallest high-impact fix is to treat `u_out=1` (expiratory phase, not scored) differently by forcing a stable/low-variance prediction there: we keep the sequential inspiratory prediction unchanged, but for expiratory rows we output the nearest allowed pressure to the per-(R,C,step) mean pressure from training (a simple deterministic aggregate). This reduces overall submission noise and indirectly improves inspiratory continuity because `last_pred` should not be updated on `u_out=1` steps (since they’re not scored and often break dynamics); we keep `last_pred` unchanged across expiratory timesteps. These changes are localized to the fallback branch only and preserve evaluation semantics, snapping, I/O paths, and runtime constraints.'
- What this solution (achieved 2.55895) has done: 'Your current MAE is far above the target, so we should improve accuracy while keeping your exact “groupby means → sequential p_lag1 lookup → snapping → write submission.csv” approach. The largest remaining weakness is that `p_lag1_r` is quantized too finely (0.01), making the `mean_p_lag` table extremely sparse so the high-priority lookup rarely hits; we coarsen that rounding slightly (to 0.05) in both train and inference so matches increase without changing the model class. To keep this minimal and stable, I only adjust how `p_lag1_r` is constructed/used and leave all other keys, fallbacks, and snapping unchanged. This should reduce MAE materially and move it closer to the 0.155 target while preserving your core logic and runtime.'
- What this solution (achieved 2.56246) has done: 'Your current score (2.55895 MAE; lower is better) is far above the target (0.15513), so we should safely improve accuracy while keeping your exact “groupby means → sequential p_lag1 lookup → snapping → submission.csv” core logic intact. The smallest high-impact issue I see is that your `id` ranges look wrong (you reported `id` up to 2000), which typically means you are reading a truncated/corrupted file path; that would completely break alignment and inflate MAE, so I add a strict data-path resolver + sanity checks to ensure we load the full 5.4M/603.6k row CSVs. Then, without changing the modeling approach, I make one minimal stability fix in the sequential loop: compute `p_lag1_r` with the same rounding behavior as in training (including `.round(2)`) to avoid tiny float-key mismatches that cause many missed hits in `mean_p_lag_index`. These changes are directly aimed at increasing key matches and fixing potential dataset misload, which should move the MAE substantially toward your target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd




## === cell 1
def resolve_competition_path(filename: str) -> str:
    candidates = [
        f"../input/ventilator-pressure-prediction/{filename}",
        f"/kaggle/input/ventilator-pressure-prediction/{filename}",
        f"../input/{filename}",
        f"/kaggle/input/{filename}",
        f"../data/{filename}",
        f"/kaggle/data/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename}. Tried: {candidates}")


sample_path = resolve_competition_path("sample_submission.csv")
sub = pd.read_csv(sample_path)

candidate_paths = [
    (
        "sub_1",
        "../input/rescaling-layer-for-discrete-output-in-tensorflow/submission.csv",
        0.1,
    ),
    (
        "sub_2",
        "../input/ventilator-pressure-prediction-lstm-gpu-infer/submission_median_round.csv",
        0.5,
    ),
    (
        "sub_3",
        "../input/a-dummy-approach-to-improve-your-score-postprocess/submission.csv",
        0.1,
    ),
    (
        "sub_4",
        "../input/ventilator-pressure-prediction-lstm-gpu-infer/submission_median.csv",
        0.3,
    ),
]

loaded = []
missing = []
for name, path, weight in candidate_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        loaded.append((name, df, weight, path))
    else:
        missing.append(path)

print(f"Loaded {len(loaded)} candidate submission(s). Missing {len(missing)}.")
if missing:
    print("Missing paths (expected in original notebook but not available here):")
    for p in missing:
        print(" -", p)




## === cell 2
def snap_to_allowed_pressures(
    pred_values: np.ndarray, allowed: np.ndarray
) -> np.ndarray:
    pred_values = np.asarray(pred_values, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=np.float64)
    idx = np.searchsorted(allowed, pred_values, side="left")
    idx0 = np.clip(idx - 1, 0, len(allowed) - 1)
    idx1 = np.clip(idx, 0, len(allowed) - 1)
    a0 = allowed[idx0]
    a1 = allowed[idx1]
    choose1 = np.abs(pred_values - a1) < np.abs(pred_values - a0)
    return np.where(choose1, a1, a0)


if loaded:
    base = sub[["id"]].copy()
    pred = pd.Series(0.0, index=base.index)

    total_w = 0.0
    for name, df, w, path in loaded:
        if "id" not in df.columns or "pressure" not in df.columns:
            raise ValueError(
                f"{name} at {path} must contain columns ['id','pressure'], got {df.columns.tolist()}"
            )

        merged = base.merge(
            df[["id", "pressure"]], on="id", how="left", validate="one_to_one"
        )
        if merged["pressure"].isna().any():
            raise ValueError(
                f"{name} at {path} is missing predictions for some ids after merge."
            )

        pred = pred + merged["pressure"].astype("float64") * w
        total_w += w

    if total_w > 0:
        pred = pred / total_w

    train_path = resolve_competition_path("train.csv")
    allowed_pressures = pd.read_csv(train_path, usecols=["pressure"])[
        "pressure"
    ].unique()
    allowed_pressures = np.sort(allowed_pressures.astype(np.float64))
    sub["pressure"] = snap_to_allowed_pressures(pred.values, allowed_pressures).astype(
        "float64"
    )

else:
    train_path = resolve_competition_path("train.csv")
    test_path = resolve_competition_path("test.csv")

    train_nrows = sum(1 for _ in open(train_path, "rb")) - 1
    test_nrows = sum(1 for _ in open(test_path, "rb")) - 1
    print(f"Resolved train.csv: {train_path} (rows={train_nrows})")
    print(f"Resolved test.csv:  {test_path} (rows={test_nrows})")
    if train_nrows < 5_000_000 or test_nrows < 600_000:
        raise RuntimeError(
            "Detected unexpectedly small train/test file. This would destroy score due to misalignment."
        )

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "u_out", "time_step", "u_in", "pressure", "id"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "breath_id", "R", "C", "u_out", "time_step", "u_in"],
    )

    global_mean = float(train["pressure"].mean())

    ROUND_DECIMALS = 3
    train["time_step_r"] = train["time_step"].round(ROUND_DECIMALS)
    test["time_step_r"] = test["time_step"].round(ROUND_DECIMALS)

    train = train.sort_values(["breath_id", "id"], kind="mergesort")
    test = test.sort_values(["breath_id", "id"], kind="mergesort")

    train["dt"] = train.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    test["dt"] = test.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)

    train["u_in_int"] = (
        (train["u_in"] * train["dt"]).groupby(train["breath_id"], sort=False).cumsum()
    )
    test["u_in_int"] = (
        (test["u_in"] * test["dt"]).groupby(test["breath_id"], sort=False).cumsum()
    )

    train["u_in_int_r"] = train["u_in_int"].round(2)
    test["u_in_int_r"] = test["u_in_int"].round(2)

    train["u_in_r"] = train["u_in"].round(1)
    test["u_in_r"] = test["u_in"].round(1)

    train["u_in_lag1_r"] = (
        train.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0).round(1)
    )
    test["u_in_lag1_r"] = (
        test.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0).round(1)
    )

    train["u_in_diff_r"] = (
        train.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).round(1)
    )
    test["u_in_diff_r"] = (
        test.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).round(1)
    )

    train["step"] = train.groupby("breath_id", sort=False).cumcount().astype("int16")
    test["step"] = test.groupby("breath_id", sort=False).cumcount().astype("int16")

    train["u_in_cum"] = train.groupby("breath_id", sort=False)["u_in"].cumsum()
    test["u_in_cum"] = test.groupby("breath_id", sort=False)["u_in"].cumsum()
    train["u_in_cum_r"] = train["u_in_cum"].round(1)
    test["u_in_cum_r"] = test["u_in_cum"].round(1)

    train["p_lag1"] = (
        train.groupby("breath_id", sort=False)["pressure"].shift(1).fillna(global_mean)
    )

    P_LAG_STEP = 0.05
    train["p_lag1_r"] = (
        np.round(train["p_lag1"].astype(np.float64) / P_LAG_STEP) * P_LAG_STEP
    ).round(2)

    key_cols_full = ["R", "C", "u_out", "time_step_r", "u_in_int_r"]
    key_cols_ts = ["R", "C", "u_out", "time_step_r"]
    key_cols_rcu = ["R", "C", "u_out"]

    key_cols_uin = ["R", "C", "u_out", "time_step_r", "u_in_r"]
    key_cols_uin_lag = ["R", "C", "u_out", "time_step_r", "u_in_r", "u_in_lag1_r"]

    key_cols_uin_cum = ["R", "C", "u_out", "step", "u_in_r", "u_in_cum_r"]
    key_cols_int_step = ["R", "C", "u_out", "step", "u_in_int_r"]

    key_cols_uin_cum_diff = [
        "R",
        "C",
        "u_out",
        "step",
        "u_in_r",
        "u_in_cum_r",
        "u_in_diff_r",
    ]

    key_cols_p_lag = ["R", "C", "u_out", "step", "u_in_int_r", "p_lag1_r"]

    mean_p_lag = (
        train.groupby(key_cols_p_lag, sort=False)["pressure"]
        .mean()
        .rename("p_p_lag")
        .reset_index()
    )

    mean_int_step_core = (
        train.groupby(key_cols_int_step, sort=False)["pressure"]
        .mean()
        .rename("p_int_step_core")
        .reset_index()
    )

    mean_uin_cum_diff = (
        train.groupby(key_cols_uin_cum_diff, sort=False)["pressure"]
        .mean()
        .rename("p_uin_cum_diff")
        .reset_index()
    )
    mean_uin_cum = (
        train.groupby(key_cols_uin_cum, sort=False)["pressure"]
        .mean()
        .rename("p_uin_cum")
        .reset_index()
    )
    mean_int_step = (
        train.groupby(key_cols_int_step, sort=False)["pressure"]
        .mean()
        .rename("p_int_step")
        .reset_index()
    )

    mean_uin_lag = (
        train.groupby(key_cols_uin_lag, sort=False)["pressure"]
        .mean()
        .rename("p_uin_lag")
        .reset_index()
    )
    mean_uin = (
        train.groupby(key_cols_uin, sort=False)["pressure"]
        .mean()
        .rename("p_uin")
        .reset_index()
    )
    mean_full = (
        train.groupby(key_cols_full, sort=False)["pressure"]
        .mean()
        .rename("p_full")
        .reset_index()
    )
    mean_ts = (
        train.groupby(key_cols_ts, sort=False)["pressure"]
        .mean()
        .rename("p_ts")
        .reset_index()
    )
    mean_rcu = (
        train.groupby(key_cols_rcu, sort=False)["pressure"]
        .mean()
        .rename("p_rcu")
        .reset_index()
    )

    key_cols_exhale = ["R", "C", "step"]
    mean_exhale = (
        train.loc[train["u_out"] == 1]
        .groupby(key_cols_exhale, sort=False)["pressure"]
        .mean()
        .rename("p_exhale")
        .reset_index()
    )

    allowed_pressures = np.sort(train["pressure"].unique().astype(np.float64))

    preds = np.empty(len(test), dtype=np.float64)

    base = test.copy()
    base = base.merge(mean_uin_cum_diff, on=key_cols_uin_cum_diff, how="left")
    base = base.merge(mean_uin_cum, on=key_cols_uin_cum, how="left")
    base = base.merge(mean_int_step_core, on=key_cols_int_step, how="left")
    base = base.merge(mean_int_step, on=key_cols_int_step, how="left")
    base = base.merge(mean_uin_lag, on=key_cols_uin_lag, how="left")
    base = base.merge(mean_uin, on=key_cols_uin, how="left")
    base = base.merge(mean_full, on=key_cols_full, how="left")
    base = base.merge(mean_ts, on=key_cols_ts, how="left")
    base = base.merge(mean_rcu, on=key_cols_rcu, how="left")
    base = base.merge(mean_exhale, on=key_cols_exhale, how="left")

    mean_p_lag_index = mean_p_lag.set_index(key_cols_p_lag)["p_p_lag"]

    base = base.reset_index(drop=True)

    last_breath = None
    last_pred = global_mean

    for i in range(len(base)):
        row = base.iloc[i]

        b = int(row["breath_id"])
        if b != last_breath:
            last_breath = b
            last_pred = global_mean  # start-of-breath lag default

        if int(row["u_out"]) == 1:
            p = row["p_exhale"]
            if pd.isna(p):
                p = global_mean
            snapped = float(
                snap_to_allowed_pressures(
                    np.array([p], dtype=np.float64), allowed_pressures
                )[0]
            )
            preds[i] = snapped
            continue

        p_lag1_r = float((np.round(float(last_pred) / P_LAG_STEP) * P_LAG_STEP))
        p_lag1_r = float(np.round(p_lag1_r, 2))

        key = (
            int(row["R"]),
            int(row["C"]),
            int(row["u_out"]),
            int(row["step"]),
            float(row["u_in_int_r"]),
            float(p_lag1_r),
        )
        p = mean_p_lag_index.get(key, np.nan)

        if pd.isna(p):
            p = row["p_uin_cum_diff"]
            if pd.isna(p):
                p = row["p_uin_cum"]
            if pd.isna(p):
                p = row["p_int_step_core"]
            if pd.isna(p):
                p = row["p_int_step"]
            if pd.isna(p):
                p = row["p_uin_lag"]
            if pd.isna(p):
                p = row["p_uin"]
            if pd.isna(p):
                p = row["p_full"]
            if pd.isna(p):
                p = row["p_ts"]
            if pd.isna(p):
                p = row["p_rcu"]
            if pd.isna(p):
                p = global_mean

        snapped = float(
            snap_to_allowed_pressures(
                np.array([p], dtype=np.float64), allowed_pressures
            )[0]
        )
        preds[i] = snapped
        last_pred = snapped

    pred_by_id = pd.Series(preds.astype("float64"), index=base["id"].values)
    sub["pressure"] = pred_by_id.reindex(sub["id"].values).astype("float64").values

    if pd.isna(sub["pressure"]).any():
        sub["pressure"] = sub["pressure"].fillna(global_mean).astype("float64")

sub = sub[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)

sub.head(5)



## === cell 3
assert list(sub.columns) == ["id", "pressure"]
assert len(sub) > 0
assert sub["id"].isna().sum() == 0
assert sub["pressure"].isna().sum() == 0

print("Wrote submission.csv")
print(sub.describe(include="all"))
