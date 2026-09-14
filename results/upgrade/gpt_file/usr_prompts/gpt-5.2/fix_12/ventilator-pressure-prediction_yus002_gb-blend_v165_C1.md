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

0.1366266368106691

# 6. Current score

6.29549

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The runtime error comes from `g()` building `flist` as a list of scalars/1D arrays with inconsistent shapes (often ending up length 1), so the blended prediction can’t be assigned to the 603600-row submission. I minimally harden the blending code to (1) only use CSVs that actually contain a `pressure` column, (2) ensure every loaded prediction is a 1D array of the correct length (matching `sample_submission.csv`), and (3) fix the split logic so it never produces empty groups. If the external dataset path doesn’t exist in your environment, the code fall back to writing the unmodified `sample_submission.csv` (all zeros) but still produce a valid `submission.csv`. These changes preserve the original “random-weight median/mean blending + nearest-pressure snapping” core logic while making it run end-to-end and write a valid `.csv`.'
- What this solution (achieved 4.10681) has done: 'Your current score (17.65486, lower-is-better) is far worse than the target (0.1366), and the main reason is that this notebook is only blending external prediction CSVs; if that dataset path is missing/empty or the CSVs don’t match the test length, it falls back to the all-zero sample submission, which scores terribly. I make the smallest change that meaningfully improves toward the target without changing the blending/snapping core: add a deterministic, leakage-free “physics-lite” fallback predictor trained on `train.csv` (KNN regression on the original features) that is only used when no valid external predictions are found. This preserves your evaluation semantics (predict pressure for each test row, then snap to nearest allowed pressure) while ensuring you never submit all zeros again. The rest of the code is kept intact, including the random-weight blending loops and nearest-pressure snapping.'
- What this solution (achieved 2.39618) has done: 'The timeout is dominated by reading the full 5.4M-row training CSV just to build the pressure grid, plus the very slow `.apply(find_nearest)` over 603,600 rows, and (when no blend files exist) the fallback model that trains on the entire training set. I (1) load only the `pressure` column for the grid, (2) replace the row-wise snapping with a vectorized nearest-neighbor snap that is mathematically identical, (3) avoid building huge `pred_list`/`vstack` by accumulating sum and keeping only one stacked array for the median (or using `np.partition` when helpful), and (4) in the fallback, train on a deterministic, bounded subset of breaths (still exact model/feature logic, just avoiding a guaranteed timeout on CPU-only environments). All file paths and the overall blending → snap → write submission semantics remain the same.'
- What this solution (achieved 2.82938) has done: 'Your current MAE (2.39618, lower-is-better) is still far above the target (0.1366), so we should improve the fallback path because it’s likely being used when `../input/gb-data-blending-recover` is missing/empty or incompatible. I keep your blending + nearest-pressure snapping logic unchanged, but replace the mislabeled/weak fallback with a stronger, still simple and deterministic regressor (HistGradientBoostingRegressor) trained on the same baseline features and a bounded, deterministic subset of breaths to stay under the 600s limit. I also fix a small robustness issue by ensuring train/test feature columns align consistently and that the fallback trains only on inspiratory rows (`u_out==0`), matching the evaluation phase and typically improving MAE without changing semantics. The script still always write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 17.65486) has done: 'Your current score (2.82938, lower-is-better) is still far worse than the target (0.1366), so the smallest meaningful improvement is to strengthen the fallback path that’s used when the external blend folder is missing/invalid. I keep your exact “predict → snap to nearest allowed pressure → write submission.csv” semantics, but I make the fallback model breath-aware by predicting per `breath_id` sequence (80 timesteps) instead of treating rows independently, using `MultiOutputRegressor(HistGradientBoostingRegressor)` on a compact set of aggregated breath-level features. This is still a simple scikit-learn regressor (no deep learning, no new training loops), but it aligns much better with the time-series structure and typically reduces MAE substantially toward your target. All paths stay the same, it remains deterministic, and it still produce a valid `submission.csv` even if no external prediction CSVs exist.'
- What this solution (achieved 17.65486) has done: 'Your current MAE (17.65, lower-is-better) is far worse than the target (0.1366), and the most likely cause is that you’re falling back to the all-zero submission because the external blend folder is missing/invalid, and the fallback model is currently broken (the “area” aggregation is a constant placeholder and train/test breath ordering can misalign). I make the smallest fixes that directly improve the fallback while preserving your overall pipeline: keep the same “predict → snap to nearest allowed pressure → write submission.csv” semantics, but correct the breath-level feature engineering (compute true area and remove the placeholder) and ensure train/test breath ordering is consistent when expanding sequences back to rows. These changes should move the score materially toward the target whenever the fallback path is used, without changing your blending logic or adding new modeling approaches. The code still write a valid `submission.csv` in all cases.'
- What this solution (achieved 17.65486) has done: 'Your current MAE (17.65, lower-is-better) is far worse than the target (0.1366), and the most likely reason is that your `_fallback_knn_predict()` is failing and returning `None`, causing the code to silently write the all-zero sample submission. I make the smallest changes that keep your overall “blend external CSVs → else fallback → snap to nearest allowed pressure → write submission.csv” logic intact, but make the fallback robust and actually usable. Concretely: (1) fix the `align()` bug where `breath_id` is being dropped (making `.set_index("breath_id")` fail), and (2) fix the expansion from breath-level predictions back to row-level so it matches each row’s within-breath timestep order deterministically. These two fixes should move the score dramatically toward your target whenever the external blend folder is missing/invalid, without changing your core blending/snapping semantics.'
- What this solution (achieved 17.65486) has done: 'Your current MAE (17.65, lower-is-better) is far worse than the target (0.1366), and the most likely reason is that you’re still falling back to a weak/broken prediction path (or producing misaligned row ordering) whenever the external blend folder is missing/invalid. I keep your overall pipeline exactly the same (blend external CSVs if present → else fallback → snap to nearest allowed pressure → write `submission.csv`), but make two minimal, score-critical fixes in the fallback: (1) use only inspiratory rows (`u_out==0`) for training (matching the metric) while still predicting all 80 timesteps per breath, and (2) ensure per-breath sequence ordering is strictly by within-breath timestep index (0..79) rather than relying on merges that can silently misorder rows. These are small changes that should materially reduce MAE toward the target without changing your blending/snapping semantics or introducing new training loops/architectures. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 4.12558) has done: 'Your current MAE (17.65486, lower-is-better) is far worse than the target (0.1366), so we need to improve the fallback path that’s used when the external blend folder is missing/invalid; right now your fallback is almost certainly failing because scikit-learn isn’t available in the provided package list, causing a silent return to the all-zero submission. I keep your overall pipeline identical (blend external CSVs if present → else fallback → snap to nearest allowed pressures → write `submission.csv`), but replace the scikit-learn-based fallback with a deterministic, fast, pure-numpy “conditional mean by (R,C,t_idx,u_out,u_in_bin)” estimator that trains on a bounded number of breaths to fit the 600s limit. This is still a simple regressor (no new training loops/architectures) and is metric-aligned (learned from inspiratory rows, while producing predictions for all timesteps), and it prevent catastrophic all-zero submissions. I also add a small robustness check so we prefer the fallback when the blend folder exists but contains no valid prediction CSVs.'
- What this solution (achieved 6.29549) has done: 'Your current MAE (4.12558, lower-is-better) is still far above the target (0.1366), so we should improve the fallback path that’s likely being used when the external blend folder is missing/invalid or low-quality. I keep your overall pipeline identical (blend external CSVs if present → else fallback → snap to nearest allowed pressures → write `submission.csv`) and make a minimal, score-relevant upgrade to the pure-pandas fallback: add breath-level conditioning (first a per-(R,C,t_idx,u_in_bin) mean, then a within-breath “pressure shape” prior) while still training only on inspiratory rows. This keeps the same evaluation semantics but typically reduces MAE materially versus a global conditional mean, without adding new packages or changing your blending logic. I also add a small robustness guard to prefer the fallback when blended files exist but are clearly poor/misaligned, ensuring you don’t regress to near-zero/garbage predictions.'

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
df_train_pressure = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
)
unique_pressures = df_train_pressure["pressure"].unique()
sorted_pressures = np.sort(unique_pressures.astype(np.float32, copy=False))
total_pressures_len = len(sorted_pressures)
del df_train_pressure

_sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
_sample_sub_df = pd.read_csv(_sample_sub_path)
EXPECTED_LEN = len(_sample_sub_df)

_test_path = "../input/ventilator-pressure-prediction/test.csv"


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


def snap_to_nearest_pressure_vec(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")

    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    idx1 = np.clip(idx, 0, total_pressures_len - 1)

    p0 = sorted_pressures[idx0]
    p1 = sorted_pressures[idx1]

    choose_lower = np.abs(p0 - pred) < np.abs(p1 - pred)
    out = np.where(choose_lower, p0, p1).astype(np.float32, copy=False)
    return out


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def _load_pressure_vector(csv_path, expected_len=EXPECTED_LEN):
    """
    Enforce that every prediction file contributes a 1D vector of the expected length.
    If not, skip it.
    """
    try:
        df = pd.read_csv(csv_path, usecols=["pressure"])
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy()
    if arr.ndim != 1 or len(arr) != expected_len:
        return None
    return arr


def _add_baseline_features(df):
    df = df.copy()

    df["R"] = df["R"].astype("int16")
    df["C"] = df["C"].astype("int16")

    grp = df.groupby("breath_id", sort=False)
    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = grp["u_out"].shift(1).fillna(0).astype(df["u_out"].dtype)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["dt"] = grp["time_step"].diff().fillna(0.0)
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()

    for c in ["u_in_lag1", "u_in_lag2", "u_in_diff1", "u_in_diff2", "dt", "area"]:
        df[c] = df[c].astype(np.float32)

    return df


def _breath_level_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix to improve fallback score toward target:
    - Compute true breath 'area' (integral of u_in over time) instead of a placeholder constant.
    - Keep aggregation breath-level and deterministic; no change to overall semantics.
    """
    df = df.copy()
    df["R"] = df["R"].astype("int16")
    df["C"] = df["C"].astype("int16")

    g = df.groupby("breath_id", sort=False)

    df["dt"] = g["time_step"].diff().fillna(0.0).astype(np.float32)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_diff1"] = (df["u_in"].astype(np.float32) - df["u_in_lag1"]).astype(
        np.float32
    )

    area = (
        (df["u_in"].astype(np.float32) * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .sum()
        .astype(np.float32)
    )

    agg = pd.DataFrame(
        {
            "breath_id": g.size().index.astype(np.int64),
            "R": g["R"].first().astype(np.int16),
            "C": g["C"].first().astype(np.int16),
            "u_in_mean": g["u_in"].mean().astype(np.float32),
            "u_in_std": g["u_in"].std(ddof=0).fillna(0.0).astype(np.float32),
            "u_in_max": g["u_in"].max().astype(np.float32),
            "u_in_min": g["u_in"].min().astype(np.float32),
            "u_in_last": g["u_in"].last().astype(np.float32),
            "u_in_first": g["u_in"].first().astype(np.float32),
            "u_out_sum": g["u_out"].sum().astype(np.int16),
            "t_last": g["time_step"].last().astype(np.float32),
        }
    )

    dyn = df.groupby("breath_id", sort=False).agg(
        u_in_diff1_abs_mean=(
            "u_in_diff1",
            lambda s: np.float32(
                np.mean(np.abs(s.to_numpy(dtype=np.float32, copy=False)))
            ),
        ),
        u_in_diff1_mean=("u_in_diff1", "mean"),
    )
    dyn["u_in_diff1_mean"] = dyn["u_in_diff1_mean"].astype(np.float32)
    dyn["area"] = area.values.astype(np.float32, copy=False)

    agg = agg.merge(dyn.reset_index(), on="breath_id", how="left")

    agg = pd.get_dummies(agg, columns=["R", "C"], prefix=["R", "C"])
    return agg


def _fallback_knn_predict(test_path=_test_path, expected_len=EXPECTED_LEN):
    """
    Score-critical improvement toward target (minimal semantic change):
    Keep the same pure-pandas/numpy "conditional mean" fallback, but make it breath-aware by:
      (1) learning a per-(R,C,t_idx,u_in_bin) mean as before (strong local signal),
      (2) learning a per-(R,C,t_idx) mean "pressure shape" prior (time-only),
      (3) learning per-(R,C,u_in_bin) mean level (pressure vs valve opening),
      (4) for each test breath, blending these sources and applying a light per-breath
          normalization so the sequence follows a realistic inspiratory trajectory.
    Still trains on inspiratory rows only (u_out==0), matching the evaluation.
    """
    try:
        test_df = pd.read_csv(
            test_path,
            usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
        )
    except Exception:
        return None

    try:
        usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
        train_it = pd.read_csv(
            "../input/ventilator-pressure-prediction/train.csv",
            usecols=usecols,
        )
    except Exception:
        return None

    max_breaths = 80000
    breath_ids = train_it["breath_id"].drop_duplicates().iloc[:max_breaths]
    train_df = train_it[train_it["breath_id"].isin(breath_ids)].copy()
    del train_it, breath_ids
    gc.collect()

    train_df = train_df.sort_values(["breath_id", "time_step"], kind="mergesort")
    test_df = test_df.sort_values(["breath_id", "time_step"], kind="mergesort")

    train_df["t_idx"] = (
        train_df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )
    test_df["t_idx"] = (
        test_df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )

    train_insp = train_df[train_df["u_out"] == 0].copy()
    if len(train_insp) == 0:
        return None

    bin_w = 0.25  # 0..100 -> 401 bins
    max_bin = int(round(100 / bin_w))
    train_insp["u_in_bin"] = np.clip(
        np.rint(train_insp["u_in"].to_numpy(np.float32) / bin_w), 0, max_bin
    ).astype(np.int16)
    test_df["u_in_bin"] = np.clip(
        np.rint(test_df["u_in"].to_numpy(np.float32) / bin_w), 0, max_bin
    ).astype(np.int16)

    global_mean = float(train_insp["pressure"].mean())

    mean_by_rctu = train_insp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[
        "pressure"
    ].mean()

    mean_by_rct = train_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"].mean()

    mean_by_rcu = train_insp.groupby(["R", "C", "u_in_bin"], sort=False)[
        "pressure"
    ].mean()

    mean_by_rc = train_insp.groupby(["R", "C"], sort=False)["pressure"].mean()

    idx_rctu = pd.MultiIndex.from_frame(test_df[["R", "C", "t_idx", "u_in_bin"]])
    idx_rct = pd.MultiIndex.from_frame(test_df[["R", "C", "t_idx"]])
    idx_rcu = pd.MultiIndex.from_frame(test_df[["R", "C", "u_in_bin"]])
    idx_rc = pd.MultiIndex.from_frame(test_df[["R", "C"]])

    v_rctu = mean_by_rctu.reindex(idx_rctu).to_numpy()
    v_rct = mean_by_rct.reindex(idx_rct).to_numpy()
    v_rcu = mean_by_rcu.reindex(idx_rcu).to_numpy()
    v_rc = mean_by_rc.reindex(idx_rc).to_numpy()

    out = v_rctu.copy()
    m = np.isnan(out)
    blend = np.where(~np.isnan(v_rct), 0.6 * v_rct, np.nan) + np.where(
        ~np.isnan(v_rcu), 0.4 * v_rcu, 0.0
    )
    out[m] = blend[m]
    m = np.isnan(out)
    out[m] = v_rc[m]
    m = np.isnan(out)
    out[m] = global_mean

    is_exp = test_df["u_out"].to_numpy(dtype=np.int8) == 1
    out_exp = np.where(~np.isnan(v_rct), v_rct, v_rc)
    out_exp = np.where(~np.isnan(out_exp), out_exp, global_mean)
    pred = np.where(is_exp, out_exp, out).astype(np.float32, copy=False)

    tmp = test_df[["breath_id", "R", "C", "t_idx", "u_out"]].copy()
    tmp["pred"] = pred.astype(np.float32, copy=False)

    shape = v_rct.astype(np.float32, copy=False)
    shape_nan = np.isnan(shape)
    if shape_nan.any():
        shape = np.where(~shape_nan, shape, v_rc.astype(np.float32, copy=False))
        shape = np.where(np.isnan(shape), np.float32(global_mean), shape).astype(
            np.float32, copy=False
        )
    tmp["shape"] = shape

    def _norm_one_breath(g):
        insp = g["u_out"].to_numpy() == 0
        if not np.any(insp):
            return g["pred"].to_numpy(dtype=np.float32, copy=False)

        p = g["pred"].to_numpy(dtype=np.float32, copy=False)
        s = g["shape"].to_numpy(dtype=np.float32, copy=False)

        p_in = p[insp]
        s_in = s[insp]

        mp = float(p_in.mean())
        ms = float(s_in.mean())
        vp = float(p_in.var())
        vs = float(s_in.var())

        scale = np.sqrt((vs + 1e-6) / (vp + 1e-6))
        outp = p.copy()
        outp[insp] = (p_in - mp) * scale + ms
        return outp.astype(np.float32, copy=False)

    tmp["pred2"] = tmp.groupby("breath_id", sort=False, group_keys=False).apply(
        lambda g: pd.Series(_norm_one_breath(g), index=g.index)
    )

    pred2 = tmp["pred2"].to_numpy(dtype=np.float32, copy=False)

    test_sorted_back = test_df[["id"]].copy()
    test_sorted_back["pressure"] = pred2
    pred_final = test_sorted_back.sort_values("id", kind="mergesort")[
        "pressure"
    ].to_numpy(dtype=np.float32, copy=False)

    if pred_final.ndim != 1 or len(pred_final) != expected_len:
        return None
    return pred_final


def wc(input_list):
    l = []
    vecs = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        v = _load_pressure_vector(input_list[i])
        if v is None:
            continue
        l.append(public_lb_score)
        vecs.append(v)

    if len(vecs) == 0:
        return None
    if len(vecs) == 1:
        return vecs[0]

    l_sum = sum(l) if sum(l) != 0 else len(l)
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = vecs[0] * weight1 + vecs[1] * weight2
    return output


def g(dp):
    files = sorted([p for p in glob.glob(os.path.join(dp, "*")) if os.path.isfile(p)])
    files = [p for p in files if p.lower().endswith(".csv")]

    loop_time = 154

    def _write_from_vec(vec):
        output = pd.read_csv(_sample_sub_path)
        output["pressure"] = snap_to_nearest_pressure_vec(vec)
        output.to_csv("submission.csv", index=False)
        return output

    if len(files) == 0:
        fallback = _fallback_knn_predict()
        output = pd.read_csv(_sample_sub_path)
        if fallback is None:
            output.to_csv("submission.csv", index=False)
            return output
        return _write_from_vec(fallback)

    splits = 2 if len(files) >= 2 else 1
    flist_groups = np.array_split(files, splits)

    flist = []
    for grp in flist_groups:
        grp = list(grp)
        if len(grp) == 0:
            continue
        combined = wc(grp)
        if combined is None:
            continue
        if len(combined) != EXPECTED_LEN:
            continue
        flist.append(combined.astype(np.float64, copy=False))

    if len(flist) == 0:
        fallback = _fallback_knn_predict()
        output = pd.read_csv(_sample_sub_path)
        if fallback is None:
            output.to_csv("submission.csv", index=False)
            return output
        return _write_from_vec(fallback)

    pred_mat = np.empty((loop_time, EXPECTED_LEN), dtype=np.float64)

    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(flist)] * len(flist)
        else:
            weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(EXPECTED_LEN, dtype=np.float64)
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_mat[t] = temp
        if (t + 1) % 25 == 0:
            gc.collect()

    output = pd.read_csv(_sample_sub_path)

    median_pred = np.median(pred_mat, axis=0)
    mean_pred = pred_mat.mean(axis=0)
    blended = 0.8 * median_pred + 0.2 * mean_pred

    if float(np.std(blended)) < 0.5:
        fallback = _fallback_knn_predict()
        if fallback is not None:
            output["pressure"] = snap_to_nearest_pressure_vec(fallback)
            output.to_csv(f"rwb {loop_time} loops.csv", index=False)
            output.to_csv("submission.csv", index=False)
            return output

    output["pressure"] = snap_to_nearest_pressure_vec(blended)

    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = snap_to_nearest_pressure_vec(a["pressure"].to_numpy())
    a.to_csv("blend.csv", index=False)
    a.to_csv("submission.csv", index=False)
    return a


def avg(dp):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    vecs = []
    for p in input_list:
        v = _load_pressure_vector(p)
        if v is not None:
            vecs.append(v)

    output = pd.read_csv(_sample_sub_path)
    if len(vecs) == 0:
        fallback = _fallback_knn_predict()
        if fallback is None:
            output.to_csv("submission.csv", index=False)
            return output
        output["pressure"] = snap_to_nearest_pressure_vec(fallback)
        output.to_csv("submission.csv", index=False)
        return output

    output.pressure = np.median(np.vstack(vecs), axis=0)
    output["pressure"] = snap_to_nearest_pressure_vec(output["pressure"].to_numpy())
    output.to_csv("avg.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output




## === cell 2
g("../input/gb-data-blending-recover")
