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

0.1358737842100663

# 6. Current score

1.43009

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the runtime failure by removing the hard dependency on a missing Kaggle dataset directory and instead blend from any prediction CSVs that actually exist under `../input`, falling back to a safe baseline submission if none are found. I also make the blending more memory-safe and deterministic (same blending logic, but without storing 154 full-length arrays at once). Finally, I ensure the script always writes a valid `submission.csv` with the required `id,pressure` columns and correct row count.'
- What this solution (achieved 9.92097) has done: 'Your current pipeline is blending arbitrary `id,pressure` CSVs found under `../input`, which almost certainly mixes in non-model files or misaligned predictions and drives the MAE to a very poor value (17.65). To move the score sharply toward the target with minimal logic change, I keep your “blend if found, else fallback” structure but add strict candidate filtering: only accept CSVs whose `id` column matches `sample_submission` exactly and whose pressure values look like real ventilator pressures (not all-zeros, not labels, not weird ranges). If no valid prediction files exist, I switch the fallback from “all zeros” to a simple, legitimate baseline: per-(R,C,time_step) median pressure from train merged onto test (and then snapped to the nearest allowed pressure), which is still simple and fast but should drastically reduce MAE. This preserves your snapping (`find_nearest`) and submission writing semantics and stays within the 600s budget.'
- What this solution (achieved 7.77976) has done: 'The crash comes from `pd.merge_asof` requiring the merge key (`time_step`) to be globally sorted even when using `by=["R","C"]`; sorting by `["R","C","time_step"]` is not sufficient for pandas’ asof join checks. I fix this by sorting both left and right primarily by `time_step` (and secondarily by `R,C`) and resetting the index before `merge_asof`, which resolves the “left keys must be sorted” error without changing the baseline logic. I also make the fallback read only needed columns from train to reduce memory and runtime risk, but keep the same median-by-(R,C,time_step) + nearest snapping semantics. The script then run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 7.77933) has done: 'Your MAE is far above the target, so the priority is a minimal, legitimate improvement in the fallback (since you typically won’t have 2+ valid prediction CSVs to blend). The biggest issue is the current fallback uses only `(R,C,time_step)` medians, which ignores the “inspiratory only” scoring and causes large errors during expiration; we can keep the same median-by-keys idea but add `u_out` to the grouping and merge keys so expiratory steps (u_out=1) get an appropriate median (near-zero-flow regime) without changing the overall approach. I also keep the same `merge_asof` fix (global sorting by `time_step`) and the same nearest-pressure snapping, so evaluation semantics remain the same. This change is small, fast, and should move the score substantially toward your target band.'
- What this solution (achieved 5.07631) has done: 'Your current score (7.78 MAE) is far above the target (0.136), so we should improve the fallback baseline (which you hit whenever there aren’t 2+ valid prediction CSVs to blend). The biggest issue is that the fallback predicts pressure for expiratory rows (u_out=1) using the same medians as inspiratory, and it also doesn’t leverage the strong auto-regressive signal that the pressure at time t is close to pressure at time t-1 within the same breath. With minimal change to your existing “median lookup + merge_asof + snap-to-allowed-pressures” core, I add `u_in` (coarsely binned) and a `pressure_lag1` feature (previous time_step pressure for the same R,C,u_out,u_in_bin) into the median table and use them in a two-stage merge to better track the trajectory, while keeping all your blending code untouched. This should materially reduce MAE toward your target while remaining fast and producing a valid `submission.csv`.'
- What this solution (achieved 4.8354) has done: 'Your current MAE (5.076) is far above the target (0.136, lower is better), so the biggest lever—without changing the overall “table lookup + merge_asof + snap-to-allowed-pressures” core—is to make the fallback lookup table much more specific to the breath trajectory. I keep your existing blending logic untouched, but improve the fallback by (1) adding `time_step_prev` + `u_in_prev` and (2) using an *asof over cumulative time within (R,C,u_out,u_in_bin)* instead of only nearest `time_step`, which better matches the auto-regressive nature of the signal while staying in the same merge/table family. I also ensure merges are strictly sorted to avoid pandas `merge_asof` pitfalls, and keep the final snapping with `find_nearest` exactly as you already do. This should materially reduce MAE toward your target while still being a fast, deterministic baseline when no valid external prediction CSVs exist.'
- What this solution (achieved 4.23832) has done: 'Your current MAE (4.8354, lower is better) is still far from the target (0.1359), so the biggest safe improvement—without changing your overall “lookup table + merge_asof + snap-to-allowed-pressures” fallback core—is to build the fallback table in true breath-time rather than in an artificial “cum_t” derived from diffs inside shuffled groups (which does not correspond to physical time and hurts matching). I keep your blending logic unchanged, but in the fallback I (1) build a denser, more reliable median table keyed by `(R,C,u_out,u_in_bin,time_bin)` computed from real `time_step`, and (2) use `merge_asof` on `time_bin` with correct global sorting to match test rows to the nearest time bin for the same keys. This is still the same family of approach (table lookup + asof + nearest pressure snapping), but it aligns the lookup with the metric’s time-series structure and should materially reduce MAE toward the target band. The script still always writes a valid `submission.csv` with `id,pressure` and correct row count.'
- What this solution (achieved 1.5863) has done: 'Your current MAE (4.23832, lower is better) is still far above the target (0.1359), and you are almost certainly hitting the fallback baseline (no valid external prediction CSVs to blend). To move the score toward the target without changing the overall “lookup table + (a)sof matching + snap-to-allowed-pressures” core, I make the fallback much more faithful to the competition’s scoring: (1) set expiratory steps (`u_out==1`) to a safe constant (nearest-to-zero pressure) because they are not scored, and (2) strengthen the inspiratory lookup by adding a simple, stable cumulative-integral feature (`u_in_cum`) binned per breath to capture trajectory shape, while still using median tables and `merge_asof`. These are minimal, deterministic additions that keep your submission semantics identical and typically reduce MAE a lot on this competition. The blending logic is left intact; only the fallback is improved because that’s what determines your score here.'
- What this solution (achieved 1.36499) has done: 'Your current MAE (1.5863, lower is better) is still far above the target (0.1359), and the fastest legitimate path within your existing “fallback table lookup + asof + snap-to-allowed-pressures” core is to make the fallback approximate the real pressure dynamics more closely. I keep your blending logic unchanged and only strengthen the fallback by adding two minimal, competition-standard physical features (breath-wise cumulative volume `u_in * dt` and instantaneous flow proxy `u_in` difference) into the median lookup keys, so inspiratory rows match to more trajectory-consistent medians. I also tighten the inspiratory-only handling by setting `u_out==1` to a constant (as you already do) and ensure all asof merges remain globally sorted to avoid pandas errors. These changes are deterministic, remain a pure lookup/asof baseline (no new model/training loop), and should move the MAE materially toward your target.'
- What this solution (achieved 1.32893) has done: 'Your current MAE (1.36499, lower is better) is still far above the target (0.1359), and since you likely have no strong external prediction CSVs to blend, the score is dominated by the fallback. To move toward the target while preserving your core “lookup table + merge_asof + snap-to-allowed-pressures” approach, I minimally add two competition-standard trajectory keys to the fallback: within-breath cumulative integral of `u_in` and within-breath cumulative count of `u_out==0` steps (a proxy for inspiratory progress), both binned and used only to refine the median table. I also keep your existing backoff merges, expiratory handling (set `u_out==1` to a constant), and final snapping unchanged, and I ensure `merge_asof` sorting remains globally correct to avoid pandas errors.'
- What this solution (achieved 1.33473) has done: 'Your current MAE (1.32893, lower is better) is still far above the target (0.13587), so we should make a small, legitimate improvement in the fallback baseline (since the blend path usually won’t find 2+ valid prediction CSVs). I keep your existing “median lookup table + merge_asof + snap-to-allowed-pressures” core unchanged, but refine the inspiratory matching with one additional breath-progress descriptor that’s very predictive here: within-breath cumulative “open valve time” (`dt_cum_insp`), binned and added to the lookup keys. This is a minimal extension of the same lookup/asof approach (no new model/training), and it should move the score down toward the target. I also keep your expiratory handling (set `u_out==1` to a constant) and your blending logic exactly as-is.'
- What this solution (achieved 1.43009) has done: 'Your current MAE (1.33473, lower is better) is still far above the target (0.13587), so we should improve the *fallback* (since the blend path typically won’t find 2+ valid aligned prediction CSVs). With minimal change to your existing “median lookup table + merge_asof + snap-to-allowed-pressures” core, I (1) make the inspiratory lookup more trajectory-consistent by adding a coarse `u_in`-integral-to-current (`u_in_cum_bin`) and `time_step` itself (continuous) back into the primary asof join (while keeping your existing keys), and (2) replace the fixed 0.88/0.12 smoothing with a slightly more lag-aware blend only when lag exists, which usually reduces noise without changing the overall approach. I keep expiratory handling (set `u_out==1` to constant), candidate CSV filtering, and final snapping unchanged, and ensure the script still writes a valid `submission.csv`. All changes are localized to the fallback function and preserve the same evaluation semantics.'

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
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Robust blending of 1-2 submission CSVs.
    Bugfix: do not rely on filename-encoded scores/weights; use fixed weights.
    """
    preds = []
    for p in input_list:
        s = pd.read_csv(p)
        if "pressure" not in s.columns:
            raise ValueError(f"File {p} does not contain 'pressure' column.")
        preds.append(s["pressure"].to_numpy().ravel())

    n = len(preds[0])
    for i, arr in enumerate(preds):
        if len(arr) != n:
            raise ValueError(
                f"Prediction length mismatch in {input_list[i]}: {len(arr)} vs {n}"
            )

    if len(preds) == 1:
        return preds[0]

    weight1 = 0.6
    weight2 = 1.0 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _pressure_sanity_ok(p_arr: np.ndarray) -> bool:
    """
    Filter out non-prediction CSVs.
    Heuristics tuned to this competition:
      - not constant (sample_submission baseline is all zeros -> terrible score)
      - values within plausible pressure range
    """
    if p_arr.ndim != 1 or p_arr.size == 0:
        return False
    p_arr = p_arr.astype(np.float64, copy=False)
    if np.nanstd(p_arr) < 1e-8:
        return False
    if np.nanmin(p_arr) < -5.0:
        return False
    if np.nanmax(p_arr) > 60.0:
        return False
    return True


def _find_candidate_prediction_csvs(dp):
    """
    Accept ONLY properly aligned submissions.
    - must be a CSV
    - must have columns id and pressure
    - must match sample_submission length
    - must have identical id order/values as sample_submission (prevents misalignment MAE blow-ups)
    - must pass pressure sanity checks (avoid all-zero / junk files)
    - must not be the official sample_submission itself
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    sample = pd.read_csv(sample_path)
    n = len(sample)
    sample_ids = sample["id"].to_numpy()

    candidates = []
    for p in glob.iglob(f"{dp}/**/*.csv", recursive=True):
        if not os.path.isfile(p):
            continue
        if os.path.abspath(p) == os.path.abspath(sample_path):
            continue

        try:
            head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if not (("pressure" in head.columns) and ("id" in head.columns)):
            continue

        try:
            full = pd.read_csv(p, usecols=["id", "pressure"])
        except Exception:
            continue
        if len(full) != n:
            continue

        ids = full["id"].to_numpy()
        if ids.dtype != sample_ids.dtype:
            try:
                ids = ids.astype(sample_ids.dtype, copy=False)
            except Exception:
                continue
        if not np.array_equal(ids, sample_ids):
            continue

        p_arr = full["pressure"].to_numpy().ravel()
        if not _pressure_sanity_ok(p_arr):
            continue

        candidates.append(p)

    candidates.sort()
    return candidates


def _fallback_baseline_by_rc_time_step():
    """
    Score-improving but still "same core logic" fallback:
    - Keep: table lookup + merge_asof + snap to allowed pressures.
    - Keep: expiratory phase is NOT scored -> set u_out==1 to a safe constant.

    Minimal improvement to move MAE toward target:
    - Add u_in_cum_bin (already computed) into the primary lookup keys to better match
      trajectory state (pressure strongly depends on delivered volume).
    - Use merge_asof on continuous time_step (not only discretized time_bin) while still
      keeping time_bin columns for the table (nearest-time matching becomes more stable).
    - Make the lag blend slightly more adaptive: only blend when lag exists, reducing
      bias in early timesteps.
    """
    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"],
    )

    exp_const = float(find_nearest(0.0))

    def _add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
            drop=True
        )
        g = df.groupby("breath_id", sort=False)

        dt = g["time_step"].diff().fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
        u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)

        incr = u_in * dt
        df["u_in_cum"] = (
            pd.Series(incr).groupby(df["breath_id"], sort=False).cumsum().to_numpy()
        )

        df["u_in_diff"] = (
            g["u_in"].diff().fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        )

        insp_mask = (u_out == 0).astype(np.int16)
        df["insp_step"] = (
            pd.Series(insp_mask)
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .to_numpy()
        )

        incr_insp = incr * (u_out == 0)
        df["u_in_cum_insp"] = (
            pd.Series(incr_insp)
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .to_numpy()
        )

        dt_insp = dt * (u_out == 0)
        df["dt_cum_insp"] = (
            pd.Series(dt_insp).groupby(df["breath_id"], sort=False).cumsum().to_numpy()
        )

        return df

    df_train_local = _add_breath_features(
        df_train[
            ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
        ].copy()
    )
    df_test_local = _add_breath_features(df_test.copy())

    bin_w = 2.5
    df_train_local["u_in_bin"] = np.floor(
        df_train_local["u_in"].to_numpy() / bin_w
    ).astype(np.int16)
    df_test_local["u_in_bin"] = np.floor(
        df_test_local["u_in"].to_numpy() / bin_w
    ).astype(np.int16)

    cum_bin_w = 0.5
    df_train_local["u_in_cum_bin"] = np.floor(
        df_train_local["u_in_cum"].to_numpy() / cum_bin_w
    ).astype(np.int16)
    df_test_local["u_in_cum_bin"] = np.floor(
        df_test_local["u_in_cum"].to_numpy() / cum_bin_w
    ).astype(np.int16)

    diff_bin_w = 1.0
    df_train_local["u_in_diff_bin"] = np.floor(
        df_train_local["u_in_diff"].to_numpy() / diff_bin_w
    ).astype(np.int16)
    df_test_local["u_in_diff_bin"] = np.floor(
        df_test_local["u_in_diff"].to_numpy() / diff_bin_w
    ).astype(np.int16)

    insp_step_w = 2
    df_train_local["insp_step_bin"] = np.floor(
        df_train_local["insp_step"].to_numpy() / insp_step_w
    ).astype(np.int16)
    df_test_local["insp_step_bin"] = np.floor(
        df_test_local["insp_step"].to_numpy() / insp_step_w
    ).astype(np.int16)

    cum_insp_bin_w = 0.5
    df_train_local["u_in_cum_insp_bin"] = np.floor(
        df_train_local["u_in_cum_insp"].to_numpy() / cum_insp_bin_w
    ).astype(np.int16)
    df_test_local["u_in_cum_insp_bin"] = np.floor(
        df_test_local["u_in_cum_insp"].to_numpy() / cum_insp_bin_w
    ).astype(np.int16)

    dt_cum_insp_w = 0.03  # roughly one original time step
    df_train_local["dt_cum_insp_bin"] = np.floor(
        df_train_local["dt_cum_insp"].to_numpy() / dt_cum_insp_w
    ).astype(np.int16)
    df_test_local["dt_cum_insp_bin"] = np.floor(
        df_test_local["dt_cum_insp"].to_numpy() / dt_cum_insp_w
    ).astype(np.int16)

    t_bin = 0.03
    df_train_local["time_bin"] = np.round(
        df_train_local["time_step"].to_numpy() / t_bin
    ).astype(np.int16)
    df_test_local["time_bin"] = np.round(
        df_test_local["time_step"].to_numpy() / t_bin
    ).astype(np.int16)

    key_cols_primary = [
        "R",
        "C",
        "u_out",
        "u_in_bin",
        "u_in_cum_bin",  # NEW: strengthens state matching with minimal extra compute
        "u_in_cum_insp_bin",
        "dt_cum_insp_bin",
        "insp_step_bin",
        "u_in_diff_bin",
    ]
    key_cols_backoff = [
        "R",
        "C",
        "u_out",
        "u_in_bin",
        "u_in_cum_bin",  # keep for backoff1
        "u_in_cum_insp_bin",
        "dt_cum_insp_bin",
        "insp_step_bin",
    ]
    key_cols_backoff2 = ["R", "C", "u_out", "u_in_bin"]

    df_train_local = df_train_local.sort_values(
        key_cols_primary + ["time_bin"], kind="mergesort"
    ).reset_index(drop=True)
    df_train_local["pressure_lag1"] = df_train_local.groupby(
        key_cols_primary, sort=False
    )["pressure"].shift(1)

    med_primary = (
        df_train_local.groupby(key_cols_primary + ["time_bin"], as_index=False)[
            ["pressure", "pressure_lag1"]
        ]
        .median()
        .rename(columns={"pressure": "pred_pressure", "pressure_lag1": "pred_lag1"})
    )

    df_train_b1 = df_train_local.sort_values(
        key_cols_backoff + ["time_bin"], kind="mergesort"
    ).reset_index(drop=True)
    df_train_b1["pressure_lag1_b1"] = df_train_b1.groupby(key_cols_backoff, sort=False)[
        "pressure"
    ].shift(1)
    med_backoff1 = (
        df_train_b1.groupby(key_cols_backoff + ["time_bin"], as_index=False)[
            ["pressure", "pressure_lag1_b1"]
        ]
        .median()
        .rename(
            columns={"pressure": "pred_pressure_b1", "pressure_lag1_b1": "pred_lag1_b1"}
        )
    )

    df_train_b2 = df_train_local.sort_values(
        key_cols_backoff2 + ["time_bin"], kind="mergesort"
    ).reset_index(drop=True)
    df_train_b2["pressure_lag1_b2"] = df_train_b2.groupby(
        key_cols_backoff2, sort=False
    )["pressure"].shift(1)
    med_backoff2 = (
        df_train_b2.groupby(key_cols_backoff2 + ["time_bin"], as_index=False)[
            ["pressure", "pressure_lag1_b2"]
        ]
        .median()
        .rename(
            columns={"pressure": "pred_pressure_b2", "pressure_lag1_b2": "pred_lag1_b2"}
        )
    )

    df_test_sorted = df_test_local.sort_values(
        ["time_step"] + key_cols_primary, kind="mergesort"
    ).reset_index(drop=True)
    med_primary_sorted = med_primary.merge(
        df_train_local[["time_bin", "time_step"]].drop_duplicates("time_bin"),
        on="time_bin",
        how="left",
    )
    med_primary_sorted = med_primary_sorted.sort_values(
        ["time_step"] + key_cols_primary, kind="mergesort"
    ).reset_index(drop=True)

    out = pd.merge_asof(
        df_test_sorted,
        med_primary_sorted.drop(columns=["time_bin"]),
        on="time_step",
        by=key_cols_primary,
        direction="nearest",
        allow_exact_matches=True,
    )

    missing = out["pred_pressure"].isna()
    if missing.any():
        tmp = out.loc[
            missing,
            [
                "id",
                "R",
                "C",
                "u_out",
                "u_in_bin",
                "u_in_cum_bin",
                "u_in_cum_insp_bin",
                "dt_cum_insp_bin",
                "insp_step_bin",
                "time_step",
            ],
        ].copy()

        tmp_sorted = tmp.sort_values(
            ["time_step"] + key_cols_backoff, kind="mergesort"
        ).reset_index(drop=True)

        med_b1_sorted = med_backoff1.merge(
            df_train_local[["time_bin", "time_step"]].drop_duplicates("time_bin"),
            on="time_bin",
            how="left",
        )
        med_b1_sorted = med_b1_sorted.sort_values(
            ["time_step"] + key_cols_backoff, kind="mergesort"
        ).reset_index(drop=True)

        tmp2 = pd.merge_asof(
            tmp_sorted,
            med_b1_sorted.drop(columns=["time_bin"]),
            on="time_step",
            by=key_cols_backoff,
            direction="nearest",
            allow_exact_matches=True,
        )
        tmp2 = tmp2[["id", "pred_pressure_b1", "pred_lag1_b1"]]
        out = out.merge(tmp2, on="id", how="left")

        out["pred_pressure"] = out["pred_pressure"].fillna(out["pred_pressure_b1"])
        out["pred_lag1"] = out["pred_lag1"].fillna(out["pred_lag1_b1"])
        out.drop(columns=["pred_pressure_b1", "pred_lag1_b1"], inplace=True)

    missing = out["pred_pressure"].isna()
    if missing.any():
        tmp = out.loc[
            missing, ["id", "R", "C", "u_out", "u_in_bin", "time_step"]
        ].copy()
        tmp_sorted = tmp.sort_values(
            ["time_step"] + key_cols_backoff2, kind="mergesort"
        ).reset_index(drop=True)

        med_b2_sorted = med_backoff2.merge(
            df_train_local[["time_bin", "time_step"]].drop_duplicates("time_bin"),
            on="time_bin",
            how="left",
        )
        med_b2_sorted = med_b2_sorted.sort_values(
            ["time_step"] + key_cols_backoff2, kind="mergesort"
        ).reset_index(drop=True)

        tmp2 = pd.merge_asof(
            tmp_sorted,
            med_b2_sorted.drop(columns=["time_bin"]),
            on="time_step",
            by=key_cols_backoff2,
            direction="nearest",
            allow_exact_matches=True,
        )
        tmp2 = tmp2[["id", "pred_pressure_b2", "pred_lag1_b2"]]
        out = out.merge(tmp2, on="id", how="left")

        out["pred_pressure"] = out["pred_pressure"].fillna(out["pred_pressure_b2"])
        out["pred_lag1"] = out["pred_lag1"].fillna(out["pred_lag1_b2"])
        out.drop(columns=["pred_pressure_b2", "pred_lag1_b2"], inplace=True)

    global_med = float(df_train_local["pressure"].median())
    out["pred_pressure"] = out["pred_pressure"].fillna(global_med)
    out["pred_lag1"] = out["pred_lag1"].fillna(out["pred_pressure"])

    lag_ok = out["pred_lag1"].notna().to_numpy()
    out.loc[lag_ok, "pred_pressure"] = (
        0.85 * out.loc[lag_ok, "pred_pressure"] + 0.15 * out.loc[lag_ok, "pred_lag1"]
    )

    out.loc[out["u_out"].to_numpy() == 1, "pred_pressure"] = exp_const

    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    sub = sub.merge(out[["id", "pred_pressure"]], on="id", how="left")
    sub["pressure"] = sub["pred_pressure"].astype(np.float64)
    sub.drop(columns=["pred_pressure"], inplace=True)

    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    """
    Only blend if we have at least two candidate prediction CSVs.
    With 0 or 1 candidates, default to the median fallback.
    """
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    n = len(output)

    if (dp is None) or (not os.path.exists(dp)):
        l = []
    else:
        l = _find_candidate_prediction_csvs(dp)

    if len(l) < 2:
        return _fallback_baseline_by_rc_time_step()

    splits = max(1, len(l) // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        flist.append(l[start:end])

    group_preds = []
    for grp in flist:
        vec = wc(grp)
        if len(vec) != n:
            raise ValueError(f"Blended vector length mismatch: {len(vec)} vs {n}")
        group_preds.append(vec.astype(np.float64, copy=False))

    loop_time = 154
    combos = np.empty((loop_time, n), dtype=np.float32)
    mean_acc = np.zeros(n, dtype=np.float64)

    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(group_preds))]
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(group_preds)):
            temp += group_preds[j] * weight[j]

        combos[seed, :] = temp.astype(np.float32)
        mean_acc += temp

    median_pred = np.median(combos, axis=0).astype(np.float64)
    mean_pred = (mean_acc / loop_time).astype(np.float64)

    output["pressure"] = 0.8 * median_pred + 0.2 * mean_pred
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = _find_candidate_prediction_csvs(dp)
    if len(input_list) == 0:
        return _fallback_baseline_by_rc_time_step()

    preds = []
    for p in input_list:
        preds.append(pd.read_csv(p, usecols=["pressure"]).pressure.to_numpy().ravel())

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(preds), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("avg.csv", index=False)
    return output




## === cell 2
g("../input")
