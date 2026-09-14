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

0.1504261141425439

# 6. Current score

5.44168

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The failure is because the code tries to read four external Kaggle dataset submissions that are not present in this environment, so the blend inputs never load and the pipeline crashes before writing `submission.csv`. To keep the core “blend multiple submissions” logic but make it runnable end-to-end, I add a small fallback that uses the provided `sample_submission.csv` whenever any of the external files are missing. I also ensure all blended frames align by `id` (to avoid silent row-order mismatches) and always write a valid `submission.csv` with the required columns. This yield a valid submission file (though with fallback inputs it won’t hit the target score, since the missing blend components are the main performance driver).'
- What this solution (achieved 3.91446) has done: 'Your current score is far worse than the target (lower-is-better), and the main issue is that your “blend of external submissions” can’t work here because those external files aren’t available—so you’re effectively submitting near-all-zeros, which performs terribly. To move the score toward the target while preserving the core “blend submissions” logic, I keep the blending approach but replace missing external submissions with a lightweight, deterministic in-notebook model that generates a reasonable per-row pressure prediction from the provided train/test (a KNN regressor on simple per-timestep features). I also ensure strict `id` alignment across all blended sources and keep the output as a valid `submission.csv`. This should substantially reduce MAE versus zeros, moving much closer to the target band without changing the overall blending semantics.'
- What this solution (achieved 4.00108) has done: 'Your current score (3.91446, lower-is-better) is far from the target (0.1504), and the biggest controllable issue inside the existing “blend submissions” framework is the weak KNN fallback: it is training on all rows (slow, noisy) and ignores the competition’s key structure (per-breath time series + inspiratory-only scoring). I keep the exact blending logic/weights intact, but replace the fallback with a fast per-(R,C,time_step,u_out,u_in rounded) lookup built from train inspiratory rows, which is deterministic and much closer to how strong baseline solutions work. This stays within the same core approach (“generate a fallback submission when external files are missing”) while materially improving the fallback predictions. I also ensure strict `id` alignment and still always write a valid `submission.csv`.'
- What this solution (achieved 4.12058) has done: 'Your score is far worse than the target (lower-is-better), so we should improve the fallback predictions (since the external blend files are missing here and you’re effectively submitting only the fallback). Keeping the same “blend submissions with fixed weights” core logic, I make the fallback closer to strong VPP baselines by (1) using exact per-breath time indices instead of rounding `time_step`, (2) quantizing to the known discrete pressure grid (a key trick for this competition), and (3) adding two minimal backoff levels plus a final nearest-neighbor-on-grid fill for any unseen keys. These are small, deterministic changes that better match the competition’s structure/metric without changing your blending semantics, and still produce a valid `submission.csv` end-to-end within the time limit.'
- What this solution (achieved 5.9386) has done: 'Your current score (4.12058, lower-is-better) is still far from the target (0.1504), and since the external blend files are missing, the score is dominated by the fallback generator. Keeping the same overall “blend fixed-weight submissions” logic, I improve only the fallback by (1) using exact `u_in` values (no rounding) for a primary lookup, (2) adding a stronger per-(R,C,t_idx,u_out,u_in) median lookup plus sensible backoffs, and (3) snapping predictions to the known discrete pressure grid after the final blend (so the blend also lands on valid pressure levels). These are minimal, deterministic changes aligned with the competition’s pressure discretization and should reduce MAE substantially versus the current fallback.'
- What this solution (achieved 5.03461) has done: 'Your current score is much worse than the target (lower-is-better), so the goal is to improve only the fallback predictions (since the external blend files are missing and you effectively submit the fallback). I keep the same overall “blend fixed-weight submissions” logic and only strengthen the lookup fallback by adding a more competition-faithful feature: per-breath cumulative inspired volume (`u_in` integrated over time), which helps distinguish states that share the same instantaneous `u_in`/`t_idx`. I then add two minimal backoff levels using this integrated feature, and keep the existing pressure-grid snapping (but now snapping uses the official, constant grid spacing to avoid any numerical mismatch). These changes are deterministic, run within the time limit, and should move MAE substantially downward toward your target.'
- What this solution (achieved 5.44168) has done: 'Your current score (5.03461, lower-is-better) is far from the target (0.1504), and since the external blend files are missing here, the score is dominated by the fallback generator. To move the MAE down without changing the overall “fixed-weight blending” core logic, I strengthen only the fallback by adding the classic ventilator engineered state features (`u_in` lag, cumulative `u_in`, and cumulative area under `u_in`) and use them in the existing hierarchical median-lookup/backoff framework. I also stop forcing `u_out==1` predictions to 0 (that heuristic can be very wrong for this competition and can heavily damage MAE) while keeping the same pressure-grid snapping and submission alignment.'
- What this solution (achieved 5.44168) has done: 'Your current MAE (5.44, lower-is-better) is far from the target (0.1504), and since the external blend files are unavailable the score is dominated by the fallback. Keeping the same “fixed-weight blending of submissions” core logic, the smallest meaningful improvement is to make the fallback closer to classic strong baselines by (1) learning a per-breath “pressure(t_idx)” prior for each (R,C) and (2) learning a simple linear correction of pressure using cumulative volume features, both computed only from inspiratory (u_out==0) train rows to match the metric. We then use these as additional backoff levels only when the existing hierarchical lookups miss, and we still snap final predictions to the official discrete pressure grid. This is deterministic, fast, preserves your blending semantics, and should move MAE down substantially toward the target.'
- What this solution (achieved 5.44168) has done: 'Your current MAE (5.44) is far from the target (0.1504), and since the external blend files aren’t available the final submission is effectively your fallback; so we should improve only the fallback while keeping the same “fixed-weight blend of submissions” core logic. The biggest low-risk gain for this competition is to enforce the known physics/metric rule: rows in expiration (`u_out==1`) are not scored and should typically copy the last inspiratory pressure within the same breath (forward-fill), rather than using a separate median model that can be very wrong. I implement this as a post-processing step on the fallback predictions (and therefore on the blended result too, since everything falls back), while keeping your existing lookups, linear backoff, and pressure-grid snapping unchanged. This is deterministic, fast, and should materially reduce MAE toward the target without changing the overall approach.'
- What this solution (achieved 5.44168) has done: 'Your MAE (5.44, lower-is-better) is far from the target, and since the external blend files aren’t available here the blended output is effectively just your fallback, so the only viable way to move toward the target is to improve the fallback while preserving the same fixed-weight blending pipeline. The biggest score-killer remaining is that the forward-fill for `u_out==1` is currently applied across the whole breath, which can incorrectly overwrite inspiratory predictions after the first expiratory point; we change it to a “expiration-only carry” that fills only within contiguous expiration segments and never modifies `u_out==0`. We also ensure the linear backoff (`p_lin`) is trained only on reliable groups (enough samples + nontrivial variance) to avoid noisy slopes that can blow up predictions. Finally, we apply the same expiration-only carry to the final blended predictions (not just the fallback) before the final grid snap, which is consistent with your existing intent and can reduce errors when any source has unstable expiration values.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.neighbors import (
    KNeighborsRegressor,
)  # kept to preserve original imports/core intent



## === cell 1
BASE = "../input/ventilator-pressure-prediction"
sub_path = f"{BASE}/sample_submission.csv"
train_path = f"{BASE}/train.csv"
test_path = f"{BASE}/test.csv"

sub = pd.read_csv(sub_path)

blend_paths = {
    "sub_1": "../input/random-weights-blending-tool-ventilator-pressure/rwb 125 loops.csv",
    "sub_2": "../input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    "sub_3": "../input/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv",
    "sub_4": "../input/vpp-lstm-baseline-median-pp/submission.csv",
}


def load_or_fallback(path: str, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """Load a submission-like CSV or fall back to provided df (must have id, pressure)."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = fallback_df.copy()

    if "id" not in df.columns or "pressure" not in df.columns:
        df = fallback_df.copy()

    df = df[["id", "pressure"]].copy()
    df = df.sort_values("id").reset_index(drop=True)
    return df


def _snap_to_pressure_grid(pred: np.ndarray, pressure_grid: np.ndarray) -> np.ndarray:
    """Snap continuous predictions to the discrete pressure grid (competition-specific)."""
    pred = pred.astype(np.float32, copy=False)
    idx = np.searchsorted(pressure_grid, pred, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    prev_idx = np.clip(idx - 1, 0, len(pressure_grid) - 1)
    next_val = pressure_grid[idx]
    prev_val = pressure_grid[prev_idx]
    pred_snapped = np.where(
        np.abs(pred - prev_val) <= np.abs(pred - next_val), prev_val, next_val
    ).astype(np.float32)
    return pred_snapped


def _apply_u_out_expiration_carry(
    test_df: pd.DataFrame, pred_snapped: np.ndarray
) -> np.ndarray:
    """
    Score-relevant post-process (minimal, preserves semantics):
    - Only expiration (u_out==1) is unscored; we can safely stabilize it.
    - IMPORTANT FIX vs prior forward-fill: do NOT let expiration values overwrite
      later inspiratory rows. We carry-forward only within contiguous expiration
      segments: for each breath, each u_out==1 row gets the last u_out==0
      prediction from earlier in the same breath; u_out==0 rows are never changed.
    This avoids corrupting inspiratory predictions (which are scored).
    """
    tmp = test_df[["breath_id", "u_out"]].copy()
    tmp["pred"] = pred_snapped.astype(np.float32)

    insp_pred = tmp["pred"].where(tmp["u_out"].to_numpy(dtype=np.int8) == 0, np.nan)
    last_insp = insp_pred.groupby(tmp["breath_id"]).ffill()

    u_out_arr = tmp["u_out"].to_numpy(dtype=np.int8)
    carried = np.where(
        u_out_arr == 1,
        last_insp.to_numpy(dtype=np.float32),
        tmp["pred"].to_numpy(dtype=np.float32),
    )
    carried = np.where(
        np.isfinite(carried), carried, tmp["pred"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    return carried


def build_lookup_fallback_submission(sample_sub_sorted: pd.DataFrame):
    """
    Fallback predictions (dominant when external blend files are missing).

    Change (score-relevant, minimal):
    - Fix expiration post-process to avoid overwriting inspiratory rows (scored).
    - Make linear backoff more stable by only using groups with enough samples
      and nontrivial cum_area variance (prevents noisy slopes).
    """
    train = pd.read_csv(
        train_path,
        usecols=["id", "R", "C", "breath_id", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "R", "C", "breath_id", "time_step", "u_in", "u_out"]
    )

    train_insp = train[train["u_out"] == 0].copy()

    pmin = np.float32(train_insp["pressure"].min())
    pmax = np.float32(train_insp["pressure"].max())
    step = np.float32(0.070302145)  # known constant step for this dataset/competition
    n_steps = int(np.rint((pmax - pmin) / step)) + 1
    pressure_grid = (pmin + step * np.arange(n_steps, dtype=np.float32)).astype(
        np.float32
    )

    train_insp["t_idx"] = train_insp.groupby("breath_id").cumcount().astype(np.int16)
    test["t_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

    for df in (train_insp, test):
        df["time_step"] = df["time_step"].astype(np.float32)
        df["u_in"] = df["u_in"].astype(np.float32)
        df["u_out"] = df["u_out"].astype(np.int8)

        df["u_in_lag1"] = (
            df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )

        df["dt"] = (
            df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
        )

        df["u_in_area"] = (df["u_in"] * df["dt"]).astype(np.float32)
        df["cum_area"] = (
            df.groupby("breath_id")["u_in_area"].cumsum().astype(np.float32)
        )
        df["cum_area_q"] = np.round(df["cum_area"] * np.float32(100.0)).astype(np.int32)

        df["cum_u_in"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
        df["cum_u_in_q"] = np.round(df["cum_u_in"] * np.float32(10.0)).astype(np.int32)

    train_insp["u_in_x"] = train_insp["u_in"].astype(np.float32)
    test["u_in_x"] = test["u_in"].astype(np.float32)
    train_insp["u_in_lag1_x"] = train_insp["u_in_lag1"].astype(np.float32)
    test["u_in_lag1_x"] = test["u_in_lag1"].astype(np.float32)

    key1 = ["R", "C", "u_out", "t_idx", "u_in_x", "u_in_lag1_x", "cum_area_q"]
    lookup1 = (
        train_insp.groupby(key1, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p1"})
    )
    test2 = test.merge(lookup1, on=key1, how="left")

    key2 = ["R", "C", "u_out", "u_in_x", "u_in_lag1_x", "cum_area_q"]
    lookup2 = (
        train_insp.groupby(key2, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p2"})
    )
    test2 = test2.merge(lookup2, on=key2, how="left")

    key3 = ["R", "C", "u_out", "t_idx", "u_in_x"]
    lookup3 = (
        train_insp.groupby(key3, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p3"})
    )
    test2 = test2.merge(lookup3, on=key3, how="left")

    key4 = ["R", "C", "u_out", "u_in_x"]
    lookup4 = (
        train_insp.groupby(key4, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p4"})
    )
    test2 = test2.merge(lookup4, on=key4, how="left")

    key5 = ["R", "C", "u_out", "t_idx"]
    lookup5 = (
        train_insp.groupby(key5, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p5"})
    )
    test2 = test2.merge(lookup5, on=key5, how="left")

    key6 = ["R", "C", "u_out"]
    lookup6 = (
        train_insp.groupby(key6, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p6"})
    )
    test2 = test2.merge(lookup6, on=key6, how="left")

    key7 = ["R", "C", "t_idx"]
    lookup7 = (
        train_insp.groupby(key7, sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p7"})
    )
    test2 = test2.merge(lookup7, on=key7, how="left")

    gcols = ["R", "C", "t_idx"]
    grp = train_insp.groupby(gcols, sort=False)

    stats = grp.agg(
        x_mean=("cum_area", "mean"),
        y_mean=("pressure", "mean"),
        xx_mean=(
            "cum_area",
            lambda s: np.mean(np.square(s.to_numpy(dtype=np.float32))),
        ),
        n=("pressure", "size"),
    ).reset_index()

    tmp = train_insp[gcols + ["cum_area", "pressure"]].copy()
    tmp["x_y"] = (
        tmp["cum_area"].to_numpy(dtype=np.float32)
        * tmp["pressure"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    xy = (
        tmp.groupby(gcols, sort=False)["x_y"]
        .mean()
        .reset_index()
        .rename(columns={"x_y": "xy_mean"})
    )
    stats = stats.merge(xy, on=gcols, how="left")

    var_x = (
        stats["xx_mean"].to_numpy(dtype=np.float32)
        - np.square(stats["x_mean"].to_numpy(dtype=np.float32))
    ).astype(np.float32)
    cov_xy = (
        stats["xy_mean"].to_numpy(dtype=np.float32)
        - (
            stats["x_mean"].to_numpy(dtype=np.float32)
            * stats["y_mean"].to_numpy(dtype=np.float32)
        )
    ).astype(np.float32)

    eps = np.float32(1e-6)
    b = cov_xy / (var_x + eps)
    a = stats["y_mean"].to_numpy(dtype=np.float32) - b * stats["x_mean"].to_numpy(
        dtype=np.float32
    )

    coef = stats[gcols].copy()
    coef["a"] = a.astype(np.float32)
    coef["b"] = b.astype(np.float32)

    min_n = 50
    min_varx = np.float32(1e-4)
    reliable = (stats["n"].to_numpy(dtype=np.int32) >= min_n) & (var_x >= min_varx)
    coef.loc[~reliable, ["a", "b"]] = np.nan

    test2 = test2.merge(coef, on=gcols, how="left")
    test2["p_lin"] = (
        test2["a"].astype(np.float32)
        + test2["b"].astype(np.float32) * test2["cum_area"].astype(np.float32)
    ).astype(np.float32)
    test2.loc[test2["a"].isna() | test2["b"].isna(), "p_lin"] = np.nan

    global_med = np.float32(train_insp["pressure"].median())

    pred = test2["p1"].astype(np.float32)
    pred = pred.fillna(test2["p2"].astype(np.float32))
    pred = pred.fillna(test2["p3"].astype(np.float32))
    pred = pred.fillna(test2["p4"].astype(np.float32))
    pred = pred.fillna(test2["p5"].astype(np.float32))
    pred = pred.fillna(test2["p6"].astype(np.float32))
    pred = pred.fillna(test2["p7"].astype(np.float32))
    pred = pred.fillna(test2["p_lin"].astype(np.float32))
    pred = pred.fillna(global_med).to_numpy(dtype=np.float32)

    pred_snapped = _snap_to_pressure_grid(pred, pressure_grid)

    pred_snapped = _apply_u_out_expiration_carry(test, pred_snapped)
    pred_snapped = _snap_to_pressure_grid(pred_snapped, pressure_grid)

    out = test2[["id"]].copy()
    out["pressure"] = pred_snapped
    out = out.sort_values("id").reset_index(drop=True)

    out = sample_sub_sorted[["id"]].merge(out, on="id", how="left")
    out["pressure"] = out["pressure"].fillna(np.float32(0.0)).astype(np.float32)
    return out, pressure_grid, test


sub_sorted = sub.sort_values("id").reset_index(drop=True)

fallback_sub, pressure_grid, test_df_full = build_lookup_fallback_submission(sub_sorted)

sub_1 = load_or_fallback(blend_paths["sub_1"], fallback_sub)
sub_2 = load_or_fallback(blend_paths["sub_2"], fallback_sub)
sub_3 = load_or_fallback(blend_paths["sub_3"], fallback_sub)
sub_4 = load_or_fallback(blend_paths["sub_4"], fallback_sub)


def align_to_ids(df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    df = df[["id", "pressure"]].copy()
    df = ids.to_frame("id").merge(df, on="id", how="left")
    df["pressure"] = df["pressure"].fillna(0.0)
    return df


ids = sub_sorted["id"]
sub_1 = align_to_ids(sub_1, ids)
sub_2 = align_to_ids(sub_2, ids)
sub_3 = align_to_ids(sub_3, ids)
sub_4 = align_to_ids(sub_4, ids)

n = len(sub_sorted)
for name, df in [
    ("sub_1", sub_1),
    ("sub_2", sub_2),
    ("sub_3", sub_3),
    ("sub_4", sub_4),
]:
    if len(df) != n:
        raise ValueError(f"{name} length {len(df)} != sample_submission length {n}")



## === cell 2
sub_out = sub_sorted[["id"]].copy()
blend_pred = (
    (sub_1["pressure"].to_numpy(dtype=np.float32) * np.float32(0.2))
    + (sub_2["pressure"].to_numpy(dtype=np.float32) * np.float32(0.6))
    + (sub_3["pressure"].to_numpy(dtype=np.float32) * np.float32(0.1))
    + (sub_4["pressure"].to_numpy(dtype=np.float32) * np.float32(0.1))
).astype(np.float32)

blend_snapped = _snap_to_pressure_grid(blend_pred, pressure_grid)

blend_snapped = _apply_u_out_expiration_carry(
    test_df_full.sort_values("id").reset_index(drop=True), blend_snapped
)
blend_snapped = _snap_to_pressure_grid(blend_snapped, pressure_grid)

sub_out["pressure"] = blend_snapped
sub_out.to_csv("submission.csv", index=False)

sub_out.head(5)
