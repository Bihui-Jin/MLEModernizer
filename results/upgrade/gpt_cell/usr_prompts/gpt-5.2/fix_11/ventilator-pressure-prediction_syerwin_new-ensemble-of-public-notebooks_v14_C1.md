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

0.1559

# 6. Current score

1.37947

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Diagnosis: Cell 1 crashes because it tries to read several public Kaggle dataset submissions from `../input/...`, but in this environment only `/kaggle/input/...` (and `/kaggle/data/...`) exists and those extra datasets are not present at all. The missing files cause a `FileNotFoundError` before any blending can occur.  
Patch summary: In cell 1, keep the same variable names (`sub`, `sub_1`..`sub_4`) but make the input path resolution robust: load the sample submission from any of the known available locations, and for the missing ensemble submission files fall back to using the sample submission (all-zero pressures) so cell 2 can still run unchanged. This is the smallest localized change that unblocks execution without altering later-cell interfaces.  
Updated cells: Only cell 1 is modified.  
Compatibility notes for cell k+1: Cell 2 expects `sub`, `sub_1`, `sub_2`, `sub_3`, `sub_4` DataFrames with a `pressure` column of matching length; the patch guarantees these exist and match `sub`’s row count.  
Assumptions: If the external ensemble submission CSVs are not available in the filesystem (as indicated by the error and provided file listing), using the sample submission as a deterministic fallback is acceptable to prevent the crash.'
- What this solution (achieved 6.43394) has done: 'Your current score is extremely far from the target because all four “ensemble” inputs are missing and you’re blending four all-zero sample submissions, yielding an essentially useless prediction. To move the MAE down toward the target while keeping the same high-level blending logic, I keep the exact weighted-ensemble structure but replace the missing external submissions with a single lightweight, fully-local model that uses only `train.csv`/`test.csv`. Specifically, I generate a per-time-step prediction by taking the median training `pressure` for the same `(R, C, time_step)` group (which is deterministic and fast), and use that as the fallback “submission” inputs so the existing blend produces non-zero, sensible pressures. This keeps the “ensemble of 4 submissions + weighted average” core logic intact, while making the outputs meaningfully aligned with the task and improving the score substantially toward 0.1559.'
- What this solution (achieved 9.92511) has done: 'Your score is far worse than the target (MAE 6.43 vs 0.1559, lower is better), so we need a small but meaningful improvement without changing the “blend 4 submissions with fixed weights” core logic. The biggest issue in your local fallback is that it does not use the breath dynamics and also mis-keys `id` (it uses per-breath 1..80 ids, not the global `id`), causing severe misalignment in the submission. I keep the same structure but (1) build the fallback prediction per `(R,C,step)` on a more informative feature set (include `u_in`, `u_out`) and (2) output predictions aligned to the sample submission’s global `id` order to ensure the blended arrays match the required row mapping. This should substantially reduce MAE while preserving your ensemble/blending semantics and staying within runtime limits.'
- What this solution (achieved 7.08925) has done: 'Your score is far above (worse than) the target MAE, so we should improve the fallback predictions while keeping the same “blend 4 submissions with fixed weights” logic unchanged. The biggest gain available without changing the overall approach is to replace the current per-row median lookup with a deterministic, ventilator-specific pressure “state table” mapping, exploiting the fact that `pressure` takes only ~950 discrete values and is strongly determined by `(R, C, time_step, u_in, u_out)`; we then round predictions to the nearest allowed pressure. To preserve your core blending semantics, we generate four slightly different local fallback submissions (using mean/median and different rounding for `u_in`/`time_step`) so the fixed weights still matter but all inputs are now meaningful. Finally, we ensure strict alignment to `sample_submission.id` order (by merging on `id`) to prevent any accidental row-order mismatch.'
- What this solution (achieved 7.11302) has done: 'Your MAE is still far above the target, so we should improve the *local fallback* predictions (which dominate because the external submissions are missing) while keeping the exact same “4 submissions + fixed weighted blend” logic. The most ventilator-specific, high-impact minimal fix is to add a deterministic “carry-forward pressure when `u_out==1`” postprocess on the fallback predictions, because the expiratory phase isn’t scored and in practice pressure should stay at/near the last inspiratory value for a large chunk of the tail; this typically reduces error without changing the model/blending structure. I also add a small, deterministic per-breath `time_step` index (`step`) and use it (in addition to your existing rounded features) for more stable grouping when rounding collapses different steps. Finally, I keep the required submission alignment by merging back onto `sample_submission.id` exactly as you already do.'
- What this solution (achieved 2.01846) has done: 'Your current MAE (7.113) is still far worse than the target (0.1559), so we need a small but high-impact correction inside the existing “build 4 fallback submissions → fixed weighted blend → snap to pressure levels” structure. The biggest issue is that your fallback table keys on exact-ish continuous values (`time_step_r`, `u_in_r`) which causes massive mismatch/NaNs and forces the global fill too often; instead, we keep the same grouping idea but switch to a much more stable discrete key (`step`, `u_out`) and also add a cumulative-volume state feature (`u_in` integral per breath) in both train/test before grouping. We keep your four fallbacks and weights unchanged, but make each fallback’s table lookup far denser by using (`R`,`C`,`step`,`u_out`,`u_in_cum_bin`) rather than raw rounded floats. This is a minimal local change that should substantially reduce MAE while preserving the ensemble/blend semantics and still finishing quickly.'
- What this solution (achieved 2.03195) has done: 'Your current MAE (2.01846, lower is better) is still far above the target (0.1559), so we should improve the *local fallback* while preserving your exact “4 submissions → fixed weighted blend → snap to pressure levels” core logic. The smallest high-impact fix is to make the lookup key much denser by adding a discretized instantaneous flow proxy (`u_in` binned) to your existing `(R,C,step,u_out,u_in_cum_b)` table, which reduces the number of global-fill rows that currently drive error. To keep changes minimal and stable, this only touches the fallback table construction (no new model/training loop), keeps your carry-forward-on-`u_out` postprocess, and keeps the blending weights unchanged. The submission writing remains identical and produces `submission.csv`.'
- What this solution (achieved 2.6704) has done: 'Your MAE is still far above the target (2.03 vs 0.1559, lower is better), so the smallest reliable way to move toward the target without changing the “4 fallback submissions → fixed weighted blend → snap-to-levels” core logic is to make the local fallback table match test rows more often. I keep your exact blending and snapping, but enrich the fallback grouping key with two very lightweight, deterministic ventilator state features: cumulative exhaled/inhaled switching context (`u_out` run-length within breath) and a short lag of `u_in` (previous step), both discretized to keep joins dense and fast. This should reduce the number of global-fill predictions (which drive large error) while preserving your overall approach and runtime constraints. The script still writes a valid `submission.csv` with `id,pressure` in the sample submission order.'
- What this solution (achieved 1.37947) has done: 'Your current MAE (2.6704) is still far worse than the target (0.1559, lower is better), so we should improve the *local fallback* predictions while keeping the same “4 fallbacks → fixed weighted blend → snap to discrete pressure levels → write submission.csv” core logic intact. The biggest remaining error source is lookup sparsity: the current table key includes run-length and lagged u_in, which can fragment groups and cause many test rows to fall back to a global median/mean. I keep the exact same pipeline but add a simple hierarchical backoff: try the full key first, then progressively drop the sparsity-driving columns while staying within the same overall table-lookup semantics. This increases match rate deterministically (no new model/training) and should move MAE downward toward the target while preserving the existing ensemble/blend structure.'
- What this solution (achieved 1.37947) has done: 'Your MAE is still far above the target (lower is better), so the best minimal improvement is to make the existing local fallback table match test rows more often without changing the overall “4 fallbacks → fixed weighted blend → snap-to-pressure-levels” logic. I keep your feature engineering and hierarchical backoff, but add one extra deterministic backoff level that drops `u_out_run_c` earlier (it’s a high-cardinality splitter that can over-fragment groups), and I also add an ultra-safe final backoff keyed only by `(R,C,step,u_out)` to reduce global-fill usage. Finally, I keep the same carry-forward postprocess and pressure snapping, and ensure the submission is still aligned to `sample_submission.id` exactly.'

# 9. Code solution

## === cell 0
import pandas as pd



## === cell 1
import os
import numpy as np


def _read_csv_first_existing(paths: list[str]) -> pd.DataFrame:
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


sub = _read_csv_first_existing(
    [
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/ventilator-pressure-prediction/sample_submission.csv",
    ]
)

train = _read_csv_first_existing(
    [
        "../input/ventilator-pressure-prediction/train.csv",
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "/kaggle/data/train.csv",
        "/kaggle/data/ventilator-pressure-prediction/train.csv",
    ]
)
test = _read_csv_first_existing(
    [
        "../input/ventilator-pressure-prediction/test.csv",
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/data/ventilator-pressure-prediction/test.csv",
    ]
)

PRESSURE_LEVELS = np.sort(train["pressure"].unique())


def _snap_to_pressure_levels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(PRESSURE_LEVELS, x, side="left")
    idx = np.clip(idx, 0, len(PRESSURE_LEVELS) - 1)
    left = np.maximum(idx - 1, 0)
    right = idx
    choose_right = np.abs(PRESSURE_LEVELS[right] - x) <= np.abs(
        PRESSURE_LEVELS[left] - x
    )
    out_idx = np.where(choose_right, right, left)
    return PRESSURE_LEVELS[out_idx]


def _postprocess_carry_on_uout(
    te: pd.DataFrame, pred_col: str = "pressure_pred"
) -> None:
    arr = te[pred_col].to_numpy(dtype=np.float64, copy=True)
    uout = te["u_out"].to_numpy(dtype=np.int8, copy=False)
    breath = te["breath_id"].to_numpy(copy=False)

    last_breath = -1
    last_insp = arr[0] if len(arr) else 0.0
    for i in range(len(arr)):
        b = breath[i]
        if b != last_breath:
            last_breath = b
            last_insp = arr[i]
        if uout[i] == 0:
            last_insp = arr[i]
        else:
            arr[i] = last_insp
    te[pred_col] = arr


def _add_engineered_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal ventilator-specific state features, deterministic and fast.
    """
    out = df.copy()

    out["step"] = out.groupby("breath_id").cumcount().astype(np.int16)

    dt = (
        out.groupby("breath_id")["time_step"]
        .diff()
        .fillna(out["time_step"])
        .astype(np.float64)
    )
    dt = dt.clip(lower=0.0)

    uin_eff = out["u_in"].astype(np.float64) * (
        out["u_out"].astype(np.int8) == 0
    ).astype(np.float64)
    out["u_in_cum"] = (uin_eff * dt).groupby(out["breath_id"]).cumsum()

    out["u_in_prev"] = (
        out.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float64)
    )

    uout_change = (
        out["u_out"].ne(out.groupby("breath_id")["u_out"].shift(1)).astype(np.int16)
    )
    out["u_out_run"] = uout_change.groupby(out["breath_id"]).cumsum().astype(np.int16)

    return out


def _hierarchical_merge_fill(
    te: pd.DataFrame,
    tr: pd.DataFrame,
    *,
    group_cols_full: list[str],
    agg: str,
) -> tuple[pd.Series, float]:
    """
    Change is aimed at improving score by reducing lookup misses:
    add more deterministic backoff levels (dropping sparsity-driving columns earlier)
    so fewer rows fall back to a global stat. This preserves the same table-lookup semantics.
    """
    if agg == "median":
        global_stat = float(tr["pressure"].median())
        reducer = "median"
    elif agg == "mean":
        global_stat = float(tr["pressure"].mean())
        reducer = "mean"
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    pred = pd.Series(np.nan, index=te.index, dtype=np.float64)

    backoff_keys = [
        group_cols_full,
        [c for c in group_cols_full if c != "u_in_prev_b"],
        [c for c in group_cols_full if c not in ("u_in_prev_b", "u_out_run_c")],
        [
            c
            for c in group_cols_full
            if c not in ("u_in_prev_b", "u_out_run_c", "u_in_b")
        ],
        [
            c
            for c in group_cols_full
            if c not in ("u_in_prev_b", "u_out_run_c", "u_in_b", "u_in_cum_b")
        ],
        ["R", "C", "step", "u_out"],
    ]

    seen = set()
    backoff_keys_unique = []
    for ks in backoff_keys:
        t = tuple(ks)
        if t not in seen:
            seen.add(t)
            backoff_keys_unique.append(ks)

    for cols in backoff_keys_unique:
        mask = pred.isna()
        if not mask.any():
            break

        stat = (
            tr.groupby(cols, sort=False)["pressure"]
            .agg(reducer)
            .rename("pressure_pred")
            .reset_index()
        )

        te_part = te.loc[mask, cols].merge(stat, on=cols, how="left")["pressure_pred"]
        pred.loc[mask] = te_part.to_numpy(dtype=np.float64, copy=False)

    pred = pred.fillna(global_stat).astype(np.float64)
    return pred, global_stat


def _build_local_fallback_submission(
    sample_sub: pd.DataFrame, *, uin_cum_bin: float, agg: str
) -> pd.DataFrame:
    tr = _add_engineered_state_features(train)
    te = _add_engineered_state_features(test)

    tr["u_in_cum_b"] = (tr["u_in_cum"] / float(uin_cum_bin)).round(0).astype(np.int16)
    te["u_in_cum_b"] = (te["u_in_cum"] / float(uin_cum_bin)).round(0).astype(np.int16)

    uin_bin = 2.0
    tr["u_in_b"] = (tr["u_in"] / uin_bin).round(0).astype(np.int16)
    te["u_in_b"] = (te["u_in"] / uin_bin).round(0).astype(np.int16)

    tr["u_in_prev_b"] = (tr["u_in_prev"] / uin_bin).round(0).astype(np.int16)
    te["u_in_prev_b"] = (te["u_in_prev"] / uin_bin).round(0).astype(np.int16)

    tr["u_out_run_c"] = tr["u_out_run"].clip(upper=6).astype(np.int16)
    te["u_out_run_c"] = te["u_out_run"].clip(upper=6).astype(np.int16)

    group_cols = [
        "R",
        "C",
        "step",
        "u_out",
        "u_out_run_c",
        "u_in_cum_b",
        "u_in_b",
        "u_in_prev_b",
    ]

    te["pressure_pred"], global_stat = _hierarchical_merge_fill(
        te, tr, group_cols_full=group_cols, agg=agg
    )

    _postprocess_carry_on_uout(te, pred_col="pressure_pred")
    te["pressure_pred"] = _snap_to_pressure_levels(te["pressure_pred"].values)

    if len(te) != len(sample_sub):
        raise ValueError(
            f"Row count mismatch: test has {len(te)} rows, sample_sub has {len(sample_sub)} rows."
        )

    pred_by_id = pd.DataFrame(
        {"id": te["id"].values, "pressure": te["pressure_pred"].values}
    )
    fallback = sample_sub[["id"]].merge(pred_by_id, on="id", how="left")
    if fallback["pressure"].isna().any():
        fallback["pressure"] = fallback["pressure"].fillna(global_stat)
        fallback["pressure"] = _snap_to_pressure_levels(fallback["pressure"].values)
    return fallback


def _read_or_fallback(path: str, fallback_df: pd.DataFrame) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return fallback_df.copy()


_local_fallback_1 = _build_local_fallback_submission(
    sub, uin_cum_bin=0.50, agg="median"
)
_local_fallback_2 = _build_local_fallback_submission(
    sub, uin_cum_bin=0.35, agg="median"
)
_local_fallback_3 = _build_local_fallback_submission(sub, uin_cum_bin=0.50, agg="mean")
_local_fallback_4 = _build_local_fallback_submission(sub, uin_cum_bin=0.35, agg="mean")

sub_1 = _read_or_fallback(
    "../input/ensemble-of-public-submissions/submission.csv", _local_fallback_1
)
sub_2 = _read_or_fallback(
    "../input/finetune-of-tensorflow-bidirectional-lstm/submission.csv",
    _local_fallback_2,
)
sub_3 = _read_or_fallback(
    "../input/a-dummy-approach-to-improve-your-score-postprocess/submission.csv",
    _local_fallback_3,
)
sub_4 = _read_or_fallback(
    "../input/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv",
    _local_fallback_4,
)



## === cell 2
sub["pressure"] = (
    (sub_1["pressure"].values * 0.2)
    + (sub_2["pressure"].values * 0.1)
    + (sub_3["pressure"].values * 0.2)
    + (sub_4["pressure"].values * 0.5)
)

sub["pressure"] = _snap_to_pressure_levels(sub["pressure"].values)

sub.to_csv("submission.csv", index=False)
sub.head(5)
