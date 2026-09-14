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

0.1437812227459371

# 6. Current score

3.45888

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.14576) has done: 'I remove the hard dependency on external “../input/*/submission*.csv” ensemble files that are not present in your environment (the root cause of the FileNotFoundError/NameError). To keep the core idea (ensembling submissions) but make it runnable end-to-end, I implement a safe fallback: if those files exist, ensemble them; otherwise generate a simple baseline prediction from the provided train/test (mean pressure per (R,C,time_step,u_out) with a global fallback). Finally, I always write a valid `submission.csv` with exactly `id,pressure` columns so Kaggle accepts it.'
- What this solution (achieved 9.91827) has done: 'Your current score (8.14576, lower-is-better) is far worse than the target (~0.1438), so we should improve substantially while keeping your core “lookup-table from train to test” fallback intact. The biggest issue is that your fallback ignores the most informative control signal `u_in` and only uses a mean; we minimally extend the same groupby-merge approach to include `u_in` and switch to the median (more robust), with a small, safe backoff hierarchy for unmatched rows. This keeps the exact same overall logic (train aggregate → merge into test → fill missing with fallback), but should move the MAE much closer to the target. The script still ensembles external submissions if they exist, otherwise produces a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.23955) has done: 'Your current fallback is a pure lookup on exact floating `time_step`/`u_in` values, which causes many unmatched rows due to tiny float representation differences, leading to widespread fallback-to-global and a very poor MAE. I keep the same “aggregate from train → merge into test → hierarchical fill” logic, but make the joins reliable by rounding `time_step` and `u_in` to fixed decimals before grouping/merging (same transformation applied to both train and test). I also ensure the submission keeps the original `sample_submission` id order (no merge that can reorder/duplicate), and clip predictions to the known discrete pressure grid from train (valid post-processing that typically reduces MAE for this competition without changing the modeling approach). These are minimal changes aimed at substantially decreasing the error toward your target.'
- What this solution (achieved 4.274) has done: 'We keep your core “train aggregate → merge into test → hierarchical fill → snap to pressure grid” logic, but make the lookup substantially more faithful to the ventilator dynamics by adding a cumulative-volume feature (`u_in_cumsum`) computed per breath and using it in the highest-priority mapping. This is still the same approach (a lookup-table built from train and merged into test), just with one extra engineered key that greatly reduces ambiguity for the same `(R,C,time_step,u_out,u_in)` across different breath trajectories. We also add a very small backoff hierarchy that includes the new key, then falls back to your existing keys, preserving stability and ensuring a complete submission. These changes are targeted to reduce MAE (lower-is-better) from 5.23955 toward 0.14378 without changing to a different modeling family.'
- What this solution (achieved 4.23404) has done: 'Your current lookup-table approach is being hurt by mixing exact-join keys (`u_in`, `time_step`, `u_in_cumsum`) that can still be slightly misaligned between train/test even after rounding, and by using a cumulative feature that is only an approximation of the true underlying “state”. To move the MAE substantially down toward the 0.1438 target while preserving the same “aggregate-from-train → merge-into-test → hierarchical fill → snap-to-pressure-grid” core logic, we add one more ventilator-consistent key: a per-breath running integral of `u_in` (area under the curve), computed deterministically from `time_step` and `u_in`. We then prioritize a mapping that uses this `u_in_area` (rounded) instead of (or before) `u_in_cumsum`, keeping your existing backoff hierarchy intact. This is a minimal extension of the same method (still just groupby-median maps and merges), but it should reduce ambiguity across breaths and improve matching fidelity. The submission writing and format stay unchanged.'
- What this solution (achieved 3.00032) has done: 'Your current score (4.23404, lower-is-better) is still far from the target (0.14378), so we should improve the fallback lookup while preserving the same core “aggregate-from-train → merge-into-test → hierarchical fill → snap-to-pressure-grid” approach. The biggest remaining issue is that exact matching on rounded `u_in_area`/`u_in_cumsum` is still too brittle; many rows won’t find a match and fall back to weak keys, inflating MAE. I keep your exact hierarchy, but add a minimal “binning” of the state keys (coarser rounded versions of `u_in_area` and `u_in_cumsum`) and insert those as intermediate backoff maps so more test rows get a physically-relevant match without changing the overall method. I also ensure merges are keyed without reordering issues by keeping `id` intact and still writing `submission.csv` as `id,pressure`.'
- What this solution (achieved 3.36527) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve the lookup’s hit-rate and reduce fallback usage while keeping the same “train aggregate → merge into test → hierarchical fill → snap to pressure grid” core logic. The most impactful minimal change is to add a per-breath discrete time index (`t_idx`) and use it (instead of float `time_step`) in the highest-priority maps, because train/test share the same 80-step structure and integer joins are much more reliable than float joins. We keep your existing rounded `time_step` joins as backoffs, but prioritize `t_idx`-based maps for both the “full + area” and “full + cumsum” keys (and their binned variants). This should substantially reduce NaNs in the strongest keys and move MAE closer to the 0.1438 target without changing the overall method.'
- What this solution (achieved 8.41701) has done: 'Your current lookup is likely failing because `id` is not globally unique (it repeats 1..80 per breath), so the final `sub.merge(pred_by_id, on="id")` creates a many-to-many join that duplicates/misaligns predictions and severely hurts MAE. I keep your entire lookup-table + hierarchical fill + snap-to-pressure-grid logic unchanged, but fix the final alignment by generating predictions in the exact same row order as `sample_submission` using a 1:1 mapping from test rows to submission rows. Concretely, we compute `pred_snapped` in the (breath_id,time_step) sorted test order (as you already do), and then assign it directly to `sub["pressure"]` after sorting `sub` by `id` (which is the canonical order). This is a minimal, semantics-preserving fix that should dramatically reduce the score toward the 0.1438 target because it removes the mismerge error without changing the model.'
- What this solution (achieved 3.36527) has done: 'Your MAE is still extremely far from the target, so we should make a minimal, high-impact correction that improves correctness without changing the core “train aggregate → merge into test → hierarchical fill → snap to pressure grid” logic. The most likely remaining issue is row-order misalignment: you currently assign `pred_snapped` in `test`’s sorted order to `sub`’s `id`-sorted order, but `id` is globally unique and already defines the required submission ordering, while `test` sorting by `(breath_id,time_step)` generally not match `id` order. I keep all feature engineering and mapping hierarchy identical, but ensure predictions are attached by `id` via a 1:1 merge (or direct reindexing) so every `id` gets its own predicted pressure without duplication. This should dramatically reduce the error toward the target while preserving evaluation semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 3.3664) has done: 'Your current MAE is far worse than the target, so we should improve the lookup hit-rate while keeping the exact same “aggregate-from-train → merge-into-test → hierarchical fill → snap-to-pressure-grid” approach. The minimal high-impact fix is to stop rounding `u_in` and instead convert it to an integer “PWM” key (`u_in_x100 = round(u_in*100)`), which makes joins exact and consistent between train/test without changing the underlying signal. We do the same for the derived state (`u_in_area`, `u_in_cumsum`) by storing integer-scaled versions and using those in the strongest maps plus the same backoff hierarchy (including your existing binned variants). This should substantially reduce NaNs in the high-priority merges and move the score down toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 3.4616) has done: 'Your current MAE is far worse than the target (lower-is-better), so we should make a small, high-impact fix that increases lookup hit-rate without changing the overall “train aggregate → merge into test → hierarchical fill → snap-to-grid → write submission” approach. The biggest remaining brittleness is that your strongest keys use cumulative integrals (`u_in_area`, `u_in_cumsum`) that depend on floating `dt`, so tiny numeric differences between train/test can still cause mismatches even after scaling/rounding. We keep your exact logic, but compute these integrals deterministically using the shared discrete step index (`t_idx`) with a fixed `dt` estimated per breath (so train and test use the same rule), which should reduce NaNs in high-priority merges and move MAE downward toward the target. Everything else (hierarchy, median maps, snapping to pressure grid, and submission format) is preserved.'
- What this solution (achieved 3.44928) has done: 'Your score is far worse than the target (lower-is-better), so we should improve correctness of the lookup keys while keeping your exact “train aggregate → merge into test → hierarchical fill → snap-to-pressure-grid → write submission” logic intact. The smallest high-impact fix is to compute `u_in_cumsum` and `u_in_area` from a truly fixed per-step `dt` derived from the known 80-step breath structure (rather than `t_end/(n_steps-1)` which varies slightly by breath and can de-sync train vs test). This makes your strongest join keys (`*_x1000`, `*_x100`) much more consistent between train and test, increasing match rate and reducing fallback usage. Everything else (maps, hierarchy order, snapping, submission writing) is preserved.'
- What this solution (achieved 3.46426) has done: 'Your current MAE (3.44928, lower-is-better) is still far above the target (~0.14378), so we should improve match-rate of the strongest lookup keys without changing your overall “train aggregate → merge into test → hierarchical fill → snap-to-pressure-grid → write submission” logic. The most likely issue is that using a hardcoded `FIXED_DT=0.03` desynchronizes the derived integrals (`u_in_cumsum`, `u_in_area`) from the original competition data’s real `time_step` spacing (which is consistent across breaths), causing many high-priority merges to miss and fall back to weaker keys. We keep the same engineered features and hierarchy, but compute `dt` deterministically from `time_step` within each breath (constant-per-breath, same rule for train and test), which should align integrals much better and reduce fallback usage. All paths, submission format, snapping-to-grid, and the ensemble-if-present behavior remain unchanged.'
- What this solution (achieved 3.45888) has done: 'Your current MAE is far worse than the target (lower-is-better), so the most likely “minimal but big” win is fixing join brittleness in the strongest lookup keys so fewer rows fall back to weak aggregates. I keep your exact lookup-table + hierarchical fill + snap-to-pressure-grid logic, but compute `dt` in a way that is identical for train and test by using the known 80-step structure: estimate a single global median `dt` from the whole training set and use that fixed `dt` for all breaths when building `u_in_cumsum` and `u_in_area`. This preserves your engineered features and merges, but removes per-breath `dt` noise that can desynchronize integrals and break exact integer-scaled joins. I also add a small safety assertion that each breath has 80 steps (if not, we fall back to the previous median-diff method), keeping runtime under the limit and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

sub = pd.read_csv(sub_path)

candidate_paths = [
    ("sub_1", "../input/private-dataset/submission (1).csv"),
    ("sub_2", "../input/vpp-a-basic-ensembling-technique/submission_pp.csv"),
    ("sub_3", "../input/basic-ensemble-of-public-notebooks/submission_mean.csv"),
]

loaded_subs = {}
for name, p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        if "pressure" in df.columns and len(df) == len(sub):
            loaded_subs[name] = df

loaded_subs.keys()



## === cell 2
if {"sub_1", "sub_2", "sub_3"}.issubset(set(loaded_subs.keys())):
    sub_1, sub_2, sub_3 = (
        loaded_subs["sub_1"],
        loaded_subs["sub_2"],
        loaded_subs["sub_3"],
    )
    sub["pressure"] = (
        (sub_1["pressure"].values * 0.670)
        + (sub_2["pressure"].values * 0.0)
        + (sub_3["pressure"].values * 0.330)
    ).astype(np.float32)
else:
    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort")

    train["t_idx"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    test["t_idx"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    def _estimate_global_dt_from_train(tr: pd.DataFrame) -> np.float32:
        dt = tr.groupby("breath_id", sort=False)["time_step"].diff()
        dt = dt[(dt > 0) & np.isfinite(dt)]
        if len(dt) == 0:
            return np.float32(0.03)
        return np.float32(np.median(dt.values.astype(np.float32)))

    global_dt = _estimate_global_dt_from_train(train)

    def _add_dt(
        df: pd.DataFrame, fixed_dt: np.float32, fallback_global_dt: np.float32
    ) -> pd.DataFrame:
        df = df.copy()
        steps = df.groupby("breath_id", sort=False).size()
        if steps.nunique() == 1 and int(steps.iloc[0]) == 80:
            df["dt"] = np.float32(fixed_dt)
            return df

        dt = df.groupby("breath_id", sort=False)["time_step"].diff().astype(np.float32)
        dt = dt.fillna(0.0)
        dt_breath = (
            dt.where(dt > 0)
            .groupby(df["breath_id"], sort=False)
            .transform("median")
            .astype(np.float32)
        )
        global_dt_fb = (
            float(np.nanmedian(dt.values[dt.values > 0]))
            if np.any(dt.values > 0)
            else float(fallback_global_dt)
        )
        dt_breath = dt_breath.fillna(np.float32(global_dt_fb))
        df["dt"] = dt_breath.astype(np.float32)
        return df

    train = _add_dt(train, fixed_dt=global_dt, fallback_global_dt=global_dt)
    test = _add_dt(test, fixed_dt=global_dt, fallback_global_dt=global_dt)

    train_u_in_f = train["u_in"].astype(np.float32)
    test_u_in_f = test["u_in"].astype(np.float32)

    train["u_in_cumsum"] = (
        (train_u_in_f * train["dt"]).groupby(train["breath_id"], sort=False).cumsum()
    )
    test["u_in_cumsum"] = (
        (test_u_in_f * test["dt"]).groupby(test["breath_id"], sort=False).cumsum()
    )

    train_u_in_prev = (
        train.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(train["u_in"])
        .astype(np.float32)
    )
    test_u_in_prev = (
        test.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(test["u_in"])
        .astype(np.float32)
    )

    train_u_in_area_inc = ((train_u_in_prev + train_u_in_f) * 0.5 * train["dt"]).astype(
        np.float32
    )
    test_u_in_area_inc = ((test_u_in_prev + test_u_in_f) * 0.5 * test["dt"]).astype(
        np.float32
    )

    train["u_in_area"] = train_u_in_area_inc.groupby(
        train["breath_id"], sort=False
    ).cumsum()
    test["u_in_area"] = test_u_in_area_inc.groupby(
        test["breath_id"], sort=False
    ).cumsum()

    train["time_step"] = train["time_step"].round(3)
    test["time_step"] = test["time_step"].round(3)

    train["u_in_x100"] = np.rint(train["u_in"].astype(np.float32) * 100.0).astype(
        np.int16
    )
    test["u_in_x100"] = np.rint(test["u_in"].astype(np.float32) * 100.0).astype(
        np.int16
    )

    train["u_in_cumsum_x100"] = np.rint(
        train["u_in_cumsum"].astype(np.float32) * 100.0
    ).astype(np.int32)
    test["u_in_cumsum_x100"] = np.rint(
        test["u_in_cumsum"].astype(np.float32) * 100.0
    ).astype(np.int32)

    train["u_in_area_x1000"] = np.rint(
        train["u_in_area"].astype(np.float32) * 1000.0
    ).astype(np.int32)
    test["u_in_area_x1000"] = np.rint(
        test["u_in_area"].astype(np.float32) * 1000.0
    ).astype(np.int32)

    train["u_in_area_b1_x100"] = np.rint(
        train["u_in_area"].astype(np.float32) * 100.0
    ).astype(np.int32)
    test["u_in_area_b1_x100"] = np.rint(
        test["u_in_area"].astype(np.float32) * 100.0
    ).astype(np.int32)

    train["u_in_area_b2_x10"] = np.rint(
        train["u_in_area"].astype(np.float32) * 10.0
    ).astype(np.int32)
    test["u_in_area_b2_x10"] = np.rint(
        test["u_in_area"].astype(np.float32) * 10.0
    ).astype(np.int32)

    train["u_in_cumsum_b1_x10"] = np.rint(
        train["u_in_cumsum"].astype(np.float32) * 10.0
    ).astype(np.int32)
    test["u_in_cumsum_b1_x10"] = np.rint(
        test["u_in_cumsum"].astype(np.float32) * 10.0
    ).astype(np.int32)

    grp_cols_full_area_idx = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_x100",
        "u_in_area_x1000",
    ]
    med_map_full_area_idx = (
        train.groupby(grp_cols_full_area_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_area_idx"})
    )
    test = test.merge(med_map_full_area_idx, on=grp_cols_full_area_idx, how="left")

    grp_cols_full_area_b1_idx = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_x100",
        "u_in_area_b1_x100",
    ]
    med_map_full_area_b1_idx = (
        train.groupby(grp_cols_full_area_b1_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_area_b1_idx"})
    )
    test = test.merge(
        med_map_full_area_b1_idx, on=grp_cols_full_area_b1_idx, how="left"
    )

    grp_cols_full_area_b2_idx = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_x100",
        "u_in_area_b2_x10",
    ]
    med_map_full_area_b2_idx = (
        train.groupby(grp_cols_full_area_b2_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_area_b2_idx"})
    )
    test = test.merge(
        med_map_full_area_b2_idx, on=grp_cols_full_area_b2_idx, how="left"
    )

    grp_cols_full_cum_idx = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_x100",
        "u_in_cumsum_x100",
    ]
    med_map_full_cum_idx = (
        train.groupby(grp_cols_full_cum_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_cum_idx"})
    )
    test = test.merge(med_map_full_cum_idx, on=grp_cols_full_cum_idx, how="left")

    grp_cols_full_cum_b1_idx = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_x100",
        "u_in_cumsum_b1_x10",
    ]
    med_map_full_cum_b1_idx = (
        train.groupby(grp_cols_full_cum_b1_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_cum_b1_idx"})
    )
    test = test.merge(med_map_full_cum_b1_idx, on=grp_cols_full_cum_b1_idx, how="left")

    grp_cols_full_idx = ["R", "C", "t_idx", "u_out", "u_in_x100"]
    med_map_full_idx = (
        train.groupby(grp_cols_full_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_idx"})
    )
    test = test.merge(med_map_full_idx, on=grp_cols_full_idx, how="left")

    grp_cols_area_no_uin_idx = ["R", "C", "t_idx", "u_out", "u_in_area_x1000"]
    med_map_area_no_uin_idx = (
        train.groupby(grp_cols_area_no_uin_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_area_no_uin_idx"})
    )
    test = test.merge(med_map_area_no_uin_idx, on=grp_cols_area_no_uin_idx, how="left")

    grp_cols_area_no_uin_b1_idx = ["R", "C", "t_idx", "u_out", "u_in_area_b1_x100"]
    med_map_area_no_uin_b1_idx = (
        train.groupby(grp_cols_area_no_uin_b1_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_area_no_uin_b1_idx"})
    )
    test = test.merge(
        med_map_area_no_uin_b1_idx, on=grp_cols_area_no_uin_b1_idx, how="left"
    )

    grp_cols_cum_no_uin_idx = ["R", "C", "t_idx", "u_out", "u_in_cumsum_x100"]
    med_map_cum_no_uin_idx = (
        train.groupby(grp_cols_cum_no_uin_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_cum_no_uin_idx"})
    )
    test = test.merge(med_map_cum_no_uin_idx, on=grp_cols_cum_no_uin_idx, how="left")

    grp_cols_cum_no_uin_b1_idx = ["R", "C", "t_idx", "u_out", "u_in_cumsum_b1_x10"]
    med_map_cum_no_uin_b1_idx = (
        train.groupby(grp_cols_cum_no_uin_b1_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_cum_no_uin_b1_idx"})
    )
    test = test.merge(
        med_map_cum_no_uin_b1_idx, on=grp_cols_cum_no_uin_b1_idx, how="left"
    )

    grp_cols_no_uin_idx = ["R", "C", "t_idx", "u_out"]
    med_map_no_uin_idx = (
        train.groupby(grp_cols_no_uin_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_no_uin_idx"})
    )
    test = test.merge(med_map_no_uin_idx, on=grp_cols_no_uin_idx, how="left")

    grp_cols_rc_idx = ["R", "C", "t_idx"]
    med_map_rc_idx = (
        train.groupby(grp_cols_rc_idx, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_rc_idx"})
    )
    test = test.merge(med_map_rc_idx, on=grp_cols_rc_idx, how="left")

    grp_cols_full_area = [
        "R",
        "C",
        "time_step",
        "u_out",
        "u_in_x100",
        "u_in_area_x1000",
    ]
    med_map_full_area = (
        train.groupby(grp_cols_full_area, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_area"})
    )
    test = test.merge(med_map_full_area, on=grp_cols_full_area, how="left")

    grp_cols_full_cum = [
        "R",
        "C",
        "time_step",
        "u_out",
        "u_in_x100",
        "u_in_cumsum_x100",
    ]
    med_map_full_cum = (
        train.groupby(grp_cols_full_cum, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_cum"})
    )
    test = test.merge(med_map_full_cum, on=grp_cols_full_cum, how="left")

    grp_cols_full = ["R", "C", "time_step", "u_out", "u_in_x100"]
    med_map_full = (
        train.groupby(grp_cols_full, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full"})
    )
    test = test.merge(med_map_full, on=grp_cols_full, how="left")

    grp_cols_full_area_b1 = [
        "R",
        "C",
        "time_step",
        "u_out",
        "u_in_x100",
        "u_in_area_b1_x100",
    ]
    med_map_full_area_b1 = (
        train.groupby(grp_cols_full_area_b1, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_area_b1"})
    )
    test = test.merge(med_map_full_area_b1, on=grp_cols_full_area_b1, how="left")

    grp_cols_full_area_b2 = [
        "R",
        "C",
        "time_step",
        "u_out",
        "u_in_x100",
        "u_in_area_b2_x10",
    ]
    med_map_full_area_b2 = (
        train.groupby(grp_cols_full_area_b2, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_area_b2"})
    )
    test = test.merge(med_map_full_area_b2, on=grp_cols_full_area_b2, how="left")

    grp_cols_full_cum_b1 = [
        "R",
        "C",
        "time_step",
        "u_out",
        "u_in_x100",
        "u_in_cumsum_b1_x10",
    ]
    med_map_full_cum_b1 = (
        train.groupby(grp_cols_full_cum_b1, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_full_cum_b1"})
    )
    test = test.merge(med_map_full_cum_b1, on=grp_cols_full_cum_b1, how="left")

    grp_cols_area_no_uin = ["R", "C", "time_step", "u_out", "u_in_area_x1000"]
    med_map_area_no_uin = (
        train.groupby(grp_cols_area_no_uin, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_area_no_uin"})
    )
    test = test.merge(med_map_area_no_uin, on=grp_cols_area_no_uin, how="left")

    grp_cols_cum_no_uin = ["R", "C", "time_step", "u_out", "u_in_cumsum_x100"]
    med_map_cum_no_uin = (
        train.groupby(grp_cols_cum_no_uin, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_cum_no_uin"})
    )
    test = test.merge(med_map_cum_no_uin, on=grp_cols_cum_no_uin, how="left")

    grp_cols_area_no_uin_b1 = ["R", "C", "time_step", "u_out", "u_in_area_b1_x100"]
    med_map_area_no_uin_b1 = (
        train.groupby(grp_cols_area_no_uin_b1, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_area_no_uin_b1"})
    )
    test = test.merge(med_map_area_no_uin_b1, on=grp_cols_area_no_uin_b1, how="left")

    grp_cols_cum_no_uin_b1 = ["R", "C", "time_step", "u_out", "u_in_cumsum_b1_x10"]
    med_map_cum_no_uin_b1 = (
        train.groupby(grp_cols_cum_no_uin_b1, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_cum_no_uin_b1"})
    )
    test = test.merge(med_map_cum_no_uin_b1, on=grp_cols_cum_no_uin_b1, how="left")

    grp_cols_no_uin = ["R", "C", "time_step", "u_out"]
    med_map_no_uin = (
        train.groupby(grp_cols_no_uin, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_no_uin"})
    )
    test = test.merge(med_map_no_uin, on=grp_cols_no_uin, how="left")

    grp_cols_rc_t = ["R", "C", "time_step"]
    med_map_rc_t = (
        train.groupby(grp_cols_rc_t, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_rc_t"})
    )
    test = test.merge(med_map_rc_t, on=grp_cols_rc_t, how="left")

    global_median = float(train["pressure"].median())

    pred = test["pressure_full_area_idx"]
    pred = pred.fillna(test["pressure_full_area_b1_idx"])
    pred = pred.fillna(test["pressure_full_area_b2_idx"])
    pred = pred.fillna(test["pressure_full_cum_idx"])
    pred = pred.fillna(test["pressure_full_cum_b1_idx"])
    pred = pred.fillna(test["pressure_full_idx"])
    pred = pred.fillna(test["pressure_area_no_uin_idx"])
    pred = pred.fillna(test["pressure_area_no_uin_b1_idx"])
    pred = pred.fillna(test["pressure_cum_no_uin_idx"])
    pred = pred.fillna(test["pressure_cum_no_uin_b1_idx"])
    pred = pred.fillna(test["pressure_no_uin_idx"])
    pred = pred.fillna(test["pressure_rc_idx"])

    pred = pred.fillna(test["pressure_full_area"])
    pred = pred.fillna(test["pressure_full_area_b1"])
    pred = pred.fillna(test["pressure_full_area_b2"])
    pred = pred.fillna(test["pressure_full_cum"])
    pred = pred.fillna(test["pressure_full_cum_b1"])
    pred = pred.fillna(test["pressure_full"])
    pred = pred.fillna(test["pressure_area_no_uin"])
    pred = pred.fillna(test["pressure_area_no_uin_b1"])
    pred = pred.fillna(test["pressure_cum_no_uin"])
    pred = pred.fillna(test["pressure_cum_no_uin_b1"])
    pred = pred.fillna(test["pressure_no_uin"])
    pred = pred.fillna(test["pressure_rc_t"])
    pred = pred.fillna(global_median).astype(np.float32)

    pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)
    pred_vals = pred.values.astype(np.float32)
    idx = np.searchsorted(pressure_grid, pred_vals, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    left = pressure_grid[np.clip(idx - 1, 0, len(pressure_grid) - 1)]
    right = pressure_grid[idx]
    choose_right = np.abs(right - pred_vals) <= np.abs(pred_vals - left)
    pred_snapped = np.where(choose_right, right, left).astype(np.float32)

    pred_by_id = pd.DataFrame({"id": test["id"].values, "pressure": pred_snapped})
    sub = sub.merge(
        pred_by_id, on="id", how="left", validate="1:1", suffixes=("", "_pred")
    )
    if sub["pressure_pred"].isna().any():
        gm = np.float32(global_median)
        gm_idx = np.searchsorted(pressure_grid, gm, side="left")
        gm_idx = int(np.clip(gm_idx, 0, len(pressure_grid) - 1))
        gm_left = pressure_grid[max(gm_idx - 1, 0)]
        gm_right = pressure_grid[gm_idx]
        gm_snapped = gm_right if abs(gm_right - gm) <= abs(gm - gm_left) else gm_left
        sub["pressure_pred"] = sub["pressure_pred"].fillna(np.float32(gm_snapped))
    sub["pressure"] = sub["pressure_pred"].astype(np.float32)
    sub = sub.drop(columns=["pressure_pred"])



## === cell 3
sub = sub[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)
sub.head()
