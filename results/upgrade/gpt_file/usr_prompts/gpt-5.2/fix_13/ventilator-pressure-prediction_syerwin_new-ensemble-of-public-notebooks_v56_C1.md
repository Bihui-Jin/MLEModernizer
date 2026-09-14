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

0.1439299956400274

# 6. Current score

2.13309

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.32522) has done: 'The runtime error comes from trying to read external Kaggle Dataset paths that don’t exist in your provided environment (`../input/vpp-a-basic-ensembling-technique/...`, etc.). To keep the core “blend submissions” logic intact while making the notebook run end-to-end, I add a small fallback that, when those files aren’t present, generates a reasonable baseline prediction directly from `train.csv`/`test.csv` by matching each test row to the mean pressure for the same `(R, C, time_step, u_in, u_out)` combination in train. This preserves the intent (produce a submission via combining sources), fixes the FileNotFound/NameError, and should yield a non-trivial score instead of failing to produce a CSV. It always write `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 3.78746) has done: 'Your current fallback is effectively predicting a near-constant global mean (because exact matching on floating `time_step` and continuous `u_in` almost never hits), which explains the very poor MAE. To move the score toward the target with minimal logic change, I keep the same “generate a fallback submission when external blend files are missing” approach but make the fallback matching realistic by using per-breath time index (`time_step` rounded + within-breath step) and binning `u_in` slightly so many more rows can match. I also add a second-stage hierarchical fill (drop `u_in`, then drop `step`) before falling back to the global mean, which improves coverage without changing the overall approach. The blending cell remains intact and still writes a valid `submission.csv`.'
- What this solution (achieved 6.08126) has done: 'We need to move the MAE down from 3.78746 toward 0.14393, so the fallback must become much more informative while keeping the same “generate fallback submissions then blend” core logic. The biggest win with minimal semantic change is to match on exact within-breath timestep (the data is fixed-length per breath) and to use a stronger, still-simple hierarchical mean encoding that keys on `(R,C,u_out,step,u_in)` without rounding/bucketing that destroys signal. I keep your blending cell intact, but I (1) stop rounding `time_step` and (2) replace the fragile `time_step_r`/`u_in_b` first key with an exact `step+u_in` key and then progressively back off to coarser keys for missing rows. This keeps runtime reasonable (single pass groupbys + merges), produces a valid `submission.csv`, and should significantly reduce MAE toward the target band.'
- What this solution (achieved 4.12219) has done: 'Your fallback is still too “exact-match” on continuous `u_in`, so most rows miss the first, most-informative key and fall back to coarse/global means, which keeps MAE very high. To move the score down toward the target with minimal core-logic change, I keep the same hierarchical mean-merge approach but discretize `u_in` slightly (small bin size) and add a lightweight “last_pressure” feature (previous timestep pressure within the same `(R,C,u_out,step)` group) to make the mapping much closer to the real dynamics without changing training loops or adding a model. I also ensure predictions are only scored on inspiratory phase by learning only from `u_out==0` rows for the primary keys and then backing off as before, which aligns better with the metric. Blending remains identical and the script still always writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.83223) has done: 'Your current fallback is still dominated by coarse mean-fills because even with small binning, matching on binned `u_in` alone doesn’t capture the strong per-breath dynamics, which keeps MAE far from the 0.1439 target. To move the score down (lower is better) with minimal logic change, I keep the same “hierarchical mean merge fallback then blend” structure but (1) add a simple within-breath cumulative integral feature `u_in_cum` (key for ventilator physics) computed per step, and (2) include it (binned) in the most-informative keys so many more test rows map to the right pressure regime. I also fix the “previous pressure” mapping bug (it was joining current `step` to `prev_step` incorrectly), so the backoff fill actually uses the prior timestep statistics as intended. The blending and output format stay identical and it still always writes a valid `submission.csv`.'
- What this solution (achieved 8.62665) has done: 'Your current fallback is still essentially a coarse mean-encoding, which struggles to approximate the discrete pressure levels and strong within-breath dynamics, keeping MAE far above the 0.1439 target. To move the score down toward the target while preserving the same “fallback submission(s) then blend” core logic, I add a minimal, competition-standard post-processing step: snap predictions to the nearest valid pressure value observed in the training data (this aligns predictions with the discrete target). I also ensure the fallback training aggregation learns from inspiratory rows but does not force `u_out` into the key when the training subset already fixes it, improving match coverage without changing the approach. The blend and output format remain identical and it still always writes a valid `submission.csv`.'
- What this solution (achieved 2.14229) has done: 'Your current score (8.62665, lower-is-better) is far worse than the target (0.14393), so we should improve the fallback predictions substantially while keeping the same overall “generate fallback submissions then blend” logic. The biggest issue is that the fallback is still essentially a mean-encoding, which cannot capture the discrete, highly structured pressure dynamics; a minimal but much stronger (and competition-standard) upgrade is to switch the fallback to a nearest-neighbor lookup over `(R,C,step,u_out,u_in,u_in_cum)` using a fast sorted-key search. This preserves the non-ML, no-training-loop approach and keeps runtime reasonable, while drastically improving match quality. We keep the existing “snap to valid pressure levels” post-processing, and we keep the blending cell semantics intact (still produces `submission.csv` with `id,pressure`).'
- What this solution (achieved 2.17868) has done: 'We need to reduce MAE from 2.14229 toward the 0.14393 target (lower is better), so the fallback must become more accurate while keeping the same “generate fallback submissions then blend” structure. Your current nearest-neighbor search is over an integer-composed key, which makes “closeness” meaningless and can return very wrong matches; we keep the LUT idea but change it to an exact hash-join first and only then do a sensible nearest lookup within the same `(R,C,u_out,step)` using sorted `u_in_cum_b` (a physically meaningful axis). We also fix the blend weights so the fallback actually passes through unchanged (currently you multiply by 0.658, unintentionally shrinking predictions), and we keep the “snap to valid pressure levels” post-processing to better align with the discrete target. All paths remain the same and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.7716) has done: 'Your current fallback relies heavily on binned `(u_in, u_in_cum)` lookups, but pressure in this competition is strongly tied to a *small discrete set* of target values and to within-breath dynamics; we can improve accuracy without changing the “fallback LUT + optional blend + snap-to-grid” core by (1) learning the snapping grid only from inspiratory-phase pressures (the scored phase) and (2) adding one more minimal backoff fill keyed on `(R,C,step,u_out,u_in_b)` when the primary key misses (this recovers many cases where `u_in_cum_b` binning causes unnecessary NaNs). These are small, deterministic additions that keep your structure intact and should reduce MAE (lower is better) toward the 0.1439 target. The submission path/format stays the same and it still produces `submission.csv` end-to-end within the time limit.'
- What this solution (achieved 2.33159) has done: 'We need to reduce MAE from 2.7716 toward the 0.14393 target (lower is better), so the fallback must become much more faithful to the known structure of this competition while keeping your “fallback LUT + snap-to-grid + (optional) blend” core intact. The smallest high-impact change is to switch the lookup target from mean pressure to the most frequent pressure (mode) per key, because pressures are discrete and mode better matches the evaluation distribution than averaging (which creates off-grid values later snapped). Next, we make the primary key slightly more robust by using a tiny `u_in` rounding (still deterministic, not a model change) and add one additional backoff that uses `(R,C,u_out,step,u_in_cum_b)` when `(…,u_in_b,…)` misses. We keep your nearest-neighbor within-group fill, but make it operate on the same “mode LUT” so it returns valid discrete-like values even before snapping, improving MAE without changing overall semantics. Output remains a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.33159) has done: 'You’re currently far worse than the target (MAE 2.33 vs 0.144, lower is better), so we should improve the fallback mapping while keeping your exact “LUT fallback + optional blend + snap-to-grid” structure. The minimal high-impact fix is to train the LUT only on the *scored phase* (`u_out==0`) and then explicitly set all `u_out==1` test predictions to a neutral value (e.g., 0) since those rows are not scored; this usually reduces error propagation without changing evaluation semantics. I also add one extra backoff key that drops `u_out` for inspiratory rows (since it’s constant there) to increase hit-rate, and keep your existing nearest-neighbor fill and snapping unchanged. These changes are deterministic, keep runtime manageable, and should move MAE downward toward the target band.'
- What this solution (achieved 2.13309) has done: 'To move MAE down toward the 0.14393 target (lower is better) without changing your overall “fallback LUT + optional blend + snap-to-grid” structure, I’m making three minimal, metric-aligned fixes. First, I stop forcing `u_out==1` predictions to 0.0 because Kaggle’s MAE is computed only on *true* inspiratory phase (`u_out==0` in the ground truth), and guessing 0 for expiratory rows can still hurt when the hidden scoring mask differs from our assumption; instead we just leave the LUT-based prediction. Second, I change the LUT aggregation from pure mode to a tiny “mode-then-median” tie-breaker per key, which preserves your discrete-pressure intent but is more stable than arbitrary tie selection. Third, I keep the snapping-to-valid-pressure post-process but snap using the full training pressure grid (not only inspiratory), which avoids off-grid bias if expiratory pressures share valid levels used in evaluation splits. These are deterministic, keep runtime reasonable, and should reduce MAE from ~2.33 toward the target band.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

paths = {
    "sub_1": "../input/vpp-a-basic-ensembling-technique/submission_pp.csv",
    "sub_2": "../input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    "sub_3": "../input/gb-vpp-whoppity-dub-dub/median_submission.csv",
    "sub_4": "../input/ensemble-without-overfitting-risk/submission_median.csv",
}

loaded = {}
missing = []
for k, p in paths.items():
    if os.path.exists(p):
        loaded[k] = pd.read_csv(p)
    else:
        missing.append(p)

if missing:
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
        dtype={
            "breath_id": np.int32,
            "R": np.int16,
            "C": np.int16,
            "u_out": np.int8,
            "time_step": np.float32,
            "u_in": np.float32,
            "pressure": np.float32,
        },
    )
    test = pd.read_csv(
        test_path,
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
        dtype={
            "id": np.int32,
            "breath_id": np.int32,
            "R": np.int16,
            "C": np.int16,
            "u_out": np.int8,
            "time_step": np.float32,
            "u_in": np.float32,
        },
    )

    train["step"] = train.groupby("breath_id").cumcount().astype(np.int16)
    test["step"] = test.groupby("breath_id").cumcount().astype(np.int16)

    train = train.sort_values(["breath_id", "step"]).reset_index(drop=True)
    test = test.sort_values(["breath_id", "step"]).reset_index(drop=True)

    dt_train = (
        train.groupby("breath_id")["time_step"]
        .diff()
        .fillna(train["time_step"])
        .astype(np.float32)
    )
    dt_test = (
        test.groupby("breath_id")["time_step"]
        .diff()
        .fillna(test["time_step"])
        .astype(np.float32)
    )
    train["u_in_cum"] = (
        (train["u_in"] * dt_train)
        .groupby(train["breath_id"])
        .cumsum()
        .astype(np.float32)
    )
    test["u_in_cum"] = (
        (test["u_in"] * dt_test).groupby(test["breath_id"]).cumsum().astype(np.float32)
    )

    UIN_BIN = np.float32(0.1)
    UINCUM_BIN = np.float32(0.06)
    train["u_in_b"] = (np.round(train["u_in"] / UIN_BIN) * UIN_BIN).astype(np.float32)
    test["u_in_b"] = (np.round(test["u_in"] / UIN_BIN) * UIN_BIN).astype(np.float32)
    train["u_in_cum_b"] = (
        np.round(train["u_in_cum"] / UINCUM_BIN) * UINCUM_BIN
    ).astype(np.float32)
    test["u_in_cum_b"] = (np.round(test["u_in_cum"] / UINCUM_BIN) * UINCUM_BIN).astype(
        np.float32
    )

    global_mean = float(train["pressure"].mean())
    insp_mean = float(train.loc[train["u_out"] == 0, "pressure"].mean())

    def build_mode_lut(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
        tmp = (
            df.groupby(key_cols + ["pressure"], sort=False, observed=True)
            .size()
            .rename("cnt")
            .reset_index()
        )
        med = (
            df.groupby(key_cols, sort=False, observed=True)["pressure"]
            .median()
            .rename("p_med")
            .reset_index()
        )
        tmp = tmp.merge(med, on=key_cols, how="left")
        tmp["tie_dist"] = (tmp["pressure"] - tmp["p_med"]).abs().astype(np.float32)

        tmp = tmp.sort_values(
            key_cols + ["cnt", "tie_dist"],
            ascending=[True] * len(key_cols) + [False, True],
        )
        return tmp.drop_duplicates(subset=key_cols, keep="first")[
            key_cols + ["pressure"]
        ].reset_index(drop=True)

    train_insp = train.loc[train["u_out"] == 0].copy()

    key_cols = ["R", "C", "u_out", "step", "u_in_b", "u_in_cum_b"]
    train_lut = build_mode_lut(train_insp, key_cols)

    merged = test[["id", "R", "C", "u_out", "step", "u_in_b", "u_in_cum_b"]].copy()
    merged = merged.merge(train_lut, on=key_cols, how="left")

    miss_mask = merged["pressure"].isna().to_numpy()
    if miss_mask.any():
        key_cols_2a = ["R", "C", "u_out", "step", "u_in_cum_b"]
        train_lut_ucum = build_mode_lut(train_insp, key_cols_2a)
        merged2a = (
            merged.loc[miss_mask, key_cols_2a]
            .merge(train_lut_ucum, on=key_cols_2a, how="left")["pressure"]
            .to_numpy(np.float32)
        )
        merged.loc[miss_mask, "pressure"] = merged2a

    miss_mask = merged["pressure"].isna().to_numpy()
    if miss_mask.any():
        key_cols_2 = ["R", "C", "u_out", "step", "u_in_b"]
        train_lut_uin = build_mode_lut(train_insp, key_cols_2)
        merged2 = (
            merged.loc[miss_mask, key_cols_2]
            .merge(train_lut_uin, on=key_cols_2, how="left")["pressure"]
            .to_numpy(np.float32)
        )
        merged.loc[miss_mask, "pressure"] = merged2

    miss_mask = merged["pressure"].isna().to_numpy()
    if miss_mask.any():
        miss_insp = miss_mask & (merged["u_out"].to_numpy() == 0)
        if miss_insp.any():
            key_cols_2b = ["R", "C", "step", "u_in_b", "u_in_cum_b"]
            train_lut_2b = build_mode_lut(train_insp, key_cols_2b)
            merged2b = (
                merged.loc[miss_insp, key_cols_2b]
                .merge(train_lut_2b, on=key_cols_2b, how="left")["pressure"]
                .to_numpy(np.float32)
            )
            merged.loc[miss_insp, "pressure"] = merged2b

    miss_mask = merged["pressure"].isna().to_numpy()
    if miss_mask.any():
        base_cols = ["R", "C", "u_out", "step"]
        lut2 = train_lut[base_cols + ["u_in_cum_b", "pressure"]].copy()
        lut2 = lut2.sort_values(base_cols + ["u_in_cum_b"]).reset_index(drop=True)

        grp_keys = lut2[base_cols].to_numpy()
        change = np.ones(len(lut2), dtype=bool)
        change[1:] = (grp_keys[1:] != grp_keys[:-1]).any(axis=1)
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(lut2)]

        group_to_slice = {}
        for s, e in zip(starts, ends):
            k = tuple(grp_keys[s])
            group_to_slice[k] = (s, e)

        ucb_arr = lut2["u_in_cum_b"].to_numpy(np.float32)
        p_arr = lut2["pressure"].to_numpy(np.float32)

        miss_idx = np.flatnonzero(miss_mask)
        filled = merged["pressure"].to_numpy(np.float32)
        for i in miss_idx:
            k = (
                int(merged.at[i, "R"]),
                int(merged.at[i, "C"]),
                int(merged.at[i, "u_out"]),
                int(merged.at[i, "step"]),
            )
            sl = group_to_slice.get(k, None)
            if sl is None:
                continue
            s, e = sl
            target = np.float32(merged.at[i, "u_in_cum_b"])
            pos = np.searchsorted(ucb_arr[s:e], target, side="left")
            pos0 = max(pos - 1, 0)
            pos1 = min(pos, (e - s) - 1)
            v0 = ucb_arr[s + pos0]
            v1 = ucb_arr[s + pos1]
            if abs(float(target - v1)) < abs(float(target - v0)):
                filled[i] = p_arr[s + pos1]
            else:
                filled[i] = p_arr[s + pos0]
        merged["pressure"] = filled

    merged["pressure"] = merged["pressure"].astype(np.float32)
    merged.loc[merged["u_out"] == 0, "pressure"] = merged.loc[
        merged["u_out"] == 0, "pressure"
    ].fillna(np.float32(insp_mean))
    merged["pressure"] = (
        merged["pressure"].fillna(np.float32(global_mean)).astype(np.float32)
    )

    merged = merged.sort_values("id").reset_index(drop=True)
    fallback = sub[["id"]].merge(merged[["id", "pressure"]], on="id", how="left")
    fallback["pressure"] = (
        fallback["pressure"]
        .fillna(np.float32(insp_mean))
        .fillna(np.float32(global_mean))
        .astype(np.float32)
    )


    pressure_values = np.sort(train["pressure"].unique()).astype(np.float32)
    pred = fallback["pressure"].to_numpy(np.float32)
    idx = np.searchsorted(pressure_values, pred, side="left")
    idx = np.clip(idx, 1, len(pressure_values) - 1)
    left = pressure_values[idx - 1]
    right = pressure_values[idx]
    snapped = np.where(pred - left <= right - pred, left, right).astype(np.float32)
    fallback["pressure"] = snapped

    sub_1 = fallback.copy()
    sub_2 = fallback.copy()
    sub_3 = fallback.copy()
    sub_4 = fallback.copy()
else:
    sub_1 = loaded["sub_1"]
    sub_2 = loaded["sub_2"]
    sub_3 = loaded["sub_3"]
    sub_4 = loaded["sub_4"]



## === cell 2
sub["pressure"] = (
    (sub_1["pressure"].values * 1.0)
    + (sub_2["pressure"].values * 0.0)
    + (sub_3["pressure"].values * 0.0)
    + (sub_4["pressure"].values * 0.0)
)

sub = sub[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)

sub.head(5)
