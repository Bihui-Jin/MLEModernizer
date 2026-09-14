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

No external packages required in the script and installed.

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

0.202392702653657

# 6. Current score

3.87686

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I remove the notebook-only `%matplotlib inline` and all TensorFlow-related imports that trigger the `MessageFactory.GetPrototype` crash, since this script only ensembles CSVs and doesn’t train a model. Then I fix the missing ensemble files by making the code automatically fall back to a single safe baseline prediction using the provided `sample_submission.csv` when the external `../input/sub-files/...` paths don’t exist. Finally, I ensure the produced `submission.csv` has the correct `id,pressure` columns, correct row count, and pressure values clipped/rounded to the valid training pressure grid so it is always a valid Kaggle submission.'
- What this solution (achieved 8.31337) has done: 'Your current score is extremely far from the target (MAE 17.65 vs 0.20, lower is better) because the script often falls back to `sample_submission.csv` (all zeros) when the external ensemble files don’t exist, which guarantees a terrible MAE. To move the score sharply toward the target while keeping changes minimal and within the competition’s rules, I replace that fallback with a legitimate “memorization” baseline: for each (R, C, time_step, u_in, u_out) combination, use the mean training pressure and apply it to matching rows in test. This preserves the overall structure (read train/test, make predictions, snap to pressure grid, write submission) and only changes the fallback prediction source. I also keep your existing pressure-grid rounding/clipping, and ensure alignment by building predictions directly in test row order.'
- What this solution (achieved 4.1221) has done: 'Your current fallback uses an exact feature lookup on continuous `u_in` and `time_step`, which almost never matches between train and test, so it degenerates to a global mean and yields a poor MAE. I keep the same overall pipeline (read train/test, build a fallback prediction only when ensemble files are missing, snap to the pressure grid, write `submission.csv`) but make the fallback lookup actually match by (1) using per‑breath `time_step` index (0–79) instead of raw float time and (2) rounding `u_in` to a fixed resolution before grouping/merging. This is a minimal change to the existing fallback logic and should move the score substantially toward the target without changing any modeling/training approach. I also add a small hierarchical backoff (exact rounded match → `(R,C,t_idx,u_out)` mean → global mean) to avoid unseen combinations.'
- What this solution (achieved 3.15529) has done: 'Your current baseline still behaves like a weak global-mean predictor because the `(t_idx, u_out, binned u_in)` lookup is too coarse to capture the pressure dynamics; the smallest meaningful improvement is to keep the same fallback “train-lookup” idea but use a stronger, still-legit key. I add two minimal features that are available in both train/test and don’t change the overall approach: `u_in_diff` (within-breath delta) and `u_in_cum` (within-breath cumulative), both lightly binned, then do a hierarchical backoff (fine → medium → coarse → global mean). I also keep your existing pressure-grid snapping and submission-writing unchanged, and I ensure feature computation is consistent across train/test to avoid merge mismatches. This should reduce MAE (lower is better) and move your score materially toward the 0.20 target without changing the script’s core “no-training, lookup baseline” logic.'
- What this solution (achieved 4.05941) has done: 'Your current score (3.155) is still far above the target (0.202; lower is better), so we should improve the fallback train-lookup predictor while keeping the same “no training, lookup + hierarchical backoff + pressure-grid snap” core logic. The minimal high-impact fix is to make the lookup key better match the competition’s inspiratory scoring by explicitly capturing valve/flow history: add light-binned `u_in_lag1`, `u_in_lag2`, and `u_out_lag1`, and (crucially) include `u_in`/`u_out` at the breath start (`u_in0`, `u_out0`) which strongly conditions the early pressure trajectory. We keep your existing features and backoff structure, simply adding one finer map level and slightly strengthening the mid level so more test rows find a close match instead of falling back to coarse/global means. Submission writing and pressure-grid rounding/clipping remain unchanged to preserve evaluation semantics and validity.'
- What this solution (achieved 3.77215) has done: 'Your current score is far worse than the target (MAE 4.06 vs 0.202, lower is better), so we should strengthen the existing fallback (train-lookup + hierarchical backoff) without changing the overall “no training, lookup then pressure-grid snap” core logic. The main low-risk gain is to make the lookup keys align better with the inspiratory dynamics by adding minimal history/physics-inspired features that are available in both train/test: cumulative exhalation indicator, within-breath cumulative u_out, and simple “state” proxies (volume-like integral of u_in and its lags). Then we add one additional finest lookup level using these features, keep your existing backoff levels, and remove a couple of ineffective/no-op merges that can misalign rows. This should increase match rates and reduce fallback-to-global-mean, moving MAE down toward the target while preserving evaluation semantics and submission format.'
- What this solution (achieved 3.77215) has done: 'I keep your “no-training, train-lookup fallback + hierarchical backoff + pressure-grid snap” core logic unchanged, but strengthen it in one targeted way: compute the “integral of u_in over time” feature without the slow/fragile `groupby.apply(...).explode()` path that can misalign rows and harm matching quality. I replace it with a vectorized within-breath cumulative sum of `u_in * delta_time`, which preserves the intended semantics but yields consistent, correctly aligned values across train/test and improves lookup hit-rate. I also add a tiny safety fix to ensure `time_step` deltas are computed deterministically per breath (using `groupby.diff()` on the column), keeping everything else (keys, backoff order, rounding/clipping, submission writing) the same. This should reduce MAE from the current 3.77 toward the 0.20 target without changing the overall approach.'
- What this solution (achieved 3.76901) has done: 'Your current score (3.77 MAE; lower is better) is still far above the target (0.202), so we need a small but meaningful upgrade to the existing “train-lookup + hierarchical backoff + pressure-grid snap” fallback without changing its core nature. The biggest low-risk gain is to make the lookup keys reflect the true state variable in this task: pressure depends heavily on delivered volume, so we add a lightly-binned `u_in_int` (∫u_in dt) **and** a lightly-binned within-breath `u_in_mean` proxy to stabilize mapping for similar breaths. We then insert one additional lookup level that uses these features (between your current “superfine” and “fine”) and keep the existing backoff order and rounding/clipping unchanged. This should increase hit-rate on informative keys and reduce fallback-to-coarse/global, moving MAE down toward the target while staying within your current approach.'
- What this solution (achieved 3.76901) has done: 'Your current MAE (3.769) is still far above the target (0.202; lower is better), so we should improve the fallback train-lookup predictor while keeping the same overall “no training, lookup + hierarchical backoff + pressure-grid snap” core logic. The most direct, minimal upgrade is to incorporate the per-breath instantaneous “flow/volume” information already implicit in the controls by adding lightly binned `u_in_dt` (u_in * Δt) and `u_in_int2` (cumulative sum of u_in_dt), then using them in a new lookup level between your current ultra and superfine levels. This increases the chance of finding a close match for inspiratory dynamics (the scored part) without changing any model architecture or introducing training. Submission writing, row alignment, and pressure-grid rounding/clipping remain unchanged.'
- What this solution (achieved 3.76901) has done: 'Your current score (3.769 MAE; lower is better) is still far above the target (0.202), so we should improve the existing fallback “train-lookup + hierarchical backoff + pressure-grid snap” without changing its overall approach. The most direct minimal fix is to align the lookup keys with what’s actually scored: inspiratory phase only, so we build the mapping using only train rows with `u_out==0` (and keep a separate exhale map for completeness), then apply the same hierarchy in test with phase-aware backoff. This preserves your feature extraction and backoff structure, but stops exhalation rows from contaminating inspiratory averages, which should reduce MAE materially. We also add one tiny stability improvement: compute `dt` within-breath as nonnegative (guard against any float artifacts) to keep integral features consistent.'
- What this solution (achieved 3.7564) has done: 'Your current score (3.769 MAE; lower is better) is still far above the target (0.202), so we should improve the fallback train-lookup predictor in the smallest way that increases exact/near-exact key matches. The main issue is that several engineered continuous features (integrals/means) are too sensitive to tiny float/bin differences, causing many test rows to miss fine maps and back off to coarse/global means. I keep the same overall “no-training, phase-aware hierarchical lookup + pressure-grid snap” logic, but (1) compute `dt` robustly per breath via `np.diff(time_step)` (less float noise than `groupby.diff()`), and (2) slightly coarsen the integral-related bins only (leave your other bins unchanged) so more rows hit the higher-quality maps instead of backing off. This should reduce MAE toward the target while preserving evaluation semantics and producing the same valid `submission.csv`.'
- What this solution (achieved 3.57321) has done: 'Your current score (3.7564 MAE; lower is better) is still far above the target (0.202), so we should modestly improve the existing phase-aware train-lookup baseline rather than changing the overall approach. The smallest high-impact change is to make the lookup keys more discriminative for inspiratory pressure by adding a lightly binned `u_in_acc` (second integral / “delivered volume-like” proxy) and using it in a new lookup level that sits just after your current best map(s). This keeps the same core logic (feature engineering → hierarchical mean lookup from train → backoff → pressure-grid snap → write submission) while increasing match quality and reducing fallback to coarse/global means. I also keep your current binning/coarsening choices intact and only add one new feature + one new map/merge in the hierarchy so runtime stays under the 600s limit.'
- What this solution (achieved 3.86537) has done: 'We keep your exact “no-training, phase-aware hierarchical train-lookup + pressure-grid snap” approach, but make one minimal change that better matches the competition’s inspiratory dynamics: add a lightly binned within-breath `u_in_slope` (Δu_in/Δt) feature and use it only in a new finest inspiratory lookup level before the current `ultra` backoff. This is directly relevant because pressure changes are strongly tied to flow changes, and your current keys don’t explicitly capture the rate-of-change of `u_in`, which causes many near-misses and backoff to coarse/global means. We compute `u_in_slope` deterministically from your existing `dt` and `u_in_diff`, bin it to avoid float noise, and add one additional map/merge (inspiratory-only) while leaving everything else (binning, hierarchy order after that, rounding/clipping, submission writing) unchanged. This should reduce MAE (lower is better) and move the score closer to 0.202 without altering the solution’s core logic.'
- What this solution (achieved 3.87686) has done: 'Your current score (3.865, lower is better) is still far above the target (0.202), so we should modestly improve the existing phase-aware hierarchical train-lookup baseline without changing the overall “no-training, lookup + backoff + pressure-grid snap” core logic. The smallest impactful fix is to stop the new `u_in_slope_bin` key from becoming noisy at very small `dt` (near-zero time deltas), which creates many mismatches and forces fallback to weaker maps. I compute slope using a robust per-breath median `dt` as the fallback denominator when `dt` is tiny, keeping everything else identical, and I also make `u_in_slope_bin` slightly coarser to increase hit-rate (only this bin, not the rest). Submission writing, column alignment, and pressure-grid snapping remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

targets = train["pressure"].to_numpy().reshape(-1, 80, 1)



## === cell 2
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = float((sorted_pressures[1] - sorted_pressures[0]).item())
PRESSURE_MIN = float(sorted_pressures[0].item())
PRESSURE_MAX = float(sorted_pressures[-1].item())



## === cell 3
Ensembles = [
    "../input/sub-files/Submission files/sub (1).csv",
    "../input/sub-files/Submission files/sub (2).csv",
    "../input/sub-files/Submission files/sub (3).csv",
    "../input/sub-files/Submission files/sub (4).csv",
    "../input/sub-files/Submission files/sub (5).csv",
    "../input/sub-files/Submission files/sub (6).csv",
    "../input/sub-files/Submission files/sub (7).csv",
    "../input/sub-files/Submission files/sub (8).csv",
    "../input/sub-files/Submission files/sub (9).csv",
    "../input/sub-files/Submission files/sub (10).csv",
    "../input/sub-files/Submission files/sub (11).csv",
    "../input/sub-files/Submission files/sub (12).csv",
    "../input/sub-files/Submission files/sub (13).csv",
    "../input/sub-files/Submission files/sub (14).csv",
    "../input/sub-files/Submission files/sub (15).csv",
    "../input/sub-files/Submission files/sub (16).csv",
    "../input/sub-files/Submission files/sub (17).csv",
    "../input/sub-files/Submission files/sub (18).csv",
    "../input/sub-files/Submission files/sub (19).csv",
    "../input/sub-files/Submission files/sub (20).csv",
    "../input/sub-files/Submission files/sub (21).csv",
    "../input/sub-files/Submission files/sub (22).csv",
]

test_preds = []
valid_files = []

for predicted in Ensembles:
    if os.path.exists(predicted):
        sub_df = pd.read_csv(predicted)
        if "pressure" not in sub_df.columns:
            continue
        test_preds.append(sub_df["pressure"].values.astype(np.float32).reshape(-1, 1))
        valid_files.append(predicted)

if len(test_preds) == 0:
    global_mean = float(train["pressure"].mean())

    def add_features(
        df: pd.DataFrame,
        uin_bin: float = 0.5,
        uin_diff_bin: float = 0.5,
        uin_cum_bin: float = 2.0,
        uin_lag_bin: float = 0.5,
        uin_int_bin: float = 1.0,
        uin_mean_bin: float = 0.5,
        uin_dt_bin: float = 0.2,
        uin_int2_bin: float = 1.0,
        uin_acc_bin: float = 5.0,
        uin_slope_bin: float = 10.0,
        slope_dt_eps: float = 0.01,
    ) -> pd.DataFrame:
        out = df.copy()
        g = out.groupby("breath_id", sort=False)

        out["t_idx"] = g.cumcount().astype(np.int16)

        uin = out["u_in"].to_numpy(np.float32)
        out["u_in_bin"] = (np.round(uin / uin_bin) * uin_bin).astype(np.float32)

        uin_prev = g["u_in"].shift(1).fillna(0.0).to_numpy(np.float32)
        uin_diff = uin - uin_prev
        out["u_in_diff_bin"] = (
            np.round(uin_diff / uin_diff_bin) * uin_diff_bin
        ).astype(np.float32)

        uin_cum = g["u_in"].cumsum().to_numpy(np.float32)
        out["u_in_cum_bin"] = (np.round(uin_cum / uin_cum_bin) * uin_cum_bin).astype(
            np.float32
        )

        uin_lag1 = g["u_in"].shift(1).fillna(0.0).to_numpy(np.float32)
        uin_lag2 = g["u_in"].shift(2).fillna(0.0).to_numpy(np.float32)
        out["u_in_lag1_bin"] = (np.round(uin_lag1 / uin_lag_bin) * uin_lag_bin).astype(
            np.float32
        )
        out["u_in_lag2_bin"] = (np.round(uin_lag2 / uin_lag_bin) * uin_lag_bin).astype(
            np.float32
        )

        uout_lag1 = g["u_out"].shift(1).fillna(0).to_numpy(np.int8)
        out["u_out_lag1"] = uout_lag1.astype(np.int8)

        out["u_in0_bin"] = (
            np.round(g["u_in"].transform("first").to_numpy(np.float32) / uin_lag_bin)
            * uin_lag_bin
        ).astype(np.float32)
        out["u_out0"] = g["u_out"].transform("first").to_numpy(np.int8).astype(np.int8)

        out["u_out_cum"] = g["u_out"].cumsum().to_numpy(np.int16).astype(np.int16)
        out["is_exhale"] = (out["u_out_cum"] > 0).astype(np.int8)

        t = out["time_step"].to_numpy(np.float32)
        dt = np.zeros(len(out), dtype=np.float32)
        dt_med = np.zeros(len(out), dtype=np.float32)
        for _, idx in g.indices.items():
            idx = np.asarray(idx)
            tt = t[idx]
            d = np.diff(tt, prepend=tt[0])
            d[0] = 0.0
            d = np.maximum(d, 0.0).astype(np.float32)
            dt[idx] = d
            pos = d[d > 0]
            med = np.median(pos).astype(np.float32) if pos.size else np.float32(0.03)
            dt_med[idx] = med

        denom = np.where(dt > slope_dt_eps, dt, dt_med).astype(np.float32)
        uin_slope = (uin_diff / denom).astype(np.float32)
        uin_slope = np.clip(uin_slope, -500.0, 500.0).astype(np.float32)
        out["u_in_slope_bin"] = (
            np.round(uin_slope / uin_slope_bin) * uin_slope_bin
        ).astype(np.float32)

        uin_int_inc = (uin * dt).astype(np.float32)
        uin_int = (
            pd.Series(uin_int_inc, index=out.index)
            .groupby(out["breath_id"], sort=False)
            .cumsum()
            .to_numpy(np.float32)
        )
        out["u_in_int_bin"] = (np.round(uin_int / uin_int_bin) * uin_int_bin).astype(
            np.float32
        )

        uin_dt = uin_int_inc
        out["u_in_dt_bin"] = (np.round(uin_dt / uin_dt_bin) * uin_dt_bin).astype(
            np.float32
        )

        uin_int2 = (
            pd.Series(uin_dt, index=out.index)
            .groupby(out["breath_id"], sort=False)
            .cumsum()
            .to_numpy(np.float32)
        )
        out["u_in_int2_bin"] = (
            np.round(uin_int2 / uin_int2_bin) * uin_int2_bin
        ).astype(np.float32)

        uin_acc = (
            pd.Series(uin_int2, index=out.index)
            .groupby(out["breath_id"], sort=False)
            .cumsum()
            .to_numpy(np.float32)
        )
        out["u_in_acc_bin"] = (np.round(uin_acc / uin_acc_bin) * uin_acc_bin).astype(
            np.float32
        )

        uin_mean = (uin_cum / (out["t_idx"].to_numpy(np.float32) + 1.0)).astype(
            np.float32
        )
        out["u_in_mean_bin"] = (
            np.round(uin_mean / uin_mean_bin) * uin_mean_bin
        ).astype(np.float32)

        return out

    train_f = add_features(
        train,
        uin_bin=0.5,
        uin_diff_bin=0.5,
        uin_cum_bin=2.0,
        uin_lag_bin=0.5,
        uin_int_bin=2.0,
        uin_mean_bin=0.5,
        uin_dt_bin=0.5,
        uin_int2_bin=2.0,
        uin_acc_bin=10.0,
        uin_slope_bin=20.0,
        slope_dt_eps=0.01,
    )
    test_f = add_features(
        test,
        uin_bin=0.5,
        uin_diff_bin=0.5,
        uin_cum_bin=2.0,
        uin_lag_bin=0.5,
        uin_int_bin=2.0,
        uin_mean_bin=0.5,
        uin_dt_bin=0.5,
        uin_int2_bin=2.0,
        uin_acc_bin=10.0,
        uin_slope_bin=20.0,
        slope_dt_eps=0.01,
    )

    key_cols_ultra_slope = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_out_lag1",
        "u_in0_bin",
        "u_out0",
        "u_in_slope_bin",
    ]

    key_cols_ultra = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_out_lag1",
        "u_in0_bin",
        "u_out0",
        "is_exhale",
        "u_out_cum",
        "u_in_int_bin",
        "u_in_mean_bin",
    ]

    key_cols_ultra_state2 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_out_lag1",
        "u_in0_bin",
        "u_out0",
        "u_in_dt_bin",
        "u_in_int2_bin",
        "u_in_int_bin",
        "u_in_mean_bin",
    ]

    key_cols_ultra_state3 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_out_lag1",
        "u_in0_bin",
        "u_out0",
        "u_in_int2_bin",
        "u_in_acc_bin",
    ]

    key_cols_superfine = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_out_lag1",
        "u_in0_bin",
        "u_out0",
    ]

    key_cols_state = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_out_lag1",
        "u_in0_bin",
        "u_out0",
        "u_in_int_bin",
        "u_in_mean_bin",
    ]

    key_cols_fine = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_diff_bin",
        "u_in_cum_bin",
        "u_in0_bin",
        "u_out0",
    ]
    key_cols_mid = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_cum_bin",
        "u_in0_bin",
    ]
    key_cols_coarse = ["R", "C", "t_idx", "u_out"]

    train_insp = train_f[train_f["u_out"] == 0].copy()
    train_exh = train_f[train_f["u_out"] == 1].copy()

    def build_map(df_src: pd.DataFrame, key_cols: list[str], pcol: str) -> pd.DataFrame:
        return (
            df_src[key_cols + ["pressure"]]
            .groupby(key_cols, sort=False)["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": pcol})
        )

    train_map_ultra_slope_insp = build_map(
        train_insp, key_cols_ultra_slope, "p_ultra_slope"
    )
    train_map_ultra_insp = build_map(train_insp, key_cols_ultra, "p_ultra")
    train_map_ultra_state2_insp = build_map(
        train_insp, key_cols_ultra_state2, "p_ultra_state2"
    )
    train_map_ultra_state3_insp = build_map(
        train_insp, key_cols_ultra_state3, "p_ultra_state3"
    )
    train_map_superfine_insp = build_map(train_insp, key_cols_superfine, "p_superfine")
    train_map_state_insp = build_map(train_insp, key_cols_state, "p_state")
    train_map_fine_insp = build_map(train_insp, key_cols_fine, "p_fine")
    train_map_mid_insp = build_map(train_insp, key_cols_mid, "p_mid")
    train_map_coarse_insp = build_map(train_insp, key_cols_coarse, "p_coarse")

    train_map_ultra_exh = build_map(train_exh, key_cols_ultra, "p_ultra")
    train_map_ultra_state2_exh = build_map(
        train_exh, key_cols_ultra_state2, "p_ultra_state2"
    )
    train_map_ultra_state3_exh = build_map(
        train_exh, key_cols_ultra_state3, "p_ultra_state3"
    )
    train_map_superfine_exh = build_map(train_exh, key_cols_superfine, "p_superfine")
    train_map_state_exh = build_map(train_exh, key_cols_state, "p_state")
    train_map_fine_exh = build_map(train_exh, key_cols_fine, "p_fine")
    train_map_mid_exh = build_map(train_exh, key_cols_mid, "p_mid")
    train_map_coarse_exh = build_map(train_exh, key_cols_coarse, "p_coarse")

    test_insp_mask = test_f["u_out"].to_numpy() == 0
    pred = np.empty((len(test_f),), dtype=np.float32)

    def predict_with_maps(mask: np.ndarray, maps: dict) -> np.ndarray:
        tf = test_f.loc[mask]

        if "ultra_slope" in maps:
            ultra_slope = tf[key_cols_ultra_slope].merge(
                maps["ultra_slope"], on=key_cols_ultra_slope, how="left", sort=False
            )["p_ultra_slope"]
        else:
            ultra_slope = None

        ultra = tf[key_cols_ultra].merge(
            maps["ultra"], on=key_cols_ultra, how="left", sort=False
        )["p_ultra"]
        ultra_state2 = tf[key_cols_ultra_state2].merge(
            maps["ultra_state2"], on=key_cols_ultra_state2, how="left", sort=False
        )["p_ultra_state2"]
        ultra_state3 = tf[key_cols_ultra_state3].merge(
            maps["ultra_state3"], on=key_cols_ultra_state3, how="left", sort=False
        )["p_ultra_state3"]
        superfine = tf[key_cols_superfine].merge(
            maps["superfine"], on=key_cols_superfine, how="left", sort=False
        )["p_superfine"]
        state = tf[key_cols_state].merge(
            maps["state"], on=key_cols_state, how="left", sort=False
        )["p_state"]
        fine = tf[key_cols_fine].merge(
            maps["fine"], on=key_cols_fine, how="left", sort=False
        )["p_fine"]
        mid = tf[key_cols_mid].merge(
            maps["mid"], on=key_cols_mid, how="left", sort=False
        )["p_mid"]
        coarse = tf[key_cols_coarse].merge(
            maps["coarse"], on=key_cols_coarse, how="left", sort=False
        )["p_coarse"]

        base = ultra_slope if ultra_slope is not None else ultra

        return (
            base.fillna(ultra)
            .fillna(ultra_state2)
            .fillna(ultra_state3)
            .fillna(superfine)
            .fillna(state)
            .fillna(fine)
            .fillna(mid)
            .fillna(coarse)
            .fillna(global_mean)
            .to_numpy(dtype=np.float32)
        )

    maps_insp = {
        "ultra_slope": train_map_ultra_slope_insp,
        "ultra": train_map_ultra_insp,
        "ultra_state2": train_map_ultra_state2_insp,
        "ultra_state3": train_map_ultra_state3_insp,
        "superfine": train_map_superfine_insp,
        "state": train_map_state_insp,
        "fine": train_map_fine_insp,
        "mid": train_map_mid_insp,
        "coarse": train_map_coarse_insp,
    }
    maps_exh = {
        "ultra": train_map_ultra_exh,
        "ultra_state2": train_map_ultra_state2_exh,
        "ultra_state3": train_map_ultra_state3_exh,
        "superfine": train_map_superfine_exh,
        "state": train_map_state_exh,
        "fine": train_map_fine_exh,
        "mid": train_map_mid_exh,
        "coarse": train_map_coarse_exh,
    }

    pred[test_insp_mask] = predict_with_maps(test_insp_mask, maps_insp)
    pred[~test_insp_mask] = predict_with_maps(~test_insp_mask, maps_exh)

    test_preds = [pred.reshape(-1, 1)]
    valid_files = [
        "train-lookup baseline (phase-aware: inspiratory-only maps) (fallback) + stabilized u_in_slope_bin"
    ]

print(f"Loaded {len(test_preds)} prediction file(s). Example source: {valid_files[0]}")



## === cell 4
predictions = np.concatenate(test_preds, axis=-1)

if predictions.shape[1] > 1:
    predictions = predictions.mean(axis=1, keepdims=True)
else:
    pass



## === cell 5
rounding_pre = (
    np.round((predictions - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX).astype(np.float32)



## === cell 6
submission_file = pd.read_csv(sample_sub_path)

if clipped_pre.shape[0] != submission_file.shape[0]:
    raise ValueError(
        f"Prediction row count ({clipped_pre.shape[0]}) does not match sample_submission rows ({submission_file.shape[0]})."
    )

submission_file["pressure"] = clipped_pre.reshape(-1)
submission_file.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission_file.head())
