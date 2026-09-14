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

0.145801162854258

# 6. Current score

3.09336

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.43372) has done: 'The current notebook fails because it tries to read four external Kaggle Dataset submission files that are not present in your environment, so `sub_1`–`sub_4` never get created and the blending cell crashes. To make it run end-to-end and still produce a reasonable baseline score without changing the intended “make a submission from available files” semantics, I replace the missing inputs with a simple, leakage-free per-(R,C,time_step) median pressure lookup built from `train.csv` and applied to `test.csv`. This uses only competition-provided data paths, writes a valid `submission.csv` with the required `id,pressure` columns, and should score meaningfully better than all-zeros. I keep the code minimal and add safe fallbacks for unseen (R,C,time_step) combinations.'
- What this solution (achieved 5.90004) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve accuracy with minimal logic changes. The simplest high-impact fix is to respect the evaluation rule that expiratory phase (`u_out==1`) is not scored by forcing those predictions to a safe constant (e.g., 0), which typically reduces MAE on inspiratory-only scoring while not harming scored rows. We also keep your median lookup but make it more robust by switching the `time_step` join key to an integer “tick” (milliseconds) to avoid floating merge misses, and then map predictions directly to `test.id` to guarantee perfect alignment. This preserves your core “leakage-free median table from train applied to test” approach and only adjusts join robustness and evaluation-consistent post-processing.'
- What this solution (achieved 6.11693) has done: 'Your current score is far from the (lower-is-better) target, so we should improve the baseline while keeping the same “train median lookup applied to test” core logic. The biggest gain with minimal change is to compute the median by the full breath position (0–79) rather than `time_step` milliseconds, because each breath has exactly 80 discrete steps and this eliminates merge noise and missing matches. We also replace the slow Python list-comprehension fallback with a vectorized merge fallback, which keeps semantics identical but avoids accidental misalignment and is more robust. Finally, we keep the evaluation-consistent handling of `u_out==1` (not scored) by forcing those predictions to 0.0 and ensure `submission.csv` aligns 1:1 with `sample_submission.csv` ids.'
- What this solution (achieved 4.12931) has done: 'Your current approach (median lookup by (R,C,pos) with fallbacks) is very weak for this competition because pressure depends strongly on the control signals `u_in` and `u_out`; ignoring `u_in` is the main reason the MAE is far from the target. To move the score substantially closer while keeping the same “leakage-free median table from train applied to test” core logic, I add `u_in` as an additional grouping key by discretizing it (rounding) so train/test match robustly. I keep your existing fallbacks (drop `u_in` then global median) and retain the evaluation-consistent handling for `u_out==1` (set prediction to 0.0). This is a minimal, safe change that should sharply reduce error without changing the overall method or requiring any new packages.'
- What this solution (achieved 3.27259) has done: 'We keep your exact “leakage-free median lookup from train applied to test” approach, but make the lookup substantially more informative with a minimal change: include a discretized `u_in` history feature (`u_in_cum`) that better captures delivered volume/pressure dynamics than instantaneous `u_in` alone. This preserves the same training approach (groupby-median tables + left merges + fallbacks) and keeps your evaluation-consistent post-processing (`u_out==1` set to 0). We also keep your existing fallbacks but add a parallel fallback chain for the new feature so coverage stays high and submission alignment remains 1:1 with `sample_submission.csv` ids. This should reduce MAE materially from 4.129 while staying within the original logic style.'
- What this solution (achieved 3.81507) has done: 'Your current median-lookup baseline is limited mainly by discretization mismatch and ignoring pressure’s strong temporal dependence on recent control history. To move the (lower-is-better) MAE closer to the target with minimal logic change, I keep the same “groupby-median tables + left-merge + fallback chain” approach but (1) add a second history feature `u_in_last2_cum_bin` (cumulative u_in over the last 2 steps) to better capture local dynamics without changing the method, and (2) slightly coarsen the binning from 0.1 to 0.2 to improve train/test key match coverage (fewer unseen bins). I keep your `u_out==1 -> 0.0` post-processing and id-aligned submission mapping exactly the same, just extending the lookup/fallback chain to use the new feature first. This should improve score materially from 3.27 while staying within the same core solution style.'
- What this solution (achieved 3.28213) has done: 'Your current score (3.815) is far worse than the target (0.1458, lower-is-better), so we should improve accuracy with minimal changes while keeping your “groupby-median lookup tables + left merges + fallback chain” core logic intact. The biggest low-risk gain is to add one more simple history feature that captures short-term dynamics better than the last-2 sum: a last-4-step rolling sum of `u_in` (binned the same way), and use it as the first/most-specific lookup key before your existing chain. This preserves your exact approach (no ML model, no new loss/loops), just extends the lookup hierarchy to reduce MAE on inspiratory timesteps. I also keep your `u_out==1 -> 0.0` handling and id-aligned mapping unchanged to preserve evaluation semantics and submission validity.'
- What this solution (achieved 3.27554) has done: 'Your current approach is a leakage-free median-lookup with a fallback chain, but it’s still missing a key piece of signal: pressure depends heavily on how long the inspiratory valve has been open, not just short rolling sums. To move the MAE materially closer to the target while preserving the exact “groupby-median tables + left merges + fallbacks” core logic, I add one additional history feature: a discretized “time-in-inspiration” counter (consecutive steps with `u_out==0`) and use it as the most-specific lookup key before your existing chain. This keeps your `u_out==1 -> 0.0` post-processing and submission alignment unchanged, and it’s a minimal extension that should reduce error on the scored inspiratory phase.'
- What this solution (achieved 3.27554) has done: 'Your score is much worse than the target (lower is better), so we should improve accuracy while keeping the exact same “groupby-median lookup tables + left merges + fallback chain” core logic. The highest-impact minimal fix is to stop forcing `u_out==1` predictions to 0.0 (even though not scored, Kaggle still computes MAE on those rows in the official metric mask logic via `u_out` in the *ground truth*, so setting them to 0 can still be harmful when the corresponding true `u_out` is 0; additionally, it can break alignment assumptions if any merge/key issues occur). Instead, we predict for all rows via the same lookup chain, and only optionally set `u_out==1` to a neutral value derived from training (median pressure at `u_out==1`) to avoid extreme errors if those rows end up scored due to any mismatch. We also make the inspiration counter feature faster and safer by replacing the slow `groupby.apply(lambda ...)` with a fully vectorized computation (same semantics), which reduces risk of subtle index misalignment and improves determinism.'
- What this solution (achieved 3.23014) has done: 'Your current score (3.27554, lower-is-better) is far worse than the target, and the biggest likely cause in this “median lookup + fallbacks” approach is over-fragmentation of keys (too many high-cardinality binned history features), which creates many unseen combinations in test and forces frequent fallback to weak/global medians. To move toward the target with minimal core-logic change, I (1) slightly coarsen the binning to increase train/test key match coverage, and (2) add one very low-risk, high-signal key (`u_out`) into the lookup hierarchy so inspiratory/expiratory regimes don’t share medians. I keep the same groupby-median tables + left merges + fallback chain semantics, preserve your inspiration counter, and still write a valid `submission.csv` with `id,pressure`. I also keep a conservative handling for `u_out==1` by using the learned `u_out`-aware medians rather than forcing a single constant.'
- What this solution (achieved 3.13991) has done: 'Your current MAE (3.23014, lower-is-better) is still far from the target, so we should improve accuracy with the smallest possible change while preserving your median-lookup + fallback-chain core logic. The most direct issue is key fragmentation/unseen combos caused by the finest-grain lookup using `insp_count` and 0.25-binned rolling sums; we increase train/test match coverage by slightly coarsening only the *highest-cardinality* rolling-sum bins, while keeping the rest of your hierarchy intact. We also add one extra low-risk fallback just below your most-specific key: the same key but with `insp_count` removed (already present later, but adding it immediately after the top merge reduces reliance on weaker later fallbacks). These changes keep the same approach (groupby median tables + merges + fillna chain) and still produce a valid `submission.csv`.'
- What this solution (achieved 3.13991) has done: 'We keep your exact “groupby-median lookup tables + left merges + fallback chain” approach, but make the most-specific lookup less sparse so test rows match it more often (reducing fallback to weaker medians), which should move MAE down toward the target. Concretely, we slightly coarsen only the top-level history key `insp_count` (bin it) and use that binned version only for the most-specific median table, leaving the rest of your hierarchy unchanged. This is a minimal change that increases key match coverage without changing the overall semantics of your method. We also add a tiny safety clip to keep predictions within the observed train pressure range (prevents rare extreme mismatches from inflating MAE) while still writing a valid `submission.csv`.'
- What this solution (achieved 3.09333) has done: 'Your current gap to the target is still very large (lower-is-better), so we should improve accuracy with the smallest change that increases train/test key match coverage in the *top* lookup tables without changing your overall “median lookup + fallback chain” method. The most likely weak point is that the most-specific tables still fragment too much (especially with `u_in_last4_cum_bin_c` and `insp_count_c`), causing many test rows to miss and fall back to weaker medians. I add one additional, slightly coarser version of the last-4 rolling-sum bin and use it only for the top two lookup levels (before your existing chain), leaving everything else intact. This preserves your approach and semantics while typically reducing MAE by increasing high-quality match rate.'
- What this solution (achieved 3.09333) has done: 'We keep your exact “groupby-median lookup tables + left merges + fallback chain” approach, but fix a key mismatch that’s likely inflating MAE: your `id` values are not globally unique across the file (they restart per breath position), so `set_index("id")` collapses many rows and `map` assigns wrong predictions to most rows. The minimal, score-relevant change is to generate predictions in the same row order as `sample_submission.csv` by aligning on `id` **and** preserving original row order (via merge on `id` with a stable ordering), instead of using a non-unique index map. This should substantially reduce error (toward the target) without changing any feature engineering or median-table logic. We also add a small assertion to guarantee submission row-count and non-nullness to avoid silent format/alignment failures.'
- What this solution (achieved 3.09336) has done: 'Your current score is still far above the target (lower-is-better), so we should improve accuracy with a minimal change that preserves your exact “groupby-median lookup + fallback chain” approach. The biggest remaining easy win is to align predictions to the competition’s discrete pressure grid: in this dataset, true pressures take on a fixed set of discrete values, so snapping your continuous median predictions to the nearest observed training pressure level typically reduces MAE without changing the modeling logic. I build the sorted unique pressure levels from `train.csv` and replace `clip()` with a nearest-level projection (still using your same median lookups and fallbacks). I keep your submission alignment-by-row-order merge and add a small sanity check that the snapped values are all valid grid values.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path, usecols=["breath_id", "R", "C", "u_in", "u_out", "pressure"]
)
test = pd.read_csv(test_path, usecols=["id", "breath_id", "R", "C", "u_in", "u_out"])
sub = pd.read_csv(sample_path, usecols=["id", "pressure"])

train["pos"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
test["pos"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

BIN_SCALE = 4.0  # 1/0.25

train["u_in_bin"] = (train["u_in"] * BIN_SCALE).round().astype(np.int16)
test["u_in_bin"] = (test["u_in"] * BIN_SCALE).round().astype(np.int16)

train["u_in_cum_bin"] = (
    (train.groupby("breath_id", sort=False)["u_in"].cumsum() * BIN_SCALE)
    .round()
    .astype(np.int32)
)
test["u_in_cum_bin"] = (
    (test.groupby("breath_id", sort=False)["u_in"].cumsum() * BIN_SCALE)
    .round()
    .astype(np.int32)
)

u_in_roll2_train = (
    train.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=2, min_periods=2)
    .sum()
    .reset_index(level=0, drop=True)
    .fillna(0.0)
)
u_in_roll2_test = (
    test.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=2, min_periods=2)
    .sum()
    .reset_index(level=0, drop=True)
    .fillna(0.0)
)
train["u_in_last2_cum_bin"] = (u_in_roll2_train * BIN_SCALE).round().astype(np.int16)
test["u_in_last2_cum_bin"] = (u_in_roll2_test * BIN_SCALE).round().astype(np.int16)

u_in_roll4_train = (
    train.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=4, min_periods=2)
    .sum()
    .reset_index(level=0, drop=True)
    .fillna(0.0)
)
u_in_roll4_test = (
    test.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=4, min_periods=2)
    .sum()
    .reset_index(level=0, drop=True)
    .fillna(0.0)
)
train["u_in_last4_cum_bin"] = (u_in_roll4_train * BIN_SCALE).round().astype(np.int16)
test["u_in_last4_cum_bin"] = (u_in_roll4_test * BIN_SCALE).round().astype(np.int16)

ROLL_COARSE = 2  # 0.25 bins -> 0.5 bins by integer floor-divide
train["u_in_last2_cum_bin_c"] = (train["u_in_last2_cum_bin"] // ROLL_COARSE).astype(
    np.int16
)
test["u_in_last2_cum_bin_c"] = (test["u_in_last2_cum_bin"] // ROLL_COARSE).astype(
    np.int16
)
train["u_in_last4_cum_bin_c"] = (train["u_in_last4_cum_bin"] // ROLL_COARSE).astype(
    np.int16
)
test["u_in_last4_cum_bin_c"] = (test["u_in_last4_cum_bin"] // ROLL_COARSE).astype(
    np.int16
)

ROLL_COARSE2 = 4
train["u_in_last4_cum_bin_cc"] = (train["u_in_last4_cum_bin"] // ROLL_COARSE2).astype(
    np.int16
)
test["u_in_last4_cum_bin_cc"] = (test["u_in_last4_cum_bin"] // ROLL_COARSE2).astype(
    np.int16
)


def add_insp_count(df: pd.DataFrame) -> pd.Series:
    seg = (
        df["u_out"].eq(1).groupby(df["breath_id"], sort=False).cumsum().astype(np.int16)
    )
    cnt_in_seg = df.groupby(["breath_id", seg], sort=False).cumcount().astype(np.int16)
    return cnt_in_seg.where(df["u_out"].eq(0), 0).astype(np.int16)


train["insp_count"] = add_insp_count(train)
test["insp_count"] = add_insp_count(test)

INSP_COARSE = 2  # 1-step -> 2-step bins
train["insp_count_c"] = (train["insp_count"] // INSP_COARSE).astype(np.int16)
test["insp_count_c"] = (test["insp_count"] // INSP_COARSE).astype(np.int16)

med_rc_pos_uo_ic_ul4cc = (
    train.groupby(
        ["R", "C", "pos", "u_out", "insp_count_c", "u_in_last4_cum_bin_cc"], sort=False
    )["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_ic_ul4cc,
    on=["R", "C", "pos", "u_out", "insp_count_c", "u_in_last4_cum_bin_cc"],
    how="left",
)

med_rc_pos_uo_ul4cc = (
    train.groupby(["R", "C", "pos", "u_out", "u_in_last4_cum_bin_cc"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("pred_rc_pos_uo_ul4cc")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_ul4cc,
    on=["R", "C", "pos", "u_out", "u_in_last4_cum_bin_cc"],
    how="left",
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_uo_ul4cc"])

med_rc_pos_uo_ic_ul4c = (
    train.groupby(
        ["R", "C", "pos", "u_out", "insp_count_c", "u_in_last4_cum_bin_c"], sort=False
    )["pressure"]
    .median()
    .rename("pred_rc_pos_uo_ic_ul4c")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_ic_ul4c,
    on=["R", "C", "pos", "u_out", "insp_count_c", "u_in_last4_cum_bin_c"],
    how="left",
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_uo_ic_ul4c"])

med_rc_pos_uo_ul4c = (
    train.groupby(["R", "C", "pos", "u_out", "u_in_last4_cum_bin_c"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("pred_rc_pos_uo_ul4c")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_ul4c,
    on=["R", "C", "pos", "u_out", "u_in_last4_cum_bin_c"],
    how="left",
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_uo_ul4c"])

med_rc_pos_uo_ul2c = (
    train.groupby(["R", "C", "pos", "u_out", "u_in_last2_cum_bin_c"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("pred_rc_pos_uo_ul2c")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_ul2c,
    on=["R", "C", "pos", "u_out", "u_in_last2_cum_bin_c"],
    how="left",
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_uo_ul2c"])

med_rc_pos_uo_ucum = (
    train.groupby(["R", "C", "pos", "u_out", "u_in_cum_bin"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos_uo_ucum")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_ucum, on=["R", "C", "pos", "u_out", "u_in_cum_bin"], how="left"
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_uo_ucum"])

med_rc_pos_uo_u = (
    train.groupby(["R", "C", "pos", "u_out", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos_uo_u")
    .reset_index()
)
test = test.merge(
    med_rc_pos_uo_u, on=["R", "C", "pos", "u_out", "u_in_bin"], how="left"
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_uo_u"])

med_rc_pos_ul4c = (
    train.groupby(["R", "C", "pos", "u_in_last4_cum_bin_c"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos_ul4c")
    .reset_index()
)
test = test.merge(
    med_rc_pos_ul4c, on=["R", "C", "pos", "u_in_last4_cum_bin_c"], how="left"
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_ul4c"])

med_rc_pos_ul2c = (
    train.groupby(["R", "C", "pos", "u_in_last2_cum_bin_c"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos_ul2c")
    .reset_index()
)
test = test.merge(
    med_rc_pos_ul2c, on=["R", "C", "pos", "u_in_last2_cum_bin_c"], how="left"
)
test["pred"] = test["pred"].fillna(test["pred_rc_pos_ul2c"])

med_rc_pos_ucum = (
    train.groupby(["R", "C", "pos", "u_in_cum_bin"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos_ucum")
    .reset_index()
)
test = test.merge(med_rc_pos_ucum, on=["R", "C", "pos", "u_in_cum_bin"], how="left")
test["pred"] = test["pred"].fillna(test["pred_rc_pos_ucum"])

med_rc_pos_u = (
    train.groupby(["R", "C", "pos", "u_in_bin"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos_u")
    .reset_index()
)
test = test.merge(med_rc_pos_u, on=["R", "C", "pos", "u_in_bin"], how="left")
test["pred"] = test["pred"].fillna(test["pred_rc_pos_u"])

med_rc_pos = (
    train.groupby(["R", "C", "pos"], sort=False)["pressure"]
    .median()
    .rename("pred_rc_pos")
    .reset_index()
)
test = test.merge(med_rc_pos, on=["R", "C", "pos"], how="left")
test["pred"] = test["pred"].fillna(test["pred_rc_pos"])

med_rc = (
    train.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .rename("pred_rc")
    .reset_index()
)
test = test.merge(med_rc, on=["R", "C"], how="left")
test["pred"] = test["pred"].fillna(test["pred_rc"])

global_med = float(train["pressure"].median())
test["pred"] = test["pred"].fillna(global_med).astype(float)

pressure_levels = np.sort(train["pressure"].unique()).astype(np.float32)

pred = test["pred"].to_numpy(dtype=np.float32, copy=False)
idx = np.searchsorted(pressure_levels, pred, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_levels) - 1)

lvl_right = pressure_levels[idx]
lvl_left = pressure_levels[idx_left]
choose_left = (pred - lvl_left) <= (lvl_right - pred)
snapped = np.where(choose_left, lvl_left, lvl_right).astype(np.float32)

test["pred"] = snapped.astype(float)

sub["_row"] = np.arange(len(sub), dtype=np.int32)
sub = sub.merge(test[["id", "pred"]], on="id", how="left", sort=False)
sub = sub.sort_values("_row", kind="mergesort").drop(columns=["_row"])

assert len(sub) == 603600, f"Unexpected submission length: {len(sub)}"
assert (
    sub["pred"].notna().all()
), "Found missing predictions after merge; check join keys."
assert sub["pred"].between(float(pressure_levels[0]), float(pressure_levels[-1])).all()

sub["pressure"] = sub["pred"].astype(float)
sub = sub[["id", "pressure"]]

sub.to_csv("submission.csv", index=False)

sub.head()
