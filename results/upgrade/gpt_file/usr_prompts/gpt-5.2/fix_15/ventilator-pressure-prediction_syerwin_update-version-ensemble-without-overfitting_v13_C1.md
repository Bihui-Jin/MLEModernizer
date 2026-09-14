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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.1383225486603057

# 6. Current score

1.58386

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.95924) has done: 'I remove the dependency on missing external Kaggle datasets (the public-notebook submissions) by replacing that ensemble input with a simple, fully local baseline prediction derived from the provided training data, so the notebook runs end-to-end. I keep your feature-engineering and scaling steps intact (they were not being used for prediction anyway) and only fix the runtime blockers (missing files/undefined `pred`) plus the final submission creation. To nudge score toward the target (lower is better) without changing model architecture/training (none exists here), I produce a leakage-free baseline: per `(R,C,time_step)` median pressure computed from train, with a global fallback and rounding/clipping to valid pressure grid (your existing calibration step). The output be a valid `median_submission.csv` with columns `id,pressure`.'
- What this solution (achieved 6.33605) has done: 'Your current baseline ignores the key scoring rule (only inspiratory phase is evaluated), so predicting pressures during expiration can badly hurt MAE; we fix this by forcing predictions to 0 when `u_out==1` (expiratory phase). To improve accuracy during inspiration with minimal logic change, we also key the median lookup on `u_out` (and keep the same `(R,C,time_step)` idea), and compute the median on train inspiratory rows only to better match the metric. Finally, we make `time_step` matching robust by rounding to a fixed precision in both train/test before grouping/merging, reducing merge-miss fallbacks that degrade score, while keeping your pressure-grid rounding/clipping step intact.'
- What this solution (achieved 6.10835) has done: 'Your current pipeline already runs and writes a valid submission, but it leaves accuracy on the table because `time_step` matching via rounding still causes merge misses and because forcing expiratory predictions to 0 is likely miscalibrated (true pressures during u_out=1 are not 0 even if not scored). To move the MAE down toward the target with minimal logic change, I (1) switch from rounding to an exact-keyed `time_step_idx` (0–79) within each breath, which matches train/test perfectly and reduces fallbacks, and (2) avoid harming inspiration while not over-penalizing expiration by setting u_out=1 predictions to the training median pressure at that same `time_step_idx` (not 0). I keep your existing feature engineering, scaling, and the pressure-grid snapping/clipping step unchanged, and still generate the same `median_submission.csv` schema.'
- What this solution (achieved 4.61224) has done: 'Your current score (6.10835, lower-is-better) is still far from the target (0.1383), so we need a legitimate accuracy lift while keeping your “median-lookup” core approach intact. The biggest gain with minimal conceptual change is to condition the median not just on `(R,C,time_step_idx)` but also on discretized `u_in`, because pressure during inspiration is strongly driven by `u_in`; this keeps the same non-ML, aggregation-based logic but reduces bias. We also ensure the submission `id` alignment is correct by building submission from `test_raw[['id']]` rather than relying on `sample_submission` ordering (safe and score-relevant if any mismatch). Finally, we keep your existing pressure-grid snapping/clipping unchanged to preserve evaluation semantics.'
- What this solution (achieved 1.72122) has done: 'Your current baseline is already producing a valid submission, but the main reason it’s far from the target MAE is that it throws away the strongest “state” signal: cumulative delivered volume (integral of `u_in` over time) and the sharp pressure drop around the first `u_out=1`. To move the score down (lower is better) without changing the core “median lookup” approach, I (1) add a lightweight, fully-local cumulative-area bin (`area = cumsum(u_in)` per breath) into the grouping keys, and (2) handle the expiration transition more realistically by using a “post-first-u_out” median curve by time index instead of a single global-by-index curve. I keep your existing feature engineering/scaling untouched (still computed but not used for prediction), keep the pressure-grid snapping/clipping step, and still write `median_submission.csv` with `id,pressure`.'
- What this solution (achieved 1.71942) has done: 'We keep your median-lookup baseline exactly as-is, but make two minimal score-relevant adjustments that typically reduce MAE in this competition without changing the overall approach: (1) replace the slow/approximate `first_u_out` detection (which can mislabel “after_first_uout” and distort the expiratory curve) with an exact, vectorized groupwise first-`u_out==1` index; and (2) snap predictions to the *true* pressure grid by computing `P_STEP` from the sorted unique pressures (your current `pressure[1]-pressure[0]` can be wrong due to row order), which improves calibration and reduces MAE. Everything else (features, scaling, binning keys, fallbacks, expiratory handling concept, and submission schema/path) remains unchanged and it still writes `median_submission.csv`. These are small, deterministic fixes aimed at moving the score down from 1.72 toward your 0.138 target without altering the core logic.'
- What this solution (achieved 3.37688) has done: 'Your current score (1.71942, lower-is-better) is still far above the target (0.1383), so we need a meaningful but still “same-core-logic” accuracy lift. Keeping your median-lookup approach intact, the minimal high-impact change is to replace the coarse `u_in`/`area` binning with exact (or near-exact) numeric keys: use `u_in` rounded to 0.1 and `u_in_area` rounded to 0.5, which dramatically reduces within-bin bias while preserving the same groupby/median + fallback structure. We keep the inspiratory-only training for the main table, keep the expiratory handling concept unchanged, and keep the same pressure-grid snapping/clipping; we only adjust the aggregation keys and add one extra fallback level to maintain coverage when exact keys miss. This should reduce MAE toward your target without changing model architecture/training (none exists) or the overall prediction semantics.'
- What this solution (achieved 1.86412) has done: 'Your current median-lookup is reasonable, but the jump from 1.72 to 3.37 strongly suggests the recent “near-exact” keys made coverage too sparse, causing many fallbacks and worse MAE. I keep the same core approach (groupby medians + hierarchical fallbacks + expiratory handling + pressure-grid snapping), but make the keys slightly less granular and more metric-aligned: compute `u_in_area` as the true integral `cumsum(u_in * delta_time)` (not just `cumsum(u_in)`), and use a coarser `u_in_key` / `area_key` to reduce merge-misses. I also restrict the expiratory fallback curve to be conditioned on `(R,C,time_step_idx)` (still a median table, not a model), which typically reduces error on the inspiratory-scored part by better separating lung types while keeping semantics intact. These are minimal, targeted changes aimed specifically at recovering the regression and moving MAE back down toward your target band.'
- What this solution (achieved 1.85329) has done: 'Your current score (1.864, lower-is-better) is still far above the target (0.138), so we should improve accuracy while keeping your “median lookup + hierarchical fallbacks + expiratory handling + pressure-grid snapping” core logic intact. The main regression risk here is sparse exact-key merges (especially `area_key`) causing many fallbacks; we reduce merge-misses by (1) computing `u_in_area` as a proper trapezoidal integral (more physically aligned and smoother), and (2) adding one additional *coarser* RC-conditioned fallback that drops `area_key` but keeps `u_in_key` and `time_step_idx`. Finally, we make submission ordering robust by sorting by `id` before writing (cheap safety) while keeping paths, schema, and post-processing identical. These are minimal, score-relevant changes that typically reduce MAE substantially for this competition without changing any ML architecture/training (none exists here).'
- What this solution (achieved 1.85329) has done: 'We keep your exact “median lookup + hierarchical fallbacks + expiratory handling + pressure-grid snapping” approach, but make the lookup tables less sparse where it currently falls back too often. Concretely, we (1) add an additional RC-conditioned fallback that keeps `area_key` but drops `u_in_key` (helpful when `u_in_key` sparsity causes misses), and (2) add another fallback that uses `(R,C,time_step_idx,area_key)` for similar coverage improvement. These are minimal, deterministic additions that preserve your semantics while typically lowering MAE by reducing reliance on the global median. We also keep submission alignment robust by continuing to build from `test_raw['id']` and sorting by `id`.'
- What this solution (achieved 1.79203) has done: 'Your current score is far above the target (lower is better), and the main issue is that the prediction table is still too sparse in many regions, forcing fallbacks that are poorly aligned with the scored inspiratory phase. I keep your exact “median lookup + hierarchical fallbacks + expiratory handling + pressure-grid snapping” core logic, but (1) add two very cheap, denser fallback tables keyed by `(R,C,time_step_idx, u_in_key)` and by `(R,C,time_step_idx, area_key)` computed on inspiratory rows, and (2) slightly adjust the key rounding (u_in to 1.0, area to 2.0) to reduce merge-misses while preserving the same approach. I not change your feature engineering/scaling blocks (they remain computed but unused for prediction), and I keep the same output filename/schema while ensuring id alignment and sorting stay correct. These minimal changes are specifically aimed at reducing fallback-to-global frequency, which should move MAE down toward your target.'
- What this solution (achieved 1.74331) has done: 'I keep your median-lookup + hierarchical fallback logic intact and only make minimal, score-relevant adjustments to reduce merge sparsity that is currently forcing too many fallbacks (a common cause of MAE staying ~1.7–1.9). Specifically, I (1) make the discretization slightly denser by using half-step bins for `u_in_key` and a bit finer bins for `area_key`, and (2) add one additional *very cheap* fallback keyed by `(R,C,time_step_idx,u_in_key,area_key)` but with coarser `area_key` to catch near-misses without changing the overall approach. Everything else—including your feature engineering/scaling blocks, expiratory handling concept, and pressure-grid snapping—remains the same, and it still write a valid `median_submission.csv`.'
- What this solution (achieved 1.57374) has done: 'We keep your median-lookup + hierarchical fallback structure intact and only make two minimal, score-relevant fixes that typically reduce MAE for this competition. First, we key the lookup on the *pressure-driving dynamics* by adding `u_in_diff1` (discretized) to the finest-grain median table (with safe fallbacks), which helps distinguish rising vs steady `u_in` at the same `u_in`/area/time index. Second, we ensure the expiration replacement curve is conditioned on `after_first_uout` **and** uses a robust fallback (`(R,C,time_step_idx)` then global-by-idx), but only applied when `u_out==1` (same as your current semantics). These changes aim to improve predictions during inspiration (the scored region) without altering your feature pipeline, pressure snapping, or the overall “groupby medians then merge/fillna” approach, and the script still write a valid `median_submission.csv`.'
- What this solution (achieved 1.58386) has done: 'Your current score (1.57374, lower-is-better) is still far above the target (0.1383), so we should improve accuracy while keeping your exact “median lookup + hierarchical fallbacks + expiratory handling + pressure-grid snapping” core logic intact. The most score-relevant minimal change is to make the inspiratory lookup less sparse without changing the approach: add an additional, denser fallback that drops the noisiest key (`u_diff_key`) but keeps (`R,C,time_step_idx,u_in_key,area_key`) and another that drops `area_key` but keeps (`R,C,time_step_idx,u_in_key,u_diff_key`), which reduces fallback-to-global frequency. We also remove two redundant merges (`pred_rc_tu2`, `pred_rc_ta2`) that are exact duplicates of earlier tables (they do not add information, only overhead), keeping prediction semantics the same. These tweaks should move MAE down toward the target while preserving your existing pipeline and still writing a valid `median_submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler

DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"



## === cell 1
sub = pd.read_csv(SAMPLE_SUB_PATH)

pred = None



## === cell 2
mean = None
med = None
std = None
clipped_mean = None



## === cell 3
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def add_features(df):
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    print("Step-1...Completed")

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
    df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
    df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
    df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
    df = df.fillna(0)
    print("Step-2...Completed")

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )
    print("Step-3...Completed")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
    print("Step-4...Completed")

    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["breath_id__u_in_lag"] = df["breath_id__u_in_lag"] * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["breath_id__u_in_lag2"] = df["breath_id__u_in_lag2"] * df["breath_id_lag2same"]
    print("Step-5...Completed")

    df["time_step_diff"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["ewm_u_in_mean"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=9)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(
            {
                "15_in_sum": "sum",
                "15_in_min": "min",
                "15_in_max": "max",
                "15_in_mean": "mean",
            }
        )
        .reset_index(level=0, drop=True)
    )
    print("Step-6...Completed")

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    print("Step-7...Completed")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")

    return df


print("Train data...\n")
train = add_features(train_df)

print("\nTest data...\n")
test = add_features(test_df)

del train_df
gc.collect()



## === cell 4
targets = train[["pressure"]].to_numpy().reshape(-1, 80)

train.drop(
    [
        "pressure",
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
    inplace=True,
)

test = test.drop(
    [
        "id",
        "breath_id",
        "one",
        "count",
        "breath_id_lag",
        "breath_id_lag2",
        "breath_id_lagsame",
        "breath_id_lag2same",
    ],
    axis=1,
)

print(f"train: {train.shape} \ntest: {test.shape}")



## === cell 5
scaler = RobustScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

print(f"train: {train.shape} \ntest: {test.shape} \ntargets: {targets.shape}")



## === cell 6
pressure = targets.squeeze().reshape(-1, 1).astype("float32")

P_MIN = float(np.min(pressure))
P_MAX = float(np.max(pressure))

_unique_p = np.unique(pressure.astype(np.float32).ravel())
_unique_p.sort()
if _unique_p.size >= 2:
    P_STEP = float(np.min(np.diff(_unique_p)))
else:
    P_STEP = 1.0  # fallback (should not happen with this dataset)

print("Min pressure: {}".format(P_MIN))
print("Max pressure: {}".format(P_MAX))
print("Pressure step: {}".format(P_STEP))
print("Unique values:  {}".format(_unique_p.shape[0]))

del pressure, _unique_p
gc.collect()



## === cell 7
submission = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 8
train_raw = pd.read_csv(
    TRAIN_PATH,
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
test_raw = pd.read_csv(
    TEST_PATH, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)

train_raw["time_step_idx"] = train_raw.groupby("breath_id").cumcount().astype(np.int16)
test_raw["time_step_idx"] = test_raw.groupby("breath_id").cumcount().astype(np.int16)

train_raw["dt"] = (
    train_raw.groupby("breath_id")["time_step"].diff().fillna(0).astype(np.float32)
)
test_raw["dt"] = (
    test_raw.groupby("breath_id")["time_step"].diff().fillna(0).astype(np.float32)
)

train_raw["u_in_prev"] = (
    train_raw.groupby("breath_id")["u_in"].shift(1).fillna(0).astype(np.float32)
)
test_raw["u_in_prev"] = (
    test_raw.groupby("breath_id")["u_in"].shift(1).fillna(0).astype(np.float32)
)

train_raw["u_in_area"] = (
    (
        0.5
        * (train_raw["u_in"].astype(np.float32) + train_raw["u_in_prev"])
        * train_raw["dt"]
    )
    .groupby(train_raw["breath_id"])
    .cumsum()
)
test_raw["u_in_area"] = (
    (
        0.5
        * (test_raw["u_in"].astype(np.float32) + test_raw["u_in_prev"])
        * test_raw["dt"]
    )
    .groupby(test_raw["breath_id"])
    .cumsum()
)

train_raw["u_in_diff1"] = train_raw["u_in"].astype(np.float32) - train_raw[
    "u_in_prev"
].astype(np.float32)
test_raw["u_in_diff1"] = test_raw["u_in"].astype(np.float32) - test_raw[
    "u_in_prev"
].astype(np.float32)

UIN_ROUND = 0.5
AREA_ROUND = 1.0
train_raw["u_in_key"] = np.round(train_raw["u_in"] / UIN_ROUND).astype(np.int16)
test_raw["u_in_key"] = np.round(test_raw["u_in"] / UIN_ROUND).astype(np.int16)
train_raw["area_key"] = np.round(train_raw["u_in_area"] / AREA_ROUND).astype(np.int32)
test_raw["area_key"] = np.round(test_raw["u_in_area"] / AREA_ROUND).astype(np.int32)

UDIFF_ROUND = 0.5
train_raw["u_diff_key"] = np.round(train_raw["u_in_diff1"] / UDIFF_ROUND).astype(
    np.int16
)
test_raw["u_diff_key"] = np.round(test_raw["u_in_diff1"] / UDIFF_ROUND).astype(np.int16)

AREA_ROUND_COARSE = 2.0
train_raw["area_key2"] = np.round(train_raw["u_in_area"] / AREA_ROUND_COARSE).astype(
    np.int32
)
test_raw["area_key2"] = np.round(test_raw["u_in_area"] / AREA_ROUND_COARSE).astype(
    np.int32
)

train_insp = train_raw[train_raw["u_out"] == 0]

rc_t_u_a_d_median = (
    train_insp.groupby(
        ["R", "C", "time_step_idx", "u_in_key", "area_key", "u_diff_key"], sort=False
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_d"})
)

rc_t_u_a_nod_median = (
    train_insp.groupby(["R", "C", "time_step_idx", "u_in_key", "area_key"], sort=False)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_nod"})
)

rc_t_u_d_median = (
    train_insp.groupby(
        ["R", "C", "time_step_idx", "u_in_key", "u_diff_key"], sort=False
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_ud"})
)

rc_t_u_a2_median = (
    train_insp.groupby(
        ["R", "C", "time_step_idx", "u_in_key", "area_key2"], sort=False
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_a2"})
)

rc_t_u_median = (
    train_insp.groupby(["R", "C", "time_step_idx", "u_in_key"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_u"})
)

rc_u_median = (
    train_insp.groupby(["R", "C", "u_in_key"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_rcu"})
)

t_u_a_median = (
    train_insp.groupby(["time_step_idx", "u_in_key", "area_key"], sort=False)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_tua"})
)

rc_t_median = (
    train_insp.groupby(["R", "C", "time_step_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_rc"})
)

rc_t_a_median = (
    train_insp.groupby(["R", "C", "time_step_idx", "area_key"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_rc_ta"})
)

rc_a_median = (
    train_insp.groupby(["R", "C", "area_key"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_rca"})
)

global_insp_median = float(train_insp["pressure"].median())

pred_df = test_raw.merge(
    rc_t_u_a_d_median,
    on=["R", "C", "time_step_idx", "u_in_key", "area_key", "u_diff_key"],
    how="left",
)

pred_df = pred_df.merge(
    rc_t_u_a_nod_median,
    on=["R", "C", "time_step_idx", "u_in_key", "area_key"],
    how="left",
)
pred_df = pred_df.merge(
    rc_t_u_d_median,
    on=["R", "C", "time_step_idx", "u_in_key", "u_diff_key"],
    how="left",
)

pred_df = pred_df.merge(
    rc_t_u_a2_median,
    on=["R", "C", "time_step_idx", "u_in_key", "area_key2"],
    how="left",
)
pred_df = pred_df.merge(
    rc_t_u_median, on=["R", "C", "time_step_idx", "u_in_key"], how="left"
)
pred_df = pred_df.merge(rc_u_median, on=["R", "C", "u_in_key"], how="left")
pred_df = pred_df.merge(
    t_u_a_median, on=["time_step_idx", "u_in_key", "area_key"], how="left"
)
pred_df = pred_df.merge(
    rc_t_a_median, on=["R", "C", "time_step_idx", "area_key"], how="left"
)
pred_df = pred_df.merge(rc_a_median, on=["R", "C", "area_key"], how="left")
pred_df = pred_df.merge(rc_t_median, on=["R", "C", "time_step_idx"], how="left")

pred_values = (
    pred_df["pred_d"]
    .fillna(pred_df["pred_nod"])
    .fillna(pred_df["pred_ud"])
    .fillna(pred_df["pred_a2"])
    .fillna(pred_df["pred_u"])
    .fillna(pred_df["pred_rcu"])
    .fillna(pred_df["pred_tua"])
    .fillna(pred_df["pred_rc_ta"])
    .fillna(pred_df["pred_rca"])
    .fillna(pred_df["pred_rc"])
    .fillna(global_insp_median)
    .to_numpy(dtype=np.float32)
)

tmp_train_first = (
    train_raw.loc[train_raw["u_out"].eq(1), ["breath_id", "time_step_idx"]]
    .groupby("breath_id", sort=False)["time_step_idx"]
    .min()
)
train_first_uout = (
    train_raw["breath_id"].map(tmp_train_first).fillna(80).astype(np.int16)
)
train_raw["after_first_uout"] = (
    train_raw["time_step_idx"].astype(np.int16) >= train_first_uout
).astype(np.int8)

tmp_test_first = (
    test_raw.loc[test_raw["u_out"].eq(1), ["breath_id", "time_step_idx"]]
    .groupby("breath_id", sort=False)["time_step_idx"]
    .min()
)
test_first_uout = test_raw["breath_id"].map(tmp_test_first).fillna(80).astype(np.int16)
test_raw["after_first_uout"] = (
    test_raw["time_step_idx"].astype(np.int16) >= test_first_uout
).astype(np.int8)

exp_rc_t_median_after = (
    train_raw[train_raw["after_first_uout"] == 1]
    .groupby(["R", "C", "time_step_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "exp_pred"})
)

test_exp = test_raw.loc[:, ["R", "C", "time_step_idx"]].merge(
    exp_rc_t_median_after, on=["R", "C", "time_step_idx"], how="left"
)

exp_median_by_idx_after = (
    train_raw[train_raw["after_first_uout"] == 1]
    .groupby(["time_step_idx"], sort=False)["pressure"]
    .median()
    .reindex(range(80))
    .ffill()
    .bfill()
    .to_numpy(dtype=np.float32)
)
exp_values = test_exp["exp_pred"].to_numpy(dtype=np.float32)
tidx_test = test_raw["time_step_idx"].to_numpy(dtype=np.int16)
exp_values = np.where(
    np.isnan(exp_values), exp_median_by_idx_after[tidx_test], exp_values
)

u_out_test = test_raw["u_out"].to_numpy(dtype=np.int8)
pred_values = np.where(u_out_test == 1, exp_values, pred_values)

submission = pd.DataFrame({"id": test_raw["id"].to_numpy(), "pressure": pred_values})

submission["pressure"] = (
    np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
)
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission = submission.sort_values("id", kind="mergesort").reset_index(drop=True)

submission.to_csv("median_submission.csv", index=False)

print(submission.head())
print("Wrote: median_submission.csv  rows:", len(submission))
print("Columns:", submission.columns.tolist())
