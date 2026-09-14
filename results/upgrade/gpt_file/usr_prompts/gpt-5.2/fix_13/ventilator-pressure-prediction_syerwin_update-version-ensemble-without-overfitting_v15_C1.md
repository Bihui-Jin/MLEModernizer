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

0.1381750237826189

# 6. Current score

1.81108

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.24114) has done: 'I remove the broken dependency on external Kaggle datasets (the missing `../input/...` submission files) and instead generate a self-contained prediction from the provided train/test using the same feature engineering you already have. To preserve your intent of “median/ensemble-like” behavior without changing the overall approach, I compute a simple per-time-step baseline (median pressure at each of the 80 time steps within a breath) from the training set and apply it to every test breath. I also fix the pressure discretization step calculation (it was using the first two raw values, which may not represent the true grid) by deriving it from sorted unique pressure levels. Finally, I ensure a valid `submission.csv` (with `.csv` suffix and correct columns) is always written end-to-end.'
- What this solution (achieved 6.11694) has done: 'Your current 6.24 MAE comes from using a global median-by-timestep baseline that ignores key known drivers (R, C, and u_out) and also predicts during expiratory (u_out=1) which is not scored but can still hurt if it misaligns. To move toward the 0.138 target with minimal logic change, I keep your “training-set aggregation baseline” approach but compute medians conditioned on (R, C, time_step_index) and apply a simple expiratory handling: set predictions to 0 when u_out==1 (a common safe baseline since expiratory isn’t scored). I also ensure the per-row mapping is correct by merging predictions back onto test rows (rather than tiling by breath count), avoiding any potential ordering mismatch. Finally, I keep your existing pressure grid snapping (discretization) but apply it after these improvements.'
- What this solution (achieved 6.40263) has done: 'Your current score is far worse than the 0.138 target (lower is better), and the biggest issue is that the model forces `pressure=0` whenever `u_out==1`, which is not aligned with how the test submission is evaluated/constructed and can create large errors around phase transitions. I keep your exact “training-set aggregation baseline” core approach, but replace that expiratory handling with a safer rule: predict the last inspiratory predicted pressure for the rest of the breath after `u_out` turns 1 (forward-fill within each breath). I also compute medians conditioned on `(R, C, t_idx, u_out)` so the baseline better matches inspiratory vs expiratory dynamics without changing the overall method. Finally, I keep your existing pressure-grid snapping/clipping and ensure the submission remains correctly aligned by `id`.'
- What this solution (achieved 6.10819) has done: 'Your current MAE is far above the target (lower is better), and the most likely cause is that your submission rows are misaligned: you sort `pred` by `(breath_id, t_idx)` but then write `pred_pressure` into `submission` in `id` order, so pressures get assigned to the wrong `id`s. To move sharply toward the target without changing your core “training-set aggregation baseline” approach, I keep your same groupby-median logic and expiratory forward-fill, but I re-align predictions by `id` (sort by `id` / merge by `id`) before writing. I also remove the unnecessary merge of `submission` with `pred[["id"]]` (it can duplicate/drop rows if anything goes wrong) and instead assert row counts match and then assign by `id`-sorted order for safety. These are minimal, score-critical fixes that should drastically reduce MAE while preserving your modeling semantics.'
- What this solution (achieved 6.10819) has done: 'Your current score (6.10819, lower is better) is still far from the 0.138 target, so we should improve performance with minimal, low-risk fixes while preserving your “training-set aggregation baseline + forward-fill + pressure snapping” core logic. The biggest remaining issue is that you’re conditioning medians on `u_out`, but the competition metric ignores expiratory (`u_out==1`) rows, so learning/predicting expiratory pressures can distort the forward-fill behavior near the transition; we instead compute medians using inspiratory rows only and use those inspiratory medians as the source for forward-fill once `u_out` turns 1. To keep robustness, we also add a per-(R,C,t_idx) fallback (without `u_out`) before falling back to coarser medians/global median. Finally, we keep the exact discretization/snapping to the pressure grid and keep strict `id` alignment when writing `submission.csv`.'
- What this solution (achieved 3.60629) has done: 'Your current score is still far above the target (lower is better), so we should improve accuracy while keeping your “training-set aggregation baseline + forward-fill + pressure snapping” core logic unchanged. The main minimal upgrade is to condition the median baseline not just on (R, C, t_idx) but also on an approximate “breath phase” derived from the cumulative integral of u_in (“area”), which strongly correlates with pressure and is already part of your feature set. We compute area per time step in both train/test raw, then groupby-median on (R, C, t_idx, area_bin) using inspiratory rows only, with the existing fallback ladder and the same expiratory forward-fill behavior. This should move MAE substantially toward the 0.138 target without changing model type or post-processing semantics.'
- What this solution (achieved 3.22244) has done: 'Your current MAE (3.60629, lower is better) is still far above the 0.138 target, so we should improve accuracy while preserving your core “training-set aggregation baseline + forward-fill + pressure-grid snapping” approach. The most direct minimal gain is to refine the conditioning of your median lookup by adding a coarse bin of `u_in` (instantaneous valve opening) alongside your existing `(R, C, t_idx, area_bin)` key, since pressure is strongly driven by current `u_in` even at the same area. To keep behavior stable, we keep the same inspiratory-only training rows, keep the same fallback ladder, and keep the same forward-fill for `u_out==1`. Finally, we keep your existing snapping to the training pressure grid and strict `id` alignment for a valid submission.'
- What this solution (achieved 2.78618) has done: 'Your core approach is an aggregation baseline with fallback + per-breath forward-fill + pressure-grid snapping; the biggest low-risk way to move MAE down toward the 0.138 target is to make the aggregation key slightly more informative without changing the method. I keep everything else the same, but refine the median lookup by adding a coarse `u_in_diff1` bin (how fast valve opening changes) alongside your existing `(R,C,t_idx,area_bin,u_in_bin)` key, with the same fallback ladder if a key is unseen. This typically helps around sharp transients where the same `u_in`/area can still map to different pressures depending on whether flow is rising or falling. I also compute the bin edges once from train inspiratory rows and apply to test to avoid train/test bin drift; submission writing and id alignment remain unchanged.'
- What this solution (achieved 2.70228) has done: 'Your current MAE (2.78618, lower is better) is still far above the 0.138 target, so we should make a small, low-risk improvement while preserving your aggregation+fallback+forward-fill+pressure-snapping core logic. The most direct minimal gain is to add a coarse conditioning on the *previous* valve opening (`u_in_lag1_bin`) in the median lookup, which helps disambiguate similar current `u_in`/area states during rising vs falling segments without changing the overall method. I keep your same inspiratory-only training aggregation, same fallback ladder structure, same per-breath forward-fill after `u_out` becomes 1, and the same pressure grid snapping. All I/O paths and the submission writing/alignment checks remain intact.'
- What this solution (achieved 2.19624) has done: 'Your current pipeline already builds a strong aggregation baseline, but it’s still leaving accuracy on the table because (1) `area` is being computed with `time_step * u_in` (instead of integrating with `delta_time`), and (2) your binning uses `pd.cut` on quantile edges, which can create empty/unstable bins and NaNs at edges in test. I keep your exact “median lookups + fallback ladder + per-breath forward-fill + pressure-grid snapping + id alignment” core logic, and only make minimal, metric-relevant tweaks: compute `area` as cumulative sum of `u_in * dt` (consistent with a physical integral) and switch to `pd.qcut(..., duplicates='drop')` for stable quantile binning shared between train/test. These changes should reduce MAE (lower is better) and move the score toward your 0.138 target without changing the overall approach or adding new models. The rest of your feature-engineering/model code is left intact to preserve semantics and runtime behavior.'
- What this solution (achieved 2.10398) has done: 'I keep your exact aggregation-baseline + fallback ladder + per-breath forward-fill + pressure-grid snapping approach, but make two minimal, metric-relevant fixes that should reduce MAE toward the 0.138 target. First, I compute `area_bin`, `u_in_bin`, and `u_in_lag1_bin` using bin edges derived from *inspiratory-only* (`u_out==0`) training rows, since the evaluation ignores expiratory and mixing phases makes bins noisier. Second, I prevent `pd.cut` from producing `-1`/NaN bins on test by extending the first/last bin edges to `(-inf, +inf)` and filling any remaining NaNs with a safe default bin, which reduces unseen-key fallbacks. Everything else (features, medians, fallbacks, forward-fill, snapping, and strict `id` alignment) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.81108) has done: 'Your current score (2.10398, lower-is-better) is still far above the target (0.138), so we should improve accuracy with a minimal change that preserves your “train aggregation medians + fallback ladder + per-breath forward-fill + pressure-grid snapping” core logic. The biggest low-risk gap is that your median keys don’t explicitly include the strong drivers `u_in_cumsum`/`time_step_cumsum` that you already compute in `add_features`, so we add a *coarse* bin of inspiratory `u_in_cumsum` (computed on raw train/test like you already do for `area`) into the median lookup key, with the same fallback structure when unseen. This keeps the same approach (groupby median lookup) but makes the lookup substantially more state-aware without introducing any new model/training loop. All paths and submission alignment checks remain unchanged, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
sub.head()



## === cell 2
train_df = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


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

gc.collect()



## === cell 3
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



## === cell 4
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

train_scaled = train_scaled.reshape(-1, 80, train_scaled.shape[-1])
test_scaled = test_scaled.reshape(-1, 80, train_scaled.shape[-1])

print(
    f"train: {train_scaled.shape} \ntest: {test_scaled.shape} \ntargets: {targets.shape}"
)



## === cell 5
pressure_flat = targets.reshape(-1).astype("float32")
P_MIN = float(np.min(pressure_flat))
P_MAX = float(np.max(pressure_flat))

unique_p = np.unique(pressure_flat)
unique_p.sort()
diffs = np.diff(unique_p)
diffs = diffs[diffs > 0]
P_STEP = float(diffs.min()) if diffs.size else 1.0

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step: {P_STEP}")
print(f"Unique values: {unique_p.shape[0]}")

del pressure_flat, unique_p, diffs
gc.collect()



## === cell 6
train_raw = train_df.copy()
test_raw = test_df.copy()

train_raw["t_idx"] = train_raw.groupby("breath_id").cumcount().astype(np.int16)
test_raw["t_idx"] = test_raw.groupby("breath_id").cumcount().astype(np.int16)

train_raw["dt"] = (
    train_raw.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
)
test_raw["dt"] = (
    test_raw.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
)

train_raw["area"] = (
    (train_raw["u_in"] * train_raw["dt"]).groupby(train_raw["breath_id"]).cumsum()
)
test_raw["area"] = (
    (test_raw["u_in"] * test_raw["dt"]).groupby(test_raw["breath_id"]).cumsum()
)

train_raw["u_in_lag1"] = train_raw.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
test_raw["u_in_lag1"] = test_raw.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
train_raw["u_in_diff1"] = (train_raw["u_in"] - train_raw["u_in_lag1"]).astype(
    np.float32
)
test_raw["u_in_diff1"] = (test_raw["u_in"] - test_raw["u_in_lag1"]).astype(np.float32)

train_raw["u_in_cumsum"] = (
    train_raw.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
)
test_raw["u_in_cumsum"] = (
    test_raw.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
)

train_insp_mask = train_raw["u_out"] == 0

area_base = train_raw.loc[train_insp_mask, "area"].to_numpy(dtype=np.float64)
area_bins_train = pd.qcut(area_base, q=5, labels=False, duplicates="drop")
n_area_bins = (
    int(area_bins_train.max() + 1) if np.isfinite(area_bins_train).any() else 1
)
_, area_edges = pd.qcut(
    area_base, q=max(n_area_bins, 2), retbins=True, duplicates="drop"
)
area_edges = np.array(area_edges, dtype=np.float64)
area_edges[0] = -np.inf
area_edges[-1] = np.inf

train_raw["area_bin"] = pd.cut(
    train_raw["area"], bins=area_edges, include_lowest=True, labels=False
).astype("float32")
test_raw["area_bin"] = pd.cut(
    test_raw["area"], bins=area_edges, include_lowest=True, labels=False
).astype("float32")
train_raw["area_bin"] = train_raw["area_bin"].fillna(0).astype("int16")
test_raw["area_bin"] = test_raw["area_bin"].fillna(0).astype("int16")

u_base = train_raw.loc[train_insp_mask, "u_in"].to_numpy(dtype=np.float64)
u_in_bins_train = pd.qcut(u_base, q=4, labels=False, duplicates="drop")
n_u_bins = int(u_in_bins_train.max() + 1) if np.isfinite(u_in_bins_train).any() else 1
_, u_edges = pd.qcut(u_base, q=max(n_u_bins, 2), retbins=True, duplicates="drop")
u_edges = np.array(u_edges, dtype=np.float64)
u_edges[0] = -np.inf
u_edges[-1] = np.inf

train_raw["u_in_bin"] = pd.cut(
    train_raw["u_in"], bins=u_edges, include_lowest=True, labels=False
).astype("float32")
test_raw["u_in_bin"] = pd.cut(
    test_raw["u_in"], bins=u_edges, include_lowest=True, labels=False
).astype("float32")
train_raw["u_in_bin"] = train_raw["u_in_bin"].fillna(0).astype("int16")
test_raw["u_in_bin"] = test_raw["u_in_bin"].fillna(0).astype("int16")

train_raw["u_in_lag1_bin"] = pd.cut(
    train_raw["u_in_lag1"], bins=u_edges, include_lowest=True, labels=False
).astype("float32")
test_raw["u_in_lag1_bin"] = pd.cut(
    test_raw["u_in_lag1"], bins=u_edges, include_lowest=True, labels=False
).astype("float32")
train_raw["u_in_lag1_bin"] = train_raw["u_in_lag1_bin"].fillna(0).astype("int16")
test_raw["u_in_lag1_bin"] = test_raw["u_in_lag1_bin"].fillna(0).astype("int16")

train_insp_for_bins = train_raw.loc[train_insp_mask, "u_in_diff1"].to_numpy(
    dtype=np.float64
)
d_q = np.quantile(train_insp_for_bins, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
d_q = np.unique(d_q)
if d_q.size < 3:
    d_q = np.linspace(
        float(np.min(train_insp_for_bins)), float(np.max(train_insp_for_bins)), 6
    )
d_q = np.array(d_q, dtype=np.float64)
d_q[0] = -np.inf
d_q[-1] = np.inf

train_raw["u_in_diff1_bin"] = pd.cut(
    train_raw["u_in_diff1"], bins=d_q, include_lowest=True, labels=False
).astype("float32")
test_raw["u_in_diff1_bin"] = pd.cut(
    test_raw["u_in_diff1"], bins=d_q, include_lowest=True, labels=False
).astype("float32")
train_raw["u_in_diff1_bin"] = train_raw["u_in_diff1_bin"].fillna(0).astype("int16")
test_raw["u_in_diff1_bin"] = test_raw["u_in_diff1_bin"].fillna(0).astype("int16")

uin_c_base = train_raw.loc[train_insp_mask, "u_in_cumsum"].to_numpy(dtype=np.float64)
uin_c_bins_train = pd.qcut(uin_c_base, q=6, labels=False, duplicates="drop")
n_uin_c_bins = (
    int(uin_c_bins_train.max() + 1) if np.isfinite(uin_c_bins_train).any() else 1
)
_, uin_c_edges = pd.qcut(
    uin_c_base, q=max(n_uin_c_bins, 2), retbins=True, duplicates="drop"
)
uin_c_edges = np.array(uin_c_edges, dtype=np.float64)
uin_c_edges[0] = -np.inf
uin_c_edges[-1] = np.inf

train_raw["u_in_cumsum_bin"] = pd.cut(
    train_raw["u_in_cumsum"], bins=uin_c_edges, include_lowest=True, labels=False
).astype("float32")
test_raw["u_in_cumsum_bin"] = pd.cut(
    test_raw["u_in_cumsum"], bins=uin_c_edges, include_lowest=True, labels=False
).astype("float32")
train_raw["u_in_cumsum_bin"] = train_raw["u_in_cumsum_bin"].fillna(0).astype("int16")
test_raw["u_in_cumsum_bin"] = test_raw["u_in_cumsum_bin"].fillna(0).astype("int16")

train_insp = train_raw.loc[
    train_raw["u_out"] == 0,
    [
        "R",
        "C",
        "t_idx",
        "area_bin",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_diff1_bin",
        "u_in_cumsum_bin",
        "pressure",
    ],
].copy()

rc_t_a_u_l_u_d_uc_median = (
    train_insp.groupby(
        [
            "R",
            "C",
            "t_idx",
            "area_bin",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_diff1_bin",
            "u_in_cumsum_bin",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("pressure_pred")
    .reset_index()
)

rc_t_a_u_d_uc_median = (
    train_insp.groupby(
        [
            "R",
            "C",
            "t_idx",
            "area_bin",
            "u_in_bin",
            "u_in_diff1_bin",
            "u_in_cumsum_bin",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("pressure_rc_t_a_u_d_uc_med")
    .reset_index()
)

rc_t_a_u_uc_median = (
    train_insp.groupby(
        ["R", "C", "t_idx", "area_bin", "u_in_bin", "u_in_cumsum_bin"], sort=False
    )["pressure"]
    .median()
    .rename("pressure_rc_t_a_u_uc_med")
    .reset_index()
)

rc_t_a_u_d_median = (
    train_insp.groupby(
        ["R", "C", "t_idx", "area_bin", "u_in_bin", "u_in_diff1_bin"], sort=False
    )["pressure"]
    .median()
    .rename("pressure_rc_t_a_u_d_med")
    .reset_index()
)

rc_t_a_u_median = (
    train_insp.groupby(["R", "C", "t_idx", "area_bin", "u_in_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("pressure_rc_t_a_u_med")
    .reset_index()
)

rc_t_a_median = (
    train_insp.groupby(["R", "C", "t_idx", "area_bin"], sort=False)["pressure"]
    .median()
    .rename("pressure_rc_t_a_med")
    .reset_index()
)

rc_t_median = (
    train_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .median()
    .rename("pressure_rc_t_med")
    .reset_index()
)

rc_median = (
    train_insp.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .rename("pressure_rc_med")
    .reset_index()
)

t_median = (
    train_insp.groupby(["t_idx"], sort=False)["pressure"]
    .median()
    .rename("pressure_t_med")
    .reset_index()
)

global_med = float(train_insp["pressure"].median())

pred = (
    test_raw[
        [
            "id",
            "breath_id",
            "R",
            "C",
            "t_idx",
            "area_bin",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_diff1_bin",
            "u_in_cumsum_bin",
            "u_out",
        ]
    ]
    .merge(
        rc_t_a_u_l_u_d_uc_median,
        on=[
            "R",
            "C",
            "t_idx",
            "area_bin",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_diff1_bin",
            "u_in_cumsum_bin",
        ],
        how="left",
    )
    .merge(
        rc_t_a_u_d_uc_median,
        on=[
            "R",
            "C",
            "t_idx",
            "area_bin",
            "u_in_bin",
            "u_in_diff1_bin",
            "u_in_cumsum_bin",
        ],
        how="left",
    )
    .merge(
        rc_t_a_u_uc_median,
        on=["R", "C", "t_idx", "area_bin", "u_in_bin", "u_in_cumsum_bin"],
        how="left",
    )
    .merge(
        rc_t_a_u_d_median,
        on=["R", "C", "t_idx", "area_bin", "u_in_bin", "u_in_diff1_bin"],
        how="left",
    )
    .merge(rc_t_a_u_median, on=["R", "C", "t_idx", "area_bin", "u_in_bin"], how="left")
    .merge(rc_t_a_median, on=["R", "C", "t_idx", "area_bin"], how="left")
    .merge(rc_t_median, on=["R", "C", "t_idx"], how="left")
    .merge(rc_median, on=["R", "C"], how="left")
    .merge(t_median, on=["t_idx"], how="left")
)

pred_pressure = pred["pressure_pred"].to_numpy(dtype=np.float32)

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_t_a_u_d_uc_med"].to_numpy(
        dtype=np.float32
    )

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_t_a_u_uc_med"].to_numpy(
        dtype=np.float32
    )

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_t_a_u_d_med"].to_numpy(
        dtype=np.float32
    )

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_t_a_u_med"].to_numpy(
        dtype=np.float32
    )

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_t_a_med"].to_numpy(
        dtype=np.float32
    )

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_t_med"].to_numpy(dtype=np.float32)

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_rc_med"].to_numpy(dtype=np.float32)

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = pred.loc[mask, "pressure_t_med"].to_numpy(dtype=np.float32)

mask = np.isnan(pred_pressure)
if mask.any():
    pred_pressure[mask] = global_med

pred["pressure_tmp"] = pred_pressure

pred.sort_values(["breath_id", "t_idx"], inplace=True, kind="mergesort")
pred["u_out"] = pred["u_out"].astype(np.int8)

pred["pressure_ffill"] = pred["pressure_tmp"]
pred.loc[pred["u_out"] == 1, "pressure_ffill"] = np.nan
pred["pressure_ffill"] = pred.groupby("breath_id")["pressure_ffill"].ffill()
pred["pressure_ffill"] = pred["pressure_ffill"].fillna(pred["pressure_tmp"])
pred_pressure = pred["pressure_ffill"].to_numpy(dtype=np.float32)

pred_pressure = (np.round((pred_pressure - P_MIN) / P_STEP) * P_STEP + P_MIN).astype(
    np.float32
)
pred_pressure = np.clip(pred_pressure, P_MIN, P_MAX)

pred_out = pred[["id"]].copy()
pred_out["pressure"] = pred_pressure
pred_out.sort_values("id", inplace=True, kind="mergesort")
pred_out.reset_index(drop=True, inplace=True)

submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission.sort_values("id", inplace=True, kind="mergesort")
submission.reset_index(drop=True, inplace=True)

if len(submission) != len(pred_out):
    raise ValueError(
        f"Row count mismatch: submission={len(submission)} pred={len(pred_out)}"
    )

if not np.array_equal(submission["id"].to_numpy(), pred_out["id"].to_numpy()):
    raise ValueError(
        "ID alignment mismatch between sample_submission and generated predictions."
    )

submission["pressure"] = pred_out["pressure"].to_numpy(dtype=np.float32)
submission.to_csv("submission.csv", index=False)
submission.head()
