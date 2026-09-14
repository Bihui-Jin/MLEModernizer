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

0.1384096203158494

# 6. Current score

2.11275

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/gb-vpp-pulp-fiction/...` etc. files), which is the root cause of the `FileNotFoundError` and the cascading `NameError`s. To preserve the original ensembling core logic (median/mean/clipped mean + pressure grid rounding), I instead build `pred` from three simple, deterministic baseline predictors computed from the provided train/test data (no new model training). I also fix pathing to use the available `/kaggle/input/ventilator-pressure-prediction/...` files, ensure `pred` is always defined before use, and write a valid `submission.csv` with columns `id,pressure`. Finally, I keep the existing feature engineering/scaler code intact (it still run end-to-end), but the submission be generated from the repaired `pred` pipeline.'
- What this solution (achieved 6.97887) has done: 'Your current 9.92 MAE comes from using simple global/grouped aggregates that ignore the breath dynamics and the fact that only inspiratory phase (u_out=0) is scored; we can move much closer to the 0.138 target with a minimal, legitimate change that keeps the “mapping/lookup ensemble + median + pressure-grid rounding” core logic. I replace the three weak base predictors with three *inspiratory-only*, *sequence-aware* predictors that use cumulative inspired volume (`u_in` integrated over time) and (R,C,u_out,time_step) to look up pressure from train, plus a safe fallback chain. This stays within the same approach (no model training; still a deterministic lookup ensemble) but uses features that are much more aligned with the physics and the competition metric. I also ensure train/test alignment for the new feature (per-row) and keep the existing pressure rounding/clipping submission post-processing unchanged.'
- What this solution (achieved 7.28803) has done: 'I fix the runtime error by ensuring the engineered feature matrices for train and test have identical columns before fitting/applying `RobustScaler` (the mismatch comes from `_cum_bin`/`_u_in_bin` existing only in `test_df`). To keep core logic unchanged and score behavior stable, I won’t alter the existing deterministic lookup-ensemble prediction pipeline; the scaler/reshaping path simply run without crashing. I also guard the `targets.reshape(-1, 80)` step with an assertion so it fails fast if row ordering/length ever changes, and I keep the final submission generation (`submission.csv` with `id,pressure`) exactly as intended.'
- What this solution (achieved 7.28803) has done: 'Your current MAE (7.288) is far worse than the target (0.138), so we should improve score substantially; the biggest easy gain with minimal logic change is to stop using expiratory-phase predictions (which are not scored) and instead force `pressure=0` wherever `u_out==1` in the final submission. This preserves your existing deterministic lookup-ensemble core logic (same predictors, same aggregation, same rounding-to-grid), but aligns post-processing with the metric. I also make the pressure-grid computation use inspiratory-only training pressures (since the metric is inspiratory-only), which improves calibration without changing the modeling approach. Everything else (data loading, feature engineering/scaler path, and CSV writing) remains intact and end-to-end.'
- What this solution (achieved 7.23589) has done: 'Your current score is far above the target (lower is better), so the smallest meaningful improvement is to make the lookup ensemble less “missy” without changing the overall approach (still deterministic lookup → 3 base predictors → mean/median/clipped-mean → pressure-grid rounding → u_out post-fix). Concretely, I keep your exact pipeline but tighten the binning used in the two sequence-aware predictors (`cum_vol` and `u_in` bins), which increases key match rate between train/test and usually reduces MAE substantially for this kind of lookup method. I also add a very small, safe extra fallback in `map_with_fallback`: when the primary lookup misses, use `(R,C,time_step,u_out)` median before dropping all the way to `(R,C)`/global; this preserves your semantics (still a fallback chain) but reduces catastrophic misses. Everything else (feature engineering/scaler path, aggregation logic, rounding/clipping, and `u_out==1 -> 0`) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 2.40617) has done: 'We need to move your MAE down toward the target (lower is better), and your current lookup-ensemble is still missing many exact keys, causing large fallback errors. I keep the same core logic (deterministic lookup predictors → mean/median/clipped mean → pressure-grid rounding → set u_out==1 to 0) but make the lookup keys more matchable by (1) using integer `time_step` index within each breath (0–79) instead of float `time_step`, and (2) switching from `.round()` binning to stable `np.floor(x/bin + 0.5)`-style integer binning to reduce boundary mismatches between train/test. I also add one additional minimal fallback level `(R,C,t_idx,u_out)` before `(R,C)` to reduce catastrophic misses without changing the overall approach. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 2.11275) has done: 'Your current MAE (2.406) is still far above the target (0.138, lower is better), so we should improve the lookup match rate without changing your core “deterministic lookup ensemble → aggregate → pressure-grid rounding → set u_out==1 to 0” logic. The most damaging mismatches now come from over-fine binning of `cum_vol` and `u_in`, which makes train/test keys miss and fall back to coarse medians; I make the bin sizes slightly coarser to increase exact-key hits (typically a large MAE drop for this approach). I also (minimally) add one more fallback level for the cum/u_in predictors that includes `t_idx` but drops `u_out` (since scoring is inspiratory-only and `u_out` can fragment keys), before falling back to `(R,C,t_idx,u_out)` and then `(R,C)`. Everything else (feature engineering/scaler path, aggregation, rounding/clipping, and CSV writing) stays intact and end-to-end.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler

np.random.seed(42)

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "../input/ventilator-pressure-prediction"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train exists:",
    os.path.exists(TRAIN_PATH),
    "| Test exists:",
    os.path.exists(TEST_PATH),
    "| Sample exists:",
    os.path.exists(SAMPLE_SUB_PATH),
)



## === cell 1
sub = pd.read_csv(SAMPLE_SUB_PATH)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def add_cum_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    df["cum_vol"] = (
        (df["u_in"].astype(np.float32) * dt).groupby(df["breath_id"]).cumsum()
    )
    return df


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["t_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)
    return df


train_df = add_cum_volume(train_df)
test_df = add_cum_volume(test_df)
train_df = add_time_index(train_df)
test_df = add_time_index(test_df)

train_insp = train_df[train_df["u_out"] == 0].copy()

keys_base = ["R", "C", "t_idx", "u_out"]
keys_rc = ["R", "C"]
keys_rct = ["R", "C", "t_idx", "u_out"]

keys_rct_no_uout = ["R", "C", "t_idx"]


def stable_bin(x: np.ndarray, bin_size: float, dtype) -> np.ndarray:
    return np.floor(x / bin_size + 0.5).astype(dtype)


cum_bin = 0.05
train_insp["_cum_bin"] = stable_bin(
    train_insp["cum_vol"].to_numpy(np.float32), cum_bin, np.int32
)
test_df["_cum_bin"] = stable_bin(
    test_df["cum_vol"].to_numpy(np.float32), cum_bin, np.int32
)

u_in_bin = 0.5
train_insp["_u_in_bin"] = stable_bin(
    train_insp["u_in"].to_numpy(np.float32), u_in_bin, np.int16
)
test_df["_u_in_bin"] = stable_bin(
    test_df["u_in"].to_numpy(np.float32), u_in_bin, np.int16
)

keys_cum = keys_base + ["_cum_bin"]
keys_uin = keys_base + ["_u_in_bin"]

grp_median_cum = train_insp.groupby(keys_cum, sort=False)["pressure"].median()
grp_median_uin = train_insp.groupby(keys_uin, sort=False)["pressure"].median()
grp_mean_cum = train_insp.groupby(keys_cum, sort=False)["pressure"].mean()

grp_median_base = train_insp.groupby(keys_base, sort=False)["pressure"].median()
grp_median_rct = train_insp.groupby(keys_rct, sort=False)["pressure"].median()

grp_median_rct_no_uout = train_insp.groupby(keys_rct_no_uout, sort=False)[
    "pressure"
].median()

rc_median = train_insp.groupby(keys_rc, sort=False)["pressure"].median()
global_median = float(train_insp["pressure"].median())


def map_with_fallback(
    df_test: pd.DataFrame,
    series_stat: pd.Series,
    keys,
    fallback_series=None,
    fallback_keys=None,
    default_value=0.0,
    fallback2_series=None,
    fallback2_keys=None,
    fallback3_series=None,
    fallback3_keys=None,
    fallback4_series=None,
    fallback4_keys=None,
):
    idx = pd.MultiIndex.from_frame(df_test[keys])
    out = series_stat.reindex(idx).to_numpy()

    if np.any(pd.isna(out)):
        if fallback_series is not None:
            idx2 = pd.MultiIndex.from_frame(df_test[fallback_keys])
            fb = fallback_series.reindex(idx2).to_numpy()
            out = np.where(pd.isna(out), fb, out)

        if np.any(pd.isna(out)) and (fallback2_series is not None):
            idx3 = pd.MultiIndex.from_frame(df_test[fallback2_keys])
            fb2 = fallback2_series.reindex(idx3).to_numpy()
            out = np.where(pd.isna(out), fb2, out)

        if np.any(pd.isna(out)) and (fallback3_series is not None):
            idx4 = pd.MultiIndex.from_frame(df_test[fallback3_keys])
            fb3 = fallback3_series.reindex(idx4).to_numpy()
            out = np.where(pd.isna(out), fb3, out)

        if np.any(pd.isna(out)) and (fallback4_series is not None):
            idx5 = pd.MultiIndex.from_frame(df_test[fallback4_keys])
            fb4 = fallback4_series.reindex(idx5).to_numpy()
            out = np.where(pd.isna(out), fb4, out)

        out = np.where(pd.isna(out), default_value, out)

    return out.astype(np.float32)


pred_0 = map_with_fallback(
    test_df,
    grp_median_cum,
    keys_cum,
    fallback_series=grp_median_base,
    fallback_keys=keys_base,
    fallback2_series=grp_median_rct_no_uout,
    fallback2_keys=keys_rct_no_uout,
    fallback3_series=grp_median_rct,
    fallback3_keys=keys_rct,
    fallback4_series=rc_median,
    fallback4_keys=keys_rc,
    default_value=global_median,
)
pred_1 = map_with_fallback(
    test_df,
    grp_median_uin,
    keys_uin,
    fallback_series=grp_median_base,
    fallback_keys=keys_base,
    fallback2_series=grp_median_rct_no_uout,
    fallback2_keys=keys_rct_no_uout,
    fallback3_series=grp_median_rct,
    fallback3_keys=keys_rct,
    fallback4_series=rc_median,
    fallback4_keys=keys_rc,
    default_value=global_median,
)
pred_2 = map_with_fallback(
    test_df,
    grp_mean_cum,
    keys_cum,
    fallback_series=grp_median_base,
    fallback_keys=keys_base,
    fallback2_series=grp_median_rct_no_uout,
    fallback2_keys=keys_rct_no_uout,
    fallback3_series=grp_median_rct,
    fallback3_keys=keys_rct,
    fallback4_series=rc_median,
    fallback4_keys=keys_rc,
    default_value=global_median,
)

pred = np.array([pred_0, pred_1, pred_2], dtype=np.float32)

print(
    "pred shape:", pred.shape, "| n_test:", len(test_df), "| submission rows:", len(sub)
)



## === cell 2
mean = np.mean(pred, axis=0)
med = np.median(pred, axis=0)
std = np.std(pred, axis=0)

print("mean/med/std:", mean.shape, med.shape, std.shape)



## === cell 3
clipped_pres = np.clip(np.vstack(pred), mean - std, mean + std)
clipped_mean = np.mean(clipped_pres, axis=0)

print("clipped_mean shape:", clipped_mean.shape)



## === cell 4
sub_mean = sub.copy()
sub_mean["pressure"] = mean
sub_mean.to_csv("submission_mean.csv", index=False)

sub_mean.head(5)



## === cell 5
sub_med = sub.copy()
sub_med["pressure"] = med
sub_med.to_csv("submission_median.csv", index=False)

sub_med.head(5)



## === cell 6
sub_clip = sub.copy()
sub_clip["pressure"] = clipped_mean
sub_clip.to_csv("submission_clipped_mean.csv", index=False)

sub_clip.head(5)




## === cell 7
def add_features(df):
    df = df.copy()
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



## === cell 8
targets = train[["pressure"]].to_numpy()
assert (
    targets.shape[0] % 80 == 0
), f"Unexpected number of rows for 80-step breaths: {targets.shape[0]}"
targets = targets.reshape(-1, 80)

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

train_cols = train.columns
test = test.reindex(columns=train_cols, fill_value=0)

print(f"train: {train.shape} \ntest: {test.shape}")



## === cell 9
scaler = RobustScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

print(f"train: {train.shape} \ntest: {test.shape} \ntargets: {targets.shape}")



## === cell 10
train_insp_pressure = (
    train_df.loc[train_df["u_out"] == 0, "pressure"].to_numpy().astype("float32")
)

P_MIN = float(np.min(train_insp_pressure))
P_MAX = float(np.max(train_insp_pressure))

uniq = np.unique(train_insp_pressure)
diffs = np.diff(uniq)
diffs = diffs[diffs > 0]
P_STEP = float(diffs.min()) if diffs.size else 0.0

print("Min pressure (insp):", P_MIN)
print("Max pressure (insp):", P_MAX)
print("Pressure step (insp):", P_STEP)
print("Unique values (insp):", uniq.shape[0])

del train_insp_pressure, uniq, diffs
gc.collect()



## === cell 11
submission = pd.read_csv(SAMPLE_SUB_PATH)

assert (
    len(submission) == pred.shape[1] == len(test_df)
), "Row count mismatch among submission/pred/test."



## === cell 12
submission["pressure"] = np.median(np.vstack(pred), axis=0)

if P_STEP > 0:
    submission["pressure"] = (
        np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
    )

submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission.loc[test_df["u_out"].to_numpy() == 1, "pressure"] = 0.0

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote: submission.csv", "| shape:", submission.shape)
