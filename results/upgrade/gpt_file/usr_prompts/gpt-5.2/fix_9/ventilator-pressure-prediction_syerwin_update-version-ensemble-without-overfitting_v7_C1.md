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

0.138236000460376

# 6. Current score

4.0055

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.00735) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/.../submission.csv` ensemble files), since that’s what prevents the notebook from running and producing any submission. Then I replace the broken `pred` ensemble with a minimal, fully self-contained baseline that uses only the provided train/test: a per-(R,C,time_step,u_in,u_out) median pressure lookup with sensible fallbacks. Finally, I keep your existing pressure-grid rounding/clipping (P_MIN/P_MAX/P_STEP) to match the competition’s discrete pressure levels and write a valid `submission.csv` file.'
- What this solution (achieved 7.54375) has done: 'Your current lookup ignores the key scoring rule (only inspiratory phase is evaluated), so it effectively learns expiratory pressures too and hurts MAE; the smallest high-impact change is to build your pressure medians using only rows where `u_out == 0` (inspiration). Then, for test rows with `u_out == 1`, we can safely output a constant (e.g., global inspiratory median) since those rows are not scored, reducing the chance of contaminating neighboring matches without changing your overall approach. I also make the merges lighter by not carrying `id` into the join frame (same predictions, less overhead), but keep your discrete pressure-grid rounding/clipping and CSV schema unchanged. These changes should move your score substantially toward the 0.138 target without altering the core “median lookup with fallbacks” logic.'
- What this solution (achieved 5.64402) has done: 'Your current approach is close to a workable baseline, but it’s leaking error through an avoidable mismatch with the metric: test predictions for inspiratory rows should come from inspiration-only statistics, and expiratory rows should not influence any fallbacks used for inspiratory rows. I keep the exact same “multi-level median lookup with fallbacks” core logic, but (1) build lookups on a *quantized* version of `time_step` to reduce join sparsity and improve match rate, and (2) add a slightly stronger fallback key that uses cumulative area (`u_in` integral) that stays within your lookup-based approach. These are minimal, metric-aligned changes that should significantly reduce MAE toward your 0.138 target without changing modeling/training (there is none) or post-processing semantics (pressure grid snapping remains). The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 4.0055) has done: 'Your current score (5.644) is far above the target (0.138) on a lower-is-better MAE, so we should improve but keep your lookup-based core intact. The biggest avoidable error source is join sparsity from using raw float `u_in` and too-fine `time_step_q`; I keep the exact same multi-level median lookup/fallback structure but quantize `u_in` (and align `time_step_q` to the dataset’s 0.03 grid) so keys match much more often. I also ensure all fallbacks are computed from inspiration-only data consistently and keep your final pressure-grid snapping unchanged. These are minimal, metric-aligned changes that should substantially reduce MAE toward the target without changing the approach.'
- What this solution (achieved 4.0055) has done: 'Your current MAE (4.0055, lower-is-better) is still far above the target (0.1382), so we should improve while keeping your “inspiration-only multi-level median lookup with fallbacks + pressure-grid snapping” core logic unchanged. The biggest remaining mismatch is that we’re keying on the raw `u_out` in the most specific lookup, even though all LUTs are built from `u_out==0` only; dropping `u_out` from the most-specific key increases match rate for inspiratory rows without changing the approach. I also make the final fallback for `u_out==1` output `0.0` (still unscored) rather than the inspiratory median, to avoid any accidental coupling and keep inspiratory statistics “clean”. Everything else (quantization, area_cum_q fallback, pressure snapping, and CSV schema) stays the same.'
- What this solution (achieved 4.0055) has done: 'We need to improve a lot (MAE 4.0055 vs target 0.1382; lower is better), while keeping your core “inspiration-only multi-level median lookup with fallbacks + pressure grid snapping” intact. The biggest remaining error source is that the lookup keys are still too sparse: using raw `u_in_q` (0.5) and `time_step_q` alone misses the important dynamic dependence on recent valve inputs, so many inspiratory rows fall back to coarse/global medians. I keep the same exact lookup-and-merge structure, but add one extra *slightly-more-informative* lookup level using quantized `u_in` plus its lag-1 (both already derivable without changing the approach), and use it only as an intermediate fallback between the most-specific LUT and the coarser ones. This should materially increase match rate for inspiratory rows and move MAE substantially toward your target, while keeping expiratory rows fixed at 0.0 (unscored) and preserving your final pressure snapping/clipping.'
- What this solution (achieved 4.0055) has done: 'Your MAE (4.0055, lower-is-better) is still far above the 0.1382 target, so we should improve while keeping your lookup/merge + fallback + pressure-grid snapping core intact. The smallest high-impact fix is to stop “polluting” the final submission with arbitrary values for `u_out==1` (expiratory, unscored): instead, reuse the most reliable inspiratory predictions but only overwrite the unscored rows after snapping (so they don’t affect inspiratory rows at all). Additionally, because `time_step` is already on a 0.03 grid, we keep that quantization but make `area_cum_q` quantization consistent with `UIN_Q` and `TS_Q` to improve key stability/match rate without changing the approach. Everything else (inspiration-only LUTs, the same fallback ordering, and the same submission schema) stays the same.'
- What this solution (achieved 4.0055) has done: 'We need to move your MAE down substantially (4.0055 → 0.1382; lower is better), but without changing the overall “inspiration-only multi-level median lookup with fallbacks + pressure-grid snapping” approach. The biggest remaining bottleneck is still key sparsity: even with `u_in_q` and `u_in_q_lag1`, many inspiratory rows won’t match and fall back to coarse medians, so I add one more intermediate lookup based on a quantized first-difference (`du_in_q = u_in_q - u_in_q_lag1`) which captures dynamics while staying within your existing lookup/merge paradigm. I also fix a subtle inconsistency: your `uout_t` LUT is built from inspiration-only data but keyed by `u_out`, so the `u_out==1` group is empty and that fallback can never help—switching it to be `time_step_q`-only makes it a real (still inspiration-only) fallback. Everything else (quantization, fallback ordering concept, pressure snapping/clipping, and CSV schema/path) remains intact.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    RobustScaler,
)  # kept (unused) to preserve original environment assumptions

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"



## === cell 2
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

print(train_df.shape, test_df.shape, submission.shape)
print(train_df.columns.tolist())




## === cell 3
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




## === cell 4
pressure_arr = train_df["pressure"].to_numpy().astype("float32").reshape(-1, 1)

P_MIN = float(np.min(pressure_arr))
P_MAX = float(np.max(pressure_arr))

unique_p = np.unique(pressure_arr.ravel())
if unique_p.size >= 2:
    diffs = np.diff(unique_p)
    P_STEP = float(diffs[diffs > 0].min())
else:
    P_STEP = 0.0

print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Pressure step: {P_STEP}")
print(f"Unique values: {unique_p.shape[0]}")

del pressure_arr
gc.collect()



## === cell 5
TS_Q = 0.03
UIN_Q = 0.5


def _add_lookup_helpers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_step_q"] = (
        np.round(df["time_step"].to_numpy(dtype="float32") / TS_Q) * TS_Q
    ).astype("float32")

    df["u_in_q"] = (
        np.round(df["u_in"].to_numpy(dtype="float32") / UIN_Q) * UIN_Q
    ).astype("float32")

    df["u_in_q_lag1"] = (
        pd.Series(df["u_in_q"], index=df.index)
        .groupby(df["breath_id"])
        .shift(1)
        .fillna(0.0)
    ).astype("float32")

    df["du_in_q"] = (df["u_in_q"] - df["u_in_q_lag1"]).astype("float32")

    dt = (
        pd.Series(df["time_step_q"], index=df.index)
        .groupby(df["breath_id"])
        .diff()
        .fillna(0.0)
        .to_numpy(dtype="float32")
    )
    u_in = df["u_in_q"].to_numpy(dtype="float32")
    df["area_cum"] = (
        pd.Series((u_in * dt), index=df.index)
        .groupby(df["breath_id"])
        .cumsum()
        .astype("float32")
    )

    AREA_Q = float(UIN_Q * TS_Q)  # 0.015 with current settings
    df["area_cum_q"] = (
        np.round(df["area_cum"].to_numpy(dtype="float32") / AREA_Q) * AREA_Q
    ).astype("float32")
    return df


for col in ["R", "C", "u_out"]:
    train_df[col] = train_df[col].astype(np.int16)
    test_df[col] = test_df[col].astype(np.int16)

train_df = _add_lookup_helpers(train_df)
test_df = _add_lookup_helpers(test_df)

train_insp = train_df[train_df["u_out"] == 0].copy()

keys_full = ["R", "C", "time_step_q", "u_in_q"]
keys_full_lag = ["R", "C", "time_step_q", "u_in_q", "u_in_q_lag1"]

keys_full_du = ["R", "C", "time_step_q", "u_in_q", "du_in_q"]

keys_rc_t = ["R", "C", "time_step_q"]

keys_t = ["time_step_q"]

keys_rc_t_area = ["R", "C", "time_step_q", "area_cum_q"]

lut_full = train_insp.groupby(keys_full, sort=False)["pressure"].median()
lut_full_lag = train_insp.groupby(keys_full_lag, sort=False)["pressure"].median()
lut_full_du = train_insp.groupby(keys_full_du, sort=False)["pressure"].median()
lut_rc_t = train_insp.groupby(keys_rc_t, sort=False)["pressure"].median()
lut_t = train_insp.groupby(keys_t, sort=False)["pressure"].median()
lut_rc_t_area = train_insp.groupby(keys_rc_t_area, sort=False)["pressure"].median()

global_insp_median = float(train_insp["pressure"].median())

tmp_full = test_df[keys_full].copy()
tmp_full_lag = test_df[keys_full_lag].copy()
tmp_full_du = test_df[keys_full_du].copy()
tmp_rc_t = test_df[keys_rc_t].copy()
tmp_t = test_df[keys_t].copy()
tmp_rc_t_area = test_df[keys_rc_t_area].copy()

pred_full = tmp_full.merge(
    lut_full.rename("p_full").reset_index(), on=keys_full, how="left"
)["p_full"].to_numpy()

pred_full_lag = tmp_full_lag.merge(
    lut_full_lag.rename("p_full_lag").reset_index(), on=keys_full_lag, how="left"
)["p_full_lag"].to_numpy()

pred_full_du = tmp_full_du.merge(
    lut_full_du.rename("p_full_du").reset_index(), on=keys_full_du, how="left"
)["p_full_du"].to_numpy()

pred_rc_t = tmp_rc_t.merge(
    lut_rc_t.rename("p_rc_t").reset_index(), on=keys_rc_t, how="left"
)["p_rc_t"].to_numpy()

pred_rc_t_area = tmp_rc_t_area.merge(
    lut_rc_t_area.rename("p_rc_t_area").reset_index(), on=keys_rc_t_area, how="left"
)["p_rc_t_area"].to_numpy()

pred_t = tmp_t.merge(lut_t.rename("p_t").reset_index(), on=keys_t, how="left")[
    "p_t"
].to_numpy()

pred = pred_full.copy()

mask = np.isnan(pred)
pred[mask] = pred_full_lag[mask]

mask = np.isnan(pred)
pred[mask] = pred_full_du[mask]

mask = np.isnan(pred)
pred[mask] = pred_rc_t[mask]
mask = np.isnan(pred)
pred[mask] = pred_rc_t_area[mask]
mask = np.isnan(pred)
pred[mask] = pred_t[mask]
mask = np.isnan(pred)
pred[mask] = global_insp_median

pred = pred.astype("float32")

u_out_mask = test_df["u_out"].to_numpy() == 1

print("Pred stats:", float(np.min(pred)), float(np.max(pred)), float(np.mean(pred)))

del train_insp, lut_full, lut_full_lag, lut_full_du, lut_rc_t, lut_t, lut_rc_t_area
gc.collect()



## === cell 6
submission = submission.copy()
submission["pressure"] = pred

if P_STEP > 0:
    submission["pressure"] = (
        np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
    )

submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission.loc[u_out_mask, "pressure"] = global_insp_median
if P_STEP > 0:
    submission.loc[u_out_mask, "pressure"] = (
        np.round((submission.loc[u_out_mask, "pressure"] - P_MIN) / P_STEP) * P_STEP
        + P_MIN
    )
submission.loc[u_out_mask, "pressure"] = np.clip(
    submission.loc[u_out_mask, "pressure"], P_MIN, P_MAX
)

submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 7
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id", "pressure"]
assert chk.shape[0] == submission.shape[0]
assert chk["pressure"].isna().sum() == 0
print("Wrote submission.csv with shape:", chk.shape)
print(chk.head())
