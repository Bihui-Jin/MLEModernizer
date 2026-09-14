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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.1578757127431918

# 6. Current score

8.46679

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The notebook fails because it tries to read a non-existent external dataset (`ensemble-of-public-submissions`), so `df_sub` is never created and all downstream cells crash. To make it run end-to-end in your environment, I replace that input with the competition’s provided `sample_submission.csv` (same required columns) so we always have a valid submission template. Since this baseline has no model predictions available, I keep the original core “round-to-known-pressures” logic but apply it to a constant initial prediction (0), which produces a valid submission file (score be poor but the pipeline work). I also remove IPython magics (`%matplotlib inline`, `%%time`) that can error in script execution and ensure the output file is named `submission.csv`.'
- What this solution (achieved 9.01216) has done: 'Your current 17.65 MAE comes from effectively predicting a constant (0) then rounding, which is far from the target 0.158, so we need a real (but minimal) predictive signal. To keep core logic intact, I only change how the initial `pressure` values are filled in `df_sub`: use a lightweight per‑(R,C,time_step,u_out) median lookup built from train (no model/loops added), then apply your existing probability-weighted rounding exactly as before. This stays aligned with the metric by learning typical inspiratory pressures at each time step and valve state, while remaining fast (<600s) and producing a valid `submission.csv`. I also keep the original rounding baseline cells, but make them operate on these filled predictions rather than zeros.'
- What this solution (achieved 7.53955) has done: 'Your current score (9.01216 MAE) is far above the target (0.1579), so we need a meaningful but still lightweight predictive improvement while keeping your “median lookup + rounding-to-known-pressures” core logic intact. The smallest high-impact fix is to make the median lookup respect the competition’s scoring rule (only inspiratory phase is scored) by building the medians using only `u_out == 0` rows from train, then using the same medians for all test rows. Additionally, using a more breath-specific key by adding `u_in` (rounded to 1 decimal to keep the map size manageable) typically improves pressure estimation without changing the approach. Everything else (rounding functions, submission format, paths) stays the same.'
- What this solution (achieved 7.84143) has done: 'Your current approach is a fast “train median lookup → fill test predictions → round to known pressure classes”, but it’s leaving accuracy on the table because the mapping is built only on inspiratory rows yet is still keyed by `u_out` (which becomes constant 0) and doesn’t use any within-breath dynamics. I keep the same core logic, but add a minimal, cheap breath-dynamics feature: cumulative inspired volume (`u_in` integrated over time) and its interaction with `(R,C)`, then rebuild the same median lookup using these keys (still only from `u_out==0` to match the metric). This typically reduces MAE substantially while preserving your overall pipeline, rounding logic, and submission format. I also keep the original fallbacks, just updated to include the new feature where appropriate, and ensure everything still runs under the 600s constraint.'
- What this solution (achieved 8.23944) has done: 'Your current MAE (7.84) is far worse than the target (0.158), so we should improve predictions while keeping your same “train median lookup → fill test → rounding to known pressures” pipeline intact. The biggest issue is that your lookup keys include raw `time_step` and fine rounded `u_in/cum_u_in`, which often won’t match between train/test due to floating precision and over-fragmentation, causing heavy fallback usage and poor accuracy. I make a minimal, metric-aligned adjustment: compute an integer time index per breath (0..79) and use that instead of float `time_step` in the grouping keys (same logic, just a stable join key). I also slightly coarsen `cum_u_in` rounding (keeping the same feature) to reduce sparsity and increase exact key hits, which should move the score substantially toward the target while preserving your overall approach and producing `submission.csv`.'
- What this solution (achieved 8.2365) has done: 'I fix the root cause of the crash in the feature→median-lookup pipeline: `test_pred` is created from a frame that already contains `R,C,t_idx,u_in_r,cum_u_in_norm_r` and then you add the same columns again, producing duplicate column labels and making `merge()` fail with “column label 'R' is not unique”. Then I ensure `test_pred` is always defined so the later rounding/probability-rounding cells run, and keep the original rounding logic intact. Finally, I make the submission-writing cells robust so `submission.csv` always contains exactly `id,pressure` and passes the final validation.'
- What this solution (achieved 8.35376) has done: 'Your score is far above the target (lower is better), and the most likely reason is that the current median-lookup keys are too sparse (especially `cum_u_in_r` and `cum_u_in_norm_r`), causing most test rows to miss the map and fall back to very coarse medians. I keep the exact same pipeline (breath dynamics features → median lookups with fallbacks → rounding to known pressure classes), but make two minimal changes to increase key hit-rate: (1) use stable float32 types and quantize the cumulative features via binning (rounded integers) rather than fragile float rounding, and (2) add one more very-cheap fallback keyed on `(R,C,t_idx,cum_bin)` to preserve dynamics without overfragmenting. This should materially reduce MAE while preserving your approach, runtime, and the final `submission.csv` format. No model, training loop, or loss changes are introduced.'
- What this solution (achieved 8.48952) has done: 'Your current MAE (8.35) is far worse than the target (0.158), so we should improve prediction accuracy while keeping your exact “median lookup with fallbacks → round to known pressure classes” pipeline intact. The biggest low-risk gain is to reduce sparsity/mismatch in the main lookup key: the combination of `u_in_r` + both cumulative bins is too specific, so most rows likely miss and fall back to coarse medians. I keep all your existing features and fallbacks, but (1) coarsen the cumulative bins (wider bins) and (2) add one additional mid-granularity fallback keyed on `(R,C,t_idx,u_in_r,cum_u_in_bin)` to preserve dynamics without requiring the normalized bin too. This is minimal, should increase key hit-rate substantially, and preserves the same evaluation semantics and submission format.'
- What this solution (achieved 8.4553) has done: 'Your current MAE (8.48952) is far worse than the target (0.1579), so we should improve accuracy while keeping the same “median lookup with fallbacks → rounding to known pressure classes” core pipeline. The biggest minimal fix is to reduce key sparsity/mismatch by removing the over-fragmenting normalized cumulative bin from the primary lookup (keep it only as a fallback), and to stabilize the breath integration by using a constant `dt` (time_step is essentially uniform per breath in this dataset). This increases exact key hit-rate in the main median map and reduces reliance on coarse fallbacks, which should move MAE down substantially. All file paths, output schema, rounding logic, and overall approach remain the same.'
- What this solution (achieved 8.42262) has done: 'Your current MAE (8.4553, lower is better) is far above the target (0.1579), so we need a real improvement while keeping your same “median lookup with fallbacks → round to known pressure classes” pipeline intact. The biggest low-risk issue is that `u_in_r` and the cumulative bins are currently computed on absolute `u_in`, which can’t capture breath dynamics as well as using changes/lagged controls; adding a single lag feature and using it only in the primary lookup (keeping your existing fallbacks unchanged) increases key match quality without changing the overall approach. I also align the median-map construction with the metric by strictly using inspiratory rows (already done) and ensure joins are stable by casting join keys to consistent dtypes across train/test. Everything still runs end-to-end, keeps the rounding logic unchanged, and writes a valid `submission.csv`.'
- What this solution (achieved 8.43576) has done: 'Your current score is far above the target (lower is better), so we need a real accuracy gain while preserving your core “median lookup with fallbacks → rounding to known pressure classes” pipeline. The smallest high-impact adjustment is to make the lookup keys less sparse and more physically aligned by replacing the absolute `u_in_prev` key with a discretized delta-control (`du_in = u_in - u_in_prev`) and using a slightly coarser `cum_u_in` bin; this increases exact key hit-rate without changing the overall approach. I also restrict the frequency-based rounding thresholds to inspiratory rows (`u_out==0`) to better match the metric, while keeping your probability-weighted rounding logic unchanged. All paths, training approach (groupby medians), and submission-writing semantics remain the same, and the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 8.46517) has done: 'Your current MAE is far above the target (lower is better), and the main minimal lever we have without changing your overall “median lookup → fallback → round-to-known-pressures” pipeline is to increase exact key hit-rate and reduce sparsity. I keep the same features and groupby-median approach, but make the primary lookup key less fragmented by dropping `du_in_r` from the main key (keep it as an extra, higher-granularity fallback instead). I also bin `u_in` to a stable integer (`u_in_bin`) rather than float rounding to avoid float-merge mismatches and further improve join stability. Everything else (inspiratory-only training, fallbacks, probability-weighted rounding, submission format/path) stays the same.'
- What this solution (achieved 8.71264) has done: 'Your current MAE (8.465) is far worse than the target (0.158, lower is better), so we need a real accuracy improvement while keeping your same “median lookup with fallbacks → round to known pressure classes” core pipeline intact. The largest minimal win is to avoid applying inspiratory-trained predictions to expiratory rows: the metric ignores expiratory phase, so we can safely set `pressure=0` when `u_out==1` in test, which typically improves public MAE substantially without changing modeling logic. Next, your lookups are likely too sparse/mismatched due to using raw cumulative bins across diverse breaths; a tiny, stable fix is to add a per-(R,C,t_idx) “median u_in” normalization and use a coarser, bounded `u_in_bin` to increase exact key hit-rate. Finally, we keep your probability-weighted rounding untouched, just applied after these more reliable base predictions, and still write a valid `submission.csv`.'
- What this solution (achieved 8.46679) has done: 'Your current MAE (8.71) is far worse than the target (0.158, lower is better), so we need a real accuracy gain while keeping your exact “median lookup with fallbacks → rounding to known pressures” pipeline intact. The biggest minimal fix is to stop forcing `u_out==1` test rows to `pressure=0`, which is almost certainly wrong and can heavily hurt MAE even if expiratory is not scored (because many `u_out==1` rows can still be in the inspiratory-scored segment for some breaths). Next, to improve lookup hit-rate without changing the approach, we add one additional low-sparsity fallback keyed on just `(R,C,t_idx,cum_u_in_bin)` before dropping to very coarse fallbacks. Everything else (features, groupby-median training, probability-weighted rounding, submission writing) remains the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"



## === cell 1
import matplotlib.pyplot as plt



## === cell 2
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_sub = pd.read_csv(SAMPLE_SUB_PATH)

if list(df_sub.columns) != ["id", "pressure"]:
    df_sub = df_sub[["id", "pressure"]].copy()

if not df_sub["id"].equals(df_test["id"]):
    df_sub = df_test[["id"]].copy()
    df_sub["pressure"] = 0.0


def add_breath_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    df["time_step"] = df["time_step"].astype(np.float32)
    df["u_in"] = df["u_in"].astype(np.float32)

    dt_const = np.float32(0.03)
    df["u_in_dt"] = (df["u_in"] * dt_const).astype(np.float32)

    df["cum_u_in"] = (
        df.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
    )

    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    df["u_in_prev"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["du_in"] = (df["u_in"] - df["u_in_prev"]).astype(np.float32)
    return df


df_train = add_breath_dynamics_features(df_train)
df_test_feat = add_breath_dynamics_features(df_test)

df_train_insp = df_train[df_train["u_out"] == 0].copy()

UIN_BIN_WIDTH = np.float32(1.0)
df_train_insp["u_in_bin"] = np.rint(df_train_insp["u_in"] / UIN_BIN_WIDTH).astype(
    np.int16
)
df_test_feat["u_in_bin"] = np.rint(df_test_feat["u_in"] / UIN_BIN_WIDTH).astype(
    np.int16
)

DUIN_BIN_WIDTH = np.float32(1.0)
df_train_insp["du_in_bin"] = np.rint(df_train_insp["du_in"] / DUIN_BIN_WIDTH).astype(
    np.int16
)
df_test_feat["du_in_bin"] = np.rint(df_test_feat["du_in"] / DUIN_BIN_WIDTH).astype(
    np.int16
)

for _df in (df_train_insp, df_test_feat):
    _df["R"] = _df["R"].astype(np.int16)
    _df["C"] = _df["C"].astype(np.int16)
    _df["t_idx"] = _df["t_idx"].astype(np.int16)
    _df["u_out"] = _df["u_out"].astype(np.int8)

rc_stats = (
    df_train_insp.groupby(["R", "C", "t_idx"], sort=False)
    .agg(cum_u_in_med_rc_t=("cum_u_in", "median"), u_in_med_rc_t=("u_in", "median"))
    .reset_index()
)
df_train_insp = df_train_insp.merge(rc_stats, on=["R", "C", "t_idx"], how="left")
df_test_feat = df_test_feat.merge(rc_stats, on=["R", "C", "t_idx"], how="left")

df_train_insp["cum_u_in_norm"] = (
    df_train_insp["cum_u_in"]
    / df_train_insp["cum_u_in_med_rc_t"].replace(0.0, 1.0).fillna(1.0)
).astype(np.float32)
df_test_feat["cum_u_in_norm"] = (
    df_test_feat["cum_u_in"]
    / df_test_feat["cum_u_in_med_rc_t"].replace(0.0, 1.0).fillna(1.0)
).astype(np.float32)

df_train_insp["u_in_norm"] = (
    df_train_insp["u_in"] / df_train_insp["u_in_med_rc_t"].replace(0.0, 1.0).fillna(1.0)
).astype(np.float32)
df_test_feat["u_in_norm"] = (
    df_test_feat["u_in"] / df_test_feat["u_in_med_rc_t"].replace(0.0, 1.0).fillna(1.0)
).astype(np.float32)

CUMUIN_BIN_WIDTH = np.float32(3.0)
CUMNORM_BIN_WIDTH = np.float32(0.06)
UINNORM_BIN_WIDTH = np.float32(0.05)

df_train_insp["cum_u_in_bin"] = np.rint(
    df_train_insp["cum_u_in"] / CUMUIN_BIN_WIDTH
).astype(np.int32)
df_test_feat["cum_u_in_bin"] = np.rint(
    df_test_feat["cum_u_in"] / CUMUIN_BIN_WIDTH
).astype(np.int32)

df_train_insp["cum_u_in_norm_bin"] = np.rint(
    df_train_insp["cum_u_in_norm"] / CUMNORM_BIN_WIDTH
).astype(np.int32)
df_test_feat["cum_u_in_norm_bin"] = np.rint(
    df_test_feat["cum_u_in_norm"] / CUMNORM_BIN_WIDTH
).astype(np.int32)

df_train_insp["u_in_norm_bin"] = np.rint(
    df_train_insp["u_in_norm"] / UINNORM_BIN_WIDTH
).astype(np.int32)
df_test_feat["u_in_norm_bin"] = np.rint(
    df_test_feat["u_in_norm"] / UINNORM_BIN_WIDTH
).astype(np.int32)

train_key_cols = ["R", "C", "t_idx", "u_in_bin", "cum_u_in_bin"]
test_key_cols = ["R", "C", "t_idx", "u_in_bin", "cum_u_in_bin"]

median_map = (
    df_train_insp.groupby(train_key_cols, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure"})
)

median_fallback_du = (
    df_train_insp.groupby(
        ["R", "C", "t_idx", "u_in_bin", "du_in_bin", "cum_u_in_bin"], sort=False
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback_du"})
)

median_fallback = (
    df_train_insp.groupby(
        ["R", "C", "t_idx", "u_in_bin", "cum_u_in_norm_bin"], sort=False
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback"})
)

median_fallback_uinnorm = (
    df_train_insp.groupby(
        ["R", "C", "t_idx", "u_in_norm_bin", "cum_u_in_bin"], sort=False
    )["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback_uinnorm"})
)

median_fallback2 = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback2"})
)

median_fallback2c = (
    df_train_insp.groupby(["R", "C", "t_idx", "cum_u_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback2c"})
)

median_fallback2b = (
    df_train_insp.groupby(["R", "C", "t_idx", "cum_u_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback2b"})
)

median_fallback3 = (
    df_train_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback3"})
)

median_fallback4 = (
    df_train_insp.groupby(["t_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_pressure_fallback4"})
)

test_pred = df_test_feat[
    [
        "id",
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_bin",
        "u_in_norm_bin",
        "du_in_bin",
        "cum_u_in_bin",
        "cum_u_in_norm_bin",
    ]
].merge(median_map, on=test_key_cols, how="left")

test_pred = test_pred.merge(
    median_fallback_du,
    on=["R", "C", "t_idx", "u_in_bin", "du_in_bin", "cum_u_in_bin"],
    how="left",
)

test_pred = test_pred.merge(
    median_fallback, on=["R", "C", "t_idx", "u_in_bin", "cum_u_in_norm_bin"], how="left"
)

test_pred = test_pred.merge(
    median_fallback_uinnorm,
    on=["R", "C", "t_idx", "u_in_norm_bin", "cum_u_in_bin"],
    how="left",
)

test_pred = test_pred.merge(
    median_fallback2, on=["R", "C", "t_idx", "u_in_bin"], how="left"
)

test_pred = test_pred.merge(
    median_fallback2c, on=["R", "C", "t_idx", "cum_u_in_bin"], how="left"
)

test_pred = test_pred.merge(
    median_fallback2b, on=["R", "C", "t_idx", "cum_u_in_bin"], how="left"
)
test_pred = test_pred.merge(median_fallback3, on=["R", "C", "t_idx"], how="left")
test_pred = test_pred.merge(median_fallback4, on=["t_idx"], how="left")

global_median = float(df_train_insp["pressure"].median())
test_pred["pred_pressure"] = (
    test_pred["pred_pressure"]
    .fillna(test_pred["pred_pressure_fallback_du"])
    .fillna(test_pred["pred_pressure_fallback"])
    .fillna(test_pred["pred_pressure_fallback_uinnorm"])
    .fillna(test_pred["pred_pressure_fallback2"])
    .fillna(test_pred["pred_pressure_fallback2c"])
    .fillna(test_pred["pred_pressure_fallback2b"])
    .fillna(test_pred["pred_pressure_fallback3"])
    .fillna(test_pred["pred_pressure_fallback4"])
    .fillna(global_median)
)


df_sub = df_test[["id"]].copy()
df_sub["pressure"] = test_pred["pred_pressure"].astype(float).to_numpy()



## === cell 3
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)



## === cell 4
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )




## === cell 5
df_sub["pressure"] = df_sub["pressure"].astype(float).apply(find_nearest)



## === cell 6
df_sub.head()



## === cell 7
df_sub.to_csv("submission.csv", index=False)
submission_round_LB154 = df_sub.copy()



## === cell 8
assert submission_round_LB154.shape[0] == df_test.shape[0]
assert list(submission_round_LB154.columns) == ["id", "pressure"]
assert (
    submission_round_LB154["id"].is_monotonic_increasing
    or submission_round_LB154["id"].nunique() == submission_round_LB154.shape[0]
)



## === cell 9
pressure_freq = df_train_insp["pressure"].value_counts().to_frame()
pressure_freq["freq"] = df_train_insp["pressure"].value_counts(normalize=True).values
pressure_freq = pressure_freq.sort_index().reset_index()
pressure_freq.columns = ["pressure", "count", "freq"]



## === cell 10
pressure_freq.head()



## === cell 11
pressure_freq["pressure_pre"] = pressure_freq["pressure"].shift(1)
pressure_freq["pressure_step"] = (
    pressure_freq["pressure"] - pressure_freq["pressure_pre"]
)

pressure_freq["freq_pre"] = pressure_freq["freq"].shift(1)
pressure_freq["freq_relative_pct"] = pressure_freq["freq"] / (
    pressure_freq["freq"] + pressure_freq["freq_pre"]
)
pressure_freq["pressure_prob"] = (
    pressure_freq["pressure_pre"]
    + pressure_freq["pressure_step"] * pressure_freq["freq_relative_pct"]
)



## === cell 12
pressure_freq["pressure_half"] = (
    pressure_freq["pressure_pre"] + pressure_freq["pressure"]
) / 2
plt.figure(figsize=(10, 6))
(pressure_freq["pressure_prob"] - pressure_freq["pressure_half"]).plot()
plt.title("Probability-weighted rounding threshold offset")
plt.tight_layout()
plt.show()



## === cell 13
total_pressures_len = len(sorted_pressures)


def find_nearest_prob(prediction: float) -> float:
    """
    Probability weighted rounding.
    Keeps the original notebook logic; selects rounding cutpoint based on empirical class frequencies.
    """
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    cut_val = pressure_freq["pressure_prob"].iloc[insert_idx]
    if pd.isna(cut_val):
        cut_val = (lower_val + upper_val) / 2.0
    return float(lower_val if prediction < cut_val else upper_val)




## === cell 14
df_sub = df_test[["id"]].copy()
df_sub["pressure"] = test_pred["pred_pressure"].astype(float).to_numpy()
df_sub["pressure"] = df_sub["pressure"].astype(float).apply(find_nearest_prob)



## === cell 15
df_sub.to_csv("submission.csv", index=False)
submission_prob = df_sub.copy()



## === cell 16
submission_prob.head()



## === cell 17
assert os.path.exists("submission.csv"), "submission.csv was not created."



## === cell 18
y_old = submission_round_LB154["pressure"]
y_new = submission_prob["pressure"]



## === cell 19
(y_new - y_old).describe().round(6)



## === cell 20
int((y_new > y_old).sum()), float((y_new > y_old).mean())



## === cell 21
int((y_new < y_old).sum()), float((y_new < y_old).mean())



## === cell 22
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == [
    "id",
    "pressure",
], f"Bad submission columns: {sub.columns.tolist()}"
assert (
    sub.shape[0] == df_test.shape[0]
), f"Bad submission rows: {sub.shape[0]} vs {df_test.shape[0]}"
sub["pressure"] = sub["pressure"].astype(float)
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape, sub.head(3).to_dict(orient="records"))
