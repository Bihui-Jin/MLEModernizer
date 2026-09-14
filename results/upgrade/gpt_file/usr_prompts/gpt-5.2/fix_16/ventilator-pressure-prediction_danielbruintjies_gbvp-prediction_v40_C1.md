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

0.1490965840003578

# 6. Current score

5.85392

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.24541) has done: 'I remove the failing internet download and the unused imports that trigger the protobuf/TensorFlow `MessageFactory.GetPrototype` crash, since the current pipeline errors before any submission is written. Because the code depends on external BiLSTM prediction files that are not present, I replace them with an in-notebook baseline that creates the missing `bilstm_pred` column from available features (minimal change to keep downstream feature engineering intact). I also fix the `add_features()` index-alignment bug caused by `groupby().apply()` returning a misaligned index by switching those assignments to `transform()` (same computation intent, but stable). Finally, I ensure a valid `submission.csv` is produced with `id,pressure` and aligned row order.'
- What this solution (achieved 4.24541) has done: 'I fix the `add_features()` crash by removing the fragile `groupby().apply(...)->to_list()->np.concatenate()` pattern that sometimes yields scalars (0‑d arrays) and instead compute those per-breath statistics with a stable `groupby().transform()`/mapping approach that preserves row alignment. This change is purely a bug fix: it keeps the same feature intent (constant per breath, repeated for all timesteps) while ensuring the pipeline runs end-to-end. I also make the `time_at_u_out` / `area_at_u_out` computation use `groupby().transform()` to avoid index reindexing pitfalls. The rest of the logic (feature set, median-based prediction, pressure clipping/rounding, and submission writing) is preserved so score changes only come from correctness/stability improvements.'
- What this solution (achieved 4.24541) has done: 'Your current 4.245 MAE is mainly because you force all `u_out==1` predictions to `0.0`, which is far outside the true pressure range and creates huge errors on many inspiratory (scored) rows where `u_out` is still 1. To move the score toward the 0.149 target with minimal changes and identical modeling intent, I remove that hard override and instead keep the same median lookup baseline while only masking predictions to zero for the *unscored expiratory phase*, defined as timesteps after the first `u_out==1` within each breath. I compute this mask using the existing `time_at_u_out` concept but in a lightweight, stable way (groupby transform) on `test_tmp` only. Everything else (feature engineering, median mapping, clipping/rounding, submission format/path) stays the same.'
- What this solution (achieved 4.04781) has done: 'Your current MAE is far from the 0.149 target mainly because you set predictions to `0.0` for all timesteps after the first `u_out==1`, but Kaggle’s metric still scores some inspiratory timesteps even when `u_out==1`; this creates large errors. With minimal change and identical baseline intent, I remove the expiratory zeroing override entirely (keep the same median lookup + pressure-grid rounding). I also align the median lookup to the competition scoring regime by building the median map using only inspiratory rows (`u_out==0`) from train, while keeping a safe fallback to the global median. Everything else (feature engineering, clustering, submission format/path) remains unchanged and it still write `submission.csv`.'
- What this solution (achieved 5.96612) has done: 'Your current 4.04781 MAE is dominated by underfitting from the very coarse median-lookup key (binned `time_step` and `u_in`), which collapses distinct trajectories into the same bucket and yields large errors. To move the score substantially closer to the 0.149 target while keeping the same “median table lookup + pressure-grid rounding” core logic, I (1) remove the binning and instead key the median map by the exact discrete values that actually repeat in this dataset (`R,C,step_index,u_in,u_out`), and (2) use a tiny hierarchical fallback (drop `u_in` then drop `step_index`) to avoid defaulting too often to the global median. This keeps the same prediction approach (groupby median lookup) but makes it much better aligned to the data granularity and should reduce MAE by a large margin. The rest of your pipeline (feature engineering cells, pressure clipping/rounding, and submission writing) remains intact and it still produce a valid `submission.csv`.'
- What this solution (achieved 5.85392) has done: 'Your current score (5.966) is far worse than the target (0.149), so we need a real improvement while keeping your “median table lookup + pressure-grid rounding” core logic. The biggest issue is that your lookup key uses raw float `u_in` values, so almost all test rows miss the median table and fall back to coarse medians/global median. I keep the same hierarchical median approach, but make the key match by quantizing `u_in` to the dataset’s natural resolution (3 decimals) and by using `u_out`-specific scored rows (train `u_out==0`) for medians, with the same fallbacks. This is a minimal change localized to the lookup cell and should move MAE substantially toward the target without altering the rest of your pipeline or submission semantics.'
- What this solution (achieved 5.96602) has done: 'Your current MAE (5.85392) is far above the target (0.1491), and the biggest score-killer in your lookup baseline is low key-match rate caused by float mismatches and an O(N) Python loop that also makes it hard to iterate on better keys. I keep your same “hierarchical per-row median table lookup + pressure-grid rounding” core logic, but (1) build the lookup on a more stable `u_in` discretization derived from the train data itself (so test keys actually hit), and (2) replace the slow per-row loop with vectorized `merge`-based joins that preserve identical semantics (same hierarchy and fallbacks). This should substantially reduce fallback-to-global-median frequency and move MAE sharply toward the target without changing the overall approach. The rest of the pipeline (feature engineering, pressure clipping/rounding, and writing `submission.csv`) is kept intact.'
- What this solution (achieved 5.96602) has done: 'Your current MAE (5.966) is far above the target (0.149), so we need a real but still minimal change that keeps your “median lookup table + pressure-grid rounding” core logic. The largest remaining score-killer is that the median tables are built from `train_scored = train[u_out==0]`, but then you still key/merge by `u_out` (always 0 in the medians), causing many test rows with `u_out==1` to miss and fall back to coarse/global medians. I keep the same hierarchical lookup approach, but (1) build medians from all rows (so both `u_out` values exist) and (2) add a scored-like fallback for `u_out==1` test rows by also merging a “u_out-agnostic” table at the same hierarchy levels; this improves key hit-rate without changing the approach. Everything else (feature pipeline cells, clipping/rounding, submission writing) remains the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 5.85392) has done: 'Your current MAE (5.966) is far above the target (0.149), so we need a meaningful improvement while keeping your existing “median lookup table + pressure-grid rounding” core logic. The biggest remaining issue is that `u_in` quantization is derived from *all* tiny float diffs (including near-duplicates), producing an extremely small `tick` and therefore poor key matching; I instead quantize `u_in` by rounding to the dataset’s natural 3-decimal resolution (minimal, stable) so train/test keys actually hit. Second, your lookup hierarchy still relies heavily on `u_out`, but Kaggle scoring ignores expiratory phase; I build median tables from inspiratory-only rows (`u_out==0`) and drop `u_out` from the keys to better match the metric while keeping the same median-merge approach. Finally, I keep the same hierarchical fallbacks and pressure-grid rounding, and still write a valid `submission.csv`.'
- What this solution (achieved 5.86775) has done: 'Your current MAE (5.85392) is far above the target (0.1491), so we need a meaningful improvement while keeping the same “hierarchical median lookup + pressure-grid rounding” core logic. The biggest score-killer left is that you build the median tables only from `u_out==0` rows, but then you predict for all test rows (including `u_out==1`), which forces many misses and fallback-to-global behavior; instead, we build the lookup tables from all train rows so both `u_out` regimes are represented. To better match the metric (only inspiratory is scored), we keep your existing keys (R,C,step,u_in_q) but also add a parallel inspiratory-only fallback table that is used when the all-rows tables miss, which improves hit rate without changing the approach. Finally, we ensure submission row alignment by explicitly merging predictions by `id` onto `sample_submission` (defensive against any ordering issues).'
- What this solution (achieved 5.86775) has done: 'I keep your current “hierarchical median lookup + pressure-grid rounding” logic, but fix the biggest remaining mismatch: you’re using `cumcount()` as `step`, which can be wrong if rows inside a breath are not strictly sorted; that destroys key matches and forces fallback-to-global medians (huge MAE). I compute `step` by sorting within each `breath_id` by `time_step` (stable) and mapping back to original rows, then build the same median tables and merges. I also ensure `R`/`C` dtypes are consistent between train/test during grouping (avoid silent join misses). These changes are localized to the lookup cell and should move the score materially toward the 0.149 target without changing the approach.'
- What this solution (achieved 5.86775) has done: 'Your current score is far above the target (lower is better), so we need a real accuracy gain while keeping your same “hierarchical median lookup + pressure-grid rounding” approach. The biggest issue is that you’re predicting for all timesteps, but the metric only scores inspiratory (`u_out==0`), and in this dataset pressure during expiratory (`u_out==1`) is essentially a fixed minimum value; using the median lookup there injects large errors without any benefit. With a minimal, localized change, we set predictions to `PRESSURE_MIN` for `u_out==1` rows in test, while keeping your current hierarchical median merges for `u_out==0`. This should move MAE substantially toward the target without changing your feature engineering, lookup keys, rounding scheme, or submission format.'
- What this solution (achieved 5.85392) has done: 'Your current MAE (5.86775; lower is better) is far above the target (0.1491), so we need a real accuracy gain while preserving your core “hierarchical median lookup + pressure-grid rounding” approach. The biggest score-killer is the hard override `pred=PRESSURE_MIN` for all `u_out==1` test rows: Kaggle scores only inspiratory phase, which is defined by `u_out==0` in this competition, so the safest minimal change is to *not* special-case `u_out==1` at all (they are not scored, and the override can only hurt if there’s any mismatch). Next, we improve key hit-rate without changing the approach by adding `u_out`-conditioned median tables (so `u_out==1` rows don’t back-propagate to global fallbacks and distort joins), while keeping your existing all-rows + inspiratory fallbacks. All changes are localized to the lookup/prediction cell and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 5.96612) has done: 'Your current MAE is far above the target (lower is better), and the biggest remaining “cheap win” without changing your core median-lookup approach is improving key hit-rate by matching `u_in` values exactly between train/test. In this dataset, `u_in` is effectively discrete on a fine grid, and rounding to 3 decimals still causes many join misses; switching to exact float32 bitwise matching (cast both train/test `u_in` to `float32` and use that value as the key) typically increases merge hits substantially while preserving the same median-table logic. I keep your same hierarchy of median tables and fallbacks, only changing the `u_in` key construction (`u_in_q`) and ensuring the merge keys use consistent dtypes. Everything else (feature engineering, clustering, pressure snapping to grid, submission writing) remains unchanged and still produces `submission.csv`.'
- What this solution (achieved 5.85392) has done: 'Your current MAE is far above the target (lower is better), and the biggest likely cause in this baseline is that the `u_in` join key is still effectively “almost unique” due to float representation, so most test rows miss `med1_all` and fall back to coarse/global medians. To move the score materially toward the target while keeping the exact same hierarchical median-merge logic, I change only the `u_in_q` construction to a stable discretization that matches train/test: integer-encoding `u_in` at 0.001 resolution (`round(u_in*1000)`), and I use that same integer key in all groupby/merge tables. I also ensure the merge key dtypes are consistent (including `step`) to avoid silent join misses, without changing the feature engineering or any prediction post-processing (clipping + pressure-grid snapping). The rest of the script and the submission writing remain identical.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import math
import random
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)


def set_seed(seeed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seeed)
    random.seed(seeed)
    np.random.seed(seeed)


set_seed(42)
gc.enable()
start_time = time.time()



## === cell 1
if os.path.exists("submission_median_round.csv"):
    print(
        "Found local submission_median_round.csv (will not be used for model output)."
    )



## === cell 2
DEBUG = False
TRAIN_MODEL = False  # Keep core pipeline runnable without heavy training; current script never reaches training anyway.

DATA_ROOT_CANDIDATES = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p) and (
        os.path.exists(os.path.join(p, "train.csv"))
        or os.path.exists(
            os.path.join(p, "ventilator-pressure-prediction", "train.csv")
        )
    ):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate ventilator-pressure-prediction dataset directory."
    )

if os.path.exists(os.path.join(DATA_ROOT, "train.csv")):
    COMP_PATH = DATA_ROOT
else:
    COMP_PATH = os.path.join(DATA_ROOT, "ventilator-pressure-prediction")

print("Using COMP_PATH:", COMP_PATH)



## === cell 3
train = pd.read_csv(os.path.join(COMP_PATH, "train.csv"))
test = pd.read_csv(os.path.join(COMP_PATH, "test.csv"))
submission = pd.read_csv(os.path.join(COMP_PATH, "sample_submission.csv"))

if DEBUG:
    train = train.iloc[: 80 * 200].copy()
    test = test.iloc[: 80 * 50].copy()
    submission = submission.iloc[: 80 * 50].copy()

train["bilstm_pred"] = train["u_in"].astype(np.float32)
test["bilstm_pred"] = test["u_in"].astype(np.float32)

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "submission shape:",
    submission.shape,
)



## === cell 4
train_gf = train[["pressure"]].copy()
all_pressure = np.sort(train_gf.pressure.unique())
PRESSURE_MIN = float(all_pressure[0])
PRESSURE_MAX = float(all_pressure[-1])
PRESSURE_STEP = float(all_pressure[1] - all_pressure[0])
del train_gf
gc.collect()

print(
    "PRESSURE_MIN:",
    PRESSURE_MIN,
    "PRESSURE_MAX:",
    PRESSURE_MAX,
    "PRESSURE_STEP:",
    PRESSURE_STEP,
)



## === cell 5
train["log_u_in"] = np.log1p(train.u_in)
test["log_u_in"] = np.log1p(test.u_in)

bins = pd.qcut(train.time_step, q=80, duplicates="drop", retbins=True)[1]
train["time_step_class"] = pd.cut(
    train.time_step, bins=bins, labels=False, include_lowest=True
)
test["time_step_class"] = pd.cut(
    test.time_step, bins=bins, labels=False, include_lowest=True
)

piv = train.pivot_table(
    index="breath_id",
    columns="time_step_class",
    values="log_u_in",
    fill_value=0,
    aggfunc="mean",
)
piv_test = test.pivot_table(
    index="breath_id",
    columns="time_step_class",
    values="log_u_in",
    fill_value=0,
    aggfunc="mean",
)

piv_test = piv_test.reindex(columns=piv.columns, fill_value=0)

pca = PCA(n_components=2, random_state=42)
pca.fit(piv)

train_pca = pd.DataFrame(pca.transform(piv), columns=["c0", "c1"], index=piv.index)
test_pca = pd.DataFrame(
    pca.transform(piv_test), columns=["c0", "c1"], index=piv_test.index
)

km = KMeans(
    n_clusters=4, random_state=42, max_iter=200, init="k-means++", tol=0.0001, n_init=10
)
y_km = km.fit_predict(train_pca)
y_km_test = km.predict(test_pca)

train_pca["cluster"] = y_km
test_pca["cluster"] = y_km_test

train_pca["breath_id"] = train_pca.index
test_pca["breath_id"] = test_pca.index
train_pca = train_pca[["breath_id", "cluster"]].reset_index(drop=True)
test_pca = test_pca[["breath_id", "cluster"]].reset_index(drop=True)

train = pd.merge(train, train_pca, how="left", on="breath_id")
test = pd.merge(test, test_pca, how="left", on="breath_id")

train.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)
test.drop(["time_step_class", "log_u_in"], axis=1, inplace=True)

print("Added cluster. train columns:", train.columns.tolist()[:12], "...")



## === cell 6
from scipy import stats


def log_return(series: pd.Series):
    return np.log1p(series).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def slope_expiratory(df):
    time_at_u_out = df.iloc[0, 1]
    u_in = df[df.iloc[:, 0] >= time_at_u_out].iloc[:, 2]
    u_in = np.where(u_in > 0, u_in, 10**-10)
    time_steps = df[df.iloc[:, 0] >= time_at_u_out].iloc[:, 0]
    if u_in.size == 0 or time_steps.size == 0:
        slope = np.nan
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_steps, u_in)
    return pd.Series([slope] * df.shape[0], index=df.index)


def realized_volatility_inspiratory(df):
    time_at_u_out = df.iloc[0, 1]
    series = df[df.iloc[:, 0] < time_at_u_out].iloc[:, 2]
    x = realized_volatility(series)
    return pd.Series([x] * df.shape[0], index=df.index)


def absolute_sum_of_changes(x):
    return np.sum(np.abs(np.diff(x)))


def std_inspiratory(df):
    time_at_u_out = df.iloc[0, 1]
    series = df[df.iloc[:, 0] < time_at_u_out].iloc[:, 2]
    x = series.std()
    return pd.Series([x] * df.shape[0], index=df.index)


def range_ratio(x):
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    return mean_median_difference / max_min_difference


def range_ratio_inspiratory(df):
    time_at_u_out = df.iloc[0, 1]
    series = df[df.iloc[:, 0] < time_at_u_out].iloc[:, 2]
    x = range_ratio(series)
    return pd.Series([x] * df.shape[0], index=df.index)


def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    return np.nan


def variation_coefficient_inspiratory(df):
    time_at_u_out = df.iloc[0, 1]
    series = df[df.iloc[:, 0] < time_at_u_out].iloc[:, 2]
    x = variation_coefficient(series)
    return pd.Series([x] * df.shape[0], index=df.index)


def variation_coefficient_expiratory(df):
    time_at_u_out = df.iloc[0, 1]
    series = df[df.iloc[:, 0] >= time_at_u_out].iloc[:, 2]
    x = variation_coefficient(series)
    return pd.Series([x] * df.shape[0], index=df.index)




## === cell 7
def add_features(dff: pd.DataFrame) -> pd.DataFrame:
    s = time.time()
    df = dff.copy()

    df["area"] = (df["time_step"] * df["u_in"]).groupby(df["breath_id"]).cumsum()

    true_area_list = []
    for bid, g in tqdm(
        df[["breath_id", "time_step", "u_in"]].groupby("breath_id"),
        desc="true_area",
        leave=False,
    ):
        ts = g["time_step"].to_numpy()
        u = g["u_in"].to_numpy()
        dts = np.diff(ts, prepend=ts[0])
        ta = np.cumsum(u * dts)
        true_area_list.append(ta)
    df["true_area"] = np.concatenate(true_area_list).astype(np.float32)

    win_mask = df["time_step"].between(0.95, 1.2) & (df["u_out"] == 1)

    df["_cand_time"] = np.where(win_mask, df["time_step"].to_numpy(), np.nan)
    df["time_at_u_out"] = (
        df.groupby("breath_id")["_cand_time"]
        .transform("min")
        .fillna(10.0)
        .astype(np.float32)
    )

    df["_cand_area"] = np.where(win_mask, df["true_area"].to_numpy(), np.nan)
    df["area_at_u_out"] = (
        df.groupby("breath_id")["_cand_area"]
        .transform("min")
        .fillna(0.0)
        .astype(np.float32)
    )

    df.drop(["_cand_time", "_cand_area"], axis=1, inplace=True)

    df["true_area_abs_sum_changes"] = df.groupby("breath_id")["true_area"].transform(
        absolute_sum_of_changes
    )
    df["true_area_max"] = df.groupby("breath_id")["true_area"].transform("max")

    df["u_in_log"] = df.groupby("breath_id")["u_in"].transform(log_return)

    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_std"] = df.groupby("breath_id")["u_in"].transform("std")
    df["u_in_mean"] = df.groupby("breath_id")["u_in"].transform("mean")

    def _map_per_breath_constant(func, colname, use_col):
        g = df.groupby("breath_id", sort=False)

        def _one_value(b):
            tmp = b[["time_step", "time_at_u_out", use_col]]
            out = func(tmp)
            return float(out.iloc[0]) if len(out) else np.nan

        const = g.apply(_one_value)
        df[colname] = df["breath_id"].map(const).astype(np.float32)

    for fn in [
        std_inspiratory,
        range_ratio_inspiratory,
        variation_coefficient_inspiratory,
        slope_expiratory,
        variation_coefficient_expiratory,
    ]:
        _map_per_breath_constant(fn, "u_in_" + fn.__name__, "u_in")

    for fn in [realized_volatility_inspiratory, range_ratio_inspiratory]:
        _map_per_breath_constant(fn, "u_in_log_" + fn.__name__, "u_in_log")

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["time_step_diff3"] = df.groupby("breath_id")["time_step"].diff(3)

    df["true_area_lag1"] = df.groupby("breath_id")["true_area"].shift(1)
    df["true_area_lag2"] = df.groupby("breath_id")["true_area"].shift(2)
    df["true_area_lag_back2"] = df.groupby("breath_id")["true_area"].shift(-2)
    df["true_area_lag3"] = df.groupby("breath_id")["true_area"].shift(3)

    df["u_in_pct"] = df.groupby("breath_id")["u_in"].pct_change()

    df["time_step_diff2"] = df.groupby("breath_id")["time_step"].diff(2)
    df["time_step_diff4"] = df.groupby("breath_id")["time_step"].diff(4)
    df["time_step_diff5"] = df.groupby("breath_id")["time_step"].diff(5)
    df["time_step_diff6"] = df.groupby("breath_id")["time_step"].diff(6)

    df["u_in_expanding10"] = (
        df.groupby("breath_id")["u_in"]
        .transform(lambda x: x.expanding(10).mean())
        .fillna(0)
    )
    df["true_area_expanding15"] = (
        df.groupby("breath_id")["true_area"]
        .transform(lambda x: x.expanding(15).std())
        .fillna(0)
    )
    df["true_area_expanding10"] = (
        df.groupby("breath_id")["true_area"]
        .transform(lambda x: x.expanding(10).mean())
        .fillna(0)
    )

    df["u_in_rolling3max"] = (
        df.groupby("breath_id")["u_in"]
        .transform(lambda x: x.rolling(window=3).max())
        .fillna(0)
    )
    df["u_in_rolling4max"] = (
        df.groupby("breath_id")["u_in"]
        .transform(lambda x: x.rolling(window=4).max())
        .fillna(0)
    )
    df["u_in_rolling5max"] = (
        df.groupby("breath_id")["u_in"]
        .transform(lambda x: x.rolling(window=5).max())
        .fillna(0)
    )
    df["u_in_rolling4"] = (
        df.groupby("breath_id")["u_in"]
        .transform(lambda x: x.rolling(window=4).mean())
        .fillna(0)
    )
    df["u_in_rolling5"] = (
        df.groupby("breath_id")["u_in"]
        .transform(lambda x: x.rolling(window=5).mean())
        .fillna(0)
    )
    df["u_out_rolling8"] = (
        df.groupby("breath_id")["u_out"]
        .transform(lambda x: x.rolling(window=8).mean())
        .fillna(0)
    )

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]

    df["bilstm_pred_lag1"] = df.groupby("breath_id")["bilstm_pred"].shift(1)
    df["bilstm_pred_lag2"] = df.groupby("breath_id")["bilstm_pred"].shift(2)
    df["bilstm_pred_lag_back1"] = df.groupby("breath_id")["bilstm_pred"].shift(-1)
    df["bilstm_pred_lag_back2"] = df.groupby("breath_id")["bilstm_pred"].shift(-2)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]
    df["cluster"] = df["cluster"].astype(str)
    df = pd.get_dummies(df, drop_first=True)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    if "u_in_lag1" in df.columns:
        df.drop(["u_in_lag1"], axis=1, inplace=True)

    print(f"add_features done in {(time.time()-s)/60:.2f} min. shape={df.shape}")
    return df




## === cell 8
train_feat = add_features(train)
test_feat = add_features(test)

target_col = "pressure"
drop_cols = ["id", "breath_id"]
y = train_feat[target_col].to_numpy().astype(np.float32)

X_train = train_feat.drop([target_col] + drop_cols, axis=1)
X_test = test_feat.drop(drop_cols, axis=1)

missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
missing_in_train = [c for c in X_test.columns if c not in X_train.columns]
for c in missing_in_test:
    X_test[c] = 0
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[X_train.columns]

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)



## === cell 9
train_tmp = train.copy()
test_tmp = test.copy()


def add_step_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_row"] = np.arange(len(df), dtype=np.int64)
    df_sorted = df.sort_values(["breath_id", "time_step", "_row"], kind="mergesort")
    df_sorted["step"] = (
        df_sorted.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )
    df = df.merge(df_sorted[["_row", "step"]], on="_row", how="left")
    df.drop(columns=["_row"], inplace=True)
    return df


train_tmp = add_step_by_time(train_tmp)
test_tmp = add_step_by_time(test_tmp)


def quantize_u_in_key(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    return np.rint(a * 1000.0).astype(np.int32)


train_tmp["u_in_q"] = quantize_u_in_key(train_tmp["u_in"].to_numpy())
test_tmp["u_in_q"] = quantize_u_in_key(test_tmp["u_in"].to_numpy())

for df in (train_tmp, test_tmp):
    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)
    df["u_out"] = df["u_out"].astype(np.int8)
    df["step"] = df["step"].astype(np.int16)
    df["u_in_q"] = df["u_in_q"].astype(np.int32)

grp1 = ["R", "C", "u_out", "step", "u_in_q"]
grp2 = ["R", "C", "u_out", "step"]
grp3 = ["R", "C", "u_out"]
grp1_uout_agn = ["R", "C", "step", "u_in_q"]
grp2_uout_agn = ["R", "C", "step"]
grp3_uout_agn = ["R", "C"]

train_all = train_tmp
med1_all = (
    train_all.groupby(grp1, sort=False, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p1_all"})
)
med2_all = (
    train_all.groupby(grp2, sort=False, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p2_all"})
)
med3_all = (
    train_all.groupby(grp3, sort=False, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p3_all"})
)
global_median_all = float(train_all["pressure"].median())

train_insp = train_tmp[train_tmp["u_out"] == 0]
med1_insp = (
    train_insp.groupby(grp1_uout_agn, sort=False, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p1_insp"})
)
med2_insp = (
    train_insp.groupby(grp2_uout_agn, sort=False, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p2_insp"})
)
med3_insp = (
    train_insp.groupby(grp3_uout_agn, sort=False, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p3_insp"})
)
global_median_insp = float(train_insp["pressure"].median())

base = test_tmp[["id", "R", "C", "u_out", "step", "u_in_q"]].copy()

base = base.merge(med1_all, how="left", on=grp1)
base = base.merge(med2_all, how="left", on=grp2)
base = base.merge(med3_all, how="left", on=grp3)

base = base.merge(med1_insp, how="left", on=grp1_uout_agn)
base = base.merge(med2_insp, how="left", on=grp2_uout_agn)
base = base.merge(med3_insp, how="left", on=grp3_uout_agn)

pred = base["p1_all"]
pred = pred.fillna(base["p2_all"])
pred = pred.fillna(base["p3_all"])

pred = pred.fillna(base["p1_insp"])
pred = pred.fillna(base["p2_insp"])
pred = pred.fillna(base["p3_insp"])

pred = pred.fillna(global_median_insp)
pred = pred.fillna(global_median_all).to_numpy(dtype=np.float32)

pred = np.clip(pred, PRESSURE_MIN, PRESSURE_MAX)
pred = np.round((pred - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN

pred_df = pd.DataFrame({"id": base["id"].to_numpy(), "pressure": pred})
sub = submission[["id"]].merge(pred_df, how="left", on="id")
sub["pressure"] = sub["pressure"].fillna(global_median_insp).astype(np.float32)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("u_in key: int(round(u_in*1000))")
print("Elapsed seconds:", int(time.time() - start_time))
