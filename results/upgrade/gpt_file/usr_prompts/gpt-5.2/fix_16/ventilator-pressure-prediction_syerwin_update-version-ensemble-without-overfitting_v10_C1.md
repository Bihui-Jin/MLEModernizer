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

0.96756

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.26422) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/gb-*` and similar paths) by switching the ensemble to use only predictions that are guaranteed to exist: a simple, deterministic model trained from `train.csv` to predict `pressure` from your engineered features. I keep your core feature engineering intact, add a lightweight scikit-learn regressor that runs within the time limit, and ensure the train/test one-hot columns are aligned to prevent shape mismatches. Finally, I preserve your original pressure “grid snapping” post-processing (P_MIN/P_MAX/P_STEP rounding/clipping) and write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 1.02331) has done: 'Your current MAE (1.26422) is far worse than the target (0.138...), and the main reason is that the evaluation ignores expiratory phase while your model is trained/evaluated on all timesteps equally. I keep your exact feature engineering and model family, but (1) train only on inspiratory timesteps (`u_out==0`) to match the metric and (2) apply the same rule at inference by setting expiratory predictions (`u_out==1`) to 0 (they are not scored, and this avoids injecting arbitrary values). I also add a simple breath-wise cross-validation split (GroupKFold by `breath_id`) to pick a slightly better `max_iter` from a tiny candidate set without changing the approach, then refit on all inspiratory data. These are minimal, metric-aligned changes that typically move this solution much closer to the target range.'
- What this solution (achieved 8.59874) has done: 'I remove the biggest bottleneck: doing full feature engineering twice and then one-hot encoding separately, which forces expensive alignment and duplicated groupby/rolling computations. Instead, I concatenate train+test once, compute the exact same features once, and run `get_dummies` once so train/test already share identical columns (no align/copies). I also replace the slow per-lag `groupby.shift` loop with an equivalent vectorized shift using NumPy that preserves semantics because the dataframe is already stably sorted by `breath_id,time_step`. Finally, I avoid repeated heavy `GroupKFold` refits by enabling `warm_start` so each candidate `max_iter` continues training from the previous iteration count (same algorithm/solution as training from scratch, just less wasted work).'
- What this solution (achieved 8.59874) has done: 'Your current score (8.59874 MAE) is far worse than the target (0.138...), so we should improve it with minimal, metric-aligned fixes. The biggest issue is in post-processing: you’re currently forcing all expiratory (`u_out==1`) predictions to the median inspiratory pressure, which is arbitrary and can strongly hurt public/private MAE when Kaggle’s scorer masks the inspiratory phase but still expects reasonable values elsewhere. I keep your exact feature engineering and model, but change expiratory handling to a safer “carry-forward last inspiratory prediction within each breath” (a standard, minimal heuristic) and fix a likely submission/id alignment pitfall by building the submission directly from `test_df[['id']]` to guarantee ordering matches predictions. Everything else (training on inspiratory only, HGBR params, grid snapping) stays the same.'
- What this solution (achieved 0.95712) has done: 'I fix the failing split after `get_dummies`: your `_is_train` flag becomes either a plain numeric column (if treated as numeric) or multiple dummy columns, so indexing `_is_train_1` is fragile and caused the KeyError. The minimal, score-neutral fix is to force `_is_train` to a categorical before `get_dummies`, then robustly detect the correct dummy column name and split on it. After that, the rest of your pipeline (feature engineering, inspiratory-only training, expiratory carry-forward, grid snapping, and submission writing) can run unchanged and produce a valid `submission.csv`. I also add a small safeguard to fall back to the original `_is_train` column if dummy creation behavior differs, keeping the logic consistent.'
- What this solution (achieved 0.96759) has done: 'Your current MAE (0.95712) is still far from the target (0.13818), so we should improve it with a small, metric-aligned fix rather than changing the model/feature pipeline. The biggest issue is that you train only on inspiratory (`u_out==0`) but at inference you still predict expiratory timesteps using features like `area` and `u_in_cumsum` that keep growing, then you carry-forward those potentially unstable values; this creates a distribution shift that can hurt even inspiratory predictions around the transition. I keep your exact model/feature logic, but apply the standard competition trick of forcing `u_in=0` for all expiratory rows in both train+test *before* feature engineering, so cumulative/rolling features stop changing during expiration and better match how pressure behaves. Everything else (inspiratory-only training, GroupKFold max_iter selection, carry-forward, grid snapping, submission writing) stays the same.'
- What this solution (achieved 0.96759) has done: 'Your current MAE (0.96759) is much worse than the target (0.13818), so we should improve it with the smallest metric-aligned changes while keeping your exact model family and feature engineering. The main issue is that you’re carrying-forward expiratory predictions, but Kaggle’s MAE only scores inspiratory timesteps (`u_out==0`), so we can make expiratory predictions completely “safe” by setting them to 0 without affecting the metric and reducing any potential side-effects from post-processing. Additionally, your post-processing “grid snapping” should be applied only on inspiratory rows (the only scored rows) to avoid unnecessary transformations. Everything else (u_out-based u_in zeroing before features, same features, RobustScaler, HGBR with GroupKFold max_iter selection) remains unchanged.'
- What this solution (achieved 0.96756) has done: 'Your current gap to the target is large (0.9676 vs 0.1382 MAE; lower is better), so we need a small but materially score-improving, metric-aligned adjustment without changing your model family or feature engineering. The most impactful issue is that you’re training only on inspiratory rows (correct) but you’re still feeding expiratory rows through the model at test time, and then overwriting them; this can still distort the model’s learned mapping because expiratory-driven engineered features exist in the training feature space (via the combined feature engineering). I keep your exact feature generation and HistGradientBoostingRegressor setup, but (1) ensure expiratory rows are made “neutral” consistently in BOTH train and test before feature engineering by also freezing `time_step`-accumulating features via an added per-breath inspiratory time counter, and (2) apply the standard Ventilator trick of snapping predictions to the known discrete pressure grid learned from training (already present) but computed robustly and used to map to the nearest grid value. These are minimal changes that typically move MAE substantially toward the target band while preserving your core approach.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

np.random.seed(42)

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

dtypes_train = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtypes_test = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train_df = pd.read_csv(TRAIN_PATH, dtype=dtypes_train, low_memory=False)
test_df = pd.read_csv(TEST_PATH, dtype=dtypes_test, low_memory=False)
sample_sub = pd.read_csv(
    SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"}, low_memory=False
)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns)




## === cell 1
def add_features_all(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy(deep=False)

    df = df.sort_values(["breath_id", "time_step"], kind="mergesort", ignore_index=True)

    df["one"] = np.int8(1)

    gb = df.groupby("breath_id", sort=False, observed=True)

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    u_out_arr = df["u_out"].to_numpy(copy=False)
    df["_insp_step"] = (u_out_arr == 0).astype("int8")

    df["area"] = (
        (df["time_step"] * df["u_in"] * df["_insp_step"])
        .groupby(df["breath_id"], sort=False, observed=True)
        .cumsum()
    )

    df["time_step_cumsum"] = (
        (df["time_step"] * df["_insp_step"])
        .groupby(df["breath_id"], sort=False, observed=True)
        .cumsum()
    )
    df["u_in_cumsum"] = gb["u_in"].cumsum()
    print("Step-1...Completed")

    breath_id = df["breath_id"].to_numpy(copy=False)

    u_in = df["u_in"].to_numpy(copy=False)
    u_out = df["u_out"].to_numpy(copy=False)

    for k in (1, 2, 3, 4):
        bid_prev = np.empty_like(breath_id)
        bid_prev[:k] = 0
        bid_prev[k:] = breath_id[:-k]
        same_prev = bid_prev == breath_id

        ui_prev = np.empty_like(u_in)
        ui_prev[:k] = 0.0
        ui_prev[k:] = u_in[:-k]
        uo_prev = np.empty_like(u_out)
        uo_prev[:k] = 0
        uo_prev[k:] = u_out[:-k]

        df[f"u_in_lag{k}"] = ui_prev * same_prev
        df[f"u_out_lag{k}"] = uo_prev * same_prev

        bid_next = np.empty_like(breath_id)
        bid_next[-k:] = 0
        bid_next[:-k] = breath_id[k:]
        same_next = bid_next == breath_id

        ui_next = np.empty_like(u_in)
        ui_next[-k:] = 0.0
        ui_next[:-k] = u_in[k:]
        uo_next = np.empty_like(u_out)
        uo_next[-k:] = 0
        uo_next[:-k] = u_out[k:]

        df[f"u_in_lag_back{k}"] = ui_next * same_next
        df[f"u_out_lag_back{k}"] = uo_next * same_next

    df = df.fillna(0)
    print("Step-2...Completed")

    u_in_g = gb["u_in"]

    u_in_max = u_in_g.transform("max")
    u_in_mean = u_in_g.transform("mean")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = u_in_max - df["u_in"]
    df["breath_id__u_in__diffmean"] = u_in_mean - df["u_in"]
    print("Step-3...Completed")

    for k in (1, 2, 3, 4):
        df[f"u_in_diff{k}"] = df["u_in"] - df[f"u_in_lag{k}"]
        df[f"u_out_diff{k}"] = df["u_out"] - df[f"u_out_lag{k}"]
    print("Step-4...Completed")

    df["count"] = gb["one"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    bid_lag = np.empty_like(breath_id)
    bid_lag[0] = 0
    bid_lag[1:] = breath_id[:-1]
    bid_lag2 = np.empty_like(breath_id)
    bid_lag2[:2] = 0
    bid_lag2[2:] = breath_id[:-2]

    lagsame = (bid_lag == breath_id).astype(np.int8, copy=False)
    lag2same = (bid_lag2 == breath_id).astype(np.int8, copy=False)

    df["breath_id_lag"] = bid_lag
    df["breath_id_lag2"] = bid_lag2
    df["breath_id_lagsame"] = lagsame
    df["breath_id_lag2same"] = lag2same

    u_in_shift1 = np.empty_like(u_in)
    u_in_shift1[0] = 0.0
    u_in_shift1[1:] = u_in[:-1]
    u_in_shift2 = np.empty_like(u_in)
    u_in_shift2[:2] = 0.0
    u_in_shift2[2:] = u_in[:-2]

    df["breath_id__u_in_lag"] = u_in_shift1 * lagsame
    df["breath_id__u_in_lag2"] = u_in_shift2 * lag2same
    print("Step-5...Completed")

    df["time_step_diff"] = gb["time_step"].diff().fillna(0)

    df["ewm_u_in_mean"] = u_in_g.ewm(halflife=9).mean().reset_index(level=0, drop=True)

    roll = u_in_g.rolling(window=15, min_periods=1)
    rolled = roll.agg(["sum", "min", "max", "mean"]).reset_index(level=0, drop=True)
    df["15_in_sum"] = rolled["sum"]
    df["15_in_min"] = rolled["min"]
    df["15_in_max"] = rolled["max"]
    df["15_in_mean"] = rolled["mean"]
    print("Step-6...Completed")

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    print("Step-7...Completed")

    df["R"] = df["R"].astype("int16").astype("category")
    df["C"] = df["C"].astype("int16").astype("category")
    df["R__C"] = (df["R"].astype(str) + "__" + df["C"].astype(str)).astype("category")

    if "_is_train" in df.columns:
        df["_is_train"] = df["_is_train"].astype("int8").astype("category")

    df = pd.get_dummies(df)
    print("Step-8...Completed")

    return df


train_df2 = train_df.copy(deep=False)
test_df2 = test_df.copy(deep=False)

train_df2.loc[train_df2["u_out"].to_numpy(copy=False) == 1, "u_in"] = 0.0
test_df2.loc[test_df2["u_out"].to_numpy(copy=False) == 1, "u_in"] = 0.0

train_df2["_is_train"] = np.int8(1)
test_df2["_is_train"] = np.int8(0)

df_all = pd.concat([train_df2, test_df2], axis=0, ignore_index=True, copy=False)
print("Combined df:", df_all.shape)

df_all_feat = add_features_all(df_all)

flag_cols = [c for c in df_all_feat.columns if c.startswith("_is_train")]
print("Flag cols:", flag_cols[:10], "count:", len(flag_cols))

if "_is_train_1" in df_all_feat.columns:
    flag_col = "_is_train_1"
elif "_is_train" in df_all_feat.columns:
    flag_col = "_is_train"
elif len(flag_cols) == 1:
    flag_col = flag_cols[0]
else:
    raise KeyError(
        f"Could not identify _is_train flag column after get_dummies. Found: {flag_cols}"
    )

train_feat = df_all_feat[df_all_feat[flag_col] == 1].copy(deep=False)
test_feat = df_all_feat[df_all_feat[flag_col] == 0].copy(deep=False)

train_feat = train_feat.sort_values(["id"], kind="mergesort").reset_index(drop=True)
test_feat = test_feat.sort_values(["id"], kind="mergesort").reset_index(drop=True)

del df_all, df_all_feat, train_df2, test_df2
gc.collect()



## === cell 2
pressure_all = train_feat["pressure"].to_numpy(copy=False).astype("float32", copy=False)

pressure_levels = np.unique(pressure_all)
pressure_levels.sort()

P_MIN = float(pressure_levels[0])
P_MAX = float(pressure_levels[-1])

if pressure_levels.size > 1:
    diffs = np.diff(pressure_levels)
    diffs = diffs[diffs > 0]
    P_STEP = float(np.min(diffs)) if diffs.size else 0.0
else:
    P_STEP = 0.0

print("Min pressure:", P_MIN)
print("Max pressure:", P_MAX)
print("Pressure step:", P_STEP)
print("Unique values:", pressure_levels.shape[0])

del pressure_all
gc.collect()



## === cell 3
y_all = train_feat["pressure"].astype("float32").to_numpy(copy=False)
groups_all = train_feat["breath_id"].to_numpy(copy=False)
train_insp_mask = train_feat["u_out"].to_numpy(copy=False) == 0

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]

X_train_all = train_feat.drop(columns=drop_cols, errors="ignore")
X_test = test_feat.drop(
    columns=[c for c in drop_cols if c != "pressure"], errors="ignore"
)

if X_train_all.shape[1] != X_test.shape[1] or not X_train_all.columns.equals(
    X_test.columns
):
    X_train_all, X_test = X_train_all.align(X_test, join="left", axis=1, fill_value=0)

X_train = X_train_all.loc[train_insp_mask]
y = y_all[train_insp_mask]
groups = groups_all[train_insp_mask]

insp_fill_value = float(np.median(y))

print("X_train (insp):", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)
print("Median insp pressure (for internal default):", insp_fill_value)

del train_feat, test_feat, X_train_all, y_all, groups_all
gc.collect()



## === cell 4
scaler = RobustScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_s = np.ascontiguousarray(X_train_s)
X_test_s = np.ascontiguousarray(X_test_s)

del X_train, X_test
gc.collect()



## === cell 5
base_params = dict(
    loss="absolute_error",
    learning_rate=0.08,
    max_depth=6,
    l2_regularization=0.0,
    random_state=42,
    warm_start=True,
)

candidates = [250, 400, 600]
gkf = GroupKFold(n_splits=3)

best_iter = candidates[0]
best_mae = float("inf")

splits = []
for tr_idx, va_idx in gkf.split(X_train_s, y, groups=groups):
    splits.append(
        (np.asarray(tr_idx, dtype=np.int32), np.asarray(va_idx, dtype=np.int32))
    )

fold_models = []
for tr_idx, _ in splits:
    m = HistGradientBoostingRegressor(max_iter=candidates[0], **base_params)
    m.fit(X_train_s[tr_idx], y[tr_idx])
    fold_models.append(m)

it0 = candidates[0]
fold_maes = []
for (tr_idx, va_idx), m in zip(splits, fold_models):
    pred_va = m.predict(X_train_s[va_idx])
    fold_maes.append(mean_absolute_error(y[va_idx], pred_va))
cv_mae = float(np.mean(fold_maes))
print(f"CV MAE (insp only) for max_iter={it0}: {cv_mae:.6f}")
best_mae = cv_mae
best_iter = it0

for it in candidates[1:]:
    fold_maes = []
    for (tr_idx, va_idx), m in zip(splits, fold_models):
        m.set_params(max_iter=it)
        m.fit(X_train_s[tr_idx], y[tr_idx])
        pred_va = m.predict(X_train_s[va_idx])
        fold_maes.append(mean_absolute_error(y[va_idx], pred_va))
    cv_mae = float(np.mean(fold_maes))
    print(f"CV MAE (insp only) for max_iter={it}: {cv_mae:.6f}")
    if cv_mae < best_mae:
        best_mae = cv_mae
        best_iter = it

print("Chosen max_iter:", best_iter, "with CV MAE:", best_mae)

final_params = dict(base_params)
final_params.pop("warm_start", None)
model = HistGradientBoostingRegressor(max_iter=best_iter, **final_params)
model.fit(X_train_s, y)

pred_test = model.predict(X_test_s).astype("float32", copy=False)
print(pred_test.shape, pred_test[:5])

del y, groups, splits, fold_models
gc.collect()



## === cell 6
pred_final = pred_test.copy()

test_u_out = test_df["u_out"].to_numpy(copy=False)

pred_final[test_u_out == 1] = 0.0

insp_mask_test = test_u_out == 0

if pressure_levels.size > 1:
    x = pred_final[insp_mask_test]
    idx = np.searchsorted(pressure_levels, x, side="left")
    idx = np.clip(idx, 0, pressure_levels.size - 1)
    idx0 = np.clip(idx - 1, 0, pressure_levels.size - 1)
    left = pressure_levels[idx0]
    right = pressure_levels[idx]
    choose_right = np.abs(right - x) <= np.abs(x - left)
    snapped = np.where(choose_right, right, left).astype("float32", copy=False)
    pred_final[insp_mask_test] = snapped
else:
    pred_final[insp_mask_test] = np.clip(
        pred_final[insp_mask_test], P_MIN, P_MAX
    ).astype("float32", copy=False)

pred_final[insp_mask_test] = np.clip(pred_final[insp_mask_test], P_MIN, P_MAX).astype(
    "float32", copy=False
)

submission = pd.DataFrame({"id": test_df["id"].astype("int32").to_numpy(copy=False)})
submission["pressure"] = pred_final.astype("float32", copy=False)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)

del X_train_s, X_test_s, pred_test, pred_final, pressure_levels
gc.collect()
