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

4.24541

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.24541) has done: 'I remove the failing internet download and the unused imports that trigger the protobuf/TensorFlow `MessageFactory.GetPrototype` crash, since the current pipeline errors before any submission is written. Because the code depends on external BiLSTM prediction files that are not present, I replace them with an in-notebook baseline that creates the missing `bilstm_pred` column from available features (minimal change to keep downstream feature engineering intact). I also fix the `add_features()` index-alignment bug caused by `groupby().apply()` returning a misaligned index by switching those assignments to `transform()` (same computation intent, but stable). Finally, I ensure a valid `submission.csv` is produced with `id,pressure` and aligned row order.'
- What this solution (achieved 4.24541) has done: 'I fix the `add_features()` crash by removing the fragile `groupby().apply(...)->to_list()->np.concatenate()` pattern that sometimes yields scalars (0‑d arrays) and instead compute those per-breath statistics with a stable `groupby().transform()`/mapping approach that preserves row alignment. This change is purely a bug fix: it keeps the same feature intent (constant per breath, repeated for all timesteps) while ensuring the pipeline runs end-to-end. I also make the `time_at_u_out` / `area_at_u_out` computation use `groupby().transform()` to avoid index reindexing pitfalls. The rest of the logic (feature set, median-based prediction, pressure clipping/rounding, and submission writing) is preserved so score changes only come from correctness/stability improvements.'

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
key_cols = ["R", "C", "time_step", "u_in", "u_out"]

train_tmp = train.copy()
test_tmp = test.copy()
train_tmp["u_in_bin"] = np.round(train_tmp["u_in"], 1)
test_tmp["u_in_bin"] = np.round(test_tmp["u_in"], 1)
train_tmp["t_bin"] = np.round(train_tmp["time_step"], 2)
test_tmp["t_bin"] = np.round(test_tmp["time_step"], 2)

grp_cols = ["R", "C", "t_bin", "u_in_bin", "u_out"]
med_map = train_tmp.groupby(grp_cols)["pressure"].median()

global_median = float(train_tmp["pressure"].median())

pred = []
for row in test_tmp[grp_cols].itertuples(index=False, name=None):
    pred.append(med_map.get(row, global_median))
pred = np.array(pred, dtype=np.float32)

pred = np.clip(pred, PRESSURE_MIN, PRESSURE_MAX)
pred = np.round((pred - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN

pred[test["u_out"].to_numpy() == 1] = 0.0



## === cell 10
sub = submission.copy()
sub["pressure"] = pred
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Elapsed seconds:", int(time.time() - start_time))
