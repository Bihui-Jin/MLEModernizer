# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge

DATA_DIR = "../input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

base_num = ["time_step", "u_in", "u_out"]
base_cat = ["R", "C"]
target_col = "pressure"
group_col = "breath_id"

usecols_train = base_num + base_cat + [target_col, group_col, "id"]
usecols_test = base_num + base_cat + [group_col, "id"]

dtype_train = {
    "id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
    "pressure": "float32",
    "breath_id": "int32",
}
dtype_test = {
    "id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
    "breath_id": "int32",
}

read_csv_kwargs = {}
try:
    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

train = pd.read_csv(
    train_path, usecols=usecols_train, dtype=dtype_train, **read_csv_kwargs
)
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test, **read_csv_kwargs)
sub = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "int32"})

train.sort_values(
    [group_col, "time_step"], inplace=True, kind="mergesort", ignore_index=True
)
test.sort_values(
    [group_col, "time_step"], inplace=True, kind="mergesort", ignore_index=True
)


def _rolling_mean_trailing(x: np.ndarray, w: int) -> np.ndarray:
    n = x.size
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(x, dtype=np.float64, out=cs[1:])
    idx = np.arange(n, dtype=np.int32)
    start = idx - (w - 1)
    start[start < 0] = 0
    s = cs[idx + 1] - cs[start]
    denom = (idx - start + 1).astype(np.float64)
    out = (s / denom).astype(np.float32)
    return out


def _rolling_max_trailing_smallw(x: np.ndarray, w: int) -> np.ndarray:
    n = x.size
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        j0 = i - (w - 1)
        if j0 < 0:
            j0 = 0
        out[i] = np.max(x[j0 : i + 1])
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    breath = df[group_col].to_numpy(copy=False)
    change = np.flatnonzero(breath[1:] != breath[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [breath.size]))

    u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)
    Rf = df["R"].to_numpy(dtype=np.float32, copy=False)
    Cf = df["C"].to_numpy(dtype=np.float32, copy=False)

    n = breath.size
    t_idx = np.empty(n, dtype=np.int16)
    u_in_cumsum = np.empty(n, dtype=np.float32)
    u_out_cumsum = np.empty(n, dtype=np.int16)

    lag1 = np.empty(n, dtype=np.float32)
    lag2 = np.empty(n, dtype=np.float32)
    lag3 = np.empty(n, dtype=np.float32)
    lag4 = np.empty(n, dtype=np.float32)
    lag5 = np.empty(n, dtype=np.float32)

    u_in_roll3 = np.empty(n, dtype=np.float32)
    u_in_roll5 = np.empty(n, dtype=np.float32)
    u_in_rmax5 = np.empty(n, dtype=np.float32)

    for s, e in zip(starts, ends):
        m = e - s
        t_idx[s:e] = np.arange(m, dtype=np.int16)

        u_in_cumsum[s:e] = np.cumsum(u_in[s:e], dtype=np.float32)
        u_out_cumsum[s:e] = np.cumsum(u_out[s:e], dtype=np.int16).astype(
            np.int16, copy=False
        )

        x = u_in[s:e]
        lag1[s] = 0.0
        lag1[s + 1 : e] = x[:-1]
        lag2[s : s + 2] = 0.0
        if m > 2:
            lag2[s + 2 : e] = x[:-2]
        lag3[s : s + 3] = 0.0
        if m > 3:
            lag3[s + 3 : e] = x[:-3]
        lag4[s : s + 4] = 0.0
        if m > 4:
            lag4[s + 4 : e] = x[:-4]
        lag5[s : s + 5] = 0.0
        if m > 5:
            lag5[s + 5 : e] = x[:-5]

        ushift = lag1[s:e].astype(np.float32, copy=False)
        u_in_roll3[s:e] = _rolling_mean_trailing(ushift, 3)
        u_in_roll5[s:e] = _rolling_mean_trailing(ushift, 5)
        u_in_rmax5[s:e] = _rolling_max_trailing_smallw(ushift, 5)

    df["t_idx"] = t_idx
    df["u_in_lag1"] = lag1
    df["u_in_lag2"] = lag2
    df["u_in_lag3"] = lag3
    df["u_in_lag4"] = lag4
    df["u_in_lag5"] = lag5

    df["u_in_diff1"] = (u_in - lag1).astype(np.float32, copy=False)

    df["u_in_cumsum"] = u_in_cumsum
    df["u_out_cumsum"] = u_out_cumsum

    df["u_in_roll3"] = u_in_roll3
    df["u_in_roll5"] = u_in_roll5
    df["u_in_rmax5"] = u_in_rmax5

    df["u_in_x_R"] = (u_in * Rf).astype(np.float32, copy=False)
    df["u_in_x_C"] = (u_in * Cf).astype(np.float32, copy=False)
    df["u_in_cumsum_x_C"] = (u_in_cumsum * Cf).astype(np.float32, copy=False)

    return df


train = add_features(train)
test = add_features(test)

feature_cols_num = base_num + [
    "t_idx",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_lag4",
    "u_in_lag5",
    "u_in_diff1",
    "u_in_cumsum",
    "u_out_cumsum",
    "u_in_roll3",
    "u_in_roll5",
    "u_in_rmax5",
    "u_in_x_R",
    "u_in_x_C",
    "u_in_cumsum_x_C",
]
feature_cols_cat = base_cat
X_cols = feature_cols_num + feature_cols_cat

train_insp = train[train["u_out"] == 0]

rng = np.random.RandomState(42)
unique_breaths = train_insp[group_col].unique()

n_breaths_fit = min(22000, unique_breaths.shape[0])
fit_breaths = rng.choice(unique_breaths, size=n_breaths_fit, replace=False)
fit_mask = train_insp[group_col].isin(fit_breaths)

X_fit = train_insp.loc[fit_mask, X_cols]
y_fit = train_insp.loc[fit_mask, target_col].to_numpy(dtype=np.float32, copy=False)

X_test = test[X_cols]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols_num),
        ("cat", categorical_transformer, feature_cols_cat),
    ],
    remainder="drop",
    sparse_threshold=1.0,
)

models = [
    (
        "sub_1",
        RandomForestRegressor(
            n_estimators=80,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        ),
    ),
    (
        "sub_2",
        ExtraTreesRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        ),
    ),
    ("sub_3", GradientBoostingRegressor(random_state=42)),
    ("sub_4", Ridge(alpha=1.0, random_state=42)),
]

X_fit_tr = preprocess.fit_transform(X_fit, y_fit)
X_test_tr = preprocess.transform(X_test)

try:
    import scipy.sparse as sp

    if sp.issparse(X_fit_tr) and not sp.isspmatrix_csc(X_fit_tr):
        X_fit_tr = X_fit_tr.tocsc()
    if sp.issparse(X_test_tr) and not sp.isspmatrix_csc(X_test_tr):
        X_test_tr = X_test_tr.tocsc()
except Exception:
    pass

preds = {}
for name, model in models:
    model.fit(X_fit_tr, y_fit)
    preds[name] = model.predict(X_test_tr).astype(np.float32, copy=False)

blended = (
    preds["sub_1"] * np.float32(0.23)
    + preds["sub_2"] * np.float32(0.30)
    + preds["sub_3"] * np.float32(0.27)
    + preds["sub_4"] * np.float32(0.20)
).astype(np.float32, copy=False)

sub = sub.copy()
test_ids = test["id"].to_numpy(dtype=np.int32, copy=False)
sub_ids = sub["id"].to_numpy(dtype=np.int32, copy=False)

if sub.shape[0] == test.shape[0] and np.array_equal(sub_ids, test_ids):
    sub["pressure"] = blended
else:
    sub["pressure"] = np.float32(0.0)
    sub_index = pd.Index(sub_ids)
    pred_index = pd.Index(test_ids)
    pos = sub_index.get_indexer(pred_index)
    if (pos < 0).any():
        pred_df = pd.DataFrame({"id": test_ids, "pressure": blended})
        sub = sub.merge(pred_df, on="id", how="left", validate="one_to_one")
        if sub["pressure"].isna().any():
            sub["pressure"] = sub["pressure"].fillna(0.0)
        sub["pressure"] = sub["pressure"].astype("float32")
    else:
        sub_pressure = sub["pressure"].to_numpy(dtype=np.float32, copy=False)
        sub_pressure[pos] = blended
        sub["pressure"] = sub_pressure

sub.to_csv("submission.csv", index=False)
sub.head(5)



## === cell 1
assert sub.shape[0] == 603600, f"Unexpected submission rows: {sub.shape[0]}"
assert list(sub.columns) == [
    "id",
    "pressure",
], f"Unexpected submission columns: {sub.columns.tolist()}"
assert np.issubdtype(sub["pressure"].dtype, np.floating), "pressure must be float"
assert (
    sub["pressure"].isna().sum() == 0
), "Found NaN predictions after alignment; id alignment issue."
sub.describe(include="all")
