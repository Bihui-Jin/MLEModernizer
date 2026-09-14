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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

SEED = 42
np.random.seed(SEED)

DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
test = pd.read_csv(
    test_path,
    dtype={
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    },
)
sub = pd.read_csv(sample_path, dtype={"id": "int32", "pressure": "float32"})

required_train_cols = {
    "id",
    "breath_id",
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "pressure",
}
required_test_cols = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}
if not required_train_cols.issubset(train.columns):
    missing = required_train_cols - set(train.columns)
    raise ValueError(f"train.csv missing columns: {missing}")
if not required_test_cols.issubset(test.columns):
    missing = required_test_cols - set(test.columns)
    raise ValueError(f"test.csv missing columns: {missing}")
if not {"id", "pressure"}.issubset(sub.columns):
    raise ValueError("sample_submission.csv must have columns: id,pressure")




## === cell 1
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["_orig_row"] = np.arange(len(df), dtype=np.int64)

    df = df.sort_values(["breath_id", "time_step", "_orig_row"], kind="mergesort")
    g = df.groupby("breath_id", sort=False)

    df["breath_time"] = g.cumcount().astype(np.int16)
    df["dt"] = g["time_step"].diff().fillna(0.0).astype(np.float32)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0).astype(np.float32)
    df["u_in_lag5"] = g["u_in"].shift(5).fillna(0.0).astype(np.float32)
    df["u_in_lead1"] = g["u_in"].shift(-1).fillna(0.0).astype(np.float32)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)
    df["u_out_lag2"] = g["u_out"].shift(2).fillna(0).astype(np.int8)

    df["du_in"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)
    df["du_in2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype(np.float32)

    df["u_in_cumsum"] = g["u_in"].cumsum().astype(np.float32)

    df["u_in_area"] = (
        (df["u_in"] * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.float32)
    )

    df["u_in_roll3_mean"] = (
        g["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["u_in_roll5_mean"] = (
        g["u_in"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["u_in_roll5_std"] = (
        g["u_in"]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["breath_time_norm"] = (df["breath_time"].astype(np.float32) / 79.0).astype(
        np.float32
    )

    df["u_in_x_R"] = (df["u_in"] * df["R"]).astype(np.float32)
    df["u_in_x_C"] = (df["u_in"] * df["C"]).astype(np.float32)

    df["u_in_lag10"] = g["u_in"].shift(10).fillna(0.0).astype(np.float32)
    df["u_in_lead2"] = g["u_in"].shift(-2).fillna(0.0).astype(np.float32)
    df["u_out_lead1"] = g["u_out"].shift(-1).fillna(0).astype(np.int8)

    df["u_in_roll10_mean"] = (
        g["u_in"]
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["u_in_roll10_std"] = (
        g["u_in"]
        .rolling(window=10, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .astype(np.float32)
    )
    df["u_in_cummax"] = g["u_in"].cummax().astype(np.float32)

    df = df.sort_values("_orig_row", kind="mergesort").drop(columns=["_orig_row"])
    return df


train_n = len(train)
combined = pd.concat(
    [train.drop(columns=["pressure"]), test], axis=0, ignore_index=True
)
combined_feat = add_breath_features(combined)

train_feat = combined_feat.iloc[:train_n].copy()
train_feat["pressure"] = train["pressure"].values
test_feat = combined_feat.iloc[train_n:].copy()

feature_cols_num = [
    "time_step",
    "breath_time",
    "breath_time_norm",
    "dt",
    "u_in",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_lag5",
    "u_in_lag10",
    "u_in_lead1",
    "u_in_lead2",
    "du_in",
    "du_in2",
    "u_in_cumsum",
    "u_in_cummax",
    "u_in_area",
    "u_in_roll3_mean",
    "u_in_roll5_mean",
    "u_in_roll5_std",
    "u_in_roll10_mean",
    "u_in_roll10_std",
    "u_in_x_R",
    "u_in_x_C",
    "u_out_lag1",
    "u_out_lag2",
    "u_out_lead1",
]
feature_cols_cat = ["R", "C"]




## === cell 2
X_all = train_feat[feature_cols_num + feature_cols_cat]
y_all = train_feat["pressure"].astype(np.float32)
groups_all = train_feat["breath_id"].to_numpy()

sample_weight = (train_feat["u_out"].to_numpy() == 0).astype(np.float32)

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
            feature_cols_num,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            feature_cols_cat,
        ),
    ],
    remainder="drop",
)

model = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.08,
    max_depth=6,
    max_iter=400,
    random_state=SEED,
)

pipe = Pipeline([("prep", preprocess), ("model", model)])

gkf = GroupKFold(n_splits=5)
oof_pred = np.empty(len(train_feat), dtype=np.float32)

X_all_values = X_all  # keep DataFrame for ColumnTransformer column selection semantics
y_all_values = y_all.to_numpy()

for tr_idx, va_idx in gkf.split(X_all_values, y_all_values, groups=groups_all):
    pipe_fold = Pipeline([("prep", preprocess), ("model", model)])
    pipe_fold.fit(
        X_all_values.iloc[tr_idx],
        y_all_values[tr_idx],
        model__sample_weight=sample_weight[tr_idx],
    )
    oof_pred[va_idx] = pipe_fold.predict(X_all_values.iloc[va_idx]).astype(np.float32)

train_resid_oof = (y_all_values - oof_pred).astype(np.float32)
train_resid_oof[train_feat["u_out"].to_numpy() == 1] = 0.0


def add_uin_bin(df: pd.DataFrame, n_bins: int = 20) -> pd.Series:
    x = df["u_in"].astype(np.float32).to_numpy()
    b = np.floor((x / 100.0) * n_bins).astype(np.int16)
    b = np.clip(b, 0, n_bins - 1)
    return pd.Series(b, index=df.index, name="u_in_bin")


train_bias_src = train_feat.copy()
train_bias_src["u_in_bin"] = add_uin_bin(train_bias_src, n_bins=20)

bias_key_cols = ["R", "C", "breath_time", "u_out_lag1", "u_out_lead1", "u_in_bin"]
bias_df = train_bias_src[bias_key_cols].copy()
bias_df["resid"] = train_resid_oof
bias_df["w"] = sample_weight

bias_df["resid_w"] = (bias_df["resid"].to_numpy() * bias_df["w"].to_numpy()).astype(
    np.float32
)
agg = (
    bias_df.groupby(bias_key_cols, sort=False, observed=True)
    .agg(
        resid_w_sum=("resid_w", "sum"),
        w_sum=("w", "sum"),
    )
    .reset_index()
)
w_sum = agg["w_sum"].to_numpy(dtype=np.float32)
resid_w_sum = agg["resid_w_sum"].to_numpy(dtype=np.float32)
agg["resid"] = (resid_w_sum / np.maximum(w_sum, np.float32(1e-6))).astype(np.float32)
bias_table = agg.drop(columns=["resid_w_sum", "w_sum"])

pipe.fit(X_all, y_all, model__sample_weight=sample_weight)

test_features = test_feat[feature_cols_num + feature_cols_cat]
pred = pipe.predict(test_features).astype(np.float32)

test_bias_df = test_feat[bias_key_cols[:-1]].copy()  # all except u_in_bin for now
test_bias_df["u_in_bin"] = add_uin_bin(test_feat, n_bins=20).astype(np.int16)
test_bias_df["_row"] = np.arange(len(test_bias_df), dtype=np.int64)

test_bias_merged = test_bias_df.merge(
    bias_table, on=bias_key_cols, how="left", sort=False
)
test_bias_merged = test_bias_merged.sort_values("_row", kind="mergesort")
test_bias = test_bias_merged["resid"].fillna(0.0).astype(np.float32).to_numpy()

pred = (pred + test_bias).astype(np.float32)

pmin = float(train["pressure"].min())
pmax = float(train["pressure"].max())
pred = np.clip(pred, pmin, pmax).astype(np.float32)

pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)
idx = np.searchsorted(pressure_grid, pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
left_idx = np.clip(idx - 1, 0, len(pressure_grid) - 1)
right_idx = idx
left_val = pressure_grid[left_idx]
right_val = pressure_grid[right_idx]
pred = np.where(
    np.abs(pred - left_val) <= np.abs(pred - right_val), left_val, right_val
).astype(np.float32)

pred = pred.copy()
pred[test_feat["u_out"].to_numpy() == 1] = 0.0

if len(test) != len(pred):
    raise ValueError(f"Row count mismatch: test={len(test)} vs predictions={len(pred)}")

sub_out = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": pred})
sub_out.to_csv("submission.csv", index=False)

sub_out.head()
