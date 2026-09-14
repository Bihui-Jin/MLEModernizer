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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.20766

# 6. Current score

5.7278

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.07789) has done: 'I fix the hard runtime errors so the notebook runs end-to-end and always writes a valid `submission.csv` in the required `key,fare_amount` format. The main scoring model be the existing XGBoost approach (kept the same training loop/semantics), but updated for XGBoost 2.0+ API (no deprecated `reg:linear` and no `best_ntree_limit` attribute). I also remove the TensorFlow Estimator section (it cannot run in TF 2.18 where `tf.estimator` is not available) so it doesn’t break execution after a valid submission is already created. Finally, I keep feature engineering identical but make datetime parsing vectorized (score-neutral) and add a small non-negative clamp on predictions (prevents obviously invalid fares and typically improves RMSE slightly).'
- What this solution (achieved 6.06599) has done: 'Your current score (6.07789 RMSE; lower is better) is far from the target (4.20766), so we should make small but meaningful improvements without changing the core approach (same simple features + XGBoost training). The biggest issue is that you train on only 10,000 rows from a 55M-row dataset; increasing the training sample size (still via `read_csv(..., nrows=...)`) typically yields a large RMSE drop while keeping the same modeling logic. I also add a minimal, competition-standard coordinate sanity filter (NYC bounding box + valid lat/lon ranges) and a fare cap filter to remove extreme outliers; this usually improves generalization for RMSE with negligible code change. Finally, I keep the non-negative clamp and ensure the submission remains `key,fare_amount` written to `submission.csv`.'
- What this solution (achieved 5.41921) has done: 'I make the pipeline always output a valid submission with the full 9,914 `key`s by removing the test-set coordinate filtering that currently drops rows and misaligns predictions/keys (a common cause of invalid submissions or poor RMSE). To keep the same core model/feature logic, I instead apply the coordinate bounding-box filter only to the training data (where it helps denoise) and “clip” test coordinates into sane ranges so feature generation stays stable without losing any rows. I also ensure datetime-derived features exist for every test row (fill missing hours/years with training medians) and keep the existing non-negative prediction clamp. These are minimal changes that typically improve RMSE substantially versus accidentally scoring on a malformed or misaligned submission.'
- What this solution (achieved 6.19804) has done: 'The timeout is dominated by (1) parsing 5M rows from CSV with default dtypes and datetime parsing, and especially (2) fitting a 500-tree `RandomForestRegressor` on millions of rows, which is far too slow in a 600s budget. To preserve the core logic (same features, same models, same training flow), the key speedups are: use fast CSV reading (explicit `usecols` + compact `dtype` + `engine="c"`), avoid expensive intermediate pandas ops by working in NumPy where equivalent, and enable Intel-optimized scikit-learn (already installed) to accelerate the RandomForest without changing semantics. XGBoost training is kept the same but is fed `float32` arrays to reduce overhead, and we avoid building `DMatrix` from pandas objects.'
- What this solution (achieved 5.86878) has done: 'Your current RMSE (6.19804; lower is better) is still far above the target (4.20766), so we should make a small, safe improvement that usually yields a meaningful RMSE drop without changing the core approach (same engineered features + XGBoost). The main issue is the extremely limited feature set: adding two standard NYC-taxi baseline features (absolute lat/lon deltas) preserves the same model/training flow but gives the tree model more signal than distance alone. I also adjust XGBoost to train for a few more rounds (still with early stopping) to let it reach a better fit, and keep the submission writing logic identical. These changes are minimal, fast, and should move the score materially toward the target band.'
- What this solution (achieved 5.86588) has done: 'Your current RMSE (5.86878, lower is better) is still far from the target (4.20766), so we should make one small, high-impact improvement without changing the overall approach (same feature engineering + XGBoost training/prediction flow). The biggest low-risk gain is to train XGBoost on a more representative validation set and include the training-set mean as a simple baseline feature via an intercept-like term (tree models often benefit from a “bias” feature when features are sparse). I also make the train/valid split size proportional (rather than a fixed 200k) so early stopping behaves consistently as you scale training rows. Finally, I keep the same submission writing logic and constraints (full 9,914 keys in original order, non-negative clamp).'
- What this solution (achieved 5.97772) has done: 'The timeout is dominated by fitting a 500-tree `RandomForestRegressor` on millions of rows plus doing extra DataFrame work and plotting; the XGBoost part is comparatively efficient. I keep the same overall pipeline and models, but remove non-essential expensive steps (EDA/plotting) and make feature construction and splitting use NumPy arrays to avoid repeated pandas overhead and copies. I also ensure we don’t hold multiple large copies of the training data in memory, and I keep determinism intact while letting sklearn-intelex accelerate where available. The core logic (same features, same RF + XGBoost training and prediction semantics, same file outputs) is preserved.'
- What this solution (achieved 5.7278) has done: 'We make one small, high-impact change that keeps your current feature set and XGBoost training flow intact: add a simple, deterministic linear blending of the RandomForest and XGBoost predictions. This preserves both models and all existing training semantics, but often reduces RMSE versus either model alone, moving you closer to the target. We choose a conservative fixed blend weight (no extra tuning loops) and keep the non-negative clamp and submission format unchanged. Everything still runs end-to-end within the same I/O paths and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

import os
import gc

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
import os

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
    TEST_PATH = "/kaggle/input/test.csv"
    SAMPLE_PATH = "/kaggle/input/sample_submission.csv"



## === cell 2
NROWS_TRAIN = 5_000_000  # keep within typical Kaggle memory/time limits

usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=NROWS_TRAIN,
    usecols=usecols_train,
    dtype=dtype_train,
    engine="c",
    low_memory=False,
)
train_df.shape



## === cell 3
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
test_df = pd.read_csv(
    TEST_PATH, usecols=usecols_test, dtype=dtype_test, engine="c", low_memory=False
)
test_df.shape



## === cell 4
pass



## === cell 5
train_df.dropna(inplace=True)



## === cell 6
pass



## === cell 7
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 150)]
train_df.shape




## === cell 8
def distance(lat1, lon1, lat2, lon2):
    a = (
        0.5
        - np.cos((lat2 - lat1) * 0.017453292519943295) / 2
        + np.cos(lat1 * 0.017453292519943295)
        * np.cos(lat2 * 0.017453292519943295)
        * (1 - np.cos((lon2 - lon1) * 0.017453292519943295))
        / 2
    )
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res




## === cell 9
def coord_filter(df):
    return df[
        (df["pickup_latitude"].between(-90, 90))
        & (df["dropoff_latitude"].between(-90, 90))
        & (df["pickup_longitude"].between(-180, 180))
        & (df["dropoff_longitude"].between(-180, 180))
        & (df["pickup_latitude"].between(40, 42))
        & (df["dropoff_latitude"].between(40, 42))
        & (df["pickup_longitude"].between(-75, -72))
        & (df["dropoff_longitude"].between(-75, -72))
    ]


train_df = coord_filter(train_df)

test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(lower=40, upper=42)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(lower=40, upper=42)
test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(lower=-75, upper=-72)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(lower=-75, upper=-72)



## === cell 10
pl = train_df["pickup_latitude"].to_numpy(copy=False)
plo = train_df["pickup_longitude"].to_numpy(copy=False)
dl = train_df["dropoff_latitude"].to_numpy(copy=False)
dlo = train_df["dropoff_longitude"].to_numpy(copy=False)

train_df["distance"] = distance(pl, plo, dl, dlo).astype(np.float32)

tpl = test_df["pickup_latitude"].to_numpy(copy=False)
tplo = test_df["pickup_longitude"].to_numpy(copy=False)
tdl = test_df["dropoff_latitude"].to_numpy(copy=False)
tdlo = test_df["dropoff_longitude"].to_numpy(copy=False)

test_df["distance"] = distance(tpl, tplo, tdl, tdlo).astype(np.float32)



## === cell 11
DIST_CAP = 30.0
train_df["distance"] = np.minimum(
    train_df["distance"].to_numpy(copy=False), DIST_CAP
).astype(np.float32)
test_df["distance"] = np.minimum(
    test_df["distance"].to_numpy(copy=False), DIST_CAP
).astype(np.float32)

train_df = train_df[train_df["distance"] >= 0.05]



## === cell 12
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]
train_df.shape



## === cell 13
test_df["passenger_count"] = test_df["passenger_count"].clip(lower=1, upper=9)



## === cell 14
train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce", utc=False)
test_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce", utc=False)

train_df["hour"] = train_dt.dt.hour.astype("float32")
train_df["year"] = train_dt.dt.year.astype("float32")

test_df["hour"] = test_dt.dt.hour.astype("float32")
test_df["year"] = test_dt.dt.year.astype("float32")

train_df = train_df.dropna(subset=["hour", "year"])

hour_med = int(train_df["hour"].median())
year_med = int(train_df["year"].median())
test_df["hour"] = test_df["hour"].fillna(hour_med).astype(int)
test_df["year"] = test_df["year"].fillna(year_med).astype(int)



## === cell 15
train_df["abs_lon_diff"] = np.abs(
    train_df["pickup_longitude"].to_numpy(copy=False)
    - train_df["dropoff_longitude"].to_numpy(copy=False)
).astype(np.float32)
train_df["abs_lat_diff"] = np.abs(
    train_df["pickup_latitude"].to_numpy(copy=False)
    - train_df["dropoff_latitude"].to_numpy(copy=False)
).astype(np.float32)

test_df["abs_lon_diff"] = np.abs(
    test_df["pickup_longitude"].to_numpy(copy=False)
    - test_df["dropoff_longitude"].to_numpy(copy=False)
).astype(np.float32)
test_df["abs_lat_diff"] = np.abs(
    test_df["pickup_latitude"].to_numpy(copy=False)
    - test_df["dropoff_latitude"].to_numpy(copy=False)
).astype(np.float32)



## === cell 16
train_df["bias"] = np.float32(1.0)
test_df["bias"] = np.float32(1.0)

feat_cols_s = [
    "distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "passenger_count",
    "hour",
    "year",
    "bias",
]

X_all = train_df[feat_cols_s].to_numpy(dtype=np.float32, copy=False)
y_all = train_df["fare_amount"].to_numpy(dtype=np.float32, copy=False)
X_test = test_df[feat_cols_s].to_numpy(dtype=np.float32, copy=False)

train_df = train_df[
    ["fare_amount"]
]  # minimal keep to preserve variable existence without large footprint
gc.collect()



## === cell 17
from sklearn.model_selection import train_test_split

valid_size = 0.05
idx = np.arange(X_all.shape[0], dtype=np.int64)
idx_train, idx_valid = train_test_split(idx, test_size=valid_size, random_state=42)

X_train = X_all[idx_train]
y_train = y_all[idx_train]
X_valid = X_all[idx_valid]
y_valid = y_all[idx_valid]

del X_all, y_all, idx, idx_train, idx_valid
gc.collect()



## === cell 18
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

r_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
r_reg.fit(X_train, y_train)

rf_pred = r_reg.predict(X_test)
rf_pred = np.maximum(rf_pred, 0.0)

rf_submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": rf_pred},
    columns=["key", "fare_amount"],
)
rf_submission.to_csv("Random Forest regression.csv", index=False)



## === cell 19
import xgboost as xgb




## === cell 20
def XGBoost(X_train, X_valid, y_train, y_valid, num_rounds=1200):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster




## === cell 21
xgbm = XGBoost(X_train, X_valid, y_train, y_valid)

dtest = xgb.DMatrix(X_test)

if hasattr(xgbm, "best_iteration") and xgbm.best_iteration is not None:
    xgbm_pred = xgbm.predict(dtest, iteration_range=(0, xgbm.best_iteration + 1))
else:
    xgbm_pred = xgbm.predict(dtest)

xgbm_pred = np.maximum(xgbm_pred, 0.0)

submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": xgbm_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
submission.to_csv("XGboost regression.csv", index=False)

submission.head()



## === cell 22
BLEND_ALPHA = (
    0.85  # weight on XGBoost; conservative since XGB usually outperforms RF here
)
blend_pred = (BLEND_ALPHA * xgbm_pred + (1.0 - BLEND_ALPHA) * rf_pred).astype(
    np.float32
)
blend_pred = np.maximum(blend_pred, 0.0)

blend_submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": blend_pred},
    columns=["key", "fare_amount"],
)
blend_submission.to_csv("submission.csv", index=False)  # keep expected output filename
blend_submission.to_csv("Blended_XGB_RF_submission.csv", index=False)

blend_submission.head()



## === cell 23
print(
    "Pipeline completed. A valid full-length submission was written to 'submission.csv' "
    "with columns ['key','fare_amount'] and the original test key order."
)
print("submission.csv rows:", len(blend_submission), " expected:", len(test_df))
print("Any NaNs in predictions:", np.isnan(blend_submission["fare_amount"]).any())
