# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.12

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

# 5. Code solution

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

DTYPES_TRAIN = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
DTYPES_TEST = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

TRAIN_USECOLS = list(DTYPES_TRAIN.keys()) + ["pickup_datetime"]
TEST_USECOLS = list(DTYPES_TEST.keys()) + ["pickup_datetime"]

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=10_000_000,
    dtype=DTYPES_TRAIN,
    usecols=TRAIN_USECOLS,
)
test_df = pd.read_csv(
    TEST_PATH,
    dtype=DTYPES_TEST,
    usecols=TEST_USECOLS,
)

train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)



## === cell 1
print("Loaded train/test:", train_df.shape, test_df.shape)



## === cell 2
print("Train dtypes:", train_df.dtypes.to_dict())



## === cell 3
pass



## === cell 4
num_rows = len(train_df)
mask_pos = train_df["fare_amount"].to_numpy(copy=False) > 0
train_df = train_df.loc[mask_pos]
print(f"Drop {num_rows - len(train_df)} rows")
del mask_pos
gc.collect()




## === cell 5
def get_valid_mask_np(df: pd.DataFrame) -> np.ndarray:
    plat = df["pickup_latitude"].to_numpy(copy=False)
    dlat = df["dropoff_latitude"].to_numpy(copy=False)
    plon = df["pickup_longitude"].to_numpy(copy=False)
    dlon = df["dropoff_longitude"].to_numpy(copy=False)
    pc = df["passenger_count"].to_numpy(copy=False)
    return (
        (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
        & (plon >= -74.3)
        & (plon <= -73.60)
        & (dlon >= -74.3)
        & (dlon <= -73.60)
        & (pc >= 1)
        & (pc <= 10)
    )


before_len = len(train_df)
train_mask = get_valid_mask_np(train_df)
train_df = train_df.loc[train_mask]
print("Dropped invalid/outlier rows from training:", before_len - len(train_df))

test_valid_mask = get_valid_mask_np(test_df)
print(
    "Test rows (unchanged):",
    len(test_df),
    "| invalid rows:",
    int((~test_valid_mask).sum()),
)
del train_mask
gc.collect()



## === cell 6
pass



## === cell 7
pass




## === cell 8
def preprocess_data(df: pd.DataFrame) -> None:
    airport_lat, airport_lon = 40.644600, -73.779700
    lga_lat, lga_lon = 40.7733, -73.8718

    plat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    plon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

    near_jfk = (
        (plat <= airport_lat + 0.005)
        & (plat >= airport_lat - 0.005)
        & (plon <= airport_lon + 0.005)
        & (plon >= airport_lon - 0.005)
    )
    near_lga = (
        (plat <= lga_lat + 0.002)
        & (plat >= lga_lat - 0.003)
        & (plon <= lga_lon + 0.005)
        & (plon >= lga_lon - 0.005)
    )
    df["near_airport"] = (near_jfk | near_lga).astype(np.int8)

    lon_diff = plon - dlon
    lat_diff = plat - dlat

    df["manhattan_distance"] = (np.abs(lon_diff) + np.abs(lat_diff)).astype(np.float32)
    df["abs_lon_diff"] = np.abs(lon_diff).astype(np.float32)
    df["abs_lat_diff"] = np.abs(lat_diff).astype(np.float32)
    df["euclidean_distance"] = np.sqrt(
        lon_diff * lon_diff + lat_diff * lat_diff
    ).astype(np.float32)

    R = np.float32(6371.0)  # km
    lat1 = np.radians(plat.astype(np.float64, copy=False))
    lon1 = np.radians(plon.astype(np.float64, copy=False))
    lat2 = np.radians(dlat.astype(np.float64, copy=False))
    lon2 = np.radians(dlon.astype(np.float64, copy=False))
    dlat_r = lat2 - lat1
    dlon_r = lon2 - lon1
    a = (
        np.sin(dlat_r / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2.0) ** 2
    )
    df["haversine_distance"] = (2.0 * float(R) * np.arcsin(np.sqrt(a))).astype(
        np.float32
    )

    dt = df["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=True)
        df["pickup_datetime"] = dt

    df["pickup_year"] = dt.dt.year.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_day"] = dt.dt.dayofweek.astype("float32")

    is_weekend = ((df["pickup_day"] >= 5) & (df["pickup_day"] <= 6)).astype(np.int8)
    df["is_weekend"] = is_weekend

    month = df["pickup_month"]
    day = dt.dt.day
    df["is_holiday"] = (
        ((month == 12) & (day == 25))
        | ((month == 12) & (day == 26))
        | ((month == 12) & (day == 31))
        | ((month == 1) & (day == 1))
        | ((month == 7) & (day == 4))
    ).astype(np.int8)


preprocess_data(train_df)
preprocess_data(test_df)

time_cols = ["pickup_year", "pickup_month", "pickup_hour", "pickup_day"]
before_len = len(train_df)
train_df = train_df.dropna(subset=time_cols)
print("Dropped train rows with invalid pickup_datetime:", before_len - len(train_df))
gc.collect()



## === cell 9
print("Train preview columns:", train_df.columns.tolist()[:10], "...")



## === cell 10
print("near_airport sum:", int(train_df["near_airport"].sum()))



## === cell 11
print("is_holiday sum:", int(train_df["is_holiday"].sum()))



## === cell 12
print("is_weekend sum:", int(train_df["is_weekend"].sum()))



## === cell 13
features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "near_airport",
    "manhattan_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "euclidean_distance",
    "haversine_distance",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "is_weekend",
    "is_holiday",
]

distance_like = [
    "manhattan_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "euclidean_distance",
    "haversine_distance",
]

dist_train = train_df[distance_like].to_numpy(dtype=np.float32, copy=False)
dist_train = np.where(np.isfinite(dist_train), dist_train, np.nan)

q_lo = np.nanquantile(dist_train, 0.001, axis=0)
q_hi = np.nanquantile(dist_train, 0.999, axis=0)

finite_min = np.nanmin(dist_train, axis=0)
finite_max = np.nanmax(dist_train, axis=0)

q_lo = np.where(np.isfinite(q_lo), q_lo, finite_min)
q_hi = np.where(np.isfinite(q_hi), q_hi, finite_max)

swap = q_hi < q_lo
q_lo2 = np.where(swap, q_hi, q_lo)
q_hi2 = np.where(swap, q_lo, q_hi)
q_lo, q_hi = q_lo2, q_hi2
del swap, q_lo2, q_hi2, finite_min, finite_max, dist_train
gc.collect()

for i, c in enumerate(distance_like):
    lo = float(q_lo[i])
    hi = float(q_hi[i])
    train_df[c] = train_df[c].clip(lo, hi)
    test_df[c] = test_df[c].clip(lo, hi)

train_medians = train_df[features].median(numeric_only=True)

for c in ["near_airport", "is_weekend", "is_holiday"]:
    test_df[c] = test_df[c].fillna(0)

test_df[features] = test_df[features].fillna(train_medians)

before_len = len(train_df)
train_df = train_df.dropna(subset=features + ["fare_amount"])
print("Dropped train rows with NaNs in features/target:", before_len - len(train_df))

X = train_df[features].to_numpy(dtype=np.float32, copy=False)
y = train_df["fare_amount"].to_numpy(dtype=np.float32, copy=False)

finite_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
if not finite_rows.all():
    X = X[finite_rows]
    y = y[finite_rows]
print("Final training rows:", X.shape[0])

fallback_fare = float(np.median(y))
print("Fallback fare (train median, original scale):", fallback_fare)

finite_y = y[np.isfinite(y)]
fare_cap_hi = float(np.quantile(finite_y, 0.999)) if finite_y.size else float(np.max(y))
if not np.isfinite(fare_cap_hi) or fare_cap_hi <= 0:
    fare_cap_hi = float(np.max(y))
print("Fare cap high (train 99.9th pct):", fare_cap_hi)

del train_df
gc.collect()



## === cell 14
X_test = test_df[features].to_numpy(dtype=np.float32, copy=False)
if not np.isfinite(X_test).all():
    med = train_medians.to_numpy(dtype=np.float32, copy=False)
    X_test = np.where(np.isfinite(X_test), X_test, med)



## === cell 15
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)



## === cell 16
print(X.shape, y.shape, X_train.shape, y_train.shape, X_val.shape, y_val.shape)



## === cell 17
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)



## === cell 18
validation_predictions_lr = lr_model.predict(X_val)
validation_rmse_lr = np.sqrt(mean_squared_error(y_val, validation_predictions_lr))
print("Validation RMSE (Linear Regression):", float(validation_rmse_lr))

train_predictions_lr = lr_model.predict(X)
train_rmse_lr = np.sqrt(mean_squared_error(y, train_predictions_lr))
print("Training RMSE (Linear Regression):", float(train_rmse_lr))



## === cell 19
test_predictions_lr = lr_model.predict(X_test)

submission_df_lr = pd.DataFrame(
    {
        "key": test_df["key"].astype("string").fillna(""),
        "fare_amount": test_predictions_lr,
    },
    columns=["key", "fare_amount"],
)
submission_df_lr.to_csv("lr_submission.csv", index=False)



## === cell 20
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "eta": 0.05,
    "min_child_weight": 3,
    "gamma": 0.1,
    "seed": 42,
    "tree_method": "hist",
    "nthread": -1,
}

y_train_log = np.log1p(np.maximum(y_train, 0.0))
y_val_log = np.log1p(np.maximum(y_val, 0.0))

cache_prefix = "/kaggle/working/xgb_cache"
dtrain = xgb.DMatrix(X_train, label=y_train_log, missing=np.nan, nthread=-1)
dval = xgb.DMatrix(X_val, label=y_val_log, missing=np.nan, nthread=-1)
dtest = xgb.DMatrix(X_test, missing=np.nan, nthread=-1)


def rmse_original_scale(preds_log, dmatrix):
    y_true_log = dmatrix.get_label()
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(preds_log)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return "rmse_orig", rmse


watchlist = [(dtrain, "train"), (dval, "valid")]
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=5000,
    evals=watchlist,
    feval=rmse_original_scale,
    maximize=False,
    early_stopping_rounds=100,
    verbose_eval=200,
)



## === cell 21
validation_predictions_xgb_log = xgb_model.predict(
    dval, iteration_range=(0, xgb_model.best_iteration + 1)
)
validation_predictions_xgb = np.expm1(validation_predictions_xgb_log)
validation_rmse_xgb = np.sqrt(mean_squared_error(y_val, validation_predictions_xgb))
print("Validation RMSE (XGBoost):", float(validation_rmse_xgb))
print("Best iteration chosen:", int(xgb_model.best_iteration))

dallTrain = xgb.DMatrix(X, missing=np.nan, nthread=-1)
train_predictions_xgb_log = xgb_model.predict(
    dallTrain, iteration_range=(0, xgb_model.best_iteration + 1)
)
train_predictions_xgb = np.expm1(train_predictions_xgb_log)
train_rmse_xgb = np.sqrt(mean_squared_error(y, train_predictions_xgb))
print("Train RMSE (XGBoost):", float(train_rmse_xgb))



## === cell 22
test_predictions_xgb_log = xgb_model.predict(
    dtest, iteration_range=(0, xgb_model.best_iteration + 1)
)
test_predictions_xgb = np.expm1(test_predictions_xgb_log)

test_predictions_xgb = np.where(test_valid_mask, test_predictions_xgb, fallback_fare)
test_predictions_xgb = np.maximum(test_predictions_xgb, 0.0)
test_predictions_xgb = np.minimum(test_predictions_xgb, fare_cap_hi)

submission_df_xgb = pd.DataFrame(
    {
        "key": test_df["key"].astype("string").fillna(""),
        "fare_amount": test_predictions_xgb,
    },
    columns=["key", "fare_amount"],
)
submission_df_xgb.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df_xgb.shape)
print("Submission dtypes:", submission_df_xgb.dtypes.to_dict())
print("Invalid test rows handled with fallback:", int((~test_valid_mask).sum()))
print("First 3 rows:\n", submission_df_xgb.head(3))
print(
    "Any NaN fare in submission?:",
    bool(pd.isna(submission_df_xgb["fare_amount"]).any()),
)
print("Fare stats:", submission_df_xgb["fare_amount"].describe().to_dict())
print("Rounds used (best_iteration+1):", int(xgb_model.best_iteration) + 1)
