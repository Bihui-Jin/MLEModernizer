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

os.environ["OMP_NUM_THREADS"] = "5"  # match rf n_jobs
os.environ["MKL_NUM_THREADS"] = "5"
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearnex import patch_sklearn

patch_sklearn()  # accelerate RandomForestRegressor & GradientBoostingRegressor




## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"
train_df = pd.read_csv(train_path, nrows=500_000)  # unchanged sample size
test_df = pd.read_csv(test_path)




## === cell 2
train_df.dropna(inplace=True)
train_df = train_df[train_df["fare_amount"] > 0]
train_df = train_df[train_df["fare_amount"] < 200]




## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    miles = km * 0.621371
    return miles


train_df["distance"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)
test_df["distance"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

train_df = train_df[train_df["distance"] < 15]
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]

train_dt = pd.to_datetime(train_df["pickup_datetime"])
train_df["hour"] = train_dt.dt.hour
train_df["minute"] = train_dt.dt.minute
train_df["dayofweek"] = train_dt.dt.dayofweek
train_df["month"] = train_dt.dt.month
train_df["year"] = train_dt.dt.year

test_dt = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_dt.dt.hour
test_df["minute"] = test_dt.dt.minute
test_df["dayofweek"] = test_dt.dt.dayofweek
test_df["month"] = test_dt.dt.month
test_df["year"] = test_dt.dt.year

train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24)
train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24)

test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)

train_df["log_distance"] = np.log1p(train_df["distance"])
test_df["log_distance"] = np.log1p(test_df["distance"])

train_df["distance_per_passenger"] = train_df["distance"] / train_df["passenger_count"]
test_df["distance_per_passenger"] = test_df["distance"] / test_df["passenger_count"]
train_df["distance_per_passenger"].replace([np.inf, -np.inf], np.nan, inplace=True)
test_df["distance_per_passenger"].replace([np.inf, -np.inf], np.nan, inplace=True)




## === cell 4
feat_cols = [
    "distance",
    "log_distance",
    "distance_per_passenger",
    "passenger_count",
    "hour",
    "minute",
    "dayofweek",
    "month",
    "year",
    "hour_sin",
    "hour_cos",
]

median_vals = train_df[feat_cols].median()
test_df[feat_cols] = test_df[feat_cols].fillna(median_vals)

X_np = train_df[feat_cols].values.astype(np.float32)
y_np = train_df["fare_amount"].values.astype(np.float32)
y_log_np = np.log1p(y_np).astype(np.float32)

X_train_np, X_val_np, y_train_log_np, y_val_np = train_test_split(
    X_np, y_log_np, test_size=0.2, random_state=42
)

del train_df, X_np, y_np, y_log_np
gc.collect()




## === cell 5
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from joblib import Parallel, delayed


def fit_rf():
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        max_features="sqrt",
        n_jobs=5,
        random_state=42,
    )
    rf.fit(X_train_np, y_train_log_np)
    return rf


def fit_gbr():
    gbr = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=6,
        random_state=42,
    )
    gbr.fit(X_train_np, y_train_log_np)
    return gbr


def fit_hgb():
    hgb = HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.03,
        max_depth=None,
        max_bins=255,
        random_state=42,
    )
    hgb.fit(X_train_np, y_train_log_np)
    return hgb


rf, gbr, hgb = Parallel(n_jobs=3, backend="threading")(
    [delayed(fit_rf)(), delayed(fit_gbr)(), delayed(fit_hgb)()]
)

rf_val_pred = np.expm1(rf.predict(X_val_np))
rf_rmse = mean_squared_error(y_val_np, rf_val_pred, squared=False)

gbr_val_pred = np.expm1(gbr.predict(X_val_np))
gbr_rmse = mean_squared_error(y_val_np, gbr_val_pred, squared=False)

hgb_val_pred = np.expm1(hgb.predict(X_val_np))
hgb_rmse = mean_squared_error(y_val_np, hgb_val_pred, squared=False)

del X_train_np, X_val_np, y_train_log_np, y_val_np
gc.collect()




## === cell 6
print(f"RF RMSE: {rf_rmse:.4f} | GBR RMSE: {gbr_rmse:.4f} | HGB RMSE: {hgb_rmse:.4f}")

test_features_np = test_df[feat_cols].values.astype(np.float32)

rf_test_log = rf.predict(test_features_np)
gbr_test_log = gbr.predict(test_features_np)
hgb_test_log = hgb.predict(test_features_np)

test_pred_log = (rf_test_log + gbr_test_log + hgb_test_log) / 3.0
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, a_min=0, a_max=None)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
