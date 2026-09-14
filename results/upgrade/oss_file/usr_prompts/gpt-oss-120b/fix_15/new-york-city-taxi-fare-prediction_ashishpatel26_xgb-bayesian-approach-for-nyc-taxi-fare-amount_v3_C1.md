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
geopy==2.4.1
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")
import geopy.distance
import os

print(os.listdir("../input"))
import gc
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error

np.random.seed(42)
import random, os

random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=10_00_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=10_00_000, low_memory=True)
    return train, test




## === cell 2
train, test = load_Data()




## === cell 3
train = train.dropna(subset=["fare_amount"])
train = train[train["fare_amount"] > 0].reset_index(drop=True)
train = train.fillna(0)
test = test.fillna(0)




## === cell 4
train["key2"] = pd.to_datetime(train["key"], errors="coerce")
test["key2"] = pd.to_datetime(test["key"], errors="coerce")




## === cell 5
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


train["distance"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
train["direction"] = np.arctan2(
    train["dropoff_latitude"] - train["pickup_latitude"],
    train["dropoff_longitude"] - train["pickup_longitude"],
)
train["distance_squared"] = train["distance"] ** 2
train["log_distance_squared"] = np.log1p(train["distance_squared"])
train["direction_sin"] = np.sin(train["direction"])
train["direction_cos"] = np.cos(train["direction"])


test["distance"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)
test["direction"] = np.arctan2(
    test["dropoff_latitude"] - test["pickup_latitude"],
    test["dropoff_longitude"] - test["pickup_longitude"],
)
test["distance_squared"] = test["distance"] ** 2
test["log_distance_squared"] = np.log1p(test["distance_squared"])
test["direction_sin"] = np.sin(test["direction"])
test["direction_cos"] = np.cos(test["direction"])




## === cell 6
train["year"] = train["key2"].dt.year
train["month"] = train["key2"].dt.month
train["day"] = train["key2"].dt.day
train["day_of_week"] = train["key2"].dt.weekday
train["hour"] = train["key2"].dt.hour
train["week"] = train["key2"].dt.isocalendar().week
train["day_of_year"] = train["key2"].dt.dayofyear
train["week_of_year"] = train["key2"].dt.isocalendar().week
train["quarter"] = train["key2"].dt.quarter
train["is_weekend"] = train["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
train["dow_sin"] = np.sin(2 * np.pi * train["day_of_week"] / 7)
train["dow_cos"] = np.cos(2 * np.pi * train["day_of_week"] / 7)
train["month_sin"] = np.sin(2 * np.pi * train["month"] / 12)
train["month_cos"] = np.cos(2 * np.pi * train["month"] / 12)




## === cell 7
test["year"] = test["key2"].dt.year
test["month"] = test["key2"].dt.month
test["day"] = test["key2"].dt.day
test["day_of_week"] = test["key2"].dt.weekday
test["hour"] = test["key2"].dt.hour
test["week"] = test["key2"].dt.isocalendar().week
test["day_of_year"] = test["key2"].dt.dayofyear
test["week_of_year"] = test["key2"].dt.isocalendar().week
test["quarter"] = test["key2"].dt.quarter
test["is_weekend"] = test["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)
test["dow_sin"] = np.sin(2 * np.pi * test["day_of_week"] / 7)
test["dow_cos"] = np.cos(2 * np.pi * test["day_of_week"] / 7)
test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)




## === cell 8
train["delta_lat"] = train["dropoff_latitude"] - train["pickup_latitude"]
train["delta_lon"] = train["dropoff_longitude"] - train["pickup_longitude"]
train["abs_delta_lat"] = train["delta_lat"].abs()
train["abs_delta_lon"] = train["delta_lon"].abs()

test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
test["delta_lon"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["abs_delta_lat"] = test["delta_lat"].abs()
test["abs_delta_lon"] = test["delta_lon"].abs()

train["log_distance"] = np.log1p(train["distance"])
train["dist_pass"] = train["distance"] * train["passenger_count"]
test["log_distance"] = np.log1p(test["distance"])
test["dist_pass"] = test["distance"] * test["passenger_count"]

train["log_passenger"] = np.log1p(train["passenger_count"])
test["log_passenger"] = np.log1p(test["passenger_count"])

column_list = [
    "passenger_count",
    "log_passenger",
    "distance",
    "log_distance",  # kept once
    "distance_squared",
    "log_distance_squared",
    "direction",
    "direction_sin",
    "direction_cos",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "week",
    "day_of_year",
    "week_of_year",
    "quarter",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "delta_lat",
    "delta_lon",
    "abs_delta_lat",
    "abs_delta_lon",
    "dist_pass",
]

y_train_ = train["fare_amount"]
X_train_ = train[column_list].astype(np.float32)
X_test = test[column_list].astype(np.float32)




## === cell 9
mask = (train["distance"] <= 100) & (train["fare_amount"] <= 200)
X_train_ = X_train_[mask]
y_train_ = y_train_[mask]
print("After outlier filtering:", X_train_.shape, y_train_.shape)




## === cell 10
X_train, X_val, y_train, y_val = train_test_split(
    X_train_, y_train_, test_size=0.1, random_state=42
)




## === cell 11
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

lgb = LGBMRegressor(
    boosting_type="gbdt",
    colsample_bytree=0.9,
    learning_rate=0.005,
    max_depth=20,
    min_child_samples=30,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=15000,
    n_jobs=-1,
    num_leaves=400,
    reg_alpha=5.0,
    reg_lambda=3.0,
    subsample=0.8,
    subsample_for_bin=200000,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)

lgb.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[
        early_stopping(stopping_rounds=100, verbose=False),
        log_evaluation(period=0),
    ],
)




## === cell 12
pred_val_log = lgb.predict(X_val)
pred_val = np.expm1(pred_val_log)

print("RMSE on validation:", np.sqrt(mean_squared_error(y_val, pred_val)))




## === cell 13
best_iter = lgb.best_iteration_ if lgb.best_iteration_ else lgb.n_estimators
lgb.set_params(n_estimators=best_iter)

y_train_log_full = np.log1p(y_train_)
lgb.fit(X_train_, y_train_log_full)




## === cell 14
pred_test_log = lgb.predict(X_test)
y_pred = np.expm1(pred_test_log)




## === cell 15
submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
