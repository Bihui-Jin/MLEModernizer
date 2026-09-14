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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))




## === cell 1
training_data = pd.read_csv("../input/train.csv", nrows=4_000_000)
test_data = pd.read_csv("../input/test.csv")

X_train = training_data.copy()
X_train["pickup_datetime"] = pd.to_datetime(X_train["pickup_datetime"])
X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek
X_train["minute"] = X_train["pickup_datetime"].dt.minute
X_train["month"] = X_train["pickup_datetime"].dt.month
X_train["is_weekend"] = (X_train["dayofweek"] >= 5).astype(int)

X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()
X_train["total_abs_distance"] = (
    X_train["latitude_distance"] + X_train["longitude_distance"]
)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometres
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


X_train["haversine_distance"] = haversine(
    X_train["pickup_latitude"],
    X_train["pickup_longitude"],
    X_train["dropoff_latitude"],
    X_train["dropoff_longitude"],
)

mean_lat = X_train[["pickup_latitude", "dropoff_latitude"]].mean().mean()
lat_km = 111.0
lon_km = 111.0 * np.cos(np.radians(mean_lat))

X_train["manhattan_distance"] = (
    X_train["latitude_distance"] * lat_km + X_train["longitude_distance"] * lon_km
)

X_train["passenger_count_sq"] = X_train["passenger_count"] ** 2

X_train["log_haversine_distance"] = np.log1p(X_train["haversine_distance"])
X_train["log_manhattan_distance"] = np.log1p(X_train["manhattan_distance"])
X_train["log_total_abs_distance"] = np.log1p(X_train["total_abs_distance"])
X_train["hour_sin"] = np.sin(2 * np.pi * X_train["hour"] / 24)
X_train["hour_cos"] = np.cos(2 * np.pi * X_train["hour"] / 24)

X_train["passenger_count_log"] = np.log1p(X_train["passenger_count"])
X_train["distance_per_passenger"] = X_train["haversine_distance"] / (
    X_train["passenger_count"] + 1
)

X_train = X_train.drop(
    columns=[
        "dropoff_longitude",
        "dropoff_latitude",
        "key",
        "pickup_datetime",
        "fare_amount",  # target removed from features
    ]
)

X_test = test_data.copy()
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"])
X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek
X_test["minute"] = X_test["pickup_datetime"].dt.minute
X_test["month"] = X_test["pickup_datetime"].dt.month
X_test["is_weekend"] = (X_test["dayofweek"] >= 5).astype(int)

X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()
X_test["total_abs_distance"] = (
    X_test["latitude_distance"] + X_test["longitude_distance"]
)

X_test["haversine_distance"] = haversine(
    X_test["pickup_latitude"],
    X_test["pickup_longitude"],
    X_test["dropoff_latitude"],
    X_test["dropoff_longitude"],
)

X_test["manhattan_distance"] = (
    X_test["latitude_distance"] * lat_km + X_test["longitude_distance"] * lon_km
)

X_test["passenger_count_sq"] = X_test["passenger_count"] ** 2

X_test["log_haversine_distance"] = np.log1p(X_test["haversine_distance"])
X_test["log_manhattan_distance"] = np.log1p(X_test["manhattan_distance"])
X_test["log_total_abs_distance"] = np.log1p(X_test["total_abs_distance"])
X_test["hour_sin"] = np.sin(2 * np.pi * X_test["hour"] / 24)
X_test["hour_cos"] = np.cos(2 * np.pi * X_test["hour"] / 24)

X_test["passenger_count_log"] = np.log1p(X_test["passenger_count"])
X_test["distance_per_passenger"] = X_test["haversine_distance"] / (
    X_test["passenger_count"] + 1
)

X_test = X_test.drop(
    columns=["dropoff_longitude", "dropoff_latitude", "key", "pickup_datetime"]
)

valid_mask = (
    (training_data["fare_amount"] > 0)
    & (training_data["fare_amount"] <= 200)  # filter extreme values
    & X_train.notnull().all(axis=1)
)

X_train = X_train[valid_mask].reset_index(drop=True)
Y_train = np.log1p(training_data.loc[valid_mask, "fare_amount"])

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)




## === cell 2
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=42
)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=5000,
    max_depth=9,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=4,
    random_state=42,
    tree_method="hist",  # fast histogram algorithm
    max_bin=256,  # limit bins for speed, same tree structure
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=False,
)




## === cell 3
Y_pred = np.expm1(model.predict(X_test))
Y_pred = np.clip(Y_pred, 0, 200)  # clip to realistic range




## === cell 4
submission = test_data[["key"]].copy()
submission["fare_amount"] = Y_pred
submission = submission[["key", "fare_amount"]]




## === cell 5
submission.to_csv("sample_submission.csv", index=False)
print("Submission saved to sample_submission.csv, shape:", submission.shape)
