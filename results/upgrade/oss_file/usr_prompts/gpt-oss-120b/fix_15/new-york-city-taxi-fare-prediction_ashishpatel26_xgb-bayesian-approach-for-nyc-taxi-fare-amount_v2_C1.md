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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb  # import the module for callbacks

plt.style.use("fivethirtyeight")
print("Input directory contents:", os.listdir("../input"))
import gc




## === cell 1
def load_Data():
    dtype_spec = {
        "key": "object",
        "fare_amount": "float32",
        "pickup_datetime": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    }
    usecols_train = list(dtype_spec.keys())
    usecols_test = [c for c in usecols_train if c != "fare_amount"]
    train = pd.read_csv(
        "../input/train.csv",
        nrows=2_000_000,
        low_memory=True,
        usecols=usecols_train,
        dtype=dtype_spec,
    )
    test = pd.read_csv(
        "../input/test.csv",
        nrows=2_000_000,
        low_memory=True,
        usecols=usecols_test,
        dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
    )
    return train, test


def prepare_distance_features(df):
    df["longitude_distance"] = np.abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ).astype(np.float32)
    df["latitude_distance"] = np.abs(
        df["pickup_latitude"] - df["dropoff_latitude"]
    ).astype(np.float32)
    df["distance_travelled"] = np.sqrt(
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ).astype(np.float32)
    df["distance_travelled_sin"] = np.sin(df["distance_travelled"]).astype(np.float32)
    df["distance_travelled_cos"] = np.cos(df["distance_travelled"]).astype(np.float32)
    df["distance_travelled_sin_sqrd"] = (df["distance_travelled_sin"] ** 2).astype(
        np.float32
    )
    df["distance_travelled_cos_sqrd"] = (df["distance_travelled_cos"] ** 2).astype(
        np.float32
    )

    R = 6371e3  # metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    dphi = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlambda = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine"] = (R * c).astype(np.float32)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    df["bearing"] = np.arctan2(y, x).astype(np.float32)
    return df


def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "", regex=False)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["hour_of_day"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["week"] = df["pickup_datetime"].dt.isocalendar().week.astype(np.int16)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear.astype(np.int16)
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype(np.int8)
    df["quarter"] = df["pickup_datetime"].dt.quarter.astype(np.int8)
    return df




## === cell 2
train, test = load_Data()

train = (
    train.dropna(subset=["fare_amount"])
    .loc[lambda df: df["fare_amount"] > 0]
    .reset_index(drop=True)
)




## === cell 3
train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)




## === cell 4
train["distance"] = (train["haversine"] / 1000.0).astype(np.float32)  # km
test["distance"] = (test["haversine"] / 1000.0).astype(np.float32)

train["distance_per_passenger"] = (
    train["distance"] / train["passenger_count"].replace(0, np.nan)
).astype(np.float32)
test["distance_per_passenger"] = (
    test["distance"] / test["passenger_count"].replace(0, np.nan)
).astype(np.float32)

train["distance_per_passenger"].fillna(0, inplace=True)
test["distance_per_passenger"].fillna(0, inplace=True)

train["log_distance"] = np.log1p(train["distance"]).astype(np.float32)
test["log_distance"] = np.log1p(test["distance"]).astype(np.float32)

train.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
        "haversine",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_datetime",
        "haversine",
    ],
    axis=1,
    inplace=True,
)




## === cell 5
train["key2"] = pd.to_datetime(train["key"], errors="coerce")
test["key2"] = pd.to_datetime(test["key"], errors="coerce")

for df in (train, test):
    df["year"] = df["key2"].dt.year.astype(np.int16)
    df["month"] = df["key2"].dt.month.astype(np.int8)
    df["day"] = df["key2"].dt.day.astype(np.int8)
    df["day_of_week"] = df["key2"].dt.weekday.astype(np.int8)
    df["hour"] = df["key2"].dt.hour.astype(np.int8)

train["passenger_count_log"] = np.log1p(train["passenger_count"]).astype(np.float32)
test["passenger_count_log"] = np.log1p(test["passenger_count"]).astype(np.float32)

train["distance_per_passenger_log"] = np.log1p(train["distance_per_passenger"]).astype(
    np.float32
)
test["distance_per_passenger_log"] = np.log1p(test["distance_per_passenger"]).astype(
    np.float32
)




## === cell 6
feature_cols = [
    "passenger_count",
    "passenger_count_log",
    "distance",
    "distance_per_passenger",
    "distance_per_passenger_log",
    "log_distance",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "hour_of_day",
    "week",
    "day_of_year",
    "weekday",
    "quarter",
    "distance_travelled",
    "distance_travelled_sin",
    "distance_travelled_cos",
    "distance_travelled_sin_sqrd",
    "distance_travelled_cos_sqrd",
    "bearing",
]
X = train[feature_cols].astype(np.float32)
y = np.log1p(train["fare_amount"]).astype(np.float32)

X = X.fillna(0)




## === cell 7
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=1023,  # slightly larger capacity
    learning_rate=0.005,
    n_estimators=8000,
    max_bin=255,
    bagging_fraction=0.9,  # increased bagging
    bagging_freq=3,
    feature_fraction=0.9,  # increased feature sampling
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    verbose=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {rmse:.4f}")

best_iter = (
    model.best_iteration_
    if hasattr(model, "best_iteration_") and model.best_iteration_
    else model.n_estimators
)

final_model = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=1023,
    learning_rate=0.005,
    n_estimators=best_iter,
    max_bin=255,
    bagging_fraction=0.9,
    bagging_freq=3,
    feature_fraction=0.9,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    verbose=-1,
)
final_model.fit(X, y)




## === cell 8
X_test = test[feature_cols].astype(np.float32).fillna(0)
test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

test_pred = np.where(test_pred < 0, 0, test_pred)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
