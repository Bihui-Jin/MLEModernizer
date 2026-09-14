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
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split




## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train = pd.read_csv(
    "../input/train.csv",
    usecols=usecols,
    dtype=dtype_spec,
    nrows=10_000_000,
    low_memory=False,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
    low_memory=False,
)

print(train.dtypes)




## === cell 2
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())




## === cell 3
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -150) & (train["pickup_longitude"] < 0)]
train = train.loc[(train["pickup_latitude"] > 0) & (train["pickup_latitude"] < 80)]
train = train.loc[
    (train["dropoff_longitude"] > -150) & (train["dropoff_longitude"] < 0)
]
train = train.loc[(train["dropoff_latitude"] > 0) & (train["dropoff_latitude"] < 80)]
train = train.loc[train["passenger_count"] <= 8]
train.describe()




## === cell 4
combine = [train, test]
for dataset in combine:
    dataset["longitude_distance"] = np.abs(
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    )
    dataset["latitude_distance"] = np.abs(
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    )
    dataset["distance_travelled"] = np.sqrt(
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    )
    dataset["distance_travelled_sin"] = np.sin(dataset["distance_travelled"])
    dataset["distance_travelled_cos"] = np.cos(dataset["distance_travelled"])
    dataset["distance_travelled_sin_sqrd"] = np.sin(dataset["distance_travelled"]) ** 2
    dataset["distance_travelled_cos_sqrd"] = np.cos(dataset["distance_travelled"]) ** 2

    R = 6371e3
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    delta_phi = np.radians(dataset["dropoff_latitude"] - dataset["pickup_latitude"])
    delta_lambda = np.radians(
        dataset["dropoff_longitude"] - dataset["pickup_longitude"]
    )
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    dataset["log_haversine"] = np.log1p(dataset["haversine"])

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    dataset["bearing"] = np.arctan2(y, x)

    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["week"] = dataset["pickup_datetime"].dt.isocalendar().week.astype(int)
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear
    dataset["week_of_year"] = (
        dataset["pickup_datetime"].dt.isocalendar().week.astype(int)
    )
    dataset["day_of_week"] = dataset["pickup_datetime"].dt.dayofweek
    dataset["is_weekend"] = dataset["day_of_week"].isin([5, 6]).astype(int)

    dataset["hour_sin"] = np.sin(2 * np.pi * dataset["hour_of_day"] / 24)
    dataset["hour_cos"] = np.cos(2 * np.pi * dataset["hour_of_day"] / 24)

train.head(3)




## === cell 5
train_features_to_keep = [
    "haversine",
    "log_haversine",
    "distance_travelled",
    "distance_travelled_sin",
    "distance_travelled_cos",
    "distance_travelled_sin_sqrd",
    "distance_travelled_cos_sqrd",
    "longitude_distance",
    "latitude_distance",
    "bearing",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "day",
    "week",
    "month",
    "day_of_year",
    "day_of_week",
    "is_weekend",
    "passenger_count",
    "fare_amount",
]
train.drop(
    columns=train.columns.difference(train_features_to_keep), axis=1, inplace=True
)

test_features_to_keep = [
    "haversine",
    "log_haversine",
    "distance_travelled",
    "distance_travelled_sin",
    "distance_travelled_cos",
    "distance_travelled_sin_sqrd",
    "distance_travelled_cos_sqrd",
    "longitude_distance",
    "latitude_distance",
    "bearing",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "day",
    "week",
    "month",
    "day_of_year",
    "day_of_week",
    "is_weekend",
    "passenger_count",
    "key",
]
test.drop(columns=test.columns.difference(test_features_to_keep), axis=1, inplace=True)

float_cols = [c for c in train.columns if c not in ("fare_amount", "key")]
train[float_cols] = train[float_cols].astype(np.float32)
test[float_cols] = test[float_cols].astype(np.float32)
train["fare_amount"] = train["fare_amount"].astype(np.float32)




## === cell 6
x_pred = test.drop("key", axis=1)

X = train.drop("fare_amount", axis=1)
y = np.log1p(train["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


def XGBmodel(x_train, x_test, y_train, y_test):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "eta": 0.015,  # slightly lower learning rate
        "max_depth": 10,  # a bit deeper tree
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "tree_method": "hist",
        "nthread": 5,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtest, "test")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

prediction_log = model.predict(xgb.DMatrix(x_pred))
prediction = np.expm1(prediction_log)
prediction = np.clip(prediction, 0, None)




## === cell 7
submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction})
submission.to_csv("sub_fare.csv", index=False)
print("Submission saved to sub_fare.csv")




## === cell 8
submission.head()
