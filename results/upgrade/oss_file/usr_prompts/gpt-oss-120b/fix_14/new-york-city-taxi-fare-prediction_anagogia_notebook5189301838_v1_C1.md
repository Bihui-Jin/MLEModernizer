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

3.10

# 3. Installed packages

folium==0.20.0
geopandas==0.14.4
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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb




## === cell 1
train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=12_000_000,  # slightly larger sample to improve learning
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "key": "object",
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    memory_map=True,
)




## === cell 2
train = train.loc[
    train["pickup_latitude"].between(40, 42)
    & train["pickup_longitude"].between(-75, -72)
    & train["dropoff_latitude"].between(40, 42)
    & train["dropoff_longitude"].between(-75, -72)
    & (train["fare_amount"] >= 2.5)
    & (train["passenger_count"] > 0)
    & (train["passenger_count"] <= 6)
].copy()




## === cell 3
train["year"] = train["pickup_datetime"].dt.year
train["month"] = train["pickup_datetime"].dt.month
train["day"] = train["pickup_datetime"].dt.day
train["weekday"] = train["pickup_datetime"].dt.weekday
train["hour"] = train["pickup_datetime"].dt.hour




## === cell 4
def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))


train["distance"] = haversine(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
)

train["log_distance"] = np.log1p(train["distance"])
train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
train["dist_per_passenger"] = train["distance"] / train["passenger_count"]
train = train[train["distance"] > 0]




## === cell 5
train = train.drop(columns=["key", "pickup_datetime"])




## === cell 6
y = np.log1p(train["fare_amount"]).astype(np.float32)
X = train.drop(columns=["fare_amount"]).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=50)

X_train_np = X_train.values
X_val_np = X_val.values
y_train_np = y_train.values
y_val_np = y_val.values




## === cell 7
lgb_params = {
    "learning_rate": 0.02,
    "objective": "regression",
    "num_leaves": 4095,
    "max_depth": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 5,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "metric": "rmse",
    "n_jobs": -1,
    "max_bin": 255,
}
train_set = lgb.Dataset(X_train_np, label=y_train_np)
val_set = lgb.Dataset(X_val_np, label=y_val_np, reference=train_set)

gbm = lgb.train(
    lgb_params,
    train_set,
    num_boost_round=1500,  # reduced upper bound; early stopping will keep optimal iter
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)




## === cell 8
best_iter = gbm.best_iteration
val_pred_log = gbm.predict(X_val_np, num_iteration=best_iter)
val_pred = np.expm1(val_pred_log)
rmse = mean_squared_error(np.expm1(y_val_np), val_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")




## === cell 9
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["weekday"] = test["pickup_datetime"].dt.weekday
test["hour"] = test["pickup_datetime"].dt.hour

test["distance"] = haversine(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)

test["log_distance"] = np.log1p(test["distance"])
test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)
test["dist_per_passenger"] = test["distance"] / test["passenger_count"]

X_test = test.drop(columns=["key", "pickup_datetime"]).astype(np.float32)
X_test_np = X_test.values




## === cell 10
test_pred_log = gbm.predict(X_test_np, num_iteration=best_iter)
test_predictions = np.expm1(test_pred_log)




## === cell 11
submission = pd.DataFrame({"key": test["key"], "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
