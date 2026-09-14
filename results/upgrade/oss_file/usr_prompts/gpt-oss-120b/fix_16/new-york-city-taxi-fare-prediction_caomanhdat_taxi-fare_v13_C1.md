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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import xgboost as xgb
from sklearn.model_selection import train_test_split




## === cell 1
train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtypes_train = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.uint8,
    "pickup_datetime": object,
}
dtypes_test = {
    "key": object,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.uint8,
    "pickup_datetime": object,
}
train = pd.read_csv(
    "../input/train.csv",
    nrows=5_000_000,
    usecols=train_usecols,
    dtype=dtypes_train,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=test_usecols,
    dtype=dtypes_test,
)




## === cell 2
def handle_date(df):
    s = df["pickup_datetime"]
    if s.str.endswith(" UTC").any():
        s = s.str.slice(stop=-4)
    dt = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", utc=False)

    df["hour_of_day"] = dt.dt.hour
    df["week"] = dt.dt.isocalendar().week.astype(np.int16)
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["day_of_year"] = dt.dt.dayofyear
    df["weekday"] = dt.dt.weekday
    df["quarter"] = dt.dt.quarter
    df["day_of_month"] = dt.dt.day

    hour = df["hour_of_day"].values
    weekday = df["weekday"].values

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    df["is_weekend"] = (weekday >= 5).astype(np.uint8)
    df["is_night"] = ((hour >= 20) | (hour <= 5)).astype(np.uint8)

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df




## === cell 3
train = handle_date(train)
test = handle_date(test)




## === cell 4
def add_all_distance_features(df):
    lon_diff = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    lat_diff = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance_travelled"] = np.sqrt(lon_diff**2 + lat_diff**2).astype(np.float32)

    r = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance_haversine"] = (r * c).astype(np.float32)

    df["distance_manhattan"] = (lon_diff + lat_diff).astype(np.float32)
    return df




## === cell 5
train = add_all_distance_features(train)
test = add_all_distance_features(test)




## === cell 6
def add_log_and_interaction_features(df):
    df["log_haversine"] = np.log1p(df["distance_haversine"])
    df["log_manhattan"] = np.log1p(df["distance_manhattan"])
    df["log_euclidean"] = np.log1p(df["distance_travelled"])
    df["passenger_distance_product"] = df["passenger_count"] * df["distance_haversine"]
    return df


train = add_log_and_interaction_features(train)
test = add_log_and_interaction_features(test)




## === cell 7
def clean_up_train(df):
    df = df.dropna()
    df = df[df["fare_amount"] > 0]
    df = df[df["passenger_count"] > 0]
    df = df[df["passenger_count"] < 7]
    df = df[df["fare_amount"] < 200]
    return df


train = clean_up_train(train)




## === cell 8
def get_samples_output(train_df):
    feature_cols = test.drop("key", axis=1).columns
    return train_df[feature_cols], train_df["fare_amount"]


samples_train, samples_label = get_samples_output(train)
samples_label = np.log1p(samples_label.astype(np.float32))

samples_train = samples_train.astype(np.float32)

X_train, X_valid, y_train, y_valid = train_test_split(
    samples_train, samples_label, test_size=0.2, random_state=0
)




## === cell 9
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

params = {
    "max_depth": 12,  # modest increase in capacity
    "eta": 0.01,  # lower learning rate for finer fitting
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_bin": 128,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "seed": 0,
    "nthread": -1,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,  # allow more rounds; early stopping will cap it
    evals=[(dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=False,
)




## === cell 10
test_features = test.drop("key", axis=1).astype(np.float32)
dtest = xgb.DMatrix(test_features)

preds = np.expm1(bst.predict(dtest))

submission = pd.DataFrame(
    {
        "key": test["key"],
        "fare_amount": preds,
    },
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)
submission.head(20)
