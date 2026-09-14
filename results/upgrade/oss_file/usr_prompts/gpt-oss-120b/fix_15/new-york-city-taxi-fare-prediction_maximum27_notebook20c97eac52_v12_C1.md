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

cudf-polars-cu12==25.6.0
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
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import polars as pl
import numpy as np
import gc  # garbage collection to free memory promptly

np.random.seed(42)

train_columns = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_df = pl.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    columns=train_columns,
)

test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

train_df = train_df.drop_nulls().sample(
    n=20000000, seed=42, with_replacement=False
)  # fast native sampling with smaller size

gc.collect()  # release any temporary memory from reading the full CSV




## === cell 2
train = train_df
test = test_df




## === cell 3
import numpy as np
import polars as pl


def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC")
        .alias("pickup_datetime")
    )

    df = df.with_columns(
        [
            df["pickup_datetime"].dt.year().alias("pickup_year"),
            df["pickup_datetime"].dt.month().alias("pickup_month"),
            df["pickup_datetime"].dt.day().alias("pickup_day"),
            df["pickup_datetime"].dt.hour().alias("pickup_hour"),
            df["pickup_datetime"].dt.minute().alias("pickup_minute"),
            df["pickup_datetime"].dt.second().alias("pickup_second"),
            df["pickup_datetime"].dt.weekday().alias("pickup_weekday"),
        ]
    )
    return df


def _haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def add_distance(df):
    dist_arr = _haversine(
        df["pickup_longitude"].to_numpy(),
        df["pickup_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
    )
    df = df.with_columns(pl.Series("distance", dist_arr))
    return df


def drop_encoding(df):
    df = df.drop(["pickup_datetime", "pickup_minute", "pickup_second"])
    return df


def drop_outliner(df):
    filtered = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41)
    )
    filtered = filtered.filter(
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
    )
    filtered = filtered.filter(
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
    )
    filtered = filtered.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41)
    )
    filtered = filtered.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6)
    )
    filtered = filtered.filter((pl.col("fare_amount") < 10000))
    return filtered


def add_cyclical_features(df):
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )
    return df


def add_one_hot(df):
    df = df.to_dummies(["pickup_year", "pickup_day"])
    return df


def add_central_flags(df):
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    df = df.with_columns(
        (
            (df["pickup_latitude"] >= left_latitude)
            & (df["pickup_latitude"] <= right_latitude)
            & (df["pickup_longitude"] >= left_longitude)
            & (df["pickup_longitude"] <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (df["dropoff_latitude"] >= left_latitude)
            & (df["dropoff_latitude"] <= right_latitude)
            & (df["dropoff_longitude"] >= left_longitude)
            & (df["dropoff_longitude"] <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central")
    )
    return df


def add_short_distance_flag(df):
    df = df.with_columns(
        (df["distance"] < 0.32).cast(pl.Int8).alias("is_short_distance")
    )
    return df


def add_interaction(df):
    df = df.with_columns(
        (df["distance"] * df["passenger_count"]).alias("distance_passenger")
    )
    return df


train = preprocess(train)
train = drop_outliner(train)  # filter early – removes many rows before heavy ops
train = add_distance(train)  # compute distance only for retained rows
train = drop_encoding(train)
train = add_cyclical_features(train)
train = add_one_hot(train)
train = add_central_flags(train)
train = add_short_distance_flag(train)
train = add_interaction(train)

test = preprocess(test)
test = add_distance(test)  # test set is small; keep order
test_key = test["key"]  # store before drop
test = drop_encoding(test)
test = add_cyclical_features(test)
test = add_one_hot(test)
test = add_central_flags(test)
test = add_short_distance_flag(test)
test = add_interaction(test)

gc.collect()  # free intermediate objects before model training




## === cell 4
import warnings

warnings.simplefilter("ignore")
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_numpy().astype(np.float32)
y = train["fare_amount"].to_numpy().astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 2047,
    "min_data_in_leaf": 5,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"

gc.collect()  # release training arrays before prediction on test set




## === cell 5
test_np = test.to_numpy().astype(np.float32)
sub_pred = bst.predict(test_np)

test_key_pd = test_key.to_pandas()
submission = pd.DataFrame({"key": test_key_pd, "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_is_central.csv", index=False)




## === cell 6
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = [f"f{i}" for i in range(X.shape[1])]
feature_importance_df["importance"] = bst.feature_importance()
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()
