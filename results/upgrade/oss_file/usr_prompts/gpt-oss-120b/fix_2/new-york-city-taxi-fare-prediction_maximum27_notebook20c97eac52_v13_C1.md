# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

3.76015

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import polars as pl

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
import numpy as np

num_rows = train_df.height

np.random.seed(42)
random_indices = np.random.choice(num_rows, size=12000000, replace=False)

train_df = train_df[random_indices]

train_df = train_df.drop_nulls()



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

    df = df.with_columns(
        [
            (df["pickup_longitude"] - df["dropoff_longitude"])
            .abs()
            .alias("abs_longitude"),
            (df["pickup_latitude"] - df["dropoff_latitude"])
            .abs()
            .alias("abs_latitude"),
        ]
    )

    return df


def distance(df):
    longitude = 85.393
    latitude = 111.034
    df = df.with_columns(
        [
            (df["abs_longitude"] * longitude + df["abs_latitude"] * latitude).alias(
                "distance"
            )
        ]
    )
    return df


def drop_encoding(df):
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    df = df.drop(drop_columns)
    return df


def cycling_encoding(df):
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )

    return df


def one_hot_encoding(df):
    one_hot_cols = ["pickup_year", "pickup_day"]
    df = df.to_dummies(one_hot_cols)
    return df


def max_min_scaling(df):
    df = df.filter(
        pl.col("pickup_longitude") >= 70 & (pl.col("pickup_longitude") <= 90)
    )


def is_central(df):
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


def diff_central(df):
    base_longitude = 85.393
    base_latitude = 111.034
    central_latitude = 40.764696
    central_longitude = -73.98065
    df = df.with_columns(
        [
            (df["pickup_longitude"] - central_longitude)
            .abs()
            .alias("central_pickup_abs_longitude"),
            (df["pickup_latitude"] - central_latitude)
            .abs()
            .alias("central_pickup_abs_latitude"),
        ]
    )
    df = df.with_columns(
        [
            (
                df["central_pickup_abs_longitude"] * base_longitude
                + df["central_pickup_abs_latitude"] * base_latitude
            ).alias("central_pickup_distance")
        ]
    )
    drop_columns = ["central_pickup_abs_longitude", "central_pickup_abs_latitude"]
    df = df.drop(drop_columns)

    df = df.with_columns(
        [
            (df["dropoff_longitude"] - central_longitude)
            .abs()
            .alias("central_dropoff_abs_longitude"),
            (df["dropoff_latitude"] - central_latitude)
            .abs()
            .alias("central_dropoff_abs_latitude"),
        ]
    )
    df = df.with_columns(
        [
            (
                df["central_dropoff_abs_longitude"] * base_longitude
                + df["central_dropoff_abs_latitude"] * base_latitude
            ).alias("central_dropoff_distance")
        ]
    )
    drop_columns = ["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"]
    df = df.drop(drop_columns)

    return df


def is_short_distance(df):
    df = df.with_columns(
        (df["distance"] < 0.32).cast(pl.Int8).alias("is_short_distance")
    )
    return df


def drop_outliner(df):
    filtered_train_df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6)
    )
    filtered_train_df = filtered_train_df.filter((pl.col("fare_amount") < 10000))

    return filtered_train_df


def rule_base(df):
    is_weekday = df["pickup_weekday"].is_in([0, 1, 2, 3, 4])
    is_weekend = df["pickup_weekday"].is_in([5, 6])

    df = df.with_columns(
        pl.when(is_weekday & df["pickup_hour"].is_in([16, 17, 18, 19, 20]))
        .then(9 * df["distance"] / 0.32)
        .when(
            is_weekday & df["pickup_hour"].is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(df["distance"] / 0.32)
        .when(is_weekday)
        .then(0.5 * df["distance"] / 0.32)
        .when(
            is_weekend & df["pickup_hour"].is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(60 * df["distance"] / 19.2)
        .otherwise(30 * df["distance"] / 19.2)
        .alias("calculated_value")
    )

    return df


train = preprocess(train)
train = distance(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = drop_outliner(train)
train = is_short_distance(train)

test = preprocess(test)
test = distance(test)
test_key = test["key"]
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)
test = is_short_distance(test)



## === cell 4
import polars as pld

dtypes = train.dtypes
float64_columns = [
    col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64
]
train = train.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])

dtypes = test.dtypes
float64_columns = [
    col for col, dtype in zip(test.columns, dtypes) if dtype == pl.Float64
]
test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])



## === cell 5
import warnings

warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()
test_pd = test.to_pandas()


def add_haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = lat2 - lat1
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine_distance"] = 6371.0 * c  # Earth radius in km
    return df


X = add_haversine(X)
test_pd = add_haversine(test_pd)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    early_stopping_rounds=30,
    verbose_eval=False,
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3856128444.py in <cell line: 0>()
     49 
     50 # Train with early stopping to avoid over‑fitting
---> 51 bst = lgb.train(
     52     params,
     53     train_data,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 6
sub_pred = bst.predict(test_pd, num_iteration=bst.best_iteration)

exp_num = "123467"
submission = pd.DataFrame({"key": test_key.to_numpy(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/103756977.py in <cell line: 0>()
      1 # Predict on the test set (using the same pandas dataframe)
----> 2 sub_pred = bst.predict(test_pd, num_iteration=bst.best_iteration)
      3 
      4 exp_num = "123467"
      5 submission = pd.DataFrame({"key": test_key.to_numpy(), "fare_amount": sub_pred})

NameError: name 'bst' is not defined

## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X.columns
feature_importance_df["importance"] = bst.feature_importance()
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
)
feature_importance_df = feature_importance_df.head(15)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2530967980.py in <cell line: 0>()
      3 feature_importance_df = pd.DataFrame()
      4 feature_importance_df["feature"] = X.columns
----> 5 feature_importance_df["importance"] = bst.feature_importance()
      6 feature_importance_df = feature_importance_df.sort_values(
      7     by="importance", ascending=False

NameError: name 'bst' is not defined
