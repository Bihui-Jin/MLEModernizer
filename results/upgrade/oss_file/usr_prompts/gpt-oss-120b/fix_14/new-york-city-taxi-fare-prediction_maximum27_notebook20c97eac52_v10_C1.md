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
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import polars as pl
import numpy as np

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_df = pl.read_csv(train_path)
test_df = pl.read_csv(test_path)

num_rows = train_df.height
sample_size = min(10_000_000, num_rows)  # larger sample improves learning
rng = np.random.default_rng(seed=42)
random_indices = rng.choice(num_rows, size=sample_size, replace=False)
train_df = train_df[random_indices]

train_df = train_df.drop_nulls()




## === cell 1
train = train_df
test = test_df




## === cell 2
import numpy as np
import polars as pl


def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
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


def haversine(df):
    """Vectorised haversine distance using NumPy on Polars Series."""
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    lon1 = np.radians(df["pickup_longitude"].to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    hav = 6371.0 * c
    return df.with_columns(pl.Series("haversine_distance", hav))


def drop_encoding(df):
    return df.drop(["key", "pickup_datetime"])


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
    return df.to_dummies(["pickup_year", "pickup_day"])


def drop_outliner(df):
    return df.filter(
        (pl.col("pickup_latitude") >= 40)
        & (pl.col("pickup_latitude") < 41)
        & (pl.col("pickup_longitude") >= -74)
        & (pl.col("pickup_longitude") < -73)
        & (pl.col("dropoff_latitude") >= 40)
        & (pl.col("dropoff_latitude") < 41)
        & (pl.col("dropoff_longitude") >= -74)
        & (pl.col("dropoff_longitude") < -73)
        & (pl.col("passenger_count") >= 1)
        & (pl.col("passenger_count") < 6)
        & (pl.col("fare_amount") < 10_000)
    )




## === cell 3
train = preprocess(train)
train = distance(train)
train = haversine(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = drop_outliner(train)

test = preprocess(test)
test = distance(test)
test = haversine(test)
test_key = test["key"]
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)




## === cell 4
float64_cols_train = [c for c, t in zip(train.columns, train.dtypes) if t == pl.Float64]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [
    c for c, t in zip(test.columns, test.dtypes) if t in (pl.Float32, pl.Float64)
]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])




## === cell 5
import warnings

warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = train.filter(pl.col("fare_amount") > 0)

feature_cols = [c for c in train.columns if c != "fare_amount"]

X = train.select(feature_cols).to_numpy()
y = np.log1p(train["fare_amount"].to_numpy())

mask = np.isfinite(y)
X = X[mask]
y = y[mask]
X = np.nan_to_num(X, nan=0.0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 1023,  # slightly larger model capacity
    "max_depth": -1,
    "feature_fraction": 0.9,  # use more features per iteration
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "lambda_l2": 0.1,  # small L2 regularisation
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=10000,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

y_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred = np.expm1(y_pred_log)
rmse = np.sqrt(mean_squared_error(np.expm1(y_val), y_pred))
print(f"rmse:{rmse}")

model_name = "lgbm_log"




## === cell 6
missing_cols = set(feature_cols) - set(test.columns)
if missing_cols:
    zero_series = pl.Series(name=None, values=[0.0] * test.height, dtype=pl.Float32)
    for col in missing_cols:
        test = test.with_columns(zero_series.alias(col))

test_features_np = test.select(feature_cols).to_numpy()

sub_pred_log = bst.predict(test_features_np, num_iteration=bst.best_iteration)
sub_pred = np.expm1(sub_pred_log)

submission = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})
submission_path = f"submission_{model_name}_haversine.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame(
    {"feature": feature_cols, "importance": bst.feature_importance()}
)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.show()
