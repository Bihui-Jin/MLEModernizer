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

3.83762

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.66886) has done: 'The fix updates the haversine distance calculation to use Polars’ `arctan2` function, removes the unsupported `verbose_eval` argument from `lgb.train`, and switches LightGBM to CPU to avoid GPU‑related failures. These changes allow the pipeline to run end‑to‑end, generate predictions, and write a proper `submission_*.csv` file while keeping the original modeling logic intact.'

# 9. Code solution

## === cell 0
import polars as pl
import numpy as np

train_lazy = pl.scan_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_lazy = pl.scan_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")


def parse_datetime(df):
    return df.with_columns(
        pl.col("pickup_datetime")
        .str.replace(" UTC", "")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        .alias("pickup_datetime")
    )


def add_time_features(df):
    return df.with_columns(
        [
            pl.col("pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime").dt.day().alias("pickup_day"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.minute().alias("pickup_minute"),
            pl.col("pickup_datetime").dt.second().alias("pickup_second"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_weekday"),
        ]
    )


def add_abs_diff(df):
    return df.with_columns(
        [
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
            .abs()
            .alias("abs_longitude"),
            (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
            .abs()
            .alias("abs_latitude"),
        ]
    )


def haversine_distance(df):
    rad = np.pi / 180.0
    dlat = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * rad
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * rad
    lat1 = pl.col("pickup_latitude") * rad
    lat2 = pl.col("dropoff_latitude") * rad

    a = ((dlat / 2).sin()) ** 2 + (lat1.cos() * lat2.cos() * ((dlon / 2).sin()) ** 2)
    c = 2 * pl.arctan2(a.sqrt(), (1 - a).sqrt())
    return df.with_columns((6371 * c).alias("distance"))  # Earth radius ≈ 6371 km


def drop_unneeded_columns(df):
    return df.drop(["pickup_datetime", "pickup_minute", "pickup_second"])


def add_cyclic_features(df):
    return df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )


def one_hot_encode(df):
    return df.to_dummies(columns=["pickup_year", "pickup_day"])


def add_central_flags(df):
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    return df.with_columns(
        [
            (
                (pl.col("pickup_latitude") >= left_latitude)
                & (pl.col("pickup_latitude") <= right_latitude)
                & (pl.col("pickup_longitude") >= left_longitude)
                & (pl.col("pickup_longitude") <= right_longitude)
            )
            .cast(pl.Int8)
            .alias("is_pickup_central"),
            (
                (pl.col("dropoff_latitude") >= left_latitude)
                & (pl.col("dropoff_latitude") <= right_latitude)
                & (pl.col("dropoff_longitude") >= left_longitude)
                & (pl.col("dropoff_longitude") <= right_longitude)
            )
            .cast(pl.Int8)
            .alias("is_dropoff_central"),
        ]
    )


def add_short_distance_flag(df):
    return df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


def add_interaction(df):
    return df.with_columns(
        (pl.col("distance") * pl.col("passenger_count")).alias("distance_passenger")
    )


def filter_invalid_rows(df):
    return (
        df.filter((pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41))
        .filter(
            (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
        )
        .filter(
            (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
        )
        .filter((pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41))
        .filter((pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6))
        .filter(pl.col("fare_amount") < 10000)
    )


train = (
    train_lazy.drop_nulls()
    .pipe(filter_invalid_rows)
    .sample(fraction=0.5, seed=0)  # early sampling to cut work
    .pipe(parse_datetime)
    .pipe(add_time_features)
    .pipe(add_abs_diff)
    .pipe(haversine_distance)
    .pipe(drop_unneeded_columns)
    .pipe(add_cyclic_features)
    .pipe(add_central_flags)
    .pipe(add_short_distance_flag)
    .pipe(add_interaction)
    .collect()
)

test = (
    test_lazy.drop_nulls()
    .pipe(parse_datetime)
    .pipe(add_time_features)
    .pipe(add_abs_diff)
    .pipe(haversine_distance)
    .pipe(drop_unneeded_columns)
    .pipe(add_cyclic_features)
    .pipe(add_central_flags)
    .pipe(add_short_distance_flag)
    .pipe(add_interaction)
    .collect()
)

train = one_hot_encode(train)
test = one_hot_encode(test)

test_key = test["key"]




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1426752693.py in <cell line: 0>()
    131     train_lazy.drop_nulls()
    132     .pipe(filter_invalid_rows)
--> 133     .sample(fraction=0.5, seed=0)  # early sampling to cut work
    134     .pipe(parse_datetime)
    135     .pipe(add_time_features)

AttributeError: 'LazyFrame' object has no attribute 'sample'

## === cell 1
import warnings

warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

float64_cols_train = [
    col for col, dtype in zip(train.columns, train.dtypes) if dtype == pl.Float64
]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [
    col for col, dtype in zip(test.columns, test.dtypes) if dtype == pl.Float64
]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])

X = train.drop(["fare_amount", "key"]).to_pandas()
y = train["fare_amount"].to_pandas()

y_log = np.log1p(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 511,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)

val_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_val)
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"rmse:{rmse}")

model_name = "lgbm"




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2215098551.py in <cell line: 0>()
      9 
     10 float64_cols_train = [
---> 11     col for col, dtype in zip(train.columns, train.dtypes) if dtype == pl.Float64
     12 ]
     13 train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

NameError: name 'train' is not defined

## === cell 2
test_pd = test.drop("key").to_pandas()
sub_pred_log = bst.predict(test_pd, num_iteration=bst.best_iteration)
sub_pred = np.expm1(sub_pred_log)

exp_num = "01"
submission = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/51657489.py in <cell line: 0>()
----> 1 test_pd = test.drop("key").to_pandas()
      2 sub_pred_log = bst.predict(test_pd, num_iteration=bst.best_iteration)
      3 sub_pred = np.expm1(sub_pred_log)
      4 
      5 exp_num = "01"

NameError: name 'test' is not defined

## === cell 3
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X.columns
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

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2783425903.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame()
----> 4 feature_importance_df["feature"] = X.columns
      5 feature_importance_df["importance"] = bst.feature_importance()
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'X' is not defined
