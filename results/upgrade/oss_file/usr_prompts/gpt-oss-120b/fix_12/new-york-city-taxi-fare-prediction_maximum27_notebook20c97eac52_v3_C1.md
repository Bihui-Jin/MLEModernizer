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

3.79127

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.43919) has done: 'The fix removes the target column `fare_amount` from the set of numeric features used for both training and prediction, preventing a `ColumnNotFoundError` when selecting columns from the test set. This change keeps the core modeling logic unchanged while ensuring a valid submission file is produced.'

# 9. Code solution

## === cell 0
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path


def resolve_path(relative_path: str) -> Path:
    """Return the first existing path among common locations."""
    candidates = [
        Path("data") / relative_path,
        Path("data") / "new-york-city-taxi-fare-prediction" / relative_path,
        Path(relative),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {relative_path} in any expected location.")


train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")

train = pl.read_csv(train_path, infer_schema_length=1000)
test = pl.read_csv(test_path, infer_schema_length=1000)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2452351630.py in <cell line: 0>()
     18 
     19 
---> 20 train_path = resolve_path("train.csv")
     21 test_path = resolve_path("test.csv")
     22 

/tmp/ipykernel_55/2452351630.py in resolve_path(relative_path)
     10         Path("data") / relative_path,
     11         Path("data") / "new-york-city-taxi-fare-prediction" / relative_path,
---> 12         Path(relative),
     13     ]
     14     for p in candidates:

NameError: name 'relative' is not defined

## === cell 1
def preprocess(df):
    dt_parsed = pl.col("pickup_datetime").str.strptime(
        pl.Datetime, fmt="%Y-%m-%d %H:%M:%S", strict=False
    )
    return df.with_columns(
        dt_parsed.dt.year().alias("pickup_year"),
        dt_parsed.dt.month().alias("pickup_month"),
        dt_parsed.dt.day().alias("pickup_day"),
        dt_parsed.dt.hour().alias("pickup_hour"),
        dt_parsed.dt.minute().alias("pickup_minute"),
        dt_parsed.dt.second().alias("pickup_second"),
        dt_parsed.dt.weekday().alias("pickup_weekday"),
        (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
        .abs()
        .alias("abs_longitude"),
        (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
        .abs()
        .alias("abs_latitude"),
    )


def distance(df):
    return df.with_columns(
        (
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude")) ** 2
            + (pl.col("pickup_latitude") - pl.col("dropoff_latitude")) ** 2
        )
        .sqrt()
        .alias("distance")
    )


def haversine_distance(df):
    R = pl.lit(6371.0)  # Earth radius in km
    deg_to_rad = pl.lit(np.pi / 180.0)

    lat1 = pl.col("pickup_latitude") * deg_to_rad
    lon1 = pl.col("pickup_longitude") * deg_to_rad
    lat2 = pl.col("dropoff_latitude") * deg_to_rad
    lon2 = pl.col("dropoff_longitude") * deg_to_rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (pl.sin(dlat / 2) ** 2) + pl.cos(lat1) * pl.cos(lat2) * (pl.sin(dlon / 2) ** 2)
    c = 2 * pl.arcsin(pl.sqrt(a))
    return df.with_columns((R * c).alias("haversine_distance"))


def drop_encoding(df):
    return df.drop(["key", "pickup_datetime", "pickup_minute", "pickup_second"])


def cycling_encoding(df):
    return df.with_columns(
        (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
        (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
        (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
        (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
    )


def one_hot_encoding(df):
    return df.to_dummies(["pickup_year", "pickup_day"])


def is_central(df):
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    return df.with_columns(
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
    )


def drop_outlier(df):
    filters = [
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41),
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73),
        (pl.col("dropoff_longitude") >= 74) & (pl.col("dropoff_longitude") < -73),
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41),
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6),
    ]
    if "fare_amount" in df.columns:
        filters.append(pl.col("fare_amount") < 10000)
    combined = filters[0]
    for f in filters[1:]:
        combined &= f
    return df.filter(combined)


def is_short_distance(df):
    return df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


train = preprocess(train)
train = distance(train)
train = haversine_distance(train)

test = preprocess(test)
test = distance(test)
test = haversine_distance(test)

test_key = test["key"].to_pandas()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2240700021.py in <cell line: 0>()
    113 
    114 # preprocessing for train and test
--> 115 train = preprocess(train)
    116 train = distance(train)
    117 train = haversine_distance(train)

NameError: name 'train' is not defined

## === cell 2
train = pl.LazyFrame(train)
test = pl.LazyFrame(test)

train = drop_encoding(train)
test = drop_encoding(test)

train = cycling_encoding(train)
test = cycling_encoding(test)

train = one_hot_encoding(train)
test = one_hot_encoding(test)

train = is_central(train)
test = is_central(test)

train = drop_outlier(train)

train = is_short_distance(train)
test = is_short_distance(test)

train = train.collect()
test = test.collect()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/148867064.py in <cell line: 0>()
      1 # Switch to lazy execution for memory‑efficiency
----> 2 train = pl.LazyFrame(train)
      3 test = pl.LazyFrame(test)
      4 
      5 # feature engineering steps

NameError: name 'train' is not defined

## === cell 3
float64_cols_train = [
    col for col, dtype in zip(train.columns, train.dtypes) if dtype == pl.Float64
]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [
    col for col, dtype in zip(test.columns, test.dtypes) if dtype == pl.Float64
]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/411782467.py in <cell line: 0>()
      1 # Reduce memory usage: down‑cast Float64 → Float32 where possible
      2 float64_cols_train = [
----> 3     col for col, dtype in zip(train.columns, train.dtypes) if dtype == pl.Float64
      4 ]
      5 train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

NameError: name 'train' is not defined

## === cell 4
import warnings

warnings.simplefilter("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

numeric_cols = train.select(
    pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Boolean)
).columns
numeric_cols = [c for c in numeric_cols if c != "fare_amount"]

X_np = train.select(numeric_cols).to_numpy()
y_original = train["fare_amount"].to_numpy()
y = np.log1p(y_original)  # log‑transform target

X_train, X_val, y_train, y_val, y_orig_train, y_orig_val = train_test_split(
    X_np, y, y_original, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
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
    num_boost_round=1500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

y_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_orig_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm_log_hav"




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3574086964.py in <cell line: 0>()
      7 
      8 # select numeric feature columns (exclude the target)
----> 9 numeric_cols = train.select(
     10     pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Boolean)
     11 ).columns

NameError: name 'train' is not defined

## === cell 5
test_features_np = test.select(numeric_cols).to_numpy()
sub_pred_log = bst.predict(test_features_np)
sub_pred = np.expm1(sub_pred_log)  # revert log transform

submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})
submission_path = f"submission_{model_name}.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4283149297.py in <cell line: 0>()
----> 1 test_features_np = test.select(numeric_cols).to_numpy()
      2 sub_pred_log = bst.predict(test_features_np)
      3 sub_pred = np.expm1(sub_pred_log)  # revert log transform
      4 
      5 submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})

NameError: name 'test' is not defined

## === cell 6
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame(
    {"feature": numeric_cols, "importance": bst.feature_importance()}
)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance")
plt.gca().invert_yaxis()
plt.show()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3098795016.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame(
----> 4     {"feature": numeric_cols, "importance": bst.feature_importance()}
      5 )
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'numeric_cols' is not defined
