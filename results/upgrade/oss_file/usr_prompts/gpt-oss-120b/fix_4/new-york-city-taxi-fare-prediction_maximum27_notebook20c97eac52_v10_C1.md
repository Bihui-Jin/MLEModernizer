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

# 5. Target score

3.79867

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.44848) has done: 'I fix the runtime errors that stop the pipeline and keep the core modeling unchanged.  
In **cell 3** I replace the unsupported `pl.sin`/`pl.cos` calls with a pandas‑based haversine calculation that returns a Polars DataFrame, so distance features are built correctly.  
In **cell 5** I adjust the LightGBM training call to use the current callback API (`early_stopping`) instead of the removed `early_stopping_rounds` argument.  
These changes let the code run end‑to‑end, produce a valid `submission_… .csv` file, and keep the original feature set and model architecture intact.'

# 9. Code solution

## === cell 0
import polars as pl

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
import numpy as np

num_rows = train_df.height

random_indices = np.random.choice(num_rows, size=10000000, replace=False)

train_df = train_df[random_indices]

train_df = train_df.drop_nulls()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3414656109.py in <cell line: 0>()
      1 import polars as pl
      2 
----> 3 train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
      4 test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
      5 import numpy as np

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
    112                 old_name, new_name, kwargs, function.__qualname__, version
    113             )
--> 114             return function(*args, **kwargs)
    115 
    116         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/io/csv/functions.py in read_csv(source, has_header, columns, new_columns, separator, comment_prefix, quote_char, skip_rows, skip_lines, schema, schema_overrides, null_values, missing_utf8_is_empty_string, ignore_errors, try_parse_dates, n_threads, infer_schema, infer_schema_length, batch_size, n_rows, encoding, low_memory, rechunk, use_pyarrow, storage_options, skip_rows_after_header, row_index_name, row_index_offset, sample_size, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, glob)
    537             storage_options=storage_options,
    538         ) as data:
--> 539             df = _read_csv_impl(
    540                 data,
    541                 has_header=has_header,

/usr/local/lib/python3.11/dist-packages/polars/io/csv/functions.py in _read_csv_impl(source, has_header, columns, separator, comment_prefix, quote_char, skip_rows, skip_lines, schema, schema_overrides, null_values, missing_utf8_is_empty_string, ignore_errors, try_parse_dates, n_threads, infer_schema_length, batch_size, n_rows, encoding, low_memory, rechunk, skip_rows_after_header, row_index_name, row_index_offset, sample_size, eol_char, raise_if_empty, truncate_ragged_lines, decimal_comma, glob)
    685     projection, columns = parse_columns_arg(columns)
    686 
--> 687     pydf = PyDataFrame.read_csv(
    688         source,
    689         infer_schema_length,

FileNotFoundError: No such file or directory (os error 2): /kaggle/input/new-york-city-taxi-fare-prediction/train.csv

## === cell 1
train = train_df
test = test_df




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2822050703.py in <cell line: 0>()
----> 1 train = train_df
      2 test = test_df
      3 
      4 

NameError: name 'train_df' is not defined

## === cell 2
import numpy as np
import polars as pl


def preprocess(df):
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S %Z", strict=False)
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
    """
    Compute haversine distance using numpy on a temporary pandas DataFrame,
    then return a Polars DataFrame so the rest of the pipeline (which expects Polars)
    works unchanged.
    """
    pdf = df.to_pandas()
    lat1 = np.radians(pdf["pickup_latitude"].astype(float))
    lat2 = np.radians(pdf["dropoff_latitude"].astype(float))
    dlat = lat2 - lat1
    dlon = np.radians(
        pdf["dropoff_longitude"].astype(float) - pdf["pickup_longitude"].astype(float)
    )

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    pdf["haversine_distance"] = 6371.0 * c
    return pl.from_pandas(pdf)


def drop_encoding(df):
    drop_columns = ["key", "pickup_datetime"]
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
    return df.filter(
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
    df = df.drop(["central_pickup_abs_longitude", "central_pickup_abs_latitude"])

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
    df = df.drop(["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"])

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
train = haversine(train)  # added more accurate distance
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = drop_outliner(train)

test = preprocess(test)
test = distance(test)
test = haversine(test)  # keep same feature for test
test_key = test["key"]
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/615186866.py in <cell line: 0>()
    228 
    229 
--> 230 train = preprocess(train)
    231 train = distance(train)
    232 train = haversine(train)  # added more accurate distance

NameError: name 'train' is not defined

## === cell 3
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/889591194.py in <cell line: 0>()
      1 import polars as pld
      2 
----> 3 dtypes = train.dtypes
      4 float64_columns = [
      5     col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64

NameError: name 'train' is not defined

## === cell 4
import warnings

warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = np.log1p(train["fare_amount"].to_pandas())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

y_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(np.expm1(y_val), y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm_log"




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1902866750.py in <cell line: 0>()
      8 from sklearn.metrics import mean_squared_error
      9 
---> 10 X = train.drop(["fare_amount"]).to_pandas()
     11 # Log‑transform the target to stabilise variance
     12 y = np.log1p(train["fare_amount"].to_pandas())

NameError: name 'train' is not defined

## === cell 5
sub_pred_log = bst.predict(test.to_pandas(), num_iteration=bst.best_iteration)
sub_pred = np.expm1(sub_pred_log)

exp_num = "distance_haversine"
submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2475307662.py in <cell line: 0>()
      1 # Predict on test set and revert the log transformation
----> 2 sub_pred_log = bst.predict(test.to_pandas(), num_iteration=bst.best_iteration)
      3 sub_pred = np.expm1(sub_pred_log)
      4 
      5 exp_num = "distance_haversine"

NameError: name 'bst' is not defined

## === cell 6
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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2530967980.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame()
----> 4 feature_importance_df["feature"] = X.columns
      5 feature_importance_df["importance"] = bst.feature_importance()
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'X' is not defined
