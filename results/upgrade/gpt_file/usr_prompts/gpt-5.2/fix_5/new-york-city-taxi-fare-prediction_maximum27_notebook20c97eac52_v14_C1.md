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

3.2814

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.10773) has done: 'Your score is worse than the target (RMSE 4.28526 vs 3.2814), so we should make small, legitimate fixes that typically improve RMSE without changing the overall approach. The biggest issue is that train/test one-hot encoding is done independently, which silently creates mismatched feature columns and degrades predictions; we align columns so the model sees the same feature set at train and inference. We also fix the datetime parsing to handle the dataset’s actual timestamp format (it usually doesn’t include `" UTC"`), and add standard NYC fare cleaning (remove obvious coordinate/fare outliers) to reduce noise while keeping the same LightGBM regression pipeline. Finally, we ensure test predictions are made on a pandas DataFrame with the exact same columns/order as training.'
- What this solution (achieved 5.10258) has done: 'I fix the Polars Haversine distance calculation by replacing the unsupported `.arctan2()` expression with a numerically-stable `2*arcsin(sqrt(a))` form so feature generation runs. Then I ensure non-numeric columns (`key`, `pickup_datetime`) are not passed into LightGBM by keeping them dropped during feature building and by converting any remaining non-numeric pandas dtypes to numeric-safe types. Finally, I remove early stopping (it was not allowed by your constraints and also blocks training when callbacks fail) while keeping the same LightGBM training approach and parameters, and I guarantee a correctly-formatted `submission_*.csv` is written.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import polars as pl

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

test_df = pl.read_csv(test_path)

CHUNK_ROWS = 10_000_000
TOTAL_ROWS_EST = 55_423_856  # from dataset description
rng = np.random.default_rng(0)
max_skip = max(0, TOTAL_ROWS_EST - CHUNK_ROWS - 1)
skip_rows = int(rng.integers(0, max_skip + 1))

train_df = pl.read_csv(
    train_path,
    skip_rows=skip_rows,
    n_rows=CHUNK_ROWS,
)

train_df = train_df.drop_nulls()



## === cell 2
train = train_df
test = test_df



## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    dt = pl.col("pickup_datetime").str.strptime(pl.Datetime, strict=False)
    df = df.with_columns(dt.alias("pickup_datetime"))

    df = df.with_columns(
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

    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
            .abs()
            .alias("abs_longitude"),
            (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
            .abs()
            .alias("abs_latitude"),
        ]
    )
    return df


def distance(df: pl.DataFrame) -> pl.DataFrame:
    r_earth_km = 6371.0
    to_rad = np.pi / 180.0

    lat1 = pl.col("pickup_latitude") * to_rad
    lon1 = pl.col("pickup_longitude") * to_rad
    lat2 = pl.col("dropoff_latitude") * to_rad
    lon2 = pl.col("dropoff_longitude") * to_rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (dlat / 2).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2)
    a = a.clip(0.0, 1.0)
    c = (a.sqrt()).arcsin() * 2.0
    return df.with_columns((pl.lit(r_earth_km) * c).alias("distance"))


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    cols_to_drop = [c for c in drop_columns if c in df.columns]
    return df.drop(cols_to_drop)


def cycling_encoding(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )
    return df


def one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    one_hot_cols = ["pickup_year", "pickup_day"]
    one_hot_cols = [c for c in one_hot_cols if c in df.columns]
    return df.to_dummies(one_hot_cols)


def is_central(df: pl.DataFrame) -> pl.DataFrame:
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521
    df = df.with_columns(
        (
            (pl.col("pickup_latitude") >= left_latitude)
            & (pl.col("pickup_latitude") <= right_latitude)
            & (pl.col("pickup_longitude") >= left_longitude)
            & (pl.col("pickup_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (pl.col("dropoff_latitude") >= left_latitude)
            & (pl.col("dropoff_latitude") <= right_latitude)
            & (pl.col("dropoff_longitude") >= left_longitude)
            & (pl.col("dropoff_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central")
    )
    return df


def is_short_distance(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


def rule_base(df: pl.DataFrame) -> pl.DataFrame:
    is_weekday = pl.col("pickup_weekday").is_in([0, 1, 2, 3, 4])
    is_weekend = pl.col("pickup_weekday").is_in([5, 6])

    df = df.with_columns(
        pl.when(is_weekday & pl.col("pickup_hour").is_in([16, 17, 18, 19, 20]))
        .then(9 * pl.col("distance") / 0.32)
        .when(
            is_weekday
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(pl.col("distance") / 0.32)
        .when(is_weekday)
        .then(0.5 * pl.col("distance") / 0.32)
        .when(
            is_weekend
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(60 * pl.col("distance") / 19.2)
        .otherwise(30 * pl.col("distance") / 19.2)
        .alias("calculated_value")
    )
    return df


def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("fare_amount").is_not_null())
    df = df.filter((pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 250))
    df = df.filter((pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6))

    df = df.filter(
        (pl.col("pickup_latitude") >= 40.0) & (pl.col("pickup_latitude") <= 42.0)
    )
    df = df.filter(
        (pl.col("dropoff_latitude") >= 40.0) & (pl.col("dropoff_latitude") <= 42.0)
    )
    df = df.filter(
        (pl.col("pickup_longitude") >= -75.0) & (pl.col("pickup_longitude") <= -72.0)
    )
    df = df.filter(
        (pl.col("dropoff_longitude") >= -75.0) & (pl.col("dropoff_longitude") <= -72.0)
    )

    df = df.with_columns(
        [
            (
                (pl.col("pickup_longitude") - pl.col("dropoff_longitude")).abs()
                + (pl.col("pickup_latitude") - pl.col("dropoff_latitude")).abs()
            ).alias("_coord_delta")
        ]
    )
    df = df.filter(~((pl.col("_coord_delta") < 1e-6) & (pl.col("fare_amount") > 5)))
    df = df.drop(["_coord_delta"])
    return df


def add_distance_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("distance") * pl.lit(0.621371)).alias("distance_miles"),
            (pl.col("abs_longitude") + pl.col("abs_latitude")).alias("manhattan_dist"),
        ]
    )


train = drop_outliner(train)

train = preprocess(train)
train = distance(train)
train = add_distance_features(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = is_short_distance(train)
train = rule_base(train)

test = preprocess(test)
test = distance(test)
test = add_distance_features(test)
test_key = test["key"]
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)
test = is_short_distance(test)
test = rule_base(test)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ColumnNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3426094151.py in <cell line: 0>()
    175 
    176 
--> 177 train = drop_outliner(train)
    178 
    179 train = preprocess(train)

/tmp/ipykernel_11/3426094151.py in drop_outliner(df)
    134 
    135 def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
--> 136     df = df.filter(pl.col("fare_amount").is_not_null())
    137     df = df.filter((pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 250))
    138     df = df.filter((pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6))

/usr/local/lib/python3.11/dist-packages/polars/dataframe/frame.py in filter(self, *predicates, **constraints)
   5099         └──────┴──────┴─────┘
   5100         """
-> 5101         return self.lazy().filter(*predicates, **constraints).collect(_eager=True)
   5102 
   5103     def remove(

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ColumnNotFoundError: unable to find column "fare_amount"; valid columns: ["2013-04-26 10:34:00.00000025", "57.33", "2013-04-26 10:34:00 UTC", "-73.780602", "40.645277", "-73.949007", "40.782592", "1"]

Resolved plan until failure:

	---> FAILED HERE RESOLVING 'sink' <---
DF ["2013-04-26 10:34:00.00000025", "57.33", "2013-04-26 10:34:00 UTC", "-73.780602", ...]; PROJECT */8 COLUMNS

## === cell 4
train_cols = set(train.columns)
test_cols = set(test.columns)

missing_in_test = sorted(list(train_cols - test_cols))
missing_in_train = sorted(list(test_cols - train_cols))

if missing_in_test:
    test = test.with_columns(
        [pl.lit(0).cast(pl.Int8).alias(c) for c in missing_in_test]
    )

if missing_in_train:
    train = train.with_columns(
        [pl.lit(0).cast(pl.Int8).alias(c) for c in missing_in_train]
    )

feature_cols = [c for c in train.columns if c != "fare_amount"]
test = test.select(feature_cols)
train = train.select(feature_cols + ["fare_amount"])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ColumnNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/963429327.py in <cell line: 0>()
     17 feature_cols = [c for c in train.columns if c != "fare_amount"]
     18 test = test.select(feature_cols)
---> 19 train = train.select(feature_cols + ["fare_amount"])
     20 

/usr/local/lib/python3.11/dist-packages/polars/dataframe/frame.py in select(self, *exprs, **named_exprs)
   9630         └──────────────┘
   9631         """
-> 9632         return self.lazy().select(*exprs, **named_exprs).collect(_eager=True)
   9633 
   9634     def select_seq(

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ColumnNotFoundError: fare_amount

Resolved plan until failure:

	---> FAILED HERE RESOLVING 'sink' <---
DF ["2013-04-26 10:34:00.00000025", "57.33", "2013-04-26 10:34:00 UTC", "-73.780602", ...]; PROJECT */15 COLUMNS

## === cell 5
import polars as pl

dtypes = train.dtypes
float64_columns = [
    col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64
]
if float64_columns:
    train = train.with_columns(
        [pl.col(col).cast(pl.Float32) for col in float64_columns]
    )

dtypes = test.dtypes
float64_columns = [
    col for col, dtype in zip(test.columns, dtypes) if dtype == pl.Float64
]
if float64_columns:
    test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])



## === cell 6
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()

for c in X.columns:
    if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_datetime64_any_dtype(X[c]):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "gpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 0,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[val_data],
)

y_pred = bst.predict(X_val, num_iteration=200)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ColumnNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/939743749.py in <cell line: 0>()
      9 from sklearn.metrics import mean_squared_error
     10 
---> 11 X = train.drop(["fare_amount"]).to_pandas()
     12 y = train["fare_amount"].to_pandas()
     13 

/usr/local/lib/python3.11/dist-packages/polars/dataframe/frame.py in drop(self, strict, *columns)
   8195         └─────┘
   8196         """
-> 8197         return self.lazy().drop(*columns, strict=strict).collect(_eager=True)
   8198 
   8199     def drop_in_place(self, name: str) -> Series:

/usr/local/lib/python3.11/dist-packages/polars/_utils/deprecation.py in wrapper(*args, **kwargs)
     86                 kwargs["engine"] = "old-streaming"
     87 
---> 88             return function(*args, **kwargs)
     89 
     90         wrapper.__signature__ = inspect.signature(function)  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/polars/lazyframe/frame.py in collect(self, type_coercion, _type_check, predicate_pushdown, projection_pushdown, simplify_expression, slice_pushdown, comm_subplan_elim, comm_subexpr_elim, cluster_with_columns, collapse_joins, no_optimization, engine, background, _check_order, _eager, **_kwargs)
   2186         # Only for testing purposes
   2187         callback = _kwargs.get("post_opt_callback", callback)
-> 2188         return wrap_df(ldf.collect(engine, callback))
   2189 
   2190     @overload

ColumnNotFoundError: "fare_amount" not found

Resolved plan until failure:

	---> FAILED HERE RESOLVING 'sink' <---
DF ["2013-04-26 10:34:00.00000025", "57.33", "2013-04-26 10:34:00 UTC", "-73.780602", ...]; PROJECT */15 COLUMNS

## === cell 7
test_X = test.to_pandas()

test_X = test_X.reindex(columns=X.columns, fill_value=0)

for c in test_X.columns:
    if pd.api.types.is_object_dtype(test_X[c]) or pd.api.types.is_datetime64_any_dtype(
        test_X[c]
    ):
        test_X[c] = pd.to_numeric(test_X[c], errors="coerce")
test_X = test_X.fillna(0)

sub_pred = bst.predict(test_X, num_iteration=200)

sub_pred = np.clip(sub_pred, 0, None)

exp_num = "123467"
submission = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})
submission_path = f"submission_{model_name}_{exp_num}.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/613381744.py in <cell line: 0>()
      1 test_X = test.to_pandas()
      2 
----> 3 test_X = test_X.reindex(columns=X.columns, fill_value=0)
      4 
      5 for c in test_X.columns:

NameError: name 'X' is not defined

## === cell 8
import matplotlib.pyplot as plt
import pandas as pd

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

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3386683095.py in <cell line: 0>()
      3 
      4 feature_importance_df = pd.DataFrame()
----> 5 feature_importance_df["feature"] = X.columns
      6 feature_importance_df["importance"] = bst.feature_importance()
      7 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'X' is not defined
