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

3.31631

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.81501) has done: 'We fix the runtime errors by (1) simplifying the haversine function to a no‑op (the original call caused an AttributeError), (2) dropping the call to that function in the preprocessing pipeline, (3) removing the unsupported GPU device setting from LightGBM parameters, and (4) ensuring the test key variable is correctly defined before column drops. These minimal changes let the script run end‑to‑end and generate a proper `submission_lgbm.csv` while preserving the original modeling logic.'
- What this solution (achieved 5.34334) has done: 'I fixed the Polars `arcsin` typo by using the expression `.sqrt().arcsin()` inside the haversine calculation, added a call to the outlier‑filtering helper (`drop_outliner`) and a simple positive‑fare filter to improve data quality, and ensured the test‑key column is saved before it gets dropped. These changes let the script run end‑to‑end, produce a proper `submission_lgbm.csv`, and modestly improve the validation RMSE toward the target.'
- What this solution (achieved 5.13169) has done: 'The changes add a log‑1p transformation of the target variable before training LightGBM and convert the predictions back with expm1, which usually lowers RMSE for skewed fare amounts. The same preprocessing pipeline is kept, and only the modeling part is adjusted, preserving the original architecture while moving the validation score closer to the target.'
- What this solution (achieved 4.89085) has done: 'I increase LightGBM capacity slightly (raise `num_leaves` and add a modest `min_data_in_leaf`) and allow a longer early‑stopping patience so the model can fit the data a bit better, which is expected to lower the validation RMSE and move the score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 5.22329) has done: 'I add a modest fare ceiling (≤ 300 USD) after the existing outlier filter to remove extreme values that hurt RMSE, and I slightly enlarge the LightGBM capacity (more leaves) while reducing the min‑data‑in‑leaf to let the model fit the data a bit better. These small, targeted changes keep the original pipeline intact but are expected to lower the validation RMSE toward the target.'

# 9. Code solution

## === cell 0
import polars as pl
import numpy as np

train = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

num_rows = train.height
random_indices = np.random.choice(num_rows, size=15_000_000, replace=False)
train = train[random_indices]
train = train.drop_nulls()




## === cell 1
import polars as pl
import numpy as np


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
    """
    Compute haversine distance (km) and its log‑transform.
    """
    rad = pl.lit(np.pi / 180.0)
    lat1 = pl.col("pickup_latitude") * rad
    lat2 = pl.col("dropoff_latitude") * rad
    lon1 = pl.col("pickup_longitude") * rad
    lon2 = pl.col("dropoff_longitude") * rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = ((dlat / 2).sin().pow(2)) + (
        lat1.cos() * lat2.cos() * ((dlon / 2).sin().pow(2))
    )
    c = 2 * a.sqrt().arcsin()
    distance_km = (6371 * c).alias("distance")  # Earth radius ≈ 6371 km
    log_distance = pl.col("distance").log1p().alias("log_distance")
    return df.with_columns([distance_km, log_distance])


def haversine_distance(df):
    """
    Placeholder kept for compatibility; actual distance is computed in `distance`.
    """
    return df


def drop_encoding(df):
    drop_columns = ["key", "pickup_datetime", "pickup_second"]
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
        (pl.col("pickup_longitude") >= 70) & (pl.col("pickup_longitude") <= 90)
    )
    return df


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




## === cell 2
float64_columns = [
    col for col, dt in zip(train.columns, train.dtypes) if dt == pl.Float64
]
train = train.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])

float64_columns_test = [
    col for col, dt in zip(test.columns, test.dtypes) if dt == pl.Float64
]
test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns_test])




## === cell 3
train = preprocess(train)
train = distance(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_central(train)
train = is_short_distance(train)

train = drop_outliner(train)
train = train.filter(pl.col("fare_amount") > 0)
train = train.filter(pl.col("fare_amount") <= 300)  # keep reasonable ceiling

test = preprocess(test)
test = distance(test)
test_key = test["key"]  # preserve IDs before dropping
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_central(test)
test = is_short_distance(test)

train = train.fill_null(strategy="mean")
test = test.fill_null(strategy="mean")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ColumnNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/2576222527.py in <cell line: 0>()
      1 train = preprocess(train)
----> 2 train = distance(train)
      3 train = drop_encoding(train)
      4 train = cycling_encoding(train)
      5 train = one_hot_encoding(train)

/tmp/ipykernel_55/2438580456.py in distance(df)
     54     distance_km = (6371 * c).alias("distance")  # Earth radius ≈ 6371 km
     55     log_distance = pl.col("distance").log1p().alias("log_distance")
---> 56     return df.with_columns([distance_km, log_distance])
     57 
     58 

/usr/local/lib/python3.11/dist-packages/polars/dataframe/frame.py in with_columns(self, *exprs, **named_exprs)
   9803         └─────┴──────┴─────────────┘
   9804         """
-> 9805         return self.lazy().with_columns(*exprs, **named_exprs).collect(_eager=True)
   9806 
   9807     def with_columns_seq(

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

ColumnNotFoundError: distance

Resolved plan until failure:

	---> FAILED HERE RESOLVING 'sink' <---
DF ["key", "fare_amount", "pickup_datetime", "pickup_longitude", ...]; PROJECT */17 COLUMNS

## === cell 4
import warnings

warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()

y_log = np.log1p(y)

numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 3071,  # a bit more capacity
    "min_data_in_leaf": 5,  # less regularisation
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}
bst = lgb.train(
    params,
    train_data,
    num_boost_round=4000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

y_pred_val_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred_val = np.expm1(y_pred_val_log)

rmse = np.sqrt(mean_squared_error(np.expm1(y_val), y_pred_val))
print(f"rmse:{rmse}")
model_name = "lgbm"




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3900448932.py in <cell line: 0>()
     47 y_pred_val = np.expm1(y_pred_val_log)
     48 
---> 49 rmse = np.sqrt(mean_squared_error(np.expm1(y_val), y_pred_val))
     50 print(f"rmse:{rmse}")
     51 model_name = "lgbm"

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_regression.py in mean_squared_error(y_true, y_pred, sample_weight, multioutput, squared)
    440     0.825...
    441     """
--> 442     y_type, y_true, y_pred, multioutput = _check_reg_targets(
    443         y_true, y_pred, multioutput
    444     )

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_regression.py in _check_reg_targets(y_true, y_pred, multioutput, dtype)
     99     """
    100     check_consistent_length(y_true, y_pred)
--> 101     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    102     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
    103 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input contains NaN.

## === cell 5
test_pd = test.to_pandas()
test_key_pd = test_key.to_pandas()
sub_pred_log = bst.predict(test_pd, num_iteration=bst.best_iteration)
sub_pred = np.expm1(sub_pred_log)

submission = pd.DataFrame({"key": test_key_pd, "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}.csv", index=False)
print("Submission saved as:", f"submission_{model_name}.csv")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/428810977.py in <cell line: 0>()
      1 test_pd = test.to_pandas()
----> 2 test_key_pd = test_key.to_pandas()
      3 sub_pred_log = bst.predict(test_pd, num_iteration=bst.best_iteration)
      4 sub_pred = np.expm1(sub_pred_log)
      5 

NameError: name 'test_key' is not defined

## === cell 6
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
plt.title("Top 15 Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
