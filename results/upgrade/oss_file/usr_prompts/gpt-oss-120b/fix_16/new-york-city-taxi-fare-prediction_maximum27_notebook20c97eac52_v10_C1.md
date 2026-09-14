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

# 8. Previous improvement plans

- What this solution (achieved 5.44848) has done: 'I fix the runtime errors that stop the pipeline and keep the core modeling unchanged.  
In **cell 3** I replace the unsupported `pl.sin`/`pl.cos` calls with a pandas‑based haversine calculation that returns a Polars DataFrame, so distance features are built correctly.  
In **cell 5** I adjust the LightGBM training call to use the current callback API (`early_stopping`) instead of the removed `early_stopping_rounds` argument.  
These changes let the code run end‑to‑end, produce a valid `submission_… .csv` file, and keep the original feature set and model architecture intact.'
- What this solution (achieved 5.20121) has done: 'I fixed the pipeline so it can run end‑to‑end and output a proper CSV submission.  
- In **cell 4** I relaxed the mask, removed the check that caused an empty training set, and filled missing numeric values with zeros.  
- In **cell 5** I aligned the test feature matrix to the training columns before prediction, guaranteeing the model sees the same feature set.  
These minimal changes keep the original modeling logic intact while ensuring a valid submission file is written.'
- What this solution (achieved 5.06712) has done: 'I increase the LightGBM boosting iterations (from 500 to 2000) and allow a slightly longer early‑stopping patience (30 → 50). This keeps the modeling pipeline unchanged while giving the model more capacity to fit the data, which should lower the RMSE and move the score closer to the target 3.79867.'
- What this solution (achieved 5.11495) has done: 'I correct the file paths, ensure the data is loaded before it is used, keep the preprocessing pipeline unchanged, and make a small hyper‑parameter tweak to help the model reach the target RMSE while still producing a valid `submission_… .csv` file.'

# 9. Code solution

## === cell 0
import os
import polars as pl
import numpy as np

train_path = "/kaggle/input/new-york-city-taxi-prediction/labels.csv"
test_path = "/kaggle/input/new-york-city-taxi-prediction/test.csv"

sample_size = 5_000_000
train_df = pl.read_csv(train_path, n_rows=sample_size)
test_df = pl.read_csv(test_path)

train_df = train_df.drop_nulls()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/3224996118.py in <cell line: 0>()
      8 # Reduce the in‑memory sample to speed up training while preserving the original pipeline.
      9 sample_size = 5_000_000
---> 10 train_df = pl.read_csv(train_path, n_rows=sample_size)
     11 test_df = pl.read_csv(test_path)
     12 

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

FileNotFoundError: No such file or directory (os error 2): /kaggle/input/new-york-city-taxi-prediction/labels.csv

## === cell 1
train = train_df
test = test_df




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2822050703.py in <cell line: 0>()
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/584532499.py in <cell line: 0>()
----> 1 train = preprocess(train)
      2 train = distance(train)
      3 train = haversine(train)
      4 train = drop_encoding(train)
      5 train = cycling_encoding(train)

NameError: name 'train' is not defined

## === cell 4
float64_cols_train = [c for c, t in zip(train.columns, train.dtypes) if t == pl.Float64]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [
    c for c, t in zip(test.columns, test.dtypes) if t in (pl.Float32, pl.Float64)
]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3558776134.py in <cell line: 0>()
----> 1 float64_cols_train = [c for c, t in zip(train.columns, train.dtypes) if t == pl.Float64]
      2 train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])
      3 
      4 float64_cols_test = [
      5     c for c, t in zip(test.columns, test.dtypes) if t in (pl.Float32, pl.Float64)

NameError: name 'train' is not defined

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
    "num_leaves": 1023,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "min_data_in_leaf": 10,
    "lambda_l2": 0.1,
    "max_bin": 255,  # faster histogram building without altering model capacity
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=3000,
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




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3821516487.py in <cell line: 0>()
      8 from sklearn.metrics import mean_squared_error
      9 
---> 10 train = train.filter(pl.col("fare_amount") > 0)
     11 
     12 feature_cols = [c for c in train.columns if c != "fare_amount"]

NameError: name 'train' is not defined

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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1072261334.py in <cell line: 0>()
----> 1 missing_cols = set(feature_cols) - set(test.columns)
      2 if missing_cols:
      3     zero_series = pl.Series(name=None, values=[0.0] * test.height, dtype=pl.Float32)
      4     for col in missing_cols:
      5         test = test.with_columns(zero_series.alias(col))

NameError: name 'feature_cols' is not defined

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

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2199342208.py in <cell line: 0>()
      2 
      3 feature_importance_df = pd.DataFrame(
----> 4     {"feature": feature_cols, "importance": bst.feature_importance()}
      5 )
      6 feature_importance_df = feature_importance_df.sort_values(

NameError: name 'feature_cols' is not defined
