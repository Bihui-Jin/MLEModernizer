# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

5.6495

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.10773) has done: 'Your score is worse than the target (RMSE 4.28526 vs 3.2814), so we should make small, legitimate fixes that typically improve RMSE without changing the overall approach. The biggest issue is that train/test one-hot encoding is done independently, which silently creates mismatched feature columns and degrades predictions; we align columns so the model sees the same feature set at train and inference. We also fix the datetime parsing to handle the dataset’s actual timestamp format (it usually doesn’t include `" UTC"`), and add standard NYC fare cleaning (remove obvious coordinate/fare outliers) to reduce noise while keeping the same LightGBM regression pipeline. Finally, we ensure test predictions are made on a pandas DataFrame with the exact same columns/order as training.'
- What this solution (achieved 5.10258) has done: 'I fix the Polars Haversine distance calculation by replacing the unsupported `.arctan2()` expression with a numerically-stable `2*arcsin(sqrt(a))` form so feature generation runs. Then I ensure non-numeric columns (`key`, `pickup_datetime`) are not passed into LightGBM by keeping them dropped during feature building and by converting any remaining non-numeric pandas dtypes to numeric-safe types. Finally, I remove early stopping (it was not allowed by your constraints and also blocks training when callbacks fail) while keeping the same LightGBM training approach and parameters, and I guarantee a correctly-formatted `submission_*.csv` is written.'
- What this solution (achieved 5.22645) has done: 'The root cause is that `pl.read_csv(..., skip_rows=...)` is skipping the header row, so Polars treats the first data row as column names and `fare_amount` doesn’t exist. I fix this by using `skip_rows_after_header` and explicitly providing the expected schema so the column names/types are correct, which unblocks all downstream feature engineering and training. I also make the datetime parsing explicit (matching the dataset’s `%Y-%m-%d %H:%M:%S%.f` format) while still allowing non-strict parsing, and I keep the existing feature logic/model unchanged. Finally, I ensure the submission is always written as a valid `.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 5.05406) has done: 'Your current RMSE (5.22645) is worse than the target (3.2814), so we should make small, legitimate fixes that usually improve Taxi Fare RMSE without changing your model or feature set. The biggest quality issue is that you train on a random 10M-row contiguous chunk without ensuring representative coverage, and you also include some noisy/invalid rows because the filter is applied after sampling; we keep the same chunk size and LightGBM training, but sample multiple smaller chunks across the file and apply the same cleaning per-chunk before concatenation. We also keep your existing feature engineering unchanged, but ensure the train chunk better matches test by using the same coordinate bounds and by adding a simple “dropoff==pickup & fare>5” filter earlier (per chunk) to reduce noise before the model sees it. These changes should improve generalization and move RMSE down toward your target while preserving the core logic and producing the same submission format.'
- What this solution (achieved 5.56713) has done: 'Your current RMSE (5.05406) is worse than the target (3.2814), so we should make small, legitimate data-quality fixes that typically reduce error without changing your model/feature logic. The biggest win with minimal disruption is to tighten training-row cleaning to remove “physically impossible / mislabeled” trips (e.g., huge distances within NYC bounds, or unrealistically low/high fare given distance), because these outliers strongly hurt RMSE for tree models. I keep your same LightGBM setup, same feature engineering, and same multi-chunk sampling, but apply a couple of additional conservative filters after distance is computed (and only on train). This should improve generalization and move RMSE down toward your target while preserving evaluation semantics and producing the same submission format.'
- What this solution (achieved 5.6495) has done: 'We make two minimal, score-relevant fixes that keep your exact feature/model approach unchanged. First, your “random chunk by skip_rows_after_header” unintentionally samples highly correlated contiguous blocks and can overfit to narrow time/area regimes; we keep the same 10M total rows and 5 chunks, but spread chunks deterministically across the file to improve representativeness (typically lowers RMSE). Second, we add one conservative, standard NYC taxi cleaning filter to remove the most damaging label noise: drop rides with fare_amount < 2.5 (below the historical NYC minimum fare), which usually improves RMSE without altering modeling logic. Everything else (features, LightGBM params, training loop, submission schema) stays the same.'

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

train_schema = {
    "key": pl.Utf8,
    "fare_amount": pl.Float32,
    "pickup_datetime": pl.Utf8,
    "pickup_longitude": pl.Float32,
    "pickup_latitude": pl.Float32,
    "dropoff_longitude": pl.Float32,
    "dropoff_latitude": pl.Float32,
    "passenger_count": pl.Int16,
}
test_schema = {
    "key": pl.Utf8,
    "pickup_datetime": pl.Utf8,
    "pickup_longitude": pl.Float32,
    "pickup_latitude": pl.Float32,
    "dropoff_longitude": pl.Float32,
    "dropoff_latitude": pl.Float32,
    "passenger_count": pl.Int16,
}

test_df = pl.read_csv(test_path, schema_overrides=test_schema)

TOTAL_ROWS_EST = 55_423_856  # from dataset description
TOTAL_TRAIN_ROWS = 10_000_000
N_CHUNKS = 5
CHUNK_ROWS = TOTAL_TRAIN_ROWS // N_CHUNKS

max_skip = max(0, TOTAL_ROWS_EST - CHUNK_ROWS - 1)
if N_CHUNKS == 1:
    skip_rows_list = [0]
else:
    skip_rows_list = [int(i * max_skip / (N_CHUNKS - 1)) for i in range(N_CHUNKS)]

train_chunks = []
for skip_rows in skip_rows_list:
    chunk = pl.read_csv(
        train_path,
        schema_overrides=train_schema,
        skip_rows_after_header=skip_rows,
        n_rows=CHUNK_ROWS,
    ).drop_nulls()
    train_chunks.append(chunk)

train_df = pl.concat(train_chunks, how="vertical_relaxed")



## === cell 2
train = train_df
test = test_df



## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    dt = pl.col("pickup_datetime").str.strptime(
        pl.Datetime,
        format="%Y-%m-%d %H:%M:%S%.f",
        strict=False,
    )
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

    df = df.filter(pl.col("fare_amount") >= 2.5)

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


def drop_outliner_after_distance(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter((pl.col("distance") >= 0.0) & (pl.col("distance") <= 100.0))
    df = df.filter(~((pl.col("distance") > 30.0) & (pl.col("fare_amount") < 10.0)))
    df = df.filter(~((pl.col("distance") < 0.5) & (pl.col("fare_amount") > 100.0)))
    return df


train = drop_outliner(train)

train = preprocess(train)
train = distance(train)
train = drop_outliner_after_distance(train)
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
