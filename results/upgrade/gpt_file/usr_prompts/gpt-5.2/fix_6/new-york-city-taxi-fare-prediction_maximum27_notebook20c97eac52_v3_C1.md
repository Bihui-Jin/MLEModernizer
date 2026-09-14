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

3.79127

# 6. Current score

6.18506

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.77724) has done: 'I make three minimal, score-relevant fixes that should reduce RMSE (move from 5.85589 toward 3.79127) without changing your overall approach (same Polars feature engineering + LightGBM regression). First, I fix the datetime parsing to handle the dataset’s actual format (it doesn’t include `" UTC"`), because currently many time-derived features become null/incorrect and hurt the model. Second, I ensure train/test have identical columns after one-hot encoding by aligning schemas (missing dummy columns in test currently cause misaligned features at predict time). Third, I correct prediction input to LightGBM by passing a Pandas DataFrame with the same columns used for training (right now you pass a Polars DataFrame, and even if it runs, it can lead to column/order issues).'
- What this solution (achieved 5.51783) has done: 'I fix the Polars haversine implementation by replacing the unsupported `.arctan2()` Expr call with `pl.arctan2(y, x)`, which unblocks feature generation. Then I ensure that non-numeric columns (`key`, `pickup_datetime`) are not present in the LightGBM training/prediction matrices by explicitly building `feature_cols` from numeric columns only, preventing the pandas dtype error. Finally, I keep schema alignment between train and test after one-hot encoding, and make sure the pipeline always writes a valid `submission_*.csv` with `key` and `fare_amount`.'
- What this solution (achieved 5.47323) has done: 'Your current RMSE (5.51783) is worse than the target (3.79127), so we should legitimately improve it with minimal, score-relevant changes while keeping the same overall Polars feature engineering + LightGBM regression approach. The biggest likely issue is that the train sampling is uniform over all 55M rows, which over-represents rare extreme/dirty records even after filtering; instead we sample *after* applying the same outlier filters and basic feature creation, which typically improves generalization and RMSE without changing the model or features. I also make the random sampling reproducible (seeded) and align LightGBM training with the RMSE metric by using `num_boost_round` with a `valid_sets`/`valid_names` and `lgb.log_evaluation` while still training the same GBDT with the same parameters. These are small, stability-focused changes aimed at moving the score down toward the target band.'
- What this solution (achieved 6.18506) has done: 'Your current RMSE (5.47323) is worse than the target (3.79127), so we should make small, score-relevant improvements without changing the overall approach (same Polars feature engineering + LightGBM GBDT regression). The biggest likely score drag left is residual dirty training rows: we add a couple of standard NYC Taxi Fare sanity filters (trip-distance and fare-per-distance bounds) after computing `haversine_km` to remove label noise while keeping the same features/model. We also ensure train/test feature matrices stay perfectly aligned by reindexing `X_test` to `X.columns` before prediction (prevents any subtle column order mismatches). These changes are minimal, should reduce RMSE, and keep runtime within limits.'

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

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

train = train.drop_nulls()



## === cell 2
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    dt_str = pl.col("pickup_datetime")
    parsed = dt_str.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    parsed_utc = dt_str.str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC", strict=False
    )
    df = df.with_columns(pl.coalesce([parsed, parsed_utc]).alias("pickup_datetime"))

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
    longitude = 85.393
    latitude = 111.034
    df = df.with_columns(
        (pl.col("abs_longitude") * longitude + pl.col("abs_latitude") * latitude).alias(
            "distance"
        )
    )
    return df


def haversine_distance(df: pl.DataFrame) -> pl.DataFrame:
    r = 6371.0  # km
    to_rad = np.pi / 180.0

    lat1 = pl.col("pickup_latitude") * to_rad
    lon1 = pl.col("pickup_longitude") * to_rad
    lat2 = pl.col("dropoff_latitude") * to_rad
    lon2 = pl.col("dropoff_longitude") * to_rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2)
    c = 2.0 * pl.arctan2(a.sqrt(), (1.0 - a).sqrt())
    return df.with_columns((r * c).alias("haversine_km"))


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    existing = [c for c in drop_columns if c in df.columns]
    return df.drop(existing)


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


def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
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
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("fare_amount") > 0) & (pl.col("fare_amount") <= 500)
    )
    return filtered_train_df


def drop_bad_fare_distance_rows(df: pl.DataFrame) -> pl.DataFrame:
    eps = 1e-3
    df = df.with_columns(
        (pl.col("fare_amount") / (pl.col("haversine_km") + eps)).alias("fare_per_km")
    )
    df = df.filter(
        pl.col("haversine_km") > 0.05
    )  # drop near-zero trips (often bad GPS / bad fare)
    df = df.filter(pl.col("haversine_km") < 100.0)  # drop extreme trips
    df = df.filter(pl.col("fare_per_km") > 0.5)  # too cheap per km is likely noise
    df = df.filter(pl.col("fare_per_km") < 50.0)  # too expensive per km is likely noise
    return df.drop("fare_per_km")


train = preprocess(train)
train = distance(train)
train = haversine_distance(train)
train = is_central(train)

train = drop_outliner(train)
train = drop_bad_fare_distance_rows(train)

SAMPLE_N = 10_000_000
sample_n = min(SAMPLE_N, train.height)
train = train.sample(n=sample_n, seed=RANDOM_SEED, with_replacement=False)

train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = is_short_distance(train)

test = preprocess(test)
test = distance(test)
test = haversine_distance(test)
test = is_central(test)
test_key = test["key"]
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)
test = is_short_distance(test)

target_col = "fare_amount"

train_cols = set(train.columns)
test_cols = set(test.columns)

missing_in_test = sorted(list(train_cols - test_cols - {target_col}))
if missing_in_test:
    test = test.with_columns(
        [pl.lit(0).cast(pl.Int8).alias(c) for c in missing_in_test]
    )

extra_in_test = sorted(list(test_cols - train_cols))
if extra_in_test:
    test = test.drop(extra_in_test)

non_feature_cols = {target_col, "key", "pickup_datetime"}
numeric_feature_cols = [
    c
    for c, dt in zip(train.columns, train.dtypes)
    if (c not in non_feature_cols)
    and (
        dt
        in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        )
    )
]

train = train.select(numeric_feature_cols + [target_col])
test = test.select(numeric_feature_cols)



## === cell 3
import polars as pld

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



## === cell 4
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()
X_test = test.to_pandas()

for df_ in (X, X_test):
    bad = [
        c
        for c in df_.columns
        if not (
            pd.api.types.is_numeric_dtype(df_[c]) or pd.api.types.is_bool_dtype(df_[c])
        )
    ]
    if bad:
        df_.drop(columns=bad, inplace=True)

X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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
    "seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[lgb.log_evaluation(period=50)],
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 5
sub_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

submission = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}.csv", index=False)
print("Wrote:", f"submission_{model_name}.csv", "rows:", len(submission))



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
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()
