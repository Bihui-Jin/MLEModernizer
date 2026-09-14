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

3.79867

# 6. Current score

5.4773

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.76597) has done: 'I make three minimal, score-relevant fixes: (1) parse `pickup_datetime` robustly (your current format doesn’t match the dataset, which silently creates null datetimes and hurts features), (2) ensure train/test one-hot columns are perfectly aligned (your independent `to_dummies` creates mismatched feature sets and degrades predictions), and (3) pass a pandas DataFrame with the same columns into `bst.predict` for test (right now you pass a Polars DF, which can lead to wrong column handling). These changes keep your model/training logic intact, but make the feature engineering consistent and correct, which should reduce RMSE toward the 3.79867 target. The pipeline still samples 10M rows and produces a valid submission CSV with the required `key,fare_amount` columns.'
- What this solution (achieved 5.85954) has done: 'Your current RMSE (5.76597) is worse than the target (3.79867), so we need small, legitimate changes that typically reduce error without changing the core model/training approach. The biggest remaining issue is that your train sampling is uncontrolled and can include many “bad” rows (label noise/outliers) because you don’t apply your `drop_outliner` rules until after feature engineering, and you also don’t remove obviously invalid fares (<=0). I make the sampling reproducible, apply a minimal extra fare filter, and (most importantly) keep all your existing feature logic but filter outliers earlier so the 10M sample contains a much higher fraction of valid NYC trips, which usually moves RMSE down substantially toward your target. The LightGBM setup, objective, and training loop remain unchanged, and the script still writes a valid `submission_*.csv`.'
- What this solution (achieved 5.68353) has done: 'Your current RMSE (5.85954) is worse than the target (3.79867), so we should make small, score-relevant fixes that reduce label noise without changing the model/training logic. The biggest remaining leakage/noise source is that you only filter obvious outliers *after* randomly sampling 10M rows, so the sample can contain many invalid/non‑NYC trips; instead, we apply the same geographic/passenger/fare filters at read-time (via `polars.read_csv` predicates) so the 10M sample is drawn from mostly-valid trips. To keep your feature/model core identical, we won’t add new features or change LightGBM settings; we only change *when* filtering happens and make the datetime parse explicitly match the dataset’s timestamp format so time features aren’t partially null. This should materially lower RMSE and move closer to the target while staying within Kaggle constraints and still producing the same submission schema.'
- What this solution (achieved 5.62981) has done: 'We need to move RMSE down (lower-is-better) from 5.68353 toward 3.79867, so we should make the smallest changes that reduce noise/outliers without changing your LightGBM training approach or feature set. The biggest remaining score drag is that the model can output negative fares and extreme fares, which are invalid and heavily penalized by RMSE; clipping predictions to a realistic range is legitimate post-processing that preserves the model and metric semantics. Also, your datetime parsing format still doesn’t match the dataset (no `" UTC"` suffix), which can silently create null time features; switching to Polars’ inference for this column is a minimal fix that usually improves generalization. Finally, we keep your one-hot/dummy alignment but make it symmetric and stable by aligning both train and test to the union of columns (instead of left-join), avoiding any accidental feature drop.'
- What this solution (achieved 5.62981) has done: 'We need to reduce RMSE (lower-is-better) from 5.62981 toward 3.79867, so the smallest high-impact change is to fix the train/test feature mismatch created by one-hot encoding `pickup_year` and `pickup_day`: the test set contains only a small set of days/years, so “outer” alignment adds many all-zero columns on test and weakens the model. I keep your exact feature set and LightGBM training approach, but change the alignment to keep only the intersection of columns (train↔test) and ensure identical column order, which usually improves generalization materially for this competition. I also make sure we keep `fare_amount` out of the alignment step explicitly to avoid any accidental column handling edge cases while preserving your semantics. Submission format and paths stay the same and a valid `.csv` is produced.'
- What this solution (achieved 5.48124) has done: 'Your RMSE (5.62981) is still far above the target (3.79867), so we should make the smallest changes that reduce label noise and obvious training/test distribution mismatch without changing your feature set or LightGBM training approach. The biggest remaining issue is that you currently train on many “bad” rows (zero-distance, extreme distance, and unusually huge fares) even after geographic filtering; these inflate RMSE heavily in this competition. I add two very standard, minimal filters applied before sampling: remove near-zero trip distance rows and cap fares to a realistic range (e.g., < 250), while keeping your exact feature engineering and LightGBM params/training loop unchanged. This typically moves RMSE down materially toward the 3.8 band, and the script still write a valid `submission_*.csv` with `key,fare_amount`.'
- What this solution (achieved 5.50843) has done: 'Your RMSE is still much worse than the 3.79867 target, so we need a small change that reduces noise without changing your model or feature set. The highest-impact minimal fix here is that your “distance” is currently a Manhattan-like linear approximation; replacing it with the standard haversine distance (in km) keeps the same single `distance` feature but makes it physically meaningful and typically improves this competition’s RMSE. To keep evaluation semantics stable and avoid any accidental feature mismatch, we also ensure the distance calculation never produces null/inf and keep your existing filtering/one-hot/model training exactly as-is. Submission writing remains unchanged and still produces `submission_lgbm_distance.csv`.'
- What this solution (achieved 5.56434) has done: 'Your current RMSE (5.50843, lower-is-better) is still far above the 3.79867 target, so we should make a small change that reduces label noise/outlier influence without changing your model, features, or training loop. The biggest remaining issue is that you apply outlier removal mostly via simple geo bounds and a distance>0.05 filter, but you still keep many “weird” trips (very long trips with unusual fares) that LightGBM struggles with and that are heavily penalized by RMSE. I add one standard, minimal filter on the already-computed `distance` feature to remove extreme distances (still within NYC bounds but often anomalous), and I also clip training fares to a realistic upper bound consistent with your earlier filter (keeping semantics consistent). This keeps the same feature set and LightGBM setup, but typically moves RMSE down toward your target band, and still writes a valid submission CSV.'
- What this solution (achieved 5.4773) has done: 'We need to reduce RMSE (lower-is-better) from 5.56434 toward 3.79867, so the smallest score-relevant change is to make your time parsing fully deterministic and non-null: your current `str.strptime(..., strict=False)` can still leave many null datetimes, which then makes several downstream time features null/garbage. I parse `pickup_datetime` with an explicit known format (the dataset uses `YYYY-MM-DD HH:MM:SS UTC`), and fall back to inference only if needed, then drop rows with null parsed datetimes in train (not test) to avoid training on broken time features. I also remove the duplicated dropoff_longitude filter in the initial scan (no semantic change, but avoids confusion) and keep your LightGBM params, features, and training loop intact. This should legitimately lower RMSE by fixing a high-impact feature-quality issue while preserving your core logic and still producing a valid submission CSV.'

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

np.random.seed(0)

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_scan = pl.scan_csv(train_path)

train_scan = train_scan.filter(
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
    & (pl.col("fare_amount") > 0)
    & (pl.col("fare_amount") < 250)
)

train_scan = train_scan.filter(
    (
        (pl.col("pickup_longitude") - pl.col("dropoff_longitude")).abs()
        + (pl.col("pickup_latitude") - pl.col("dropoff_latitude")).abs()
    )
    > 1e-4
)

train_df = train_scan.collect(streaming=True)

test_df = pl.read_csv(test_path)

num_rows = train_df.height
sample_size = min(10_000_000, num_rows)
random_indices = np.random.choice(num_rows, size=sample_size, replace=False)
train_df = train_df[random_indices]
train_df = train_df.drop_nulls()



## === cell 2
train = train_df
test = test_df



## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame, is_train: bool) -> pl.DataFrame:
    parsed = pl.coalesce(
        [
            pl.col("pickup_datetime").str.strptime(
                pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC", strict=False
            ),
            pl.col("pickup_datetime").str.strptime(pl.Datetime, strict=False),
        ]
    ).alias("pickup_datetime")

    df = df.with_columns(parsed)

    if is_train:
        df = df.drop_nulls(["pickup_datetime"])

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
    R = 6371.0  # Earth radius in km

    lat1 = pl.col("pickup_latitude") * (np.pi / 180.0)
    lon1 = pl.col("pickup_longitude") * (np.pi / 180.0)
    lat2 = pl.col("dropoff_latitude") * (np.pi / 180.0)
    lon2 = pl.col("dropoff_longitude") * (np.pi / 180.0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (dlat / 2.0).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2.0).sin() ** 2
    c = 2.0 * a.sqrt().arcsin()

    df = df.with_columns((R * c).alias("distance"))
    return df


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    return df.drop(drop_columns)


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


def diff_central(df: pl.DataFrame) -> pl.DataFrame:
    base_longitude = 85.393
    base_latitude = 111.034
    central_latitude = 40.764696
    central_longitude = -73.98065

    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - central_longitude)
            .abs()
            .alias("central_pickup_abs_longitude"),
            (pl.col("pickup_latitude") - central_latitude)
            .abs()
            .alias("central_pickup_abs_latitude"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("central_pickup_abs_longitude") * base_longitude
                + pl.col("central_pickup_abs_latitude") * base_latitude
            ).alias("central_pickup_distance")
        ]
    ).drop(["central_pickup_abs_longitude", "central_pickup_abs_latitude"])

    df = df.with_columns(
        [
            (pl.col("dropoff_longitude") - central_longitude)
            .abs()
            .alias("central_dropoff_abs_longitude"),
            (pl.col("dropoff_latitude") - central_latitude)
            .abs()
            .alias("central_dropoff_abs_latitude"),
        ]
    )
    df = df.with_columns(
        [
            (
                pl.col("central_dropoff_abs_longitude") * base_longitude
                + pl.col("central_dropoff_abs_latitude") * base_latitude
            ).alias("central_dropoff_distance")
        ]
    ).drop(["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"])

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
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6)
    )
    filtered_train_df = filtered_train_df.filter((pl.col("fare_amount") < 10000))
    return filtered_train_df


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


train = drop_outliner(train)
train = train.filter(pl.col("fare_amount") > 0)

train = preprocess(train, is_train=True)
train = distance(train)

train = train.filter((pl.col("distance") > 0.05) & (pl.col("distance") < 60.0))
train = train.filter(pl.col("fare_amount") < 250)

train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)

test = preprocess(test, is_train=False)
test = distance(test)
test_key = test["key"]
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)



## === cell 4
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

train_pd = train.to_pandas()
test_pd = test.to_pandas()

y = train_pd["fare_amount"]
X = train_pd.drop(columns=["fare_amount"])

common_cols = X.columns.intersection(test_pd.columns)
X = X.loc[:, common_cols]
test_pd = test_pd.loc[:, common_cols]

test_pd = test_pd.reindex(columns=X.columns)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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
}

bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mse = mean_squared_error(y_val, y_pred)
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 6
sub_pred = bst.predict(test_pd, num_iteration=bst.best_iteration)

sub_pred = np.clip(sub_pred, 0.0, 500.0)

exp_num = "distance"
submission = pd.DataFrame({"key": test_key.to_list(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)



## === cell 7
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
