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

3.60484

# 6. Current score

7.15406

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.6468) has done: 'Your score is worse than the target (RMSE 4.63551 vs 3.60484), so we should make small, legitimate fixes that improve generalization without changing the core model/training approach. The biggest issue is feature mismatch between train/test caused by one-hot encoding `pickup_year`/`pickup_day` separately, which leads to inconsistent columns (and LightGBM may silently mis-handle). I fix this by aligning train/test columns after encoding (add missing columns with 0, same order), and also correct datetime parsing to handle the dataset’s actual format (no literal `" UTC"`), avoiding null-heavy parses. Finally, I ensure predictions use a pandas DataFrame with the exact same feature columns used for training, and clip negative fares to 0 to reduce RMSE on obviously invalid outputs.'
- What this solution (achieved 5.33907) has done: 'Your current RMSE (4.6468) is worse than the target (3.60484), so we should make small, legitimate fixes that typically improve generalization without changing the model/training approach. The biggest win with minimal risk is to clean obviously bad rows (invalid lat/lon ranges, zero passenger_count, extreme fares) and add one core feature (haversine distance) that is standard for this competition, while keeping LightGBM and your preprocessing structure intact. I also make datetime parsing robust for the dataset’s actual format (it includes microseconds and no literal `" UTC"`), which prevents null/garbled time features. Finally, I keep your train/test dummy-column alignment and submission generation unchanged in semantics, just ensuring types are consistent.'
- What this solution (achieved 5.63145) has done: 'Your RMSE (5.33907) is worse than the target (3.60484), so we should make small, legitimate fixes that typically improve generalization without changing your LightGBM approach. The biggest issue is likely that `pickup_month`/`pickup_hour` were being used twice (raw + cyclic), which can add noise, and that rows with null `pickup_datetime` parsing weren’t being removed after feature extraction. I (1) parse datetime robustly and drop rows where datetime-derived features are null, (2) tighten cleaning with a standard “NYC bounding box + distance sanity” filter, and (3) avoid duplicating time features by dropping the raw `pickup_month`/`pickup_hour` once cyclic features are added (core model/training stays identical). These are minimal preprocessing adjustments that usually reduce RMSE on this competition without altering the training loop or model family.'
- What this solution (achieved 5.28197) has done: 'Your RMSE (5.63145) is still far from the target (3.60484), so we should make a small, legitimate improvement that usually yields a meaningful RMSE drop for this competition without changing the model family or training loop: add the standard “distance in meters” feature and a simple log1p-distance feature, and apply the exact same distance sanity filtering to the test feature generation (without dropping rows) to avoid extreme outliers at inference. I also ensure the one-hot encoded train/test columns are aligned in a deterministic order (sorted feature columns) so LightGBM sees consistent feature ordering, which can otherwise hurt generalization. Finally, I slightly tighten the “zero-distance” handling by allowing small distances (very short trips) rather than filtering them out completely, which reduces bias on short fares.'
- What this solution (achieved 5.32595) has done: 'Your current RMSE (5.28197) is worse than the target (3.60484), so we need a small, high-impact improvement without changing the overall LightGBM approach. The main issue is that your one-hot encoding creates extremely sparse/high-dimensional time features (pickup_day and pickup_year), which tends to hurt generalization on this competition; replacing them with numeric (non-one-hot) time features preserves the same model/training loop but usually reduces RMSE materially. I also add two standard, minimal geospatial features (Manhattan distance in degrees and bearing) derived from the same coordinates, which is a common low-risk boost for taxi fare RMSE. Finally, I keep your column alignment/ordering safeguards and submission format unchanged.'
- What this solution (achieved 5.31594) has done: 'Your current RMSE is worse than the target, so we should make a small, high-impact fix without changing the model family or training loop: add a minimal set of widely-used location features that LightGBM exploits well (pickup/dropoff distance to NYC center, and absolute coordinates), while keeping your existing distance/bearing/time pipeline intact. I also make datetime parsing robust to the dataset’s actual timezone suffix (it’s often `" UTC"` or `"+00:00"`), which prevents silent nulls in time features that hurt generalization. Finally, I keep your column alignment and submission logic the same, just ensuring these added features are computed identically for train and test.'
- What this solution (achieved 6.69873) has done: 'I fix the Polars feature bug in `add_nyc_bbox_flags()` where a newly created column is referenced in the same `with_columns()` call (Polars doesn’t allow that), by splitting it into two sequential `with_columns()` calls. Then I ensure LightGBM only receives numeric features by explicitly selecting `feature_cols` (which already excludes `key` / `pickup_datetime`) when building the pandas matrices, preventing the “bad pandas dtypes” error. Finally, I make test-time column ordering deterministic using the same `feature_cols` list (instead of `X.columns`, which differed due to the earlier error), and write a valid `submission_*.csv` file.'
- What this solution (achieved 7.15406) has done: 'Your current RMSE (6.69873) is much worse than the target (3.60484), so we should make a small, legitimate change that typically improves this competition without changing the model family or training loop: add a simple passenger-count interaction and a distance-squared feature (both derived from existing inputs), which LightGBM can exploit to model nonlinear fare-vs-distance and per-passenger effects. I also make test-time datetime parsing safer by explicitly filling null-derived time parts with a default (0) so the model doesn’t see nulls at inference. Finally, I keep your column alignment/ordering and submission format identical, only extending the feature set in a consistent way for train and test.'

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

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

num_rows = train_df.height
rng = np.random.default_rng(0)
random_indices = rng.choice(num_rows, size=10000000, replace=False)

train_df = train_df[random_indices]
train_df = train_df.drop_nulls()



## === cell 2
train = train_df
test = test_df



## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    dt = (
        pl.col("pickup_datetime")
        .str.replace(" UTC", "", literal=True)
        .str.replace("+00:00", "", literal=True)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
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
    ).with_columns(
        [
            pl.col("pickup_year").fill_null(0).cast(pl.Int16),
            pl.col("pickup_month").fill_null(0).cast(pl.Int8),
            pl.col("pickup_day").fill_null(0).cast(pl.Int8),
            pl.col("pickup_hour").fill_null(0).cast(pl.Int8),
            pl.col("pickup_minute").fill_null(0).cast(pl.Int8),
            pl.col("pickup_second").fill_null(0).cast(pl.Int8),
            pl.col("pickup_weekday").fill_null(0).cast(pl.Int8),
        ]
    )
    return df


def clean_train(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("fare_amount") >= 0.0)
        & (pl.col("fare_amount") <= 500.0)
        & (pl.col("passenger_count") >= 1)
        & (pl.col("passenger_count") <= 6)
        & (pl.col("pickup_longitude") >= -75.0)
        & (pl.col("pickup_longitude") <= -72.0)
        & (pl.col("dropoff_longitude") >= -75.0)
        & (pl.col("dropoff_longitude") <= -72.0)
        & (pl.col("pickup_latitude") >= 40.0)
        & (pl.col("pickup_latitude") <= 42.0)
        & (pl.col("dropoff_latitude") >= 40.0)
        & (pl.col("dropoff_latitude") <= 42.0)
        & (pl.col("pickup_year").is_not_null())
        & (pl.col("pickup_month").is_not_null())
        & (pl.col("pickup_day").is_not_null())
        & (pl.col("pickup_hour").is_not_null())
    )


def add_distance(df: pl.DataFrame) -> pl.DataFrame:
    r_km = 6371.0
    to_rad = np.pi / 180.0
    lat1 = pl.col("pickup_latitude") * to_rad
    lat2 = pl.col("dropoff_latitude") * to_rad
    dlat = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * to_rad
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * to_rad

    a = (dlat / 2).sin().pow(2) + (lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2))
    c = (a.sqrt().arcsin()) * 2.0
    hav_km = (pl.lit(r_km) * c).alias("haversine_km")

    df = df.with_columns(hav_km)

    df = df.with_columns(
        [
            (pl.col("haversine_km") * 1000.0).alias("haversine_m"),
            (pl.col("haversine_km").log1p()).alias("log1p_haversine_km"),
        ]
    )
    return df


def add_geo_features(df: pl.DataFrame) -> pl.DataFrame:
    to_rad = np.pi / 180.0
    dlat_deg = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")).alias(
        "delta_lat"
    )
    dlon_deg = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")).alias(
        "delta_lon"
    )

    df = df.with_columns([dlat_deg, dlon_deg])

    df = df.with_columns(
        [
            (pl.col("delta_lat").abs() + pl.col("delta_lon").abs()).alias(
                "manhattan_deg"
            ),
        ]
    )

    lat1 = pl.col("pickup_latitude") * to_rad
    lat2 = pl.col("dropoff_latitude") * to_rad
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * to_rad
    y = dlon.sin() * lat2.cos()
    x = (lat1.cos() * lat2.sin()) - (lat1.sin() * lat2.cos() * dlon.cos())
    df = df.with_columns(pl.arctan2(y, x).alias("bearing"))
    return df


def add_center_features(df: pl.DataFrame) -> pl.DataFrame:
    nyc_lon = -73.985428  # Times Sq-ish
    nyc_lat = 40.748817

    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - pl.lit(nyc_lon))
            .abs()
            .alias("pickup_lon_abs_center_diff"),
            (pl.col("pickup_latitude") - pl.lit(nyc_lat))
            .abs()
            .alias("pickup_lat_abs_center_diff"),
            (pl.col("dropoff_longitude") - pl.lit(nyc_lon))
            .abs()
            .alias("dropoff_lon_abs_center_diff"),
            (pl.col("dropoff_latitude") - pl.lit(nyc_lat))
            .abs()
            .alias("dropoff_lat_abs_center_diff"),
            (
                (pl.col("pickup_longitude") - pl.lit(nyc_lon)).pow(2)
                + (pl.col("pickup_latitude") - pl.lit(nyc_lat)).pow(2)
            )
            .sqrt()
            .alias("pickup_center_euclid_deg"),
            (
                (pl.col("dropoff_longitude") - pl.lit(nyc_lon)).pow(2)
                + (pl.col("dropoff_latitude") - pl.lit(nyc_lat)).pow(2)
            )
            .sqrt()
            .alias("dropoff_center_euclid_deg"),
        ]
    )
    return df


def add_abs_coord_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col("pickup_longitude").abs().alias("pickup_longitude_abs"),
            pl.col("pickup_latitude").abs().alias("pickup_latitude_abs"),
            pl.col("dropoff_longitude").abs().alias("dropoff_longitude_abs"),
            pl.col("dropoff_latitude").abs().alias("dropoff_latitude_abs"),
        ]
    )
    return df


def add_airport_distance_features(df: pl.DataFrame) -> pl.DataFrame:
    def haversine_to_point(
        lon_col: str, lat_col: str, lon0: float, lat0: float, out: str
    ):
        r_km = 6371.0
        to_rad = np.pi / 180.0
        lat1 = pl.col(lat_col) * to_rad
        lon1 = pl.col(lon_col) * to_rad
        lat2 = pl.lit(lat0) * to_rad
        lon2 = pl.lit(lon0) * to_rad
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2).sin().pow(2) + (
            lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2)
        )
        c = (a.sqrt().arcsin()) * 2.0
        return (pl.lit(r_km) * c).alias(out)

    jfk_lon, jfk_lat = -73.7781, 40.6413
    lga_lon, lga_lat = -73.8740, 40.7769
    ewr_lon, ewr_lat = -74.1745, 40.6895

    df = df.with_columns(
        [
            haversine_to_point(
                "pickup_longitude",
                "pickup_latitude",
                jfk_lon,
                jfk_lat,
                "pickup_to_jfk_km",
            ),
            haversine_to_point(
                "dropoff_longitude",
                "dropoff_latitude",
                jfk_lon,
                jfk_lat,
                "dropoff_to_jfk_km",
            ),
            haversine_to_point(
                "pickup_longitude",
                "pickup_latitude",
                lga_lon,
                lga_lat,
                "pickup_to_lga_km",
            ),
            haversine_to_point(
                "dropoff_longitude",
                "dropoff_latitude",
                lga_lon,
                lga_lat,
                "dropoff_to_lga_km",
            ),
            haversine_to_point(
                "pickup_longitude",
                "pickup_latitude",
                ewr_lon,
                ewr_lat,
                "pickup_to_ewr_km",
            ),
            haversine_to_point(
                "dropoff_longitude",
                "dropoff_latitude",
                ewr_lon,
                ewr_lat,
                "dropoff_to_ewr_km",
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.min_horizontal(
                ["pickup_to_jfk_km", "pickup_to_lga_km", "pickup_to_ewr_km"]
            ).alias("pickup_to_any_airport_km"),
            pl.min_horizontal(
                ["dropoff_to_jfk_km", "dropoff_to_lga_km", "dropoff_to_ewr_km"]
            ).alias("dropoff_to_any_airport_km"),
        ]
    )
    return df


def add_nyc_bbox_flags(df: pl.DataFrame) -> pl.DataFrame:
    lon_min, lon_max = -74.5, -72.8
    lat_min, lat_max = 40.5, 41.8

    pickup_in = (
        (
            (pl.col("pickup_longitude") >= lon_min)
            & (pl.col("pickup_longitude") <= lon_max)
            & (pl.col("pickup_latitude") >= lat_min)
            & (pl.col("pickup_latitude") <= lat_max)
        )
        .cast(pl.Int8)
        .alias("pickup_in_nyc_bbox")
    )

    dropoff_in = (
        (
            (pl.col("dropoff_longitude") >= lon_min)
            & (pl.col("dropoff_longitude") <= lon_max)
            & (pl.col("dropoff_latitude") >= lat_min)
            & (pl.col("dropoff_latitude") <= lat_max)
        )
        .cast(pl.Int8)
        .alias("dropoff_in_nyc_bbox")
    )

    df = df.with_columns([pickup_in, dropoff_in])
    df = df.with_columns(
        [
            (pl.col("pickup_in_nyc_bbox") & pl.col("dropoff_in_nyc_bbox"))
            .cast(pl.Int8)
            .alias("both_in_nyc_bbox")
        ]
    )
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
    return df.drop(["pickup_month", "pickup_hour"])


def one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    return df


def add_simple_interactions(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col("haversine_km") ** 2).alias("haversine_km_sq"),
            (pl.col("haversine_km") * pl.col("passenger_count")).alias(
                "hav_km_x_passengers"
            ),
            (pl.col("manhattan_deg") * pl.col("passenger_count")).alias(
                "manhattan_x_passengers"
            ),
        ]
    )
    return df


train = preprocess(train)
train = clean_train(train)
train = add_distance(train)
train = add_geo_features(train)
train = add_center_features(train)
train = add_abs_coord_features(train)
train = add_airport_distance_features(train)
train = add_nyc_bbox_flags(train)

train = train.filter((pl.col("haversine_km") >= 0.0) & (pl.col("haversine_km") < 200.0))

train = add_simple_interactions(train)

train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)

test = preprocess(test)
test_key = test["key"]
test = add_distance(test)
test = add_geo_features(test)
test = add_center_features(test)
test = add_abs_coord_features(test)
test = add_airport_distance_features(test)
test = add_nyc_bbox_flags(test)

test = add_simple_interactions(test)

test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)

train_cols = set(train.columns)
test_cols = set(test.columns)

missing_in_test = sorted(list(train_cols - test_cols))
missing_in_train = sorted(list(test_cols - train_cols))

if missing_in_test:
    test = test.with_columns(
        [pl.lit(0).cast(pl.Float32).alias(c) for c in missing_in_test]
    )
if missing_in_train:
    train = train.with_columns(
        [pl.lit(0).cast(pl.Float32).alias(c) for c in missing_in_train]
    )

feature_cols = sorted([c for c in train.columns if c != "fare_amount"])
train = train.select(["fare_amount"] + feature_cols)
test = test.select(feature_cols)



## === cell 4
import polars as pl

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

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train.select(feature_cols).to_pandas()
y = train["fare_amount"].to_pandas()

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
}

bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mse = mean_squared_error(y_val, y_pred)
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 6
X_test = test.select(feature_cols).to_pandas()

sub_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
sub_pred = np.clip(sub_pred, 0, None)

exp_num = "base"
submission = pd.DataFrame({"key": test_key.to_list(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
print(f"Wrote submission_{model_name}_{exp_num}.csv with shape {submission.shape}")



## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = feature_cols
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



## === cell 8
train.head()
