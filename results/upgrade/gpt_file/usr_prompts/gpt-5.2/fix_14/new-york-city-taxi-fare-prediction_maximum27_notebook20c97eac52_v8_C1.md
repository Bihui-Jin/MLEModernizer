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

4.31221

# 6. Current score

7.36209

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.52631) has done: 'Your score is worse than the target (RMSE 5.83 vs 4.31), so we should make small, legitimate fixes that typically reduce Taxi Fare RMSE without changing the overall approach (still: Polars feature engineering + LightGBM regression). The biggest likely issue is datetime parsing (your format expects “ UTC” but the competition strings are usually without it), which can silently create null datetimes and degrade all time features; we parse robustly. Next, your train/test one-hot columns can mismatch (train has many pickup_day/pickup_year levels that test may not, and vice versa), causing feature misalignment; we align columns before training/predicting. Finally, we use LightGBM’s standard early stopping (doesn’t change core logic, just picks the best iteration) and ensure we predict using the same pandas feature matrix for test.'
- What this solution (achieved 5.62848) has done: 'Your current gap to the target is large (RMSE 5.526 → 4.312), so we should make small but high-impact fixes that keep the same overall approach (Polars feature engineering + LightGBM regression). The biggest likely score drag is that `pickup_datetime` strings include fractional seconds (e.g. `.0000002`), so your current `strptime("%Y-%m-%d %H:%M:%S")` produces many null datetimes; we parse robustly to retain time features. Next, we add a single, standard distance feature (haversine) using existing latitude/longitude columns—this doesn’t change the modeling approach, but usually materially improves NYC taxi RMSE. Finally, we tighten outlier filtering to remove zero/negative fares and obvious coordinate errors (still the same filtering step), which typically helps RMSE without changing semantics.'
- What this solution (achieved 5.13578) has done: 'Your RMSE is still far above target, so we make two minimal, high-impact adjustments that keep the same overall approach (Polars feature engineering + LightGBM regression). First, we tighten and correct the train/test feature alignment: after `align(join="outer")` we reindex test to the exact training column order (this avoids subtle column order mismatches that can hurt predictions). Second, we add a standard NYC “airport/center distance” pair of features (distance to NYC center for pickup/dropoff) using the same existing coordinate columns; this is still simple feature engineering and typically reduces RMSE without changing the modeling method. Everything else (sampling, filters, LightGBM objective, early stopping, submission format/path) stays the same.'
- What this solution (achieved 5.75015) has done: 'Your current RMSE (5.13578) is worse than the target (4.31221), so we should make small, legitimate improvements that typically reduce NYC Taxi Fare error without changing the overall approach (same Polars feature engineering + LightGBM regression). The biggest likely score drag is that you apply one-hot encoding on `pickup_day` and `pickup_year`, which creates many sparse/dummy columns and can generalize poorly to test; we keep the same information but encode these as numeric instead (while still keeping your cyclic encodings). Next, we add two very standard, low-risk coordinate interaction features (`abs_lon_diff`, `abs_lat_diff`) that usually help alongside haversine/manhattan. Finally, we make the train/test split deterministic with `shuffle=True` and keep GPU LightGBM/early stopping unchanged.'
- What this solution (achieved 5.91548) has done: 'Your RMSE (5.75) is still far above the target (4.31), so we need a small but high-impact improvement while keeping the same pipeline (Polars feature engineering + LightGBM regression). The most reliable boost for this competition, without changing the modeling approach, is to add standard geographic features that capture NYC-specific structure: (1) Haversine distance to the main airports (JFK/LGA/EWR) for pickup/dropoff and (2) a simple bearing feature between pickup and dropoff. These are minimal feature-engineering additions using existing columns and typically reduce RMSE materially. Everything else (sampling, filtering approach, train/val split, LightGBM objective/loop, submission format/path) is kept intact.'
- What this solution (achieved 5.36145) has done: 'Your RMSE is still far above the target (5.915 → 4.312, lower is better), so we make the smallest high-impact, competition-standard fixes without changing the overall approach (same Polars feature engineering + LightGBM regression). The main issue is that raw haversine/manhattan distances are in km/deg, while the target fare is in dollars; adding a single multiplicative scaling feature (distance × typical $/km) and a basic passenger interaction helps LightGBM learn the right magnitude more directly. We also add one minimal outlier filter for “near-zero distance but large fare” which is common label noise in this dataset and usually hurts RMSE. Everything else (sampling size, model type, training loop/early stopping, submission format) is kept intact.'
- What this solution (achieved 5.73723) has done: 'Your current RMSE (5.361) is worse than the target (4.312, lower is better), so we should make small, high-impact fixes that don’t change the overall pipeline (same Polars feature engineering + LightGBM regression). The most likely remaining drag is training on a very large but still noisy sample without the classic “fare vs distance” consistency filters; we add two standard, minimal filters: remove near-zero-distance trips with non-trivial fare (and vice versa) after distances are computed. We also clip negative predictions to zero at submission time (a legitimate post-process for a non-negative target) which usually improves RMSE slightly without changing the model. Everything else (sampling size, features, LightGBM objective/loop/early stopping, submission format/path) stays the same.'
- What this solution (achieved 7.37869) has done: 'Your RMSE is still much worse than the target, so the most likely remaining issue is label noise/outliers that slip through your current filters and dominate RMSE. I make one minimal, competition-standard improvement: add a “reasonable fare per km” filter after you compute `haversine_km`, which removes trips with impossible $/km ratios while keeping your model/feature pipeline unchanged. I also add two tiny stability fixes that typically help a bit without changing core logic: fill null engineered time fields (from occasional datetime parse failures) with safe defaults, and set a deterministic seed for the row sampling so your score doesn’t swing between runs. Everything else (LightGBM, features, training loop, submission format/path) stays the same.'
- What this solution (achieved 6.61698) has done: 'Your RMSE is much worse than the target, so the smallest likely “big win” is fixing a data-quality bug: you compute key features (haversine etc.) on rows that still include null/invalid coordinates, then filter later—this can create null/inf values that LightGBM handles poorly and inflates error. I (1) move `drop_nulls` and basic coordinate/passenger/fare sanity filtering to happen before any distance/ratio features are computed, (2) make `drop_outliner` robust by also dropping null/NaN/inf distances and using a slightly less aggressive $/km filter (your last change likely over-filtered and hurt generalization), and (3) keep everything else (model, training loop, feature set, submission format/path) identical so the semantics don’t change beyond cleaning invalid rows. These changes should move RMSE down toward the 4.31 target without altering the core approach.'
- What this solution (achieved 6.71117) has done: 'Your current script likely doesn’t yield a valid Kaggle submission because `basic_sanity_filter_test()` filters rows out of the test set, so `test_key` and `sub_pred` no longer have the required 9914 rows/keys. I keep your model and features intact, but change test handling to never drop test rows: instead, we keep all test keys, compute features, and fill invalid/missing coordinates/features with safe defaults so prediction length always matches the submission template. This is a minimal semantic change (no model/loop/loss changes) and is directly aimed at producing a valid .csv and improving RMSE versus “missing rows / wrong alignment” failures. I also ensure column alignment uses the exact training column order (already mostly done) and write the submission using `sample_submission.csv` keys order for perfect alignment.'
- What this solution (achieved 7.7956) has done: 'Your current RMSE (6.71) is far worse than the 4.31 target, so we should make a small, high-impact fix that doesn’t change the modeling approach: your preprocessing never removes extreme-but-in-range label noise that dominates RMSE, and the `fare_per_km` filter is currently too permissive for this competition. I tighten `drop_outliner()` with standard NYC Taxi sanity bounds on fare-per-km and distance, and I add one minimal filter for unrealistic coordinate deltas (still just data cleaning, not a model change). Everything else stays the same (same Polars feature pipeline, same LightGBM regressor/params/early stopping, same submission alignment), so the change is directly aimed at reducing RMSE toward the target.'
- What this solution (achieved 7.36209) has done: 'Your current RMSE is much worse than the target (7.7956 vs 4.31221, lower is better), so we should make the smallest fixes that typically yield a large, reliable RMSE drop without changing the overall pipeline (same Polars feature engineering + LightGBM regression). The biggest likely issue now is that the tightened outlier rules are over-filtering/warping the training distribution; I relax the fare-per-km and distance bounds back toward standard NYC Taxi “reasonable” ranges while keeping the same filter mechanism. I also add one classic, minimal geographic feature that’s missing in your current set: a “manhattan distance in km” approximation (lat/lon scaled), which usually materially improves taxi-fare RMSE while staying within the same feature-engineering approach. Everything else (sampling size, model type/params, early stopping, submission alignment) stays the same.'

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

np.random.seed(42)

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

num_rows = train_df.height
random_indices = np.random.choice(num_rows, size=10000000, replace=False)
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
        pl.col("pickup_datetime").str.replace(r"\s+UTC$", "", literal=False)
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
    )

    df = df.with_columns(
        [
            pl.col("pickup_year").fill_null(2010),
            pl.col("pickup_month").fill_null(1),
            pl.col("pickup_day").fill_null(1),
            pl.col("pickup_hour").fill_null(0),
            pl.col("pickup_minute").fill_null(0),
            pl.col("pickup_second").fill_null(0),
            pl.col("pickup_weekday").fill_null(0),
        ]
    )
    return df


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    return df.drop(drop_columns)


def cycling_encoding(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )


def one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("pickup_year").cast(pl.Int16),
            pl.col("pickup_day").cast(pl.Int8),
        ]
    )


def add_distance_features(df: pl.DataFrame) -> pl.DataFrame:
    r = 6371.0  # km
    lat1 = pl.col("pickup_latitude") * np.pi / 180.0
    lat2 = pl.col("dropoff_latitude") * np.pi / 180.0
    dlat = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * np.pi / 180.0
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * np.pi / 180.0

    a = (dlat / 2.0).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2.0).sin() ** 2
    c = 2.0 * a.sqrt().arcsin()
    hav_km = (r * c).alias("haversine_km")

    man_deg = (
        (pl.col("dropoff_latitude") - pl.col("pickup_latitude")).abs()
        + (pl.col("dropoff_longitude") - pl.col("pickup_longitude")).abs()
    ).alias("manhattan_deg")

    abs_lat_diff = (
        (pl.col("dropoff_latitude") - pl.col("pickup_latitude"))
        .abs()
        .alias("abs_lat_diff")
    )
    abs_lon_diff = (
        (pl.col("dropoff_longitude") - pl.col("pickup_longitude"))
        .abs()
        .alias("abs_lon_diff")
    )

    mean_lat = (
        ((pl.col("pickup_latitude") + pl.col("dropoff_latitude")) / 2.0) * np.pi / 180.0
    )
    dlat_km = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")).abs() * 111.32
    dlon_km = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")).abs() * (
        111.32 * mean_lat.cos()
    )
    man_km = (dlat_km + dlon_km).alias("manhattan_km")

    return df.with_columns([hav_km, man_deg, abs_lat_diff, abs_lon_diff, man_km])


def add_nyc_center_features(df: pl.DataFrame) -> pl.DataFrame:
    nyc_lat = 40.7580
    nyc_lon = -73.9855
    r = 6371.0  # km

    def haversine_km(lat_col: str, lon_col: str, out_name: str) -> pl.Expr:
        lat1 = pl.col(lat_col) * np.pi / 180.0
        lon1 = pl.col(lon_col) * np.pi / 180.0
        lat2 = pl.lit(nyc_lat) * np.pi / 180.0
        lon2 = pl.lit(nyc_lon) * np.pi / 180.0
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2.0).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2.0).sin() ** 2
        c = 2.0 * a.sqrt().arcsin()
        return (r * c).alias(out_name)

    return df.with_columns(
        [
            haversine_km("pickup_latitude", "pickup_longitude", "pickup_to_center_km"),
            haversine_km(
                "dropoff_latitude", "dropoff_longitude", "dropoff_to_center_km"
            ),
        ]
    )


def add_airport_features(df: pl.DataFrame) -> pl.DataFrame:
    r = 6371.0  # km

    airports = {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
    }

    def haversine_to(
        lat_col: str, lon_col: str, lat0: float, lon0: float, out_name: str
    ) -> pl.Expr:
        lat1 = pl.col(lat_col) * np.pi / 180.0
        lon1 = pl.col(lon_col) * np.pi / 180.0
        lat2 = pl.lit(lat0) * np.pi / 180.0
        lon2 = pl.lit(lon0) * np.pi / 180.0
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2.0).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2.0).sin() ** 2
        c = 2.0 * a.sqrt().arcsin()
        return (r * c).alias(out_name)

    exprs = []
    for code, (alat, alon) in airports.items():
        exprs.append(
            haversine_to(
                "pickup_latitude",
                "pickup_longitude",
                alat,
                alon,
                f"pickup_to_{code}_km",
            )
        )
        exprs.append(
            haversine_to(
                "dropoff_latitude",
                "dropoff_longitude",
                alat,
                alon,
                f"dropoff_to_{code}_km",
            )
        )

    return df.with_columns(exprs)


def add_bearing_feature(df: pl.DataFrame) -> pl.DataFrame:
    lat1 = pl.col("pickup_latitude") * np.pi / 180.0
    lat2 = pl.col("dropoff_latitude") * np.pi / 180.0
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * np.pi / 180.0

    y = dlon.sin() * lat2.cos()
    x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos()
    bearing = pl.arctan2(y, x).alias("bearing_rad")
    return df.with_columns([bearing])


def basic_sanity_filter_train(df: pl.DataFrame) -> pl.DataFrame:
    df = df.drop_nulls(
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "fare_amount",
        ]
    )
    df = df.filter((pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41))
    df = df.filter(
        (pl.col("pickup_longitude") >= -74.5) & (pl.col("pickup_longitude") < -72.5)
    )
    df = df.filter(
        (pl.col("dropoff_longitude") >= -74.5) & (pl.col("dropoff_longitude") < -72.5)
    )
    df = df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41)
    )
    df = df.filter((pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6))
    df = df.filter((pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 500))
    return df


def basic_sanity_prepare_test_no_drop(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("passenger_count").fill_null(1).cast(pl.Int64))
    return df


def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
    df = df.drop_nulls(
        [
            "haversine_km",
            "manhattan_deg",
            "abs_lat_diff",
            "abs_lon_diff",
            "manhattan_km",
        ]
    )
    df = df.filter(
        pl.col("haversine_km").is_finite()
        & pl.col("manhattan_deg").is_finite()
        & pl.col("abs_lat_diff").is_finite()
        & pl.col("abs_lon_diff").is_finite()
        & pl.col("manhattan_km").is_finite()
    )

    df = df.filter((pl.col("abs_lat_diff") <= 1.0) & (pl.col("abs_lon_diff") <= 1.0))

    df = df.filter(~((pl.col("haversine_km") < 0.10) & (pl.col("fare_amount") > 15.0)))
    df = df.filter(~((pl.col("haversine_km") > 30.0) & (pl.col("fare_amount") < 5.0)))
    df = df.filter(~((pl.col("haversine_km") < 0.05) & (pl.col("fare_amount") > 20)))

    fare_per_km = pl.col("fare_amount") / (pl.col("haversine_km") + 1e-3)

    df = df.filter((pl.col("haversine_km") >= 0.01) & (pl.col("haversine_km") <= 200.0))
    df = df.filter((fare_per_km >= 0.5) & (fare_per_km <= 60.0))

    return df


def add_fare_scale_interactions(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("haversine_km") * 2.5).alias("hav_km_x_2p5"),
            (pl.col("haversine_km") * 3.0).alias("hav_km_x_3p0"),
            (pl.col("haversine_km") * pl.col("passenger_count")).alias(
                "hav_km_x_passenger"
            ),
        ]
    )


train = basic_sanity_filter_train(train)

train = preprocess(train)
train = add_distance_features(train)
train = add_nyc_center_features(train)
train = add_airport_features(train)
train = add_bearing_feature(train)
train = add_fare_scale_interactions(train)
train = drop_encoding(train)
train = cycling_encoding(train)
train = one_hot_encoding(train)
train = drop_outliner(train)

test = basic_sanity_prepare_test_no_drop(test)
test_key = test["key"]

test = preprocess(test)
test = add_distance_features(test)
test = add_nyc_center_features(test)
test = add_airport_features(test)
test = add_bearing_feature(test)
test = add_fare_scale_interactions(test)
test = drop_encoding(test)
test = cycling_encoding(test)
test = one_hot_encoding(test)

num_cols = [
    c
    for c, dt in zip(test.columns, test.dtypes)
    if dt in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
]
fix_exprs = []
for c in num_cols:
    if c == "fare_amount":
        continue
    fix_exprs.append(
        pl.when(pl.col(c).cast(pl.Float64).is_finite() & pl.col(c).is_not_null())
        .then(pl.col(c))
        .otherwise(0.0)
        .alias(c)
    )
test = test.with_columns(fix_exprs)



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

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_pd = train.to_pandas()
test_pd = test.to_pandas()

y = train_pd["fare_amount"]
X = train_pd.drop(columns=["fare_amount"])

test_pd = test_pd.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0, shuffle=True
)

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

bst = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 6
sub_pred = bst.predict(test_pd, num_iteration=bst.best_iteration)
sub_pred = np.clip(sub_pred, 0.0, None)

exp_num = "base"
sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

pred_df = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})
submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")

submission["fare_amount"] = submission["fare_amount"].fillna(float(np.mean(sub_pred)))

submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
print(
    f"Wrote: submission_{model_name}_{exp_num}.csv  rows={len(submission)}  cols={list(submission.columns)}"
)



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



## === cell 8
train.head()
