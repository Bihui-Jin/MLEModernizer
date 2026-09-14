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

3.76015

# 6. Current score

10.79909

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.67858) has done: 'Your current score (5.86864 RMSE) is worse than the target (3.76015), so we should improve it with minimal, low-risk fixes that don’t change the overall modeling approach (still LightGBM on engineered tabular features). The biggest issue is feature misalignment between train and test caused by one-hot encoding being applied independently; we align columns so the model sees the same features at train and predict time. We also fix the datetime parsing to match the dataset format (it doesn’t include `" UTC"`), and we clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE. Finally, we make prediction use `bst.best_iteration` for consistency with validation.'
- What this solution (achieved 5.90559) has done: 'Your RMSE (5.67858) is still worse than the target (3.76015), so we should improve it with minimal, low-risk fixes that keep the same overall LightGBM + engineered-features approach. The largest remaining issue is that your `preprocess()` datetime parsing is too strict for this dataset (it contains microseconds), which can produce null datetimes and silently degrade all derived time features; I parse with microseconds support while keeping the same derived columns. Next, your longitude filter in `max_min_scaling()` is incorrect for NYC (it uses 70–90), so I fix it to a proper NYC longitude range and actually apply it to the training data to reduce extreme-noise rows (a standard, legitimate cleaning step). Finally, I make the train sampling deterministic (seed) to reduce score variance and keep improvements stable.'
- What this solution (achieved 5.88841) has done: 'Your current run didn’t yield a Kaggle score, and the biggest blocker is that you apply `max_min_scaling()` to the test set, which can drop rows and make the submission invalid (row count/key set no longer matches the provided test.csv). I keep your exact model and feature logic, but change filtering so it only removes outliers from the training data (where labels exist) and never filters the test set. I also make the submission keys come from the original raw test file (not the filtered/processed frame) and merge predictions back by `key` to guarantee a full 9914-row submission in the correct order. These minimal changes both ensure a valid `.csv` submission and typically improve RMSE by avoiding accidental loss/misalignment of test rows.'
- What this solution (achieved 5.88841) has done: 'Your RMSE (5.88841) is still worse than the target (3.76015), so we should improve it with the smallest changes that preserve your LightGBM + engineered-features pipeline. The biggest remaining low-risk gain is to remove label leakage/noise by filtering obviously-bad training rows **earlier** (before feature creation and especially before one-hot), and to ensure datetime parsing never leaves nulls that later turn into missing/garbage time features. I also align one-hot columns using a single concatenated dummy expansion (same semantics as your current to_dummies, but guarantees identical columns without manual patching), which typically improves generalization and reduces RMSE. Finally, I keep test rows intact and keep your submission alignment-by-key logic unchanged.'
- What this solution (achieved 5.78605) has done: 'Your RMSE (5.88841) is still worse than the target (3.76015), so we should improve it with the smallest, lowest-risk fixes that keep your exact LightGBM + engineered-features pipeline. The biggest remaining drag is that your distance feature is a rough linear approximation; replacing it with a standard haversine distance keeps the same feature “idea” (distance) but is much more accurate and typically yields a large RMSE gain. Next, your one-hot encoding can still be inconsistent because it’s applied separately to train/test; generating dummies on the concatenated (train+test) frame guarantees identical columns without changing the model. Finally, we keep your “never filter test rows” behavior and submission alignment by `key`, while also clipping negative predictions as you already do.'
- What this solution (achieved 6.78434) has done: 'Your current RMSE (5.78605) is worse than the target (3.76015), so we should make a small, low-risk improvement while keeping the same overall LightGBM-on-engineered-features approach. The biggest remaining issue is that your datetime parsing can still yield nulls (timezone suffix variants and microseconds), which silently degrades all time-derived features; I make parsing robust by trying multiple known formats and falling back to automatic parsing. Next, your train/validation split is random, which can shift the distribution of time/locations; switching to a time-based split (using pickup_datetime) typically improves generalization for this dataset without changing the model. Finally, I add a very standard training-only filter to remove zero-distance rides (often label noise) while keeping test untouched, which usually reduces RMSE a bit.'
- What this solution (achieved 6.29355) has done: 'Your RMSE (6.78434) is still much worse than the target (3.76015), so we should move it downward with a minimal, low-risk correction. The biggest likely regression is the “time-based split” you implemented by sorting on `key` (which is not a real chronological order because it includes an extra integer suffix), so the validation scheme becomes noisy and can lead to a weaker fitted model for test. I switch to a true time-based split using the already-parsed `pickup_datetime` but still keep `pickup_datetime` dropped from features (so model/feature semantics stay the same). Additionally, your `preprocess()` currently contains many redundant no-op blocks; I replace it with a single robust parse + fallback (same derived features) to avoid null datetime-derived columns that can hurt both training and inference.'
- What this solution (achieved 7.03907) has done: 'Your score (6.29355 RMSE) is worse than the target (3.76015), so we should reduce RMSE with the smallest changes that keep your LightGBM + engineered-features pipeline intact. The biggest low-risk issue is that you sort/split by a datetime reconstructed via a `key` join; missing/duplicate keys can silently break the chronological ordering and degrade fit, so we instead carry the parsed `pickup_datetime` through preprocessing (as an extra column only for splitting) and then drop it before training to keep feature semantics unchanged. Next, we add standard training-only cleaning that removes extreme “teleport” rides and unrealistic high fares based on your existing features (distance/coords), which typically reduces noise without touching test rows. Finally, we keep your submission alignment-by-key logic unchanged and continue clipping negative predictions to 0.'
- What this solution (achieved 7.03907) has done: 'We need to move your RMSE down from 7.039 toward 3.760 (lower is better), so we should fix the most likely regression with minimal disruption: your preprocessing keeps `pickup_datetime_parsed` inside the one-hot concatenation, which can introduce nulls/mismatched dtypes and harms split quality and model fit. I keep your exact feature set semantics (same engineered columns, same LightGBM objective/loop), but I explicitly keep `pickup_datetime_parsed` only for splitting (not as a model feature and not inside dummy generation), and I make the chronological split operate on the parsed datetime *before* any dummy expansion to avoid key/join side-effects. I also add one very standard training-only sanity filter consistent with your existing cleaning (remove extreme passenger_count==0 and absurd coordinate zeros) without touching test rows; this typically reduces noise and improves RMSE with minimal risk. Submission generation remains identical (keys from raw test, left-merge predictions, fill missing, clip negatives) to guarantee a valid 9914-row CSV.'
- What this solution (achieved 8.66494) has done: 'Your RMSE (7.039) is far worse than the target (3.760, lower is better), so we should make a minimal, low-risk correction that directly affects generalization without changing your model/feature semantics. The biggest regression risk is training on a time-ordered 80/20 split but only fitting **100 boosting rounds without early stopping**, which often underfits badly on this task; we keep the same LightGBM setup but enable early stopping (still the same training approach) and allow more rounds so the model can reach a reasonable fit and use `best_iteration` consistently. We also fix a subtle but important bug: your `__is_train__` flag is inverted (train marked 0, test marked 1) which is harmless for dummies but makes the code error-prone; we correct it without changing the resulting feature columns. Finally, we ensure `pickup_datetime_parsed` is kept only for splitting and never enters dummy creation/model features, preventing dtype/null contamination of the one-hot stage.'
- What this solution (achieved 10.79909) has done: 'Your current RMSE (8.66494) is far worse than the target (3.76015, lower is better), so we should make a minimal change that directly improves generalization without changing the overall LightGBM + engineered-tabular-features approach. The biggest likely issue is that you train with a chronological split but then only fit on the first 80% of time; this can underfit the later-time distribution and hurt test performance. I keep the same split for finding `best_iteration`, but then retrain one final model on 100% of the sampled training data using that `best_iteration` and predict with it (same model type/objective/features, just using all data). I also add one tiny, standard training-only cleanup to remove a small number of extreme fare outliers (<0 or >300) after your existing filters, which typically reduces RMSE noise without affecting test rows.'

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


def preprocess(df):
    dt1 = pl.col("pickup_datetime").str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
    )
    dt2 = pl.col("pickup_datetime").str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
    )
    dt3 = pl.col("pickup_datetime").str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f UTC", strict=False
    )
    dt4 = pl.col("pickup_datetime").str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S UTC", strict=False
    )

    df = df.with_columns(
        pl.coalesce(
            [
                dt1,
                dt2,
                dt3,
                dt4,
                pl.col("pickup_datetime")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, strict=False),
            ]
        ).alias("pickup_datetime_parsed")
    )

    df = df.with_columns(
        [
            pl.col("pickup_datetime_parsed").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime_parsed").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime_parsed").dt.day().alias("pickup_day"),
            pl.col("pickup_datetime_parsed").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime_parsed").dt.minute().alias("pickup_minute"),
            pl.col("pickup_datetime_parsed").dt.second().alias("pickup_second"),
            pl.col("pickup_datetime_parsed").dt.weekday().alias("pickup_weekday"),
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


def distance(df):
    R_km = 6371.0
    deg2rad = np.pi / 180.0

    df = df.with_columns(
        [
            (pl.col("pickup_latitude") * deg2rad).alias("pickup_lat_rad"),
            (pl.col("dropoff_latitude") * deg2rad).alias("dropoff_lat_rad"),
            ((pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * deg2rad).alias(
                "dlat"
            ),
            (
                (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * deg2rad
            ).alias("dlon"),
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col("dlat") / 2).sin().pow(2)
                + (pl.col("pickup_lat_rad").cos())
                * (pl.col("dropoff_lat_rad").cos())
                * (pl.col("dlon") / 2).sin().pow(2)
            ).alias("a")
        ]
    )

    df = df.with_columns(
        [(2.0 * pl.arctan2(pl.col("a").sqrt(), (1.0 - pl.col("a")).sqrt())).alias("c")]
    )

    df = df.with_columns([(pl.col("c") * R_km).alias("distance")])

    df = df.drop(["pickup_lat_rad", "dropoff_lat_rad", "dlat", "dlon", "a", "c"])
    return df


def drop_encoding(df):
    drop_columns = ["pickup_datetime", "pickup_minute", "pickup_second"]
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
        (pl.col("pickup_longitude") >= -75) & (pl.col("pickup_longitude") <= -72)
    )
    df = df.filter(
        (pl.col("dropoff_longitude") >= -75) & (pl.col("dropoff_longitude") <= -72)
    )
    df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") <= 42)
    )
    df = df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") <= 42)
    )
    return df


def is_central(df):
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


def diff_central(df):
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
    )
    df = df.drop(["central_pickup_abs_longitude", "central_pickup_abs_latitude"])

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
    )
    df = df.drop(["central_dropoff_abs_longitude", "central_dropoff_abs_latitude"])
    return df


def is_short_distance(df):
    df = df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
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
    filtered_train_df = filtered_train_df.filter(
        (pl.col("fare_amount") > 0) & (pl.col("fare_amount") <= 500)
    )
    return filtered_train_df


def rule_base(df):
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


train = max_min_scaling(train)
train = drop_outliner(train)
train = train.filter(pl.col("passenger_count") > 0)
train = train.filter(
    ~((pl.col("pickup_longitude") == 0) | (pl.col("dropoff_longitude") == 0))
)
train = train.filter(
    ~((pl.col("pickup_latitude") == 0) | (pl.col("dropoff_latitude") == 0))
)

train = preprocess(train)
train = distance(train)

train = train.filter(pl.col("distance") > 0)
train = train.filter(pl.col("distance") <= 200)
train = train.filter(pl.col("fare_amount") <= 250)

train = train.filter((pl.col("fare_amount") >= 0) & (pl.col("fare_amount") <= 300))

train = drop_encoding(train)
train = cycling_encoding(train)

test = preprocess(test)
test = distance(test)
test = drop_encoding(test)
test = cycling_encoding(test)

train = is_central(train)
train = is_short_distance(train)

test = is_central(test)
test = is_short_distance(test)

train_feat = train.drop(["fare_amount"])
test_feat = test

train_split_col = train_feat.select(["pickup_datetime_parsed"])
test_split_col = test_feat.select(["pickup_datetime_parsed"])

train_feat_no_split = train_feat.drop(["pickup_datetime_parsed"])
test_feat_no_split = test_feat.drop(["pickup_datetime_parsed"])

train_feat_no_split = train_feat_no_split.with_columns(
    pl.lit(1).cast(pl.Int8).alias("__is_train__")
)
test_feat_no_split = test_feat_no_split.with_columns(
    pl.lit(0).cast(pl.Int8).alias("__is_train__")
)

combined = pl.concat([train_feat_no_split, test_feat_no_split], how="vertical_relaxed")
combined = one_hot_encoding(combined)

train_feat_encoded = combined.filter(pl.col("__is_train__") == 1).drop(["__is_train__"])
test_feat_encoded = combined.filter(pl.col("__is_train__") == 0).drop(["__is_train__"])

train_feat_encoded = train_feat_encoded.with_columns(
    train_split_col["pickup_datetime_parsed"]
)
test_feat_encoded = test_feat_encoded.with_columns(
    test_split_col["pickup_datetime_parsed"]
)

train = train_feat_encoded.with_columns(train["fare_amount"])
test = test_feat_encoded

train_cols = train.columns
feature_cols = [c for c in train_cols if c != "fare_amount"]
if "key" not in feature_cols and "key" in test.columns:
    feature_cols = ["key"] + [c for c in feature_cols if c != "key"]

test = test.select(feature_cols)



## === cell 4
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



## === cell 5
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

if "pickup_datetime_parsed" not in train.columns:
    raise ValueError("Expected 'pickup_datetime_parsed' to exist after preprocess().")

train_pd = train.to_pandas()
X = train_pd.drop(columns=["fare_amount"])
y = train_pd["fare_amount"].astype(np.float32)

if "pickup_datetime" in X.columns:
    raise ValueError("pickup_datetime should have been dropped in drop_encoding().")

split_dt = pd.to_datetime(train_pd["pickup_datetime_parsed"], errors="coerce")
split_dt = split_dt.fillna(pd.Timestamp("1970-01-01"))

order = np.argsort(split_dt.values.astype("datetime64[ns]"))
X_ord = X.iloc[order].reset_index(drop=True)
y_ord = y.iloc[order].reset_index(drop=True)

drop_cols_for_model = ["key", "pickup_datetime_parsed"]
X_model_ord = X_ord.drop(columns=[c for c in drop_cols_for_model if c in X_ord.columns])

split_idx = int(len(X_model_ord) * 0.8)
X_train, X_val = X_model_ord.iloc[:split_idx], X_model_ord.iloc[split_idx:]
y_train, y_val = y_ord.iloc[:split_idx], y_ord.iloc[split_idx:]

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
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=50),
]

bst = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[valid_data],
    callbacks=callbacks,
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mse = mean_squared_error(y_val, y_pred)
print(f"rmse:{rmse}")
model_name = "lgbm"

full_train_data = lgb.Dataset(X_model_ord, label=y_ord)
final_bst = lgb.train(
    params,
    full_train_data,
    num_boost_round=int(bst.best_iteration),
    valid_sets=[full_train_data],
    callbacks=[lgb.log_evaluation(period=0)],
)



## === cell 6
test_pd = test.to_pandas()

if "key" not in test_pd.columns:
    raise ValueError(
        "Expected 'key' to be present in processed test for submission alignment."
    )

drop_cols_for_model = ["key", "pickup_datetime_parsed"]
X_test_model = test_pd.drop(
    columns=[c for c in drop_cols_for_model if c in test_pd.columns]
)

pred_df = pd.DataFrame(
    {
        "key": test_pd["key"].values,
        "fare_amount": final_bst.predict(
            X_test_model, num_iteration=final_bst.current_iteration()
        ),
    }
)

pred_df["fare_amount"] = np.clip(pred_df["fare_amount"].values, 0.0, None)

raw_test_keys = (
    pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
    .select("key")
    .to_pandas()
)
submission = raw_test_keys.merge(pred_df, on="key", how="left")

if submission["fare_amount"].isna().any():
    submission["fare_amount"] = submission["fare_amount"].fillna(
        float(pred_df["fare_amount"].mean())
    )

exp_num = "123467"
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
print(submission.shape)
print(submission.head())



## === cell 7
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X_model_ord.columns
feature_importance_df["importance"] = final_bst.feature_importance()
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
