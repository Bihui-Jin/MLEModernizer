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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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

4.19963

# 6. Current score

5.83536

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.50764) has done: 'I fix the submission generation so it always creates a valid `.csv` with the correct `key` values and filename extension, since your current code overwrites `key` with row indices and writes a non-CSV filename. To move RMSE toward the target, I keep your XGBoost setup but add the standard NYC taxi distance feature (haversine) using the same lat/long inputs; this is minimal feature engineering that doesn’t change the model type or training approach. I also ensure train/test columns align exactly (same feature set, same order) and clip negative fare predictions to 0 for stability (a safe post-process for a nonnegative target). These changes are directly tied to producing a valid submission and improving score without altering the core modeling logic.'
- What this solution (achieved 6.56179) has done: 'Your current RMSE (7.50764) is worse than the target (4.19963), so we need a modest, safe improvement without changing your core model/training approach. The biggest low-risk gain here is adding the standard datetime-derived features (hour, day-of-week, month, year) from `pickup_datetime`, because fare strongly depends on time patterns and this preserves your XGBoost pipeline. I also apply the same geographic bounds filtering to the test set (with row-preserving clipping rather than dropping) to avoid out-of-distribution coordinates at prediction time. Finally, I fix a performance bug: you are using `early_stopping_rounds` (disallowed by your requirements), so I remove it and keep `num_boost_round` unchanged to preserve the intended training procedure while making it compliant.'
- What this solution (achieved 7.14307) has done: 'Your current RMSE (6.56179) is worse than the target (4.19963), so we should make a small, low-risk improvement without changing the model type or training loop. The biggest gain for NYC taxi fares while preserving your pipeline is adding a few standard location-based features derived from the same inputs: straight-line deltas, manhattan distance in degrees, and a simple bearing. I also make the geographic bounds handling consistent by filtering out-of-bounds rows from the training sample (as you already do) and clipping the test rows (row-preserving) exactly as you already intended, while ensuring no NaNs remain after datetime parsing. These changes keep XGBoost and the same training procedure intact, but typically move RMSE materially toward your target on this competition.'
- What this solution (achieved 5.30689) has done: 'Your current RMSE (7.14307) is far worse than the target (4.19963), so we need a meaningful but still core-logic-preserving improvement. The most direct issue is that training uses only 400k rows while the dataset is huge; increasing the training sample (within the 600s limit) usually yields a large RMSE drop without changing your model type, loss, or training loop. I also apply the same coordinate clipping you already do for test to the training data (row-preserving, no extra filtering), and remove a small set of obvious outliers using a distance/fare sanity rule to reduce noise without altering the modeling approach. Finally, I ensure all engineered features are finite (no NaN/inf), keeping train/test columns perfectly aligned, and still write a valid `my_submission.csv`.'
- What this solution (achieved 5.63191) has done: 'I fix the runtime error in `haversine_np` so it works with both NumPy arrays and scalar airport coordinates, which is what currently stops the pipeline before training and submission. I also ensure that the same feature engineering runs for both train and test, that NaNs/Infs are handled safely, and that `X_test` is aligned to `X` without dropping any test rows. These changes are score-neutral in intent except that they allow the intended airport-distance features to be computed correctly (which should improve RMSE versus a broken run). Finally, I make sure a valid `my_submission.csv` with the required `key,fare_amount` columns is always written.'
- What this solution (achieved 5.54086) has done: 'Your current RMSE (5.63191) is still far above the target (4.19963), so we need a small, legitimate improvement without changing the model type or training loop. The biggest low-risk gain here is to add the standard “center” coordinate features (midpoint lat/lon) plus distances to NYC landmarks for both pickup and dropoff; this keeps the same inputs (lat/lon) and the same XGBoost training, but gives the model better structure to learn. I also make the train/test cleaning fully symmetric for non-finite engineered features (fill in test with train medians after feature creation) to avoid silent distribution mismatches. Finally, I keep the submission creation unchanged in format but add an assertion to guarantee row alignment.'
- What this solution (achieved 5.61722) has done: 'Your RMSE (5.54086) is still worse than the target (4.19963), so we should make a small, legitimate improvement without changing your model type, objective, or training loop. The biggest low-risk gain for this competition is to add a proper trip distance feature in kilometers (you currently only use raw degree deltas plus engineered landmark distances), and to add a simple “night/peak” time indicator to help the model learn fare structure without altering the approach. To keep train/test semantics aligned and stable, we compute these features for both sets, keep the same filtering/cleaning logic, and continue to write the same `my_submission.csv` format. All other choices (XGBoost params, num_boost_round, clipping negatives, submission schema) remain unchanged.'
- What this solution (achieved 5.74023) has done: 'To move RMSE down toward your target with minimal disruption, I keep your exact XGBoost training loop and engineered feature set, but fix two high-impact data issues: (1) train/test time features are currently computed in UTC, while this dataset’s fares align better with NYC local time patterns, and (2) the model benefits from a simple, standard geodesic-to-fare linearization by adding `log1p` transforms of your distance features (without changing the model type or objective). I also make the median fill map strictly feature-aligned (train medians computed on the exact numeric feature columns used for training) to avoid subtle train/test mismatch. These are small, legitimate changes that typically reduce RMSE materially on this competition while preserving your overall approach and still producing a valid `my_submission.csv`.'
- What this solution (achieved 5.65166) has done: 'Your current RMSE (5.74023) is still well above the target (4.19963), so we should make a small, legitimate improvement without changing the core XGBoost training setup. The biggest low-risk gain is to align training with how fares are charged by adding a simple “NYC meter” baseline feature (2.5 + 1.56 * haversine_km) and a few interaction terms between distance and time-of-day/weekend, while keeping the model/loop/objective identical. I also remove a subtle mismatch by filling NaNs in both train and test using the same median map (currently only test is median-filled, while train rows are dropped), which usually improves generalization without changing semantics. Finally, I keep submission creation identical but add a strict feature alignment check to guarantee no silent column drift.'
- What this solution (achieved 5.72802) has done: 'Your RMSE (5.65166) is still above the target (4.19963), so we should make a small, legitimate improvement that preserves your XGBoost training loop and overall feature approach. The biggest low-risk gain here is correcting a subtle but important bug in `add_trip_km_features`: it currently treats radians as if they were kilometers, which corrupts several distance-derived features and harms RMSE; fixing this keeps the same feature intent but makes the values physically meaningful. I also add one standard, minimal geo feature (`euclidean_km`) derived from the same already-computed km deltas, and include its log1p transform in your existing log-distance block, which typically helps the model fit fare curvature without changing the model or objective. Everything else (data loading, filters, params, num_boost_round, submission writing) remains the same to keep changes minimal and stable.'
- What this solution (achieved 5.82487) has done: 'Your current RMSE (5.72802) is still above the target (4.19963), so we need a small, legitimate improvement that keeps your exact XGBoost training loop and feature set intact. The biggest low-risk gain without changing model/loop is to make the training data quality closer to what the metric rewards by removing clearly invalid coordinates and zero-distance/high-fare anomalies more consistently (these are label-noise/outlier removals, not new modeling). I also add one standard, minimal passenger_count sanitization (cap to [1,6]) applied symmetrically to train/test to reduce extreme leverage points without dropping test rows. Finally, I keep your submission generation unchanged but add a strict numeric dtype enforcement for XGBoost inputs to avoid any silent object dtypes hurting fit.'
- What this solution (achieved 5.83536) has done: 'Your current RMSE (5.82487) is worse than the target (4.19963), so we need a modest, low-risk improvement while keeping the same XGBoost training loop and feature pipeline. The biggest minimal change with high impact here is to make the train/valid split “time-aware” by splitting on `pickup_datetime` rather than randomly; this matches the competition’s temporal generalization better and typically improves public LB RMSE without changing model architecture or loss. I also remove a small source of label noise by filtering clearly invalid `passenger_count` > 6 in training (you already cap to 6 later, but those rows still carry noisy labels/behavior). Finally, I keep submission creation identical and add a deterministic sort-by-time before feature engineering so the split is stable and reproducible.'
- What this solution (achieved 5.83536) has done: 'Your current RMSE (5.83536) is worse than the target (4.19963), so we need a cautious, minimal improvement without changing your XGBoost training loop or the existing feature set. The biggest low-risk issue in your code is that you sort `df_test` by `pickup_datetime` and then write predictions paired to that sorted order, which can misalign predictions with the original test `key` order expected by Kaggle; fixing this preserves identical modeling but corrects submission row alignment. I keep your exact feature engineering and training setup, but compute features in sorted order for stability while saving/restoring the original test row order before writing `my_submission.csv`. This should move the score down (better) toward the target without altering the model or adding new features.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_NROWS = 2000000

df_train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=TRAIN_NROWS
)
df_train.head()



## === cell 2
df_train.shape



## === cell 3
df_test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df_test.head()



## === cell 4
df_train.info()
df_test.info()



## === cell 5
df_train.describe()



## === cell 6
df_test.describe()



## === cell 7
df_train.drop(df_train[df_train["passenger_count"] > 6].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["passenger_count"] > 5].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 8
df_train.drop(df_train[df_train["fare_amount"] < 3].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["fare_amount"] > 200].index, axis=0, inplace=True)



## === cell 9
df_train.dropna(inplace=True)

df_train.drop(
    df_train.index[
        (df_train.pickup_longitude < -75.0)
        | (df_train.pickup_longitude > -72.0)
        | (df_train.pickup_latitude < 40.0)
        | (df_train.pickup_latitude > 42.0)
    ],
    inplace=True,
)

df_train.drop(
    df_train.index[
        (df_train.dropoff_longitude < -75.0)
        | (df_train.dropoff_longitude > -72.0)
        | (df_train.dropoff_latitude < 40.0)
        | (df_train.dropoff_latitude > 42.0)
    ],
    inplace=True,
)



## === cell 10
df_train.describe()




## === cell 11
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # km


def bearing_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x))


def add_datetime_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True).dt.tz_convert(
        "America/New_York"
    )
    df["pickup_year"] = dt.dt.year.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_day"] = dt.dt.day.astype("float32")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")

    df["is_night"] = ((dt.dt.hour <= 6) | (dt.dt.hour >= 20)).astype("float32")
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
    return df


def add_geo_features(df):
    df["abs_lon_diff"] = (
        (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype("float32")
    )
    df["manhattan_deg"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
    df["bearing"] = bearing_np(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    ).astype("float32")
    return df


def add_airport_features(df):
    JFK = (-73.7781, 40.6413)
    LGA = (-73.8740, 40.7769)
    EWR = (-74.1745, 40.6895)

    df["pickup_to_jfk_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, JFK[0], JFK[1]
    ).astype("float32")
    df["dropoff_to_jfk_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, JFK[0], JFK[1]
    ).astype("float32")

    df["pickup_to_lga_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, LGA[0], LGA[1]
    ).astype("float32")
    df["dropoff_to_lga_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, LGA[0], LGA[1]
    ).astype("float32")

    df["pickup_to_ewr_km"] = haversine_np(
        df["pickup_longitude"].values, df["pickup_latitude"].values, EWR[0], EWR[1]
    ).astype("float32")
    df["dropoff_to_ewr_km"] = haversine_np(
        df["dropoff_longitude"].values, df["dropoff_latitude"].values, EWR[0], EWR[1]
    ).astype("float32")
    return df


def add_center_and_landmark_features(df):
    df["center_longitude"] = (
        (df["pickup_longitude"] + df["dropoff_longitude"]) * 0.5
    ).astype("float32")
    df["center_latitude"] = (
        (df["pickup_latitude"] + df["dropoff_latitude"]) * 0.5
    ).astype("float32")

    TIMES_SQ = (-73.9855, 40.7580)
    EMPIRE = (-73.9857, 40.7484)
    WALL_ST = (-74.0090, 40.7060)
    CENTRAL_PARK = (-73.9654, 40.7829)

    for name, (lon, lat) in {
        "times_sq": TIMES_SQ,
        "empire": EMPIRE,
        "wall_st": WALL_ST,
        "central_park": CENTRAL_PARK,
    }.items():
        df[f"pickup_to_{name}_km"] = haversine_np(
            df["pickup_longitude"].values, df["pickup_latitude"].values, lon, lat
        ).astype("float32")
        df[f"dropoff_to_{name}_km"] = haversine_np(
            df["dropoff_longitude"].values, df["dropoff_latitude"].values, lon, lat
        ).astype("float32")

    for name, (lon, lat) in {
        "times_sq": TIMES_SQ,
        "empire": EMPIRE,
        "wall_st": WALL_ST,
        "central_park": CENTRAL_PARK,
    }.items():
        df[f"center_to_{name}_km"] = haversine_np(
            df["center_longitude"].values, df["center_latitude"].values, lon, lat
        ).astype("float32")

    return df


def add_trip_km_features(df):
    R = 6371.0
    lat1 = np.radians(np.asarray(df["pickup_latitude"].values, dtype="float64"))
    lat2 = np.radians(np.asarray(df["dropoff_latitude"].values, dtype="float64"))
    dlat = lat2 - lat1
    dlon = np.radians(
        np.asarray(
            (df["dropoff_longitude"].values - df["pickup_longitude"].values),
            dtype="float64",
        )
    )

    df["delta_lat_km"] = (R * np.abs(dlat)).astype("float32")
    df["delta_lon_km"] = (R * np.abs(dlon) * np.cos((lat1 + lat2) * 0.5)).astype(
        "float32"
    )
    df["manhattan_km"] = (df["delta_lat_km"] + df["delta_lon_km"]).astype("float32")

    df["euclidean_km"] = np.sqrt(
        df["delta_lat_km"].astype("float32") ** 2
        + df["delta_lon_km"].astype("float32") ** 2
    ).astype("float32")
    return df


def add_log_distance_features(df):
    for c in [
        "haversine_km",
        "manhattan_km",
        "euclidean_km",
        "delta_lat_km",
        "delta_lon_km",
        "pickup_to_jfk_km",
        "dropoff_to_jfk_km",
        "pickup_to_lga_km",
        "dropoff_to_lga_km",
        "pickup_to_ewr_km",
        "dropoff_to_ewr_km",
        "pickup_to_times_sq_km",
        "dropoff_to_times_sq_km",
        "pickup_to_empire_km",
        "dropoff_to_empire_km",
        "pickup_to_wall_st_km",
        "dropoff_to_wall_st_km",
        "pickup_to_central_park_km",
        "dropoff_to_central_park_km",
        "center_to_times_sq_km",
        "center_to_empire_km",
        "center_to_wall_st_km",
        "center_to_central_park_km",
    ]:
        if c in df.columns:
            df[f"log1p_{c}"] = np.log1p(
                np.maximum(df[c].astype("float32"), 0.0)
            ).astype("float32")
    return df


def add_fare_structure_features(df):
    df["meter_base_plus_dist"] = (
        2.5 + 1.56 * df["haversine_km"].astype("float32")
    ).astype("float32")
    df["dist_x_hour"] = (
        df["haversine_km"].astype("float32") * df["pickup_hour"].astype("float32")
    ).astype("float32")
    df["dist_x_weekend"] = (
        df["haversine_km"].astype("float32") * df["is_weekend"].astype("float32")
    ).astype("float32")
    df["dist_x_night"] = (
        df["haversine_km"].astype("float32") * df["is_night"].astype("float32")
    ).astype("float32")
    df["manhattan_x_hour"] = (
        df["manhattan_km"].astype("float32") * df["pickup_hour"].astype("float32")
    ).astype("float32")
    return df


df_train["pickup_datetime"] = pd.to_datetime(
    df_train["pickup_datetime"], errors="coerce", utc=True
)
df_test["pickup_datetime"] = pd.to_datetime(
    df_test["pickup_datetime"], errors="coerce", utc=True
)

df_train = (
    df_train.dropna(subset=["pickup_datetime"])
    .sort_values("pickup_datetime")
    .reset_index(drop=True)
)

df_test["_orig_row_id"] = np.arange(len(df_test), dtype=np.int64)
df_test = df_test.sort_values("pickup_datetime").reset_index(drop=True)

df_train["passenger_count"] = df_train["passenger_count"].clip(1, 6)
df_test["passenger_count"] = df_test["passenger_count"].clip(1, 6)

for col, lo, hi in [
    ("pickup_longitude", -75.0, -72.0),
    ("dropoff_longitude", -75.0, -72.0),
    ("pickup_latitude", 40.0, 42.0),
    ("dropoff_latitude", 40.0, 42.0),
]:
    df_train[col] = df_train[col].clip(lo, hi)
    df_test[col] = df_test[col].clip(lo, hi)

df_train = add_datetime_features(df_train)
df_test = add_datetime_features(df_test)

df_train.dropna(
    subset=[
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_dayofweek",
        "is_night",
        "is_weekend",
    ],
    inplace=True,
)

df_train["haversine_km"] = haversine_np(
    df_train["pickup_longitude"].values,
    df_train["pickup_latitude"].values,
    df_train["dropoff_longitude"].values,
    df_train["dropoff_latitude"].values,
).astype("float32")

df_test["haversine_km"] = haversine_np(
    df_test["pickup_longitude"].values,
    df_test["pickup_latitude"].values,
    df_test["dropoff_longitude"].values,
    df_test["dropoff_latitude"].values,
).astype("float32")

df_train = add_trip_km_features(df_train)
df_test = add_trip_km_features(df_test)

df_train = add_geo_features(df_train)
df_test = add_geo_features(df_test)

df_train = add_airport_features(df_train)
df_test = add_airport_features(df_test)

df_train = add_center_and_landmark_features(df_train)
df_test = add_center_and_landmark_features(df_test)

df_train = add_log_distance_features(df_train)
df_test = add_log_distance_features(df_test)

df_train = add_fare_structure_features(df_train)
df_test = add_fare_structure_features(df_test)

coord_valid = (
    (df_train["pickup_longitude"].between(-75.0, -72.0))
    & (df_train["dropoff_longitude"].between(-75.0, -72.0))
    & (df_train["pickup_latitude"].between(40.0, 42.0))
    & (df_train["dropoff_latitude"].between(40.0, 42.0))
)
df_train = df_train[coord_valid].copy()

df_train = df_train[np.isfinite(df_train["haversine_km"])].copy()
df_train = df_train[
    (df_train["haversine_km"] >= 0.0) & (df_train["haversine_km"] <= 150.0)
].copy()
df_train = df_train[
    (df_train["fare_amount"] >= 2.5) & (df_train["fare_amount"] <= 200.0)
].copy()
df_train = df_train[
    ~((df_train["haversine_km"] < 0.2) & (df_train["fare_amount"] > 60.0))
].copy()
df_train = df_train[
    ~((df_train["haversine_km"] > 50.0) & (df_train["fare_amount"] < 10.0))
].copy()
df_train = df_train[
    ~((df_train["haversine_km"] < 0.05) & (df_train["fare_amount"] > 30.0))
].copy()

num_cols = [c for c in df_train.columns if c not in ["key", "pickup_datetime"]]
df_train[num_cols] = df_train[num_cols].replace([np.inf, -np.inf], np.nan)

num_cols_test = [c for c in df_test.columns if c not in ["key", "pickup_datetime"]]
df_test[num_cols_test] = df_test[num_cols_test].replace([np.inf, -np.inf], np.nan)

X = df_train.drop(["fare_amount", "pickup_datetime", "key"], axis=1)
y = df_train["fare_amount"]

X_test = df_test.drop(["pickup_datetime", "key", "_orig_row_id"], axis=1)

fill_map = X.median(numeric_only=True).to_dict()
X = X.fillna(fill_map)
X_test = X_test.fillna(fill_map)

X_test = X_test[X.columns]
assert list(X_test.columns) == list(X.columns), "Train/test feature columns misaligned."

X = X.astype("float32")
X_test = X_test.astype("float32")
y = y.astype("float32")

split_idx = int(len(X) * 0.8)
train_x, valid_x = X.iloc[:split_idx], X.iloc[split_idx:]
train_y, valid_y = y.iloc[:split_idx], y.iloc[split_idx:]



## === cell 12
train_x.dtypes



## === cell 13
import xgboost as xgb

dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)
dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, "train"), (dvalid, "valid")]

params = {
    "objective": "reg:squarederror",
    "min_child_weight": 1,
    "learning_rate": 0.005,
    "colsample_bytree": 0.7,
    "max_depth": 10,
    "subsample": 0.7,
    "n_jobs": -1,
    "booster": "gbtree",
    "eval_metric": "rmse",
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=700,
    evals=watchlist,
    maximize=False,
    verbose_eval=50,
)



## === cell 14
prediction = model.predict(dtest)
prediction = np.clip(prediction, 0.0, None)



## === cell 15
submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission.head()



## === cell 16
pred_df = pd.DataFrame(
    {
        "_orig_row_id": df_test["_orig_row_id"].values,
        "key": df_test["key"].values,
        "fare_amount": prediction,
    }
).sort_values("_orig_row_id")

output = pred_df[["key", "fare_amount"]].reset_index(drop=True)

assert output.shape[0] == submission.shape[0], "Row mismatch vs sample_submission."
assert output.shape[0] == df_test.shape[0], "Row mismatch between test and predictions."
assert output["key"].isna().sum() == 0, "Missing keys in output."
assert output["fare_amount"].isna().sum() == 0, "Missing predictions in output."

output.to_csv("my_submission.csv", index=False)
print("Wrote my_submission.csv with shape:", output.shape)
print("Columns:", list(output.columns))
print("Any NA fare_amount:", output["fare_amount"].isna().any())
print("Any duplicated keys:", output["key"].duplicated().any())
print("Test rows:", df_test.shape[0], "Pred rows:", output.shape[0])
