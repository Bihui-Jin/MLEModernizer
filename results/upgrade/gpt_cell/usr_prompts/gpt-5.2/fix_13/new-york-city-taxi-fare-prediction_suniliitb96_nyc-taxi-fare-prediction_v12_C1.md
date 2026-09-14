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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

3.93524

# 6. Current score

5.90323

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.83413) has done: 'Diagnosis: The crash occurs in cell 31 because the installed `xgboost` version’s `Booster.predict()` does not accept the `ntree_limit` keyword argument (API mismatch across xgboost versions). The code currently branches into a path that always supplies `ntree_limit` when `best_iteration` exists, triggering `TypeError`. We should keep the same “use best iteration if available” semantics, but pass the correct keyword supported by this version (`iteration_range`) and only fall back to plain `predict()` if neither is usable.

Patch summary: Update only cell 31 to call `model.predict()` using `iteration_range=(0, best_iteration+1)` when `best_iteration` is available, and otherwise call plain `predict()` (no tree limit). This preserves early-stopping behavior without changing the model or features.

Updated cells: cell 31 only (below).

Compatibility notes for cell k+1: `prediction` and `test_key` remain defined exactly as before; cell 32 (`len(test_key)`) is unaffected.

Assumptions: The local `xgboost` build is old enough to not support `ntree_limit` but new enough to support `iteration_range` on `Booster.predict()`; if `best_iteration` is absent/None, using all trees matches the original fallback behavior.'
- What this solution (achieved 5.83413) has done: 'Your current gap to the target RMSE is large (5.83413 vs 3.93524; lower is better), so we need a modest, low-risk improvement without changing the overall approach (XGBoost on engineered features). The biggest score hit in this notebook is that the model is trained on `dense_DBSCAN_trips` and `airport_bound` but those features are never created for `df_holdout` after the later column dropping, causing either misalignment/NaNs or weaker generalization; we ensure the test set has the exact same final feature columns as training. Additionally, the line `df_city_trips = df_train.loc[-airportTripsIds]` is logically wrong (boolean negation) and can silently corrupt diagnostics; we fix it to `~airportTripsIds` without affecting training data. Finally, we add a tiny, metric-consistent postprocess: clip negative fare predictions to 0, which usually reduces RMSE by removing impossible negatives while keeping the same model and loss.'
- What this solution (achieved 5.83413) has done: 'Your current RMSE (5.83413) is worse than the target (3.93524), so we should make a small, low-risk change that tends to improve generalization without changing the model or feature set. The biggest score drag here is that `df_holdout` still contains unrealistic `passenger_count` values (0 or >=7) that were filtered out of training, creating a train/test distribution mismatch; we clip `passenger_count` in the test set into the valid training range [1,6] right after loading. This preserves the same features and XGBoost training loop, but avoids out-of-distribution inputs at inference time, which typically reduces RMSE. Everything else (DBSCAN feature, airport features, training params, and the best-iteration prediction logic) is kept the same, and the script still writes a valid `taxi_fare_submission.csv`.'
- What this solution (achieved 5.83413) has done: 'Your current RMSE (5.83413) is worse than the target (3.93524), so we should make the smallest changes that usually improve this classic NYC Taxi Fare baseline without changing the overall approach (XGBoost regressor on engineered geo/time features). The biggest score drag left is that the model is trained with time-derived features, but the raw `pickup_datetime` parsing uses a fixed `" ... UTC"` format that can silently create `NaT` (or inconsistent parsing) and degrade those features; we parse robustly while preserving the same feature set. Next, we remove a common source of large errors by filtering clearly invalid coordinates in training (zeros/out-of-range), matching typical competition baselines, while keeping the same model/training loop. Finally, we apply the same lightweight coordinate sanity handling to test (clip into valid ranges) to reduce out-of-distribution inputs and stabilize predictions, keeping submission format unchanged.'
- What this solution (achieved 5.83413) has done: 'To move your RMSE down toward the 3.935 target with minimal disruption, I keep the same XGBoost training loop and feature set but fix a key feature mismatch: you train with `dense_DBSCAN_trips` and `airport_bound`, yet `x_pred` currently drops neither and relies on reindexing that can silently zero/NaN mismatched columns. I make the train/test feature drop lists identical by deriving a single `DROP_COLS` list and using it for both, ensuring the model sees the same schema at inference. I also ensure the boolean engineered features are cast to small integers (0/1) consistently in both train and test, which typically improves XGBoost split behavior without changing the core logic. Finally, I keep your best-iteration prediction path and the non-negative clipping to avoid impossible negative fares.'
- What this solution (achieved 5.73736) has done: 'Your current RMSE (5.83413) is worse than the target (3.93524), so we should make a small, low-risk change that typically improves this exact NYC Taxi Fare baseline without changing your model/training loop: add a couple of standard, geometry-consistent features (straight-line “manhattan” distance and absolute lat/lon deltas) derived from the same pickup/dropoff coordinates you already use. This preserves the overall approach (XGBoost on engineered geo/time features) and doesn’t alter the learning procedure, but it usually reduces large errors because XGBoost can model piecewise relationships on these simple deltas more easily than on haversine alone. To keep schema consistent, the same new features are created for both train and test before DBSCAN/airport processing, and the existing DROP_COLS logic remains unchanged. Submission writing is kept identical and still produces `taxi_fare_submission.csv`.'
- What this solution (achieved 5.90323) has done: 'The main timeout driver is the two DBSCAN fits on ~500k+ rows with a BallTree haversine metric; we keep DBSCAN exactly but make it faster by feeding it the correct (lat, lon) order, converting to radians once, and using contiguous float64 arrays to avoid repeated pandas slicing/copying. The next largest cost is pandas row-wise `apply` for `trip_rate`; we replace it with a fully vectorized identical computation. We also remove heavy plotting/statistics cells that don’t affect the final model or predictions, and we reduce repeated dataframe filtering passes by combining masks (same semantics) to cut overhead. All changes preserve the same features, training approach, model, and evaluation behavior, with only negligible floating-point differences.'
- What this solution (achieved 5.90323) has done: 'To move your RMSE down toward the 3.935 target with minimal disruption, I keep your exact XGBoost training loop and current feature set, but fix a key train/test mismatch: you clip invalid coordinates only in the test set, while training still contains borderline/out-of-range coordinates that can create noisy splits and worsen generalization. I apply the same coordinate sanity filtering/clipping to the training data (consistent with your existing bounding-box logic) and also clip the test data into that same NYC bounding box to avoid out-of-distribution inference values. These are small, metric-aligned data-quality tweaks that typically improve this competition baseline without changing the model architecture, loss, or training semantics. The script still run end-to-end and write `taxi_fare_submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)



## === cell 1
MAX_TRAINING_SIZE = 500_000

EPS_IN_KM = 0.5  ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

RADIUS_VICINITY_AIRPORTS = 1.0

THERSHOLD_TRIP_FARE_RATE = 50.0

THRESHOLD_TRIP_DISTANCE = 25.0



## === cell 2
import os
import math
import timeit
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import xgboost as xgb

from sklearn import metrics  # noqa: F401

np.random.seed(0)
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    from haversine import haversine  # type: ignore
except Exception:
    haversine = None



## === cell 3
start_time = timeit.default_timer()

df_train = pd.read_csv(
    "../input/train.csv", nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_holdout = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_key = df_holdout["key"]
df_train.drop(columns=["key"], inplace=True)
df_holdout.drop(columns=["key"], inplace=True)

df_holdout["passenger_count"] = df_holdout["passenger_count"].clip(lower=1, upper=6)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 4
print("Old size: %d" % len(df_train))

df_train = df_train[df_train.fare_amount >= 0]
df_train = df_train.dropna(how="any", axis="rows")

pc = df_train["passenger_count"]
df_train = df_train[(pc > 0) & (pc < 7)]

print("New size: %d" % len(df_train))




## === cell 5
def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


BB = (-74.5, -72.8, 40.5, 41.8)

print("Old size: %d" % len(df_train))

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

coords = df_train[coord_cols]
mask_notnull = coords.notnull().all(axis=1)
mask_nonzero = (
    (df_train["pickup_longitude"].abs() > 1e-6)
    & (df_train["dropoff_longitude"].abs() > 1e-6)
    & (df_train["pickup_latitude"].abs() > 1e-6)
    & (df_train["dropoff_latitude"].abs() > 1e-6)
)

mask_valid_ranges = (
    df_train["pickup_latitude"].between(-90, 90)
    & df_train["dropoff_latitude"].between(-90, 90)
    & df_train["pickup_longitude"].between(-180, 180)
    & df_train["dropoff_longitude"].between(-180, 180)
)

df_train = df_train[mask_notnull & mask_nonzero & mask_valid_ranges]
df_train = df_train[select_within_boundingbox(df_train, BB)]

print("New size: %d" % len(df_train))



## === cell 6
if haversine is None:

    def haversine(point1, point2):
        lat1, lon1 = point1
        lat2, lon2 = point2
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return KMS_PER_RADIAN * c




## === cell 7
def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return KMS_PER_RADIAN * c


def add_simple_geo_features(df):
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["manhattan_dist"] = df["abs_lon_diff"] + df["abs_lat_diff"]
    return df


def addPickDropDistanceFeature(df):
    df["trip_distance"] = haversine_vectorized(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def addAirportDistanceFeatures(df):
    df["pickup_distance_to_jfk"] = haversine_vectorized(
        df["pickup_latitude"],
        df["pickup_longitude"],
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )
    df["drop_distance_to_jfk"] = haversine_vectorized(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )

    df["pickup_distance_to_lgr"] = haversine_vectorized(
        df["pickup_latitude"],
        df["pickup_longitude"],
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )
    df["drop_distance_to_lgr"] = haversine_vectorized(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )

    df["pickup_distance_to_ewr"] = haversine_vectorized(
        df["pickup_latitude"],
        df["pickup_longitude"],
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    df["drop_distance_to_ewr"] = haversine_vectorized(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    return df


def getAirportTrips(df, airportVicinity):
    ids = (
        (df.pickup_distance_to_jfk < airportVicinity)
        | (df.drop_distance_to_jfk < airportVicinity)
        | (df.pickup_distance_to_lgr < airportVicinity)
        | (df.drop_distance_to_lgr < airportVicinity)
        | (df.pickup_distance_to_ewr < airportVicinity)
        | (df.drop_distance_to_ewr < airportVicinity)
    )
    return ids


start_time = timeit.default_timer()

df_train = add_simple_geo_features(df_train)
df_holdout = add_simple_geo_features(df_holdout)

df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 8
bucketsCount = 100
feat = "trip_distance"
_ = bucketsCount, feat



## === cell 9
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE]
print("New size: %d" % len(df_train))




## === cell 10
def add_datetime_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )

    df["hour"] = df.pickup_datetime.dt.hour
    df["day"] = df.pickup_datetime.dt.day
    df["month"] = df.pickup_datetime.dt.month
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["year"] = df.pickup_datetime.dt.year

    return df


start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_holdout = add_datetime_features(df_holdout)

df_train = df_train.dropna(subset=["pickup_datetime"])

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 11
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)



## === cell 12
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 13
start_time = timeit.default_timer()

pickup_latlon_rad = np.radians(
    df_nyc_taxi[["pickup_latitude", "pickup_longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )
)
pickup_latlon_rad = np.ascontiguousarray(pickup_latlon_rad)

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(pickup_latlon_rad)
labels_pick = dbscan_pick.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 14
start_time = timeit.default_timer()

drop_latlon_rad = np.radians(
    df_nyc_taxi[["dropoff_latitude", "dropoff_longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )
)
drop_latlon_rad = np.ascontiguousarray(drop_latlon_rad)

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(drop_latlon_rad)
labels_drop = dbscan_drop.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 15
df_nyc_taxi["dense_DBSCAN_trips"] = (labels_pick != -1) & (labels_drop != -1)



## === cell 16
_ = df_nyc_taxi.loc[
    df_nyc_taxi.dense_DBSCAN_trips == 1, ["pickup_longitude", "pickup_latitude"]
]



## === cell 17
df_train = df_nyc_taxi.iloc[:train_len, :]
df_holdout = df_nyc_taxi.iloc[train_len:, :].iloc[
    :, df_nyc_taxi.columns != "fare_amount"
]

(len(df_train), len(df_holdout))



## === cell 18
df_train.loc[df_train.trip_distance < 0.2, "trip_distance"] = 0.2

_ = bucketsCount



## === cell 19
df_train["trip_rate"] = df_train["fare_amount"].to_numpy(dtype=np.float64) / df_train[
    "trip_distance"
].to_numpy(dtype=np.float64)



## === cell 20
len(df_train.loc[df_train.trip_rate > THERSHOLD_TRIP_FARE_RATE])



## === cell 21
ids = df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train[ids]
print("New size: %d" % len(df_train))



## === cell 22
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 23
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[~airportTripsIds]



## === cell 24
_ = df_airport_trips, df_city_trips



## === cell 25
_ = df_train



## === cell 26
for _df in (df_train, df_holdout):
    if "dense_DBSCAN_trips" in _df.columns:
        _df["dense_DBSCAN_trips"] = _df["dense_DBSCAN_trips"].astype(np.int8)
    if "airport_bound" in _df.columns:
        _df["airport_bound"] = _df["airport_bound"].astype(np.int8)

DROP_COLS = [
    "pickup_datetime",
    "pickup_distance_to_jfk",
    "drop_distance_to_jfk",
    "pickup_distance_to_lgr",
    "drop_distance_to_lgr",
    "pickup_distance_to_ewr",
    "drop_distance_to_ewr",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_rate",
]

df_train = df_train.drop(columns=DROP_COLS)
df_train.info()



## === cell 27
y = df_train["fare_amount"]
train = df_train.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.01
)



## === cell 28
params = {
    "max_depth": 8,  # Result of tuning with CV
    "eta": 0.03,  # Result of tuning with CV
    "subsample": 1,  # Result of tuning with CV
    "colsample_bytree": 0.8,  # Result of tuning with CV
    "objective": "reg:linear",
    "eval_metric": "rmse",
    "silent": 1,
    "nthread": max(1, os.cpu_count() or 1),
    "seed": 0,
}

CV = False
if CV:
    dtrain = xgb.DMatrix(train, label=y)
    gridsearch_params = [(eta) for eta in np.arange(0.04, 0.12, 0.02)]

    min_rmse = float("Inf")
    best_params = None
    for eta in gridsearch_params:
        print("CV with eta={} ".format(eta))

        params["eta"] = eta

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={"rmse"},
            early_stopping_rounds=10,
        )

        mean_rmse = cv_results["test-rmse-mean"].min()
        boost_rounds = cv_results["test-rmse-mean"].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = eta

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    params["silent"] = 0  # Turn on output
    print(params)




## === cell 29
def XGBmodel(x_train, x_test, y_train, y_test, params):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test, params)



## === cell 30
df_holdout["pickup_latitude"] = (
    df_holdout["pickup_latitude"].clip(BB[2], BB[3]).clip(-90, 90)
)
df_holdout["dropoff_latitude"] = (
    df_holdout["dropoff_latitude"].clip(BB[2], BB[3]).clip(-90, 90)
)
df_holdout["pickup_longitude"] = (
    df_holdout["pickup_longitude"].clip(BB[0], BB[1]).clip(-180, 180)
)
df_holdout["dropoff_longitude"] = (
    df_holdout["dropoff_longitude"].clip(BB[0], BB[1]).clip(-180, 180)
)

x_pred = df_holdout.drop(columns=[c for c in DROP_COLS if c in df_holdout.columns])

x_pred = x_pred.reindex(columns=train.columns, fill_value=0)

dtest = xgb.DMatrix(x_pred)

if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(
        dtest, iteration_range=(0, int(model.best_iteration) + 1)
    )
else:
    prediction = model.predict(dtest)

prediction = np.clip(prediction, 0, None)



## === cell 31
len(test_key)



## === cell 32
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(prediction, 2)})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
