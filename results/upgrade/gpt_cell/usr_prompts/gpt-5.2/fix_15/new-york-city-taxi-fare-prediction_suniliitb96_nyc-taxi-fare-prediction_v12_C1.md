# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from sklearn.neighbors import (
    NearestNeighbors,
)  # keep DBSCAN feature but avoid train+test leakage
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

BB = (-74.5, -72.8, 40.5, 41.8)
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



## === cell 12
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 13
start_time = timeit.default_timer()

train_pick_rad = np.radians(
    df_train[["pickup_latitude", "pickup_longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )
)
train_pick_rad = np.ascontiguousarray(train_pick_rad)

train_drop_rad = np.radians(
    df_train[["dropoff_latitude", "dropoff_longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )
)
train_drop_rad = np.ascontiguousarray(train_drop_rad)

test_pick_rad = np.radians(
    df_holdout[["pickup_latitude", "pickup_longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )
)
test_pick_rad = np.ascontiguousarray(test_pick_rad)

test_drop_rad = np.radians(
    df_holdout[["dropoff_latitude", "dropoff_longitude"]].to_numpy(
        dtype=np.float64, copy=True
    )
)
test_drop_rad = np.ascontiguousarray(test_drop_rad)

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(train_pick_rad)
labels_pick_train = dbscan_pick.labels_

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(train_drop_rad)
labels_drop_train = dbscan_drop.labels_

train_pick_dense = labels_pick_train != -1
train_drop_dense = labels_drop_train != -1

nn_pick = NearestNeighbors(
    radius=EPS_IN_RADIAN, algorithm="ball_tree", metric="haversine"
)
nn_pick.fit(train_pick_rad[train_pick_dense])
test_pick_dense = np.zeros(len(df_holdout), dtype=bool)
if train_pick_dense.any():
    neigh = nn_pick.radius_neighbors(test_pick_rad, return_distance=False)
    test_pick_dense = np.fromiter(
        (len(x) > 0 for x in neigh), dtype=bool, count=len(df_holdout)
    )

nn_drop = NearestNeighbors(
    radius=EPS_IN_RADIAN, algorithm="ball_tree", metric="haversine"
)
nn_drop.fit(train_drop_rad[train_drop_dense])
test_drop_dense = np.zeros(len(df_holdout), dtype=bool)
if train_drop_dense.any():
    neigh = nn_drop.radius_neighbors(test_drop_rad, return_distance=False)
    test_drop_dense = np.fromiter(
        (len(x) > 0 for x in neigh), dtype=bool, count=len(df_holdout)
    )

df_train["dense_DBSCAN_trips"] = train_pick_dense & train_drop_dense
df_holdout["dense_DBSCAN_trips"] = test_pick_dense & test_drop_dense

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 14
_ = df_train.loc[
    df_train.dense_DBSCAN_trips == 1, ["pickup_longitude", "pickup_latitude"]
]



## === cell 15
df_train.loc[df_train.trip_distance < 0.2, "trip_distance"] = 0.2

_ = bucketsCount



## === cell 16
df_train["trip_rate"] = df_train["fare_amount"].to_numpy(dtype=np.float64) / df_train[
    "trip_distance"
].to_numpy(dtype=np.float64)



## === cell 17
len(df_train.loc[df_train.trip_rate > THERSHOLD_TRIP_FARE_RATE])



## === cell 18
ids = df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train[ids]
print("New size: %d" % len(df_train))



## === cell 19
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 20
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[~airportTripsIds]



## === cell 21
_ = df_airport_trips, df_city_trips



## === cell 22
_ = df_train



## === cell 23
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



## === cell 24
y = df_train["fare_amount"]
train = df_train.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.01
)



## === cell 25
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




## === cell 26
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



## === cell 27
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



## === cell 28
len(test_key)



## === cell 29
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(prediction, 2)})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
