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



## === cell 2
import os
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb

from pandas.tseries.holiday import (
    USFederalHolidayCalendar as calendar,
)  # unused but kept to preserve original structure

import timeit
from sklearn import metrics

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

NTHREAD = int(os.environ.get("OMP_NUM_THREADS", "4"))
if NTHREAD <= 0:
    NTHREAD = 4



## === cell 3
import timeit

start_time = timeit.default_timer()

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int8",
}

df_train = pd.read_csv(
    "../input/train.csv",
    nrows=MAX_TRAINING_SIZE,
    usecols=train_usecols,
    dtype=dtypes,
    parse_dates=["pickup_datetime"],
)
df_test = pd.read_csv(
    "../input/test.csv",
    usecols=test_usecols,
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
)

test_key = df_test["key"]
df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 4
_ = df_train.head()



## === cell 5
print("Old size: %d" % len(df_train))

fare_v = df_train["fare_amount"].to_numpy(copy=False)
pc_v = df_train["passenger_count"].to_numpy(copy=False)
no_na = (
    ~df_train[
        [
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ]
    .isna()
    .any(axis=1)
    .to_numpy()
)

mask = (fare_v >= 0.0) & no_na & (pc_v < 7) & (pc_v != 0)
df_train = df_train.loc[mask]

print("New size: %d" % len(df_train))




## === cell 6
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
df_train = df_train.loc[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))




## === cell 7
def haversine_rad(lat1r, lon1r, lat2r, lon2r):
    lat1r = np.asarray(lat1r, dtype="float64")
    lon1r = np.asarray(lon1r, dtype="float64")
    lat2r = np.asarray(lat2r, dtype="float64")
    lon2r = np.asarray(lon2r, dtype="float64")

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return KMS_PER_RADIAN * c


def add_precomputed_radians(df):
    df["_pickup_lat_rad"] = np.radians(
        df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    )
    df["_pickup_lon_rad"] = np.radians(
        df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    )
    df["_dropoff_lat_rad"] = np.radians(
        df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    )
    df["_dropoff_lon_rad"] = np.radians(
        df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    )
    return df


def addPickDropDistanceFeature(df):
    df["trip_distance"] = haversine_rad(
        df["_pickup_lat_rad"].values,
        df["_pickup_lon_rad"].values,
        df["_dropoff_lat_rad"].values,
        df["_dropoff_lon_rad"].values,
    )
    return df


_JFK_LAT_RAD, _JFK_LON_RAD = np.radians(JFK_GEO_LOCATION[0]), np.radians(
    JFK_GEO_LOCATION[1]
)
_LGR_LAT_RAD, _LGR_LON_RAD = np.radians(LGR_GEO_LOCATION[0]), np.radians(
    LGR_GEO_LOCATION[1]
)
_EWR_LAT_RAD, _EWR_LON_RAD = np.radians(EWR_GEO_LOCATION[0]), np.radians(
    EWR_GEO_LOCATION[1]
)


def addAirportDistanceFeatures(df):
    df["pickup_distance_to_jfk"] = haversine_rad(
        df["_pickup_lat_rad"].values,
        df["_pickup_lon_rad"].values,
        _JFK_LAT_RAD,
        _JFK_LON_RAD,
    )
    df["drop_distance_to_jfk"] = haversine_rad(
        df["_dropoff_lat_rad"].values,
        df["_dropoff_lon_rad"].values,
        _JFK_LAT_RAD,
        _JFK_LON_RAD,
    )

    df["pickup_distance_to_lgr"] = haversine_rad(
        df["_pickup_lat_rad"].values,
        df["_pickup_lon_rad"].values,
        _LGR_LAT_RAD,
        _LGR_LON_RAD,
    )
    df["drop_distance_to_lgr"] = haversine_rad(
        df["_dropoff_lat_rad"].values,
        df["_dropoff_lon_rad"].values,
        _LGR_LAT_RAD,
        _LGR_LON_RAD,
    )

    df["pickup_distance_to_ewr"] = haversine_rad(
        df["_pickup_lat_rad"].values,
        df["_pickup_lon_rad"].values,
        _EWR_LAT_RAD,
        _EWR_LON_RAD,
    )
    df["drop_distance_to_ewr"] = haversine_rad(
        df["_dropoff_lat_rad"].values,
        df["_dropoff_lon_rad"].values,
        _EWR_LAT_RAD,
        _EWR_LON_RAD,
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




## === cell 8
start_time = timeit.default_timer()

df_train = add_precomputed_radians(df_train)
df_test = add_precomputed_radians(df_test)

df_train = addPickDropDistanceFeature(df_train)
df_test = addPickDropDistanceFeature(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 9
print("Old size: %d" % len(df_train))
df_train = df_train.loc[df_train.trip_distance < 25.0]
print("New size: %d" % len(df_train))



## === cell 10
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_test = addAirportDistanceFeatures(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 11
airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
df_test["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[~airportTripsIds]

_ = pd.DataFrame(
    data={
        "Airport Trips": df_airport_trips.fare_amount,
        "City Trips": df_city_trips.fare_amount,
    }
).describe()




## === cell 12
def add_datetime_features(df):
    from pandas.api.types import is_datetime64_any_dtype

    if not is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], utc=True, errors="coerce"
        )

    df["hour"] = df.pickup_datetime.dt.hour
    df["day"] = df.pickup_datetime.dt.day
    df["month"] = df.pickup_datetime.dt.month
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["year"] = df.pickup_datetime.dt.year

    return df


start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_test = add_datetime_features(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 13
df_train.info()



## === cell 14
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 15
start_time = timeit.default_timer()

pickup_coords_train_rad = np.column_stack(
    (
        df_train["_pickup_lon_rad"].to_numpy(dtype=np.float32, copy=False),
        df_train["_pickup_lat_rad"].to_numpy(dtype=np.float32, copy=False),
    )
)

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
    n_jobs=NTHREAD,
).fit(pickup_coords_train_rad)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 16
labels_pick_train = dbscan_pick.labels_
df_train["is_good_dbscan_pick"] = labels_pick_train != -1

n_clusters_pick = len(set(labels_pick_train)) - (1 if -1 in labels_pick_train else 0)
n_clusters_pick



## === cell 17
start_time = timeit.default_timer()

core_pick_idx = dbscan_pick.core_sample_indices_
core_pick_rad = pickup_coords_train_rad[core_pick_idx]

if core_pick_rad.shape[0] == 0:
    df_test["is_good_dbscan_pick"] = False
else:
    pick_tree = BallTree(core_pick_rad, metric="haversine")
    pickup_coords_test_rad = np.column_stack(
        (
            df_test["_pickup_lon_rad"].to_numpy(dtype=np.float32, copy=False),
            df_test["_pickup_lat_rad"].to_numpy(dtype=np.float32, copy=False),
        )
    )
    cnt = pick_tree.query_radius(
        pickup_coords_test_rad, r=EPS_IN_RADIAN, count_only=True
    )
    df_test["is_good_dbscan_pick"] = cnt > 0

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 18
start_time = timeit.default_timer()

dropoff_coords_train_rad = np.column_stack(
    (
        df_train["_dropoff_lon_rad"].to_numpy(dtype=np.float32, copy=False),
        df_train["_dropoff_lat_rad"].to_numpy(dtype=np.float32, copy=False),
    )
)

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
    n_jobs=NTHREAD,
).fit(dropoff_coords_train_rad)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 19
labels_drop_train = dbscan_drop.labels_
df_train["is_good_dbscan_drop"] = labels_drop_train != -1

n_clusters_drop = len(set(labels_drop_train)) - (1 if -1 in labels_drop_train else 0)
n_clusters_drop



## === cell 20
start_time = timeit.default_timer()

core_drop_idx = dbscan_drop.core_sample_indices_
core_drop_rad = dropoff_coords_train_rad[core_drop_idx]

if core_drop_rad.shape[0] == 0:
    df_test["is_good_dbscan_drop"] = False
else:
    drop_tree = BallTree(core_drop_rad, metric="haversine")
    dropoff_coords_test_rad = np.column_stack(
        (
            df_test["_dropoff_lon_rad"].to_numpy(dtype=np.float32, copy=False),
            df_test["_dropoff_lat_rad"].to_numpy(dtype=np.float32, copy=False),
        )
    )
    cnt = drop_tree.query_radius(
        dropoff_coords_test_rad, r=EPS_IN_RADIAN, count_only=True
    )
    df_test["is_good_dbscan_drop"] = cnt > 0

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 21
df_train["dense_DBSCAN_trips"] = (
    df_train["is_good_dbscan_drop"].values & df_train["is_good_dbscan_pick"].values
)
df_test["dense_DBSCAN_trips"] = (
    df_test["is_good_dbscan_drop"].values & df_test["is_good_dbscan_pick"].values
)

len(df_train.loc[df_train.dense_DBSCAN_trips == 1])



## === cell 22
(len(df_train), len(df_test))



## === cell 23
fare = df_train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
dist = df_train["trip_distance"].to_numpy(dtype=np.float64, copy=False)
dist_clip = np.maximum(dist, 0.2)
trip_rate = fare / dist_clip



## === cell 24
ids = trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train.loc[ids]
print("New size: %d" % len(df_train))



## === cell 25
DROP_COLS = [
    "pickup_datetime",
    "is_good_dbscan_drop",
    "is_good_dbscan_pick",
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
    "_pickup_lat_rad",
    "_pickup_lon_rad",
    "_dropoff_lat_rad",
    "_dropoff_lon_rad",
]

df_tmp = df_train.drop(columns=DROP_COLS)
df_tmp.info()



## === cell 26
y = df_tmp["fare_amount"]
train = df_tmp.drop(columns=["fare_amount"])

if "trip_distance" in train.columns:
    td = train["trip_distance"].to_numpy(copy=False)
    n_bins = 20
    try:
        strat_bins = pd.qcut(td, q=n_bins, labels=False, duplicates="drop")
    except Exception:
        strat_bins = np.minimum((td / 1.0).astype(np.int32), n_bins - 1)
else:
    strat_bins = None

x_train, x_valid, y_train, y_valid = train_test_split(
    train, y, random_state=SEED, test_size=0.2, stratify=strat_bins
)



## === cell 27
params = {
    "max_depth": 8,  # Result of tuning with CV
    "eta": 0.03,  # Result of tuning with CV
    "subsample": 1,  # Result of tuning with CV
    "colsample_bytree": 0.8,  # Result of tuning with CV
    "objective": "reg:linear",
    "eval_metric": "rmse",
    "silent": 1,
    "nthread": NTHREAD,
    "tree_method": "hist",
    "seed": SEED,
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




## === cell 28
def XGBmodel(x_train, x_test, y_train, y_test, params):
    matrix_train = xgb.DMatrix(x_train, label=y_train, nthread=NTHREAD)
    matrix_test = xgb.DMatrix(x_test, label=y_test, nthread=NTHREAD)
    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_valid, y_train, y_valid, params)




## === cell 29
def get_best_num_boost_round(m):
    if hasattr(m, "best_iteration") and m.best_iteration is not None:
        return int(m.best_iteration) + 1
    if hasattr(m, "best_ntree_limit") and m.best_ntree_limit is not None:
        return int(m.best_ntree_limit)
    return None


best_num_boost_round = get_best_num_boost_round(model)

dtrain_full = xgb.DMatrix(train, label=y, nthread=NTHREAD)

if best_num_boost_round is None:
    model_full = xgb.train(params=params, dtrain=dtrain_full, num_boost_round=5000)
else:
    model_full = xgb.train(
        params=params, dtrain=dtrain_full, num_boost_round=best_num_boost_round
    )



## === cell 30
x_pred = df_test.drop(columns=DROP_COLS)

dmatrix_pred = xgb.DMatrix(x_pred, nthread=NTHREAD)

if hasattr(model_full, "best_ntree_limit") and model_full.best_ntree_limit is not None:
    prediction = model_full.predict(
        dmatrix_pred, ntree_limit=model_full.best_ntree_limit
    )
elif hasattr(model_full, "best_iteration") and model_full.best_iteration is not None:
    prediction = model_full.predict(
        dmatrix_pred, iteration_range=(0, int(model_full.best_iteration) + 1)
    )
else:
    prediction = model_full.predict(dmatrix_pred)



## === cell 31
len(test_key)



## === cell 32
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
