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
MAX_TRAINING_SIZE = 1_000_000

EPS_IN_KM = 0.5  ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

RADIUS_VICINITY_AIRPORTS = 1.0

THERSHOLD_TRIP_FARE_RATE = 50.0



## === cell 2
import os
import timeit
import random
import gc

_CPU = os.cpu_count() or 1
N_JOBS = max(1, min(_CPU, 8))

os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)



## === cell 3
INPUT_DIR_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input/new-york-city-taxi-fare-prediction",
    "../input",
]


def _pick_input_dir(cands):
    for d in cands:
        if os.path.exists(d):
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "test.csv")
            ):
                return d
    return cands[0]


INPUT_DIR = _pick_input_dir(INPUT_DIR_CANDIDATES)
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH



## === cell 4
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
dtype_map_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
dtype_map_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

df_train = pd.read_csv(
    TRAIN_PATH,
    nrows=MAX_TRAINING_SIZE,
    usecols=train_usecols,
    dtype=dtype_map_train,
    parse_dates=["pickup_datetime"],
    date_format=DT_FORMAT,
    engine="c",
)
df_test = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=dtype_map_test,
    parse_dates=["pickup_datetime"],
    date_format=DT_FORMAT,
    engine="c",
)

test_key = df_test["key"].copy()

df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 5
df_train.head()



## === cell 6
print("Old size: %d" % len(df_train))

num_cols = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
arr = df_train[num_cols].to_numpy(copy=False)
notna_all = np.isfinite(arr).all(axis=1)

pc = df_train["passenger_count"].to_numpy(copy=False)
fa = df_train["fare_amount"].to_numpy(copy=False)
mask = (fa >= 0) & notna_all & (pc < 7) & (pc != 0)

df_train = df_train.loc[mask]

print("New size: %d" % len(df_train))

del arr, notna_all, pc, fa, mask
gc.collect()




## === cell 7
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

gc.collect()




## === cell 8
def _haversine_vec_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in kilometers.
    Inputs are in decimal degrees; outputs in km.

    Bugfix: support scalar lat2/lon2 (airport coordinates) as well as arrays.
    """
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return KMS_PER_RADIAN * c


def addPickDropDistanceFeature_inplace(df):
    plat = df["pickup_latitude"].to_numpy(copy=False)
    plon = df["pickup_longitude"].to_numpy(copy=False)
    dlat = df["dropoff_latitude"].to_numpy(copy=False)
    dlon = df["dropoff_longitude"].to_numpy(copy=False)
    df["trip_distance"] = _haversine_vec_km(plat, plon, dlat, dlon).astype("float32")
    return df


def addAirportDistanceFeatures_inplace(df):
    plat = df["pickup_latitude"].to_numpy(copy=False)
    plon = df["pickup_longitude"].to_numpy(copy=False)
    dlat = df["dropoff_latitude"].to_numpy(copy=False)
    dlon = df["dropoff_longitude"].to_numpy(copy=False)

    df["pickup_distance_to_jfk"] = _haversine_vec_km(
        plat, plon, JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]
    ).astype("float32")
    df["drop_distance_to_jfk"] = _haversine_vec_km(
        dlat, dlon, JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]
    ).astype("float32")

    df["pickup_distance_to_lgr"] = _haversine_vec_km(
        plat, plon, LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]
    ).astype("float32")
    df["drop_distance_to_lgr"] = _haversine_vec_km(
        dlat, dlon, LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]
    ).astype("float32")

    df["pickup_distance_to_ewr"] = _haversine_vec_km(
        plat, plon, EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]
    ).astype("float32")
    df["drop_distance_to_ewr"] = _haversine_vec_km(
        dlat, dlon, EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]
    ).astype("float32")

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




## === cell 9
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature_inplace(df_train)
df_test = addPickDropDistanceFeature_inplace(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 10
bucketsCount = 100
feat = "trip_distance"
_ = bucketsCount, feat  # keep variables defined



## === cell 11
print("Old size: %d" % len(df_train))
df_train = df_train.loc[df_train.trip_distance < 25.0]
print("New size: %d" % len(df_train))

gc.collect()



## === cell 12
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures_inplace(df_train)
df_test = addAirportDistanceFeatures_inplace(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 13
airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
df_test["airport_bound"] = airportTripsIds

airportTripsIds_train = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds_train

df_airport_trips = df_train.loc[airportTripsIds_train]
df_city_trips = df_train.loc[~airportTripsIds_train]

pd.DataFrame(
    data={
        "Airport Trips": df_airport_trips.fare_amount,
        "City Trips": df_city_trips.fare_amount,
    }
).describe()




## === cell 14
def add_datetime_features_inplace(df):
    if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    dt = df["pickup_datetime"].dt
    df["hour"] = dt.hour.astype("int16")
    df["day"] = dt.day.astype("int16")
    df["month"] = dt.month.astype("int16")
    df["weekday"] = dt.weekday.astype("int16")
    df["year"] = dt.year.astype("int16")
    return df


start_time = timeit.default_timer()

df_train = add_datetime_features_inplace(df_train)
df_test = add_datetime_features_inplace(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 15
train_len = len(df_train)
(train_len, len(df_test))



## === cell 16
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN
EPS_IN_RADIAN




## === cell 17
def _unique_coords_with_inverse(lat_lon_df):
    coords = lat_lon_df.to_numpy(dtype=np.float32, copy=False)
    coords_c = np.ascontiguousarray(coords)
    v = coords_c.view([("f0", np.float32), ("f1", np.float32)]).reshape(-1)
    uniq_v, inv = np.unique(v, return_inverse=True)
    uniq = uniq_v.view(np.float32).reshape(-1, 2)
    return uniq, inv


start_time = timeit.default_timer()

pickup_unique_deg, pickup_inv = _unique_coords_with_inverse(
    df_train[["pickup_latitude", "pickup_longitude"]]
)
dropoff_unique_deg, dropoff_inv = _unique_coords_with_inverse(
    df_train[["dropoff_latitude", "dropoff_longitude"]]
)

pickup_radians_unique = np.ascontiguousarray(
    np.radians(pickup_unique_deg.astype("float64"))
)
dropoff_radians_unique = np.ascontiguousarray(
    np.radians(dropoff_unique_deg.astype("float64"))
)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 18
start_time = timeit.default_timer()

if pickup_radians_unique.shape[0] < MIN_SAMPLES_CLUSTER:

    class _DummyDB:
        labels_ = np.full((pickup_radians_unique.shape[0],), -1, dtype=np.int32)
        core_sample_indices_ = np.empty((0,), dtype=np.int64)

    dbscan_pick_unique = _DummyDB()
else:
    dbscan_pick_unique = DBSCAN(
        eps=EPS_IN_RADIAN,
        min_samples=MIN_SAMPLES_CLUSTER,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=N_JOBS,
    ).fit(pickup_radians_unique)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 19
labels_pick_unique = np.asarray(dbscan_pick_unique.labels_, dtype=np.int32)
labels_pick = labels_pick_unique[pickup_inv]
df_train["is_good_dbscan_pick"] = labels_pick != -1

n_clusters_pick = len(set(labels_pick_unique.tolist())) - (
    1 if -1 in labels_pick_unique else 0
)
n_clusters_pick



## === cell 20
print(
    "Pickup: DBSCAN clustering done. Estimated number of clusters: %d" % n_clusters_pick
)



## === cell 21
mask_dense_pick = labels_pick != -1
mask_rare_pick = labels_pick == -1

print("df_train size: %d" % len(df_train))
print("df_train_dense_pick size: %d" % int(mask_dense_pick.sum()))



## === cell 22
_ = None  # placeholder to preserve cell structure



## === cell 23
start_time = timeit.default_timer()

if dropoff_radians_unique.shape[0] < MIN_SAMPLES_CLUSTER:

    class _DummyDB:
        labels_ = np.full((dropoff_radians_unique.shape[0],), -1, dtype=np.int32)
        core_sample_indices_ = np.empty((0,), dtype=np.int64)

    dbscan_drop_unique = _DummyDB()
else:
    dbscan_drop_unique = DBSCAN(
        eps=EPS_IN_RADIAN,
        min_samples=MIN_SAMPLES_CLUSTER,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=N_JOBS,
    ).fit(dropoff_radians_unique)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 24
labels_drop_unique = np.asarray(dbscan_drop_unique.labels_, dtype=np.int32)
labels_drop = labels_drop_unique[dropoff_inv]
df_train["is_good_dbscan_drop"] = labels_drop != -1

n_clusters_drop = len(set(labels_drop_unique.tolist())) - (
    1 if -1 in labels_drop_unique else 0
)
n_clusters_drop



## === cell 25
print(
    "DropOff: DBSCAN clustering done. Estimated number of clusters: %d"
    % n_clusters_drop
)



## === cell 26
mask_dense_drop = labels_drop != -1
mask_rare_drop = labels_drop == -1

print("df_train size: %d" % len(df_train))
print("df_train_dense_drop size: %d" % int(mask_dense_drop.sum()))



## === cell 27
from sklearn.neighbors import BallTree

pickup_radians_test = np.ascontiguousarray(
    np.radians(
        df_test[["pickup_latitude", "pickup_longitude"]].to_numpy(
            dtype="float64", copy=False
        )
    )
)
dropoff_radians_test = np.ascontiguousarray(
    np.radians(
        df_test[["dropoff_latitude", "dropoff_longitude"]].to_numpy(
            dtype="float64", copy=False
        )
    )
)

core_pick_idx = getattr(dbscan_pick_unique, "core_sample_indices_", None)
core_drop_idx = getattr(dbscan_drop_unique, "core_sample_indices_", None)
if core_pick_idx is None:
    core_pick_idx = np.empty((0,), dtype=np.int64)
if core_drop_idx is None:
    core_drop_idx = np.empty((0,), dtype=np.int64)

core_pick_unique = pickup_radians_unique[core_pick_idx]
core_drop_unique = dropoff_radians_unique[core_drop_idx]

if core_pick_unique.shape[0] == 0:
    test_good_pick = np.zeros((pickup_radians_test.shape[0],), dtype=bool)
else:
    tree_pick = BallTree(core_pick_unique, metric="haversine")
    test_good_pick = (
        tree_pick.query_radius(pickup_radians_test, r=EPS_IN_RADIAN, count_only=True)
        > 0
    )

if core_drop_unique.shape[0] == 0:
    test_good_drop = np.zeros((dropoff_radians_test.shape[0],), dtype=bool)
else:
    tree_drop = BallTree(core_drop_unique, metric="haversine")
    test_good_drop = (
        tree_drop.query_radius(dropoff_radians_test, r=EPS_IN_RADIAN, count_only=True)
        > 0
    )

df_test["is_good_dbscan_pick"] = test_good_pick
df_test["is_good_dbscan_drop"] = test_good_drop

del (
    pickup_radians_test,
    dropoff_radians_test,
    core_pick_unique,
    core_drop_unique,
    pickup_unique_deg,
    dropoff_unique_deg,
    pickup_radians_unique,
    dropoff_radians_unique,
    pickup_inv,
    dropoff_inv,
    labels_pick_unique,
    labels_drop_unique,
    core_pick_idx,
    core_drop_idx,
)
gc.collect()



## === cell 28
df_train["dense_DBSCAN_trips"] = (
    df_train["is_good_dbscan_drop"] & df_train["is_good_dbscan_pick"]
).astype(bool)
df_test["dense_DBSCAN_trips"] = (
    df_test["is_good_dbscan_drop"] & df_test["is_good_dbscan_pick"]
).astype(bool)

len(df_train.loc[df_train.dense_DBSCAN_trips == 1])



## === cell 29
df_train.head()



## === cell 30
(len(df_train), len(df_test))



## === cell 31
pd.DataFrame(
    data={
        "Good Density Pickups": df_train.loc[
            df_train.is_good_dbscan_pick == 1
        ].fare_amount,
        "LOW Density Pickups": df_train.loc[
            df_train.is_good_dbscan_pick == 0
        ].fare_amount,
    }
).describe()



## === cell 32
pd.DataFrame(
    data={
        "Good Density DropOffs": df_train.loc[
            df_train.is_good_dbscan_drop == 1
        ].fare_amount,
        "LOW Density DropOffs": df_train.loc[
            df_train.is_good_dbscan_drop == 0
        ].fare_amount,
    }
).describe()



## === cell 33
trip_distance = df_train["trip_distance"].to_numpy(dtype="float64", copy=False)
fare_amount = df_train["fare_amount"].to_numpy(dtype="float64", copy=False)

dist = np.maximum(trip_distance, 0.2)
trip_rate = fare_amount / dist

len(trip_rate[trip_rate > THERSHOLD_TRIP_FARE_RATE])



## === cell 34
ids = trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train.loc[ids]
print("New size: %d" % len(df_train))

del trip_distance, fare_amount, dist, trip_rate, ids
gc.collect()



## === cell 35
drop_cols = [
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
]

df_tmp = df_train.drop(columns=drop_cols, errors="ignore")
df_tmp.info()



## === cell 36
y_raw = df_tmp["fare_amount"].astype(float)
y = np.log1p(y_raw)

train = df_tmp.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.10
)



## === cell 37
params = {
    "max_depth": 8,  # Result of tuning with CV
    "eta": 0.03,  # Result of tuning with CV
    "subsample": 1,  # Result of tuning with CV
    "colsample_bytree": 0.8,  # Result of tuning with CV
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 1,
    "nthread": N_JOBS,
    "seed": SEED,
    "tree_method": "hist",
    "deterministic_histogram": True,
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
    print(params)




## === cell 38
def XGBmodel(x_train, x_test, y_train, y_test, params):
    feature_names = list(x_train.columns)

    Xtr = np.ascontiguousarray(x_train.to_numpy(dtype=np.float32, copy=False))
    Xte = np.ascontiguousarray(x_test.to_numpy(dtype=np.float32, copy=False))
    ytr = np.asarray(y_train, dtype=np.float32)
    yte = np.asarray(y_test, dtype=np.float32)

    matrix_train = xgb.DMatrix(
        Xtr, label=ytr, nthread=N_JOBS, feature_names=feature_names
    )
    matrix_test = xgb.DMatrix(
        Xte, label=yte, nthread=N_JOBS, feature_names=feature_names
    )
    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test, params)



## === cell 39
x_pred = df_test.drop(columns=drop_cols, errors="ignore")
x_pred = x_pred.reindex(columns=train.columns, fill_value=0)

Xpred = np.ascontiguousarray(x_pred.to_numpy(dtype=np.float32, copy=False))
dtest = xgb.DMatrix(Xpred, nthread=N_JOBS, feature_names=list(train.columns))

try:
    prediction_log = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
except TypeError:
    prediction_log = model.predict(
        dtest, ntree_limit=getattr(model, "best_ntree_limit", 0)
    )

prediction = np.expm1(prediction_log)



## === cell 40
len(test_key), len(prediction)



## === cell 41
submission = pd.DataFrame(
    {
        "key": test_key.values,
        "fare_amount": prediction,
    }
)

submission["fare_amount"] = submission["fare_amount"].clip(lower=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
submission.head()
