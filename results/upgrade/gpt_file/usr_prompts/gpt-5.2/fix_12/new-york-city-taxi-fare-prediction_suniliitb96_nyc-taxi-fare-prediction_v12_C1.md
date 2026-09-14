# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

5.92286

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.83398) has done: 'I fix the `haversine_np` function so it correctly handles scalar airport coordinates (floats) as well as vector inputs, which unblocks the airport-distance feature generation and downstream filtering. Then I ensure we drop non-numeric columns (notably `pickup_datetime`) from the XGBoost matrices and update deprecated XGBoost params (`reg:linear`, `silent`) so training runs in the current Kaggle runtime. Finally, I make the train/test feature columns align deterministically (same columns/order) and ensure a valid `taxi_fare_submission.csv` is written with `key,fare_amount` and the correct row count.'
- What this solution (achieved 5.92286) has done: 'The timeout is dominated by two DBSCAN fits on up to ~1M points (ball-tree haversine clustering) plus repeated large DataFrame materializations. I keep the exact feature set and XGBoost training logic, but make DBSCAN faster by fitting it on a much smaller *provably irrelevant-filtered* subset (only trips near NYC + with realistic distances already enforced) and by reusing precomputed contiguous radians arrays. I also reduce pandas overhead by avoiding repeated `.reset_index()`/temporary DataFrames, precomputing numpy arrays once, and using boolean masks directly (semantics unchanged). Finally, I cap thread oversubscription for BLAS/OpenMP to prevent CPU thrash while keeping the same algorithms and determinism.'

# 9. Code solution

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

THRESHOLD_TRIP_DISTANCE = 25.0



## === cell 2
import os
import timeit

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import xgboost as xgb

from pandas.tseries.holiday import (
    USFederalHolidayCalendar as calendar,
)  # kept (core logic unchanged)
from sklearn import metrics  # kept (core logic unchanged)


def haversine_np(lat1, lon1, lat2, lon2, radius=KMS_PER_RADIAN):
    """
    Vectorized haversine distance in kilometers.
    Inputs can be numpy arrays / pandas Series OR scalars.
    """
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c


BASE_INPUT = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

if not os.path.exists(TRAIN_PATH):
    BASE_INPUT = "/kaggle/input"
    TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
    TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)

np.random.seed(0)
NTHREAD = max(1, (os.cpu_count() or 1))



## === cell 3
start_time = timeit.default_timer()

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

df_train = pd.read_csv(
    TRAIN_PATH,
    nrows=MAX_TRAINING_SIZE,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
df_holdout = pd.read_csv(
    TEST_PATH,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)

test_key = df_holdout.pop("key")
_ = df_train.pop("key")

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 4
print("Old size: %d" % len(df_train))

pc = df_train["passenger_count"].to_numpy(copy=False)
fa = df_train["fare_amount"].to_numpy(copy=False)

na_mask = (
    ~np.isnan(fa)
    & ~np.isnan(df_train["pickup_longitude"].to_numpy(copy=False))
    & ~np.isnan(df_train["pickup_latitude"].to_numpy(copy=False))
    & ~np.isnan(df_train["dropoff_longitude"].to_numpy(copy=False))
    & ~np.isnan(df_train["dropoff_latitude"].to_numpy(copy=False))
)

mask = (fa >= 0) & na_mask & (pc < 7) & (pc != 0)
df_train = df_train.loc[mask].copy()

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
df_train = df_train.loc[select_within_boundingbox(df_train, BB)].copy()
print("New size: %d" % len(df_train))




## === cell 6
def addPickDropDistanceFeature(df):
    df["trip_distance"] = haversine_np(
        df["pickup_latitude"].to_numpy(copy=False),
        df["pickup_longitude"].to_numpy(copy=False),
        df["dropoff_latitude"].to_numpy(copy=False),
        df["dropoff_longitude"].to_numpy(copy=False),
    ).astype("float32")
    return df


def addAirportDistanceFeatures(df):
    pu_lat = df["pickup_latitude"].to_numpy(copy=False)
    pu_lon = df["pickup_longitude"].to_numpy(copy=False)
    do_lat = df["dropoff_latitude"].to_numpy(copy=False)
    do_lon = df["dropoff_longitude"].to_numpy(copy=False)

    df["pickup_distance_to_jfk"] = haversine_np(
        pu_lat, pu_lon, JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]
    ).astype("float32")
    df["drop_distance_to_jfk"] = haversine_np(
        do_lat, do_lon, JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]
    ).astype("float32")

    df["pickup_distance_to_lgr"] = haversine_np(
        pu_lat, pu_lon, LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]
    ).astype("float32")
    df["drop_distance_to_lgr"] = haversine_np(
        do_lat, do_lon, LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]
    ).astype("float32")

    df["pickup_distance_to_ewr"] = haversine_np(
        pu_lat, pu_lon, EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]
    ).astype("float32")
    df["drop_distance_to_ewr"] = haversine_np(
        do_lat, do_lon, EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]
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




## === cell 7
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 8
bucketsCount = 100
feat = "trip_distance"



## === cell 9
print("Old size: %d" % len(df_train))
df_train = df_train.loc[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE].copy()
print("New size: %d" % len(df_train))




## === cell 10
def add_datetime_features(df):
    dt = df["pickup_datetime"].dt
    df["hour"] = dt.hour.astype("int16")
    df["day"] = dt.day.astype("int16")
    df["month"] = dt.month.astype("int16")
    df["weekday"] = dt.weekday.astype("int16")
    df["year"] = dt.year.astype("int16")
    return df


start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_holdout = add_datetime_features(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 11
train_len = len(df_train)



## === cell 12
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 13
_DBSCAN_NJOBS = NTHREAD

start_time = timeit.default_timer()

pickup_latlon_train = df_train[["pickup_latitude", "pickup_longitude"]].to_numpy(
    dtype=np.float64, copy=False
)
drop_latlon_train = df_train[["dropoff_latitude", "dropoff_longitude"]].to_numpy(
    dtype=np.float64, copy=False
)
pickup_coords_train = np.ascontiguousarray(np.radians(pickup_latlon_train))
dropoff_coords_train = np.ascontiguousarray(np.radians(drop_latlon_train))

pu_view = pickup_latlon_train.view([("lat", np.float64), ("lon", np.float64)])
do_view = drop_latlon_train.view([("lat", np.float64), ("lon", np.float64)])

_, pu_unique_idx, pu_inv = np.unique(pu_view, return_index=True, return_inverse=True)
_, do_unique_idx, do_inv = np.unique(do_view, return_index=True, return_inverse=True)

pickup_coords_train_u = pickup_coords_train[pu_unique_idx]
dropoff_coords_train_u = dropoff_coords_train[do_unique_idx]

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
    n_jobs=_DBSCAN_NJOBS,
).fit(pickup_coords_train_u)
labels_pick_train_u = dbscan_pick.labels_
labels_pick_train = labels_pick_train_u[pu_inv]

elapsed = timeit.default_timer() - start_time
elapsed



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2174406463.py in <cell line: 0>()
     23 # are identical points and do not change DBSCAN neighborhood counts (they only add identical samples).
     24 # We map labels back to full data exactly.
---> 25 pu_view = pickup_latlon_train.view([("lat", np.float64), ("lon", np.float64)])
     26 do_view = drop_latlon_train.view([("lat", np.float64), ("lon", np.float64)])
     27 

ValueError: To change to a dtype of a different size, the last axis must be contiguous

## === cell 14
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
    n_jobs=_DBSCAN_NJOBS,
).fit(dropoff_coords_train_u)
labels_drop_train_u = dbscan_drop.labels_
labels_drop_train = labels_drop_train_u[do_inv]

elapsed = timeit.default_timer() - start_time
elapsed




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/570458730.py in <cell line: 0>()
      7     metric="haversine",
      8     n_jobs=_DBSCAN_NJOBS,
----> 9 ).fit(dropoff_coords_train_u)
     10 labels_drop_train_u = dbscan_drop.labels_
     11 labels_drop_train = labels_drop_train_u[do_inv]

NameError: name 'dropoff_coords_train_u' is not defined

## === cell 15
def _within_eps_of_any_core(dbscan_model, coords_radians, eps, batch_size=65536):
    if getattr(dbscan_model, "ball_tree_", None) is None:
        return np.zeros(coords_radians.shape[0], dtype=bool)
    n = coords_radians.shape[0]
    out = np.zeros(n, dtype=bool)
    bt = dbscan_model.ball_tree_
    for i in range(0, n, batch_size):
        sl = slice(i, min(i + batch_size, n))
        counts = bt.query_radius(
            coords_radians[sl], r=eps, count_only=True, n_jobs=_DBSCAN_NJOBS
        )
        out[sl] = counts > 0
    return out


start_time = timeit.default_timer()

hold_pick_latlon = df_holdout[["pickup_latitude", "pickup_longitude"]].to_numpy(
    dtype=np.float64, copy=False
)
hold_drop_latlon = df_holdout[["dropoff_latitude", "dropoff_longitude"]].to_numpy(
    dtype=np.float64, copy=False
)

pickup_coords_hold = np.ascontiguousarray(np.radians(hold_pick_latlon))
dropoff_coords_hold = np.ascontiguousarray(np.radians(hold_drop_latlon))

train_pick_dense = labels_pick_train != -1
train_drop_dense = labels_drop_train != -1

hold_pick_dense = _within_eps_of_any_core(
    dbscan_pick, pickup_coords_hold, EPS_IN_RADIAN
)
hold_drop_dense = _within_eps_of_any_core(
    dbscan_drop, dropoff_coords_hold, EPS_IN_RADIAN
)

df_train["dense_DBSCAN_trips"] = train_pick_dense & train_drop_dense
df_holdout["dense_DBSCAN_trips"] = hold_pick_dense & hold_drop_dense

elapsed = timeit.default_timer() - start_time
elapsed



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1655440386.py in <cell line: 0>()
     26 dropoff_coords_hold = np.ascontiguousarray(np.radians(hold_drop_latlon))
     27 
---> 28 train_pick_dense = labels_pick_train != -1
     29 train_drop_dense = labels_drop_train != -1
     30 

NameError: name 'labels_pick_train' is not defined

## === cell 16
df_tmp = df_train.loc[df_train.dense_DBSCAN_trips == 1]



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3762755993.py in <cell line: 0>()
----> 1 df_tmp = df_train.loc[df_train.dense_DBSCAN_trips == 1]
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'dense_DBSCAN_trips'

## === cell 17
df_holdout = df_holdout.loc[:, df_holdout.columns != "fare_amount"]
(len(df_train), len(df_holdout))



## === cell 18
df_train["trip_distance"] = df_train["trip_distance"].clip(lower=0.2)



## === cell 19
df_train["trip_rate"] = (df_train["fare_amount"] / df_train["trip_distance"]).astype(
    "float32"
)



## === cell 20
len(df_train.loc[df_train.trip_rate > THERSHOLD_TRIP_FARE_RATE])



## === cell 21
ids = df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train.loc[ids].copy()
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
_ = None



## === cell 25
_ = None



## === cell 26
df_train = df_train.drop(
    columns=[
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
    ],
    errors="ignore",
)
df_train.info()



## === cell 27
y = df_train["fare_amount"]
train = df_train.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.05
)



## === cell 28
params = {
    "max_depth": 8,  # Result of tuning with CV
    "eta": 0.03,  # Result of tuning with CV
    "subsample": 1,  # Result of tuning with CV
    "colsample_bytree": 0.8,  # Result of tuning with CV
    "objective": "reg:squarederror",  # was reg:linear (deprecated)
    "eval_metric": "rmse",
    "verbosity": 1,  # replaces silent
    "nthread": NTHREAD,
    "seed": 0,
    "tree_method": "hist",
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




## === cell 29
def _numeric_only(df):
    df2 = df
    dt_cols = [c for c in df2.columns if np.issubdtype(df2[c].dtype, np.datetime64)]
    if dt_cols:
        df2 = df2.drop(columns=dt_cols)
    return df2


def _as_float32_matrix(df, cols):
    X = df.loc[:, cols].to_numpy(dtype=np.float32, copy=False)
    return np.ascontiguousarray(X)


def XGBmodel(x_train, x_test, y_train, y_test, params):
    x_train_n = _numeric_only(x_train)
    x_test_n = _numeric_only(x_test)

    cols = x_train_n.columns.tolist()
    Xtr = _as_float32_matrix(x_train_n, cols)
    Xte = _as_float32_matrix(x_test_n, cols)

    ytr = y_train.to_numpy(dtype=np.float32, copy=False)
    yte = y_test.to_numpy(dtype=np.float32, copy=False)

    try:
        matrix_train = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=cols)
        matrix_test = xgb.QuantileDMatrix(Xte, label=yte, feature_names=cols)
    except Exception:
        matrix_train = xgb.DMatrix(Xtr, label=ytr, feature_names=cols)
        matrix_test = xgb.DMatrix(Xte, label=yte, feature_names=cols)

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model, cols


model, train_feature_cols = XGBmodel(x_train, x_test, y_train, y_test, params)



## === cell 30
x_pred = df_holdout.drop(
    columns=[
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
    ],
    errors="ignore",
)
x_pred = _numeric_only(x_pred)

x_pred = x_pred.reindex(columns=train_feature_cols, fill_value=0)
Xpred = np.ascontiguousarray(x_pred.to_numpy(dtype=np.float32, copy=False))

try:
    dpre = xgb.QuantileDMatrix(Xpred, feature_names=train_feature_cols)
except Exception:
    dpre = xgb.DMatrix(Xpred, feature_names=train_feature_cols)

prediction = model.predict(
    dpre,
    iteration_range=(0, model.best_iteration + 1),
)



## === cell 31
len(test_key)



## === cell 32
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(prediction, 2)})

submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote submission to:", submission_path, "rows:", len(submission))
submission.head()
