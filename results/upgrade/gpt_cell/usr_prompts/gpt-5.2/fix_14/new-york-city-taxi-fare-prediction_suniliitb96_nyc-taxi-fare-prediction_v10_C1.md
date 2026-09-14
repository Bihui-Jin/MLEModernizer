# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

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
    if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
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



## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1319151853.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m [0mstart_time[0m [0;34m=[0m [0mtimeit[0m[0;34m.[0m[0mdefault_timer[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m [0;34m[0m[0m
[0;32m---> 18[0;31m [0mdf_train[0m [0;34m=[0m [0madd_datetime_features[0m[0;34m([0m[0mdf_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0mdf_test[0m [0;34m=[0m [0madd_datetime_features[0m[0;34m([0m[0mdf_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1319151853.py[0m in [0;36madd_datetime_features[0;34m(df)[0m
[1;32m      1[0m [0;32mdef[0m [0madd_datetime_features[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0;32mif[0m [0;32mnot[0m [0mnp[0m[0;34m.[0m[0missubdtype[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mdatetime64[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m         df["pickup_datetime"] = pd.to_datetime(
[1;32m      4[0m             [0mdf[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m,[0m [0mformat[0m[0;34m=[0m[0;34m"%Y-%m-%d %H:%M:%S UTC"[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py[0m in [0;36missubdtype[0;34m(arg1, arg2)[0m
[1;32m    415[0m     """
[1;32m    416[0m     [0;32mif[0m [0;32mnot[0m [0missubclass_[0m[0;34m([0m[0marg1[0m[0;34m,[0m [0mgeneric[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 417[0;31m         [0marg1[0m [0;34m=[0m [0mdtype[0m[0;34m([0m[0marg1[0m[0;34m)[0m[0;34m.[0m[0mtype[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    418[0m     [0;32mif[0m [0;32mnot[0m [0missubclass_[0m[0;34m([0m[0marg2[0m[0;34m,[0m [0mgeneric[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    419[0m         [0marg2[0m [0;34m=[0m [0mdtype[0m[0;34m([0m[0marg2[0m[0;34m)[0m[0;34m.[0m[0mtype[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 13
df_train.info()
