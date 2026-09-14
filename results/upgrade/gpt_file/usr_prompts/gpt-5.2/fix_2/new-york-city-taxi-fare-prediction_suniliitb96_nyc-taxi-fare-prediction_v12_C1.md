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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)



## === cell 1
MAX_TRAINING_SIZE = 1_000_00

EPS_IN_KM = 0.5  ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

RADIUS_VICINITY_AIRPORTS = 1.0

THERSHOLD_TRIP_FARE_RATE = 50.0

THRESHOLD_TRIP_DISTANCE = 25.0



## === cell 2
import os
import timeit

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tseries.holiday import (
    USFederalHolidayCalendar as calendar,
)  # kept (core logic unchanged)
from sklearn import metrics  # kept (core logic unchanged)

plt.style.use("seaborn-whitegrid")


def haversine_np(lat1, lon1, lat2, lon2, radius=KMS_PER_RADIAN):
    """
    Vectorized haversine distance in kilometers.
    Inputs can be numpy arrays or pandas Series.
    """
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

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



## === cell 3
start_time = timeit.default_timer()

df_train = pd.read_csv(
    TRAIN_PATH, nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_holdout = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])

test_key = df_holdout["key"].copy()

df_train.drop(columns=["key"], inplace=True)
df_holdout.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 4
print("Old size: %d" % len(df_train))

df_train = df_train[df_train.fare_amount >= 0]
df_train = df_train.dropna(how="any", axis="rows")

df_train = df_train.drop(
    index=df_train[df_train.passenger_count >= 7].index, axis="rows"
)
df_train = df_train.drop(
    index=df_train[df_train.passenger_count == 0].index, axis="rows"
)

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
df_train = df_train[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))




## === cell 6
def addPickDropDistanceFeature(df):
    df = df.copy()
    df["trip_distance"] = haversine_np(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def addAirportDistanceFeatures(df):
    df = df.copy()

    df["pickup_distance_to_jfk"] = haversine_np(
        df["pickup_latitude"],
        df["pickup_longitude"],
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )
    df["drop_distance_to_jfk"] = haversine_np(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )

    df["pickup_distance_to_lgr"] = haversine_np(
        df["pickup_latitude"],
        df["pickup_longitude"],
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )
    df["drop_distance_to_lgr"] = haversine_np(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )

    df["pickup_distance_to_ewr"] = haversine_np(
        df["pickup_latitude"],
        df["pickup_longitude"],
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    df["drop_distance_to_ewr"] = haversine_np(
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




## === cell 7
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 8
bucketsCount = 100
feat = "trip_distance"

try:
    df_train[feat].hist(bins=bucketsCount, figsize=(15, 8))
    df_holdout[feat].hist(bins=bucketsCount, figsize=(15, 8))
    plt.yscale("log")
    plt.xlabel(feat)
    plt.ylabel("Frequency Log")
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 9
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.trip_distance < THRESHOLD_TRIP_DISTANCE]
print("New size: %d" % len(df_train))




## === cell 10
def add_datetime_features(df):
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    df["hour"] = df.pickup_datetime.dt.hour
    df["day"] = df.pickup_datetime.dt.day
    df["month"] = df.pickup_datetime.dt.month
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["year"] = df.pickup_datetime.dt.year

    return df


start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_holdout = add_datetime_features(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 11
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)



## === cell 12
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 13
start_time = timeit.default_timer()

pickup_coords = np.radians(
    df_nyc_taxi[["pickup_latitude", "pickup_longitude"]].astype(float).values
)
dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(pickup_coords)
labels_pick = dbscan_pick.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 14
start_time = timeit.default_timer()

dropoff_coords = np.radians(
    df_nyc_taxi[["dropoff_latitude", "dropoff_longitude"]].astype(float).values
)
dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(dropoff_coords)
labels_drop = dbscan_drop.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 15
df_nyc_taxi["dense_DBSCAN_trips"] = (labels_pick != -1) & (labels_drop != -1)



## === cell 16
df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
try:
    plt.figure(figsize=(8, 6))
    plt.plot(
        df_tmp.pickup_longitude, df_tmp.pickup_latitude, "o", markersize=1, alpha=0.3
    )
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 17
df_train = df_nyc_taxi.iloc[:train_len, :].copy()
df_holdout = df_nyc_taxi.iloc[train_len:, :].copy()
df_holdout = df_holdout.loc[:, df_nyc_taxi.columns != "fare_amount"]

(len(df_train), len(df_holdout))



## === cell 18
df_train.loc[df_train.trip_distance < 0.2, "trip_distance"] = 0.2

try:
    (df_train.fare_amount / df_train.trip_distance).hist(
        bins=bucketsCount, figsize=(15, 8)
    )
    plt.yscale("log")
    plt.xlabel("trip_rate")
    plt.ylabel("Log Frequency")
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 19
df_train["trip_rate"] = df_train["fare_amount"] / df_train["trip_distance"]



## === cell 20
len(df_train.loc[df_train.trip_rate > THERSHOLD_TRIP_FARE_RATE])



## === cell 21
ids = df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train[ids].copy()
print("New size: %d" % len(df_train))



## === cell 22
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/512964251.py in <cell line: 0>()
      1 start_time = timeit.default_timer()
      2 
----> 3 df_train = addAirportDistanceFeatures(df_train)
      4 df_holdout = addAirportDistanceFeatures(df_holdout)
      5 

/tmp/ipykernel_11/535644952.py in addAirportDistanceFeatures(df)
     14     df = df.copy()
     15 
---> 16     df["pickup_distance_to_jfk"] = haversine_np(
     17         df["pickup_latitude"],
     18         df["pickup_longitude"],

/tmp/ipykernel_11/3971809124.py in haversine_np(lat1, lon1, lat2, lon2, radius)
     30     lat1 = np.radians(lat1.astype(float))
     31     lon1 = np.radians(lon1.astype(float))
---> 32     lat2 = np.radians(lat2.astype(float))
     33     lon2 = np.radians(lon2.astype(float))
     34 

AttributeError: 'float' object has no attribute 'astype'

## === cell 23
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds

df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[~airportTripsIds]



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1165458671.py in <cell line: 0>()
----> 1 airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
      2 df_holdout["airport_bound"] = airportTripsIds
      3 
      4 airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
      5 df_train["airport_bound"] = airportTripsIds

/tmp/ipykernel_11/535644952.py in getAirportTrips(df, airportVicinity)
     58 def getAirportTrips(df, airportVicinity):
     59     ids = (
---> 60         (df.pickup_distance_to_jfk < airportVicinity)
     61         | (df.drop_distance_to_jfk < airportVicinity)
     62         | (df.pickup_distance_to_lgr < airportVicinity)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'pickup_distance_to_jfk'

## === cell 24
pd.DataFrame(
    data={
        "Airport Trips": df_airport_trips.trip_rate,
        "City Trips": df_city_trips.trip_rate,
    }
).describe()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3262420191.py in <cell line: 0>()
      1 pd.DataFrame(
      2     data={
----> 3         "Airport Trips": df_airport_trips.trip_rate,
      4         "City Trips": df_city_trips.trip_rate,
      5     }

NameError: name 'df_airport_trips' is not defined

## === cell 25
pd.DataFrame(
    data={
        "Good Density Trips": df_train.loc[df_train.dense_DBSCAN_trips == 1].trip_rate,
        "LOW Density Pickups": df_train.loc[df_train.dense_DBSCAN_trips == 0].trip_rate,
    }
).describe()



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
    ]
)
df_train.info()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3648582925.py in <cell line: 0>()
----> 1 df_train = df_train.drop(
      2     columns=[
      3         "pickup_datetime",
      4         "pickup_distance_to_jfk",
      5         "drop_distance_to_jfk",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr'] not found in axis"

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



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3929071051.py in <cell line: 0>()
     12 
     13 
---> 14 model = XGBmodel(x_train, x_test, y_train, y_test, params)
     15 

/tmp/ipykernel_11/3929071051.py in XGBmodel(x_train, x_test, y_train, y_test, params)
      1 def XGBmodel(x_train, x_test, y_train, y_test, params):
----> 2     matrix_train = xgb.DMatrix(x_train, label=y_train)
      3     matrix_test = xgb.DMatrix(x_test, label=y_test)
      4     model = xgb.train(
      5         params=params,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1087         data = pd.DataFrame(data)
   1088     if _is_pandas_df(data):
-> 1089         return _from_pandas_df(
   1090             data, enable_categorical, missing, threads, feature_names, feature_types
   1091         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_pandas_df(data, enable_categorical, missing, nthread, feature_names, feature_types)
    520     feature_types: Optional[FeatureTypes],
    521 ) -> DispatchedDataBackendReturnType:
--> 522     data, feature_names, feature_types = _transform_pandas_df(
    523         data, enable_categorical, feature_names, feature_types
    524     )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:pickup_datetime: datetime64[ns, UTC]

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
    ]
)

prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit=model.best_ntree_limit)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/429303779.py in <cell line: 0>()
----> 1 x_pred = df_holdout.drop(
      2     columns=[
      3         "pickup_datetime",
      4         "pickup_distance_to_jfk",
      5         "drop_distance_to_jfk",

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr'] not found in axis"

## === cell 31
len(test_key)



## === cell 32
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(prediction, 2)})

submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)

submission.head()
print("Wrote submission to:", submission_path, "rows:", len(submission))

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/552852699.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(prediction, 2)})
      2 
      3 # Ensure correct filename and .csv suffix in Kaggle working directory
      4 submission_path = "taxi_fare_submission.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'prediction' is not defined
