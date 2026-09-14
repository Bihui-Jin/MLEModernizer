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

3.95939

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



## === cell 2
import os
import timeit

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn import metrics

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")



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

df_train = pd.read_csv(
    TRAIN_PATH, nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_test = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])

test_key = df_test["key"].copy()

df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 5
df_train.head()



## === cell 6
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
df_train = df_train[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))




## === cell 8
def _haversine_vec_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in kilometers.
    Inputs are in decimal degrees; outputs in km.
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
    c = 2.0 * np.arcsin(np.sqrt(a))
    return KMS_PER_RADIAN * c


def addPickDropDistanceFeature(df):
    df = df.copy()
    df["trip_distance"] = _haversine_vec_km(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    return df


def addAirportDistanceFeatures(df):
    df = df.copy()

    df["pickup_distance_to_jfk"] = _haversine_vec_km(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )
    df["drop_distance_to_jfk"] = _haversine_vec_km(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )

    df["pickup_distance_to_lgr"] = _haversine_vec_km(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )
    df["drop_distance_to_lgr"] = _haversine_vec_km(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )

    df["pickup_distance_to_ewr"] = _haversine_vec_km(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    df["drop_distance_to_ewr"] = _haversine_vec_km(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
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




## === cell 9
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_test = addPickDropDistanceFeature(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 10
bucketsCount = 100
feat = "trip_distance"

_ = df_train[feat].hist(bins=bucketsCount, figsize=(15, 8))
_ = df_test[feat].hist(bins=bucketsCount, figsize=(15, 8))
plt.yscale("log")
plt.xlabel(feat)
plt.ylabel("Frequency Log")
plt.close()



## === cell 11
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.trip_distance < 25.0]
print("New size: %d" % len(df_train))



## === cell 12
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_test = addAirportDistanceFeatures(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1930263541.py in <cell line: 0>()
      1 start_time = timeit.default_timer()
      2 
----> 3 df_train = addAirportDistanceFeatures(df_train)
      4 df_test = addAirportDistanceFeatures(df_test)
      5 

/tmp/ipykernel_11/1899439894.py in addAirportDistanceFeatures(df)
     34     df = df.copy()
     35 
---> 36     df["pickup_distance_to_jfk"] = _haversine_vec_km(
     37         df["pickup_latitude"].values,
     38         df["pickup_longitude"].values,

/tmp/ipykernel_11/1899439894.py in _haversine_vec_km(lat1, lon1, lat2, lon2)
      7     lat1 = np.radians(lat1.astype(float))
      8     lon1 = np.radians(lon1.astype(float))
----> 9     lat2 = np.radians(lat2.astype(float))
     10     lon2 = np.radians(lon2.astype(float))
     11 

AttributeError: 'float' object has no attribute 'astype'

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




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/965272535.py in <cell line: 0>()
      1 # Fix: boolean negation was wrong in original (-airportTripsIds). Use ~airportTripsIds.
----> 2 airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
      3 df_test["airport_bound"] = airportTripsIds
      4 
      5 airportTripsIds_train = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)

/tmp/ipykernel_11/1899439894.py in getAirportTrips(df, airportVicinity)
     78 def getAirportTrips(df, airportVicinity):
     79     ids = (
---> 80         (df.pickup_distance_to_jfk < airportVicinity)
     81         | (df.drop_distance_to_jfk < airportVicinity)
     82         | (df.pickup_distance_to_lgr < airportVicinity)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'pickup_distance_to_jfk'

## === cell 14
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
df_test = add_datetime_features(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 15
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_test], axis=0, ignore_index=False, sort=False)
df_nyc_taxi.info()



## === cell 16
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN
EPS_IN_RADIAN



## === cell 17
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(
    np.radians(
        df_nyc_taxi[["pickup_latitude", "pickup_longitude"]].astype(float).values
    )
)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 18
mask_pick_all = np.zeros_like(dbscan_pick.labels_, dtype=bool)
mask_pick_all[dbscan_pick.core_sample_indices_] = True
labels_pick = dbscan_pick.labels_
df_nyc_taxi["is_good_dbscan_pick"] = labels_pick != -1

n_clusters_pick = len(set(labels_pick)) - (1 if -1 in labels_pick else 0)
n_clusters_pick



## === cell 19
try:
    print("Pickup: DBSCAN clustering accuracy indicators")
    print("Estimated number of clusters: %d" % n_clusters_pick)
    train_mask = df_nyc_taxi["fare_amount"].notna().values
    print(
        "Homogeneity: %0.3f"
        % metrics.homogeneity_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_pick[train_mask]
        )
    )
    print(
        "Completeness: %0.3f"
        % metrics.completeness_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_pick[train_mask]
        )
    )
    print(
        "V-measure: %0.3f"
        % metrics.v_measure_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_pick[train_mask]
        )
    )
    print(
        "Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_pick[train_mask]
        )
    )
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_pick[train_mask]
        )
    )
except Exception as e:
    print("Skipping clustering indicator metrics due to:", repr(e))



## === cell 20
mask_dense_pick = labels_pick != -1
mask_rare_pick = labels_pick == -1

print("df_nyc_taxi size: %d" % len(df_nyc_taxi))
df_train_dense_pick = df_nyc_taxi[mask_dense_pick]
df_train_rare_pick = df_nyc_taxi[mask_rare_pick]
print("df_train_dense_pick size: %d" % len(df_train_dense_pick))



## === cell 21
plt.figure(figsize=(8, 6))
plt.plot(
    df_train_dense_pick.pickup_longitude,
    df_train_dense_pick.pickup_latitude,
    "o",
    markersize=1,
    alpha=0.2,
)
plt.close()



## === cell 22
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(
    np.radians(
        df_nyc_taxi[["dropoff_latitude", "dropoff_longitude"]].astype(float).values
    )
)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 23
labels_drop = dbscan_drop.labels_
df_nyc_taxi["is_good_dbscan_drop"] = labels_drop != -1

n_clusters_drop = len(set(labels_drop)) - (1 if -1 in labels_drop else 0)
n_clusters_drop



## === cell 24
try:
    print("DropOff: DBSCAN clustering accuracy indicators")
    print("Estimated number of clusters: %d" % n_clusters_drop)
    train_mask = df_nyc_taxi["fare_amount"].notna().values
    print(
        "Homogeneity: %0.3f"
        % metrics.homogeneity_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_drop[train_mask]
        )
    )
    print(
        "Completeness: %0.3f"
        % metrics.completeness_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_drop[train_mask]
        )
    )
    print(
        "V-measure: %0.3f"
        % metrics.v_measure_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_drop[train_mask]
        )
    )
    print(
        "Adjusted Rand Index: %0.3f"
        % metrics.adjusted_rand_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_drop[train_mask]
        )
    )
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics.adjusted_mutual_info_score(
            df_nyc_taxi.loc[train_mask, "fare_amount"], labels_drop[train_mask]
        )
    )
except Exception as e:
    print("Skipping clustering indicator metrics due to:", repr(e))



## === cell 25
mask_dense_drop = labels_drop != -1
mask_rare_drop = labels_drop == -1

print("df_nyc_taxi size: %d" % len(df_nyc_taxi))
df_train_dense_drop = df_nyc_taxi[mask_dense_drop]
df_train_rare_drop = df_nyc_taxi[mask_rare_drop]
print("df_train_dense_drop size: %d" % len(df_train_dense_drop))



## === cell 26
df_nyc_taxi["dense_DBSCAN_trips"] = (
    df_nyc_taxi["is_good_dbscan_drop"] & df_nyc_taxi["is_good_dbscan_pick"]
).astype(bool)
len(df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1])



## === cell 27
df_nyc_taxi.head()



## === cell 28
df_train = df_nyc_taxi.iloc[:train_len, :].copy()
df_test = (
    df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != "fare_amount"].copy()
)

(len(df_train), len(df_test))



## === cell 29
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



## === cell 30
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



## === cell 31
df_temp = df_train[["fare_amount", "trip_distance"]].copy()
df_temp.loc[df_temp.trip_distance < 0.2, "trip_distance"] = 0.2

trip_rate = (df_temp["fare_amount"] / df_temp["trip_distance"]).values
df_temp["trip_rate"] = trip_rate

len(df_temp.loc[df_temp.trip_rate > THERSHOLD_TRIP_FARE_RATE])



## === cell 32
ids = df_temp.trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train.loc[ids.values].copy()
print("New size: %d" % len(df_train))



## === cell 33
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

df_tmp = df_train.drop(columns=drop_cols)
df_tmp.info()



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1690813858.py in <cell line: 0>()
     16 ]
     17 
---> 18 df_tmp = df_train.drop(columns=drop_cols)
     19 df_tmp.info()
     20 

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

## === cell 34
y = df_tmp["fare_amount"]
train = df_tmp.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.01
)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1502404153.py in <cell line: 0>()
----> 1 y = df_tmp["fare_amount"]
      2 train = df_tmp.drop(columns=["fare_amount"])
      3 
      4 x_train, x_test, y_train, y_test = train_test_split(
      5     train, y, random_state=0, test_size=0.01

NameError: name 'df_tmp' is not defined

## === cell 35
params = {
    "max_depth": 8,  # Result of tuning with CV
    "eta": 0.03,  # Result of tuning with CV
    "subsample": 1,  # Result of tuning with CV
    "colsample_bytree": 0.8,  # Result of tuning with CV
    "objective": "reg:squarederror",  # Fix: 'reg:linear' deprecated; equivalent for RMSE regression.
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




## === cell 36
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



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3929071051.py in <cell line: 0>()
     12 
     13 
---> 14 model = XGBmodel(x_train, x_test, y_train, y_test, params)
     15 

NameError: name 'x_train' is not defined

## === cell 37
x_pred = df_test.drop(columns=drop_cols)

prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit=model.best_ntree_limit)



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1975980767.py in <cell line: 0>()
----> 1 x_pred = df_test.drop(columns=drop_cols)
      2 
      3 prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit=model.best_ntree_limit)
      4 

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

## === cell 38
len(test_key), len(prediction)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3028859299.py in <cell line: 0>()
----> 1 len(test_key), len(prediction)
      2 

NameError: name 'prediction' is not defined

## === cell 39
submission = pd.DataFrame(
    {
        "key": test_key.values,
        "fare_amount": np.round(prediction, 2),
    }
)

submission["fare_amount"] = submission["fare_amount"].clip(lower=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
submission.head()

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/236037854.py in <cell line: 0>()
      3     {
      4         "key": test_key.values,
----> 5         "fare_amount": np.round(prediction, 2),
      6     }
      7 )

NameError: name 'prediction' is not defined
