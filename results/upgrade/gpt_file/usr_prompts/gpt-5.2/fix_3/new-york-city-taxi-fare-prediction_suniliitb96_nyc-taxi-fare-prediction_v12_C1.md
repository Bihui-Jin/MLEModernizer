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

5.83398

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.83398) has done: 'I fix the `haversine_np` function so it correctly handles scalar airport coordinates (floats) as well as vector inputs, which unblocks the airport-distance feature generation and downstream filtering. Then I ensure we drop non-numeric columns (notably `pickup_datetime`) from the XGBoost matrices and update deprecated XGBoost params (`reg:linear`, `silent`) so training runs in the current Kaggle runtime. Finally, I make the train/test feature columns align deterministically (same columns/order) and ensure a valid `taxi_fare_submission.csv` is written with `key,fare_amount` and the correct row count.'

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
    Inputs can be numpy arrays / pandas Series OR scalars.
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



## === cell 23
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds

df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[~airportTripsIds]



## === cell 24
pd.DataFrame(
    data={
        "Airport Trips": df_airport_trips.trip_rate,
        "City Trips": df_city_trips.trip_rate,
    }
).describe()



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
    ],
    errors="ignore",
)
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
    "objective": "reg:squarederror",  # was reg:linear (deprecated)
    "eval_metric": "rmse",
    "verbosity": 1,  # replaces silent
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
    df2 = df.copy()
    for c in df2.columns:
        if np.issubdtype(df2[c].dtype, np.datetime64):
            df2 = df2.drop(columns=[c])
    return df2


def XGBmodel(x_train, x_test, y_train, y_test, params):
    x_train_n = _numeric_only(x_train)
    x_test_n = _numeric_only(x_test)
    matrix_train = xgb.DMatrix(x_train_n, label=y_train)
    matrix_test = xgb.DMatrix(x_test_n, label=y_test)
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

train_feature_cols = _numeric_only(train).columns.tolist()
for c in train_feature_cols:
    if c not in x_pred.columns:
        x_pred[c] = 0
x_pred = x_pred[train_feature_cols]

prediction = model.predict(
    xgb.DMatrix(x_pred), iteration_range=(0, model.best_iteration + 1)
)



## === cell 31
len(test_key)



## === cell 32
submission = pd.DataFrame({"key": test_key, "fare_amount": np.round(prediction, 2)})

submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote submission to:", submission_path, "rows:", len(submission))
submission.head()
