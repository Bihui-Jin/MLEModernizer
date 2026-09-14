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

3.95459

# 6. Current score

5.53504

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.61708) has done: 'Diagnosis: The crash happens because this environment’s `xgboost.Booster` object does not expose `best_ntree_limit` (API/version difference), even though early stopping was used during training. Using that attribute is therefore unsafe and raises `AttributeError`.  
Patch summary: In cell 31 only, keep the same prediction logic but make it robust by using `best_ntree_limit` when available, otherwise falling back to `best_iteration` (converted to a valid `iteration_range`) or finally defaulting to a plain `predict` with no limit. This preserves the intended “use best model from early stopping” behavior as closely as possible without changing model/training.  
Updated cells: Only cell 31 is changed.  
Compatibility notes for cell k+1: Variables `prediction` and `test_key` remain unchanged in name/type; cell 32 (`len(test_key)`) is unaffected.  
Assumptions: `xgboost` is installed and provides either `best_ntree_limit` or `best_iteration` when early stopping is used; if neither exists, predicting with the full model is acceptable to keep execution unblocked.'
- What this solution (achieved 5.61708) has done: 'Your current RMSE (5.617) is far from the target (3.954) so we need a small but meaningful quality improvement without changing the model/training loop. The biggest low-risk gain here is fixing a bug in the airport/city split: `df_train.loc[-airportTripsIds]` is incorrect boolean negation and silently breaks the intended feature/analysis logic; switching to `~airportTripsIds` restores correct semantics and prevents downstream oddities. Next, the datetime parsing format is too strict and can create NaTs or inconsistencies; letting pandas parse the already-parsed datetimes more safely keeps the same features but avoids accidental missing values. Finally, because RMSE penalizes negatives and XGBoost can predict them, clipping predictions to a small non-negative floor is a minimal post-processing step that typically improves public RMSE for this competition without altering training.'
- What this solution (achieved 5.61708) has done: 'Your current RMSE (5.617) is worse than the target (3.955), so we should make a small, low-risk improvement that doesn’t change the core model/training loop. The biggest likely gain is preventing train/test feature mismatch by ensuring the exact same feature columns (and order) are used for training and test prediction; right now `airport_bound` and `dense_DBSCAN_trips` can behave inconsistently if any column gets dropped differently. Next, we remove the rounding-to-cents in the submission (rounding almost always slightly worsens RMSE) while keeping the same predictions otherwise. Finally, we add a minimal, competition-standard upper clip (e.g., 500) in addition to the non-negative clip to reduce the impact of rare extreme predictions on RMSE without altering training.'
- What this solution (achieved 5.53504) has done: 'Your current RMSE (5.617) is worse than the target (3.955), so we should make a small, low-risk improvement without changing the XGBoost training loop or features used. The biggest likely win is fixing the DBSCAN coordinate slicing bug: `loc[:, "pickup_longitude":"pickup_latitude"]` is an invalid column range (end column comes before start), which can silently produce wrong/no clustering features; switching to an explicit column list makes `dense_DBSCAN_trips` meaningful again. Next, remove plotting cells’ heavy work by gating them (no effect on predictions) to help the notebook finish reliably within time. Finally, keep the existing train/test column alignment and prediction clipping unchanged to preserve evaluation semantics while benefiting from the corrected clustering feature.'

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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import timeit
from sklearn import metrics

from haversine import haversine



## === cell 3
import timeit

start_time = timeit.default_timer()

df_train = pd.read_csv(
    "../input/train.csv", nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_holdout = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_key = df_holdout["key"]
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

    df["trip_distance"] = df.apply(
        (
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (row["dropoff_latitude"], row["dropoff_longitude"]),
            )
        ),
        axis="columns",
    )
    return df


def addAirportDistanceFeatures(df):

    df["pickup_distance_to_jfk"] = df.apply(
        (
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]),
            )
        ),
        axis="columns",
    )

    df["drop_distance_to_jfk"] = df.apply(
        (
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]),
                (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]),
            )
        ),
        axis="columns",
    )

    df["pickup_distance_to_lgr"] = df.apply(
        (
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]),
            )
        ),
        axis="columns",
    )

    df["drop_distance_to_lgr"] = df.apply(
        (
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]),
                (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]),
            )
        ),
        axis="columns",
    )

    df["pickup_distance_to_ewr"] = df.apply(
        (
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]),
            )
        ),
        axis="columns",
    )

    df["drop_distance_to_ewr"] = df.apply(
        (
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]),
                (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]),
            )
        ),
        axis="columns",
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

try:
    haversine  # noqa: F821
except NameError:
    import math

    def haversine(point1, point2):
        lat1, lon1 = point1
        lat2, lon2 = point2

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * math.asin(math.sqrt(a))

        return KMS_PER_RADIAN * c


df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 8
DO_PLOTS = False

bucketsCount = 100
feat = "trip_distance"

if DO_PLOTS:
    df_train[feat].hist(bins=bucketsCount, figsize=(15, 8))
    df_holdout[feat].hist(bins=bucketsCount, figsize=(15, 8))
    plt.yscale("log")
    plt.xlabel(feat)
    plt.ylabel("Frequency Log")



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

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 11
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)



## === cell 12
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 13
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(np.radians(df_nyc_taxi[["pickup_latitude", "pickup_longitude"]].values))
labels_pick = dbscan_pick.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 14
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(np.radians(df_nyc_taxi[["dropoff_latitude", "dropoff_longitude"]].values))
labels_drop = dbscan_drop.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 15
df_nyc_taxi["dense_DBSCAN_trips"] = (labels_pick != -1) & (labels_drop != -1)



## === cell 16
df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
if DO_PLOTS:
    plt.plot(df_tmp.pickup_longitude, df_tmp.pickup_latitude, "o")



## === cell 17
df_train = df_nyc_taxi.iloc[:train_len, :]
df_holdout = df_nyc_taxi.iloc[train_len:, :].iloc[
    :, df_nyc_taxi.columns != "fare_amount"
]

(len(df_train), len(df_holdout))



## === cell 18
df_train.loc[df_train.trip_distance < 0.2, "trip_distance"] = 0.2

if DO_PLOTS:
    (df_train.fare_amount / df_train.trip_distance).hist(
        bins=bucketsCount, figsize=(15, 8)
    )
    plt.yscale("log")
    plt.xlabel("trip_rate")
    plt.ylabel("Log Frequency")



## === cell 19
df_train["trip_rate"] = df_train.apply(
    (lambda row: (row.fare_amount / row.trip_distance)), axis="columns"
)



## === cell 20
"""
ids = (df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))
"""



## === cell 21
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 22
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]

df_city_trips = df_train.loc[~airportTripsIds]



## === cell 23
if DO_PLOTS:
    pd.DataFrame(
        data={
            "Airport Trips": df_airport_trips.trip_rate,
            "City Trips": df_city_trips.trip_rate,
        }
    ).describe()



## === cell 24
if DO_PLOTS:
    pd.DataFrame(
        data={
            "Good Density Trips": df_train.loc[
                df_train.dense_DBSCAN_trips == 1
            ].trip_rate,
            "LOW Density Pickups": df_train.loc[
                df_train.dense_DBSCAN_trips == 0
            ].trip_rate,
        }
    ).describe()



## === cell 25
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



## === cell 26
y = df_train["fare_amount"]
train = df_train.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.01
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



## === cell 29
x_pred = df_holdout.drop(
    columns=[c for c in DROP_COLS if c in df_holdout.columns], errors="ignore"
)

x_pred = x_pred.reindex(columns=train.columns, fill_value=0)

dtest = xgb.DMatrix(x_pred)
if hasattr(model, "best_ntree_limit"):
    prediction = model.predict(dtest, ntree_limit=model.best_ntree_limit)
elif hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(
        dtest, iteration_range=(0, int(model.best_iteration) + 1)
    )
else:
    prediction = model.predict(dtest)

prediction = np.clip(prediction, 0.0, 500.0)



## === cell 30
len(test_key)



## === cell 31
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
