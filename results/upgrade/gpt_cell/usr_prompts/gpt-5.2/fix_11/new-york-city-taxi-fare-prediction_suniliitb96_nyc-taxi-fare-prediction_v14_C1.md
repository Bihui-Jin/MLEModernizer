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

from haversine import haversine, Unit



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
                unit=Unit.KILOMETERS,
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
                unit=Unit.KILOMETERS,
            )
        ),
        axis="columns",
    )

    df["drop_distance_to_jfk"] = df.apply(
        (
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]),
                (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]),
                unit=Unit.KILOMETERS,
            )
        ),
        axis="columns",
    )

    df["pickup_distance_to_lgr"] = df.apply(
        (
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]),
                unit=Unit.KILOMETERS,
            )
        ),
        axis="columns",
    )

    df["drop_distance_to_lgr"] = df.apply(
        (
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]),
                (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]),
                unit=Unit.KILOMETERS,
            )
        ),
        axis="columns",
    )

    df["pickup_distance_to_ewr"] = df.apply(
        (
            lambda row: haversine(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]),
                unit=Unit.KILOMETERS,
            )
        ),
        axis="columns",
    )

    df["drop_distance_to_ewr"] = df.apply(
        (
            lambda row: haversine(
                (row["dropoff_latitude"], row["dropoff_longitude"]),
                (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]),
                unit=Unit.KILOMETERS,
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

df_holdout["trip_distance"] = df_holdout["trip_distance"].clip(
    upper=THRESHOLD_TRIP_DISTANCE
)




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

dt_cols = ["hour", "day", "month", "weekday", "year"]
df_train = df_train.dropna(subset=["pickup_datetime"] + dt_cols)

train_dt_medians = df_train[dt_cols].median()
for c in dt_cols:
    df_holdout[c] = df_holdout[c].fillna(train_dt_medians[c])

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 11
df_holdout = df_holdout.dropna(how="any", axis="rows")
df_holdout = df_holdout[
    (df_holdout.passenger_count > 0) & (df_holdout.passenger_count < 7)
]
df_holdout = df_holdout[select_within_boundingbox(df_holdout, BB)].copy()



## === cell 12
train_len = len(df_train)

df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=True, sort=False)



## === cell 13
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 14
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



## === cell 15
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



## === cell 16
df_nyc_taxi["dense_DBSCAN_trips"] = ((labels_pick != -1) & (labels_drop != -1)).astype(
    np.int8
)



## === cell 17
df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
if DO_PLOTS:
    plt.plot(df_tmp.pickup_longitude, df_tmp.pickup_latitude, "o")



## === cell 18
df_train = df_nyc_taxi.iloc[:train_len, :].copy()
df_holdout = (
    df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != "fare_amount"].copy()
)

(len(df_train), len(df_holdout))



## === cell 19
df_train.loc[df_train.trip_distance < 0.2, "trip_distance"] = 0.2

if DO_PLOTS:
    (df_train.fare_amount / df_train.trip_distance).hist(
        bins=bucketsCount, figsize=(15, 8)
    )
    plt.yscale("log")
    plt.xlabel("trip_rate")
    plt.ylabel("Log Frequency")



## === cell 20
df_train["trip_rate"] = df_train.apply(
    (lambda row: (row.fare_amount / row.trip_distance)), axis="columns"
)



## === cell 21
"""
ids = (df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))
"""



## === cell 22
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 23
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds.astype(np.int8)

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds.astype(np.int8)
df_airport_trips = df_train.loc[airportTripsIds]

df_city_trips = df_train.loc[~airportTripsIds]



## === cell 24
if DO_PLOTS:
    pd.DataFrame(
        data={
            "Airport Trips": df_airport_trips.trip_rate,
            "City Trips": df_city_trips.trip_rate,
        }
    ).describe()



## === cell 25
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



## === cell 26
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



## === cell 31
len(test_key)



## === cell 32
holdout_idx = df_holdout.index.to_numpy()
order = np.argsort(holdout_idx)

submission = pd.DataFrame(
    {
        "key": test_key.iloc[holdout_idx[order]].values,
        "fare_amount": np.asarray(prediction)[order],
    }
)

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()


## --- ERROR in cell 32, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_list_axis[0;34m(self, key, axis)[0m
[1;32m   1713[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1714[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_take_with_is_copy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1715[0m         [0;32mexcept[0m [0mIndexError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_take_with_is_copy[0;34m(self, indices, axis)[0m
[1;32m   4152[0m         """
[0;32m-> 4153[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0mindices[0m[0;34m=[0m[0mindices[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4154[0m         [0;31m# Maybe set copy if we didn't actually change the index.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mtake[0;34m(self, indices, axis, **kwargs)[0m
[1;32m   4132[0m [0;34m[0m[0m
[0;32m-> 4133[0;31m         new_data = self._mgr.take(
[0m[1;32m   4134[0m             [0mindices[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mtake[0;34m(self, indexer, axis, verify)[0m
[1;32m    890[0m         [0mn[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0maxis[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 891[0;31m         [0mindexer[0m [0;34m=[0m [0mmaybe_convert_indices[0m[0;34m([0m[0mindexer[0m[0;34m,[0m [0mn[0m[0;34m,[0m [0mverify[0m[0;34m=[0m[0mverify[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    892[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexers/utils.py[0m in [0;36mmaybe_convert_indices[0;34m(indices, n, verify)[0m
[1;32m    281[0m         [0;32mif[0m [0mmask[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m             [0;32mraise[0m [0mIndexError[0m[0;34m([0m[0;34m"indices are out-of-bounds"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m     [0;32mreturn[0m [0mindices[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: indices are out-of-bounds

The above exception was the direct cause of the following exception:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1388262010.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m submission = pd.DataFrame(
[1;32m      8[0m     {
[0;32m----> 9[0;31m         [0;34m"key"[0m[0;34m:[0m [0mtest_key[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mholdout_idx[0m[0;34m[[0m[0morder[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m         [0;34m"fare_amount"[0m[0;34m:[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mprediction[0m[0;34m)[0m[0;34m[[0m[0morder[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m     }

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1189[0m             [0mmaybe_callable[0m [0;34m=[0m [0mcom[0m[0;34m.[0m[0mapply_if_callable[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1190[0m             [0mmaybe_callable[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_check_deprecated_callable_usage[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmaybe_callable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1191[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_axis[0m[0;34m([0m[0mmaybe_callable[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1192[0m [0;34m[0m[0m
[1;32m   1193[0m     [0;32mdef[0m [0m_is_scalar_access[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m:[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_axis[0;34m(self, key, axis)[0m
[1;32m   1741[0m         [0;31m# a list of integers[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1742[0m         [0;32melif[0m [0mis_list_like_indexer[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1743[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_get_list_axis[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1744[0m [0;34m[0m[0m
[1;32m   1745[0m         [0;31m# a single integer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_list_axis[0;34m(self, key, axis)[0m
[1;32m   1715[0m         [0;32mexcept[0m [0mIndexError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1716[0m             [0;31m# re-raise with different error message, e.g. test_getitem_ndarray_3d[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1717[0;31m             [0;32mraise[0m [0mIndexError[0m[0;34m([0m[0;34m"positional indexers are out-of-bounds"[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1718[0m [0;34m[0m[0m
[1;32m   1719[0m     [0;32mdef[0m [0m_getitem_axis[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m,[0m [0maxis[0m[0;34m:[0m [0mAxisInt[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: positional indexers are out-of-bounds
