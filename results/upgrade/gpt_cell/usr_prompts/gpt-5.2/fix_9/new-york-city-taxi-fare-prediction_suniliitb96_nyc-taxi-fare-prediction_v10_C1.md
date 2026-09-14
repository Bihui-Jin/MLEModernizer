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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import timeit
from sklearn import metrics



## === cell 3
import timeit

start_time = timeit.default_timer()

df_train = pd.read_csv(
    "../input/train.csv", nrows=MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"]
)
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_key = df_test["key"]
df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 4
df_train.head()



## === cell 5
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
df_train = df_train[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))




## === cell 7
def haversine_np(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype("float64"))
    lon1 = np.radians(lon1.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return KMS_PER_RADIAN * c


def addPickDropDistanceFeature(df):
    df["trip_distance"] = haversine_np(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    return df


def addAirportDistanceFeatures(df):
    df["pickup_distance_to_jfk"] = haversine_np(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )
    df["drop_distance_to_jfk"] = haversine_np(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )

    df["pickup_distance_to_lgr"] = haversine_np(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )
    df["drop_distance_to_lgr"] = haversine_np(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )

    df["pickup_distance_to_ewr"] = haversine_np(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    df["drop_distance_to_ewr"] = haversine_np(
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




## === cell 8
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_test = addPickDropDistanceFeature(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 9
bucketsCount = 100
feat = "trip_distance"

df_train[feat].hist(bins=bucketsCount, figsize=(15, 8))
df_test[feat].hist(bins=bucketsCount, figsize=(15, 8))
plt.yscale("log")
plt.xlabel(feat)
plt.ylabel("Frequency Log")



## === cell 10
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.trip_distance < 25.0]
print("New size: %d" % len(df_train))



## === cell 11
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_test = addAirportDistanceFeatures(df_test)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 12
airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
df_test["airport_bound"] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[~airportTripsIds]

pd.DataFrame(
    data={
        "Airport Trips": df_airport_trips.fare_amount,
        "City Trips": df_city_trips.fare_amount,
    }
).describe()




## === cell 13
def add_datetime_features(df):
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



## === cell 14
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_test], axis=0, ignore_index=False, sort=False)
df_nyc_taxi.info()



## === cell 15
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 16
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(np.radians(df_nyc_taxi.loc[:, "pickup_longitude":"pickup_latitude"]))

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 17
mask_pick_all = np.zeros_like(dbscan_pick.labels_, dtype=bool)
mask_pick_all[dbscan_pick.core_sample_indices_] = True
labels_pick = dbscan_pick.labels_
df_nyc_taxi["is_good_dbscan_pick"] = labels_pick != -1

n_clusters_pick = len(set(labels_pick)) - (1 if -1 in labels_pick else 0)
n_clusters_pick



## === cell 18
from sklearn import metrics

print("Pickup: DBSCAN clustering accuracy indicators")
print("Estimated number of clusters: %d" % n_clusters_pick)

valid_fare_mask = df_nyc_taxi["fare_amount"].notna().values
fare_valid = df_nyc_taxi.loc[valid_fare_mask, "fare_amount"].values
labels_pick_valid = labels_pick[valid_fare_mask]

print("Homogeneity: %0.3f" % metrics.homogeneity_score(fare_valid, labels_pick_valid))
print("Completeness: %0.3f" % metrics.completeness_score(fare_valid, labels_pick_valid))
print("V-measure: %0.3f" % metrics.v_measure_score(fare_valid, labels_pick_valid))
print(
    "Adjusted Rand Index: %0.3f"
    % metrics.adjusted_rand_score(fare_valid, labels_pick_valid)
)
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(fare_valid, labels_pick_valid)
)

start_time = timeit.default_timer()
elapsed = timeit.default_timer() - start_time
elapsed



## === cell 19
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(np.radians(df_nyc_taxi.loc[:, "dropoff_longitude":"dropoff_latitude"]))

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 20
labels_drop = dbscan_drop.labels_
df_nyc_taxi["is_good_dbscan_drop"] = labels_drop != -1

n_clusters_drop = len(set(labels_drop)) - (1 if -1 in labels_drop else 0)
n_clusters_drop



## === cell 21
print("DropOff: DBSCAN clustering accuracy indicators")
print("Estimated number of clusters: %d" % n_clusters_drop)

valid_fare_mask = df_nyc_taxi["fare_amount"].notna().values
fare_valid = df_nyc_taxi.loc[valid_fare_mask, "fare_amount"].values
labels_drop_valid = labels_drop[valid_fare_mask]

print("Homogeneity: %0.3f" % metrics.homogeneity_score(fare_valid, labels_drop_valid))
print("Completeness: %0.3f" % metrics.completeness_score(fare_valid, labels_drop_valid))
print("V-measure: %0.3f" % metrics.v_measure_score(fare_valid, labels_drop_valid))
print(
    "Adjusted Rand Index: %0.3f"
    % metrics.adjusted_rand_score(fare_valid, labels_drop_valid)
)
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(fare_valid, labels_drop_valid)
)

start_time = timeit.default_timer()
elapsed = timeit.default_timer() - start_time
elapsed



## === cell 22
df_nyc_taxi["dense_DBSCAN_trips"] = df_nyc_taxi.apply(
    (lambda row: (row.is_good_dbscan_drop & row.is_good_dbscan_pick)), axis="columns"
)

len(df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1])



## === cell 23
df_train = df_nyc_taxi.iloc[:train_len, :]
df_test = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != "fare_amount"]

(len(df_train), len(df_test))



## === cell 24
df_temp = df_train[["fare_amount", "trip_distance"]].copy()
df_temp.loc[df_temp.trip_distance < 0.2, "trip_distance"] = 0.2

(df_temp.fare_amount / df_temp.trip_distance).hist(bins=bucketsCount, figsize=(15, 8))
plt.yscale("log")
plt.xlabel(feat)
plt.ylabel("Frequency")



## === cell 25
df_temp["trip_rate"] = df_temp.apply(
    (lambda row: (row.fare_amount / row.trip_distance)), axis="columns"
)



## === cell 26
ids = df_temp.trip_rate < THERSHOLD_TRIP_FARE_RATE

print("Old size: %d" % len(df_train))
df_train = df_train[ids]
print("New size: %d" % len(df_train))



## === cell 27
df_tmp = df_train.drop(
    columns=[
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
)
df_tmp.info()



## === cell 28
y = df_tmp["fare_amount"]
train = df_tmp.drop(columns=["fare_amount"])

x_train, x_valid, y_train, y_valid = train_test_split(
    train, y, random_state=0, test_size=0.2
)



## === cell 29
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




## === cell 30
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


model = XGBmodel(x_train, x_valid, y_train, y_valid, params)




## === cell 31
def get_best_num_boost_round(m):
    if hasattr(m, "best_iteration") and m.best_iteration is not None:
        return int(m.best_iteration) + 1
    if hasattr(m, "best_ntree_limit") and m.best_ntree_limit is not None:
        return int(m.best_ntree_limit)
    return None


best_num_boost_round = get_best_num_boost_round(model)

dtrain_full = xgb.DMatrix(train, label=y)

if best_num_boost_round is None:
    model_full = xgb.train(params=params, dtrain=dtrain_full, num_boost_round=5000)
else:
    model_full = xgb.train(
        params=params, dtrain=dtrain_full, num_boost_round=best_num_boost_round
    )



## === cell 32
x_pred = df_test.drop(
    columns=[
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
)

dmatrix_pred = xgb.DMatrix(x_pred)

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



## === cell 33
len(test_key)



## === cell 34
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
