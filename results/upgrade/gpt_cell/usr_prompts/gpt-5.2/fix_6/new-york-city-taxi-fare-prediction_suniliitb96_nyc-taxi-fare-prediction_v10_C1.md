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

EPS_IN_KM = 0.5           ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

RADIUS_VICINITY_AIRPORTS = 1.0

THERSHOLD_TRIP_FARE_RATE = 50.0


## === cell 2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')

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
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_key = df_test["key"]
df_train.drop(columns=["key"], inplace=True)
df_test.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 4
df_train.head()


## === cell 5
print('Old size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount >=0]

df_train = df_train.dropna(how='any', axis='rows')

df_train = df_train.drop(index= df_train[df_train.passenger_count >= 7].index, axis='rows')
df_train = df_train.drop(index= df_train[df_train.passenger_count == 0].index, axis='rows')

print('New size: %d' % len(df_train))


## === cell 6


def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            

BB = (-74.5, -72.8, 40.5, 41.8)

print('Old size: %d' % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print('New size: %d' % len(df_train))


## === cell 7
def addPickDropDistanceFeature(df):

    df['trip_distance'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude']))
        ),
        axis='columns'
    )
    return df

def addAirportDistanceFeatures(df):

    df['pickup_distance_to_jfk'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_jfk'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (JFK_GEO_LOCATION[0], JFK_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['pickup_distance_to_lgr'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_lgr'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (LGR_GEO_LOCATION[0], LGR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['pickup_distance_to_ewr'] = df.apply(
        (lambda row: haversine(
            (row['pickup_latitude'], row['pickup_longitude']),
            (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )

    df['drop_distance_to_ewr'] = df.apply(
        (lambda row: haversine(
            (row['dropoff_latitude'], row['dropoff_longitude']),
            (EWR_GEO_LOCATION[0], EWR_GEO_LOCATION[1]))
        ),
        axis='columns'
    )
    
    return df

def getAirportTrips(df, airportVicinity):
    ids = (df.pickup_distance_to_jfk < airportVicinity) | (df.drop_distance_to_jfk < airportVicinity) | (df.pickup_distance_to_lgr < airportVicinity) | (df.drop_distance_to_lgr < airportVicinity) | (df.pickup_distance_to_ewr < airportVicinity) | (df.drop_distance_to_ewr < airportVicinity)
    
    return ids


## === cell 8
if "haversine" not in globals():
    import math

    def haversine(point1, point2):
        lat1, lon1 = point1
        lat2, lon2 = point2

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (
            math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return KMS_PER_RADIAN * c


start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_test = addPickDropDistanceFeature(df_test)

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 9

bucketsCount = 100
feat = 'trip_distance'


df_train[feat].hist(bins=bucketsCount, figsize = (15,8))
df_test[feat].hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency Log")
'''
#newFeat = 'catFare'
#df_train[newFeat] = pd.cut(df_train[feat], custom_bucket_array)
#labels, levels = pd.factorize(df_train[newFeat])
#df_train[newFeat] = labels
'''


## === cell 10

print('Old size: %d' % len(df_train))
df_train = df_train[df_train.trip_distance < 25.0]
print('New size: %d' % len(df_train))


## === cell 11
'''
# Ceiling near-zero fare values to 0.2 to check fare/dist behavior
df_temp = df_train[['fare_amount', 'trip_distance']]
df_temp.loc[df_temp.trip_distance < 0.2, 'trip_distance'] = 0.2

(df_temp.fare_amount / df_temp.trip_distance).hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency")
'''


## === cell 12
'''
df_temp['trip_rate'] = df_temp.apply(
    (lambda row: (row.fare_amount / row.trip_distance)),
    axis='columns'
)
'''


## === cell 14
'''
ids = (df_temp.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))
'''


## === cell 16
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_test = addAirportDistanceFeatures(df_test)

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 17

airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
df_test['airport_bound'] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train['airport_bound'] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[-airportTripsIds]

pd.DataFrame(data={'Airport Trips' : df_airport_trips.fare_amount, 'City Trips' : df_city_trips.fare_amount}).describe()


## === cell 18
def add_datetime_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    df['hour'] = df.pickup_datetime.dt.hour
    df['day'] = df.pickup_datetime.dt.day
    df['month'] = df.pickup_datetime.dt.month
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['year'] = df.pickup_datetime.dt.year
    
    return df

start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_test = add_datetime_features(df_test)

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 19
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_test], axis=0, ignore_index=False, sort=False)
df_nyc_taxi.info()


## === cell 20
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN


## === cell 22
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'pickup_longitude':'pickup_latitude']))

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 23

mask_pick_all = np.zeros_like(dbscan_pick.labels_, dtype=bool)
mask_pick_all[dbscan_pick.core_sample_indices_] = True
labels_pick = dbscan_pick.labels_
df_nyc_taxi['is_good_dbscan_pick'] = (labels_pick != -1)

n_clusters_pick = len(set(labels_pick)) - (1 if -1 in labels_pick else 0)
n_clusters_pick


## === cell 24
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


## === cell 25
mask_dense_pick = (labels_pick != -1)
mask_rare_pick = (labels_pick == -1)

print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
df_train_dense_pick = df_nyc_taxi[mask_dense_pick]
df_train_rare_pick = df_nyc_taxi[mask_rare_pick]
print('df_train_dense_pick size: %d' % len(df_train_dense_pick))


## === cell 26
plt.plot(df_train_dense_pick.pickup_longitude, df_train_dense_pick.pickup_latitude, 'o')

'''
unique_labels = set(labels_pick)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

class_member_mask = (labels_pick == -1)

for k, col in zip(unique_labels, colors):
    if k == -1:
        break
        # Black used for noise.
        #col = [0, 0, 0, 1]

    class_member_mask = (labels_pick == k)

    xy = df_nyc_taxi[class_member_mask & mask_pick_all]
    plt.plot(xy.pickup_longitude, xy.pickup_latitude, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = df_nyc_taxi[class_member_mask & ~mask_pick_all]
    plt.plot(xy.pickup_longitude, xy.pickup_latitude, 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
'''


## === cell 27
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'dropoff_longitude':'dropoff_latitude']))

elapsed = timeit.default_timer() - start_time
elapsed


## === cell 28

labels_drop = dbscan_drop.labels_
df_nyc_taxi['is_good_dbscan_drop'] = (labels_drop != -1)

n_clusters_drop = len(set(labels_drop)) - (1 if -1 in labels_drop else 0)
n_clusters_drop


## === cell 29
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


## === cell 30
mask_dense_drop = (labels_drop != -1)
mask_rare_drop = (labels_drop == -1)

print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
df_train_dense_drop = df_nyc_taxi[mask_dense_drop]
df_train_rare_drop = df_nyc_taxi[mask_rare_drop]
print('df_train_dense_drop size: %d' % len(df_train_dense_drop))


## === cell 32
df_nyc_taxi['dense_DBSCAN_trips'] = df_nyc_taxi.apply(
    (lambda row: (row.is_good_dbscan_drop & row.is_good_dbscan_pick)),
    axis='columns'
)

len(df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1])


## === cell 33
df_nyc_taxi.head()


## === cell 34

df_train = df_nyc_taxi.iloc[:train_len, :]
df_test = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != 'fare_amount']

(len(df_train), len(df_test))


## === cell 35
pd.DataFrame(data={'Good Density Pickups' : df_train.loc[df_train.is_good_dbscan_pick == 1].fare_amount, 'LOW Density Pickups' : df_train.loc[df_train.is_good_dbscan_pick == 0].fare_amount}).describe()


## === cell 36
pd.DataFrame(data={'Good Density DropOffs' : df_train.loc[df_train.is_good_dbscan_drop == 1].fare_amount, 'LOW Density DropOffs' : df_train.loc[df_train.is_good_dbscan_drop == 0].fare_amount}).describe()


## === cell 37
df_temp = df_train[['fare_amount', 'trip_distance']]
df_temp.loc[df_temp.trip_distance < 0.2, 'trip_distance'] = 0.2

(df_temp.fare_amount / df_temp.trip_distance).hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency")


## === cell 38
df_temp['trip_rate'] = df_temp.apply(
    (lambda row: (row.fare_amount / row.trip_distance)),
    axis='columns'
)


## === cell 39
len(df_temp.loc[df_temp.trip_rate > THERSHOLD_TRIP_FARE_RATE])


## === cell 40
ids = (df_temp.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))


## === cell 41

df_tmp = df_train.drop(columns = ['pickup_datetime', 'is_good_dbscan_drop', 'is_good_dbscan_pick', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
df_tmp.info()


## === cell 42
y = df_tmp['fare_amount']
train = df_tmp.drop(columns=['fare_amount'])

x_train, x_test, y_train, y_test = train_test_split(train, y, random_state=0, test_size=0.01)


## === cell 43
params = {
    'max_depth': 8, #Result of tuning with CV
    'eta':.03, #Result of tuning with CV
    'subsample': 1, #Result of tuning with CV
    'colsample_bytree': 0.8, #Result of tuning with CV
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 1
}

CV=False
if CV:
    dtrain = xgb.DMatrix(train,label=y)
    gridsearch_params = [
        (eta)
        for eta in np.arange(.04, 0.12, .02)
    ]

    min_rmse = float("Inf")
    best_params = None
    for (eta) in gridsearch_params:
        print("CV with eta={} ".format(
                                 eta))

        params['eta'] = eta

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={'rmse'},
            early_stopping_rounds=10
        )

        mean_rmse = cv_results['test-rmse-mean'].min()
        boost_rounds = cv_results['test-rmse-mean'].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (eta)

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    params['silent'] = 0 #Turn on output
    print(params)


## === cell 44
def XGBmodel(x_train,x_test,y_train,y_test,params):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=5000, 
                    early_stopping_rounds=10,evals=[(matrix_test,'test')])
    return model

model = XGBmodel(x_train,x_test,y_train,y_test,params)


## === cell 45
x_pred = df_test.drop(columns = ['pickup_datetime', 'is_good_dbscan_drop', 'is_good_dbscan_pick', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 45, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3589173125.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0;31m#Predict from test set[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mprediction[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mx_pred[0m[0;34m)[0m[0;34m,[0m [0mntree_limit[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mbest_ntree_limit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 46
len(test_key)
