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
print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_nyc_taxi.fare_amount, labels_drop))
print("Completeness: %0.3f" % metrics.completeness_score(df_nyc_taxi.fare_amount, labels_drop))
print("V-measure: %0.3f" % metrics.v_measure_score(df_nyc_taxi.fare_amount, labels_drop))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(df_nyc_taxi.fare_amount, labels_drop))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(df_nyc_taxi.fare_amount, labels_drop))

start_time = timeit.default_timer()
elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 29, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/316351583.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mprint[0m[0;34m([0m[0;34m"DropOff: DBSCAN clustering accuracy indicators"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mprint[0m[0;34m([0m[0;34m"Estimated number of clusters: %d"[0m [0;34m%[0m [0mn_clusters_drop[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mprint[0m[0;34m([0m[0;34m"Homogeneity: %0.3f"[0m [0;34m%[0m [0mmetrics[0m[0;34m.[0m[0mhomogeneity_score[0m[0;34m([0m[0mdf_nyc_taxi[0m[0;34m.[0m[0mfare_amount[0m[0;34m,[0m [0mlabels_drop[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mprint[0m[0;34m([0m[0;34m"Completeness: %0.3f"[0m [0;34m%[0m [0mmetrics[0m[0;34m.[0m[0mcompleteness_score[0m[0;34m([0m[0mdf_nyc_taxi[0m[0;34m.[0m[0mfare_amount[0m[0;34m,[0m [0mlabels_drop[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mprint[0m[0;34m([0m[0;34m"V-measure: %0.3f"[0m [0;34m%[0m [0mmetrics[0m[0;34m.[0m[0mv_measure_score[0m[0;34m([0m[0mdf_nyc_taxi[0m[0;34m.[0m[0mfare_amount[0m[0;34m,[0m [0mlabels_drop[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/metrics/cluster/_supervised.py[0m in [0;36mhomogeneity_score[0;34m(labels_true, labels_pred)[0m
[1;32m    562[0m       [0;36m0.0[0m[0;34m...[0m[0;34m[0m[0;34m[0m[0m
[1;32m    563[0m     """
[0;32m--> 564[0;31m     [0;32mreturn[0m [0mhomogeneity_completeness_v_measure[0m[0;34m([0m[0mlabels_true[0m[0;34m,[0m [0mlabels_pred[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    565[0m [0;34m[0m[0m
[1;32m    566[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/metrics/cluster/_supervised.py[0m in [0;36mhomogeneity_completeness_v_measure[0;34m(labels_true, labels_pred, beta)[0m
[1;32m    469[0m     [0mv_measure_score[0m [0;34m:[0m [0mV[0m[0;34m-[0m[0mMeasure[0m [0;34m([0m[0mNMI[0m [0;32mwith[0m [0marithmetic[0m [0mmean[0m [0moption[0m[0;34m)[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    470[0m     """
[0;32m--> 471[0;31m     [0mlabels_true[0m[0;34m,[0m [0mlabels_pred[0m [0;34m=[0m [0mcheck_clusterings[0m[0;34m([0m[0mlabels_true[0m[0;34m,[0m [0mlabels_pred[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    472[0m [0;34m[0m[0m
[1;32m    473[0m     [0;32mif[0m [0mlen[0m[0;34m([0m[0mlabels_true[0m[0;34m)[0m [0;34m==[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/metrics/cluster/_supervised.py[0m in [0;36mcheck_clusterings[0;34m(labels_true, labels_pred)[0m
[1;32m     39[0m         [0mThe[0m [0mpredicted[0m [0mlabels[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m     """
[0;32m---> 41[0;31m     labels_true = check_array(
[0m[1;32m     42[0m         [0mlabels_true[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m         [0mensure_2d[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    919[0m [0;34m[0m[0m
[1;32m    920[0m         [0;32mif[0m [0mforce_all_finite[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 921[0;31m             _assert_all_finite(
[0m[1;32m    922[0m                 [0marray[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    923[0m                 [0minput_name[0m[0;34m=[0m[0minput_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36m_assert_all_finite[0;34m(X, allow_nan, msg_dtype, estimator_name, input_name)[0m
[1;32m    159[0m                 [0;34m"#estimators-that-handle-nan-values"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    160[0m             )
[0;32m--> 161[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mmsg_err[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    162[0m [0;34m[0m[0m
[1;32m    163[0m [0;34m[0m[0m

[0;31mValueError[0m: Input contains NaN.

## === cell 30
mask_dense_drop = (labels_drop != -1)
mask_rare_drop = (labels_drop == -1)

print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
df_train_dense_drop = df_nyc_taxi[mask_dense_drop]
df_train_rare_drop = df_nyc_taxi[mask_rare_drop]
print('df_train_dense_drop size: %d' % len(df_train_dense_drop))
