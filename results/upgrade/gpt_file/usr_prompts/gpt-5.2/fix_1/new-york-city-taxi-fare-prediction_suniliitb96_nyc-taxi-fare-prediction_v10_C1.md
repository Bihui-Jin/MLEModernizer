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
start_time = timeit.default_timer()

df_train =  pd.read_csv('../input/train.csv', nrows = MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"])
df_test =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])
test_key = df_test['key']
df_train.drop(columns = ['key'], inplace=True)
df_test.drop(columns = ['key'], inplace=True)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3328292387.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # read data in pandas dataframe
      4 df_train =  pd.read_csv('../input/train.csv', nrows = MAX_TRAINING_SIZE, parse_dates=["pickup_datetime"])
      5 df_test =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"])

NameError: name 'timeit' is not defined

## === cell 4
df_train.head()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/897408518.py in <cell line: 0>()
----> 1 df_train.head()

NameError: name 'df_train' is not defined

## === cell 5
print('Old size: %d' % len(df_train))

df_train = df_train[df_train.fare_amount >=0]

df_train = df_train.dropna(how='any', axis='rows')

df_train = df_train.drop(index= df_train[df_train.passenger_count >= 7].index, axis='rows')
df_train = df_train.drop(index= df_train[df_train.passenger_count == 0].index, axis='rows')

print('New size: %d' % len(df_train))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379924533.py in <cell line: 0>()
----> 1 print('Old size: %d' % len(df_train))
      2 
      3 ### Ignore -ve fare
      4 df_train = df_train[df_train.fare_amount >=0]
      5 

NameError: name 'df_train' is not defined

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


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1535071740.py in <cell line: 0>()
     16 BB = (-74.5, -72.8, 40.5, 41.8)
     17 
---> 18 print('Old size: %d' % len(df_train))
     19 df_train = df_train[select_within_boundingbox(df_train, BB)]
     20 print('New size: %d' % len(df_train))

NameError: name 'df_train' is not defined

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
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_test = addPickDropDistanceFeature(df_test)

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2866252144.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # Add pickup-dropoff distance feature
      4 df_train = addPickDropDistanceFeature(df_train)
      5 df_test = addPickDropDistanceFeature(df_test)

NameError: name 'timeit' is not defined

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


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3272970911.py in <cell line: 0>()
     11 #pd.cut(df_train[feat], custom_bucket_array).value_counts()
     12 
---> 13 df_train[feat].hist(bins=bucketsCount, figsize = (15,8))
     14 df_test[feat].hist(bins=bucketsCount, figsize = (15,8))
     15 plt.yscale('log')

NameError: name 'df_train' is not defined

## === cell 10

print('Old size: %d' % len(df_train))
df_train = df_train[df_train.trip_distance < 25.0]
print('New size: %d' % len(df_train))


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1084183929.py in <cell line: 0>()
      1 #(len(df_train[df_train[feat] > 25.0]), len(df_test[df_test[feat] > 25.0]))
      2 
----> 3 print('Old size: %d' % len(df_train))
      4 df_train = df_train[df_train.trip_distance < 25.0]
      5 print('New size: %d' % len(df_train))

NameError: name 'df_train' is not defined

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


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4242227462.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # Add airport trips distance features
      4 df_train = addAirportDistanceFeatures(df_train)
      5 df_test = addAirportDistanceFeatures(df_test)

NameError: name 'timeit' is not defined

## === cell 17

airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
df_test['airport_bound'] = airportTripsIds

airportTripsIds = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train['airport_bound'] = airportTripsIds
df_airport_trips = df_train.loc[airportTripsIds]
df_city_trips = df_train.loc[-airportTripsIds]

pd.DataFrame(data={'Airport Trips' : df_airport_trips.fare_amount, 'City Trips' : df_city_trips.fare_amount}).describe()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3411865564.py in <cell line: 0>()
      1 # Split training data into Airport & City trips
      2 
----> 3 airportTripsIds = getAirportTrips(df_test, RADIUS_VICINITY_AIRPORTS)
      4 df_test['airport_bound'] = airportTripsIds
      5 

NameError: name 'df_test' is not defined

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


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/251390255.py in <cell line: 0>()
     11     return df
     12 
---> 13 start_time = timeit.default_timer()
     14 
     15 df_train = add_datetime_features(df_train)

NameError: name 'timeit' is not defined

## === cell 19
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_test], axis=0, ignore_index=False, sort=False)
df_nyc_taxi.info()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2175684748.py in <cell line: 0>()
----> 1 train_len = len(df_train)
      2 df_nyc_taxi = pd.concat([df_train, df_test], axis=0, ignore_index=False, sort=False)
      3 df_nyc_taxi.info()

NameError: name 'df_train' is not defined

## === cell 20
from sklearn.cluster import DBSCAN

EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN


## === cell 22
start_time = timeit.default_timer()

dbscan_pick = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'pickup_longitude':'pickup_latitude']))

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/889358349.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 # https://stackoverflow.com/questions/34579213/dbscan-for-clustering-of-geographic-location-data
      4 dbscan_pick = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'pickup_longitude':'pickup_latitude']))
      5 

NameError: name 'timeit' is not defined

## === cell 23

mask_pick_all = np.zeros_like(dbscan_pick.labels_, dtype=bool)
mask_pick_all[dbscan_pick.core_sample_indices_] = True
labels_pick = dbscan_pick.labels_
df_nyc_taxi['is_good_dbscan_pick'] = (labels_pick != -1)

n_clusters_pick = len(set(labels_pick)) - (1 if -1 in labels_pick else 0)
n_clusters_pick


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/153204913.py in <cell line: 0>()
      1 # NOTE that each of the clusters' core samples -or- center point are not so much useful in this context => because our goal here is reduce problem space by filtering low-density-drop-pick-both
      2 
----> 3 mask_pick_all = np.zeros_like(dbscan_pick.labels_, dtype=bool)
      4 mask_pick_all[dbscan_pick.core_sample_indices_] = True
      5 labels_pick = dbscan_pick.labels_

NameError: name 'dbscan_pick' is not defined

## === cell 24
print("Pickup: DBSCAN clustering accuracy indicators")
print("Estimated number of clusters: %d" % n_clusters_pick)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_nyc_taxi.fare_amount, labels_pick))
print("Completeness: %0.3f" % metrics.completeness_score(df_nyc_taxi.fare_amount, labels_pick))
print("V-measure: %0.3f" % metrics.v_measure_score(df_nyc_taxi.fare_amount, labels_pick))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(df_nyc_taxi.fare_amount, labels_pick))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(df_nyc_taxi.fare_amount, labels_pick))

start_time = timeit.default_timer()
elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4103165153.py in <cell line: 0>()
      1 print("Pickup: DBSCAN clustering accuracy indicators")
----> 2 print("Estimated number of clusters: %d" % n_clusters_pick)
      3 print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_nyc_taxi.fare_amount, labels_pick))
      4 print("Completeness: %0.3f" % metrics.completeness_score(df_nyc_taxi.fare_amount, labels_pick))
      5 print("V-measure: %0.3f" % metrics.v_measure_score(df_nyc_taxi.fare_amount, labels_pick))

NameError: name 'n_clusters_pick' is not defined

## === cell 25
mask_dense_pick = (labels_pick != -1)
mask_rare_pick = (labels_pick == -1)

print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
df_train_dense_pick = df_nyc_taxi[mask_dense_pick]
df_train_rare_pick = df_nyc_taxi[mask_rare_pick]
print('df_train_dense_pick size: %d' % len(df_train_dense_pick))


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3036701164.py in <cell line: 0>()
----> 1 mask_dense_pick = (labels_pick != -1)
      2 mask_rare_pick = (labels_pick == -1)
      3 
      4 print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
      5 df_train_dense_pick = df_nyc_taxi[mask_dense_pick]

NameError: name 'labels_pick' is not defined

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


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2724128856.py in <cell line: 0>()
      1 # NOTE that our focus of DBSCAN is not to differentiate levels of clusters, i.e., set of connected clusters => hence we are plotting all clusters together
----> 2 plt.plot(df_train_dense_pick.pickup_longitude, df_train_dense_pick.pickup_latitude, 'o')
      3 
      4 '''
      5 unique_labels = set(labels_pick)

NameError: name 'df_train_dense_pick' is not defined

## === cell 27
start_time = timeit.default_timer()

dbscan_drop = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'dropoff_longitude':'dropoff_latitude']))

elapsed = timeit.default_timer() - start_time
elapsed


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3853609935.py in <cell line: 0>()
----> 1 start_time = timeit.default_timer()
      2 
      3 dbscan_drop = DBSCAN(eps=EPS_IN_RADIAN, min_samples=MIN_SAMPLES_CLUSTER, algorithm='ball_tree', metric='haversine').fit(np.radians(df_nyc_taxi.loc[:,'dropoff_longitude':'dropoff_latitude']))
      4 
      5 elapsed = timeit.default_timer() - start_time

NameError: name 'timeit' is not defined

## === cell 28

labels_drop = dbscan_drop.labels_
df_nyc_taxi['is_good_dbscan_drop'] = (labels_drop != -1)

n_clusters_drop = len(set(labels_drop)) - (1 if -1 in labels_drop else 0)
n_clusters_drop


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/377182110.py in <cell line: 0>()
      3 #mask_drop_all = np.zeros_like(dbscan_drop.labels_, dtype=bool)
      4 #mask_drop_all[dbscan_drop.core_sample_indices_] = True
----> 5 labels_drop = dbscan_drop.labels_
      6 df_nyc_taxi['is_good_dbscan_drop'] = (labels_drop != -1)
      7 

NameError: name 'dbscan_drop' is not defined

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
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/316351583.py in <cell line: 0>()
      1 print("DropOff: DBSCAN clustering accuracy indicators")
----> 2 print("Estimated number of clusters: %d" % n_clusters_drop)
      3 print("Homogeneity: %0.3f" % metrics.homogeneity_score(df_nyc_taxi.fare_amount, labels_drop))
      4 print("Completeness: %0.3f" % metrics.completeness_score(df_nyc_taxi.fare_amount, labels_drop))
      5 print("V-measure: %0.3f" % metrics.v_measure_score(df_nyc_taxi.fare_amount, labels_drop))

NameError: name 'n_clusters_drop' is not defined

## === cell 30
mask_dense_drop = (labels_drop != -1)
mask_rare_drop = (labels_drop == -1)

print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
df_train_dense_drop = df_nyc_taxi[mask_dense_drop]
df_train_rare_drop = df_nyc_taxi[mask_rare_drop]
print('df_train_dense_drop size: %d' % len(df_train_dense_drop))


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/151052839.py in <cell line: 0>()
----> 1 mask_dense_drop = (labels_drop != -1)
      2 mask_rare_drop = (labels_drop == -1)
      3 
      4 print('df_nyc_taxi size: %d' % len(df_nyc_taxi))
      5 df_train_dense_drop = df_nyc_taxi[mask_dense_drop]

NameError: name 'labels_drop' is not defined

## === cell 32
df_nyc_taxi['dense_DBSCAN_trips'] = df_nyc_taxi.apply(
    (lambda row: (row.is_good_dbscan_drop & row.is_good_dbscan_pick)),
    axis='columns'
)

len(df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3914368271.py in <cell line: 0>()
----> 1 df_nyc_taxi['dense_DBSCAN_trips'] = df_nyc_taxi.apply(
      2     (lambda row: (row.is_good_dbscan_drop & row.is_good_dbscan_pick)),
      3     axis='columns'
      4 )
      5 

NameError: name 'df_nyc_taxi' is not defined

## === cell 33
df_nyc_taxi.head()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3730666567.py in <cell line: 0>()
----> 1 df_nyc_taxi.head()

NameError: name 'df_nyc_taxi' is not defined

## === cell 34

df_train = df_nyc_taxi.iloc[:train_len, :]
df_test = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != 'fare_amount']

(len(df_train), len(df_test))


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/566930790.py in <cell line: 0>()
      2 #df_test = df_nyc_taxi.loc[df_nyc_taxi['key'] > train_len].iloc[:, df_nyc_taxi.columns != 'fare_amount']
      3 
----> 4 df_train = df_nyc_taxi.iloc[:train_len, :]
      5 df_test = df_nyc_taxi.iloc[train_len:, :].iloc[:, df_nyc_taxi.columns != 'fare_amount']
      6 

NameError: name 'df_nyc_taxi' is not defined

## === cell 35
pd.DataFrame(data={'Good Density Pickups' : df_train.loc[df_train.is_good_dbscan_pick == 1].fare_amount, 'LOW Density Pickups' : df_train.loc[df_train.is_good_dbscan_pick == 0].fare_amount}).describe()


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1113671364.py in <cell line: 0>()
----> 1 pd.DataFrame(data={'Good Density Pickups' : df_train.loc[df_train.is_good_dbscan_pick == 1].fare_amount, 'LOW Density Pickups' : df_train.loc[df_train.is_good_dbscan_pick == 0].fare_amount}).describe()

NameError: name 'df_train' is not defined

## === cell 36
pd.DataFrame(data={'Good Density DropOffs' : df_train.loc[df_train.is_good_dbscan_drop == 1].fare_amount, 'LOW Density DropOffs' : df_train.loc[df_train.is_good_dbscan_drop == 0].fare_amount}).describe()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1571734895.py in <cell line: 0>()
----> 1 pd.DataFrame(data={'Good Density DropOffs' : df_train.loc[df_train.is_good_dbscan_drop == 1].fare_amount, 'LOW Density DropOffs' : df_train.loc[df_train.is_good_dbscan_drop == 0].fare_amount}).describe()

NameError: name 'df_train' is not defined

## === cell 37
df_temp = df_train[['fare_amount', 'trip_distance']]
df_temp.loc[df_temp.trip_distance < 0.2, 'trip_distance'] = 0.2

(df_temp.fare_amount / df_temp.trip_distance).hist(bins=bucketsCount, figsize = (15,8))
plt.yscale('log')
plt.xlabel(feat)
plt.ylabel("Frequency")


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2401379494.py in <cell line: 0>()
      1 # Ceiling near-zero fare values to 0.2 to check fare/dist behavior
----> 2 df_temp = df_train[['fare_amount', 'trip_distance']]
      3 df_temp.loc[df_temp.trip_distance < 0.2, 'trip_distance'] = 0.2
      4 
      5 (df_temp.fare_amount / df_temp.trip_distance).hist(bins=bucketsCount, figsize = (15,8))

NameError: name 'df_train' is not defined

## === cell 38
df_temp['trip_rate'] = df_temp.apply(
    (lambda row: (row.fare_amount / row.trip_distance)),
    axis='columns'
)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2820447132.py in <cell line: 0>()
----> 1 df_temp['trip_rate'] = df_temp.apply(
      2     (lambda row: (row.fare_amount / row.trip_distance)),
      3     axis='columns'
      4 )

NameError: name 'df_temp' is not defined

## === cell 39
len(df_temp.loc[df_temp.trip_rate > THERSHOLD_TRIP_FARE_RATE])


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3674740667.py in <cell line: 0>()
----> 1 len(df_temp.loc[df_temp.trip_rate > THERSHOLD_TRIP_FARE_RATE])

NameError: name 'df_temp' is not defined

## === cell 40
ids = (df_temp.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print('Old size: %d' % len(df_train))
df_train = df_train[ids]
print('New size: %d' % len(df_train))


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2993387820.py in <cell line: 0>()
----> 1 ids = (df_temp.trip_rate < THERSHOLD_TRIP_FARE_RATE)
      2 
      3 print('Old size: %d' % len(df_train))
      4 df_train = df_train[ids]
      5 print('New size: %d' % len(df_train))

NameError: name 'df_temp' is not defined

## === cell 41

df_tmp = df_train.drop(columns = ['pickup_datetime', 'is_good_dbscan_drop', 'is_good_dbscan_pick', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
df_tmp.info()


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1851627956.py in <cell line: 0>()
      2 #df_train.info()
      3 
----> 4 df_tmp = df_train.drop(columns = ['pickup_datetime', 'is_good_dbscan_drop', 'is_good_dbscan_pick', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
      5 df_tmp.info()

NameError: name 'df_train' is not defined

## === cell 42
y = df_tmp['fare_amount']
train = df_tmp.drop(columns=['fare_amount'])

x_train, x_test, y_train, y_test = train_test_split(train, y, random_state=0, test_size=0.01)


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1249552587.py in <cell line: 0>()
----> 1 y = df_tmp['fare_amount']
      2 train = df_tmp.drop(columns=['fare_amount'])
      3 
      4 x_train, x_test, y_train, y_test = train_test_split(train, y, random_state=0, test_size=0.01)

NameError: name 'df_tmp' is not defined

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


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2363490757.py in <cell line: 0>()
      7     return model
      8 
----> 9 model = XGBmodel(x_train,x_test,y_train,y_test,params)

NameError: name 'x_train' is not defined

## === cell 45
x_pred = df_test.drop(columns = ['pickup_datetime', 'is_good_dbscan_drop', 'is_good_dbscan_pick', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3589173125.py in <cell line: 0>()
      1 #Read and preprocess test set
----> 2 x_pred = df_test.drop(columns = ['pickup_datetime', 'is_good_dbscan_drop', 'is_good_dbscan_pick', 'pickup_distance_to_jfk', 'drop_distance_to_jfk', 'pickup_distance_to_lgr', 'drop_distance_to_lgr', 'pickup_distance_to_ewr', 'drop_distance_to_ewr', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])
      3 
      4 #Predict from test set
      5 prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)

NameError: name 'df_test' is not defined

## === cell 46
len(test_key)


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3634691910.py in <cell line: 0>()
----> 1 len(test_key)

NameError: name 'test_key' is not defined

## === cell 47
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('taxi_fare_submission.csv',index=False)
submission.head()


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1489528079.py in <cell line: 0>()
      1 #Create submission file
      2 submission = pd.DataFrame({
----> 3         "key": test_key,
      4         "fare_amount": prediction.round(2)
      5 })

NameError: name 'test_key' is not defined
